"""
ResumeSync AI Search Backend
=============================

An AI-powered candidate search API that lets HR describe a hire in plain
English ("Need a backend developer with Razorpay experience") and get back
ranked, explainable candidate matches from an existing MongoDB Atlas
collection of pre-structured resumes.

Design summary (see inline section comments for detail):

  1. Gemini is used ONLY for query understanding (NL -> structured intent).
     It never sees candidate data and never produces candidate names.
  2. A static skill ontology expands the recruiter's intent into related
     technologies/concepts (e.g. "Backend" -> FastAPI, Django, REST API...).
     This is what lets a "Razorpay" search surface a "Stripe" candidate.
  3. MongoDB is used for cheap structural pre-filtering (experience,
     location, role keyword) to shrink the candidate pool before any
     expensive scoring happens.
  4. Semantic similarity is computed with `sentence-transformers`
     (all-MiniLM-L6-v2 -- small, CPU-friendly, Render/Railway-compatible).
     Candidate embeddings are NEVER persisted to MongoDB. They are kept in
     a process-local, TTL-based in-memory cache so that a given resume
     isn't re-embedded on every single search request, while nothing is
     written back to the database. Cold start / cache-miss simply means
     "embed it now, keep it in RAM for a while."
  5. Final ranking combines multiple independent, explainable sub-scores
     (role, skill, semantic, experience, achievements, location, notice)
     into one weighted match_score, with human-readable reasons.
"""

from __future__ import annotations

import os
import re
import json
import time
import logging
import hashlib
import uuid
import contextvars
import threading
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from bson import ObjectId
from bson.errors import InvalidId

# ---------------------------------------------------------------------------
# SECTION: Configuration & Logging Setup
# ---------------------------------------------------------------------------

load_dotenv()

request_id_context = contextvars.ContextVar("request_id", default="system")

class RequestIdFilter(logging.Filter):
    """Injects the request ID into log records."""
    def filter(self, record):
        record.request_id = request_id_context.get()
        return True

# Set up logging with Request ID
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | [%(request_id)s] | %(message)s",
)
logger = logging.getLogger("resumesync")
logger.addFilter(RequestIdFilter())


class Settings:
    """Centralised, validated environment configuration."""
    def __init__(self) -> None:
        self.mongo_uri: str | None = os.getenv("MONGO_URI")
        self.database_name: str = os.getenv("DATABASE_NAME", "recruitment_db")
        self.collection_name: str = os.getenv("COLLECTION_NAME", "candidates")
        self.gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
        self.gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.embedding_model_name: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.candidate_pool_limit: int = int(os.getenv("CANDIDATE_POOL_LIMIT", "300"))
        self.max_results: int = int(os.getenv("MAX_RESULTS", "20"))
        self.embedding_cache_ttl: int = int(os.getenv("EMBEDDING_CACHE_TTL", "1800"))
        self.cors_origins: list[str] = [
            o.strip()
            for o in os.getenv("CORS_ORIGINS", "https://resumesync.in").split(",")
            if o.strip()
        ]

    def validate(self) -> list[str]:
        problems = []
        if not self.mongo_uri:
            problems.append("MONGO_URI is not set")
        if not self.gemini_api_key:
            problems.append("GEMINI_API_KEY is not set")
        return problems

settings = Settings()

# ---------------------------------------------------------------------------
# SECTION: MongoDB Client
# ---------------------------------------------------------------------------

_mongo_client: MongoClient | None = None
_mongo_lock = threading.Lock()

def get_mongo_client() -> MongoClient:
    global _mongo_client
    if _mongo_client is None:
        with _mongo_lock:
            # Check again in case another thread initialized it while waiting
            if _mongo_client is None:
                start_time = time.perf_counter()
                if not settings.mongo_uri:
                    raise RuntimeError("MONGO_URI is not configured")
                _mongo_client = MongoClient(
                    settings.mongo_uri,
                    serverSelectionTimeoutMS=8000,
                    connectTimeoutMS=8000,
                    maxPoolSize=50, # Keep a healthy pool for Railway
                )
                logger.info("MongoDB client connected in %.2f sec", time.perf_counter() - start_time)
    return _mongo_client

def get_candidates_collection():
    client = get_mongo_client()
    db = client[settings.database_name]
    return db[settings.collection_name]


# ---------------------------------------------------------------------------
# SECTION: Gemini Client (Query Understanding ONLY)
# ---------------------------------------------------------------------------

_gemini_configured = False
_gemini_lock = threading.Lock()

def _ensure_gemini_configured() -> None:
    global _gemini_configured
    if _gemini_configured:
        return
    with _gemini_lock:
        if _gemini_configured:
            return
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        import google.generativeai as genai
        genai.configure(api_key=settings.gemini_api_key)
        _gemini_configured = True
        logger.info("Gemini client configured successfully.")

GEMINI_SYSTEM_PROMPT = """You are a query-understanding engine for a recruiting \
search system. You convert a recruiter's natural language request into a \
strict JSON object describing their intent.

Rules:
- Output ONLY valid JSON. No markdown fences, no commentary, no preamble.
- Never invent a candidate name. You have no access to any candidate data.
- Never guess at a field you cannot reasonably infer from the text -- use \
null (or an empty list for list fields) instead of fabricating a value.
- "experience_min" is a number of years (integer) or null if unspecified. \
Phrases like "minimum 2 years" or "at least 3+ years" should populate this.
- "location" should be a city/region name or null.
- "skills" is a list of explicit technology / tool names mentioned or \
directly implied by the role itself (e.g. "Flutter developer" implies \
"Flutter" even if not separately stated).
- "domain" is a short label like "FinTech", "Healthcare", "E-commerce", \
"AI/ML", "Cloud Infrastructure", "Data Engineering", or null if unclear.
- "concepts" is a list of broader technical concepts implied by the request \
(e.g. "Payment Gateway", "Microservices", "REST API"), separate from the \
literal skills list.

Output schema:
{
  "role": string or null,
  "experience_min": integer or null,
  "location": string or null,
  "skills": [string],
  "domain": string or null,
  "concepts": [string]
}
"""

class QueryIntent(BaseModel):
    role: Optional[str] = None
    experience_min: Optional[int] = None
    location: Optional[str] = None
    skills: list[str] = Field(default_factory=list)
    domain: Optional[str] = None
    concepts: list[str] = Field(default_factory=list)

def extract_query_intent(query: str) -> QueryIntent:
    step_start = time.perf_counter()
    try:
        _ensure_gemini_configured()
        import google.generativeai as genai

        model = genai.GenerativeModel(
            settings.gemini_model,
            system_instruction=GEMINI_SYSTEM_PROMPT,
            generation_config={"response_mime_type": "application/json"},
        )
        
        # PREVENT HANGING: Force a strict timeout
        response = model.generate_content(
            query, 
            request_options={"timeout": 5.0} 
        )
        
        raw_text = (response.text or "").strip()
        data = json.loads(raw_text)
        intent = QueryIntent(**data)
        logger.info("Gemini extraction completed successfully in %.2f sec", time.perf_counter() - step_start)
        return intent
    except Exception:
        logger.exception("Gemini query understanding failed or timed out. Switching to fallback.")
        return _fallback_query_intent(query)

_EXPERIENCE_PATTERN = re.compile(r"(\d+)\s*\+?\s*(?:years?|yrs?)", re.IGNORECASE)

def _fallback_query_intent(query: str) -> QueryIntent:
    step_start = time.perf_counter()
    experience_min = None
    match = _EXPERIENCE_PATTERN.search(query)
    if match:
        experience_min = int(match.group(1))

    lowered = query.lower()
    found_skills = [
        term for term in ALL_KNOWN_TERMS if term.lower() in lowered
    ]
    logger.info("Fallback extraction completed in %.2f sec", time.perf_counter() - step_start)
    return QueryIntent(
        role=None,
        experience_min=experience_min,
        location=None,
        skills=found_skills,
        domain=None,
        concepts=[],
    )

# ---------------------------------------------------------------------------
# SECTION: Intelligent Skill Ontology
# ---------------------------------------------------------------------------

SKILL_ONTOLOGY: dict[str, list[str]] = {
    "Flutter": ["Dart", "Firebase", "Android", "iOS", "Cross Platform", "Material UI", "Mobile Development"],
    "Mobile Development": ["Flutter", "React Native", "Swift", "Kotlin", "Android", "iOS", "Dart"],
    "Frontend": ["React", "Angular", "Vue", "Next.js", "JavaScript", "TypeScript", "HTML", "CSS", "Redux", "Tailwind"],
    "Backend": ["Java", "Python", "Node.js", "Go", "Spring Boot", "FastAPI", "Express", "Django", "Flask", "REST API", "GraphQL", "Microservices", "SQL"],
    "Cloud": ["AWS", "Azure", "GCP", "Terraform", "Docker", "Kubernetes", "CloudFormation", "DevOps", "Linux", "NGINX", "CI/CD"],
    "DevOps": ["Docker", "Kubernetes", "CI/CD", "Terraform", "Jenkins", "GitHub Actions", "Ansible", "Linux", "Monitoring"],
    "Data Engineering": ["Spark", "Kafka", "Airflow", "Snowflake", "AWS Glue", "Redshift", "Databricks", "ETL", "SQL", "Python"],
    "AI/ML": ["TensorFlow", "PyTorch", "LLM", "LangChain", "Transformers", "RAG", "Gemini", "OpenAI", "Hugging Face", "Embeddings", "Scikit-learn", "NLP", "Computer Vision"],
    "Payment Gateway": ["Razorpay", "Stripe", "PayPal", "Webhook", "Transaction Processing", "PCI DSS", "FinTech", "API Security"],
    "FinTech": ["Payment Gateway", "Razorpay", "Stripe", "Transaction Processing", "Banking APIs", "Compliance", "KYC"],
    "Database": ["MongoDB", "PostgreSQL", "MySQL", "Redis", "DynamoDB", "Cassandra", "SQL", "NoSQL"],
    "Security": ["OAuth", "JWT", "API Security", "Authentication", "Encryption", "PCI DSS"],
    "Testing": ["PyTest", "JUnit", "Selenium", "Cypress", "Jest", "Unit Testing", "Integration Testing"],
}

ROLE_ALIASES: dict[str, str] = {
    "ai engineer": "AI/ML", "ai developer": "AI/ML", "ml engineer": "AI/ML",
    "machine learning engineer": "AI/ML", "llm engineer": "AI/ML", "data scientist": "AI/ML",
    "cloud engineer": "Cloud", "cloud architect": "Cloud", "devops engineer": "DevOps",
    "site reliability engineer": "DevOps", "sre": "DevOps", "data engineer": "Data Engineering",
    "frontend developer": "Frontend", "front-end developer": "Frontend", "ui developer": "Frontend",
    "backend developer": "Backend", "backend engineer": "Backend", "full stack developer": "Backend",
    "fullstack developer": "Backend", "mobile developer": "Mobile Development", "app developer": "Mobile Development",
    "qa engineer": "Testing", "test engineer": "Testing", "database administrator": "Database",
    "dba": "Database", "security engineer": "Security",
}

_TERM_TO_CLUSTERS: dict[str, set[str]] = {}
for _cluster, _members in SKILL_ONTOLOGY.items():
    _TERM_TO_CLUSTERS.setdefault(_cluster.lower(), set()).add(_cluster)
    for _m in _members:
        _TERM_TO_CLUSTERS.setdefault(_m.lower(), set()).add(_cluster)
for _alias, _cluster in ROLE_ALIASES.items():
    _TERM_TO_CLUSTERS.setdefault(_alias, set()).add(_cluster)

ALL_KNOWN_TERMS: list[str] = sorted(
    {SKILL_ONTOLOGY_KEY for SKILL_ONTOLOGY_KEY in SKILL_ONTOLOGY.keys()}
    | {member for members in SKILL_ONTOLOGY.values() for member in members}
)

def _clusters_for_term(term: str) -> set[str]:
    key = term.strip().lower()
    if not key:
        return set()
    exact = _TERM_TO_CLUSTERS.get(key)
    if exact:
        return set(exact)
    matches: set[str] = set()
    for known_term, clusters in _TERM_TO_CLUSTERS.items():
        if len(known_term) < 3:
            if key == known_term:
                matches.update(clusters)
            continue
        if known_term in key or key in known_term:
            matches.update(clusters)
    return matches

_term_cluster_cache: dict[str, set[str]] = {}

def _cached_clusters_for_term(term: str) -> set[str]:
    key = term.strip().lower()
    if key not in _term_cluster_cache:
        _term_cluster_cache[key] = _clusters_for_term(term)
    return _term_cluster_cache[key]

def expand_terms(terms: list[str]) -> list[str]:
    expanded: set[str] = set()
    for term in terms:
        clusters = _cached_clusters_for_term(term)
        for cluster in clusters:
            expanded.add(cluster)
            expanded.update(SKILL_ONTOLOGY.get(cluster, []))
    return sorted(expanded)

def are_related(skill_a: str, skill_b: str) -> bool:
    a, b = skill_a.strip().lower(), skill_b.strip().lower()
    if a == b:
        return True
    clusters_a = _cached_clusters_for_term(a)
    clusters_b = _cached_clusters_for_term(b)
    return bool(clusters_a & clusters_b)

# ---------------------------------------------------------------------------
# SECTION: Pydantic Models (API contract)
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)

class CandidateResult(BaseModel):
    id: str
    name: str
    experience: Optional[float] = None
    location: Optional[str] = None
    match_score: int
    matched_skills: list[str] = Field(default_factory=list)
    matched_concepts: list[str] = Field(default_factory=list)
    reason: str
    score_breakdown: dict[str, int] = Field(default_factory=dict)

class SearchResponse(BaseModel):
    query: str
    result_count: int
    results: list[CandidateResult]

class HealthResponse(BaseModel):
    status: str
    mongo_connected: bool
    gemini_configured: bool

# ---------------------------------------------------------------------------
# SECTION: Embedding Engine (in-memory only)
# ---------------------------------------------------------------------------

_embedding_model = None
_embedding_model_lock = threading.Lock()

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        with _embedding_model_lock:
            # Prevent race conditions during concurrent cold starts
            if _embedding_model is None:
                step_start = time.perf_counter()
                logger.info("Loading embedding model: %s...", settings.embedding_model_name)
                from sentence_transformers import SentenceTransformer
                _embedding_model = SentenceTransformer(settings.embedding_model_name)
                logger.info("Embedding model loaded in %.2f sec", time.perf_counter() - step_start)
    return _embedding_model

class _TTLEmbeddingCache:
    def __init__(self, ttl: int) -> None:
        self._ttl = ttl
        self._store: dict[str, tuple[float, np.ndarray]] = {}

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> np.ndarray | None:
        key = self._key(text)
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, vector = entry
        if time.time() > expires_at:
            self._store.pop(key, None)
            return None
        return vector

    def set(self, text: str, vector: np.ndarray) -> None:
        key = self._key(text)
        self._store[key] = (time.time() + self._ttl, vector)

    def get_or_compute_many(self, texts: list[str]) -> np.ndarray:
        step_start_lookup = time.perf_counter()
        results: list[np.ndarray | None] = [self.get(t) for t in texts]
        miss_indices = [i for i, v in enumerate(results) if v is None]
        hits = len(texts) - len(miss_indices)
        
        logger.info(
            "Embedding cache stats: %d hits, %d misses. Lookup time: %.4f sec. Cache size: %d",
            hits, len(miss_indices), time.perf_counter() - step_start_lookup, len(self._store)
        )

        if miss_indices:
            step_start_compute = time.perf_counter()
            model = get_embedding_model()
            miss_texts = [texts[i] for i in miss_indices]
            computed = model.encode(miss_texts, show_progress_bar=False)
            
            for idx, vec in zip(miss_indices, computed):
                vec = np.asarray(vec, dtype=np.float32)
                self.set(texts[idx], vec)
                results[idx] = vec
                
            logger.info("Computed %d new embeddings in %.2f sec", len(miss_indices), time.perf_counter() - step_start_compute)

        return np.vstack(results)  # type: ignore[arg-type]

_embedding_cache = _TTLEmbeddingCache(ttl=settings.embedding_cache_ttl)

def cosine_similarity_batch(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    step_start = time.perf_counter()
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    res = matrix_norms @ query_norm
    logger.debug("Cosine similarity computed in %.4f sec", time.perf_counter() - step_start)
    return res

# ---------------------------------------------------------------------------
# SECTION: Query Expansion
# ---------------------------------------------------------------------------

class ExpandedQuery(BaseModel):
    intent: QueryIntent
    expanded_terms: list[str]
    semantic_text: str

def build_expanded_query(intent: QueryIntent, raw_query: str) -> ExpandedQuery:
    step_start = time.perf_counter()
    seed_terms = list({*intent.skills, *intent.concepts, *( [intent.role] if intent.role else [] )})
    expanded = expand_terms(seed_terms)

    semantic_text_parts = [raw_query]
    if intent.role:
        semantic_text_parts.append(f"Role: {intent.role}")
    if intent.domain:
        semantic_text_parts.append(f"Domain: {intent.domain}")
    if intent.skills:
        semantic_text_parts.append("Skills: " + ", ".join(intent.skills))
    if intent.concepts:
        semantic_text_parts.append("Concepts: " + ", ".join(intent.concepts))
    if expanded:
        semantic_text_parts.append("Related: " + ", ".join(expanded))

    res = ExpandedQuery(
        intent=intent,
        expanded_terms=expanded,
        semantic_text=" | ".join(semantic_text_parts),
    )
    logger.info("Query expansion completed in %.4f sec", time.perf_counter() - step_start)
    return res

# ---------------------------------------------------------------------------
# SECTION: Mongo Filtering (cheap structural pre-filter)
# ---------------------------------------------------------------------------

def build_mongo_filter(intent: QueryIntent) -> dict:
    conditions: list[dict] = []
    if intent.location:
        conditions.append({
            "output.candidate.location": {
                "$regex": re.escape(intent.location),
                "$options": "i",
            }
        })
    if intent.experience_min:
        conditions.append({"exp_years_num": {"$gte": max(intent.experience_min - 1, 0)}})
    if not conditions:
        return {}
    return {"$and": conditions}

def fetch_candidate_pool(mongo_filter: dict, limit: int) -> list[dict]:
    step_start = time.perf_counter()
    collection = get_candidates_collection()
    projection = {
        "output.candidate": 1, "output.summary": 1, "output.fit_score": 1,
        "exp_years_num": 1, "search_text": 1,
    }
    
    logger.info("Executing Mongo query with filter: %s", json.dumps(mongo_filter, default=str))
    try:
        cursor = collection.find(mongo_filter, projection).limit(limit)
        docs = list(cursor)
        duration = time.perf_counter() - step_start
        logger.info(
            "Mongo query completed. Candidates fetched: %d | Mongo time: %.0f ms (%.4f sec)", 
            len(docs), duration * 1000, duration
        )
        return docs
    except PyMongoError as exc:
        logger.exception("Mongo query failed miserably")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Candidate database is temporarily unavailable.",
        ) from exc


# ---------------------------------------------------------------------------
# SECTION: Match Score Calculation (multi-stage, explainable ranking)
# ---------------------------------------------------------------------------

SCORE_WEIGHTS = {
    "role": 0.20, "skill": 0.30, "semantic": 0.20, "experience": 0.15,
    "achievements": 0.05, "location": 0.05, "notice": 0.05,
}
assert abs(sum(SCORE_WEIGHTS.values()) - 1.0) < 1e-6

def _safe_lower(value: Any) -> str:
    return str(value).lower() if value else ""

def score_role_match(intent: QueryIntent, current_role: str) -> tuple[float, list[str]]:
    if not intent.role or not current_role:
        return 0.5, []
    role_q, role_c = _safe_lower(intent.role), _safe_lower(current_role)
    if role_q in role_c or role_c in role_q:
        return 1.0, [f"Current role '{current_role}' matches requested role"]
    q_tokens, c_tokens = set(role_q.split()), set(role_c.split())
    overlap = q_tokens & c_tokens
    if overlap:
        return min(1.0, 0.4 + 0.2 * len(overlap)), [f"Role overlap with '{current_role}'"]
    return 0.2, []

def score_skill_match(query_skills: list[str], candidate_skills: list[str]) -> tuple[float, list[str], list[str]]:
    if not query_skills:
        return 0.5, [], []
    candidate_skills_lower = {s.lower(): s for s in candidate_skills}
    matched: list[str] = []
    related_matched: list[str] = []
    total_weight = 0.0

    for q_skill in query_skills:
        q_lower = q_skill.lower()
        if q_lower in candidate_skills_lower:
            matched.append(candidate_skills_lower[q_lower])
            total_weight += 1.0
            continue
        found_related = False
        for c_lower, c_original in candidate_skills_lower.items():
            if are_related(q_skill, c_lower):
                related_matched.append(c_original)
                total_weight += 0.6
                found_related = True
                break
        if not found_related:
            continue

    score = min(1.0, total_weight / max(len(query_skills), 1))
    return score, list(dict.fromkeys(matched)), list(dict.fromkeys(related_matched))

def score_semantic_match(similarity: float) -> tuple[float, list[str]]:
    clamped = max(0.0, min(1.0, similarity))
    reasons = []
    if clamped >= 0.6:
        reasons.append("Strong overall semantic match to the role description")
    elif clamped >= 0.4:
        reasons.append("Moderate semantic alignment with the role description")
    return clamped, reasons

def score_experience_match(required_min: Optional[int], candidate_years: Optional[float]) -> tuple[float, list[str]]:
    if required_min is None: return 0.7, []
    if candidate_years is None: return 0.3, []
    if candidate_years >= required_min:
        return 1.0, ["Experience exceeds requirement" if candidate_years > required_min else "Meets required experience"]
    ratio = candidate_years / required_min if required_min > 0 else 1.0
    return max(0.0, ratio * 0.8), []

def score_achievements_match(query_terms: list[str], achievements: list[str]) -> tuple[float, list[str]]:
    if not achievements or not query_terms: return 0.0, []
    text = " ".join(str(a) for a in achievements).lower()
    hits = [t for t in query_terms if t.lower() in text]
    if not hits: return 0.0, []
    return min(1.0, 0.3 + 0.2 * len(hits)), [f"Key achievement involving {hits[0]}"]

def score_location_match(required_location: Optional[str], candidate_location: Optional[str]) -> tuple[float, list[str]]:
    if not required_location: return 1.0, []
    if not candidate_location: return 0.3, []
    if required_location.lower() in candidate_location.lower():
        return 1.0, [f"Located in {candidate_location}"]
    return 0.1, []

def score_notice_match(notice_period: Optional[str]) -> tuple[float, list[str]]:
    if not notice_period: return 0.5, []
    digits = re.findall(r"\d+", str(notice_period))
    if not digits: return 0.5, []
    days = int(digits[0])
    if days <= 15: return 1.0, ["Short notice period"]
    if days <= 30: return 0.8, []
    if days <= 60: return 0.6, []
    return 0.4, []

def calculate_match_score(
    expanded_query: ExpandedQuery, candidate_doc: dict, semantic_similarity: float
) -> tuple[int, dict[str, int], list[str], list[str], str]:
    output = candidate_doc.get("output", {}) or {}
    candidate_info = output.get("candidate", {}) or {}
    summary = output.get("summary", {}) or {}
    intent = expanded_query.intent
    
    candidate_skills = summary.get("technical_skills", []) or []
    achievements = summary.get("key_achievements", []) or []
    current_role = summary.get("current_role", "") or ""
    candidate_location = candidate_info.get("location")
    notice_period = candidate_info.get("notice_period")
    candidate_years = candidate_doc.get("exp_years_num")

    role_score, role_reasons = score_role_match(intent, current_role)
    query_skills_for_match = list({*intent.skills, *intent.concepts})
    skill_score, matched_skills, related_skills = score_skill_match(query_skills_for_match, candidate_skills)
    
    skill_reasons = []
    if matched_skills: skill_reasons.append("Strong " + ", ".join(matched_skills[:4]) + " experience")
    if related_skills: skill_reasons.append("Related technology experience: " + ", ".join(related_skills[:3]))

    semantic_score, semantic_reasons = score_semantic_match(semantic_similarity)
    experience_score, experience_reasons = score_experience_match(intent.experience_min, candidate_years)
    all_query_terms = list({*intent.skills, *intent.concepts, *expanded_query.expanded_terms})
    achievement_score, achievement_reasons = score_achievements_match(all_query_terms, achievements)
    location_score, location_reasons = score_location_match(intent.location, candidate_location)
    notice_score, notice_reasons = score_notice_match(notice_period)

    breakdown_raw = {
        "role": role_score, "skill": skill_score, "semantic": semantic_score,
        "experience": experience_score, "achievements": achievement_score,
        "location": location_score, "notice": notice_score,
    }

    final_score = sum(breakdown_raw[k] * SCORE_WEIGHTS[k] for k in SCORE_WEIGHTS)
    matched_concepts = [c for c in intent.concepts if c.lower() in " ".join(candidate_skills + achievements).lower()] or related_skills[:3]
    
    all_reasons = role_reasons + skill_reasons + semantic_reasons + experience_reasons + achievement_reasons + location_reasons + notice_reasons
    reason_text = ". ".join(all_reasons) if all_reasons else "General profile alignment with the search criteria."
    combined_skills = list(dict.fromkeys(matched_skills + related_skills))
    
    return round(final_score * 100), {k: round(v * 100) for k, v in breakdown_raw.items()}, combined_skills, matched_concepts, reason_text

# ---------------------------------------------------------------------------
# SECTION: Semantic Ranking 
# ---------------------------------------------------------------------------

def rank_candidates(expanded_query: ExpandedQuery, candidate_docs: list[dict]) -> list[CandidateResult]:
    if not candidate_docs:
        return []

    step_start = time.perf_counter()
    model = get_embedding_model()
    
    step_encode = time.perf_counter()
    query_vector = np.asarray(
        model.encode([expanded_query.semantic_text], show_progress_bar=False)[0],
        dtype=np.float32,
    )
    logger.debug("Query encoded in %.4f sec", time.perf_counter() - step_encode)

    search_texts = [doc.get("search_text", "") or "" for doc in candidate_docs]
    candidate_matrix = _embedding_cache.get_or_compute_many(search_texts)

    similarities = cosine_similarity_batch(query_vector, candidate_matrix)

    results: list[CandidateResult] = []
    for doc, similarity in zip(candidate_docs, similarities):
        score_pct, breakdown, matched_skills, matched_concepts, reason = calculate_match_score(
            expanded_query, doc, float(similarity)
        )
        candidate_info = (doc.get("output", {}) or {}).get("candidate", {}) or {}
        first_name = candidate_info.get("first_name", "") or ""
        last_name = candidate_info.get("last_name", "") or ""
        full_name = f"{first_name} {last_name}".strip() or "Unknown Candidate"

        results.append(
            CandidateResult(
                id=str(doc.get("_id")), name=full_name, experience=doc.get("exp_years_num"),
                location=candidate_info.get("location"), match_score=score_pct,
                matched_skills=matched_skills[:8], matched_concepts=matched_concepts[:5],
                reason=reason, score_breakdown=breakdown,
            )
        )

    results.sort(key=lambda r: r.match_score, reverse=True)
    logger.info("Candidate ranking process completed entirely in %.2f sec", time.perf_counter() - step_start)
    return results

def format_search_response(query: str, ranked: list[CandidateResult], max_results: int) -> SearchResponse:
    top = ranked[:max_results]
    return SearchResponse(query=query, result_count=len(top), results=top)

# ---------------------------------------------------------------------------
# SECTION: Utilities
# ---------------------------------------------------------------------------

def to_object_id(raw_id: str) -> ObjectId:
    try:
        return ObjectId(raw_id)
    except (InvalidId, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{raw_id}' is not a valid candidate id.",
        ) from exc

def serialize_mongo_doc(doc: dict) -> dict:
    if isinstance(doc, dict):
        return {k: serialize_mongo_doc(v) for k, v in doc.items()}
    if isinstance(doc, list):
        return [serialize_mongo_doc(v) for v in doc]
    if isinstance(doc, ObjectId):
        return str(doc)
    return doc

# ---------------------------------------------------------------------------
# SECTION: FastAPI Endpoints
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ResumeSync AI Search API",
    description="AI-powered natural language candidate search backend.",
    version="1.0.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """Middleware to inject a UUID for traceability."""
    req_id = str(uuid.uuid4())[:8]
    request_id_context.set(req_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response

@app.on_event("startup")
def on_startup() -> None:
    problems = settings.validate()
    if problems:
        for p in problems:
            logger.warning("Configuration issue: %s", p)
    else:
        logger.info("Configuration OK. Application starting.")

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    mongo_ok = False
    try:
        get_mongo_client().admin.command("ping")
        mongo_ok = True
    except Exception:
        logger.exception("Health check: Mongo not reachable")

    return HealthResponse(
        status="ok" if mongo_ok else "degraded",
        mongo_connected=mongo_ok,
        gemini_configured=bool(settings.gemini_api_key),
    )

@app.post("/search", response_model=SearchResponse, tags=["search"])
def search_candidates(request: SearchRequest) -> SearchResponse:
    start_total = time.perf_counter()
    query_text = request.query.strip()
    
    if not query_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must not be empty.",
        )

    logger.info("STEP 1: Request received. Query: '%s'", query_text)

    # Step 2: Query Understanding (Gemini with failsafe)
    step = time.perf_counter()
    intent = extract_query_intent(query_text)
    logger.info("STEP 2 (Intent Extraction) completed in %.2f sec", time.perf_counter() - step)

    # Step 3: Ontology Expansion
    step = time.perf_counter()
    expanded_query = build_expanded_query(intent, query_text)
    logger.info("STEP 3 (Query Expansion) completed in %.2f sec", time.perf_counter() - step)

    # Step 4: Mongo Pre-filter
    step = time.perf_counter()
    mongo_filter = build_mongo_filter(intent)
    
    try:
        candidate_pool = fetch_candidate_pool(mongo_filter, settings.candidate_pool_limit)
        
        if not candidate_pool and mongo_filter:
            logger.info("No candidates from filtered query; retrying unfiltered fallback.")
            candidate_pool = fetch_candidate_pool({}, settings.candidate_pool_limit)
            
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error during Mongo fetch")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve candidates from database.",
        )
        
    logger.info("STEP 4 (Mongo Fetch) completed in %.2f sec", time.perf_counter() - step)

    if not candidate_pool:
        logger.info("TOTAL SEARCH TIME %.2f sec (0 Results)", time.perf_counter() - start_total)
        return SearchResponse(query=query_text, result_count=0, results=[])

    # Steps 5: Semantic Ranking
    step = time.perf_counter()
    try:
        ranked = rank_candidates(expanded_query, candidate_pool)
    except Exception:
        logger.exception("Ranking execution failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to rank candidates.",
        )
    logger.info("STEP 5 (Semantic Ranking) completed in %.2f sec", time.perf_counter() - step)

    # Telemetry and Final Return
    total_time = time.perf_counter() - start_total
    logger.info("TOTAL SEARCH TIME %.2f sec", total_time)
    
    if total_time > 5.0:
        logger.warning("WARNING: Slow search detected | Total time: %.2f sec", total_time)

    return format_search_response(query_text, ranked, settings.max_results)

@app.get("/candidate/{candidate_id}", tags=["search"])
def get_candidate(candidate_id: str) -> dict:
    object_id = to_object_id(candidate_id)
    collection = get_candidates_collection()
    try:
        doc = collection.find_one({"_id": object_id})
    except PyMongoError as exc:
        logger.exception("Mongo get_candidate failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Candidate database is temporarily unavailable.",
        ) from exc

    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No candidate found with id '{candidate_id}'.",
        )

    return serialize_mongo_doc(doc)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=bool(os.getenv("RELOAD", "")),
    )
