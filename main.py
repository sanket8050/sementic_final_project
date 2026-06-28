"""
ResumeSync AI Search Backend – Production-Grade v2
====================================================

Changelog vs v1
-----------------
*  Mandatory-skill gate    AUTOSAR / Rust / MISRA-C / Kafka etc. require exact
                           match; candidates missing ALL mandatory skills are
                           hard-eliminated before any embedding work.
*  Hybrid retrieval        BM25 lexical layer (rank_bm25) combined with cosine
                           semantic similarity for better keyword precision.
*  Calibrated thresholds   Raw cosine similarity re-scaled so low values (<0.20)
                           contribute nothing.  0.32 raw ≠ 32% relevance.
*  Dynamic weights         Skills dominate (≥45%).  Location / notice weight = 0
                           unless the recruiter explicitly mentioned them.
*  Fuzzy skill matching    rapidfuzz handles misspellings: Fluter→Flutter,
                           Pythn→Python, Ract→React (80–85 % threshold for
                           regular skills, 92 % for mandatory ones).
*  Role stop-word filter   "Engineer", "Developer", "Lead" etc. are filtered out
                           so "Backend Engineer" ≠ "Security Engineer" on the
                           word "Engineer" alone.
*  Confidence gate         If the top-ranked candidate is below NO_RESULT_THRESHOLD
                           the API returns a clear "No suitable candidates" message
                           instead of weak matches that destroy recruiter trust.
*  Smarter Mongo pre-filter  Mandatory skills are pre-filtered at the DB layer
                              using $regex/$or so irrelevant candidates never reach
                              the embedding stage → lower latency.
*  Negative skill support  "No PHP" / "Without Java" excludes matching candidates.
*  Evidence-driven reason  Score breakdown now separates mandatory, exact, fuzzy,
                           and related-skill matches with individual labels.

New pip dependencies (add to requirements.txt):
    rank-bm25>=0.2.2
    rapidfuzz>=3.6.1
"""

from __future__ import annotations

import os
import re
import json
import time
import logging
import hashlib
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from bson import ObjectId
from bson.errors import InvalidId
from rank_bm25 import BM25Okapi                      # NEW: lexical retrieval
from rapidfuzz import fuzz, process as rfprocess     # NEW: fuzzy skill matching


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: Configuration
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("resumesync")


class Settings:
    """Centralised, validated environment configuration."""

    def __init__(self) -> None:
        self.mongo_uri: str | None = os.getenv("MONGO_URI")
        self.database_name: str = os.getenv("DATABASE_NAME", "recruitment_db")
        self.collection_name: str = os.getenv("COLLECTION_NAME", "candidates")
        self.gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
        self.gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

        # Embedding model. Small + fast + good enough for resume-length text.
        self.embedding_model_name: str = os.getenv(
            "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )

        # How many candidates survive Mongo pre-filtering.
        self.candidate_pool_limit: int = int(os.getenv("CANDIDATE_POOL_LIMIT", "300"))

        # How many ranked results are returned to the frontend.
        self.max_results: int = int(os.getenv("MAX_RESULTS", "20"))

        # In-memory embedding cache TTL (seconds).
        self.embedding_cache_ttl: int = int(os.getenv("EMBEDDING_CACHE_TTL", "1800"))

        # Minimum match score (0-100) a candidate must reach to appear in
        # results. Anything below this → "No suitable candidates found."
        self.no_result_threshold: int = int(os.getenv("NO_RESULT_THRESHOLD", "42"))

        self.cors_origins: list[str] = [
            o.strip()
            for o in os.getenv("CORS_ORIGINS", "https://resumesync.in").split(",")
            if o.strip()
        ]

    def validate(self) -> list[str]:
        """Returns a list of human-readable problems with the config."""
        problems = []
        if not self.mongo_uri:
            problems.append("MONGO_URI is not set")
        if not self.gemini_api_key:
            problems.append("GEMINI_API_KEY is not set")
        return problems


settings = Settings()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: MongoDB Client
# ─────────────────────────────────────────────────────────────────────────────

_mongo_client: MongoClient | None = None


def get_mongo_client() -> MongoClient:
    global _mongo_client
    if _mongo_client is None:
        if not settings.mongo_uri:
            raise RuntimeError("MONGO_URI is not configured")
        _mongo_client = MongoClient(
            settings.mongo_uri,
            serverSelectionTimeoutMS=8000,
            connectTimeoutMS=8000,
        )
    return _mongo_client


def get_candidates_collection():
    client = get_mongo_client()
    db = client[settings.database_name]
    return db[settings.collection_name]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: Gemini Client (query understanding ONLY)
# ─────────────────────────────────────────────────────────────────────────────
# Gemini's sole job is converting free-text recruiter queries into structured
# intent. It never sees candidate data and never names or ranks candidates.

_gemini_configured = False


def _ensure_gemini_configured() -> None:
    global _gemini_configured
    if _gemini_configured:
        return
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    import google.generativeai as genai
    genai.configure(api_key=settings.gemini_api_key)
    _gemini_configured = True


GEMINI_SYSTEM_PROMPT = """You are a query-understanding engine for a recruiting \
search system. You convert a recruiter's natural language request into a \
strict JSON object describing their intent.

Rules:
- Output ONLY valid JSON. No markdown fences, no commentary, no preamble.
- Never invent a candidate name. You have no access to any candidate data.
- Never guess at a field you cannot reasonably infer -- use null (or empty \
list) instead of fabricating a value.
- "experience_min" is an integer number of years, or null if unspecified.
- "location" is a city/region name or null.
- "skills" lists explicit technology/tool names mentioned or directly implied \
by the role (e.g. "Flutter developer" implies "Flutter").
- "domain" is a short label: "FinTech", "Healthcare", "E-commerce", "AI/ML", \
"Cloud Infrastructure", "Data Engineering", or null.
- "concepts" lists broader technical concepts implied by the request (e.g. \
"Payment Gateway", "Microservices"), separate from the literal skills list.
- "negative_skills" lists skills explicitly excluded by the recruiter, e.g. \
"No PHP" → ["PHP"], "Without Java" → ["Java"].

Output schema:
{
  "role": string or null,
  "experience_min": integer or null,
  "location": string or null,
  "skills": [string],
  "domain": string or null,
  "concepts": [string],
  "negative_skills": [string]
}
"""


class QueryIntent(BaseModel):
    """Structured recruiter intent, as extracted by Gemini."""
    role: Optional[str] = None
    experience_min: Optional[int] = None
    location: Optional[str] = None
    skills: list[str] = Field(default_factory=list)
    domain: Optional[str] = None
    concepts: list[str] = Field(default_factory=list)
    negative_skills: list[str] = Field(default_factory=list)  # "No PHP", "Without Java"


def extract_query_intent(query: str) -> QueryIntent:
    """
    Calls Gemini in JSON mode to convert a recruiter's free-text query into
    structured intent. Falls back to a conservative heuristic if Gemini is
    unavailable, so the pipeline degrades gracefully rather than failing.
    """
    try:
        _ensure_gemini_configured()
        import google.generativeai as genai
        model = genai.GenerativeModel(
            settings.gemini_model,
            system_instruction=GEMINI_SYSTEM_PROMPT,
            generation_config={"response_mime_type": "application/json"},
        )
        response = model.generate_content(query)
        raw_text = (response.text or "").strip()
        data = json.loads(raw_text)
        return QueryIntent(**data)
    except Exception as exc:   # noqa: BLE001
        logger.warning("Gemini query understanding failed, using fallback: %s", exc)
        return _fallback_query_intent(query)


_EXPERIENCE_PATTERN = re.compile(
    r"(\d+)\s*\+?\s*(?:years?|yrs?)", re.IGNORECASE
)
_NEGATIVE_SKILL_PATTERN = re.compile(
    r"\b(?:no|not|without|exclude|excluding)\s+([A-Za-z0-9#\+\.]+)", re.IGNORECASE
)


def _fallback_query_intent(query: str) -> QueryIntent:
    """
    Lightweight heuristic used only when Gemini is down. Pulls an experience
    number via regex, detects negative skills, and checks known ontology
    terms in the query text.
    """
    experience_min = None
    m = _EXPERIENCE_PATTERN.search(query)
    if m:
        experience_min = int(m.group(1))

    negative_skills = [m.group(1) for m in _NEGATIVE_SKILL_PATTERN.finditer(query)]

    lowered = query.lower()
    found_skills = [term for term in ALL_KNOWN_TERMS if term.lower() in lowered]

    return QueryIntent(
        role=None,
        experience_min=experience_min,
        location=None,
        skills=found_skills,
        domain=None,
        concepts=[],
        negative_skills=negative_skills,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: Mandatory Skill Classification
# ─────────────────────────────────────────────────────────────────────────────
# Skills in this set require EXACT presence in the candidate's profile.
# They are never expanded via the ontology. A candidate that has ZERO of the
# mandatory skills queried is eliminated with score = 0, before any embedding
# computation.

MANDATORY_SKILLS: frozenset[str] = frozenset({
    # ── Automotive / Embedded ────────────────────────────────────────────────
    "autosar", "can bus", "misra c", "misra-c", "freertos", "rtos",
    "plc", "lin bus", "flexray", "arm cortex", "uds", "candb++",
    # ── Systems / Native languages ───────────────────────────────────────────
    "rust", "assembly", "embedded c", "verilog", "vhdl", "fpga",
    # ── EDA / Simulation / Scientific ───────────────────────────────────────
    "matlab", "simulink", "labview", "qt",
    # ── Enterprise / Legacy ──────────────────────────────────────────────────
    "sap", "sap abap", "abap", "cobol", "fortran",
    # ── Big Data / Streaming (rare, employer-specific) ───────────────────────
    "pyspark", "kafka", "snowflake", "airflow",
    # ── GPU / HPC ────────────────────────────────────────────────────────────
    "cuda", "opencl",
    # ── Blockchain ───────────────────────────────────────────────────────────
    "solidity",
})

# Normalised lookup (strip spaces / hyphens) to catch "MISRA-C" == "misra c"
_MANDATORY_NORMALISED: frozenset[str] = frozenset(
    re.sub(r"[\s\-]", "", s) for s in MANDATORY_SKILLS
)


def _is_mandatory(skill: str) -> bool:
    """True if a skill token belongs to the mandatory-exact-match set."""
    s = skill.strip().lower()
    if s in MANDATORY_SKILLS:
        return True
    if re.sub(r"[\s\-]", "", s) in _MANDATORY_NORMALISED:
        return True
    return False


def classify_query_skills(skills: list[str]) -> dict[str, list[str]]:
    """Split a list of queried skills into mandatory vs regular buckets."""
    mandatory: list[str] = []
    regular: list[str] = []
    for skill in skills:
        (mandatory if _is_mandatory(skill) else regular).append(skill)
    return {"mandatory": mandatory, "regular": regular}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: Skill Ontology
# ─────────────────────────────────────────────────────────────────────────────
# A static map of technology clusters used for:
#   1. Query expansion  – "Razorpay" expands to "Payment Gateway", "Stripe", etc.
#   2. Related-skill scoring  – candidate has Stripe when recruiter asked Razorpay.
#
# Mandatory skills (above) are intentionally NOT expanded -- they must match
# exactly, never by proxy. expand_terms() enforces this.

SKILL_ONTOLOGY: dict[str, list[str]] = {
    "Flutter": ["Dart", "Firebase", "Android", "iOS", "Cross Platform", "Material UI", "Mobile Development"],
    "Mobile Development": ["Flutter", "React Native", "Swift", "Kotlin", "Android", "iOS", "Dart"],
    "Frontend": ["React", "Angular", "Vue", "Next.js", "JavaScript", "TypeScript", "HTML", "CSS", "Redux", "Tailwind"],
    "Backend": ["Java", "Python", "Node.js", "Go", "Spring Boot", "FastAPI", "Express", "Django", "Flask", "REST API", "GraphQL", "Microservices", "SQL"],
    "Cloud": ["AWS", "Azure", "GCP", "Terraform", "Docker", "Kubernetes", "CloudFormation", "DevOps", "Linux", "NGINX", "CI/CD"],
    "DevOps": ["Docker", "Kubernetes", "CI/CD", "Terraform", "Jenkins", "GitHub Actions", "Ansible", "Linux", "Monitoring"],
    "Data Engineering": ["Spark", "AWS Glue", "Redshift", "Databricks", "ETL", "SQL", "Python"],
    "AI/ML": ["TensorFlow", "PyTorch", "LLM", "LangChain", "Transformers", "RAG", "Gemini", "OpenAI", "Hugging Face", "Embeddings", "Scikit-learn", "NLP", "Computer Vision"],
    "Payment Gateway": ["Razorpay", "Stripe", "PayPal", "Webhook", "Transaction Processing", "PCI DSS", "FinTech", "API Security"],
    "FinTech": ["Payment Gateway", "Razorpay", "Stripe", "Transaction Processing", "Banking APIs", "Compliance", "KYC"],
    "Database": ["MongoDB", "PostgreSQL", "MySQL", "Redis", "DynamoDB", "Cassandra", "SQL", "NoSQL"],
    "Security": ["OAuth", "JWT", "API Security", "Authentication", "Encryption", "PCI DSS"],
    "Testing": ["PyTest", "JUnit", "Selenium", "Cypress", "Jest", "Unit Testing", "Integration Testing"],
}

# Common role phrasings → ontology cluster
ROLE_ALIASES: dict[str, str] = {
    "ai engineer": "AI/ML",
    "ai developer": "AI/ML",
    "ml engineer": "AI/ML",
    "machine learning engineer": "AI/ML",
    "llm engineer": "AI/ML",
    "data scientist": "AI/ML",
    "cloud engineer": "Cloud",
    "cloud architect": "Cloud",
    "devops engineer": "DevOps",
    "site reliability engineer": "DevOps",
    "sre": "DevOps",
    "data engineer": "Data Engineering",
    "frontend developer": "Frontend",
    "front-end developer": "Frontend",
    "ui developer": "Frontend",
    "backend developer": "Backend",
    "backend engineer": "Backend",
    "full stack developer": "Backend",
    "fullstack developer": "Backend",
    "mobile developer": "Mobile Development",
    "app developer": "Mobile Development",
    "qa engineer": "Testing",
    "test engineer": "Testing",
    "database administrator": "Database",
    "dba": "Database",
    "security engineer": "Security",
}

# Generic tokens that must NOT drive role matching by themselves.
# "Backend Engineer" vs "Security Engineer" share "Engineer" – that must not
# count as a match.
ROLE_STOP_WORDS: frozenset[str] = frozenset({
    "engineer", "developer", "dev", "manager", "lead", "senior", "junior",
    "associate", "principal", "staff", "head", "director", "intern",
    "architect", "specialist", "consultant", "analyst", "professional",
    "expert", "team", "member", "fresher", "trainee", "officer",
})

# Reverse index: term → set of cluster names it belongs to.
_TERM_TO_CLUSTERS: dict[str, set[str]] = {}
for _cluster, _members in SKILL_ONTOLOGY.items():
    _TERM_TO_CLUSTERS.setdefault(_cluster.lower(), set()).add(_cluster)
    for _m in _members:
        _TERM_TO_CLUSTERS.setdefault(_m.lower(), set()).add(_cluster)
for _alias, _cluster in ROLE_ALIASES.items():
    _TERM_TO_CLUSTERS.setdefault(_alias, set()).add(_cluster)

ALL_KNOWN_TERMS: list[str] = sorted(
    {k for k in SKILL_ONTOLOGY.keys()}
    | {member for members in SKILL_ONTOLOGY.values() for member in members}
)


def _clusters_for_term(term: str) -> set[str]:
    """
    Resolve a free-text term to the set of ontology clusters it touches.
    Strategy: exact lookup first, then substring containment.
    """
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
    """
    Expand terms via the skill ontology.

    KEY RULE: Mandatory skills are intentionally NOT used as expansion seeds.
    "Rust" must never expand into generic Backend technologies.
    "Backend" still expands correctly.
    Members of expanded clusters that are themselves mandatory are also
    excluded from the expansion output (to avoid contaminating the semantic
    query text with specialised jargon the candidate may not have).
    """
    expanded: set[str] = set()
    for term in terms:
        if _is_mandatory(term):
            continue  # Mandatory: never expand, require exact
        clusters = _cached_clusters_for_term(term)
        for cluster in clusters:
            expanded.add(cluster)
            for member in SKILL_ONTOLOGY.get(cluster, []):
                if not _is_mandatory(member):
                    expanded.add(member)
    return sorted(expanded)


def are_related(skill_a: str, skill_b: str) -> bool:
    """
    True if two skills share at least one ontology cluster.
    Mandatory skills are NEVER considered "related" to anything –
    they demand exact presence, not approximate similarity.
    """
    if _is_mandatory(skill_a) or _is_mandatory(skill_b):
        return False
    a, b = skill_a.strip().lower(), skill_b.strip().lower()
    if a == b:
        return True
    return bool(_cached_clusters_for_term(a) & _cached_clusters_for_term(b))


def meaningful_role_tokens(role: str) -> set[str]:
    """Strip stop-words from a role string, keeping only meaningful tokens."""
    return {t for t in role.lower().split() if t not in ROLE_STOP_WORDS and len(t) > 1}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: Pydantic Models (API contract – backward-compatible additions only)
# ─────────────────────────────────────────────────────────────────────────────

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
    message: Optional[str] = None   # Set when no suitable candidates found


class HealthResponse(BaseModel):
    status: str
    mongo_connected: bool
    gemini_configured: bool


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: Embedding Engine (in-memory only, never persisted)
# ─────────────────────────────────────────────────────────────────────────────

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", settings.embedding_model_name)
        _embedding_model = SentenceTransformer(settings.embedding_model_name)
    return _embedding_model


class _TTLEmbeddingCache:
    """
    Process-local cache of text → embedding vector, keyed by SHA-256 of the
    text. Entries expire after `ttl` seconds. Nothing here ever touches MongoDB.
    """

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
        """Batch-embed cache misses only; return full matrix in input order."""
        results: list[np.ndarray | None] = [self.get(t) for t in texts]
        miss_indices = [i for i, v in enumerate(results) if v is None]
        if miss_indices:
            model = get_embedding_model()
            miss_texts = [texts[i] for i in miss_indices]
            computed = model.encode(miss_texts, show_progress_bar=False)
            for idx, vec in zip(miss_indices, computed):
                vec = np.asarray(vec, dtype=np.float32)
                self.set(texts[idx], vec)
                results[idx] = vec
        return np.vstack(results)  # type: ignore[arg-type]


_embedding_cache = _TTLEmbeddingCache(ttl=settings.embedding_cache_ttl)


def cosine_similarity_batch(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector and N candidate vectors."""
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return matrix_norms @ query_norm


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: BM25 Lexical Engine
# ─────────────────────────────────────────────────────────────────────────────
# BM25 provides keyword-level recall that semantic embeddings miss.
# It rewards candidates whose search_text contains the exact tokens the
# recruiter typed ("AUTOSAR", "CAN Bus") rather than semantically similar
# but unrelated text.

_TOKEN_RE = re.compile(r"\b[a-z][a-z0-9#\+\.\-]{1,}\b")  # pre-compiled once


def _tokenize(text: str) -> list[str]:
    """Lower-case tokeniser that preserves technical tokens: C++, .NET, Node.js."""
    return _TOKEN_RE.findall(text.lower())


def build_bm25_index(candidate_docs: list[dict]) -> BM25Okapi:
    """Build a BM25 index from the current candidate pool."""
    corpus = []
    for doc in candidate_docs:
        text = doc.get("search_text", "") or ""
        tokens = _tokenize(text) or ["__empty__"]
        corpus.append(tokens)
    return BM25Okapi(corpus)


def build_bm25_query_tokens(intent: QueryIntent, raw_query: str) -> list[str]:
    """Produce a de-duplicated token list for the BM25 query."""
    parts: list[str] = []
    if intent.role:
        parts.extend(intent.role.lower().split())
    parts.extend(s.lower() for s in intent.skills)
    parts.extend(c.lower() for c in intent.concepts)
    if intent.domain:
        parts.extend(intent.domain.lower().split())
    parts.extend(_tokenize(raw_query))

    seen: set[str] = set()
    result: list[str] = []
    for t in parts:
        if t not in seen and len(t) > 1:
            seen.add(t)
            result.append(t)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: Query Expansion
# ─────────────────────────────────────────────────────────────────────────────

class ExpandedQuery(BaseModel):
    intent: QueryIntent
    expanded_terms: list[str]
    semantic_text: str
    raw_query: str   # Preserved for BM25 tokenisation


def build_expanded_query(intent: QueryIntent, raw_query: str) -> ExpandedQuery:
    """
    Combine the recruiter's literal skills/concepts/role with ontology
    expansion into a rich semantic query string.
    Mandatory skills are included as-is in the semantic text (so the embedding
    captures them) but are NOT used as expansion seeds.
    """
    classified = classify_query_skills(intent.skills + intent.concepts)
    expanded = expand_terms(classified["regular"])  # mandatory skills skipped inside

    parts = [raw_query]
    if intent.role:
        parts.append(f"Role: {intent.role}")
    if intent.domain:
        parts.append(f"Domain: {intent.domain}")
    if intent.skills:
        parts.append("Skills: " + ", ".join(intent.skills))
    if intent.concepts:
        parts.append("Concepts: " + ", ".join(intent.concepts))
    if expanded:
        parts.append("Related: " + ", ".join(expanded))

    return ExpandedQuery(
        intent=intent,
        expanded_terms=expanded,
        semantic_text=" | ".join(parts),
        raw_query=raw_query,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: Dynamic Weight Engine
# ─────────────────────────────────────────────────────────────────────────────

def compute_dynamic_weights(intent: QueryIntent) -> dict[str, float]:
    """
    Return normalised scoring weights that reflect what the recruiter specified.

    Rules:
    * Skills always dominate (base ≥ 45 %).
    * Role weight drops when no role was mentioned (recruiter listed skills only).
    * Experience weight drops when no minimum was stated.
    * Location weight is 0 unless the recruiter mentioned a location; budget
      is redistributed to skill + semantic proportionally.
    * Notice period is always 0 (not in the current QueryIntent schema).
    * All weights are normalised so they sum exactly to 1.0.
    """
    raw: dict[str, float] = {
        "skill":        0.45,
        "role":         0.15,
        "semantic":     0.18,
        "bm25":         0.12,
        "experience":   0.10,
        "achievements": 0.00,   # additive bonus only; normally 0 base weight
        "location":     0.00,   # 0 unless recruiter specified
        "notice":       0.00,   # always 0 (schema doesn't carry this)
    }

    # No role mentioned → redistribute role weight to skill + semantic
    if not intent.role:
        raw["skill"]    += raw["role"] * 0.70
        raw["semantic"] += raw["role"] * 0.30
        raw["role"]      = 0.02   # tiny residual to credit perfect role hits

    # No experience requirement → redistribute to skill + semantic
    if intent.experience_min is None:
        raw["skill"]    += raw["experience"] * 0.60
        raw["semantic"] += raw["experience"] * 0.40
        raw["experience"] = 0.0

    # Location requested → carve 8 % proportionally from skill + semantic
    if intent.location:
        loc_budget   = 0.08
        donor_total  = raw["skill"] + raw["semantic"]
        raw["skill"]    -= loc_budget * (raw["skill"]    / (donor_total + 1e-9))
        raw["semantic"] -= loc_budget * (raw["semantic"] / (donor_total + 1e-9))
        raw["location"]  = loc_budget

    # Normalise to sum to 1.0
    total = sum(v for v in raw.values())
    return {k: (v / total if total > 0 else 0.0) for k, v in raw.items()}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: MongoDB Filtering
# ─────────────────────────────────────────────────────────────────────────────

def build_mongo_filter(intent: QueryIntent, mandatory_skills: list[str]) -> dict:
    """
    Deliberately loose pre-filter: shrink the collection cheaply before scoring.

    Mandatory skills are OR-filtered at the MongoDB layer so candidates that
    don't mention ANY of the required specialised skills are never fetched —
    this provides a fast early exit (e.g. "AUTOSAR engineer" when nobody in the
    DB has AUTOSAR in their search_text) without expensive embedding work.
    """
    conditions: list[dict] = []

    if intent.location:
        conditions.append({
            "output.candidate.location": {
                "$regex": re.escape(intent.location),
                "$options": "i",
            }
        })

    if intent.experience_min:
        # Allow 1-year slack below the stated minimum (avoid over-filtering).
        conditions.append(
            {"exp_years_num": {"$gte": max(intent.experience_min - 1, 0)}}
        )

    # Mandatory skill pre-filter: candidate must mention at least ONE of the
    # mandatory skills in their search_text (word-boundary match).
    if mandatory_skills:
        mandatory_or = [
            {"search_text": {"$regex": r"(?i)\b" + re.escape(s) + r"\b"}}
            for s in mandatory_skills
        ]
        conditions.append({"$or": mandatory_or})

    return {"$and": conditions} if conditions else {}


def fetch_candidate_pool(mongo_filter: dict, limit: int) -> list[dict]:
    collection = get_candidates_collection()
    projection = {
        "output.candidate": 1,
        "output.summary": 1,
        "output.fit_score": 1,
        "exp_years_num": 1,
        "search_text": 1,
    }
    try:
        cursor = collection.find(mongo_filter, projection).limit(limit)
        return list(cursor)
    except PyMongoError as exc:
        logger.error("Mongo query failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Candidate database is temporarily unavailable.",
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: Sub-Score Functions
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_semantic(raw: float) -> float:
    """
    Re-scale raw cosine similarity to a meaningful [0, 1] score.

    Raw cosine similarity is NOT a probability. Two unrelated professional
    texts routinely score 0.25–0.40 purely due to shared vocabulary.
    This piecewise linear map converts raw sim so that low values correctly
    contribute near-zero, not a misleadingly high "32%".

    Breakpoints (tuned for sentence-transformers all-MiniLM-L6-v2):
      < 0.20   →  0.00  (noise / completely unrelated domains)
      0.20–0.40  →  0.00–0.15  (weak, mostly vocabulary overlap)
      0.40–0.55  →  0.15–0.50  (some domain alignment)
      0.55–0.70  →  0.50–0.85  (clear domain match)
      0.70+    →  0.85–1.00  (strong match)
    """
    if raw < 0.20:
        return 0.0
    elif raw < 0.40:
        return (raw - 0.20) / 0.20 * 0.15
    elif raw < 0.55:
        return 0.15 + (raw - 0.40) / 0.15 * 0.35
    elif raw < 0.70:
        return 0.50 + (raw - 0.55) / 0.15 * 0.35
    else:
        return min(1.0, 0.85 + (raw - 0.70) / 0.30 * 0.15)


def score_mandatory_skills(
    query_mandatory: list[str],
    candidate_skills_lower: dict[str, str],   # {lowercase: original}
) -> tuple[float, list[str], list[str]]:
    """
    Mandatory skills require EXACT match (or very-high-confidence fuzzy ≥92).
    Returns (score 0–1, matched, missed).
    score == 0.0  means the candidate must be hard-eliminated.
    """
    if not query_mandatory:
        return 1.0, [], []

    candidate_lower_list = list(candidate_skills_lower.keys())
    matched: list[str] = []
    missed: list[str] = []

    for ms in query_mandatory:
        ms_l = ms.lower()
        if ms_l in candidate_skills_lower:
            matched.append(candidate_skills_lower[ms_l])
            continue
        # Very strict fuzzy for mandatory (≥92 token ratio)
        hit = rfprocess.extractOne(ms_l, candidate_lower_list, scorer=fuzz.ratio)
        if hit and hit[1] >= 92:
            matched.append(candidate_skills_lower[hit[0]])
        else:
            missed.append(ms)

    score = len(matched) / len(query_mandatory)
    return score, matched, missed


def score_regular_skills(
    query_regular: list[str],
    candidate_skills_lower: dict[str, str],
) -> tuple[float, list[str], list[str], list[str]]:
    """
    Regular (non-mandatory) skills: exact > fuzzy > ontology-related.
    Returns (score 0–1, exact_matched, fuzzy_matched, related_matched).

    Weight per skill:
      Exact    1.00
      Fuzzy    0.80  (handles misspellings like "Fluter", "Pythn", "Ract")
      Related  0.45  (ontology link, e.g. Stripe when Razorpay was asked)
      None     0.00
    """
    if not query_regular:
        return 0.5, [], [], []   # neutral when no regular skills were queried

    candidate_lower_list = list(candidate_skills_lower.keys())
    exact_matched: list[str] = []
    fuzzy_matched: list[str] = []
    related_matched: list[str] = []
    total_weight = 0.0

    for qs in query_regular:
        qs_l = qs.lower()
        if qs_l in candidate_skills_lower:
            exact_matched.append(candidate_skills_lower[qs_l])
            total_weight += 1.0
        else:
            hit = rfprocess.extractOne(
                qs_l, candidate_lower_list, scorer=fuzz.token_sort_ratio
            )
            if hit and hit[1] >= 82:
                fuzzy_matched.append(candidate_skills_lower[hit[0]])
                total_weight += 0.80
            else:
                # Ontology-related match (e.g. Stripe ↔ Razorpay)
                for cl, c_orig in candidate_skills_lower.items():
                    if are_related(qs, cl):
                        related_matched.append(c_orig)
                        total_weight += 0.45
                        break   # take only the best related match per query skill

    score = min(1.0, total_weight / len(query_regular))
    return score, exact_matched, fuzzy_matched, related_matched


def score_role_match(intent: QueryIntent, current_role: str) -> tuple[float, list[str]]:
    """
    Compare the queried role against the candidate's current role.
    Generic stop-words ("Engineer", "Developer") are stripped before
    token overlap so "Backend Engineer" ≠ "Security Engineer" on
    the word "Engineer" alone.
    """
    if not intent.role or not current_role:
        return 0.5, []

    role_q = intent.role.lower()
    role_c = current_role.lower()

    # Full substring containment (fast path)
    if role_q in role_c or role_c in role_q:
        return 1.0, [f"Current role '{current_role}' directly matches requested role"]

    # Meaningful token overlap (stop-words removed)
    q_tokens = meaningful_role_tokens(role_q)
    c_tokens = meaningful_role_tokens(role_c)
    overlap = q_tokens & c_tokens

    if overlap:
        score = min(1.0, 0.50 + 0.25 * len(overlap))
        return score, [f"Role overlap — {', '.join(sorted(overlap))} in '{current_role}'"]

    # Same ontology cluster
    if _cached_clusters_for_term(role_q) & _cached_clusters_for_term(role_c):
        return 0.45, [f"Role in same domain as '{current_role}'"]

    return 0.10, []


def score_experience_match(
    required_min: Optional[int], candidate_years: Optional[float]
) -> tuple[float, list[str]]:
    if required_min is None:
        return 0.7, []
    if candidate_years is None:
        return 0.3, []
    if candidate_years >= required_min:
        label = (
            "Experience exceeds requirement"
            if candidate_years > required_min
            else "Meets required experience"
        )
        return 1.0, [label]
    ratio = candidate_years / required_min if required_min > 0 else 1.0
    return max(0.0, ratio * 0.8), []


def score_achievements_match(
    query_terms: list[str], achievements: list[str]
) -> tuple[float, list[str]]:
    if not achievements or not query_terms:
        return 0.0, []
    text = " ".join(str(a) for a in achievements).lower()
    hits = [t for t in query_terms if t.lower() in text]
    if not hits:
        return 0.0, []
    score = min(1.0, 0.30 + 0.15 * len(hits))
    return score, [f"Key achievement involving {hits[0]}"]


def score_location_match(
    required_location: Optional[str], candidate_location: Optional[str]
) -> tuple[float, list[str]]:
    if not required_location:
        return 1.0, []   # no preference → don't penalise anyone
    if not candidate_location:
        return 0.3, []
    if required_location.lower() in candidate_location.lower():
        return 1.0, [f"Located in {candidate_location}"]
    return 0.1, []


def score_notice_match(notice_period: Optional[str]) -> tuple[float, list[str]]:
    if not notice_period:
        return 0.5, []
    digits = re.findall(r"\d+", str(notice_period))
    if not digits:
        return 0.5, []
    days = int(digits[0])
    if days <= 15:
        return 1.0, ["Short notice period"]
    if days <= 30:
        return 0.8, []
    if days <= 60:
        return 0.6, []
    return 0.4, []


def _has_negative_skill(
    candidate_skills: list[str], negative_skills: list[str]
) -> bool:
    """Return True if the candidate should be excluded via negative-skill filter."""
    if not negative_skills:
        return False
    c_lower = {s.lower() for s in candidate_skills}
    return any(ns.lower() in c_lower for ns in negative_skills)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: Master Score Calculation
# ─────────────────────────────────────────────────────────────────────────────

def calculate_match_score(
    expanded_query: ExpandedQuery,
    candidate_doc: dict,
    semantic_similarity: float,
    bm25_score: float,
    weights: dict[str, float],
) -> tuple[int, dict[str, int], list[str], list[str], str, bool]:
    """
    Compute the full match score for one candidate.

    Returns
    -------
    (final_score_pct, score_breakdown_pct, matched_skills,
     matched_concepts, reason_text, should_eliminate)

    should_eliminate is True when:
      * Mandatory skill gate fails (candidate has NONE of the required skills).
      * Negative skill filter fires ("No PHP" but candidate has PHP).
    Callers must discard candidates where should_eliminate is True.
    """
    output         = candidate_doc.get("output", {}) or {}
    candidate_info = output.get("candidate", {}) or {}
    summary        = output.get("summary", {}) or {}

    intent             = expanded_query.intent
    candidate_skills   = summary.get("technical_skills", []) or []
    achievements       = summary.get("key_achievements", []) or []
    current_role       = summary.get("current_role", "") or ""
    candidate_location = candidate_info.get("location")
    notice_period      = candidate_info.get("notice_period")
    candidate_years    = candidate_doc.get("exp_years_num")

    # ── Negative skill filter ────────────────────────────────────────────────
    if _has_negative_skill(candidate_skills, intent.negative_skills):
        return 0, {}, [], [], "Excluded: contains explicitly unwanted technology.", True

    # Fast lowercase lookup for candidate skills
    candidate_skills_lower: dict[str, str] = {s.lower(): s for s in candidate_skills}

    # ── Classify query skills ────────────────────────────────────────────────
    all_query_skills = list(dict.fromkeys(intent.skills + intent.concepts))
    classified       = classify_query_skills(all_query_skills)
    query_mandatory  = classified["mandatory"]
    query_regular    = classified["regular"]

    # ── Mandatory skill gate ─────────────────────────────────────────────────
    mandatory_score, mandatory_matched, mandatory_missed = score_mandatory_skills(
        query_mandatory, candidate_skills_lower
    )
    if query_mandatory and mandatory_score == 0.0:
        missing_str = ", ".join(mandatory_missed[:3])
        return (
            0, {}, [], [],
            f"Eliminated: missing mandatory skill(s) — {missing_str}.",
            True
        )

    # ── Regular skill scoring ────────────────────────────────────────────────
    regular_score, exact_matched, fuzzy_matched, related_matched = score_regular_skills(
        query_regular, candidate_skills_lower
    )

    # Combined skill score: mandatory skills dominate when both are present
    if query_mandatory and query_regular:
        skill_score = 0.60 * mandatory_score + 0.40 * regular_score
    elif query_mandatory:
        skill_score = mandatory_score
    else:
        skill_score = regular_score

    # ── Role ─────────────────────────────────────────────────────────────────
    role_score, role_reasons = score_role_match(intent, current_role)

    # ── Semantic (calibrated) ────────────────────────────────────────────────
    calibrated_sem    = calibrate_semantic(semantic_similarity)
    semantic_reasons: list[str] = []
    if calibrated_sem >= 0.70:
        semantic_reasons.append("Strong semantic alignment with role description")
    elif calibrated_sem >= 0.40:
        semantic_reasons.append("Moderate semantic alignment")

    # ── Experience ───────────────────────────────────────────────────────────
    experience_score, experience_reasons = score_experience_match(
        intent.experience_min, candidate_years
    )

    # ── Achievements ─────────────────────────────────────────────────────────
    all_query_terms = list({*intent.skills, *intent.concepts, *expanded_query.expanded_terms})
    achievement_score, achievement_reasons = score_achievements_match(
        all_query_terms, achievements
    )

    # ── Location ─────────────────────────────────────────────────────────────
    location_score, location_reasons = score_location_match(
        intent.location, candidate_location
    )

    # ── Notice ───────────────────────────────────────────────────────────────
    notice_score, notice_reasons = score_notice_match(notice_period)

    # ── Weighted final score ──────────────────────────────────────────────────
    sub_scores: dict[str, float] = {
        "skill":        skill_score,
        "role":         role_score,
        "semantic":     calibrated_sem,
        "bm25":         bm25_score,
        "experience":   experience_score,
        "achievements": achievement_score,
        "location":     location_score,
        "notice":       notice_score,
    }
    final_score     = sum(sub_scores[k] * weights.get(k, 0.0) for k in sub_scores)
    final_score_pct = round(final_score * 100)
    breakdown_pct   = {k: round(v * 100) for k, v in sub_scores.items()}

    # ── Human-readable reason ─────────────────────────────────────────────────
    reason_parts: list[str] = []
    if mandatory_matched:
        reason_parts.append(
            f"{len(mandatory_matched)} mandatory skill(s) matched: "
            f"{', '.join(mandatory_matched[:3])}"
        )
    if exact_matched:
        reason_parts.append(f"Exact skill match: {', '.join(exact_matched[:4])}")
    if fuzzy_matched:
        reason_parts.append(f"Close skill match: {', '.join(fuzzy_matched[:3])}")
    if related_matched:
        reason_parts.append(f"Related technologies: {', '.join(related_matched[:3])}")
    reason_parts.extend(role_reasons)
    reason_parts.extend(semantic_reasons)
    reason_parts.extend(experience_reasons)
    reason_parts.extend(achievement_reasons)
    reason_parts.extend(location_reasons)
    if not reason_parts:
        reason_parts.append("General profile alignment with the search criteria")
    reason_text = ". ".join(reason_parts) + "."

    # ── Matched skills / concepts for UI ─────────────────────────────────────
    all_matched = list(dict.fromkeys(
        mandatory_matched + exact_matched + fuzzy_matched + related_matched
    ))
    matched_concepts = [
        c for c in intent.concepts
        if c.lower() in " ".join(candidate_skills + achievements).lower()
    ] or related_matched[:3]

    return (
        final_score_pct,
        breakdown_pct,
        all_matched[:8],
        matched_concepts[:5],
        reason_text,
        False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: Hybrid Ranking (semantic + BM25 + skill scoring)
# ─────────────────────────────────────────────────────────────────────────────

def rank_candidates(
    expanded_query: ExpandedQuery,
    candidate_docs: list[dict],
) -> list[CandidateResult]:
    """
    Orchestrate hybrid retrieval (semantic + BM25) and multi-factor scoring
    over the candidate pool. Hard-eliminated candidates are excluded before
    building the result list.
    """
    if not candidate_docs:
        return []

    # ── Semantic embeddings ───────────────────────────────────────────────────
    model        = get_embedding_model()
    query_vector = np.asarray(
        model.encode([expanded_query.semantic_text], show_progress_bar=False)[0],
        dtype=np.float32,
    )
    search_texts     = [doc.get("search_text", "") or "" for doc in candidate_docs]
    candidate_matrix = _embedding_cache.get_or_compute_many(search_texts)
    semantic_sims    = cosine_similarity_batch(query_vector, candidate_matrix)

    # ── BM25 lexical scores ───────────────────────────────────────────────────
    bm25_index        = build_bm25_index(candidate_docs)
    bm25_query_tokens = build_bm25_query_tokens(
        expanded_query.intent, expanded_query.raw_query
    )
    bm25_raw       = bm25_index.get_scores(bm25_query_tokens)
    bm25_max       = bm25_raw.max() if bm25_raw.max() > 0 else 1.0
    bm25_normalised = bm25_raw / bm25_max   # → [0, 1]

    # ── Dynamic weights ───────────────────────────────────────────────────────
    weights = compute_dynamic_weights(expanded_query.intent)

    # ── Per-candidate scoring ─────────────────────────────────────────────────
    results: list[CandidateResult] = []
    for doc, sem_sim, bm25_score in zip(candidate_docs, semantic_sims, bm25_normalised):
        (
            score_pct, breakdown, matched_skills,
            matched_concepts, reason, should_eliminate
        ) = calculate_match_score(
            expanded_query, doc, float(sem_sim), float(bm25_score), weights
        )

        if should_eliminate:
            continue   # Mandatory gate or negative-skill filter fired

        candidate_info = (doc.get("output", {}) or {}).get("candidate", {}) or {}
        first_name = candidate_info.get("first_name", "") or ""
        last_name  = candidate_info.get("last_name",  "") or ""
        full_name  = f"{first_name} {last_name}".strip() or "Unknown Candidate"

        results.append(
            CandidateResult(
                id=str(doc.get("_id")),
                name=full_name,
                experience=doc.get("exp_years_num"),
                location=candidate_info.get("location"),
                match_score=score_pct,
                matched_skills=matched_skills,
                matched_concepts=matched_concepts,
                reason=reason,
                score_breakdown=breakdown,
            )
        )

    results.sort(key=lambda r: r.match_score, reverse=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: Response Formatter (confidence gate)
# ─────────────────────────────────────────────────────────────────────────────

def format_search_response(
    query: str,
    ranked: list[CandidateResult],
    max_results: int,
    mandatory_skills_queried: list[str],
) -> SearchResponse:
    """
    Apply the confidence gate: only return candidates that clear
    settings.no_result_threshold.  If none qualify, return an empty result
    with a descriptive message rather than silently serving weak matches.
    """
    qualified = [r for r in ranked if r.match_score >= settings.no_result_threshold]
    top = qualified[:max_results]

    if not top:
        if mandatory_skills_queried:
            msg = (
                f"No suitable candidates found. This search requires specialised "
                f"skill(s) — {', '.join(mandatory_skills_queried)} — that are not "
                f"present in the current candidate pool."
            )
        else:
            msg = (
                "No suitable candidates found for this search. "
                "The skills or role requested do not closely match any current profiles."
            )
        return SearchResponse(query=query, result_count=0, results=[], message=msg)

    return SearchResponse(query=query, result_count=len(top), results=top)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: Utilities
# ─────────────────────────────────────────────────────────────────────────────

def to_object_id(raw_id: str) -> ObjectId:
    try:
        return ObjectId(raw_id)
    except (InvalidId, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{raw_id}' is not a valid candidate id.",
        ) from exc


def serialize_mongo_doc(doc: dict) -> dict:
    """Recursively convert ObjectId / non-JSON-native values for the API response."""
    if isinstance(doc, dict):
        return {k: serialize_mongo_doc(v) for k, v in doc.items()}
    if isinstance(doc, list):
        return [serialize_mongo_doc(v) for v in doc]
    if isinstance(doc, ObjectId):
        return str(doc)
    return doc


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: FastAPI Application
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ResumeSync AI Search API",
    description="AI-powered natural language candidate search backend.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    problems = settings.validate()
    if problems:
        for p in problems:
            logger.warning("Configuration issue: %s", p)
    else:
        logger.info(
            "ResumeSync v2 ready. NO_RESULT_THRESHOLD=%d mandatory_skills=%d",
            settings.no_result_threshold,
            len(MANDATORY_SKILLS),
        )


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    mongo_ok = False
    try:
        get_mongo_client().admin.command("ping")
        mongo_ok = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Health check: Mongo not reachable: %s", exc)
    return HealthResponse(
        status="ok" if mongo_ok else "degraded",
        mongo_connected=mongo_ok,
        gemini_configured=bool(settings.gemini_api_key),
    )


@app.post("/search", response_model=SearchResponse, tags=["search"])
def search_candidates(request: SearchRequest) -> SearchResponse:
    query_text = request.query.strip()
    if not query_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must not be empty.",
        )

    # Step 1: structured intent extraction (Gemini → fallback heuristic)
    intent = extract_query_intent(query_text)

    # Step 2: classify mandatory vs regular skills early (drives filter + gate)
    all_query_skills = list(dict.fromkeys(intent.skills + intent.concepts))
    classified       = classify_query_skills(all_query_skills)
    mandatory_skills = classified["mandatory"]

    logger.info(
        "Query '%s' → role=%s  skills=%s  mandatory=%s  negative=%s",
        query_text[:60], intent.role, intent.skills,
        mandatory_skills, intent.negative_skills,
    )

    # Step 3: ontology expansion (mandatory skills are skipped inside expand_terms)
    expanded_query = build_expanded_query(intent, query_text)

    # Step 4: Mongo pre-filter (mandatory skill OR filter + location/exp filter)
    mongo_filter   = build_mongo_filter(intent, mandatory_skills)
    candidate_pool = fetch_candidate_pool(mongo_filter, settings.candidate_pool_limit)

    # Retry with relaxed filter if nothing came back.
    # If there are mandatory skills, keep that constraint; only drop
    # location / experience filters.
    if not candidate_pool and mongo_filter:
        logger.info("No results with full filter; retrying with relaxed filter.")
        relaxed_intent    = QueryIntent()            # empty → no loc/exp filters
        relaxed_filter    = build_mongo_filter(relaxed_intent, mandatory_skills)
        candidate_pool    = fetch_candidate_pool(relaxed_filter, settings.candidate_pool_limit)

    if not candidate_pool:
        msg = (
            f"No candidates with required skill(s) ({', '.join(mandatory_skills)}) in the pool."
            if mandatory_skills
            else "No candidates found in the current pool."
        )
        return SearchResponse(query=query_text, result_count=0, results=[], message=msg)

    # Steps 5-6: hybrid ranking (BM25 + semantic embedding + skill scoring)
    try:
        ranked = rank_candidates(expanded_query, candidate_pool)
    except Exception as exc:  # noqa: BLE001
        logger.error("Ranking failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to rank candidates.",
        ) from exc

    # Step 7: confidence gate → return best results or "no suitable candidates"
    return format_search_response(
        query_text, ranked, settings.max_results, mandatory_skills
    )


@app.get("/candidate/{candidate_id}", tags=["search"])
def get_candidate(candidate_id: str) -> dict:
    object_id  = to_object_id(candidate_id)
    collection = get_candidates_collection()
    try:
        doc = collection.find_one({"_id": object_id})
    except PyMongoError as exc:
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
