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
     (all-MiniLM-L6-v2 -- small, CPU-friendly, Render-compatible).
     Candidate embeddings are NEVER persisted to MongoDB. They are kept in
     a process-local, TTL-based in-memory cache so that a given resume
     isn't re-embedded on every single search request, while nothing is
     written back to the database. Cold start / cache-miss simply means
     "embed it now, keep it in RAM for a while."
  5. Final ranking combines multiple independent, explainable sub-scores
     (role, skill, semantic, experience, achievements, location, notice)
     into one weighted match_score, with human-readable reasons.

This file is intentionally organized into clearly delimited sections so it
can be split into a proper package (config.py, db.py, ai.py, ontology.py,
scoring.py, routers/...) later without changing any logic -- only imports
would need to move.
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

# ---------------------------------------------------------------------------
# SECTION: Configuration
# ---------------------------------------------------------------------------
# All runtime configuration comes from environment variables (.env locally,
# Render's dashboard env vars in production). Nothing is ever hardcoded.

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

        # How many candidates survive Mongo pre-filtering before the
        # (more expensive) semantic scoring stage runs on them.
        self.candidate_pool_limit: int = int(os.getenv("CANDIDATE_POOL_LIMIT", "300"))

        # How many ranked results are returned to the frontend.
        self.max_results: int = int(os.getenv("MAX_RESULTS", "20"))

        # In-memory embedding cache TTL (seconds). Embeddings are never
        # persisted to Mongo -- this is purely a request-to-request,
        # in-process speed optimisation that disappears on restart.
        self.embedding_cache_ttl: int = int(os.getenv("EMBEDDING_CACHE_TTL", "1800"))

        self.cors_origins: list[str] = [
            o.strip()
            for o in os.getenv(
                "CORS_ORIGINS", "https://resumesync.in"
            ).split(",")
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


# ---------------------------------------------------------------------------
# SECTION: MongoDB Client
# ---------------------------------------------------------------------------
# A single, lazily-validated client shared across the app's lifetime.
# We do NOT touch the existing schema in any way -- read-only filtering
# queries plus a single findOne by _id for the detail endpoint.

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


# ---------------------------------------------------------------------------
# SECTION: Gemini Client (Query Understanding ONLY)
# ---------------------------------------------------------------------------
# Gemini's sole responsibility is turning a recruiter's free-text query into
# a structured intent object. It never sees the candidate database and is
# never asked to name or rank candidates -- that would risk hallucination.

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
    """Structured recruiter intent, as extracted by Gemini."""

    role: Optional[str] = None
    experience_min: Optional[int] = None
    location: Optional[str] = None
    skills: list[str] = Field(default_factory=list)
    domain: Optional[str] = None
    concepts: list[str] = Field(default_factory=list)


def extract_query_intent(query: str) -> QueryIntent:
    """
    Calls Gemini in JSON mode to convert a recruiter's free-text query into
    structured intent. Falls back to a conservative heuristic extraction if
    Gemini is unavailable or returns something unparseable, so the search
    pipeline degrades gracefully rather than failing outright.
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
    except Exception as exc:  # noqa: BLE001 -- intentionally broad: this is a
        # best-effort enrichment step, not a hard dependency for search to work.
        logger.warning("Gemini query understanding failed, using fallback: %s", exc)
        return _fallback_query_intent(query)


_EXPERIENCE_PATTERN = re.compile(
    r"(\d+)\s*\+?\s*(?:years?|yrs?)", re.IGNORECASE
)


def _fallback_query_intent(query: str) -> QueryIntent:
    """
    Lightweight, dependency-free heuristic used only if Gemini is down.
    Pulls an experience number via regex and treats capitalised / known
    ontology tokens in the query as candidate skills, so search remains
    functional (degraded, not broken) during a Gemini outage.
    """
    experience_min = None
    match = _EXPERIENCE_PATTERN.search(query)
    if match:
        experience_min = int(match.group(1))

    lowered = query.lower()
    found_skills = [
        term for term in ALL_KNOWN_TERMS if term.lower() in lowered
    ]

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
# A static map of technology clusters used for two purposes:
#   1. Query expansion -- so "Razorpay" pulls in "Payment Gateway", "Stripe",
#      "Webhook", etc., letting a Stripe-only resume still match well.
#   2. Skill-match scoring -- a candidate skill doesn't need to be an exact
#      string match to the query skill; being in the same cluster counts as
#      a (lower-weighted) related match instead of zero.
#
# This is intentionally a plain Python dict so it's trivial to extend, and
# easy to later move into its own ontology.py / a small JSON config file or
# even a Mongo "ontology" collection without touching scoring logic.

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

# Common recruiter role phrasings that don't literally contain an ontology
# key as a substring (e.g. "AI Engineer" doesn't contain "ai/ml"; "Cloud
# Engineer" doesn't contain "cloud" as a clean substring once pluralised
# variants are considered). Mapping these explicitly is more reliable than
# relying purely on substring containment for role-style queries.
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

# A reverse index: term -> set of cluster names it belongs to (as a key or
# as a listed member). Lets us answer "are skill A and skill B related?" in
# O(1)-ish time without scanning the whole ontology repeatedly.
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
    """
    Resolves a free-text term (which may be a multi-word phrase like
    "Frontend Developer" or "Backend Development", not just a bare
    ontology key like "Frontend") to the set of ontology clusters it
    touches.

    Strategy, cheapest-first:
      1. Exact match against the reverse index (fast path for the common
         case of a single clean term like "Razorpay" or "AWS").
      2. Substring containment in either direction against every known
         ontology key/member -- this is what lets "Frontend Developer"
         resolve to the "Frontend" cluster, and "Backend Development"
         resolve to "Backend".
    Results are cached per-process since the ontology is static and query
    vocabulary repeats heavily across recruiters.
    """
    key = term.strip().lower()
    if not key:
        return set()

    exact = _TERM_TO_CLUSTERS.get(key)
    if exact:
        return set(exact)

    matches: set[str] = set()
    for known_term, clusters in _TERM_TO_CLUSTERS.items():
        # Guard against accidental matches on very short tokens (e.g. "Go"
        # inside "Going") by requiring exact equality for short tokens.
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
    Expands a list of skills/concepts/role-words into a richer set of
    related technologies using the ontology above. Deduplicated, original
    casing of newly-added terms preserved from the ontology definition.

    Handles multi-word phrases (e.g. "Frontend Developer", "Cloud
    Engineer") by resolving them to ontology clusters via substring
    matching, not just exact key lookup.

    This expanded list is used internally for semantic scoring context and
    is NEVER returned to the frontend (per the "never expose expansion"
    requirement) -- callers should keep `expanded` separate from the raw
    `skills`/`concepts` that do get echoed back as matched_skills.
    """
    expanded: set[str] = set()
    for term in terms:
        clusters = _cached_clusters_for_term(term)
        for cluster in clusters:
            expanded.add(cluster)
            expanded.update(SKILL_ONTOLOGY.get(cluster, []))
    return sorted(expanded)


def are_related(skill_a: str, skill_b: str) -> bool:
    """True if two skill strings share at least one ontology cluster."""
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
# SECTION: Embedding Engine (in-memory only, never persisted to Mongo)
# ---------------------------------------------------------------------------
# Loaded lazily on first use so `/health` and app startup don't pay the
# model-load cost, and so a Gemini-only deployment that hasn't been hit yet
# starts fast. The model itself stays resident in process memory for the
# life of the worker -- that's normal and fine; what we avoid is writing any
# embedding vectors back into MongoDB documents.

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
    Process-local cache of text -> embedding vector, keyed by a hash of the
    text itself (so if a candidate's search_text changes, it's treated as a
    new cache entry automatically -- no stale-data risk). Entries expire
    after `ttl` seconds. Nothing here ever touches MongoDB; restarting the
    proc