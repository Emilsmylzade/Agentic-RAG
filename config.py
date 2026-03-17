"""
config.py — All constants and paths in one place.
Change these to match your setup.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load environment variables ──────────────────────────────────────
# Looks for .env in common locations
for env_path in [Path(".env"), Path("../.env"), Path("../../.env")]:
    if env_path.exists():
        load_dotenv(env_path)
        break

# ── Paths ───────────────────────────────────────────────────────────
# Point this to your InsurElm knowledge base folder from the course.
# Typical location: llm_engineering/week5/knowledge-base
KNOWLEDGE_BASE_PATH = Path(
    os.getenv("KNOWLEDGE_BASE_PATH", "/Users/emilismayilzada/Desktop/Agentic_RAG/knowledge-base")
)

# Where ChromaDB stores its data on disk
CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", "./chroma_db"))

# ChromaDB collection name
CHROMA_COLLECTION = "agentic_rag"

# ── Models ──────────────────────────────────────────────────────────
# The main agent model (the "brain" that decides what tools to call)
AGENT_MODEL = "gpt-4.1"

# Cheaper model for chunking, reranking, evaluation
UTILITY_MODEL = "gpt-4.1-nano"

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-large"

# ── Chunking ────────────────────────────────────────────────────────
# Rough target for average chunk size (in characters).
# The LLM decides the actual splits, but we give it a hint.
AVG_CHUNK_SIZE = 300

# ── Retrieval ───────────────────────────────────────────────────────
# How many chunks to retrieve per vector search
RETRIEVAL_K = 15

# After reranking, keep only this many
FINAL_K = 8

# ── Self-evaluation ─────────────────────────────────────────────────
# Minimum score (out of 5) to accept an answer without retrying
MIN_ACCEPTABLE_SCORE = 4

# Max times the agent can retry after a bad self-evaluation
MAX_EVAL_RETRIES = 3

# ── Parallel processing ────────────────────────────────────────────
# Number of parallel workers for ingest (calling the LLM for chunking)
INGEST_WORKERS = 3