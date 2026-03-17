import os
from pathlib import Path
from dotenv import load_dotenv

for env_path in [Path(".env"), Path("../.env"), Path("../../.env")]:
    if env_path.exists():
        load_dotenv(env_path)
        break

KNOWLEDGE_BASE_PATH = Path(
    os.getenv("KNOWLEDGE_BASE_PATH", "/Users/emilismayilzada/Desktop/Agentic_RAG/knowledge-base")
)

CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", "./chroma_db"))
CHROMA_COLLECTION = "agentic_rag"

AGENT_MODEL = "gpt-4.1"
UTILITY_MODEL = "gpt-4.1-nano"
EMBEDDING_MODEL = "text-embedding-3-large"

AVG_CHUNK_SIZE = 300
RETRIEVAL_K = 15
FINAL_K = 8
MIN_ACCEPTABLE_SCORE = 4
MAX_EVAL_RETRIES = 3
INGEST_WORKERS = 3