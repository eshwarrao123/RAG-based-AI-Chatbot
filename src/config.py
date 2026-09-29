"""
Configuration module for the Agentic AI RAG Chatbot.

Loads environment variables and defines constants used
across the ingestion pipeline and RAG workflow.
"""

import os
from dotenv import load_dotenv

# Load .env file (if present) into os.environ
load_dotenv()

# ── API Keys ──────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")

# ── Pinecone ──────────────────────────────────────────────
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-rag")
PINECONE_CLOUD: str = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")

# ── OpenAI Models ─────────────────────────────────────────
EMBEDDING_MODEL: str = "text-embedding-3-small"
EMBEDDING_DIMENSIONS: int = 1536
LLM_MODEL: str = "gpt-4o-mini"
LLM_TEMPERATURE: float = 0.0  # deterministic for grounded answers

# ── PDF / Ingestion ──────────────────────────────────────
PDF_PATH: str = os.path.join("data", "Ebook-Agentic-AI.pdf")
CHUNK_SIZE: int = 1000      # characters per chunk
CHUNK_OVERLAP: int = 200    # character overlap between chunks

# ── Retrieval ─────────────────────────────────────────────
TOP_K: int = 4                         # number of chunks to retrieve
CONFIDENCE_THRESHOLD: float = 0.60     # below this, flag low relevance


def validate_config() -> None:
    """Raise early if required environment variables are missing."""
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Copy .env.example to .env and fill in your keys."
        )
