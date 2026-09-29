"""
FastAPI Application for the Agentic AI RAG Chatbot.

Endpoints:
    POST /chat    — Submit a question and receive a grounded answer
    GET  /health  — Health check with Pinecone connectivity status
    POST /ingest  — Trigger PDF ingestion pipeline

Usage:
    uvicorn app:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import validate_config, PINECONE_API_KEY, PINECONE_INDEX_NAME
from src.graph import query_rag
from src.ingestion import run_ingestion

logger = logging.getLogger(__name__)


# ── Pydantic Models ───────────────────────────────────────

class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The question to ask about Agentic AI",
        examples=["What is Agentic AI?"],
    )


class ContextChunk(BaseModel):
    """A single retrieved context chunk with metadata."""
    text: str
    source: str
    page: int
    relevance_score: float


class ChatResponse(BaseModel):
    """Response body for the /chat endpoint."""
    answer: str
    context: list[ContextChunk]
    confidence_score: float


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""
    status: str
    pinecone_connected: bool
    index_name: str


class IngestResponse(BaseModel):
    """Response body for the /ingest endpoint."""
    status: str
    total_pages: int
    total_chunks: int
    vectors_upserted: int
    index_name: str


# ── Application Lifespan ─────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate configuration on startup."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        validate_config()
        logger.info("Configuration validated successfully")
    except EnvironmentError as e:
        logger.error("Configuration error: %s", e)
        raise
    yield


# ── FastAPI App ───────────────────────────────────────────

app = FastAPI(
    title="Agentic AI RAG Chatbot",
    description=(
        "A Retrieval-Augmented Generation chatbot that answers questions "
        "about Agentic AI using the provided eBook as its knowledge base. "
        "Powered by LangGraph, OpenAI, and Pinecone."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Submit a question and receive a grounded answer from the Agentic AI eBook.

    The response includes:
    - The generated answer (grounded in retrieved context)
    - Retrieved context chunks with source/page metadata
    - A confidence score based on retrieval similarity
    """
    try:
        logger.info("Chat request: %s", request.question)
        result = query_rag(request.question)

        return ChatResponse(
            answer=result["answer"],
            context=[ContextChunk(**chunk) for chunk in result["context"]],
            confidence_score=result["confidence_score"],
        )

    except Exception as e:
        logger.exception("Error processing chat request")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.

    Verifies Pinecone connectivity and returns status.
    """
    pinecone_ok = False
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        indexes = [idx.name for idx in pc.list_indexes()]
        pinecone_ok = PINECONE_INDEX_NAME in indexes
    except Exception as e:
        logger.warning("Pinecone health check failed: %s", e)

    return HealthResponse(
        status="healthy" if pinecone_ok else "degraded",
        pinecone_connected=pinecone_ok,
        index_name=PINECONE_INDEX_NAME,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest():
    """
    Trigger the PDF ingestion pipeline.

    Loads the eBook PDF, chunks it, generates embeddings,
    and upserts vectors into Pinecone.
    """
    try:
        logger.info("Starting ingestion pipeline via API")
        result = run_ingestion()

        return IngestResponse(
            status="success",
            total_pages=result["total_pages"],
            total_chunks=result["total_chunks"],
            vectors_upserted=result["vectors_upserted"],
            index_name=result["index_name"],
        )

    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")
