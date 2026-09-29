"""
PDF Ingestion Pipeline for the Agentic AI RAG Chatbot.

Handles:
    1. Loading the PDF document
    2. Splitting text into meaningful chunks with metadata
    3. Generating OpenAI embeddings
    4. Upserting vectors into Pinecone

Usage:
    python -m src.ingestion
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

from src.config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    PDF_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    validate_config,
)

logger = logging.getLogger(__name__)


def load_pdf(pdf_path: str | None = None) -> list:
    """Load PDF and return a list of LangChain Document objects (one per page)."""
    path = pdf_path or PDF_PATH
    if not Path(path).exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    logger.info("Loading PDF from %s", path)
    loader = PyPDFLoader(path)
    pages = loader.load()
    logger.info("Loaded %d pages from PDF", len(pages))
    return pages


def split_documents(pages: list) -> list:
    """Split page-level documents into smaller chunks with metadata preserved."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(pages)

    # Enrich metadata with chunk index and 1-indexed page number
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        # PyPDFLoader uses 0-indexed 'page'; convert to 1-indexed
        if "page" in chunk.metadata:
            chunk.metadata["page"] = chunk.metadata["page"] + 1

    logger.info("Split into %d chunks (size=%d, overlap=%d)", len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)
    return chunks


def get_embeddings_model() -> OpenAIEmbeddings:
    """Return the configured OpenAI embeddings model."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )


def init_pinecone_index():
    """Initialize the Pinecone client and ensure the target index exists."""
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check if index already exists
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        logger.info("Creating Pinecone index '%s' ...", PINECONE_INDEX_NAME)
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        logger.info("Index '%s' created successfully", PINECONE_INDEX_NAME)
    else:
        logger.info("Pinecone index '%s' already exists", PINECONE_INDEX_NAME)

    return pc.Index(PINECONE_INDEX_NAME)


def upsert_to_pinecone(chunks: list, embeddings_model: OpenAIEmbeddings, index) -> int:
    """
    Generate embeddings for each chunk and upsert into Pinecone.

    Returns the number of vectors upserted.
    """
    texts = [chunk.page_content for chunk in chunks]

    logger.info("Generating embeddings for %d chunks ...", len(texts))
    embeddings = embeddings_model.embed_documents(texts)
    logger.info("Generated %d embeddings (dim=%d)", len(embeddings), len(embeddings[0]))

    # Prepare vectors for upsert: (id, values, metadata)
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        metadata = {
            "text": chunk.page_content,
            "source": chunk.metadata.get("source", PDF_PATH),
            "page": chunk.metadata.get("page", 0),
            "chunk_index": chunk.metadata.get("chunk_index", i),
        }
        vectors.append({
            "id": f"chunk-{i}",
            "values": embedding,
            "metadata": metadata,
        })

    # Upsert in batches of 100
    batch_size = 100
    for start in range(0, len(vectors), batch_size):
        batch = vectors[start : start + batch_size]
        index.upsert(vectors=batch)
        logger.info("Upserted batch %d-%d", start, start + len(batch) - 1)

    logger.info("Total vectors upserted: %d", len(vectors))
    return len(vectors)


def run_ingestion(pdf_path: str | None = None) -> dict:
    """
    Execute the full ingestion pipeline:
        PDF → pages → chunks → embeddings → Pinecone

    Returns a summary dict with pipeline statistics.
    """
    validate_config()

    # Step 1: Load PDF
    pages = load_pdf(pdf_path)

    # Step 2: Split into chunks
    chunks = split_documents(pages)

    # Step 3: Initialize embedding model
    embeddings_model = get_embeddings_model()

    # Step 4: Initialize Pinecone index
    index = init_pinecone_index()

    # Step 5: Upsert embeddings
    num_vectors = upsert_to_pinecone(chunks, embeddings_model, index)

    summary = {
        "pdf_path": pdf_path or PDF_PATH,
        "total_pages": len(pages),
        "total_chunks": len(chunks),
        "vectors_upserted": num_vectors,
        "index_name": PINECONE_INDEX_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }

    logger.info("Ingestion complete: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    result = run_ingestion()
    print("\n✅ Ingestion Pipeline Complete")
    print("-" * 40)
    for key, value in result.items():
        print(f"  {key}: {value}")
