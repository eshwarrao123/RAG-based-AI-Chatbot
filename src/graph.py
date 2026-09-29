"""
LangGraph RAG Workflow for the Agentic AI Chatbot.

Implements a two-node state machine:
    START → retrieve → generate → END

The retrieve node queries Pinecone for relevant chunks.
The generate node produces a grounded answer using the retrieved context.
"""

import logging
from typing import TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, START, END
from pinecone import Pinecone

from src.config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    TOP_K,
    CONFIDENCE_THRESHOLD,
    validate_config,
)

logger = logging.getLogger(__name__)


# ── State Definition ──────────────────────────────────────

class RAGState(TypedDict):
    """Shared state passed through the LangGraph nodes."""
    question: str
    context: list[dict]          # list of {text, source, page, relevance_score}
    retrieval_scores: list[float]
    answer: str
    confidence_score: float


# ── System Prompt ─────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about Agentic AI based ONLY \
on the provided context from the "Agentic AI" eBook.

RULES:
1. Answer ONLY using information found in the CONTEXT below.
2. If the context does not contain sufficient information to answer the question, \
respond with: "I don't have enough information in the eBook to answer this question."
3. Do NOT use any prior knowledge or training data.
4. Cite the page number(s) when the information is available in the metadata.
5. Be concise and accurate.

CONTEXT:
{context}
"""

LOW_RELEVANCE_ADVISORY = """
NOTE: The retrieved context has LOW relevance to this question (confidence: {confidence:.2f}). \
If you cannot find a direct answer in the context, state that clearly.
"""


# ── Helper Functions ──────────────────────────────────────

def compute_confidence(retrieval_scores: list[float]) -> float:
    """
    Compute confidence from Pinecone cosine similarity scores.

    Returns the mean of top-K scores as a relevance indicator.
    Thresholds:
        >= 0.80 : High confidence — context strongly relevant
        0.60-0.79 : Medium confidence — context partially relevant
        < 0.60 : Low confidence — context may not support answer
    """
    if not retrieval_scores:
        return 0.0
    return round(sum(retrieval_scores) / len(retrieval_scores), 4)


def _format_context(context_chunks: list[dict]) -> str:
    """Format context chunks into a single string for the LLM prompt."""
    parts = []
    for i, chunk in enumerate(context_chunks, 1):
        page = chunk.get("page", "?")
        text = chunk.get("text", "")
        parts.append(f"[Chunk {i} | Page {page}]\n{text}")
    return "\n\n".join(parts)


# ── Graph Nodes ───────────────────────────────────────────

def _get_pinecone_index():
    """Return the Pinecone index handle."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)


def _get_embeddings_model() -> OpenAIEmbeddings:
    """Return the configured embeddings model."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )


def _get_llm() -> ChatOpenAI:
    """Return the configured LLM."""
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=OPENAI_API_KEY,
    )


def retrieve(state: RAGState) -> dict:
    """
    Retrieve node: embed the user query and search Pinecone for relevant chunks.

    Updates state with:
        - context: list of retrieved chunk dicts
        - retrieval_scores: list of similarity scores
        - confidence_score: mean similarity score
    """
    question = state["question"]
    logger.info("Retrieving context for: %s", question)

    # Embed the query
    embeddings_model = _get_embeddings_model()
    query_embedding = embeddings_model.embed_query(question)

    # Query Pinecone
    index = _get_pinecone_index()
    results = index.query(
        vector=query_embedding,
        top_k=TOP_K,
        include_metadata=True,
    )

    # Extract context and scores
    context_chunks = []
    scores = []

    for match in results.get("matches", []):
        metadata = match.get("metadata", {})
        score = float(match.get("score", 0.0))

        context_chunks.append({
            "text": metadata.get("text", ""),
            "source": metadata.get("source", ""),
            "page": int(metadata.get("page", 0)),
            "relevance_score": round(score, 4),
        })
        scores.append(score)

    confidence = compute_confidence(scores)
    logger.info(
        "Retrieved %d chunks, confidence=%.4f (scores=%s)",
        len(context_chunks), confidence, [round(s, 4) for s in scores],
    )

    return {
        "context": context_chunks,
        "retrieval_scores": scores,
        "confidence_score": confidence,
    }


def generate(state: RAGState) -> dict:
    """
    Generate node: produce a grounded answer using the retrieved context.

    Uses the system prompt to constrain the LLM to answer only from context.
    Adds a low-relevance advisory when confidence is below threshold.
    """
    question = state["question"]
    context_chunks = state["context"]
    confidence = state["confidence_score"]

    logger.info("Generating answer (confidence=%.4f)", confidence)

    # Build the prompt
    formatted_context = _format_context(context_chunks)
    system_message = SYSTEM_PROMPT.format(context=formatted_context)

    # Add low-relevance advisory if confidence is below threshold
    if confidence < CONFIDENCE_THRESHOLD:
        system_message += LOW_RELEVANCE_ADVISORY.format(confidence=confidence)
        logger.warning("Low relevance detected (%.4f < %.2f)", confidence, CONFIDENCE_THRESHOLD)

    # Call the LLM
    llm = _get_llm()
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": question},
    ]
    response = llm.invoke(messages)
    answer = response.content

    logger.info("Generated answer (%d chars)", len(answer))

    return {
        "answer": answer,
    }


# ── Graph Construction ────────────────────────────────────

def build_rag_graph() -> StateGraph:
    """
    Build and compile the LangGraph RAG workflow.

    Flow: START → retrieve → generate → END
    """
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    # Add edges
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    compiled = graph.compile()
    logger.info("RAG graph compiled: START → retrieve → generate → END")
    return compiled


# ── Convenience Function ──────────────────────────────────

def query_rag(question: str) -> RAGState:
    """
    Run a question through the full RAG pipeline.

    Args:
        question: The user's question string.

    Returns:
        Final RAGState with answer, context, and confidence_score.
    """
    validate_config()

    graph = build_rag_graph()
    initial_state: RAGState = {
        "question": question,
        "context": [],
        "retrieval_scores": [],
        "answer": "",
        "confidence_score": 0.0,
    }

    result = graph.invoke(initial_state)
    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    test_question = "What is Agentic AI?"
    print(f"\n🔍 Query: {test_question}\n")

    result = query_rag(test_question)

    print(f"📊 Confidence: {result['confidence_score']}")
    print(f"📄 Context chunks: {len(result['context'])}")
    print(f"\n💬 Answer:\n{result['answer']}")
