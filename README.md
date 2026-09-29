# Agentic AI RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about Agentic AI using a provided eBook as its knowledge base. Built with **LangGraph** for workflow orchestration, **OpenAI** for embeddings and generation, **Pinecone** for vector storage, and **FastAPI** for the API layer.

---

## Architecture

```
PDF (Ebook-Agentic-AI.pdf)
  ↓
Document Loader (pypdf)
  ↓
Text Chunks + Metadata (RecursiveCharacterTextSplitter)
  ↓
OpenAI Embeddings (text-embedding-3-small)
  ↓
Pinecone Vector Store (cosine similarity)
  ↓
Retrieval (similarity search with scores)
  ↓
LangGraph (START → retrieve → generate → END)
  ↓
Grounded Generation (gpt-4o-mini)
  ↓
FastAPI /chat
  ↓
Structured Response (answer + context + confidence)
```

### LangGraph Workflow

The RAG pipeline uses a two-node LangGraph `StateGraph`:

| Node | Responsibility |
|---|---|
| **retrieve** | Embeds the user query, queries Pinecone for top-K similar chunks, computes confidence score |
| **generate** | Constructs a grounded prompt with retrieved context, calls the LLM, enforces refusal on insufficient context |

### Key Design Decisions

- **Confidence scoring:** Uses the **mean cosine similarity** of retrieved chunks (not an arbitrary constant). Scores below 0.60 trigger a low-relevance advisory.
- **Grounding:** The system prompt strictly constrains the LLM to answer only from retrieved context and to refuse when information is unavailable.
- **Direct Pinecone SDK:** Uses the `pinecone` SDK directly instead of `langchain-pinecone` (which has dependency conflicts on Python 3.14).

---

## Project Structure

```
rag-agentic-ai/
├── data/
│   └── Ebook-Agentic-AI.pdf       # Source eBook (60 pages)
├── src/
│   ├── __init__.py
│   ├── config.py                   # Environment variables & constants
│   ├── ingestion.py                # PDF → chunks → embeddings → Pinecone
│   └── graph.py                    # LangGraph RAG workflow
├── app.py                          # FastAPI application
├── tests_sample_queries.py         # 6 benchmark test queries
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
├── .gitignore
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- OpenAI API key ([platform.openai.com](https://platform.openai.com))
- Pinecone API key ([pinecone.io](https://www.pinecone.io))

### 1. Clone and Navigate

```bash
git clone <repository-url>
cd rag-agentic-ai
```

### 2. Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
OPENAI_API_KEY=sk-your-openai-key
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=agentic-ai-rag
```

---

## Ingestion

Ingest the eBook PDF into Pinecone. This extracts text, chunks it, generates embeddings, and upserts vectors.

### Via CLI

```bash
python -m src.ingestion
```

### Via API (after starting the server)

```bash
curl -X POST http://localhost:8000/ingest
```

### What Happens

1. Loads the 60-page PDF using `PyPDFLoader`
2. Splits into ~100-120 chunks (1000 chars, 200 overlap)
3. Generates embeddings using `text-embedding-3-small` (1536 dims)
4. Creates the Pinecone index (if it doesn't exist) — serverless, cosine metric
5. Upserts all vectors with metadata (source, page number, chunk index)

---

## Running the Application

### Start the FastAPI Server

```bash
uvicorn app:app --reload --port 8000
```

The API documentation is available at: [http://localhost:8000/docs](http://localhost:8000/docs)

### API Endpoints

#### `POST /chat` — Ask a Question

**Request:**
```json
{
    "question": "What is Agentic AI?"
}
```

**Response:**
```json
{
    "answer": "Agentic AI refers to AI systems that can autonomously plan, reason, and take actions...",
    "context": [
        {
            "text": "Agentic AI represents a paradigm shift...",
            "source": "data/Ebook-Agentic-AI.pdf",
            "page": 5,
            "relevance_score": 0.87
        }
    ],
    "confidence_score": 0.845
}
```

#### `GET /health` — Health Check

```bash
curl http://localhost:8000/health
```

#### `POST /ingest` — Trigger Ingestion

```bash
curl -X POST http://localhost:8000/ingest
```

---

## Testing

### Benchmark Queries

The test suite includes 6 benchmark queries:

| # | Query | Type | Purpose |
|---|---|---|---|
| 1 | "What is Agentic AI?" | In-domain | Core concept retrieval |
| 2 | "What are the key components of an AI agent?" | In-domain | Factual enumeration |
| 3 | "How do multi-agent systems communicate?" | In-domain | Specific topic retrieval |
| 4 | "What is the difference between LLMs and AI agents?" | In-domain | Comparison task |
| 5 | "What are the risks and challenges of Agentic AI?" | In-domain | Analytical retrieval |
| 6 | "Who won the 2022 FIFA World Cup?" | **Out-of-domain** | **Refusal behavior test** |

### Run Tests

**Against the live API:**
```bash
python tests_sample_queries.py
```

**Directly (no server needed):**
```bash
python tests_sample_queries.py --direct
```

### Validation Criteria

- **In-domain:** Confidence ≥ 0.70, answer contains expected keywords, context is non-empty
- **Out-of-domain:** Confidence < 0.50, answer contains refusal language, does NOT fabricate an answer

---

## Confidence Scoring

The confidence score is computed as the **mean cosine similarity** of the top-K retrieved chunks from Pinecone:

| Range | Interpretation |
|---|---|
| ≥ 0.80 | **High confidence** — context strongly supports the answer |
| 0.60 – 0.79 | **Medium confidence** — context partially relevant |
| < 0.60 | **Low confidence** — context may not support an answer; refusal likely |

When confidence falls below 0.60, the system adds an explicit low-relevance advisory to the LLM prompt, reinforcing refusal behavior.

---

## Configuration

All settings are centralized in `src/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI chat model (swap to `gpt-4o` for higher quality) |
| `LLM_TEMPERATURE` | `0.0` | Deterministic generation for grounded answers |
| `CHUNK_SIZE` | `1000` | Characters per text chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `TOP_K` | `4` | Number of chunks to retrieve |
| `CONFIDENCE_THRESHOLD` | `0.60` | Below this, low-relevance advisory is added |

---

## Tech Stack

| Component | Technology |
|---|---|
| Workflow Orchestration | LangGraph (`StateGraph`) |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o-mini` |
| Vector Store | Pinecone (serverless, cosine) |
| PDF Processing | pypdf via LangChain `PyPDFLoader` |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| API Framework | FastAPI |
| Server | Uvicorn |

---

## License

This project was created as a take-home assignment submission.
