# RAG Vector Database Pipeline

A multimodal retrieval-augmented generation pipeline with text and image support,
running entirely on local models. Documents are chunked, embedded into a shared
text+image vector space via Jina CLIP v1 (FastEmbed/ONNX), stored in ChromaDB, and
queried through a FastAPI surface with Gemma4 generation via Ollama.

```
document ─→ chunk ─→ jina-clip-v1 embed ─→ ChromaDB store
(text/PDF/image)                              │
                                               ↓
query ──→ jina-clip-v1 embed ─→ top-k retrieve ─→ Gemma4 generate ─→ answer
```

## Quick Start (Docker)

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Ollama](https://ollama.com/) running somewhere reachable, with `gemma4:latest` pulled

### 1. Configure environment

```bash
cp .env.example .env
```

Edit `.env` to point at your Ollama instance:

```env
RAG_LLM_PROVIDER=ollama
RAG_LLM_BASE_URL=<ollama server>:11434/v1
RAG_LLM_MODEL=gemma4:latest
RAG_EMBEDDING_PROVIDER=fastembed
RAG_EMBEDDING_MODEL=jinaai/jina-clip-v1
```

### 2. Build and start

```bash
docker compose up --build -d
```

The API will be available at `http://localhost:8000`.
Swagger UI at `http://localhost:8000/docs`.

First request will be slow — FastEmbed downloads the ONNX model (~900MB for
text + vision) on first embed call, then caches it in the Docker volume.

### 3. Ingest documents

**Text / Markdown:**

```bash
# Inline text
curl -X POST http://localhost:8000/documents \
  -F "title=My Document" \
  -F "text=Some content to ingest" \
  -F "source=manual"

# File upload (.txt, .md)
curl -X POST http://localhost:8000/documents \
  -F "title=RAG Article" \
  -F "file=@article.txt" \
  -F "source=wikipedia"

# PDF
curl -X POST http://localhost:8000/documents \
  -F "title=Research Paper" \
  -F "file=@paper.pdf"
```

**Images (PNG, JPEG, GIF, WebP):**

```bash
curl -X POST http://localhost:8000/documents \
  -F "title=Diagram" \
  -F "file=@diagram.png"
```

Images are embedded into the **same 768-dim vector space** as text via Jina CLIP,
so text queries can find relevant images and vice versa.

### 4. Query

```bash
# Write JSON body to file (avoids PowerShell quoting issues)
echo '{"query":"What is retrieval augmented generation?","top_k":3}' > query.json

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d "@query.json"
```

Response includes the Gemma-generated answer plus the retrieved context chunks
with similarity scores.

### 5. List / inspect documents

```bash
# List all documents
curl http://localhost:8000/documents

# Get a specific document with its chunks
curl http://localhost:8000/documents/1
```

### 6. Run tests

```bash
docker compose exec api python -m pytest -q
```

### 7. Stop / clean up

```bash
# Stop containers (data persists in volumes)
docker compose down

# Stop and wipe all data (Chroma vectors + SQLite DB)
docker compose down -v
```

> **Note:** `docker compose down -v` deletes the ChromaDB volume. After wiping,
> you must re-ingest all documents before queries will return results.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/documents` | Ingest text, file (.txt/.pdf), or image (png/jpg/gif/webp) |
| `GET` | `/documents` | List all documents |
| `GET` | `/documents/{id}` | Get document detail with chunks |
| `POST` | `/query` | RAG query — retrieve top-k chunks + generate answer |

## Architecture

| Component | Technology | Details |
|-----------|-----------|---------|
| **API** | FastAPI 0.141 | Async, auto-docs at `/docs` |
| **Embeddings** | FastEmbed + jina-clip-v1 | 768-dim shared text+image space, ONNX CPU inference |
| **Vector Store** | ChromaDB 1.5.9 | Persistent HTTP server mode (Docker), L2 distance |
| **Generation** | Ollama (Gemma4) | OpenAI-compatible `/v1/chat/completions` endpoint |
| **Database** | SQLite + SQLAlchemy | Document/chunk metadata tracking |
| **Container** | Docker Compose | `api` + `vectordb` services |

### Configuration

All settings are loaded from `.env` (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_LLM_PROVIDER` | `ollama` | `dummy`, `ollama`, or `openai` |
| `RAG_LLM_BASE_URL` | `http://localhost:11434/v1` | Ollama API endpoint |
| `RAG_LLM_MODEL` | `gemma4:latest` | Model name in Ollama |
| `RAG_EMBEDDING_PROVIDER` | `fastembed` | `local` (hash fallback), `fastembed`, or `openai` |
| `RAG_EMBEDDING_MODEL` | `jinaai/jina-clip-v1` | FastEmbed model name |
| `RAG_DATABASE_URL` | `sqlite:///./rag.db` | SQLAlchemy connection string |
| `RAG_CHROMA_HOST` | `vectordb` | ChromaDB server hostname (Docker service name). Leave empty for standalone. |
| `RAG_CHROMA_PORT` | `8000` | ChromaDB server port |
| `RAG_CHROMA_PERSIST_DIRECTORY` | *(empty)* | Local persistence path for standalone mode (when `RAG_CHROMA_HOST` is empty) |

## Project Structure  

```
.
├── app/
│   ├── api/
│   │   ├── routes_documents.py   # POST/GET /documents
│   │   └── routes_query.py       # POST /query
│   ├── services/
│   │   ├── embeddings.py         # FastEmbed text + image providers
│   │   ├── ingestion.py          # Chunking + embedding pipeline
│   │   ├── vector_store.py       # ChromaDB wrapper
│   │   ├── retrieval.py          # Top-k vector search
│   │   ├── llm.py                # Ollama / Dummy / OpenAI clients
│   │   ├── rag.py                # Retrieve → generate orchestration
│   │   └── chunking.py           # Text splitter
│   ├── persistence/
│   │   ├── models.py             # SQLAlchemy Document + Chunk
│   │   └── repositories.py       # DB CRUD
│   ├── core/
│   │   ├── db.py                 # Engine + session factory
│   │   ├── models.py             # Pydantic schemas
│   │   └── logging.py            # Logger setup
│   ├── tests/                    # pytest (ingestion, retrieval, API)
│   ├── config.py                 # Pydantic Settings (.env loading)
│   └── main.py                   # FastAPI app factory
├── src/
│   └── graph_rag.py              # GraphRAG spike (not integrated — see below)
├── .env.example                  # Environment template
├── docker-compose.yml            # Compose config
├── Dockerfile                    # python:3.11-slim + vim
├── requirements.txt              # Pinned dependencies
└── README.md
```

## GraphRAG (Roadmap)

`src/graph_rag.py` is an isolated research spike using NetworkX. It is **not**
wired into the live pipeline — no entity extraction runs at ingest time, and
no graph traversal participates in retrieval.

**Planned integration:**

1. Entity extraction on ingested chunks via LLM (extract Entity-Relationship-Entity triplets)
2. Persist triplets in a graph store alongside the vector index
3. Hybrid retrieval: expand seed entities via graph neighbors before dense top-k fallback

## Tech Stack

- **Backend:** FastAPI, Python 3.11
- **Embeddings:** FastEmbed (ONNX) with jina-clip-v1
- **Vector Store:** ChromaDB
- **Generation:** Ollama (Gemma4)
- **Graph (planned):** NetworkX
- **Container:** Docker Compose
