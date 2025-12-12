<<<<<<< ours
# Codex Prompt: RAG + Vector Database Pipeline Project

**Goal:**
Build a production-style **Retrieval-Augmented Generation (RAG)** system that ingests documents, stores them in a **vector database**, and exposes an API where a client can ask questions and get grounded answers based on the ingested content.

The project should be structured as a small, clean backend service with clear modules, tests, and documentation.

## 1. Tech Stack

Use the following technologies:

- **Language:** Python 3.10+
- **Web framework:** FastAPI
- **Vector DB:** Qdrant (self-hosted via Docker) OR ChromaDB (local) – pick one and integrate it cleanly behind an interface
- **Embeddings:** Use a pluggable embedding provider abstraction
  - Implement one provider using OpenAI embeddings (stub config/env for API key)
  - Also implement a simple local embedding provider using `sentence-transformers` if possible
- **LLM (for generation):**
  - Implement an abstraction for a “LLM client”
  - Provide a dummy/local implementation (e.g., a simple template-based response) so the code runs without external keys
  - Optionally stub integration with OpenAI ChatCompletion-style models
- **Storage for metadata:**
  - Use SQLite via SQLAlchemy for basic metadata (documents, chunks, etc.)
- **Containerization:** Dockerfile for the API + docker-compose for API + vector DB

## 2. Project Structure

Create a well-organized repository with this approximate structure:

```
rag_pipeline/
  app/
    __init__.py
    main.py                  # FastAPI entrypoint
    config.py                # Settings (env vars, etc.)
    api/
      __init__.py
      routes_documents.py    # Upload/list documents
      routes_query.py        # Query endpoint
    core/
      models.py              # Pydantic models / schemas
      db.py                  # DB session, init
      logging.py             # Logging setup
    services/
      __init__.py
      embeddings.py          # Embedding provider interfaces + impls
      llm.py                 # LLM interfaces + impls
      vector_store.py        # Vector DB interface + impls
      chunking.py            # Text splitting logic
      ingestion.py           # High-level ingestion pipeline
      retrieval.py           # Retrieval + ranking logic
      rag.py                 # Orchestrates retrieval + LLM call
    persistence/
      __init__.py
      models.py              # SQLAlchemy models
      repositories.py        # CRUD for documents/chunks
  tests/
    test_ingestion.py
    test_retrieval.py
    test_rag_api.py
  scripts/
    dev_seed.py              # Optionally seed some test docs
  Dockerfile
  docker-compose.yml
  pyproject.toml or requirements.txt
  README.md
```

Use type hints everywhere and docstrings for all public functions.

## 3. Core Features

### 3.1. Document Ingestion API

Implement an endpoint:

- `POST /documents`
  - Accepts:
    - `title` (string)
    - `source` (string, optional)
    - `tags` (list of strings, optional)
    - `file` (PDF or plain text) OR raw text in JSON
  - Steps:
    1. Extract text from file if PDF (use a simple PDF text extractor like `pypdf`).
    2. Normalize text (strip, normalize whitespace).
    3. Chunk text into overlapping chunks (e.g., 512–1024 tokens or characters with overlap).
    4. Generate embeddings for each chunk using the configured embedding provider.
    5. Store:
       - Document metadata in SQLite
       - Chunks metadata in SQLite
       - Embedding vectors in the vector DB, associated with:
         - document_id
         - chunk_id
         - tags
         - title
  - Return: Document ID + number of chunks created.

Also implement:

- `GET /documents`
  - List ingested documents with basic metadata and chunk counts.
- `GET /documents/{document_id}`
  - Get full metadata + summary of chunks for one document.

### 3.2. Query API (RAG endpoint)

Implement an endpoint:

- `POST /query`
  - Accepts:
    - `query` (string)
    - `top_k` (int, optional, default e.g. 5)
    - `filters` (optional; e.g., by tags or document_ids)
  - Steps:
    1. Embed the query using the same embedding provider.
    2. Search the vector DB for most similar chunks (respecting `top_k`, filters).
    3. Return:
       - The retrieved chunks (text, source, score).
       - A final answer from the LLM (or dummy LLM) that:
         - Reads the query
         - Reads the top-k context chunks
         - Produces an answer grounded in the provided context.
  - The answer pipeline should be in `services/rag.py` and call:
    - `retrieval.retrieve(...)`
    - `llm.generate_answer(context_chunks, query)`

The LLM layer must be easily swappable (e.g., base class + concrete implementation).

## 4. Abstractions & Interfaces

### 4.1. Embedding Provider

Create an abstract class / protocol like:

```python
class EmbeddingProvider(Protocol):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...
```

Implement:

- `OpenAIEmbeddingProvider`
- `LocalEmbeddingProvider` (using sentence-transformers, if available)

Make embedding provider configurable via environment variable (e.g., `EMBEDDING_PROVIDER=openai|local`).

### 4.2. Vector Store

Create an interface like:

```python
class VectorStore(Protocol):
    async def index_embeddings(
        self,
        embeddings: list[list[float]],
        metadatas: list[dict],
        ids: list[str],
    ) -> None:
        ...

    async def query(
        self,
        embedding: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        ...
```

Implement one concrete class, e.g.:

- `QdrantVectorStore` OR
- `ChromaVectorStore`

Make the implementation configurable via environment variable.

### 4.3. LLM Client

Create interface:

```python
class LLMClient(Protocol):
    async def generate_answer(self, query: str, context: list[str]) -> str:
        ...
```

Implement:

- `DummyLLMClient` that returns a templated response combining context snippets.
- Optionally `OpenAILLMClient` (stub with OpenAI ChatCompletion-style call).

## 5. Chunking Logic

Implement `services/chunking.py`:

- A function that splits long text into overlapping chunks by characters (e.g., 1000 chars with 200-char overlap).
- Each chunk should carry:
  - an index
  - a reference to the original document
  - position metadata (e.g., start/end offsets)

Make it easy to swap to token-based chunking in the future.

## 6. Configuration & Environment

Create a centralized `config.py` using Pydantic’s `BaseSettings` or similar. Support:

- `EMBEDDING_PROVIDER`
- `VECTOR_STORE`
- `OPENAI_API_KEY` (if used)
- `QDRANT_URL`, `QDRANT_API_KEY` (if used)
- `DATABASE_URL` (SQLite by default)
- App port, debug mode, etc.

## 7. Testing

Add tests in `tests/`:

- `test_ingestion.py`
  - Test text-only ingestion (no PDF).
  - Assert that chunks are stored and vector store index is called.
- `test_retrieval.py`
  - Insert a fake document + embeddings.
  - Query for related content and assert correct chunk(s) are retrieved.
- `test_rag_api.py`
  - Use FastAPI TestClient.
  - Ingest a small document.
  - Call `/query` and assert:
    - 200 OK
    - Response includes answer + context.

Use pytest.

## 8. Docker & Local Dev

Create:

- `Dockerfile` for the FastAPI app:
  - Install dependencies
  - Run with `uvicorn app.main:app`
- `docker-compose.yml`:
  - Service: `api` (FastAPI app)
  - Service: `vectordb` (Qdrant container or similar)
  - Map ports with sensible defaults.

## 9. README

Generate a `README.md` that includes:

- Project overview (RAG + VectorDB, what it does)
- Tech stack
- How to run locally:
  - `docker-compose up` or `uvicorn`
- How to:
  - Ingest a document (sample curl or HTTPie command)
  - Query the system
- How to switch embedding provider / vector store via ENV
- High-level architecture diagram (ASCII or description)

## 10. Code Quality

- Use type hints and docstrings.
- Apply basic logging for:
  - ingestion steps
  - retrieval operations
  - errors/exceptions
- Make the service resilient to:
  - Missing configurations
  - Vector DB not available (clear error message)
- Keep functions small, focused, and testable.

You can feed this whole specification into Codex / a code model and say:

> “Generate the full Python project according to this specification.”

If you’d like, next I can do the same Codex-style write-up for another project (e.g., AI analytics dashboard, AI security lab, or LLM observability service).
=======
# RAG + Vector Database Pipeline

A production-style Retrieval-Augmented Generation (RAG) service built with FastAPI, SQLite metadata storage, and a configurable embedding/vector store pipeline. The service ingests documents, chunks and embeds their contents, indexes them in ChromaDB, and exposes query endpoints that retrieve relevant context before generating an answer with a pluggable LLM client.

## Features
- Document ingestion via HTTP (raw text or uploaded PDF/plaintext files)
- Character-based overlapping chunking with offsets
- Embedding providers: local deterministic encoder (with optional `sentence-transformers`) or stubbed OpenAI embeddings
- Vector store interface with a ChromaDB implementation
- SQLite metadata persistence (documents + chunks) through SQLAlchemy
- Retrieval + RAG orchestration with a dummy LLM client (swap for OpenAI if desired)
- Containerization via Docker and docker-compose
- Pytest suite covering ingestion, retrieval, and API integration

## Project Structure
```
app/
  api/                # FastAPI routes
  core/               # config, db, logging, schemas
  persistence/        # SQLAlchemy models and repositories
  services/           # embeddings, vector store, chunking, ingestion, retrieval, rag, llm
scripts/              # utilities (dev seeding)
```

## Getting Started
### Install dependencies
```bash
pip install -r requirements.txt
```

### Run locally
```bash
uvicorn app.main:app --reload
```

### With Docker
```bash
docker-compose up --build
```

### Environment variables
- `RAG_EMBEDDING_PROVIDER` (default: `local`)
- `RAG_VECTOR_STORE` (default: `chroma`)
- `RAG_OPENAI_API_KEY` (if using OpenAI embedding/LLM stubs)
- `RAG_DATABASE_URL` (default: `sqlite:///./rag.db`)
- `RAG_CHROMA_PERSIST_DIRECTORY` (optional, for persistent Chroma storage)

## API
### Ingest a document
```bash
curl -X POST "http://localhost:8000/documents" \
  -F "title=Example" \
  -F "text=This is a sample document for ingestion" \
  -F "source=local"
```

### Query
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the sample about?","top_k":3}'
```

## Testing
```bash
pytest
```

## Notes
- The default local embedding provider is deterministic and lightweight. If `sentence-transformers` is installed, it will be used automatically.
- The OpenAI embedding/LLM clients are stubs that validate configuration and return synthetic outputs to keep tests offline.
>>>>>>> theirs
