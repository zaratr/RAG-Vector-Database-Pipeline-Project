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
