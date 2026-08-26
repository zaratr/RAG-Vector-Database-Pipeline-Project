# RAG Vector Database Pipeline

A Docker-first multimodal retrieval-augmented generation pipeline with persistent vector search and provenance-backed GraphRAG.

The supported runtime uses:

- FastAPI for document ingestion and querying
- FastEmbed with `jinaai/jina-clip-v1` for 768-dimensional text and image embeddings
- ChromaDB in persistent HTTP-server mode
- SQLite, SQLAlchemy, and Alembic for durable relational metadata and graph provenance
- Ollama with Gemma for structured relationship extraction and grounded answer generation
- Deterministic vector, graph, or hybrid retrieval

## How It Works

![Ingestion Flow](public/Diagram_GraphRag%20(2).png)
![Extraction Lifecycle](public/Diagram_GraphRag%20(1).png)
![Document LIfecycle](public/Diagram_GraphRag%20(3).png)


```text
Text / Markdown / PDF
        │
        ├── chunk ── FastEmbed ── deterministic upsert ── ChromaDB
        │                                      │
        └── chunk ── Gemma extraction ── entities / mentions / edges / evidence
                                                   │
                                                   ▼
                                      SQLite relational graph

Image ── Jina CLIP image embedding ── ChromaDB

Query ── vector search ───────────────┐
      └── bounded graph traversal ────┼── deterministic hybrid fusion
                                      └── Gemma grounded answer
```

Text ingestion records an explicit `staged → ready` lifecycle. A document becomes query-visible only after its deterministic Chroma vectors and relational graph provenance have been written successfully. Failed or interrupted cross-store operations can be repaired with the reconciliation command documented below.

## Quick Start with Docker Compose

Docker Compose is the authoritative runtime for this project.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Ollama](https://ollama.com/) reachable from Docker
- A Gemma model available in Ollama under the name configured in `.env`

The committed example uses `gemma4`. Adjust the model name if your Ollama installation uses a tag such as `gemma4:latest`.

### 1. Configure the environment

```bash
cp .env.example .env
```

Copying `.env.example` enables the production recipe below (the hermetic/local recipe is documented in the same file and needs no services):

```env
RAG_LLM_PROVIDER=ollama
RAG_LLM_BASE_URL=http://host.docker.internal:11434/v1
RAG_LLM_MODEL=gemma4

RAG_GRAPH_EXTRACTION_ENABLED=true
RAG_GRAPH_EXTRACTION_MODEL=gemma4
RAG_GRAPH_MAX_HOPS=2

RAG_EMBEDDING_PROVIDER=fastembed
RAG_EMBEDDING_MODEL=jinaai/jina-clip-v1

RAG_CHROMA_HOST=vectordb
RAG_CHROMA_PORT=8000
RAG_DATABASE_URL=sqlite:////data/rag.db
```

For an Ollama server on another machine, replace `host.docker.internal` with that server's hostname or IP address.

> The supported and production-validated generation/extraction path is Ollama. The dummy provider is for development. The OpenAI answer client is currently a placeholder and is not a production implementation.

### 2. Build and start

```bash
docker compose up --build -d
```

Compose starts three services:

| Service | Role |
|---|---|
| `migrate` | One-shot, adoption-aware Alembic migration; must succeed before API startup |
| `vectordb` | Persistent ChromaDB HTTP server |
| `api` | FastAPI application; starts after migration success and Chroma health |

Endpoints:

- API health: <http://localhost:8000/>
- Swagger UI: <http://localhost:8000/docs>
- Host-only Chroma endpoint: <http://127.0.0.1:8001>

The first embedding request can be slow because FastEmbed downloads and initializes the ONNX model. Relational data and Chroma data persist in separate named Docker volumes.

### 3. Ingest documents

#### Inline text

```bash
curl -X POST http://localhost:8000/documents \
  -F "title=Architecture Notes" \
  -F "source=manual" \
  -F "tags=rag,architecture" \
  -F "text=Project Helios uses Vector Engine. Vector Engine powers Search Portal."
```

A successful text response includes the number of chunks and extracted relationships:

```json
{
  "document_id": 1,
  "chunks": 1,
  "relations": 2
}
```

#### Text, Markdown, or PDF upload

```bash
curl -X POST http://localhost:8000/documents \
  -F "title=RAG Article" \
  -F "source=reference" \
  -F "file=@article.md"

curl -X POST http://localhost:8000/documents \
  -F "title=Research Paper" \
  -F "file=@paper.pdf"
```

Supported textual inputs include UTF-8 plain text, `.md`/`.markdown`, and PDFs with extractable text. Empty text, invalid UTF-8, corrupt PDFs, and PDFs without extractable text are rejected explicitly.

#### Image upload

```bash
curl -X POST http://localhost:8000/documents \
  -F "title=System Diagram" \
  -F "file=@diagram.png"
```

Supported image media types are PNG, JPEG, GIF, and WebP. Images are embedded into the same 768-dimensional Jina CLIP space as text, enabling text-to-image retrieval. Graph relationship extraction currently applies to textual chunks, not image contents.

### 4. Query

The API supports three retrieval modes:

| Mode | Behavior |
|---|---|
| `vector` | Dense Chroma retrieval; compatibility default |
| `graph` | Direction-preserving traversal over persisted relational graph evidence |
| `hybrid` | Deterministic reciprocal-rank fusion of vector and graph contexts |

`top_k` accepts `1–50`. The HTTP API accepts `graph_max_hops` from `1–3` and defaults to `2`.

Create a request file to avoid shell quoting problems:

```json
{
  "query": "How is Aria connected to Vector Engine?",
  "retrieval_mode": "hybrid",
  "graph_max_hops": 2,
  "top_k": 5
}
```

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d "@query.json"
```

On Windows PowerShell, use `curl.exe` rather than the `curl` alias:

```powershell
curl.exe -X POST http://localhost:8000/query `
  -H "Content-Type: application/json" `
  -d "@query.json"
```

The response contains a Gemma-generated answer and source contexts. Depending on retrieval mode, context metadata can include:

- SQL-authoritative document, chunk, title, and index identity
- Native vector distance
- `retrieval_sources` such as `vector`, `graph`, or both
- Deterministic `hybrid_score`
- Graph paths with source, predicate, target, hop, evidence, confidence, and extraction model

Only relational documents in the `ready` state are query-visible. Stale Chroma metadata cannot override relational provenance, and a vector hit must carry the exact deterministic ID assigned to its SQL chunk.

### 5. Inspect documents

```bash
curl http://localhost:8000/documents
curl http://localhost:8000/documents/1
```

### 6. Reconcile SQL and Chroma

SQLite and ChromaDB cannot participate in one ACID transaction. The ingestion protocol therefore uses explicit lifecycle state, deterministic vector IDs, compensating cleanup, and an operator reconciliation command:

```bash
docker compose exec api python scripts/reconcile_ingestion.py
```

Example output:

```json
{
  "nonready_vectors_deleted": 0,
  "orphan_vectors_deleted": 0,
  "pending_extractions_failed": 0,
  "ready_chunks_upserted": 3,
  "staged_documents_failed": 0
}
```

Reconciliation treats ready SQL chunks as authoritative. It:

1. Marks interrupted `staged` documents as failed (`reconciled_incomplete`) and terminalizes their still-pending graph extraction runs as failed.
2. Inventories the configured Chroma collection.
3. Deletes vectors not represented by ready SQL chunks, including legacy UUID aliases.
4. Re-embeds and idempotently upserts every ready chunk under `chunk:<chunk_id>`.

> Run reconciliation only against the collection paired with the configured relational database. It intentionally removes collection records that are not authoritative ready SQL chunks.

### 7. Backfill graph extractions

Ready text chunks that predate GraphRAG or lack the current extraction identity can be backfilled idempotently:

```bash
docker compose exec api python scripts/backfill_graph.py --dry-run
docker compose exec api python scripts/backfill_graph.py --document-id 3
docker compose exec api python scripts/backfill_graph.py --retry-failed
```

`--batch-size` accepts `1–100` (default 20). `--document-id` restricts the scan to one document. `--dry-run` classifies eligibility without calling the extraction provider. `--retry-failed` re-attempts chunks whose current identity owner previously failed or whose lease has expired. Backfill never modifies vector IDs, Chroma records, document readiness, or provenance rows owned by other extraction identities. Each run prints one sorted JSON counter report (`scanned`, `eligible`, `processed`, `succeeded`, `skipped`, `empty`, `failed`, `lease_lost`, `relations`, and `skip_reasons`).

Exit codes: `0` when no chunk failed (including no-op runs), `1` when one or more chunks failed (successful chunks remain committed), and `2` for invalid arguments, configuration failure, or a fatal database error.

### 8. Query the persisted graph from the operator CLI

```bash
docker compose exec api python src/graph_rag.py "Aria" --hops 3 --limit 10
```

The CLI traverses the persisted relational graph through the SQL-authoritative `retrieve_graph_paths` traversal and prints complete JSON path objects with per-step provenance. `--hops` accepts `1–3` (default 2), `--direction` accepts `outbound`, `inbound`, or `both` (default `outbound`), and repeatable `--filters KEY=VALUE` applies the scalar document filter matrix (`document_id`, `title`, `source`, `tags`). Unsupported filter keys or malformed `document_id` values exit with an error.

### 9. Run tests

```bash
docker compose exec api python -m pytest -q
```

#### Running the tests without any services

The suite is hermetic: it needs no Ollama, no Chroma server, and no
`fastembed` model downloads. From a clean checkout, with only the pinned
dependencies installed (for example in a project venv):

```bash
python -m pytest -q
```

The application defaults already select the hermetic configuration
(`RAG_EMBEDDING_PROVIDER=local`, no Chroma host → ephemeral client). If a
local `.env` selects the production recipe instead, override it for the run:

```bash
RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest -q
```

One test is skipped by default: `test_rag_api.py::test_query_answer_lane_with_live_llm_optin`
is an opt-in lane that exercises the real LLM only when the `RAG_LIVE_LLM`
environment variable is set and the configured `RAG_LLM_BASE_URL` answers.


### 10. Stop or reset

```bash
# Stop services; named-volume data remains
docker compose down

# Permanently remove relational and Chroma data
docker compose down -v
```

> `docker compose down -v` permanently deletes both the SQLite and Chroma named volumes. Re-ingestion is required afterward.

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | API health check |
| `POST` | `/documents` | Ingest inline text, text/Markdown/PDF files, or images |
| `GET` | `/documents` | List documents |
| `GET` | `/documents/{id}` | Return a document and its chunks |
| `POST` | `/query` | Generate an answer using vector, graph, or hybrid retrieval |
| `GET` | `/graph/entities` | List graph entities visible through ready-document evidence |
| `GET` | `/graph/entities/{id}/relationships` | List an entity's relationships that have ready evidence |
| `POST` | `/graph/paths` | Run bounded directional path traversal over persisted graph evidence |

Important error behavior:

- `400`: invalid/empty document input or unsupported file
- `404`: document or graph entity not found
- `422`: invalid query controls, unsupported or malformed graph filters (including non-integer `document_id` filter values), or invalid graph-inspection parameters
- `502`: graph extraction provider returned unusable structured output
- `503`: vector index unavailable during ingestion, graph provider unavailable, or traversal safety limit reached

### Graph inspection endpoints

The `/graph/*` routes are read-only over ready-document evidence and never touch the embedding or Chroma layer. Entities, relationships, and paths evidenced only by non-ready documents are invisible.

- `GET /graph/entities` — optional `name` needle with `match` mode (`exact` default, or `prefix`/`contains`; `%`/`_` are matched literally), optional `entity_type` (one of `person`, `organization`, `location`, `product`, `project`, `technology`, `concept`, `event`, `other`), `limit` `1–100` (default 20), and `offset`. Results are ordered by canonical name, entity type, then entity ID, and each item carries ready-document mention and evidence counts. An unsupported `entity_type` returns `422`.
- `GET /graph/entities/{id}/relationships` — `direction` (`outbound` default, `inbound`, or `both`), `limit` `1–100` (default 20), and `offset`. Items are sorted by source canonical name, predicate, target canonical name, then edge ID; only edges with ready evidence are returned. Unknown entity IDs return `404`.
- `POST /graph/paths` — JSON body: `query`, `max_hops` (default 3; values outside `1–3` are rejected), `direction` (default `outbound`), `limit` `1–50` (default 20; values outside the range are rejected rather than clamped), and optional scalar `filters` (`document_id`, `title`, `source`, `tags`). Returns complete `GraphPath` objects with per-step provenance. Exceeding a traversal safety cap returns `503`; unsupported or malformed filters return `422`.

## Architecture

| Component | Technology | Current behavior |
|---|---|---|
| API | FastAPI 0.141.1 | Async HTTP surface and OpenAPI docs |
| Embeddings | FastEmbed 0.8.0 + `jinaai/jina-clip-v1` | 768-dimensional text/image embeddings through ONNX |
| Vector store | ChromaDB 1.5.9 | Persistent HTTP server, deterministic IDs, idempotent upserts |
| Generation | Ollama + Gemma | Grounded answers through OpenAI-compatible chat completions |
| Graph extraction | Ollama + Gemma | Strict JSON-schema relationships, grounding, normalization, lease-guarded retries, explicit empty/failure states |
| Relational store | SQLite + SQLAlchemy 2.0.51 | Documents, chunks, lifecycle state, and normalized graph provenance |
| Migrations | Alembic 1.18.5 | One-shot Compose migrator, legacy adoption, exact-schema drift validation |
| Graph retrieval | Relational bounded traversal | Directional, deterministic, cycle-safe, provenance-preserving expansion |
| Hybrid retrieval | Reciprocal-rank fusion | Chunk-level deduplication with native and fusion scores retained separately |
| Runtime | Docker Compose | Immutable API/migrator images plus named relational and Chroma volumes |

### Persisted graph provenance

The relational graph separates logical identity from per-extraction evidence:

| Table | Purpose |
|---|---|
| `graph_extractions` | Provider/model/prompt/schema version, status, error, and chunk provenance |
| `graph_entities` | Canonical entity identity keyed by normalized name and entity type |
| `entity_mentions` | Exact source surface form and offsets for each extraction |
| `graph_edges` | Deduplicated logical source–predicate–target relationships |
| `graph_edge_evidence` | Per-extraction evidence text, offsets, confidence, and provenance |

This separation preserves repeated evidence without duplicating logical edges and prevents same-name entities of different types from being merged.

### Ingestion consistency model

The relational database is the visibility authority:

- `documents.ingestion_status` is constrained to `staged`, `ready`, or `failed`.
- Chunks receive unique deterministic vector IDs.
- Retrieval excludes non-ready documents.
- Chroma hits are accepted only when the returned record ID matches the SQL chunk's vector ID.
- Reconciliation converges Chroma to the complete set of ready SQL chunks.

### Migration lineage

```text
dee48bc24a7f  frozen documents/chunks baseline
      ↓
4c9a8d7e6f5b  graph provenance and ingestion lifecycle
      ↓
a6e2c4f8b1d9  database constraint for document lifecycle states
      ↓
b7f3d5a9c2e1  hardened graph extraction lifecycle: idempotent identity owner, attempt counters, pending lease (head)
```

Application import and API startup do not run migrations. The one-shot `migrate` Compose service owns schema adoption and upgrades.

## Configuration

All application settings use the `RAG_` prefix and load from `.env`.

| Variable | Application default | Docker example | Description |
|---|---:|---:|---|
| `RAG_LLM_PROVIDER` | `ollama` | `ollama` | Answer provider; Ollama is the validated path |
| `RAG_LLM_BASE_URL` | `http://localhost:11434/v1` | `http://host.docker.internal:11434/v1` | OpenAI-compatible Ollama base URL |
| `RAG_LLM_MODEL` | `gemma4:latest` | `gemma4` | Answer-generation model |
| `RAG_GRAPH_EXTRACTION_ENABLED` | `true` | `true` | Enable structured graph extraction for text ingestion |
| `RAG_GRAPH_EXTRACTION_MODEL` | answer model | `gemma4` | Optional extraction-specific model |
| `RAG_GRAPH_MAX_HOPS` | `2` | `2` | Configured hop bound; HTTP requests currently use their own validated `graph_max_hops` field (default 2, range 1–3) |
| `RAG_EXTRACTION_LEASE_SECONDS` | `600` | `600` | Graph extraction lease duration in seconds (range 60–3600); an expired pending lease can be reclaimed by a backfill retry, and reconciliation terminalizes still-pending extractions as failed |
| `RAG_EMBEDDING_PROVIDER` | `local` | `fastembed` | `local`, `fastembed`, or `openai`; `local` uses deterministic hash embeddings (no model download), `fastembed` produces real 768-dimensional CLIP embeddings |
| `RAG_EMBEDDING_MODEL` | `jinaai/jina-clip-v1` | same | Embedding model |
| `RAG_DATABASE_URL` | `sqlite:///./rag.db` | `sqlite:////data/rag.db` | SQLAlchemy database URL |
| `RAG_CHROMA_HOST` | empty | `vectordb` | Chroma host; empty selects standalone modes |
| `RAG_CHROMA_PORT` | `8000` | `8000` | Chroma server port |
| `RAG_CHROMA_PERSIST_DIRECTORY` | empty | empty | Standalone persistent-client path when host is empty |
| `RAG_OPENAI_API_KEY` | empty | placeholder | Used only by OpenAI-oriented provider paths |

## Project Structure

```text
.
├── app/
│   ├── api/
│   │   ├── routes_documents.py       # ingestion and document inspection
│   │   ├── routes_graph.py           # read-only graph inspection endpoints
│   │   └── routes_query.py           # vector/graph/hybrid query API
│   ├── core/
│   │   ├── db.py                     # engine, sessions, SQLite FK enforcement
│   │   ├── migrations.py             # adoption-aware migration entrypoint
│   │   └── models.py                 # API schemas
│   ├── persistence/
│   │   ├── alembic/                  # frozen baseline and forward revisions
│   │   ├── graph_repository.py       # normalized graph persistence
│   │   ├── models.py                 # relational document/graph models
│   │   └── repositories.py           # document/chunk operations
│   ├── services/
│   │   ├── embeddings.py             # text and image embeddings
│   │   ├── graph_backfill.py         # idempotent graph extraction backfill
│   │   ├── graph_extraction.py       # strict Gemma relationship extraction
│   │   ├── graph_retrieval.py        # bounded persisted traversal
│   │   ├── ingestion.py              # staged SQL/Chroma ingestion lifecycle
│   │   ├── llm.py                    # Ollama and development answer clients
│   │   ├── reconciliation.py         # cross-store convergence
│   │   ├── retrieval.py              # vector/graph/hybrid retrieval
│   │   └── vector_store.py           # persistent Chroma adapter
│   ├── tests/                        # migration, failure, retrieval, API, persistence tests
│   ├── config.py
│   └── main.py
├── scripts/
│   ├── backfill_graph.py             # operator graph backfill command
│   └── reconcile_ingestion.py        # operator reconciliation command
├── src/
│   └── graph_rag.py                  # persisted graph traversal CLI
├── .env.example
├── alembic.ini
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```
