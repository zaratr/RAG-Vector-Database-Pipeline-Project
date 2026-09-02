# RAG Vector Database Pipeline

## 1. What This Is

A single-user, Docker-first retrieval-augmented generation (RAG) knowledge base.
Its distinctive feature is GraphRAG: a real LLM extracts entities and relations
from ingested text into a SQL graph, queries traverse that graph over multiple
hops, and answers blend vector similarity with graph evidence — always with
citations back to the exact source chunk and character offsets. The stack is
FastAPI, FastEmbed (`jinaai/jina-clip-v1`, 768-dim), persistent ChromaDB,
SQLite/SQLAlchemy/Alembic, and Ollama with Gemma for extraction and answer
generation. This is a portfolio/reference-grade project, not a production
service — see *Scope* below for the honest boundaries.

## 2. What 10A Gives You

The core loop, plain then precise:

1. **Ingest** — text, Markdown, or PDF text goes through chunking; images are
   embedded as vectors.
2. **Extract** — each text chunk is sent to Gemma (`gemma4` via Ollama) under a
   strict JSON schema; every extracted entity, mention, and relationship records
   provenance to its chunk and character offsets.
3. **Persist** — chunks become deterministic ChromaDB vectors (idempotent
   upserts, stable IDs `chunk:<chunk_id>`); entities and edges land in the
   normalized SQLite graph.
4. **Traverse** — bounded multi-hop graph traversal (hops 1–3), implemented as
   relational SQL (NetworkX was the original F2 technology choice, was removed
   under amendment 10A.5, and is history only).
5. **Retrieve and answer** — vector, graph, or hybrid mode with deterministic
   reciprocal-rank fusion (RRF, k=60); Gemma generates a grounded answer with
   citations.

One worked example. Ingest this sentence:

> "Aria manages Project Helios. Project Helios uses Vector Engine."

Extraction persists two edges — `Aria —manages→ Project Helios` and
`Project Helios —uses→ Vector Engine` — each with the evidence text and offsets.
The query `"How is Aria connected to Vector Engine?"` in `hybrid` mode fuses
dense vector hits with the two-hop graph path
`Aria → Project Helios → Vector Engine`, and the answer cites the source chunk
for every claim.

## 3. How It Works

![Ingestion Flow](public/Diagram_GraphRag%20(2).png)
![Extraction Lifecycle](public/Diagram_GraphRag%20(1).png)
![Document Lifecycle](public/Diagram_GraphRag%20(3).png)

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

### SQL authority

SQLite is the visibility authority; Chroma records and LLM output are untrusted
hints. Only documents in the `ready` state are query-visible, retrieval excludes
non-ready documents, a Chroma hit is accepted only when its record ID matches
the SQL chunk's deterministic vector ID, and stale Chroma metadata can never
override relational provenance. The `/graph/*` inspection routes are read-only
over ready-document evidence and never touch the embedding layer; entities,
relationships, and paths evidenced only by non-ready documents are invisible.

### Extraction lifecycle (lease-fenced and idempotent)

Graph extraction identity is the chunk's `input_sha256` together with the
provider/model/prompt/schema version. State transitions use compare-and-swap
ownership: a run takes a time-bounded pending lease (`RAG_EXTRACTION_LEASE_SECONDS`,
default 600 s, range 60–3600), and a worker that loses its lease to another run
observes `ExtractionLeaseLost` and stops writing — an expired lease can be
reclaimed by a backfill retry. Extraction runs are terminal-protected: completed
or failed runs are never re-executed by the same identity. Reconciliation
terminalizes still-pending extractions of failed documents.

### Ingestion state machine and reconciliation

Documents move through an explicit `staged → ready | failed` lifecycle enforced
by a database CHECK constraint. A document becomes `ready` — and therefore
query-visible — only after both its Chroma vectors and its graph provenance have
been written successfully. SQLite and ChromaDB cannot participate in one ACID
transaction, so cross-store divergence is repaired by the operator
reconciliation command (`scripts/reconcile_ingestion.py`, below in Quickstart):
ready SQL chunks are authoritative; interrupted `staged` documents are marked
failed; vectors not represented by ready SQL chunks (including legacy UUID
aliases) are deleted; every ready chunk is re-embedded and idempotently
upserted. Backfill (`scripts/backfill_graph.py`) re-runs extraction for ready
chunks that predate GraphRAG or lack the current extraction identity — it never
modifies vector IDs, Chroma records, document readiness, or provenance rows
owned by other extraction identities.

### Traversal caps and scoring

Graph traversal is bounded by hop count (`max_hops` 1–3; HTTP default 2, CLI
default 2, `/graph/paths` default 3) and a traversal safety limit that returns
`503` when exceeded. Edges carry extraction confidence, and traversal preserves
direction and per-step provenance. Hybrid retrieval fuses vector and graph
result lists with exact reciprocal-rank fusion, RRF k=60
(`score = Σ 1 / (60 + rank)`), computed per deduplicated chunk with native and
fusion scores retained separately — the `hybrid_score` is never rounded, so
fusion is deterministic.

### Security and safety layers

- **Server-assigned source trust.** Document trust fields come from a pinned
  source-trust policy (`RAG_SOURCE_TRUST_POLICY_PATH`); a missing or invalid
  policy refuses startup — the control fails closed.
- **Prompt-injection quarantine.** A pinned context-security policy
  (`RAG_CONTEXT_SECURITY_POLICY_PATH`) governs retrieval-time context
  inspection; failures fail closed with typed errors (`context_detector_failed`).
- **Calibrated poisoning controls.** The retrieval-security policy
  (`config/retrieval-security-policy.json`, `max_distance` 0.643872, l2) is
  calibrated to the fastembed/Jina regime; loading it under a mismatched
  embedding regime is a typed, fail-closed error naming both regimes. Ingestion
  is rate limited (atomic cross-worker buckets, default 30 requests/minute) with
  persisted security audits under bounded retention
  (`RAG_SECURITY_AUDIT_RETENTION_DAYS`, default 30), prunable with
  `scripts/prune_security_audits.py`.
- **Content safety.** A byte-pinned content-safety policy
  (`RAG_CONTENT_SAFETY_POLICY_PATH`) is enforced at generation time and fails
  closed: when every retrieved candidate is blocked, `/query` still returns
  `200` with the deterministic answer
  `No safe context was available to answer the query.`

### Key modules

| Component | Technology | Behavior |
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

The relational graph separates logical identity from per-extraction evidence:

| Table | Purpose |
|---|---|
| `graph_extractions` | Provider/model/prompt/schema version, status, error, and chunk provenance |
| `graph_entities` | Canonical entity identity keyed by normalized name and entity type |
| `entity_mentions` | Exact source surface form and offsets for each extraction |
| `graph_edges` | Deduplicated logical source–predicate–target relationships |
| `graph_edge_evidence` | Per-extraction evidence text, offsets, confidence, and provenance |

This separation preserves repeated evidence without duplicating logical edges
and prevents same-name entities of different types from being merged.

Source layout (see also *Configuration* below):

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
│   │   ├── ingestion_limits.py       # envelope/file/extracted byte caps, rate buckets
│   │   ├── llm.py                    # Ollama and development answer clients
│   │   ├── reconciliation.py         # cross-store convergence
│   │   ├── retrieval.py              # vector/graph/hybrid retrieval (RRF k=60)
│   │   └── vector_store.py           # persistent Chroma adapter
│   ├── tests/                        # migration, failure, retrieval, API, persistence tests
│   ├── config.py
│   └── main.py
├── scripts/
│   ├── backfill_graph.py             # operator graph backfill command
│   ├── prune_security_audits.py      # bounded-retention audit pruning
│   ├── reconcile_ingestion.py        # operator reconciliation command
│   ├── run_redteam.py                # one-shot isolated red-team harness
│   └── validate_*.py                 # phase validators and image hygiene
├── src/
│   └── graph_rag.py                  # persisted graph traversal CLI
├── .env.example
├── alembic.ini
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

### The four API surfaces

| Method | Path | Description |
|---|---|---|
| `GET` | `/graph/entities` | List graph entities visible through ready-document evidence |
| `GET` | `/graph/entities/{id}/relationships` | List an entity's relationships that have ready evidence |
| `POST` | `/graph/paths` | Run bounded directional path traversal over persisted graph evidence |
| `POST` | `/query` | Generate an answer using vector, graph, or hybrid retrieval |

(Plus the document surface: `GET /` health, `POST /documents` ingest,
`GET /documents`, `GET /documents/{id}`.)

- `GET /graph/entities` — optional `name` needle with `match` mode (`exact`
  default, or `prefix`/`contains`; `%`/`_` are matched literally), optional
  `entity_type` (one of `person`, `organization`, `location`, `product`,
  `project`, `technology`, `concept`, `event`, `other`), `limit` `1–100`
  (default 20), and `offset`. Results are ordered by canonical name, entity
  type, then entity ID, and each item carries ready-document mention and
  evidence counts. An unsupported `entity_type` returns `422`.
- `GET /graph/entities/{id}/relationships` — `direction` (`outbound` default,
  `inbound`, or `both`), `limit` `1–100` (default 20), and `offset`. Items are
  sorted by source canonical name, predicate, target canonical name, then edge
  ID; only edges with ready evidence are returned. Unknown entity IDs return
  `404`.
- `POST /graph/paths` — JSON body: `query`, `max_hops` (default 3; values
  outside `1–3` are rejected), `direction` (default `outbound`), `limit` `1–50`
  (default 20; values outside the range are rejected rather than clamped), and
  optional scalar `filters` (`document_id`, `title`, `source`, `tags`). Returns
  complete `GraphPath` objects with per-step provenance. Exceeding a traversal
  safety cap returns `503`; unsupported or malformed filters return `422`.
- `POST /query` — modes `vector` (dense Chroma retrieval, compatibility
  default), `graph` (direction-preserving traversal over persisted relational
  evidence), or `hybrid` (deterministic RRF fusion). `top_k` accepts `1–50`;
  `graph_max_hops` accepts `1–3` and defaults to `2`. Responses carry
  SQL-authoritative document/chunk identity, native vector distance,
  `retrieval_sources` (`vector`, `graph`, or both), deterministic
  `hybrid_score`, and graph paths with source, predicate, target, hop, evidence,
  confidence, and extraction model.

Error behavior: `400` invalid/empty document input or unsupported file; `404`
unknown document or graph entity; `413` request envelope, uploaded file, or
extracted text over the configured byte limits
(`request_envelope_too_large` / `ingestion_too_large`); `422` invalid query
controls or malformed graph filters; `429` ingestion rate limit exceeded
(`ingestion_rate_limited`); `502` extraction provider returned unusable
structured output; `503` vector index unavailable, provider unavailable,
traversal safety cap, or fail-closed security/safety failures
(`retrieval_failed`, `context_detector_failed`, `generation_provider_failed`,
audit-persistence codes).

### Migration lineage

```text
dee48bc24a7f  frozen documents/chunks baseline
      ↓
4c9a8d7e6f5b  graph provenance and ingestion lifecycle
      ↓
a6e2c4f8b1d9  database constraint for document lifecycle states
      ↓
b7f3d5a9c2e1  hardened graph extraction lifecycle: idempotent identity owner, attempt counters, pending lease
      ↓
c8a4e6b0d3f2  security provenance and audits: document trust fields, retrieval audits/decisions, ingestion rate buckets
      ↓
c9f5b3e7a1d8  candidate decision CHECK: retrieval_candidate_decisions.decision restricted to the 10B.3 decision enum
      ↓
d9b5f7c1e4a3  safety reviews: safety_review_runs/safety_findings, safety_blocked skip reason, rejected_safety decision value (current head)
```

Application import and API startup do not run migrations. The one-shot
`migrate` Compose service owns schema adoption and upgrades.

### Configuration

All application settings use the `RAG_` prefix and load from `.env`.

| Variable | Application default | Docker example | Description |
|---|---:|---:|---|
| `RAG_LLM_PROVIDER` | `ollama` | `ollama` | Answer provider; Ollama is the supported path |
| `RAG_LLM_BASE_URL` | `http://localhost:11434/v1` | `http://host.docker.internal:11434/v1` | OpenAI-compatible Ollama base URL |
| `RAG_LLM_MODEL` | `gemma4:latest` | `gemma4` | Answer-generation model |
| `RAG_GRAPH_EXTRACTION_ENABLED` | `true` | `true` | Enable structured graph extraction for text ingestion |
| `RAG_GRAPH_EXTRACTION_MODEL` | answer model | `gemma4` | Optional extraction-specific model |
| `RAG_GRAPH_MAX_HOPS` | `2` | `2` | Configured hop bound; HTTP requests use their own validated `graph_max_hops` field (default 2, range 1–3) |
| `RAG_EXTRACTION_LEASE_SECONDS` | `600` | `600` | Graph extraction lease duration in seconds (range 60–3600); an expired pending lease can be reclaimed by a backfill retry |
| `RAG_EMBEDDING_PROVIDER` | `local` | `fastembed` | `local`, `fastembed`, or `openai`; `local` uses deterministic hash embeddings (no model download), `fastembed` produces real 768-dimensional CLIP embeddings |
| `RAG_EMBEDDING_MODEL` | `jinaai/jina-clip-v1` | same | Embedding model (ignored by `local`) |
| `RAG_DATABASE_URL` | `sqlite:///./rag.db` | `sqlite:////data/rag.db` | SQLAlchemy database URL |
| `RAG_CHROMA_HOST` | empty | `vectordb` | Chroma host; empty selects standalone modes |
| `RAG_CHROMA_PORT` | `8000` | `8000` | Chroma server port |
| `RAG_CHROMA_PERSIST_DIRECTORY` | empty | empty | Standalone persistent-client path when host is empty |
| `RAG_OPENAI_API_KEY` | empty | placeholder | Used only by OpenAI-oriented provider paths |
| `RAG_OPERATOR_API_ENABLED` | `false` | `true` | Enable the single-operator API surface (see Scope) |
| `RAG_OPERATOR_TOKEN` | empty | set in `.env` | The one static operator bearer token; minimum 32 characters when enabled |
| `RAG_SECURITY_AUDIT_RETENTION_DAYS` | `30` | `30` | Bounded retention for persisted security audits; prune with `scripts/prune_security_audits.py` |
| `RAG_SOURCE_TRUST_POLICY_PATH` | `config/source-trust-policy.json` | `/app/config/source-trust-policy.json` | Source-trust policy (startup fails when missing/invalid) |
| `RAG_RETRIEVAL_SECURITY_POLICY_PATH` | `config/retrieval-security-policy.json` | `/app/config/retrieval-security-policy.json` | Distance-calibrated retrieval policy; must match the embedding regime |
| `RAG_CONTEXT_SECURITY_POLICY_PATH` | `config/context-security-policy.json` | `/app/config/context-security-policy.json` | Context-security policy (injection controls) |
| `RAG_CONTENT_SAFETY_ENABLED` | `true` | `true` | Content-safety enforcement toggle |
| `RAG_CONTENT_SAFETY_POLICY_PATH` | `config/content-safety-policy.json` | `/app/config/content-safety-policy.json` | Byte-pinned content-safety policy |
| `RAG_SAFETY_LLM_MODE` | provider default | set in `.env` | Safety-review LLM mode for 10C enforcement |

**Embedding regimes and the retrieval-security policy.** The pipeline has two
embedding regimes, selected with `RAG_EMBEDDING_PROVIDER`:

- **fastembed (normalized, the calibrated regime)** — `fastembed` with
  `jinaai/jina-clip-v1` produces normalized real-model vectors. The
  retrieval-security policy (`max_distance` 0.643872, l2) was calibrated under
  exactly this regime, and the API refuses to start (and any process refuses to
  load the policy) with a typed, fail-closed error naming both regimes when the
  runtime regime does not match the policy's calibration — e.g.
  `RAG_EMBEDDING_PROVIDER=local` under the committed policy. Hermetic tests
  that install their own policy through the `_POLICY_CACHE` hook are a full
  bypass of that check.
- **local hash (hermetic/deterministic)** — `local` is a deterministic SHA-256
  hash embedding used by hermetic and deterministic tests. Its vectors are
  unnormalized (l2 distances around 1e1), so every calibrated distance
  threshold is meaningless under it; retrieval under this regime requires a
  policy calibrated (or scoped) for it.

Validator requirements:

- `scripts/validate_phase10b.py` judges exact distance-calibrated retrieval
  outcomes and fails fast with a machine-readable `provider_mismatch` (exit 2)
  unless the live settings resolve `fastembed` plus the policy's calibrated
  model.
- `scripts/validate_phase10a.py` is regime-independent: its deterministic lane
  pins topology/RRF-fusion semantics with the hash embedder under a lane-scoped
  policy, and its live lane asserts ingestion/provider behavior only.
- `scripts/validate_phase10c.py` judges content-safety enforcement and imposes
  no embedding-regime precondition.

## 4. What Goes In — Supported Inputs and Limits

| Input | Support | Behavior |
|---|---|---|
| Text (UTF-8) / Markdown (`.md`, `.markdown`) | Full pipeline | Chunk → embed → extract → graph; empty text and invalid UTF-8 are rejected explicitly |
| PDF | Text layer only, via pypdf | PDFs with extractable text run the full pipeline; **no OCR** — scanned PDFs yield nothing useful and PDFs without extractable text are rejected |
| Images (PNG, JPEG, GIF, WebP) | Vector search only | Embedded into the same 768-dim Jina CLIP space as text (text-to-image retrieval); **not reasoned about** — graph extraction is honestly skipped for images, with no description or OCR |
| Video / audio / EPUB | Not supported | Rejected with `400` |

Ingestion byte and rate limits (enforced before handlers on both the
`Content-Length` fast-reject and streamed-count paths):

| Limit | Default | Setting |
|---|---:|---|
| Uploaded file | 10 MiB (10,485,760 bytes) | `RAG_INGESTION_FILE_MAX_BYTES` |
| Extracted text | 5 MiB (5,242,880 bytes) | `RAG_INGESTION_EXTRACTED_MAX_BYTES` |
| Whole request envelope | 11 MiB (11,534,336 bytes) | `RAG_INGESTION_REQUEST_MAX_BYTES` |
| Ingestion rate | 30 requests / 60 s | `RAG_INGEST_RATE_LIMIT_REQUESTS` / `RAG_INGEST_RATE_LIMIT_WINDOW_SECONDS` |

These settings are raisable (`RAG_INGESTION_REQUEST_MAX_BYTES` up to
53,477,376; file up to 52,428,800; extracted text up to 26,214,400), but the
bounds scale with extraction cost — larger inputs mean more chunks and more
LLM extraction calls. Startup requires the envelope limit to exceed the file
limit.

## 5. Scope

Stated as facts:

- **Single SQLite database and a single Chroma collection** (`rag-collection`)
  on one Compose host; named Docker volumes persist relational data, Chroma
  data, and the FastEmbed model cache.
- **Single-operator, static-token authorization.** The operator API (when
  enabled with `RAG_OPERATOR_API_ENABLED=true`) authenticates exactly one
  static bearer token (`RAG_OPERATOR_TOKEN`); there are no per-user identities,
  no rotation, and no scopes.
- **No multi-tenancy.** All documents, vectors, graph provenance, audits, and
  safety findings live in one shared store with no tenant isolation; any client
  of the API can read all ingested content.
- **No TLS, CORS configuration, monitoring, or load testing** — deferred by
  design (plan E-01 A-02).
- The test suite is validated on a Windows-bare environment only; the
  POSIX-only lanes documented below have never been executed on POSIX.
- The system is not production ready for public, multi-tenant deployment, and
  no separate release certification has been performed.
- The supported generation/extraction path is Ollama; the dummy provider is for
  development, and the OpenAI answer client is currently a placeholder, not a
  production implementation.

## 6. Quickstart

Docker Compose is the authoritative runtime.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Ollama](https://ollama.com/) reachable from Docker
- A Gemma model available in Ollama under the name configured in `.env`
  (the committed example uses `gemma4`; adjust for tags like `gemma4:latest`)

### Configure the environment

```bash
cp .env.example .env
```

Copying `.env.example` enables the production recipe below (the
hermetic/local recipe for tests is documented further down and needs no
services):

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

For an Ollama server on another machine, replace `host.docker.internal` with
that server's hostname or IP address.

### Build and start

```bash
docker compose up --build -d
```

Compose starts three services:

| Service | Role |
|---|---|
| `migrate` | One-shot, adoption-aware Alembic migration; must succeed before API startup |
| `vectordb` | Persistent ChromaDB HTTP server |
| `api` | FastAPI application; starts after migration success and Chroma health |

Endpoints: API health <http://localhost:8000/>, Swagger UI
<http://localhost:8000/docs>, host-only Chroma endpoint
<http://127.0.0.1:8001>. The first embedding request can be slow because
FastEmbed downloads and initializes the ONNX model (~500MB for
`jinaai/jina-clip-v1`); the model cache and both data stores persist in named
volumes, so `docker compose up --force-recreate` reuses the download.

### Ingest documents

```bash
# Inline text
curl -X POST http://localhost:8000/documents \
  -F "title=Architecture Notes" \
  -F "source=manual" \
  -F "tags=rag,architecture" \
  -F "text=Project Helios uses Vector Engine. Vector Engine powers Search Portal."

# Text / Markdown / PDF upload
curl -X POST http://localhost:8000/documents \
  -F "title=RAG Article" \
  -F "source=reference" \
  -F "file=@article.md"

curl -X POST http://localhost:8000/documents \
  -F "title=Research Paper" \
  -F "file=@paper.pdf"

# Image upload
curl -X POST http://localhost:8000/documents \
  -F "title=System Diagram" \
  -F "file=@diagram.png"
```

A successful text response includes the number of chunks and extracted
relationships:

```json
{
  "document_id": 1,
  "chunks": 1,
  "relations": 2
}
```

### Query

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

### Inspect documents and the graph

```bash
curl http://localhost:8000/documents
curl http://localhost:8000/documents/1
curl "http://localhost:8000/graph/entities?name=Aria&match=contains"
```

The operator CLI traverses the persisted relational graph through the
SQL-authoritative traversal and prints complete JSON path objects with
per-step provenance:

```bash
docker compose exec api python src/graph_rag.py "Aria" --hops 3 --limit 10
```

`--hops` accepts `1–3` (default 2), `--direction` accepts `outbound`,
`inbound`, or `both` (default `outbound`), and repeatable `--filters KEY=VALUE`
applies the scalar document filter matrix (`document_id`, `title`, `source`,
`tags`). Unsupported filter keys or malformed `document_id` values exit with an
error.

### Reconcile SQL and Chroma

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

Run reconciliation only against the collection paired with the configured
relational database; it intentionally removes collection records that are not
authoritative ready SQL chunks.

### Backfill graph extractions

```bash
docker compose exec api python scripts/backfill_graph.py --dry-run
docker compose exec api python scripts/backfill_graph.py --document-id 3
docker compose exec api python scripts/backfill_graph.py --retry-failed
```

Each run prints one sorted JSON counter report (`scanned`, `eligible`,
`processed`, `succeeded`, `skipped`, `empty`, `failed`, `lease_lost`,
`relations`, `skip_reasons`). Exit codes: `0` when no chunk failed (including
no-op runs), `1` when one or more chunks failed (successful chunks remain
committed), `2` for invalid arguments, configuration failure, or a fatal
database error.

### Run tests

In the container:

```bash
docker compose exec api python -m pytest -q
```

Without any services — the suite is hermetic (no Ollama, no Chroma server, no
`fastembed` model downloads). From a clean checkout with only the pinned
dependencies installed:

```bash
python -m pytest -q
```

The application defaults already select the hermetic configuration
(`RAG_EMBEDDING_PROVIDER=local`, no Chroma host → ephemeral client). If a
local `.env` selects the production recipe instead, override it for the run:

```bash
RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest -q
```

A clean clone collects the full hermetic suite with zero collection errors —
955 tests, of which 933 pass and 22 are skipped (19 POSIX-only lanes plus 3
opt-in live lanes, each skipped with an explicit reason):

- Opt-in live lanes, executed only when their variable is set:
  `test_rag_api.py::test_query_answer_lane_with_live_llm_optin`
  (`RAG_LIVE_LLM` plus a reachable `RAG_LLM_BASE_URL`), the
  `test_validate_phase10c.py` live-API lane (`RAG_PHASE10C_LIVE_BASE_URL`
  against a running safety-enabled API), and
  `test_named_volume_durability.py::test_validate_named_volume_durability_argv_exact`
  (`RAG_LIVE_DURABILITY` against a live Docker Compose deployment).
- POSIX-only lanes: the retrieval-security calibration runs, the phase-10D
  attack-harness disposable-store lanes, and one symlink-refusal lane pin
  their disposable/production database URLs to POSIX absolute `sqlite:////`
  paths as a destructive-tool path guard; they are skipped on Windows.

### Stop or reset

```bash
# Stop services; named-volume data remains
docker compose down

# Permanently remove relational and Chroma data
docker compose down -v
```

> `docker compose down -v` permanently deletes both the SQLite and Chroma named
> volumes. Re-ingestion is required afterward.

## 7. Evidence Map

Quantitative and behavioral claims in this README and in
[`docs/phase10-evidence.md`](docs/phase10-evidence.md) are anchored with
`[EVID-*]` identifiers; each maps to a row in that document, which cites the
source path/symbol and a hermetic command a clean clone can re-run.

Foundations (prerequisite work):

- `[EVID-F1]` — durable persistent Chroma vector store.
- `[EVID-F2AMENDED]` — runnable graph traversal (F2's original NetworkX choice
  was amended under 10A.5 and removed; production traversal is relational).
- `[EVID-F3]` — truthful document ingestion.
- `[EVID-F4]` — Alembic-owned schema evolution.

10A (core loop):

- `[EVID-10A-01]` — deterministic hybrid RRF retrieval (k=60, never rounded).
- `[EVID-10A-02]` — persisted graph provenance (entities/mentions/edges/evidence).
- `[EVID-10A-03]` — exact durable migration head `d9b5f7c1e4a3`.
- `[EVID-10A-06]` — SQL-authoritative query visibility and reconciliation
  convergence.

10B (security engineering):

- `[EVID-10B-01]` — complete-envelope ingestion byte limits enforced before
  handlers on both the `Content-Length` fast-reject and streamed-count paths.
- `[EVID-10B-02]` — atomic cross-worker rate limiting.
- `[EVID-10B-04]` — persisted security audits with bounded retention, pruned
  by `scripts/prune_security_audits.py`.
- `[EVID-10B-05]` — retrieval-security policy regime pinning that fails closed.
- `[EVID-10B-OPAUTH]` — single-operator static-token authorization surface.

10C (content safety):

- `[EVID-10C-01]` — byte-pinned content-safety policy.
- `[EVID-10C-02]` — fail-closed generation-time safety enforcement; the
  all-candidates-blocked path returns `200` with the deterministic no-safe-
  context answer.

10D (red team):

- `[EVID-10D-01]` — isolated, byte-pinned attack corpus.
- `[EVID-10D-02]` — closed control registry mapped to its owner phases.
- `[EVID-10D-03]` — reproducible report schema; the harness
  (`scripts/run_redteam.py`) runs one-shot containers against disposable
  per-mode SQLite databases and Chroma collections — never the production
  store — and reports are validated against
  `app/tests/fixtures/redteam-report.schema.json`
  (`scripts/validate_redteam_report.py`,
  `scripts/normalize_redteam_report.py`). Methodology and metric definitions
  live in `docs/red-team-methodology.md`.
- `[EVID-10D-04]` — documented defense-effectiveness acceptance bound of
  relative ASR reduction ≥ 0.60 [EVID-10D-04].

DOC.1 (documentation and image hygiene):

- `[EVID-DOC1-04]`, `[EVID-DOC1-05]`, `[EVID-DOC1-06]`, `[EVID-DOC1-07]` —
  `scripts/validate_image_hygiene.py`, a host-side scanner (it refuses to run
  inside a container) that verifies the built images' OCI source-provenance
  labels and scans the final merged rootfs of each service — via
  `docker create --entrypoint /bin/true` plus `docker export` — for forbidden
  artifacts (`.git`, `.env*`, `.hermes`, local databases, report files, host
  absolute paths, credential sentinels, out-of-allowlist attack fixtures) and
  verifies the pinned policy hash inventory. It never mounts the Docker socket
  and never echoes matched credential bytes:

  ```bash
  python scripts/validate_image_hygiene.py --manifest <source-manifest.json> --services api migrate
  ```

  Its behavior is pinned by 31 hermetic lanes with zero Docker dependency.
- `[EVID-DOC1-SUITE]` — the hermetic suite collection/passage counts and the
  bare-Windows validation of the evidence map itself.
