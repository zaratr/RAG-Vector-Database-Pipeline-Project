# Local RAG Pipeline with Gemma Generation

A standard retrieve-augmented generation pipeline running entirely against
local models: FastEmbed dense embeddings persisted in ChromaDB, with a
pluggable LLM client (Ollama / Gemma by default) for the generation step.
Ingest markdown/text/documents → chunk → embed → store → retrieve top-k →
generate.

A separate NetworkX script (`src/graph_rag.py`) is included as a **research
spike** that sketches how a graph layer would look; see the _Graph layer
spike_ section below for its (intentionally limited) scope.

## What is actually implemented

| component | status | location |
|---|---|---|
| Document ingestion + persistence | implemented | `app/services/ingestion.py`, `app/persistence/` |
| Text chunking | implemented | `app/services/chunking.py` |
| FastEmbed embedding provider | implemented | `app/services/embeddings.py` |
| ChromaDB vector store | implemented | `app/services/vector_store.py` |
| Top-k retrieval | implemented | `app/services/retrieval.py` |
| RAG orchestration (retrieve → generate) | implemented | `app/services/rag.py` |
| Pluggable LLM client (Ollama/Gemma) | scaffolded (thin) | `app/services/llm.py` |
| FastAPI surface (documents + query) | implemented | `app/api/routes_documents.py`, `app/api/routes_query.py`, `app/main.py` |
| Tests (ingestion, retrieval, RAG API) | implemented | `app/tests/` |

### Pipeline shape

```
document → chunking → FastEmbed embed → ChromaDB upsert
                                                ↓
query    → FastEmbed embed → top-k retrieve → Gemma generate → answer
```

## Graph layer spike (NOT wired into the pipeline)

An earlier headline framed this repository as **GraphRAG** with multi-hop
knowledge-graph reasoning over extracted (Entity, Relationship, Entity)
triplets. The code does not deliver that end-to-end. What exists is
`src/graph_rag.py`: a 30-line **isolated spike** that:

- imports `networkx` and constructs an in-memory `DiGraph`,
- **returns hardcoded mock triplets** from `extract_entities_with_gemma`
  (the Ollama call is stubbed in a comment), and
- is **not imported by anything under `app/`** — it does not participate
  in ingestion or query.

So the live pipeline is **standard vector RAG**, not GraphRAG. Multi-hop
reasoning, entity extraction at ingest time, and graph-augmented retrieval
are **not implemented**. The spike is kept as a starting point for future
work, not as a claim of capability.

## What is NOT implemented (explicitly)

- **No multi-hop reasoning.** Retrieval is single-shot top-k vector search.
- **No live entity extraction.** `src/graph_rag.py` returns canned triplets;
  no document is parsed into a graph at ingest time.
- **No graph-augmented retrieval.** The ChromaDB path and the NetworkX spike
  do not share an index.

## Roadmap

1. Wire `extract_entities_with_gemma` in `src/graph_rag.py` to a real
   Ollama/Gemma call producing triplets from chunk text.
2. Persist triplets alongside chunks at ingest time and expose a graph
   index that the retriever can consult.
3. Add a hybrid retriever that expands seed entities via graph neighbours
   before falling back to dense top-k.

## Tech stack

- **Backend:** FastAPI, Python
- **Embeddings:** FastEmbed
- **Vector store:** ChromaDB
- **Generation:** Ollama (Gemma)
- **Graph spike (unwired):** NetworkX
