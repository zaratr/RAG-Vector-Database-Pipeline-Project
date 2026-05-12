# GraphRAG & Knowledge Graph Pipeline

## ?? 2026 Architecture Modernization
Standard semantic search (chunk & embed) is insufficient for complex reasoning. This repository implements **GraphRAG**—a paradigm where knowledge graphs are mapped alongside vector spaces to enable multi-hop reasoning.

### Key Features
1. **Dynamic Knowledge Graphs:** Utilizes NetworkX to construct directional relationship graphs from unstructured documents.
2. **Local LLM Entity Extraction:** Uses local inference (Ollama + Gemma) to dynamically extract (Entity, Relationship, Entity) triplets on ingestion.
3. **Multi-Hop Querying:** When a user queries the system, the pipeline traverses the Knowledge Graph to retrieve deep context before generating an answer, drastically reducing hallucination on complex queries.

## ??? Tech Stack
*   **AI / ML:** Ollama (Gemma), GraphRAG, FastEmbed
*   **Data Structures:** NetworkX, ChromaDB
*   **Backend:** FastAPI, Python
