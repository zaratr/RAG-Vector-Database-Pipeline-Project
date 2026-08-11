FROM python:3.11-slim

# Phase 10 source provenance build arguments (Task 10.0).
# Parse-only defaults that every approval rejects; real builds supply validated
# scalars from scripts/source_manifest.py.
ARG SOURCE_REVISION=unknown
ARG SOURCE_CONTEXT_SHA256=unknown
ARG SOURCE_DIRTY=unknown

LABEL org.opencontainers.image.revision="${SOURCE_REVISION}"
LABEL org.opencontainers.image.source-context-sha256="${SOURCE_CONTEXT_SHA256}"
LABEL org.opencontainers.image.source-dirty="${SOURCE_DIRTY}"

RUN apt-get update && apt-get install -y --no-install-recommends vim git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV RAG_EMBEDDING_PROVIDER=local \
    RAG_VECTOR_STORE=chroma

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
