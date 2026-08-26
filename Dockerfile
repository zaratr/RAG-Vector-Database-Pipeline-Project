FROM python:3.11-slim

# Source provenance build arguments. Defaults are placeholders; real builds
# may supply validated revision scalars to stamp into the labels below.
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

# Copy only what the image uses: the application package (including its
# tests), the Alembic configuration plus in-package migration scripts used
# by the one-shot migrator, the operator scripts, the graph traversal CLI,
# and the pytest configuration for the documented in-container test run.
# Local working-tree files (env files, databases, query payloads, diagrams)
# are never baked into the image.
COPY alembic.ini pytest.ini ./
COPY app/ app/
COPY scripts/ scripts/
COPY src/ src/

ENV RAG_EMBEDDING_PROVIDER=local \
    RAG_VECTOR_STORE=chroma

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
