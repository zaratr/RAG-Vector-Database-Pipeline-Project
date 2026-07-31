"""Reconcile relational ingestion state with the Chroma collection."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.db import session_scope
from app.services.embeddings import get_embedding_provider
from app.services.reconciliation import reconcile_ingestion
from app.services.vector_store import get_vector_store


async def _main() -> None:
    with session_scope() as session:
        result = await reconcile_ingestion(
            session=session,
            embedding_provider=get_embedding_provider(),
            vector_store=get_vector_store(),
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(_main())
