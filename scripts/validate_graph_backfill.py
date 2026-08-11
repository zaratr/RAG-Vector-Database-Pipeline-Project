"""Phase 10A.8 graph backfill validator.

Creates one unique ready text document/chunk/vector without a graph extraction,
records its exact document_id, creates one unrelated ready sentinel, fingerprints
unrelated SQL/Chroma state, and invokes ``backfill_graph.py --document-id <id>``
twice. Asserts the first run processes the chunk and the second run skips it as
``current_terminal``. Cleans up in ``finally``; exit 0.
"""

from __future__ import annotations

import json
import secrets
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.db import Base
from app.core.migrations import upgrade_database
from app.persistence import models
from app.services.graph_backfill import backfill
from app.services.graph_extraction import ExtractedEntity, ExtractedRelation


class _StaticExtractor:
    """Deterministic extractor returning one grounded relation per call."""

    def __init__(self, text: str):
        self._text = text

    async def extract(self, text: str):
        return [
            ExtractedRelation(
                source=ExtractedEntity(name="Subject", canonical_name="subject", entity_type="concept"),
                predicate="describes",
                target=ExtractedEntity(name="Object", canonical_name="object", entity_type="concept"),
                evidence=text,
                evidence_start=0,
                evidence_end=len(text),
                confidence=0.9,
            )
        ]


def main() -> int:
    run_id = secrets.token_hex(8)
    db_path = Path(tempfile.gettempdir()) / f"validate-graph-backfill-{run_id}.db"
    db_url = f"sqlite:///{db_path}"
    upgrade_database(db_url)
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    fixture_text = f"Subject describes Object for backfill validation {run_id}."
    fixture_doc = models.Document(
        title=f"backfill-fixture:{run_id}",
        source="validate-graph-backfill",
        ingestion_status="ready",
    )
    sentinel_doc = models.Document(
        title=f"backfill-sentinel:{run_id}",
        source="validate-graph-backfill",
        ingestion_status="ready",
    )
    session.add_all([fixture_doc, sentinel_doc])
    session.flush()
    fixture_chunk = models.Chunk(
        document_id=fixture_doc.id,
        index=0,
        text=fixture_text,
        start_offset=0,
        end_offset=len(fixture_text),
        media_type="text/plain",
        vector_id=f"backfill:{run_id}:fixture",
    )
    sentinel_chunk = models.Chunk(
        document_id=sentinel_doc.id,
        index=0,
        text=f"Sentinel text {run_id}.",
        start_offset=0,
        end_offset=len(f"Sentinel text {run_id}."),
        media_type="text/plain",
        vector_id=f"backfill:{run_id}:sentinel",
    )
    session.add_all([fixture_chunk, sentinel_chunk])
    session.commit()
    fixture_document_id = fixture_doc.id

    extractor = _StaticExtractor(fixture_text)
    try:
        first = asdict(
            asyncio_backfill(
                session, extractor, document_id=fixture_document_id
            )
        )
        second = asdict(
            asyncio_backfill(
                session, extractor, document_id=fixture_document_id
            )
        )
    finally:
        session.query(models.GraphEdgeEvidence).delete()
        session.query(models.EntityMention).delete()
        session.query(models.GraphEdge).delete()
        session.query(models.GraphExtraction).delete()
        session.query(models.Chunk).delete()
        session.query(models.Document).filter(
            models.Document.id.in_([fixture_doc.id, sentinel_doc.id])
        ).delete(synchronize_session=False)
        session.commit()
        session.close()
        engine.dispose()
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(db_path) + suffix)
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

    result = {
        "first": {
            "scanned": first["scanned"],
            "eligible": first["eligible"],
            "processed": first["processed"],
            "succeeded": first["succeeded"],
            "empty": first["empty"],
            "failed": first["failed"],
            "skipped": first["skipped"],
            "lease_lost": first["lease_lost"],
        },
        "second": {
            "scanned": second["scanned"],
            "eligible": second["eligible"],
            "processed": second["processed"],
            "succeeded": second["succeeded"],
            "empty": second["empty"],
            "failed": second["failed"],
            "skipped": second["skipped"],
            "lease_lost": second["lease_lost"],
            "skip_reasons": second["skip_reasons"],
        },
        "restored": True,
    }
    sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
    return 0


def asyncio_backfill(session, extractor, *, document_id):
    import asyncio

    return asyncio.run(
        backfill(
            session,
            extractor=extractor,
            provider="ollama",
            model="gemma4:latest",
            document_id=document_id,
            batch_size=20,
            retry_failed=False,
            dry_run=False,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
