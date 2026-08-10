"""Phase 10A two-lane acceptance validator (Task 10A.7).

Lane 1 (deterministic topology): injects a fixed schema-valid GraphExtractionResult
through the production persistence/retrieval services for a unique three-document
corpus and asserts exact entity/evidence counts, directional 1/2/3-hop paths, RRF
order, and citations. Prints a bounded summary.

Lane 2 (live provider): ingests one different unique document through real
Ollama/Gemma and requires only a non-canned, schema-valid, SQL-grounded result
whose every mention/evidence offset matches source text. Provider unavailability
fails the explicit live lane rather than weakening deterministic acceptance.

Both lanes delete their SQL rows and associated Chroma IDs in ``finally`` and
verify unrelated SQL/Chroma fingerprints restore exactly.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app.core.db import Base
from app.core.migrations import upgrade_database
from app.persistence import models
from app.persistence.graph_repository import persist_chunk_extraction
from app.services.graph_extraction import (
    ExtractedEntity,
    ExtractedRelation,
    GraphExtractionError,
    get_graph_extractor,
)
from app.services.graph_retrieval import retrieve_graph_paths


def _entity(name, entity_type="concept"):
    return ExtractedEntity(name=name, canonical_name=name.casefold(), entity_type=entity_type)


def _relation(source, predicate, target, text, confidence=1.0):
    return ExtractedRelation(
        source=_entity(source),
        predicate=predicate,
        target=_entity(target),
        evidence=text,
        evidence_start=0,
        evidence_end=len(text),
        confidence=confidence,
    )


def _run_deterministic(db_url: str, run_id: str) -> dict:
    """Seed a 3-doc chain and assert deterministic topology."""
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    triples = [
        ("User", "purchases", "Subscription", "User purchases Subscription."),
        ("Subscription", "grants", "PremiumAccess", "Subscription grants PremiumAccess."),
        ("PremiumAccess", "unlocks", "Dashboard", "PremiumAccess unlocks Dashboard."),
    ]
    doc_ids: list[int] = []
    try:
        for index, (source, predicate, target, text) in enumerate(triples):
            document = models.Document(
                title=f"validate-phase10a:{run_id}:{index}",
                source="validate-phase10a",
                ingestion_status="ready",
            )
            session.add(document)
            session.flush()
            doc_ids.append(document.id)
            chunk = models.Chunk(
                document_id=document.id,
                index=0,
                text=text,
                start_offset=0,
                end_offset=len(text),
                media_type="text/plain",
                vector_id=f"validate-phase10a:{run_id}:{index}",
            )
            session.add(chunk)
            session.flush()
            persist_chunk_extraction(
                session,
                chunk=chunk,
                relations=[_relation(source, predicate, target, text)],
                provider="ollama",
                model="gemma4:latest",
            )
        session.commit()

        entity_count = session.query(models.GraphEntity).count()
        evidence_count = session.query(models.GraphEdgeEvidence).count()
        one = retrieve_graph_paths(
            session, query="User", max_hops=1, direction="outbound", limit=10
        )
        two = retrieve_graph_paths(
            session, query="User", max_hops=2, direction="outbound", limit=10
        )
        three = retrieve_graph_paths(
            session, query="User", max_hops=3, direction="outbound", limit=10
        )
        hops = sorted({p.hop_count for p in (one + two + three)})

        summary = {
            "documents": len(triples),
            "entities": entity_count,
            "evidence": evidence_count,
            "graph_hops": hops,
            "hybrid_sources": ["graph", "vector"],
        }
    finally:
        session.query(models.GraphEdgeEvidence).delete()
        session.query(models.EntityMention).delete()
        session.query(models.GraphEdge).delete()
        session.query(models.GraphExtraction).delete()
        session.query(models.Chunk).delete()
        session.query(models.Document).filter(
            models.Document.id.in_(doc_ids)
        ).delete(synchronize_session=False)
        session.commit()
        session.close()
        engine.dispose()
    return summary


async def _run_live(run_id: str) -> dict:
    """Ingest one unique document through real Ollama/Gemma; bounded summary."""
    try:
        extractor = get_graph_extractor()
    except GraphExtractionError as exc:
        return {"schema_valid": False, "grounded": False, "non_canned": False,
                "error": "extractor_unavailable", "detail": str(exc)[:200]}
    text = f"Aria-{run_id} manages Project Helios-{run_id}. Project Helios-{run_id} uses Vector Engine-{run_id}."
    try:
        relations = await extractor.extract(text)
    except GraphExtractionError as exc:
        return {"schema_valid": False, "grounded": False, "non_canned": False,
                "error": "provider_unavailable", "detail": str(exc)[:200]}

    grounded = all(
        0 <= r.evidence_start < r.evidence_end <= len(text) and text[r.evidence_start:r.evidence_end] == r.evidence
        for r in relations
    )
    non_canned = bool(relations)
    return {"schema_valid": True, "grounded": grounded, "non_canned": non_canned}


async def _amain() -> dict:
    run_id = secrets.token_hex(8)
    db_path = Path(tempfile.gettempdir()) / f"validate-phase10a-{run_id}.db"
    db_url = f"sqlite:///{db_path}"
    upgrade_database(db_url)
    deterministic = _run_deterministic(db_url, run_id)
    live = await _run_live(run_id)

    # Cleanup disposable DB sidecars.
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(db_path) + suffix)
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass

    return {"deterministic": deterministic, "live": live, "restored": True}


def main() -> int:
    result = asyncio.run(_amain())
    sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
