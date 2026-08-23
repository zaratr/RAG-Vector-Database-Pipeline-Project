"""Phase 10A two-lane acceptance validator (Task 10A.7).

Lane 1 (deterministic topology): injects a fixed schema-valid GraphExtractionResult
through the production persistence/retrieval services for a unique three-document
corpus into a disposable migrated SQLite database plus a disposable in-process
(ephemeral) Chroma collection, and ASSERTS exact entity/evidence counts,
directional 1/2/3-hop paths with full citations, genuinely executed hybrid
retrieval (sources derived from the real fused results), and RRF-60 fusion
semantics (each fused hybrid_score must equal the sum of 1/(60+rank) over the
sides that returned the chunk, with ranks derived from real vector-only and
graph-only ``retrieve_contexts`` calls; a chunk retrievable by both sides must
outrank every single-side chunk at fused top_k=2).

Lane 2 (live provider): ingests one different unique document through the
PRODUCTION ingestion path (``app.services.ingestion.ingest_text``) with the
real Ollama/Gemma extractor, a disposable ephemeral Chroma collection, and the
production local (hash) embedding provider — the image's own
``RAG_EMBEDDING_PROVIDER=local`` default — so provider dependence is isolated
to Ollama/Gemma exactly as the plan names it. It requires a non-canned
(no token-free canned echo: some persisted surface/evidence must contain the
unique run token), schema-valid, SQL-grounded result whose every mention and
evidence offset matches the persisted source text, and a document that reached
``ready`` through the 10A.4 state machine. Provider unavailability or invalid
output FAILS the explicit live lane (exit 2) rather than weakening
deterministic acceptance.

Both lanes delete their SQL rows and associated Chroma IDs in ``finally`` and
verify their disposable row-count/PK fingerprints restore exactly; the
configured production SQL and Chroma-ID fingerprints are captured read-only
before and after the whole run and must restore exactly. ``"restored": true``
is emitted only when every restoration check genuinely passed.

Exit codes (parsed by the recorded phase10a gate):

* ``0`` — success; stdout is exactly the plan-pinned summary JSON.
* ``1`` — acceptance assertion failure (machine-readable JSON on stderr).
* ``2`` — provider/configuration/infrastructure failure, including provider
  unavailability and unreadable configured production fingerprints.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import sqlite3
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.core.db import Base
from app.core.migrations import upgrade_database
from app.persistence import models
from app.persistence.graph_repository import persist_chunk_extraction
from app.services.embeddings import HashEmbeddingProvider
from app.services.graph_extraction import (
    DisabledGraphExtractor,
    ExtractedEntity,
    ExtractedRelation,
    GraphExtractionError,
    GraphProviderOutputError,
    GraphProviderUnavailable,
    get_graph_extractor,
)
from app.services.graph_retrieval import retrieve_graph_paths
from app.services.ingestion import ingest_text
from app.services.retrieval import retrieve_contexts
from app.services.vector_store import ChromaVectorStore

# Plan 10A.6 pins the reciprocal-rank-fusion constant at 60.
RRF_K = 60

# Plan-pinned deterministic expectations for the seeded 3-hop chain
# User -purchases-> Subscription -grants-> PremiumAccess -unlocks-> Dashboard:
# 3 documents, 4 distinct entities, 3 edge-evidence rows, hops 1..3 present.
EXPECTED_DOCUMENTS = 3
EXPECTED_ENTITIES = 4
EXPECTED_EVIDENCE = 3
EXPECTED_HOPS = [1, 2, 3]

# Hybrid query chosen so lexical seed resolution matches every seeded entity;
# hybrid's graph side then has the same seed set as a graph-only derivation
# call (lexical seeds plus vector-derived seeds collapse to the same set).
HYBRID_QUERY = "User purchases Subscription grants PremiumAccess unlocks Dashboard"
HYBRID_MAX_HOPS = 2

# The seeded chain yields exactly 5 outbound paths within 2 hops; every hybrid
# internal graph limit used below (top_k=2 -> 6, top_k=10 -> 30) exceeds this,
# so per-side candidate sets are limit-independent. Soundness is asserted.
_MAX_HYBRID_PATHS_FOR_LIMIT_INDEPENDENCE = 6

_EXIT2_CODES = {
    "provider_unavailable",
    "provider_output_error",
    "configuration_error",
    "fingerprint_unreadable",
    "live_ingestion_error",
}

_PRODUCTION_COLLECTION = "rag-collection"


class ValidatorFailure(RuntimeError):
    """Machine-readable acceptance failure; the gate parses the exit code."""

    def __init__(self, code: str, lane: str, detail: dict | None = None):
        self.code = code
        self.lane = lane
        self.detail = {key: str(value)[:200] for key, value in (detail or {}).items()}
        if "check" not in self.detail:
            self.detail["check"] = code
        super().__init__(f"{code}:{lane}:{self.detail.get('check')}")


def _check(lane: str, check: str, condition: bool, **detail) -> None:
    if not condition:
        raise ValidatorFailure("assertion_failed", lane, {"check": check, **detail})


def _rrf_score(ranks: dict) -> float:
    """Reciprocal-rank-fusion score with the plan-pinned constant ``RRF_K``.

    ``ranks`` maps side name -> 1-based rank of one candidate on that side;
    a candidate retrieved by a side at rank ``r`` contributes 1/(RRF_K + r).
    Production emits the exact sum (no rounding), so the mirror does too.
    """
    return sum(1.0 / (RRF_K + rank) for rank in ranks.values())


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


def _disposable_store(run_id: str, label: str):
    """Disposable in-process (ephemeral) Chroma collection; never the
    configured production client, host, or collection.

    Uses the plain default ``EphemeralClient()`` like the rest of the
    codebase: chromadb keys its shared system by settings, so a custom
    Settings object here would break later default-client creation in the
    same process (observed as cross-module fixture errors).
    """
    import chromadb

    client = chromadb.EphemeralClient()
    name = f"validate-phase10a-{label}-{run_id}"
    return name, client, ChromaVectorStore(collection_name=name, client=client)


def _db_fingerprint(engine) -> dict:
    """Row-count/PK fingerprint: [row count, max id] per application table."""
    inspector = inspect(engine)
    fingerprint: dict[str, list] = {}
    with engine.connect() as conn:
        for table in sorted(inspector.get_table_names()):
            if table == "alembic_version":
                continue
            count = conn.exec_driver_sql(f"SELECT COUNT(*) FROM {table}").scalar()
            try:
                max_id = conn.exec_driver_sql(f"SELECT MAX(id) FROM {table}").scalar()
            except Exception:  # pragma: no cover - tables without an id PK
                max_id = None
            fingerprint[table] = [count, max_id]
    return fingerprint


def _delete_lane_rows(session, doc_ids: list[int]) -> None:
    """Delete every SQL row a lane created (disposable DBs are lane-exclusive).

    GraphEntity rows are deleted too: they are shared/global rows created by
    the lane's extractions, and leaving them would leak lane state (the old
    script's cleanup missed them — its hardcoded ``restored`` hid the leak).
    """
    session.query(models.GraphEdgeEvidence).delete()
    session.query(models.EntityMention).delete()
    session.query(models.GraphEdge).delete()
    session.query(models.GraphEntity).delete()
    session.query(models.GraphExtraction).delete()
    session.query(models.Chunk).delete()
    session.query(models.Document).filter(
        models.Document.id.in_(doc_ids)
    ).delete(synchronize_session=False)
    session.commit()


def _chunk_id(context: dict) -> int | None:
    try:
        return int((context.get("metadata") or {})["chunk_id"])
    except (KeyError, TypeError, ValueError):
        return None


def _rank_by_chunk(contexts: list[dict]) -> dict[int, int]:
    return {
        cid: rank for rank, cid in enumerate((_chunk_id(c) for c in contexts), start=1)
    }


async def _run_deterministic(db_url: str, run_id: str) -> tuple[dict, bool]:
    """Seed the 3-doc chain, ASSERT topology/hybrid semantics, self-clean."""
    lane = "deterministic"
    upgrade_database(db_url)
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    collection_name, client, store = _disposable_store(run_id, "det")
    baseline = _db_fingerprint(engine)

    triples = [
        ("User", "purchases", "Subscription", "User purchases Subscription."),
        ("Subscription", "grants", "PremiumAccess", "Subscription grants PremiumAccess."),
        ("PremiumAccess", "unlocks", "Dashboard", "PremiumAccess unlocks Dashboard."),
    ]
    doc_ids: list[int] = []
    chunk_ids: list[int] = []
    chunk_texts: dict[int, str] = {}
    chunk_models: dict[int, models.Chunk] = {}
    vector_ids: list[str] = []
    restored = False
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
            chunk_ids.append(chunk.id)
            chunk_texts[chunk.id] = text
            chunk_models[chunk.id] = chunk
            vector_ids.append(chunk.vector_id)
            persist_chunk_extraction(
                session,
                chunk=chunk,
                relations=[_relation(source, predicate, target, text)],
                provider="ollama",
                model="gemma4:latest",
            )
        session.commit()

        # --- Exact count assertions (derived from SQL, never printed only). ---
        documents = (
            session.query(models.Document)
            .filter(models.Document.id.in_(doc_ids))
            .count()
        )
        entity_count = session.query(models.GraphEntity).count()
        evidence_count = session.query(models.GraphEdgeEvidence).count()
        _check(lane, "documents", documents == EXPECTED_DOCUMENTS,
               expected=EXPECTED_DOCUMENTS, actual=documents)
        _check(lane, "entities", entity_count == EXPECTED_ENTITIES,
               expected=EXPECTED_ENTITIES, actual=entity_count)
        _check(lane, "evidence", evidence_count == EXPECTED_EVIDENCE,
               expected=EXPECTED_EVIDENCE, actual=evidence_count)

        # --- Directional 1/2/3-hop path assertions with full citations. ---
        one = retrieve_graph_paths(
            session, query="User", max_hops=1, direction="outbound", limit=10
        )
        two = retrieve_graph_paths(
            session, query="User", max_hops=2, direction="outbound", limit=10
        )
        three = retrieve_graph_paths(
            session, query="User", max_hops=3, direction="outbound", limit=10
        )
        _check(lane, "hop1_set", sorted({p.hop_count for p in one}) == [1])
        _check(lane, "hop2_set", sorted({p.hop_count for p in two}) == [1, 2])
        _check(lane, "hop3_set", sorted({p.hop_count for p in three}) == [1, 2, 3])

        three_hop_paths = [p for p in three if p.hop_count == 3]
        _check(lane, "three_hop_path_present", len(three_hop_paths) >= 1)
        path = three_hop_paths[0]
        _check(lane, "three_hop_predicates",
               [s.predicate for s in path.steps] == ["purchases", "grants", "unlocks"],
               actual=[s.predicate for s in path.steps])
        _check(lane, "three_hop_chain",
               [s.source for s in path.steps] == ["User", "Subscription", "PremiumAccess"]
               and [s.target for s in path.steps] == ["Subscription", "PremiumAccess", "Dashboard"],
               actual=[(s.source, s.target) for s in path.steps])
        for paths in (one, two, three):
            for hop_path in paths:
                for step in hop_path.steps:
                    _check(lane, "citation_document",
                           step.document_id in doc_ids, document_id=step.document_id)
                    _check(lane, "citation_chunk",
                           step.chunk_id in chunk_texts, chunk_id=step.chunk_id)
                    _check(lane, "citation_evidence_text",
                           step.evidence == chunk_texts[step.chunk_id])
                    _check(lane, "citation_evidence_ids", step.evidence_id > 0 and step.edge_id > 0)
                    _check(lane, "citation_extraction_model",
                           step.extraction_model == "gemma4:latest",
                           actual=step.extraction_model)

        # --- Genuinely executed hybrid retrieval + RRF-60 fusion semantics. ---
        embedder = HashEmbeddingProvider()
        embeddings = await embedder.embed_texts([chunk_texts[cid] for cid in chunk_ids])
        await store.upsert_embeddings(
            embeddings,
            [chunk_models[cid].get_chunk_metadata() for cid in chunk_ids],
            vector_ids,
            documents=[chunk_texts[cid] for cid in chunk_ids],
        )

        hybrid_all = await retrieve_contexts(
            query=HYBRID_QUERY, embedding_provider=embedder, vector_store=store,
            session=session, mode="hybrid", top_k=10, graph_max_hops=HYBRID_MAX_HOPS,
        )
        vector_ctxs = await retrieve_contexts(
            query=HYBRID_QUERY, embedding_provider=embedder, vector_store=store,
            session=session, mode="vector", top_k=10,
        )
        graph_ctxs = await retrieve_contexts(
            query=HYBRID_QUERY, embedding_provider=embedder, vector_store=store,
            session=session, mode="graph", top_k=10, graph_max_hops=HYBRID_MAX_HOPS,
        )

        _check(lane, "hybrid_executed", len(hybrid_all) == len(chunk_ids),
               expected=len(chunk_ids), actual=len(hybrid_all))
        sources = sorted(
            {source for ctx in hybrid_all for source in ctx["metadata"]["retrieval_sources"]}
        )
        _check(lane, "hybrid_sources_graph_and_vector",
               "graph" in sources and "vector" in sources, actual=sources)

        # Derivation soundness: hybrid's internal graph limits (6 for top_k=2,
        # 30 for top_k=10) must not truncate the path set, so per-side
        # candidate lists are identical across the calls compared below.
        path_count = len(retrieve_graph_paths(
            session, query=HYBRID_QUERY, max_hops=HYBRID_MAX_HOPS,
            direction="outbound", limit=50,
        ))
        _check(lane, "hybrid_limit_independence",
               path_count <= _MAX_HYBRID_PATHS_FOR_LIMIT_INDEPENDENCE,
               actual=path_count)

        vrank = _rank_by_chunk(vector_ctxs)
        grank = _rank_by_chunk(graph_ctxs)

        def _expected_score(chunk_id: int) -> float:
            ranks = {}
            if chunk_id in vrank:
                ranks["vector"] = vrank[chunk_id]
            if chunk_id in grank:
                ranks["graph"] = grank[chunk_id]
            return _rrf_score(ranks)

        # RRF-60: every fused score equals sum(1/(60+rank)) over its sides.
        for ctx in hybrid_all:
            cid = _chunk_id(ctx)
            _check(lane, "rrf60_score", cid is not None)
            expected = _expected_score(cid)
            _check(lane, "rrf60_score",
                   ctx["metadata"]["hybrid_score"] == expected,
                   chunk_id=cid, expected=expected,
                   actual=ctx["metadata"]["hybrid_score"])
        scores = [ctx["metadata"]["hybrid_score"] for ctx in hybrid_all]
        _check(lane, "rrf60_order", scores == sorted(scores, reverse=True), actual=scores)

        # A chunk retrievable by BOTH sides must outrank single-side chunks:
        # at fused top_k=2 the two both-sides chunks win and the graph-only
        # chunk (absent from the vector top-2) is demoted below them.
        vector_top2 = await retrieve_contexts(
            query=HYBRID_QUERY, embedding_provider=embedder, vector_store=store,
            session=session, mode="vector", top_k=2,
        )
        hybrid_top2 = await retrieve_contexts(
            query=HYBRID_QUERY, embedding_provider=embedder, vector_store=store,
            session=session, mode="hybrid", top_k=2, graph_max_hops=HYBRID_MAX_HOPS,
        )
        vector2_ids = {_chunk_id(ctx) for ctx in vector_top2}
        graph_ids = {_chunk_id(ctx) for ctx in graph_ctxs}
        graph_only = graph_ids - vector2_ids
        _check(lane, "hybrid_single_side_chunk_exists", len(graph_only) == 1,
               actual=sorted(graph_only))
        top2_ids = {_chunk_id(ctx) for ctx in hybrid_top2}
        _check(lane, "rrf60_both_sides_outrank_single_side",
               top2_ids == vector2_ids and not (graph_only & top2_ids),
               expected=sorted(vector2_ids), actual=sorted(top2_ids))
        vrank2 = _rank_by_chunk(vector_top2)
        for ctx in hybrid_top2:
            cid = _chunk_id(ctx)
            _check(lane, "rrf60_both_sides_sources",
                   set(ctx["metadata"]["retrieval_sources"]) == {"vector", "graph"},
                   chunk_id=cid,
                   actual=ctx["metadata"]["retrieval_sources"])
            expected = _rrf_score({"vector": vrank2[cid], "graph": grank[cid]})
            _check(lane, "rrf60_both_sides_score",
                   ctx["metadata"]["hybrid_score"] == expected,
                   chunk_id=cid, expected=expected,
                   actual=ctx["metadata"]["hybrid_score"])

        hops = sorted({p.hop_count for p in (one + two + three)})
        _check(lane, "graph_hops", hops == EXPECTED_HOPS, actual=hops)
        summary = {
            "documents": documents,
            "entities": entity_count,
            "evidence": evidence_count,
            "graph_hops": hops,
            "hybrid_sources": sources,
        }
    finally:
        session.rollback()
        try:
            await store.delete(vector_ids)
            client.delete_collection(collection_name)
        except Exception:  # pragma: no cover - best-effort disposable cleanup
            pass
        _delete_lane_rows(session, doc_ids)
        session.close()
        engine.dispose()
        try:
            fresh = create_engine(db_url)
            restored = _db_fingerprint(fresh) == baseline
            fresh.dispose()
        except Exception:  # pragma: no cover - fingerprint must be readable
            restored = False
    return summary, restored


async def _run_live(db_url: str, run_id: str) -> tuple[dict, bool]:
    """Ingest one unique document through the production ingestion path with
    the real Ollama/Gemma extractor; ASSERT the persisted SQL result."""
    lane = "live"
    upgrade_database(db_url)
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    collection_name, client, store = _disposable_store(run_id, "live")
    baseline = _db_fingerprint(engine)

    try:
        try:
            extractor = get_graph_extractor()
        except GraphExtractionError as exc:
            raise ValidatorFailure("configuration_error", lane, {"detail": str(exc)[:200]}) from exc
        if isinstance(extractor, DisabledGraphExtractor):
            raise ValidatorFailure(
                "configuration_error", lane, {"detail": "graph extraction disabled"}
            )

        text = (
            f"Aria-{run_id} manages Project Helios-{run_id}. "
            f"Project Helios-{run_id} uses Vector Engine-{run_id}."
        )
        try:
            result = await ingest_text(
                title=f"validate-phase10a-live:{run_id}",
                source="validate-phase10a",
                tags=None,
                text=text,
                embedding_provider=HashEmbeddingProvider(),
                vector_store=store,
                session=session,
                graph_extractor=extractor,
            )
        except GraphProviderUnavailable as exc:
            raise ValidatorFailure(
                "provider_unavailable", lane, {"detail": str(exc)[:200]}
            ) from exc
        except GraphProviderOutputError as exc:
            raise ValidatorFailure(
                "provider_output_error", lane, {"detail": str(exc)[:200]}
            ) from exc
        except GraphExtractionError as exc:
            raise ValidatorFailure(
                "provider_unavailable", lane, {"detail": str(exc)[:200]}
            ) from exc
        except Exception as exc:
            raise ValidatorFailure(
                "live_ingestion_error", lane,
                {"detail": f"{type(exc).__name__}: {exc}"[:200]},
            ) from exc

        document = session.get(models.Document, result["document_id"])
        _check(lane, "ingestion_result_chunks", result["chunks"] >= 1)
        _check(lane, "document_ready",
               document is not None and document.ingestion_status == "ready",
               actual=getattr(document, "ingestion_status", None))

        chunks = (
            session.query(models.Chunk)
            .filter(models.Chunk.document_id == document.id)
            .all()
        )
        chunk_text_by_id = {chunk.id: chunk.text for chunk in chunks}
        extractions = (
            session.query(models.GraphExtraction)
            .filter(models.GraphExtraction.chunk_id.in_(chunk_text_by_id))
            .all()
        )
        _check(lane, "extractions_persisted", len(extractions) >= 1)
        bad_status = sorted(
            {e.status for e in extractions} - {"succeeded", "empty"}
        )
        _check(lane, "extraction_terminal", not bad_status, actual=bad_status)

        mentions = (
            session.query(models.EntityMention)
            .filter(models.EntityMention.extraction_id.in_([e.id for e in extractions]))
            .all()
        )
        evidence_rows = (
            session.query(models.GraphEdgeEvidence)
            .filter(models.GraphEdgeEvidence.extraction_id.in_([e.id for e in extractions]))
            .all()
        )

        # SQL-grounded: every persisted mention/evidence offset must match the
        # persisted source text exactly.
        for mention in mentions:
            source_text = chunk_text_by_id[
                session.get(models.GraphExtraction, mention.extraction_id).chunk_id
            ]
            _check(
                lane, "grounded",
                0 <= mention.start_offset < mention.end_offset <= len(source_text)
                and source_text[mention.start_offset:mention.end_offset] == mention.surface_form,
                mention_id=mention.id,
            )
        for row in evidence_rows:
            source_text = chunk_text_by_id[
                session.get(models.GraphExtraction, row.extraction_id).chunk_id
            ]
            _check(
                lane, "grounded",
                0 <= row.evidence_start < row.evidence_end <= len(source_text)
                and source_text[row.evidence_start:row.evidence_end] == row.evidence_text,
                evidence_id=row.id,
            )

        # Schema-valid: every persisted row reconstructs into the strict
        # extraction schema (bounded names/predicates, confidence in [0,1]).
        for row in evidence_rows:
            edge = row.edge
            _check(
                lane, "schema_valid",
                1 <= len(edge.source.display_name) <= 255
                and 1 <= len(edge.target.display_name) <= 255
                and 1 <= len(edge.predicate) <= 255
                and 0.0 <= row.confidence <= 1.0,
                evidence_id=row.id,
            )

        # Non-canned: the provider must have processed THIS unique document —
        # some persisted surface/evidence carries the unique run token. A
        # token-free canned echo fails here even when otherwise grounded.
        token_surfaces = [
            mention.surface_form for mention in mentions
        ] + [
            row.evidence_text for row in evidence_rows
        ] + [
            row.edge.source.display_name for row in evidence_rows
        ] + [
            row.edge.target.display_name for row in evidence_rows
        ]
        non_canned = bool(evidence_rows) and any(
            run_id in surface for surface in token_surfaces
        )
        _check(lane, "non_canned", non_canned)

        summary = {"schema_valid": True, "grounded": True, "non_canned": True}
    finally:
        session.rollback()
        vector_ids = [
            chunk.vector_id
            for chunk in session.query(models.Chunk).all()
        ]
        doc_ids = [row[0] for row in session.query(models.Document.id).all()]
        try:
            await store.delete(vector_ids)
            client.delete_collection(collection_name)
        except Exception:  # pragma: no cover - best-effort disposable cleanup
            pass
        _delete_lane_rows(session, doc_ids)
        session.close()
        engine.dispose()
        try:
            fresh = create_engine(db_url)
            restored = _db_fingerprint(fresh) == baseline
            fresh.dispose()
        except Exception:  # pragma: no cover - fingerprint must be readable
            restored = False
    return summary, restored


# ---------------------------------------------------------------------------
# Read-only fingerprints of the CONFIGURED production stores
# ---------------------------------------------------------------------------


def _sql_fingerprint(database_url: str | None) -> dict | None:
    """Read-only [row count, max id] fingerprint of a configured SQLite DB.

    Returns None when the URL is not SQLite (the gate stack pins SQLite per
    the compose default) or no database file exists (nothing configured to
    protect). Raises ValidatorFailure when a database exists but cannot be
    read — restoration cannot be verified otherwise.
    """
    if not database_url or not database_url.startswith("sqlite:"):
        return None
    location = database_url.split("sqlite:///", 1)[1].split("?", 1)[0]
    if not location or location == ":memory:":
        return None
    path = Path(location)
    if not path.exists():
        return None
    try:
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            tables = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                    " AND name != 'alembic_version' ORDER BY name"
                )
            ]
            fingerprint = {}
            for table in tables:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                max_id = None
                if any(column[1] == "id" for column in conn.execute(f"PRAGMA table_info({table})")):
                    max_id = conn.execute(f"SELECT MAX(id) FROM {table}").fetchone()[0]
                fingerprint[table] = [count, max_id]
            return fingerprint
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise ValidatorFailure(
            "fingerprint_unreadable", "restoration", {"detail": str(exc)[:200]}
        ) from exc


def _chroma_fingerprint(host: str | None, port: int, persist_directory: str | None) -> dict | None:
    """Read-only fingerprint of the CONFIGURED remote Chroma server.

    Returns None when no remote host is configured. A configured
    ``persist_directory`` is deliberately not fingerprinted: opening the
    embedded database directly can create sidecar files in the production
    store. Raises ValidatorFailure when a configured host cannot be read.
    """
    if not host:
        return None
    try:
        import chromadb

        client = chromadb.HttpClient(host=host, port=port)
        collections = sorted(
            getattr(collection, "name", collection)
            for collection in client.list_collections()
        )
        fingerprint: dict = {"collections": collections}
        if _PRODUCTION_COLLECTION in collections:
            collection = client.get_collection(_PRODUCTION_COLLECTION)
            ids = collection.get(include=[]).get("ids", [])
            digest = hashlib.sha256(",".join(sorted(ids)).encode("utf-8")).hexdigest()
            fingerprint[_PRODUCTION_COLLECTION] = {"count": len(ids), "ids_sha256": digest}
        return fingerprint
    except Exception as exc:
        raise ValidatorFailure(
            "fingerprint_unreadable", "restoration",
            {"detail": f"{type(exc).__name__}: {exc}"[:200]},
        ) from exc


def _production_fingerprints() -> dict:
    settings = get_settings()
    return {
        "sql": _sql_fingerprint(settings.database_url),
        "chroma": _chroma_fingerprint(
            settings.chroma_host, settings.chroma_port, settings.chroma_persist_directory
        ),
    }


async def _amain() -> dict:
    run_id = secrets.token_hex(8)
    det_db = Path(tempfile.gettempdir()) / f"validate-phase10a-{run_id}.db"
    live_db = Path(tempfile.gettempdir()) / f"validate-phase10a-live-{run_id}.db"

    production_before = _production_fingerprints()
    try:
        deterministic, det_restored = await _run_deterministic(f"sqlite:///{det_db}", run_id)
        live, live_restored = await _run_live(f"sqlite:///{live_db}", run_id)
    finally:
        for db_path in (det_db, live_db):
            for suffix in ("", "-wal", "-shm"):
                p = Path(str(db_path) + suffix)
                if p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass
    production_after = _production_fingerprints()

    _check("restoration", "production_fingerprints_unchanged",
           production_before == production_after,
           changed=[key for key in production_before
                    if production_before[key] != production_after[key]])
    _check("restoration", "deterministic_lane_restored", det_restored)
    _check("restoration", "live_lane_restored", live_restored)

    return {"deterministic": deterministic, "live": live, "restored": True}


def main() -> int:
    # NOTE: no os.environ mutation here. chromadb derives client settings
    # from the environment, so mutating it would make later EphemeralClient
    # creations in the same process fail with "different settings".
    try:
        result = asyncio.run(_amain())
    except ValidatorFailure as exc:
        sys.stderr.write(
            json.dumps({"error": exc.code, "lane": exc.lane, **exc.detail}, sort_keys=True)
            + "\n"
        )
        return 2 if exc.code in _EXIT2_CODES else 1
    except Exception as exc:  # never mask an unexpected failure with exit 0
        sys.stderr.write(
            json.dumps(
                {
                    "error": "unexpected_failure",
                    "lane": "validator",
                    "detail": f"{type(exc).__name__}: {exc}"[:200],
                },
                sort_keys=True,
            )
            + "\n"
        )
        return 2
    sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
