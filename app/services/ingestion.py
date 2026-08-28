"""Document ingestion pipeline.

Ingestion is aligned with the durable extraction lifecycle. The text
ingestion state machine is:

1. Normalize/chunk input.
2. Insert document as ``staged``, chunks with deterministic vector IDs and text
   media type, and extraction identities as ``pending`` or ``skipped``; commit.
3. Compute embeddings and run eligible extraction outside the SQL transaction.
4. Persist ``succeeded``/``empty`` or ``failed`` extraction status.
5. Upsert deterministic Chroma records.
6. Set document ``ready`` only after every required vector exists and every
   eligible extraction is ``succeeded`` or ``empty``.
7. On any exception, mark document ``failed``; attempt vector deletion for all
   document vector IDs; preserve the extraction failure audit row; re-raise.

Provider/vector failures leave a truthful, operator-visible (never
query-visible) failed state. No pending extraction may remain after a handled
request failure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from app.core.logging import logger
from app.persistence import graph_repository, models, repositories
from app.services.chunking import chunk_text
from app.services.embeddings import EmbeddingProvider, ImageEmbeddingProvider
from app.services.graph_extraction import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    DisabledGraphExtractor,
    GraphExtractionError,
    GraphExtractor,
)
from app.services.vector_store import VectorStore
from app.services.safety_review import (
    IngestionSafetyBlocked,
    SafetyReviewSubsystemFailure,
)
from app.services.safety_policy import get_safety_policy

EXTRACTION_PROVIDER = "ollama"
FAILED_AFTER_CHUNK = "aborted_after_chunk_failure"
EMBEDDING_FAILED = "embedding_failed"


class VectorIndexIncomplete(RuntimeError):
    """Raised when required vectors are absent after upsert (compensation trigger).

    Maps to HTTP 503 with the stable public detail ``Vector index unavailable``;
    distinct from graph-provider failures so the route handler can produce the
    exact status/detail.
    """


async def ingest_text(
    *,
    title: str,
    source: Optional[str],
    tags: Optional[Sequence[str]],
    text: str,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    session,
    graph_extractor: GraphExtractor | None = None,
    graph_extraction_provider: str = EXTRACTION_PROVIDER,
    graph_extraction_model: str = "unknown",
    trust_tier: str = "untrusted",
    trust_score: float = 0.0,
    trust_policy_version: str = "unassigned",
    ingestion_origin: str = "api",
) -> dict:
    """Ingest raw text through the staged lifecycle state machine."""
    # A disabled extractor passed directly (service level) is treated exactly
    # like the route does: no provider call, extraction recorded as skipped
    # with reason extraction_disabled — never mislabeled as an empty provider
    # result (a disabled run is never labeled empty).
    if isinstance(graph_extractor, DisabledGraphExtractor):
        graph_extractor = None
    normalized = " ".join(text.split())
    logger.info("Chunking document '%s'", title)
    chunk_payloads = chunk_text(normalized)
    chunk_texts = [chunk["text"] for chunk in chunk_payloads]

    # Step 2: staged commit first. Provider/vector failures during external work
    # leave a truthful staged state rather than vanishing.
    document = repositories.create_document(session, title=title, source=source, tags=tags)
    document.ingestion_status = "staged"
    # 10B.2: server-assigned provenance fields.
    document.trust_tier = trust_tier
    document.trust_score = trust_score
    document.trust_policy_version = trust_policy_version
    document.ingestion_origin = ingestion_origin
    chunk_models: List[models.Chunk] = repositories.create_chunks(
        session, document=document, chunks=chunk_payloads
    )
    for chunk_model in chunk_models:
        chunk_model.vector_id = f"chunk:{chunk_model.id}"
        chunk_model.media_type = "text/plain"
    session.flush()
    # 10C.4: stage WITHOUT extraction leases first; the ingestion-scope
    # safety review runs on the staged rows before any identity exists.
    session.commit()

    document_id = document.id
    vector_ids = [chunk.vector_id for chunk in chunk_models]
    relation_count = 0

    try:
        # 10C.4 ingestion-scope safety enforcement: block|filter creates
        # direct terminal skipped identities (safety_blocked,
        # attempt_count=0), marks the document failed, writes no vectors,
        # and raises for HTTP 422.
        from app.config import get_settings as _get_settings

        _settings = _get_settings()
        if _settings.content_safety_enabled:
            from app.services.safety_review import SafetyReviewService

            safety_policy = get_safety_policy()
            safety_service = SafetyReviewService(session)
            run = safety_service.review_ingestion(
                document=document, text=normalized, policy=safety_policy,
                mode=_settings.safety_llm_mode,
            )
            if run.status != "succeeded":
                raise SafetyReviewSubsystemFailure(
                    f"ingestion safety review {run.status}: {run.failure_code}")
            if run.final_action in ("block", "filter"):
                for chunk_model in chunk_models:
                    graph_repository.skip_chunk_extraction(
                        session,
                        chunk=chunk_model,
                        provider=graph_extraction_provider,
                        model=graph_extraction_model,
                        prompt_version=PROMPT_VERSION,
                        schema_version=SCHEMA_VERSION,
                        reason_code="safety_blocked",
                    )
                document.ingestion_status = "failed"
                document.failure_code = "safety_blocked"
                session.commit()
                raise IngestionSafetyBlocked(run.final_action)

        # Establish extraction identities as pending/skipped before external
        # work.
        lease_by_index: dict[int, object] = {}
        for idx, chunk_model in enumerate(chunk_models):
            if graph_extractor is not None:
                lease = graph_repository.begin_chunk_extraction(
                    session,
                    chunk=chunk_model,
                    provider=graph_extraction_provider,
                    model=graph_extraction_model,
                    prompt_version=PROMPT_VERSION,
                    schema_version=SCHEMA_VERSION,
                )
                lease_by_index[idx] = lease
            else:
                lease = graph_repository.skip_chunk_extraction(
                    session,
                    chunk=chunk_model,
                    provider=graph_extraction_provider,
                    model=graph_extraction_model,
                    prompt_version=PROMPT_VERSION,
                    schema_version=SCHEMA_VERSION,
                    reason_code="extraction_disabled",
                )
                lease_by_index[idx] = lease
        session.commit()

        # Step 3: external work (embeddings + extraction) outside the SQL txn.
        try:
            embeddings = await embedding_provider.embed_texts(chunk_texts)
        except Exception as embed_exc:
            # Embedding failed before/independent of extraction: fail every
            # pending run so none remains pending.
            _fail_pending_leases(
                session, lease_by_index, only_pending=True, code=EMBEDDING_FAILED
            )
            session.commit()
            raise embed_exc

        extracted_by_chunk: list[list] = []
        for idx, text_value in enumerate(chunk_texts):
            lease = lease_by_index.get(idx)
            if graph_extractor is not None and lease is not None and lease.should_call_provider:
                try:
                    relations = await graph_extractor.extract(text_value)
                except Exception as extract_exc:
                    graph_repository.fail_chunk_extraction(
                        session,
                        extraction=lease.extraction,
                        error_code=type(extract_exc).__name__[:100],
                        error_detail=str(extract_exc)[:1000],
                        expected_attempt_count=lease.lease_attempt_count,
                    )
                    session.commit()
                    # Remaining pending runs become failed/aborted.
                    _fail_pending_leases(
                        session, lease_by_index, only_pending=True, code=FAILED_AFTER_CHUNK
                    )
                    session.commit()
                    raise extract_exc
                graph_repository.complete_chunk_extraction(
                    session, extraction=lease.extraction, relations=relations,
                    expected_attempt_count=lease.lease_attempt_count,
                )
                extracted_by_chunk.append(relations)
            else:
                extracted_by_chunk.append([])

        for relations in extracted_by_chunk:
            relation_count += len(relations)
        session.commit()

        # Step 5: upsert deterministic Chroma records.
        metadata_entries = [chunk.get_chunk_metadata() for chunk in chunk_models]
        await vector_store.upsert_embeddings(
            embeddings,
            metadata_entries,
            vector_ids,
            documents=chunk_texts,
        )

        # Step 6: readiness requires every expected vector to exist.
        existing_ids = set(await vector_store.list_ids())
        missing = [vid for vid in vector_ids if vid not in existing_ids]
        if missing:
            raise VectorIndexIncomplete(
                f"vector index incomplete: missing {len(missing)} required ids"
            )

        # Step 6b: every eligible extraction must be terminal succeeded/empty.
        terminal = all(
            lease.extraction.status in ("succeeded", "empty", "skipped")
            for lease in lease_by_index.values()
        )
        if not terminal:
            raise VectorIndexIncomplete("extraction left non-terminal runs")

        document.ingestion_status = "ready"
        document.failure_code = None
        session.commit()
    except Exception as exc:
        await _handle_ingestion_failure(session, document_id, vector_ids, vector_store, exc)
        raise

    logger.info("Ingested document %s with %s chunks", document_id, len(chunk_models))
    return {
        "document_id": document_id,
        "chunks": len(chunk_models),
        "relations": relation_count,
    }


def _fail_pending_leases(
    session, lease_by_index: dict[int, object], *, only_pending: bool, code: str
) -> None:
    """Transition every still-pending leased extraction to failed."""
    for lease in lease_by_index.values():
        extraction = lease.extraction
        if extraction.status == "pending":
            graph_repository.fail_chunk_extraction(
                session,
                extraction=extraction,
                error_code=code,
                error_detail="aborted during ingestion",
                expected_attempt_count=lease.lease_attempt_count,
            )


async def _handle_ingestion_failure(
    session, document_id: int, vector_ids: list[str], vector_store: VectorStore, exc: Exception
) -> None:
    """Mark the document failed, attempt vector cleanup, preserve audit rows."""
    try:
        failed_document = session.get(models.Document, document_id)
        if failed_document is not None:
            failed_document.ingestion_status = "failed"
            # Preserve an explicitly set stable code (e.g. safety_blocked);
            # only synthesize the exception-class code when none was set.
            if not failed_document.failure_code:
                failed_document.failure_code = type(exc).__name__[:100]
        session.commit()
    except Exception:  # pragma: no cover - best-effort failure marking
        session.rollback()
    # Attempt vector deletion for all document vector IDs (compensation).
    try:
        await vector_store.delete(vector_ids)
    except Exception:  # pragma: no cover - compensation best-effort
        logger.warning("compensation vector delete failed for document %s", document_id)


async def ingest_image(
    *,
    title: str,
    source: Optional[str],
    tags: Optional[Sequence[str]],
    image_path: str | Path,
    media_type: str,
    embedding_provider: ImageEmbeddingProvider,
    vector_store: VectorStore,
    session,
    graph_extraction_provider: str = EXTRACTION_PROVIDER,
    graph_extraction_model: str = "unknown",
) -> dict:
    """Ingest an image: actual image media type, graph extraction skipped.

    Never sends placeholder image text to the extraction provider.
    """
    path = Path(image_path)
    logger.info("Embedding image '%s' (%s)", title, media_type)

    document = repositories.create_document(session, title=title, source=source, tags=tags)
    document.ingestion_status = "staged"
    chunk_text_value = f"[image:{media_type}] {path.name}"
    chunk = repositories.create_chunks(
        session,
        document=document,
        chunks=[
            {
                "index": 0,
                "text": chunk_text_value,
                "start_offset": 0,
                "end_offset": len(chunk_text_value),
            }
        ],
    )[0]
    chunk.vector_id = f"chunk:{chunk.id}"
    chunk.media_type = media_type
    # Image ingestion records graph extraction as skipped (unsupported media).
    skip_lease = graph_repository.skip_chunk_extraction(
        session,
        chunk=chunk,
        provider=graph_extraction_provider,
        model=graph_extraction_model,
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        reason_code="unsupported_media_type",
    )
    session.flush()
    document_id = document.id
    session.commit()

    vector_id = chunk.vector_id
    try:
        embeddings = await embedding_provider.embed_images([path])
        embedding = embeddings[0]
        metadata = chunk.get_chunk_metadata()
        metadata["media_type"] = media_type
        metadata["image_name"] = path.name
        await vector_store.upsert_embeddings(
            [embedding],
            [metadata],
            [vector_id],
            documents=[chunk_text_value],
        )
        existing_ids = set(await vector_store.list_ids())
        if vector_id not in existing_ids:
            raise VectorIndexIncomplete("vector index missing image vector id")
        document.ingestion_status = "ready"
        document.failure_code = None
        session.commit()
    except Exception as exc:
        await _handle_ingestion_failure(session, document_id, [vector_id], vector_store, exc)
        raise

    logger.info("Ingested image document %s", document_id)
    return {"document_id": document_id, "chunks": 1}


def chunks_for_document(chunks: Iterable[models.Chunk]) -> List[dict]:
    return [
        {
            "id": chunk.id,
            "index": chunk.index,
            "text": chunk.text,
            "start_offset": chunk.start_offset,
            "end_offset": chunk.end_offset,
        }
        for chunk in chunks
    ]
