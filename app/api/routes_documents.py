"""Document endpoints."""
from __future__ import annotations

import io
from typing import List
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, Request
from fastapi import status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.models import DocumentDetail, DocumentSummary
from app.persistence import models, repositories
from app.services.embeddings import get_embedding_provider
from app.services.ingestion import chunks_for_document, ingest_text, ingest_image, VectorIndexIncomplete
from app.services.safety_review import (
    IngestionSafetyBlocked,
    SafetyReviewSubsystemFailure,
)
from app.services.graph_extraction import (
    DisabledGraphExtractor,
    GraphExtractionError,
    GraphExtractor,
    GraphProviderUnavailable,
    get_graph_extractor,
)
from app.services.vector_store import get_vector_store
from app.services.embeddings import get_image_embedding_provider 
from app.config import get_settings

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentCreateResponse(BaseModel):
    document_id: int
    chunks: int
    relations: int = 0


_bearer_scheme = HTTPBearer(auto_error=False)


@router.post("", response_model=DocumentCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_document(
    request: Request,
    response: Response,
    title: str = Form(...),
    source: str | None = Form(default=None),
    tags: str | None = Form(default=None),
    text: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    session: Session = Depends(get_db),
    graph_extractor: GraphExtractor = Depends(get_graph_extractor),
):
    embedding_provider = get_embedding_provider()
    vector_store = get_vector_store()
    text_content = text
    settings = get_settings()

    parsed_tags = [t.strip() for t in tags.split(",")] if tags else None

    # Physical size limit first: an oversized file creates no SQL/Chroma rows,
    # including no rate-bucket row (streamed in 64 KiB chunks, B-10/B-20).
    image_payload: tuple[bytes, str] | None = None
    if file:
        file_max = settings.ingestion_file_max_bytes
        chunks = []
        total = 0
        while True:
            chunk = await file.read(65536)
            if not chunk:
                break
            total += len(chunk)
            if total > file_max:
                raise HTTPException(
                    status_code=413,
                    detail={"code": "ingestion_too_large", "limit_bytes": file_max, "measured": "file"},
                )
            chunks.append(chunk)
        content = b"".join(chunks)

        if file.content_type in {"image/png", "image/jpeg", "image/gif", "image/webp"}:
            # Decode/dispatch happens after the rate limiter (D-17).
            image_payload = (content, file.content_type)
        elif file.content_type == "application/pdf":
            if not PdfReader:
                raise HTTPException(status_code=500, detail="PDF support not available")
            try:
                reader = PdfReader(io.BytesIO(content))
                text_content = "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception:
                raise HTTPException(status_code=400, detail="Could not parse PDF file")
        elif file.content_type in {"text/plain", "text/markdown"}:
            try:
                text_content = content.decode("utf-8")
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="File is not valid UTF-8 text")
        elif file.filename and Path(file.filename).suffix.lower() in {".md", ".markdown", ".txt"}:
            try:
                text_content = content.decode("utf-8")
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="File is not valid UTF-8 text")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

    # Auth matrix BEFORE the limiter: authentication rejections consume no slot.
    import hmac

    is_operator = False
    if credentials is not None:
        token_val = settings.operator_token.get_secret_value()
        if not token_val or not hmac.compare_digest(credentials.credentials, token_val):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        is_operator = True

    # 10B.3: rate limiting AFTER auth, BEFORE application validation, in a
    # SEPARATE transaction (D-8): successful AND application-invalid upload
    # attempts consume a slot; only authentication rejections do not.
    from app.services.ingestion_limits import check_rate_limit_http
    from app.core.db import create_database_engine, sessionmaker as _sm
    remote_host = request.client.host if request.client else "unknown"
    # Rate identity (plan §10B.3): operator:<token> only for a VALID operator
    # bearer; every other caller — authenticated or not — is client:<host>.
    # Never seed the limiter with the server-configured token for anonymous
    # requests (D-37): that would couple all clients into one shared bucket.
    op_token = credentials.credentials if is_operator else None
    # Use a separate engine/session so the rate increment survives ingestion rollback.
    _rl_engine = create_database_engine(settings.database_url)
    _rl_session = _sm(bind=_rl_engine)()
    try:
        allowed, rl_headers = check_rate_limit_http(
            _rl_session, operator_token=op_token, remote_host=remote_host,
            limit=settings.ingest_rate_limit_requests,
            window_seconds=settings.ingest_rate_limit_window_seconds,
        )
        _rl_session.commit()
    except Exception:
        _rl_session.rollback()
        raise
    finally:
        _rl_session.close()
        _rl_engine.dispose()
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={"code": "ingestion_rate_limited"},
            headers=rl_headers,
        )

    # 10B.3: extracted/normalized UTF-8 bytes are measured before chunking,
    # after decode/extraction finalizes the text (D-16).
    if text_content is not None:
        extracted_bytes = len(text_content.encode("utf-8"))
        if extracted_bytes > settings.ingestion_extracted_max_bytes:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "ingestion_too_large",
                    "limit_bytes": settings.ingestion_extracted_max_bytes,
                    "measured": "extracted",
                },
            )

    if image_payload is None and (not text_content or not text_content.strip()):
        raise HTTPException(status_code=400, detail="No text content provided")

    # 10B.2: server-assigned trust assessment using lifespan-cached policy.
    from app.services.provenance import SourceTrustPolicy

    policy: SourceTrustPolicy = getattr(request.app.state, "source_trust_policy", None)
    if policy is None:
        from app.services.provenance import load_source_trust_policy
        try:
            policy = load_source_trust_policy(settings.source_trust_policy_path)
        except Exception:
            raise HTTPException(status_code=503, detail="Trust policy unavailable")

    # Check for blocked source (any bearer).
    if policy.is_blocked(source):
        raise HTTPException(status_code=403, detail="source_blocked")

    # Check for protected trusted source without operator.
    if policy.requires_operator(source) and not is_operator:
        raise HTTPException(status_code=403, detail="protected_source_requires_operator")

    if image_payload is not None:
        content, media_type = image_payload
        suffix = Path(file.filename).suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        image_provider = get_image_embedding_provider()
        try:
            result = await ingest_image(
                title=title,
                source=source,
                tags=parsed_tags,
                image_path=tmp_path,
                media_type=media_type,
                embedding_provider=image_provider,
                vector_store=vector_store,
                session=session,
            )
        except VectorIndexIncomplete as exc:
            raise HTTPException(
                status_code=503, detail="Vector index unavailable"
            ) from exc
        except RuntimeError as exc:
            # Vector-store/embedding failures (VectorIndexIncomplete is a
            # RuntimeError) share the text route's stable public detail.
            raise HTTPException(
                status_code=503, detail="Vector index unavailable"
            ) from exc
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        response.headers.update(rl_headers)
        return DocumentCreateResponse(**result)

    # Assess trust tier with the resolved operator flag.
    trust_tier, trust_score = policy.assess(source, is_operator=is_operator)

    try:
        result = await ingest_text(
            title=title,
            source=source,
            tags=parsed_tags,
            text=text_content,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            graph_extractor=(
                None if isinstance(graph_extractor, DisabledGraphExtractor) else graph_extractor
            ),
            graph_extraction_model=(
                settings.graph_extraction_model or settings.llm_model
            ),
            session=session,
            trust_tier=trust_tier,
            trust_score=trust_score,
            trust_policy_version=policy.version,
            ingestion_origin="api",
        )
    except GraphProviderUnavailable as exc:
        raise HTTPException(
            status_code=503, detail="Graph extraction provider unavailable"
        ) from exc
    except VectorIndexIncomplete as exc:
        raise HTTPException(
            status_code=503, detail="Vector index unavailable"
        ) from exc
    except GraphExtractionError as exc:
        raise HTTPException(
            status_code=502, detail="Graph extraction provider failed"
        ) from exc
    except SafetyReviewSubsystemFailure as exc:
        # 10C.4: safety subsystem failure fails closed with 503.
        raise HTTPException(
            status_code=503, detail="safety_review_failed",
        ) from exc
    except IngestionSafetyBlocked as exc:
        # 10C.4: ingestion-scope block|filter refuses with 422; the failed
        # document, skipped safety_blocked identities, and the safety review
        # are already persisted by the ingestion pipeline.
        raise HTTPException(
            status_code=422,
            detail={"code": "ingestion_safety_blocked",
                    "final_action": exc.final_action},
        ) from exc
    except RuntimeError as exc:
        # Remaining vector-store failures (e.g. upsert/list raised by the
        # backend) map to the plan's stable public detail; the internal
        # exception type is already stored as the bounded failure code.
        # Typed subclasses above (VectorIndexIncomplete,
        # SafetyReviewSubsystemFailure) intercept their own mapped statuses.
        raise HTTPException(
            status_code=503, detail="Vector index unavailable"
        ) from exc
    # Accepted requests carry the same rate-limit headers as rejections.
    response.headers.update(rl_headers)
    return DocumentCreateResponse(**result)


@router.get("", response_model=List[DocumentSummary])
async def list_documents(session: Session = Depends(get_db)):
    docs = repositories.list_documents(session)
    summaries = [
        DocumentSummary(
            id=doc.id,
            title=doc.title,
            source=doc.source,
            tags=doc.tags.split(",") if doc.tags else [],
            chunk_count=repositories.chunk_count(session, doc.id),
        )
        for doc in docs
    ]
    return summaries


@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(document_id: int, session: Session = Depends(get_db)):
    document: models.Document | None = repositories.get_document(session, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    chunk_summaries = chunks_for_document(document.chunks)
    return DocumentDetail(
        id=document.id,
        title=document.title,
        source=document.source,
        tags=document.tags.split(",") if document.tags else [],
        chunks=chunk_summaries,
    )
