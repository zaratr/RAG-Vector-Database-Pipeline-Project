"""Document endpoints."""
from __future__ import annotations

import io
from typing import List
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi import status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.models import DocumentDetail, DocumentSummary
from app.persistence import models, repositories
from app.services.embeddings import get_embedding_provider
from app.services.ingestion import chunks_for_document, ingest_text, ingest_image, VectorIndexIncomplete
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


@router.post("", response_model=DocumentCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_document(
    title: str = Form(...),
    source: str | None = Form(default=None),
    tags: str | None = Form(default=None),
    text: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    session: Session = Depends(get_db),
    graph_extractor: GraphExtractor = Depends(get_graph_extractor),
):
    embedding_provider = get_embedding_provider()
    vector_store = get_vector_store()
    text_content = text

    parsed_tags = [t.strip() for t in tags.split(",")] if tags else None

    if file:
        content = await file.read()

        if file.content_type  in {"image/png", "image/jpeg", "image/gif", "image/webp" }:
            suffix = Path(file.filename).suffix or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                tmp_path= tmp.name
            image_provider = get_image_embedding_provider()

            try:
                result = await ingest_image(
                        title = title,
                        source = source,
                        tags = parsed_tags,
                        image_path = tmp_path,
                        media_type = file.content_type,
                        embedding_provider = image_provider,
                        vector_store = vector_store,
                        session = session
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
            return DocumentCreateResponse(**result)
        
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

    if not text_content or not text_content.strip():
        raise HTTPException(status_code=400, detail="No text content provided")




    settings = get_settings()
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
    except RuntimeError as exc:
        # Remaining vector-store failures (e.g. upsert/list raised by the
        # backend) map to the plan's stable public detail; the internal
        # exception type is already stored as the bounded failure code.
        raise HTTPException(
            status_code=503, detail="Vector index unavailable"
        ) from exc
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
