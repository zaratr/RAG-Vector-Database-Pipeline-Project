"""Document endpoints."""
from __future__ import annotations

import io
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi import status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.models import DocumentCreate, DocumentDetail, DocumentSummary
from app.persistence import models, repositories
from app.services.embeddings import get_embedding_provider
from app.services.ingestion import chunks_for_document, ingest_text
from app.services.vector_store import get_vector_store

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentCreateResponse(BaseModel):
    document_id: int
    chunks: int


@router.post("", response_model=DocumentCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_document(
    payload: DocumentCreate = Depends(),
    file: UploadFile | None = File(default=None),
    session: Session = Depends(get_db),
):
    embedding_provider = get_embedding_provider()
    vector_store = get_vector_store()
    text_content = payload.text

    if file:
        if file.content_type not in {"application/pdf", "text/plain"}:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        content = await file.read()
        if file.content_type == "application/pdf":
            if not PdfReader:
                raise HTTPException(status_code=500, detail="PDF support not available")
            reader = PdfReader(io.BytesIO(content))
            text_content = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            text_content = content.decode("utf-8")

    if not text_content:
        raise HTTPException(status_code=400, detail="No text provided")

    result = await ingest_text(
        title=payload.title,
        source=payload.source,
        tags=payload.tags,
        text=text_content,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        session=session,
    )
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
