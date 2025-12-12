"""Repositories for database operations."""
from typing import Iterable, List, Optional, Sequence

from sqlalchemy.orm import Session

from app.persistence import models


def create_document(
    session: Session, *, title: str, source: Optional[str], tags: Optional[Sequence[str]]
) -> models.Document:
    document = models.Document(title=title, source=source, tags=",".join(tags) if tags else None)
    session.add(document)
    session.flush()
    return document


def create_chunks(
    session: Session, *, document: models.Document, chunks: Iterable[dict]
) -> List[models.Chunk]:
    created = []
    for chunk in chunks:
        chunk_model = models.Chunk(
            document_id=document.id,
            index=chunk["index"],
            text=chunk["text"],
            start_offset=chunk["start_offset"],
            end_offset=chunk["end_offset"],
        )
        session.add(chunk_model)
        created.append(chunk_model)
    session.flush()
    return created


def list_documents(session: Session) -> List[models.Document]:
    return session.query(models.Document).all()


def get_document(session: Session, document_id: int) -> Optional[models.Document]:
    return session.query(models.Document).filter(models.Document.id == document_id).first()


def chunk_count(session: Session, document_id: int) -> int:
    return session.query(models.Chunk).filter(models.Chunk.document_id == document_id).count()
