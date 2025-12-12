"""SQLAlchemy models for documents and chunks."""
from __future__ import annotations

from sqlalchemy import Column, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.core.db import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    source = Column(String, nullable=True)
    tags = Column(String, nullable=True)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    start_offset = Column(Integer, nullable=False)
    end_offset = Column(Integer, nullable=False)

    document = relationship("Document", back_populates="chunks")

    def metadata(self) -> dict:
        return {
            "document_id": self.document_id,
            "chunk_id": self.id,
            "index": self.index,
            "title": self.document.title if self.document else None,
            "tags": self.document.tags.split(",") if self.document and self.document.tags else [],
        }
