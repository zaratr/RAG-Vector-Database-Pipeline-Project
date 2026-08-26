"""SQLAlchemy models for documents and chunks."""
from __future__ import annotations

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import relationship

from app.core.db import Base


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(
            "ingestion_status IN ('staged', 'ready', 'failed')",
            name="ck_documents_ingestion_status",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    source = Column(String, nullable=True)
    tags = Column(String, nullable=True)
    ingestion_status = Column(
        String(20), nullable=False, default="ready", server_default="ready", index=True
    )
    failure_code = Column(String(100), nullable=True)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")



class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "index", name="uq_chunks_document_index"),
    )

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    start_offset = Column(Integer, nullable=False)
    end_offset = Column(Integer, nullable=False)
    vector_id = Column(String(255), nullable=True, unique=True, index=True)
    media_type = Column(String(100), nullable=False, default="text/plain", server_default="text/plain")

    document = relationship("Document", back_populates="chunks")
    graph_extractions = relationship(
        "GraphExtraction", back_populates="chunk", cascade="all, delete-orphan"
    )

    def get_chunk_metadata(self) -> dict:
        meta = {
            "document_id": self.document_id,
            "chunk_id": self.id,
            "index": self.index,
        }
        if self.document and self.document.title:
            meta["title"] = self.document.title
        if self.document and self.document.tags:
            tags = [t.strip() for t in self.document.tags.split(",") if t.strip()]
            if tags:
                meta["tags"] = tags
        return meta


class GraphEntity(Base):
    __tablename__ = "graph_entities"
    __table_args__ = (
        UniqueConstraint(
            "canonical_name", "entity_type", name="uq_graph_entities_name_type"
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    canonical_name = Column(String(255), nullable=False, index=True)
    display_name = Column(String(255), nullable=False)
    entity_type = Column(String(100), nullable=False)

    mentions = relationship("EntityMention", back_populates="entity")
    outgoing_edges = relationship(
        "GraphEdge",
        foreign_keys="GraphEdge.source_entity_id",
        back_populates="source",
    )
    incoming_edges = relationship(
        "GraphEdge",
        foreign_keys="GraphEdge.target_entity_id",
        back_populates="target",
    )


class GraphExtraction(Base):
    __tablename__ = "graph_extractions"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'succeeded', 'failed', 'empty', 'skipped')",
            name="ck_graph_extractions_status",
        ),
        CheckConstraint(
            "attempt_count >= 0",
            name="ck_graph_extractions_attempt_count",
        ),
        CheckConstraint(
            "is_identity_owner IN (0,1)",
            name="ck_graph_extractions_is_identity_owner",
        ),
        CheckConstraint(
            "length(input_sha256) = 64 AND input_sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_graph_extractions_input_sha256_hex",
        ),
        CheckConstraint(
            # Per-status lifecycle rules; mirrors the
            # b7f3d5a9c2e1 migration CHECK of the same name.
            "CASE status "
            "WHEN 'pending' THEN completed_at IS NULL AND error_code IS NULL "
            "AND error_detail IS NULL AND attempt_count >= 1 "
            "WHEN 'succeeded' THEN completed_at IS NOT NULL AND error_code IS NULL "
            "AND error_detail IS NULL AND attempt_count >= 1 "
            "WHEN 'empty' THEN completed_at IS NOT NULL AND error_code IS NULL "
            "AND error_detail IS NULL AND attempt_count >= 1 "
            "WHEN 'failed' THEN completed_at IS NOT NULL AND error_code IS NOT NULL "
            "AND attempt_count >= 1 "
            "WHEN 'skipped' THEN completed_at IS NOT NULL AND error_code IN "
            "('extraction_disabled', 'unsupported_media_type') AND attempt_count = 0 "
            "ELSE 0 END",
            name="ck_graph_extractions_lifecycle",
        ),
        Index(
            "uq_graph_extractions_identity_owner",
            "chunk_id",
            "provider",
            "model",
            "prompt_version",
            "schema_version",
            "input_sha256",
            unique=True,
            sqlite_where=text("is_identity_owner = 1"),
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(Integer, ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False)
    provider = Column(String(100), nullable=False)
    model = Column(String(255), nullable=False)
    prompt_version = Column(String(50), nullable=False)
    schema_version = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False)
    error_code = Column(String(100), nullable=True)
    error_detail = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    # Lifecycle/identity columns.
    input_sha256 = Column(String(64), nullable=False)
    attempt_count = Column(Integer, nullable=False, default=0, server_default="0")
    completed_at = Column(DateTime(timezone=True), nullable=True)
    attempt_started_at = Column(DateTime(timezone=True), nullable=True)
    is_identity_owner = Column(
        Boolean, nullable=False, default=True, server_default="1"
    )

    chunk = relationship("Chunk", back_populates="graph_extractions")
    mentions = relationship(
        "EntityMention", back_populates="extraction", cascade="all, delete-orphan"
    )
    edge_evidence = relationship(
        "GraphEdgeEvidence", back_populates="extraction", cascade="all, delete-orphan"
    )


class EntityMention(Base):
    __tablename__ = "entity_mentions"
    __table_args__ = (
        UniqueConstraint(
            "entity_id", "extraction_id", name="uq_entity_mentions_entity_extraction"
        ),
        CheckConstraint(
            "start_offset >= 0 AND end_offset > start_offset",
            name="ck_entity_mentions_offsets",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    entity_id = Column(
        Integer, ForeignKey("graph_entities.id", ondelete="CASCADE"), nullable=False
    )
    extraction_id = Column(
        Integer, ForeignKey("graph_extractions.id", ondelete="CASCADE"), nullable=False
    )
    surface_form = Column(String(255), nullable=False)
    start_offset = Column(Integer, nullable=False)
    end_offset = Column(Integer, nullable=False)

    entity = relationship("GraphEntity", back_populates="mentions")
    extraction = relationship("GraphExtraction", back_populates="mentions")


class GraphEdge(Base):
    __tablename__ = "graph_edges"
    __table_args__ = (
        UniqueConstraint(
            "source_entity_id",
            "predicate",
            "target_entity_id",
            name="uq_graph_edges_triplet",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    source_entity_id = Column(
        Integer, ForeignKey("graph_entities.id", ondelete="CASCADE"), nullable=False
    )
    target_entity_id = Column(
        Integer, ForeignKey("graph_entities.id", ondelete="CASCADE"), nullable=False
    )
    predicate = Column(String(255), nullable=False, index=True)

    source = relationship(
        "GraphEntity",
        foreign_keys=[source_entity_id],
        back_populates="outgoing_edges",
    )
    target = relationship(
        "GraphEntity",
        foreign_keys=[target_entity_id],
        back_populates="incoming_edges",
    )
    evidence = relationship(
        "GraphEdgeEvidence", back_populates="edge", cascade="all, delete-orphan"
    )


class GraphEdgeEvidence(Base):
    __tablename__ = "graph_edge_evidence"
    __table_args__ = (
        UniqueConstraint(
            "edge_id",
            "extraction_id",
            "evidence_start",
            "evidence_end",
            name="uq_graph_edge_evidence_location",
        ),
        CheckConstraint(
            "evidence_start >= 0 AND evidence_end > evidence_start",
            name="ck_graph_edge_evidence_offsets",
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_graph_edge_evidence_confidence",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    edge_id = Column(
        Integer, ForeignKey("graph_edges.id", ondelete="CASCADE"), nullable=False
    )
    extraction_id = Column(
        Integer, ForeignKey("graph_extractions.id", ondelete="CASCADE"), nullable=False
    )
    evidence_text = Column(Text, nullable=False)
    evidence_start = Column(Integer, nullable=False)
    evidence_end = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)

    edge = relationship("GraphEdge", back_populates="evidence")
    extraction = relationship("GraphExtraction", back_populates="edge_evidence")
