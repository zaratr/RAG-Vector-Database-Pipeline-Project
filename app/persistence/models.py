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
        CheckConstraint(
            "trust_tier IN ('trusted', 'standard', 'untrusted', 'blocked')",
            name="ck_documents_trust_tier",
        ),
        CheckConstraint(
            "trust_score >= 0 AND trust_score <= 1",
            name="ck_documents_trust_score",
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
    # 10B.2 server-assigned provenance fields.
    trust_tier = Column(
        String(20), nullable=False, default="untrusted", server_default="untrusted", index=True
    )
    trust_score = Column(Float, nullable=False, default=0.0, server_default="0")
    trust_policy_version = Column(
        String(50), nullable=False, default="unassigned", server_default="unassigned", index=True
    )
    ingestion_origin = Column(
        String(50), nullable=False, default="api", server_default="api"
    )

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
        if self.document and self.document.source:
            # 10B.3: per-source caps need the authoritative source in the
            # vector metadata; without it every context shares "unknown".
            meta["source"] = self.document.source
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
            # Plan-exact per-status lifecycle rules (10A.3 W4); mirrors the
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
    # 10A.3 lifecycle/identity columns.
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


class RetrievalAudit(Base):
    """End-to-end /query lifecycle audit (10B.2)."""
    __tablename__ = "retrieval_audits"
    __table_args__ = (
        CheckConstraint(
            "retrieval_mode IN ('vector', 'graph', 'hybrid')",
            name="ck_retrieval_audits_mode",
        ),
        CheckConstraint(
            "status IN ('pending', 'completed', 'failed')",
            name="ck_retrieval_audits_status",
        ),
        CheckConstraint(
            "candidate_count >= 0 AND selected_count >= 0 AND rejected_count >= 0",
            name="ck_retrieval_audits_counts_nonneg",
        ),
        CheckConstraint(
            "candidate_count = selected_count + rejected_count",
            name="ck_retrieval_audits_counter_equality",
        ),
        # Lifecycle: pending has no completion/failure; completed has completion/no failure;
        # failed has completion and failure code.
        CheckConstraint(
            "(status = 'pending' AND completed_at IS NULL AND failure_code IS NULL) "
            "OR (status = 'completed' AND completed_at IS NOT NULL AND failure_code IS NULL) "
            "OR (status = 'failed' AND completed_at IS NOT NULL AND failure_code IS NOT NULL)",
            name="ck_retrieval_audits_lifecycle",
        ),
    )

    id = Column(String(36), primary_key=True)
    query_sha256 = Column(String(64), nullable=False)
    retrieval_mode = Column(String(10), nullable=False)
    status = Column(String(20), nullable=False)
    provenance_policy_version = Column(String(50), nullable=False)
    retrieval_policy_version = Column(String(50), nullable=False)
    context_policy_version = Column(String(50), nullable=False)
    candidate_count = Column(Integer, nullable=False, default=0, server_default="0")
    selected_count = Column(Integer, nullable=False, default=0, server_default="0")
    rejected_count = Column(Integer, nullable=False, default=0, server_default="0")
    failure_code = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    decisions = relationship(
        "RetrievalCandidateDecision", back_populates="audit", cascade="all, delete-orphan"
    )


class RetrievalCandidateDecision(Base):
    """Durable evidence snapshot for a retrieval candidate (10B.2)."""
    __tablename__ = "retrieval_candidate_decisions"
    __table_args__ = (
        UniqueConstraint(
            "audit_id", "chunk_id_snapshot", name="uq_candidate_decisions_audit_chunk"
        ),
        CheckConstraint(
            "provenance_score >= 0 AND provenance_score <= 1",
            name="ck_candidate_decisions_provenance_score",
        ),
        Index("ix_candidate_decisions_audit_decision", "audit_id", "decision"),
        Index("ix_candidate_decisions_live_doc", "document_id"),
        Index("ix_candidate_decisions_live_chunk", "chunk_id"),
        Index("ix_candidate_decisions_snapshot_doc", "document_id_snapshot"),
        Index("ix_candidate_decisions_snapshot_chunk", "chunk_id_snapshot"),
    )

    id = Column(Integer, primary_key=True)
    audit_id = Column(
        String(36), ForeignKey("retrieval_audits.id", ondelete="CASCADE"), nullable=False
    )
    document_id = Column(
        Integer, ForeignKey("documents.id", ondelete="SET NULL"), nullable=True
    )
    chunk_id = Column(
        Integer, ForeignKey("chunks.id", ondelete="SET NULL"), nullable=True
    )
    document_id_snapshot = Column(Integer, nullable=False)
    chunk_id_snapshot = Column(Integer, nullable=False)
    decision = Column(String(50), nullable=False)
    native_score = Column(Float, nullable=True)
    provenance_score = Column(Float, nullable=False)
    reason_codes = Column(Text, nullable=False)
    content_sha256 = Column(String(64), nullable=False)
    bounded_excerpt = Column(String(200), nullable=True)

    audit = relationship("RetrievalAudit", back_populates="decisions")


class IngestionRateBucket(Base):
    """Fixed-window ingestion rate limiter (10B.2/10B.3)."""
    __tablename__ = "ingestion_rate_buckets"
    __table_args__ = (
        CheckConstraint("request_count > 0", name="ck_ingestion_rate_buckets_count"),
    )

    identity_sha256 = Column(String(64), primary_key=True)
    window_start_epoch = Column(Integer, primary_key=True)
    request_count = Column(Integer, nullable=False)
