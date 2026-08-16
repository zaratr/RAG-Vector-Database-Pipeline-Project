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
            "is_identity_owner IN (0, 1)",
            name="ck_graph_extractions_is_identity_owner",
        ),
        CheckConstraint(
            "length(input_sha256) = 64 AND input_sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_graph_extractions_input_sha256_hex",
        ),
        CheckConstraint(
            "status != 'skipped' OR (error_code IN "
            "('extraction_disabled', 'unsupported_media_type', 'safety_blocked') "
            "AND attempt_count = 0)",
            name="ck_graph_extractions_skip_reason",
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
        CheckConstraint(
            "decision IN ('selected', 'rejected_distance', "
            "'rejected_blocked_source', 'rejected_source_cap', "
            "'rejected_document_cap', 'rejected_duplicate', "
            "'rejected_injection', 'rejected_safety')",
            name="ck_candidate_decisions_decision",
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


class SafetyReviewRun(Base):
    """Persisted safety review run (10C.4).

    Provenance is immutable through the snapshot columns: ingestion reviews
    hang off a document snapshot, context reviews off a retrieval audit plus
    chunk snapshot, answer reviews off the retrieval audit alone. Partial
    unique indexes make ``begin`` idempotent per review target.
    """
    __tablename__ = "safety_review_runs"
    __table_args__ = (
        CheckConstraint(
            "scope IN ('ingestion', 'context', 'answer')",
            name="ck_safety_runs_scope",
        ),
        CheckConstraint(
            "status IN ('pending', 'succeeded', 'failed')",
            name="ck_safety_runs_status",
        ),
        CheckConstraint(
            "llm_status IN ('skipped', 'succeeded', 'failed')",
            name="ck_safety_runs_llm_status",
        ),
        CheckConstraint(
            "final_action IS NULL OR final_action IN "
            "('allow', 'warn', 'filter', 'block')",
            name="ck_safety_runs_action",
        ),
        CheckConstraint(
            "length(input_sha256) = 64 AND input_sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_safety_runs_hash",
        ),
        CheckConstraint(
            "(scope = 'ingestion' AND document_id_snapshot IS NOT NULL "
            "AND chunk_id_snapshot IS NULL AND chunk_id IS NULL "
            "AND retrieval_audit_id IS NULL) OR "
            "(scope = 'context' AND document_id_snapshot IS NOT NULL "
            "AND chunk_id_snapshot IS NOT NULL "
            "AND retrieval_audit_id IS NOT NULL) OR "
            "(scope = 'answer' AND document_id_snapshot IS NULL "
            "AND chunk_id_snapshot IS NULL AND document_id IS NULL "
            "AND chunk_id IS NULL AND retrieval_audit_id IS NOT NULL)",
            name="ck_safety_runs_provenance",
        ),
        CheckConstraint(
            "(document_id IS NULL OR document_id = document_id_snapshot) "
            "AND (chunk_id IS NULL OR chunk_id = chunk_id_snapshot)",
            name="ck_safety_runs_live_ids",
        ),
        CheckConstraint(
            "(status = 'pending' AND completed_at IS NULL AND final_action "
            "IS NULL AND failure_code IS NULL AND llm_status = 'skipped') OR "
            "(status = 'succeeded' AND completed_at IS NOT NULL AND "
            "final_action IS NOT NULL AND failure_code IS NULL AND "
            "llm_status IN ('skipped', 'succeeded', 'failed')) OR "
            "(status = 'failed' AND completed_at IS NOT NULL AND final_action "
            "IS NULL AND failure_code IS NOT NULL AND llm_status IN "
            "('skipped', 'failed'))",
            name="ck_safety_runs_lifecycle",
        ),
        Index("ix_safety_runs_scope_status_created",
              "scope", "status", "created_at"),
        Index("ix_safety_runs_document", "document_id"),
        Index("ix_safety_runs_chunk", "chunk_id"),
        Index("ix_safety_runs_document_snapshot", "document_id_snapshot"),
        Index("ix_safety_runs_chunk_snapshot", "chunk_id_snapshot"),
        Index("ix_safety_runs_retrieval_audit", "retrieval_audit_id"),
        Index("ix_safety_runs_policy_version", "policy_version"),
        Index(
            "uq_safety_runs_ingestion_document",
            "document_id_snapshot", unique=True,
            sqlite_where=text("scope = 'ingestion'"),
        ),
        Index(
            "uq_safety_runs_context_audit_chunk",
            "retrieval_audit_id", "chunk_id_snapshot", unique=True,
            sqlite_where=text("scope = 'context'"),
        ),
        Index(
            "uq_safety_runs_answer_audit",
            "retrieval_audit_id", unique=True,
            sqlite_where=text("scope = 'answer'"),
        ),
    )

    id = Column(Integer, primary_key=True)
    scope = Column(String(20), nullable=False)
    status = Column(String(20), nullable=False, default="pending",
                    server_default="pending")
    document_id = Column(
        Integer, ForeignKey("documents.id", ondelete="SET NULL"), nullable=True)
    chunk_id = Column(
        Integer, ForeignKey("chunks.id", ondelete="SET NULL"), nullable=True)
    document_id_snapshot = Column(Integer, nullable=True)
    chunk_id_snapshot = Column(Integer, nullable=True)
    retrieval_audit_id = Column(
        String(36), ForeignKey("retrieval_audits.id", ondelete="CASCADE"),
        nullable=True)
    input_sha256 = Column(String(64), nullable=False)
    policy_version = Column(String(50), nullable=False)
    detector_version = Column(String(50), nullable=False)
    provider = Column(String(50), nullable=True)
    model = Column(String(255), nullable=True)
    prompt_version = Column(String(50), nullable=True)
    schema_version = Column(String(50), nullable=True)
    llm_status = Column(String(20), nullable=False, default="skipped",
                        server_default="skipped")
    final_action = Column(String(10), nullable=True)
    failure_code = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False,
                        server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    findings = relationship(
        "SafetyFinding", back_populates="review_run",
        cascade="all, delete-orphan",
    )


class SafetyFinding(Base):
    """Persisted safety finding for a review run (10C.4).

    Never carries full reviewed input: only bounded excerpts (v1 stores none)
    and the excerpt hash. ``source_rule_ids`` is the canonical sorted JSON
    array; source (deterministic/llm/merged) derives from the ID prefixes.
    """
    __tablename__ = "safety_findings"
    __table_args__ = (
        CheckConstraint(
            "category IN ('violence', 'self_harm', 'sexual_content', "
            "'hate_harassment', 'illegal_activity', 'privacy_credentials')",
            name="ck_safety_findings_category",
        ),
        CheckConstraint(
            "severity >= 0 AND severity <= 4",
            name="ck_safety_findings_severity",
        ),
        CheckConstraint(
            "action IN ('allow', 'warn', 'filter', 'block')",
            name="ck_safety_findings_action",
        ),
        CheckConstraint(
            "start_offset >= 0 AND end_offset > start_offset",
            name="ck_safety_findings_offsets",
        ),
        CheckConstraint(
            "length(excerpt_sha256) = 64 AND excerpt_sha256 NOT GLOB "
            "'*[^0-9a-f]*'",
            name="ck_safety_findings_hash",
        ),
        CheckConstraint(
            "bounded_excerpt IS NULL OR length(bounded_excerpt) <= 200",
            name="ck_safety_findings_excerpt_length",
        ),
        UniqueConstraint(
            "review_run_id", "category", "start_offset", "end_offset",
            "source_rule_ids", name="uq_safety_findings_run_span_rules",
        ),
        Index("ix_safety_findings_run_category_action",
              "review_run_id", "category", "action"),
    )

    id = Column(Integer, primary_key=True)
    review_run_id = Column(
        Integer, ForeignKey("safety_review_runs.id", ondelete="CASCADE"),
        nullable=False)
    category = Column(String(50), nullable=False)
    severity = Column(Integer, nullable=False)
    action = Column(String(10), nullable=False)
    start_offset = Column(Integer, nullable=False)
    end_offset = Column(Integer, nullable=False)
    source_rule_ids = Column(Text, nullable=False)
    excerpt_sha256 = Column(String(64), nullable=False)
    bounded_excerpt = Column(String(200), nullable=True)

    review_run = relationship("SafetyReviewRun", back_populates="findings")
