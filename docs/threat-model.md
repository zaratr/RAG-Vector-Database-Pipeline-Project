# Phase 10 Threat Model — RAG Vector Database Pipeline

**OWASP Top 10 for LLM Applications 2025**
- Edition URL: https://genai.owasp.org/llm-top-10/
- Retrieved: 2026-08-01
- Retained excerpt SHA-256: e024da7f5a562882e3ba8c8eae62d74db29d9aa86ec20b5512ad6be58c0c6200

---

## OWASP LLM Top 10 2025 ID Mapping

| OWASP ID | Full Name | Phase 10 Role |
|---|---|---|
| LLM01:2025 Prompt Injection | LLM01:2025 Prompt Injection | **Primary** — retrieved_prompt_injection |
| LLM02:2025 Sensitive Information Disclosure | LLM02:2025 Sensitive Information Disclosure | **Primary** — credential_leakage_llm02 |
| LLM03:2025 Supply Chain | LLM03:2025 Supply Chain | not_primary_for_phase10 (dependency integrity is out of scope; no third-party model/dependency attestation in this portfolio system) |
| LLM04:2025 Data and Model Poisoning | LLM04:2025 Data and Model Poisoning | **Primary** — corpus_or_data_poisoning |
| LLM05:2025 Improper Output Handling | LLM05:2025 Improper Output Handling | not_primary_for_phase10 (no downstream system consuming structured LLM output beyond the grounded answer; output is text returned to the caller) |
| LLM06:2025 Excessive Agency | LLM06:2025 Excessive Agency | not_primary_for_phase10 (no autonomous tool use, function calling, or agentic execution; the LLM only generates grounded answers from retrieved context) |
| LLM07:2025 System Prompt Leakage | LLM07:2025 System Prompt Leakage | **Primary** — credential_leakage_llm07 |
| LLM08:2025 Vector and Embedding Weaknesses | LLM08:2025 Vector and Embedding Weaknesses | **Primary** — vector_or_embedding_attack |
| LLM09:2025 Misinformation | LLM09:2025 Misinformation | not_primary_for_phase10 (grounded generation with explicit no-evidence responses; misinformation detection as a distinct control is not implemented in this phase) |
| LLM10:2025 Unbounded Consumption | LLM10:2025 Unbounded Consumption | **Primary** — resource_flooding |

---

## Threat Scenarios

### TFS-01-untrusted-uploader-trusted-source

- **protected_asset:** Document trust tier assignment and retrieval provenance
- **trust_boundary:** Public API ingestion → server-assigned trust policy
- **precondition:** Attacker has public (unauthenticated) access to POST /documents
- **attack:** Uploader labels content with `source=operator-curated` (a trusted source) to influence retrieval ranking
- **expected_control:** Server-assigned trust; client `source` is descriptive only; `requires_operator=true` rules reject unauthenticated trusted assignment (LLM04)
- **audit_evidence:** retrieval_audits row with `provenance_policy_version` and candidate decisions showing the untrusted tier
- **residual_risk:** Operator credential compromise could assign trust manually
- **fixture_id:** TFS-01

### TFS-02-context-flooding-single-source

- **protected_asset:** Retrieval diversity and context window budget
- **trust_boundary:** Retrieval → LLM context construction
- **precondition:** Attacker ingests many chunks from one source/document
- **attack:** Flooding the vector index with chunks from one source to dominate top-k results
- **expected_control:** Per-query cap, per-document cap, source cap; deduplication and poisoning controls (LLM04)
- **audit_evidence:** retrieval_candidate_decisions showing rejected_duplicate and cap-based rejections
- **residual_risk:** A determined attacker with high ingestion volume may partially saturate results before caps engage
- **fixture_id:** TFS-02

### TFS-03-exact-near-duplicate-poisoning

- **protected_asset:** Retrieval result diversity and relevance
- **trust_boundary:** Ingestion → vector index → retrieval
- **precondition:** Attacker can ingest documents with near-identical content
- **attack:** Submitting exact or near-duplicate poisoned chunks to crowd out legitimate results
- **expected_control:** Exact-duplicate and near-duplicate (Jaccard similarity) detection at retrieval time; deduplication by content SHA-256 (LLM04)
- **audit_evidence:** retrieval_candidate_decisions with reason_code `exact_duplicate` or `near_duplicate`
- **residual_risk:** Semantically similar but lexically distinct content bypasses Jaccard threshold
- **fixture_id:** TFS-03

### TFS-04-low-relevance-poisoned-chunks

- **protected_asset:** Retrieval precision and answer grounding
- **trust_boundary:** Vector similarity → candidate selection
- **precondition:** Poisoned chunks are ingested and indexed
- **attack:** Injecting low-relevance chunks that match the query embedding but contain adversarial content
- **expected_control:** Distance threshold calibration; low-distance rejection; provenance scoring (LLM04)
- **audit_evidence:** retrieval_candidate_decisions with reason_code `rejected_distance`
- **residual_risk:** Threshold may reject legitimate low-distance results if miscalibrated
- **fixture_id:** TFS-04

### TFS-05-chroma-metadata-record-id-aliasing

- **protected_asset:** SQL-authoritative chunk identity and text integrity
- **trust_boundary:** Chroma vector store → SQL hydration
- **precondition:** Chroma record IDs or metadata can be manipulated (aliasing, reuse)
- **attack:** Creating Chroma records with aliased or duplicate vector IDs that map to different SQL chunks
- **expected_control:** SQL-authoritative hydration: every Chroma hit must resolve to a ready SQL chunk whose vector_id matches; alias/stale IDs rejected (LLM08)
- **audit_evidence:** retrieval_candidate_decisions showing rejected candidates due to vector_id mismatch
- **residual_risk:** Direct Chroma manipulation outside the API bypasses controls
- **fixture_id:** TFS-05

### TFS-06-graph-entity-alias-collision-relationship-poisoning

- **protected_asset:** Graph provenance and multi-hop retrieval correctness
- **trust_boundary:** Graph extraction → graph traversal → retrieval
- **precondition:** Attacker can ingest documents that extract graph entities/relationships
- **attack:** Creating entities with colliding canonical names but different types, or injecting false relationships
- **expected_control:** Canonical entity identity by (canonical_name, entity_type); grounded evidence with exact source offsets; ready-only filtering (LLM08)
- **audit_evidence:** Graph extraction evidence rows with source offsets; retrieval paths traceable to ready chunks
- **residual_risk:** Entity aliasing across types could create unexpected multi-hop paths
- **fixture_id:** TFS-06

### TFS-07-retrieved-prompt-injection

- **protected_asset:** LLM instruction integrity and system prompt confidentiality
- **trust_boundary:** Retrieved context → LLM prompt construction
- **precondition:** Poisoned content containing prompt-injection payloads is retrieved
- **attack:** Embedded instructions in retrieved chunks (e.g., "ignore previous instructions") hijack the LLM
- **expected_control:** Context-security policy detection (CTX001–CTX006); quarantine/block actions; untrusted evidence wrapping; system prompt immutability (LLM01)
- **audit_evidence:** retrieval_audits with context-security findings; blocked/quarantined candidate decisions
- **residual_risk:** Novel injection patterns not covered by the literal-pattern policy may bypass detection
- **fixture_id:** TFS-07

### TFS-08-audit-tampering-sensitive-data-disclosure

- **protected_asset:** Audit record integrity and sensitive-data confidentiality
- **trust_boundary:** Audit persistence → operator API → external visibility
- **precondition:** Audits contain query hashes, candidate hashes, and bounded excerpts
- **attack:** Attempting to tamper with audit records or extract sensitive data (credentials, queries, content) from audit responses
- **expected_control:** Immutable audit records; candidate content SHA-256 without raw text; `bounded_excerpt=NULL` for untrusted candidate text; secret scanning in reports (LLM02, LLM07)
- **audit_evidence:** No raw query, document text, or credential in any audit or report artifact
- **residual_risk:** Operator with direct DB access could read audit tables
- **fixture_id:** TFS-08

### TFS-09-cross-tenant-retrieval-deferred

- **protected_asset:** Tenant data isolation in retrieval
- **trust_boundary:** Multi-tenant query routing (not implemented)
- **precondition:** Multiple tenants share the same vector index and SQL database
- **attack:** Querying for content belonging to another tenant
- **expected_control:** None in Phase 10 — explicitly deferred until multi-tenancy exists, not claimed solved (see Formal Amendment A-01)
- **audit_evidence:** N/A (no tenant-scoped audit exists)
- **residual_risk:** Cross-tenant data leakage is possible in a shared-index deployment
- **fixture_id:** TFS-09
