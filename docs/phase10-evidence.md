# Phase 10 requirement-to-evidence map

This document is repo-self-contained: every claim below is re-derivable from a
clean clone by running the cited test or live command. All live commands are
the hermetic recipe — run from the repository root with the pinned
dependencies installed — and need no Ollama, no Chroma server, no model
downloads, and no Docker daemon. Validation of this map was performed on a
bare Windows host; see *Limitations*.

Columns: claim ID, phase/task, requirement summary, source path + symbol,
test path :: name, live command (hermetic), expected invariant.

## Requirement-to-evidence table

| Claim ID | Phase/task | Requirement summary | Source path + symbol | Test path :: name | Live command (hermetic) | Expected invariant |
|---|---|---|---|---|---|---|
| [EVID-F1] | F1 | Durable vector foundation: persistent Chroma, not an ephemeral process-local client | `app/services/vector_store.py` :: `ChromaVectorStore` | `app/tests/test_vector_store.py` :: `test_http_mode_when_host_set` | `RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest -q app/tests/test_vector_store.py` | Host/persist/directory selection yields HTTP, persistent, or ephemeral mode exactly as configured; upserts idempotent under deterministic chunk IDs |
| [EVID-F2AMENDED] | F2 (amended by 10A.5) | Graph traversal no longer uses NetworkX; production traversal is relational | `app/services/graph_retrieval.py` (no networkx import) | `app/tests/test_graph_retrieval.py` (module) | `RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest -q app/tests/test_graph_retrieval.py` | Multi-hop traversal runs over persisted relational evidence with no NetworkX dependency in production paths |
| [EVID-F3] | F3 | Truthful ingestion: markdown/input failures match documented behavior | `app/services/ingestion.py` :: staged→ready lifecycle | `app/tests/test_ingestion.py` (module) | `RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest -q app/tests/test_ingestion.py` | Explicit staged/ready/failed states; empty text, invalid UTF-8, corrupt PDFs rejected explicitly |
| [EVID-F4] | F4 | Alembic owns schema evolution safely | `app/persistence/alembic/versions/` (chain) | `app/tests/test_migrations.py` :: `test_baseline_creates_exact_schema_on_empty_db` | `RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest -q app/tests/test_migrations.py` | Fresh, legacy-adoption, upgrade, downgrade/re-upgrade, and drift paths preserve data and pin the exact schema |
| [EVID-10A-01] | 10A | Deterministic hybrid (RRF) retrieval | `app/services/retrieval.py` :: hybrid fusion | `app/tests/test_hybrid_retrieval.py` :: `test_hybrid_rrf_deduplicates_chunk_and_preserves_native_score_and_paths` | `RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest -q app/tests/test_hybrid_retrieval.py` | Chunk-level dedup with native and fusion scores retained separately; identical inputs produce identical results |
| [EVID-10A-02] | 10A | Persisted graph provenance (extractions, entities, mentions, edges, evidence) | `app/persistence/graph_repository.py` | `app/tests/test_graph_persistence.py` (module) | `RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest -q app/tests/test_graph_persistence.py` | Logical identity separated from per-extraction evidence; same-name entities of different types never merge |
| [EVID-10A-03] | 10A / DOC.1 | Exact durable migration head is `d9b5f7c1e4a3` | `app/persistence/alembic/versions/d9b5f7c1e4a3_add_safety_reviews.py` :: `revision` | `app/tests/test_nonsecret_scripts.py` :: `test_fingerprint_reports_heads_counts_and_pks_without_content` | `RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest -q app/tests/test_nonsecret_scripts.py` | Fingerprint payload reports `alembic_head == "d9b5f7c1e4a3"`; chain head has no children |
| [EVID-10A-06] | 10A.6 | SQL authority: only ready documents are query-visible; reconciliation converges Chroma to ready SQL chunks | `app/services/reconciliation.py` | `app/tests/test_reconciliation.py` :: `test_reconciliation_hides_nonready_and_idempotently_repairs_ready_vectors` | `RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest -q app/tests/test_reconciliation.py` | Non-ready vectors deleted, orphans removed, ready chunks idempotently re-upserted; second run mutates nothing |
| [EVID-10B-01] | 10B.3 | Complete-envelope byte limits (request/file/extracted) enforced before handlers | `app/main.py` :: `BoundedReceiveMiddleware` | `app/tests/test_ingestion_limits.py` :: `test_request_envelope_one_byte_over_limit_rejected_before_handler` (Content-Length fast-reject, `request_too_large`) and `::test_request_envelope_streamed_count_over_limit_rejected_before_handler` (streamed count, `request_envelope_too_large`) | `RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest -q app/tests/test_ingestion_limits.py` | 413 before any route handler at limit+1 on both paths; exact-limit requests pass; no partial SQL/Chroma/audit rows |
| [EVID-10B-02] | 10B.3 | Atomic fixed-window rate limiting shared across workers and restarts | `app/services/ingestion_limits.py` :: `acquire_slot` | `app/tests/test_ingestion_limits.py` :: `test_rate_limit_concurrent_workers_produce_distinct_counts` | `RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest -q app/tests/test_ingestion_limits.py` | Under N concurrent workers every accepted request has a distinct count 1..limit; losers capped; 429 with `Retry-After` and rate headers |
| [EVID-10B-OPAUTH] | 10B | Operator API is single-operator, static-token, disabled by default | `app/tests/test_auth.py` (operator bearer enforcement) | `app/tests/test_auth.py` :: `test_operator_api_enabled_invalid_bearer_returns_401` | `RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest -q app/tests/test_auth.py` | Disabled → 404; enabled without/with invalid bearer → 401; token never compared in non-constant-time or logged |
| [EVID-10B-04] | 10B | Security-audit persistence and bounded retention pruning | `scripts/prune_security_audits.py` | `app/tests/test_prune_security_audits_cli.py` :: `test_prune_retains_pending_audits` | `RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest -q app/tests/test_prune_security_audits_cli.py` | Old completed audits deleted, pending retained, dry-run preserves all data, disposable-DB path guard enforced |
| [EVID-10B-05] | 10B.3 | Retrieval-security policy is regime-pinned and fails closed | `scripts/validate_phase10b.py` + `config/retrieval-security-policy.json` | `app/tests/test_validate_phase10b.py` :: `test_regime_precondition_fails_fast_under_local_provider` | `RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest -q app/tests/test_validate_phase10b.py` | Wrong embedding regime/model fails fast with typed `provider_mismatch` (exit 2); no distance verdicts under an uncalibrated regime |
| [EVID-10C-01] | 10C.1 | Content-safety policy is immutable and pinned by bytes | `config/content-safety-policy.json` | `app/tests/test_safety_policy.py` :: `test_policy_byte_count_and_sha256_immutable` | `RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest -q app/tests/test_safety_policy.py` | Policy byte count and SHA-256 match pins; unknown rule IDs rejected |
| [EVID-10C-02] | 10C | Generation-time safety enforcement, fail-closed | `app/api/routes_query.py` (safety gating) | `app/tests/test_safety_api.py` (module) | `RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest -q app/tests/test_safety_api.py` | Blocked-all retrieval still returns 200 with the deterministic no-safe-context answer; safety failures fail closed |
| [EVID-10D-01] | 10D.1 | Isolated attack corpus pinned by bytes and schema | `app/tests/fixtures/attack_payloads.json` | `app/tests/test_attack_corpus.py` :: `test_corpus_byte_count_and_sha256_immutable` | `RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest -q app/tests/test_attack_corpus.py` | Corpus bytes/schema pinned; fixture and document IDs unique; every required category present |
| [EVID-10D-02] | 10D | Closed control registry maps each control ID to its owner phase | `app/tests/test_attack_corpus.py` :: `CLOSED_CONTROL_REGISTRY` | `app/tests/test_attack_corpus.py` :: `test_control_registry_mapping_to_10b_10c` | `RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest -q app/tests/test_attack_corpus.py` | Registry keys equal the 10A/10B/10C owner mapping exactly (closed set) |
| [EVID-10D-03] | 10D | Reproducible red-team report schema | `app/tests/fixtures/redteam-report.schema.json` | `app/tests/test_redteam_report.py` :: `test_report_schema_is_draft2020_12_with_id` | `RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest -q app/tests/test_redteam_report.py` | Draft 2020-12 schema with `$id`, `additionalProperties: false` at every level, cross-field numerator ≤ denominator |
| [EVID-10D-04] | 10D | Red-team defense-effectiveness threshold: relative ASR reduction ≥ 0.60 | `docs/red-team-methodology.md` (metric definitions) | `app/tests/test_redteam_metrics.py` (module) | `RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest -q app/tests/test_redteam_metrics.py` | ASR and relative-reduction metrics computed per methodology; threshold ≥ 0.60 is the documented acceptance bound |
| [EVID-DOC1-01] | DOC.1 | Evidence map exists with the required columns | `docs/phase10-evidence.md` (header row) | `app/tests/test_documentation_truth.py` :: `test_evidence_file_exists_and_has_required_columns` | `RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest -q app/tests/test_documentation_truth.py` | Header contains claim id, phase/task, source path, test path, live command, expected invariant |
| [EVID-DOC1-02] | DOC.1 | Every README completion/quantitative claim maps to an evidence row | `README.md` (`[EVID-*]` anchors) | `app/tests/test_documentation_truth.py` :: `test_every_completion_claim_in_readme_maps_to_evidence_row` | `RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest -q app/tests/test_documentation_truth.py` | All `[EVID-*]` IDs in README exist here; every quantitative line carries an `[EVID-` anchor |
| [EVID-DOC1-03] | DOC.1 | Image hygiene scanner is host-side; refuses in-container execution | `scripts/validate_image_hygiene.py` :: `_is_inside_container` | `app/tests/test_image_hygiene_script.py` :: `test_script_is_host_side_never_runs_inside_api` | `RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest -q app/tests/test_image_hygiene_script.py` | Container-marker presence → exit 2 with host/container refusal; host guard passes on the host; zero Docker dependency in the lane |
| [EVID-DOC1-04] | DOC.1 | Final-rootfs scan rejects forbidden artifact classes | `scripts/validate_image_hygiene.py` :: `_scan_tarball` | `app/tests/test_image_hygiene_script.py` :: `test_forbidden_git_directory_detected` (and the `.env`/`.hermes`/db/report/host-path/credential-sentinel/fixture-allowlist sibling lanes) | `RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest -q app/tests/test_image_hygiene_script.py` | `.git`, `.env*`, `.hermes`, `rag.db`/-wal/-shm, report files, host absolute paths, credential sentinels, and out-of-allowlist fixtures each fail the scan; matched secret bytes never echoed |
| [EVID-DOC1-05] | DOC.1 | Pinned policy hash inventory verified inside the image | `scripts/validate_image_hygiene.py` :: `PINNED_POLICY_HASHES` | `app/tests/test_image_hygiene_script.py` :: `test_pinned_policy_hash_inventory_verified` | `RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest -q app/tests/test_image_hygiene_script.py` | Tampered policy content at a pinned path → exit 2 naming the path; the four pins equal SHA-256 of the committed `config/*.json` |
| [EVID-DOC1-06] | DOC.1 | Cleanup of temporary container and tarball in `finally` | `scripts/validate_image_hygiene.py` :: `main` (finally) | `app/tests/test_image_hygiene_script.py` :: `test_cleanup_removes_container_and_tarball_in_finally` | `RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest -q app/tests/test_image_hygiene_script.py` | `docker rm` issued even on scan failure; `cleanup_complete: true`; rm/export failure → exit 2 |
| [EVID-DOC1-07] | DOC.1 | Subprocess argv discipline: exact docker argv, `shell=False`, no socket mount | `scripts/validate_image_hygiene.py` :: `_run` | `app/tests/test_image_hygiene_script.py` :: `test_export_uses_docker_export_to_host_tar`, `::test_no_shell_true_in_any_subprocess_call`, `::test_no_docker_socket_mounted` | `RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest -q app/tests/test_image_hygiene_script.py` | `docker create --entrypoint /bin/true`, exactly `["docker","export",<id>]`, `docker rm`; no `shell=True`; no `/var/run/docker.sock` in any argv |
| [EVID-DOC1-08] | DOC.1 (Amendment 2 split) | Envelope over-limit rejection is pinned per path | `app/main.py` :: `BoundedReceiveMiddleware` | `app/tests/test_ingestion_limits.py` :: `test_request_envelope_one_byte_over_limit_rejected_before_handler` (pins `request_too_large`) and `::test_request_envelope_streamed_count_over_limit_rejected_before_handler` (pins `request_envelope_too_large`) | `RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest -q app/tests/test_ingestion_limits.py` | Production behavior unchanged: 413 before the handler on both the Content-Length fast path and the streamed-count path |
| [EVID-DOC1-SUITE] | DOC.1 (Amendment 1 R1) | Hermetic full-suite collection count | repository test suite | `app/tests/` (full suite) | `RAG_EMBEDDING_PROVIDER=local RAG_CHROMA_HOST= python -m pytest --collect-only -q` | A clean clone collects the full suite with zero errors (955 tests after the post-approval scanner defect fixes and their four regression lanes, plus the dev-seed child-env hermeticity fix); full run passes with only declared environment-conditional skips — including with a production-selecting `.env` present, since the dev-seed lane builds its child environment explicitly |

## Limitations

Windows-bare only; 19 lanes never executed on POSIX (POSIX CI deferred by
owner). Every command recorded in this map was executed on a bare Windows
host with the project virtualenv interpreter. The suite's POSIX-only lanes
(retrieval-security calibration runs, the phase-10D attack-harness
disposable-store lanes, and one symlink lane) pin disposable
database URLs to POSIX absolute `sqlite:////` paths as a destructive-tool
path guard and are skipped on Windows; no POSIX execution is claimed
anywhere in this document. In the recorded full-suite run, 22 lanes were
skipped in total: 19 POSIX-only lanes (15 attack-harness lanes, 3
calibration lanes, 1 symlink lane) and 3 opt-in live lanes that require an
explicitly set environment variable and are skipped by design.

## Adaptation record (owner ruling D2)

The appendix specifies 41 DOC.1 test functions; 12 + 27 are implemented
here with exact appendix names. Five were ADAPT-REQUIRED; the exact
adaptation versus the appendix sketch:

1. **T02** (`test_readme_cites_current_migration_head`) — unchanged in
   substance; the literal head `d9b5f7c1e4a3` was verified to be the actual
   Alembic head at the DOC.1 branch point (no child revisions) before
   pinning, and is asserted both in README text and by
   `app/tests/test_nonsecret_scripts.py::test_fingerprint_reports_heads_counts_and_pks_without_content`.
2. **T09** (`test_readme_does_not_equate_phase10_with_public_production_readiness`)
   — adapted wording: the appendix's alternative clause referenced the
   superseded separate final-release gate; the implemented assertion pins
   the residual-risk truth only — README must not equate Phase 10 with
   public production readiness.
3. **T10** (`test_evidence_file_exists_and_has_required_columns`) — adapted
   column set: the appendix's "validator report path" and "approval
   verdict" columns referenced superseded gate artifacts; the evidence doc
   instead carries exactly seven columns — claim ID, phase/task,
   requirement summary, source path + symbol, test path :: name, live
   command (hermetic), and expected invariant — per owner ruling Q2
   (repo-self-contained evidence). The test locates the document's first
   markdown table row and asserts that header row matches those seven
   columns exactly (the appendix sketch read line 1, which is the document
   title, not the table).
4. **H01** (`test_script_is_host_side_never_runs_inside_api`) — the
   appendix lane executed the script expecting the in-API-container
   refusal, contradicting the host-side scanner contract; the adapted lane
   asserts the same host-guard substance runnable anywhere: with container
   markers forced on, the scanner exits 2 with a host/container refusal;
   with the guard off, a clean mocked scan proceeds to exit 0.
5. **H16** (`test_pinned_policy_hash_inventory_verified`) — pins
   recomputed against the current `config/*.json` at the DOC.1 branch
   point: the appendix values for source-trust, content-safety, and
   context-security still match, and retrieval-security
   (`1cc5310fffaf28bbefcf2debf6aa5fbf31ff88d81d7997d77d5c96f0c2acf1bf`)
   was added so the inventory covers all four committed policies.

Post-approval corrections recorded here: (a) the owner's "20 lanes never
executed on POSIX" figure was corrected to the verified on-disk actuals
(19 POSIX-only + 3 opt-in = 22 skipped); (b) the 10D D4 envelope
over-limit lane split (Amendment 2) is necessarily net +1 test, which is
the ratified basis for the final collected count; and (c) the fix batch
after the 2026-09-02 program audit adds the docker-absent regression lane
(`test_regression_docker_absent_clean_exit_two`, net +1 → 955 collected,
31 hygiene lanes) and makes the dev-seed provenance lane build its child
environment explicitly so the README §9 `.env`-override recipe holds in a
tree with `.env` present (Windows drops empty-valued inherited variables;
previously the empty `RAG_CHROMA_HOST` override never reached the child).

Additionally, the `test_allowed_attack_fixtures_pass` lane maps image paths
to repository paths by stripping the image root prefix (`/app/`) — the
appendix sketch's `lstrip("/")` mapping double-prefixed `app/` and could
not resolve the fixture files.

**SUPERSEDED (not implemented, by owner ruling D2):**
`test_required_approval_reports_exist` (T12) and
`test_approval_reports_have_appproved_verdict` (T13) asserted the deleted
approval-pair registry machinery; under the amended contract the evidence
of record is the hermetic suite (R1) plus this requirement-to-evidence map
(R2), so both functions are superseded and intentionally absent.
