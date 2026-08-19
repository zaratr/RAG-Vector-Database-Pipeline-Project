# Red-Team Methodology (Phase 10D)

## Purpose

Phase 10D measures, in a disposable and isolated environment, how much the
defenses built in Phases 10A–10C reduce attack success against the RAG
pipeline. It never attacks production: every run uses fresh, UUID-named
SQL databases and Chroma collections under `/tmp`, and production stores
are opened read-only for before/after fingerprints.

## Attack corpus

The corpus (`app/tests/fixtures/attack_payloads.json`) is versioned and
immutable: exactly 17757 UTF-8 bytes (sorted keys, minified, final LF)
with SHA-256
`bca0c8eed73c02a346b92658b9545620aa4565e5decb8befc6a74f541f8d03ce`,
`schema_version=attack-corpus-v1`, and `seed=42`. It contains one benign
control fixture and one malicious fixture for each required category:

| Category | Attack shape |
|---|---|
| vector_poisoning | Poisoned chunk closer in L2 than the clean chunk |
| low_relevance_poisoning | Poisoned chunk far away (distance 5.0) vs clean (0.1) |
| duplicate_flooding (exact/near-duplicate flooding) | Near-duplicate poison chunks at tiny distances crowding out clean evidence |
| source_spoofing | Client requests a trusted source; server assigns untrusted/blocked |
| retrieved_prompt_injection | Retrieved chunk injects an `OVERRIDE_ACCEPTED` instruction |
| metadata_aliasing | Chroma metadata claims another document's ID; SQL is authoritative |
| graph_relationship_poisoning | Poisoned evidence rides a path through a clean document |
| entity_alias_collision | Same canonical entity fed by clean and poisoned documents |
| context_flooding | Multiple poison chunks from one source flood the context window |
| safety_filter_evasion | Fullwidth (NFKC) text evading naive literal safety matching |

## Closed control registry

Every fixture names the control(s) expected to stop it; unknown IDs are
rejected. The registry maps to the owning phase contract:

| Control ID | Owner |
|---|---|
| `retrieval_distance` | 10B.3 distance |
| `retrieval_duplicate_cap` | 10B.3 duplicate/caps |
| `source_trust` | 10B.2 server trust |
| `context_injection` | 10B.4 |
| `sql_authority` | 10A.6/10B.3 SQL hydration |
| `graph_evidence` | 10A.2 evidence validation |
| `graph_identity` | 10A.3 canonical identity |
| `context_caps` | 10B.3 caps |
| `content_safety` | 10C.1–10C.4 |

## Evaluators

Outcome judgments are deterministic Python predicates over structured
retrieval/audit/answer output; LLM judgment is forbidden. Five types:
`poisoned_chunk_selected`, `answer_contains_literal`,
`graph_path_contains_poisoned_evidence`, `unsafe_answer_returned`, and
`control_not_triggered` (benign controls). `validate_attack_corpus` in
`app/services/attack_simulator.py` enforces the structural schema
(Draft 2020-12, closed at every level, type-conditional branches) plus
the semantic invariants: lexical fixture/document order, globally unique
IDs, reference resolution, exactly one malicious + one benign fixture per
category, exactly one non-poisoned required-clean document per fixture,
severity 4/1 pairing, finite nonnegative L2 distances, and absence of
credential-shaped material.

## Determinism

All arrays execute in lexical fixture/document ID order using seed 42.
Disabled and enabled modes consume byte-identical fixture inputs; the
`FixtureInputManifest` (Task 10D.2) proves input equality before any
measurement, and each mode's result derives only from its own outputs.

## Disabled/enabled modes (Task 10D.2 preview)

Each report run executes the corpus twice through the exact production
ingestion path: a `disabled` mode that observes but does not enforce the
measured trust/retrieval/context/content-safety controls, and an
`enabled` mode that enforces them. Baseline invariants (SQL authority,
readiness, bounded work, escaping, citation grounding) stay active in
both. Task 10D.3 derives the metrics (attack success rate,
poisoned-context share, clean retrieval recall, false-positive rate,
graph-path contamination, blocked-generation count, latency overhead)
and the acceptance thresholds from these paired runs.
