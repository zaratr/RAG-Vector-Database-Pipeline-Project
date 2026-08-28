# Content Safety Policy (Phase 10C)

## Scope

The content-safety policy is the versioned, immutable artifact that governs
deterministic first-pass classification of operator input, retrieved context,
and generated answers (scopes: `ingestion`, `context`, `answer`).

- Artifact: `config/content-safety-policy.json` — 1785 bytes (UTF-8, LF,
  sorted-key minified, final LF), SHA-256
  `2a9c9c5d4d44cce8ecb02bbf2b8586f6dd86dc410e474b93552e22180637d4f1`,
  policy version `safety-v1`.
- Fixture corpus: `app/tests/fixtures/content_safety.json` — 4627 bytes,
  SHA-256 `2560db13b33f42c986229d522702540a50a1a842f3fbb71bd8e9f25a94aa0758`,
  schema `content-safety-fixtures-v1`. Cases are lexical by ID and are the
  authoritative evidence for rule behavior — never regenerate them.

Both payloads are pinned bytes from the Phase 10 plan;
`app/tests/test_safety_policy.py` locates and byte-verifies them.

## Categories (stable IDs)

`violence`, `self_harm`, `sexual_content`, `hate_harassment`,
`illegal_activity`, `privacy_credentials`. Benign educational or contextual
discussions are mandatory negative-control fixtures, not a category.

## Actions and precedence

Actions are `allow`, `warn`, `filter`, `block`. Precedence is fixed:
`allow < warn < filter < block`. **Severity never overrides action** — a
severity-1 `block` outranks a severity-4 `warn` at the same span. Severity is
an integer 0–4 used for reporting and operator review only.

## Rules

Literal patterns (NFKC + casefold normalized matching with an index map back
to original-string offsets), each scoped to all three scopes in v1:
`SAF001_violence` (warn/3), `SAF002_self_harm` (block/4),
`SAF003_sexual_content` (block/4), `SAF004_hate_harassment` (filter/3),
`SAF005_illegal_activity` (block/4), `SAF006_privacy_credentials` (filter/4).

## Configuration

```
RAG_CONTENT_SAFETY_ENABLED=false
RAG_CONTENT_SAFETY_POLICY_PATH=/app/config/content-safety-policy.json
RAG_SAFETY_LLM_MODE=disabled|rules_only|fail_closed
```

Defaults render in API Compose and `.env.example`; the Dockerfile ships the
policy at the immutable-image path above.

## Loading and failure behavior

`load_safety_policy(path, fixture_path=None)` validates exact key sets,
stable category IDs, duplicate rule/category/case IDs, unknown
actions/scopes, severity ranges, credential-like tokens, and (with a fixture)
rule references, span offsets, corpus emptiness, case ordering, unmapped
matches, and unreferenced rules — any violation raises `ValueError`.

The API lifespan loads the policy exactly once when
`RAG_CONTENT_SAFETY_ENABLED=true` and injects the immutable object into
`app.state.content_safety_policy`. A missing, unreadable, or invalid policy
**aborts startup**. No request can replace the policy or its version.
