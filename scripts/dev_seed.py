"""Seed the API with a sample document (B-18 non-bypass contract).

Loads the active server source-trust policy, calls IngestionService with an
explicitly non-operator ``ingestion_origin='dev_seed_cli'`` and ``trust_tier``
resolved by the server policy (default ``untrusted``, never ``trusted``).
Refuses any source label matching a ``requires_operator=true`` or ``blocked``
rule, returning exit 2 with a bounded diagnostic.
"""
import asyncio
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.config import get_settings
from app.services.embeddings import get_embedding_provider
from app.services.ingestion import ingest_text
from app.services.vector_store import get_vector_store
from app.core.db import session_scope


def main() -> int:
    settings = get_settings()
    from app.services.provenance import load_source_trust_policy

    try:
        policy = load_source_trust_policy(settings.source_trust_policy_path)
    except Exception as exc:
        sys.stderr.write(f"dev_seed: policy load failed: {exc}\n")
        return 2

    source = "seed"

    # Refuse protected/blocked sources (B-18 non-bypass).
    if policy.is_blocked(source):
        sys.stderr.write(f"dev_seed: source '{source}' is blocked\n")
        return 2
    if policy.requires_operator(source):
        sys.stderr.write(f"dev_seed: source '{source}' requires operator; dev_seed is non-operator\n")
        return 2

    # Server-assigned trust: always untrusted for dev_seed_cli.
    trust_tier, trust_score = policy.assess(source, is_operator=False)

    with session_scope() as session:
        asyncio.run(
            ingest_text(
                title="Sample",
                source=source,
                tags=["example"],
                text="Hello world. This is a sample document for seeding.",
                embedding_provider=get_embedding_provider(),
                vector_store=get_vector_store(),
                session=session,
                trust_tier=trust_tier,
                trust_score=trust_score,
                trust_policy_version=policy.version,
                ingestion_origin="dev_seed_cli",
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
