"""Idempotent graph backfill CLI.

Exit codes:
* ``0`` — completed with ``failed=0`` (including no-op/idempotent runs).
* ``1`` — one or more chunk failures; successful chunks remain committed.
* ``2`` — invalid arguments, configuration failure, or database-wide fatal error.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.core.db import session_scope
from app.config import get_settings
from app.services.graph_backfill import backfill
from app.services.graph_extraction import (
    GraphExtractionError,
    get_graph_extractor,
)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Idempotent graph backfill.")
    parser.add_argument("--document-id", type=int, default=None)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    provider = "ollama"
    model = settings.graph_extraction_model or settings.llm_model
    try:
        extractor = get_graph_extractor()
    except GraphExtractionError as exc:
        sys.stderr.write(
            json.dumps({"error": "extractor_unavailable", "detail": str(exc)[:200]}) + "\n"
        )
        return 2

    try:
        with session_scope() as session:
            report = asyncio.run(
                backfill(
                    session,
                    extractor=extractor,
                    provider=provider,
                    model=model,
                    document_id=args.document_id,
                    retry_failed=args.retry_failed,
                    dry_run=args.dry_run,
                )
            )
    except ValueError as exc:
        sys.stderr.write(f"backfill_graph: {exc}\n")
        return 2

    payload = asdict(report)
    # sorted-key minified JSON.
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return 1 if payload["failed"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
