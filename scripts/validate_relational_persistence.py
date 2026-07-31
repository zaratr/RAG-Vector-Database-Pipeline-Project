"""Verify relational rows and Alembic state survive API container replacement."""
from __future__ import annotations

import json
import subprocess
import sys
import uuid


def run(*args: str) -> str:
    result = subprocess.run(
        ["docker", "compose", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"docker compose {' '.join(args)} failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def database_probe(action: str, marker: str) -> dict:
    code = r'''
import json
import sqlite3
import sys
from sqlalchemy.engine import make_url
from app.config import get_settings

action, marker = sys.argv[1], sys.argv[2]
path = make_url(get_settings().database_url).database
connection = sqlite3.connect(path)
connection.execute("PRAGMA foreign_keys=ON")
try:
    if action == "insert":
        cursor = connection.execute(
            "INSERT INTO documents (title, source, tags) VALUES (?, 'f4-persistence', 'validation')",
            (marker,),
        )
        document_id = cursor.lastrowid
        connection.execute(
            'INSERT INTO chunks (document_id, "index", text, start_offset, end_offset) VALUES (?, 0, ?, 0, ?)',
            (document_id, marker, len(marker)),
        )
        connection.commit()
    elif action == "cleanup":
        connection.execute("DELETE FROM documents WHERE title = ?", (marker,))
        connection.commit()

    document_count = connection.execute(
        "SELECT COUNT(*) FROM documents WHERE title = ?", (marker,)
    ).fetchone()[0]
    chunk_count = connection.execute(
        "SELECT COUNT(*) FROM chunks WHERE text = ?", (marker,)
    ).fetchone()[0]
    revision = connection.execute("SELECT version_num FROM alembic_version").fetchone()[0]
    print(json.dumps({"documents": document_count, "chunks": chunk_count, "revision": revision}))
finally:
    connection.close()
'''
    output = run(
        "exec",
        "-T",
        "api",
        "python",
        "-c",
        code,
        action,
        marker,
    )
    return json.loads(output)


def main() -> int:
    marker = f"f4-persistence-{uuid.uuid4()}"
    expected_revision = "dee48bc24a7f"
    try:
        before = database_probe("insert", marker)
        if before != {"documents": 1, "chunks": 1, "revision": expected_revision}:
            raise AssertionError(f"Unexpected state before recreation: {before}")

        run("up", "-d", "--force-recreate", "api")
        after = database_probe("read", marker)
        if after != before:
            raise AssertionError(
                f"Relational state did not survive API recreation: before={before}, after={after}"
            )
        print(json.dumps({"status": "passed", **after}, sort_keys=True))
        return 0
    finally:
        try:
            database_probe("cleanup", marker)
        except Exception as exc:
            print(f"cleanup failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())