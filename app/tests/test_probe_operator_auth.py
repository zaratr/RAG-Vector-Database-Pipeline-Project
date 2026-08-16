"""Phase 10B — operator auth probe script tests (B-12, D-38/D-42).

Subprocess-level tests running scripts/probe_operator_auth.py against a
threaded local HTTP server whose status the test controls, plus assertions
on the OPERATOR_BEARER env-only contract.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _StatusHandler(BaseHTTPRequestHandler):
    status = 200

    def do_GET(self):  # noqa: N802 — http.server interface
        self.send_response(self.status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):  # silence
        pass


def _serve(status: int):
    server = HTTPServer(("127.0.0.1", 0), _StatusHandler)
    server.RequestHandlerClass.status = status
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_port}/security/audits/x"


def _run_probe(url: str, *extra: str, bearer: str = "probe-bearer-token-0123456789") -> subprocess.CompletedProcess:
    env = {**os.environ, "OPERATOR_BEARER": bearer}
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "probe_operator_auth.py"),
         "--url", url, *extra],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT), env=env, check=False,
    )


def test_probe_reports_404_after_successful_auth_as_authorized():
    server, url = _serve(404)
    try:
        result = _run_probe(
            url, "--assert-authorized", "--expected-status", "404",
            "--assertion-id", "valid-auth-unknown-id",
        )
    finally:
        server.shutdown()
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == 404
    assert payload["authorized"] is True


def test_probe_reports_401_as_unauthorized():
    server, url = _serve(401)
    try:
        result = _run_probe(
            url, "--assert-unauthorized", "--expected-status", "401",
            "--assertion-id", "invalid-bearer",
        )
    finally:
        server.shutdown()
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["authorized"] is False


def test_probe_reports_403_as_unauthorized():
    server, url = _serve(403)
    try:
        result = _run_probe(url, "--expected-status", "403")
    finally:
        server.shutdown()
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["authorized"] is False


def test_probe_status_mismatch_exits_two_without_secrets():
    server, url = _serve(404)
    try:
        result = _run_probe(
            url, "--expected-status", "200", "--assertion-id", "mismatch",
        )
    finally:
        server.shutdown()
    assert result.returncode == 2
    assert "probe-bearer-token" not in result.stdout + result.stderr


def test_probe_authorized_assertion_fails_for_unauthorized_status():
    server, url = _serve(401)
    try:
        result = _run_probe(url, "--assert-authorized", "--assertion-id", "should-fail")
    finally:
        server.shutdown()
    assert result.returncode == 2
    assert json.loads(result.stderr)["error"] == "expected_authorized"
