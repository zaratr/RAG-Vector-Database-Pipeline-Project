"""Operator auth probe (B-12).

Reads bearer token from OPERATOR_BEARER env var only — never from argv.
Constructs the Authorization header internally using urllib.
On success emits one sorted-key JSON line and exits 0.
On mismatch exits 2 without emitting any secret.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import urllib.error


def _probe(url: str, bearer: str, expected_status: int, assert_authorized: bool) -> dict:
    """Probe the operator auth endpoint and return the result."""
    req = urllib.request.Request(url, method="GET")
    req.add_header("Authorization", f"Bearer {bearer}")
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        status = resp.getcode()
    except urllib.error.HTTPError as exc:
        status = exc.code
    except Exception:
        status = 0

    # "Authorized" means the request was NOT rejected by authentication:
    # a valid credential on an unknown resource yields 404 AFTER auth
    # succeeds, so only 401/403 count as unauthorized (D-38).
    authorized = status not in (0, 401, 403)
    return {"status": status, "authorized": authorized}


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe operator auth.")
    parser.add_argument("--assert-authorized", action="store_true")
    parser.add_argument("--assert-unauthorized", action="store_true")
    parser.add_argument("--expected-status", type=int, default=None)
    parser.add_argument("--assertion-id", default="probe")
    parser.add_argument("--url", default="http://127.0.0.1:8000/security/audits/probe-test")
    parser.add_argument("--debug-argv", action="store_true")
    args = parser.parse_args()

    if args.debug_argv:
        # Print argv without any environment secrets for debugging.
        print(json.dumps({"argv": sys.argv}))
        return 0

    bearer = os.environ.get("OPERATOR_BEARER", "")
    # Clear from own process environment before any network call.
    if "OPERATOR_BEARER" in os.environ:
        del os.environ["OPERATOR_BEARER"]

    result = _probe(args.url, bearer, args.expected_status or 0, args.assert_authorized)

    if args.expected_status is not None and result["status"] != args.expected_status:
        sys.stderr.write(json.dumps({"assertion_id": args.assertion_id, "status": result["status"], "error": "status_mismatch"}))
        return 2

    if args.assert_authorized and not result["authorized"]:
        sys.stderr.write(json.dumps({"assertion_id": args.assertion_id, "error": "expected_authorized"}))
        return 2

    if args.assert_unauthorized and result["authorized"]:
        sys.stderr.write(json.dumps({"assertion_id": args.assertion_id, "error": "expected_unauthorized"}))
        return 2

    output = {
        "assertion_id": args.assertion_id,
        "status": result["status"],
        "authorized": result["authorized"],
    }
    sys.stdout.write(json.dumps(output, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
