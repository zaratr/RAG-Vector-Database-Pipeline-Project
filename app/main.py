"""FastAPI application entrypoint."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from app.api import routes_documents, routes_graph, routes_query, routes_security
from app.config import get_settings
from app.core.logging import logger

settings = get_settings()


class BoundedReceiveMiddleware:
    """ASGI middleware that counts all incoming body bytes before any handler (B-10).

    If the total body exceeds ``max_bytes``, the middleware sends a 413 response
    and does not call the downstream app. This bounds the entire HTTP/multipart
    envelope — ``Content-Length`` is a hint; the middleware counts actual bytes.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = 10 * 1024 * 1024) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Pre-check Content-Length header if present.
        headers = dict(
            (k.decode("latin-1").lower(), v.decode("latin-1"))
            for k, v in scope.get("headers", [])
        )
        content_length = headers.get("content-length")
        if content_length:
            try:
                cl = int(content_length)
                if cl > self.max_bytes:
                    await self._send_413(send, code="request_too_large")
                    return
            except ValueError:
                pass  # untrusted header; fall through to byte counting

        # Eagerly drain the request stream so the complete envelope is bounded
        # even when the downstream app never reads the body (B-10). Missing or
        # invalid Content-Length is untrusted: every chunk is counted.
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.request":
                chunk = message.get("body", b"")
                if chunk:
                    body.extend(chunk)
                    if len(body) > self.max_bytes:
                        await self._send_413(send, code="request_envelope_too_large")
                        return
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                return  # client went away mid-stream; nothing to serve

        payload = bytes(body)

        async def replay_receive() -> Receive:
            return {"type": "http.request", "body": payload, "more_body": False}

        await self.app(scope, replay_receive, send)

    async def _send_413(self, send: Send, code: str = "request_envelope_too_large") -> None:
        import json
        body = json.dumps({"detail": {"code": code}}).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})


class _BodyTooLarge(Exception):
    """Retained for backward compatibility with earlier 10B.3 imports."""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and validate all security policies before readiness.

    Missing/invalid/unreadable policy aborts startup.
    """
    # 10B.3: startup hard-aborts on invalid ingestion-limit configuration
    # (ranges and the envelope > file cross-field rule live in Settings).
    from app.config import get_settings as _get_settings

    boot_settings = _get_settings()

    from app.services.provenance import load_source_trust_policy

    try:
        policy = load_source_trust_policy(settings.source_trust_policy_path)
        logger.info("Source-trust policy loaded: version=%s, rules=%d",
                     policy.version, len(policy.rules))
        app.state.source_trust_policy = policy
    except Exception as exc:
        logger.error("FATAL: source-trust policy load failed: %s", exc)
        raise

    # 10B.3: load retrieval security policy — hard abort on missing/invalid.
    # Strict validation (exact keys, ranges, metric, fixture hash) lives in the
    # service loader; the immutable object is shared with app.services.retrieval.
    try:
        from app.services.retrieval import reset_retrieval_security_policy_cache
        from app.services.retrieval_security import load_retrieval_security_policy_strict

        reset_retrieval_security_policy_cache()
        policy = load_retrieval_security_policy_strict(
            settings.retrieval_security_policy_path
        )
        app.state.retrieval_security_policy = policy
        logger.info("Retrieval security policy loaded: version=%s, max_distance=%s",
                     policy.version, policy.max_distance)
    except Exception as exc:
        logger.error("FATAL: retrieval security policy load failed: %s", exc)
        raise

    # 10B.4: load context security policy — hard abort on missing/invalid
    # (strict loader validates exact key sets, rule IDs, actions, version).
    try:
        from app.services.context_security import (
            get_context_security_policy,
            reset_context_security_policy_cache,
        )
        reset_context_security_policy_cache()
        ctx_policy = get_context_security_policy()
        app.state.context_security_policy = ctx_policy
        logger.info("Context security policy loaded: version=%s", ctx_policy.version)
    except Exception as exc:
        logger.error("FATAL: context security policy load failed: %s", exc)
        raise

    # 10C.1: load the content-safety policy when enabled — missing/invalid/
    # unreadable policy aborts startup. Disabled by default; no request can
    # replace the injected immutable policy object. Uses the freshly built
    # settings, never the import-time snapshot, so env changes in tests and
    # recreations are honored per startup.
    if boot_settings.content_safety_enabled:
        try:
            from app.services.safety_policy import load_safety_policy

            safety_policy = load_safety_policy(boot_settings.content_safety_policy_path)
            app.state.content_safety_policy = safety_policy
            logger.info(
                "Content safety policy loaded: version=%s, rules=%d",
                safety_policy.version, len(safety_policy.rules),
            )
        except Exception as exc:
            logger.error("FATAL: content safety policy load failed: %s", exc)
            raise

    yield


app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 10B.3: bounded envelope middleware (B-10).
app.add_middleware(BoundedReceiveMiddleware, max_bytes=settings.ingestion_request_max_bytes)

# The envelope middleware is the authoritative body bound; the multipart
# parser's per-part default (1 MiB) must not undercut the configured envelope.
# Starlette 1.3 reads max_part_size at the Request.form() call site, so wrap it.
import functools  # noqa: E402

import starlette.requests  # noqa: E402

_original_request_form = starlette.requests.Request.form


@functools.wraps(_original_request_form)
async def _bounded_request_form(self, **kwargs):
    kwargs.setdefault("max_part_size", settings.ingestion_request_max_bytes)
    return await _original_request_form(self, **kwargs)


starlette.requests.Request.form = _bounded_request_form

app.include_router(routes_documents.router)
app.include_router(routes_query.router)
app.include_router(routes_graph.router)
app.include_router(routes_security.router)


@app.get("/")
async def root():
    logger.info("Health check")
    return {"status": "ok", "app": settings.app_name}
