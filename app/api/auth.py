"""Operator authentication dependency (Task 10B.2).

Implements the auth precedence matrix: evaluate operator-API enabled/route
status first, then bearer syntax/validity when supplied, then source policy.
Authentication logs contain only event code and request ID—never token values.
"""
from __future__ import annotations

import hmac

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import Settings, get_settings

_bearer_scheme = HTTPBearer(auto_error=False)


def require_operator(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> bool:
    """Return True if the caller is an authenticated operator.

    Raises HTTPException for invalid/missing credentials. When the operator API
    is disabled, any operator-route access returns 404 before credential
    disclosure.
    """
    if not settings.operator_api_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Operator API disabled")

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = settings.operator_token.get_secret_value()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Operator token not configured",
        )

    if not hmac.compare_digest(credentials.credentials, token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True
