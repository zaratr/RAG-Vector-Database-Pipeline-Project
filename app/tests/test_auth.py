"""Phase 10B.2 — operator authentication tests."""
from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.api.auth import require_operator
from app.config import Settings


def test_operator_api_disabled_returns_404():
    settings = Settings(operator_api_enabled=False)
    with pytest.raises(HTTPException) as exc:
        require_operator(credentials=None, settings=settings)
    assert exc.value.status_code == 404


def test_operator_api_enabled_missing_bearer_returns_401():
    settings = Settings(operator_api_enabled=True, operator_token="x" * 32)
    with pytest.raises(HTTPException) as exc:
        require_operator(credentials=None, settings=settings)
    assert exc.value.status_code == 401


def test_operator_api_enabled_invalid_bearer_returns_401():
    settings = Settings(operator_api_enabled=True, operator_token="x" * 32)
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="wrong-token")
    with pytest.raises(HTTPException) as exc:
        require_operator(credentials=creds, settings=settings)
    assert exc.value.status_code == 401


def test_operator_api_enabled_valid_bearer_returns_true():
    token = "x" * 32
    settings = Settings(operator_api_enabled=True, operator_token=token)
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    result = require_operator(credentials=creds, settings=settings)
    assert result is True
