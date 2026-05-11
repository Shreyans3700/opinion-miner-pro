"""API-key authentication dependency."""

from __future__ import annotations

from os import getenv

from fastapi import Header, HTTPException, status

from src.utils.env_loader import load_project_env

load_project_env()


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Validate request API key from the x-api-key header."""
    expected_api_key = getenv("API_KEY", "").strip()
    if not expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_KEY is not configured on the server.",
        )

    if not x_api_key or x_api_key.strip() != expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )

