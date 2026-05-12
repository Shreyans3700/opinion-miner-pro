"""Schemas for training endpoint responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class TrainResponse(BaseModel):
    """Training endpoint response model."""

    status: str
    outputs: dict[str, Any]
