"""Request/response schemas for analysis endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnalyseRequest(BaseModel):
    """Single-review analysis request body."""

    review: str = Field(..., min_length=1, description="Single review text.")


class AnalyseResponse(BaseModel):
    """Single-review analysis response body."""

    input_text: str
    cleaned_text: str
    tokens: list[str]
    token_count: int
    embedding_shape: list[int]
    predicted_label: str
    overall_sentiment: str | None = None
    predicted_raw: Any
    positive_score: float | None = None
    negative_score: float | None = None
    sentiment_breakdown: dict[str, float] | None = None
    aspect_sentiments: dict[str, str] | None = None
    aspect_sentiment_scores: dict[str, dict[str, float]] | None = None
    main_feature_points: list[dict[str, Any]] | None = None
    confidence: float | None = None
    decision_margin: float | None = None
    class_probabilities: dict[str, float] | None = None
