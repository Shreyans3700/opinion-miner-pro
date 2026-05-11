"""Service layer around the sentiment prediction pipeline."""

from __future__ import annotations

from typing import Any

from src.pipeline.predict_pipeline import PredictPipeline


class PredictionService:
    """Facade for single and batch prediction operations."""

    def __init__(self) -> None:
        self._pipeline = PredictPipeline()

    def predict_one(self, review: str) -> dict[str, Any]:
        """Run inference for a single review."""
        return self._pipeline.run(review)

    def predict_many(self, reviews: list[str]) -> list[dict[str, Any]]:
        """Run inference for multiple reviews."""
        return [self.predict_one(review) for review in reviews]

