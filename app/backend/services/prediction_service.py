"""Service layer around the sentiment prediction pipeline."""

from __future__ import annotations

from typing import Any

from src.pipeline.predict_pipeline import PredictPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionService:
    """Facade for single and batch prediction operations."""

    def __init__(self) -> None:
        self._pipeline = PredictPipeline()

    def predict_one(self, review: str) -> dict[str, Any]:
        """Run inference for a single review."""
        logger.info("Running prediction for review: %s...", review[:100])
        result = self._pipeline.run(review)
        logger.info("Prediction complete: %s", result.get("overall_sentiment", result.get("predicted_label", "")))
        return result

    def predict_many(self, reviews: list[str]) -> list[dict[str, Any]]:
        """Run inference for multiple reviews."""
        return [self.predict_one(review) for review in reviews]

