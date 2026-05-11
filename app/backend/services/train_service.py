"""Service layer around the model training pipeline."""

from __future__ import annotations

from typing import Any

from src.pipeline.train_pipeline import TrainPipeline


class TrainService:
    """Facade for running the train pipeline from the API layer."""

    def __init__(self) -> None:
        self._pipeline = TrainPipeline()

    def run_training(self) -> dict[str, Any]:
        """Execute train pipeline and return stage outputs."""
        return self._pipeline.run()

