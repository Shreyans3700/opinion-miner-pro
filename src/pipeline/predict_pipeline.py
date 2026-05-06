"""Prediction pipeline entrypoint for single-sentence sentiment inference."""

from __future__ import annotations

import os
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    )

from src.components.prediction import Prediction
from src.config.data_preprocessing_config import DataPreprocessingConfig
from src.config.predict_config import PredictConfig
from src.utils.exception import CustomException


class PredictPipeline:
    """Inference pipeline that returns model observations for input text."""

    def __init__(
        self,
        predict_config: PredictConfig | None = None,
        preprocessing_config: DataPreprocessingConfig | None = None,
    ) -> None:
        self.predict_config = predict_config or PredictConfig()
        self.preprocessing_config = preprocessing_config or DataPreprocessingConfig()
        self._predictor = Prediction(
            predict_config=self.predict_config,
            preprocessing_config=self.preprocessing_config,
        )

    def run(self, sentence: str) -> dict[str, Any]:
        """Execute single-text inference and return observations."""
        try:
            return self._predictor.run(sentence)
        except Exception as error:
            raise CustomException(error, sys) from error


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError('Usage: python src/pipeline/predict_pipeline.py "your text"')
    input_text = " ".join(sys.argv[1:])
    output = PredictPipeline().run(input_text)
    for key, value in output.items():
        print(f"{key}={value}")
