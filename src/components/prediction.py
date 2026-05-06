"""Class-based prediction component for single-sentence sentiment inference."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.components.data_preprocessing import DataPreprocessing
from src.config.data_preprocessing_config import DataPreprocessingConfig
from src.config.predict_config import PredictConfig
from src.utils.exception import CustomException
from src.utils.logger import logging


class Prediction:
    """Run text preprocessing, embedding, and model inference for one sentence."""

    def __init__(
        self,
        predict_config: PredictConfig,
        preprocessing_config: DataPreprocessingConfig | None = None,
    ) -> None:
        self.predict_config = predict_config
        self.preprocessing_config = preprocessing_config or DataPreprocessingConfig()
        self._preprocessor = DataPreprocessing(self.preprocessing_config)
        self._vectorizer: Any | None = None
        self._model: Any | None = None
        self._label_encoder: Any | None = None

    def run(self, sentence: str) -> dict[str, Any]:
        """Predict sentiment from input sentence and return observations."""
        try:
            if not isinstance(sentence, str):
                raise ValueError("Input sentence must be a string.")
            if sentence.strip() == "":
                raise ValueError("Input sentence cannot be empty.")

            self._ensure_artifacts_loaded()

            cleaned_text = self._preprocessor.preprocess_text(sentence)
            if cleaned_text.strip() == "":
                raise ValueError(
                    "Sentence became empty after preprocessing. Provide richer text."
                )
            tokens = cleaned_text.split()

            embedding = self._vectorizer.transform([cleaned_text])
            predicted = self._model.predict(embedding)[0]
            predicted_label = self._decode_label(predicted)

            observations = {
                "input_text": sentence,
                "cleaned_text": cleaned_text,
                "tokens": tokens,
                "token_count": len(tokens),
                "embedding_shape": [int(embedding.shape[0]), int(embedding.shape[1])],
                "predicted_label": predicted_label,
                "predicted_raw": self._to_python_scalar(predicted),
            }
            observations.update(self._probability_or_score_observations(embedding))
            return observations
        except Exception as error:
            raise CustomException(error, sys) from error

    def _ensure_artifacts_loaded(self) -> None:
        """Load and cache model/vectorizer/label-encoder artifacts."""
        if self._model is not None and self._vectorizer is not None:
            return

        required = [
            self.predict_config.model_path,
            self.predict_config.vectorizer_path,
        ]
        for artifact_path in required:
            if not Path(artifact_path).exists():
                raise FileNotFoundError(
                    f"Required prediction artifact not found: {artifact_path}"
                )

        self._model = joblib.load(self.predict_config.model_path)
        self._vectorizer = joblib.load(self.predict_config.vectorizer_path)

        if Path(self.predict_config.label_encoder_path).exists():
            self._label_encoder = joblib.load(self.predict_config.label_encoder_path)
        else:
            self._label_encoder = None
            logging.warning(
                "Label encoder not found at '%s'; returning raw predicted labels.",
                self.predict_config.label_encoder_path,
            )

    def _decode_label(self, predicted_value: Any) -> str:
        """Decode model output to original class label if encoder is available."""
        if self._label_encoder is None:
            return str(self._to_python_scalar(predicted_value))
        decoded = self._label_encoder.inverse_transform([predicted_value])[0]
        return str(decoded)

    def _probability_or_score_observations(self, embedding: Any) -> dict[str, Any]:
        """Return confidence observations from predict_proba or decision_function."""
        if hasattr(self._model, "predict_proba"):
            probabilities = self._model.predict_proba(embedding)[0]
            max_idx = int(np.argmax(probabilities))
            class_map = self._class_probability_map(probabilities)
            return {
                "confidence": float(probabilities[max_idx]),
                "class_probabilities": class_map,
            }

        if hasattr(self._model, "decision_function"):
            decision = self._model.decision_function(embedding)
            if np.ndim(decision) == 1:
                margin = float(decision[0])
            else:
                margin = float(np.max(decision[0]))
            return {
                "decision_margin": margin,
            }

        return {}

    def _class_probability_map(self, probabilities: np.ndarray) -> dict[str, float]:
        """Map probability vector to human-readable class names."""
        if self._label_encoder is not None and hasattr(self._label_encoder, "classes_"):
            class_names = [str(label) for label in self._label_encoder.classes_]
            if len(class_names) == len(probabilities):
                return {
                    class_names[idx]: float(probabilities[idx])
                    for idx in range(len(probabilities))
                }

        return {
            str(idx): float(probabilities[idx]) for idx in range(len(probabilities))
        }

    @staticmethod
    def _to_python_scalar(value: Any) -> Any:
        """Convert numpy scalar values to native Python types."""
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return value
        return value
