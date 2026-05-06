"""Configuration for the prediction pipeline/component."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PredictConfig:
    """Dataclass config for loading inference artifacts."""

    artifacts_dir: str = "artifacts"
    model_file_name: str = "model.pkl"
    vectorizer_file_name: str = "vectorizer.pkl"
    label_encoder_file_name: str = "label_encoder.pkl"

    @property
    def model_path(self) -> str:
        return str(Path(self.artifacts_dir) / self.model_file_name)

    @property
    def vectorizer_path(self) -> str:
        return str(Path(self.artifacts_dir) / self.vectorizer_file_name)

    @property
    def label_encoder_path(self) -> str:
        return str(Path(self.artifacts_dir) / self.label_encoder_file_name)
