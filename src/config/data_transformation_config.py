"""Configuration for the data transformation component."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataTransformationConfig:
    """Dataclass config for TF-IDF transformation stage."""

    local_input_dir: str = "data/interim"
    local_input_file_name: str = "reviews_cleaned.parquet"

    text_column: str = "text_clean"
    label_column: str = "sentiment"

    test_size: float = 0.2
    random_state: int = 42
    stratify_split: bool = True

    tfidf_max_features: int = 50000
    tfidf_ngram_min: int = 1
    tfidf_ngram_max: int = 2
    tfidf_min_df: int = 2
    tfidf_max_df: float = 0.95

    artifacts_dir: str = "artifacts"
    vectorizer_file_name: str = "vectorizer.pkl"
    label_encoder_file_name: str = "label_encoder.pkl"
    x_train_file_name: str = "X_train.npz"
    x_test_file_name: str = "X_test.npz"
    y_train_file_name: str = "y_train.npy"
    y_test_file_name: str = "y_test.npy"

    @property
    def local_input_path(self) -> str:
        """Local parquet destination used after MongoDB download."""
        return str(Path(self.local_input_dir) / self.local_input_file_name)

    @property
    def vectorizer_path(self) -> str:
        return str(Path(self.artifacts_dir) / self.vectorizer_file_name)

    @property
    def label_encoder_path(self) -> str:
        return str(Path(self.artifacts_dir) / self.label_encoder_file_name)

    @property
    def x_train_path(self) -> str:
        return str(Path(self.artifacts_dir) / self.x_train_file_name)

    @property
    def x_test_path(self) -> str:
        return str(Path(self.artifacts_dir) / self.x_test_file_name)

    @property
    def y_train_path(self) -> str:
        return str(Path(self.artifacts_dir) / self.y_train_file_name)

    @property
    def y_test_path(self) -> str:
        return str(Path(self.artifacts_dir) / self.y_test_file_name)
