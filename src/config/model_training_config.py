"""Configuration for the model training component."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelTrainingConfig:
    """Dataclass config for model training and evaluation."""

    transformed_artifacts_dir: str = "data/transformed"
    x_train_file_name: str = "X_train.npz"
    x_test_file_name: str = "X_test.npz"
    y_train_file_name: str = "y_train.npy"
    y_test_file_name: str = "y_test.npy"

    artifacts_dir: str = "artifacts"
    trained_model_file_name: str = "model.pkl"

    report_dir: str = "report"
    model_scores_file_name: str = "model_scores.yaml"

    primary_metric: str = "f1_weighted"
    cv_folds: int = 3
    random_state: int = 42
    n_jobs: int = 1
    pre_dispatch: str = "1*n_jobs"
    grid_search_verbose: int = 0

    @property
    def x_train_path(self) -> str:
        return str(Path(self.transformed_artifacts_dir) / self.x_train_file_name)

    @property
    def x_test_path(self) -> str:
        return str(Path(self.transformed_artifacts_dir) / self.x_test_file_name)

    @property
    def y_train_path(self) -> str:
        return str(Path(self.transformed_artifacts_dir) / self.y_train_file_name)

    @property
    def y_test_path(self) -> str:
        return str(Path(self.transformed_artifacts_dir) / self.y_test_file_name)

    @property
    def trained_model_path(self) -> str:
        return str(Path(self.artifacts_dir) / self.trained_model_file_name)

    @property
    def model_scores_path(self) -> str:
        return str(Path(self.report_dir) / self.model_scores_file_name)
