"""Class-based model training component with tuning and reporting."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.config.model_training_config import ModelTrainingConfig
from src.utils.exception import CustomException
from src.utils.logger import logging


class ModelTraining:
    """Train multiple models with hyperparameter search and persist outputs."""

    def __init__(self, training_config: ModelTrainingConfig) -> None:
        self.training_config = training_config

    def run(
        self, transformed_artifacts: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Execute model training, evaluation, best-model save, and YAML report."""
        try:
            paths = self._resolve_input_paths(transformed_artifacts)
            X_train, X_test, y_train, y_test = self._load_transformed_artifacts(paths)

            model_search_space = self._get_model_search_space()
            model_scores, best_model_name, best_model, best_score = (
                self._train_and_select_best(
                    X_train, X_test, y_train, y_test, model_search_space
                )
            )

            best_model_path = self._save_best_model(best_model)
            report_path = self._save_scores_report(
                model_scores=model_scores,
                best_model_name=best_model_name,
                best_model=best_model,
                best_score=best_score,
                best_model_path=best_model_path,
            )

            logging.info(
                "Model training completed successfully with best model: %s",
                best_model_name,
            )
            return {
                "best_model_name": best_model_name,
                "best_model_score": best_score,
                "best_model_path": best_model_path,
                "model_scores_report_path": report_path,
                "all_model_scores": model_scores,
            }
        except Exception as error:
            raise CustomException(error, sys) from error

    def _resolve_input_paths(
        self, transformed_artifacts: dict[str, str] | None
    ) -> dict[str, str]:
        """Resolve transformed artifact paths from inputs or defaults."""
        if transformed_artifacts is None:
            return {
                "x_train_path": self.training_config.x_train_path,
                "x_test_path": self.training_config.x_test_path,
                "y_train_path": self.training_config.y_train_path,
                "y_test_path": self.training_config.y_test_path,
            }
        return {
            "x_train_path": transformed_artifacts.get(
                "x_train_path", self.training_config.x_train_path
            ),
            "x_test_path": transformed_artifacts.get(
                "x_test_path", self.training_config.x_test_path
            ),
            "y_train_path": transformed_artifacts.get(
                "y_train_path", self.training_config.y_train_path
            ),
            "y_test_path": transformed_artifacts.get(
                "y_test_path", self.training_config.y_test_path
            ),
        }

    def _load_transformed_artifacts(
        self, paths: dict[str, str]
    ) -> tuple[Any, Any, np.ndarray, np.ndarray]:
        """Load sparse features and labels produced by the transformation stage."""
        required_paths = [
            paths["x_train_path"],
            paths["x_test_path"],
            paths["y_train_path"],
            paths["y_test_path"],
        ]
        for path in required_paths:
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"Required transformed artifact not found: {path}"
                )

        X_train = load_npz(paths["x_train_path"])
        X_test = load_npz(paths["x_test_path"])
        # allow_pickle=True keeps compatibility with older saved label arrays
        y_train = np.asarray(np.load(paths["y_train_path"], allow_pickle=True)).ravel()
        y_test = np.asarray(np.load(paths["y_test_path"], allow_pickle=True)).ravel()
        return X_train, X_test, y_train, y_test

    def _get_model_search_space(
        self,
    ) -> dict[str, tuple[BaseEstimator, dict[str, list[Any]]]]:
        """Define candidate models and hyperparameter grids."""
        random_state = self.training_config.random_state
        return {
            "LogisticRegression": (
                LogisticRegression(
                    max_iter=2000, solver="liblinear", random_state=random_state
                ),
                {"C": [0.1, 1.0, 5.0], "class_weight": [None, "balanced"]},
            ),
            "LinearSVC": (
                LinearSVC(random_state=random_state),
                {"C": [0.1, 1.0, 5.0]},
            ),
            "MultinomialNB": (
                MultinomialNB(),
                {"alpha": [0.1, 0.5, 1.0]},
            ),
            "SGDClassifier": (
                SGDClassifier(random_state=random_state),
                {
                    "loss": ["hinge", "log_loss"],
                    "alpha": [1e-4, 1e-3, 1e-2],
                    "penalty": ["l2", "elasticnet"],
                },
            ),
        }

    def _train_and_select_best(
        self,
        X_train: Any,
        X_test: Any,
        y_train: np.ndarray,
        y_test: np.ndarray,
        model_search_space: dict[str, tuple[BaseEstimator, dict[str, list[Any]]]],
    ) -> tuple[dict[str, dict[str, Any]], str, BaseEstimator, float]:
        """Tune all models and select best one using the configured primary metric."""
        model_scores: dict[str, dict[str, Any]] = {}
        best_model_name = ""
        best_model: BaseEstimator | None = None
        best_score = float("-inf")
        target_metric_key = self._metric_to_score_key(
            self.training_config.primary_metric
        )

        for model_name, (estimator, param_grid) in model_search_space.items():
            logging.info("Tuning model: %s", model_name)
            search = self._fit_grid_search_with_fallback(
                model_name=model_name,
                estimator=estimator,
                param_grid=param_grid,
                X_train=X_train,
                y_train=y_train,
            )

            tuned_model = search.best_estimator_
            y_pred = tuned_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average="weighted", zero_division=0
            )

            score_row = {
                "cv_best_score": float(search.best_score_),
                "test_accuracy": float(accuracy),
                "test_precision_weighted": float(precision),
                "test_recall_weighted": float(recall),
                "test_f1_weighted": float(f1),
                "best_params": self._normalize_for_yaml(search.best_params_),
            }
            model_scores[model_name] = score_row

            metric_value = float(score_row[target_metric_key])
            if metric_value > best_score:
                best_score = metric_value
                best_model_name = model_name
                best_model = tuned_model

        if best_model is None:
            raise ValueError("No model could be trained successfully.")

        return model_scores, best_model_name, best_model, best_score

    def _fit_grid_search_with_fallback(
        self,
        model_name: str,
        estimator: BaseEstimator,
        param_grid: dict[str, list[Any]],
        X_train: Any,
        y_train: np.ndarray,
    ) -> GridSearchCV:
        """Fit GridSearchCV and retry in serial mode on Windows resource failures."""
        search = self._build_grid_search(
            estimator=estimator,
            param_grid=param_grid,
            n_jobs=self.training_config.n_jobs,
            pre_dispatch=self.training_config.pre_dispatch,
        )

        try:
            search.fit(X_train, y_train)
            return search
        except Exception as error:
            if self._should_fallback_to_serial(error, self.training_config.n_jobs):
                logging.warning(
                    "Resource-heavy grid-search failed for %s; retrying with serial mode (n_jobs=1).",
                    model_name,
                )
                serial_search = self._build_grid_search(
                    estimator=estimator,
                    param_grid=param_grid,
                    n_jobs=1,
                    pre_dispatch=1,
                )
                serial_search.fit(X_train, y_train)
                return serial_search
            raise

    def _build_grid_search(
        self,
        estimator: BaseEstimator,
        param_grid: dict[str, list[Any]],
        n_jobs: int,
        pre_dispatch: str | int,
    ) -> GridSearchCV:
        """Create a configured GridSearchCV instance."""
        return GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=self.training_config.primary_metric,
            cv=self.training_config.cv_folds,
            n_jobs=n_jobs,
            pre_dispatch=pre_dispatch,
            verbose=self.training_config.grid_search_verbose,
            refit=True,
        )

    def _should_fallback_to_serial(
        self, error: Exception, configured_n_jobs: int
    ) -> bool:
        """Detect parallel execution failures and decide whether serial retry is safe."""
        if configured_n_jobs == 1:
            return False
        if isinstance(error, PermissionError):
            return True
        if isinstance(error, MemoryError):
            return True
        if isinstance(error, OSError):
            winerror = getattr(error, "winerror", None)
            message = str(error).lower()
            if winerror == 1450:
                return True
            if "insufficient system resources" in message:
                return True
        return False

    def _save_best_model(self, best_model: BaseEstimator) -> str:
        """Persist best model artifact into artifacts directory."""
        model_path = Path(self.training_config.trained_model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, model_path)
        logging.info("Saved best model at '%s'.", model_path)
        return str(model_path)

    def _save_scores_report(
        self,
        model_scores: dict[str, dict[str, Any]],
        best_model_name: str,
        best_model: BaseEstimator,
        best_score: float,
        best_model_path: str,
    ) -> str:
        """Save all model scores and best-model summary into a YAML report."""
        report_path = Path(self.training_config.model_scores_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report_payload = {
            "primary_metric": self.training_config.primary_metric,
            "best_model": {
                "name": best_model_name,
                "score": float(best_score),
                "artifact_path": best_model_path,
                "estimator": type(best_model).__name__,
                "best_params": model_scores[best_model_name]["best_params"],
            },
            "models": model_scores,
        }

        yaml_content = self._to_yaml(report_payload)
        report_path.write_text(yaml_content, encoding="utf-8")
        logging.info("Saved model score report at '%s'.", report_path)
        return str(report_path)

    def _metric_to_score_key(self, metric_name: str) -> str:
        """Map sklearn scoring names to score table fields."""
        metric_map = {
            "accuracy": "test_accuracy",
            "precision_weighted": "test_precision_weighted",
            "recall_weighted": "test_recall_weighted",
            "f1_weighted": "test_f1_weighted",
        }
        if metric_name not in metric_map:
            raise ValueError(
                f"Unsupported primary_metric '{metric_name}'. Supported: {list(metric_map.keys())}"
            )
        return metric_map[metric_name]

    def _normalize_for_yaml(self, value: Any) -> Any:
        """Convert numpy scalars/containers into YAML-friendly Python values."""
        if isinstance(value, dict):
            return {str(k): self._normalize_for_yaml(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._normalize_for_yaml(item) for item in value]
        if isinstance(value, tuple):
            return [self._normalize_for_yaml(item) for item in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return str(value)
        return value

    def _to_yaml(self, payload: dict[str, Any]) -> str:
        """Serialize basic nested dict/list payload into YAML text."""
        lines = self._yaml_lines(payload, indent=0)
        return "\n".join(lines) + "\n"

    def _yaml_lines(self, value: Any, indent: int) -> list[str]:
        spaces = " " * indent

        if isinstance(value, dict):
            lines: list[str] = []
            for key, item in value.items():
                yaml_key = str(key)
                if isinstance(item, (dict, list)):
                    lines.append(f"{spaces}{yaml_key}:")
                    lines.extend(self._yaml_lines(item, indent + 2))
                else:
                    lines.append(f"{spaces}{yaml_key}: {self._yaml_scalar(item)}")
            return lines

        if isinstance(value, list):
            lines = []
            for item in value:
                if isinstance(item, (dict, list)):
                    lines.append(f"{spaces}-")
                    lines.extend(self._yaml_lines(item, indent + 2))
                else:
                    lines.append(f"{spaces}- {self._yaml_scalar(item)}")
            return lines

        return [f"{spaces}{self._yaml_scalar(value)}"]

    def _yaml_scalar(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "null"
        if isinstance(value, float):
            return f"{value:.8f}"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, str):
            if value == "" or value != value.strip():
                return json.dumps(value)
            reserved = [
                ":",
                "#",
                "{",
                "}",
                "[",
                "]",
                ",",
                "&",
                "*",
                "?",
                "|",
                ">",
                "%",
                "@",
                "`",
            ]
            if any(token in value for token in reserved):
                return json.dumps(value)
            return value
        return json.dumps(str(value))
