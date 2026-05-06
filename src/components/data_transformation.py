"""Class-based data transformation component for sentiment model training."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config.data_transformation_config import DataTransformationConfig
from src.utils.exception import CustomException
from src.utils.logger import logging


class DataTransformation:
    """Build model-ready features from cleaned parquet."""

    def __init__(self, transformation_config: DataTransformationConfig) -> None:
        self.transformation_config = transformation_config

    def run(self, parquet_path: str | None = None) -> dict[str, str]:
        """Execute load -> transform -> save artifacts workflow."""
        try:
            input_path = parquet_path or self.transformation_config.local_input_path
            df = self.load_cleaned_dataframe(input_path)
            self.validate_schema(df)

            transformed = self.transform_dataframe(df)
            artifact_paths = self.save_transformed_artifacts(*transformed)

            artifact_paths["input_parquet_path"] = input_path
            logging.info("Data transformation completed successfully.")
            return artifact_paths
        except Exception as error:
            raise CustomException(error, sys) from error

    def load_cleaned_dataframe(self, parquet_path: str) -> pd.DataFrame:
        """Load cleaned parquet dataframe from local path."""
        try:
            if not Path(parquet_path).exists():
                raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
            df = pd.read_parquet(parquet_path)
            if df.empty:
                raise ValueError("Cleaned parquet is empty.")
            return df
        except Exception as error:
            raise CustomException(error, sys) from error

    def validate_schema(self, df: pd.DataFrame) -> None:
        """Validate required transformation columns."""
        required_columns = {
            self.transformation_config.text_column,
            self.transformation_config.label_column,
        }
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise CustomException(
                ValueError(
                    f"Missing required columns: {sorted(missing_columns)}. "
                    f"Available columns: {list(df.columns)}"
                ),
                sys,
            )

    def transform_dataframe(
        self, df: pd.DataFrame
    ) -> tuple[Any, Any, np.ndarray, np.ndarray, TfidfVectorizer, LabelEncoder]:
        """Split data and build TF-IDF train/test matrices with encoded labels."""
        try:
            text_series = (
                df[self.transformation_config.text_column].fillna("").astype(str)
            )
            label_series = (
                df[self.transformation_config.label_column].fillna("").astype(str)
            )

            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(label_series)

            stratify_target = (
                encoded_labels if self.transformation_config.stratify_split else None
            )

            X_train_text, X_test_text, y_train, y_test = train_test_split(
                text_series,
                encoded_labels,
                test_size=self.transformation_config.test_size,
                random_state=self.transformation_config.random_state,
                stratify=stratify_target,
            )

            vectorizer = TfidfVectorizer(
                max_features=self.transformation_config.tfidf_max_features,
                ngram_range=(
                    self.transformation_config.tfidf_ngram_min,
                    self.transformation_config.tfidf_ngram_max,
                ),
                min_df=self.transformation_config.tfidf_min_df,
                max_df=self.transformation_config.tfidf_max_df,
            )

            X_train = vectorizer.fit_transform(X_train_text)
            X_test = vectorizer.transform(X_test_text)

            return X_train, X_test, y_train, y_test, vectorizer, label_encoder
        except Exception as error:
            raise CustomException(error, sys) from error

    def save_transformed_artifacts(
        self,
        X_train: Any,
        X_test: Any,
        y_train: np.ndarray,
        y_test: np.ndarray,
        vectorizer: TfidfVectorizer,
        label_encoder: LabelEncoder,
    ) -> dict[str, str]:
        """Save TF-IDF matrices, labels, and reusable transformation artifacts."""
        try:
            artifacts_dir = Path(self.transformation_config.artifacts_dir)
            transformed_data_dir = Path(self.transformation_config.transformed_data_dir)
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            transformed_data_dir.mkdir(parents=True, exist_ok=True)

            save_npz(self.transformation_config.x_train_path, X_train)
            save_npz(self.transformation_config.x_test_path, X_test)
            np.save(self.transformation_config.y_train_path, y_train)
            np.save(self.transformation_config.y_test_path, y_test)
            joblib.dump(vectorizer, self.transformation_config.vectorizer_path)
            joblib.dump(label_encoder, self.transformation_config.label_encoder_path)

            logging.info(
                "Saved transformed data under '%s' and artifacts under '%s'.",
                transformed_data_dir,
                artifacts_dir,
            )
            return {
                "x_train_path": self.transformation_config.x_train_path,
                "x_test_path": self.transformation_config.x_test_path,
                "y_train_path": self.transformation_config.y_train_path,
                "y_test_path": self.transformation_config.y_test_path,
                "vectorizer_path": self.transformation_config.vectorizer_path,
                "label_encoder_path": self.transformation_config.label_encoder_path,
            }
        except Exception as error:
            raise CustomException(error, sys) from error
