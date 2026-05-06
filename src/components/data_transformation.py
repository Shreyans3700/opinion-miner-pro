"""Class-based data transformation component for sentiment model training."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from gridfs import GridFSBucket
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.config.data_transformation_config import DataTransformationConfig
from src.config.mongodb_data_exchange_config import MongoDBDataExchangeConfig
from src.utils.exception import CustomException
from src.utils.logger import logging
from src.utils.mongo_uri import normalize_mongo_uri


class DataTransformation:
    """Load cleaned parquet from MongoDB and build model-ready features."""

    def __init__(
        self,
        transformation_config: DataTransformationConfig,
        mongo_config: MongoDBDataExchangeConfig,
    ) -> None:
        self.transformation_config = transformation_config
        self.mongo_config = mongo_config

    def run(self) -> dict[str, str]:
        """Execute MongoDB download -> transform -> save artifacts workflow."""
        try:
            parquet_path = self.download_cleaned_parquet_from_mongodb()
            df = self.load_cleaned_dataframe(parquet_path)
            self.validate_schema(df)

            transformed = self.transform_dataframe(df)
            artifact_paths = self.save_transformed_artifacts(*transformed)

            artifact_paths["input_parquet_path"] = parquet_path
            logging.info("Data transformation completed successfully.")
            return artifact_paths
        except Exception as error:
            raise CustomException(error, sys) from error

    def download_cleaned_parquet_from_mongodb(self) -> str:
        """Download cleaned parquet file from MongoDB GridFS."""
        try:
            local_path = Path(self.transformation_config.local_input_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)

            bucket = self._get_bucket()
            with local_path.open("wb") as file_obj:
                bucket.download_to_stream_by_name(
                    self.mongo_config.cleaned_parquet_gridfs_filename,
                    file_obj,
                )

            logging.info(
                "Downloaded cleaned parquet '%s' from MongoDB to '%s'.",
                self.mongo_config.cleaned_parquet_gridfs_filename,
                local_path,
            )
            return str(local_path)
        except Exception as error:
            raise CustomException(error, sys) from error

    def load_cleaned_dataframe(self, parquet_path: str) -> pd.DataFrame:
        """Load cleaned parquet dataframe from local path."""
        try:
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
    ) -> tuple[Any, Any, np.ndarray, np.ndarray, TfidfVectorizer]:
        """Split data and build TF-IDF train/test matrices."""
        try:
            text_series = df[self.transformation_config.text_column].fillna("").astype(str)
            label_series = df[self.transformation_config.label_column].astype(str)

            stratify_target = label_series if self.transformation_config.stratify_split else None

            X_train_text, X_test_text, y_train, y_test = train_test_split(
                text_series,
                label_series,
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

            return X_train, X_test, y_train.to_numpy(), y_test.to_numpy(), vectorizer
        except Exception as error:
            raise CustomException(error, sys) from error

    def save_transformed_artifacts(
        self,
        X_train: Any,
        X_test: Any,
        y_train: np.ndarray,
        y_test: np.ndarray,
        vectorizer: TfidfVectorizer,
    ) -> dict[str, str]:
        """Save TF-IDF matrices, labels, and vectorizer artifacts."""
        try:
            artifacts_dir = Path(self.transformation_config.artifacts_dir)
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            save_npz(self.transformation_config.x_train_path, X_train)
            save_npz(self.transformation_config.x_test_path, X_test)
            np.save(self.transformation_config.y_train_path, y_train)
            np.save(self.transformation_config.y_test_path, y_test)
            joblib.dump(vectorizer, self.transformation_config.vectorizer_path)

            logging.info("Saved transformed artifacts under '%s'.", artifacts_dir)
            return {
                "x_train_path": self.transformation_config.x_train_path,
                "x_test_path": self.transformation_config.x_test_path,
                "y_train_path": self.transformation_config.y_train_path,
                "y_test_path": self.transformation_config.y_test_path,
                "vectorizer_path": self.transformation_config.vectorizer_path,
            }
        except Exception as error:
            raise CustomException(error, sys) from error

    def _get_bucket(self) -> GridFSBucket:
        """Create MongoDB GridFS bucket from configured DB details."""
        try:
            client = MongoClient(normalize_mongo_uri(self.mongo_config.mongo_uri))
            db = client[self.mongo_config.database_name]
            return GridFSBucket(db, bucket_name=self.mongo_config.bucket_name)
        except PyMongoError as error:
            raise CustomException(error, sys) from error
