"""MongoDB GridFS component for raw CSV download and cleaned parquet upload."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

from gridfs import GridFSBucket
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from src.components.data_preprocessing import DataPreprocessing
from src.config.data_preprocessing_config import DataPreprocessingConfig
from src.config.mongodb_storage_config import MongoDBStorageConfig
from src.utils.exception import CustomException
from src.utils.logger import logging


class MongoDBStorage:
    """Manage file storage roundtrips between local workspace and MongoDB GridFS."""

    def __init__(self, config: MongoDBStorageConfig) -> None:
        self.config = config

    def download_raw_csv(self) -> str:
        """Download raw CSV from GridFS to local path and return local path."""
        try:
            bucket = self._get_bucket()
            local_path = Path(self.config.raw_csv_local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)

            with local_path.open("wb") as file_obj:
                bucket.download_to_stream_by_name(
                    self.config.raw_csv_gridfs_filename, file_obj
                )

            logging.info(
                "Downloaded GridFS file '%s' to '%s'.",
                self.config.raw_csv_gridfs_filename,
                local_path,
            )
            return str(local_path)
        except Exception as error:
            raise CustomException(error, sys) from error

    def upload_cleaned_parquet(self, parquet_path: str) -> str:
        """Upload cleaned parquet file to GridFS and return inserted file id."""
        try:
            bucket = self._get_bucket()
            source_path = Path(parquet_path)
            if not source_path.exists():
                raise FileNotFoundError(f"Parquet file not found: {source_path}")

            if self.config.replace_existing_cleaned_file:
                for grid_out in bucket.find(
                    {"filename": self.config.cleaned_parquet_gridfs_filename}
                ):
                    bucket.delete(grid_out._id)

            with source_path.open("rb") as file_obj:
                file_id = bucket.upload_from_stream(
                    self.config.cleaned_parquet_gridfs_filename,
                    file_obj,
                    metadata={
                        "source": "data_preprocessing_component",
                        "original_local_path": str(source_path),
                    },
                )

            logging.info(
                "Uploaded cleaned parquet '%s' to GridFS as '%s' (file_id=%s).",
                source_path,
                self.config.cleaned_parquet_gridfs_filename,
                file_id,
            )
            return str(file_id)
        except Exception as error:
            raise CustomException(error, sys) from error

    def run_preprocess_roundtrip(
        self,
        preprocessing_config: DataPreprocessingConfig | None = None,
    ) -> Tuple[str, str]:
        """Download raw CSV from MongoDB, preprocess locally, upload parquet back."""
        try:
            raw_csv_local_path = self.download_raw_csv()

            pre_cfg = preprocessing_config or DataPreprocessingConfig()
            pre_cfg.raw_data_path = raw_csv_local_path

            cleaned_parquet_path = DataPreprocessing(pre_cfg).run()
            uploaded_file_id = self.upload_cleaned_parquet(cleaned_parquet_path)

            return cleaned_parquet_path, uploaded_file_id
        except Exception as error:
            raise CustomException(error, sys) from error

    def _get_bucket(self) -> GridFSBucket:
        """Create and return a GridFS bucket handle."""
        try:
            client = MongoClient(self.config.mongo_uri)
            db = client[self.config.database_name]
            return GridFSBucket(db, bucket_name=self.config.bucket_name)
        except PyMongoError as error:
            raise CustomException(error, sys) from error


# Backward-compatible alias.
MongoDBDataExchange = MongoDBStorage
