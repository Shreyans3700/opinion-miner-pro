"""Utility to upload a local CSV file to MongoDB GridFS."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    )

from gridfs import GridFSBucket
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from src.config.mongodb_storage_config import MongoDBStorageConfig
from src.utils.env_loader import load_project_env
from src.utils.exception import CustomException
from src.utils.logger import logging
from src.utils.mongo_uri import normalize_mongo_uri

load_project_env()


class MongoDBStorageUploader:
    """Uploads CSV files into MongoDB GridFS."""

    def __init__(self, config: MongoDBStorageConfig | None = None) -> None:
        self.config = config or MongoDBStorageConfig()

    def upload_csv(
        self,
        csv_path: str,
        gridfs_filename: str | None = None,
        replace_existing: bool = True,
    ) -> str:
        """Upload the CSV at csv_path to GridFS and return inserted file id."""
        try:
            source_path = Path(csv_path)
            if not source_path.exists():
                raise FileNotFoundError(f"CSV file not found: {source_path}")
            if source_path.suffix.lower() != ".csv":
                raise ValueError(f"Expected a .csv file, got: {source_path.name}")

            upload_name = gridfs_filename or self.config.raw_csv_gridfs_filename
            bucket = self._get_bucket()

            if replace_existing:
                for grid_out in bucket.find({"filename": upload_name}):
                    bucket.delete(grid_out._id)

            with source_path.open("rb") as file_obj:
                file_id = bucket.upload_from_stream(
                    upload_name,
                    file_obj,
                    metadata={
                        "contentType": "text/csv",
                        "source": "mongodb_storage_uploader",
                        "original_local_path": str(source_path),
                    },
                )

            logging.info(
                "Uploaded CSV '%s' to GridFS filename '%s' (file_id=%s).",
                source_path,
                upload_name,
                file_id,
            )
            return str(file_id)
        except CustomException:
            raise
        except Exception as error:
            raise CustomException(error, sys) from error

    def _get_bucket(self) -> GridFSBucket:
        """Return GridFS bucket configured for this project."""
        try:
            client = MongoClient(normalize_mongo_uri(self.config.mongo_uri))
            db = client[self.config.database_name]
            return GridFSBucket(db, bucket_name=self.config.bucket_name)
        except CustomException:
            raise
        except PyMongoError as error:
            raise CustomException(error, sys) from error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a CSV file to MongoDB GridFS.")
    parser.add_argument("--csv-path", required=True, help="Path to local CSV file.")
    parser.add_argument(
        "--mongo-uri",
        default=None,
        help="MongoDB URI override. If omitted, reads MONGODB_URI from environment/.env.",
    )
    parser.add_argument(
        "--gridfs-filename",
        default=None,
        help="Target filename in GridFS. Defaults to config raw_csv_gridfs_filename.",
    )
    parser.add_argument(
        "--no-replace-existing",
        action="store_true",
        help="Keep existing file revisions with same GridFS filename.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for uploading CSV file to MongoDB GridFS."""
    args = _parse_args()
    config = MongoDBStorageConfig()
    if args.mongo_uri:
        config.mongo_uri = normalize_mongo_uri(args.mongo_uri)

    uploader = MongoDBStorageUploader(config=config)
    uploaded_file_id = uploader.upload_csv(
        csv_path=args.csv_path,
        gridfs_filename=args.gridfs_filename,
        replace_existing=not args.no_replace_existing,
    )
    print(f"uploaded_file_id={uploaded_file_id}")


# Backward-compatible alias.
MongoDBCSVUploader = MongoDBStorageUploader


if __name__ == "__main__":
    main()
