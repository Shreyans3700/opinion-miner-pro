"""Main training pipeline orchestrator.

Flow:
1) MongoDB raw CSV download
2) Local preprocessing
3) Cleaned parquet upload to MongoDB
4) Transformation + artifact persistence for reuse
"""

from __future__ import annotations

import os
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    )

from src.components.data_preprocessing import DataPreprocessing
from src.components.data_transformation import DataTransformation
from src.components.mongodb_storage import MongoDBStorage
from src.config.data_preprocessing_config import DataPreprocessingConfig
from src.config.data_transformation_config import DataTransformationConfig
from src.config.mongodb_storage_config import MongoDBStorageConfig
from src.utils.exception import CustomException
from src.utils.logger import logging


class TrainPipeline:
    """Single training data pipeline with an explicit four-step flow."""

    def __init__(
        self,
        mongo_config: MongoDBStorageConfig | None = None,
        preprocessing_config: DataPreprocessingConfig | None = None,
        transformation_config: DataTransformationConfig | None = None,
    ) -> None:
        self.mongo_config = mongo_config or MongoDBStorageConfig()
        self.preprocessing_config = preprocessing_config or DataPreprocessingConfig()
        self.transformation_config = transformation_config or DataTransformationConfig()

    def run(self) -> dict[str, Any]:
        """Run full training data preparation flow and return outputs."""
        try:
            logging.info("Starting simplified train pipeline.")
            mongo_storage = MongoDBStorage(self.mongo_config)

            raw_csv_local_path = mongo_storage.download_raw_csv()
            self.preprocessing_config.raw_data_path = raw_csv_local_path

            cleaned_path = DataPreprocessing(self.preprocessing_config).run()
            cleaned_file_id = mongo_storage.upload_cleaned_parquet(cleaned_path)

            transformation_outputs = DataTransformation(self.transformation_config).run(
                cleaned_path
            )

            result = {
                "preprocessing": {
                    "raw_local_path": raw_csv_local_path,
                    "cleaned_local_path": cleaned_path,
                    "cleaned_gridfs_file_id": cleaned_file_id,
                },
                "transformation": transformation_outputs,
            }
            logging.info("Simplified train pipeline completed.")
            return result
        except Exception as error:
            raise CustomException(error, sys) from error


if __name__ == "__main__":
    outputs = TrainPipeline().run()
    for stage, stage_outputs in outputs.items():
        if isinstance(stage_outputs, dict):
            for key, value in stage_outputs.items():
                print(f"{stage}.{key}={value}")
        else:
            print(f"{stage}={stage_outputs}")
