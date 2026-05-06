"""Main training pipeline orchestrator.

Flow:
1) MongoDB raw CSV -> preprocessing -> cleaned parquet back to MongoDB
2) MongoDB cleaned parquet -> transformation -> training artifacts
"""

from __future__ import annotations

import os
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config.data_preprocessing_config import DataPreprocessingConfig
from src.config.data_transformation_config import DataTransformationConfig
from src.config.mongodb_data_exchange_config import MongoDBDataExchangeConfig
from src.pipeline.mongodb_preprocess_pipeline import MongoDBPreprocessPipeline
from src.pipeline.mongodb_transformation_pipeline import MongoDBTransformationPipeline
from src.utils.exception import CustomException
from src.utils.logger import logging


class TrainPipeline:
    """Orchestrates preprocessing + transformation as one training flow."""

    def __init__(
        self,
        mongo_config: MongoDBDataExchangeConfig | None = None,
        preprocessing_config: DataPreprocessingConfig | None = None,
        transformation_config: DataTransformationConfig | None = None,
    ) -> None:
        self.mongo_config = mongo_config or MongoDBDataExchangeConfig()
        self.preprocessing_config = preprocessing_config or DataPreprocessingConfig()
        self.transformation_config = transformation_config or DataTransformationConfig()

    def run(self) -> dict[str, Any]:
        """Run full training data preparation flow and return outputs."""
        try:
            logging.info("Starting main train pipeline.")

            preprocess_pipeline = MongoDBPreprocessPipeline(
                mongo_config=self.mongo_config,
                preprocessing_config=self.preprocessing_config,
            )
            cleaned_path, cleaned_file_id = preprocess_pipeline.run()

            transform_pipeline = MongoDBTransformationPipeline(
                transformation_config=self.transformation_config,
                mongo_config=self.mongo_config,
            )
            transformation_outputs = transform_pipeline.run()

            result = {
                "preprocessing": {
                    "cleaned_local_path": cleaned_path,
                    "cleaned_gridfs_file_id": cleaned_file_id,
                },
                "transformation": transformation_outputs,
            }
            logging.info("Main train pipeline completed.")
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
