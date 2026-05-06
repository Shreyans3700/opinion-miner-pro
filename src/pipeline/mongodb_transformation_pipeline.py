"""Pipeline to run data transformation by reading cleaned parquet from MongoDB."""

from __future__ import annotations

import os
import sys

if __package__ is None or __package__ == "":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.components.data_transformation import DataTransformation
from src.config.data_transformation_config import DataTransformationConfig
from src.config.mongodb_data_exchange_config import MongoDBDataExchangeConfig
from src.utils.exception import CustomException
from src.utils.logger import logging


class MongoDBTransformationPipeline:
    """Orchestrates transformation workflow sourced from MongoDB."""

    def __init__(
        self,
        transformation_config: DataTransformationConfig | None = None,
        mongo_config: MongoDBDataExchangeConfig | None = None,
    ) -> None:
        self.transformation_config = transformation_config or DataTransformationConfig()
        self.mongo_config = mongo_config or MongoDBDataExchangeConfig()

    def run(self) -> dict[str, str]:
        """Run transformation and return artifact paths."""
        try:
            logging.info("Starting MongoDB transformation pipeline.")
            component = DataTransformation(
                transformation_config=self.transformation_config,
                mongo_config=self.mongo_config,
            )
            output_paths = component.run()
            logging.info("MongoDB transformation pipeline completed.")
            return output_paths
        except Exception as error:
            raise CustomException(error, sys) from error

