"""Pipeline to fetch raw CSV from MongoDB, preprocess, and upload parquet."""

from __future__ import annotations

import os
import sys
from typing import Tuple

if __package__ is None or __package__ == "":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.components.mongodb_data_exchange import MongoDBDataExchange
from src.config.data_preprocessing_config import DataPreprocessingConfig
from src.config.mongodb_data_exchange_config import MongoDBDataExchangeConfig
from src.utils.exception import CustomException
from src.utils.logger import logging


class MongoDBPreprocessPipeline:
    """Orchestrates MongoDB download -> preprocess -> MongoDB upload."""

    def __init__(
        self,
        mongo_config: MongoDBDataExchangeConfig | None = None,
        preprocessing_config: DataPreprocessingConfig | None = None,
    ) -> None:
        self.mongo_config = mongo_config or MongoDBDataExchangeConfig()
        self.preprocessing_config = preprocessing_config or DataPreprocessingConfig()

    def run(self) -> Tuple[str, str]:
        """Run full roundtrip and return cleaned parquet path and GridFS file id."""
        try:
            logging.info("Starting MongoDB preprocess pipeline.")
            component = MongoDBDataExchange(self.mongo_config)
            cleaned_path, file_id = component.run_preprocess_roundtrip(self.preprocessing_config)
            logging.info("MongoDB preprocess pipeline completed.")
            return cleaned_path, file_id
        except Exception as error:
            raise CustomException(error, sys) from error
