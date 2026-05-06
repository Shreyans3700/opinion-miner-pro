"""Configuration for MongoDB raw/cleaned file exchange."""

from dataclasses import dataclass, field
from os import getenv

from src.utils.env_loader import load_project_env
from src.utils.mongo_uri import normalize_mongo_uri

load_project_env()


@dataclass
class MongoDBDataExchangeConfig:
    """Dataclass config for MongoDB GridFS file exchange."""

    mongo_uri: str = field(
        default_factory=lambda: normalize_mongo_uri(
            getenv("MONGODB_URI", "mongodb://localhost:27017")
        )
    )
    database_name: str = "opinion_miner"
    bucket_name: str = "review_files"

    raw_csv_gridfs_filename: str = "raw.csv"
    raw_csv_local_path: str = "data/raw.csv"

    cleaned_parquet_gridfs_filename: str = "reviews_cleaned.parquet"
    replace_existing_cleaned_file: bool = True
