"""Configuration for the data preprocessing component."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataPreprocessingConfig:
    """Dataclass config for review text preprocessing."""

    raw_data_path: str = "data/raw.csv"
    cleaned_dir: str = "data/cleaned"
    cleaned_file_name: str = "reviews_cleaned.parquet"

    text_column: str = "text"
    label_column: str = "sentiment"
    output_text_column: str = "text_clean"

    lowercase: bool = True
    remove_html: bool = True
    remove_urls: bool = True
    remove_non_alpha: bool = True
    remove_stopwords: bool = True
    lemmatize: bool = True
    min_token_length: int = 2

    drop_empty_rows: bool = True
    deduplicate_on_text: bool = False
    batch_size: int = 50000

    @property
    def output_path(self) -> str:
        """Resolved output parquet path."""
        return str(Path(self.cleaned_dir) / self.cleaned_file_name)
