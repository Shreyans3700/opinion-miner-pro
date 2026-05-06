"""Class-based data preprocessing component for review text."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

import nltk
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from src.config.data_preprocessing_config import DataPreprocessingConfig
from src.utils.exception import CustomException
from src.utils.logger import logging


class DataPreprocessing:
    """Prepare raw review data for downstream ABSA components."""

    def __init__(self, config: DataPreprocessingConfig) -> None:
        self.config = config
        self._lemmatizer: WordNetLemmatizer | None = None
        self._stop_words: set[str] = set()
        self._initialize_nlp_assets()

    def run(self) -> str:
        """Execute load -> validate -> preprocess -> save workflow."""
        try:
            logging.info("Starting data preprocessing component.")
            if self.config.batch_size and self.config.batch_size > 0:
                output_path = self.run_in_batches()
            else:
                df = self.load_data()
                self.validate_schema(df)
                processed_df = self.preprocess_dataframe(df)
                output_path = self.save_data(processed_df)
            logging.info("Data preprocessing completed: %s", output_path)
            return output_path
        except Exception as error:
            raise CustomException(error, sys) from error

    def load_data(self) -> pd.DataFrame:
        """Load raw CSV input."""
        try:
            logging.info("Loading raw data from %s", self.config.raw_data_path)
            df = pd.read_csv(self.config.raw_data_path)
            if df.empty:
                raise ValueError("Input dataset is empty.")
            return df
        except Exception as error:
            raise CustomException(error, sys) from error

    def validate_schema(self, df: pd.DataFrame) -> None:
        """Validate required input schema."""
        required_columns = {self.config.text_column, self.config.label_column}
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise CustomException(
                ValueError(
                    f"Missing required columns: {sorted(missing_columns)}. "
                    f"Available columns: {list(df.columns)}"
                ),
                sys,
            )

    def preprocess_dataframe(
        self, df: pd.DataFrame, strict_non_empty: bool = True
    ) -> pd.DataFrame:
        """Apply text normalization and build model-ready dataframe."""
        try:
            output_df = df.copy()
            output_df[self.config.text_column] = output_df[
                self.config.text_column
            ].fillna("")

            output_df[self.config.output_text_column] = output_df[
                self.config.text_column
            ].map(self.preprocess_text)

            if self.config.drop_empty_rows:
                output_df = output_df[
                    output_df[self.config.output_text_column].str.strip() != ""
                ]

            if self.config.deduplicate_on_text:
                output_df = output_df.drop_duplicates(subset=[self.config.text_column])

            if strict_non_empty and output_df.empty:
                raise ValueError("All rows were dropped after preprocessing.")

            output_df = output_df.reset_index(drop=True)
            return output_df
        except Exception as error:
            raise CustomException(error, sys) from error

    def save_data(self, df: pd.DataFrame) -> str:
        """Persist cleaned dataframe in parquet format."""
        try:
            output_dir = self.config.cleaned_dir
            output_path = self.config.output_path

            Path(output_dir).mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path, index=False)
            return output_path
        except Exception as error:
            raise CustomException(error, sys) from error

    def run_in_batches(self) -> str:
        """Process raw CSV in chunks and write parquet incrementally."""
        try:
            output_dir = Path(self.config.cleaned_dir)
            output_path = Path(self.config.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            writer: pq.ParquetWriter | None = None
            total_rows_written = 0

            logging.info(
                "Running preprocessing in batches with chunk size=%s",
                self.config.batch_size,
            )

            for chunk_idx, chunk in enumerate(
                pd.read_csv(
                    self.config.raw_data_path, chunksize=self.config.batch_size
                ),
                start=1,
            ):
                if chunk_idx == 1:
                    self.validate_schema(chunk)

                processed_chunk = self.preprocess_dataframe(
                    chunk, strict_non_empty=False
                )
                if processed_chunk.empty:
                    logging.info("Skipping empty processed chunk=%s", chunk_idx)
                    continue

                table = pa.Table.from_pandas(processed_chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(str(output_path), table.schema)

                writer.write_table(table)
                total_rows_written += len(processed_chunk)

                logging.info(
                    "Processed chunk=%s rows_in_chunk=%s total_rows_written=%s",
                    chunk_idx,
                    len(processed_chunk),
                    total_rows_written,
                )

            if writer is not None:
                writer.close()

            if total_rows_written == 0:
                raise ValueError("No rows were written after preprocessing all chunks.")

            return str(output_path)
        except Exception as error:
            raise CustomException(error, sys) from error

    def preprocess_text(self, text: str) -> str:
        """Normalize a review string into model-ready text."""
        if text is None:
            return ""

        normalized_text = str(text)

        if self.config.remove_html:
            normalized_text = re.sub(r"<[^>]+>", " ", normalized_text)

        if self.config.remove_urls:
            normalized_text = re.sub(r"https?://\S+|www\.\S+", " ", normalized_text)

        if self.config.lowercase:
            normalized_text = normalized_text.lower()

        if self.config.remove_non_alpha:
            normalized_text = re.sub(r"[^a-zA-Z\s]", " ", normalized_text)

        normalized_text = re.sub(r"\s+", " ", normalized_text).strip()
        if not normalized_text:
            return ""

        tokens = word_tokenize(normalized_text)
        filtered_tokens = self._filter_tokens(tokens)
        return " ".join(filtered_tokens)

    def _initialize_nlp_assets(self) -> None:
        """Download and initialize NLTK resources required for preprocessing."""
        try:
            self._download_nltk_resources_if_needed()
            self._stop_words = (
                set(stopwords.words("english"))
                if self.config.remove_stopwords
                else set()
            )
            self._lemmatizer = WordNetLemmatizer() if self.config.lemmatize else None
        except Exception as error:
            raise CustomException(error, sys) from error

    def _download_nltk_resources_if_needed(self) -> None:
        """Ensure required NLTK resources are available."""
        required_resources = {
            "punkt_tab": ["tokenizers/punkt_tab/english/"],
            "stopwords": ["corpora/stopwords", "corpora/stopwords.zip"],
            "wordnet": ["corpora/wordnet", "corpora/wordnet.zip"],
            "omw-1.4": ["corpora/omw-1.4", "corpora/omw-1.4.zip"],
        }

        for package_name, resource_paths in required_resources.items():
            if package_name in {"stopwords"} and not self.config.remove_stopwords:
                continue
            if package_name in {"wordnet", "omw-1.4"} and not self.config.lemmatize:
                continue

            try:
                self._find_any_resource_path(resource_paths)
            except LookupError:
                nltk.download(package_name, quiet=True)
                self._find_any_resource_path(resource_paths)

    @staticmethod
    def _find_any_resource_path(resource_paths: list[str]) -> None:
        """Find at least one valid NLTK resource path from candidates."""
        last_error: Exception | None = None
        for resource_path in resource_paths:
            try:
                nltk.data.find(resource_path)
                return
            except LookupError as error:
                last_error = error

        if last_error is not None:
            raise last_error

    def _filter_tokens(self, tokens: Iterable[str]) -> list[str]:
        """Apply stopword removal, lemmatization, and length filtering."""
        cleaned_tokens: list[str] = []

        for token in tokens:
            if self.config.remove_stopwords and token in self._stop_words:
                continue

            if self._lemmatizer is not None:
                token = self._lemmatizer.lemmatize(token)

            if len(token) < self.config.min_token_length:
                continue

            cleaned_tokens.append(token)

        return cleaned_tokens
