"""Analysis API routes for single and bulk review prediction."""

from __future__ import annotations

import io
import json
from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from app.backend.deps.auth import verify_api_key
from app.backend.schemas.analyse import AnalyseRequest, AnalyseResponse
from app.backend.schemas.train import TrainResponse
from app.backend.services.prediction_service import PredictionService
from app.backend.services.train_service import TrainService
from src.utils.exception import CustomException
from src.utils.logger import logging

router = APIRouter(tags=["analysis"], dependencies=[Depends(verify_api_key)])


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    """Lazily construct a shared prediction service instance."""
    return PredictionService()


@lru_cache(maxsize=1)
def get_train_service() -> TrainService:
    """Lazily construct a shared train service instance."""
    return TrainService()


@router.post("/analyse", response_model=AnalyseResponse)
def analyse(payload: AnalyseRequest) -> dict[str, Any]:
    """Analyse a single review and return inference observations."""
    try:
        return get_prediction_service().predict_one(payload.review)
    except CustomException as error:
        original_error = getattr(error, "original_exception", None)
        if isinstance(original_error, ValueError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(original_error),
            ) from error
        logging.error("Failed to analyse review: %s", error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process the review.",
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except Exception as error:
        logging.error("Failed to analyse review: %s", error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process the review.",
        ) from error


@router.post("/bulkAnalyse")
async def bulk_analyse(
    file: UploadFile = File(...),
    text_column: str = Form(...),
) -> StreamingResponse:
    """Analyse reviews from uploaded CSV and return a downloadable CSV."""
    if not text_column.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="text_column cannot be empty.",
        )

    uploaded_name = file.filename or ""
    if not uploaded_name.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only CSV files are supported.",
        )

    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded CSV is empty.",
            )

        input_df = pd.read_csv(io.BytesIO(file_bytes))
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid CSV file: {error}",
        ) from error

    if text_column not in input_df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Column '{text_column}' not found in CSV. "
                f"Available columns: {list(input_df.columns)}"
            ),
        )

    output_df = input_df.copy()
    predictions: list[dict[str, Any]] = []

    for value in output_df[text_column].fillna("").astype(str):
        try:
            result = get_prediction_service().predict_one(value)
            predictions.append(result)
        except CustomException as error:
            original_error = getattr(error, "original_exception", None)
            error_message = (
                str(original_error) if original_error is not None else str(error)
            )
            predictions.append(
                {
                    "input_text": value,
                    "cleaned_text": "",
                    "tokens": [],
                    "token_count": 0,
                    "embedding_shape": [],
                    "predicted_label": "",
                    "predicted_raw": "",
                    "confidence": None,
                    "decision_margin": None,
                    "class_probabilities": None,
                    "error": error_message,
                }
            )
        except Exception as error:
            predictions.append(
                {
                    "input_text": value,
                    "cleaned_text": "",
                    "tokens": [],
                    "token_count": 0,
                    "embedding_shape": [],
                    "predicted_label": "",
                    "predicted_raw": "",
                    "confidence": None,
                    "decision_margin": None,
                    "class_probabilities": None,
                    "error": str(error),
                }
            )

    prediction_df = pd.DataFrame(
        [flatten_prediction_for_csv(row) for row in predictions]
    )
    merged_df = pd.concat([output_df.reset_index(drop=True), prediction_df], axis=1)

    csv_bytes = merged_df.to_csv(index=False).encode("utf-8")
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="bulk_analysis.csv"'},
    )


@router.post("/train", response_model=TrainResponse, tags=["training"])
def train() -> dict[str, Any]:
    """Run full training pipeline and regenerate model artifacts."""
    try:
        outputs = get_train_service().run_training()
        get_prediction_service.cache_clear()
        return {
            "status": "success",
            "outputs": outputs,
        }
    except CustomException as error:
        logging.error("Failed to run training pipeline: %s", error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run training pipeline.",
        ) from error
    except Exception as error:
        logging.error("Failed to run training pipeline: %s", error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run training pipeline.",
        ) from error


def flatten_prediction_for_csv(observation: dict[str, Any]) -> dict[str, Any]:
    """Convert nested prediction observations to CSV-friendly scalar fields."""
    flattened = dict(observation)

    for field_name in ("tokens", "embedding_shape", "class_probabilities"):
        field_value = flattened.get(field_name)
        if field_value is not None:
            flattened[field_name] = json.dumps(field_value, ensure_ascii=True)

    return flattened
