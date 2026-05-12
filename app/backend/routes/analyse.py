"""Analysis API routes for single and bulk review prediction."""

from __future__ import annotations

import io
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
from src.utils.logger import get_logger

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
        logger = get_logger(__name__)
        logger.info("Received single review analysis request")
        result = get_prediction_service().predict_one(payload.review)
        logger.info("Single review analysis complete")
        return result
    except CustomException as error:
        original_error = getattr(error, "original_exception", None)
        if isinstance(original_error, ValueError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(original_error),
            ) from error
        logger.error("Failed to analyse review: %s", error)
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
        logger.error("Failed to analyse review: %s", error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process the review.",
        ) from error


@router.post("/bulkAnalyseResults")
async def bulk_analyse_results(
    file: UploadFile = File(...),
    text_column: str = Form(...),
    page: int = Form(1),
    per_page: int = Form(50),
) -> dict[str, Any]:
    """Analyse reviews from uploaded CSV and return paginated JSON results."""
    logger = get_logger(__name__)
    logger.info(
        "Starting bulk analysis: file=%s, text_column=%s", file.filename, text_column
    )
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

    results: list[dict[str, Any]] = []

    for value in input_df[text_column].fillna("").astype(str):
        if not value.strip():
            continue
        try:
            result = get_prediction_service().predict_one(value)
            results.append(
                {
                    "review": value,
                    "overall_sentiment": result.get(
                        "overall_sentiment", result.get("predicted_label", "")
                    ),
                    "positive_score": result.get("positive_score", 0),
                    "negative_score": result.get("negative_score", 0),
                    "confidence": result.get("confidence", 0),
                    "features": [
                        {
                            "feature": fp.get("feature", ""),
                            "sentiment": fp.get("sentiment", ""),
                            "evidence": fp.get("evidence", ""),
                            "scores": fp.get("scores", {}),
                        }
                        for fp in result.get("main_feature_points", [])
                    ],
                }
            )
        except Exception as e:
            logger.warning("Failed to analyze row: %s", e)
            continue

    total = len(results)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_results = results[start:end]
    logger.info(
        "Bulk analysis complete: total=%d, returned=%d", total, len(paginated_results)
    )

    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "results": paginated_results,
    }


@router.post("/bulkAnalyse")
async def bulk_analyse(
    file: UploadFile = File(...),
    text_column: str = Form(...),
) -> StreamingResponse:
    """Analyse reviews from uploaded CSV and return a downloadable CSV."""
    logger = get_logger(__name__)
    logger.info(
        "Starting bulk analyse (CSV download): file=%s, text_column=%s",
        file.filename,
        text_column,
    )
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
            logger = get_logger(__name__)
            original_error = getattr(error, "original_exception", None)
            error_message = (
                str(original_error) if original_error is not None else str(error)
            )
            logger.warning("CustomException during bulk analyse: %s", error_message)
            predictions.append(
                {
                    "input_text": value,
                    "cleaned_text": "",
                    "tokens": [],
                    "token_count": 0,
                    "embedding_shape": [],
                    "predicted_label": "",
                    "overall_sentiment": "",
                    "predicted_raw": "",
                    "positive_score": None,
                    "negative_score": None,
                    "sentiment_breakdown": None,
                    "aspect_sentiments": None,
                    "aspect_sentiment_scores": None,
                    "main_feature_points": None,
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
                    "overall_sentiment": "",
                    "predicted_raw": "",
                    "positive_score": None,
                    "negative_score": None,
                    "sentiment_breakdown": None,
                    "aspect_sentiments": None,
                    "aspect_sentiment_scores": None,
                    "main_feature_points": None,
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
    logger.info(
        "Bulk analyse (CSV download) complete: %d rows processed", len(predictions)
    )
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="bulk_analysis.csv"'},
    )


@router.post("/train", response_model=TrainResponse, tags=["training"])
def train() -> dict[str, Any]:
    """Run full training pipeline and regenerate model artifacts."""
    logger = get_logger(__name__)
    logger.info("Starting training pipeline")
    try:
        outputs = get_train_service().run_training()
        get_prediction_service.cache_clear()
        return {
            "status": "success",
            "outputs": outputs,
        }
        logger.info("Training pipeline complete")
    except CustomException as error:
        logger.error("Failed to run training pipeline: %s", error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run training pipeline.",
        ) from error
    except Exception as error:
        logger.error("Failed to run training pipeline: %s", error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run training pipeline.",
        ) from error


def flatten_prediction_for_csv(observation: dict[str, Any]) -> dict[str, Any]:
    """Convert nested prediction observations to CSV-friendly scalar fields."""
    features_list = observation.get("main_feature_points", [])
    features_str = (
        ", ".join(
            f"{fp.get('feature', 'unknown')}: {fp.get('sentiment', 'unknown')}"
            for fp in features_list
        )
        if features_list
        else ""
    )

    return {
        "overall_sentiment": observation.get(
            "overall_sentiment", observation.get("predicted_label", "")
        ),
        "positive_score": observation.get("positive_score", 0),
        "negative_score": observation.get("negative_score", 0),
        "confidence": observation.get("confidence", 0),
        "features_and_sentiments": features_str,
    }
