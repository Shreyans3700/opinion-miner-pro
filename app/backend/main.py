"""FastAPI application entrypoint."""

from __future__ import annotations

from pathlib import Path
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.backend.routes import analyse_router
from src.utils.logger import logging

app = FastAPI(
    title="Opinion Miner API",
    version="1.0.0",
    description="FastAPI backend for single and bulk sentiment analysis.",
)

app.include_router(analyse_router)

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
STATIC_DIR = FRONTEND_DIR

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def log_requests(request, call_next):
    """Log incoming requests and their status codes."""
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        logging.exception(
            "%s %s failed after %.2f ms",
            request.method,
            request.url.path,
            duration_ms,
        )
        raise

    duration_ms = (time.perf_counter() - start) * 1000
    logging.info(
        "%s %s -> %s (%.2f ms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    """Serve frontend entry page."""
    index_file = FRONTEND_DIR / "index.html"
    response = FileResponse(index_file)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Basic health endpoint."""
    return {"status": "ok"}
