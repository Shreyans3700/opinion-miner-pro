"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from app.backend.routes import analyse_router

app = FastAPI(
    title="Opinion Miner API",
    version="1.0.0",
    description="FastAPI backend for single and bulk sentiment analysis.",
)

app.include_router(analyse_router)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Basic health endpoint."""
    return {"status": "ok"}

