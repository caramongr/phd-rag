"""Health check endpoints.

These routes let operators and load balancers confirm the service is running and
surface which build/environment answered the request.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Payload returned by the health endpoint."""

    status: str
    application: str


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report that the API process is up."""
    settings = get_settings()
    return HealthResponse(status="ok", application=settings.app_name)
