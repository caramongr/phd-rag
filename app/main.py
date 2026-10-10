"""FastAPI application entry point.

Builds the application, wires configuration, and mounts the API routers. No
retrieval, storage, or model integrations are registered at this stage.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.api.health import router as health_router
from app.config import get_settings


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
    )
    application.include_router(health_router)
    return application


app = create_app()
