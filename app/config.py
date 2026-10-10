"""Application configuration.

Settings are read from environment variables at process start and cached. The
layer stays deliberately small and dependency-free (no settings library) so the
values and their defaults are easy to audit.

OCR settings (RAG-005) are centralised here so thresholds and the language/DPI
configuration are never scattered through parser code.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import BaseModel, ConfigDict

DEFAULT_APP_NAME = "PhD Academic RAG"
DEFAULT_APP_VERSION = "0.1.0"
DEFAULT_ENVIRONMENT = "development"
DEFAULT_API_PREFIX = "/api"

# --- OCR defaults (RAG-005) ---
DEFAULT_OCR_ENABLED = True
DEFAULT_OCR_LANGUAGE = "eng"
DEFAULT_OCR_DPI = 300
DEFAULT_OCR_MIN_NATIVE_CHARS = 50


class Settings(BaseModel):
    """Runtime configuration for the backend.

    Fields are sourced from ``PHD_RAG_*`` environment variables, falling back to
    safe development defaults.
    """

    model_config = ConfigDict(frozen=True)

    app_name: str = DEFAULT_APP_NAME
    app_version: str = DEFAULT_APP_VERSION
    environment: str = DEFAULT_ENVIRONMENT
    debug: bool = False
    api_prefix: str = DEFAULT_API_PREFIX

    # OCR (RAG-005)
    ocr_enabled: bool = DEFAULT_OCR_ENABLED
    ocr_language: str = DEFAULT_OCR_LANGUAGE
    ocr_dpi: int = DEFAULT_OCR_DPI
    ocr_min_native_chars: int = DEFAULT_OCR_MIN_NATIVE_CHARS


def _parse_bool(value: str) -> bool:
    """Interpret common truthy strings from the environment."""
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    """Build a :class:`Settings` instance from the current environment."""
    return Settings(
        app_name=os.getenv("PHD_RAG_APP_NAME", DEFAULT_APP_NAME),
        app_version=os.getenv("PHD_RAG_APP_VERSION", DEFAULT_APP_VERSION),
        environment=os.getenv("PHD_RAG_ENVIRONMENT", DEFAULT_ENVIRONMENT),
        debug=_parse_bool(os.getenv("PHD_RAG_DEBUG", "false")),
        api_prefix=os.getenv("PHD_RAG_API_PREFIX", DEFAULT_API_PREFIX),
        ocr_enabled=_parse_bool(os.getenv("PHD_RAG_OCR_ENABLED", "true")),
        ocr_language=os.getenv("PHD_RAG_OCR_LANGUAGE", DEFAULT_OCR_LANGUAGE),
        ocr_dpi=int(os.getenv("PHD_RAG_OCR_DPI", str(DEFAULT_OCR_DPI))),
        ocr_min_native_chars=int(
            os.getenv("PHD_RAG_OCR_MIN_NATIVE_CHARS", str(DEFAULT_OCR_MIN_NATIVE_CHARS))
        ),
    )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached settings instance."""
    return load_settings()
