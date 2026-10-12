"""One place that builds a :class:`ModeService` over the real pipeline (RAG-014).

The RAG-013 CLI and the RAG-014 web UI both need the same thing: RAG-009 retrieval,
RAG-011 generation over the local Ollama model, RAG-012 validation and RAG-010 discovery,
wired into one :class:`~app.modes.service.ModeService`. Building that in two places would
be two places to get the safety wiring wrong, so it happens here and both callers delegate.

Everything is injectable, so tests never load a model or open a Qdrant store.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings
from app.modes.service import ModeService


def build_mode_service(
    settings: Settings | None = None,
    *,
    model: str | None = None,
    rerank: bool = True,
    retrieval_service: Any | None = None,
    generation_service: Any | None = None,
    citation_service: Any | None = None,
    discovery_service: Any | None = None,
    build_retrieval: Any | None = None,
) -> ModeService:
    """Build a mode service over the real services, with optional overrides.

    ``build_retrieval`` lets a caller supply the RAG-009 builder (the web app uses it to
    reuse an already open Qdrant client); when it is omitted the standard factory is used.
    """
    settings = settings or get_settings()
    if retrieval_service is None:
        from app.retrieval.factory import build_retrieval_service

        builder = build_retrieval or build_retrieval_service
        retrieval_service = builder(settings, rerank=rerank)
    if generation_service is None:
        from app.generation.ollama import OllamaClient
        from app.generation.service import GenerationService

        model_client = OllamaClient(
            base_url=settings.ollama_base_url,
            model=model or settings.generation_model,
            timeout=settings.generation_timeout,
            keep_alive=settings.generation_keep_alive,
            think=settings.generation_think,
        )
        generation_service = GenerationService(
            retrieval_service=retrieval_service, model=model_client, settings=settings
        )
    return ModeService(
        retrieval_service=retrieval_service,
        generation_service=generation_service,
        citation_service=citation_service,
        discovery_service=discovery_service,
        settings=settings,
    )


__all__ = ["build_mode_service"]
