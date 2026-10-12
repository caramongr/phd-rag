"""Status for the web UI (RAG-014 §15).

Two questions the user should be able to answer before typing: is the local model reachable,
and is the index there? Both answers are cheap and neither loads a model:

* **Ollama** is asked over HTTP (``/api/version``, ``/api/tags``) with a short timeout. The
  models themselves are never loaded — that happens on the first real run.
* **Qdrant** is counted. If a run already happened, the client the retrieval service owns is
  reused, because a local Qdrant store may be opened once per process; otherwise a temporary
  client is opened, counted and closed. A locked store is reported as locked, not as an
  exception.

The result is a plain dictionary: the UI shows one line from it, and the collapsed debug
panel shows the rest.
"""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, Request

from app.config import Settings, get_settings
from app.web.service import ModeRunner

router = APIRouter(prefix="/api", tags=["status"])

#: How long a status probe may take before it reports "unavailable". Short on purpose: a
#: status line that blocks for 30 seconds is not a status line.
STATUS_TIMEOUT_SECONDS = 2.0


def ollama_client_factory(**kwargs: Any) -> Any:
    """The Ollama client the status probe uses (patched in tests)."""
    from app.generation.ollama import OllamaClient

    return OllamaClient(**kwargs)


def qdrant_client_factory(**kwargs: Any) -> Any:
    """The Qdrant client a status probe opens when no run has built one (patched in tests)."""
    from app.indexing.collection import build_client

    return build_client(**kwargs)


def _friendly_index_error(exc: Exception) -> str:
    message = str(exc) or type(exc).__name__
    lowered = message.casefold()
    if "already accessed" in lowered or "lock" in lowered:
        return (
            "the local Qdrant store is locked by another process (or by this process's "
            "own pipeline): " + message
        )
    if "no qdrant target" in lowered:
        return "no Qdrant target is configured"
    return message


def ollama_status(
    settings: Settings,
    *,
    client_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Whether Ollama answers, which version, and whether the model is installed."""
    client_factory = client_factory or ollama_client_factory
    payload: dict[str, Any] = {
        "base_url": settings.ollama_base_url,
        "model": settings.generation_model,
        "available": False,
        "version": None,
        "model_present": None,
    }
    try:
        client = client_factory(
            base_url=settings.ollama_base_url,
            model=settings.generation_model,
            timeout=STATUS_TIMEOUT_SECONDS,
        )
        version = client.version()
        models = client.available_models(timeout=STATUS_TIMEOUT_SECONDS)
    except Exception as exc:  # never fatal: status must always answer
        payload["error"] = _friendly_index_error(exc)
        return payload
    payload["available"] = bool(version or models)
    payload["version"] = version
    if models:
        payload["model_present"] = settings.generation_model in models
    if not payload["available"]:
        payload["error"] = "Ollama did not answer on its base URL"
    return payload


def index_status(
    settings: Settings,
    *,
    runner: ModeRunner | None = None,
    client_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Whether the Qdrant index can be read, and how many points it holds."""
    payload: dict[str, Any] = {
        "collection": settings.qdrant_collection,
        "target": settings.qdrant_url or settings.qdrant_path,
        "available": False,
        "points": None,
        "collection_present": None,
        "reused_client": False,
        "busy": False,
        "error": None,
    }
    held = runner.probe() if runner is not None else True
    if not held:
        # A run owns the store right now; opening a second client would break it.
        payload["busy"] = True
        payload["error"] = "a run is in progress and the index is in use"
        return payload
    try:
        return _index_status_locked(
            settings, runner=runner, client_factory=client_factory, payload=payload
        )
    finally:
        if runner is not None:
            runner.probe_done()


def _index_status_locked(
    settings: Settings,
    *,
    runner: ModeRunner | None,
    client_factory: Callable[..., Any] | None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """The count itself, called while the run lock is held (or when there is no runner)."""
    service = runner.retrieval_service() if runner is not None else None
    if service is not None:
        retriever = getattr(service, "retriever", None)
        if retriever is not None:
            payload["reused_client"] = True
            payload["collection"] = getattr(retriever, "collection", payload["collection"])
            try:
                # RAG-009's retriever exposes ``point_count`` as a method; accept either form
                # rather than assuming, because a status line is not worth a broken probe.
                count = getattr(retriever, "point_count")
                payload["points"] = int(count() if callable(count) else count)
                payload["collection_present"] = True
                payload["available"] = True
            except Exception as exc:
                payload["error"] = _friendly_index_error(exc)
            return payload

    if client_factory is None:
        client_factory = qdrant_client_factory
    client = None
    try:
        client = client_factory(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            path=None if settings.qdrant_url else settings.qdrant_path,
            timeout=STATUS_TIMEOUT_SECONDS,
        )
        present = bool(client.collection_exists(settings.qdrant_collection))
        payload["collection_present"] = present
        payload["available"] = True
        if present:
            payload["points"] = int(
                client.count(settings.qdrant_collection, exact=False).count
            )
        else:
            payload["error"] = (
                f"the collection {settings.qdrant_collection!r} is not in the store "
                "(build the index with RAG-008)"
            )
    except Exception as exc:
        payload["error"] = _friendly_index_error(exc)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    return payload


def collect_status(
    *,
    settings: Settings | None = None,
    runner: ModeRunner | None = None,
    ollama_client_factory: Callable[..., Any] | None = None,
    qdrant_client_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Everything the status area shows, without loading a model."""
    settings = settings or get_settings()
    return {
        "ollama": ollama_status(settings, client_factory=ollama_client_factory),
        "index": index_status(
            settings, runner=runner, client_factory=qdrant_client_factory
        ),
        "generation_model": settings.generation_model,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "locality": "localhost only (127.0.0.1) unless explicitly configured otherwise",
    }


@router.get("/status")
def status(request: Request) -> dict[str, Any]:
    """Report model and index availability for the UI."""
    runner = getattr(request.app.state, "mode_runner", None)
    settings = getattr(request.app.state, "settings", None) or get_settings()
    return collect_status(settings=settings, runner=runner)


__all__ = [
    "STATUS_TIMEOUT_SECONDS",
    "collect_status",
    "index_status",
    "ollama_client_factory",
    "ollama_status",
    "qdrant_client_factory",
    "router",
    "status",
]
