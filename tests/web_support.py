"""Support for the RAG-014 web tests (RAG-014 §19).

The web layer is tested against the *real* RAG-013 mode service wired to the fakes the
RAG-013 tests already use, so what the API returns is the pipeline's own result, and not a
hand-written JSON fixture that could drift away from it.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from app.web.service import ModeRunner
from tests.modes_support import ModeHarness, build_harness


class RecordingFactory:
    """A service factory that counts builds and returns one prepared harness."""

    def __init__(self, harness: ModeHarness) -> None:
        self.harness = harness
        self.calls: list[dict[str, Any]] = []

    def __call__(self, settings: Any, **kwargs: Any) -> Any:
        self.calls.append({"settings": settings, **kwargs})
        return self.harness.service


def build_web_app(harness: ModeHarness | None = None, **harness_kwargs: Any):
    """Return ``(client, harness, runner, factory)`` over the real mode service."""
    harness = harness or build_harness(**harness_kwargs)
    factory = RecordingFactory(harness)
    runner = ModeRunner(settings=get_settings(), service_factory=factory)
    application = create_app(settings=get_settings(), mode_runner=runner)
    return TestClient(application), harness, runner, factory


def build_client(harness: ModeHarness | None = None, **harness_kwargs: Any) -> TestClient:
    """Just the client, for tests that do not need the fakes."""
    client, _, _, _ = build_web_app(harness, **harness_kwargs)
    return client


class FakeOllamaClient:
    """An Ollama client that answers (or does not) without a server."""

    def __init__(self, *, version: str | None = "0.35.1", models: tuple[str, ...] = (), **_: Any) -> None:
        self._version = version
        self._models = models

    def version(self) -> str | None:
        return self._version

    def available_models(self, *, timeout: float | None = None) -> tuple[str, ...]:
        return self._models


class FakeQdrantClient:
    """A Qdrant client with a point count, an optional collection and a lock flag."""

    def __init__(
        self,
        *,
        points: int = 1234,
        present: bool = True,
        error: Exception | None = None,
        **_: Any,
    ) -> None:
        self.points = points
        self.present = present
        self.error = error
        self.closed = False

    def collection_exists(self, collection: str) -> bool:
        if self.error is not None:
            raise self.error
        return self.present

    def count(self, collection: str, exact: bool = False) -> Any:
        if self.error is not None:
            raise self.error

        class _Result:
            def __init__(self, count: int) -> None:
                self.count = count

        return _Result(self.points)

    def close(self) -> None:
        self.closed = True


def static_files() -> dict[str, str]:
    """The three frontend files, as the server serves them."""
    from app.web import STATIC_DIR

    return {
        name: (STATIC_DIR / name).read_text(encoding="utf-8")
        for name in ("index.html", "app.js", "style.css")
    }
