"""Status reporting (RAG-014 §15).

The status area has two jobs: answer quickly, and never lie. That means it must not load a
model (the first real run does), it must report a locked store as locked, and it must still
answer when nothing is reachable at all.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.api import status as status_api
from app.config import get_settings
from app.web.service import ModeRunner
from tests.modes_support import build_harness
from tests.web_support import FakeOllamaClient, FakeQdrantClient, build_web_app


def _settings(**overrides):
    return get_settings().model_copy(update=overrides)


# --- ollama -----------------------------------------------------------------


def test_ollama_status_reports_the_version_and_that_the_model_is_present() -> None:
    settings = _settings(generation_model="qwen3.5:35b")
    payload = status_api.ollama_status(
        settings,
        client_factory=lambda **_: FakeOllamaClient(models=("qwen3.5:35b",)),
    )
    assert payload["available"] is True
    assert payload["version"] == "0.35.1"
    assert payload["model"] == "qwen3.5:35b"
    assert payload["model_present"] is True


def test_ollama_status_reports_a_missing_model_without_failing() -> None:
    settings = _settings(generation_model="qwen3.5:35b")
    payload = status_api.ollama_status(
        settings, client_factory=lambda **_: FakeOllamaClient(models=("llama3.2:3b",))
    )
    assert payload["available"] is True
    assert payload["model_present"] is False


def test_ollama_status_reports_an_unreachable_server() -> None:
    payload = status_api.ollama_status(
        get_settings(),
        client_factory=lambda **_: FakeOllamaClient(version=None, models=()),
    )
    assert payload["available"] is False
    assert payload["error"]


def test_ollama_status_never_raises_even_if_the_client_explodes() -> None:
    def explode(**_: object) -> object:
        raise RuntimeError("connection refused")

    payload = status_api.ollama_status(get_settings(), client_factory=explode)
    assert payload["available"] is False
    assert "connection refused" in payload["error"]


# --- qdrant -----------------------------------------------------------------


def test_index_status_counts_a_temporary_client_and_closes_it() -> None:
    clients: list[FakeQdrantClient] = []

    def factory(**kwargs: object) -> FakeQdrantClient:
        client = FakeQdrantClient(points=4321)
        clients.append(client)
        return client

    payload = status_api.index_status(_settings(), client_factory=factory)
    assert payload["available"] is True
    assert payload["points"] == 4321
    assert payload["collection_present"] is True
    assert payload["reused_client"] is False
    assert clients[0].closed is True


def test_index_status_reports_a_missing_collection() -> None:
    payload = status_api.index_status(
        _settings(), client_factory=lambda **_: FakeQdrantClient(present=False)
    )
    assert payload["available"] is True
    assert payload["collection_present"] is False
    assert payload["points"] is None
    assert "not in the store" in payload["error"]


def test_index_status_explains_a_locked_store() -> None:
    def factory(**_: object) -> FakeQdrantClient:
        return FakeQdrantClient(
            error=RuntimeError(
                "Storage folder .qdrant_data is already accessed by another instance of "
                "QdrantClient"
            )
        )

    payload = status_api.index_status(_settings(), client_factory=factory)
    assert payload["available"] is False
    assert "locked" in payload["error"]


def test_index_status_reuses_the_client_of_a_built_pipeline() -> None:
    """A local Qdrant store can be opened once: status must not open a second one."""

    class _Retriever:
        collection = "phd_rag_chunks_v1"
        point_count = 987

    class _Service:
        retriever = _Retriever()

        def retrieval_service(self) -> object:
            return self

    runner = ModeRunner(service_factory=lambda *_, **__: _Service())
    runner.service()  # a run happened, so the pipeline (and its client) exists

    def forbidden(**_: object) -> object:
        raise AssertionError("status opened a second Qdrant client")

    payload = status_api.index_status(
        _settings(), runner=runner, client_factory=forbidden
    )
    assert payload["points"] == 987
    assert payload["reused_client"] is True


def test_index_status_reuses_the_real_mode_service_client() -> None:
    """The same check against the real ``ModeService``, whose attribute is a property."""
    from app.modes.service import ModeService
    from tests.modes_support import FakeModeRetrieval

    retrieval = FakeModeRetrieval(evidence=[], my_work=[], guidance=[], supervision=[])

    class _Retriever:
        """Shaped like RAG-009's retriever: an attribute and a *method*."""

        collection = "phd_rag_chunks_v1"

        def point_count(self) -> int:
            return 42

    retrieval.retriever = _Retriever()
    service = ModeService(retrieval_service=retrieval)
    runner = ModeRunner(service_factory=lambda *_, **__: service)
    runner.run("evidence", "anything")  # builds the retrieval service

    def forbidden(**_: object) -> object:
        raise AssertionError("status opened a second Qdrant client")

    payload = status_api.index_status(
        _settings(), runner=runner, client_factory=forbidden
    )
    assert payload["points"] == 42
    assert payload["reused_client"] is True
    assert payload["available"] is True
    assert payload["error"] is None


def test_the_retriever_api_the_status_probe_reads_is_still_the_real_one() -> None:
    """`point_count` is a method on RAG-009's retriever; the probe must keep working."""
    from app.retrieval.dense import DenseRetriever

    retriever = DenseRetriever(
        embedder=object(), client=object(), collection="phd_rag_chunks_v1"
    )
    assert callable(retriever.point_count)
    assert retriever.collection == "phd_rag_chunks_v1"


def test_a_status_probe_while_a_run_is_in_flight_reports_busy_not_broken() -> None:
    """Opening a second client on a store a run already holds would break the run."""
    runner = ModeRunner(service_factory=lambda *_, **__: object())
    assert runner.probe() is True
    try:
        payload = status_api.index_status(
            _settings(),
            runner=runner,
            client_factory=lambda **_: FakeQdrantClient(points=1),
        )
    finally:
        runner.probe_done()
    assert payload["busy"] is True
    assert payload["points"] is None
    assert "in progress" in payload["error"]
    # And the lock is free again afterwards.
    assert runner.probe() is True
    runner.probe_done()


# --- the whole picture ------------------------------------------------------


def test_collect_status_contains_what_the_ui_shows() -> None:
    payload = status_api.collect_status(
        settings=_settings(),
        ollama_client_factory=lambda **_: FakeOllamaClient(models=("qwen3.5:35b",)),
        qdrant_client_factory=lambda **_: FakeQdrantClient(points=10),
    )
    assert set(payload) == {
        "ollama",
        "index",
        "generation_model",
        "embedding_model",
        "reranker_model",
        "locality",
    }
    assert payload["index"]["points"] == 10
    assert "127.0.0.1" in payload["locality"]


def test_status_endpoint_does_not_build_the_pipeline(monkeypatch) -> None:
    """Displaying status must never pay for model loading."""
    monkeypatch.setattr(
        status_api,
        "ollama_client_factory",
        lambda **_: FakeOllamaClient(models=("qwen3.5:35b",)),
    )
    monkeypatch.setattr(
        status_api, "qdrant_client_factory", lambda **_: FakeQdrantClient(points=5)
    )
    client, _, runner, factory = build_web_app()

    response = client.get("/api/status")

    assert response.status_code == 200
    assert runner.build_count == 0
    assert factory.calls == []
    body = response.json()
    assert body["ollama"]["available"] is True
    assert body["index"]["points"] == 5


def test_status_endpoint_still_answers_when_nothing_is_reachable(monkeypatch) -> None:
    monkeypatch.setattr(
        status_api, "ollama_client_factory", lambda **_: FakeOllamaClient(version=None, models=())
    )

    def broken(**_: object) -> object:
        raise RuntimeError("no Qdrant target configured")

    monkeypatch.setattr(status_api, "qdrant_client_factory", broken)
    body = build_web_app()[0].get("/api/status").json()
    assert body["ollama"]["available"] is False
    assert body["index"]["available"] is False
    assert body["index"]["error"]


def test_status_reports_the_index_the_real_configuration_points_at(monkeypatch) -> None:
    """The default configuration must describe the real index, not a guess."""
    captured: list[dict[str, object]] = []

    def factory(**kwargs: object) -> FakeQdrantClient:
        captured.append(kwargs)
        return FakeQdrantClient(points=3)

    monkeypatch.setattr(status_api, "qdrant_client_factory", factory)
    settings = _settings()
    status_api.index_status(settings, client_factory=factory)
    assert captured[0]["path"] == settings.qdrant_path
    assert captured[0]["timeout"] == status_api.STATUS_TIMEOUT_SECONDS


def test_the_status_probe_has_a_short_timeout() -> None:
    """A status line that blocks for 30 seconds is not a status line."""
    captured: list[dict[str, object]] = []

    def factory(**kwargs: object) -> FakeOllamaClient:
        captured.append(kwargs)
        return FakeOllamaClient()

    status_api.ollama_status(get_settings(), client_factory=factory)
    assert captured[0]["timeout"] == status_api.STATUS_TIMEOUT_SECONDS
    assert status_api.STATUS_TIMEOUT_SECONDS <= 5
