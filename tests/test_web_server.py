"""The server entry point, the binding and the safety of the served frontend (RAG-014 §17, §22).

Two things are being protected here. The first is the default: this interface can read the
whole corpus, including the author's own drafts and supervision material, so it binds to
``127.0.0.1`` and only leaves the machine when someone asks for it explicitly. The second is
the rendering contract: every value the frontend shows goes through ``textContent``, so a
document that contains markup is displayed, never executed.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.modes import ADVANCED_OPTIONS
from app.web import DEFAULT_HOST, DEFAULT_PORT, STATIC_DIR, create_web_app
from app.web import __main__ as web_main
from app.web.service import ModeRunner
from tests.web_support import build_web_app, static_files


# --- the default binding ----------------------------------------------------


def test_the_default_binding_is_localhost() -> None:
    assert DEFAULT_HOST == "127.0.0.1"
    assert DEFAULT_PORT == 8000


def test_the_command_line_defaults_to_localhost() -> None:
    args = web_main.parse_args([])
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.model is None
    assert args.no_rerank is False


def test_exposing_the_interface_requires_the_explicit_host_flag() -> None:
    args = web_main.parse_args(["--host", "0.0.0.0", "--port", "9000"])
    assert args.host == "0.0.0.0"
    assert args.port == 9000


def test_the_banner_says_so_when_the_interface_is_exposed(capsys, monkeypatch) -> None:
    """Starting on 0.0.0.0 must be visible, not silent."""
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *_, **__: None)
    assert web_main.main(["--host", "0.0.0.0", "--port", "8123"]) == 0
    out = capsys.readouterr().out
    assert "http://0.0.0.0:8123/" in out
    assert "WARNING" in out


def test_the_banner_shows_the_url_and_the_model(capsys, monkeypatch) -> None:
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *_, **__: None)
    assert web_main.main([]) == 0
    out = capsys.readouterr().out
    assert "http://127.0.0.1:8000/" in out
    assert "qwen3.5:35b" in out
    assert "WARNING" not in out


def test_one_command_builds_the_whole_app() -> None:
    """`python -m app.web` serves the API and the page from the same application."""
    runner = ModeRunner(service_factory=lambda *_, **__: None)
    application = create_web_app(mode_runner=runner)
    assert application.state.mode_runner is runner
    paths = set(application.openapi()["paths"])
    assert {"/health", "/api/modes", "/api/mode", "/api/status"} <= paths


# --- the static frontend ----------------------------------------------------


def test_the_frontend_files_exist() -> None:
    for name in ("index.html", "app.js", "style.css"):
        assert (STATIC_DIR / name).is_file(), name


def test_the_page_has_the_controls_the_brief_asks_for() -> None:
    page = static_files()["index.html"]
    for element in (
        'id="mode"',          # mode selector
        'id="query"',         # large question box
        'id="run"',           # run
        'id="clear"',         # clear
        'id="loading"',       # status/loading indicator
        'id="mode-description"',  # mode explanation
        'id="body"',          # answer panel
        'id="sources-panel"',  # expandable sources
        'id="validation-panel"',  # expandable validation
        'id="external-panel"',  # external discoveries
        'id="warnings"',      # warnings
        'id="references"',    # references
        'data-copy="answer"',
        'data-copy="answer-references"',
        'data-copy="references"',
        'id="advanced"',      # advanced options (collapsed)
    ):
        assert element in page, element
    assert '<details id="advanced"' in page  # collapsed by default: no `open`


def test_the_advanced_panel_offers_only_the_documented_controls() -> None:
    page = static_files()["index.html"]
    for option in ADVANCED_OPTIONS:
        assert f'id="opt-{option.replace("_", "-")}"' in page
    # Nothing internal is exposed as a control.
    for forbidden in (
        'id="opt-context-tokens"',
        'id="opt-temperature"',
        'id="opt-provider"',
        'id="opt-external-k"',
        'id="opt-qdrant"',
        'id="opt-local-k"',
    ):
        assert forbidden not in page


def test_the_frontend_never_interprets_data_as_markup() -> None:
    script = static_files()["app.js"]
    assert "innerHTML" not in script
    assert "outerHTML" not in script
    assert "insertAdjacentHTML" not in script
    assert "document.write" not in script
    assert "eval(" not in script
    # Every data field is written as text.
    assert "textContent" in script


def test_the_frontend_renders_the_sections_the_modes_produce() -> None:
    """The renderers exist for each mode's output, keyed on the server's own values."""
    script = static_files()["app.js"]
    for expected in (
        '"evidence"',
        '"writing"',
        '"consistency"',
        "unsupported_claim",
        "numeric_discrepancy",
        '"supervisor"',
        '"research"',
        "ALREADY IN CORPUS",
        "NOT IN LOCAL CORPUS",
        "metadata only — never evidence",
        "method:",
        "Retrieved text, scores and provenance",
    ):
        assert expected in script, expected


def test_the_frontend_offers_the_documented_keyboard_shortcuts() -> None:
    script = static_files()["app.js"]
    assert "event.ctrlKey" in script and 'event.key === "Enter"' in script
    assert 'event.key === "Escape"' in script
    assert "requestSubmit()" in script


def test_every_mode_the_api_serves_has_a_title_in_the_frontend() -> None:
    script = static_files()["app.js"]
    for mode in ("evidence", "writing", "consistency", "supervisor", "research"):
        assert f"{mode}: \"" in script, mode


# --- the runner -------------------------------------------------------------


def test_the_runner_builds_the_pipeline_once() -> None:
    calls: list[dict[str, object]] = []

    class _Service:
        def run_mode(self, mode, query, **kwargs):
            return {"mode": mode, "query": query}

        def close(self):
            calls.append({"closed": True})

    runner = ModeRunner(service_factory=lambda settings, **kwargs: calls.append(kwargs) or _Service())
    for index in range(3):
        runner.run("evidence", f"question {index}")
    assert runner.build_count == 1
    assert len(calls) == 1  # only the factory call; nothing closed yet
    runner.close()
    assert runner.is_built is False
    assert calls[-1] == {"closed": True}


def test_the_runner_runs_one_request_at_a_time() -> None:
    """Two concurrent runs would be two model forwards on one local Qdrant store."""
    import threading
    import time

    state = {"active": 0, "max_active": 0}

    class _Service:
        def run_mode(self, mode, query, **kwargs):
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.05)
            state["active"] -= 1
            return {"mode": mode}

        def close(self):
            pass

    runner = ModeRunner(service_factory=lambda settings, **kwargs: _Service())
    threads = [
        threading.Thread(target=runner.run, args=("evidence", f"q{index}"))
        for index in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert state["max_active"] == 1
    assert runner.build_count == 1


def test_the_runner_passes_options_and_the_query_through() -> None:
    seen: list[dict[str, object]] = []

    class _Service:
        def run_mode(self, mode, query, **kwargs):
            seen.append({"mode": mode, "query": query, **kwargs})
            return {"mode": mode}

        def close(self):
            pass

    runner = ModeRunner(service_factory=lambda settings, **kwargs: _Service())
    runner.run("research", "find papers", options={"year_min": 2015, "rerank": False})
    assert seen[0] == {
        "mode": "research",
        "query": "find papers",
        "year_min": 2015,
        "rerank": False,
    }


# --- integration: the served app answers ------------------------------------


def test_health_still_works_alongside_the_ui() -> None:
    client, _, _, _ = build_web_app()
    assert client.get("/health").json() == {
        "status": "ok",
        "application": "PhD Academic RAG",
    }
    assert client.get("/").status_code == 200


def test_a_missing_static_file_is_a_plain_404() -> None:
    client, _, _, _ = build_web_app()
    assert client.get("/nope.txt").status_code == 404


def test_an_unexpected_failure_stays_json(monkeypatch) -> None:
    """An HTML error page in a JSON UI leaves the user with nothing to read."""
    client, harness, _, _ = build_web_app()

    def explode(*_: object, **__: object) -> object:
        raise RuntimeError("unexpected internal failure")

    monkeypatch.setattr(harness.service, "run_mode", explode)
    # TestClient re-raises server exceptions by default; a browser does not.
    honest = TestClient(client.app, raise_server_exceptions=False)
    response = honest.post("/api/mode", json={"mode": "evidence", "query": "ai"})
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["kind"] == "internal_error"
    assert body["error"]["title"]
    # The traceback stays out of a non-debug response.
    assert "traceback" not in body["error"]


@pytest.mark.parametrize("path", ["/api/modes", "/health"])
def test_read_only_endpoints_reject_nothing_but_work(path: str) -> None:
    assert build_web_app()[0].get(path).status_code == 200
