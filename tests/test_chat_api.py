"""The simple-chat API (RAG-015 §13, §16, §17 example 4).

Endpoints, error envelopes, security and — the point of a repair rather than a rewrite — the
proof that the RAG-014 interface still answers next to the new chat, on the same application.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from app.api.modes import MAX_QUERY_CHARS
from app.chat.service import DEFAULT_CHAT_CORPUS_FINAL_K
from app.config import DEFAULT_CHAT_NUM_CTX, DEFAULT_GENERATION_NUM_CTX
from app.main import create_app
from app.modes.models import ModeName
from app.web.service import ChatRunner, ModeRunner, make_corpus_provider
from tests.chat_support import (
    ScriptedModel,
    answer_result,
    attachment_body,
    build_client,
    chat_settings,
    docx_bytes,
    failure_result,
    text_bytes,
)
from tests.modes_support import build_harness
from tests.web_support import static_files

HOSTILE = '<script>alert("x")</script> & <img src=x onerror=alert(1)>'


# --- chat info ----------------------------------------------------------------


def test_chat_info_reports_the_full_window_and_the_limits(tmp_path: Path) -> None:
    client, _, _, _ = build_client(tmp_path)
    payload = client.get("/api/chat/info").json()

    assert payload["num_ctx"] == 262144 == DEFAULT_CHAT_NUM_CTX
    assert payload["num_ctx"] != DEFAULT_GENERATION_NUM_CTX
    assert payload["max_output_tokens"] == 4096  # FIX1: the conversation-sized output cap
    assert set(payload["accepted_formats"]) == {".pdf", ".docx", ".txt", ".md"}
    assert payload["max_attachment_bytes"] > 0
    assert payload["max_message_chars"] > MAX_QUERY_CHARS
    assert payload["corpus_retrieval_available"] is True


# --- chat lifecycle -----------------------------------------------------------


def test_chat_lifecycle_over_the_api(tmp_path: Path) -> None:
    client, _, model, _ = build_client(tmp_path)

    created = client.post("/api/chats", json={})
    assert created.status_code == 201
    chat_id = created.json()["id"]
    assert created.json()["title"] == "New chat"

    assert client.get("/api/chats").json()["chats"][0]["id"] == chat_id

    answered = client.post(f"/api/chats/{chat_id}/messages", json={"message": "Hello there."})
    body = answered.json()
    assert answered.status_code == 200
    assert body["user_message"]["content"] == "Hello there."
    assert body["assistant_message"]["content"] == "Answer from the conversation."
    assert body["context"]["num_ctx"] == 262144
    assert model.last_prompt.rstrip().endswith("Hello there.")

    renamed = client.patch(f"/api/chats/{chat_id}", json={"title": "Renamed chat"})
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Renamed chat"

    reopened = client.get(f"/api/chats/{chat_id}").json()
    assert [item["role"] for item in reopened["messages"]] == ["user", "assistant"]
    assert reopened["attachments"] == []

    deleted = client.delete(f"/api/chats/{chat_id}")
    assert deleted.status_code == 200 and deleted.json()["deleted"] is True
    assert client.get("/api/chats").json()["chats"] == []
    assert client.get(f"/api/chats/{chat_id}").status_code == 404


def test_an_explicit_title_can_be_given_at_creation(tmp_path: Path) -> None:
    client, _, _, _ = build_client(tmp_path)
    created = client.post("/api/chats", json={"title": "Chapter three"}).json()
    assert created["title"] == "Chapter three"


# --- attachments --------------------------------------------------------------


def test_attachment_round_trip_through_the_api(tmp_path: Path) -> None:
    client, _, model, settings = build_client(tmp_path)
    chat_id = client.post("/api/chats", json={}).json()["id"]
    data = docx_bytes(
        (
            "Paragraph one explains the research context.",
            "Paragraph two argues that governance shapes adoption.",
        ),
        heading="Chapter three",
    )

    uploaded = client.post(
        f"/api/chats/{chat_id}/attachments", json=attachment_body("chapter.docx", data)
    )

    assert uploaded.status_code == 201
    attachment = uploaded.json()
    assert attachment["filename"] == "chapter.docx"
    assert attachment["extension"] == ".docx"
    assert attachment["text_status"] == "parsed"
    assert attachment["char_count"] > 0
    # The API never exposes a filesystem path.
    assert "stored_path" not in attachment
    assert json.dumps(attachment)

    assert (
        client.get(f"/api/chats/{chat_id}").json()["attachments"][0]["id"] == attachment["id"]
    )

    client.post(f"/api/chats/{chat_id}/messages", json={"message": "What does paragraph two say?"})
    assert "Paragraph two argues that governance shapes adoption." in model.last_prompt

    removed = client.delete(f"/api/chats/{chat_id}/attachments/{attachment['id']}")
    assert removed.status_code == 200
    assert client.get(f"/api/chats/{chat_id}").json()["attachments"] == []
    assert not list(Path(settings.chat_attachment_dir).rglob("*.docx"))


def test_an_unsupported_upload_gets_a_safe_readable_error(tmp_path: Path) -> None:
    client, _, _, _ = build_client(tmp_path)
    chat_id = client.post("/api/chats", json={}).json()["id"]

    response = client.post(
        f"/api/chats/{chat_id}/attachments",
        json=attachment_body("huge.zip", b"PK\x03\x04"),
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["kind"] == "unsupported_attachment"
    assert "PDF" in error["detail"]
    assert "Traceback" not in response.text
    assert client.get(f"/api/chats/{chat_id}").json()["attachments"] == []


def test_an_oversized_upload_gets_a_readable_413(tmp_path: Path) -> None:
    client, _, _, _ = build_client(tmp_path, chat_max_attachment_bytes=1024 * 1024)
    chat_id = client.post("/api/chats", json={}).json()["id"]

    response = client.post(
        f"/api/chats/{chat_id}/attachments",
        json=attachment_body("big.txt", b"x" * (2 * 1024 * 1024)),
    )

    assert response.status_code == 413
    assert response.json()["error"]["kind"] == "attachment_too_large"
    assert "1.0 MB" in response.json()["error"]["detail"]


def test_invalid_base64_is_reported_without_a_traceback(tmp_path: Path) -> None:
    client, _, _, _ = build_client(tmp_path)
    chat_id = client.post("/api/chats", json={}).json()["id"]

    response = client.post(
        f"/api/chats/{chat_id}/attachments",
        json={"filename": "notes.txt", "content_base64": "not base64 at all!!"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["kind"] == "attachment_rejected"
    assert "base64" in response.json()["error"]["detail"]


def test_a_traversal_filename_through_the_api_stays_inside_the_directory(tmp_path: Path) -> None:
    client, _, _, settings = build_client(tmp_path)
    chat_id = client.post("/api/chats", json={}).json()["id"]

    uploaded = client.post(
        f"/api/chats/{chat_id}/attachments",
        json=attachment_body("../../../../etc/passwd.txt", text_bytes("nope")),
    )

    assert uploaded.status_code == 201
    assert uploaded.json()["filename"] == "passwd.txt"
    stored = list(Path(settings.chat_attachment_dir).rglob("*.txt"))
    assert len(stored) == 1
    assert not (tmp_path.parent / "etc").exists()


def test_deleting_an_unknown_attachment_is_a_readable_error(tmp_path: Path) -> None:
    client, _, _, _ = build_client(tmp_path)
    chat_id = client.post("/api/chats", json={}).json()["id"]

    response = client.delete(f"/api/chats/{chat_id}/attachments/deadbeef")

    assert response.status_code == 400
    assert response.json()["error"]["kind"] == "attachment_rejected"


# --- request validation -------------------------------------------------------


def test_unknown_chat_is_a_404_with_the_standard_envelope(tmp_path: Path) -> None:
    client, _, _, _ = build_client(tmp_path)

    for method, url, body in (
        ("get", "/api/chats/nope", None),
        ("patch", "/api/chats/nope", {"title": "x"}),
        ("delete", "/api/chats/nope", None),
        ("post", "/api/chats/nope/messages", {"message": "hi"}),
        ("post", "/api/chats/nope/attachments", attachment_body("a.txt", b"a")),
    ):
        response = getattr(client, method)(url, json=body) if body else getattr(client, method)(url)
        assert response.status_code == 404, (method, url)
        assert response.json()["error"]["kind"] == "chat_not_found"


def test_an_empty_message_is_refused_with_a_422(tmp_path: Path) -> None:
    client, _, _, _ = build_client(tmp_path)
    chat_id = client.post("/api/chats", json={}).json()["id"]

    response = client.post(f"/api/chats/{chat_id}/messages", json={"message": "   "})

    assert response.status_code == 422
    assert response.json()["error"]["kind"] == "invalid_request"


def test_an_empty_rename_is_refused(tmp_path: Path) -> None:
    client, _, _, _ = build_client(tmp_path)
    chat_id = client.post("/api/chats", json={}).json()["id"]

    response = client.patch(f"/api/chats/{chat_id}", json={"title": "  "})

    assert response.status_code == 422


def test_a_model_failure_is_reported_and_the_message_is_kept(tmp_path: Path) -> None:
    from app.generation.models import ModelFailure
    from tests.generation_support import FakeGenerationModel

    model = FakeGenerationModel(failure=ModelFailure(kind="server_error", message="boom"))
    client, _, _, _ = build_client(tmp_path, model=model)
    chat_id = client.post("/api/chats", json={}).json()["id"]

    body = client.post(f"/api/chats/{chat_id}/messages", json={"message": "Still saved?"}).json()

    assert body["assistant_message"] is None
    assert body["failure"]["kind"] == "server_error"
    stored = client.get(f"/api/chats/{chat_id}").json()["messages"]
    assert [item["content"] for item in stored] == ["Still saved?"]


# --- the failed-turn retry (FIX1 §6) ------------------------------------------


def test_the_retry_endpoint_answers_the_existing_turn_only_once(tmp_path: Path) -> None:
    model = ScriptedModel([failure_result(), failure_result(), failure_result(), answer_result("Answered on retry.")])
    client, service, _, _ = build_client(tmp_path, model=model)
    chat_id = client.post("/api/chats", json={}).json()["id"]

    first = client.post(f"/api/chats/{chat_id}/messages", json={"message": "About lattice?"}).json()
    assert first["assistant_message"] is None
    assert first["failure"]["retries"] == 1  # the automatic retry was already used
    assert first["failure"]["retry_reason"]

    retried = client.post(f"/api/chats/{chat_id}/retry")

    assert retried.status_code == 200
    body = retried.json()
    assert body["assistant_message"]["content"] == "Answered on retry."
    stored = client.get(f"/api/chats/{chat_id}").json()["messages"]
    assert [(item["role"], item["content"]) for item in stored] == [
        ("user", "About lattice?"),
        ("assistant", "Answered on retry."),
    ]


def test_a_turn_that_needs_a_retry_says_so_in_the_report(tmp_path: Path) -> None:
    model = ScriptedModel([failure_result(), answer_result("Recovered.")])
    client, _, _, _ = build_client(tmp_path, model=model)
    chat_id = client.post("/api/chats", json={}).json()["id"]

    body = client.post(f"/api/chats/{chat_id}/messages", json={"message": "A question."}).json()

    assert body["retried"] is True
    assert body["context"]["generation_retries"] == 1
    assert "token repeat limit reached" in body["context"]["retry_reason"]
    assert body["assistant_message"]["content"] == "Recovered."


def test_retrying_an_answered_chat_is_a_409(tmp_path: Path) -> None:
    client, _, _, _ = build_client(tmp_path)
    chat_id = client.post("/api/chats", json={}).json()["id"]
    client.post(f"/api/chats/{chat_id}/messages", json={"message": "A question."})

    response = client.post(f"/api/chats/{chat_id}/retry")

    assert response.status_code == 409
    error = response.json()["error"]
    assert error["kind"] == "nothing_to_retry"
    assert "nothing to retry" in error["detail"].lower()


def test_retrying_an_empty_chat_is_a_409(tmp_path: Path) -> None:
    client, _, _, _ = build_client(tmp_path)
    chat_id = client.post("/api/chats", json={}).json()["id"]

    response = client.post(f"/api/chats/{chat_id}/retry")

    assert response.status_code == 409
    assert response.json()["error"]["kind"] == "nothing_to_retry"


def test_retrying_a_missing_chat_is_a_404(tmp_path: Path) -> None:
    client, _, _, _ = build_client(tmp_path)

    response = client.post("/api/chats/nope/retry")

    assert response.status_code == 404
    assert response.json()["error"]["kind"] == "chat_not_found"


def test_user_content_is_returned_as_data_not_markup(tmp_path: Path) -> None:
    client, _, model, _ = build_client(tmp_path)
    chat_id = client.post("/api/chats", json={"title": HOSTILE}).json()["id"]

    body = client.post(f"/api/chats/{chat_id}/messages", json={"message": HOSTILE}).json()

    assert body["user_message"]["content"] == HOSTILE
    assert client.get(f"/api/chats/{chat_id}").json()["title"] == HOSTILE
    assert HOSTILE in model.last_prompt  # data reaches the model verbatim


# --- corpus retrieval through the mode runner ---------------------------------


def test_the_chat_corpus_provider_uses_the_shared_general_retrieval(tmp_path: Path) -> None:
    harness = build_harness()
    runner = ModeRunner(
        settings=None, service_factory=lambda *_, **__: harness.service
    )
    provider = make_corpus_provider(runner)

    response = provider("what is ai governance")

    assert response is not None
    assert len(response.results) > 0
    assert harness.retrieval.calls[-1]["mode"].value == "general"
    assert harness.retrieval.calls[-1]["final_k"] == DEFAULT_CHAT_CORPUS_FINAL_K
    assert runner.build_count == 1


def test_consulting_the_corpus_from_the_chat_reaches_the_pipeline(tmp_path: Path) -> None:
    """The whole chain: chat message -> mode runner -> RAG-009 fakes -> labelled prompt."""
    harness = build_harness()
    client, _, model, _ = build_client(tmp_path, mode_harness=harness)
    chat_id = client.post("/api/chats", json={}).json()["id"]

    body = client.post(
        f"/api/chats/{chat_id}/messages", json={"message": "What does the literature say?"}
    ).json()

    assert body["context"]["corpus_invoked"] is True
    assert body["context"]["corpus_sources"] >= 1
    prompt = model.last_prompt
    assert "CORPUS MATERIAL" in prompt
    assert "evidence (peer-reviewed or published scholarship; citable)" in prompt
    assert harness.retrieval.calls[-1]["mode"].value == "general"


def test_ordinary_chat_never_touches_retrieval(tmp_path: Path) -> None:
    harness = build_harness()
    client, _, _, _ = build_client(tmp_path, mode_harness=harness)
    chat_id = client.post("/api/chats", json={}).json()["id"]

    body = client.post(
        f"/api/chats/{chat_id}/messages", json={"message": "Give me three observations."}
    ).json()

    assert body["context"]["corpus_invoked"] is False
    assert harness.retrieval.calls == []


# --- Advanced compatibility (RAG-014, unchanged) ------------------------------


def test_the_mode_api_still_works_next_to_the_chat(tmp_path: Path) -> None:
    """§17 example 4: use Advanced after the simple chat, on the same application."""
    harness = build_harness()
    client, _, _, _ = build_client(tmp_path, mode_harness=harness)

    chat_id = client.post("/api/chats", json={}).json()["id"]
    client.post(f"/api/chats/{chat_id}/messages", json={"message": "A first chat message."})

    modes = client.get("/api/modes").json()
    assert [mode["name"] for mode in modes["modes"]] == [mode.value for mode in ModeName]
    assert modes["max_query_chars"] == MAX_QUERY_CHARS

    result = client.post(
        "/api/mode", json={"mode": "evidence", "query": "what is ai governance"}
    )
    assert result.status_code == 200
    payload = result.json()
    assert payload["mode"] == "evidence"
    assert payload["local_sources"]
    assert harness.retrieval.calls[-1]["mode"].value in {"evidence", "general", "my_work"}


def test_the_advanced_interface_and_the_chat_share_one_application(tmp_path: Path) -> None:
    harness = build_harness()
    client, _, _, _ = build_client(tmp_path, mode_harness=harness)

    paths = set(client.get("/openapi.json").json()["paths"])
    assert {"/api/modes", "/api/mode", "/api/status", "/health"} <= paths
    assert {"/api/chat/info", "/api/chats", "/api/chats/{chat_id}"} <= paths
    assert client.get("/").status_code == 200
    assert client.get("/app.js").status_code == 200


# --- the runners --------------------------------------------------------------


def test_the_chat_runner_builds_the_service_once(tmp_path: Path) -> None:
    settings = chat_settings(tmp_path)
    calls: list[dict[str, Any]] = []

    class _Service:
        def close(self) -> None:  # pragma: no cover - the runner calls it on shutdown
            pass

    def factory(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return _Service()

    runner = ChatRunner(settings=settings, service_factory=factory, corpus_provider=lambda _: None)

    first = runner.service()
    second = runner.service()

    assert first is second
    assert runner.build_count == 1
    assert calls[0]["settings"] is settings
    runner.close()
    assert runner.is_built is False


def test_the_command_line_model_applies_to_the_chat_too(tmp_path: Path) -> None:
    """``python -m app.web --model X`` must not leave the chat on a different model."""
    settings = chat_settings(tmp_path)
    mode_runner = ModeRunner(settings=settings, model="qwen3.8:27b")

    application = create_app(settings=settings, mode_runner=mode_runner)
    chat_runner = application.state.chat_runner

    assert mode_runner.model == "qwen3.8:27b"
    assert chat_runner.service().model_name == "qwen3.8:27b"
    chat_runner.close()


# --- the frontend -------------------------------------------------------------


def test_the_page_has_the_chat_shell_and_keeps_the_advanced_interface() -> None:
    page = static_files()["index.html"]
    for element in (
        'id="new-chat"',  # New chat
        'id="chat-list"',  # saved chats
        'id="conversation"',  # the messages
        'id="composer"',  # the composer
        'id="composer-input"',
        'id="attach"',  # Attach
        'id="attach-input"',
        'id="send"',  # Send
        'id="attachments"',  # attached-file chips
        'id="advanced-toggle"',  # the secondary entry
        'id="advanced-view"',  # the RAG-014 interface, kept
        "+ New chat",
        ">Rename</button>",
        ">Delete</button>",
        "Advanced options",
    ):
        assert element in page, element
    # The RAG-014 controls are all still there, behind Advanced.
    for element in (
        'id="mode"',
        'id="query"',
        'id="run"',
        'id="sources-panel"',
        'id="validation-panel"',
        'id="external-panel"',
        '<details id="advanced"',
    ):
        assert element in page, element
    assert "onclick=" not in page and "onerror=" not in page


def test_the_chat_frontend_never_interprets_data_as_markup() -> None:
    page = static_files()["index.html"]
    script = (Path(__file__).resolve().parents[1] / "app" / "web" / "static" / "chat.js").read_text(
        encoding="utf-8"
    )

    assert "chat.js" in page
    for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert forbidden not in script, forbidden
    assert "textContent" in script
    # Filenames and messages are written as text, and the mode result keeps its own renderer.
    assert "function appendMessage" in script


def test_the_chat_frontend_offers_new_chat_rename_and_delete() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "app" / "web" / "static" / "chat.js"
    ).read_text(encoding="utf-8")
    for expected in (
        '"/api/chats"',
        '"/api/chat/info"',
        "/messages",
        "/attachments",
        "/retry",
        "renderChatList",
        "renameChat",
        "deleteChat",
        "attachFiles",
        "showAdvanced",
        "retryLast",
        "Retry this turn",
    ):
        assert expected in script, expected


def test_oversized_messages_are_not_rejected_at_the_old_web_limit(tmp_path: Path) -> None:
    client, _, model, _ = build_client(tmp_path)
    chat_id = client.post("/api/chats", json={}).json()["id"]
    message = "A long paragraph about governance. " * 800  # ~29 600 characters
    assert len(message) > MAX_QUERY_CHARS

    response = client.post(f"/api/chats/{chat_id}/messages", json={"message": message})

    assert response.status_code == 200
    assert response.json()["user_message"]["content"] == message.strip()
    assert message.strip() in model.last_prompt
    # The old limit still applies to the mode API, which the repair leaves alone.
    assert MAX_QUERY_CHARS == 20_000
    mode_response = client.post("/api/mode", json={"mode": "evidence", "query": message})
    assert mode_response.status_code == 422


def test_an_attachment_can_be_posted_as_plain_base64(tmp_path: Path) -> None:
    client, _, _, _ = build_client(tmp_path)
    chat_id = client.post("/api/chats", json={}).json()["id"]

    response = client.post(
        f"/api/chats/{chat_id}/attachments",
        json={
            "filename": "note.txt",
            "content_base64": base64.b64encode(b"governance note").decode("ascii"),
        },
    )

    assert response.status_code == 201
    assert response.json()["text_status"] == "parsed"
