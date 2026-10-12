"""Large-context behaviour for the simple chat (RAG-015 §2, §6, §15, §16).

These tests are the ones the repair exists for: the chat path asks for the full window
(262144) instead of the evidence path's 8192, it does not inherit the ~6000-token evidence
budget, a long message is not rejected at the old 20 000-character web limit, the answer keeps
its headroom, the assembled prompt never exceeds the window, and anything left out is recorded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.chat.context import (
    TRUNCATION_NOTE,
    ChatContextRequest,
    WorkingDocument,
    build_chat_context,
    truncate_text_to_tokens,
)
from app.chat.models import ChatMessage, ChatRole, MessageTooLargeError
from app.chat.service import ChatService
from app.config import (
    DEFAULT_CHAT_MAX_OUTPUT_TOKENS,
    DEFAULT_CHAT_NUM_CTX,
    DEFAULT_GENERATION_CONTEXT_TOKENS,
    DEFAULT_GENERATION_NUM_CTX,
)
from app.generation.context import estimate_tokens
from app.retrieval.models import RetrievalMode
from tests.chat_support import build_chat, candidate, chat_settings, retrieval_response
from tests.generation_support import FakeGenerationModel

SYSTEM = "You are the local research assistant."
#: The cap the simple chat used before FIX1. 8192 is no longer the default.
DEFAULT_CHAT_PREVIOUS_OUTPUT_TOKENS = 8192


def _message(role: ChatRole, content: str, ordering: int) -> ChatMessage:
    return ChatMessage(
        id=f"m{ordering}",
        chat_id="c1",
        role=role,
        content=content,
        created_at="2026-01-01T00:00:00+00:00",
        ordering=ordering,
    )


def _request(tmp_path: Path, **overrides: object) -> ChatContextRequest:
    settings = chat_settings(tmp_path)
    values: dict[str, object] = {
        "system_prompt": SYSTEM,
        "question": "What do you think about this paragraph?",
        "settings": settings,
    }
    values.update(overrides)
    return ChatContextRequest(**values)  # type: ignore[arg-type]


# --- the old limits are not inherited -----------------------------------------


def test_chat_defaults_to_the_full_context_window(tmp_path: Path) -> None:
    settings = chat_settings(tmp_path)
    assert settings.chat_num_ctx == DEFAULT_CHAT_NUM_CTX == 262144
    assert settings.chat_num_ctx != DEFAULT_GENERATION_NUM_CTX


def test_chat_does_not_inherit_the_evidence_context_budget(tmp_path: Path) -> None:
    settings = chat_settings(tmp_path)
    assert settings.chat_context_tokens != DEFAULT_GENERATION_CONTEXT_TOKENS
    # Roughly forty times the evidence budget: the window, not the top-8 chunk package.
    assert settings.chat_context_tokens > 20 * DEFAULT_GENERATION_CONTEXT_TOKENS


def test_a_message_far_beyond_the_old_web_limit_is_accepted(tmp_path: Path) -> None:
    """The old mode API stopped at 20 000 characters; a chat message is not a mode query."""
    service, model, settings = build_chat(tmp_path)
    chat = service.new_chat()
    long_message = "This paragraph needs work. " * 2000  # ~56 000 characters
    assert len(long_message) > 20_000

    answer = service.send_message(chat.id, long_message)

    assert answer.failure is None
    assert answer.user_message.content == long_message.strip()
    assert long_message.strip() in model.last_prompt
    assert len(long_message) < settings.chat_max_message_chars
    service.close()


def test_the_context_report_states_the_configured_window_and_headroom(tmp_path: Path) -> None:
    context = build_chat_context(_request(tmp_path))

    report = context.report
    assert report.num_ctx == 262144
    assert report.output_headroom_tokens == 4096 == DEFAULT_CHAT_MAX_OUTPUT_TOKENS
    assert report.input_budget_tokens <= report.num_ctx - report.output_headroom_tokens


def test_the_assembled_prompt_never_exceeds_the_window(tmp_path: Path) -> None:
    document = "\n\n".join(f"Paragraph {index} of a very long chapter." * 40 for index in range(400))
    context = build_chat_context(
        _request(
            tmp_path,
            question="Summarise this." * 500,
            documents=[WorkingDocument("a1", "huge.txt", document)],
            history=[
                _message(ChatRole.USER, "earlier question " * 200, 1),
                _message(ChatRole.ASSISTANT, "earlier answer " * 200, 2),
            ],
        )
    )

    budget = context.report.input_budget_tokens
    sent = estimate_tokens(context.messages[-1].content) + sum(
        estimate_tokens(message.content) for message in context.messages[:-1]
    )
    assert sent <= budget
    assert context.report.total_tokens <= budget
    assert budget + context.report.output_headroom_tokens <= context.report.num_ctx


def test_a_message_that_cannot_fit_is_reported_rather_than_truncated(tmp_path: Path) -> None:
    settings = chat_settings(tmp_path, chat_context_tokens=500)
    with pytest.raises(MessageTooLargeError):
        build_chat_context(
            ChatContextRequest(system_prompt=SYSTEM, question="x " * 5000, settings=settings)
        )


# --- priority and trimming ----------------------------------------------------


def test_a_whole_document_is_included_when_it_fits(tmp_path: Path) -> None:
    document = "Paragraph about governance.\n\n" * 200
    context = build_chat_context(
        _request(tmp_path, documents=[WorkingDocument("a1", "chapter.txt", document)])
    )

    assert document.strip() in context.user_prompt
    usage = context.report.attachments[0]
    assert usage.truncated is False
    assert usage.included_chars == len(document)
    assert context.report.omitted == ()


def test_trimming_keeps_the_document_start_and_records_the_omission(tmp_path: Path) -> None:
    settings = chat_settings(tmp_path, chat_context_tokens=6000)
    document = "\n\n".join(
        f"Paragraph {index}: " + ("governance and adoption " * 40) for index in range(60)
    )
    context = build_chat_context(
        ChatContextRequest(
            system_prompt=SYSTEM,
            question="Comment on this chapter.",
            documents=[WorkingDocument("a1", "chapter.txt", document)],
            settings=settings,
        )
    )

    assert context.user_prompt.startswith("WORKING DOCUMENTS")
    assert "Paragraph 0:" in context.user_prompt
    assert "Paragraph 59:" not in context.user_prompt  # cut at the end, not in the middle
    assert TRUNCATION_NOTE.format(tokens=0).split(":")[0] in context.user_prompt
    notes = [item for item in context.report.omitted if item.kind == "attachment"]
    assert notes and notes[0].reason == "context limit"
    assert notes[0].tokens > 0
    assert context.report.attachments[0].truncated is True


def test_history_has_priority_over_corpus_material(tmp_path: Path) -> None:
    settings = chat_settings(tmp_path, chat_context_tokens=8000)
    history = [
        _message(ChatRole.USER, "previous user message " * 10, 1),
        _message(ChatRole.ASSISTANT, "previous assistant message " * 10, 2),
    ]
    corpus = retrieval_response(
        [candidate(f"chunk-{index}", text="governance evidence " * 40) for index in range(12)],
        mode=RetrievalMode.GENERAL,
    )

    context = build_chat_context(
        ChatContextRequest(
            system_prompt=SYSTEM,
            question="Keep going.",
            history=history,
            corpus=corpus,
            corpus_requested=True,
            settings=settings,
        )
    )

    assert context.report.history_messages_included == 2
    assert context.report.history_messages_omitted == 0
    assert context.report.corpus_tokens > 0
    kinds = [item.kind for item in context.report.omitted]
    assert "history" not in kinds  # conversation survived
    assert "corpus" in kinds  # the tail of the corpus did not fit, and says so
    assert context.report.total_tokens <= context.report.input_budget_tokens


def test_oldest_history_is_dropped_first_and_reported(tmp_path: Path) -> None:
    settings = chat_settings(tmp_path, chat_context_tokens=1200)
    history = [
        _message(ChatRole.USER, f"old message {index} " * 40, index) for index in range(1, 8)
    ]

    context = build_chat_context(
        ChatContextRequest(
            system_prompt=SYSTEM,
            question="And then?",
            history=history,
            settings=settings,
        )
    )

    assert context.report.history_messages_included >= 1
    assert context.report.history_messages_omitted == len(history) - context.report.history_messages_included
    dropped = [item for item in context.report.omitted if item.kind == "history"]
    assert dropped and "earlier message" in dropped[0].label
    # The newest turn is always the one that survives.
    assert "old message 7" in context.messages[1].content


def test_corpus_material_is_labeled_by_role(tmp_path: Path) -> None:
    corpus = retrieval_response(
        [
            candidate("c1", text="A published finding.", corpus_role="literature"),
            candidate("c2", text="My own draft sentence.", corpus_role="my_work"),
            candidate("c3", text="A writing rule.", corpus_role="rules"),
        ],
        mode=RetrievalMode.GENERAL,
    )
    context = build_chat_context(
        _request(tmp_path, question="What does the literature say?", corpus=corpus, corpus_requested=True)
    )

    prompt = context.user_prompt
    assert "CORPUS MATERIAL" in prompt
    assert "evidence (peer-reviewed or published scholarship; citable)" in prompt
    assert "my_work (the author's own draft; NOT citable as evidence" in prompt
    assert "guidance (supervision notes, writing rules, plans; NOT citable evidence" in prompt
    assert context.report.corpus_invoked is True
    assert context.report.corpus_sources == 3


def test_no_corpus_material_appears_when_none_was_supplied(tmp_path: Path) -> None:
    """A turn that did not ask for the corpus is built without it (§8)."""
    context = build_chat_context(
        _request(tmp_path, question="Do you think governance belongs here?", corpus=None)
    )
    assert "CORPUS MATERIAL" not in context.user_prompt
    assert context.report.corpus_requested is False
    assert context.report.corpus_invoked is False
    assert context.report.corpus_sources == 0


def test_the_prompt_layout_is_system_history_documents_corpus_message(tmp_path: Path) -> None:
    context = build_chat_context(
        _request(
            tmp_path,
            question="The user's own message.",
            history=[_message(ChatRole.USER, "an earlier turn", 1)],
            documents=[WorkingDocument("a1", "chapter.txt", "A working document.")],
            corpus=retrieval_response([candidate("c1")], mode=RetrievalMode.GENERAL),
            corpus_requested=True,
        )
    )

    assert context.messages[0].role == "system"
    assert context.messages[1].role == "user" and context.messages[1].content == "an earlier turn"
    last = context.messages[-1].content
    assert last.index("WORKING DOCUMENTS") < last.index("CORPUS MATERIAL")
    assert last.rstrip().endswith("The user's own message.")


# --- truncation mechanics -----------------------------------------------------


def test_truncation_prefers_a_paragraph_boundary_and_never_cuts_in_the_middle() -> None:
    text = "\n\n".join(f"Paragraph {index} " + "word " * 200 for index in range(10))

    prefix, omitted, truncated = truncate_text_to_tokens(text, 300)

    assert truncated is True
    assert omitted > 0
    assert prefix.startswith("Paragraph 0")
    assert text.startswith(prefix)
    assert estimate_tokens(prefix) <= 300


def test_truncation_of_an_empty_document_is_empty() -> None:
    prefix, omitted, truncated = truncate_text_to_tokens("", 100)
    assert (prefix, omitted, truncated) == ("", 0, False)


def test_a_document_with_no_parsed_text_is_reported_as_such(tmp_path: Path) -> None:
    context = build_chat_context(
        _request(tmp_path, documents=[WorkingDocument("a1", "scanned.pdf", "")])
    )
    assert "scanned.pdf" not in context.user_prompt
    reasons = [item.reason for item in context.report.omitted if item.kind == "attachment"]
    assert reasons == ["no parsed text"]


# --- the model call itself ----------------------------------------------------


def test_the_model_is_asked_for_the_full_window(tmp_path: Path) -> None:
    service, model, _ = build_chat(tmp_path)
    chat = service.new_chat()

    service.send_message(chat.id, "Hello.")

    options = model.calls[-1]["options"]
    assert options.num_ctx == 262144
    assert options.num_ctx != DEFAULT_GENERATION_NUM_CTX
    assert options.max_output_tokens == 4096
    assert options.max_output_tokens != DEFAULT_CHAT_PREVIOUS_OUTPUT_TOKENS
    assert model.calls[-1]["think"] is False  # no hidden reasoning in the conversation
    service.close()


def test_the_ollama_request_itself_carries_think_false_and_the_full_window() -> None:
    """FIX1 §3: the request disables thinking; it is not discarded after generation.

    The assertion is on the JSON that reaches Ollama, not on the service's intent, so this
    cannot pass while the wire request is wrong.
    """
    recorded: list[tuple[str, dict[str, Any]]] = []

    class _Response:
        def __init__(self, payload: dict[str, Any]) -> None:
            self.status_code = 200
            self.text = ""
            self._payload = payload

        def json(self) -> dict[str, Any]:
            return self._payload

    class _Session:
        def post(self, url: str, json: dict[str, Any] | None = None, timeout: float | None = None):
            recorded.append((url, json or {}))
            if url.endswith("/api/chat"):
                return _Response(
                    {
                        "model": "qwen3.5:35b",
                        "message": {"role": "assistant", "content": "Hello."},
                        "done_reason": "stop",
                        "prompt_eval_count": 12,
                        "eval_count": 3,
                    }
                )
            return _Response({"model": "qwen3.5:35b"})

        def get(self, url: str, timeout: float | None = None):  # pragma: no cover - unused
            return _Response({})

    from app.config import Settings
    from app.generation.ollama import OllamaClient

    settings = Settings(chat_db_path=":memory:")
    client = OllamaClient(model="qwen3.5:35b", think=True, session=_Session())  # hostile default
    service = ChatService(settings=settings, model=client)
    chat = service.new_chat()

    service.send_message(chat.id, "Say hello.")

    chat_calls = [payload for url, payload in recorded if url.endswith("/api/chat")]
    assert len(chat_calls) == 1
    payload = chat_calls[-1]
    assert payload["think"] is False  # the client's own True must be overridden per call
    assert payload["options"]["num_ctx"] == 262144
    assert payload["options"]["num_predict"] == 4096
    assert payload["model"] == "qwen3.5:35b"
    assert [message["role"] for message in payload["messages"]] == ["system", "user"]
    service.close()


def test_the_service_reports_its_context_configuration(tmp_path: Path) -> None:
    service, model, settings = build_chat(tmp_path)
    assert settings.chat_num_ctx == 262144
    assert isinstance(service, ChatService)
    assert service.model_name == "fake:qwen"
    assert model.describe()["parameter_size"] == "34.7B"
    service.close()


def test_a_scripted_model_still_returns_a_usable_shape(tmp_path: Path) -> None:
    """A guard against the fake drifting away from what the service expects to read."""
    model = FakeGenerationModel(content="An answer.")
    service, _, _ = build_chat(tmp_path, model=model)
    chat = service.new_chat()
    answer = service.send_message(chat.id, "A question.")
    assert answer.assistant_message is not None
    assert answer.assistant_message.content == "An answer."
    service.close()
