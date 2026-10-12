"""The one-retry recovery and the failed-turn rule (RAG-015 FIX1 §5, §6, §8).

The failure these tests model is real and reproducible against this Ollama build: its runner
can end up emitting one token forever, and its own repeat guard then aborts the prediction with
"prediction aborted, token repeat limit reached". What is asserted here:

* exactly one retry, and only for that failure;
* a successful retry yields one assistant message and does **not** add a second user message;
* a failing retry surfaces the server's own error, and stops;
* an unanswered turn can be answered again through ``retry_last`` without duplicating it;
* an already-answered turn is refused, so the UI can never quietly rewrite history.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.chat.models import ChatRole, NothingToRetryError
from app.chat.service import REPEAT_ABORT_MARKER, _is_repeat_abort
from tests.chat_support import (
    ScriptedModel,
    answer_result,
    build_chat,
    failure_result,
    persisted_messages,
    repeat_abort_failure,
)
from tests.generation_support import FakeGenerationModel, ModelFailure


def test_a_repeat_abort_is_retried_exactly_once_and_then_answers(tmp_path: Path) -> None:
    model = ScriptedModel([failure_result(), answer_result("Recovered answer.")])
    service, _, _ = build_chat(tmp_path, model=model)
    chat = service.new_chat()

    answer = service.send_message(chat.id, "hi, can you please tell me about lattice?")

    assert len(model.calls) == 2  # one attempt, one retry: never more
    assert model.unloads == 1  # the retry asked for a fresh runner
    assert answer.assistant_message is not None
    assert answer.assistant_message.content == "Recovered answer."
    assert answer.failure is None
    assert persisted_messages(service, chat.id) == [
        ("user", "hi, can you please tell me about lattice?"),
        ("assistant", "Recovered answer."),
    ]
    service.close()


def test_the_retry_is_recorded_in_the_context_report(tmp_path: Path) -> None:
    model = ScriptedModel([failure_result(), answer_result()])
    service, _, _ = build_chat(tmp_path, model=model)
    chat = service.new_chat()

    answer = service.send_message(chat.id, "A question.")

    assert answer.report is not None
    assert answer.report.generation_retries == 1
    assert REPEAT_ABORT_MARKER in answer.report.retry_reason
    # The retry is recorded with the stored turn as well, so reopening the chat shows it.
    stored = service.store.list_messages(chat.id)[-1].context_report
    assert stored is not None and stored["generation_retries"] == 1
    assert stored["retry_reason"]
    service.close()


def test_a_second_repeat_abort_is_surfaced_cleanly(tmp_path: Path) -> None:
    model = ScriptedModel([failure_result(), failure_result()])
    service, _, _ = build_chat(tmp_path, model=model)
    chat = service.new_chat()

    answer = service.send_message(chat.id, "A question.")

    assert len(model.calls) == 2  # the retry happened, and stopped there
    assert model.unloads == 1
    assert answer.assistant_message is None
    assert answer.failure is not None
    assert REPEAT_ABORT_MARKER in answer.failure["message"]
    assert answer.failure["retries"] == 1
    assert "Traceback" not in answer.failure["message"]
    # One user turn, no assistant turn: the failed turn is still answerable.
    assert persisted_messages(service, chat.id) == [("user", "A question.")]
    service.close()


@pytest.mark.parametrize(
    "failure",
    [
        ModelFailure(kind="model_not_found", message="model 'qwen3.5:35b' not found"),
        ModelFailure(kind="server_error", message="Connection refused"),
        ModelFailure(kind="invalid_request", message="the model server rejected the request"),
    ],
)
def test_unrelated_errors_are_not_retried(tmp_path: Path, failure: ModelFailure) -> None:
    model = ScriptedModel([failure_result(failure), answer_result()])
    service, _, _ = build_chat(tmp_path, model=model)
    chat = service.new_chat()

    answer = service.send_message(chat.id, "A question.")

    assert len(model.calls) == 1  # no retry for anything else
    assert model.unloads == 0
    assert answer.failure is not None
    assert answer.failure["retries"] == 0
    assert answer.report is not None and answer.report.generation_retries == 0
    service.close()


def test_the_retry_can_be_switched_off(tmp_path: Path) -> None:
    model = ScriptedModel([failure_result(), answer_result()])
    service, _, _ = build_chat(tmp_path, model=model, chat_retry_on_repeat_abort=False)
    chat = service.new_chat()

    answer = service.send_message(chat.id, "A question.")

    assert len(model.calls) == 1
    assert answer.failure is not None and answer.failure["retries"] == 0
    service.close()


def test_retrying_answers_the_existing_turn_without_duplicating_it(tmp_path: Path) -> None:
    """§6: a failed generation leaves one user turn, and retry answers that same turn."""
    model = ScriptedModel([failure_result(), failure_result(), answer_result("Answered later.")])
    service, _, _ = build_chat(tmp_path, model=model)
    chat = service.new_chat()
    first = service.send_message(chat.id, "Please comment on paragraph two.")
    assert first.failure is not None

    retried = service.retry_last(chat.id)

    assert retried.assistant_message is not None
    assert retried.assistant_message.content == "Answered later."
    assert retried.user_message.content == "Please comment on paragraph two."
    assert persisted_messages(service, chat.id) == [
        ("user", "Please comment on paragraph two."),
        ("assistant", "Answered later."),
    ]
    # Both the failed attempt and the retry were sent as a fresh generation request.
    assert len(model.calls) == 3
    service.close()


def test_retry_keeps_the_conversation_order_with_earlier_turns(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            answer_result("First answer."),
            failure_result(),  # the second turn fails
            failure_result(),  # and its automatic retry fails too
            answer_result("Second answer."),  # the user's Retry succeeds later
        ]
    )
    service, _, _ = build_chat(tmp_path, model=model)
    chat = service.new_chat()
    service.send_message(chat.id, "First question.")
    second = service.send_message(chat.id, "Second question.")
    assert second.failure is not None and second.assistant_message is None

    service.retry_last(chat.id)

    assert persisted_messages(service, chat.id) == [
        ("user", "First question."),
        ("assistant", "First answer."),
        ("user", "Second question."),
        ("assistant", "Second answer."),
    ]
    # The retry's prompt carried the conversation that existed before the retried turn.
    last_call = model.calls[-1]["messages"]
    assert [message.role for message in last_call] == ["system", "user", "assistant", "user"]
    service.close()


def test_retrying_an_answered_turn_is_refused(tmp_path: Path) -> None:
    model = ScriptedModel([answer_result()])
    service, _, _ = build_chat(tmp_path, model=model)
    chat = service.new_chat()
    service.send_message(chat.id, "A question.")

    with pytest.raises(NothingToRetryError):
        service.retry_last(chat.id)

    assert len(model.calls) == 1  # nothing was sent
    service.close()


def test_retrying_an_empty_chat_is_refused(tmp_path: Path) -> None:
    model = ScriptedModel([answer_result()])
    service, _, _ = build_chat(tmp_path, model=model)
    chat = service.new_chat()

    with pytest.raises(NothingToRetryError):
        service.retry_last(chat.id)
    service.close()


def test_retrying_a_missing_chat_is_a_not_found(tmp_path: Path) -> None:
    from app.chat.models import ChatNotFoundError

    model = ScriptedModel([answer_result()])
    service, _, _ = build_chat(tmp_path, model=model)

    with pytest.raises(ChatNotFoundError):
        service.retry_last("nope")
    service.close()


def test_the_marker_match_is_the_repeat_abort_only() -> None:
    """The retry trigger is one exact phrase, so an unrelated 500 is never retried."""
    assert _is_repeat_abort(failure_result())
    assert _is_repeat_abort(
        failure_result(
            ModelFailure(kind="server_error", message="PREDICTION ABORTED, TOKEN REPEAT LIMIT REACHED")
        )
    )
    assert not _is_repeat_abort(answer_result())
    assert not _is_repeat_abort(failure_result(ModelFailure(kind="server_error", message="boom")))


def test_a_fake_model_without_an_unload_still_retries(tmp_path: Path) -> None:
    """The recovery does not depend on the client exposing an unload (test doubles, older servers)."""
    model = FakeGenerationModel(content="")
    model.failure = repeat_abort_failure()
    service, _, _ = build_chat(tmp_path, model=model)
    chat = service.new_chat()

    answer = service.send_message(chat.id, "A question.")

    assert len(model.calls) == 2  # attempted twice
    assert answer.failure is not None  # and surfaced, because the fake always fails
    service.close()
