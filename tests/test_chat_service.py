"""Conversation behaviour of the simple chat (RAG-015 §7, §8, §9, §14, §16, §17).

Ordinary conversation: previous turns reach later ones, attached documents stay available on
follow-ups, a new chat starts empty, one chat never leaks into another, the whole thing
survives a restart, the corpus is consulted only when it is asked for, and a model failure
costs the answer but never the user's own message.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.chat.models import ChatNotFoundError, EmptyMessageError, MessageTooLargeError
from app.chat.service import ChatService, wants_corpus
from app.generation.models import ModelFailure
from tests.chat_support import (
    FakeCorpusRetrieval,
    build_chat,
    docx_bytes,
    last_messages,
    persisted_messages,
    text_bytes,
)
from tests.generation_support import FakeGenerationModel, candidate

PARAGRAPHS = (
    "Paragraph one explains the research context of the study.",
    "Paragraph two argues that governance mechanisms shape adoption decisions.",
)


def test_the_third_turn_receives_both_previous_messages(tmp_path: Path) -> None:
    """§17 example 2."""
    service, model, _ = build_chat(tmp_path)
    chat = service.new_chat()

    service.send_message(chat.id, "Here is my first question.")
    service.send_message(chat.id, "And a follow-up.")
    service.send_message(chat.id, "One more follow-up, with the earlier turns in mind.")

    messages = last_messages(model)
    roles = [role for role, _ in messages]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    assert messages[1][1] == "Here is my first question."
    assert messages[2][1] == "Answer from the conversation."
    assert messages[3][1] == "And a follow-up."
    assert messages[-1][1].rstrip().endswith("with the earlier turns in mind.")
    service.close()


def test_an_attached_document_stays_available_on_follow_up_turns(tmp_path: Path) -> None:
    service, model, _ = build_chat(tmp_path)
    chat = service.new_chat()
    service.add_attachment(chat.id, "chapter.docx", docx_bytes(PARAGRAPHS, heading="Chapter three"))

    service.send_message(chat.id, "Read this chapter first.")
    service.send_message(chat.id, "Now rewrite only the second paragraph.")

    for call in model.calls:
        last = call["messages"][-1].content
        assert "WORKING DOCUMENT 1: chapter.docx" in last
        assert "Paragraph two argues that governance" in last
    service.close()


def test_new_chat_starts_with_clean_history(tmp_path: Path) -> None:
    service, model, _ = build_chat(tmp_path)
    first = service.new_chat()
    service.send_message(first.id, "Something private about chapter five.")

    second = service.new_chat()
    service.send_message(second.id, "A completely fresh question.")

    messages = last_messages(model)
    assert len(messages) == 2  # system + this message
    assert "chapter five" not in messages[-1][1]
    service.close()


def test_one_chat_never_leaks_into_another(tmp_path: Path) -> None:
    service, model, _ = build_chat(tmp_path)
    chat_a = service.new_chat("A")
    chat_b = service.new_chat("B")
    service.add_attachment(chat_a.id, "a.txt", text_bytes("only chat A has this document"))

    service.send_message(chat_a.id, "A question with a confidential phrase.")
    service.send_message(chat_b.id, "A question in the other chat.")

    prompt_b = model.calls[-1]["messages"][-1].content
    assert "only chat A has this document" not in prompt_b
    assert "confidential phrase" not in prompt_b
    assert "confidential phrase" in model.calls[-2]["messages"][-1].content
    service.close()


def test_the_conversation_survives_a_restart(tmp_path: Path) -> None:
    """§17 example 3: a new service over the same storage reloads messages and attachments."""
    service, _, settings = build_chat(tmp_path)
    chat = service.new_chat("Chapter three")
    service.add_attachment(chat.id, "chapter.docx", docx_bytes(PARAGRAPHS, heading="Chapter three"))
    service.send_message(chat.id, "Remember this question.")
    attachment_id = service.list_attachments(chat.id)[0].id
    service.close()

    model = FakeGenerationModel(content="Still here.")
    reopened = ChatService(settings=settings, model=model)
    messages = persisted_messages(reopened, chat.id)
    assert ("user", "Remember this question.") in messages

    answer = reopened.send_message(chat.id, "Do you still have the document?")

    prompt = model.calls[-1]["messages"][-1].content
    assert "Paragraph one explains the research context" in prompt  # no re-upload needed
    assert reopened.attachment_text(chat.id, attachment_id) != ""
    assert answer.assistant_message is not None
    reopened.close()


# --- corpus behaviour (§8) ----------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "What does the literature say about governance?",
        "Search my corpus for adoption barriers.",
        "Check the corpus and compare it with my chapter.",
    ],
)
def test_an_explicit_request_triggers_corpus_retrieval(tmp_path: Path, message: str) -> None:
    corpus = FakeCorpusRetrieval([candidate("c1", text="A published finding.")])
    service, model, _ = build_chat(tmp_path, corpus=corpus)
    chat = service.new_chat()

    answer = service.send_message(chat.id, message)

    assert corpus.queries == [message]
    assert answer.report is not None and answer.report.corpus_invoked is True
    assert "CORPUS MATERIAL" in model.calls[-1]["messages"][-1].content
    service.close()


def test_ordinary_conversation_does_not_search_the_corpus(tmp_path: Path) -> None:
    corpus = FakeCorpusRetrieval([candidate("c1")])
    service, model, _ = build_chat(tmp_path, corpus=corpus)
    chat = service.new_chat()

    service.send_message(chat.id, "Do you think governance belongs in this paragraph?")
    service.send_message(chat.id, "Don't rewrite yet. Give me three observations.")

    assert corpus.queries == []
    assert "CORPUS MATERIAL" not in model.calls[-1]["messages"][-1].content
    service.close()


def test_the_option_forces_and_suppresses_corpus_retrieval(tmp_path: Path) -> None:
    corpus = FakeCorpusRetrieval([candidate("c1")])
    service, _, _ = build_chat(tmp_path, corpus=corpus)
    chat = service.new_chat()

    forced = service.send_message(chat.id, "Just a comment.", consult_corpus=True)
    suppressed = service.send_message(
        chat.id, "What does the literature say?", consult_corpus=False
    )

    assert corpus.queries == ["Just a comment."]
    assert forced.report is not None and forced.report.corpus_invoked is True
    assert suppressed.report is not None and suppressed.report.corpus_requested is False
    service.close()


def test_retrieval_failure_is_recorded_and_the_answer_still_happens(tmp_path: Path) -> None:
    corpus = FakeCorpusRetrieval(error=RuntimeError("the Qdrant store is locked"))
    service, model, _ = build_chat(tmp_path, corpus=corpus)
    chat = service.new_chat()

    answer = service.send_message(chat.id, "Search my corpus for governance.")

    assert answer.failure is None
    assert answer.assistant_message is not None
    assert "CORPUS MATERIAL" not in model.calls[-1]["messages"][-1].content
    omitted = [item for item in answer.report.omitted if item.kind == "corpus"]
    assert omitted and omitted[0].reason == "retrieval failed"
    assert "locked" in omitted[0].detail
    service.close()


def test_corpus_material_keeps_its_role_labels_in_the_prompt(tmp_path: Path) -> None:
    corpus = FakeCorpusRetrieval(
        [
            candidate("c1", text="A published finding.", corpus_role="literature"),
            candidate("c2", text="A sentence from my own draft.", corpus_role="my_work"),
        ]
    )
    service, model, _ = build_chat(tmp_path, corpus=corpus)
    chat = service.new_chat()

    service.send_message(chat.id, "Consult the corpus about governance.")

    prompt = model.calls[-1]["messages"][-1].content
    assert "evidence (peer-reviewed or published scholarship; citable)" in prompt
    assert "my_work (the author's own draft; NOT citable as evidence" in prompt
    service.close()


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("What does the literature say?", True),
        ("Search my corpus.", True),
        ("Give me three observations about this paragraph.", False),
        ("Rewrite the second paragraph only.", False),
    ],
)
def test_the_corpus_trigger_is_a_fixed_phrase_list(message: str, expected: bool) -> None:
    assert wants_corpus(message) is expected


# --- failures and limits ------------------------------------------------------


def test_a_model_failure_keeps_the_users_message(tmp_path: Path) -> None:
    failure = ModelFailure(kind="model_not_found", message="model 'qwen3.5:35b' not found")
    model = FakeGenerationModel(failure=failure)
    service, _, _ = build_chat(tmp_path, model=model)
    chat = service.new_chat()

    answer = service.send_message(chat.id, "An unanswered question.")

    assert answer.assistant_message is None
    assert answer.failure is not None and answer.failure["kind"] == "model_not_found"
    assert persisted_messages(service, chat.id) == [("user", "An unanswered question.")]
    service.close()


def test_hidden_reasoning_is_never_part_of_the_conversation(tmp_path: Path) -> None:
    model = FakeGenerationModel(content="The answer.", thinking="internal reasoning trace")
    service, _, _ = build_chat(tmp_path, model=model)
    chat = service.new_chat()

    answer = service.send_message(chat.id, "A question.")

    assert answer.assistant_message is not None
    assert answer.assistant_message.content == "The answer."
    assert "internal reasoning trace" not in str(answer.model_dump())
    service.close()


def test_an_empty_message_is_refused(tmp_path: Path) -> None:
    service, _, _ = build_chat(tmp_path)
    chat = service.new_chat()
    with pytest.raises(EmptyMessageError):
        service.send_message(chat.id, "   ")
    assert persisted_messages(service, chat.id) == []
    service.close()


def test_a_message_beyond_the_limit_is_refused_readably(tmp_path: Path) -> None:
    service, _, _ = build_chat(tmp_path, chat_max_message_chars=100)
    chat = service.new_chat()
    with pytest.raises(MessageTooLargeError) as raised:
        service.send_message(chat.id, "x" * 200)
    assert "Attach it as a document instead" in raised.value.detail
    service.close()


def test_sending_to_a_missing_chat_is_a_not_found(tmp_path: Path) -> None:
    service, _, _ = build_chat(tmp_path)
    with pytest.raises(ChatNotFoundError):
        service.send_message("nope", "hello")
    service.close()


def test_every_turn_stores_its_context_report(tmp_path: Path) -> None:
    service, _, _ = build_chat(tmp_path)
    chat = service.new_chat()

    service.send_message(chat.id, "A question with a report.")

    messages = service.store.list_messages(chat.id)
    assert messages[0].context_report is None  # a user message has no report
    report = messages[1].context_report
    assert report is not None
    assert report["num_ctx"] == 262144
    assert report["input_budget_tokens"] > 0
    service.close()


def test_conversation_history_is_sent_as_roles_not_as_one_blob(tmp_path: Path) -> None:
    service, model, _ = build_chat(tmp_path)
    chat = service.new_chat()
    service.send_message(chat.id, "First.")
    service.send_message(chat.id, "Second.")

    roles = [role for role, _ in last_messages(model)]
    assert roles.count("assistant") == 1
    assert roles.count("user") == 2
    service.close()


def test_a_mode_style_evidence_request_is_ordinary_conversation(tmp_path: Path) -> None:
    """§7: no [S1] markers, no references and no validation are forced on a chat turn."""
    service, model, _ = build_chat(tmp_path)
    chat = service.new_chat()

    answer = service.send_message(chat.id, "Explain the difference between the two approaches.")

    system = model.calls[-1]["messages"][0].content
    assert "Do not add citation markers" in system
    assert "reference list" in system
    assert answer.report is not None
    assert answer.report.corpus_invoked is False
    service.close()


def test_the_chat_system_prompt_asks_for_short_answers(tmp_path: Path) -> None:
    """Brevity is the default: generation length, not context, dominates how long a turn takes."""
    service, model, _ = build_chat(tmp_path)
    chat = service.new_chat()

    service.send_message(chat.id, "What is a lattice?")

    system = model.calls[-1]["messages"][0].content
    assert "Be brief by default" in system
    assert "Do not restate the question" in system
    service.close()


def test_deleting_a_chat_after_a_conversation_leaves_nothing_behind(tmp_path: Path) -> None:
    service, _, _ = build_chat(tmp_path)
    chat = service.new_chat()
    service.add_attachment(chat.id, "notes.txt", text_bytes("notes"))
    service.send_message(chat.id, "A question.")

    service.delete_chat(chat.id)

    assert service.list_chats() == ()
    with pytest.raises(ChatNotFoundError):
        service.get_chat(chat.id)
    service.close()
