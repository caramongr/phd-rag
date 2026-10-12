"""Chat persistence (RAG-015 §4, §16).

Create, save, reopen, rename and delete — plus the two properties that make "persistent"
mean something: a new store over the same file sees the same conversation, and one chat's
content never appears in another.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.chat.models import ChatNotFoundError, ChatRole
from app.chat.store import DEFAULT_CHAT_TITLE, ChatStore
from tests.chat_support import build_chat, chat_settings, persisted_messages


def test_new_chat_is_empty_and_listed(tmp_path: Path) -> None:
    service, _, _ = build_chat(tmp_path)
    chat = service.new_chat()

    assert chat.title == DEFAULT_CHAT_TITLE
    assert chat.messages == ()
    listed = service.list_chats()
    assert [item.id for item in listed] == [chat.id]
    assert listed[0].message_count == 0
    service.close()


def test_messages_are_saved_in_order_and_reopened(tmp_path: Path) -> None:
    service, _, _ = build_chat(tmp_path)
    chat = service.new_chat()
    service.store.add_message(chat.id, ChatRole.USER, "first question")
    service.store.add_message(chat.id, ChatRole.ASSISTANT, "first answer")
    service.store.add_message(chat.id, ChatRole.USER, "second question")

    reopened = service.get_chat(chat.id)

    assert [(m.role.value, m.content) for m in reopened.messages] == [
        ("user", "first question"),
        ("assistant", "first answer"),
        ("user", "second question"),
    ]
    assert [m.ordering for m in reopened.messages] == [1, 2, 3]
    service.close()


def test_history_survives_a_new_service_over_the_same_database(tmp_path: Path) -> None:
    """Closing the service and reopening the chat is the restart case (§17 example 3)."""
    settings = chat_settings(tmp_path)
    first = ChatStore(settings.chat_db_path)
    chat = first.create_chat("Chapter three")
    first.add_message(chat.id, ChatRole.USER, "we will work paragraph by paragraph")
    first.close()

    second = ChatStore(settings.chat_db_path)
    reopened = second.get_chat(chat.id)

    assert reopened.title == "Chapter three"
    assert [m.content for m in reopened.messages] == ["we will work paragraph by paragraph"]
    second.close()


def test_rename_changes_the_title_and_the_timestamp(tmp_path: Path) -> None:
    service, _, _ = build_chat(tmp_path)
    chat = service.new_chat()
    before = service.get_chat(chat.id).updated_at

    renamed = service.rename_chat(chat.id, "Governance chapter")

    assert renamed.title == "Governance chapter"
    assert service.get_chat(chat.id).title == "Governance chapter"
    assert service.get_chat(chat.id).updated_at >= before
    service.close()


def test_rename_refuses_an_empty_title(tmp_path: Path) -> None:
    service, _, _ = build_chat(tmp_path)
    chat = service.new_chat()
    with pytest.raises(ValueError):
        service.rename_chat(chat.id, "   ")
    service.close()


def test_a_chat_titles_itself_from_the_first_message(tmp_path: Path) -> None:
    service, _, _ = build_chat(tmp_path)
    chat = service.new_chat()

    service.send_message(chat.id, "Do you think governance belongs in this paragraph? " * 3)

    title = service.get_chat(chat.id).title
    assert title != DEFAULT_CHAT_TITLE
    assert title.startswith("Do you think governance belongs")
    assert len(title) <= 61
    service.close()


def test_an_explicit_title_is_never_overwritten(tmp_path: Path) -> None:
    service, _, _ = build_chat(tmp_path)
    chat = service.new_chat("My own title")

    service.send_message(chat.id, "A completely different message")

    assert service.get_chat(chat.id).title == "My own title"
    service.close()


def test_delete_removes_the_chat_and_its_messages(tmp_path: Path) -> None:
    service, _, _ = build_chat(tmp_path)
    chat = service.new_chat()
    service.send_message(chat.id, "a question")

    service.delete_chat(chat.id)

    assert service.list_chats() == ()
    with pytest.raises(ChatNotFoundError):
        service.get_chat(chat.id)
    service.close()


def test_unknown_chat_is_a_typed_not_found(tmp_path: Path) -> None:
    service, _, _ = build_chat(tmp_path)
    with pytest.raises(ChatNotFoundError):
        service.get_chat("missing")
    with pytest.raises(ChatNotFoundError):
        service.rename_chat("missing", "anything")
    with pytest.raises(ChatNotFoundError):
        service.delete_chat("missing")
    service.close()


def test_chats_are_listed_most_recently_used_first(tmp_path: Path) -> None:
    service, _, _ = build_chat(tmp_path)
    first = service.new_chat("First")
    second = service.new_chat("Second")

    service.send_message(first.id, "back to the first conversation")

    assert [item.id for item in service.list_chats()][0] == first.id
    assert {item.id for item in service.list_chats()} == {first.id, second.id}
    service.close()


def test_one_chats_content_never_reaches_another(tmp_path: Path) -> None:
    service, _, _ = build_chat(tmp_path)
    chat_a = service.new_chat("A")
    chat_b = service.new_chat("B")

    service.send_message(chat_a.id, "chat A secret content")

    assert persisted_messages(service, chat_b.id) == []
    assert "chat A secret content" not in str(service.get_chat(chat_b.id).messages)
    service.close()
