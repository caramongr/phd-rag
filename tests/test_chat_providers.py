"""RAG-016: manual provider selection in the simple chat.

The whole stage is small, so the tests are deliberately few and each one checks a stated
requirement rather than a code path: an existing chat still loads, a turn goes to the model that
was selected for it, the metadata survives reopening, a failed turn can be answered by the other
provider without duplicating the user's message, and nothing ever falls back to the cloud on its
own.

Every test is offline. Ollama is the project's existing ``FakeGenerationModel`` and DeepSeek is
``FakeCloudModel``: which double received the request *is* the assertion about which provider
answered, so "no automatic failover" is checkable rather than assumed.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from app.chat.models import ProviderNotConfiguredError, UnsupportedProviderError
from app.chat.store import BACKUP_PREFIX, ChatStore
from app.generation.models import ModelFailure
from tests.chat_support import (
    FAKE_DEEPSEEK_KEY,
    FakeCloudModel,
    ScriptedModel,
    answer_result,
    build_chat,
    build_client,
    failure_result,
    repeat_abort_failure,
)

#: A local failure that is *not* Ollama's repeat guard, so the local retry never fires and the
#: turn is simply unanswered — which is the state a manual fallback recovers from.
LOCAL_OUTAGE = ModelFailure(
    kind="connection_error",
    message="could not reach the model server at http://127.0.0.1:11434",
    retryable=True,
)


def legacy_database(path: Path) -> None:
    """Write a pre-RAG-016 chat database: the old ``messages`` shape, with real content."""
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE chats (
            id          TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );
        CREATE TABLE messages (
            id              TEXT PRIMARY KEY,
            chat_id         TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
            role            TEXT NOT NULL,
            content         TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            ordering        INTEGER NOT NULL,
            context_report  TEXT
        );
        INSERT INTO chats VALUES
            ('chat-legacy', 'Old chat', '2026-01-01T00:00:00.000000+00:00',
             '2026-01-01T00:00:01.000000+00:00');
        INSERT INTO messages VALUES
            ('m1', 'chat-legacy', 'user', 'An old question', '2026-01-01T00:00:00.000000+00:00',
             1, NULL),
            ('m2', 'chat-legacy', 'assistant', 'An old answer', '2026-01-01T00:00:01.000000+00:00',
             2, '{"num_ctx": 262144, "total_tokens": 42}');
        """
    )
    connection.commit()
    connection.close()


def backups_of(path: Path) -> list[Path]:
    """The timestamped backups taken next to a database."""
    directory = path.parent / "backups"
    return sorted(directory.glob(f"{BACKUP_PREFIX}-*.sqlite3")) if directory.exists() else []


# --- 1. existing chats still load, and the migration is additive --------------


def test_legacy_chat_loads_after_an_additive_migration(tmp_path: Path) -> None:
    """A RAG-015 database opens unchanged, and a backup is taken before it is altered (§2)."""
    database = tmp_path / "chat.sqlite3"
    legacy_database(database)

    store = ChatStore(database)
    try:
        migration = store.migration
        assert migration.applied is True
        assert set(migration.columns) == {"provider", "model"}
        # The copy is a real database holding the pre-migration content, taken before the ALTER.
        assert migration.backup_path is not None
        backup = Path(migration.backup_path)
        assert backup.exists()
        assert backups_of(database) == [backup]
        copied = sqlite3.connect(backup)
        try:
            assert copied.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
        finally:
            copied.close()

        chat = store.get_chat("chat-legacy")
        assert [message.content for message in chat.messages] == [
            "An old question",
            "An old answer",
        ]
        # A message written before this stage has no provider, and keeps loading that way.
        assert [message.provider for message in chat.messages] == [None, None]
        assert [message.model for message in chat.messages] == [None, None]
        # Nothing else about the stored turn changed.
        assert chat.messages[1].context_report == {"num_ctx": 262144, "total_tokens": 42}
        assert chat.title == "Old chat"
    finally:
        store.close()

    # Reopening an already-migrated database changes nothing and copies nothing.
    again = ChatStore(database)
    try:
        assert again.migration.applied is False
        assert backups_of(database) == [backup]
    finally:
        again.close()


def test_fresh_database_needs_no_migration_or_backup(tmp_path: Path) -> None:
    """A new database is created current: no migration, no backup, no file to lose."""
    database = tmp_path / "chat.sqlite3"
    store = ChatStore(database)
    try:
        assert store.migration.applied is False
        assert store.migration.backup_path is None
        assert backups_of(database) == []
    finally:
        store.close()


# --- 2, 5. the local model is still the default and still reaches Ollama ------


def test_local_turn_goes_to_ollama_by_default(tmp_path: Path) -> None:
    """Nothing about a turn without a provider changed: it is the local model (RAG-015, §7)."""
    cloud = FakeCloudModel()
    client, _, scripted, _ = build_client(tmp_path, cloud_model=cloud)
    chat = client.post("/api/chats", json={}).json()

    answer = client.post(
        f"/api/chats/{chat['id']}/messages", json={"message": "A plain question"}
    ).json()

    assert len(scripted.calls) == 1
    assert cloud.calls == []
    assert answer["provider"] == "ollama"
    assert answer["assistant_message"]["provider"] == "ollama"
    assert answer["assistant_message"]["model"] == "fake:qwen"
    # The context report records which provider answered and what it reported (§19).
    assert answer["context"]["provider"] == "ollama"
    assert answer["context"]["input_tokens"] == 100
    assert answer["context"]["output_tokens"] == 20
    assert answer["context"]["cached_input_tokens"] is None
    assert answer["context"]["reasoning_tokens"] is None


# --- 4. selecting DeepSeek sends the request to DeepSeek ----------------------


def test_cloud_turn_goes_to_deepseek(tmp_path: Path) -> None:
    """Selecting DeepSeek for one message routes that message — and only that one — there."""
    cloud = FakeCloudModel(content="DEEPSEEK OK", reasoning_tokens=12)
    client, _, scripted, _ = build_client(
        tmp_path, cloud_model=cloud, deepseek_api_key=FAKE_DEEPSEEK_KEY
    )
    chat = client.post("/api/chats", json={}).json()

    answer = client.post(
        f"/api/chats/{chat['id']}/messages",
        json={"message": "Reply exactly: DEEPSEEK OK", "provider": "deepseek"},
    ).json()

    assert cloud.calls, "the cloud client was not used"
    assert scripted.calls == []
    assert answer["assistant_message"]["content"] == "DEEPSEEK OK"
    assert answer["provider"] == "deepseek"
    assert answer["assistant_message"]["provider"] == "deepseek"
    # With no model named, the provider's configured default answers: deepseek-flash (FIX1).
    assert answer["assistant_message"]["model"] == "deepseek-flash"
    # Provider-reported numbers are shown as reported, and reasoning tokens are a number only
    # — no reasoning text is ever requested, read or stored (§15, §19).
    assert answer["context"]["input_tokens"] == 1200
    assert answer["context"]["cached_input_tokens"] == 900
    assert answer["context"]["reasoning_tokens"] == 12
    assert answer["context"]["request_id"] == "req-fake-1"
    # No reasoning *text* is requested, read or stored (§15): only the billed token count is.
    assert "reasoning_content" not in json.dumps(answer)
    assert "reasoning" not in json.dumps(answer["assistant_message"]["content"])


def test_cloud_turn_is_budgeted_against_the_cloud_window(tmp_path: Path) -> None:
    """The same context builder runs for a cloud turn, against the cloud window (§9).

    FIX1 raised that window to the provider's current 1M tokens and nothing else: the input
    budget of a turn is unchanged, so no request grew and a million tokens are never sent by
    default.
    """
    cloud = FakeCloudModel()
    client, _, _, settings = build_client(
        tmp_path, cloud_model=cloud, deepseek_api_key=FAKE_DEEPSEEK_KEY
    )
    chat = client.post("/api/chats", json={}).json()

    answer = client.post(
        f"/api/chats/{chat['id']}/messages",
        json={"message": "Hello", "provider": "deepseek"},
    ).json()

    assert settings.deepseek_num_ctx == 1_000_000
    assert answer["context"]["num_ctx"] == 1_000_000
    assert answer["context"]["num_ctx"] != settings.chat_num_ctx
    # The ceiling moved; the amount actually sent did not.
    assert answer["context"]["input_budget_tokens"] == settings.deepseek_context_tokens == 96_000
    assert answer["context"]["total_tokens"] < 1_000
    # The local model still asks for the full local window.
    local = client.post(
        f"/api/chats/{chat['id']}/messages", json={"message": "And again"}
    ).json()
    assert local["context"]["num_ctx"] == settings.chat_num_ctx


# --- FIX1: both DeepSeek models, chosen per message --------------------------


def test_both_deepseek_models_are_offered_with_their_display_names(tmp_path: Path) -> None:
    """The selector gets Flash and Pro, Flash first, and Flash is the default (FIX1)."""
    client, _, _, settings = build_client(
        tmp_path, cloud_model=FakeCloudModel(), deepseek_api_key=FAKE_DEEPSEEK_KEY
    )

    info = client.get("/api/chat/info").json()

    assert settings.deepseek_model == "deepseek-flash"
    cloud = info["providers"][1]
    assert cloud["group"] == "DeepSeek"
    assert cloud["models"] == [
        {"id": "deepseek-flash", "label": "DeepSeek V4.1 Flash"},
        {"id": "deepseek-v4-pro", "label": "DeepSeek V4 Pro"},
    ]
    # The local entry is named the same way, so the selector reads consistently.
    assert info["providers"][0]["group"] == "Local"
    assert info["providers"][0]["models"] == [
        {"id": "qwen3.5:35b", "label": "Qwen 3.5 35B"}
    ]
    # No model the provider does not have is invented, and the old id is gone.
    assert "deepseek-chat" not in json.dumps(info)


def test_pro_can_answer_one_message_without_starting_a_new_chat(tmp_path: Path) -> None:
    """Switching Flash → Pro only changes who answers the next message (FIX1)."""
    cloud = FakeCloudModel()
    client, service, _, _ = build_client(
        tmp_path, cloud_model=cloud, deepseek_api_key=FAKE_DEEPSEEK_KEY
    )
    chat = client.post("/api/chats", json={}).json()

    flash = client.post(
        f"/api/chats/{chat['id']}/messages",
        json={"message": "One", "provider": "deepseek", "model": "deepseek-flash"},
    ).json()
    pro = client.post(
        f"/api/chats/{chat['id']}/messages",
        json={"message": "Two", "provider": "deepseek", "model": "deepseek-v4-pro"},
    ).json()
    back = client.post(
        f"/api/chats/{chat['id']}/messages",
        json={"message": "Three", "provider": "deepseek", "model": "deepseek-flash"},
    ).json()

    assert [item["assistant_message"]["model"] for item in (flash, pro, back)] == [
        "deepseek-flash",
        "deepseek-v4-pro",
        "deepseek-flash",
    ]
    # Each turn really named the model it was asked for on the wire.
    assert [call["options"] is not None for call in cloud.calls] == [True, True, True]
    assert cloud.models_requested() == ["deepseek-flash", "deepseek-v4-pro", "deepseek-flash"]
    # One chat, six messages, no new conversation.
    assert len(client.get("/api/chats").json()["chats"]) == 1
    assert len(service.store.list_messages(chat["id"])) == 6
    assert [item["model"] for item in client.get(f"/api/chats/{chat['id']}").json()["messages"]] == [
        None,
        "deepseek-flash",
        None,
        "deepseek-v4-pro",
        None,
        "deepseek-flash",
    ]


def test_a_model_the_provider_does_not_offer_is_refused(tmp_path: Path) -> None:
    """An unknown model is a readable 400 before the message is stored (FIX1)."""
    client, service, _, _ = build_client(
        tmp_path, cloud_model=FakeCloudModel(), deepseek_api_key=FAKE_DEEPSEEK_KEY
    )
    chat = client.post("/api/chats", json={}).json()

    response = client.post(
        f"/api/chats/{chat['id']}/messages",
        json={"message": "Hello", "provider": "deepseek", "model": "deepseek-chat"},
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["kind"] == "unsupported_model"
    assert "deepseek-flash, deepseek-v4-pro" in error["detail"]
    assert client.get(f"/api/chats/{chat['id']}").json()["messages"] == []


def test_a_retry_can_switch_between_the_two_deepseek_models(tmp_path: Path) -> None:
    """Retry answers the existing turn with the other model, in the same chat (FIX1, §8)."""
    cloud = FakeCloudModel(
        failure=ModelFailure(kind="server_error", message="DeepSeek service is unavailable.")
    )
    client, service, _, _ = build_client(
        tmp_path, cloud_model=cloud, deepseek_api_key=FAKE_DEEPSEEK_KEY
    )
    chat = client.post("/api/chats", json={}).json()

    failed = client.post(
        f"/api/chats/{chat['id']}/messages",
        json={"message": "Anything", "provider": "deepseek", "model": "deepseek-flash"},
    ).json()
    assert failed["assistant_message"] is None

    cloud.failure = None
    cloud.content = "Pro answered instead."
    retried = client.post(
        f"/api/chats/{chat['id']}/retry",
        json={"provider": "deepseek", "model": "deepseek-v4-pro"},
    ).json()

    assert retried["assistant_message"]["model"] == "deepseek-v4-pro"
    stored = service.store.list_messages(chat["id"])
    assert [item.role.value for item in stored] == ["user", "assistant"]
    assert len(client.get("/api/chats").json()["chats"]) == 1


# --- 6. the provider can change inside one chat ------------------------------


def test_provider_changes_inside_one_chat(tmp_path: Path) -> None:
    """One conversation, three turns, three different selections, no new chats (§1, §6)."""
    cloud = FakeCloudModel(content="The phrase was BLUE SHIP.")
    client, service, scripted, _ = build_client(
        tmp_path, cloud_model=cloud, deepseek_api_key=FAKE_DEEPSEEK_KEY
    )
    chat = client.post("/api/chats", json={}).json()

    first = client.post(
        f"/api/chats/{chat['id']}/messages",
        json={"message": "Remember the phrase BLUE SHIP."},
    ).json()
    second = client.post(
        f"/api/chats/{chat['id']}/messages",
        json={"message": "What phrase did I ask you to remember?", "provider": "deepseek"},
    ).json()
    third = client.post(
        f"/api/chats/{chat['id']}/messages",
        json={"message": "Thank you", "provider": "ollama"},
    ).json()

    assert [first["provider"], second["provider"], third["provider"]] == [
        "ollama",
        "deepseek",
        "ollama",
    ]
    stored = service.store.get_chat(chat["id"])
    assert [message.provider for message in stored.messages] == [
        None,
        "ollama",
        None,
        "deepseek",
        None,
        "ollama",
    ]
    # Both providers saw the whole local conversation: the cloud turn carried the local answer
    # to the earlier message, so continuity is provider-independent (§20).
    cloud_prompt = "\n".join(
        message.content for message in cloud.calls[-1]["messages"]
    )
    assert "Remember the phrase BLUE SHIP." in cloud_prompt
    # And the chat list still shows one chat.
    assert len(client.get("/api/chats").json()["chats"]) == 1


# --- 7. provider metadata survives reopening ---------------------------------


def test_provider_metadata_survives_reopening(tmp_path: Path) -> None:
    """Reopening the chat from a new store instance returns the same provider metadata (§7)."""
    cloud = FakeCloudModel()
    client, service, _, settings = build_client(
        tmp_path, cloud_model=cloud, deepseek_api_key=FAKE_DEEPSEEK_KEY
    )
    chat = client.post("/api/chats", json={}).json()
    client.post(f"/api/chats/{chat['id']}/messages", json={"message": "Local"})
    client.post(
        f"/api/chats/{chat['id']}/messages",
        json={"message": "Cloud", "provider": "deepseek"},
    )

    reopened = ChatStore(settings.chat_db_path)
    try:
        messages = reopened.list_messages(chat["id"])
        assert [(message.role.value, message.content) for message in messages] == [
            ("user", "Local"),
            ("assistant", "Answer from the conversation."),
            ("user", "Cloud"),
            ("assistant", "Answer from the cloud."),
        ]
        assert [message.provider for message in messages] == [
            None,
            "ollama",
            None,
            "deepseek",
        ]
        assert messages[3].model == "deepseek-flash"
        # The stored context report keeps the provider facts too.
        assert messages[3].context_report["provider"] == "deepseek"
    finally:
        reopened.close()

    # The API agrees after a fresh read.
    served = client.get(f"/api/chats/{chat['id']}").json()
    assert [item["provider"] for item in served["messages"]] == [
        None,
        "ollama",
        None,
        "deepseek",
    ]


# --- 8, 9. retry with the other provider, no duplicate user message ----------


def test_retry_answers_a_failed_local_turn_with_deepseek(tmp_path: Path) -> None:
    """A turn Ollama could not answer is answered by DeepSeek, as one turn, not two (§8)."""
    cloud = FakeCloudModel(content="DEEPSEEK OK")
    local = ScriptedModel([failure_result(LOCAL_OUTAGE), answer_result("Local, later")])
    client, service, _, _ = build_client(
        tmp_path, model=local, cloud_model=cloud, deepseek_api_key=FAKE_DEEPSEEK_KEY
    )
    chat = client.post("/api/chats", json={}).json()

    failed = client.post(
        f"/api/chats/{chat['id']}/messages", json={"message": "Reply exactly: DEEPSEEK OK"}
    ).json()
    assert failed["assistant_message"] is None
    assert failed["failure"]["kind"] == "connection_error"
    assert cloud.calls == [], "a failed local turn must not fall back to the cloud"
    assert len(service.store.list_messages(chat["id"])) == 1

    retried = client.post(
        f"/api/chats/{chat['id']}/retry", json={"provider": "deepseek"}
    ).json()

    assert retried["provider"] == "deepseek"
    assert retried["assistant_message"]["content"] == "DEEPSEEK OK"
    messages = service.store.list_messages(chat["id"])
    assert [(message.role.value, message.content) for message in messages] == [
        ("user", "Reply exactly: DEEPSEEK OK"),
        ("assistant", "DEEPSEEK OK"),
    ]


def test_retry_answers_a_failed_cloud_turn_with_local(tmp_path: Path) -> None:
    """A failed cloud turn is answered by the local model, as one turn, not two (§8)."""
    cloud = FakeCloudModel(
        failure=ModelFailure(kind="rate_limit", message="DeepSeek rate limit reached.")
    )
    local = ScriptedModel([answer_result("LOCAL OK")])
    client, service, _, _ = build_client(
        tmp_path, model=local, cloud_model=cloud, deepseek_api_key=FAKE_DEEPSEEK_KEY
    )
    chat = client.post("/api/chats", json={}).json()

    failed = client.post(
        f"/api/chats/{chat['id']}/messages",
        json={"message": "Reply exactly: LOCAL OK", "provider": "deepseek"},
    ).json()
    assert failed["assistant_message"] is None
    assert failed["provider"] == "deepseek"
    assert len(service.store.list_messages(chat["id"])) == 1

    retried = client.post(
        f"/api/chats/{chat['id']}/retry", json={"provider": "ollama"}
    ).json()

    assert retried["provider"] == "ollama"
    assert retried["assistant_message"]["content"] == "LOCAL OK"
    assert [(item.role.value, item.content) for item in service.store.list_messages(chat["id"])] == [
        ("user", "Reply exactly: LOCAL OK"),
        ("assistant", "LOCAL OK"),
    ]


def test_retry_without_a_body_still_works(tmp_path: Path) -> None:
    """The pre-RAG-016 retry call is unchanged: no body, one user message, one answer."""
    local = ScriptedModel([failure_result(LOCAL_OUTAGE), answer_result("Second try")])
    client, service, _, _ = build_client(tmp_path, model=local)
    chat = client.post("/api/chats", json={}).json()
    client.post(f"/api/chats/{chat['id']}/messages", json={"message": "Anything"})

    retried = client.post(f"/api/chats/{chat['id']}/retry").json()

    assert retried["provider"] == "ollama"
    assert [(item.role.value, item.content) for item in service.store.list_messages(chat["id"])] == [
        ("user", "Anything"),
        ("assistant", "Second try"),
    ]


# --- 10. no automatic failover, in either direction ---------------------------


def test_a_failed_local_turn_is_never_sent_to_the_cloud(tmp_path: Path) -> None:
    """Ollama's repeat guard is retried locally and never escalates to a paid provider (§8)."""
    cloud = FakeCloudModel()
    local = ScriptedModel([failure_result(repeat_abort_failure())])
    client, service, _, _ = build_client(
        tmp_path, model=local, cloud_model=cloud, deepseek_api_key=FAKE_DEEPSEEK_KEY
    )
    chat = client.post("/api/chats", json={}).json()

    answer = client.post(
        f"/api/chats/{chat['id']}/messages", json={"message": "A private chapter"}
    ).json()

    assert answer["assistant_message"] is None
    assert answer["failure"]["retries"] == 1  # the one local retry is still there
    assert cloud.calls == [], "private material must never leave the machine on its own"
    assert [item.role.value for item in service.store.list_messages(chat["id"])] == ["user"]


def test_a_failed_cloud_turn_is_not_retried_automatically(tmp_path: Path) -> None:
    """A cloud failure returns to the user: one request, one message saved (§18)."""
    cloud = FakeCloudModel(failure=ModelFailure(kind="server_error", message="HTTP 503"))
    client, service, _, _ = build_client(
        tmp_path, cloud_model=cloud, deepseek_api_key=FAKE_DEEPSEEK_KEY
    )
    chat = client.post("/api/chats", json={}).json()

    answer = client.post(
        f"/api/chats/{chat['id']}/messages",
        json={"message": "Anything", "provider": "deepseek"},
    ).json()

    assert answer["assistant_message"] is None
    assert len(cloud.calls) == 1
    assert answer["failure"]["retries"] == 0
    assert len(service.store.list_messages(chat["id"])) == 1


# --- 3. the API key stays on the server ---------------------------------------


def test_the_api_key_is_never_exposed(tmp_path: Path) -> None:
    """No endpoint, answer or message body contains the configured key (§4)."""
    cloud = FakeCloudModel(
        failure=ModelFailure(kind="auth_error", message="DeepSeek authentication failed.")
    )
    client, _, _, _ = build_client(
        tmp_path, cloud_model=cloud, deepseek_api_key=FAKE_DEEPSEEK_KEY
    )
    chat = client.post("/api/chats", json={}).json()

    info = client.get("/api/chat/info")
    failed = client.post(
        f"/api/chats/{chat['id']}/messages",
        json={"message": "Hello", "provider": "deepseek"},
    )
    served = client.get(f"/api/chats/{chat['id']}")

    for response in (info, failed, served):
        assert FAKE_DEEPSEEK_KEY not in response.text
    # The failing turn still tells the user something useful, without the key.
    assert failed.json()["failure"]["kind"] == "auth_error"
    # The selector is still fully described: ids, labels, models, availability, privacy note.
    providers = info.json()["providers"]
    assert [item["id"] for item in providers] == ["ollama", "deepseek"]
    assert info.json()["default_provider"] == "ollama"
    cloud_option = providers[1]
    assert cloud_option["kind"] == "cloud"
    assert cloud_option["available"] is True
    assert "sent to DeepSeek" in cloud_option["note"]
    assert "remains on this computer" in providers[0]["note"]
    assert [item["id"] for item in cloud_option["models"]] == [
        "deepseek-flash",
        "deepseek-v4-pro",
    ]


def test_an_unconfigured_cloud_provider_is_listed_but_unavailable(tmp_path: Path) -> None:
    """Without a key, DeepSeek is visible and explainable, and cannot be selected (§4)."""
    client, _, _, _ = build_client(tmp_path)
    info = client.get("/api/chat/info").json()

    option = info["providers"][1]
    assert option["available"] is False
    # Both configured models are still described, so the page can explain what is missing.
    assert [item["id"] for item in option["models"]] == ["deepseek-flash", "deepseek-v4-pro"]
    chat = client.post("/api/chats", json={}).json()
    response = client.post(
        f"/api/chats/{chat['id']}/messages",
        json={"message": "Hello", "provider": "deepseek"},
    )
    assert response.status_code == 503
    assert response.json()["error"]["kind"] == "provider_not_configured"
    # The message the user typed is saved, so Retry can still answer it locally.
    served = client.get(f"/api/chats/{chat['id']}").json()
    assert [item["role"] for item in served["messages"]] == ["user"]


def test_an_unknown_provider_is_refused_without_storing_the_message(tmp_path: Path) -> None:
    """A provider this build does not have is a readable error, not a saved turn (§6)."""
    client, _, _, _ = build_client(tmp_path)
    chat = client.post("/api/chats", json={}).json()

    response = client.post(
        f"/api/chats/{chat['id']}/messages",
        json={"message": "Hello", "provider": "openai"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["kind"] == "unsupported_provider"
    assert client.get(f"/api/chats/{chat['id']}").json()["messages"] == []


# --- the service-level helpers themselves -------------------------------------


def test_service_refuses_an_unknown_provider(tmp_path: Path) -> None:
    service, _, _ = build_chat(tmp_path)
    with pytest.raises(UnsupportedProviderError):
        service.client("gemini")


def test_service_reports_a_missing_key_before_calling_out(tmp_path: Path) -> None:
    service, _, _ = build_chat(tmp_path)
    with pytest.raises(ProviderNotConfiguredError) as error:
        service.client("deepseek")
    assert "DEEPSEEK_API_KEY" in error.value.detail


def test_configured_model_ids_are_what_the_selector_offers(tmp_path: Path) -> None:
    """Model ids are configuration: the default first, the rest after it (FIX1)."""
    cloud = FakeCloudModel(models=("deepseek-flash", "deepseek-v4-pro"))
    client, _, _, _ = build_client(
        tmp_path,
        cloud_model=cloud,
        deepseek_api_key=FAKE_DEEPSEEK_KEY,
        deepseek_model="deepseek-v4-pro",
        deepseek_models=("deepseek-flash", "deepseek-flash"),  # a duplicate changes nothing
    )

    option = client.get("/api/chat/info").json()["providers"][1]

    assert [item["id"] for item in option["models"]] == ["deepseek-v4-pro", "deepseek-flash"]
    # The provider's own list is reported, and never decides what can be selected.
    assert option["reported_models"] == ["deepseek-flash", "deepseek-v4-pro"]
    assert cloud.listed == 1, "the provider's model list is asked for once per process"


def test_the_listing_is_reported_but_never_changes_the_selector(tmp_path: Path) -> None:
    """A failed — or surprising — listing neither hides nor adds a model (FIX1)."""
    cloud = FakeCloudModel(
        models=("deepseek-flash", "deepseek-v4-pro", "something-new"),
        available_models_error=None,
    )
    client, _, _, _ = build_client(
        tmp_path, cloud_model=cloud, deepseek_api_key=FAKE_DEEPSEEK_KEY
    )
    first = client.get("/api/chat/info").json()["providers"][1]
    assert [item["id"] for item in first["models"]] == ["deepseek-flash", "deepseek-v4-pro"]
    assert first["reported_models"] == [
        "deepseek-flash",
        "deepseek-v4-pro",
        "something-new",
    ]

    offline = FakeCloudModel(available_models_error=RuntimeError("offline"))
    second_client, _, _, _ = build_client(
        tmp_path / "offline", cloud_model=offline, deepseek_api_key=FAKE_DEEPSEEK_KEY
    )
    second = second_client.get("/api/chat/info").json()["providers"][1]

    assert [item["id"] for item in second["models"]] == ["deepseek-flash", "deepseek-v4-pro"]
    assert second["reported_models"] == []
    assert second["available"] is True


def test_corpus_material_is_retrieved_locally_for_a_cloud_turn(tmp_path: Path) -> None:
    """Retrieval stays local; only the selected excerpts are sent to the cloud (§12)."""
    from tests.chat_support import FakeCorpusRetrieval, candidate

    corpus = FakeCorpusRetrieval([candidate(text="A retrieved excerpt about governance.")])
    cloud = FakeCloudModel()
    client, _, _, _ = build_client(
        tmp_path,
        cloud_model=cloud,
        corpus=corpus,
        deepseek_api_key=FAKE_DEEPSEEK_KEY,
    )
    chat = client.post("/api/chats", json={}).json()

    answer = client.post(
        f"/api/chats/{chat['id']}/messages",
        json={
            "message": "Consult the corpus about governance",
            "provider": "deepseek",
        },
    ).json()

    assert corpus.queries, "the corpus was not searched locally"
    assert answer["context"]["corpus_invoked"] is True
    sent = "\n".join(message.content for message in cloud.calls[-1]["messages"])
    assert "A retrieved excerpt about governance." in sent
    # Only the selected excerpt is sent: the index itself is never uploaded (§10).
    assert sent.count("A retrieved excerpt about governance.") == 1


def test_attachment_text_is_sent_instead_of_the_file(tmp_path: Path) -> None:
    """An attachment contributes its parsed text, never the binary and never a file API (§11)."""
    from tests.chat_support import attachment_body, text_bytes

    cloud = FakeCloudModel()
    client, _, _, _ = build_client(
        tmp_path, cloud_model=cloud, deepseek_api_key=FAKE_DEEPSEEK_KEY
    )
    chat = client.post("/api/chats", json={}).json()
    client.post(
        f"/api/chats/{chat['id']}/attachments",
        json=attachment_body("lattice.txt", text_bytes("THE LATTICE PARAGRAPH.")),
    )

    client.post(
        f"/api/chats/{chat['id']}/messages",
        json={"message": "Rewrite paragraph 2", "provider": "deepseek"},
    )

    sent = "\n".join(message.content for message in cloud.calls[-1]["messages"])
    assert "THE LATTICE PARAGRAPH." in sent
