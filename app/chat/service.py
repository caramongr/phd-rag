"""The simple chat service (RAG-015 §7, §8, §14; RAG-016 §7, §8, §9).

One conversation turn is: read the chat, assemble the context, ask the selected model, store the
answer. Nothing here is a retrieval mode and nothing here is automatic RAG-011: there is no
marker requirement, no reference list, and no citation validation on ordinary turns.

What it *does* reuse, deliberately, rather than reimplementing:

* :func:`app.generation.prompts.build_*`'s sibling — the chat prompt templates, loaded with
  ``app.generation.prompts.load_template`` so the version is content-addressed;
* :class:`app.generation.ollama.OllamaClient` — the project's one Ollama HTTP implementation,
  asked for the full 262144-token window instead of the evidence path's 8192;
* :func:`app.chat.context.build_chat_context` — the one context builder, for *both* providers.
  A cloud turn is budgeted against the cloud window rather than the local one, by handing the
  builder a settings copy; the assembly itself is not duplicated (RAG-016 §9);
* :class:`app.generation.ollama.GenerationOptions` and the ``GenerationModel`` protocol, so a
  test can drive the service with the same fake the generation tests use;
* the RAG-009 retrieval service, injected as a callable, when the user explicitly asks the
  chat to consult the permanent corpus. Retrieval is always local, whatever answers (RAG-016
  §12);
* :class:`app.generation.deepseek.DeepSeekClient` — the one cloud provider (RAG-016), selected
  manually for a turn and never reached automatically. There is no failover in either
  direction: a failed turn stays a single unanswered user message, and the user chooses who
  answers it next with "Retry this turn" (RAG-016 §8).

The model is called once per turn and errors are reported, never retried: a retry would write
a second, different answer for the same message. The one exception is Ollama's own token-repeat
abort, which is a local runner fault rather than a model answer (RAG-015 FIX1) and is never
extended to the cloud provider, whose requests cost money (RAG-016 §18). A user message is
always stored *before* generation, so a model failure never loses what the user typed.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Sequence

from app.chat.attachments import AttachmentStore
from app.chat.context import ChatContextRequest, WorkingDocument, build_chat_context
from app.chat.models import (
    Chat,
    ChatAnswer,
    ChatAttachment,
    ChatMessage,
    ChatRole,
    ChatSummary,
    EmptyMessageError,
    MessageTooLargeError,
    NothingToRetryError,
    OmittedContextItem,
    ProviderNotConfiguredError,
)
from app.chat.prompts import build_system_prompt, prompt_version
from app.chat.providers import (
    PROVIDER_DEEPSEEK,
    PROVIDER_LOCAL,
    ProviderOption,
    normalise_provider,
    provider_options,
    resolve_model,
)
from app.chat.store import ChatStore
from app.config import Settings, get_settings
from app.generation.ollama import ChatResult, GenerationOptions

#: Phrases that mean "look in my permanent PhD corpus" (§8). A fixed list, not a
#: classifier: the chat only reaches for the corpus when the user asks for it in words, or
#: when the explicit option is set. "consult the corpus"/"the literature" style requests are
#: the ones worth recognising, and a false negative costs nothing (the option is right there).
CORPUS_TRIGGERS: tuple[str, ...] = (
    "consult the corpus",
    "consult my corpus",
    "consult the literature",
    "consult my library",
    "check the corpus",
    "check my corpus",
    "check the literature",
    "in my corpus",
    "in the corpus",
    "in my library",
    "search the corpus",
    "search my corpus",
    "search the literature",
    "search my library",
    "the literature says",
    "the literature on",
    "does the literature",
    "what does the literature",
    "from the literature",
    "which papers",
    "what papers",
    "find papers",
    "my published work",
    "the phd corpus",
    "phd corpus",
    "my corpus",
    "my library",
    "my thesis says",
    "in my thesis",
    "my own work says",
    "supervision notes say",
    "the writing rules",
    "my writing rules",
)

#: How many retrieved chunks a chat turn may bring in when the corpus is consulted.
DEFAULT_CHAT_CORPUS_FINAL_K = 6


def wants_corpus(text: str) -> bool:
    """Whether the user's own words ask for the permanent corpus (§8)."""
    lowered = " ".join((text or "").lower().split())
    return any(trigger in lowered for trigger in CORPUS_TRIGGERS)


#: Ollama's own repeat guard, in its own words (RAG-015 FIX1 §5). The phrase is matched
#: exactly because it is the *only* failure this repair retries: a runner that emits one token
#: forever is recoverable, a missing model or a refused connection is not retried at all.
REPEAT_ABORT_MARKER = "token repeat limit reached"


def _failure_message(result: ChatResult) -> str:
    """The model's failure text, whitespace collapsed (empty when there was no failure)."""
    raw = result.failure
    if raw is None:
        return ""
    return " ".join(str(getattr(raw, "message", "") or "").split())


def _is_repeat_abort(result: ChatResult) -> bool:
    """Whether Ollama aborted this prediction with its token-repeat guard."""
    if result.failure is None:
        return False
    return REPEAT_ABORT_MARKER in _failure_message(result).lower()


class ChatService:
    """Persistent conversations over the local model."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        store: ChatStore | None = None,
        attachments: AttachmentStore | None = None,
        model: Any | None = None,
        model_name: str | None = None,
        cloud_model: Any | None = None,
        corpus_retrieval: Callable[[str], Any] | None = None,
        prompt_directory: str | Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or ChatStore(self.settings.chat_db_path)
        self.attachments = attachments or AttachmentStore(settings=self.settings)
        self._model = model
        #: ``--model`` on the command line overrides the configured model, for the chat too.
        self._model_name = model_name or self.settings.generation_model
        #: The cloud client (RAG-016), injectable so tests never reach the network.
        self._cloud_model = cloud_model
        self._cloud_discovered: tuple[str, ...] | None = None
        self._corpus = corpus_retrieval
        self._prompt_directory = str(prompt_directory) if prompt_directory else None
        # One turn at a time: a single local model, and two interleaved turns in one
        # conversation would be written to the history in an order the user never typed.
        self._lock = threading.Lock()

    # --- components -----------------------------------------------------------
    @property
    def model(self) -> Any:
        """The local generation model, built on first use."""
        if self._model is None:
            from app.generation.ollama import OllamaClient

            self._model = OllamaClient(
                base_url=self.settings.ollama_base_url,
                model=self._model_name,
                timeout=self.settings.generation_timeout,
                keep_alive=self.settings.generation_keep_alive,
                think=False,
            )
        return self._model

    @property
    def model_name(self) -> str:
        """The local model the chat answers with."""
        return str(getattr(self.model, "name", self._model_name))

    def client(self, provider: str | None = None, model: str | None = None) -> Any:
        """The generation client for one provider and model (RAG-016 §7, FIX1).

        ``model`` selects *which* model answers: the provider's default when it is ``None``, one
        of the models the selector offers otherwise. The provider decides, not this method — an
        id it does not offer is refused rather than sent on to fail at the other end.

        Raises:
            UnsupportedProviderError: when the provider is not one this build knows.
            UnsupportedModelError: when the model is not one the provider offers.
            ProviderNotConfiguredError: when the provider exists but cannot be used here.
        """
        provider_id = normalise_provider(provider)
        model_id = resolve_model(
            provider_id, self.settings, model, local_model_id=self._model_name
        )
        if provider_id == PROVIDER_LOCAL:
            return self.model
        cloud = self._cloud_provider()
        use_model = getattr(cloud, "use_model", None)
        if callable(use_model):
            use_model(model_id)
        return cloud

    def _cloud_provider(self) -> Any:
        """The cloud client, built on first use. Never changes which model it points at."""
        if self._cloud_model is None:
            if not self.settings.deepseek_api_key:
                raise ProviderNotConfiguredError(
                    PROVIDER_DEEPSEEK,
                    "DeepSeek was selected for this message, but no API key is configured on "
                    "the server. Set DEEPSEEK_API_KEY (or PHD_RAG_DEEPSEEK_API_KEY) and "
                    "restart, or choose the local model.",
                )
            from app.generation.deepseek import DeepSeekClient

            self._cloud_model = DeepSeekClient(
                api_key=self.settings.deepseek_api_key,
                base_url=self.settings.deepseek_base_url,
                model=self.settings.deepseek_model,
                timeout=self.settings.deepseek_timeout,
            )
        return self._cloud_model

    def provider_options(self) -> tuple[ProviderOption, ...]:
        """Which providers and models the composer may offer, and what each is called (§6).

        The cloud entries are the configured models, so the selector is stable and always holds
        the models this application is set up for (FIX1). Asking the provider for its own list
        is best-effort, cached for the life of the process, and only *reported*: an unreachable
        or out-of-date listing must never take a model out of the selector.
        """
        return provider_options(
            self.settings,
            local_model_id=self._model_name,
            reported=self.discovered_cloud_models(),
        )

    def discovered_cloud_models(self) -> tuple[str, ...]:
        """Model ids the cloud provider reported for this process (empty when unasked)."""
        if self._cloud_discovered is None:
            self._cloud_discovered = ()
            try:
                client = self._cloud_provider()
                available = getattr(client, "available_models", None)
                if callable(available):
                    self._cloud_discovered = tuple(available() or ())
            except Exception:  # pragma: no cover - a listing is never worth a failure
                self._cloud_discovered = ()
        return self._cloud_discovered

    @property
    def prompt_version(self) -> str:
        """The chat prompt pair's version."""
        return prompt_version(self._prompt_directory)

    @property
    def corpus_available(self) -> bool:
        """Whether this service can consult the permanent corpus at all."""
        return self._corpus is not None

    def close(self) -> None:
        """Close the store (the model client keeps no resources of its own)."""
        self.store.close()

    def __enter__(self) -> "ChatService":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # --- chats ----------------------------------------------------------------
    def new_chat(self, title: str | None = None) -> Chat:
        """Start a clean conversation."""
        return self.store.create_chat(title)

    def list_chats(self) -> tuple[ChatSummary, ...]:
        """The saved conversations, most recently used first."""
        return self.store.list_chats()

    def get_chat(self, chat_id: str) -> Chat:
        """Reopen a conversation with its messages and attachments."""
        return self.store.get_chat(chat_id)

    def rename_chat(self, chat_id: str, title: str) -> Chat:
        """Rename a conversation."""
        return self.store.rename_chat(chat_id, title)

    def delete_chat(self, chat_id: str) -> None:
        """Delete a conversation, its messages and its attachment files."""
        self.store.delete_chat(chat_id)
        self.attachments.delete_chat_files(chat_id)

    # --- attachments ----------------------------------------------------------
    def add_attachment(self, chat_id: str, filename: str, data: bytes) -> ChatAttachment:
        """Attach one working document to a chat.

        The file is copied into the chat's own attachment directory and its text is stored
        with the chat. It is never added to the permanent corpus (§5).
        """
        existing = len(self.store.list_attachments(chat_id))
        stored = self.attachments.save(chat_id, filename, data, existing=existing)
        return self.store.add_attachment(
            chat_id,
            attachment_id=stored.id,
            filename=stored.filename,
            stored_path=stored.stored_path,
            content_type=stored.content_type,
            extension=stored.extension,
            size_bytes=stored.size_bytes,
            text_status=stored.text_status,
            parsed_text=stored.text or None,
            char_count=stored.char_count,
            token_estimate=stored.token_estimate,
            parse_error=stored.parse_error,
        )

    def list_attachments(self, chat_id: str) -> tuple[ChatAttachment, ...]:
        """The documents attached to a chat."""
        return self.store.list_attachments(chat_id)

    def delete_attachment(self, chat_id: str, attachment_id: str) -> ChatAttachment | None:
        """Detach one document, removing the stored file."""
        removed = self.store.delete_attachment(chat_id, attachment_id)
        if removed is not None:
            self.attachments.delete_file(removed)
        return removed

    def attachment_text(self, chat_id: str, attachment_id: str) -> str:
        """The stored parsed text of one attachment (``""`` when there is none)."""
        return self.store.attachment_text(chat_id, attachment_id)

    # --- conversation ---------------------------------------------------------
    def send_message(
        self,
        chat_id: str,
        text: str,
        *,
        consult_corpus: bool | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> ChatAnswer:
        """Answer one user message in a chat, saving both turns.

        ``consult_corpus``: ``True`` consults the permanent corpus, ``False`` never does, and
        ``None`` (the default) consults it only when the message asks for it in words.

        ``provider`` and ``model``: who answers *this* message (RAG-016 §7, FIX1). ``None``
        means the local model and its configured id; ``"deepseek"`` with ``"deepseek-flash"`` or
        ``"deepseek-v4-pro"`` means that cloud model. Both are validated before the user's text
        is stored — a misspelled choice must not leave a stray turn behind — but a provider that
        is merely unusable is reported after storing it, exactly like a model failure, so
        nothing the user typed is lost.
        """
        message = (text or "").strip()
        if not message:
            raise EmptyMessageError("Type a message before sending.")
        limit = int(self.settings.chat_max_message_chars)
        if len(message) > limit:
            raise MessageTooLargeError(
                f"That message is {len(message):,} characters; the limit is {limit:,}. "
                "Attach it as a document instead."
            )
        provider_id = normalise_provider(provider)
        model_id = resolve_model(
            provider_id, self.settings, model, local_model_id=self._model_name
        )

        with self._lock:
            chat = self.store.get_chat(chat_id)
            # The user's own words are saved first: a model failure must not lose them, and a
            # failure must leave exactly one user turn so it can be answered again (§6, §8).
            user_message = self.store.add_message(chat_id, ChatRole.USER, message)
            self.store.name_chat_from_message(
                chat_id, message, limit=int(self.settings.chat_title_chars)
            )
            history = chat.messages
            return self._answer(
                chat,
                user_message,
                message,
                history,
                consult_corpus=consult_corpus,
                provider=provider_id,
                model=model_id,
            )

    def retry_last(
        self, chat_id: str, *, provider: str | None = None, model: str | None = None
    ) -> ChatAnswer:
        """Answer the last user turn again, without adding a second copy of it (§6, FIX1).

        This is the clean recovery for a failed turn: the conversation keeps exactly one user
        message, and the assistant answer is written for that existing turn. ``provider`` and
        ``model`` may name a *different* model from the one that failed, which is the whole point
        of the manual fallback (RAG-016 §8): a turn Ollama could not answer can be answered by
        DeepSeek — or by the other DeepSeek model — and vice versa, with no duplicate user
        message and no new chat.

        Raises:
            NothingToRetryError: when the chat is empty or its last turn already has an answer.
        """
        provider_id = normalise_provider(provider)
        model_id = resolve_model(
            provider_id, self.settings, model, local_model_id=self._model_name
        )
        with self._lock:
            chat = self.store.get_chat(chat_id)
            if not chat.messages:
                raise NothingToRetryError(
                    "This chat has no message to answer yet. Type one and send it."
                )
            last = chat.messages[-1]
            if last.role is not ChatRole.USER:
                raise NothingToRetryError(
                    "The last message in this chat already has an answer, so there is nothing "
                    "to retry. Type a new message instead."
                )
            return self._answer(
                chat,
                last,
                last.content,
                chat.messages[:-1],
                consult_corpus=None,
                provider=provider_id,
                model=model_id,
            )

    def _answer(
        self,
        chat: Chat,
        user_message: ChatMessage,
        question: str,
        history: Sequence[ChatMessage],
        *,
        consult_corpus: bool | None,
        provider: str = PROVIDER_LOCAL,
        model: str | None = None,
    ) -> ChatAnswer:
        """Assemble the context, ask the selected model and store the answer.

        Called for a new message and for a retry of an existing one; the caller has already
        written the user message, so this never touches it.
        """
        chat_id = chat.id
        # Raises before any cloud request, so a missing key is a readable sentence rather than
        # a network error. The user message is already stored, so Retry can still answer it.
        client = self.client(provider, model)
        context_settings = self._context_settings(provider)
        corpus_requested = (
            wants_corpus(question) if consult_corpus is None else bool(consult_corpus)
        )
        corpus, retrieval_note = self._retrieve(question, corpus_requested)

        documents = tuple(
            WorkingDocument(
                attachment_id=attachment.id,
                filename=attachment.filename,
                text=self.store.attachment_text(chat_id, attachment.id),
            )
            for attachment in chat.attachments
        )
        context = build_chat_context(
            ChatContextRequest(
                system_prompt=build_system_prompt(self._prompt_directory),
                question=question,
                history=history,
                documents=documents,
                corpus=corpus,
                corpus_requested=corpus_requested,
                settings=context_settings,
                prompt_directory=self._prompt_directory,
            )
        )
        report = context.report
        if retrieval_note is not None:
            report = report.model_copy(update={"omitted": report.omitted + (retrieval_note,)})

        # The request itself disables thinking (§3): the flag is on the wire, not filtered out
        # of the answer afterwards. Use (2) below is the same request repeated once.
        result = self._model_chat(client, context.messages, settings=context_settings)
        retries = 0
        retry_reason = ""
        # Ollama's repeat guard is a local runner fault, and only the local runner is retried
        # (RAG-015 FIX1). A cloud failure is never retried automatically: every cloud request is
        # billed, and the user's own "Retry this turn" is the retry (RAG-016 §18).
        if (
            provider == PROVIDER_LOCAL
            and _is_repeat_abort(result)
            and bool(self.settings.chat_retry_on_repeat_abort)
        ):
            retries = 1
            retry_reason = _failure_message(result)
            # One fresh runner, then the same turn once more. Nothing in the conversation
            # changes, so a successful retry yields exactly one assistant message.
            self._reset_runner(client)
            result = self._model_chat(client, context.messages, settings=context_settings)
        report = report.model_copy(
            update=self._provider_metadata(result, provider, retries, retry_reason)
        )

        if result.failure is not None or not result.content.strip():
            raw = result.failure
            failure = {
                "kind": getattr(raw, "kind", "empty_response"),
                "message": getattr(raw, "message", "the model returned no text"),
                "retryable": bool(getattr(raw, "retryable", False)),
                "retries": retries,
                "retry_reason": retry_reason,
                "provider": provider,
                "detail": getattr(raw, "detail", ""),
            }
            return ChatAnswer(
                chat_id=chat_id,
                user_message=user_message,
                report=report,
                failure=failure,
                provider=provider,
                model=result.model,
                wall_seconds=result.wall_seconds,
                prompt_tokens=result.prompt_tokens,
            )

        # Thinking output, when a model produces it, is deliberately dropped: it is not part
        # of the conversation (§14). Provider reasoning is never even read (RAG-016 §15).
        assistant_message = self.store.add_message(
            chat_id,
            ChatRole.ASSISTANT,
            result.content.strip(),
            context_report=report.model_dump(mode="json"),
            provider=provider,
            model=result.model or getattr(client, "name", ""),
        )
        return ChatAnswer(
            chat_id=chat_id,
            user_message=user_message,
            assistant_message=assistant_message,
            report=report,
            provider=provider,
            model=result.model,
            wall_seconds=result.wall_seconds,
            prompt_tokens=result.prompt_tokens,
            generated_tokens=result.generated_tokens,
        )

    # --- internals ------------------------------------------------------------
    def _context_settings(self, provider: str) -> Settings:
        """The settings the context builder budgets against for this provider (RAG-016 §9).

        The cloud window is smaller than the local one, so the same builder is given the cloud
        numbers instead of being reimplemented for the cloud. Nothing else about the request
        changes: the same system prompt, history, attachments and corpus excerpts go out.
        """
        if provider != PROVIDER_DEEPSEEK:
            return self.settings
        return self.settings.model_copy(
            update={
                "chat_num_ctx": int(self.settings.deepseek_num_ctx),
                "chat_context_tokens": int(self.settings.deepseek_context_tokens),
                "chat_max_output_tokens": int(self.settings.deepseek_max_output_tokens),
            }
        )

    @staticmethod
    def _provider_metadata(
        result: ChatResult, provider: str, retries: int, retry_reason: str
    ) -> dict[str, Any]:
        """The provider facts one answer contributes to its context report (§19).

        Only fields the provider actually returned are filled in: an unreported token count
        stays ``None`` rather than being estimated, and nothing is ever priced here.
        """
        return {
            "provider": provider,
            "model": result.model,
            "input_tokens": result.prompt_tokens,
            "cached_input_tokens": result.cached_input_tokens,
            "output_tokens": result.generated_tokens,
            "reasoning_tokens": result.reasoning_tokens,
            "latency_seconds": round(result.wall_seconds, 3) if result.wall_seconds else None,
            "request_id": result.request_id,
            "generation_retries": retries,
            "retry_reason": retry_reason,
        }

    def _model_chat(
        self, client: Any, messages: Any, *, settings: Settings | None = None
    ) -> ChatResult:
        """One generation call with the selected provider's own window and sampling settings."""
        active = settings or self.settings
        options = GenerationOptions(
            temperature=float(self.settings.generation_temperature),
            top_p=float(self.settings.generation_top_p),
            num_ctx=int(active.chat_num_ctx),
            max_output_tokens=int(active.chat_max_output_tokens),
            seed=self.settings.generation_seed,
        )
        # ``think=False`` goes on the wire (§3): the request itself disables the reasoning
        # channel, so nothing has to be filtered out of the answer afterwards.
        return client.chat(messages, options=options, think=False)

    def _reset_runner(self, client: Any) -> None:
        """Ask Ollama to drop the model, so the retry gets a fresh runner (FIX1 §5).

        Best effort: a client that cannot unload (a test double, the cloud provider, or an older
        server) is simply skipped, and the retry is still attempted.
        """
        unload = getattr(client, "unload", None)
        if callable(unload):
            try:
                unload()
            except Exception:  # pragma: no cover - defensive: the retry still runs
                return

    def _retrieve(self, message: str, requested: bool) -> tuple[Any | None, OmittedContextItem | None]:
        """Corpus material for this turn, or the reason there is none (§8)."""
        if not requested:
            return None, None
        if self._corpus is None:
            return None, OmittedContextItem(
                kind="corpus",
                label="permanent PhD corpus",
                reason="not available",
                detail=(
                    "You asked to consult the PhD corpus, but this service has no retrieval "
                    "pipeline attached, so the answer uses only your message and your attached "
                    "documents."
                ),
            )
        try:
            return self._corpus(message), None
        except Exception as exc:
            return None, OmittedContextItem(
                kind="corpus",
                label="permanent PhD corpus",
                reason="retrieval failed",
                detail=(
                    "The corpus could not be searched for this message, so the answer uses "
                    "only your message and your attached documents. "
                    f"({type(exc).__name__}: {' '.join(str(exc).split())[:200]})"
                ),
            )

    def ensure_chat(self, chat_id: str) -> Chat:
        """Reopen a chat (raises :class:`~app.chat.models.ChatNotFoundError` when gone)."""
        return self.store.get_chat(chat_id)


def build_chat_service(
    settings: Settings | None = None,
    *,
    store: ChatStore | None = None,
    attachments: AttachmentStore | None = None,
    model: Any | None = None,
    model_name: str | None = None,
    cloud_model: Any | None = None,
    corpus_retrieval: Callable[[str], Any] | None = None,
) -> ChatService:
    """The standard chat service over local storage and the configured models."""
    settings = settings or get_settings()
    return ChatService(
        settings=settings,
        store=store,
        attachments=attachments,
        model=model,
        model_name=model_name,
        cloud_model=cloud_model,
        corpus_retrieval=corpus_retrieval,
    )


__all__ = ["CORPUS_TRIGGERS", "ChatService", "build_chat_service", "wants_corpus"]
