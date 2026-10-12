"""The web process's mode and chat runners (RAG-014, RAG-015).

A web server is not a CLI: the process lives for hours, and the expensive part of the
pipeline (BGE-M3, the cross-encoder, the Qdrant client) must be built once and reused. This
module owns exactly one :class:`~app.modes.service.ModeService`, builds it on first use, and
runs one request at a time.

Why one at a time: the service holds two local models and a local Qdrant store, and a local
Qdrant store may be opened by one client per process. Two concurrent runs would be two
concurrent model forwards on the same objects and one store shared by two callers through
different code paths. The lock is deliberate and cheap — the pipeline is compute bound, so
serialising runs costs almost nothing and removes a whole class of failures.

The simple chat (RAG-015) is a second runner over cheap, always-available resources (SQLite
and the Ollama client), but its *corpus* retrieval is not cheap and must not open a second
Qdrant store, so it borrows the mode runner's retrieval service through
:meth:`ModeRunner.retrieve` and therefore shares the same lock.

Nothing here knows about HTTP. The API layer converts requests into calls and results into
transport; the runners only own their services and their lifetimes.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Mapping

from app.config import Settings, get_settings
from app.modes.factory import build_mode_service
from app.modes.models import ModeResult

#: Builds a mode service from settings. Injectable so tests never touch a real model.
ServiceFactory = Callable[..., Any]
#: Builds a chat service from settings and a corpus provider. Injectable, for the same reason.
ChatServiceFactory = Callable[..., Any]
#: Runs one corpus retrieval for the chat: ``f(query) -> RetrievalResponse``.
CorpusProvider = Callable[[str], Any]


class ModeRunner:
    """One lazily built :class:`ModeService`, used by one run at a time."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        service_factory: ServiceFactory | None = None,
        model: str | None = None,
        rerank: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self._factory = service_factory or build_mode_service
        self._model = model
        self._rerank = rerank
        self._service: Any | None = None
        self._lock = threading.Lock()
        self._builds = 0

    # --- lifecycle ------------------------------------------------------------
    @property
    def is_built(self) -> bool:
        """Whether the pipeline has been built (status must not build it)."""
        return self._service is not None

    @property
    def model(self) -> str:
        """The model this runner uses: the command-line override, else the configured one.

        Exposed so the simple chat uses the same model: ``--model`` on the command line must
        not leave the chat answering with a different model from the modes.
        """
        return self._model or self.settings.generation_model

    @property
    def build_count(self) -> int:
        """How many times the service was built (should stay at 1)."""
        return self._builds

    def service(self) -> Any:
        """The mode service, built on first use."""
        with self._lock:
            return self._ensure_service()

    def _ensure_service(self) -> Any:
        if self._service is None:
            self._service = self._factory(
                self.settings, model=self._model, rerank=self._rerank
            )
            self._builds += 1
        return self._service

    def retrieval_service(self) -> Any | None:
        """The RAG-009 service once a run has built it, so status can reuse its client.

        ``ModeService`` exposes it as a property, so this reads it as a value when it is not
        callable — asking the service for something is not the same as asking it to build it.
        """
        service = self._service
        if service is None:
            return None
        attribute = getattr(service, "retrieval_service", None)
        if attribute is None:
            return None
        return attribute() if callable(attribute) else attribute

    def probe(self) -> bool:
        """Take the run lock briefly, so a status probe cannot collide with a run.

        A local Qdrant store is opened once per process: if a run is in flight, a probe must
        report "in use" instead of opening a second client and breaking the run.
        """
        return self._lock.acquire(blocking=False)

    def probe_done(self) -> None:
        """Release the lock taken by :meth:`probe`."""
        self._lock.release()

    def close(self) -> None:
        """Close the pipeline this runner built, if any."""
        with self._lock:
            if self._service is not None:
                close = getattr(self._service, "close", None)
                if callable(close):
                    close()
                self._service = None

    def __enter__(self) -> "ModeRunner":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # --- the one call the API makes -------------------------------------------
    def run(
        self,
        mode: str,
        query: str,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> ModeResult:
        """Run one mode for one query, serialised against other runs."""
        with self._lock:
            service = self._ensure_service()
            return service.run_mode(mode, query, **dict(options or {}))

    def retrieve(self, query: str, *, mode: Any = "evidence", **kwargs: Any) -> Any:
        """Run one RAG-009 retrieval through the shared pipeline (RAG-015 §8).

        The simple chat asks for corpus material this way, so a chat turn and a mode run share
        one embedder, one reranker and one Qdrant client — and one lock, because those objects
        are not safe to use from two code paths at once.
        """
        with self._lock:
            service = self._ensure_service()
            retrieval = getattr(service, "retrieval", None)
            if retrieval is None:  # pragma: no cover - ModeService always exposes it
                raise RuntimeError("the mode service exposes no retrieval service")
            return retrieval.retrieve(query, mode=mode, **kwargs)


class ChatRunner:
    """One lazily built :class:`~app.chat.service.ChatService` for the web process.

    The chat service itself is cheap (SQLite and an Ollama client), but its corpus retrieval is
    not: it is handed a provider that runs a retrieval through the mode runner, so the Qdrant
    store is still opened exactly once per process.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        service_factory: ChatServiceFactory | None = None,
        model: str | None = None,
        corpus_provider: CorpusProvider | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._factory = service_factory or _build_chat_service
        #: An Ollama model name (``--model``), not a client: the service builds its own client.
        self._model = model
        self._corpus_provider = corpus_provider
        self._service: Any | None = None
        self._lock = threading.Lock()
        self._builds = 0

    @property
    def model(self) -> str:
        """The model the chat answers with: the command-line override, else the configured one."""
        return self._model or self.settings.generation_model

    @property
    def is_built(self) -> bool:
        """Whether the chat service has been built."""
        return self._service is not None

    @property
    def build_count(self) -> int:
        """How many times the service was built (should stay at 1)."""
        return self._builds

    def service(self) -> Any:
        """The chat service, built on first use."""
        with self._lock:
            if self._service is None:
                self._service = self._factory(
                    settings=self.settings,
                    model_name=self._model,
                    corpus_retrieval=self._corpus_provider,
                )
                self._builds += 1
            return self._service

    def close(self) -> None:
        """Close the chat service this runner built, if any."""
        with self._lock:
            if self._service is not None:
                close = getattr(self._service, "close", None)
                if callable(close):
                    close()
                self._service = None

    def __enter__(self) -> "ChatRunner":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _build_chat_service(**kwargs: Any) -> Any:
    """The real factory, imported lazily so importing this module stays cheap."""
    from app.chat.service import build_chat_service

    return build_chat_service(**kwargs)


def make_corpus_provider(mode_runner: ModeRunner) -> CorpusProvider:
    """A chat corpus provider over the one shared retrieval pipeline (§8).

    ``GENERAL`` is the whole-corpus mode, and each excerpt is labelled with its own corpus role
    by the chat context builder, so ``my_work`` and the rules material are never presented as
    scholarly evidence.
    """

    def provider(query: str) -> Any:
        from app.chat.service import DEFAULT_CHAT_CORPUS_FINAL_K
        from app.retrieval.models import RetrievalMode

        return mode_runner.retrieve(
            query, mode=RetrievalMode.GENERAL, final_k=DEFAULT_CHAT_CORPUS_FINAL_K
        )

    return provider


__all__ = [
    "ChatRunner",
    "ChatServiceFactory",
    "CorpusProvider",
    "ModeRunner",
    "ServiceFactory",
    "make_corpus_provider",
]
