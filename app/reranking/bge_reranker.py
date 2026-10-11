"""BGE cross-encoder reranker wrapper (RAG-009).

The wrapper is the only place that knows about ``CrossEncoder``. It:

* loads ``BAAI/bge-reranker-v2-m3`` once and reuses the local snapshot (so a
  second run needs no network at all),
* records the exact model revision (the snapshot commit) instead of guessing,
* consumes ``retrieval_text`` - the same text the dense stage embedded, labels
  included - so the two stages see the same passage,
* scores every pair once and returns the scores **in input order**, leaving
  identity and ordering decisions to the retrieval service,
* keeps the model's own ``max_length`` (8192 for this checkpoint) and reports it,
  because silent truncation would change what a score means.

It is deliberately not a general-purpose reranking framework: one model, one
score, no fusion with the dense score.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from app.model_loading import library_version, resolve_device, resolve_snapshot
from app.reranking.models import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LIBRARY,
    DEFAULT_RERANKER_MODEL,
    RerankInput,
    RerankerFailure,
    RerankerLoadError,
    RerankerSpec,
    RerankScore,
)


def _default_loader(path: str, device: str, *, max_length: int | None = None) -> Any:
    """Load a ``CrossEncoder`` from ``path`` on ``device``."""
    from sentence_transformers import CrossEncoder

    kwargs: dict[str, Any] = {}
    if max_length is not None:
        kwargs["max_length"] = max_length
    return CrossEncoder(path, device=device, **kwargs)


class BgeReranker:
    """BGE cross-encoder reranker with lazy loading and an injectable backend.

    ``model`` / ``loader`` exist for tests and for advanced callers: production
    code passes neither and gets the real cross-encoder.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_RERANKER_MODEL,
        revision: str | None = None,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int | None = None,
        local_files_only: bool = False,
        cache_dir: str | Path | None = None,
        loader: Callable[..., Any] | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.requested_revision = revision
        self.batch_size = max(1, batch_size)
        self.requested_max_length = max_length
        self.local_files_only = local_files_only
        self.cache_dir = cache_dir
        self._loader = loader or _default_loader
        self._model = model
        self._device = device
        self._snapshot: str | None = None
        self._revision = revision
        self._last_seconds: float | None = None

    # --- loading --------------------------------------------------------------
    def load(self) -> None:
        """Load the model if it is not loaded yet (idempotent)."""
        if self._model is not None:
            return
        device = resolve_device(self._device, error_cls=RerankerLoadError)
        self._device = device
        if self.model_name and Path(self.model_name).is_dir():
            self._snapshot = self.model_name
        else:
            self._snapshot, self._revision = resolve_snapshot(
                self.model_name,
                revision=self.requested_revision,
                local_files_only=self.local_files_only,
                cache_dir=self.cache_dir,
                error_cls=RerankerLoadError,
            )
        try:
            self._model = self._loader(
                self._snapshot, device, max_length=self.requested_max_length
            )
        except RerankerLoadError:
            raise
        except Exception as exc:
            raise RerankerLoadError(
                f"could not load reranker {self.model_name!r} on {device}: {exc}"
            ) from exc
        if self._model is None:  # pragma: no cover - defensive
            raise RerankerLoadError(f"loader returned no model for {self.model_name!r}")

    @property
    def model(self) -> Any:
        """The underlying backend model (loading it first)."""
        self.load()
        return self._model

    @property
    def local_path(self) -> str | None:
        """Local snapshot directory the weights were loaded from."""
        return self._snapshot

    @property
    def max_length(self) -> int | None:
        """Maximum pair length the model will score (``None`` when unknown)."""
        value = getattr(self.model, "max_length", None)
        return int(value) if value else None

    @property
    def spec(self) -> RerankerSpec:
        """Identity, device, batch size and max length of the loaded reranker."""
        self.load()
        return RerankerSpec(
            model_name=self.model_name,
            model_revision=self._revision,
            device=self._device or resolve_device(self._device),
            batch_size=self.batch_size,
            library=DEFAULT_LIBRARY,
            library_version=library_version(DEFAULT_LIBRARY),
            max_length=self.max_length,
        )

    @property
    def last_seconds(self) -> float | None:
        """Wall-clock duration of the most recent :meth:`rerank` call."""
        return self._last_seconds

    # --- scoring --------------------------------------------------------------
    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int | None = None,
    ) -> list[float]:
        """Score ``(query, passage)`` pairs in order."""
        if not pairs:
            return []
        try:
            raw = self.model.predict(
                list(pairs),
                batch_size=batch_size or self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise RerankerFailure(f"reranking failed: {exc}") from exc
        scores = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        if len(scores) != len(pairs):
            raise RerankerFailure(
                f"reranker returned {len(scores)} scores for {len(pairs)} pairs"
            )
        out: list[float] = []
        for score in scores:
            value = score
            if isinstance(value, (list, tuple)):  # a multi-label head would nest
                if len(value) != 1:
                    raise RerankerFailure(
                        f"reranker returned {len(value)} labels; RAG-009 expects one"
                    )
                value = value[0]
            out.append(float(value))
        return out

    def rerank(self, query: str, candidates: Sequence[RerankInput]) -> list[RerankScore]:
        """Score ``candidates`` against ``query``, preserving input identity/order."""
        started = time.perf_counter()
        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self.score_pairs(pairs)
        self._last_seconds = time.perf_counter() - started
        return [
            RerankScore(chunk_id=candidate.chunk_id, score=score)
            for candidate, score in zip(candidates, scores, strict=True)
        ]

    # --- diagnostics ----------------------------------------------------------
    def count_truncated(self, pairs: Sequence[tuple[str, str]]) -> int:
        """How many pairs exceed the model's ``max_length`` (i.e. get truncated).

        Used when reporting a benchmark: a reranker that silently truncates most
        passages should be visible in the report rather than assumed harmless.
        """
        limit = self.max_length
        tokenizer = getattr(self.model, "tokenizer", None)
        if limit is None or tokenizer is None:
            return 0
        truncated = 0
        for query, passage in pairs:
            try:
                encoded = tokenizer(query, passage, truncation=False)
                ids = encoded["input_ids"]
            except Exception:  # pragma: no cover - diagnostic only
                return truncated
            if len(ids) > limit:
                truncated += 1
        return truncated

    def warm_up(self) -> float:
        """Score one tiny pair and return how long it took (seconds)."""
        started = time.perf_counter()
        self.score_pairs([("warm up", "warm up")], batch_size=1)
        return time.perf_counter() - started
