"""BGE-M3 dense embedding wrapper (RAG-008).

The wrapper is the only place that knows about sentence-transformers. It:

* loads ``BAAI/bge-m3`` once and reuses the local snapshot (so a second run needs
  no network at all),
* records the exact model revision (the snapshot commit) instead of guessing,
* reads the vector dimension from the loaded model instead of hard-coding 1024,
* produces L2-normalized vectors (cosine-ready) and can verify that numerically,
* falls back to CPU when CUDA is unavailable, and refuses to silently fall back to
  a *different* device or model when one was explicitly requested.

Dense vectors only: BGE-M3's sparse and ColBERT heads are deliberately unused
(RAG-009 decides whether retrieval needs them).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from app.embedding.cache import unit_norm
from app.embedding.models import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LIBRARY,
    DEFAULT_MODEL_NAME,
    EmbeddingFailure,
    EmbeddingModelLoadError,
    EmbeddingSpec,
    as_vector_list,
    count_tokens_many,
)
from app.model_loading import resolve_device as _resolve_device
from app.model_loading import resolve_snapshot as _resolve_snapshot
from app.model_loading import library_version as _library_version


def _library() -> str:
    """RAG-008's library string (kept for the embedding spec identifier)."""
    return _library_version(DEFAULT_LIBRARY) or ""


def resolve_device(requested: str | None) -> str:
    """Return the device to load on, refusing an explicit unavailable device.

    Thin embedding-flavoured wrapper around :func:`app.model_loading.resolve_device`
    so callers keep getting :class:`EmbeddingModelLoadError` on failure.
    """
    return _resolve_device(requested, error_cls=EmbeddingModelLoadError)


def resolve_snapshot(
    model_name: str,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
    cache_dir: str | Path | None = None,
) -> tuple[str, str | None]:
    """Resolve ``model_name`` to a local path plus the exact revision.

    A hub id is downloaded/re-used through the Hugging Face cache; the snapshot
    directory name is the commit hash, which is the revision recorded in the
    manifest. A local directory is used as-is (``revision=None``).
    """
    return _resolve_snapshot(
        model_name,
        revision=revision,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
        error_cls=EmbeddingModelLoadError,
    )


def _default_loader(path: str, device: str, **_: Any) -> Any:
    """Load a SentenceTransformer from ``path`` on ``device``."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(path, device=device)


def _extract_input_ids(encoded: Any) -> Any | None:
    """Pull ``input_ids`` out of a tokenizer result.

    ``BatchEncoding`` is a ``UserDict``, not a ``dict``, and a plain object is also
    possible, so neither ``isinstance(..., dict)`` nor attribute access alone is
    enough (RAG-008 learned this the hard way: the measurement silently read zero).
    """
    if encoded is None:
        return None
    try:
        return encoded["input_ids"]
    except Exception:
        return getattr(encoded, "input_ids", None)


def _has_shape(value: Any, dimensions: int) -> bool:
    """Whether ``value`` looks like a nested sequence of ``dimensions`` levels."""
    current = value
    for _ in range(dimensions - 1):
        if not hasattr(current, "__len__") or not len(current):
            return False
        current = current[0]
    return True


class BgeM3Embedder:
    """Dense BGE-M3 embedder with lazy loading and an injectable backend.

    ``model`` / ``loader`` exist for tests and for advanced callers: production code
    passes neither and gets the real sentence-transformer.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        revision: str | None = None,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize: bool = True,
        local_files_only: bool = False,
        cache_dir: str | Path | None = None,
        loader: Callable[..., Any] | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.requested_revision = revision
        self.batch_size = max(1, batch_size)
        self.normalize = normalize
        self.local_files_only = local_files_only
        self.cache_dir = cache_dir
        self._loader = loader or _default_loader
        self._model = model
        self._device = device
        self._snapshot: str | None = None
        self._revision = revision

    # --- loading --------------------------------------------------------------
    def load(self) -> None:
        """Load the model if it is not loaded yet (idempotent)."""
        if self._model is not None:
            return
        device = resolve_device(self._device)
        self._device = device
        if self.model_name and Path(self.model_name).is_dir():
            self._snapshot = self.model_name
        else:
            self._snapshot, self._revision = resolve_snapshot(
                self.model_name,
                revision=self.requested_revision,
                local_files_only=self.local_files_only,
                cache_dir=self.cache_dir,
            )
        try:
            self._model = self._loader(self._snapshot, device)
        except EmbeddingModelLoadError:
            raise
        except Exception as exc:
            raise EmbeddingModelLoadError(
                f"could not load embedding model {self.model_name!r} on {device}: {exc}"
            ) from exc
        if self._model is None:  # pragma: no cover - defensive
            raise EmbeddingModelLoadError(
                f"loader returned no model for {self.model_name!r}"
            )

    @property
    def model(self) -> Any:
        """The underlying backend model (loading it first)."""
        self.load()
        return self._model

    # --- identity -------------------------------------------------------------
    @property
    def dimension(self) -> int:
        """Vector dimension reported by the loaded model."""
        model = self.model
        getter = getattr(model, "get_embedding_dimension", None) or getattr(
            model, "get_sentence_embedding_dimension", None
        )
        if getter is None:
            raise EmbeddingModelLoadError("model does not expose an embedding dimension")
        size = getter()
        if size is None:
            raise EmbeddingModelLoadError("model does not report an embedding dimension")
        return int(size)

    @property
    def local_path(self) -> str | None:
        """Local snapshot directory the weights were loaded from."""
        return self._snapshot

    @property
    def spec(self) -> EmbeddingSpec:
        """Identity, dimension and normalization of the loaded model."""
        max_tokens = getattr(self.model, "max_seq_length", None)
        return EmbeddingSpec(
            model_name=self.model_name,
            model_revision=self._revision,
            dimension=self.dimension,
            normalized=self.normalize,
            device=self._device or resolve_device(self._device),
            library=DEFAULT_LIBRARY,
            library_version=_library() or None,
            max_tokens=int(max_tokens) if max_tokens else None,
        )

    # --- embedding ------------------------------------------------------------
    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """Embed ``texts`` in order, returning one normalized vector per input."""
        if not texts:
            return []
        model = self.model
        try:
            matrix = model.encode(
                list(texts),
                batch_size=batch_size or self.batch_size,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingFailure(f"embedding failed: {exc}") from exc
        rows = matrix if hasattr(matrix, "__len__") else [matrix]
        if len(rows) != len(texts):
            raise EmbeddingFailure(
                f"model returned {len(rows)} vectors for {len(texts)} texts"
            )
        return [as_vector_list(row) for row in rows]

    def embed_query(self, text: str) -> list[float]:
        """Embed one query string."""
        vectors = self.embed_documents([text], batch_size=1)
        return vectors[0]

    def count_tokens(self, text: str) -> int | None:
        """Token count per the model's own tokenizer (no truncation)."""
        counts = self.count_tokens_many([text])
        return counts[0] if counts else None

    def count_tokens_many(self, texts: Sequence[str]) -> list[int | None]:
        """Token counts for several texts, in one tokenizer call.

        The BGE-M3 tokenizer is the slow (Python) XLM-Roberta implementation, so
        batching matters: one call per batch avoids per-chunk Python overhead.
        """
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None or not texts:
            return [None] * len(texts)
        try:
            encoded = tokenizer(list(texts), add_special_tokens=True, truncation=False)
        except Exception:
            return [None] * len(texts)
        ids = _extract_input_ids(encoded)
        if ids is None:
            return [None] * len(texts)
        if _has_shape(ids, 2):
            return [len(row) for row in ids]
        return [len(ids)] * len(texts)

    @property
    def max_tokens(self) -> int | None:
        """Maximum input length the model supports (tokens)."""
        value = getattr(self.model, "max_seq_length", None)
        return int(value) if value else None

    def verify_normalization(self, sample: str = "normalization check") -> float:
        """Embed a sample and return its L2 norm (must be ~1.0 for cosine)."""
        vector = self.embed_query(sample)
        return unit_norm(vector)

    def warm_up(self) -> float:
        """Run one tiny encode and return how long it took (seconds)."""
        started = time.perf_counter()
        self.embed_query("warm up")
        return time.perf_counter() - started
