"""Embedding-layer models and the embedding-model interface (RAG-008).

RAG-008 embeds exactly one thing: ``Chunk.retrieval_text`` (falling back to
``Chunk.text`` only when a chunk has no retrieval text). The source text is never
rewritten, and the synthetic ``[Document: …]`` / ``[Section: …]`` labels stay
where RAG-007 put them.

This module defines *what* an embedder must expose. Concrete implementations live
next to it (:mod:`app.embedding.bge_m3`), so indexing code never calls a
sentence-transformer directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

#: Dense embedding model used for the corpus.
DEFAULT_MODEL_NAME = "BAAI/bge-m3"
#: Library the dense vectors are produced with.
DEFAULT_LIBRARY = "sentence-transformers"
#: Default number of texts handed to the model per forward pass.
DEFAULT_BATCH_SIZE = 8


class EmbeddingError(RuntimeError):
    """Base class for embedding-layer failures."""


class EmbeddingModelLoadError(EmbeddingError):
    """The embedding model could not be loaded (missing weights, bad device, ...)."""


class EmbeddingFailure(EmbeddingError):
    """A batch of texts could not be embedded."""


class EmbeddingSpec(BaseModel):
    """Exactly what produced a vector.

    The :attr:`identifier` is what the embedding cache and the index manifest key
    on, so a model, revision, dimension or normalization change invalidates cached
    vectors instead of silently mixing them.
    """

    model_config = ConfigDict(frozen=True)

    model_name: str = DEFAULT_MODEL_NAME
    model_revision: str | None = None
    dimension: int = Field(gt=0)
    normalized: bool = True
    device: str = "cpu"
    library: str = DEFAULT_LIBRARY
    library_version: str | None = None
    max_tokens: int | None = None

    @property
    def identifier(self) -> str:
        """Stable identifier for this embedder configuration (no timestamps)."""
        library = f"{self.library}{self.library_version or ''}"
        return "|".join(
            (
                self.model_name,
                f"rev={self.model_revision or 'unknown'}",
                f"dim={self.dimension}",
                f"norm={int(self.normalized)}",
                f"lib={library}",
            )
        )


@runtime_checkable
class EmbeddingModel(Protocol):
    """Minimal interface every embedder must satisfy."""

    @property
    def spec(self) -> EmbeddingSpec:
        """Identity, dimension and normalization of the loaded model."""

    @property
    def dimension(self) -> int:
        """Vector dimension reported by the loaded model."""

    @property
    def normalized(self) -> bool:
        """Whether produced vectors are L2-normalized (cosine-ready)."""

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """Embed ``texts`` in order, returning one vector per input."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""

    def count_tokens(self, text: str) -> int | None:
        """Token count per the model's own tokenizer, when available."""


def count_tokens_many(model: object, texts: Sequence[str]) -> list[int | None]:
    """Token counts for ``texts``, using the model's bulk path when it has one.

    Bulk counting matters because the BGE-M3 tokenizer is the slow Python
    implementation; embedders that only offer :meth:`count_tokens` still work.
    """
    bulk = getattr(model, "count_tokens_many", None)
    if callable(bulk):
        return list(bulk(texts))
    counter = getattr(model, "count_tokens", None)
    if not callable(counter):
        return [None] * len(texts)
    return [counter(text) for text in texts]


def as_vector_list(vector: object) -> list[float]:
    """Convert a numpy row (or any sequence) into a plain float list.

    Values are rounded through float32 because that is what the vector store keeps:
    embedding once and re-using a cached vector then produce *identical* payloads
    instead of values that differ in the last decimal place.
    """
    if hasattr(vector, "tolist"):
        values = vector.tolist()  # type: ignore[attr-defined]
    else:
        values = list(vector)  # type: ignore[arg-type]
    return [float(value) for value in np.asarray(values, dtype=np.float32)]
