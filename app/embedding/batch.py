"""Streaming, cache-aware batching for embeddings (RAG-008).

The indexer must never hold all retrieval texts (or all vectors) in memory, so
embedding happens in batches that are consumed as a stream: texts go in, vectors
come out, and the caller uploads and forgets.

Failures are contained per item: if a batch fails, its texts are retried one by
one so a single pathological chunk cannot destroy the run, and whatever still
fails is reported instead of being silently dropped.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

from pydantic import BaseModel, ConfigDict

from app.embedding.cache import EmbeddingCache
from app.embedding.models import EmbeddingFailure, EmbeddingModel


class EmbedRequest(BaseModel):
    """One text to embed, identified by the chunk it belongs to."""

    model_config = ConfigDict(extra="forbid")

    key: str
    text: str


class EmbedOutcome(BaseModel):
    """The result of embedding one request (either a vector or an error)."""

    model_config = ConfigDict(extra="forbid")

    key: str
    vector: list[float] | None = None
    error: str | None = None
    from_cache: bool = False

    @property
    def ok(self) -> bool:
        """Whether a vector was produced."""
        return self.vector is not None


def iter_batches(items: Iterable[object], size: int) -> Iterator[list]:
    """Yield ``items`` in lists of at most ``size`` (streaming, no full copy)."""
    if size < 1:
        raise ValueError("batch size must be at least 1")
    batch: list = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def embed_requests(
    model: EmbeddingModel,
    requests: Sequence[EmbedRequest],
    *,
    batch_size: int,
    cache: EmbeddingCache | None = None,
) -> list[EmbedOutcome]:
    """Embed one batch of requests, consulting and filling ``cache``.

    Order is preserved: the returned list matches ``requests`` one to one, so a
    caller can zip vectors with chunk ids without tracking indices.
    """
    outcomes: list[EmbedOutcome | None] = [None] * len(requests)
    pending: list[int] = []

    for index, request in enumerate(requests):
        cached = cache.get(request.key, request.text) if cache is not None else None
        if cached is not None:
            outcomes[index] = EmbedOutcome(key=request.key, vector=cached, from_cache=True)
        else:
            pending.append(index)

    if pending:
        texts = [requests[index].text for index in pending]
        vectors = _embed_with_fallback(model, texts, batch_size=batch_size)
        for offset, index in enumerate(pending):
            vector = vectors[offset]
            if vector is None:
                outcomes[index] = EmbedOutcome(
                    key=requests[index].key,
                    error=f"embedding failed for chunk {requests[index].key}",
                )
                continue
            if cache is not None:
                cache.put(requests[index].key, requests[index].text, vector)
            outcomes[index] = EmbedOutcome(key=requests[index].key, vector=vector)

    return [outcome for outcome in outcomes if outcome is not None]


def _embed_with_fallback(
    model: EmbeddingModel, texts: list[str], *, batch_size: int
) -> list[list[float] | None]:
    """Embed ``texts``; on a batch failure retry item by item to isolate the cause."""
    try:
        vectors = model.embed_documents(texts, batch_size=batch_size)
        if len(vectors) != len(texts):
            raise EmbeddingFailure(
                f"model returned {len(vectors)} vectors for {len(texts)} texts"
            )
        return [list(vector) for vector in vectors]
    except Exception:
        # Isolate the offending item(s) instead of losing the whole batch.
        results: list[list[float] | None] = []
        for text in texts:
            try:
                single = model.embed_documents([text], batch_size=1)
                results.append(list(single[0]) if single else None)
            except Exception:
                results.append(None)
        return results


def embed_stream(
    model: EmbeddingModel,
    requests: Iterable[EmbedRequest],
    *,
    batch_size: int,
    cache: EmbeddingCache | None = None,
) -> Iterator[list[EmbedOutcome]]:
    """Yield embedding outcomes batch by batch, never materialising all texts."""
    for batch in iter_batches(requests, batch_size):
        yield embed_requests(model, batch, batch_size=batch_size, cache=cache)
