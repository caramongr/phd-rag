"""RAG-008 tests 19-21: the embedding cache, and batching behaviour."""

from __future__ import annotations

from pathlib import Path

import pytest
from embedding_support import FakeEmbedder

from app.embedding.batch import EmbedRequest, embed_requests, embed_stream, iter_batches
from app.embedding.cache import EmbeddingCache, cache_key, text_hash, unit_norm


def _cache(tmp_path: Path, identifier: str = "modelA") -> EmbeddingCache:
    return EmbeddingCache(tmp_path / "cache.npz", model_identifier=identifier)


def test_cache_hit_avoids_re_embedding(tmp_path: Path) -> None:
    model = FakeEmbedder()
    requests = [EmbedRequest(key="chunk-1", text="first text"), EmbedRequest(key="chunk-2", text="second text")]

    cache = _cache(tmp_path)
    first = embed_requests(model, requests, batch_size=2, cache=cache)
    cache.save()
    calls_after_first = model.embed_calls

    reloaded = _cache(tmp_path)
    second = embed_requests(model, requests, batch_size=2, cache=reloaded)

    assert calls_after_first == 1
    assert model.embed_calls == calls_after_first
    assert [outcome.from_cache for outcome in second] == [True, True]
    assert [outcome.vector for outcome in second] == [outcome.vector for outcome in first]
    assert reloaded.stats["hits"] == 2
    assert reloaded.stats["misses"] == 0


def test_changed_retrieval_text_invalidates_cache(tmp_path: Path) -> None:
    model = FakeEmbedder()
    cache = _cache(tmp_path)

    embed_requests(model, [EmbedRequest(key="chunk-1", text="original")], batch_size=1, cache=cache)
    calls = model.embed_calls
    misses = cache.misses
    outcome = embed_requests(
        model, [EmbedRequest(key="chunk-1", text="edited retrieval text")], batch_size=1, cache=cache
    )[0]

    assert outcome.from_cache is False
    assert model.embed_calls == calls + 1
    assert cache.misses == misses + 1


def test_changed_model_id_invalidates_cache(tmp_path: Path) -> None:
    model = FakeEmbedder()
    first = _cache(tmp_path, identifier="modelA")
    embed_requests(model, [EmbedRequest(key="chunk-1", text="same text")], batch_size=1, cache=first)
    first.save()
    calls = model.embed_calls

    other_model_cache = _cache(tmp_path, identifier="modelB")
    outcome = embed_requests(
        model, [EmbedRequest(key="chunk-1", text="same text")], batch_size=1, cache=other_model_cache
    )[0]

    assert outcome.from_cache is False
    assert model.embed_calls == calls + 1


def test_cache_keys_include_model_chunk_and_text_hash() -> None:
    key = cache_key("modelA", "chunk-1", text_hash("text"))

    assert key.startswith("modelA")
    assert "chunk-1" in key
    assert text_hash("text") in key
    assert text_hash("text") != text_hash("other text")


def test_corrupt_cache_file_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "cache.npz"
    path.write_bytes(b"not an npz file")

    cache = _cache(tmp_path)

    assert cache.get("chunk-1", "text") is None
    assert cache.size == 0


def test_iter_batches_streams_without_copying_everything() -> None:
    batches = list(iter_batches(range(5), 2))

    assert batches == [[0, 1], [2, 3], [4]]
    with pytest.raises(ValueError):
        list(iter_batches(range(3), 0))


def test_embed_stream_yields_batches_in_order(tmp_path: Path) -> None:
    model = FakeEmbedder()
    requests = [EmbedRequest(key=f"chunk-{index}", text=f"text {index}") for index in range(5)]

    batches = list(embed_stream(model, requests, batch_size=2))

    assert [len(batch) for batch in batches] == [2, 2, 1]
    keys = [outcome.key for batch in batches for outcome in batch]
    assert keys == [f"chunk-{index}" for index in range(5)]
    assert all(unit_norm(outcome.vector or []) == pytest.approx(1.0, abs=1e-6) for batch in batches for outcome in batch)


def test_single_bad_chunk_does_not_lose_the_batch(tmp_path: Path) -> None:
    model = FakeEmbedder(fail_for=["poison"])
    requests = [
        EmbedRequest(key="good-1", text="fine"),
        EmbedRequest(key="bad", text="poison"),
        EmbedRequest(key="good-2", text="also fine"),
    ]

    outcomes = embed_requests(model, requests, batch_size=3)

    assert {outcome.key: outcome.ok for outcome in outcomes} == {
        "good-1": True,
        "bad": False,
        "good-2": True,
    }
    assert outcomes[1].error is not None
