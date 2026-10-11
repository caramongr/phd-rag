"""Shared fixtures for the RAG-008 embedding/indexing tests.

Nothing here downloads BGE-M3 or contacts a Qdrant server: the embedder is a
deterministic local double and the vector store is Qdrant's local mode (or the
in-memory flavour of it), which is enough to test point identity, payload content
and synchronisation behaviour.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from qdrant_client import QdrantClient

from app.embedding.models import EmbeddingFailure, EmbeddingSpec
from app.models.chunk import Chunk

#: Dimension of the deterministic fake embedder.
FAKE_DIMENSION = 8
FAKE_MODEL = "fake/bge-m3"
FAKE_REVISION = "test-revision"


def deterministic_vector(text: str, dimension: int = FAKE_DIMENSION) -> list[float]:
    """A reproducible unit-norm vector derived from the text (float32 precision)."""
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    generator = np.random.default_rng(seed)
    vector = generator.normal(size=dimension)
    norm = float(np.linalg.norm(vector))
    if not norm:
        return [0.0] * dimension
    unit = np.asarray(vector / norm, dtype=np.float32)
    return [float(value) for value in unit]


class FakeEmbedder:
    """Deterministic embedding model double satisfying ``EmbeddingModel``."""

    def __init__(
        self,
        *,
        dimension: int = FAKE_DIMENSION,
        model_name: str = FAKE_MODEL,
        revision: str | None = FAKE_REVISION,
        normalized: bool = True,
        device: str = "cpu",
        max_tokens: int | None = 8192,
        fail_for: Sequence[str] = (),
        token_counter: Any | None = None,
    ) -> None:
        self._dimension = dimension
        self.model_name = model_name
        self.revision = revision
        self._normalized = normalized
        self._device = device
        self.max_tokens = max_tokens
        self.fail_for = set(fail_for)
        self.token_counter = token_counter
        self.embed_calls = 0
        self.embedded_texts: list[str] = []

    # --- identity -------------------------------------------------------------
    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def normalized(self) -> bool:
        return self._normalized

    @property
    def spec(self) -> EmbeddingSpec:
        return EmbeddingSpec(
            model_name=self.model_name,
            model_revision=self.revision,
            dimension=self._dimension,
            normalized=self._normalized,
            device=self._device,
            library="fake",
            library_version="0",
            max_tokens=self.max_tokens,
        )

    # --- embedding ------------------------------------------------------------
    def embed_documents(
        self, texts: Sequence[str], *, batch_size: int | None = None
    ) -> list[list[float]]:
        self.embed_calls += 1
        if len(texts) > 1 and any(text in self.fail_for for text in texts):
            raise EmbeddingFailure("simulated batch failure")
        for text in texts:
            if text in self.fail_for:
                raise EmbeddingFailure(f"simulated failure for {text!r}")
        self.embedded_texts.extend(texts)
        return [deterministic_vector(text, self._dimension) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text], batch_size=1)[0]

    def count_tokens(self, text: str) -> int | None:
        if self.token_counter is not None:
            return self.token_counter(text)
        return len(text.split())

    def count_tokens_many(self, texts: Sequence[str]) -> list[int | None]:
        return [self.count_tokens(text) for text in texts]


class FakeSentenceTransformer:
    """Minimal sentence-transformers stand-in for the BGE-M3 wrapper tests."""

    def __init__(
        self,
        *,
        dimension: int = 4,
        max_seq_length: int = 8192,
        normalize: bool = True,
        fail: bool = False,
    ) -> None:
        self.dimension = dimension
        self.max_seq_length = max_seq_length
        self.normalize = normalize
        self.fail = fail
        self.calls: list[list[str]] = []
        self.tokenizer = _FakeTokenizer()

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 8,
        normalize_embeddings: bool = False,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> Any:
        if self.fail:
            raise RuntimeError("backend exploded")
        self.calls.append(list(texts))
        rows = []
        for text in texts:
            vector = np.asarray(deterministic_vector(text, self.dimension), dtype=np.float32)
            if normalize_embeddings and self.normalize:
                vector = vector / np.linalg.norm(vector)
            rows.append(vector)
        return np.vstack(rows)


class _FakeTokenizer:
    """Tokenizes on whitespace and reports a plausible length."""

    def __call__(self, text: Any, **_: Any) -> dict[str, list[Any]]:
        if isinstance(text, (list, tuple)):
            return {
                "input_ids": [[1] + list(range(2, 2 + len(item.split()))) for item in text]
            }
        return {"input_ids": [1] + list(range(2, 2 + len(text.split())))}


class RecordingClient:
    """A Qdrant client proxy that records the calls a test cares about."""

    def __init__(self, client: QdrantClient) -> None:
        self._client = client
        self.payload_index_calls: list[tuple[str, str]] = []
        self.upsert_calls = 0
        self.delete_calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def create_payload_index(self, *, collection_name: str, field_name: str, field_schema: Any):
        self.payload_index_calls.append((collection_name, field_name))
        return self._client.create_payload_index(
            collection_name=collection_name, field_name=field_name, field_schema=field_schema
        )

    def upsert(self, **kwargs: Any):
        self.upsert_calls += 1
        return self._client.upsert(**kwargs)

    def delete(self, **kwargs: Any):
        self.delete_calls += 1
        return self._client.delete(**kwargs)


def local_client(tmp_path: Path, name: str = "qdrant") -> QdrantClient:
    """A local persistent Qdrant client rooted in ``tmp_path``."""
    path = tmp_path / name
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def in_memory_client() -> QdrantClient:
    """An isolated in-memory Qdrant client (no files, no server)."""
    return QdrantClient(location=":memory:")


def chunk_text_of(chunk: Chunk) -> str:
    """The text RAG-008 embeds for ``chunk``."""
    return chunk.retrieval_text or chunk.text
