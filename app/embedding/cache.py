"""Disk-backed embedding cache (RAG-008).

Vectors are cached per ``(model identifier, chunk id, retrieval-text hash)``, so
re-running the indexer over an unchanged corpus re-uses every vector and never
re-embeds a chunk whose retrieval text is byte-identical.

The cache is deliberately independent of Qdrant: it is what makes a rebuild cheap,
not a copy of the vector store. It is a single numpy ``.npz`` file (float32
vectors), which keeps 11k × 1024 vectors at ~45 MB instead of a
multi-hundred-megabyte JSON.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

#: Format marker, bumped when the on-disk layout changes.
CACHE_FORMAT_VERSION = "rag008-embedding-cache-v1"
_KEY_SEPARATOR = "\x1f"


def text_hash(text: str) -> str:
    """Return the stable hash of a retrieval text (cache invalidation key)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_key(model_identifier: str, chunk_id: str, text_sha: str) -> str:
    """Build the cache key; the model identifier makes model changes invalidate."""
    return _KEY_SEPARATOR.join((model_identifier, chunk_id, text_sha))


def unit_norm(vector: Any) -> float:
    """L2 norm of a vector, for the numerical normalization checks."""
    array = np.asarray(vector, dtype=np.float64)
    return float(np.linalg.norm(array))


class EmbeddingCache:
    """A local, model-aware vector cache.

    Entries for other models stay on disk but are never returned, so switching
    embedding models silently re-embeds instead of mixing vector spaces.
    """

    def __init__(self, path: str | Path, *, model_identifier: str) -> None:
        self.path = Path(path)
        self.model_identifier = model_identifier
        self.hits = 0
        self.misses = 0
        self._rows: dict[str, Any] = {}
        self._loaded = False

    # --- loading / saving -----------------------------------------------------
    def load(self) -> None:
        """Load the cache file when it exists (a missing file is an empty cache)."""
        if self._loaded:
            return
        self._loaded = True
        if not self.path.is_file():
            return
        try:
            with np.load(self.path, allow_pickle=False) as data:
                keys = [str(key) for key in data["keys"]]
                matrix = np.asarray(data["vectors"], dtype=np.float32)
        except Exception:
            # A corrupt cache must never break indexing: start from empty.
            self._rows = {}
            return
        if len(keys) != len(matrix):
            self._rows = {}
            return
        self._rows = {key: matrix[index] for index, key in enumerate(keys)}

    def save(self) -> None:
        """Persist the cache atomically (temporary file, then replace)."""
        self.load()
        if not self._rows:
            return
        keys = list(self._rows)
        matrix = np.vstack([self._rows[key] for key in keys]).astype(np.float32)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp.npz")
        np.savez(temporary, keys=np.array(keys, dtype="U"), vectors=matrix)
        temporary.replace(self.path)

    # --- access ---------------------------------------------------------------
    def get(self, chunk_id: str, text: str) -> list[float] | None:
        """Return the cached vector for ``chunk_id``/``text``, or ``None``."""
        self.load()
        row = self._rows.get(cache_key(self.model_identifier, chunk_id, text_hash(text)))
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return [float(value) for value in row]

    def put(self, chunk_id: str, text: str, vector: list[float]) -> None:
        """Store a vector for ``chunk_id``/``text`` (overwriting any older value)."""
        self.load()
        key = cache_key(self.model_identifier, chunk_id, text_hash(text))
        self._rows[key] = np.asarray(vector, dtype=np.float32)

    def contains(self, chunk_id: str, text: str) -> bool:
        """Whether a vector is cached, without counting a hit or a miss."""
        self.load()
        return cache_key(self.model_identifier, chunk_id, text_hash(text)) in self._rows

    # --- reporting ------------------------------------------------------------
    @property
    def size(self) -> int:
        """Number of cached entries (all models)."""
        self.load()
        return len(self._rows)

    @property
    def stats(self) -> dict[str, int]:
        """Cache counters for the current run."""
        return {"hits": self.hits, "misses": self.misses, "entries": self.size}
