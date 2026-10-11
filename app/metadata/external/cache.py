"""A lightweight local JSON response cache for external lookups.

The cache is keyed by DOI (or normalized lookup parameters) and stores the raw
parsed JSON. It is a single JSON file — no database is introduced for metadata
caching.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonCache:
    """A best-effort JSON file cache.

    Read/write failures never propagate: a missing or corrupt cache simply means
    no cached entries are available.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._entries: dict[str, Any] = self._load()
        #: Instrumentation for the enrichment reports.
        self.hits = 0
        self.misses = 0

    @property
    def entry_count(self) -> int:
        """Return the number of cached entries."""
        return len(self._entries)

    def _load(self) -> dict[str, Any]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError:
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def has(self, key: str) -> bool:
        """Return whether ``key`` is cached."""
        return key in self._entries

    def get(self, key: str) -> Any | None:
        """Return the cached value for ``key``, or ``None``."""
        entry = self._entries.get(key)
        if not isinstance(entry, dict):
            self.misses += 1
            return None
        self.hits += 1
        return entry.get("value")

    def set(self, key: str, value: Any) -> None:
        """Store ``value`` under ``key`` and flush the cache to disk."""
        self._entries[key] = {
            "stored_at": datetime.now(timezone.utc).isoformat(),
            "value": value,
        }
        self._flush()

    def _flush(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self._entries, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            # A cache write failure must never break metadata extraction.
            pass
