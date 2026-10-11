"""Deterministic corpus fingerprint for the index (RAG-008).

The fingerprint answers "which chunk set produced these vectors?" without trusting
the collection name. It is derived only from stable values - the chunking
normalization version, the sorted source file hashes and the sorted chunk ids - so
the same unchanged corpus reproduces it exactly, and no timestamp or iteration
order can leak into it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

#: Version marker so a change to the fingerprint recipe is visible.
FINGERPRINT_VERSION = "rag008-corpus-fingerprint-v1"


def _digest(lines: Iterable[str]) -> str:
    hasher = hashlib.sha256()
    for line in lines:
        hasher.update(line.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def document_hashes_digest(file_hashes: Iterable[str]) -> str:
    """Digest of the sorted source file hashes (order independent)."""
    return _digest(sorted(set(file_hashes)))


def chunk_ids_digest(chunk_ids: Iterable[str]) -> str:
    """Digest of the sorted chunk ids (order independent)."""
    return _digest(sorted(chunk_ids))


def corpus_fingerprint(
    *,
    file_hashes: Iterable[str],
    chunk_ids: Iterable[str],
    chunking_version: str,
) -> str:
    """Fingerprint of one indexed corpus: hashes + chunk ids + chunking version."""
    hasher = hashlib.sha256()
    hasher.update(f"{FINGERPRINT_VERSION}\n".encode("utf-8"))
    hasher.update(f"chunking={chunking_version}\n".encode("utf-8"))
    hasher.update(f"documents={document_hashes_digest(file_hashes)}\n".encode("utf-8"))
    hasher.update(f"chunks={chunk_ids_digest(chunk_ids)}\n".encode("utf-8"))
    return hasher.hexdigest()
