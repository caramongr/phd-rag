"""SHA-256 hashing for discovered files.

Hashing is the authoritative signal for binary duplicate detection. Files are
read in fixed-size blocks so that very large documents never need to be loaded
fully into memory.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: Number of bytes read per iteration. 1 MiB balances syscall overhead and
#: memory use for multi-hundred-megabyte files.
CHUNK_SIZE: int = 1024 * 1024


def calculate_sha256(path: Path) -> str:
    """Return the lowercase hex SHA-256 digest of the file at ``path``.

    The file is streamed in :data:`CHUNK_SIZE` blocks. Any I/O problem (missing
    file, permission error, unreadable device) is propagated as an ``OSError``
    rather than being swallowed, so callers can decide how to react.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()
