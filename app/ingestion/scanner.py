"""Recursive, read-only discovery of research files.

The scanner walks a directory tree, hashes every supported file, and groups
files with identical SHA-256 hashes into a single logical
:class:`~app.ingestion.models.DiscoveredDocument`. It never opens a file for
writing: the corpus is treated as strictly read-only.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypeVar

from app.ingestion.file_hash import calculate_sha256
from app.ingestion.folder_hints import related_project_hint_for, topic_hint_for
from app.ingestion.models import DiscoveredDocument, ScanError, ScanResult
from app.models.physical_location import PhysicalLocation

#: File extensions the scanner recognises. Matching is case-insensitive because
#: every encountered extension is lower-cased before comparison.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".docx", ".doc", ".rtf", ".txt", ".md", ".xlsx", ".xls"}
)

_T = TypeVar("_T")


class ScanRootError(Exception):
    """Raised when the supplied scan root is missing, inaccessible, or not a directory."""


def _append_unique(items: list[_T], value: _T) -> None:
    """Append ``value`` to ``items`` only if it is not already present."""
    if value not in items:
        items.append(value)


def _relative_folder_components(parent: Path, root: Path) -> list[str]:
    """Return directory names from ``root`` down to ``parent`` (root excluded).

    Hints are evaluated against the *whole* ancestor path relative to the scan
    root, so a file such as ``root\\literature\\AI\\adaptive-learning\\paper.pdf``
    still contributes the ``AI`` hint. When ``parent`` is the root itself, the
    list is empty.
    """
    try:
        relative = parent.relative_to(root)
    except ValueError:
        return []
    return [part for part in relative.parts if part not in ("", ".")]


def scan_directory(root: Path) -> ScanResult:
    """Recursively scan ``root`` and return a typed :class:`ScanResult`.

    Raises:
        ScanRootError: if ``root`` does not exist, is inaccessible, or is not a
            directory. Individual file failures do **not** raise; they are
            recorded in :attr:`ScanResult.errors` and scanning continues.
    """
    root_path = Path(root)

    try:
        exists = root_path.exists()
    except OSError as exc:  # e.g. an unreadable drive or malformed path
        raise ScanRootError(f"Scan root is inaccessible: {root_path}") from exc
    if not exists:
        raise ScanRootError(f"Scan root does not exist: {root_path}")
    if not root_path.is_dir():
        raise ScanRootError(f"Scan root is not a directory: {root_path}")

    result = ScanResult()
    documents_by_hash: dict[str, DiscoveredDocument] = {}

    def _on_walk_error(exc: OSError) -> None:
        """Record directory-level walk failures without aborting the scan."""
        failed = getattr(exc, "filename", None) or str(root_path)
        result.errors.append(ScanError(path=str(failed), message=str(exc)))

    for dirpath, dirnames, filenames in os.walk(root_path, onerror=_on_walk_error):
        # Sort for deterministic ordering and reproducible canonical paths.
        dirnames.sort()
        filenames.sort()

        for name in filenames:
            file_path = Path(dirpath) / name
            result.scanned_files += 1
            extension = file_path.suffix.lower()

            if extension not in SUPPORTED_EXTENSIONS:
                result.unsupported_files += 1
                result.unsupported_paths.append(str(file_path))
                continue

            result.supported_files += 1

            try:
                file_hash = calculate_sha256(file_path)
            except OSError as exc:
                result.errors.append(ScanError(path=str(file_path), message=str(exc)))
                continue

            folder = file_path.parent.name or None
            location = PhysicalLocation(path=str(file_path), folder=folder)

            document = documents_by_hash.get(file_hash)
            if document is None:
                document = DiscoveredDocument(
                    file_hash=file_hash,
                    filename=file_path.name,
                    canonical_path=str(file_path),
                    extension=extension,
                    physical_locations=[location],
                    original_folders=[folder] if folder else [],
                )
                documents_by_hash[file_hash] = document
            else:
                document.physical_locations.append(location)
                if folder is not None:
                    _append_unique(document.original_folders, folder)
                result.duplicate_copies += 1

            for component in _relative_folder_components(file_path.parent, root_path):
                topic_hint = topic_hint_for(component)
                if topic_hint is not None:
                    _append_unique(document.topic_hints, topic_hint)

                project_hint = related_project_hint_for(component)
                if project_hint is not None:
                    _append_unique(document.related_project_hints, project_hint)

    result.documents = list(documents_by_hash.values())
    result.unique_documents = len(result.documents)
    return result
