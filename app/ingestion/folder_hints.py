"""Deterministic folder hints for the research corpus.

Hints are derived purely from the physical folder a file was found in. They are
*not* a semantic classification and must never replace the future semantic
classification stage; they merely seed useful metadata from the way the corpus is
currently organised on disk.
"""

from __future__ import annotations

from app.models.enums import RelatedProject

#: Folder name (case-insensitive) -> topic hint.
TOPIC_HINTS: dict[str, str] = {
    "ai": "artificial_intelligence",
    "vr": "virtual_reality",
    "teaching": "teaching",
    "algorithms": "algorithms",
}

#: Folder name (case-insensitive) -> related project hint. Only "Second paper"
#: yields a deterministic project hint in RAG-002.
RELATED_PROJECT_HINTS: dict[str, RelatedProject] = {
    "second paper": RelatedProject.SECOND_PAPER,
}


def _normalise(folder: str | None) -> str | None:
    """Return a case-insensitive key for a folder label, or ``None``."""
    if folder is None:
        return None
    normalised = folder.strip().casefold()
    return normalised or None


def topic_hint_for(folder: str | None) -> str | None:
    """Return the deterministic topic hint for a folder, if one is defined."""
    key = _normalise(folder)
    if key is None:
        return None
    return TOPIC_HINTS.get(key)


def related_project_hint_for(folder: str | None) -> RelatedProject | None:
    """Return the deterministic related-project hint for a folder, if any."""
    key = _normalise(folder)
    if key is None:
        return None
    return RELATED_PROJECT_HINTS.get(key)
