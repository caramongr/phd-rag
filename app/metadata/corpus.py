"""Corpus-role detection and enrichment gating (RAG-006).

The folder a document lives in determines its role, and the role gates whether
the document may be sent to an external bibliographic service. The author's own
work, rules, supervision records, guidance, future chapters and internal
summaries are never queried externally by default.

A logical document is one set of identical bytes (SHA-256), but it can exist in
several folders. :func:`resolve_corpus_role` therefore resolves the role from
*every* physical location and, when those locations disagree, lets the most
conservative role win: a draft that is also filed under ``literature`` must never
become scholarly evidence because a second copy exists there.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from app.ingestion.scanner import scan_directory
from app.metadata.enums import CorpusRole
from app.metadata.models import PhysicalRole, RoleResolution

#: Folder name (case-insensitive) -> corpus role, checked in priority order.
_ROLE_FOLDERS: tuple[tuple[str, CorpusRole], ...] = (
    ("my_work", CorpusRole.MY_WORK),
    ("rules", CorpusRole.RULES),
    ("supervision", CorpusRole.SUPERVISION),
    ("future_chapters", CorpusRole.FUTURE_CHAPTERS),
    ("summaries", CorpusRole.SUMMARIES),
    ("guidance", CorpusRole.GUIDANCE),
    ("literature", CorpusRole.LITERATURE),
)

#: Precedence used when the physical copies of one logical document (identical
#: SHA-256 bytes) sit in folders that imply different roles: the first role
#: present wins. It reuses ``_ROLE_FOLDERS`` order, so detection and resolution
#: share one documented ordering, and appends ``OTHER`` last.
#:
#: Every non-literature role precedes ``LITERATURE`` on purpose: a draft that is
#: also filed under ``literature`` must never become scholarly evidence merely
#: because a second copy exists there. Alphabetical discovery order therefore has
#: no influence on the outcome.
ROLE_PRECEDENCE: tuple[CorpusRole, ...] = tuple(
    role for _folder, role in _ROLE_FOLDERS
) + (CorpusRole.OTHER,)


def corpus_role_for_path(path: str | Path) -> CorpusRole:
    """Return the corpus role for ``path`` from its folder names.

    When a ``data`` directory appears in the path only the components below it
    are considered, so an unrelated parent folder cannot influence the role.
    """
    parts = [part.casefold() for part in Path(path).parts]
    scope = parts
    if "data" in parts:
        index = len(parts) - 1 - parts[::-1].index("data")
        scope = parts[index + 1 :]

    for folder, role in _ROLE_FOLDERS:
        if folder in scope:
            return role
    return CorpusRole.OTHER


def resolve_corpus_role(paths: Iterable[str | Path]) -> RoleResolution:
    """Resolve one logical document's role from *all* of its physical locations.

    Binary-identical copies are one logical document, so the role must not depend
    on which copy happened to be discovered first. Every location contributes its
    role; when they disagree, :data:`ROLE_PRECEDENCE` decides and the decision is
    recorded together with the reason.
    """
    unique_paths: dict[str, str] = {}
    for path in paths:
        text = str(path)
        unique_paths.setdefault(text, text)

    physical_roles = [
        PhysicalRole(path=text, role=corpus_role_for_path(text))
        for text in sorted(unique_paths)
    ]
    if not physical_roles:
        return RoleResolution(
            effective_role=CorpusRole.OTHER,
            reason="no physical location known",
            applied_precedence=list(ROLE_PRECEDENCE),
        )

    detected = {physical.role for physical in physical_roles}
    effective = next(role for role in ROLE_PRECEDENCE if role in detected)

    if len(physical_roles) == 1:
        reason = "single physical location"
    elif len(detected) == 1:
        reason = f"all {len(physical_roles)} physical locations agree"
    else:
        losing = [role.value for role in ROLE_PRECEDENCE if role in detected and role is not effective]
        reason = f"conservative precedence: {effective.value} preferred over {', '.join(losing)}"

    return RoleResolution(
        effective_role=effective,
        reason=reason,
        physical_roles=physical_roles,
        applied_precedence=list(ROLE_PRECEDENCE),
    )


def physical_paths_for_hash(root: str | Path, file_hash: str) -> list[str]:
    """Return every physical path under ``root`` whose bytes hash to ``file_hash``.

    Used by the development CLIs so a single file can be inspected with the role
    its *duplicates* imply, not just the role of the path that was passed in.
    """
    for document in scan_directory(Path(root)).documents:
        if document.file_hash == file_hash:
            return sorted(location.path for location in document.physical_locations)
    return []


def is_enrichable(role: CorpusRole, *, has_doi: bool) -> bool:
    """Return whether external enrichment is permitted for ``role``.

    Literature is always eligible. The author's own work is eligible only when a
    DOI is already known (i.e. it is plausibly a published work). Every other
    role is never enriched.
    """
    if role is CorpusRole.LITERATURE:
        return True
    if role is CorpusRole.MY_WORK:
        return has_doi
    return False
