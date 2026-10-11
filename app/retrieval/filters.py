"""Qdrant filter construction and the defensive payload predicate (RAG-009).

Safety works at two independent levels:

1. :func:`build_qdrant_filter` pushes the mode and the explicit filters into the
   vector search itself, so non-evidence points are never even candidates;
2. :func:`rejection_reason` re-checks every returned payload locally, so a
   collection that ignores a filter - or a future collection that still contains
   REFERENCES points - cannot leak into a result.

The second level is not redundant: it is what turns "we asked Qdrant nicely" into
"we verified what came back".
"""

from __future__ import annotations

from typing import Any

from qdrant_client import models

from app.retrieval.models import GUIDANCE_ROLES, RetrievalFilters, RetrievalMode

#: Content type that must never be returned, whatever the collection contains.
REFERENCES_CONTENT_TYPE = "references"

#: Rejection reasons recorded in the audit trail.
REASON_REFERENCES = "references_content_type"
REASON_INELIGIBLE = "not_retrieval_eligible"
REASON_MODE = "mode_mismatch"
REASON_FILTER = "filter_mismatch"
REASON_BIBLIOGRAPHY = "bibliography_like"

#: Modes whose purpose is scholarly evidence and content-bearing prose, so a
#: reference list is never a wanted result. GUIDANCE and SUPERVISION are excluded
#: from this policy on purpose: they surface the author's own scaffolding material,
#: where a reference-format table or a form's citation list can be the actual
#: content (e.g. the "Reference checklist" rules document).
BIBLIOGRAPHY_EXCLUDING_MODES: tuple[RetrievalMode, ...] = (
    RetrievalMode.EVIDENCE,
    RetrievalMode.MY_WORK,
    RetrievalMode.GENERAL,
)

_FILTER_KEYS: tuple[tuple[str, str], ...] = (
    ("document_id", "document_id"),
    ("file_hash", "file_hash"),
    ("doi", "doi"),
    ("corpus_role", "corpus_role"),
    ("publication_type", "publication_type"),
)


def _match(key: str, value: Any) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def mode_conditions(mode: RetrievalMode) -> list[Any]:
    """Qdrant conditions that encode ``mode`` on their own."""
    if mode is RetrievalMode.EVIDENCE:
        return [_match("evidence_eligible", True)]
    if mode is RetrievalMode.MY_WORK:
        return [_match("corpus_role", "my_work")]
    if mode is RetrievalMode.SUPERVISION:
        return [_match("corpus_role", "supervision")]
    if mode is RetrievalMode.GUIDANCE:
        return [models.FieldCondition(key="corpus_role", match=models.MatchAny(any=list(GUIDANCE_ROLES)))]
    return []


def build_qdrant_filter(
    mode: RetrievalMode = RetrievalMode.EVIDENCE,
    filters: RetrievalFilters | None = None,
) -> models.Filter | None:
    """Build the Qdrant filter for ``mode`` plus explicit ``filters``.

    RAG-008 did not index REFERENCES chunks, but the exclusion is requested anyway:
    a filter that depends on "the indexer did not do X" is an assumption, not a
    guarantee.
    """
    filters = filters or RetrievalFilters()
    must: list[Any] = list(mode_conditions(mode))

    # Always demand the point be retrieval-eligible (defensive; RAG-008 indexed only
    # eligible chunks) and never a reference chunk (Qdrant has no "not equal" in
    # `must`, so the exclusion lives in `must_not`).
    must.append(_match("retrieval_eligible", True))
    must_not: list[Any] = [
        models.FieldCondition(
            key="content_type",
            match=models.MatchValue(value=REFERENCES_CONTENT_TYPE),
        )
    ]

    for field, payload_key in _FILTER_KEYS:
        value = getattr(filters, field)
        if value not in (None, ""):
            must.append(_match(payload_key, value))
    if filters.evidence_eligible is not None:
        must.append(_match("evidence_eligible", bool(filters.evidence_eligible)))
    if filters.ocr_involved is not None:
        must.append(_match("ocr_involved", bool(filters.ocr_involved)))
    if filters.content_types:
        must.append(
            models.FieldCondition(
                key="content_type",
                match=models.MatchAny(any=list(filters.content_types)),
            )
        )
    if filters.year_min is not None or filters.year_max is not None:
        must.append(
            models.FieldCondition(
                key="year",
                range=models.Range(gte=filters.year_min, lte=filters.year_max),
            )
        )

    if not must and not must_not:
        return None
    return models.Filter(must=must or None, must_not=must_not or None)


def _value(payload: dict[str, Any], key: str) -> Any:
    value = payload.get(key)
    return value


def _mode_reason(payload: dict[str, Any], mode: RetrievalMode) -> str | None:
    if mode is RetrievalMode.EVIDENCE:
        return None if payload.get("evidence_eligible") is True else REASON_MODE
    if mode is RetrievalMode.MY_WORK:
        return None if payload.get("corpus_role") == "my_work" else REASON_MODE
    if mode is RetrievalMode.SUPERVISION:
        return None if payload.get("corpus_role") == "supervision" else REASON_MODE
    if mode is RetrievalMode.GUIDANCE:
        return None if payload.get("corpus_role") in GUIDANCE_ROLES else REASON_MODE
    return None


def _filter_reason(payload: dict[str, Any], filters: RetrievalFilters) -> str | None:
    for field, payload_key in _FILTER_KEYS:
        wanted = getattr(filters, field)
        if wanted in (None, ""):
            continue
        if _value(payload, payload_key) != wanted:
            return REASON_FILTER
    if filters.evidence_eligible is not None:
        if bool(payload.get("evidence_eligible")) is not bool(filters.evidence_eligible):
            return REASON_FILTER
    if filters.ocr_involved is not None:
        if bool(payload.get("ocr_involved")) is not bool(filters.ocr_involved):
            return REASON_FILTER
    if filters.content_types and payload.get("content_type") not in filters.content_types:
        return REASON_FILTER
    year = payload.get("year")
    if filters.year_min is not None and (year is None or year < filters.year_min):
        return REASON_FILTER
    if filters.year_max is not None and (year is None or year > filters.year_max):
        return REASON_FILTER
    return None


def rejection_reason(
    payload: dict[str, Any],
    mode: RetrievalMode = RetrievalMode.EVIDENCE,
    filters: RetrievalFilters | None = None,
) -> str | None:
    """Why ``payload`` must not be returned, or ``None`` when it may be.

    Defense in depth: applied to everything Qdrant returned, before scoring is
    trusted.
    """
    if payload.get("content_type") == REFERENCES_CONTENT_TYPE:
        return REASON_REFERENCES
    if payload.get("retrieval_eligible") is not True:
        return REASON_INELIGIBLE
    reason = _mode_reason(payload, mode)
    if reason is not None:
        return reason
    return _filter_reason(payload, filters or RetrievalFilters())


def is_allowed(
    payload: dict[str, Any],
    mode: RetrievalMode = RetrievalMode.EVIDENCE,
    filters: RetrievalFilters | None = None,
) -> bool:
    """Whether ``payload`` passes every mode and filter rule."""
    return rejection_reason(payload, mode, filters) is None


def mode_excludes_bibliography(mode: RetrievalMode) -> bool:
    """Whether bibliography-like chunks are withheld in ``mode`` by default."""
    return mode in BIBLIOGRAPHY_EXCLUDING_MODES
