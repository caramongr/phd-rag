"""Deterministic ranking of external discoveries (RAG-010).

Ranking answers "which of these external papers should the reader look at first?", not
"which is true". The score is a fixed weighted sum of measurable properties:

* **query relevance of the title** (0.55) - the strongest signal for a search;
* **query relevance of the abstract** (0.20), when a provider supplied one;
* **metadata completeness** (0.15) - a record with authors, year, venue and a DOI is
  more usable than a bare title;
* **recency** (0.10) - a small, bounded nudge towards newer work, never a reordering
  by age alone.

Citation count is deliberately *not* in the score (§19): it is displayed as context,
because a highly cited paper is not automatically relevant to this query. Ties are
broken deterministically (year descending, title, DOI), so identical inputs always give
the identical order.
"""

from __future__ import annotations

from typing import Sequence

from app.discovery.models import ExternalWork
from app.metadata.text import normalize_for_comparison, title_similarity

#: Weights: they sum to 1.0, and changing them changes every score, so they live here.
WEIGHT_TITLE = 0.55
WEIGHT_ABSTRACT = 0.20
WEIGHT_COMPLETENESS = 0.15
WEIGHT_RECENCY = 0.10

#: The year at or above which recency is maximal (kept fixed for determinism).
RECENCY_REFERENCE_YEAR = 2026
#: Years over which recency decays to zero.
RECENCY_WINDOW = 12

_COMPLETENESS_FIELDS = ("authors", "year", "venue", "doi", "abstract", "publication_type")


def _relevance(query: str, text: str | None) -> float:
    if not query or not text:
        return 0.0
    if normalize_for_comparison(query) == normalize_for_comparison(text):
        return 1.0
    return max(0.0, min(1.0, title_similarity(query, text)))


def completeness(work: ExternalWork) -> float:
    """Share of the bibliographic fields a reader expects that are present."""
    present = 0
    for field in _COMPLETENESS_FIELDS:
        value = getattr(work, field)
        if value not in (None, "", ()):
            present += 1
    return present / len(_COMPLETENESS_FIELDS)


def recency(work: ExternalWork, *, reference_year: int = RECENCY_REFERENCE_YEAR) -> float:
    """A bounded 0..1 recency value derived from the publication year."""
    if work.year is None:
        return 0.0
    age = reference_year - work.year
    if age <= 0:
        return 1.0
    if age >= RECENCY_WINDOW:
        return 0.0
    return round(1.0 - (age / RECENCY_WINDOW), 4)


def relevance_score(work: ExternalWork, query: str) -> float:
    """The composite deterministic relevance score."""
    score = (
        WEIGHT_TITLE * _relevance(query, work.title)
        + WEIGHT_ABSTRACT * _relevance(query, work.abstract)
        + WEIGHT_COMPLETENESS * completeness(work)
        + WEIGHT_RECENCY * recency(work)
    )
    return round(score, 6)


def rank_records(
    records: Sequence[ExternalWork], query: str, *, limit: int | None = None
) -> list[ExternalWork]:
    """Attach relevance scores and return records in deterministic rank order."""
    scored = [
        work.model_copy(update={"relevance_score": relevance_score(work, query)})
        for work in records
    ]
    scored.sort(
        key=lambda work: (
            -work.relevance_score,
            -(work.year or 0),
            (work.title or "").casefold(),
            work.doi or "",
            work.providers,
        )
    )
    return scored[:limit] if limit is not None else scored


def describe_weights() -> dict[str, float]:
    """The weights used, for the report and the audit trail."""
    return {
        "title_relevance": WEIGHT_TITLE,
        "abstract_relevance": WEIGHT_ABSTRACT,
        "completeness": WEIGHT_COMPLETENESS,
        "recency": WEIGHT_RECENCY,
        "recency_reference_year": float(RECENCY_REFERENCE_YEAR),
        "recency_window": float(RECENCY_WINDOW),
    }
