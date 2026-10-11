"""Matching rules for external records (RAG-006).

An external result is never accepted merely because it was returned. A DOI
lookup must return the same DOI (``EXACT``). A title lookup must clear a
similarity threshold and agree on year and first author; otherwise the result is
``AMBIGUOUS`` or ``NO_MATCH`` and local metadata is left untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.metadata.doi import doi_cache_key, normalize_doi
from app.metadata.enums import ExternalMatchStatus
from app.metadata.external.clients import ExternalRecord
from app.metadata.text import title_similarity

#: Similarity at or above which a title match is considered strong.
STRONG_TITLE_SIMILARITY = 0.92
#: Similarity below which a candidate is rejected outright.
MIN_TITLE_SIMILARITY = 0.75
#: Two candidates closer than this are treated as indistinguishable.
AMBIGUITY_MARGIN = 0.03
#: Allowed year discrepancy before a match is considered ambiguous.
MAX_YEAR_DELTA = 1
#: Below this similarity an exact-DOI record is treated as *materially
#: inconsistent* with a strong local title (``CONFLICT``) rather than merely
#: having minor punctuation/subtitle differences. Chosen to sit clearly below
#: the level produced by subtitle additions and abridgements, and clearly above
#: the level produced by two genuinely different papers that share a few words.
CONFLICT_TITLE_SIMILARITY = 0.60


def title_conflict(local_title: str, external_title: str) -> bool:
    """Return ``True`` when two titles are materially inconsistent.

    Used only to guard DOI-exact resolutions against a strong local title: minor
    punctuation or subtitle differences stay well above the threshold, whereas a
    different paper's title falls below it.
    """
    return title_similarity(local_title, external_title) < CONFLICT_TITLE_SIMILARITY


def doi_match_status(
    record: ExternalRecord, requested_doi: str
) -> ExternalMatchStatus:
    """Return ``EXACT`` when ``record`` carries the requested DOI, else ``NO_MATCH``."""
    requested = normalize_doi(requested_doi)
    if requested is None or record.doi is None:
        return ExternalMatchStatus.NO_MATCH
    if doi_cache_key(record.doi) == doi_cache_key(requested):
        return ExternalMatchStatus.EXACT
    return ExternalMatchStatus.NO_MATCH


#: Machine-readable reasons a title match was not accepted as ``STRONG``.
REASON_NO_CANDIDATES = "no_external_candidates"
REASON_BELOW_MINIMUM = "similarity_below_minimum"
REASON_BELOW_STRONG = "similarity_below_strong"
REASON_MULTIPLE_RECORDS = "multiple_external_records"
REASON_YEAR = "local_year_disagreement"
REASON_AUTHOR = "local_author_disagreement"


@dataclass(frozen=True)
class TitleMatchExplanation:
    """The full title-match decision, including *why* a match was withheld.

    ``title_match_status`` is a thin wrapper over this, so reporting code can
    classify withheld matches without re-implementing (and drifting from) the
    acceptance rules.
    """

    record: ExternalRecord | None
    status: ExternalMatchStatus
    reason: str | None = None
    best_similarity: float | None = None
    runner_up_similarity: float | None = None
    year_delta: int | None = None
    author_mismatch: bool = False
    candidate_count: int = 0
    matched_title: str | None = None


def _first_author_family(record: ExternalRecord) -> str | None:
    if not record.authors:
        return None
    first = record.authors[0]
    if first.family_name:
        return first.family_name
    tokens = first.full_name.split()
    if len(tokens) >= 2:
        return tokens[-1]
    return None


def explain_title_match(
    records: list[ExternalRecord],
    *,
    local_title: str,
    local_year: int | None = None,
    local_family: str | None = None,
) -> TitleMatchExplanation:
    """Return the title-match decision and the reason for it."""
    scored: list[tuple[float, ExternalRecord]] = []
    for record in records:
        if not record.title:
            continue
        scored.append((title_similarity(local_title, record.title), record))

    if not scored:
        return TitleMatchExplanation(
            record=None,
            status=ExternalMatchStatus.NO_MATCH,
            reason=REASON_NO_CANDIDATES,
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    best_similarity, best = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else None

    if best_similarity < MIN_TITLE_SIMILARITY:
        return TitleMatchExplanation(
            record=None,
            status=ExternalMatchStatus.NO_MATCH,
            reason=REASON_BELOW_MINIMUM,
            best_similarity=best_similarity,
            runner_up_similarity=runner_up,
            candidate_count=len(scored),
        )

    year_delta: int | None = None
    if local_year is not None and best.year is not None:
        year_delta = abs(local_year - best.year)
    author_mismatch = False
    if local_family:
        record_family = _first_author_family(best)
        if record_family and record_family.casefold() != local_family.casefold():
            author_mismatch = True

    common = dict(
        best_similarity=best_similarity,
        runner_up_similarity=runner_up,
        year_delta=year_delta,
        author_mismatch=author_mismatch,
        candidate_count=len(scored),
        matched_title=best.title,
    )

    if runner_up is not None and (best_similarity - runner_up) < AMBIGUITY_MARGIN:
        return TitleMatchExplanation(
            record=best, status=ExternalMatchStatus.AMBIGUOUS,
            reason=REASON_MULTIPLE_RECORDS, **common,
        )
    if year_delta is not None and year_delta > MAX_YEAR_DELTA:
        return TitleMatchExplanation(
            record=best, status=ExternalMatchStatus.AMBIGUOUS,
            reason=REASON_YEAR, **common,
        )
    if author_mismatch:
        return TitleMatchExplanation(
            record=best, status=ExternalMatchStatus.AMBIGUOUS,
            reason=REASON_AUTHOR, **common,
        )
    if best_similarity >= STRONG_TITLE_SIMILARITY:
        return TitleMatchExplanation(
            record=best, status=ExternalMatchStatus.STRONG, **common
        )
    return TitleMatchExplanation(
        record=best, status=ExternalMatchStatus.AMBIGUOUS,
        reason=REASON_BELOW_STRONG, **common,
    )


def title_match_status(
    records: list[ExternalRecord],
    *,
    local_title: str,
    local_year: int | None = None,
    local_family: str | None = None,
) -> tuple[ExternalRecord | None, ExternalMatchStatus]:
    """Return the best matching record and its :class:`ExternalMatchStatus`."""
    explanation = explain_title_match(
        records, local_title=local_title, local_year=local_year, local_family=local_family
    )
    return explanation.record, explanation.status
