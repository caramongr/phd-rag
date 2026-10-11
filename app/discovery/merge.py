"""Deduplication, field-level provenance and conflict preservation (RAG-010).

Several providers return the same paper. Merging them is not just deduplication: the
merged record must say *which provider* supplied each field, and must keep a material
disagreement instead of silently picking a winner.

Rules:

* grouping is by normalized DOI first, then by very strong normalized title identity
  (spec §9);
* field priority follows what each provider is actually good at (§10): DOI/venue/
  publisher from Crossref, citation count and open-access status from OpenAlex, an
  abstract from whoever supplies one;
* a *material* conflict (different title, year apart by more than a year, different
  first author) marks the record ``CONFLICT`` and stores both values (§10);
* the merged order is deterministic: DOI, then title, then provider list (spec §30).

Field precedence is a *preference*, not a claim of authority: every field keeps the
list of providers that supplied it, so a consumer can always see where a value came
from. RAG-006's title-similarity function is reused so the two stages agree on what
"the same title" means.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from app.discovery.models import (
    DiscoveryStatus,
    ExternalWork,
    FieldConflict,
    FulltextStatus,
)
from app.metadata.doi import doi_cache_key, normalize_doi
from app.metadata.text import normalize_for_comparison, title_similarity

#: Titles at or above this similarity are treated as the same work when no DOI exists.
SAME_WORK_TITLE_SIMILARITY = 0.95
#: Similarity below which two titles are a *material* conflict, not a variation.
CONFLICT_TITLE_SIMILARITY = 0.80
#: Years further apart than this are a material conflict.
CONFLICT_YEAR_DELTA = 1

#: Which provider a field is normally taken from, before completeness is considered.
FIELD_PRECEDENCE: dict[str, tuple[str, ...]] = {
    "title": ("crossref", "openalex", "semanticscholar"),
    "authors": ("crossref", "openalex", "semanticscholar"),
    "year": ("crossref", "openalex", "semanticscholar"),
    "doi": ("crossref", "openalex", "semanticscholar"),
    "venue": ("crossref", "openalex", "semanticscholar"),
    "publisher": ("crossref", "openalex", "semanticscholar"),
    "publication_type": ("crossref", "openalex", "semanticscholar"),
    "abstract": ("crossref", "openalex", "semanticscholar"),
    "cited_by_count": ("openalex", "semanticscholar", "crossref"),
    "open_access": ("openalex", "semanticscholar", "crossref"),
    "open_access_status": ("openalex", "semanticscholar", "crossref"),
    "fulltext_url": ("openalex", "crossref", "semanticscholar"),
}

MERGED_FIELDS: tuple[str, ...] = (
    "title",
    "authors",
    "year",
    "doi",
    "url",
    "venue",
    "publisher",
    "publication_type",
    "abstract",
    "cited_by_count",
    "open_access",
    "open_access_status",
    "fulltext_url",
)

#: Provider order used whenever a deterministic choice among equals is needed.
PROVIDER_ORDER = ("crossref", "openalex", "semanticscholar")


def _provider_rank(provider: str) -> int:
    try:
        return PROVIDER_ORDER.index(provider)
    except ValueError:
        return len(PROVIDER_ORDER)


def _first_author_family(work: ExternalWork) -> str | None:
    if not work.authors:
        return None
    tokens = work.authors[0].split()
    return tokens[-1].casefold() if tokens else None


def _doi_key(value: str) -> str:
    """Case-insensitive DOI identity key, tolerant of URL/``doi:`` wrappers.

    Providers normally normalize before this point, but merging is the layer that
    decides identity, so it normalizes defensively: ``10.x/y``,
    ``https://doi.org/10.x/y`` and ``doi:10.X/Y`` must be one key (spec §9).
    """
    return doi_cache_key(normalize_doi(value) or value)


def same_work(left: ExternalWork, right: ExternalWork) -> bool:
    """Whether two external records describe the same paper.

    DOI equality wins outright. Without a DOI, a very strong normalized title match
    is required (§9) - a merely similar title is *not* enough, because merging two
    different papers would create a record neither provider supplied.
    """
    if left.doi and right.doi:
        if _doi_key(left.doi) == _doi_key(right.doi):
            return True
        # Two different DOIs are two different papers, whatever the titles say.
        return False
    if not left.title or not right.title:
        return False
    if normalize_for_comparison(left.title) == normalize_for_comparison(right.title):
        return True
    return title_similarity(left.title, right.title) >= SAME_WORK_TITLE_SIMILARITY


def _merge_group(group: Sequence[ExternalWork]) -> ExternalWork:
    """Merge records that describe one paper."""
    providers = tuple(
        dict.fromkeys(
            provider
            for work in group
            for provider in work.providers
        )
    )
    ordered_providers = tuple(
        sorted(providers, key=lambda name: (_provider_rank(name), name))
    )

    conflicts: list[FieldConflict] = []
    field_provenance: dict[str, tuple[str, ...]] = {}
    merged: dict[str, object] = {}

    # --- material conflict detection first: it changes the status ---
    titles = [(work.providers[0] if work.providers else "unknown", work.title) for work in group if work.title]
    if len(titles) > 1:
        base_title = titles[0][1]
        for provider, title in titles[1:]:
            if title and base_title and title_similarity(base_title, title) < CONFLICT_TITLE_SIMILARITY:
                conflicts.append(
                    FieldConflict(
                        field_name="title",
                        values=tuple(dict.fromkeys(str(t) for _p, t in titles)),
                        providers=tuple(provider for provider, _t in titles),
                        detail="provider titles differ materially",
                    )
                )
                break
    years = [(work.providers[0] if work.providers else "unknown", work.year) for work in group if work.year is not None]
    if len(years) > 1:
        values = [year for _p, year in years]
        if max(values) - min(values) > CONFLICT_YEAR_DELTA:
            conflicts.append(
                FieldConflict(
                    field_name="year",
                    values=tuple(str(year) for year in values),
                    providers=tuple(provider for provider, _y in years),
                    detail="provider years differ by more than one year",
                )
            )
    families = {
        (work.providers[0] if work.providers else "unknown"): _first_author_family(work)
        for work in group
        if work.authors
    }
    distinct_families = {family for family in families.values() if family}
    if len(distinct_families) > 1:
        conflicts.append(
            FieldConflict(
                field_name="authors",
                values=tuple(sorted(distinct_families)),
                providers=tuple(sorted(families)),
                detail="first-author family names disagree",
            )
        )

    # --- field-by-field merge with provenance ---
    by_provider = {
        (work.providers[0] if work.providers else "unknown"): work for work in group
    }
    for field in MERGED_FIELDS:
        supplied_by: list[str] = []
        for provider in ordered_providers:
            work = by_provider.get(provider)
            if work is None:
                continue
            value = getattr(work, field)
            if value in (None, "", ()):
                continue
            supplied_by.append(provider)
        if not supplied_by:
            continue
        preferred = FIELD_PRECEDENCE.get(field, PROVIDER_ORDER)
        chosen_provider = next(
            (
                provider
                for provider in preferred
                if provider in supplied_by
            ),
            supplied_by[0],
        )
        merged[field] = getattr(by_provider[chosen_provider], field)
        field_provenance[field] = tuple(supplied_by)

    # A DOI is stored in one canonical form so "https://doi.org/10.x" and "10.x" are
    # the same string everywhere downstream (spec §9).
    merged_doi = merged.get("doi")
    if isinstance(merged_doi, str):
        merged["doi"] = normalize_doi(merged_doi) or merged_doi

    raw_ids: dict[str, str] = {}
    for work in group:
        for key, value in work.raw_provider_ids.items():
            raw_ids.setdefault(key, value)
    missing = tuple(
        sorted(
            {
                name
                for work in group
                for name in work.missing_fields
                if name in MERGED_FIELDS and merged.get(name) in (None, "", ())
            }
        )
    )
    provenance_notes = tuple(
        dict.fromkeys(note for work in group for note in work.source_provenance)
    )
    fulltext_status = FulltextStatus.UNKNOWN
    for work in group:
        if work.fulltext_status is FulltextStatus.FULLTEXT_OPEN:
            fulltext_status = FulltextStatus.FULLTEXT_OPEN
            break
        if work.fulltext_status is FulltextStatus.LANDING_PAGE_ONLY:
            fulltext_status = FulltextStatus.LANDING_PAGE_ONLY
    merged.update(
        {
            "providers": ordered_providers,
            "raw_provider_ids": raw_ids,
            "field_provenance": field_provenance,
            "source_provenance": provenance_notes,
            "conflicts": tuple(conflicts),
            "missing_fields": missing,
            "fulltext_status": fulltext_status,
            "discovery_status": (
                DiscoveryStatus.CONFLICT if conflicts else DiscoveryStatus.EXTERNAL_DISCOVERY
            ),
            "doi": merged_doi,
        }
    )
    return ExternalWork(**merged)  # type: ignore[arg-type]


def merge_records(records: Iterable[ExternalWork]) -> tuple[list[ExternalWork], int]:
    """Merge duplicate external records.

    Returns the merged records in deterministic order plus the number of records that
    were merged away (i.e. ``input_count - output_count``).
    """
    items = list(records)
    # Sort before grouping so the grouping cannot depend on provider response order
    # (spec §30): the same inputs produce the same groups and the same field choices.
    items.sort(
        key=lambda work: (
            _doi_key(work.doi) if work.doi else "~",
            (work.title or "~").casefold(),
            work.providers,
        )
    )
    groups: list[list[ExternalWork]] = []
    for work in items:
        for group in groups:
            if any(same_work(work, member) for member in group):
                group.append(work)
                break
        else:
            groups.append([work])
    merged = [_merge_group(group) for group in groups]
    merged.sort(
        key=lambda work: (
            _doi_key(work.doi) if work.doi else "",
            (work.title or "").casefold(),
            work.providers,
        )
    )
    return merged, len(items) - len(merged)


def identity_confidence(work: ExternalWork, *, doi_lookup: bool = False) -> float:
    """Deterministic identity confidence for a merged record.

    The weights are fixed and explainable: a DOI is worth more than any other field,
    because a DOI *is* the identity claim; several providers agreeing adds evidence; a
    direct DOI lookup is stronger than a search hit. Nothing here is learned or random.
    """
    score = 0.0
    if work.doi:
        score += 0.35
    if doi_lookup:
        score += 0.15
    if work.title:
        score += 0.15
    if work.authors:
        score += 0.10
    if work.year is not None:
        score += 0.05
    if work.venue or work.publisher or work.publication_type:
        score += 0.05
    if len(work.providers) >= 2:
        score += 0.15
    return round(min(score, 1.0), 4)


def status_for(work: ExternalWork, *, doi_lookup: bool = False) -> tuple[DiscoveryStatus, float]:
    """The status and identity confidence a merged record should carry."""
    confidence = identity_confidence(work, doi_lookup=doi_lookup)
    if work.conflicts:
        return DiscoveryStatus.CONFLICT, confidence
    if confidence >= 0.80:
        return DiscoveryStatus.EXTERNAL_METADATA_VERIFIED, confidence
    if not work.title and not work.doi:
        return DiscoveryStatus.UNAVAILABLE, confidence
    return DiscoveryStatus.EXTERNAL_DISCOVERY, confidence


def apply_status(
    records: Sequence[ExternalWork], *, doi_lookup: bool = False
) -> list[ExternalWork]:
    """Attach status and identity confidence to every merged record."""
    return [
        work.model_copy(
            update={
                "discovery_status": status_for(work, doi_lookup=doi_lookup)[0],
                "identity_confidence": status_for(work, doi_lookup=doi_lookup)[1],
            }
        )
        for work in records
    ]


def merge_conflicts(
    records: Sequence[ExternalWork], *, doi_lookup: bool = False
) -> tuple[list[ExternalWork], int]:
    """Merge and then assign status/confidence (the usual pipeline order).

    ``doi_lookup`` marks the records as coming from a direct DOI lookup, which is
    stronger identity evidence than a search hit (see :func:`identity_confidence`).
    """
    merged, merged_away = merge_records(records)
    return apply_status(merged, doi_lookup=doi_lookup), merged_away


def normalized_doi_or_none(value: str | None) -> str | None:
    """Convenience wrapper keeping RAG-006's DOI normalization as the only parser."""
    return normalize_doi(value) if value else None
