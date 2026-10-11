"""Crossref and OpenAlex metadata clients (RAG-006).

Both clients fetch bibliographic metadata only, cache raw responses by DOI (or
normalized lookup parameters), and normalise the result into a single
:class:`ExternalRecord` shape. They never raise on a failed lookup; callers see
``None`` or an empty list and act conservatively.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.metadata.doi import doi_cache_key, normalize_doi
from app.metadata.enums import MetadataSource, PublicationType
from app.metadata.external.cache import JsonCache
from app.metadata.external.http import HttpClient
from app.metadata.external.hygiene import clean_external_text, has_artefacts
from app.metadata.models import Author
from app.metadata.text import collapse_whitespace

_DEFAULT_HEADERS = {"User-Agent": "phd-rag/0.1 (academic metadata extraction)"}


def _headers_with_contact(mailto: str | None) -> dict[str, str]:
    """Identify this client to the bibliographic APIs (Crossref/OpenAlex polite pool)."""
    headers = dict(_DEFAULT_HEADERS)
    if mailto:
        headers["User-Agent"] = f"{_DEFAULT_HEADERS['User-Agent']} mailto:{mailto}"
    return headers

#: Crossref ``type`` -> :class:`PublicationType`.
_CROSSREF_TYPES: dict[str, PublicationType] = {
    "journal-article": PublicationType.JOURNAL_ARTICLE,
    "proceedings-article": PublicationType.CONFERENCE_PAPER,
    "book": PublicationType.BOOK,
    "monograph": PublicationType.BOOK,
    "edited-book": PublicationType.BOOK,
    "reference-book": PublicationType.BOOK,
    "book-chapter": PublicationType.BOOK_CHAPTER,
    "reference-entry": PublicationType.BOOK_CHAPTER,
    "dissertation": PublicationType.THESIS,
    "report": PublicationType.REPORT,
    "standard": PublicationType.REPORT,
    "posted-content": PublicationType.PREPRINT,
    "component": PublicationType.UNKNOWN,
    "dataset": PublicationType.UNKNOWN,
    "peer-review": PublicationType.UNKNOWN,
    "proceedings": PublicationType.CONFERENCE_PAPER,
}

#: OpenAlex ``type`` -> :class:`PublicationType`.
_OPENALEX_TYPES: dict[str, PublicationType] = {
    "article": PublicationType.JOURNAL_ARTICLE,
    "review": PublicationType.JOURNAL_ARTICLE,
    "editorial": PublicationType.JOURNAL_ARTICLE,
    "letter": PublicationType.JOURNAL_ARTICLE,
    "book": PublicationType.BOOK,
    "book-chapter": PublicationType.BOOK_CHAPTER,
    "dissertation": PublicationType.THESIS,
    "report": PublicationType.REPORT,
    "preprint": PublicationType.PREPRINT,
    "proceedings-article": PublicationType.CONFERENCE_PAPER,
    "standard": PublicationType.REPORT,
    "paratext": PublicationType.UNKNOWN,
    "erratum": PublicationType.UNKNOWN,
    "dataset": PublicationType.UNKNOWN,
}


@dataclass
class ExternalRecord:
    """A normalised bibliographic record from an external service."""

    source: MetadataSource
    doi: str | None = None
    title: str | None = None
    authors: list[Author] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    publisher: str | None = None
    publication_type: PublicationType = PublicationType.UNKNOWN
    url: str | None = None
    abstract: str | None = None
    #: The provider's text before hygiene, kept when it differed (provenance).
    raw_title: str | None = None
    #: Names of the fields whose provider text had to be normalised.
    normalised: tuple[str, ...] = ()


def _strip_markup(text: str | None) -> str | None:
    """Remove markup/encoding artefacts from one external value."""
    return clean_external_text(text)


def _note_normalised(normalised: list[str], field_name: str, *raws: object) -> None:
    """Record that a field's provider text contained artefacts."""
    if any(has_artefacts(raw) for raw in raws):
        normalised.append(field_name)


def _authors_from_crossref(items: list[dict], normalised: list[str]) -> list[Author]:
    authors: list[Author] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        _note_normalised(normalised, "authors", item.get("given"), item.get("family"), item.get("name"))
        given = clean_external_text(item.get("given"))
        family = clean_external_text(item.get("family"))
        name = clean_external_text(item.get("name"))
        full_name = collapse_whitespace(
            " ".join(part for part in (given, family) if part)
        ) or name
        if not full_name:
            continue
        orcid = item.get("ORCID")
        authors.append(
            Author(
                full_name=full_name,
                given_name=given,
                family_name=family,
                orcid=orcid.strip() if isinstance(orcid, str) and orcid.strip() else None,
            )
        )
    return authors


def _authors_from_openalex(items: list[dict], normalised: list[str]) -> list[Author]:
    authors: list[Author] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        author = item.get("author") or {}
        _note_normalised(normalised, "authors", author.get("display_name"))
        name = clean_external_text(author.get("display_name"))
        if not name:
            continue
        orcid = author.get("orcid")
        orcid_value = None
        if isinstance(orcid, str) and orcid.strip():
            orcid_value = orcid.rsplit("/", 1)[-1]
        authors.append(Author(full_name=name, orcid=orcid_value))
    return authors


def _abstract_from_openalex(inverted: dict | None) -> str | None:
    """Reconstruct an abstract from OpenAlex's inverted index, if present."""
    if not isinstance(inverted, dict) or not inverted:
        return None
    positions: list[tuple[int, str]] = []
    for word, indexes in inverted.items():
        if not isinstance(indexes, list):
            continue
        for position in indexes:
            if isinstance(position, int):
                positions.append((position, word))
    if not positions:
        return None
    positions.sort(key=lambda pair: pair[0])
    return collapse_whitespace(" ".join(word for _pos, word in positions)) or None


class CrossrefClient:
    """A minimal Crossref REST client with caching."""

    def __init__(
        self,
        http: HttpClient,
        cache: JsonCache | None = None,
        *,
        base_url: str = "https://api.crossref.org",
        mailto: str | None = None,
    ) -> None:
        self._http = http
        self._cache = cache
        self._base_url = base_url.rstrip("/")
        self._headers = _headers_with_contact(mailto)

    def lookup_doi(self, doi: str) -> ExternalRecord | None:
        """Look up a single DOI, returning a record or ``None``."""
        normalized = normalize_doi(doi) or doi.strip()
        key = f"crossref:doi:{doi_cache_key(normalized)}"
        cached = self._cache.get(key) if self._cache else None
        if cached is None:
            result = self._http.get_json(
                f"{self._base_url}/works/{normalized}", headers=self._headers
            )
            if not result.ok or not isinstance(result.data, dict):
                return None
            cached = result.data
            if self._cache:
                self._cache.set(key, cached)
        message = cached.get("message") if isinstance(cached, dict) else None
        if not isinstance(message, dict):
            return None
        return self._record_from_message(message)

    def search_title(self, title: str, *, rows: int = 5) -> list[ExternalRecord]:
        """Search Crossref bibliographically by title, returning candidates."""
        query = collapse_whitespace(title)
        if not query:
            return []
        key = f"crossref:title:{query.casefold()}:{rows}"
        cached = self._cache.get(key) if self._cache else None
        if cached is None:
            result = self._http.get_json(
                f"{self._base_url}/works",
                params={"query.bibliographic": query, "rows": rows},
                headers=self._headers,
            )
            if not result.ok or not isinstance(result.data, dict):
                return []
            cached = result.data
            if self._cache:
                self._cache.set(key, cached)
        message = cached.get("message") if isinstance(cached, dict) else None
        items = message.get("items") if isinstance(message, dict) else None
        if not isinstance(items, list):
            return []
        return [self._record_from_message(item) for item in items if isinstance(item, dict)]

    @staticmethod
    def _record_from_message(message: dict) -> ExternalRecord:
        normalised: list[str] = []

        titles = message.get("title") or []
        raw_title = titles[0] if titles and isinstance(titles[0], str) else None
        _note_normalised(normalised, "title", raw_title)
        title = clean_external_text(raw_title)

        venue = None
        container = message.get("container-title") or []
        if container:
            _note_normalised(normalised, "venue", container[0])
            venue = clean_external_text(container[0])
        if venue is None:
            event = message.get("event") or {}
            _note_normalised(normalised, "venue", event.get("name"))
            venue = clean_external_text(event.get("name"))

        _note_normalised(normalised, "publisher", message.get("publisher"))
        _note_normalised(normalised, "abstract", message.get("abstract"))
        year = _crossref_year(message)
        return ExternalRecord(
            source=MetadataSource.CROSSREF,
            doi=normalize_doi(message.get("DOI") or "") or None,
            title=title,
            authors=_authors_from_crossref(message.get("author") or [], normalised),
            year=year,
            venue=venue,
            publisher=clean_external_text(message.get("publisher")),
            publication_type=_CROSSREF_TYPES.get(
                str(message.get("type") or "").casefold(), PublicationType.UNKNOWN
            ),
            url=collapse_whitespace(message.get("URL") or "") or None,
            abstract=_strip_markup(message.get("abstract")),
            raw_title=raw_title if raw_title and raw_title != title else None,
            normalised=tuple(dict.fromkeys(normalised)),
        )


def _crossref_year(message: dict) -> int | None:
    for field_name in ("issued", "published-print", "published-online", "created"):
        parts = (message.get(field_name) or {}).get("date-parts")
        if parts and isinstance(parts, list) and parts[0]:
            year = parts[0][0]
            if isinstance(year, int):
                return year
    return None


class OpenAlexClient:
    """A minimal OpenAlex client with caching."""

    def __init__(
        self,
        http: HttpClient,
        cache: JsonCache | None = None,
        *,
        base_url: str = "https://api.openalex.org",
        mailto: str | None = None,
    ) -> None:
        self._http = http
        self._cache = cache
        self._base_url = base_url.rstrip("/")
        self._mailto = mailto
        self._headers = _headers_with_contact(mailto)

    def _params(self, extra: dict) -> dict:
        params = dict(extra)
        if self._mailto:
            params["mailto"] = self._mailto
        return params

    def lookup_doi(self, doi: str) -> ExternalRecord | None:
        """Look up a single DOI, returning a record or ``None``."""
        normalized = normalize_doi(doi) or doi.strip()
        key = f"openalex:doi:{doi_cache_key(normalized)}"
        cached = self._cache.get(key) if self._cache else None
        if cached is None:
            result = self._http.get_json(
                f"{self._base_url}/works/https://doi.org/{normalized}",
                params=self._params({}),
                headers=self._headers,
            )
            if not result.ok or not isinstance(result.data, dict):
                return None
            cached = result.data
            if self._cache:
                self._cache.set(key, cached)
        return self._record_from_work(cached)

    def search_title(self, title: str, *, per_page: int = 5) -> list[ExternalRecord]:
        """Search OpenAlex by title, returning candidate records."""
        query = collapse_whitespace(title)
        if not query:
            return []
        key = f"openalex:title:{query.casefold()}:{per_page}"
        cached = self._cache.get(key) if self._cache else None
        if cached is None:
            result = self._http.get_json(
                f"{self._base_url}/works",
                params=self._params({"search": query, "per-page": per_page}),
                headers=self._headers,
            )
            if not result.ok or not isinstance(result.data, dict):
                return []
            cached = result.data
            if self._cache:
                self._cache.set(key, cached)
        results = cached.get("results") if isinstance(cached, dict) else None
        if not isinstance(results, list):
            return []
        return [self._record_from_work(item) for item in results if isinstance(item, dict)]

    @staticmethod
    def _record_from_work(work: dict) -> ExternalRecord:
        normalised: list[str] = []

        raw_title = work.get("title") or work.get("display_name")
        raw_title = raw_title if isinstance(raw_title, str) else None
        _note_normalised(normalised, "title", raw_title)
        title = clean_external_text(raw_title)

        source = work.get("primary_location") or {}
        source_meta = (source.get("source") or {}) if isinstance(source, dict) else {}
        _note_normalised(normalised, "venue", source_meta.get("display_name"))
        venue = clean_external_text(source_meta.get("display_name"))
        _note_normalised(normalised, "publisher", source_meta.get("host_organization_name"))
        publisher = clean_external_text(source_meta.get("host_organization_name"))

        year = work.get("publication_year")
        if not isinstance(year, int):
            year = None

        doi = normalize_doi(work.get("doi") or "") or None
        return ExternalRecord(
            source=MetadataSource.OPENALEX,
            doi=doi,
            title=title,
            authors=_authors_from_openalex(work.get("authorships") or [], normalised),
            year=year,
            venue=venue,
            publisher=publisher,
            publication_type=_OPENALEX_TYPES.get(
                str(work.get("type") or "").casefold(), PublicationType.UNKNOWN
            ),
            url=collapse_whitespace(work.get("id") or "") or None,
            abstract=_abstract_from_openalex(work.get("abstract_inverted_index")),
            raw_title=raw_title if raw_title and raw_title != title else None,
            normalised=tuple(dict.fromkeys(normalised)),
        )
