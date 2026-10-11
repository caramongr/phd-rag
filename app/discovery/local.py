"""Comparison of external discoveries against the local corpus (RAG-010).

Every external record is compared with the documents the local RAG already has, so a
result can say "already in your corpus" instead of presenting it as new. This module
does **not** invent a second matching system: it reuses

* RAG-006's DOI normalization (:func:`app.metadata.doi.normalize_doi`) so
  ``10.x/y``, ``https://doi.org/10.x/y`` and ``DOI:10.x/y`` are one DOI (§8);
* RAG-006's comparison key and title similarity
  (:func:`app.metadata.text.normalize_for_comparison`,
  :func:`app.metadata.text.title_similarity`);
* RAG-006's acceptance constants (strong/ambiguous thresholds, ±1 year), so the two
  stages cannot drift apart.

Matching priority (§7): exact normalized DOI, then exact/very strong normalized title,
then title + year + first-author consistency.

The local document set is read from the *indexed* payloads (the same metadata RAG-009
retrieves with), because that is what the corpus contains as far as retrieval is
concerned. Nothing here writes to the corpus, the index or the embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from app.discovery.models import ExternalWork, LocalMatch, LocalMatchKind
from app.metadata.doi import doi_cache_key, normalize_doi
from app.metadata.external.matching import MAX_YEAR_DELTA, STRONG_TITLE_SIMILARITY
from app.metadata.text import normalize_for_comparison, title_similarity

#: Similarity required for a title-only match to count as the same document.
EXACT_TITLE_SIMILARITY = 0.97
#: Similarity required, together with year and author agreement, for POSSIBLE_MATCH.
POSSIBLE_TITLE_SIMILARITY = 0.90
#: Corpus roles whose documents are scholarly evidence.
EVIDENCE_ROLE = "literature"


@dataclass(frozen=True)
class LocalDocument:
    """One document of the local corpus, as the index knows it."""

    document_id: str
    filename: str
    file_hash: str
    title: str | None = None
    doi: str | None = None
    year: int | None = None
    first_author: str | None = None
    corpus_role: str | None = None
    evidence_eligible: bool = False

    @property
    def normalized_title(self) -> str:
        """Comparison key for the title (empty when there is no title)."""
        return normalize_for_comparison(self.title) if self.title else ""

    @property
    def author_family(self) -> str | None:
        """Family name of the first author, when one is known."""
        if not self.first_author:
            return None
        tokens = self.first_author.replace(",", " ").split()
        return tokens[-1].casefold() if tokens else None


@dataclass
class LocalCorpusIndex:
    """A DOI and title index over the local documents."""

    documents: tuple[LocalDocument, ...] = ()
    _by_doi: dict[str, LocalDocument] = field(default_factory=dict, repr=False)
    _by_title: dict[str, LocalDocument] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        # Insert in document_id order so that when two documents share a DOI or a
        # normalized title, the winner is decided by a stable content key rather than
        # by the order the payloads happened to arrive in (determinism, §12/§30).
        for document in sorted(self.documents, key=lambda item: item.document_id):
            if document.doi:
                canonical = normalize_doi(document.doi)
                if canonical:
                    self._by_doi.setdefault(doi_cache_key(canonical), document)
            key = document.normalized_title
            if key:
                self._by_title.setdefault(key, document)

    # --- lookups --------------------------------------------------------------
    @property
    def document_count(self) -> int:
        """Number of indexed local documents."""
        return len(self.documents)

    @property
    def doi_count(self) -> int:
        """Number of local documents with a DOI."""
        return len(self._by_doi)

    @property
    def title_count(self) -> int:
        """Number of local documents with a usable title."""
        return len(self._by_title)

    def by_doi(self, doi: str | None) -> LocalDocument | None:
        """Find a document by DOI (any accepted DOI spelling)."""
        canonical = normalize_doi(doi) if doi else None
        if canonical is None:
            return None
        return self._by_doi.get(doi_cache_key(canonical))

    def by_title(self, title: str | None) -> LocalDocument | None:
        """Find a document whose normalized title is identical."""
        if not title:
            return None
        return self._by_title.get(normalize_for_comparison(title))

    def best_title_candidates(self, title: str, *, limit: int = 5) -> list[tuple[float, LocalDocument]]:
        """Title-similarity candidates, best first (deterministic)."""
        if not title:
            return []
        scored = [
            (title_similarity(title, document.title or ""), document)
            for document in self.documents
            if document.title
        ]
        scored.sort(key=lambda item: (-item[0], item[1].document_id))
        return scored[:limit]

    # --- matching -------------------------------------------------------------
    def match(self, work: ExternalWork) -> LocalMatch:
        """Compare one external record with the corpus.

        Returns a :class:`LocalMatch` describing whether the corpus contains this
        document, which local document it is, and *why* the match was accepted.
        """
        document = self.by_doi(work.doi)
        if document is not None:
            return _match_from(document, "doi_identity", title_similarity=1.0)

        document = self.by_title(work.title)
        if document is not None:
            year_delta = _year_delta(work, document)
            # The year guard applies to a title identity too: an identical title from a
            # materially different year is a different paper (a later edition or a
            # follow-up study), so it must not be silently swallowed as "already here".
            if year_delta is None or year_delta <= MAX_YEAR_DELTA:
                return _match_from(
                    document,
                    "normalized_title_identity",
                    title_similarity=1.0,
                    year_delta=year_delta,
                )

        for similarity, candidate in self.best_title_candidates(work.title):
            if similarity < POSSIBLE_TITLE_SIMILARITY:
                break
            year_delta = _year_delta(work, candidate)
            family = _family(work.authors)
            author_consistent = None
            if family and candidate.author_family:
                author_consistent = family == candidate.author_family
            strong_title = similarity >= STRONG_TITLE_SIMILARITY
            year_ok = year_delta is None or year_delta <= MAX_YEAR_DELTA
            author_ok = author_consistent is not False
            if similarity >= EXACT_TITLE_SIMILARITY and year_ok:
                return _match_from(
                    candidate,
                    "near_identical_title",
                    title_similarity=similarity,
                    year_delta=year_delta,
                    author_consistent=author_consistent,
                )
            if strong_title and year_ok and author_ok:
                return _match_from(
                    candidate,
                    "title_year_author_consistency",
                    kind=LocalMatchKind.POSSIBLE_MATCH,
                    title_similarity=similarity,
                    year_delta=year_delta,
                    author_consistent=author_consistent,
                )
        return LocalMatch(kind=LocalMatchKind.NEW, match_reason="no_local_match")


def _year_delta(work: ExternalWork, document: LocalDocument) -> int | None:
    """Absolute year difference, when both sides state a year."""
    if work.year is None or document.year is None:
        return None
    return abs(work.year - document.year)


def _family(authors: Sequence[str]) -> str | None:
    if not authors:
        return None
    tokens = authors[0].replace(",", " ").split()
    return tokens[-1].casefold() if tokens else None


def _match_from(
    document: LocalDocument,
    reason: str,
    *,
    kind: LocalMatchKind | None = None,
    title_similarity: float | None = None,
    year_delta: int | None = None,
    author_consistent: bool | None = None,
) -> LocalMatch:
    """Build a :class:`LocalMatch` for a matched local document.

    A match against a non-evidence document (the author's own draft, a summary) is
    reported as ``POSSIBLE_MATCH`` unless the identity is a DOI/title identity *and*
    the local document is scholarly evidence: an external paper is never promoted to
    "already have it as evidence" just because it resembles the author's own work.
    """
    is_evidence = document.evidence_eligible and document.corpus_role == EVIDENCE_ROLE
    resolved_kind = kind
    if resolved_kind is None:
        resolved_kind = LocalMatchKind.ALREADY_IN_CORPUS if is_evidence else LocalMatchKind.POSSIBLE_MATCH
    return LocalMatch(
        already_in_corpus=True,
        kind=resolved_kind,
        local_document_id=document.document_id,
        local_filename=document.filename,
        local_file_hash=document.file_hash,
        local_title=document.title,
        local_year=document.year,
        local_corpus_role=document.corpus_role,
        local_evidence_eligible=is_evidence,
        match_reason=reason if is_evidence else f"{reason} (local role: {document.corpus_role or 'unknown'})",
        title_similarity=title_similarity,
        year_delta=year_delta,
        author_consistent=author_consistent,
    )


def local_documents_from_payloads(payloads: Iterable[dict[str, Any]]) -> list[LocalDocument]:
    """Build the document list from indexed payloads (one entry per document id)."""
    documents: dict[str, LocalDocument] = {}
    for payload in payloads:
        document_id = str(payload.get("document_id") or "")
        if not document_id or document_id in documents:
            continue
        documents[document_id] = LocalDocument(
            document_id=document_id,
            filename=str(payload.get("filename") or ""),
            file_hash=str(payload.get("file_hash") or ""),
            title=payload.get("title"),
            doi=payload.get("doi"),
            year=payload.get("year"),
            first_author=payload.get("first_author"),
            corpus_role=payload.get("corpus_role"),
            evidence_eligible=bool(payload.get("evidence_eligible")),
        )
    return list(documents.values())


def build_index(payloads: Iterable[dict[str, Any]]) -> LocalCorpusIndex:
    """Build a :class:`LocalCorpusIndex` from indexed payloads."""
    return LocalCorpusIndex(documents=tuple(local_documents_from_payloads(payloads)))


def payloads_from_collection(
    client: Any, collection: str, *, limit: int = 20000
) -> list[dict[str, Any]]:
    """Read the document-level payloads of a Qdrant collection (read-only)."""
    fields = [
        "document_id",
        "filename",
        "file_hash",
        "title",
        "doi",
        "year",
        "first_author",
        "corpus_role",
        "evidence_eligible",
    ]
    payloads: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=1000,
            offset=offset,
            with_payload=fields,
            with_vectors=False,
        )
        payloads.extend(dict(point.payload or {}) for point in points)
        if offset is None or len(payloads) >= limit:
            break
    return payloads
