"""Retrieval-layer models (RAG-009).

Three shapes matter here:

* :class:`RetrievalMode` plus :class:`RetrievalFilters` - *what* may be returned;
* :class:`Candidate` - one retrieved chunk with **two** scores that are never
  mixed up (``dense_score`` and ``rerank_score``) and all metadata RAG-010 will
  need for citation;
* :class:`QueryAudit` / :class:`RetrievalStats` - *how* the answer was produced,
  so a result can be reproduced and explained.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Roles GUIDANCE mode exposes: the user's own scaffolding material, never
#: scholarly evidence.
GUIDANCE_ROLES: tuple[str, ...] = ("rules", "guidance", "future_chapters", "summaries")

#: Content types RAG-009 supports filtering on (REFERENCES is never indexed, and
#: is rejected defensively if it ever appears).
KNOWN_CONTENT_TYPES: tuple[str, ...] = (
    "body",
    "abstract",
    "table",
    "conclusion",
    "appendix",
    "summary_entry",
    "caption",
    "references",
)

#: How many widening search rounds one query may use when the first stage is
#: exhausted by duplicate copies of a few works. Each round adds ``candidate_k``.
DEFAULT_MAX_CANDIDATE_ROUNDS = 3
#: Absolute ceiling on the candidate pool considered for one query.
DEFAULT_MAX_CANDIDATES = 150


class RetrievalMode(str, Enum):
    """Which slice of the corpus a query may see."""

    EVIDENCE = "evidence"
    GENERAL = "general"
    MY_WORK = "my_work"
    SUPERVISION = "supervision"
    GUIDANCE = "guidance"


class RetrievalFilters(BaseModel):
    """Optional, explicit, individually testable filters.

    Nothing here is a scoring knob: filters decide membership, never ranking.
    """

    model_config = ConfigDict(frozen=True)

    year_min: int | None = None
    year_max: int | None = None
    document_id: str | None = None
    file_hash: str | None = None
    doi: str | None = None
    corpus_role: str | None = None
    evidence_eligible: bool | None = None
    ocr_involved: bool | None = None
    publication_type: str | None = None
    content_types: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> "RetrievalFilters":
        if (
            self.year_min is not None
            and self.year_max is not None
            and self.year_min > self.year_max
        ):
            raise ValueError("year_min must not be greater than year_max")
        unknown = [value for value in self.content_types if value not in KNOWN_CONTENT_TYPES]
        if unknown:
            raise ValueError(f"unknown content_type(s): {', '.join(sorted(unknown))}")
        return self

    @property
    def is_empty(self) -> bool:
        """Whether no filter is set at all."""
        return all(
            (
                self.year_min is None,
                self.year_max is None,
                self.document_id is None,
                self.file_hash is None,
                self.doi is None,
                self.corpus_role is None,
                self.evidence_eligible is None,
                self.ocr_involved is None,
                self.publication_type is None,
                not self.content_types,
            )
        )

    def describe(self) -> dict[str, object]:
        """Audit-friendly description with empty values dropped."""
        raw = self.model_dump()
        return {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in raw.items()
            if value not in (None, ())
        }


class Candidate(BaseModel):
    """One retrieved chunk, with every field a citation needs."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    file_hash: str = ""
    filename: str = ""
    chunk_index: int | None = None

    dense_score: float | None = None
    rerank_score: float | None = None
    dense_rank: int | None = None
    final_rank: int | None = None

    text: str = ""
    retrieval_text: str = ""

    title: str | None = None
    first_author: str | None = None
    year: int | None = None
    doi: str | None = None
    publication_type: str | None = None

    page_start: int | None = None
    page_end: int | None = None
    section_title: str | None = None
    heading_path: tuple[str, ...] = ()
    content_type: str = ""

    corpus_role: str | None = None
    authority: str | None = None
    evidence_eligible: bool = False
    retrieval_eligible: bool = False

    #: Retrieval-time bibliography guard (RAG-009 correction). ``content_type`` is
    #: not trustworthy for this: some reference lists were classified as ``body``.
    bibliography_like: bool = False
    bibliography_reason: str | None = None

    text_sources: tuple[str, ...] = ()
    ocr_involved: bool = False
    source_locators: tuple[str, ...] = ()
    token_count: int | None = None

    @property
    def pages(self) -> str:
        """Human-readable page range (1-based, inclusive)."""
        if self.page_start is None:
            return "?"
        if self.page_end is None or self.page_end == self.page_start:
            return str(self.page_start)
        return f"{self.page_start}-{self.page_end}"


class DroppedCandidate(BaseModel):
    """A candidate that was retrieved and then deliberately not returned."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    reason: str
    detail: str | None = None
    dense_rank: int | None = None
    rerank_score: float | None = None


class QueryAudit(BaseModel):
    """Everything needed to reproduce a retrieval request."""

    model_config = ConfigDict(frozen=True)

    raw_query: str
    normalized_query: str
    retrieval_mode: RetrievalMode
    filters: dict[str, object] = Field(default_factory=dict)
    candidate_k: int
    final_k: int
    rerank_enabled: bool
    rerank_model: str | None = None
    rerank_revision: str | None = None
    rerank_device: str | None = None
    embedding_model: str
    embedding_revision: str | None = None
    embedding_device: str | None = None
    collection: str
    max_chunks_per_document: int
    max_candidate_rounds: int = DEFAULT_MAX_CANDIDATE_ROUNDS
    suppress_duplicates: bool
    duplicate_overlap_threshold: float
    exclude_bibliography: bool = True
    min_dense_score: float | None = None
    min_rerank_score: float | None = None
    timestamp: str


class RetrievalStats(BaseModel):
    """Counts and timings for one retrieval request."""

    model_config = ConfigDict(frozen=True)

    candidates_retrieved: int = 0
    candidates_after_filter: int = 0
    candidate_rounds: int = 1
    candidate_pool: int = 0
    dropped_by_post_filter: int = 0
    dropped_as_bibliography: int = 0
    dropped_by_min_score: int = 0
    dropped_as_duplicate: int = 0
    dropped_by_diversity: int = 0
    returned: int = 0
    embedding_seconds: float = 0.0
    search_seconds: float = 0.0
    rerank_seconds: float = 0.0
    total_seconds: float = 0.0


class RetrievalResponse(BaseModel):
    """The complete result of one retrieval request."""

    model_config = ConfigDict(frozen=True)

    audit: QueryAudit
    results: list[Candidate] = Field(default_factory=list)
    dense_candidates: list[Candidate] = Field(default_factory=list)
    dropped: list[DroppedCandidate] = Field(default_factory=list)
    stats: RetrievalStats = Field(default_factory=RetrievalStats)

    @property
    def chunk_ids(self) -> list[str]:
        """Returned chunk ids, in final rank order."""
        return [candidate.chunk_id for candidate in self.results]

    @property
    def dense_order(self) -> list[str]:
        """Returned chunk ids in dense-score order (for rerank comparison)."""
        return [candidate.chunk_id for candidate in self.dense_candidates]
