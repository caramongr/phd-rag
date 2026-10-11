"""Data model for grounded answer generation (RAG-011).

Generation has one job: turn retrieved, policy-filtered evidence into an answer whose
every substantive claim can be traced back to an exact chunk. The models here make that
traceability explicit:

* a :class:`SourceBlock` is one labelled piece of context with a stable ``S<n>`` id and
  a *class* that says whether it may support scholarly claims at all;
* a :class:`ContextPackage` records what was included, what was dropped and why;
* a :class:`SourceMarkerAudit` records which markers the model actually produced and
  whether each one exists;
* a :class:`GenerationAudit` records the model, prompt version and real token counts.

Nothing here imports Qwen or Ollama: the model is reached through the
:class:`~app.generation.ollama.GenerationModel` protocol, so this module stays
importable without a running server.
"""

from __future__ import annotations

import datetime as _datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GenerationMode(str, Enum):
    """Which source classes may be placed in the context (development modes, §18)."""

    EVIDENCE_ONLY = "evidence_only"
    EVIDENCE_PLUS_MY_WORK = "evidence_plus_my_work"
    EVIDENCE_PLUS_GUIDANCE = "evidence_plus_guidance"

    @property
    def includes_my_work(self) -> bool:
        """Whether the author's own writing may be included as context."""
        return self in (
            GenerationMode.EVIDENCE_PLUS_MY_WORK,
            GenerationMode.EVIDENCE_PLUS_GUIDANCE,
        )

    @property
    def includes_guidance(self) -> bool:
        """Whether supervision/guidance material may be included as context."""
        return self is GenerationMode.EVIDENCE_PLUS_GUIDANCE


class SourceClass(str, Enum):
    """What kind of material a context block is, and therefore what it may support."""

    #: Published scholarship from the local corpus: the only citable evidence.
    EVIDENCE = "evidence"
    #: The author's own drafts. Continuity and terminology only, never evidence.
    MY_WORK = "my_work"
    #: Supervision notes, writing rules, future-chapter guidance. Instructions only.
    GUIDANCE = "guidance"
    #: External record found on the Internet, metadata only (RAG-010).
    EXTERNAL_DISCOVERY = "external_discovery"
    #: External record whose metadata two providers corroborated. Still metadata only.
    EXTERNAL_METADATA_VERIFIED = "external_metadata_verified"
    #: External record whose full text was actually retrieved and identity-checked.
    EXTERNAL_FULLTEXT_VERIFIED = "external_fulltext_verified"

    @property
    def is_external(self) -> bool:
        """Whether this block came from outside the local corpus."""
        return self.value.startswith("external_")

    @property
    def is_metadata_only(self) -> bool:
        """Whether nothing but bibliographic metadata was actually retrieved."""
        return self in (
            SourceClass.EXTERNAL_DISCOVERY,
            SourceClass.EXTERNAL_METADATA_VERIFIED,
        )


class OutputLength(str, Enum):
    """Simple answer-length control (§34)."""

    CONCISE = "concise"
    NORMAL = "normal"
    DETAILED = "detailed"

    @property
    def target_tokens(self) -> int:
        """Approximate answer length, used for the prompt and for ``num_predict``."""
        return {"concise": 220, "normal": 500, "detailed": 1100}[self.value]

    @property
    def instruction(self) -> str:
        """The one-line length instruction placed in the user prompt."""
        return {
            "concise": "Answer in 3 to 5 sentences.",
            "normal": "Answer in 2 to 4 short paragraphs.",
            "detailed": (
                "Write a fuller academic answer of about 6 to 10 short paragraphs, "
                "still using only the supplied evidence."
            ),
        }[self.value]


class GenerationStatus(str, Enum):
    """The outcome of one generation request."""

    #: An answer was produced and no grounding problem was detected.
    OK = "ok"
    #: An answer was produced, but markers/claims need attention (see the audit).
    GROUNDING_WARNINGS = "grounding_warnings"
    #: The model stated the evidence is insufficient. This is a correct outcome.
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    #: No citable context was available, so generation was skipped on purpose.
    NO_CITABLE_CONTEXT = "no_citable_context"
    #: The model could not be reached, timed out or answered malformed output.
    MODEL_ERROR = "model_error"

    @property
    def is_answer(self) -> bool:
        """Whether an answer text was produced by the model."""
        return self in (
            GenerationStatus.OK,
            GenerationStatus.GROUNDING_WARNINGS,
            GenerationStatus.INSUFFICIENT_EVIDENCE,
        )


class SourceBlock(BaseModel):
    """One labelled context block (``[S1]``...) built from retrieved material."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    source_class: SourceClass
    #: Whether a factual claim may cite this block.
    citable: bool
    text: str
    token_estimate: int
    #: RAG-009 provenance, preserved exactly as retrieved (§4).
    chunk_id: str | None = None
    document_id: str | None = None
    filename: str | None = None
    title: str | None = None
    first_author: str | None = None
    year: int | None = None
    doi: str | None = None
    #: Venue is only known for external records (the corpus stores no venue field).
    venue: str | None = None
    pages: str | None = None
    section_title: str | None = None
    corpus_role: str | None = None
    evidence_eligible: bool = False
    ocr_involved: bool = False
    content_type: str | None = None
    dense_score: float | None = None
    rerank_score: float | None = None
    #: External provenance (RAG-010), when this block is an external record.
    external: bool = False
    fulltext_verified: bool = False
    external_source_url: str | None = None
    external_providers: tuple[str, ...] = ()
    header: str = ""

    @property
    def is_table(self) -> bool:
        """Whether this block is a table (values must not be invented, §17)."""
        return self.content_type == "table"

    @property
    def has_page_provenance(self) -> bool:
        """Whether the block can be pointed at a page."""
        return bool(self.pages)

    def citation_metadata(self) -> dict[str, Any]:
        """The traceable metadata for ``source_map`` (§12)."""
        return {
            "source_id": self.source_id,
            "source_class": self.source_class.value,
            "citable": self.citable,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "title": self.title,
            "first_author": self.first_author,
            "year": self.year,
            "doi": self.doi,
            "venue": self.venue,
            "pages": self.pages,
            "section": self.section_title,
            "corpus_role": self.corpus_role,
            "evidence_eligible": self.evidence_eligible,
            "ocr_involved": self.ocr_involved,
            "content_type": self.content_type,
            "external": self.external,
            "fulltext_verified": self.fulltext_verified,
        }


class DroppedBlock(BaseModel):
    """A candidate that did not make it into the context, with the reason."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str | None = None
    source_id: str | None = None
    title: str | None = None
    reason: str = ""
    token_estimate: int | None = None
    rerank_score: float | None = None


class ContextPackage(BaseModel):
    """The deterministic context handed to the model (§6-§7)."""

    model_config = ConfigDict(frozen=True)

    blocks: tuple[SourceBlock, ...] = ()
    rendered: str = ""
    retrieved_chunks: int = 0
    included_chunks: int = 0
    excluded_chunks: int = 0
    dropped: tuple[DroppedBlock, ...] = ()
    context_tokens_estimated: int = 0
    budget_tokens: int = 0
    #: Evidence blocks may use at most this much of the budget.
    evidence_budget_tokens: int = 0
    #: Tokens held back for my_work/guidance blocks when they were requested.
    reserved_non_evidence_tokens: int = 0
    chars_per_token: float = 4.0
    estimator: str = "chars"
    prompt_version: str = ""
    mode: GenerationMode = GenerationMode.EVIDENCE_ONLY
    #: Always empty: a block is either included whole or dropped whole (§7).
    truncated_blocks: tuple[str, ...] = ()
    retrieval_seconds: float = 0.0

    @property
    def citable_blocks(self) -> tuple[SourceBlock, ...]:
        """Blocks that may support factual claims."""
        return tuple(block for block in self.blocks if block.citable)

    @property
    def non_citable_blocks(self) -> tuple[SourceBlock, ...]:
        """Blocks supplied as context but not citable."""
        return tuple(block for block in self.blocks if not block.citable)

    @property
    def source_ids(self) -> tuple[str, ...]:
        """Every supplied source id, in order."""
        return tuple(block.source_id for block in self.blocks)

    @property
    def citable_source_ids(self) -> tuple[str, ...]:
        """Source ids that may be cited."""
        return tuple(block.source_id for block in self.citable_blocks)

    @property
    def ocr_source_ids(self) -> tuple[str, ...]:
        """Source ids whose text came from OCR (§16)."""
        return tuple(block.source_id for block in self.blocks if block.ocr_involved)

    @property
    def table_source_ids(self) -> tuple[str, ...]:
        """Source ids that are tables (§17)."""
        return tuple(block.source_id for block in self.blocks if block.is_table)

    def block_by_id(self, source_id: str) -> SourceBlock | None:
        """Look up a block by its ``S<n>`` id."""
        for block in self.blocks:
            if block.source_id == source_id:
                return block
        return None

    def source_map(self) -> dict[str, dict[str, Any]]:
        """``S1 -> chunk/document metadata`` for later citation validation (§12)."""
        return {block.source_id: block.citation_metadata() for block in self.blocks}


class QuotationEvidence(BaseModel):
    """A quotation found in the answer, with whatever provenance can be checked (§15)."""

    model_config = ConfigDict(frozen=True)

    text: str
    source_ids: tuple[str, ...] = ()
    found_in_source: bool = False
    page_provenance: str | None = None
    note: str = ""


class SourceMarkerAudit(BaseModel):
    """What the model actually cited, checked against what it was given (§11-§13)."""

    model_config = ConfigDict(frozen=True)

    markers_found: tuple[str, ...] = ()
    marker_counts: dict[str, int] = Field(default_factory=dict)
    invalid_markers: tuple[str, ...] = ()
    #: Sentences making a claim without any marker.
    unmarked_claims: tuple[str, ...] = ()
    #: Sentences citing a block that is not citable in this mode.
    non_evidence_citations: tuple[str, ...] = ()
    #: Sentences that describe supplied non-evidence material and are therefore fine
    #: (for example "your draft states X [S6]" in evidence_plus_my_work mode).
    non_evidence_reports: tuple[str, ...] = ()
    #: Sentences that read as findings but cite metadata-only external records.
    external_finding_claims: tuple[str, ...] = ()
    quotations: tuple[QuotationEvidence, ...] = ()
    #: Non-trivial sentences without markers that make no explicit finding claim
    #: (interpretation and framing). Reported separately from ``unmarked_claims``.
    unmarked_statements: tuple[str, ...] = ()
    #: House-style problems (for example the em dash character).
    style_warnings: tuple[str, ...] = ()
    sentences_total: int = 0
    sentences_with_markers: int = 0

    @property
    def has_problems(self) -> bool:
        """Whether any grounding problem was detected."""
        return bool(
            self.invalid_markers
            or self.unmarked_claims
            or self.non_evidence_citations
            or self.external_finding_claims
            or [q for q in self.quotations if not q.found_in_source]
        )


class ModelFailure(BaseModel):
    """Why the model produced nothing, in the provider's own terms (§3)."""

    model_config = ConfigDict(frozen=True)

    kind: str
    message: str = ""
    status_code: int | None = None
    endpoint: str | None = None
    retryable: bool = False

    @property
    def is_unavailable(self) -> bool:
        """Whether the server could not be reached at all."""
        return self.kind in ("connection_error", "unavailable")


class GenerationAudit(BaseModel):
    """Everything needed to reproduce and time one generation (§3, §22, §30)."""

    model_config = ConfigDict(frozen=True)

    provider: str = "ollama"
    model: str = ""
    base_url: str = ""
    ollama_version: str | None = None
    prompt_version: str = ""
    prompt_sha256: str = ""
    system_chars: int = 0
    context_chars: int = 0
    question_chars: int = 0
    temperature: float = 0.1
    top_p: float = 0.9
    num_ctx: int = 8192
    max_output_tokens: int = 900
    think: bool = False
    keep_alive: str | None = None
    #: Real counts reported by Ollama.
    prompt_tokens: int | None = None
    prompt_tokens_estimated: int = 0
    generated_tokens: int | None = None
    prompt_eval_seconds: float | None = None
    generation_seconds: float | None = None
    load_seconds: float | None = None
    total_seconds: float | None = None
    tokens_per_second: float | None = None
    done_reason: str | None = None
    retrieval_seconds: float = 0.0
    context_seconds: float = 0.0
    wall_seconds: float = 0.0
    model_details: dict[str, Any] = Field(default_factory=dict)
    failure: ModelFailure | None = None

    @property
    def token_estimate_ratio(self) -> float | None:
        """Real prompt tokens divided by the estimate, for calibrating the estimator."""
        if not self.prompt_tokens or not self.prompt_tokens_estimated:
            return None
        return round(self.prompt_tokens / self.prompt_tokens_estimated, 3)


class GenerationResponse(BaseModel):
    """The structured result of one generation request (§12)."""

    model_config = ConfigDict(frozen=True)

    question: str
    answer: str = ""
    mode: GenerationMode = GenerationMode.EVIDENCE_ONLY
    output_length: OutputLength = OutputLength.NORMAL
    status: GenerationStatus = GenerationStatus.OK
    source_map: dict[str, dict[str, Any]] = Field(default_factory=dict)
    used_source_ids: tuple[str, ...] = ()
    unused_source_ids: tuple[str, ...] = ()
    marker_audit: SourceMarkerAudit = Field(default_factory=SourceMarkerAudit)
    unsupported_claim_warnings: tuple[str, ...] = ()
    external_discovery_notes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    retrieval_audit: dict[str, Any] = Field(default_factory=dict)
    context: ContextPackage = Field(default_factory=ContextPackage)
    generation_audit: GenerationAudit = Field(default_factory=GenerationAudit)
    system_prompt: str = ""
    user_prompt: str = ""
    timestamp: str = Field(
        default_factory=lambda: _datetime.datetime.now(_datetime.timezone.utc).isoformat(
            timespec="seconds"
        )
    )

    @property
    def ocr_involved(self) -> bool:
        """Whether any used source came from OCR (preserved for RAG-012, §16)."""
        return any(
            self.source_map[source_id].get("ocr_involved")
            for source_id in self.used_source_ids
            if source_id in self.source_map
        )

    @property
    def used_sources(self) -> tuple[dict[str, Any], ...]:
        """Citation metadata for each used source, in marker order."""
        return tuple(
            self.source_map[source_id]
            for source_id in self.used_source_ids
            if source_id in self.source_map
        )
