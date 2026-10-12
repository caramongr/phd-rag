"""Data model for the academic working modes (RAG-013).

A *mode* is an orchestration preset, nothing more. It says which slice of the corpus may
be retrieved, which material may enter the context and in what role, whether the local
model is asked for prose, whether RAG-012 validates that prose, and how the result is
presented. The presets live in :data:`MODES` and are frozen data, so one mode always
produces the same configuration and a mode's safety rules can be read (and tested) without
following code paths.

The one rule every mode keeps: **only evidence-eligible literature may support a scholarly
claim.** ``my_work`` stays the author's own material, supervision and guidance stay
instructions, and metadata-only Internet records stay records. :class:`ModeSource` records
the role each block was allowed to play, so a result can be checked rather than trusted.
"""

from __future__ import annotations

import datetime as _datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.citations.models import CitationMapEntry, ReferenceEntry, ValidationSummary
from app.generation.models import GenerationMode
from app.retrieval.models import RetrievalMode


class ModeName(str, Enum):
    """The five user-facing modes."""

    EVIDENCE = "evidence"
    WRITING = "writing"
    CONSISTENCY = "consistency"
    SUPERVISOR = "supervisor"
    RESEARCH = "research"


class OutputStyle(str, Enum):
    """How a mode presents what it found."""

    #: A generated answer, validated and cited (EVIDENCE, WRITING).
    VALIDATED_ANSWER = "validated_answer"
    #: A deterministic comparison of the author's work with the literature (CONSISTENCY).
    STRUCTURED_COMPARISON = "structured_comparison"
    #: Clearly labelled excerpts of supervision/guidance material (SUPERVISOR).
    LABELLED_EXCERPTS = "labelled_excerpts"
    #: Local results and external records, side by side and never merged (RESEARCH).
    SEPARATED_RESULTS = "separated_results"


class SourceRole(str, Enum):
    """What a block of corpus material may do in a mode."""

    #: May support scholarly claims.
    EVIDENCE = "evidence"
    #: Retrieval subject of a mode (the author's own work in CONSISTENCY).
    SUBJECT = "subject"
    #: Continuity and terminology only; never cited.
    CONTEXT_ONLY = "context_only"
    #: Instructions about the project; never cited.
    INSTRUCTIONS_ONLY = "instructions_only"
    #: Material to surface as-is; never cited.
    GUIDANCE_ONLY = "guidance_only"


class ModeStatus(str, Enum):
    """The outcome of one mode run."""

    OK = "ok"
    #: The model reported that the retrieved evidence is insufficient.
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    #: Nothing citable was retrieved, so no answer was generated.
    NO_CITABLE_CONTEXT = "no_citable_context"
    #: The model could not produce an answer.
    MODEL_ERROR = "model_error"
    #: Nothing at all was found for the query.
    NO_RESULTS = "no_results"


class ConsistencyFindingKind(str, Enum):
    """The kinds of difference CONSISTENCY mode reports (§4)."""

    TERMINOLOGY = "terminology"
    DUPLICATION = "duplication"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    NUMERIC_DISCREPANCY = "numeric_discrepancy"
    STYLE = "style"


class ModeConfig(BaseModel):
    """The preset behind one mode (§7)."""

    model_config = ConfigDict(frozen=True)

    name: ModeName
    purpose: str
    retrieval_mode: RetrievalMode
    generation_mode: GenerationMode
    include_my_work: bool = False
    include_guidance: bool = False
    include_supervision: bool = False
    #: Whether RAG-010 discovery runs and its records are shown separately.
    use_external_discovery: bool = False
    run_generation: bool = False
    run_citation_validation: bool = False
    output_style: OutputStyle = OutputStyle.VALIDATED_ANSWER
    #: Whether the author's own work is compared against the retrieved literature.
    compare_with_literature: bool = False
    #: Corpus role to the role it plays in this mode. The value is what the mode promises.
    context_policy: dict[str, SourceRole] = Field(default_factory=dict)

    def role_for(self, corpus_role: str | None) -> SourceRole:
        """The role a corpus role plays here (``INSTRUCTIONS_ONLY`` when undeclared)."""
        return self.context_policy.get(corpus_role or "", SourceRole.INSTRUCTIONS_ONLY)

    @property
    def allows_scholarly_claims(self) -> bool:
        """Whether this mode can produce a cited answer at all."""
        return self.run_generation and self.run_citation_validation


#: The user's writing preferences (§3). The CONSISTENCY style checks are written against
#: this list, not against an imagined rule set.
WRITING_PREFERENCES: tuple[str, ...] = (
    "plain academic prose",
    "short varied sentences",
    "clarity over sophistication",
    "no em dash",
    "Harvard citations",
    "no invented claims",
    "no promotional language",
)

#: Words that read as promotional language in academic prose.
PROMOTIONAL_TERMS: tuple[str, ...] = (
    "revolutionary",
    "groundbreaking",
    "game-changing",
    "game changing",
    "cutting-edge",
    "cutting edge",
    "unparalleled",
    "world-class",
    "best-in-class",
    "seamless",
    "effortless",
    "unprecedented",
    "must-have",
    "state-of-the-art",
)

#: Sentences longer than this are reported against "short varied sentences".
LONG_SENTENCE_WORDS = 35

#: The character the writing preferences rule out.
EM_DASH = "\u2014"

EVIDENCE_CONFIG = ModeConfig(
    name=ModeName.EVIDENCE,
    purpose="Answer a research question from scholarly evidence only.",
    retrieval_mode=RetrievalMode.EVIDENCE,
    generation_mode=GenerationMode.EVIDENCE_ONLY,
    run_generation=True,
    run_citation_validation=True,
    output_style=OutputStyle.VALIDATED_ANSWER,
    context_policy={"literature": SourceRole.EVIDENCE},
)

WRITING_CONFIG = ModeConfig(
    name=ModeName.WRITING,
    purpose="Draft PhD prose grounded in literature, respecting the author's work and rules.",
    retrieval_mode=RetrievalMode.EVIDENCE,
    generation_mode=GenerationMode.EVIDENCE_PLUS_GUIDANCE,
    include_my_work=True,
    include_guidance=True,
    run_generation=True,
    run_citation_validation=True,
    output_style=OutputStyle.VALIDATED_ANSWER,
    context_policy={
        "literature": SourceRole.EVIDENCE,
        "my_work": SourceRole.CONTEXT_ONLY,
        "rules": SourceRole.INSTRUCTIONS_ONLY,
        "guidance": SourceRole.INSTRUCTIONS_ONLY,
        "summaries": SourceRole.INSTRUCTIONS_ONLY,
        "future_chapters": SourceRole.INSTRUCTIONS_ONLY,
        "supervision": SourceRole.INSTRUCTIONS_ONLY,
    },
)

CONSISTENCY_CONFIG = ModeConfig(
    name=ModeName.CONSISTENCY,
    purpose="Compare the author's existing work with the literature and the rules.",
    retrieval_mode=RetrievalMode.MY_WORK,
    generation_mode=GenerationMode.EVIDENCE_PLUS_GUIDANCE,
    include_my_work=True,
    include_guidance=True,
    compare_with_literature=True,
    run_generation=False,
    run_citation_validation=False,
    output_style=OutputStyle.STRUCTURED_COMPARISON,
    context_policy={
        "my_work": SourceRole.SUBJECT,
        "literature": SourceRole.EVIDENCE,
        "rules": SourceRole.INSTRUCTIONS_ONLY,
        "guidance": SourceRole.INSTRUCTIONS_ONLY,
        "summaries": SourceRole.INSTRUCTIONS_ONLY,
        "future_chapters": SourceRole.INSTRUCTIONS_ONLY,
        "supervision": SourceRole.INSTRUCTIONS_ONLY,
    },
)

SUPERVISOR_CONFIG = ModeConfig(
    name=ModeName.SUPERVISOR,
    purpose="Surface what supervision and guidance material says about a topic.",
    retrieval_mode=RetrievalMode.SUPERVISION,
    generation_mode=GenerationMode.EVIDENCE_PLUS_GUIDANCE,
    include_supervision=True,
    include_guidance=True,
    run_generation=False,
    run_citation_validation=False,
    output_style=OutputStyle.LABELLED_EXCERPTS,
    context_policy={
        "supervision": SourceRole.GUIDANCE_ONLY,
        "rules": SourceRole.GUIDANCE_ONLY,
        "guidance": SourceRole.GUIDANCE_ONLY,
        "summaries": SourceRole.GUIDANCE_ONLY,
        "future_chapters": SourceRole.GUIDANCE_ONLY,
        "my_work": SourceRole.CONTEXT_ONLY,
    },
)

RESEARCH_CONFIG = ModeConfig(
    name=ModeName.RESEARCH,
    purpose="Search the local corpus and external academic sources, kept apart.",
    retrieval_mode=RetrievalMode.EVIDENCE,
    generation_mode=GenerationMode.EVIDENCE_ONLY,
    use_external_discovery=True,
    run_generation=False,
    run_citation_validation=False,
    output_style=OutputStyle.SEPARATED_RESULTS,
    context_policy={"literature": SourceRole.EVIDENCE},
)

#: The presets, keyed by mode. Frozen data: one mode, one configuration.
MODES: dict[ModeName, ModeConfig] = {
    ModeName.EVIDENCE: EVIDENCE_CONFIG,
    ModeName.WRITING: WRITING_CONFIG,
    ModeName.CONSISTENCY: CONSISTENCY_CONFIG,
    ModeName.SUPERVISOR: SUPERVISOR_CONFIG,
    ModeName.RESEARCH: RESEARCH_CONFIG,
}


def mode_config(mode: ModeName | str) -> ModeConfig:
    """The preset for ``mode`` (accepts the enum or its value)."""
    return MODES[ModeName(mode)]


class ModeSource(BaseModel):
    """One local block a mode retrieved, with the role it was allowed to play."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    chunk_id: str | None = None
    document_id: str | None = None
    filename: str | None = None
    title: str | None = None
    first_author: str | None = None
    year: int | None = None
    doi: str | None = None
    pages: str | None = None
    section_title: str | None = None
    corpus_role: str | None = None
    evidence_eligible: bool = False
    ocr_involved: bool = False
    content_type: str | None = None
    dense_score: float | None = None
    rerank_score: float | None = None
    #: What the mode was allowed to do with this block.
    role: SourceRole = SourceRole.EVIDENCE
    #: The retrieval mode that produced it.
    retrieval_mode: str = ""
    excerpt: str = ""

    @property
    def is_evidence(self) -> bool:
        """Whether this block could support a scholarly claim."""
        return self.role is SourceRole.EVIDENCE and self.evidence_eligible


class ExternalSourceView(BaseModel):
    """One external (Internet) record, always labelled as external (§6)."""

    model_config = ConfigDict(frozen=True)

    group: str = "possible_match"
    title: str | None = None
    authors: tuple[str, ...] = ()
    year: int | None = None
    doi: str | None = None
    url: str | None = None
    venue: str | None = None
    publisher: str | None = None
    providers: tuple[str, ...] = ()
    #: Preview only; metadata-only records must not look like evidence.
    abstract: str | None = None
    open_access: bool | None = None
    fulltext_status: str = ""
    discovery_status: str = ""
    identity_confidence: float = 0.0
    already_in_corpus: bool = False
    local_match_kind: str = ""
    #: Whether the corpus document this record matches is itself scholarly evidence.
    local_evidence_eligible: bool = False
    relevance_score: float = 0.0
    fulltext_verified: bool = False
    #: External metadata is never evidence; this is recorded, not assumed.
    usable_as_evidence: bool = False

    @property
    def key(self) -> str:
        """A stable identity for rendering and de-duplication."""
        return self.doi or self.title or self.url or "(untitled)"


class ConsistencyFinding(BaseModel):
    """One difference found between the author's work and the literature or rules (§4)."""

    model_config = ConfigDict(frozen=True)

    kind: ConsistencyFindingKind
    summary: str
    detail: str = ""
    #: The author's own text, when the finding is about it.
    your_text: str | None = None
    #: The literature or rule text the finding refers to.
    other_text: str | None = None
    your_chunk_id: str | None = None
    other_chunk_id: str | None = None
    #: What backs the finding: a literature chunk, a rules chunk, or the built-in list.
    evidence: tuple[str, ...] = ()
    #: How the finding was produced, so nothing looks like a model judgement.
    method: str = "lexical_overlap"

    @property
    def label(self) -> str:
        """A short human-readable label."""
        return self.kind.value.replace("_", " ")


class ModeSection(BaseModel):
    """A named part of a mode's output, referring to ``local_sources``/``external_sources``."""

    model_config = ConfigDict(frozen=True)

    name: str
    note: str = ""
    local_indices: tuple[int, ...] = ()
    external_indices: tuple[int, ...] = ()

    @property
    def is_empty(self) -> bool:
        """Whether this section has nothing in it."""
        return not self.local_indices and not self.external_indices


class ModeResult(BaseModel):
    """The common structured result every mode returns (§9)."""

    model_config = ConfigDict(frozen=True)

    mode: ModeName
    query: str
    status: ModeStatus = ModeStatus.OK
    #: Generated prose when the mode generates (EVIDENCE, WRITING). For CONSISTENCY this
    #: is the deterministic comparison report; for SUPERVISOR the labelled excerpt list.
    answer: str = ""
    #: The answer with Harvard citations (RAG-012). Empty when no citations were produced.
    validated_answer: str = ""
    references: tuple[str, ...] = ()
    reference_entries: tuple[ReferenceEntry, ...] = ()
    citation_map: tuple[CitationMapEntry, ...] = ()
    validation_summary: ValidationSummary | None = None
    #: The claim-by-claim audit from RAG-012, present when citation validation ran.
    claim_validations: tuple[Any, ...] = ()
    local_sources: tuple[ModeSource, ...] = ()
    external_sources: tuple[ExternalSourceView, ...] = ()
    sections: tuple[ModeSection, ...] = ()
    findings: tuple[ConsistencyFinding, ...] = ()
    warnings: tuple[str, ...] = ()
    audits: dict[str, Any] = Field(default_factory=dict)
    config: ModeConfig
    generation_status: str = ""
    seconds: float = 0.0
    timestamp: str = Field(
        default_factory=lambda: _datetime.datetime.now(_datetime.timezone.utc).isoformat(
            timespec="seconds"
        )
    )

    # --- convenience ----------------------------------------------------------
    @property
    def evidence_sources(self) -> tuple[ModeSource, ...]:
        """Local blocks that were allowed to support scholarly claims."""
        return tuple(source for source in self.local_sources if source.is_evidence)

    @property
    def context_only_sources(self) -> tuple[ModeSource, ...]:
        """Local blocks supplied as context or instructions, never as evidence."""
        return tuple(source for source in self.local_sources if not source.is_evidence)

    @property
    def new_external_papers(self) -> tuple[ExternalSourceView, ...]:
        """External records that are not already in the corpus."""
        return tuple(
            source for source in self.external_sources if source.group == "new_external"
        )

    @property
    def unsupported_claims(self) -> tuple[str, ...]:
        """Sentences RAG-012 withheld, when citation validation ran."""
        withheld = self.audits.get("withheld") or ()
        return tuple(str(item) for item in withheld)

    @property
    def has_citations(self) -> bool:
        """Whether this result carries Harvard citations."""
        return bool(self.references)

    def to_json(self) -> dict[str, Any]:
        """The whole result as JSON-ready data."""
        return self.model_dump(mode="json")


__all__ = [
    "CONSISTENCY_CONFIG",
    "ConsistencyFinding",
    "ConsistencyFindingKind",
    "EM_DASH",
    "EVIDENCE_CONFIG",
    "ExternalSourceView",
    "LONG_SENTENCE_WORDS",
    "MODES",
    "ModeConfig",
    "ModeName",
    "ModeResult",
    "ModeSection",
    "ModeSource",
    "ModeStatus",
    "OutputStyle",
    "PROMOTIONAL_TERMS",
    "RESEARCH_CONFIG",
    "SUPERVISOR_CONFIG",
    "SourceRole",
    "WRITING_CONFIG",
    "WRITING_PREFERENCES",
    "mode_config",
]
