"""Shared fixtures for the RAG-013 mode tests.

Everything here is offline and deterministic. The mode service is wired to
:class:`~tests.generation_support.FakeRetrievalService`-style fakes built from the *real*
RAG-009 models, the real RAG-011 generation service (with a scripted model instead of
Ollama) and the real RAG-012 citation service, so a mode test exercises the same objects a
mode run does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from app.discovery.models import (
    DiscoveryAudit,
    DiscoveryResult,
    DiscoveryStatus,
    ExternalWork,
    LocalMatch,
    LocalMatchKind,
)
from app.generation.ollama import ChatMessage, ChatResult, GenerationOptions
from app.modes.models import ModeName
from app.modes.service import ModeService
from app.retrieval.models import RetrievalMode, RetrievalResponse
from tests.citations_support import SENIOR_CLAIM, SENIOR_TEXT
from tests.generation_support import candidate, retrieval_response

#: The evidence chunk every mode test starts from (a real RAG-011 evidence sentence).
EVIDENCE_TEXT = SENIOR_TEXT
EVIDENCE_CLAIM = SENIOR_CLAIM

#: The author's own draft: one long sentence, no em dash, and a coinage of their own.
MY_WORK_TEXT = (
    "In this chapter I argue that AI governance in higher education should be understood as "
    "an institutional responsibility that runs through policy, teaching practice and "
    "supervision, and I call that stance algorithmic stewardship."
)
#: A second draft chunk that repeats the first one's material.
MY_WORK_NEAR_DUPLICATE = (
    "In this chapter I argue that AI governance in higher education is an institutional "
    "responsibility that runs through policy, teaching practice and supervision; I call that "
    "stance algorithmic stewardship."
)
#: A draft chunk about material the shared evidence chunk does not cover, with a figure.
MY_WORK_WITH_FIGURE = (
    "The survey reported that 92% of universities had adopted a mandatory AI literacy "
    "module by 2024."
)
#: A draft chunk with promotional language and an em dash.
MY_WORK_PROMOTIONAL = (
    "The platform offers a revolutionary and seamless experience \u2014 one that transforms "
    "how students learn."
)
#: The retrieved writing rules (guidance role, non-citable).
RULES_TEXT = (
    "Writing rules: use plain academic prose, keep sentences short and varied, avoid the em "
    "dash, use Harvard citations, never invent claims, and avoid promotional language."
)
#: The retrieved supervision note.
SUPERVISION_TEXT = (
    "Supervision note on methodology: justify the design science research approach, explain "
    "the evaluation strategy, and keep the artefact description separate from the findings."
)


@dataclass
class FakeModeRetrieval:
    """A RAG-009 double that returns a scripted response per retrieval mode."""

    evidence: list[Any] = field(default_factory=list)
    my_work: list[Any] = field(default_factory=list)
    guidance: list[Any] = field(default_factory=list)
    supervision: list[Any] = field(default_factory=list)
    error: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def retrieve(self, query: str, **kwargs: Any) -> RetrievalResponse:
        """Return the scripted candidates for the requested mode."""
        self.calls.append({"query": query, **kwargs})
        if self.error is not None:
            raise self.error
        mode = RetrievalMode(kwargs.get("mode") or RetrievalMode.EVIDENCE)
        candidates = {
            RetrievalMode.EVIDENCE: self.evidence,
            RetrievalMode.GENERAL: self.evidence,
            RetrievalMode.MY_WORK: self.my_work,
            RetrievalMode.SUPERVISION: self.supervision,
            RetrievalMode.GUIDANCE: self.guidance,
        }[mode]
        return retrieval_response(
            candidates,
            query=query,
            mode=mode,
            candidate_k=kwargs.get("candidate_k") or 25,
            final_k=kwargs.get("final_k") or 8,
        )

    def close(self) -> None:
        """Match the real service lifecycle."""

    def modes_called(self) -> list[str]:
        """The retrieval modes this fake was asked for, in order."""
        return [str(call.get("mode")) for call in self.calls]


@dataclass
class FakeDiscoveryService:
    """A RAG-010 double: scripted external records, no HTTP, no cache."""

    records: tuple[ExternalWork, ...] = ()
    available: bool = True
    error: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)
    shared_client: bool = False
    has_local_index: bool = False

    def discover(self, query: str, **kwargs: Any) -> DiscoveryResult:
        """Return the scripted discovery result."""
        self.calls.append({"query": query, **kwargs})
        if self.error is not None:
            raise self.error
        audit = DiscoveryAudit(
            raw_query=query,
            normalized_query=query,
            operation="search",
            providers_requested=("crossref", "openalex"),
            providers_succeeded=("crossref",) if self.available else (),
            providers_failed=() if self.available else ("crossref", "openalex"),
            merged_records=len(self.records),
            new_records=len([r for r in self.records if not r.local_match.already_in_corpus]),
            local_matches=len([r for r in self.records if r.local_match.already_in_corpus]),
            all_providers_failed=not self.available,
        )
        return DiscoveryResult(audit=audit, records=self.records)

    def share_client(self, client: Any, collection: str | None = None) -> None:
        """Record that the retrieval client was shared (no-op otherwise)."""
        self.shared_client = True


def evidence_candidate(**kwargs: Any) -> Any:
    """A RAG-009 evidence candidate carrying the shared evidence text."""
    kwargs.setdefault("text", EVIDENCE_TEXT)
    kwargs.setdefault("title", "Generative AI in Higher Education")
    kwargs.setdefault("first_author", "Cecilia Ka Yuk Chan")
    kwargs.setdefault("year", 2024)
    kwargs.setdefault("doi", "10.1234/governance")
    kwargs.setdefault("corpus_role", "literature")
    kwargs.setdefault("evidence_eligible", True)
    kwargs.setdefault("chunk_id", "chunk-evidence")
    return candidate(**kwargs)


def my_work_candidate(text: str = MY_WORK_TEXT, **kwargs: Any) -> Any:
    """A RAG-009 my_work candidate (never evidence)."""
    kwargs.setdefault("title", "Thesis chapter 3 draft")
    kwargs.setdefault("corpus_role", "my_work")
    kwargs.setdefault("evidence_eligible", False)
    kwargs.setdefault("chunk_id", "chunk-my-work")
    return candidate(text=text, **kwargs)


def guidance_candidate(text: str = RULES_TEXT, **kwargs: Any) -> Any:
    """A RAG-009 guidance candidate (rules: instructions only)."""
    kwargs.setdefault("title", "Writing rules")
    kwargs.setdefault("corpus_role", "rules")
    kwargs.setdefault("evidence_eligible", False)
    kwargs.setdefault("chunk_id", "chunk-rules")
    return candidate(text=text, **kwargs)


def supervision_candidate(text: str = SUPERVISION_TEXT, **kwargs: Any) -> Any:
    """A RAG-009 supervision candidate."""
    kwargs.setdefault("title", "Supervision notes: methodology")
    kwargs.setdefault("corpus_role", "supervision")
    kwargs.setdefault("evidence_eligible", False)
    kwargs.setdefault("chunk_id", "chunk-supervision")
    return candidate(text=text, **kwargs)


def external_record(
    title: str = "Governing AI in Universities",
    *,
    doi: str | None = "10.5555/external",
    year: int | None = 2025,
    already_in_corpus: bool = False,
    status: DiscoveryStatus = DiscoveryStatus.EXTERNAL_METADATA_VERIFIED,
    providers: Sequence[str] = ("crossref", "openalex"),
    conflicts: tuple[Any, ...] = (),
    authors: Sequence[str] = ("Jane Doe",),
) -> ExternalWork:
    """A real RAG-010 record, built from the real model."""
    return ExternalWork(
        title=title,
        doi=doi,
        year=year,
        authors=tuple(authors),
        venue="Journal of Higher Education Policy",
        providers=tuple(providers),
        discovery_status=status,
        relevance_score=0.8,
        conflicts=conflicts,
        local_match=LocalMatch(
            already_in_corpus=already_in_corpus,
            kind=LocalMatchKind.ALREADY_IN_CORPUS
            if already_in_corpus
            else LocalMatchKind.NEW,
        ),
    )


@dataclass
class ModeHarness:
    """A wired mode service plus the fakes it was built from."""

    service: ModeService
    retrieval: FakeModeRetrieval
    model: Any
    discovery: FakeDiscoveryService

    def run(self, mode: ModeName | str, query: str = "what is ai governance", **kwargs: Any):
        """Run a mode through the wired service."""
        return self.service.run_mode(mode, query, **kwargs)


def build_harness(
    *,
    evidence: Sequence[Any] | None = None,
    my_work: Sequence[Any] | None = None,
    guidance: Sequence[Any] | None = None,
    supervision: Sequence[Any] | None = None,
    model_content: str = EVIDENCE_CLAIM + ".",
    discovery_records: Sequence[ExternalWork] = (),
    discovery_available: bool = True,
    retrieval_error: Exception | None = None,
    discovery_error: Exception | None = None,
) -> ModeHarness:
    """A mode service over scripted retrieval, a scripted model and scripted discovery."""
    from app.generation.service import GenerationService
    from tests.generation_support import FakeGenerationModel

    retrieval = FakeModeRetrieval(
        evidence=list(evidence if evidence is not None else [evidence_candidate()]),
        my_work=list(my_work if my_work is not None else [my_work_candidate()]),
        guidance=list(guidance if guidance is not None else [guidance_candidate()]),
        supervision=list(
            supervision if supervision is not None else [supervision_candidate()]
        ),
        error=retrieval_error,
    )
    model = FakeGenerationModel(content=model_content)
    generation = GenerationService(retrieval_service=retrieval, model=model)
    discovery = FakeDiscoveryService(
        records=tuple(discovery_records),
        available=discovery_available,
        error=discovery_error,
    )
    service = ModeService(
        retrieval_service=retrieval,
        generation_service=generation,
        discovery_service=discovery,
    )
    return ModeHarness(service=service, retrieval=retrieval, model=model, discovery=discovery)
