"""Shared helpers for the RAG-011 generation tests.

Everything here is offline and deterministic: a scripted generation model replaces
Ollama, real RAG-009 ``Candidate``/``RetrievalResponse`` objects replace a fake
retrieval shape (so the retrieval audit under test is the real one), and the local
corpus is an in-memory Qdrant collection built by the RAG-009 test helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from app.generation.models import ModelFailure
from app.generation.ollama import ChatMessage, ChatResult, GenerationOptions
from app.retrieval.models import (
    Candidate,
    DroppedCandidate,
    QueryAudit,
    RetrievalMode,
    RetrievalResponse,
    RetrievalStats,
)


def candidate(
    chunk_id: str = "chunk-1",
    *,
    text: str = "Artificial intelligence governance is discussed in this passage.",
    title: str = "A Paper About AI",
    first_author: str = "Jane Doe",
    year: int | None = 2024,
    doi: str | None = "10.1234/abc",
    page_start: int | None = 4,
    page_end: int | None = 6,
    section_title: str | None = "Introduction",
    corpus_role: str = "literature",
    evidence_eligible: bool = True,
    content_type: str = "body",
    ocr_involved: bool = False,
    rerank_score: float | None = 0.75,
    dense_score: float | None = 0.66,
    final_rank: int = 1,
    document_id: str | None = None,
    filename: str | None = None,
) -> Candidate:
    """A real RAG-009 candidate, so generation consumes the same shape it will see."""
    return Candidate(
        chunk_id=chunk_id,
        document_id=document_id or f"doc-{chunk_id}",
        file_hash=f"hash-{chunk_id}",
        filename=filename or f"{chunk_id}.pdf",
        chunk_index=0,
        dense_score=dense_score,
        rerank_score=rerank_score,
        dense_rank=final_rank,
        final_rank=final_rank,
        text=text,
        retrieval_text=text,
        title=title,
        first_author=first_author,
        year=year,
        doi=doi,
        page_start=page_start,
        page_end=page_end,
        section_title=section_title,
        content_type=content_type,
        corpus_role=corpus_role,
        evidence_eligible=evidence_eligible,
        retrieval_eligible=True,
        ocr_involved=ocr_involved,
    )


def query_audit(
    query: str = "what is ai governance",
    *,
    mode: RetrievalMode = RetrievalMode.EVIDENCE,
    candidate_k: int = 25,
    final_k: int = 8,
) -> QueryAudit:
    """A minimal but real RAG-009 query audit."""
    return QueryAudit(
        raw_query=query,
        normalized_query=query,
        retrieval_mode=mode,
        filters={},
        candidate_k=candidate_k,
        final_k=final_k,
        rerank_enabled=True,
        rerank_model="fake/bge-reranker-v2-m3",
        embedding_model="fake/bge-m3",
        collection="test_collection",
        max_chunks_per_document=3,
        suppress_duplicates=True,
        duplicate_overlap_threshold=0.85,
        timestamp="2026-01-01T00:00:00+00:00",
    )


def retrieval_response(
    results: Sequence[Candidate] = (),
    *,
    query: str = "what is ai governance",
    mode: RetrievalMode = RetrievalMode.EVIDENCE,
    dropped: Sequence[DroppedCandidate] = (),
    candidate_k: int = 25,
    final_k: int = 8,
) -> RetrievalResponse:
    """A real RAG-009 response carrying the supplied candidates."""
    return RetrievalResponse(
        audit=query_audit(query, mode=mode, candidate_k=candidate_k, final_k=final_k),
        results=list(results),
        dense_candidates=list(results),
        dropped=list(dropped),
        stats=RetrievalStats(
            candidates_retrieved=len(results),
            candidates_after_filter=len(results),
            returned=len(results),
            embedding_seconds=0.01,
            search_seconds=0.02,
            rerank_seconds=0.03,
            total_seconds=0.06,
        ),
    )


class FakeRetrievalService:
    """A RAG-009 double that returns scripted responses per mode."""

    def __init__(
        self,
        *,
        evidence: Sequence[Candidate] = (),
        my_work: Sequence[Candidate] = (),
        guidance: Sequence[Candidate] = (),
        dropped: Sequence[DroppedCandidate] = (),
        error: Exception | None = None,
    ) -> None:
        self.evidence = list(evidence)
        self.my_work = list(my_work)
        self.guidance = list(guidance)
        self.dropped = list(dropped)
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def retrieve(self, query: str, **kwargs: Any) -> RetrievalResponse:
        """Return the scripted response for the requested mode."""
        self.calls.append({"query": query, **kwargs})
        if self.error is not None:
            raise self.error
        mode = kwargs.get("mode") or RetrievalMode.EVIDENCE
        mode = RetrievalMode(mode)
        if mode is RetrievalMode.MY_WORK:
            return retrieval_response(self.my_work, query=query, mode=mode)
        if mode in (RetrievalMode.GUIDANCE, RetrievalMode.SUPERVISION):
            return retrieval_response(self.guidance, query=query, mode=mode)
        return retrieval_response(
            self.evidence,
            query=query,
            mode=mode,
            dropped=self.dropped,
            candidate_k=kwargs.get("candidate_k") or 25,
            final_k=kwargs.get("final_k") or 8,
        )

    def close(self) -> None:
        """Match the real service lifecycle."""


@dataclass
class FakeGenerationModel:
    """A scripted generation model: no Ollama, no network, fully deterministic."""

    content: str = "Grounded answer [S1]."
    failure: ModelFailure | None = None
    generated_tokens: int = 20
    generation_seconds: float = 0.5
    prompt_tokens: int = 100
    prompt_eval_seconds: float = 0.02
    done_reason: str = "stop"
    thinking: str | None = None
    model_details: dict[str, Any] = field(
        default_factory=lambda: {
            "model": "fake/qwen",
            "family": "fake",
            "parameter_size": "34.7B",
            "quantization_level": "Q4_K_M",
            "ollama_version": "0.35.1",
        }
    )
    think: bool = False
    provider: str = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    name: str = "fake:qwen"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        options: GenerationOptions | None = None,
        think: bool | None = None,
    ) -> ChatResult:
        """Record the call and return the scripted result."""
        self.calls.append(
            {
                "messages": list(messages),
                "options": options or GenerationOptions(),
                "think": self.think if think is None else think,
            }
        )
        return ChatResult(
            model=self.name,
            content="" if self.failure else self.content,
            thinking=self.thinking,
            done_reason=self.done_reason,
            prompt_tokens=self.prompt_tokens,
            generated_tokens=self.generated_tokens,
            prompt_eval_seconds=self.prompt_eval_seconds,
            generation_seconds=self.generation_seconds,
            load_seconds=0.0,
            total_seconds=self.generation_seconds,
            wall_seconds=self.generation_seconds,
            model_details=dict(self.model_details),
            failure=self.failure,
        )

    def describe(self) -> dict[str, Any]:
        """Report the scripted model metadata."""
        return dict(self.model_details)

    @property
    def last_prompt(self) -> str:
        """The user message of the most recent call."""
        if not self.calls:
            return ""
        return self.calls[-1]["messages"][-1].content

    @property
    def last_system_prompt(self) -> str:
        """The system message of the most recent call."""
        if not self.calls:
            return ""
        return self.calls[0]["messages"][0].content


def build_service(
    *,
    retrieval: Any | None = None,
    model: FakeGenerationModel | None = None,
    **settings_overrides: Any,
) -> Any:
    """A :class:`GenerationService` wired to the doubles, with test-friendly settings."""
    from app.config import Settings
    from app.generation.service import GenerationService

    settings = Settings(**settings_overrides)
    return GenerationService(
        retrieval_service=retrieval if retrieval is not None else FakeRetrievalService(),
        model=model if model is not None else FakeGenerationModel(),
        settings=settings,
    )


@dataclass(frozen=True)
class FakeExternalRecord:
    """An external record shaped like a RAG-010 result (duck-typed on purpose).

    The generation layer reads external records through ``getattr``, so this stands in
    for :class:`app.discovery.models.ExternalWork` and carries the extra
    ``fulltext_text`` attribute a caller may attach when it really did retrieve content.
    """

    title: str | None = None
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str | None = None
    publisher: str | None = None
    doi: str | None = None
    url: str | None = None
    abstract: str | None = None
    providers: tuple[str, ...] = ()
    discovery_status: Any = None
    fulltext_verified: bool = False
    fulltext_text: str | None = None
    fulltext_source_url: str | None = None


def external_record(
    *,
    title: str = "A 2026 Paper On AI Governance",
    year: int = 2026,
    doi: str = "10.1234/external",
    venue: str = "Journal of External Studies",
    authors: Sequence[str] = ("Jane Doe",),
    abstract: str | None = "An abstract that mentions findings.",
    status: str = "EXTERNAL_METADATA_VERIFIED",
    fulltext_verified: bool = False,
    fulltext_text: str | None = None,
) -> FakeExternalRecord:
    """An external record (metadata-only unless a retrieved full text is attached)."""
    from app.discovery.models import DiscoveryStatus

    return FakeExternalRecord(
        title=title,
        authors=tuple(authors),
        year=year,
        doi=doi,
        venue=venue,
        abstract=abstract,
        providers=("crossref", "openalex"),
        discovery_status=DiscoveryStatus(status),
        fulltext_verified=fulltext_verified,
        fulltext_text=fulltext_text,
        fulltext_source_url="https://example.org/paper.pdf" if fulltext_verified else None,
    )


def real_external_work(**kwargs: Any) -> Any:
    """A real RAG-010 ``ExternalWork``, to prove the adapter works on the real shape."""
    from app.discovery.models import DiscoveryStatus, ExternalWork

    status = kwargs.pop("status", "EXTERNAL_METADATA_VERIFIED")
    return ExternalWork(discovery_status=DiscoveryStatus(status), **kwargs)
