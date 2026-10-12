"""Shared fixtures for the RAG-012 citation tests.

Everything here is offline and deterministic. The context package is built with the real
RAG-011 :class:`~app.generation.models.SourceBlock`/:class:`~app.generation.models.
ContextPackage` models (not stand-ins), so the validator is exercised against the exact
shape the generation stage produces.
"""

from __future__ import annotations

from typing import Any, Sequence

from app.generation.models import (
    ContextPackage,
    GenerationAudit,
    GenerationMode,
    GenerationResponse,
    GenerationStatus,
    SourceBlock,
    SourceClass,
    SourceMarkerAudit,
)

#: The RAG-011 evidence sentence used in the one-source tests, and its claim.
SENIOR_TEXT = (
    "Senior management holds significant responsibility in the governance dimension as "
    "they are instrumental in policy formulation, the establishment of ethical "
    "guidelines, and overall institutional direction of AI integration."
)
SENIOR_CLAIM = (
    "AI governance refers to senior management responsibility for policy, ethical "
    "guidelines and institutional direction [S1]"
)

#: Two findings from one source, which RAG-011 answered as a single sentence (§15). The
#: first three sentences mirror the real chunk the RAG-011 answer cited (S3 of the
#: academic-integrity question): one finding about faculty training, one about assignments
#: and critical thinking, and one about regulating access to prevent over-reliance.
FACULTY_TEXT = (
    "The majority of faculty, 56.7%, consider training on ethical AI usage as highly "
    "important. These strategies include designing assignments that encourage critical "
    "thinking and creativity, which received 69 endorsements (69%). Finally, 12% advocate "
    "for regulating AI access to prevent over-reliance and maintain critical thinking "
    "skills."
)
FACULTY_CLAIM = (
    "A majority of faculty consider ethical AI training highly important, with many "
    "advocating assignments that foster critical thinking to mitigate overreliance [S3]"
)


def block(
    source_id: str = "S1",
    *,
    text: str = SENIOR_TEXT,
    source_class: SourceClass = SourceClass.EVIDENCE,
    citable: bool = True,
    title: str | None = "A Paper About AI Governance",
    first_author: str | None = "Jane Doe",
    year: int | None = 2024,
    doi: str | None = None,
    pages: str | None = "4-6",
    chunk_id: str | None = None,
    document_id: str | None = None,
    venue: str | None = None,
    section_title: str | None = "Introduction",
    corpus_role: str | None = "literature",
    evidence_eligible: bool = True,
    ocr_involved: bool = False,
    content_type: str | None = "body",
) -> SourceBlock:
    """A real source block, built the way RAG-011 builds one."""
    return SourceBlock(
        source_id=source_id,
        source_class=source_class,
        citable=citable,
        text=text,
        token_estimate=64,
        chunk_id=chunk_id or f"chunk-{source_id.casefold()}",
        document_id=document_id or f"doc-{source_id.casefold()}",
        filename=f"{source_id.casefold()}.pdf",
        title=title,
        first_author=first_author,
        year=year,
        doi=doi,
        pages=pages,
        section_title=section_title,
        corpus_role=corpus_role,
        evidence_eligible=evidence_eligible,
        ocr_involved=ocr_involved,
        content_type=content_type,
        dense_score=0.66,
        rerank_score=0.75,
        venue=venue,
    )


def make_context(
    blocks: Sequence[SourceBlock],
    *,
    mode: GenerationMode = GenerationMode.EVIDENCE_ONLY,
) -> ContextPackage:
    """A real context package over the supplied blocks."""
    return ContextPackage(
        blocks=tuple(blocks),
        rendered="",
        retrieved_chunks=len(blocks),
        included_chunks=len(blocks),
        excluded_chunks=0,
        context_tokens_estimated=64 * len(blocks),
        budget_tokens=6000,
        mode=mode,
    )


def make_response(
    answer: str,
    blocks: Sequence[SourceBlock],
    *,
    question: str = "What is AI governance in higher education?",
    status: GenerationStatus = GenerationStatus.OK,
    non_evidence_reports: Sequence[str] = (),
    invalid_markers: Sequence[str] = (),
    warnings: Sequence[str] = (),
    mode: GenerationMode = GenerationMode.EVIDENCE_ONLY,
) -> GenerationResponse:
    """A RAG-011 response carrying the supplied answer and sources."""
    context = make_context(blocks, mode=mode)
    return GenerationResponse(
        question=question,
        answer=answer,
        mode=mode,
        status=status,
        source_map=context.source_map(),
        used_source_ids=context.citable_source_ids,
        marker_audit=SourceMarkerAudit(
            marker_counts={block_.source_id: 1 for block_ in blocks},
            invalid_markers=tuple(invalid_markers),
            non_evidence_reports=tuple(non_evidence_reports),
        ),
        warnings=tuple(warnings),
        context=context,
        generation_audit=GenerationAudit(model="test-model", prompt_version="v2"),
    )


def validation_of(response: GenerationResponse, **kwargs: Any):
    """Validate a response with the default (deterministic) citation service."""
    from app.citations.service import CitationService

    return CitationService(**kwargs).validate(response)
