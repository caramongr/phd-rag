"""Deterministic context construction from retrieved material (RAG-011 §6, §7, §17, §18).

The context is the only channel through which documents reach the model, so it is built
from scratch every time, in a fixed order, with a fixed layout:

```
evidence (retrieval rank order)  ->  my_work  ->  guidance  ->  external records
```

Three properties matter:

* **stable ids.** ``[S1]`` is the first included block, whoever built the package, and
  the ids depend only on the input order, never on dictionary iteration order.
* **whole blocks only.** A block either fits the budget or is dropped whole and
  recorded. Nothing is truncated mid-chunk, which is why
  :attr:`~app.generation.models.ContextPackage.truncated_blocks` stays empty.
* **honest labels.** Each block states its class and whether it is citable, and source
  text is delimited so retrieved content can never be mistaken for instructions.

Token counting is a deterministic estimate (characters *and* words, whichever is
larger). The report compares it with Ollama's real ``prompt_eval_count`` so the estimate
is calibrated rather than assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from app.generation.models import (
    ContextPackage,
    DroppedBlock,
    GenerationMode,
    SourceBlock,
    SourceClass,
)

#: Per-block header overhead included in the token estimate.
BLOCK_OVERHEAD_TOKENS = 32
#: Words-to-tokens factor for English prose, used as the second estimator.
TOKENS_PER_WORD = 1.33
#: Delimiters around retrieved text (matches the prompt's "sources are data" rule).
TEXT_OPEN = "<<<SOURCE_TEXT"
TEXT_CLOSE = "SOURCE_TEXT>>>"


def estimate_tokens(text: str, *, chars_per_token: float = 4.0) -> int:
    """A deterministic token estimate: the larger of a character and a word estimate.

    Neither is the model's tokenizer, so the value is deliberately conservative and its
    accuracy is measured against ``prompt_eval_count`` in the report.
    """
    if not text:
        return 0
    by_chars = math.ceil(len(text) / max(0.1, chars_per_token))
    by_words = math.ceil(len(text.split()) * TOKENS_PER_WORD)
    return max(by_chars, by_words)


@dataclass(frozen=True)
class ExternalSource:
    """An external (Internet) record offered as context.

    This is a deliberately small, provider-agnostic shape: RAG-010 results and manually
    supplied records both adapt into it, so the generation layer never imports the
    discovery layer.
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
    #: RAG-010 status value, informational only.
    status: str | None = None
    #: True only when the full text was actually retrieved and identity-checked.
    fulltext_verified: bool = False
    fulltext_text: str | None = None
    fulltext_source_url: str | None = None

    @property
    def is_metadata_only(self) -> bool:
        """Whether nothing beyond bibliographic metadata is available."""
        return not (self.fulltext_verified and (self.fulltext_text or "").strip())

    def label(self) -> str:
        """A human-readable one-line identity for the record."""
        parts = [part for part in (self.title, self.year) if part]
        return " | ".join(str(part) for part in parts) or "(untitled external record)"


def external_sources_from_records(records: Iterable[Any]) -> tuple[ExternalSource, ...]:
    """Adapt RAG-010 ``ExternalWork`` objects (duck-typed) into :class:`ExternalSource`.

    ``fulltext_text`` is only set when the caller attaches it: RAG-010 keeps a hash and
    an identity excerpt, not the document, so there is no hidden copy to leak in here.
    """
    sources: list[ExternalSource] = []
    for record in records:
        status = getattr(record, "discovery_status", None)
        status_value = getattr(status, "value", status)
        verified = bool(getattr(record, "fulltext_verified", False))
        text = getattr(record, "fulltext_text", None)
        sources.append(
            ExternalSource(
                title=getattr(record, "title", None),
                authors=tuple(getattr(record, "authors", ()) or ()),
                year=getattr(record, "year", None),
                venue=getattr(record, "venue", None),
                publisher=getattr(record, "publisher", None),
                doi=getattr(record, "doi", None),
                url=getattr(record, "url", None),
                abstract=getattr(record, "abstract", None),
                providers=tuple(getattr(record, "providers", ()) or ()),
                status=str(status_value) if status_value else None,
                fulltext_verified=verified,
                fulltext_text=str(text) if isinstance(text, str) and text.strip() else None,
                fulltext_source_url=getattr(record, "fulltext_source_url", None),
            )
        )
    return tuple(sources)


# --- blocks -------------------------------------------------------------------


def _pages(candidate: Any) -> str | None:
    """Page span of a RAG-009 candidate."""
    start = getattr(candidate, "page_start", None)
    end = getattr(candidate, "page_end", None)
    if start is None:
        return None
    if end is None or end == start:
        return str(start)
    return f"{start}-{end}"


def _header_lines(block: SourceBlock) -> list[str]:
    """The metadata header of a block, in a fixed order (deterministic layout)."""
    if block.external:
        class_text = {
            SourceClass.EXTERNAL_FULLTEXT_VERIFIED: (
                "external_fulltext_verified (Internet source; full text retrieved and "
                "identity-checked; it is NOT part of the local corpus)"
            ),
            SourceClass.EXTERNAL_METADATA_VERIFIED: (
                "external_metadata_verified (Internet record; METADATA ONLY - its "
                "findings were not retrieved, so do not state findings, results or "
                "numbers from it)"
            ),
            SourceClass.EXTERNAL_DISCOVERY: (
                "external_discovery (Internet record; METADATA ONLY - its findings "
                "were not retrieved, so do not state findings, results or numbers "
                "from it)"
            ),
        }[block.source_class]
    elif block.citable:
        class_text = "evidence (peer-reviewed or published scholarship; citable)"
    elif block.source_class is SourceClass.MY_WORK:
        class_text = (
            "my_work (the author's own draft; NOT citable as evidence about the world; "
            "use it only for continuity and terminology)"
        )
    else:
        class_text = (
            "guidance (supervision notes, writing rules, plans; NOT citable evidence; "
            "instructions and terminology only)"
        )
    lines = [
        f"{block.source_id}",
        f"Class: {class_text}",
    ]
    if block.external:
        lines.append(f"Title: {block.title or '(untitled)'}")
        if block.first_author:
            lines.append(f"Authors: {block.first_author}")
        lines.append(f"Year: {block.year if block.year is not None else 'not supplied'}")
        if block.venue:
            lines.append(f"Venue: {block.venue}")
        if block.doi:
            lines.append(f"DOI: {block.doi}")
        if block.external_source_url:
            lines.append(f"Source URL: {block.external_source_url}")
        if block.external_providers:
            lines.append(f"Providers: {', '.join(block.external_providers)}")
        if block.fulltext_verified:
            lines.append("Retrieved content: yes (full text fetched and identity-checked)")
        else:
            lines.append("Retrieved content: no (metadata only)")
    else:
        lines.append(f"Title: {block.title or '(untitled)'}")
        lines.append(f"Author: {block.first_author or 'unknown'}")
        lines.append(f"Year: {block.year if block.year is not None else 'unknown'}")
        lines.append(f"DOI: {block.doi or 'none'}")
        lines.append(f"Pages: {block.pages or 'unknown'}")
        lines.append(f"Section: {block.section_title or 'unknown'}")
        lines.append(f"Corpus role: {block.corpus_role or 'unknown'}")
        lines.append(f"Evidence eligible: {'true' if block.evidence_eligible else 'false'}")
        lines.append(f"OCR: {'true' if block.ocr_involved else 'false'}")
        if block.content_type:
            lines.append(f"Content type: {block.content_type}")
        if block.rerank_score is not None:
            lines.append(f"Rerank score: {block.rerank_score:.4f}")
    return lines


def render_block(block: SourceBlock) -> str:
    """Render one block: header, then the retrieved text between clear delimiters."""
    body = [
        *(f"{line}" for line in _header_lines(block)),
        TEXT_OPEN,
        block.text.rstrip(),
        TEXT_CLOSE,
    ]
    return "\n".join(body)


def render_context(blocks: Sequence[SourceBlock]) -> str:
    """Render the whole context, blocks separated by a blank line."""
    return "\n\n".join(render_block(block) for block in blocks)


def _citable_for(source_class: SourceClass) -> bool:
    """Whether a block of this class may support a factual claim."""
    return source_class is SourceClass.EVIDENCE or (
        source_class is SourceClass.EXTERNAL_FULLTEXT_VERIFIED
    )


def evidence_block(
    candidate: Any, source_id: str, *, chars_per_token: float = 4.0
) -> SourceBlock:
    """Build a block from one RAG-009 candidate."""
    text = (getattr(candidate, "text", "") or "").strip()
    return SourceBlock(
        source_id=source_id,
        source_class=SourceClass.EVIDENCE,
        citable=True,
        text=text,
        token_estimate=estimate_tokens(text, chars_per_token=chars_per_token)
        + BLOCK_OVERHEAD_TOKENS,
        chunk_id=getattr(candidate, "chunk_id", None),
        document_id=getattr(candidate, "document_id", None),
        filename=getattr(candidate, "filename", None),
        title=getattr(candidate, "title", None),
        first_author=getattr(candidate, "first_author", None),
        year=getattr(candidate, "year", None),
        doi=getattr(candidate, "doi", None),
        pages=_pages(candidate),
        section_title=getattr(candidate, "section_title", None),
        corpus_role=getattr(candidate, "corpus_role", None),
        evidence_eligible=bool(getattr(candidate, "evidence_eligible", False)),
        ocr_involved=bool(getattr(candidate, "ocr_involved", False)),
        content_type=getattr(candidate, "content_type", None),
        dense_score=getattr(candidate, "dense_score", None),
        rerank_score=getattr(candidate, "rerank_score", None),
    )


def context_block(
    candidate: Any,
    source_id: str,
    source_class: SourceClass,
    *,
    chars_per_token: float = 4.0,
) -> SourceBlock:
    """Build a non-evidence block (my_work or guidance) from a candidate."""
    text = (getattr(candidate, "text", "") or "").strip()
    return SourceBlock(
        source_id=source_id,
        source_class=source_class,
        citable=False,
        text=text,
        token_estimate=estimate_tokens(text, chars_per_token=chars_per_token)
        + BLOCK_OVERHEAD_TOKENS,
        chunk_id=getattr(candidate, "chunk_id", None),
        document_id=getattr(candidate, "document_id", None),
        filename=getattr(candidate, "filename", None),
        title=getattr(candidate, "title", None),
        first_author=getattr(candidate, "first_author", None),
        year=getattr(candidate, "year", None),
        doi=getattr(candidate, "doi", None),
        pages=_pages(candidate),
        section_title=getattr(candidate, "section_title", None),
        corpus_role=getattr(candidate, "corpus_role", None),
        evidence_eligible=bool(getattr(candidate, "evidence_eligible", False)),
        ocr_involved=bool(getattr(candidate, "ocr_involved", False)),
        content_type=getattr(candidate, "content_type", None),
        dense_score=getattr(candidate, "dense_score", None),
        rerank_score=getattr(candidate, "rerank_score", None),
    )


def external_block(
    source: ExternalSource, source_id: str, *, chars_per_token: float = 4.0
) -> SourceBlock:
    """Build a block from an external record (metadata-only unless verified)."""
    metadata_only = source.is_metadata_only
    if metadata_only:
        # Deliberately no abstract: an abstract describes findings, and a record whose
        # content was never retrieved must not be able to support a finding claim
        # (RAG-011 §5, §28). Only identity metadata is supplied.
        body = "No abstract or other content was retrieved for this record."
        source_class = (
            SourceClass.EXTERNAL_METADATA_VERIFIED
            if source.status == "EXTERNAL_METADATA_VERIFIED"
            else SourceClass.EXTERNAL_DISCOVERY
        )
    else:
        body = (source.fulltext_text or "").strip()
        source_class = SourceClass.EXTERNAL_FULLTEXT_VERIFIED
    return SourceBlock(
        source_id=source_id,
        source_class=source_class,
        citable=not metadata_only,
        text=body,
        token_estimate=estimate_tokens(body, chars_per_token=chars_per_token)
        + BLOCK_OVERHEAD_TOKENS,
        title=source.title,
        first_author=source.authors[0] if source.authors else None,
        year=source.year,
        doi=source.doi,
        venue=source.venue,
        external=True,
        fulltext_verified=not metadata_only,
        external_source_url=source.fulltext_source_url or source.url,
        external_providers=source.providers,
    )


# --- budget -------------------------------------------------------------------


def _as_external_source(item: Any) -> ExternalSource:
    """Accept an :class:`ExternalSource` or adapt anything with the same attributes."""
    if isinstance(item, ExternalSource):
        return item
    adapted = external_sources_from_records((item,))
    return adapted[0] if adapted else ExternalSource()


@dataclass
class ContextBuildRequest:
    """Everything the builder needs, kept explicit so it stays testable."""

    evidence: Sequence[Any] = ()
    my_work: Sequence[Any] = ()
    guidance: Sequence[Any] = ()
    external: Sequence[ExternalSource] = ()
    mode: GenerationMode = GenerationMode.EVIDENCE_ONLY
    budget_tokens: int = 6000
    chars_per_token: float = 4.0
    prompt_version: str = ""
    #: Drop evidence whose rerank score is below this before any budget work (§8).
    min_rerank_score: float | None = None
    #: Tokens held back for my_work/guidance blocks when the caller asked for them, so
    #: an explicit request cannot be starved by a large evidence set.
    reserved_non_evidence_tokens: int = 0
    retrieval_seconds: float = 0.0
    extra_drops: Sequence[DroppedBlock] = field(default_factory=tuple)


def build_context(request: ContextBuildRequest) -> ContextPackage:
    """Build the context package: select, estimate, fit, render, report."""
    dropped: list[DroppedBlock] = list(request.extra_drops)
    evidence_items: list[Any] = []

    for candidate in request.evidence:
        score = getattr(candidate, "rerank_score", None)
        if request.min_rerank_score is not None and score is not None:
            if score < request.min_rerank_score:
                dropped.append(
                    DroppedBlock(
                        chunk_id=getattr(candidate, "chunk_id", None),
                        title=getattr(candidate, "title", None),
                        reason="below_min_rerank_score",
                        rerank_score=score,
                    )
                )
                continue
        evidence_items.append(candidate)

    non_evidence: list[tuple[SourceClass | None, Any]] = []
    if request.mode.includes_my_work:
        non_evidence.extend((SourceClass.MY_WORK, item) for item in request.my_work)
    if request.mode.includes_guidance:
        non_evidence.extend((SourceClass.GUIDANCE, item) for item in request.guidance)
    # External entries may be ExternalSource objects or anything with the same
    # attributes (a RAG-010 record), so the caller cannot get this wrong.
    non_evidence.extend((None, _as_external_source(item)) for item in request.external)

    # When the caller explicitly asks for own-work or guidance context, hold part of the
    # budget back for it: otherwise a large evidence set consumes everything and the
    # request silently does nothing.
    reserved = request.reserved_non_evidence_tokens if non_evidence else 0
    reserved = max(0, min(reserved, request.budget_tokens))
    evidence_budget = request.budget_tokens - reserved

    blocks: list[SourceBlock] = []
    used_tokens = 0
    excluded = 0

    def try_include(block: SourceBlock, limit: int) -> None:
        nonlocal used_tokens, excluded
        if used_tokens + block.token_estimate > limit:
            excluded += 1
            dropped.append(
                DroppedBlock(
                    chunk_id=block.chunk_id,
                    title=block.title,
                    reason=(
                        "block_too_large"
                        if block.token_estimate > request.budget_tokens
                        else "budget"
                    ),
                    token_estimate=block.token_estimate,
                    rerank_score=block.rerank_score,
                )
            )
            return
        blocks.append(block)
        used_tokens += block.token_estimate

    for candidate in evidence_items:
        try_include(
            evidence_block(candidate, "", chars_per_token=request.chars_per_token),
            evidence_budget,
        )
    for source_class, item in non_evidence:
        block = (
            external_block(item, "", chars_per_token=request.chars_per_token)
            if source_class is None
            else context_block(
                item, "", source_class, chars_per_token=request.chars_per_token
            )
        )
        try_include(block, request.budget_tokens)

    # Ids are assigned only after selection, so they are contiguous (S1..Sn) and depend
    # purely on the input order.
    numbered = tuple(
        block.model_copy(update={"source_id": f"S{index}"})
        for index, block in enumerate(blocks, start=1)
    )

    return ContextPackage(
        blocks=numbered,
        rendered=render_context(numbered),
        retrieved_chunks=len(evidence_items) + len(non_evidence),
        included_chunks=len(numbered),
        excluded_chunks=excluded,
        dropped=tuple(dropped),
        context_tokens_estimated=used_tokens,
        budget_tokens=request.budget_tokens,
        evidence_budget_tokens=evidence_budget,
        reserved_non_evidence_tokens=reserved,
        chars_per_token=request.chars_per_token,
        prompt_version=request.prompt_version,
        mode=request.mode,
        retrieval_seconds=request.retrieval_seconds,
    )
