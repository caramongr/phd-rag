"""Structure-preserving chunking of parsed documents (RAG-007).

The chunker consumes what RAG-003/RAG-004/RAG-005 already produced
(:class:`~app.parsing.models.ParsedDocument`) plus, when available, the RAG-006
metadata for the same logical document. It never re-parses a file and never
touches the parsers.

How it works: content items are consumed in the canonical source order; headings
are *structure*, not content, so they set the section context (``section_title``
and ``heading_path``) instead of becoming isolated chunks; prose and captions
accumulate paragraph by paragraph until the soft target is reached, never
crossing a section boundary and never exceeding the hard maximum; tables stand
alone; a paragraph larger than the hard maximum is split at sentence boundaries;
and a short section is merged into an adjacent sibling only when both sit
together under one parent and the merge stays inside the target band. Every chunk
keeps its page range, source locators, extraction sources and corpus-role policy,
so no chunk is ever anonymous text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Sequence
from uuid import UUID, uuid5

from app.chunking.classify import (
    SectionKind,
    looks_like_caption,
    reference_intro_line,
    section_kind,
)
from app.chunking.models import (
    REASON_DUPLICATE_TABLE_TEXT,
    REASON_EMPTY_AFTER_NORMALIZATION,
    REASON_HEADING_WITHOUT_CONTENT,
    ChunkedDocument,
    ChunkingOptions,
    SkippedItem,
)
from app.chunking.normalize import (
    NORMALIZATION_VERSION,
    RunningLineReport,
    detect_running_lines,
    normalize_block_text,
    normalize_inline,
    strip_running_lines,
)
from app.chunking.policy import eligibility_for, policy_for_role
from app.metadata.corpus import corpus_role_for_path
from app.metadata.enums import CorpusRole, PublicationType
from app.metadata.models import AcademicMetadata
from app.models.chunk import Chunk, ChunkMetadataRef
from app.models.enums import ContentType
from app.parsing.models import (
    ContentItem,
    ContentItemType,
    ParsedBlock,
    ParsedBlockType,
    ParsedDocument,
    ParsedTable,
    TextSource,
)

#: Fixed namespaces, so chunk and document ids are reproducible from content alone.
DOCUMENT_ID_NAMESPACE = UUID("6d1f0b7a-3f0e-4d4c-9a2b-7c8e9f0a1b2c")
CHUNK_ID_NAMESPACE = UUID("9b2a7c4e-8d1f-4a3b-8c6e-2f4d7e9a1c3b")

#: Sentence boundary used when a paragraph must be split.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

#: Shortest collapsed text the table-duplication substring rule may act on.
_MIN_DUPLICATE_SUBSTRING_CHARS = 20

_PARAGRAPH_JOIN = "\n\n"


def document_id_for(file_hash: str) -> UUID:
    """Return the deterministic logical-document id for ``file_hash``."""
    return uuid5(DOCUMENT_ID_NAMESPACE, file_hash)


def chunk_id_for(file_hash: str, chunk_index: int, text: str) -> UUID:
    """Return the deterministic id for a chunk's identity and content."""
    digest = sha256(text.encode("utf-8")).hexdigest()[:32]
    return uuid5(CHUNK_ID_NAMESPACE, f"{file_hash}:{chunk_index}:{digest}")


@dataclass
class _TableCoverage:
    """The text a page's tables already represent, for duplicate detection."""

    corpus: str
    cells: frozenset[str]


@dataclass
class _Item:
    """One normalized, classified piece of content ready for accumulation."""

    sequence_index: int
    text: str
    content_type: ContentType
    section: SectionKind
    heading_path: tuple[str, ...]
    page_number: int | None = None
    source_locator: str | None = None
    text_source: TextSource | None = None
    token_count: int = 0


@dataclass
class _Buffer:
    """Accumulated items for one section of one content type."""

    heading_path: tuple[str, ...]
    section_title: str | None
    section: SectionKind
    items: list[_Item] = field(default_factory=list)
    tokens: int = 0
    split_from_previous: bool = False

    @property
    def pages(self) -> list[int]:
        """Page numbers of the contributing items."""
        return [item.page_number for item in self.items if item.page_number is not None]

    @property
    def content_types(self) -> set[ContentType]:
        """Content types of the contributing items."""
        return {item.content_type for item in self.items}

    @property
    def merged_sections(self) -> list[str]:
        """Section titles this chunk covers beyond its own section title.

        Derived from the items themselves, so the list is empty unless a short
        sibling section was genuinely merged in.
        """
        titles = [
            item.heading_path[-1] for item in self.items if item.heading_path
        ]
        return [
            title
            for title in dict.fromkeys(titles)
            if title and title != self.section_title
        ]

    @property
    def text(self) -> str:
        """The chunk text: contributing paragraphs joined with a blank line."""
        return _PARAGRAPH_JOIN.join(item.text for item in self.items if item.text)


def _table_rows_text(table: ParsedTable) -> str:
    """Render a table deterministically as pipe-separated rows."""
    lines: list[str] = []
    for row in table.rows:
        cells = [normalize_inline(cell.text).replace("|", "\\|") for cell in row.cells]
        if not any(cell.strip() for cell in cells):
            continue
        lines.append(" | ".join(cells).strip())
    return "\n".join(lines)


def _table_coverage(tables: Sequence[ParsedTable]) -> _TableCoverage:
    """Build the searchable text of the tables found on one page."""
    cells: list[str] = []
    normalized_cells: set[str] = set()
    for table in tables:
        for row in table.rows:
            for cell in row.cells:
                cleaned = normalize_inline(cell.text)
                if not cleaned:
                    continue
                cells.append(cleaned)
                normalized_cells.add(cleaned)
    return _TableCoverage(corpus=" ".join(cells), cells=frozenset(normalized_cells))


def _is_duplicated_by_table(block_text: str, coverage: _TableCoverage) -> bool:
    """Whether a prose block's text is already represented by a table.

    Two deterministic rules, both conservative: the whole block appears verbatim
    inside the page's table cell text, or every line of the block equals a table
    cell exactly.
    """
    if not coverage.cells:
        return False
    normalized = normalize_block_text(block_text)
    if not normalized:
        return False
    collapsed = " ".join(normalized.split())
    if len(collapsed) >= _MIN_DUPLICATE_SUBSTRING_CHARS and collapsed in coverage.corpus:
        return True
    lines = [line for line in normalized.split("\n") if line.strip()]
    if not lines or len(lines) > 40:
        return False
    return all(line in coverage.cells for line in lines)


def _ordered_content_items(parsed: ParsedDocument) -> list[ContentItem]:
    """Return the canonical content stream, synthesising one if the parser left none."""
    if parsed.content_items:
        return sorted(parsed.content_items, key=lambda item: item.sequence_index)
    items: list[ContentItem] = [
        ContentItem(
            sequence_index=block.sequence_index,
            item_type=ContentItemType.BLOCK,
            block=block,
        )
        for block in parsed.blocks
    ]
    items.extend(
        ContentItem(
            sequence_index=table.sequence_index,
            item_type=ContentItemType.TABLE,
            table=table,
        )
        for table in parsed.tables
    )
    return sorted(items, key=lambda item: item.sequence_index)


def split_oversized_text(
    text: str, *, limit: int, options: ChunkingOptions
) -> list[str]:
    """Split oversized text at sentence boundaries, falling back to a token cut.

    Sentences are packed back up to ``limit`` tokens; a single sentence that still
    exceeds ``limit`` (pathological prose, or a table row with one huge cell) is
    cut at an exact token boundary, which is why no chunk can exceed the ceiling.
    """
    tokenizer = options.tokenizer
    pieces: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if tokenizer.count(sentence) <= limit:
            if (
                pieces
                and tokenizer.count(pieces[-1]) + 1 + tokenizer.count(sentence) <= limit
            ):
                pieces[-1] = f"{pieces[-1]} {sentence}"
            else:
                pieces.append(sentence)
            continue
        remaining = sentence
        while tokenizer.count(remaining) > limit:
            head = tokenizer.truncate(remaining, max_tokens=limit).strip()
            if not head:
                break
            pieces.append(head)
            remaining = remaining[len(head) :].strip()
        if remaining:
            pieces.append(remaining)
    return pieces or [text]


def _context_label(heading_path: tuple[str, ...], document_title: str | None) -> str:
    """Build the synthetic context label used for a chunk's retrieval text."""
    lines: list[str] = []
    if document_title:
        lines.append(f"[Document: {document_title}]")
    if heading_path:
        lines.append(f"[Section: {' > '.join(heading_path)}]")
    return "\n".join(lines)


def _common_parent(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    """Return the shared heading prefix of two heading trails."""
    shared: list[str] = []
    for left_part, right_part in zip(left, right):
        if left_part != right_part:
            break
        shared.append(left_part)
    return tuple(shared)


def _buffer_content_type(buffer: _Buffer) -> ContentType:
    """Derive a chunk's content type from the items it actually contains."""
    types = buffer.content_types
    if types == {ContentType.TABLE}:
        return ContentType.TABLE
    if types == {ContentType.CAPTION}:
        return ContentType.CAPTION
    types = types - {ContentType.CAPTION}
    if len(types) == 1:
        return next(iter(types))
    if len(types) > 1:
        return ContentType.OTHER
    return ContentType.OTHER


def _metadata_ref(
    parsed: ParsedDocument,
    metadata: AcademicMetadata | None,
    *,
    file_hash: str,
) -> ChunkMetadataRef:
    """Build the minimal bibliographic payload carried by every chunk."""
    title = None
    year = None
    first_author = None
    doi = None
    publication_type: PublicationType | None = None
    duplicate_locations: list[str] = []
    role_resolution: str | None = None

    if metadata is not None:
        if metadata.title is not None:
            title = metadata.title.value
        if metadata.year is not None:
            year = metadata.year.value
        if metadata.authors is not None and metadata.authors.value:
            first_author = metadata.authors.value[0].full_name
        if metadata.doi is not None:
            doi = metadata.doi.value
        if metadata.publication_type is not None:
            publication_type = metadata.publication_type.value
        resolution = metadata.role_resolution
        if resolution is not None and len(resolution.physical_roles) > 1:
            duplicate_locations = resolution.physical_paths
            role_resolution = resolution.describe()
    if title is None and parsed.title:
        title = normalize_inline(parsed.title)

    return ChunkMetadataRef(
        file_hash=file_hash,
        filename=parsed.filename,
        title=title,
        year=year,
        first_author=first_author,
        doi=doi,
        publication_type=publication_type,
        duplicate_locations=duplicate_locations,
        role_resolution=role_resolution,
    )


class DocumentChunker:
    """Chunk one parsed document according to the RAG-007 rules."""

    def __init__(self, options: ChunkingOptions | None = None) -> None:
        self.options = options or ChunkingOptions()
        self._reset()

    def _reset(self) -> None:
        self._buffers: list[_Buffer] = []
        self._current: _Buffer | None = None
        self._skipped: list[SkippedItem] = []
        self._duplicates_removed = 0
        self._running_lines_removed = 0
        self._headings_without_content = 0
        self._pending_heading_path: tuple[str, ...] = ()
        self._pending_heading_active = False
        self._content_since_heading = False
        self._document_extension = ""

    # -- public API ---------------------------------------------------------------
    def chunk(
        self,
        parsed: ParsedDocument,
        *,
        metadata: AcademicMetadata | None = None,
        file_hash: str | None = None,
        corpus_role: CorpusRole | None = None,
    ) -> ChunkedDocument:
        """Chunk ``parsed`` and return the chunks together with their audit trail."""
        self._reset()
        self._document_extension = parsed.extension

        effective_hash = (
            file_hash
            or parsed.file_hash
            or sha256(parsed.raw_text.encode("utf-8")).hexdigest()
        )
        role = corpus_role
        if role is None and metadata is not None:
            role = metadata.corpus_role
        if role is None:
            role = corpus_role_for_path(parsed.source_path)

        running = (
            detect_running_lines(self._page_texts(parsed))
            if self.options.strip_running_lines
            else RunningLineReport(frozenset(), 0, 0)
        )

        self._consume(parsed, running, role)
        self._close_buffer()
        self._finish_heading_tracking()
        if self.options.merge_short_sections:
            self._merge_short_sections()

        document_id = document_id_for(effective_hash)
        chunks = self._build_chunks(
            parsed, metadata, document_id=document_id, file_hash=effective_hash, role=role
        )
        return ChunkedDocument(
            document_id=document_id,
            file_hash=effective_hash,
            filename=parsed.filename,
            extension=parsed.extension,
            chunks=chunks,
            skipped_items=self._skipped,
            duplicate_table_blocks_removed=self._duplicates_removed,
            running_lines_removed=self._running_lines_removed,
            headings_without_content=self._headings_without_content,
            normalization_version=NORMALIZATION_VERSION,
            tokenizer_name=self.options.tokenizer.name,
        )

    @staticmethod
    def _page_texts(parsed: ParsedDocument) -> list[tuple[int, str]]:
        """Effective page texts for running-header/footer detection."""
        return [
            (page.page_number, page.raw_text)
            for page in parsed.pages
            if page.raw_text and page.raw_text.strip()
        ]

    # -- consumption --------------------------------------------------------------
    def _consume(
        self, parsed: ParsedDocument, running: RunningLineReport, role: CorpusRole
    ) -> None:
        coverage_by_page: dict[int, _TableCoverage] = {}
        if self.options.deduplicate_table_text:
            tables_by_page: dict[int, list[ParsedTable]] = {}
            for table in parsed.tables:
                if table.page_number is not None:
                    tables_by_page.setdefault(table.page_number, []).append(table)
            coverage_by_page = {
                page: _table_coverage(tables) for page, tables in tables_by_page.items()
            }

        page_sources = {page.page_number: page.text_source for page in parsed.pages}

        for item in _ordered_content_items(parsed):
            if item.item_type is ContentItemType.TABLE and item.table is not None:
                self._consume_table(item.table, role, page_sources)
                continue
            block = item.block
            if block is None:
                continue
            if block.block_type is ParsedBlockType.HEADING:
                self._consume_heading(block)
                continue
            self._consume_block(
                block, role, running, coverage_by_page.get(block.page_number or -1)
            )

    def _consume_heading(self, block: ParsedBlock) -> None:
        """Headings give structure: they mark a boundary and set the context."""
        title = normalize_inline(block.text)
        if not title:
            return
        parser_path = tuple(normalize_inline(part) for part in block.heading_path if part)
        if parser_path and parser_path[-1] == title:
            path = parser_path
        elif parser_path:
            path = parser_path + (title,)
        else:
            path = self._pending_heading_path + (title,)

        if self._pending_heading_active and not self._content_since_heading:
            self._headings_without_content += 1
            self._skipped.append(
                SkippedItem(
                    sequence_index=block.sequence_index,
                    reason=REASON_HEADING_WITHOUT_CONTENT,
                    source_locator=block.source_locator,
                )
            )
        self._close_buffer()
        self._pending_heading_path = path
        self._pending_heading_active = True
        self._content_since_heading = False

    def _finish_heading_tracking(self) -> None:
        """Count a trailing heading that never received content."""
        if self._pending_heading_active and not self._content_since_heading:
            self._headings_without_content += 1

    def _consume_table(
        self,
        table: ParsedTable,
        role: CorpusRole,
        page_sources: dict[int, TextSource],
    ) -> None:
        text = _table_rows_text(table)
        if not text:
            return
        path = tuple(normalize_inline(part) for part in table.heading_path if part)
        if not path:
            path = self._pending_heading_path
        text_source = page_sources.get(table.page_number) if table.page_number else None
        section = section_kind(path)
        item = _Item(
            sequence_index=table.sequence_index,
            text=text,
            # A table inside a reference list is still bibliography content, so it
            # must not become retrieval-eligible evidence.
            content_type=(
                ContentType.REFERENCES if section is SectionKind.REFERENCES else ContentType.TABLE
            ),
            section=section,
            heading_path=path,
            page_number=table.page_number,
            source_locator=table.source_locator,
            text_source=text_source,
            token_count=self.options.tokenizer.count(text),
        )
        self._content_since_heading = True
        self._close_buffer()
        limit = self.options.hard_max_tokens
        if item.token_count <= limit:
            parts = [item]
        else:
            parts = self._split_table_item(item, limit=limit)
        for index, part in enumerate(parts):
            buffer = self._new_buffer(item.heading_path, item.section)
            buffer.items.append(part)
            buffer.tokens = part.token_count
            buffer.split_from_previous = index > 0
            self._buffers.append(buffer)

    def _split_table_item(self, item: _Item, *, limit: int) -> list[_Item]:
        """Split an oversized table by rows, never exceeding the hard maximum."""
        parts: list[_Item] = []
        current: list[str] = []
        current_tokens = 0
        for row in item.text.split("\n"):
            row_tokens = self.options.tokenizer.count(row)
            if row_tokens > limit:
                if current:
                    parts.append(_with_text(item, "\n".join(current), self.options))
                    current, current_tokens = [], 0
                for piece in split_oversized_text(row, limit=limit, options=self.options):
                    parts.append(_with_text(item, piece, self.options))
                continue
            if current and current_tokens + row_tokens > limit:
                parts.append(_with_text(item, "\n".join(current), self.options))
                current, current_tokens = [], 0
            current.append(row)
            current_tokens += row_tokens
        if current:
            parts.append(_with_text(item, "\n".join(current), self.options))
        return parts

    def _consume_block(
        self,
        block: ParsedBlock,
        role: CorpusRole,
        running: RunningLineReport,
        coverage: _TableCoverage | None,
    ) -> None:
        raw = block.text or ""
        if running.lines:
            stripped = strip_running_lines(raw, running)
            if stripped != raw:
                self._running_lines_removed += len(raw.split("\n")) - len(
                    stripped.split("\n")
                )
                raw = stripped
        if coverage is not None and _is_duplicated_by_table(raw, coverage):
            self._duplicates_removed += 1
            self._skipped.append(
                SkippedItem(
                    sequence_index=block.sequence_index,
                    reason=REASON_DUPLICATE_TABLE_TEXT,
                    source_locator=block.source_locator,
                )
            )
            return

        text = normalize_block_text(raw)
        if not text:
            self._skipped.append(
                SkippedItem(
                    sequence_index=block.sequence_index,
                    reason=REASON_EMPTY_AFTER_NORMALIZATION,
                    source_locator=block.source_locator,
                )
            )
            return

        path = tuple(normalize_inline(part) for part in block.heading_path if part)
        if not path:
            path = self._pending_heading_path

        intro = reference_intro_line(text)
        if intro is not None and section_kind(path) is not SectionKind.REFERENCES:
            # "References" glued to its first entry inside a single paragraph block.
            path = path + (intro,)
            rest = text.split("\n", 1)[1].strip() if "\n" in text else ""
            self._pending_heading_path = path
            self._pending_heading_active = True
            self._content_since_heading = False
            if not rest:
                return
            text = rest

        section = section_kind(path)
        content_type = self._content_type_for(section, role)
        if looks_like_caption(text) and section is not SectionKind.REFERENCES:
            content_type = ContentType.CAPTION

        self._content_since_heading = True
        self._accumulate(
            _Item(
                sequence_index=block.sequence_index,
                text=text,
                content_type=content_type,
                section=section,
                heading_path=path,
                page_number=block.page_number,
                source_locator=block.source_locator,
                text_source=block.text_source,
                token_count=self.options.tokenizer.count(text),
            )
        )

    @staticmethod
    def _content_type_for(section: SectionKind, role: CorpusRole) -> ContentType:
        """Map a section kind (and role) to a chunk content type.

        A reference list is always a reference list. Otherwise an internal
        summary file (role ``summaries``) is navigation material end to end, so it
        stays distinguishable from the literature it summarises.
        """
        if section is SectionKind.REFERENCES:
            return ContentType.REFERENCES
        if role is CorpusRole.SUMMARIES:
            return ContentType.SUMMARY_ENTRY
        mapping = {
            SectionKind.ABSTRACT: ContentType.ABSTRACT,
            SectionKind.CONCLUSION: ContentType.CONCLUSION,
            SectionKind.APPENDIX: ContentType.APPENDIX,
        }
        return mapping.get(section, ContentType.BODY)

    # -- accumulation -------------------------------------------------------------
    def _limits(self) -> dict[str, int]:
        return {
            "min": self.options.target_min_tokens,
            "target": self.options.target_max_tokens,
            "hard": self.options.hard_max_tokens,
            "extended": self.options.extended_max_tokens,
        }

    def _new_buffer(self, path: tuple[str, ...], section: SectionKind) -> _Buffer:
        return _Buffer(
            heading_path=path,
            section_title=path[-1] if path else None,
            section=section,
        )

    def _push(self, item: _Item) -> None:
        if self._current is None:
            self._current = self._new_buffer(item.heading_path, item.section)
        self._current.items.append(item)
        self._current.tokens += item.token_count

    def _close_buffer(self) -> None:
        if self._current is not None and self._current.items:
            self._buffers.append(self._current)
        self._current = None

    def _accumulate(self, item: _Item) -> None:
        limits = self._limits()
        current = self._current
        if current is None or current.heading_path != item.heading_path:
            self._close_buffer()
            self._current = self._new_buffer(item.heading_path, item.section)
            current = self._current

        if item.token_count > limits["hard"]:
            # A single paragraph that exceeds the hard maximum: split it and let
            # each part stand as its own chunk.
            self._close_buffer()
            parts = split_oversized_text(
                item.text, limit=limits["target"], options=self.options
            )
            for index, part in enumerate(parts):
                buffer = self._new_buffer(item.heading_path, item.section)
                piece = _with_text(item, part, self.options)
                buffer.items.append(piece)
                buffer.tokens = piece.token_count
                buffer.split_from_previous = index > 0
                self._buffers.append(buffer)
            return

        if current.items and current.tokens + item.token_count > limits["hard"]:
            self._close_buffer()
            self._current = self._new_buffer(item.heading_path, item.section)
        self._push(item)
        if self._current is not None and self._current.tokens >= limits["target"]:
            self._close_buffer()

    # -- short-section merging ----------------------------------------------------
    def _merge_short_sections(self) -> None:
        limits = self._limits()
        merged: list[_Buffer] = []
        for buffer in self._buffers:
            if (
                merged
                and buffer.tokens < limits["min"]
                and _can_merge(merged[-1], buffer, limits)
            ):
                previous = merged[-1]
                previous.items.extend(buffer.items)
                previous.tokens += buffer.tokens
                common = _common_parent(previous.heading_path, buffer.heading_path)
                previous.heading_path = common
                previous.section_title = common[-1] if common else None
                continue
            merged.append(buffer)
        self._buffers = merged

    # -- chunk construction -------------------------------------------------------
    def _build_chunks(
        self,
        parsed: ParsedDocument,
        metadata: AcademicMetadata | None,
        *,
        document_id: UUID,
        file_hash: str,
        role: CorpusRole,
    ) -> list[Chunk]:
        created_at = self.options.resolve_created_at()
        metadata_ref = _metadata_ref(parsed, metadata, file_hash=file_hash)
        document_title = metadata_ref.title
        policy = policy_for_role(role)
        chunks: list[Chunk] = []
        previous_section: tuple[SectionKind, tuple[str, ...]] | None = None

        for index, buffer in enumerate(self._buffers):
            text = buffer.text.strip()
            if not text:
                continue
            content_type = _buffer_content_type(buffer)
            if (
                content_type is ContentType.ABSTRACT
                and previous_section is not None
                and previous_section == (buffer.section, buffer.heading_path)
            ):
                # A heading detector can fail to close an "Abstract" section, which
                # would label whole body paragraphs as an abstract. Only the first
                # chunk of an abstract section is an abstract; anything after it is
                # body text, so the label never over-claims.
                content_type = ContentType.BODY
            if content_type is not ContentType.TABLE:
                previous_section = (buffer.section, buffer.heading_path)
            retrieval, evidence = eligibility_for(role, content_type)
            pages = buffer.pages
            chunks.append(
                Chunk(
                    id=chunk_id_for(file_hash, index, text),
                    document_id=document_id,
                    chunk_index=index,
                    text=text,
                    page_start=min(pages) if pages else None,
                    page_end=max(pages) if pages else None,
                    section_title=buffer.section_title,
                    heading_path=list(buffer.heading_path),
                    token_count=self.options.tokenizer.count(text),
                    content_type=content_type,
                    created_at=created_at,
                    retrieval_text=self._retrieval_text(
                        text, buffer.heading_path, document_title
                    ),
                    text_sources=list(
                        dict.fromkeys(
                            item.text_source
                            for item in buffer.items
                            if item.text_source is not None
                        )
                    ),
                    source_locators=list(
                        dict.fromkeys(
                            item.source_locator
                            for item in buffer.items
                            if item.source_locator
                        )
                    ),
                    sequence_start=min(item.sequence_index for item in buffer.items),
                    sequence_end=max(item.sequence_index for item in buffer.items),
                    metadata_ref=metadata_ref,
                    corpus_role=role,
                    authority=policy.authority,
                    retrieval_eligible=retrieval,
                    evidence_eligible=evidence,
                    merged_sections=buffer.merged_sections,
                    split_from_previous=buffer.split_from_previous,
                    overlap_tokens=0,
                )
            )
        return chunks

    def _retrieval_text(
        self, text: str, heading_path: tuple[str, ...], document_title: str | None
    ) -> str:
        """Contextualised text for embedding (never the quote-capable ``text``)."""
        if not self.options.include_context_labels:
            return text
        label = _context_label(heading_path, document_title)
        return f"{label}\n\n{text}" if label else text


def _with_text(item: _Item, text: str, options: ChunkingOptions) -> _Item:
    """Copy an item with replacement text and a recomputed token count."""
    return _Item(
        sequence_index=item.sequence_index,
        text=text,
        content_type=item.content_type,
        section=item.section,
        heading_path=item.heading_path,
        page_number=item.page_number,
        source_locator=item.source_locator,
        text_source=item.text_source,
        token_count=options.tokenizer.count(text),
    )


def _can_merge(previous: _Buffer, following: _Buffer, limits: dict[str, int]) -> bool:
    """Whether two sibling buffers may become one chunk (RAG-007 §14).

    Merging happens only inside one parent section: ``following`` must sit under
    ``previous``'s own heading (its child), or both must share a non-empty parent
    heading. Two unrelated top-level sections are never merged, because the
    merged chunk would lose its section label entirely.
    """
    if not previous.items or not following.items:
        return False
    if ContentType.TABLE in (previous.content_types | following.content_types):
        return False
    if previous.section is not following.section:
        return False
    if previous.section is SectionKind.REFERENCES:
        return False
    common = _common_parent(previous.heading_path, following.heading_path)
    if common != previous.heading_path and not (
        common and len(common) == len(previous.heading_path) - 1
    ):
        return False
    return previous.tokens + following.tokens <= limits["target"]


def chunk_document(
    parsed: ParsedDocument,
    *,
    metadata: AcademicMetadata | None = None,
    file_hash: str | None = None,
    corpus_role: CorpusRole | None = None,
    options: ChunkingOptions | None = None,
) -> ChunkedDocument:
    """Chunk one parsed document into retrieval chunks with full provenance.

    Args:
        parsed: the RAG-003/RAG-004/RAG-005 parse result (never re-parsed here).
        metadata: optional RAG-006 metadata for the same logical document.
        file_hash: the logical document hash; the parse result's own hash is used
            when omitted, and a content hash is derived only as a last resort.
        corpus_role: overrides the role derived from the metadata or the path.
        options: chunking configuration (token limits, merging, labels).
    """
    return DocumentChunker(options).chunk(
        parsed,
        metadata=metadata,
        file_hash=file_hash,
        corpus_role=corpus_role,
    )
