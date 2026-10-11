"""Development CLI: inspect the chunks of a single document (RAG-007).

Usage::

    python -m app.chunking.inspect "<path>" [--show-text] [--show-retrieval-text]
                                            [--show-locators] [--max-chunks N] [--no-ocr]
                                            [--corpus <root>]

The default output is a compact per-chunk overview (index, token count, pages,
section, content type, text sources, retrieval/evidence eligibility and a short
preview), so a document is never dumped in full by accident.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from app.chunking.chunker import chunk_document
from app.chunking.models import ChunkingOptions
from app.ingestion.file_hash import calculate_sha256
from app.metadata.corpus import physical_paths_for_hash
from app.metadata.extractor import extract_metadata
from app.models.chunk import Chunk
from app.parsing.exceptions import (
    DocumentParseError,
    EncryptedDocumentError,
    UnsupportedDocumentFormat,
)
from app.parsing.parser import parse_document

_DEFAULT_MAX_CHUNKS = 10
_PREVIEW_CHARS = 160


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.chunking.inspect",
        description="Inspect the RAG-007 chunks and provenance of one document.",
    )
    parser.add_argument("path", type=Path, help="Document to chunk (PDF, DOCX or Markdown).")
    parser.add_argument(
        "--show-text",
        action="store_true",
        help="Print each chunk's normalized source text.",
    )
    parser.add_argument(
        "--show-retrieval-text",
        action="store_true",
        help="Print each chunk's contextualised retrieval text.",
    )
    parser.add_argument(
        "--show-locators",
        action="store_true",
        help="Print each chunk's source locators and sequence range.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=_DEFAULT_MAX_CHUNKS,
        help=f"Print at most N chunks (default {_DEFAULT_MAX_CHUNKS}).",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Skip the OCR fallback for PDFs (native text only).",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help=(
            "Corpus root used to find every physical copy of this file, so the "
            "corpus role and evidence policy are resolved across all of them."
        ),
    )
    return parser


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _preview(text: str, *, width: int = _PREVIEW_CHARS) -> str:
    collapsed = " ".join(text.split())
    return collapsed[:width] + ("…" if len(collapsed) > width else "")


def render_chunk(chunk: Chunk, *, show_text: bool, show_retrieval_text: bool, show_locators: bool) -> list[str]:
    """Render one chunk as report lines."""
    pages = (
        f"{chunk.page_start}-{chunk.page_end}"
        if chunk.page_start is not None
        else "–"
    )
    sources = "+".join(source.value for source in chunk.text_sources) or "–"
    section = " > ".join(chunk.heading_path) if chunk.heading_path else "–"
    lines = [
        f"  [{chunk.chunk_index:3d}] {chunk.token_count or 0:5d} tokens  pages {pages:>9}"
        f"  type {chunk.content_type.value:<12} section {section}",
        f"        sources {sources:<18} retrieval {_yes_no(chunk.retrieval_eligible):<3}"
        f" evidence {_yes_no(chunk.evidence_eligible):<3}"
        f" role {chunk.corpus_role.value if chunk.corpus_role else '–'}",
    ]
    if chunk.merged_sections:
        lines.append(f"        merged sections: {' | '.join(chunk.merged_sections)}")
    if chunk.split_from_previous:
        lines.append("        continues a split paragraph/table from the previous chunk")
    lines.append(f"        preview: {_preview(chunk.text)}")
    if show_text:
        lines.append("        text:")
        lines.extend(f"          {line}" for line in chunk.text.split("\n"))
    if show_retrieval_text:
        lines.append("        retrieval_text:")
        lines.extend(
            f"          {line}" for line in (chunk.retrieval_text or "").split("\n")
        )
    if show_locators:
        locators = ", ".join(chunk.source_locators) or "–"
        lines.append(
            f"        locators: {locators} (sequence {chunk.sequence_start}…{chunk.sequence_end})"
        )
    return lines


def render(
    path: Path,
    *,
    options: ChunkingOptions | None = None,
    show_text: bool = False,
    show_retrieval_text: bool = False,
    show_locators: bool = False,
    max_chunks: int = _DEFAULT_MAX_CHUNKS,
    ocr: bool | None = None,
    corpus_root: Path | None = None,
) -> str:
    """Chunk ``path`` and render the inspection report."""
    file_hash = calculate_sha256(path)
    parsed = parse_document(path, file_hash=file_hash, ocr=ocr)
    physical_paths: list[str] | None = None
    if corpus_root is not None:
        locations = physical_paths_for_hash(corpus_root, file_hash)
        physical_paths = locations or None
    metadata = extract_metadata(path, file_hash=file_hash, physical_paths=physical_paths)
    result = chunk_document(parsed, metadata=metadata, options=options)

    lines = [
        f"File: {path}",
        f"Extension: {parsed.extension}",
        f"Corpus role: {metadata.corpus_role.value}",
    ]
    resolution = metadata.role_resolution
    if resolution is not None:
        lines.append(f"Role resolution: {resolution.describe()}")
        if resolution.has_conflicting_roles:
            for physical in resolution.physical_roles:
                lines.append(f"  {physical.role.value:<16} {physical.path}")
    lines.extend(
        [
            f"Document id: {result.document_id}",
            f"Pages: {parsed.page_count if parsed.page_count is not None else '–'}",
            "",
            f"Chunks: {result.chunk_count} ({result.total_tokens} tokens, tokenizer "
            f"{result.tokenizer_name or 'unknown'})",
            f"Duplicate table blocks removed: {result.duplicate_table_blocks_removed}",
            f"Running header/footer lines removed: {result.running_lines_removed}",
            f"Headings without content: {result.headings_without_content}",
            "",
        ]
    )
    if result.skipped_items:
        reasons: dict[str, int] = {}
        for item in result.skipped_items:
            reasons[item.reason] = reasons.get(item.reason, 0) + 1
        summary = ", ".join(f"{reason}={count}" for reason, count in sorted(reasons.items()))
        lines.extend([f"Skipped items: {len(result.skipped_items)} ({summary})", ""])

    for chunk in result.chunks[:max_chunks]:
        lines.extend(
            render_chunk(
                chunk,
                show_text=show_text,
                show_retrieval_text=show_retrieval_text,
                show_locators=show_locators,
            )
        )
    if result.chunk_count > max_chunks:
        lines.append(f"  … {result.chunk_count - max_chunks} further chunks not shown")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the chunk inspector and return an exit code."""
    args = _build_parser().parse_args(argv)
    path: Path = args.path
    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 2

    try:
        report = render(
            path,
            show_text=args.show_text,
            show_retrieval_text=args.show_retrieval_text,
            show_locators=args.show_locators,
            max_chunks=args.max_chunks,
            ocr=False if args.no_ocr else None,
            corpus_root=args.corpus,
        )
    except UnsupportedDocumentFormat as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except (DocumentParseError, EncryptedDocumentError) as exc:
        print(f"Error: could not read document: {exc}", file=sys.stderr)
        return 3

    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
