"""RAG-007 tests 9-18: structure, page provenance and chunk sizing."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from chunking_support import (
    PageSpec,
    build_document,
    heading,
    long_text,
    paragraph,
    table,
)

from app.chunking.chunker import chunk_document
from app.chunking.models import ChunkingOptions
from app.chunking.tokenizer import count_tokens
from app.models.enums import ContentType

FIXED = datetime(2026, 1, 1, tzinfo=timezone.utc)


def options(**overrides: object) -> ChunkingOptions:
    """Chunking options with a fixed timestamp for deterministic comparisons."""
    values: dict[str, object] = {"created_at": FIXED}
    values.update(overrides)
    return ChunkingOptions(**values)  # type: ignore[arg-type]


def test_explicit_heading_path_is_preserved() -> None:
    document = build_document(
        [
            heading("4 Virtual Reality in Higher Education", 1),
            paragraph(long_text(30), path=("4 Virtual Reality in Higher Education",)),
            heading(
                "4.2 Educational Applications",
                2,
                path=("4 Virtual Reality in Higher Education", "4.2 Educational Applications"),
            ),
            paragraph(
                long_text(30),
                path=("4 Virtual Reality in Higher Education", "4.2 Educational Applications"),
            ),
            heading(
                "4.2.1 Medical Training",
                3,
                path=(
                    "4 Virtual Reality in Higher Education",
                    "4.2 Educational Applications",
                    "4.2.1 Medical Training",
                ),
            ),
            paragraph(
                long_text(30),
                path=(
                    "4 Virtual Reality in Higher Education",
                    "4.2 Educational Applications",
                    "4.2.1 Medical Training",
                ),
            ),
        ]
    )

    result = chunk_document(document, options=options())

    medical_path = [
        "4 Virtual Reality in Higher Education",
        "4.2 Educational Applications",
        "4.2.1 Medical Training",
    ]
    medical = [c for c in result.chunks if c.heading_path == medical_path]

    assert medical, "the deepest section must keep its full heading path"
    assert all(c.section_title == "4.2.1 Medical Training" for c in medical)
    # No other section may claim the medical heading path.
    assert all(
        "4.2.1 Medical Training" not in chunk.heading_path
        for chunk in result.chunks
        if chunk.heading_path != medical_path
    )


def test_inferred_heading_path_is_preserved() -> None:
    intro = heading("Introduction", 1)
    intro.heading_source = None  # the parser's inferred headings keep the same path
    document = build_document([intro, paragraph("Body.", path=("Introduction",))])

    result = chunk_document(document, options=options())

    assert result.chunks[0].heading_path == ["Introduction"]
    assert result.chunks[0].section_title == "Introduction"


def test_page_provenance_is_preserved() -> None:
    document = build_document([paragraph("Page three prose.", page=3)])

    chunk = chunk_document(document, options=options()).chunks[0]

    assert chunk.page_start == 3
    assert chunk.page_end == 3
    assert chunk.source_locators == ["page:3:block:0"]


def test_multi_page_chunk_records_page_range() -> None:
    document = build_document(
        [paragraph("First page prose.", page=2), paragraph("Second page prose.", page=3)]
    )

    chunk = chunk_document(document, options=options()).chunks[0]

    assert (chunk.page_start, chunk.page_end) == (2, 3)
    assert chunk.source_locators == ["page:2:block:0", "page:3:block:1"]
    assert (chunk.sequence_start, chunk.sequence_end) == (0, 1)


def test_source_locators_and_sequence_range_are_retained() -> None:
    document = build_document(
        [
            heading("Results", 1),
            paragraph("Alpha.", path=("Results",)),
            paragraph("Beta.", path=("Results",)),
        ]
    )

    chunk = chunk_document(document, options=options()).chunks[0]

    assert chunk.source_locators == ["page:1:block:1", "page:1:block:2"]
    assert (chunk.sequence_start, chunk.sequence_end) == (1, 2)
    assert chunk.text == "Alpha.\n\nBeta."


def test_checking_the_heading_block_itself_is_not_a_chunk() -> None:
    document = build_document([heading("References", 1), paragraph("Entry one.", path=("References",))])

    result = chunk_document(document, options=options())

    assert len(result.chunks) == 1
    assert result.chunks[0].text == "Entry one."


def test_paragraphs_are_accumulated_toward_the_target() -> None:
    body = long_text(6)
    per_paragraph = count_tokens(body)
    document = build_document([paragraph(body, path=("Results",)) for _ in range(6)])

    result = chunk_document(document, options=options())
    merged = [c for c in result.chunks if c.token_count >= 700]

    assert body.strip() != ""
    assert per_paragraph > 100
    assert merged, "paragraphs should be merged into target-sized chunks"
    assert merged[0].token_count >= 700
    assert merged[0].text.count("\n\n") >= 2, "the chunk should hold several paragraphs"
    assert len(result.chunks) < 6


def test_hard_maximum_is_respected() -> None:
    document = build_document([paragraph(long_text(5), path=("Body",)) for _ in range(12)])

    result = chunk_document(document, options=options())

    assert result.chunks
    assert max(chunk.token_count for chunk in result.chunks) <= 1200


def test_oversized_paragraph_is_split_at_sentences() -> None:
    document = build_document([paragraph(long_text(60), path=("Discussion",))])

    result = chunk_document(document, options=options())

    assert len(result.chunks) >= 3
    assert all(chunk.token_count <= 1200 for chunk in result.chunks)
    assert result.chunks[0].split_from_previous is False
    assert all(chunk.split_from_previous for chunk in result.chunks[1:])
    assert result.chunks[0].text.endswith(".")
    assert all(chunk.overlap_tokens == 0 for chunk in result.chunks)


def test_no_chunk_exceeds_the_ceiling_with_tables_and_long_paragraphs() -> None:
    big_table = table(
        [["Row header", "Second column", "Third column"]]
        + [[f"cell {row} a", f"cell {row} b", long_text(3)] for row in range(40)],
        page=2,
        path=("Results",),
    )
    document = build_document(
        [paragraph(long_text(80), path=("Results",)), big_table, paragraph(long_text(40), path=("Results",))]
    )

    result = chunk_document(document, options=options())

    assert result.chunks
    assert max(chunk.token_count for chunk in result.chunks) <= 1500
    assert all(chunk.token_count <= 1200 for chunk in result.chunks)


def test_a_large_abstract_section_only_labels_its_first_chunk() -> None:
    document = build_document(
        [
            heading("Abstract", 1),
            paragraph(long_text(60), path=("Abstract",)),
            paragraph(long_text(60), path=("Abstract",)),
        ]
    )

    chunks = chunk_document(document, options=options()).chunks

    assert len(chunks) >= 2
    assert chunks[0].content_type is ContentType.ABSTRACT
    assert all(chunk.content_type is ContentType.BODY for chunk in chunks[1:])


def test_a_short_abstract_may_stand_alone() -> None:
    abstract = "This study reviews immersive virtual reality in higher education."
    body = long_text(10)
    document = build_document(
        [
            heading("Abstract", 1),
            paragraph(abstract, path=("Abstract",)),
            heading("1 Introduction", 1),
            paragraph(body, path=("1 Introduction",)),
        ]
    )

    result = chunk_document(document, options=options())
    abstracts = [c for c in result.chunks if c.content_type is ContentType.ABSTRACT]

    assert len(abstracts) == 1
    assert abstracts[0].text == abstract
    assert abstracts[0].token_count < 200
    assert all(body[:40] not in chunk.text for chunk in abstracts)


def test_short_sibling_sections_are_merged_within_their_parent() -> None:
    short = long_text(3)
    document = build_document(
        [
            heading("4 Results", 1),
            paragraph(long_text(3), path=("4 Results",)),
            heading("4.1 First topic", 2, path=("4 Results", "4.1 First topic")),
            paragraph(short, path=("4 Results", "4.1 First topic")),
            heading("4.2 Second topic", 2, path=("4 Results", "4.2 Second topic")),
            paragraph(short, path=("4 Results", "4.2 Second topic")),
        ]
    )

    result = chunk_document(document, options=options())
    merged = [c for c in result.chunks if c.merged_sections]

    assert len(result.chunks) == 1
    assert merged[0].heading_path == ["4 Results"]
    assert merged[0].section_title == "4 Results"
    assert merged[0].merged_sections == ["4.1 First topic", "4.2 Second topic"]


def test_unrelated_top_level_sections_are_not_merged() -> None:
    short = long_text(3)
    document = build_document(
        [
            heading("3 Methods", 1),
            paragraph(short, path=("3 Methods",)),
            heading("4 Results", 1),
            paragraph(short, path=("4 Results",)),
        ]
    )

    result = chunk_document(document, options=options())

    assert len(result.chunks) == 2
    assert [c.section_title for c in result.chunks] == ["3 Methods", "4 Results"]
    assert all(not c.merged_sections for c in result.chunks)


def test_running_header_lines_do_not_become_chunk_text() -> None:
    document = build_document(
        [paragraph(f"Journal of Testing\nBody prose for page {page}.", page=page) for page in range(1, 6)],
        pages=[PageSpec(number=page, text=f"Journal of Testing\nBody prose for page {page}.\n{page}") for page in range(1, 6)],
    )

    result = chunk_document(document, options=options())

    assert result.running_lines_removed >= 5
    assert all("Journal of Testing" not in chunk.text for chunk in result.chunks)


@pytest.mark.parametrize("tokens", [1, 50, 180])
def test_small_chunks_are_allowed_and_reported(tokens: int) -> None:
    text = " ".join(["word"] * tokens)
    document = build_document([paragraph(text, path=("Body",))])

    chunk = chunk_document(document, options=options()).chunks[0]

    assert chunk.token_count == count_tokens(text)
    assert chunk.token_count <= 1200
