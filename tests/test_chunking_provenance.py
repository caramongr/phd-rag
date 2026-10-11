"""RAG-007 tests 28-31: OCR provenance and table handling."""

from __future__ import annotations

from datetime import datetime, timezone

from chunking_support import PageSpec, build_document, heading, long_text, paragraph, table

from app.chunking.chunker import chunk_document
from app.chunking.models import ChunkingOptions
from app.parsing.models import TextSource
from app.models.enums import ContentType

FIXED = datetime(2026, 1, 1, tzinfo=timezone.utc)
OPTIONS = ChunkingOptions(created_at=FIXED)


def test_ocr_text_source_survives_to_the_chunk() -> None:
    document = build_document(
        [paragraph("Text recovered by OCR.", page=4, path=("Results",), text_source=TextSource.OCR)]
    )

    chunk = chunk_document(document, options=OPTIONS).chunks[0]

    assert chunk.text_sources == [TextSource.OCR]
    assert chunk.ocr_involved is True
    assert chunk.page_start == 4


def test_mixed_native_and_ocr_chunk_is_marked() -> None:
    document = build_document(
        [
            paragraph("Native paragraph.", page=1, path=("Results",)),
            paragraph("OCR paragraph.", page=2, path=("Results",), text_source=TextSource.OCR),
        ]
    )

    chunk = chunk_document(document, options=OPTIONS).chunks[0]

    assert chunk.text_sources == [TextSource.NATIVE_PDF, TextSource.OCR]
    assert chunk.ocr_involved is True
    assert (chunk.page_start, chunk.page_end) == (1, 2)


def test_ocr_page_provenance_is_kept_for_the_page_itself() -> None:
    document = build_document(
        [paragraph("Recovered text.", page=7, path=("Results",), text_source=TextSource.OCR)],
        pages=[
            PageSpec(
                number=7,
                text="Recovered text.",
                text_source=TextSource.OCR,
                ocr_applied=True,
                native_text="",
                ocr_text="Recovered text.",
            )
        ],
    )

    result = chunk_document(document, options=OPTIONS)

    assert result.chunks[0].text_sources == [TextSource.OCR]
    assert result.chunks[0].source_locators == ["page:7:block:0"]


def test_structured_table_is_represented_deterministically() -> None:
    document = build_document(
        [
            heading("Results", 1),
            paragraph("Prose before the table.", path=("Results",)),
            table(
                [["Condition", "Effect size", "N"], ["VR training", "0.72", "120"]],
                page=3,
                path=("Results",),
            ),
        ]
    )

    chunks = chunk_document(document, options=OPTIONS).chunks
    tables = [chunk for chunk in chunks if chunk.content_type is ContentType.TABLE]

    assert len(tables) == 1
    assert tables[0].text == "Condition | Effect size | N\nVR training | 0.72 | 120"
    assert tables[0].page_start == 3
    assert tables[0].source_locators == ["page:3:table:0"]


def test_table_text_is_not_duplicated_when_the_parser_emits_it_as_prose() -> None:
    rows = [["Condition", "Effect size", "N"], ["VR training", "0.72", "120"]]
    # RAG-004 keeps table text in the prose blocks as well; the same row text is
    # therefore present as a paragraph block and as structured table data.
    duplicated_row = "Condition Effect size N"
    document = build_document(
        [
            heading("Results", 1),
            paragraph(duplicated_row, page=2, path=("Results",)),
            paragraph("VR training 0.72 120", page=2, path=("Results",)),
            table(rows, page=2, path=("Results",)),
            paragraph("Prose after the table.", page=2, path=("Results",)),
        ]
    )

    result = chunk_document(document, options=OPTIONS)

    assert result.duplicate_table_blocks_removed == 2
    assert [item.reason for item in result.skipped_items] == [
        "duplicate_table_text",
        "duplicate_table_text",
    ]
    joined = "\n".join(chunk.text for chunk in result.chunks)
    assert joined.count("Condition | Effect size | N") == 1
    assert "Prose after the table." in joined
    assert "VR training | 0.72 | 120" in joined


def test_table_text_on_a_different_page_is_not_deduplicated() -> None:
    document = build_document(
        [
            paragraph("Condition Effect size N here on page one.", page=1, path=("Results",)),
            table([["Condition", "Effect size", "N"], ["VR", "0.72", "120"]], page=2, path=("Results",)),
        ]
    )

    result = chunk_document(document, options=OPTIONS)

    assert result.duplicate_table_blocks_removed == 0
    assert any("Condition Effect size N" in chunk.text for chunk in result.chunks)


def test_oversized_table_is_split_by_rows_within_the_ceiling() -> None:
    rows = [["Header A", "Header B"]] + [
        [f"row {index}", long_text(3)] for index in range(40)
    ]
    document = build_document([table(rows, page=5, path=("Results",))])

    chunks = chunk_document(document, options=OPTIONS).chunks

    assert len(chunks) > 1
    assert all(chunk.token_count <= 1200 for chunk in chunks)
    assert all(chunk.content_type is ContentType.TABLE for chunk in chunks)
    assert chunks[0].split_from_previous is False
    assert all(chunk.split_from_previous for chunk in chunks[1:])
    # Splitting a table does not repeat its header row.
    assert sum(chunk.text.count("Header A") for chunk in chunks) == 1


def test_a_caption_stays_context_not_a_lone_tiny_chunk() -> None:
    document = build_document(
        [
            heading("Results", 1),
            paragraph(long_text(6), path=("Results",)),
            paragraph("Figure 2: Reported learning outcomes across the three cohorts.", path=("Results",)),
        ]
    )

    chunks = chunk_document(document, options=OPTIONS).chunks

    assert len(chunks) == 1
    assert chunks[0].content_type is ContentType.BODY
    assert "Figure 2:" in chunks[0].text


def test_in_text_figure_reference_is_not_treated_as_a_caption() -> None:
    document = build_document(
        [
            heading("Results", 1),
            paragraph(long_text(6), path=("Results",)),
            paragraph("Figure 2 shows the reported learning outcomes across cohorts.", path=("Results",)),
        ]
    )

    chunks = chunk_document(document, options=OPTIONS).chunks

    assert chunks[0].content_type is ContentType.BODY


def test_caption_only_content_is_tagged_caption() -> None:
    document = build_document(
        [paragraph("Table 4. Descriptive statistics for the sample.", page=6, path=("Results",))]
    )

    chunk = chunk_document(document, options=OPTIONS).chunks[0]

    assert chunk.content_type is ContentType.CAPTION
    assert chunk.page_start == 6
