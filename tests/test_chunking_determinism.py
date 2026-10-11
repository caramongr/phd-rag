"""RAG-007 tests 32-34: deterministic chunk ids, order and text."""

from __future__ import annotations

from datetime import datetime, timezone

from chunking_support import build_document, heading, long_text, paragraph, table

from app.chunking.chunker import chunk_document, chunk_id_for, document_id_for
from app.chunking.models import ChunkingOptions

FIXED = datetime(2026, 1, 1, tzinfo=timezone.utc)
LATER = datetime(2026, 6, 30, tzinfo=timezone.utc)
OPTIONS = ChunkingOptions(created_at=FIXED)


def sample_document():
    """A document with headings, prose, a table and a reference list."""
    return build_document(
        [
            heading("Abstract", 1),
            paragraph("Short abstract.", path=("Abstract",)),
            heading("1 Introduction", 1),
            paragraph(long_text(12), path=("1 Introduction",)),
            paragraph(long_text(12), path=("1 Introduction",)),
            table([["A", "B"], ["1", "2"]], page=2, path=("1 Introduction",)),
            heading("References", 1),
            paragraph("Smith, J. (2022). A study. Journal, 1(1), 1-10.", path=("References",)),
        ],
        file_hash="b" * 64,
    )


def test_same_input_produces_the_same_chunk_ids() -> None:
    first = chunk_document(sample_document(), options=OPTIONS)
    second = chunk_document(sample_document(), options=OPTIONS)

    assert [chunk.id for chunk in first.chunks] == [chunk.id for chunk in second.chunks]
    assert first.document_id == second.document_id


def test_same_input_produces_the_same_order_and_text() -> None:
    first = chunk_document(sample_document(), options=OPTIONS)
    second = chunk_document(sample_document(), options=OPTIONS)

    assert [chunk.chunk_index for chunk in first.chunks] == list(range(first.chunk_count))
    assert [chunk.text for chunk in first.chunks] == [chunk.text for chunk in second.chunks]
    assert [chunk.heading_path for chunk in first.chunks] == [
        chunk.heading_path for chunk in second.chunks
    ]
    assert [chunk.page_start for chunk in first.chunks] == [
        chunk.page_start for chunk in second.chunks
    ]


def test_payloads_are_identical_apart_from_the_timestamp() -> None:
    first = chunk_document(sample_document(), options=OPTIONS)
    second = chunk_document(sample_document(), options=ChunkingOptions(created_at=LATER))

    def payloads(result):
        return [
            chunk.model_dump(mode="json", exclude={"created_at"}) for chunk in result.chunks
        ]

    assert payloads(first) == payloads(second)
    assert first.chunks[0].created_at != second.chunks[0].created_at


def test_chunk_id_depends_on_the_document_and_the_content() -> None:
    assert chunk_id_for("a" * 64, 0, "text") == chunk_id_for("a" * 64, 0, "text")
    assert chunk_id_for("a" * 64, 0, "text") != chunk_id_for("a" * 64, 1, "text")
    assert chunk_id_for("a" * 64, 0, "text") != chunk_id_for("b" * 64, 0, "text")
    assert chunk_id_for("a" * 64, 0, "text") != chunk_id_for("a" * 64, 0, "other text")


def test_document_id_is_derived_from_the_file_hash() -> None:
    document = build_document([paragraph("Body.", path=("Body",))], file_hash="c" * 64)

    result = chunk_document(document, options=OPTIONS)

    assert result.document_id == document_id_for("c" * 64)
    assert all(chunk.document_id == result.document_id for chunk in result.chunks)


def test_normalization_is_idempotent_for_chunk_text() -> None:
    from app.chunking.normalize import normalize_block_text

    document = build_document(
        [paragraph("Wrapped prose that continues\non a second line with educa-\ntion.", path=("Body",))]
    )

    chunk = chunk_document(document, options=OPTIONS).chunks[0]

    assert normalize_block_text(chunk.text) == chunk.text
