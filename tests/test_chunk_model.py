"""Tests for the :class:`Chunk` domain model and its validation rules."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.models.chunk import Chunk
from app.models.enums import ContentType


def make_chunk(**overrides: object) -> Chunk:
    """Build a valid Chunk, overriding individual fields as needed."""
    defaults: dict[str, object] = {
        "id": uuid4(),
        "document_id": uuid4(),
        "chunk_index": 0,
        "text": "Some body text.",
        "content_type": ContentType.BODY,
        "created_at": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    return Chunk(**defaults)  # type: ignore[arg-type]


def test_minimal_chunk_defaults() -> None:
    chunk = make_chunk()
    assert chunk.heading_path == []
    assert chunk.page_start is None
    assert chunk.page_end is None
    assert chunk.section_title is None
    assert chunk.token_count is None


def test_heading_path_defaults_to_independent_lists() -> None:
    first = make_chunk()
    second = make_chunk()
    first.heading_path.append("Introduction")
    assert second.heading_path == []


def test_negative_chunk_index_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_chunk(chunk_index=-1)


def test_page_numbers_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        make_chunk(page_start=0)
    with pytest.raises(ValidationError):
        make_chunk(page_end=0)


def test_page_end_before_page_start_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_chunk(page_start=5, page_end=4)


def test_valid_page_range_is_accepted() -> None:
    chunk = make_chunk(page_start=2, page_end=5)
    assert chunk.page_start == 2
    assert chunk.page_end == 5


def test_single_page_range_is_accepted() -> None:
    chunk = make_chunk(page_start=3, page_end=3)
    assert chunk.page_start == 3
    assert chunk.page_end == 3


def test_page_start_without_page_end_is_accepted() -> None:
    chunk = make_chunk(page_start=3)
    assert chunk.page_start == 3
    assert chunk.page_end is None


def test_content_type_round_trips_through_json() -> None:
    chunk = make_chunk(content_type=ContentType.ABSTRACT)
    payload = chunk.model_dump(mode="json")
    assert payload["content_type"] == "abstract"
    assert Chunk.model_validate(payload) == chunk


def test_extra_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_chunk(unexpected="boom")
