"""Tests for the :class:`PhysicalLocation` model."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.models.physical_location import PhysicalLocation


def test_path_is_required_and_not_empty() -> None:
    with pytest.raises(ValidationError):
        PhysicalLocation(path="")
    with pytest.raises(ValidationError):
        PhysicalLocation(path="   ")


def test_accepts_path_and_folder() -> None:
    location = PhysicalLocation(path="F:/data/literature/AI/a.pdf", folder="AI")
    assert location.path == "F:/data/literature/AI/a.pdf"
    assert location.folder == "AI"


def test_folder_is_optional() -> None:
    assert PhysicalLocation(path="a.pdf").folder is None


def test_discovered_at_defaults_to_utc_now() -> None:
    before = datetime.now(timezone.utc)
    location = PhysicalLocation(path="a.pdf")
    after = datetime.now(timezone.utc)
    assert before <= location.discovered_at <= after
    assert location.discovered_at.tzinfo is not None


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        PhysicalLocation(path="a.pdf", unexpected="boom")
