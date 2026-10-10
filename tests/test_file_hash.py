"""Tests for SHA-256 file hashing."""

from __future__ import annotations

import hashlib

import pytest

from app.ingestion.file_hash import CHUNK_SIZE, calculate_sha256


def test_sha256_matches_hashlib(tmp_path) -> None:
    target = tmp_path / "sample.bin"
    payload = b"PhD Academic RAG"
    target.write_bytes(payload)
    assert calculate_sha256(target) == hashlib.sha256(payload).hexdigest()


def test_sha256_is_lowercase_hex_of_expected_length(tmp_path) -> None:
    target = tmp_path / "sample.bin"
    target.write_bytes(b"hello")
    digest = calculate_sha256(target)
    assert len(digest) == 64
    assert digest == digest.lower()
    assert all(character in "0123456789abcdef" for character in digest)


def test_sha256_output_is_stable(tmp_path) -> None:
    target = tmp_path / "sample.bin"
    target.write_bytes(b"stable content")
    assert calculate_sha256(target) == calculate_sha256(target)


def test_sha256_spans_multiple_chunks(tmp_path) -> None:
    payload = b"a" * (CHUNK_SIZE + 4096)
    target = tmp_path / "large.bin"
    target.write_bytes(payload)
    assert calculate_sha256(target) == hashlib.sha256(payload).hexdigest()


def test_sha256_raises_for_missing_file(tmp_path) -> None:
    with pytest.raises(OSError):
        calculate_sha256(tmp_path / "does-not-exist.bin")
