"""Tests for recursive file discovery, hashing, dedup, and folder hints."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.ingestion import scanner
from app.ingestion.scanner import SUPPORTED_EXTENSIONS, ScanRootError, scan_directory
from app.models.enums import RelatedProject


def write_file(path: Path, content: bytes = b"content") -> Path:
    """Create ``path`` (and parents) containing ``content``; return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


# --- 1-8: discovery per supported extension ---------------------------------


@pytest.mark.parametrize(
    "filename",
    ["a.pdf", "a.docx", "a.doc", "a.rtf", "a.txt", "a.md", "a.xlsx", "a.xls"],
)
def test_supported_extensions_are_discovered(tmp_path: Path, filename: str) -> None:
    write_file(tmp_path / filename)
    result = scan_directory(tmp_path)
    assert result.scanned_files == 1
    assert result.supported_files == 1
    assert result.unique_documents == 1
    assert result.documents[0].extension == Path(filename).suffix


def test_supported_extensions_exact_set() -> None:
    assert SUPPORTED_EXTENSIONS == frozenset(
        {".pdf", ".docx", ".doc", ".rtf", ".txt", ".md", ".xlsx", ".xls"}
    )


# --- 9: case-insensitive extension matching ---------------------------------


@pytest.mark.parametrize("filename", ["A.PDF", "B.DocX", "C.XLSX", "D.TXT"])
def test_extension_matching_is_case_insensitive(tmp_path: Path, filename: str) -> None:
    write_file(tmp_path / filename)
    result = scan_directory(tmp_path)
    assert result.supported_files == 1
    assert result.documents[0].extension == Path(filename).suffix.lower()


# --- 10: unsupported files --------------------------------------------------


def test_unsupported_files_are_counted_without_failing(tmp_path: Path) -> None:
    write_file(tmp_path / "notes.pdf")
    write_file(tmp_path / "image.png")
    write_file(tmp_path / "archive.zip")
    result = scan_directory(tmp_path)
    assert result.scanned_files == 3
    assert result.supported_files == 1
    assert result.unsupported_files == 2
    assert len(result.unsupported_paths) == 2
    assert len(result.documents) == 1


# --- 11: stable SHA-256 -----------------------------------------------------


def test_document_hash_matches_content(tmp_path: Path) -> None:
    payload = b"identical bytes"
    write_file(tmp_path / "a.pdf", payload)
    result = scan_directory(tmp_path)
    assert result.documents[0].file_hash == hashlib.sha256(payload).hexdigest()


# --- 12-14: duplicates ------------------------------------------------------


def test_same_content_in_different_folders_is_one_logical_document(tmp_path: Path) -> None:
    write_file(tmp_path / "literature" / "AI" / "smith2024.pdf", b"same")
    write_file(tmp_path / "literature" / "Algorithms" / "smith2024-copy.pdf", b"same")
    result = scan_directory(tmp_path)
    assert result.unique_documents == 1
    assert result.duplicate_copies == 1
    assert result.supported_files == 2


def test_duplicate_physical_paths_are_preserved(tmp_path: Path) -> None:
    first = write_file(tmp_path / "AI" / "smith2024.pdf", b"same")
    second = write_file(tmp_path / "Algorithms" / "smith2024-copy.pdf", b"same")
    result = scan_directory(tmp_path)
    document = result.documents[0]
    assert {location.path for location in document.physical_locations} == {
        str(first),
        str(second),
    }
    assert set(document.original_folders) == {"AI", "Algorithms"}


def test_duplicate_folder_labels_are_deduplicated(tmp_path: Path) -> None:
    write_file(tmp_path / "AI" / "a.pdf", b"same")
    write_file(tmp_path / "AI" / "b.pdf", b"same")
    result = scan_directory(tmp_path)
    assert result.unique_documents == 1
    assert result.duplicate_copies == 1
    assert result.documents[0].original_folders == ["AI"]


def test_same_filename_different_content_is_not_duplicate(tmp_path: Path) -> None:
    write_file(tmp_path / "AI" / "paper.pdf", b"one")
    write_file(tmp_path / "VR" / "paper.pdf", b"two")
    result = scan_directory(tmp_path)
    assert result.unique_documents == 2
    assert result.duplicate_copies == 0


# --- 15-20: folder hints ----------------------------------------------------


@pytest.mark.parametrize(
    ("folder", "expected"),
    [
        ("AI", "artificial_intelligence"),
        ("VR", "virtual_reality"),
        ("Teaching", "teaching"),
        ("Algorithms", "algorithms"),
    ],
)
def test_topic_hints(tmp_path: Path, folder: str, expected: str) -> None:
    write_file(tmp_path / folder / "a.pdf", folder.encode())
    result = scan_directory(tmp_path)
    assert result.documents[0].topic_hints == [expected]


def test_second_paper_produces_project_hint(tmp_path: Path) -> None:
    write_file(tmp_path / "Second paper" / "a.pdf")
    result = scan_directory(tmp_path)
    document = result.documents[0]
    assert document.related_project_hints == [RelatedProject.SECOND_PAPER]
    assert document.original_folders == ["Second paper"]


def test_algorithms_alone_does_not_produce_second_paper(tmp_path: Path) -> None:
    write_file(tmp_path / "Algorithms" / "a.pdf")
    result = scan_directory(tmp_path)
    document = result.documents[0]
    assert document.topic_hints == ["algorithms"]
    assert document.related_project_hints == []


def test_hints_accumulate_across_duplicate_locations(tmp_path: Path) -> None:
    write_file(tmp_path / "AI" / "a.pdf", b"same")
    write_file(tmp_path / "VR" / "b.pdf", b"same")
    result = scan_directory(tmp_path)
    document = result.documents[0]
    assert set(document.topic_hints) == {"artificial_intelligence", "virtual_reality"}
    assert set(document.original_folders) == {"AI", "VR"}


def test_folder_hints_are_case_insensitive(tmp_path: Path) -> None:
    write_file(tmp_path / "second PAPER" / "a.pdf")
    result = scan_directory(tmp_path)
    assert result.documents[0].related_project_hints == [RelatedProject.SECOND_PAPER]


# --- nested folder hints (hints use the whole ancestor path) ----------------


def test_nested_folder_hint_uses_ancestor(tmp_path: Path) -> None:
    write_file(tmp_path / "literature" / "AI" / "adaptive-learning" / "paper.pdf")
    document = scan_directory(tmp_path).documents[0]
    assert document.topic_hints == ["artificial_intelligence"]
    # Immediate parent is still recorded as the folder label.
    assert document.original_folders == ["adaptive-learning"]
    assert document.physical_locations[0].folder == "adaptive-learning"


def test_nested_second_paper_and_algorithms_hints(tmp_path: Path) -> None:
    write_file(tmp_path / "literature" / "Second paper" / "Algorithms" / "paper.pdf")
    document = scan_directory(tmp_path).documents[0]
    assert document.topic_hints == ["algorithms"]
    assert document.related_project_hints == [RelatedProject.SECOND_PAPER]


def test_nested_hints_accumulate_from_multiple_ancestors(tmp_path: Path) -> None:
    write_file(tmp_path / "literature" / "AI" / "VR" / "paper.pdf")
    document = scan_directory(tmp_path).documents[0]
    assert set(document.topic_hints) == {"artificial_intelligence", "virtual_reality"}


def test_nested_hints_accumulate_across_duplicate_locations(tmp_path: Path) -> None:
    write_file(tmp_path / "literature" / "AI" / "deep" / "a.pdf", b"same")
    write_file(tmp_path / "literature" / "Second paper" / "deep" / "b.pdf", b"same")
    document = scan_directory(tmp_path).documents[0]
    assert document.topic_hints == ["artificial_intelligence"]
    assert document.related_project_hints == [RelatedProject.SECOND_PAPER]


# --- recursion & empty folder -----------------------------------------------


def test_recurses_into_nested_folders(tmp_path: Path) -> None:
    write_file(tmp_path / "literature" / "AI" / "deep" / "a.pdf")
    result = scan_directory(tmp_path)
    assert result.unique_documents == 1
    assert result.documents[0].original_folders == ["deep"]


def test_empty_folder_scan(tmp_path: Path) -> None:
    result = scan_directory(tmp_path)
    assert result.scanned_files == 0
    assert result.supported_files == 0
    assert result.unsupported_files == 0
    assert result.unique_documents == 0
    assert result.duplicate_copies == 0
    assert result.documents == []
    assert result.errors == []


# --- error handling ---------------------------------------------------------


def test_one_file_error_does_not_stop_the_scan(tmp_path: Path, monkeypatch) -> None:
    write_file(tmp_path / "bad.pdf", b"a")
    write_file(tmp_path / "good.pdf", b"b")

    real_calculate = scanner.calculate_sha256

    def flaky(path: Path) -> str:
        if Path(path).name == "bad.pdf":
            raise OSError("cannot read file")
        return real_calculate(path)

    monkeypatch.setattr(scanner, "calculate_sha256", flaky)

    result = scan_directory(tmp_path)
    assert result.supported_files == 2
    assert result.unique_documents == 1
    assert len(result.errors) == 1
    assert result.errors[0].path.endswith("bad.pdf")


def test_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(ScanRootError):
        scan_directory(tmp_path / "missing")


def test_file_as_root_raises(tmp_path: Path) -> None:
    target = write_file(tmp_path / "a.pdf")
    with pytest.raises(ScanRootError):
        scan_directory(target)


# --- read-only guarantee ----------------------------------------------------


def test_scan_does_not_modify_source_files(tmp_path: Path) -> None:
    target = write_file(tmp_path / "AI" / "a.pdf", b"read-only")
    before = (target.stat().st_mtime_ns, target.read_bytes())
    scan_directory(tmp_path)
    after = (target.stat().st_mtime_ns, target.read_bytes())
    assert before == after
