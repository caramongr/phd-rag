"""RAG-007 CLI tests: chunk inspection and corpus reporting."""

from __future__ import annotations

from pathlib import Path

from chunking_support import build_document, heading, paragraph
from metadata_support import make_docx, make_markdown, make_pdf_pages

from app.chunking.inspect import main as inspect_main
from app.chunking.inspect import render as render_inspection
from app.chunking.report import collect, main as report_main, render

_HEADING_LINE = "Introduction"


def _sample_pdf(tmp_path: Path) -> Path:
    lines = [
        ("Introduction", 16.0, 90.0),
        ("This paper reviews immersive virtual reality in higher education.", 10.0, 130.0),
        ("It reports the outcomes of a systematic review of the literature.", 10.0, 150.0),
        ("References", 16.0, 200.0),
        ("Smith, J. (2022). A study. Journal, 1(1), 1-10.", 10.0, 230.0),
    ]
    return make_pdf_pages(tmp_path, "sample.pdf", [lines], metadata={"title": "Sample paper"})


def test_inspect_renders_a_compact_overview(tmp_path: Path) -> None:
    path = _sample_pdf(tmp_path)

    output = render_inspection(path, max_chunks=5, ocr=False)

    assert "File:" in output
    assert "Chunks:" in output
    assert "tokenizer subword-estimator-v1" in output
    assert "type body" in output
    assert "section Introduction" in output
    assert "type references" in output
    assert "retrieval no" in output
    assert "preview:" in output


def test_inspect_respects_max_chunks_and_sections(tmp_path: Path) -> None:
    path = _sample_pdf(tmp_path)

    output = render_inspection(path, max_chunks=1, ocr=False)

    assert "[  0]" in output
    assert "further chunks not shown" in output


def test_inspect_can_show_text_retrieval_text_and_locators(tmp_path: Path) -> None:
    path = _sample_pdf(tmp_path)

    output = render_inspection(
        path, show_text=True, show_retrieval_text=True, show_locators=True, ocr=False
    )

    assert "text:" in output
    assert "retrieval_text:" in output
    assert "[Section: Introduction]" in output
    assert "locators: page:1:block:1" in output


def test_inspect_cli_reports_a_missing_file(tmp_path: Path, capsys) -> None:
    exit_code = inspect_main([str(tmp_path / "missing.pdf")])

    assert exit_code == 2
    assert "file not found" in capsys.readouterr().err


def test_inspect_cli_runs_on_a_real_file(tmp_path: Path) -> None:
    path = _sample_pdf(tmp_path)

    assert inspect_main([str(path), "--max-chunks", "2", "--no-ocr"]) == 0


def _temp_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    (root / "literature" / "AI").mkdir(parents=True)
    (root / "my_work").mkdir(parents=True)
    (root / "rules").mkdir(parents=True)
    make_pdf_pages(
        root / "literature" / "AI",
        "paper.pdf",
        [[("Introduction", 16.0, 90.0), ("Body text about virtual reality.", 10.0, 130.0)]],
    )
    make_docx(root / "my_work", "draft.docx", title="My draft", paragraphs=["My own draft text."])
    make_markdown(root / "rules", "rules.md", text="# Submission rules\n\nSubmit on time.\n")
    # One spreadsheet: unsupported for chunking in RAG-007, counted not discarded.
    (root / "rules" / "grades.xlsx").write_bytes(b"not really a spreadsheet")
    return root


def test_report_collects_documents_and_flags_spreadsheets(tmp_path: Path) -> None:
    root = _temp_corpus(tmp_path)

    report = collect(root, ocr=False)

    assert report.unique_documents == 4
    assert report.documents_chunked == 3
    assert report.unsupported_for_chunking == {"xlsx": 1}
    assert report.parse_errors == 0
    assert report.chunking_errors == 0
    assert report.total_chunks >= 3
    assert report.token_max <= 1500
    assert report.by_corpus_role["literature"] >= 1
    assert report.by_corpus_role["my_work"] >= 1
    assert report.by_corpus_role["rules"] >= 1
    assert report.evidence_eligible_chunks < report.total_chunks


def test_report_render_includes_the_required_sections(tmp_path: Path) -> None:
    root = _temp_corpus(tmp_path)

    text = render(collect(root, ocr=False))

    for expected in (
        "unique documents considered",
        "documents successfully chunked",
        "documents skipped",
        "total chunks",
        "retrieval-eligible chunks",
        "evidence-eligible chunks",
        "median:",
        "p95:",
        "max:",
        "700-1000 tokens",
        "by content type:",
        "by corpus role:",
        "by text source:",
        "chunks crossing page boundaries:",
        "chunks containing OCR text:",
        "reference chunks:",
        "table chunks:",
        "unsupported for chunking in RAG-007",
        "No source file was modified",
    ):
        assert expected in text


def test_report_cli_writes_the_artifact(tmp_path: Path) -> None:
    root = _temp_corpus(tmp_path)
    out = tmp_path / "report.txt"

    exit_code = report_main([str(root), "--out", str(out), "--no-ocr"])

    assert exit_code == 0
    written = out.read_text(encoding="utf-8")
    assert "RAG-007 chunking report" in written
    assert "No source file was modified" in written


def test_report_text_source_buckets_are_canonical(tmp_path: Path) -> None:
    root = tmp_path / "data"
    (root / "literature").mkdir(parents=True)
    make_pdf_pages(
        root / "literature",
        "mixed.pdf",
        [[("Body text about virtual reality.", 10.0, 130.0)]],
    )

    report = collect(root, ocr=False)

    # No scan-only keys such as "ocr+native_pdf" may appear alongside the canonical
    # order; the label is sorted so the same combination is one bucket.
    for key in report.by_text_source:
        if "+" in key:
            assert key == "+".join(sorted(key.split("+")))


def test_report_limit_restricts_processing(tmp_path: Path) -> None:
    root = _temp_corpus(tmp_path)

    limited = collect(root, limit=1, ocr=False)

    assert limited.documents_chunked + limited.documents_skipped == 1
    assert limited.unique_documents == 4


def test_chunk_document_survives_a_document_without_content() -> None:
    document = build_document([paragraph("", path=("Body",))])

    from app.chunking.chunker import chunk_document
    from app.chunking.models import ChunkingOptions

    result = chunk_document(document, options=ChunkingOptions())

    assert result.chunk_count == 0
    assert result.skipped_items


def test_heading_without_content_is_counted() -> None:
    from app.chunking.chunker import chunk_document
    from app.chunking.models import ChunkingOptions

    document = build_document([heading(_HEADING_LINE, 1), heading("Methods", 1), paragraph("Body.", path=("Methods",))])

    result = chunk_document(document, options=ChunkingOptions())

    assert result.headings_without_content == 1
