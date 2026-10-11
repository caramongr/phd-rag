"""RAG-006: metadata inspection CLI."""

from __future__ import annotations

from pathlib import Path

from metadata_support import (
    FakeTransport,
    build_test_enricher,
    crossref_doi_handler,
    crossref_message,
    make_pdf,
)

from app.metadata import inspect as inspect_cli


def test_cli_offline_prints_metadata_and_provenance(tmp_path: Path, capsys) -> None:
    path = make_pdf(
        tmp_path,
        metadata={"title": "A Meaningful Academic Title"},
        lines=[("Ordinary body text on the first page.", 10.0, 120.0)],
    )

    exit_code = inspect_cli.main([str(path), "--offline"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Preferred metadata:" in output
    assert "Title: A Meaningful Academic Title" in output
    assert "Provenance:" in output
    assert "Title: PDF_METADATA / MEDIUM" in output
    # The CLI must not dump document contents.
    assert "Ordinary body text" not in output


def test_cli_offline_builds_no_enricher(tmp_path: Path, monkeypatch, capsys) -> None:
    path = make_pdf(
        tmp_path,
        metadata={"title": "A Meaningful Academic Title"},
        lines=[("Ordinary body text on the first page.", 10.0, 120.0)],
    )
    calls = {"count": 0}

    def _fail(*args, **kwargs):
        calls["count"] += 1
        raise AssertionError("enricher must not be built in offline mode")

    monkeypatch.setattr(inspect_cli, "build_default_enricher", _fail)

    assert inspect_cli.main([str(path), "--offline"]) == 0
    capsys.readouterr()
    assert calls["count"] == 0


def test_cli_missing_file_returns_error(tmp_path: Path, capsys) -> None:
    exit_code = inspect_cli.main([str(tmp_path / "missing.pdf"), "--offline"])
    assert exit_code == 2
    assert "not found" in capsys.readouterr().err


def test_cli_show_candidates(tmp_path: Path, capsys) -> None:
    path = make_pdf(
        tmp_path,
        metadata={"title": "A Meaningful Academic Title"},
        lines=[("Ordinary body text on the first page.", 10.0, 120.0)],
    )
    assert inspect_cli.main([str(path), "--offline", "--show-candidates"]) == 0
    output = capsys.readouterr().out
    assert "Candidate history:" in output


def test_cli_prints_doi_conflict(tmp_path: Path, monkeypatch, capsys) -> None:
    literature = tmp_path / "data" / "literature"
    literature.mkdir(parents=True)
    path = make_pdf(
        literature,
        lines=[
            ("A Prominent First Page Title", 18.0, 80.0),
            ("Ordinary body text with doi:10.1000/xyz here.", 10.0, 130.0),
        ],
    )

    def _enricher(*args, **kwargs):
        return build_test_enricher(
            FakeTransport(
                crossref_doi_handler(
                    crossref_message(
                        doi="10.1000/xyz",
                        title="A Totally Different Work About Widgets",
                    )
                )
            ),
            crossref=True,
            openalex=False,
        )

    monkeypatch.setattr(inspect_cli, "build_default_enricher", _enricher)

    assert inspect_cli.main([str(path)]) == 0
    output = capsys.readouterr().out
    assert "External match status: CONFLICT" in output
    assert "Conflicts" in output
    assert "A Prominent First Page Title" in output
