"""RAG-008: the indexing and embedding CLIs (dry run, pilot, inspection)."""

from __future__ import annotations

from pathlib import Path

import pytest
from embedding_support import FakeEmbedder, FakeSentenceTransformer, local_client

from app.embedding.inspect import main as embedding_inspect_main
from app.embedding.inspect import render as render_embedding
from app.indexing.build import EXIT_OK, EXIT_TOKEN_LIMIT, EXIT_USAGE, main as build_main
from app.indexing.collection import count_points, fetch_payloads, scroll_point_ids
from app.indexing.inspect import collect_statistics, main as inspect_main, render as render_inspection

_BODY = (
    "Academic-aware chunking preserves page, section and OCR provenance, which keeps "
    "every retrieved passage traceable to the source document. "
) * 4


def _corpus(root: Path) -> Path:
    for folder in ("literature", "my_work", "future_chapters"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    (root / "literature" / "paper.md").write_text(
        f"# A Literature Paper\n\n{_BODY}\n## References\n\nSmith, J. (2022). Cited work.\n",
        encoding="utf-8",
    )
    (root / "my_work" / "draft.md").write_text(f"# My Draft\n\n{_BODY}\n", encoding="utf-8")
    (root / "future_chapters" / "plan.md").write_text(
        f"# Future Plan\n\n{_BODY}\n", encoding="utf-8"
    )
    return root


def _isolated(tmp_path: Path) -> list[str]:
    """Cache and manifest paths under tmp_path, so tests never touch the repo."""
    return [
        "--cache-path",
        str(tmp_path / "embedding-cache.npz"),
        "--manifest-path",
        str(tmp_path / "index-manifest.json"),
    ]


def test_dry_run_reports_population_without_touching_qdrant(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "data")
    qdrant_path = tmp_path / "qdrant"

    exit_code = build_main(
        [str(root), "--qdrant-path", str(qdrant_path), "--dry-run", *_isolated(tmp_path)]
    )

    assert exit_code == EXIT_OK
    assert not qdrant_path.exists()


def test_dry_run_pilot_reports_kinds(tmp_path: Path, capsys) -> None:
    root = _corpus(tmp_path / "data")

    exit_code = build_main(
        [
            str(root),
            "--qdrant-path",
            str(tmp_path / "qdrant"),
            "--dry-run",
            "--pilot",
            "3",
            *_isolated(tmp_path),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == EXIT_OK
    assert "pilot chunks:" in output
    assert "my_work" in output and "future_chapters" in output


def test_dry_run_requires_a_qdrant_target(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "data")

    exit_code = build_main([str(root), "--qdrant-path", "", "--qdrant-url", "", "--dry-run"])

    assert exit_code == EXIT_USAGE


def test_build_runs_pilot_end_to_end_with_injected_doubles(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "data")
    qdrant_path = tmp_path / "qdrant"
    out = tmp_path / "report.txt"
    manifest = tmp_path / "manifest.json"
    embedder = FakeEmbedder()

    exit_code = build_main(
        [
            str(root),
            "--qdrant-path",
            str(qdrant_path),
            "--collection",
            "pilot_chunks",
            "--pilot",
            "3",
            "--report-out",
            str(out),
            "--cache-path",
            str(tmp_path / "cache.npz"),
            "--manifest-path",
            str(manifest),
        ],
        embedder_factory=lambda _args: embedder,
        client_factory=lambda _args: local_client(tmp_path),
    )
    text = out.read_text(encoding="utf-8")

    assert exit_code == EXIT_OK
    assert "RAG-008 indexing report" in text
    assert "points indexed:             3" in text
    assert embedder.embed_calls == 1
    assert manifest.is_file()
    assert "corpus fingerprint" in text


def test_build_measure_tokens_reports_tokenizer_stats(tmp_path: Path, capsys) -> None:
    root = _corpus(tmp_path / "data")

    exit_code = build_main(
        [
            str(root),
            "--qdrant-path",
            str(tmp_path / "qdrant"),
            "--dry-run",
            "--measure-tokens",
            *_isolated(tmp_path),
        ],
        embedder_factory=lambda _args: FakeEmbedder(token_counter=lambda text: len(text.split())),
    )
    output = capsys.readouterr().out

    assert exit_code == EXIT_OK
    assert "BGE-M3 tokenizer measurement" in output
    assert "chunks above model limit:   0" in output
    assert "real tokens/second:" in output
    assert "peak allocated VRAM:" in output


def test_inspection_cli_reports_distributions(tmp_path: Path, capsys) -> None:
    root = _corpus(tmp_path / "data")
    qdrant_path = tmp_path / "qdrant"
    build_main(
        [
            str(root),
            "--qdrant-path",
            str(qdrant_path),
            "--collection",
            "inspect_me",
            *_isolated(tmp_path),
        ],
        embedder_factory=lambda _args: FakeEmbedder(),
        client_factory=lambda _args: local_client(tmp_path),
    )

    exit_code = inspect_main(
        ["--qdrant-path", str(qdrant_path), "--collection", "inspect_me", "--limit", "2"]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "[ok] no reference chunk indexed" in output
    assert "[ok] my_work present" in output
    assert "[ok] future_chapters present" in output
    assert "[ok] my_work is retrievable but not evidence" in output


def test_inspection_reports_missing_collection(tmp_path: Path, capsys) -> None:
    exit_code = inspect_main(
        ["--qdrant-path", str(tmp_path / "empty"), "--collection", "missing"]
    )

    assert exit_code == 3
    assert "not readable" in capsys.readouterr().err


def test_embedding_inspection_renders_model_identity() -> None:
    from app.embedding.bge_m3 import BgeM3Embedder

    embedder = BgeM3Embedder(
        model_name="fake-model",
        device="cpu",
        model=FakeSentenceTransformer(dimension=5),
    )

    text = render_embedding(embedder, "sample input text", show_vector=3)

    assert "vector dimension:      5" in text
    assert "l2 (unit norm)" in text
    assert "sample vector norm:    1.0" in text
    assert "sample vector (first 3 dims)" in text


def test_embedding_inspection_cli_with_injected_embedder(capsys) -> None:
    from app.embedding.bge_m3 import BgeM3Embedder

    def factory(**_kwargs):
        return BgeM3Embedder(
            model_name="fake-model", device="cpu", model=FakeSentenceTransformer(dimension=3)
        )

    exit_code = embedding_inspect_main(["--text", "hello"], embedder_factory=factory)
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "RAG-008 embedding inspection" in output
    assert "vector dimension:      3" in output


def test_payloads_expose_roles_for_filtering(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "data")
    qdrant_path = tmp_path / "qdrant"
    build_main(
        [
            str(root),
            "--qdrant-path",
            str(qdrant_path),
            "--collection",
            "roles",
            *_isolated(tmp_path),
        ],
        embedder_factory=lambda _args: FakeEmbedder(),
        client_factory=lambda _args: local_client(tmp_path),
    )
    client = local_client(tmp_path)
    payloads = [
        fetch_payloads(client, "roles", [point_id])[0]
        for point_id in scroll_point_ids(client, "roles")
    ]

    assert count_points(client, "roles") == len(payloads) == 3
    roles = {payload["corpus_role"]: payload for payload in payloads}
    assert roles["literature"]["evidence_eligible"] is True
    assert roles["my_work"]["evidence_eligible"] is False
    assert roles["future_chapters"]["evidence_eligible"] is False
    assert all(payload["retrieval_eligible"] is True for payload in payloads)
