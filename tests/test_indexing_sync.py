"""RAG-008 tests 22-24: stale point detection and explicit synchronisation."""

from __future__ import annotations

from pathlib import Path

from embedding_support import FakeEmbedder, local_client

from app.chunking.models import ChunkingOptions
from app.indexing.collection import count_points, scroll_point_ids
from app.indexing.indexer import run_index
from app.indexing.models import IndexingOptions

_BODY = (
    "Chunking academic text keeps document, page and section provenance intact so "
    "answers can point back to the exact place they came from. "
) * 4


def _corpus(root: Path, filenames: list[str]) -> Path:
    (root / "literature").mkdir(parents=True, exist_ok=True)
    for name in filenames:
        (root / "literature" / name).write_text(f"# {name}\n\n{_BODY}\n", encoding="utf-8")
    return root


def test_stale_points_are_detected_and_reported(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "data", ["one.md", "two.md"])
    client = local_client(tmp_path)
    options = IndexingOptions(collection="stale", use_cache=False)

    first = run_index(root, options=options, embedder=FakeEmbedder(), client=client)
    assert first.stale_points == 0
    before = count_points(client, "stale")

    (root / "literature" / "two.md").unlink()
    second = run_index(root, options=options, embedder=FakeEmbedder(), client=client)

    assert before == 2
    assert second.stale_points == 1
    assert second.stale_point_ids
    assert second.deleted_stale_points == 0
    # Without `--delete-stale` the stale point is still there - never a silent delete.
    assert count_points(client, "stale") == 2
    assert set(second.stale_point_ids) <= set(scroll_point_ids(client, "stale"))


def test_stale_point_deleted_with_explicit_flag(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "data", ["one.md", "two.md"])
    client = local_client(tmp_path)
    options = IndexingOptions(collection="sync", use_cache=False)
    run_index(root, options=options, embedder=FakeEmbedder(), client=client)

    (root / "literature" / "two.md").unlink()
    report = run_index(
        root,
        options=options.model_copy(update={"delete_stale": True}),
        embedder=FakeEmbedder(),
        client=client,
    )

    assert report.stale_points == 1
    assert report.deleted_stale_points == 1
    assert count_points(client, "sync") == 1
    assert report.points_after == 1


def test_stale_detection_is_scoped_to_the_collection(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "data", ["one.md"])
    client = local_client(tmp_path)
    run_index(
        root,
        options=IndexingOptions(collection="alpha", use_cache=False),
        embedder=FakeEmbedder(),
        client=client,
    )
    run_index(
        root,
        options=IndexingOptions(collection="beta", use_cache=False),
        embedder=FakeEmbedder(),
        client=client,
    )

    report = run_index(
        root,
        options=IndexingOptions(collection="beta", use_cache=False, delete_stale=True),
        embedder=FakeEmbedder(),
        client=client,
    )

    assert report.stale_points == 0
    assert count_points(client, "alpha") == 1
    assert count_points(client, "beta") == 1
