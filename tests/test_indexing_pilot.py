"""RAG-008: deterministic selection of the indexing population and the pilot sample."""

from __future__ import annotations

from pathlib import Path

from app.chunking.models import ChunkingOptions
from app.indexing.indexer import iter_population
from app.indexing.models import IndexingOptions

_BODY = (
    "The chapter reviews how immersive virtual reality is used in higher education "
    "and reports the learning outcomes that the studies describe. "
) * 5


def _corpus(root: Path) -> Path:
    for folder in ("literature", "my_work", "future_chapters"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    for index in range(3):
        (root / "literature" / f"paper-{index}.md").write_text(
            f"# Literature {index}\n\n{_BODY}\n\n## References\n\nSmith, J. (2022). Cited.\n",
            encoding="utf-8",
        )
    (root / "my_work" / "draft.md").write_text(f"# My Draft\n\n{_BODY}\n", encoding="utf-8")
    (root / "future_chapters" / "plan.md").write_text(
        f"# Future Plan\n\n{_BODY}\n", encoding="utf-8"
    )
    return root


def test_full_population_yields_every_chunk_with_eligibility(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "data")

    population = list(iter_population(root, options=IndexingOptions()))

    assert len(population) == 8  # 4 body chunks x 2 + 3 documents + 1 reference per paper
    eligible = [item for item, flag in population if flag]
    excluded = [item for item, flag in population if not flag]
    assert len(eligible) == 5
    assert len(excluded) == 3
    assert all(item.chunk.content_type.value == "references" for item in excluded)


def test_pilot_takes_one_chunk_per_document_and_covers_roles(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "data")

    pilot = list(iter_population(root, options=IndexingOptions(pilot=4)))
    filenames = [item.filename for item, _flag in pilot]
    roles = {item.chunk.corpus_role.value for item, _flag in pilot}

    # Non-literature roles are visited first, so four chunks cover every required
    # role without needing a top-up.
    assert len(pilot) == 4
    assert len(set(filenames)) == 4
    assert {"literature", "my_work", "future_chapters"} <= roles
    assert {"draft.md", "plan.md"} <= set(filenames)
    assert all(item.chunk.retrieval_eligible for item, _flag in pilot)


def test_pilot_selection_is_deterministic(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "data")
    options = IndexingOptions(pilot=6)

    first = [item.chunk_id for item, _flag in iter_population(root, options=options)]
    second = [item.chunk_id for item, _flag in iter_population(root, options=options)]

    assert first == second


def test_document_limit_restricts_the_visit_order(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "data")

    limited = list(iter_population(root, options=IndexingOptions(limit=1)))

    assert {item.filename for item, _flag in limited} == {"draft.md"}


def test_iteration_uses_rag007_chunks(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "data")

    population = list(iter_population(root, options=IndexingOptions(limit=1)))

    assert population
    for item, _flag in population:
        assert item.chunk.id is not None
        assert item.chunk.document_id is not None
        assert item.retrieval_text  # never empty: retrieval text is what gets embedded
        assert item.retrieval_text == (item.chunk.retrieval_text or item.chunk.text)


def test_chunking_options_are_passed_through(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "data")
    options = IndexingOptions()
    without_labels = ChunkingOptions(include_context_labels=False)

    with_labels = list(iter_population(root, options=options))
    without = list(iter_population(root, options=options, chunking=without_labels))

    assert "[Document:" in with_labels[0][0].retrieval_text
    assert "[Document:" not in without[0][0].retrieval_text
