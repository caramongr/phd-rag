"""RAG-009 checks 22-24: determinism, filter-order invariance and the CLI."""

from __future__ import annotations

import json

import pytest

from app.retrieval.inspect import build_filters, format_response, main, parse_args
from app.retrieval.models import RetrievalFilters, RetrievalMode
from app.retrieval.service import RetrievalService
from tests.retrieval_support import TEST_COLLECTION, FakeReranker, QueryEmbedder, corpus_points, local_collection, memory_collection


def test_same_query_gives_the_same_ordered_ids() -> None:
    """Check 22."""
    client = memory_collection(corpus_points())
    service = RetrievalService(
        embedder=QueryEmbedder(),
        reranker=FakeReranker(score_fn=lambda _q, cid, _t: float(len(cid))),
        client=client,
        collection=TEST_COLLECTION,
        final_k=6,
    )
    first = service.retrieve("artificial intelligence", mode=RetrievalMode.GENERAL)
    second = service.retrieve("artificial intelligence", mode=RetrievalMode.GENERAL)
    third = service.retrieve("artificial intelligence", mode=RetrievalMode.GENERAL)
    assert first.chunk_ids == second.chunk_ids == third.chunk_ids
    assert [c.final_rank for c in first.results] == list(range(1, len(first.results) + 1))
    assert [c.rerank_score for c in first.results] == [c.rerank_score for c in third.results]
    service.close()


def test_filter_order_does_not_change_the_result_set() -> None:
    """Check 23: filters are conjunctions; the order they are set in is irrelevant."""
    client = memory_collection(corpus_points())
    service = RetrievalService(
        embedder=QueryEmbedder(), client=client, collection=TEST_COLLECTION, final_k=10
    )
    combos = [
        RetrievalFilters(year_min=2000, content_types=("body",), publication_type="journal-article"),
        RetrievalFilters(publication_type="journal-article", content_types=("body",), year_min=2000),
        RetrievalFilters(content_types=("body",), year_min=2000, publication_type="journal-article"),
    ]
    results = [
        service.retrieve("anything", mode=RetrievalMode.GENERAL, filters=filters).chunk_ids
        for filters in combos
    ]
    assert results[0] == results[1] == results[2]
    assert results[0]
    service.close()


def test_determinism_survives_a_persistent_client_restart(tmp_path) -> None:
    """The same index read from a new process yields the same ordering."""
    client = local_collection(tmp_path, corpus_points())
    service = RetrievalService(
        embedder=QueryEmbedder(), client=client, collection=TEST_COLLECTION, final_k=6
    )
    first = service.retrieve("artificial intelligence", mode=RetrievalMode.GENERAL).chunk_ids
    client.close()

    from qdrant_client import QdrantClient

    reopened = QdrantClient(path=str(tmp_path / "q"))
    service = RetrievalService(
        embedder=QueryEmbedder(), client=reopened, collection=TEST_COLLECTION, final_k=6
    )
    second = service.retrieve("artificial intelligence", mode=RetrievalMode.GENERAL).chunk_ids
    reopened.close()
    assert first == second


def test_audit_record_reproduces_the_configuration() -> None:
    """Check 24."""
    client = memory_collection(corpus_points())
    service = RetrievalService(
        embedder=QueryEmbedder(),
        reranker=FakeReranker(batch_size=4, device="cpu"),
        client=client,
        collection=TEST_COLLECTION,
        candidate_k=11,
        final_k=4,
        max_chunks_per_document=2,
    )
    response = service.retrieve(
        "  adaptive   learning ",
        mode=RetrievalMode.GUIDANCE,
        filters=RetrievalFilters(year_min=2010),
        rerank=True,
    )
    audit = response.audit
    assert audit.normalized_query == "adaptive learning"
    assert audit.filters == {"year_min": 2010}
    assert (audit.candidate_k, audit.final_k) == (11, 4)
    assert audit.max_chunks_per_document == 2
    assert audit.suppress_duplicates is True
    assert audit.duplicate_overlap_threshold == pytest.approx(0.85)
    assert audit.rerank_enabled is True
    # a second, identical call produces the same audit configuration (only the
    # timestamp may differ)
    again = service.retrieve(
        "  adaptive   learning ",
        mode=RetrievalMode.GUIDANCE,
        filters=RetrievalFilters(year_min=2010),
        rerank=True,
    ).audit
    assert again.model_dump(exclude={"timestamp"}) == audit.model_dump(exclude={"timestamp"})
    service.close()


def test_cli_flags_map_to_filters_and_modes() -> None:
    args = parse_args(
        [
            "anything",
            "--mode",
            "guidance",
            "--candidate-k",
            "30",
            "--final-k",
            "5",
            "--year-min",
            "2015",
            "--year-max",
            "2025",
            "--content-type",
            "body",
            "--content-type",
            "table",
            "--corpus-role",
            "literature",
            "--max-rounds",
            "2",
            "--no-rerank",
            "--ocr-only",
        ]
    )
    filters = build_filters(args)
    assert filters.year_min == 2015 and filters.year_max == 2025
    assert filters.content_types == ("body", "table")
    assert filters.corpus_role == "literature"
    assert filters.ocr_involved is True
    assert args.no_rerank is True
    assert args.max_rounds == 2
    assert RetrievalMode(args.mode) is RetrievalMode.GUIDANCE


def test_cli_rejects_an_unknown_mode() -> None:
    with pytest.raises(SystemExit):
        parse_args(["anything", "--mode", "nonsense"])


def test_cli_reports_results_readably() -> None:
    client = memory_collection(corpus_points())
    service = RetrievalService(
        embedder=QueryEmbedder(),
        reranker=FakeReranker(score=0.7),
        client=client,
        collection=TEST_COLLECTION,
        final_k=3,
    )
    response = service.retrieve("artificial intelligence", mode=RetrievalMode.GENERAL)
    text = format_response(response)
    assert "mode       : general" in text
    assert "evidence=yes" in text or "evidence=no" in text
    assert "dense=" in text and "rerank=" in text
    assert "top 3 results" in text or "top " in text
    service.close()


def test_cli_main_end_to_end_on_a_local_collection(tmp_path, monkeypatch, capsys) -> None:
    """The CLI runs against a real client path with the fake models patched in."""
    client = local_collection(tmp_path, corpus_points())
    client.close()

    import app.embedding.bge_m3 as bge
    import app.reranking.bge_reranker as reranker_module
    import app.retrieval.inspect as inspect_module

    monkeypatch.setattr(bge, "BgeM3Embedder", lambda **_: QueryEmbedder())
    monkeypatch.setattr(
        reranker_module,
        "BgeReranker",
        lambda **_: FakeReranker(score_fn=lambda _q, cid, _t: float(len(cid))),
    )
    assert reranker_module.BgeReranker is not None

    exit_code = main(
        [
            "artificial intelligence in higher education",
            "--mode",
            "general",
            "--qdrant-path",
            str(tmp_path / "q"),
            "--collection",
            TEST_COLLECTION,
            "--final-k",
            "3",
            "--show-text",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "query      : artificial intelligence in higher education" in captured.out
    assert "role    :" in captured.out
    assert inspect_module is not None


def test_cli_json_output_is_valid(tmp_path, monkeypatch, capsys) -> None:
    client = local_collection(tmp_path, corpus_points())
    client.close()

    import app.embedding.bge_m3 as bge
    import app.reranking.bge_reranker as reranker_module

    monkeypatch.setattr(bge, "BgeM3Embedder", lambda **_: QueryEmbedder())
    monkeypatch.setattr(reranker_module, "BgeReranker", lambda **_: FakeReranker(score=0.5))

    main(
        [
            "anything",
            "--mode",
            "general",
            "--qdrant-path",
            str(tmp_path / "q"),
            "--collection",
            TEST_COLLECTION,
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["audit"]["collection"] == TEST_COLLECTION
    assert payload["audit"]["rerank_enabled"] is True
    assert payload["results"]
