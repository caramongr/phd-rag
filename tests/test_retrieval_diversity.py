"""RAG-009 checks 19-21: document diversity and near-duplicate suppression."""

from __future__ import annotations

import pytest

from app.retrieval.diversity import (
    REASON_DOCUMENT_LIMIT,
    REASON_DUPLICATE_NEIGHBOUR,
    apply_document_limit,
    normalized_text_hash,
    suppress_near_duplicates,
    token_overlap,
    work_key,
)
from app.retrieval.models import Candidate, RetrievalMode
from app.retrieval.service import RetrievalService
from tests.retrieval_support import (
    TEST_COLLECTION,
    FakeReranker,
    QueryEmbedder,
    corpus_points,
    direction,
    memory_collection,
)


def candidate(
    chunk_id: str,
    *,
    document_id: str = "doc-1",
    chunk_index: int = 0,
    text: str = "text",
    rerank_score: float | None = None,
    title: str | None = None,
) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        retrieval_text=text,
        title=title,
        rerank_score=rerank_score,
    )


def test_max_chunks_per_document_is_enforced() -> None:
    """Check 19."""
    candidates = [
        candidate(f"c{index}", chunk_index=index, text=f"unique passage {index}", rerank_score=1 - index / 10)
        for index in range(6)
    ]
    kept, dropped = apply_document_limit(candidates, max_chunks_per_document=3)
    assert [item.chunk_id for item in kept] == ["c0", "c1", "c2"]
    assert len(dropped) == 3
    assert {item.reason for item in dropped} == {REASON_DOCUMENT_LIMIT}
    assert all(item.detail and "limit 3" in item.detail for item in dropped)


def test_document_limit_is_a_cap_not_a_quota() -> None:
    """Check 21: distinct chunks from one document may coexist up to the limit."""
    candidates = [
        candidate("a", document_id="d1", chunk_index=0, text="alpha passage"),
        candidate("b", document_id="d1", chunk_index=5, text="beta passage"),
        candidate("c", document_id="d1", chunk_index=9, text="gamma passage"),
        candidate("d", document_id="d2", chunk_index=0, text="delta passage"),
    ]
    kept, dropped = apply_document_limit(candidates, max_chunks_per_document=3)
    assert [item.chunk_id for item in kept] == ["a", "b", "c", "d"]
    assert dropped == []
    unlimited, _ = apply_document_limit(candidates, max_chunks_per_document=None)
    assert len(unlimited) == 4


def test_lower_ranked_neighbour_with_high_overlap_is_suppressed() -> None:
    """Check 20: adjacent chunks of one document with near-identical text collapse."""
    first = candidate(
        "k0",
        chunk_index=3,
        text="artificial intelligence in higher education improves learning outcomes for students",
        rerank_score=0.9,
    )
    second = candidate(
        "k1",
        chunk_index=4,
        text="artificial intelligence in higher education improves learning outcomes for students today",
        rerank_score=0.8,
    )
    kept, dropped = suppress_near_duplicates([first, second])
    assert [item.chunk_id for item in kept] == ["k0"]
    assert len(dropped) == 1
    assert dropped[0].reason == REASON_DUPLICATE_NEIGHBOUR
    assert dropped[0].chunk_id == "k1"
    assert "kept k0" in (dropped[0].detail or "")


def test_identical_text_is_suppressed_even_when_not_adjacent() -> None:
    text = "the same passage appearing twice"
    kept, dropped = suppress_near_duplicates(
        [
            candidate("k0", chunk_index=1, text=text, rerank_score=0.7),
            candidate("k9", chunk_index=40, text=text, rerank_score=0.6),
        ]
    )
    assert [item.chunk_id for item in kept] == ["k0"]
    assert dropped[0].reason == "identical_text"
    assert normalized_text_hash(text) == normalized_text_hash(f"  {text}  ")


def test_identical_passage_from_two_copies_of_one_work_is_suppressed() -> None:
    """Copies of one work are different documents (different hashes), same text.

    The corpus really contains several physical copies of some published papers, so
    the identical-text rule cannot be scoped to one document id.
    """
    text = "4.5.2 Research design framework we used existing research design categories"
    kept, dropped = suppress_near_duplicates(
        [
            candidate("copy-a", document_id="doc-a", chunk_index=11, text=text),
            candidate("copy-b", document_id="doc-b", chunk_index=13, text=text),
            candidate("copy-c", document_id="doc-c", chunk_index=11, text=text),
        ]
    )
    assert [item.chunk_id for item in kept] == ["copy-a"]
    assert [item.chunk_id for item in dropped] == ["copy-b", "copy-c"]
    assert all("another copy of the work" in (item.detail or "") for item in dropped)

    relaxed, relaxed_dropped = suppress_near_duplicates(
        [
            candidate("copy-a", document_id="doc-a", chunk_index=11, text=text),
            candidate("copy-b", document_id="doc-b", chunk_index=13, text=text),
        ],
        suppress_identical_across_documents=False,
    )
    assert len(relaxed) == 2 and relaxed_dropped == []


def test_similar_but_distinct_passages_survive() -> None:
    """Suppression must stay conservative: different content is different evidence."""
    kept, dropped = suppress_near_duplicates(
        [
            candidate(
                "k0",
                chunk_index=1,
                text="Virtual reality increases student engagement in laboratory courses.",
            ),
            candidate(
                "k1",
                chunk_index=2,
                text="Adaptive learning systems adjust task difficulty using knowledge tracing models.",
            ),
        ]
    )
    assert len(kept) == 2
    assert dropped == []

    # neighbouring but only partially overlapping text stays as well
    kept, dropped = suppress_near_duplicates(
        [
            candidate("k0", chunk_index=1, text="alpha beta gamma delta epsilon zeta"),
            candidate("k1", chunk_index=2, text="alpha beta gamma iota kappa lambda"),
        ]
    )
    assert len(kept) == 2


def test_overlap_helper_is_containment_of_the_smaller_set() -> None:
    assert token_overlap("alpha beta gamma", "alpha beta gamma delta") == pytest.approx(1.0)
    assert token_overlap("alpha beta gamma", "delta epsilon zeta") == pytest.approx(0.0)
    assert token_overlap("", "anything") == pytest.approx(0.0)


def test_document_limit_groups_copies_of_one_work() -> None:
    """Different copies of one paper (different ids) share one work key."""
    copies = [
        candidate(
            f"copy-{index}",
            document_id=f"doc-{index}",
            chunk_index=index,
            text=f"distinct passage {index}",
            title="A systematic review of immersive virtual reality applications",
        )
        for index in range(5)
    ]
    assert len({work_key(item) for item in copies}) == 1
    kept, dropped = apply_document_limit(copies, max_chunks_per_document=3)
    assert [item.chunk_id for item in kept] == ["copy-0", "copy-1", "copy-2"]
    assert len(dropped) == 2
    assert all("title:" in (item.detail or "") for item in dropped)

    # without work grouping the cap falls back to the physical document
    per_document, per_document_dropped = apply_document_limit(
        copies, max_chunks_per_document=3, group_by_work=False
    )
    assert len(per_document) == 5
    assert per_document_dropped == []


def test_work_key_does_not_merge_generic_or_missing_titles() -> None:
    """Short generic titles are shared by unrelated documents, so they never group."""
    for title in (None, "", "Form", "Book Reviews", "PRIVACY POLICY", "Windows User"):
        assert work_key(candidate("c1", document_id="doc-1", title=title)) == "document:doc-1"
    distinctive = "Positioning and Presenting Design Science Research for Maximum Impact"
    assert work_key(candidate("c1", document_id="doc-1", title=distinctive)).startswith("title:")
    # ... while two documents with the same distinctive title group together
    first = work_key(candidate("c1", document_id="doc-1", title=distinctive))
    second = work_key(candidate("c2", document_id="doc-2", title=distinctive.upper()))
    assert first == second
    # a ligature is a spelling variant, not a different work
    assert work_key(
        candidate("c3", document_id="doc-3", title="Adaptive Learning Using Artiﬁcial Intelligence")
    ) == work_key(
        candidate("c4", document_id="doc-4", title="Adaptive learning using artificial intelligence")
    )


def test_service_reports_duplicates_and_diversity_drops() -> None:
    """Checks 19-21 through the full pipeline, including the audit trail."""
    points = list(corpus_points())
    base = next(item for item in points if item["payload"]["chunk_id"] == "lit-a1")
    # a near-copy of lit-a1 with an adjacent index: overlapping neighbour
    neighbour = dict(base["payload"])
    neighbour.update(
        {
            "chunk_id": "lit-a1b",
            "chunk_index": 1,
            "text": base["payload"]["text"] + " today",
            "retrieval_text": base["payload"]["text"] + " today",
        }
    )
    points.append({"vector": base["vector"], "payload": neighbour})
    # two more distinct chunks of the same document, to force the per-document cap
    for index, angle in ((7, 60.0), (8, 61.0)):
        extra = dict(base["payload"])
        extra.update(
            {
                "chunk_id": f"lit-a{index}",
                "chunk_index": index,
                "text": f"a distinct passage number {index} about tutoring systems",
                "retrieval_text": f"a distinct passage number {index} about tutoring systems",
            }
        )
        points.append({"vector": direction(angle), "payload": extra})

    client = memory_collection(points)
    service = RetrievalService(
        embedder=QueryEmbedder(),
        reranker=FakeReranker(
            score_fn=lambda _q, cid, _t: 1.0 if cid == "lit-a1" else 0.5 if cid == "lit-a1b" else 0.1
        ),
        client=client,
        collection=TEST_COLLECTION,
        final_k=10,
        max_chunks_per_document=2,
    )
    response = service.retrieve("anything", mode=RetrievalMode.GENERAL)
    reasons = {item.reason for item in response.dropped}
    assert REASON_DUPLICATE_NEIGHBOUR in reasons
    assert REASON_DOCUMENT_LIMIT in reasons
    per_document: dict[str, int] = {}
    for item in response.results:
        per_document[item.document_id] = per_document.get(item.document_id, 0) + 1
    assert max(per_document.values()) <= 2
    assert response.stats.dropped_as_duplicate >= 1
    assert response.stats.dropped_by_diversity >= 1
    service.close()
