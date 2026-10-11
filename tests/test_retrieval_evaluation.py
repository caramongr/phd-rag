"""RAG-009 §19/§20: the evaluation harness and its metrics (no real models)."""

from __future__ import annotations

import pytest

from app.retrieval.evaluate import hit_at, mrr_at
from app.retrieval.evaluation_set import EVAL_QUERIES, counts, gold_needles, query_set
from app.retrieval.models import RetrievalMode


def test_evaluation_set_is_large_enough_and_documented() -> None:
    """§19: at least 20 queries, each with a mode and a category."""
    assert len(EVAL_QUERIES) >= 20
    for entry in EVAL_QUERIES:
        assert entry.query.strip()
        assert isinstance(entry.mode, RetrievalMode)
        assert entry.category
        assert entry.gold_titles or entry.gold_filenames or entry.is_exploratory


def test_evaluation_set_covers_the_required_topics() -> None:
    """§19: the brief's topic list is represented, and non-evidence modes appear."""
    categories = {entry.category for entry in EVAL_QUERIES}
    for required in (
        "ai-in-he",
        "vr-engagement",
        "adaptive-learning",
        "intelligent-tutoring",
        "learning-analytics",
        "dsr-methodology",
        "ai-literacy",
        "vr-medical-education",
        "academic-integrity",
        "generative-ai",
        "personalization",
        "governance",
        "ethics",
    ):
        assert required in categories, required
    modes = {entry.mode for entry in EVAL_QUERIES}
    assert {RetrievalMode.EVIDENCE, RetrievalMode.MY_WORK, RetrievalMode.SUPERVISION, RetrievalMode.GUIDANCE} <= modes


def test_required_example_queries_are_in_the_set() -> None:
    """§28's verbatim queries must be part of the set (or deliberately additive)."""
    queries = {entry.query for entry in EVAL_QUERIES}
    for required in (
        "artificial intelligence governance in higher education",
        "virtual reality student engagement higher education",
    ):
        assert required in queries
    assert any("generative AI academic integrity" in entry.query for entry in EVAL_QUERIES)
    assert any("design science research methodology" in entry.query for entry in EVAL_QUERIES)
    assert any("historical development of AI" in entry.query for entry in EVAL_QUERIES)
    assert any("supervisor say about methodology" in entry.query for entry in EVAL_QUERIES)


def test_exploratory_queries_assert_no_gold() -> None:
    exploratory = [entry for entry in EVAL_QUERIES if entry.is_exploratory]
    assert exploratory, "some queries must honestly admit they have no gold document"
    for entry in exploratory:
        assert entry.gold_titles == () and entry.gold_filenames == ()


def test_gold_needles_are_lowercase_and_distinct() -> None:
    needles = gold_needles()
    assert needles
    assert len(needles) == len(set(needles))
    for needle in needles:
        assert needle == needle.casefold()
        assert needle.strip()


def test_counts_summarises_the_set() -> None:
    summary = counts()
    assert summary["total"] == len(EVAL_QUERIES)
    assert summary["gold"] + summary["exploratory"] == summary["total"]
    assert summary["mode_evidence"] >= 20
    assert query_set() == EVAL_QUERIES


def test_hit_at_and_mrr_at_maths() -> None:
    ranks = [1, 2, 6, None]
    assert hit_at(ranks, 5) == pytest.approx(0.5)
    assert hit_at(ranks, 10) == pytest.approx(0.75)
    assert mrr_at(ranks, 10) == pytest.approx((1 + 0.5 + 1 / 6) / 4)
    assert hit_at([], 5) == 0.0
    assert mrr_at([], 10) == 0.0
    assert hit_at([None, None], 10) == 0.0
    assert mrr_at([2], 1) == 0.0
