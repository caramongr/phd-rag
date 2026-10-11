"""Context construction tests (RAG-011 §31 items 1-5, 22-23).

The context is the only path documents take into the model, so these tests check the
things that make it trustworthy: correct conversion, stable ids, an honest budget, and
labels that keep non-evidence material out of factual claims.
"""

from __future__ import annotations

from app.generation.context import (
    BLOCK_OVERHEAD_TOKENS,
    TEXT_CLOSE,
    TEXT_OPEN,
    ContextBuildRequest,
    ExternalSource,
    build_context,
    estimate_tokens,
    external_sources_from_records,
)
from app.generation.models import GenerationMode, SourceClass
from tests.generation_support import candidate, external_record


def _package(**kwargs: object):
    """Build a context package with test defaults."""
    request = ContextBuildRequest(**kwargs)  # type: ignore[arg-type]
    return build_context(request)


# --- 1. evidence results become source blocks correctly ----------------------


def test_evidence_candidates_become_labelled_source_blocks() -> None:
    """Every retrieved field a citation needs survives into the block."""
    result = _package(
        evidence=[
            candidate(
                "c1",
                text="Governance is described here.",
                title="AI Governance in Universities",
                first_author="Jane Doe",
                year=2024,
                doi="10.1234/abc",
                page_start=4,
                page_end=6,
                section_title="Introduction",
                rerank_score=0.81,
            )
        ]
    )

    assert result.included_chunks == 1
    block = result.blocks[0]
    assert block.source_id == "S1"
    assert block.source_class is SourceClass.EVIDENCE
    assert block.citable is True
    assert block.chunk_id == "c1"
    assert block.title == "AI Governance in Universities"
    assert block.first_author == "Jane Doe"
    assert block.year == 2024
    assert block.doi == "10.1234/abc"
    assert block.pages == "4-6"
    assert block.section_title == "Introduction"
    assert block.corpus_role == "literature"
    assert block.evidence_eligible is True
    assert block.ocr_involved is False
    assert block.content_type == "body"
    assert block.rerank_score == 0.81
    assert block.text == "Governance is described here."


def test_source_map_points_every_marker_at_an_exact_chunk() -> None:
    """``source_map`` is the trace RAG-012 will need (§12)."""
    result = _package(evidence=[candidate("c1"), candidate("c2")])
    source_map = result.source_map()

    assert set(source_map) == {"S1", "S2"}
    assert source_map["S1"]["chunk_id"] == "c1"
    assert source_map["S2"]["chunk_id"] == "c2"
    assert source_map["S1"]["corpus_role"] == "literature"
    assert source_map["S1"]["document_id"] == "doc-c1"


def test_ocr_and_table_provenance_is_preserved() -> None:
    """OCR and table facts must reach the audit so later stages can be stricter."""
    result = _package(
        evidence=[
            candidate("ocr-1", ocr_involved=True),
            candidate("table-1", content_type="table"),
        ]
    )

    assert result.ocr_source_ids == ("S1",)
    assert result.table_source_ids == ("S2",)
    assert result.blocks[1].is_table is True


# --- 2. source ids are deterministic ----------------------------------------


def test_source_ids_are_assigned_in_input_order() -> None:
    """S1..Sn follow retrieval order, contiguous, never renumbered later."""
    result = _package(evidence=[candidate(f"c{index}") for index in range(5)])

    assert result.source_ids == ("S1", "S2", "S3", "S4", "S5")
    assert [block.chunk_id for block in result.blocks] == [f"c{index}" for index in range(5)]


def test_ids_are_contiguous_after_a_budget_drop() -> None:
    """A dropped block leaves no hole: ids describe what was actually supplied."""
    small = candidate("c1", text="x" * 40)
    huge = candidate("c2", text="y" * 40000)
    result = _package(evidence=[small, huge, candidate("c3", text="z" * 40)], budget_tokens=600)

    assert result.source_ids == ("S1", "S2")
    assert [block.chunk_id for block in result.blocks] == ["c1", "c3"]


# --- 3-4. budget behaviour ---------------------------------------------------


def test_highest_ranked_evidence_is_retained_under_a_tight_budget() -> None:
    """The first-ranked blocks are kept and the tail is dropped, not the reverse."""
    unit = estimate_tokens("A" * 400) + BLOCK_OVERHEAD_TOKENS
    evidence = [
        candidate("c1", text="A" * 400, rerank_score=0.9),
        candidate("c2", text="B" * 400, rerank_score=0.8),
        candidate("c3", text="C" * 400, rerank_score=0.7),
        candidate("c4", text="D" * 400, rerank_score=0.1),
    ]
    result = _package(evidence=evidence, budget_tokens=unit * 2 + 5)

    kept = [block.chunk_id for block in result.blocks]
    assert kept == ["c1", "c2"]
    assert result.excluded_chunks == 2
    assert result.context_tokens_estimated <= unit * 2 + 5


def test_dropped_blocks_are_recorded_with_a_reason() -> None:
    """Every exclusion is explained, including a block that cannot fit at all."""
    budget = 200
    fitted = estimate_tokens("a" * 100) + BLOCK_OVERHEAD_TOKENS
    assert fitted < budget
    oversized = estimate_tokens("b" * 40000) + BLOCK_OVERHEAD_TOKENS
    assert oversized > budget
    candidate_too_large_to_add = estimate_tokens("c" * 500) + BLOCK_OVERHEAD_TOKENS
    assert candidate_too_large_to_add <= budget
    assert fitted + candidate_too_large_to_add > budget

    result = _package(
        evidence=[
            candidate("fits", text="a" * 100),
            candidate("too-big", text="b" * 40000),
            candidate("no-room", text="c" * 500),
        ],
        budget_tokens=budget,
    )

    reasons = {item.chunk_id: item.reason for item in result.dropped}
    assert reasons["too-big"] == "block_too_large"
    assert reasons["no-room"] == "budget"
    assert result.retrieved_chunks == 3
    assert result.included_chunks == 1


def test_evidence_below_the_rerank_threshold_is_dropped_and_recorded() -> None:
    """The optional threshold drops weak tail evidence before any budget work (§8)."""
    result = _package(
        evidence=[
            candidate("strong", rerank_score=0.9),
            candidate("weak", rerank_score=0.02),
        ],
        min_rerank_score=0.25,
    )

    assert [block.chunk_id for block in result.blocks] == ["strong"]
    dropped = result.dropped[0]
    assert dropped.chunk_id == "weak"
    assert dropped.reason == "below_min_rerank_score"
    assert dropped.rerank_score == 0.02


def test_no_block_is_ever_truncated_in_the_middle() -> None:
    """Whole blocks only: the truncation field stays empty by construction."""
    long_text = " ".join(f"word{index}" for index in range(4000))
    result = _package(evidence=[candidate("c1", text=long_text)], budget_tokens=200)

    assert result.blocks == ()
    assert result.truncated_blocks == ()
    assert result.dropped[0].reason == "block_too_large"


def test_token_estimate_is_conservative_and_deterministic() -> None:
    """The estimator is a pure function and never underestimates a word count."""
    text = "one two three four five six seven eight nine ten"
    first = estimate_tokens(text)
    assert first == estimate_tokens(text)
    assert first >= 10  # ten words cannot be fewer than ten tokens
    assert estimate_tokens("") == 0


# --- 5. evidence / non-evidence labels --------------------------------------


def test_mode_labels_keep_non_evidence_material_out_of_claims() -> None:
    """my_work and guidance blocks are present but explicitly not citable."""
    result = _package(
        evidence=[candidate("c1")],
        my_work=[candidate("m1", text="My own draft.", corpus_role="my_work")],
        guidance=[candidate("g1", text="Supervisor note.", corpus_role="guidance")],
        mode=GenerationMode.EVIDENCE_PLUS_GUIDANCE,
    )

    assert result.source_ids == ("S1", "S2", "S3")
    assert result.citable_source_ids == ("S1",)
    classes = {block.source_id: block.source_class for block in result.blocks}
    assert classes["S2"] is SourceClass.MY_WORK
    assert classes["S3"] is SourceClass.GUIDANCE
    rendered = result.rendered
    assert "my_work (the author's own draft; NOT citable as evidence" in rendered
    assert "guidance (supervision notes, writing rules, plans; NOT citable evidence" in rendered


def test_evidence_only_mode_ignores_supplied_non_evidence_context() -> None:
    """In the default mode, own-work and guidance candidates are not included at all."""
    result = _package(
        evidence=[candidate("c1")],
        my_work=[candidate("m1")],
        guidance=[candidate("g1")],
        mode=GenerationMode.EVIDENCE_ONLY,
    )

    assert result.source_ids == ("S1",)
    assert result.included_chunks == 1


def test_external_fulltext_verified_is_citable_but_labelled_external() -> None:
    """Retrieved external content may be used, but it is never corpus evidence (§28)."""
    result = _package(
        evidence=[candidate("c1")],
        external=[
            ExternalSource(
                title="An Open Access Paper",
                year=2026,
                fulltext_verified=True,
                fulltext_text="The paper reports that institutions adopt policies slowly.",
                fulltext_source_url="https://example.org/paper.pdf",
                providers=("openalex",),
            )
        ],
    )

    block = result.blocks[1]
    assert block.source_class is SourceClass.EXTERNAL_FULLTEXT_VERIFIED
    assert block.citable is True
    assert block.external is True
    assert block.fulltext_verified is True
    assert "external_fulltext_verified" in result.rendered
    assert "NOT part of the local corpus" in result.rendered


def test_external_metadata_only_is_not_citable() -> None:
    """Metadata-only records are supplied but must not support findings (§5)."""
    result = _package(external=[external_record(status="EXTERNAL_METADATA_VERIFIED")])

    block = result.blocks[0]
    assert block.citable is False
    assert block.source_class is SourceClass.EXTERNAL_METADATA_VERIFIED
    assert "METADATA ONLY" in result.rendered
    assert "Retrieved content: no (metadata only)" in result.rendered


def test_real_rag010_record_adapts_to_a_context_block() -> None:
    """A genuine RAG-010 ``ExternalWork`` flows through the adapter unchanged."""
    from tests.generation_support import real_external_work

    record = real_external_work(
        title="AI Governance in Universities", year=2025, doi="10.1234/real", venue="Journal"
    )
    sources = external_sources_from_records([record])
    result = _package(external=sources)

    block = result.blocks[0]
    assert block.title == "AI Governance in Universities"
    assert block.year == 2025
    assert block.doi == "10.1234/real"
    assert block.venue == "Journal"
    assert block.citable is False


# --- 22-23. determinism ------------------------------------------------------


def test_same_order_produces_the_same_ids_and_rendering() -> None:
    """Identical inputs give byte-identical context."""
    build = lambda: _package(  # noqa: E731 - a local builder keeps the test short
        evidence=[candidate("c1"), candidate("c2")],
        my_work=[candidate("m1")],
        external=[external_record()],
        mode=GenerationMode.EVIDENCE_PLUS_MY_WORK,
    )

    first, second = build(), build()
    assert first.source_ids == second.source_ids
    assert first.rendered == second.rendered
    assert first.context_tokens_estimated == second.context_tokens_estimated


def test_context_does_not_depend_on_mapping_order() -> None:
    """Re-ordered equal inputs still produce a well-defined, stable layout."""
    forward = _package(
        evidence=[candidate("a", rerank_score=0.9), candidate("b", rerank_score=0.8)]
    )
    # The same two candidates supplied in the other order must swap ids, nothing else.
    backward = _package(
        evidence=[candidate("b", rerank_score=0.8), candidate("a", rerank_score=0.9)]
    )

    assert [block.chunk_id for block in forward.blocks] == ["a", "b"]
    assert [block.chunk_id for block in backward.blocks] == ["b", "a"]
    assert [block.token_estimate for block in forward.blocks] == [
        block.token_estimate for block in backward.blocks
    ]


def test_block_overhead_is_counted_so_a_just_fitting_block_is_not_advertised_as_free() -> None:
    """The header costs tokens too; the estimate includes it."""
    result = _package(evidence=[candidate("c1", text="short")])

    assert result.blocks[0].token_estimate >= BLOCK_OVERHEAD_TOKENS


def test_requesting_own_work_reserves_budget_for_it() -> None:
    """An explicit my_work request is not starved by a large evidence set."""
    evidence = [candidate(f"c{index}", text="word " * 900) for index in range(4)]
    my_work = [candidate("m1", text="My own draft wording.", corpus_role="my_work")]
    unit = estimate_tokens("word " * 900) + BLOCK_OVERHEAD_TOKENS

    result = _package(
        evidence=evidence,
        my_work=my_work,
        mode=GenerationMode.EVIDENCE_PLUS_MY_WORK,
        budget_tokens=unit * 3,
        reserved_non_evidence_tokens=unit,
    )

    assert result.reserved_non_evidence_tokens == unit
    assert result.evidence_budget_tokens == unit * 2
    classes = [block.source_class for block in result.blocks]
    assert SourceClass.MY_WORK in classes
    assert result.context_tokens_estimated <= unit * 3
    # Evidence still gets the larger share.
    assert classes.count(SourceClass.EVIDENCE) == 2


def test_no_reservation_when_no_non_evidence_context_was_requested() -> None:
    """In evidence-only mode the whole budget belongs to evidence."""
    result = _package(
        evidence=[candidate("c1")],
        my_work=[candidate("m1")],
        mode=GenerationMode.EVIDENCE_ONLY,
        reserved_non_evidence_tokens=1000,
    )

    assert result.reserved_non_evidence_tokens == 0
    assert result.evidence_budget_tokens == result.budget_tokens


def test_rendered_context_delimits_every_source_text() -> None:
    """Each block wraps its text in explicit delimiters (§23 / §31 item 17)."""
    result = _package(evidence=[candidate("c1", text="Body one."), candidate("c2", text="Body two.")])

    assert result.rendered.count(TEXT_OPEN) == 2
    assert result.rendered.count(TEXT_CLOSE) == 2
    assert "Body one." in result.rendered
