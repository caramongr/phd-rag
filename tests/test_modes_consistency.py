"""CONSISTENCY comparison tests (RAG-013 §4, §13)."""

from __future__ import annotations

from app.modes.consistency import (
    BUILTIN_PREFERENCE_SOURCE,
    ComparisonPolicy,
    compare,
    duplication_findings,
    finding_counts,
    numeric_discrepancy_findings,
    render_report,
    style_findings,
    terminology_findings,
    unsupported_claim_findings,
)
from app.modes.models import ConsistencyFindingKind
from tests.generation_support import candidate

LITERATURE = (
    "Generative AI is reshaping higher education. Institutions increasingly rely on "
    "generative systems for assessment support, which raises questions about academic "
    "integrity and the governance of teaching practice."
)
DRAFT = (
    "In this chapter I argue that AI governance is an institutional responsibility, and I "
    "call that stance algorithmic stewardship of teaching practice."
)


def chunk(text: str, *, chunk_id: str = "chunk-1", title: str = "A work", **kwargs):
    """A RAG-009 candidate usable as a comparison input."""
    kwargs.setdefault("corpus_role", "my_work")
    kwargs.setdefault("evidence_eligible", False)
    return candidate(chunk_id=chunk_id, text=text, title=title, **kwargs)


# --- terminology ---------------------------------------------------------------


def test_terminology_reports_both_directions():
    findings = terminology_findings(
        [chunk(DRAFT)],
        [chunk(LITERATURE, corpus_role="literature", evidence_eligible=True)],
        # The top-N cap is lifted so the assertion does not depend on how ties are broken.
        ComparisonPolicy(terminology_min_count=1, terminology_top=20),
    )
    assert len(findings) == 2
    assert all(finding.kind is ConsistencyFindingKind.TERMINOLOGY for finding in findings)
    yours, theirs = findings
    assert "stewardship" in yours.summary or "algorithmic" in yours.summary
    assert "academic" in theirs.summary or "assessment" in theirs.summary or "integrity" in theirs.summary


def test_terminology_shows_surface_words_not_stems():
    """The comparison canonicalises, but the report must show the word the author wrote."""
    literature = chunk(
        "Generative systems reshape assessment tasks in universities worldwide.",
        corpus_role="literature",
        evidence_eligible=True,
    )
    draft = chunk(
        "Governance shapes governance: the governance of teaching practice is contested.",
        chunk_id="chunk-2",
    )
    findings = terminology_findings(
        [draft], [literature], ComparisonPolicy(terminology_min_count=1)
    )
    yours = findings[0]
    assert "governance" in yours.summary
    assert "governanc" not in yours.summary.replace("governance", "")


def test_terminology_ignores_terms_mentioned_only_once():
    """The default threshold keeps a stray single mention from becoming a finding."""
    literature = chunk(LITERATURE, corpus_role="literature", evidence_eligible=True)
    draft = chunk(
        "AI governance in higher education is about assurance and integrity.",
        chunk_id="chunk-2",
    )
    assert terminology_findings([draft], [literature], ComparisonPolicy()) == ()


def test_terminology_is_quiet_when_the_vocabularies_agree():
    literature = chunk(LITERATURE, corpus_role="literature", evidence_eligible=True)
    draft = chunk(
        "Generative AI is reshaping higher education. Institutions rely on generative "
        "systems for assessment support, and that raises governance questions.",
        chunk_id="chunk-2",
    )
    findings = terminology_findings(
        [draft], [literature], ComparisonPolicy(terminology_min_count=1)
    )
    assert [finding for finding in findings if "stewardship" in finding.summary] == []


def test_terminology_needs_literature_to_compare_against():
    assert terminology_findings([chunk(DRAFT)], [], ComparisonPolicy()) == ()


def test_terminology_records_where_it_looked():
    literature = chunk(LITERATURE, chunk_id="chunk-lit", corpus_role="literature")
    findings = terminology_findings(
        [chunk(DRAFT)], [literature], ComparisonPolicy(terminology_min_count=1)
    )
    assert findings
    assert all("chunk-lit" in finding.evidence for finding in findings)


# --- duplication ---------------------------------------------------------------


def test_duplication_flags_two_drafts_that_repeat_each_other():
    first = chunk(DRAFT, chunk_id="chunk-a")
    second = chunk(DRAFT.replace("I argue that", "I maintain that"), chunk_id="chunk-b")
    findings = duplication_findings([first, second], [], ComparisonPolicy())
    assert findings
    finding = findings[0]
    assert finding.kind is ConsistencyFindingKind.DUPLICATION
    assert finding.your_chunk_id == "chunk-a"
    assert finding.other_chunk_id == "chunk-b"
    assert finding.method == "content_term_containment"


def test_duplication_flags_a_draft_passage_that_occurs_in_the_literature():
    literature = chunk(LITERATURE, chunk_id="chunk-lit", corpus_role="literature")
    copied = chunk(LITERATURE, chunk_id="chunk-copy")
    findings = duplication_findings([copied], [literature], ComparisonPolicy())
    assert findings
    assert findings[0].other_chunk_id == "chunk-lit"
    assert "chunk-lit" in findings[0].evidence


def test_duplication_is_quiet_for_distinct_drafts():
    first = chunk(DRAFT, chunk_id="chunk-a")
    second = chunk(
        "The methodology chapter describes a design science approach with two evaluation "
        "cycles and a documented artefact.",
        chunk_id="chunk-b",
    )
    assert duplication_findings([first, second], [], ComparisonPolicy()) == ()


def test_duplication_threshold_is_configurable():
    first = chunk(DRAFT, chunk_id="chunk-a")
    second = chunk(DRAFT.replace("In this chapter", "In the introduction"), chunk_id="chunk-b")
    strict = duplication_findings(
        [first, second], [], ComparisonPolicy(duplication_overlap=0.99)
    )
    assert strict == ()


# --- unsupported claims --------------------------------------------------------


def test_unsupported_claim_when_no_literature_passage_covers_the_sentence():
    draft = chunk(
        "The survey reported that 92% of universities had adopted a mandatory AI literacy "
        "module by 2024.",
        chunk_id="chunk-draft",
    )
    literature = chunk(LITERATURE, chunk_id="chunk-lit", corpus_role="literature")
    findings = unsupported_claim_findings([draft], [literature], ComparisonPolicy())
    assert findings
    assert findings[0].kind is ConsistencyFindingKind.UNSUPPORTED_CLAIM
    assert "no retrieved literature passage" in findings[0].summary
    assert findings[0].your_chunk_id == "chunk-draft"


def test_supported_draft_sentence_is_not_flagged():
    draft = chunk(
        "Institutions increasingly rely on generative systems for assessment support.",
        chunk_id="chunk-draft",
    )
    literature = chunk(LITERATURE, chunk_id="chunk-lit", corpus_role="literature")
    assert unsupported_claim_findings([draft], [literature], ComparisonPolicy()) == ()


def test_unsupported_claim_check_needs_literature():
    draft = chunk("Something entirely unsupported by anything at all.", chunk_id="chunk-d")
    assert unsupported_claim_findings([draft], [], ComparisonPolicy()) == ()


def test_unsupported_claim_wording_does_not_judge_the_claim():
    draft = chunk(
        "The survey reported that 92% of universities had adopted a mandatory AI literacy "
        "module by 2024.",
        chunk_id="chunk-draft",
    )
    literature = chunk(LITERATURE, chunk_id="chunk-lit", corpus_role="literature")
    finding = unsupported_claim_findings([draft], [literature], ComparisonPolicy())[0]
    assert "not that the claim is wrong" in finding.detail


# --- numeric discrepancy -------------------------------------------------------


def test_numeric_discrepancy_when_the_literature_states_other_figures():
    draft = chunk(
        "A survey of 150 universities found that 92% had adopted an AI governance policy.",
        chunk_id="chunk-draft",
    )
    literature = chunk(
        "A survey of 150 universities found that 40% had adopted an AI governance policy.",
        chunk_id="chunk-lit",
        corpus_role="literature",
    )
    findings = numeric_discrepancy_findings([draft], [literature], ComparisonPolicy())
    assert findings
    assert findings[0].kind is ConsistencyFindingKind.NUMERIC_DISCREPANCY
    assert "92%" in findings[0].summary
    assert "40" in findings[0].detail


def test_no_discrepancy_when_the_figure_matches():
    sentence = (
        "A survey of 150 universities found that 40% had adopted an AI governance policy."
    )
    draft = chunk(sentence, chunk_id="chunk-draft")
    literature = chunk(sentence, chunk_id="chunk-lit", corpus_role="literature")
    assert numeric_discrepancy_findings([draft], [literature], ComparisonPolicy()) == ()


def test_discrepancy_check_needs_a_matching_passage():
    draft = chunk("The survey reported that 92% of universities had adopted a policy.")
    literature = chunk(LITERATURE, chunk_id="chunk-lit", corpus_role="literature")
    assert numeric_discrepancy_findings([draft], [literature], ComparisonPolicy()) == ()


# --- style ---------------------------------------------------------------------


def test_style_flags_the_em_dash():
    draft = chunk("The platform transforms learning \u2014 and teaching too.")
    findings = style_findings([draft], [], ComparisonPolicy())
    assert any("em dash" in finding.summary for finding in findings)
    assert findings[0].evidence == (BUILTIN_PREFERENCE_SOURCE,)


def test_style_shows_the_dash_in_context_not_the_chunk_head():
    """A character-level finding must quote the character it fires on."""
    draft = chunk(
        "The chapter reviews the tools in turn without any stylistic surprise at the start "
        "\u2014 and then the dash appears."
    )
    em_dash = next(
        finding for finding in style_findings([draft], [], ComparisonPolicy())
        if "em dash" in finding.summary
    )
    assert "\u2014" in em_dash.your_text


def test_style_reports_the_length_of_the_longest_sentence():
    draft = chunk(
        "The chapter argues that " + " ".join(f"point{index}" for index in range(45)) + "."
    )
    longest = next(
        finding for finding in style_findings([draft], [], ComparisonPolicy())
        if "longer than" in finding.summary
    )
    assert "longest 49 words" in longest.summary


def test_style_findings_name_the_draft_chunk_they_came_from():
    draft = chunk(
        "The result is revolutionary.", chunk_id="chunk-draft"
    )
    findings = style_findings([draft], [], ComparisonPolicy())
    assert findings
    assert all(finding.your_chunk_id == "chunk-draft" for finding in findings)


def test_style_flags_promotional_language():
    draft = chunk("The platform offers a revolutionary and seamless experience for students.")
    findings = style_findings([draft], [], ComparisonPolicy())
    promotional = [finding for finding in findings if "promotional" in finding.summary]
    assert promotional
    assert "revolutionary" in promotional[0].summary
    assert "seamless" in promotional[0].summary


def test_style_flags_long_sentences():
    draft = chunk(
        "The chapter argues that " + " ".join(f"point{index}" for index in range(45)) + "."
    )
    findings = style_findings([draft], [], ComparisonPolicy())
    assert any("longer than" in finding.summary for finding in findings)


def test_style_is_quiet_for_plain_prose():
    draft = chunk("The chapter describes the research design in three short sections.")
    assert style_findings([draft], [], ComparisonPolicy()) == ()


def test_style_uses_the_retrieved_rules_chunk_when_there_is_one():
    draft = chunk("The result is revolutionary.")
    rules = chunk("Writing rules: avoid promotional language.", chunk_id="chunk-rules")
    findings = style_findings([draft], [rules], ComparisonPolicy())
    assert all("chunk-rules" in finding.evidence for finding in findings)


# --- the whole comparison ------------------------------------------------------


def test_compare_produces_findings_in_a_fixed_order():
    draft = chunk("The platform is revolutionary \u2014 a ninety-two percent improvement survey reported.")
    literature = chunk(LITERATURE, chunk_id="chunk-lit", corpus_role="literature")
    findings = compare([draft], literature=[literature], rules=[])
    order = [finding.kind.value for finding in findings]
    assert order == sorted(
        order,
        key=lambda kind: [
            "terminology",
            "duplication",
            "unsupported_claim",
            "numeric_discrepancy",
            "style",
        ].index(kind),
    )


def test_finding_counts_summarises_by_kind():
    draft = chunk("The platform is revolutionary \u2014 a ninety-two percent improvement.")
    counts = finding_counts(compare([draft], literature=[], rules=[]))
    assert counts.get("style", 0) >= 2
    assert set(counts) <= {kind.value for kind in ConsistencyFindingKind}


def test_render_report_lists_findings_and_inputs():
    draft = chunk("The platform is revolutionary.")
    report = render_report("platform", compare([draft]), literature_count=0, rules_count=0, your_count=1)
    assert "CONSISTENCY COMPARISON" in report
    assert "STYLE" in report
    assert "no model was used" in report


def test_render_report_says_so_when_nothing_was_found():
    report = render_report("empty", (), literature_count=2, rules_count=1, your_count=1)
    assert "No differences were found" in report


def test_the_findings_cap_admits_that_it_truncated():
    """A truncated list must not read as a complete list."""
    draft = chunk(
        " ".join(
            f"Widget{index} stewardship carries substantial institutional responsibilities "
            "regarding governance oversight."
            for index in range(20)
        )
    )
    literature = chunk(
        "Generative systems reshape assessment tasks in universities worldwide.",
        chunk_id="chunk-lit",
        corpus_role="literature",
        evidence_eligible=True,
    )
    findings = unsupported_claim_findings(
        [draft], [literature], ComparisonPolicy(max_findings_per_kind=3)
    )
    assert len(findings) == 3
    assert "showing 3 of 20 findings of this kind" in findings[-1].detail


def test_the_findings_cap_says_nothing_when_it_did_not_bite():
    draft = chunk("The platform is revolutionary.", chunk_id="chunk-1")
    findings = style_findings([draft], [], ComparisonPolicy(max_findings_per_kind=10))
    assert findings
    assert all("showing" not in finding.detail for finding in findings)


def test_comparison_is_deterministic():
    draft = chunk(DRAFT, chunk_id="chunk-a")
    literature = chunk(LITERATURE, chunk_id="chunk-lit", corpus_role="literature")
    first = compare([draft], literature=[literature], rules=[])
    second = compare([draft], literature=[literature], rules=[])
    assert [finding.model_dump() for finding in first] == [
        finding.model_dump() for finding in second
    ]
