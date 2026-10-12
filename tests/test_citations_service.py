"""End-to-end citation service tests (RAG-012 §13-§15, §19)."""

from __future__ import annotations

import re

import pytest

from app.citations.models import ClaimDecision, SupportVerdict
from app.citations.service import POLICY_NOTE, PARTIAL_ANNOTATION, WITHHELD_NOTE
from app.generation.models import GenerationMode, SourceClass, SourceMarkerAudit
from tests.citations_support import (
    FACULTY_CLAIM,
    FACULTY_TEXT,
    SENIOR_CLAIM,
    SENIOR_TEXT,
    block,
    make_response,
    validation_of,
)


def answer_with(*sentences: str) -> str:
    return " ".join(sentences)


# --- marker conversion ---------------------------------------------------------


def test_supported_claims_get_harvard_citations():
    response = make_response(
        answer_with(SENIOR_CLAIM + "."),
        [block("S1", doi="10.1234/abc")],
    )
    result = validation_of(response)
    assert "(Doe, 2024)" in result.validated_answer
    assert "[S1]" not in result.validated_answer
    assert result.validation_summary.supported == 1
    assert result.validation_summary.markers_replaced == 1


def test_original_answer_is_preserved_untouched():
    answer = answer_with(SENIOR_CLAIM + ".")
    response = make_response(answer, [block("S1")])
    result = validation_of(response)
    assert result.original_answer == answer
    assert result.validated_answer != answer
    assert result.question == response.question


def test_paragraph_layout_is_preserved():
    first = SENIOR_CLAIM + "."
    second = SENIOR_CLAIM + "."
    response = make_response(f"{first}\n\n{second}", [block("S1")])
    result = validation_of(response)
    assert "\n\n" in result.validated_answer
    assert result.validated_answer.count("(Doe, 2024)") == 2


def test_marker_groups_record_every_substitution():
    blocks = [
        block(
            "S1",
            text="Brown (2022) reports that governance frameworks protect student privacy.",
            doi="10.1234/a",
        ),
        block(
            "S2",
            text="Smith (2024) confirms that governance frameworks protect student privacy.",
            first_author="John Smith",
            doi="10.1234/b",
        ),
    ]
    response = make_response(
        "Governance frameworks protect student privacy [S1,S2].", blocks
    )
    result = validation_of(response)
    assert result.marker_groups["[S1,S2]"] == "(Doe, 2024; Smith, 2024)"
    assert "[(Doe, 2024; Smith, 2024)]" not in result.validated_answer


def test_multi_source_citation_is_alphabetically_ordered():
    blocks = [
        block(
            "S1",
            text="Adams (2020) found that governance frameworks protect student privacy.",
            first_author="Zoe Adams",
            year=2020,
            doi="10.1234/a",
        ),
        block(
            "S2",
            text="Chan (2024) found that governance frameworks prevent academic misconduct.",
            first_author="Bob Chan",
            year=2024,
            doi="10.1234/b",
        ),
    ]
    response = make_response(
        "Governance frameworks protect student privacy and prevent misconduct [S1,S2].",
        blocks,
    )
    result = validation_of(response)
    assert "(Adams, 2020; Chan, 2024)" in result.validated_answer


# --- the publication policy ----------------------------------------------------


def test_unsupported_claim_is_withheld():
    response = make_response(
        answer_with(
            "Quantum tunnelling governs Antarctic krill populations [S1]."
        ),
        [block("S1")],
    )
    result = validation_of(response)
    assert result.validated_answer.strip() == WITHHELD_NOTE
    assert result.validation_summary.unsupported == 1
    assert result.validation_summary.withheld_claims == 1
    assert result.claim_validations[0].decision is ClaimDecision.WITHHELD
    assert result.unsupported_sentences == (
        "Quantum tunnelling governs Antarctic krill populations [S1].",
    )
    assert any(
        warning.startswith("unsupported_claim:") for warning in result.warnings
    )


def test_only_the_unsupported_sentence_is_withheld():
    response = make_response(
        answer_with(
            SENIOR_CLAIM + ".",
            "Quantum tunnelling governs Antarctic krill populations [S1].",
            SENIOR_CLAIM + ".",
        ),
        [block("S1")],
    )
    result = validation_of(response)
    assert result.validated_answer.count(WITHHELD_NOTE) == 1
    assert result.validation_summary.supported == 2
    assert result.validation_summary.unsupported == 1


def test_partially_supported_claim_is_annotated_and_not_presented_as_validated():
    response = make_response(
        answer_with(FACULTY_CLAIM + "."),
        [block("S3", text=FACULTY_TEXT, first_author="Anna Chan", year=2024, pages="17")],
    )
    result = validation_of(response)
    claim = result.claim_validations[0]
    assert claim.verdict is SupportVerdict.PARTIALLY_SUPPORTED
    assert PARTIAL_ANNOTATION in result.validated_answer
    assert "(Chan, 2024). [partially supported]" in result.validated_answer
    assert result.validation_summary.partially_supported == 1
    assert result.validation_summary.annotated_claims == 1
    assert result.has_problems is True
    assert any(
        warning.startswith("partially_supported_claim:") for warning in result.warnings
    )


def test_uncertain_claim_is_annotated_unverified():
    source = block(
        "S1",
        text="Virtual reality increases learner engagement in laboratory courses.",
    )
    response = make_response(
        "Immersive headset use reduces examination anxiety among psychology "
        "undergraduates while improving procedural retention [S1].",
        [source],
    )
    result = validation_of(response)
    claim = result.claim_validations[0]
    if claim.verdict is SupportVerdict.UNCERTAIN:
        assert "[unverified]" in result.validated_answer
        assert claim.decision is ClaimDecision.ANNOTATED


def test_policy_note_is_recorded():
    result = validation_of(make_response(SENIOR_CLAIM + ".", [block("S1")]))
    assert result.policy_note == POLICY_NOTE
    assert "withheld" in result.policy_note


def test_invalid_marker_is_recorded_and_its_sentence_withheld():
    response = make_response(
        "Governance frameworks protect student privacy [S9].",
        [block("S1")],
        invalid_markers=("S9",),
    )
    result = validation_of(response)
    assert result.validation_summary.invalid_markers == ("S9",)
    assert result.validated_answer.strip() == WITHHELD_NOTE
    assert any(
        warning.startswith("invalid_source_marker") for warning in result.warnings
    )


def test_non_evidence_source_never_becomes_a_scholarly_citation():
    my_work = block(
        "S6",
        text=SENIOR_TEXT,
        source_class=SourceClass.MY_WORK,
        citable=False,
        corpus_role="my_work",
        evidence_eligible=False,
        first_author=None,
    )
    response = make_response(
        "Governance is the responsibility of senior management [S6].",
        [block("S1"), my_work],
        mode=GenerationMode.EVIDENCE_PLUS_MY_WORK,
        non_evidence_reports=(),
    )
    result = validation_of(response)
    assert result.validation_summary.non_evidence_sources_cited == ("S6",)
    assert "Doe" not in result.validated_answer.replace("(Doe, 2024)", "")
    assert result.reference_list == () or all(
        "S6" not in entry.source_ids for entry in result.references
    )
    assert result.validated_answer.strip() == WITHHELD_NOTE


def test_reported_own_work_keeps_its_wording_and_gets_a_provenance_note():
    my_work = block(
        "S6",
        text=SENIOR_TEXT,
        source_class=SourceClass.MY_WORK,
        citable=False,
        corpus_role="my_work",
        evidence_eligible=False,
    )
    sentence = "Your draft states that senior management is responsible [S6]."
    response = make_response(
        sentence,
        [my_work],
        mode=GenerationMode.EVIDENCE_PLUS_MY_WORK,
        non_evidence_reports=(sentence,),
    )
    result = validation_of(response)
    claim = result.claim_validations[0]
    assert claim.kind.value == "non_evidence_report"
    assert claim.decision is ClaimDecision.NON_CITABLE_REPORT
    assert "[author's own work]" in result.validated_answer
    assert "[S6]" not in result.validated_answer
    assert result.reference_list == ()


def test_heading_with_a_marker_loses_the_marker():
    response = make_response(
        "## Findings [S1]\n\n" + SENIOR_CLAIM + ".",
        [block("S1")],
    )
    result = validation_of(response)
    assert "## Findings" in result.validated_answer
    assert "[S1]" not in result.validated_answer
    assert result.validation_summary.markers_removed >= 1


def test_no_internal_marker_survives_anywhere_in_the_validated_answer():
    my_work = block(
        "S6",
        text=SENIOR_TEXT,
        source_class=SourceClass.MY_WORK,
        citable=False,
        corpus_role="my_work",
        evidence_eligible=False,
    )
    own_report = "Your draft states that senior management is responsible [S6]."
    response = make_response(
        "\n\n".join(
            [
                "## Findings [S1]",
                SENIOR_CLAIM + ".",
                "Quantum tunnelling governs Antarctic krill populations [S1].",
                "Governance frameworks protect student privacy [S9].",
                own_report,
                "In summary, the evidence points one way [S1].",
            ]
        ),
        [block("S1"), my_work],
        mode=GenerationMode.EVIDENCE_PLUS_MY_WORK,
        non_evidence_reports=(own_report,),
        invalid_markers=("S9",),
    )
    result = validation_of(response)
    assert re.search(r"\[\s*S\d+", result.validated_answer) is None
    assert "[author's own work]" in result.validated_answer


# --- references ----------------------------------------------------------------


def test_chunks_of_one_paper_produce_one_reference():
    blocks = [
        block("S1", doi="10.1234/shared", pages="4-6", chunk_id="chunk-a"),
        block("S2", doi="10.1234/shared", pages="7-9", chunk_id="chunk-b"),
    ]
    response = make_response(
        answer_with(SENIOR_CLAIM + ".", SENIOR_CLAIM.replace("[S1]", "[S2]") + "."),
        blocks,
    )
    result = validation_of(response)
    assert len(result.references) == 1
    assert len(result.reference_list) == 1
    assert result.references[0].source_ids == ("S1", "S2")
    assert result.validation_summary.references_total == 1
    assert result.validation_summary.sources_cited == 2


def test_reference_list_uses_the_available_metadata_only():
    response = make_response(
        SENIOR_CLAIM + ".",
        [block("S1", doi="10.1234/abc", pages="4-6", venue=None, year=2024)],
    )
    result = validation_of(response)
    assert result.reference_list == (
        "Doe, J. (2024) 'A Paper About AI Governance', pp. 4-6. doi:10.1234/abc",
    )


def test_reference_metadata_warnings_are_reported():
    response = make_response(
        SENIOR_CLAIM + ".",
        [block("S1", doi=None, year=None, title=None)],
    )
    result = validation_of(response)
    assert any(
        warning.startswith("reference_metadata_warning:") for warning in result.warnings
    )
    assert result.validation_summary.references_with_metadata_warning == 1


def test_unsupported_claim_contributes_no_reference():
    response = make_response(
        "Quantum tunnelling governs Antarctic krill populations [S1].",
        [block("S1", doi="10.1234/abc")],
    )
    result = validation_of(response)
    assert result.reference_list == ()
    assert result.citation_map == ()


def test_citation_map_covers_every_cited_source():
    blocks = [
        block("S1", doi="10.1234/a"),
        block("S2", doi="10.1234/b", first_author="John Smith"),
    ]
    response = make_response(
        answer_with(SENIOR_CLAIM + ".", SENIOR_CLAIM.replace("[S1]", "[S2]") + "."),
        blocks,
    )
    result = validation_of(response)
    mapped = {entry.source_id: entry for entry in result.citation_map}
    assert set(mapped) == {"S1", "S2"}
    assert mapped["S1"].reference_key == "doi:10.1234/a"
    assert mapped["S1"].chunk_id == "chunk-s1"
    assert mapped["S2"].in_text == "(Smith, 2024)"


# --- numeric, quotation and risk reporting --------------------------------------


def test_missing_number_is_reported_and_withheld():
    source = block(
        "S1",
        text="A mixed-methods approach with 150 participants found that 75% cited "
        "academic integrity as a primary concern.",
    )
    response = make_response(
        "A study of 150 participants found that 88% cited academic integrity as a "
        "primary concern [S1].",
        [source],
    )
    result = validation_of(response)
    assert result.validation_summary.numeric_mentions_missing == 1
    assert any(
        warning.startswith("numeric_claim_not_in_evidence:") for warning in result.warnings
    )
    assert result.validated_answer.strip() == WITHHELD_NOTE


def test_verified_quotation_is_recorded_in_the_summary():
    response = make_response(
        'The source states "Senior management holds significant responsibility" [S1].',
        [block("S1")],
    )
    result = validation_of(response)
    assert result.validation_summary.quotations_checked == 1
    assert result.validation_summary.quotations_failed == 0
    assert "(Doe, 2024)" in result.validated_answer


def test_unverified_quotation_fails_the_claim():
    response = make_response(
        'The source states "senior managers are liable for every decision" [S1].',
        [block("S1")],
    )
    result = validation_of(response)
    assert result.validation_summary.quotations_failed == 1
    assert result.validation_summary.unsupported == 1
    assert any(
        warning.startswith("quotation_not_in_cited_source:") for warning in result.warnings
    )


def test_ocr_source_is_flagged_as_elevated_risk():
    response = make_response(
        SENIOR_CLAIM + ".", [block("S1", ocr_involved=True)]
    )
    result = validation_of(response)
    assert result.validation_summary.elevated_risk_claims == 1
    assert any(warning.startswith("elevated_validation_risk") for warning in result.warnings)
    assert result.claim_validations[0].elevated_validation_risk is True


def test_table_source_is_flagged_as_elevated_risk():
    response = make_response(SENIOR_CLAIM + ".", [block("S1", content_type="table")])
    result = validation_of(response)
    assert result.claim_validations[0].elevated_validation_risk is True


# --- audits ---------------------------------------------------------------------


def test_generation_warnings_and_audit_are_preserved_separately():
    response = make_response(
        SENIOR_CLAIM + ".",
        [block("S1")],
        warnings=("unmarked_statement: something",),
    )
    response = response.model_copy(
        update={
            "marker_audit": SourceMarkerAudit(
                invalid_markers=(), unmarked_statements=("something",)
            )
        }
    )
    result = validation_of(response)
    assert result.generation_warnings == ("unmarked_statement: something",)
    assert not any(
        warning.startswith("unmarked_statement") for warning in result.warnings
    )
    assert result.generation_status == response.status.value


def test_source_map_is_preserved_for_traceability():
    response = make_response(SENIOR_CLAIM + ".", [block("S1", chunk_id="chunk-xyz")])
    result = validation_of(response)
    assert result.source_map["S1"]["chunk_id"] == "chunk-xyz"
    assert result.validation_seconds >= 0.0


def test_validator_name_reports_the_active_layers():
    result = validation_of(make_response(SENIOR_CLAIM + ".", [block("S1")]))
    assert result.validator == "deterministic"


def test_summary_counts_add_up():
    response = make_response(
        answer_with(
            SENIOR_CLAIM + ".",
            FACULTY_CLAIM + ".",
            "Quantum tunnelling governs Antarctic krill populations [S1].",
            "## Heading [S1]",
        ),
        [block("S1", text=SENIOR_TEXT), block("S3", text=FACULTY_TEXT, first_author="Anna Chan")],
    )
    result = validation_of(response)
    summary = result.validation_summary
    assert summary.claims_total == 3
    assert summary.supported + summary.partially_supported + summary.unsupported == 3
    assert summary.unmarked_claims == 0
    assert summary.sentences_total == 4


# --- the recorded RAG-011 failure mode ------------------------------------------


def test_known_rag011_partial_claim_is_not_marked_fully_supported():
    """The single PARTIALLY_SUPPORTED claim of the RAG-011 evaluation (§15, §18)."""
    source = block(
        "S3",
        text=FACULTY_TEXT,
        first_author="Md Alamin",
        year=2024,
        pages="17",
        doi="10.1234/faculty",
    )
    integrity = block(
        "S1",
        text=(
            "Generative AI models can be used to generate essays, assignments, or answers, "
            "enabling academic dishonesty and plagiarism."
        ),
        first_author="Junaid Khan",
        year=2024,
        doi="10.1234/integrity",
    )
    supported = (
        "Generative AI models can generate essays that enable plagiarism and academic "
        "dishonesty [S1]."
    )
    response = make_response(
        answer_with(supported, FACULTY_CLAIM + "."),
        [integrity, source],
    )
    result = validation_of(response)
    verdicts = {claim.sentence: claim.verdict for claim in result.claim_validations}
    assert verdicts[supported] is SupportVerdict.SUPPORTED
    assert verdicts[FACULTY_CLAIM + "."] is not SupportVerdict.SUPPORTED
    assert not result.validation_summary.all_supported
    assert result.has_problems is True
    job = next(
        claim for claim in result.claim_validations if claim.sentence == FACULTY_CLAIM + "."
    )
    assert job.validator_method.value == "lexical_merged_findings"
    assert job.verdict is SupportVerdict.PARTIALLY_SUPPORTED


def test_claim_validation_carries_method_confidence_reason_and_sources():
    response = make_response(SENIOR_CLAIM + ".", [block("S1")])
    result = validation_of(response)
    claim = result.claim_validations[0]
    assert claim.validator_method.value == "lexical_single_sentence"
    assert 0.0 < claim.confidence <= 1.0
    assert claim.reason
    assert claim.source_ids == ("S1",)
    assert claim.per_source[0].source_id == "S1"


@pytest.mark.parametrize("verdict", list(SupportVerdict))
def test_every_verdict_has_a_defined_decision(verdict):
    from app.citations.models import ClaimDecision as Decision

    assert verdict.is_failure is (verdict is SupportVerdict.UNSUPPORTED)
    assert verdict.needs_annotation is (
        verdict in (SupportVerdict.PARTIALLY_SUPPORTED, SupportVerdict.UNCERTAIN)
    )
    assert Decision.WITHHELD.value == "withheld"
