"""Claim support validation tests (RAG-012 §2-§8, §15, §17)."""

from __future__ import annotations

from app.citations.claims import classify_sentence
from app.citations.models import (
    ClaimKind,
    NumericKind,
    SupportVerdict,
    ValidationPolicy,
    ValidatorMethod,
)
from app.citations.support import lexical_result, validate_claim
from app.generation.models import SourceClass
from tests.citations_support import (
    FACULTY_CLAIM,
    FACULTY_TEXT,
    SENIOR_CLAIM,
    SENIOR_TEXT,
    block,
    make_context,
)


def validate(sentence: str, *blocks, policy: ValidationPolicy | None = None):
    """Validate one sentence against the supplied blocks, classified as the service does."""
    context = make_context(blocks)
    return validate_claim(
        sentence,
        context=context,
        policy=policy,
        kind=classify_sentence(
            sentence,
            supplied_ids=context.source_ids,
            citable_ids=context.citable_source_ids,
        ),
    )


# --- one source ---------------------------------------------------------------


def test_valid_one_source_claim_is_supported():
    result = validate(SENIOR_CLAIM, block("S1"))
    assert result.verdict is SupportVerdict.SUPPORTED
    assert result.validator_method is ValidatorMethod.LEXICAL_SINGLE_SENTENCE
    assert result.confidence > 0.5
    assert result.cited_ids == ("S1",)
    assert result.per_source[0].verdict is SupportVerdict.SUPPORTED


def test_supported_claim_keeps_the_source_metadata_checks():
    result = validate(SENIOR_CLAIM, block("S1", doi="10.1234/abc"))
    assert result.per_source[0].metadata_warnings == ("page provenance: 4-6",)


def test_claim_with_no_overlap_with_the_source_fails():
    result = validate(
        "Quantum tunnelling governs the thermal stability of Antarctic krill populations [S1]",
        block("S1"),
    )
    assert result.verdict is SupportVerdict.UNSUPPORTED
    assert result.validator_method is ValidatorMethod.LEXICAL_NO_OVERLAP
    assert result.decision.value == "withheld"


def test_short_claim_iscertain_when_the_source_shares_its_content():
    result = validate("Ethical guidelines matter [S1]", block("S1"))
    assert result.verdict is SupportVerdict.SUPPORTED


def test_unknown_marker_fails_even_with_valid_text():
    result = validate(SENIOR_CLAIM.replace("[S1]", "[S7]"), block("S1"))
    assert result.verdict is SupportVerdict.UNSUPPORTED
    assert result.validator_method is ValidatorMethod.INVALID_MARKER
    assert result.invalid_source_ids == ("S7",)


def test_invalid_marker_beside_a_valid_one_is_recorded_but_not_fatal():
    result = validate(SENIOR_CLAIM.replace("[S1]", "[S1,S9]"), block("S1"))
    assert result.verdict is SupportVerdict.SUPPORTED
    assert result.invalid_source_ids == ("S9",)
    assert any("invalid_source_marker" in warning for warning in result.warnings)


def test_non_evidence_source_is_unsupported():
    my_work = block(
        "S6",
        text=SENIOR_TEXT,
        source_class=SourceClass.MY_WORK,
        citable=False,
        corpus_role="my_work",
        evidence_eligible=False,
    )
    result = validate(SENIOR_CLAIM.replace("[S1]", "[S6]"), my_work)
    assert result.verdict is SupportVerdict.UNSUPPORTED
    assert result.validator_method is ValidatorMethod.NON_EVIDENCE_SOURCE
    assert result.non_evidence_source_ids == ("S6",)
    assert result.cited_ids == ()


def test_metadata_only_external_record_cannot_support_a_claim():
    external = block(
        "S9",
        text="No abstract or other content was retrieved for this record.",
        source_class=SourceClass.EXTERNAL_DISCOVERY,
        citable=False,
        corpus_role=None,
        evidence_eligible=False,
    )
    result = validate(SENIOR_CLAIM.replace("[S1]", "[S9]"), external)
    assert result.verdict is SupportVerdict.UNSUPPORTED
    assert result.validator_method is ValidatorMethod.NON_EVIDENCE_SOURCE


def test_source_without_text_cannot_support_a_claim():
    result = validate(SENIOR_CLAIM, block("S1", text="   "))
    assert result.verdict is SupportVerdict.UNSUPPORTED
    assert result.validator_method is ValidatorMethod.MISSING_SOURCE_TEXT


def test_bibliography_like_chunk_is_rejected():
    reference_list = (
        "References\n"
        "Smith, J. (2020) 'A study of AI governance', Journal of AI, 12(3), pp. 10-25. "
        "doi:10.1234/abc\n"
        "Jones, A. and Lee, B. (2019) 'Another study', Journal of Ethics, 4(1), pp. 1-9. "
        "doi:10.1234/def\n"
        "Chan, C. (2024) 'Higher education and AI', Higher Education, 7(2), pp. 30-44. "
        "doi:10.1234/ghi\n"
        "Brown, D. (2022) 'Governance in practice', Policy Review, 9(1), pp. 5-15. "
        "doi:10.1234/jkl\n"
    )
    result = validate(SENIOR_CLAIM, block("S1", text=reference_list))
    assert result.verdict is SupportVerdict.UNSUPPORTED
    assert result.validator_method is ValidatorMethod.BIBLIOGRAPHY_LIKE_SOURCE
    assert "bibliography-like chunk cited" in " ".join(result.warnings)


# --- multiple sources ---------------------------------------------------------


def test_multi_source_claim_supported_by_one_source_alone():
    first = block(
        "S1",
        text="Brown (2022) reports that governance frameworks protect student privacy.",
    )
    second = block(
        "S2",
        text=FACULTY_TEXT,
    )
    result = validate(
        "Governance frameworks protect student privacy [S1,S2]", first, second
    )
    assert result.verdict is SupportVerdict.SUPPORTED
    assert result.combination_required is False
    assert result.cited_ids == ("S1", "S2")


def test_multi_source_claim_supported_only_by_the_combination():
    first = block(
        "S1",
        text="Adaptive learning platforms adjust task difficulty from learner responses.",
    )
    second = block(
        "S2",
        text="A review of higher education institutions reports improved retention rates.",
    )
    result = validate(
        "Adaptive learning platforms adjust task difficulty and improve retention rates "
        "in higher education institutions [S1,S2]",
        first,
        second,
    )
    assert result.verdict is SupportVerdict.SUPPORTED
    assert result.combination_required is True
    assert all(
        record.verdict is not SupportVerdict.SUPPORTED for record in result.per_source
    )
    assert any("the combination does" in warning for warning in result.warnings)


def test_multi_source_with_one_weak_source_still_follows_the_strong_one():
    strong = block("S1")
    weak = block("S2", text="Krill feeding habits vary with the season and the sea ice.")
    result = validate(SENIOR_CLAIM.replace("[S1]", "[S1,S2]"), strong, weak)
    assert result.verdict is SupportVerdict.SUPPORTED
    verdicts = {record.source_id: record.verdict for record in result.per_source}
    assert verdicts["S1"] is SupportVerdict.SUPPORTED
    assert verdicts["S2"] is not SupportVerdict.SUPPORTED


def test_partially_supported_claim_from_one_source_merging_two_findings():
    result = validate(FACULTY_CLAIM, block("S3", text=FACULTY_TEXT, pages="12"))
    assert result.verdict is SupportVerdict.PARTIALLY_SUPPORTED
    assert result.validator_method is ValidatorMethod.LEXICAL_MERGED_FINDINGS
    assert result.combination_required is False
    assert result.decision.value == "annotated"
    record = result.per_source[0]
    assert record.union_coverage > record.best_sentence_coverage


def test_claim_whose_content_is_only_partly_in_the_evidence_is_partial():
    source = block(
        "S1",
        text=(
            "Adaptive tutorials in mathematics adjust task difficulty from learner "
            "responses during the course."
        ),
    )
    result = validate(
        "Adaptive tutorials adjust task difficulty from learner responses and reduce "
        "dropout among first-year engineering undergraduates in large lecture halls [S1]",
        source,
    )
    assert result.verdict is SupportVerdict.PARTIALLY_SUPPORTED
    assert result.validator_method is ValidatorMethod.LEXICAL_MULTI_SENTENCE


def test_claim_below_the_partial_threshold_is_uncertain():
    source = block(
        "S1",
        text="Virtual reality increases learner engagement in laboratory courses.",
    )
    result = validate(
        "Immersive virtual reality reduces examination anxiety among psychology "
        "undergraduates and improves their retention of procedural knowledge [S1]",
        source,
    )
    assert result.verdict in (SupportVerdict.UNCERTAIN, SupportVerdict.UNSUPPORTED)


# --- numeric claims -----------------------------------------------------------


def test_supported_numeric_claim_is_verified():
    source = block(
        "S1",
        text=(
            "A mixed-methods approach with 150 participants found that 75% cited "
            "academic integrity as a primary concern (p < 0.05)."
        ),
    )
    result = validate(
        "A study of 150 participants found that 75% cited academic integrity as a "
        "primary concern [S1]",
        source,
    )
    assert result.verdict is SupportVerdict.SUPPORTED
    assert {mention.kind for mention in result.numeric_mentions} == {
        NumericKind.COUNT,
        NumericKind.PERCENT,
    }
    assert all(mention.found for mention in result.numeric_mentions)


def test_unsupported_numeric_claim_is_caught_despite_similar_wording():
    source = block(
        "S1",
        text=(
            "A mixed-methods approach with 150 participants found that 75% cited "
            "academic integrity as a primary concern."
        ),
    )
    result = validate(
        "A study of 150 participants found that 88% cited academic integrity as a "
        "primary concern [S1]",
        source,
    )
    assert result.verdict is SupportVerdict.UNSUPPORTED
    assert result.validator_method is ValidatorMethod.NUMERIC_MISSING
    missing = [mention for mention in result.numeric_mentions if mention.found is False]
    assert [mention.raw for mention in missing] == ["88%"]


def test_p_value_must_be_present_in_the_evidence():
    source = block(
        "S1",
        text="Statistically significant increases in engagement were observed (p < 0.05).",
    )
    result = validate(
        "VR exposure produced a significant increase in engagement (p < 0.01) [S1]",
        source,
    )
    assert result.verdict is SupportVerdict.UNSUPPORTED
    assert result.validator_method is ValidatorMethod.NUMERIC_MISSING


def test_p_value_formatting_difference_still_matches():
    source = block(
        "S1",
        text="Statistically significant increases in engagement were observed (p<0.05).",
    )
    result = validate(
        "VR exposure produced a significant increase in engagement (p < 0.05) [S1]",
        source,
    )
    assert result.verdict is SupportVerdict.SUPPORTED


def test_percentage_is_derivable_from_its_decimal_equivalent():
    source = block("S1", text="The completion rate was 0.75 across the cohort.")
    result = validate("The completion rate was 75% across the cohort [S1]", source)
    assert result.verdict is SupportVerdict.SUPPORTED
    assert result.numeric_mentions[0].derivation == "percent_written_as_decimal"


def test_year_in_a_citation_matches_the_work_metadata():
    source = block("S1", text="The framework has three components.", year=2011)
    result = validate(
        "Mikropoulos and Natsis (2011) describe a framework with three components [S1]",
        source,
    )
    assert result.verdict is SupportVerdict.SUPPORTED
    year = next(
        mention for mention in result.numeric_mentions if mention.kind is NumericKind.YEAR
    )
    assert year.found is True
    assert year.derivation == "citation_metadata_year"


def test_ranges_are_checked_as_ranges():
    source = block("S1", text="Scenes contained between 20-62 images each.")
    good = validate("Each scene held 20-62 images [S1]", source)
    assert good.verdict is SupportVerdict.SUPPORTED
    bad = validate("Each scene held 30-80 images [S1]", source)
    assert bad.verdict is SupportVerdict.UNSUPPORTED
    assert bad.validator_method is ValidatorMethod.NUMERIC_MISSING


def test_source_marker_digits_are_not_treated_as_numbers():
    result = validate(SENIOR_CLAIM, block("S1"))
    assert result.numeric_mentions == ()


# --- quotations ---------------------------------------------------------------


def test_exact_quotation_is_found_and_reported_verbatim():
    result = validate(
        'The source states "Senior management holds significant responsibility" [S1]',
        block("S1"),
    )
    assert result.verdict is SupportVerdict.SUPPORTED
    assert result.validator_method is ValidatorMethod.QUOTATION_EXACT
    check = result.quotations[0]
    assert check.found_in_source is True
    assert check.match_level == "verbatim"
    assert check.matched_source_id == "S1"
    assert check.page_provenance == "4-6"


def test_quotation_that_is_not_in_the_source_fails():
    result = validate(
        'The source states "senior managers are liable for every algorithmic decision" [S1]',
        block("S1"),
    )
    assert result.verdict is SupportVerdict.UNSUPPORTED
    assert result.validator_method is ValidatorMethod.QUOTATION_MISMATCH
    assert result.quotations[0].found_in_source is False
    assert result.quotations[0].match_level == "not_found"


def test_quotation_present_only_in_another_source_fails():
    first = block("S1")
    second = block(
        "S2", text="A separate paper notes that senior management sets the agenda."
    )
    result = validate(
        'The paper states "senior management sets the agenda" [S1]', first, second
    )
    assert result.verdict is SupportVerdict.UNSUPPORTED
    assert result.quotations[0].outside_cited_sources is True
    assert result.quotations[0].matched_source_id == "S2"


def test_elided_quotation_is_accepted_when_every_fragment_is_present():
    result = validate(
        'The source states "Senior management holds significant responsibility ... '
        'policy formulation" [S1]',
        block("S1"),
    )
    assert result.verdict is SupportVerdict.SUPPORTED
    assert result.quotations[0].match_level == "elided"
    assert result.quotations[0].elided is True


def test_quotation_with_changed_wording_fails():
    result = validate(
        'The source states "senior management bears significant responsibility" [S1]',
        block("S1", text=SENIOR_TEXT),
    )
    assert result.verdict is SupportVerdict.UNSUPPORTED
    assert result.validator_method is ValidatorMethod.QUOTATION_MISMATCH


# --- OCR and table risk --------------------------------------------------------


def test_ocr_source_sets_elevated_validation_risk():
    result = validate(SENIOR_CLAIM, block("S1", ocr_involved=True))
    assert result.elevated_validation_risk is True
    assert any("OCR" in reason for reason in result.risk_reasons)
    assert result.verdict is SupportVerdict.SUPPORTED


def test_table_source_sets_elevated_validation_risk():
    result = validate(SENIOR_CLAIM, block("S1", content_type="table"))
    assert result.elevated_validation_risk is True
    assert any("table" in reason for reason in result.risk_reasons)


def test_ocr_claim_near_the_threshold_becomes_partial_not_supported():
    source = block("S1", text=SENIOR_TEXT, ocr_involved=True)
    result = validate(SENIOR_CLAIM, source)
    assert result.elevated_validation_risk is True
    assert "stricter thresholds applied" in " ".join(result.risk_reasons)


def test_risk_is_not_set_for_plain_text():
    result = validate(SENIOR_CLAIM, block("S1"))
    assert result.elevated_validation_risk is False
    assert result.risk_reasons == ()


# --- sentence kinds ------------------------------------------------------------


def test_heading_with_a_marker_loses_the_marker_without_becoming_a_claim():
    result = validate("## Findings [S1]", block("S1"))
    assert result.kind is ClaimKind.NON_CLAIM
    assert result.decision.value == "marker_removed"


def test_non_claim_sentence_without_a_marker_is_untouched():
    result = validate("In summary, the evidence points one way.", block("S1"))
    assert result.kind is ClaimKind.NON_CLAIM
    assert result.decision.value == "unchanged"


def test_unmarked_claim_is_reported_as_uncertain():
    result = validate(
        "The study found that governance frameworks protect student privacy.", block("S1")
    )
    assert result.kind is ClaimKind.UNMARKED_CLAIM
    assert result.verdict is SupportVerdict.UNCERTAIN


def test_reported_own_work_sentence_is_not_citable():
    my_work = block(
        "S6",
        text=SENIOR_TEXT,
        source_class=SourceClass.MY_WORK,
        citable=False,
        corpus_role="my_work",
        evidence_eligible=False,
    )
    result = validate_claim(
        "Senior management is responsible for AI policy [S6]",
        context=make_context([my_work]),
        kind=ClaimKind.NON_EVIDENCE_REPORT,
    )
    assert result.decision.value == "non_citable_report"
    assert result.verdict is SupportVerdict.UNCERTAIN


# --- policy --------------------------------------------------------------------


def test_lexical_result_thresholds_are_configurable():
    policy = ValidationPolicy(support_coverage=0.95)
    result = lexical_result(
        "AI governance refers to senior management responsibility for policy [S1]",
        [SENIOR_TEXT],
        policy=policy,
    )
    assert result.verdict is not SupportVerdict.SUPPORTED


def test_thresholds_are_recorded_in_the_result():
    policy = ValidationPolicy(support_coverage=0.5)
    result = validate(SENIOR_CLAIM, block("S1"), policy=policy)
    assert result.verdict is SupportVerdict.SUPPORTED


def test_lower_support_threshold_accepts_less_overlap():
    source = block(
        "S1",
        text="Adaptive tutorials adjust task difficulty from learner responses.",
    )
    claim = (
        "Adaptive tutorials adjust task difficulty from learner responses and reduce "
        "dropout among first-year engineering undergraduates [S1]"
    )
    strict = validate(claim, source, policy=ValidationPolicy())
    relaxed = validate(claim, source, policy=ValidationPolicy(support_coverage=0.4))
    assert strict.verdict is SupportVerdict.PARTIALLY_SUPPORTED
    assert relaxed.verdict is SupportVerdict.SUPPORTED


# --- the merged-findings detector ----------------------------------------------


def test_merged_findings_detector_fires_on_two_quantified_findings():
    from app.citations.support import merged_findings

    detected = merged_findings(
        "A majority of faculty consider ethical AI training highly important, with many "
        "advocating assignments that foster critical thinking to mitigate overreliance",
        [FACULTY_TEXT],
    )
    assert detected is not None
    assert detected.distinct_passages >= 2
    assert detected.single_sentence_coverage < 0.6


def test_merged_findings_detector_stays_quiet_when_one_sentence_states_the_claim():
    from app.citations.support import merged_findings

    claim = (
        "A majority of faculty consider training on ethical AI usage as highly important"
    )
    assert merged_findings(claim, [FACULTY_TEXT]) is None


def test_merged_findings_detector_stays_quiet_without_quantified_units():
    from app.citations.support import merged_findings

    assert (
        merged_findings(
            "Senior management is responsible for ethical guidelines and institutional "
            "direction",
            [SENIOR_TEXT],
        )
        is None
    )


def test_merged_findings_detector_forgets_a_unit_the_evidence_never_grounds():
    from app.citations.support import merged_findings

    assert (
        merged_findings(
            "A majority of faculty consider training important, with many advocating "
            "sabbatical reform in Antarctic research stations",
            [FACULTY_TEXT],
        )
        is None
    )


# --- optional semantic layer ----------------------------------------------------


class FakeEmbedder:
    """A stand-in embedder: the first dimension marks the passage it belongs to."""

    def __init__(self, claim_vector, passage_vectors) -> None:
        self._claim = claim_vector
        self._passages = passage_vectors

    def embed_documents(self, texts, **_kwargs):
        vectors = [self._claim]
        for index, _text in enumerate(texts[1:]):
            vectors.append(self._passages[min(index, len(self._passages) - 1)])
        return vectors


def test_embedding_checker_reports_similarity_and_verdict():
    from app.citations.support import EmbeddingEntailmentChecker

    checker = EmbeddingEntailmentChecker(
        FakeEmbedder([1.0, 0.0], [[0.99, 0.01], [0.9, 0.1]])
    )
    result = checker.check("claim", ["passage one", "passage two"])
    assert result.verdict is SupportVerdict.SUPPORTED
    assert result.method is ValidatorMethod.EMBEDDING_SEMANTIC
    assert result.score > 0.9


def test_embedding_checker_can_never_upgrade_partial_support():
    from app.citations.support import EmbeddingEntailmentChecker, validate_claim

    checker = EmbeddingEntailmentChecker(
        FakeEmbedder([1.0, 0.0], [[1.0, 0.0], [1.0, 0.0]])
    )
    result = validate_claim(
        FACULTY_CLAIM,
        context=make_context([block("S3", text=FACULTY_TEXT)]),
        checker=checker,
    )
    assert result.verdict is SupportVerdict.PARTIALLY_SUPPORTED
    assert any("semantic layer" in warning for warning in result.warnings)


def test_embedding_checker_can_resolve_an_uncertain_claim():
    from app.citations.support import EmbeddingEntailmentChecker, validate_claim

    checker = EmbeddingEntailmentChecker(
        FakeEmbedder([1.0, 0.0], [[1.0, 0.0]])
    )
    sentence = (
        "Balanced strategies and regulatory frameworks are needed to keep the risks to "
        "academic honesty manageable [S1]"
    )
    source = block(
        "S1",
        text=(
            "Benefits require mitigating risks through ethical considerations and "
            "guidelines for academic honesty."
        ),
    )
    without = validate_claim(sentence, context=make_context([source]))
    assert without.verdict is not SupportVerdict.SUPPORTED
    with_checker = validate_claim(
        sentence, context=make_context([source]), checker=checker
    )
    assert with_checker.verdict is SupportVerdict.SUPPORTED
    assert with_checker.validator_method is ValidatorMethod.EMBEDDING_SEMANTIC
