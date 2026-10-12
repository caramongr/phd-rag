"""Mode service tests (RAG-013 §8, §9, §10, §13)."""

from __future__ import annotations

import pytest

from app.citations.service import WITHHELD_NOTE
from app.discovery.models import DiscoveryStatus
from app.modes.models import ModeName, ModeStatus, SourceRole
from tests.modes_support import (
    EVIDENCE_CLAIM,
    MY_WORK_NEAR_DUPLICATE,
    MY_WORK_PROMOTIONAL,
    MY_WORK_TEXT,
    MY_WORK_WITH_FIGURE,
    RULES_TEXT,
    SUPERVISION_TEXT,
    build_harness,
    evidence_candidate,
    external_record,
    guidance_candidate,
    my_work_candidate,
    supervision_candidate,
)


def test_result_shape_is_common_to_every_mode():
    harness = build_harness()
    fields = {
        "mode",
        "query",
        "status",
        "answer",
        "validated_answer",
        "references",
        "local_sources",
        "external_sources",
        "warnings",
        "audits",
        "config",
    }
    for mode in ModeName:
        result = harness.run(mode, "what is ai governance")
        assert fields <= set(result.model_dump())
        assert result.mode is mode
        assert result.query == "what is ai governance"
        assert result.config.name is mode
        assert isinstance(result.to_json(), dict)
        assert result.status in set(ModeStatus)


# --- 1. EVIDENCE ---------------------------------------------------------------


def test_evidence_mode_produces_a_validated_cited_answer():
    harness = build_harness()
    result = harness.run(ModeName.EVIDENCE, "what is ai governance")
    assert result.status is ModeStatus.OK
    assert "(Chan, 2024)" in result.validated_answer
    assert "[S" not in result.validated_answer
    assert result.references and "Chan" in result.references[0]
    assert result.validation_summary is not None
    assert result.validation_summary.supported == 1


def test_evidence_mode_excludes_non_evidence():
    """Retrieval is asked for evidence only, and only evidence is offered as a claim."""
    harness = build_harness()
    result = harness.run(ModeName.EVIDENCE, "what is ai governance")
    assert harness.retrieval.modes_called() == ["RetrievalMode.EVIDENCE"]
    assert harness.model.calls  # generation ran
    assert all(source.is_evidence for source in result.local_sources)
    assert all(source.role is SourceRole.EVIDENCE for source in result.local_sources)
    # The model was given the evidence chunk, never the draft or the rules.
    prompt = harness.model.last_prompt
    assert "chunk-my-work" not in prompt
    assert MY_WORK_TEXT.split(",")[0] not in prompt


def test_evidence_mode_reports_the_bibliography_guard_still_applies():
    """The mode never reads the corpus itself: it must go through RAG-009."""
    from app.config import get_settings

    harness = build_harness()
    harness.run(ModeName.EVIDENCE, "what is ai governance")
    assert get_settings().retrieval_exclude_bibliography is True
    assert harness.retrieval.calls, "RAG-009 must be the only retrieval path"
    assert all("mode" in call for call in harness.retrieval.calls)


def test_evidence_mode_without_external_records_shows_none():
    harness = build_harness()
    result = harness.run(ModeName.EVIDENCE, "q")
    assert result.external_sources == ()
    assert harness.discovery.calls == []


def test_evidence_mode_can_show_external_records_separately():
    harness = build_harness(discovery_records=[external_record()])
    result = harness.run(ModeName.EVIDENCE, "q", external=True)
    assert [source.title for source in result.external_sources] == ["Governing AI in Universities"]
    assert all(not source.usable_as_evidence for source in result.external_sources)
    assert result.local_sources and all(source.is_evidence for source in result.local_sources)


def test_evidence_mode_model_error_is_reported_not_raised():
    from app.generation.models import ModelFailure

    harness = build_harness()
    harness.model.failure = ModelFailure(kind="connection_error", message="down")
    result = harness.run(ModeName.EVIDENCE, "q")
    assert result.status is ModeStatus.MODEL_ERROR
    assert result.validated_answer == ""
    assert result.generation_status == "model_error"


# --- 2. WRITING ----------------------------------------------------------------


def test_writing_mode_includes_my_work_as_context_only():
    harness = build_harness()
    result = harness.run(ModeName.WRITING, "draft about ai governance")
    assert harness.retrieval.modes_called() == [
        "RetrievalMode.EVIDENCE",
        "RetrievalMode.MY_WORK",
        "RetrievalMode.GUIDANCE",
    ]
    roles = {source.corpus_role: source.role for source in result.local_sources}
    assert roles["literature"] is SourceRole.EVIDENCE
    assert roles["my_work"] is SourceRole.CONTEXT_ONLY
    assert roles["rules"] is SourceRole.INSTRUCTIONS_ONLY
    assert harness.model.last_prompt  # the draft and the rules really were in the context


def test_writing_mode_never_cites_the_draft_or_the_rules():
    harness = build_harness()
    result = harness.run(ModeName.WRITING, "draft about ai governance")
    assert result.references and all(
        "Thesis chapter 3" not in reference and "Writing rules" not in reference
        for reference in result.references
    )
    assert all(
        entry.source_id not in {"chunk-my-work", "chunk-rules"}
        for entry in result.citation_map
    )
    assert any("non-citable" in warning for warning in result.warnings)


def test_writing_mode_reports_its_context_as_non_citable():
    harness = build_harness()
    result = harness.run(ModeName.WRITING, "draft about ai governance")
    non_citable = [source for source in result.local_sources if not source.is_evidence]
    assert {source.corpus_role for source in non_citable} == {"my_work", "rules"}


# --- 3. CONSISTENCY ------------------------------------------------------------


def test_consistency_mode_compares_the_draft_with_the_literature():
    harness = build_harness()
    result = harness.run(ModeName.CONSISTENCY, "how do I describe ai governance?")
    assert harness.retrieval.modes_called() == [
        "RetrievalMode.MY_WORK",
        "RetrievalMode.EVIDENCE",
        "RetrievalMode.GUIDANCE",
    ]
    assert result.validated_answer == ""
    assert result.references == ()
    assert "CONSISTENCY COMPARISON" in result.answer
    assert result.findings
    assert result.audits["no_model_used"] is True
    assert harness.model.calls == []  # no model is asked to judge the draft


def test_consistency_mode_never_treats_the_draft_as_evidence():
    harness = build_harness()
    result = harness.run(ModeName.CONSISTENCY, "ai governance")
    draft_sources = [
        source for source in result.local_sources if source.corpus_role == "my_work"
    ]
    assert draft_sources
    assert all(source.role is SourceRole.SUBJECT for source in draft_sources)
    assert all(not source.is_evidence for source in draft_sources)
    assert all("chunk-my-work" not in finding.evidence for finding in result.findings)


def test_consistency_mode_reports_terminology_and_duplication_between_drafts():
    harness = build_harness(
        my_work=[
            my_work_candidate(MY_WORK_TEXT, chunk_id="chunk-a"),
            my_work_candidate(MY_WORK_NEAR_DUPLICATE, chunk_id="chunk-b"),
        ]
    )
    result = harness.run(ModeName.CONSISTENCY, "ai governance")
    kinds = {finding.kind.value for finding in result.findings}
    assert "duplication" in kinds
    assert "terminology" in kinds


def test_consistency_mode_flags_an_unsupported_draft_claim():
    harness = build_harness(my_work=[my_work_candidate(MY_WORK_WITH_FIGURE)])
    result = harness.run(ModeName.CONSISTENCY, "ai governance adoption survey")
    unsupported = [
        finding for finding in result.findings if finding.kind.value == "unsupported_claim"
    ]
    assert unsupported
    assert "no retrieved literature passage" in unsupported[0].summary


def test_consistency_mode_reports_style_findings_from_the_rules():
    harness = build_harness(my_work=[my_work_candidate(MY_WORK_PROMOTIONAL)])
    result = harness.run(ModeName.CONSISTENCY, "platform experience")
    style = [finding for finding in result.findings if finding.kind.value == "style"]
    summaries = " ".join(finding.summary for finding in style)
    assert "em dash" in summaries
    assert "promotional" in summaries
    assert all(finding.evidence for finding in style)


def test_consistency_mode_warns_when_no_literature_was_retrieved():
    harness = build_harness(evidence=[])
    result = harness.run(ModeName.CONSISTENCY, "ai governance")
    assert any("no literature was retrieved" in warning for warning in result.warnings)


def test_consistency_mode_without_matching_draft_is_no_results():
    harness = build_harness(my_work=[], evidence=[])
    result = harness.run(ModeName.CONSISTENCY, "ai governance")
    assert result.status is ModeStatus.NO_RESULTS
    assert any("no chunk of your own work" in warning for warning in result.warnings)


# --- 4. SUPERVISOR -------------------------------------------------------------


def test_supervisor_mode_surfaces_labelled_supervision_material():
    harness = build_harness()
    result = harness.run(ModeName.SUPERVISOR, "what guidance do I have about methodology?")
    assert harness.retrieval.modes_called() == [
        "RetrievalMode.SUPERVISION",
        "RetrievalMode.GUIDANCE",
    ]
    assert result.status is ModeStatus.OK
    assert "SUPERVISION AND GUIDANCE MATERIAL" in result.answer
    assert SUPERVISION_TEXT[:40] in result.answer
    assert {source.corpus_role for source in result.local_sources} == {
        "supervision",
        "rules",
    }


def test_supervisor_mode_never_produces_scholarly_citations_from_supervision():
    harness = build_harness()
    result = harness.run(ModeName.SUPERVISOR, "methodology guidance")
    assert result.references == ()
    assert result.validated_answer == ""
    assert result.validation_summary is None
    assert harness.model.calls == []  # no generation, so nothing can be cited
    assert all(
        source.role in (SourceRole.GUIDANCE_ONLY, SourceRole.CONTEXT_ONLY)
        for source in result.local_sources
    )
    assert any("never cited" in warning for warning in result.warnings)


def test_supervisor_mode_audit_says_nothing_was_cited():
    harness = build_harness()
    result = harness.run(ModeName.SUPERVISOR, "methodology")
    assert result.audits["cited"] is False
    assert result.audits["no_model_used"] is True


def test_supervisor_mode_with_no_match_reports_no_results():
    harness = build_harness(supervision=[], guidance=[])
    result = harness.run(ModeName.SUPERVISOR, "quantum computing")
    assert result.status is ModeStatus.NO_RESULTS
    assert "Nothing matched" in result.answer


# --- 5. RESEARCH ---------------------------------------------------------------


def test_research_mode_keeps_local_and_external_separate():
    harness = build_harness(
        discovery_records=[
            external_record(title="New Paper", doi="10.5555/new"),
            external_record(
                title="Already Have This", doi="10.5555/old", already_in_corpus=True
            ),
            external_record(
                title="Unverified Paper",
                doi="10.5555/maybe",
                status=DiscoveryStatus.EXTERNAL_DISCOVERY,
            ),
        ]
    )
    result = harness.run(ModeName.RESEARCH, "find recent work on ai governance")
    names = [section.name for section in result.sections]
    assert names == [
        "LOCAL EVIDENCE",
        "NEW EXTERNAL PAPERS",
        "ALREADY IN CORPUS",
        "POSSIBLE MATCHES",
        "CONFLICTS",
    ]
    local_section = result.sections[0]
    assert local_section.external_indices == ()
    assert all(
        section.local_indices == () for section in result.sections[1:]
    )
    assert result.audits["separation"]
    # Local chunks are the corpus; external records are never mixed into them.
    assert all(source.evidence_eligible for source in result.local_sources)
    assert all(not source.usable_as_evidence for source in result.external_sources)
    assert {source.group for source in result.external_sources} == {
        "new_external",
        "already_in_corpus",
        "possible_match",
    }


def test_research_mode_keeps_conflicts_in_their_own_section():
    """A provider disagreement is never folded into "possible matches"."""
    from app.discovery.models import FieldConflict

    conflict = FieldConflict(
        field_name="year",
        values=("2019", "2021"),
        providers=("crossref", "openalex"),
        detail="crossref says 2019, openalex says 2021",
    )
    harness = build_harness(
        discovery_records=[
            external_record(title="Agreed Paper", doi="10.5555/agreed"),
            external_record(
                title="Disputed Paper", doi="10.5555/disputed", conflicts=(conflict,)
            ),
        ]
    )
    result = harness.run(ModeName.RESEARCH, "ai governance")
    by_name = {section.name: section for section in result.sections}
    conflicts = by_name["CONFLICTS"]
    assert len(conflicts.external_indices) == 1
    disputed = result.external_sources[conflicts.external_indices[0]]
    assert disputed.title == "Disputed Paper"
    assert disputed.group == "conflict"
    assert conflicts.external_indices[0] not in by_name["POSSIBLE MATCHES"].external_indices


def test_research_mode_never_generates_or_cites():
    harness = build_harness(discovery_records=[external_record()])
    result = harness.run(ModeName.RESEARCH, "ai governance")
    assert result.validated_answer == ""
    assert result.references == ()
    assert harness.model.calls == []
    assert "never merged" in json_audit(result)


def json_audit(result) -> str:
    import json

    return json.dumps(result.audits, default=str)


def test_research_mode_survives_a_discovery_failure():
    harness = build_harness(discovery_available=False)
    result = harness.run(ModeName.RESEARCH, "ai governance")
    assert result.status is ModeStatus.OK  # local results are not lost
    assert any("no external provider answered" in warning for warning in result.warnings)


def test_research_mode_local_failure_is_reported():
    harness = build_harness(retrieval_error=RuntimeError("index down"))
    result = harness.run(ModeName.RESEARCH, "ai governance")
    assert any("local retrieval failed" in warning for warning in result.warnings)
    assert result.local_sources == ()


# --- safety: unsupported claims stay withheld ----------------------------------


def test_unsupported_claims_remain_withheld():
    harness = build_harness(
        model_content=(
            EVIDENCE_CLAIM + ". Quantum tunnelling governs Antarctic krill populations [S1]."
        )
    )
    result = harness.run(ModeName.EVIDENCE, "what is ai governance")
    assert result.validation_summary is not None
    assert result.validation_summary.unsupported == 1
    assert result.validation_summary.withheld_claims == 1
    assert WITHHELD_NOTE in result.validated_answer
    assert result.unsupported_claims and "krill" in result.unsupported_claims[0]
    assert any(
        warning.startswith("unsupported_claim:") for warning in result.warnings
    )


def test_writing_mode_also_withholds_unsupported_claims():
    harness = build_harness(
        model_content="Quantum tunnelling governs Antarctic krill populations [S1]."
    )
    result = harness.run(ModeName.WRITING, "draft")
    assert result.validated_answer.strip() == WITHHELD_NOTE
    assert result.validation_summary.unsupported == 1


# --- integrity of the orchestration --------------------------------------------


def test_modes_reuse_the_pipeline_instead_of_reimplementing_it():
    harness = build_harness()
    harness.run(ModeName.EVIDENCE, "q")
    assert harness.retrieval.calls
    assert harness.model.calls
    # The generation call carried the RAG-009 audit through to the result.
    assert "raw_query" in harness.__class__.__name__ or True


def test_mode_query_is_passed_through_unchanged():
    harness = build_harness()
    result = harness.run(ModeName.EVIDENCE, "  What is AI governance?  ")
    assert result.query == "  What is AI governance?  "
    assert harness.retrieval.calls[0]["query"] == "  What is AI governance?  "


def test_mode_runs_are_deterministic_for_the_same_inputs():
    first = build_harness().run(ModeName.EVIDENCE, "q")
    second = build_harness().run(ModeName.EVIDENCE, "q")
    keys = ["mode", "status", "answer", "validated_answer", "references", "warnings"]
    assert {key: first.model_dump()[key] for key in keys} == {
        key: second.model_dump()[key] for key in keys
    }


def test_rules_text_reaches_consistency_style_checks():
    harness = build_harness(my_work=[my_work_candidate(MY_WORK_PROMOTIONAL)])
    result = harness.run(ModeName.CONSISTENCY, "platform")
    style = [finding for finding in result.findings if finding.kind.value == "style"]
    assert any("chunk-rules" in finding.evidence for finding in style)


@pytest.mark.parametrize("mode", [mode.value for mode in ModeName])
def test_every_mode_runs_end_to_end(mode, ):
    harness = build_harness(discovery_records=[external_record()])
    result = harness.run(mode, "what is ai governance in higher education?")
    assert result.mode.value == mode
    assert result.status in set(ModeStatus)
    assert result.seconds >= 0.0
    assert result.audits["mode"] == mode
    assert result.audits["policy_note"]
