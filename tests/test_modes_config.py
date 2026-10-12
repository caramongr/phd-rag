"""Mode preset tests (RAG-013 §7, §10, §13)."""

from __future__ import annotations

import pytest

from app.generation.models import GenerationMode
from app.modes.models import (
    MODES,
    ModeConfig,
    ModeName,
    OutputStyle,
    SourceRole,
    mode_config,
)
from app.retrieval.models import RetrievalMode


def test_every_mode_has_a_preset():
    assert set(MODES) == set(ModeName)


def test_preset_configuration_per_mode():
    """The ten configuration decisions of §7, pinned per mode."""
    expected = {
        ModeName.EVIDENCE: (RetrievalMode.EVIDENCE, False, False, False, False, True, True),
        ModeName.WRITING: (RetrievalMode.EVIDENCE, True, True, False, False, True, True),
        ModeName.CONSISTENCY: (RetrievalMode.MY_WORK, True, True, False, False, False, False),
        ModeName.SUPERVISOR: (
            RetrievalMode.SUPERVISION,
            False,
            True,
            True,
            False,
            False,
            False,
        ),
        ModeName.RESEARCH: (RetrievalMode.EVIDENCE, False, False, False, True, False, False),
    }
    for mode, (
        retrieval_mode,
        include_my_work,
        include_guidance,
        include_supervision,
        use_external,
        run_generation,
        run_validation,
    ) in expected.items():
        config = MODES[mode]
        assert config.retrieval_mode is retrieval_mode
        assert config.include_my_work is include_my_work
        assert config.include_guidance is include_guidance
        assert config.include_supervision is include_supervision
        assert config.use_external_discovery is use_external
        assert config.run_generation is run_generation
        assert config.run_citation_validation is run_validation


def test_mode_configuration_is_deterministic():
    first = mode_config("evidence")
    second = mode_config(ModeName.EVIDENCE)
    assert first is second  # frozen data, one object per mode
    assert first.model_dump() == second.model_dump()


def test_evidence_mode_treats_only_literature_as_evidence():
    config = MODES[ModeName.EVIDENCE]
    assert config.role_for("literature") is SourceRole.EVIDENCE
    assert config.role_for("my_work") is not SourceRole.EVIDENCE
    assert config.role_for("supervision") is not SourceRole.EVIDENCE
    assert config.role_for(None) is not SourceRole.EVIDENCE


def test_writing_mode_keeps_my_work_and_rules_non_citable():
    config = MODES[ModeName.WRITING]
    assert config.role_for("literature") is SourceRole.EVIDENCE
    assert config.role_for("my_work") is SourceRole.CONTEXT_ONLY
    assert config.role_for("rules") is SourceRole.INSTRUCTIONS_ONLY
    assert config.generation_mode is GenerationMode.EVIDENCE_PLUS_GUIDANCE


def test_supervisor_mode_never_promises_evidence_roles():
    config = MODES[ModeName.SUPERVISOR]
    assert set(config.context_policy.values()) <= {
        SourceRole.GUIDANCE_ONLY,
        SourceRole.CONTEXT_ONLY,
    }
    assert config.allows_scholarly_claims is False
    assert config.output_style is OutputStyle.LABELLED_EXCERPTS


def test_consistency_mode_has_my_work_as_subject_and_literature_as_reference():
    config = MODES[ModeName.CONSISTENCY]
    assert config.role_for("my_work") is SourceRole.SUBJECT
    assert config.role_for("literature") is SourceRole.EVIDENCE
    assert config.compare_with_literature is True
    assert config.allows_scholarly_claims is False


def test_research_mode_separates_results():
    config = MODES[ModeName.RESEARCH]
    assert config.use_external_discovery is True
    assert config.run_generation is False
    assert config.output_style.value == "separated_results"


def test_only_generation_modes_validate_citations():
    for config in MODES.values():
        if config.run_citation_validation:
            assert config.run_generation, config.name
            assert config.output_style is OutputStyle.VALIDATED_ANSWER


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        mode_config("nonsense")


def test_mode_config_is_frozen():
    config = MODES[ModeName.EVIDENCE]
    with pytest.raises(Exception):
        config.run_generation = False  # type: ignore[misc]


def test_context_policy_defaults_to_instructions_only():
    config = ModeConfig(
        name=ModeName.EVIDENCE,
        purpose="test",
        retrieval_mode=RetrievalMode.EVIDENCE,
        generation_mode=GenerationMode.EVIDENCE_ONLY,
    )
    assert config.role_for("literature") is SourceRole.INSTRUCTIONS_ONLY
