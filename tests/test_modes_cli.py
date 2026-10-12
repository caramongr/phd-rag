"""Mode CLI tests (RAG-013 §11)."""

from __future__ import annotations

import json

import pytest

from app.modes import inspect as modes_cli
from app.modes.models import ModeName
from tests.modes_support import build_harness, external_record


@pytest.fixture
def run(monkeypatch):
    """Run the CLI with a scripted mode service."""

    def _run(harness, *argv):
        monkeypatch.setattr(modes_cli, "build_service", lambda args: harness.service)
        return modes_cli.main(list(argv))

    return _run


def test_parse_args_defaults():
    args = modes_cli.parse_args(["evidence", "a question"])
    assert args.mode == "evidence"
    assert args.query == "a question"
    assert args.show_sources is False
    assert args.show_audit is False
    assert args.json is False
    assert args.external is False


def test_parse_args_rejects_an_unknown_mode():
    with pytest.raises(SystemExit):
        modes_cli.parse_args(["nonsense", "q"])


def test_parse_args_accepts_the_documented_flags():
    args = modes_cli.parse_args(
        ["research", "q", "--show-sources", "--show-audit", "--json", "--external", "--local-k", "3"]
    )
    assert args.json and args.show_sources and args.show_audit and args.external
    assert args.local_k == 3


def test_evidence_output_shows_the_validated_answer_and_references(run, capsys):
    harness = build_harness()
    assert run(harness, "evidence", "what is ai governance?") == 0
    out = capsys.readouterr().out
    assert "MODE   : evidence" in out
    assert "VALIDATED ANSWER" in out
    assert "(Chan, 2024)" in out
    assert "REFERENCES" in out
    assert "VALIDATION" in out
    assert "supported        : 1" in out


def test_show_sources_prints_roles_and_retrieval_modes(run, capsys):
    harness = build_harness()
    run(harness, "writing", "draft a paragraph", "--show-sources")
    out = capsys.readouterr().out
    assert "SOURCES" in out
    assert "used_as=evidence" in out
    assert "used_as=context_only" in out
    assert "used_as=instructions_only" in out
    assert "via=my_work" in out


def test_consistency_output_shows_findings_not_citations(run, capsys):
    harness = build_harness()
    run(harness, "consistency", "how do I describe ai governance?")
    out = capsys.readouterr().out
    assert "CONSISTENCY REPORT" in out
    assert "FINDINGS" in out
    assert "REFERENCES" not in out
    assert "VALIDATION" not in out


def test_supervisor_output_is_labelled_as_non_evidence(run, capsys):
    harness = build_harness()
    run(harness, "supervisor", "what guidance do I have about methodology?")
    out = capsys.readouterr().out
    assert "SUPERVISION EXCERPTS (not evidence)" in out
    assert "Supervision note on methodology" in out
    assert "REFERENCES" not in out


def test_research_output_shows_the_separated_sections(run, capsys):
    harness = build_harness(
        discovery_records=[
            external_record(title="New Paper", doi="10.5555/new"),
            external_record(title="Have It", doi="10.5555/old", already_in_corpus=True),
        ]
    )
    run(harness, "research", "find recent work on ai governance")
    out = capsys.readouterr().out
    assert "LOCAL EVIDENCE" in out
    assert "NEW EXTERNAL PAPERS (1)" in out
    assert "ALREADY IN CORPUS (1)" in out
    assert "POSSIBLE MATCHES" in out
    assert "CONFLICTS" in out
    assert "never evidence" in out
    assert "chunk-evidence" in out


def test_show_audit_prints_the_audit_trail(run, capsys):
    harness = build_harness()
    run(harness, "evidence", "q", "--show-audit")
    out = capsys.readouterr().out
    assert "AUDIT" in out
    assert '"mode": "evidence"' in out
    assert "GENERATION STATUS" in out


def test_json_output_is_the_structured_result(run, capsys):
    harness = build_harness()
    run(harness, "evidence", "what is ai governance?", "--json")
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "evidence"
    assert payload["query"] == "what is ai governance?"
    assert "(Chan, 2024)" in payload["validated_answer"]
    assert payload["references"]
    assert payload["validation_summary"]["supported"] == 1
    assert payload["config"]["name"] == "evidence"
    assert payload["audits"]["policy_note"]


def test_json_output_for_every_mode_is_serialisable(run, capsys):
    harness = build_harness(discovery_records=[external_record()])
    for mode in ModeName:
        run(harness, mode.value, "q", "--json")
        payload = json.loads(capsys.readouterr().out)
        assert payload["mode"] == mode.value


def test_exit_code_is_one_on_a_model_error(run):
    from app.generation.models import ModelFailure

    harness = build_harness()
    harness.model.failure = ModelFailure(kind="connection_error", message="down")
    assert run(harness, "evidence", "q") == 1


def test_exit_code_is_three_when_a_claim_is_withheld(run, capsys):
    harness = build_harness(
        model_content="Quantum tunnelling governs Antarctic krill populations [S1]."
    )
    assert run(harness, "evidence", "q") == 3
    assert "[claim withheld" in capsys.readouterr().out


def test_exit_code_is_zero_for_the_non_generating_modes(run):
    harness = build_harness(discovery_records=[external_record()])
    for mode in ("consistency", "supervisor", "research"):
        assert run(harness, mode, "ai governance") == 0


def test_external_flag_enables_discovery_for_evidence(run, capsys):
    harness = build_harness(discovery_records=[external_record()])
    run(harness, "evidence", "q", "--external", "--show-sources")
    out = capsys.readouterr().out
    assert harness.discovery.calls, "the external flag must run discovery"
    assert "EXTERNAL RECORDS" in out
