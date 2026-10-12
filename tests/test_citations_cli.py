"""Citation CLI tests (RAG-012 §16)."""

from __future__ import annotations

import json

import pytest

from app.citations import inspect as citations_cli
from app.citations.models import ValidationPolicy
from app.citations.service import CitationService
from app.generation.models import GenerationStatus
from tests.citations_support import (
    FACULTY_CLAIM,
    FACULTY_TEXT,
    SENIOR_CLAIM,
    block,
    make_response,
)


class FakeService:
    """A citation service that returns a prepared response without any model."""

    def __init__(self, response) -> None:
        self._response = response
        self.closed = False
        self.policy = ValidationPolicy()

    def answer_and_validate(self, question, **kwargs):
        response = self._response.model_copy(update={"question": question})
        result = CitationService().validate(response)
        return response, result

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def run(monkeypatch):
    """Run the CLI with a scripted generation response."""

    def _run(response, *argv):
        service = FakeService(response)
        monkeypatch.setattr(citations_cli, "build_service", lambda args: service)
        code = citations_cli.main(list(argv))
        return code

    return _run


SUPPORTED_RESPONSE = make_response(
    SENIOR_CLAIM + ".",
    [
        block(
            "S1",
            doi="10.1234/abc",
            title="Governing AI in universities",
            first_author="Anna Chan",
            year=2024,
            pages="175",
        )
    ],
)


def test_default_output_has_the_three_required_sections(run, capsys):
    run(SUPPORTED_RESPONSE, "What is AI governance in higher education?")
    out = capsys.readouterr().out
    assert "VALIDATED ANSWER" in out
    assert "REFERENCES" in out
    assert "VALIDATION" in out
    assert "(Chan, 2024)" in out
    assert "supported        : 1" in out


def test_exit_code_is_zero_when_every_claim_is_supported(run):
    assert run(SUPPORTED_RESPONSE, "Q") == 0


def test_exit_code_is_three_when_a_claim_is_withheld(run, capsys):
    response = make_response(
        "Quantum tunnelling governs Antarctic krill populations [S1].",
        [block("S1")],
    )
    assert run(response, "Q") == 3
    assert "[claim withheld" in capsys.readouterr().out


def test_exit_code_is_one_when_the_model_failed(run):
    response = make_response(
        "", [block("S1")], status=GenerationStatus.MODEL_ERROR
    )
    assert run(response, "Q") == 1


def test_show_claims_prints_the_claim_audit(run, capsys):
    run(SUPPORTED_RESPONSE, "Q", "--show-claims")
    out = capsys.readouterr().out
    assert "CLAIMS" in out
    assert "method=lexical_single_sentence" in out
    assert "[S1] -> (Chan, 2024)" in out


def test_show_claims_reports_numbers_and_quotations(run, capsys):
    response = make_response(
        'The source states "Senior management holds significant responsibility" [S1].',
        [block("S1")],
    )
    run(response, "Q", "--show-claims")
    out = capsys.readouterr().out
    assert "quote: found (verbatim)" in out


def test_show_source_map_prints_provenance_and_citation(run, capsys):
    run(SUPPORTED_RESPONSE, "Q", "--show-source-map")
    out = capsys.readouterr().out
    assert "SOURCE MAP" in out
    assert "chunk=chunk-s1" in out
    assert "citation=(Chan, 2024) reference=doi:10.1234/abc" in out


def test_show_warnings_prints_only_the_warnings_and_validation(run, capsys):
    response = make_response(
        FACULTY_CLAIM + ".", [block("S3", text=FACULTY_TEXT, first_author="Anna Chan")]
    )
    run(response, "Q", "--show-warnings")
    out = capsys.readouterr().out
    assert "WARNINGS" in out
    assert "partially_supported_claim" in out
    assert "VALIDATION" in out
    assert "VALIDATED ANSWER" not in out


def test_json_output_is_the_structured_result(run, capsys):
    run(SUPPORTED_RESPONSE, "What is AI governance?", "--json")
    payload = json.loads(capsys.readouterr().out)
    assert payload["question"] == "What is AI governance?"
    assert payload["original_answer"] == SENIOR_CLAIM + "."
    assert "(Chan, 2024)" in payload["validated_answer"]
    assert payload["validation_summary"]["supported"] == 1
    assert payload["reference_list"][0].startswith("Chan, A. (2024)")
    assert payload["citation_map"][0]["source_id"] == "S1"
    assert payload["generation"]["status"] == "ok"


def test_json_survives_the_model_error_case(run, capsys):
    response = make_response("", [block("S1")], status=GenerationStatus.MODEL_ERROR)
    run(response, "Q", "--json")
    payload = json.loads(capsys.readouterr().out)
    assert payload["generation_status"] == "model_error"


def test_show_original_prints_the_rag011_answer(run, capsys):
    run(SUPPORTED_RESPONSE, "Q", "--show-original")
    out = capsys.readouterr().out
    assert "ORIGINAL RAG-011 ANSWER" in out
    assert "[S1]" in out


def test_parse_args_defaults():
    args = citations_cli.parse_args(["Q"])
    assert args.question == "Q"
    assert args.length == "normal"
    assert args.semantic is False
    assert args.support_coverage is None
    assert args.show_claims is False


def test_parse_args_accepts_the_documented_flags():
    args = citations_cli.parse_args(
        ["Q", "--show-claims", "--show-source-map", "--show-warnings", "--json"]
    )
    assert args.show_claims and args.show_source_map and args.show_warnings and args.json


def test_build_service_uses_the_support_coverage_override(monkeypatch):
    from app.citations import inspect as module

    args = module.parse_args(["Q", "--support-coverage", "0.42", "--semantic"])
    service = module.build_service(args)
    try:
        assert service.policy.support_coverage == 0.42
        assert service.checker is not None
        assert service.validator.startswith("deterministic+")
    finally:
        service.close()
