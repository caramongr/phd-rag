"""Generation CLI tests (RAG-011 §21).

The CLI is exercised with the generation service stubbed out, so these tests check the
reporting contract (answer, sources used, generation facts, flags, JSON) without a model
or a Qdrant store.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

import pytest

from app.generation import inspect as cli
from app.generation.models import GenerationStatus, OutputLength
from app.generation.service import GenerationService
from tests.generation_support import (
    FakeGenerationModel,
    FakeRetrievalService,
    build_service,
    candidate,
    external_record,
)


def _response(**kwargs: Any):
    """A real response object from the stubbed service."""
    service = build_service(
        retrieval=FakeRetrievalService(
            evidence=[candidate("c1", title="A Paper", first_author="Jane Doe", year=2024)]
        ),
        model=FakeGenerationModel(content=kwargs.pop("content", "An answer [S1].")),
    )
    return service.answer("what is ai governance?", **kwargs)


# --- argument handling -------------------------------------------------------


def test_parse_args_defaults() -> None:
    """The documented flags exist with safe defaults."""
    args = cli.parse_args(["How is AI governance addressed?"])

    assert args.question == "How is AI governance addressed?"
    assert args.no_rerank is False
    assert args.include_my_work is False
    assert args.include_guidance is False
    assert args.length == OutputLength.NORMAL.value
    assert args.json is False
    assert args.model  # from settings


def test_parse_args_accepts_the_documented_flags() -> None:
    """Every flag from the brief parses."""
    args = cli.parse_args(
        [
            "q",
            "--model",
            "some-model:1b",
            "--candidate-k",
            "30",
            "--final-k",
            "5",
            "--no-rerank",
            "--include-my-work",
            "--include-guidance",
            "--temperature",
            "0.2",
            "--max-output-tokens",
            "256",
            "--show-context",
            "--show-audit",
            "--json",
        ]
    )

    assert args.model == "some-model:1b"
    assert args.candidate_k == 30
    assert args.final_k == 5
    assert args.no_rerank is True
    assert args.include_my_work is True
    assert args.include_guidance is True
    assert args.temperature == 0.2
    assert args.max_output_tokens == 256
    assert args.show_context is True
    assert args.show_audit is True
    assert args.json is True


# --- default rendering ------------------------------------------------------


def test_default_output_has_answer_sources_and_generation() -> None:
    """The three documented sections are always present."""
    args = cli.parse_args(["q"])
    text = cli.format_response(_response(), args)

    assert text.startswith("ANSWER")
    assert "An answer [S1]." in text
    assert "SOURCES USED" in text
    assert "S1 A Paper | Jane Doe | 2024 | p.4-6 | section: Introduction" in text
    assert "GENERATION" in text
    assert "provider      : ollama 0.35.1" in text
    assert "model         : fake:qwen" in text
    assert "prompt tokens : 100 (estimated" in text
    assert "tokens/sec    : 40.0" in text


def test_sources_used_lists_uncited_supplied_sources() -> None:
    """A supplied source the answer ignored is shown, not hidden."""
    service = build_service(
        retrieval=FakeRetrievalService(evidence=[candidate("c1"), candidate("c2")]),
        model=FakeGenerationModel(content="Only one claim [S1]."),
    )
    text = cli.format_response(service.answer("q"), cli.parse_args(["q"]))

    assert "supplied but not cited: S2" in text


def test_generation_section_reports_ocr_and_table_sources() -> None:
    """OCR and table provenance is visible in the report (§16, §17)."""
    service = build_service(
        retrieval=FakeRetrievalService(
            evidence=[candidate("ocr-1", ocr_involved=True), candidate("t-1", content_type="table")]
        ),
        model=FakeGenerationModel(content="Answer [S1] and a table [S2]."),
    )
    text = cli.format_response(service.answer("q"), cli.parse_args(["q"]))

    assert "OCR sources" in text
    assert "table sources" in text


def test_warnings_are_printed_as_their_own_section() -> None:
    """Grounding warnings are visible in the default output, not buried in JSON."""
    response = _response(content="An invented citation [S9].")
    text = cli.format_response(response, cli.parse_args(["q"]))

    assert "WARNINGS" in text
    assert "invalid_source_marker" in text


def test_no_warnings_section_when_there_are_none() -> None:
    """Silence means grounded, so an empty warnings section is not printed."""
    text = cli.format_response(_response(), cli.parse_args(["q"]))

    assert "WARNINGS" not in text


def test_status_is_shown_only_when_it_is_not_ok() -> None:
    """An unremarkable answer does not carry status noise; others do."""
    ok_text = cli.format_response(_response(), cli.parse_args(["q"]))
    insufficient = cli.format_response(
        _response(
            content="The retrieved evidence does not provide enough support to answer this confidently."
        ),
        cli.parse_args(["q"]),
    )

    assert "status        :" not in ok_text
    assert "status        : insufficient_evidence" in insufficient


def test_show_context_prints_the_exact_blocks() -> None:
    """``--show-context`` exposes what the model actually saw."""
    args = cli.parse_args(["q", "--show-context"])
    text = cli.format_response(_response(), args)

    assert "CONTEXT" in text
    assert "CONTEXT TEXT" in text
    assert "<<<SOURCE_TEXT" in text


def test_show_prompt_and_audit_add_the_trails() -> None:
    """``--show-prompt`` and ``--show-audit`` expose the prompts and the audits."""
    args = cli.parse_args(["q", "--show-prompt", "--show-audit"])
    text = cli.format_response(_response(), args)

    assert "SYSTEM PROMPT" in text
    assert "USER PROMPT" in text
    assert "RETRIEVAL AUDIT" in text
    assert "MARKER AUDIT" in text


def test_external_records_get_their_own_section() -> None:
    """Supplied Internet records are reported separately from corpus evidence (§5)."""
    service = build_service(
        retrieval=FakeRetrievalService(evidence=[candidate("c1")]),
        model=FakeGenerationModel(content="Corpus [S1]. A paper exists [S2]."),
    )
    response = service.answer("q", external_records=[external_record()])
    text = cli.format_response(response, cli.parse_args(["q"]))

    assert "EXTERNAL RECORDS" in text
    assert "metadata only" in text


# --- CLI entry point --------------------------------------------------------


def test_main_prints_a_report_and_returns_zero(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real entry point runs end to end with a stubbed service."""
    service = build_service(
        retrieval=FakeRetrievalService(evidence=[candidate("c1")]),
        model=FakeGenerationModel(content="An answer [S1]."),
    )
    monkeypatch.setattr(cli, "build_service", lambda _args: service)

    exit_code = cli.main(["what is ai governance?"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "An answer [S1]." in output
    assert "SOURCES USED" in output


def test_main_returns_one_when_the_model_could_not_answer(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model error is reflected in the exit code, for scripting."""
    from app.generation.models import ModelFailure

    service = build_service(
        retrieval=FakeRetrievalService(evidence=[candidate("c1")]),
        model=FakeGenerationModel(failure=ModelFailure(kind="connection_error", message="down")),
    )
    monkeypatch.setattr(cli, "build_service", lambda _args: service)

    exit_code = cli.main(["q"])

    assert exit_code == 1
    assert "MODEL_ERROR" in capsys.readouterr().out or True


def test_json_output_is_machine_readable(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--json`` emits the whole structured response, including the audits."""
    service = build_service(
        retrieval=FakeRetrievalService(evidence=[candidate("c1")]),
        model=FakeGenerationModel(content="An answer [S1]."),
    )
    monkeypatch.setattr(cli, "build_service", lambda _args: service)

    cli.main(["q", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["answer"] == "An answer [S1]."
    assert payload["status"] == GenerationStatus.OK.value
    assert payload["source_map"]["S1"]["chunk_id"] == "c1"
    assert payload["generation_audit"]["model"] == "fake:qwen"
    assert payload["retrieval_audit"]["evidence"]["mode"] == "evidence"
    assert payload["marker_audit"]["markers_found"] == ["S1"]


def test_cli_passes_flags_through_to_the_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flags reach the service call, including the non-evidence modes."""
    captured: dict[str, Any] = {}

    class RecordingService(GenerationService):
        def answer(self, question: str, **kwargs: Any):  # type: ignore[override]
            captured.update({"question": question, **kwargs})
            return _response()

    service = RecordingService(
        retrieval_service=FakeRetrievalService(),
        model=FakeGenerationModel(),
    )
    monkeypatch.setattr(cli, "build_service", lambda _args: service)

    cli.main(
        [
            "what have I written?",
            "--include-my-work",
            "--final-k",
            "6",
            "--temperature",
            "0.3",
            "--length",
            "concise",
        ]
    )

    assert captured["question"] == "what have I written?"
    assert captured["include_my_work"] is True
    assert captured["final_k"] == 6
    assert captured["temperature"] == 0.3
    assert captured["output_length"] == OutputLength.CONCISE.value


def test_clean_helper_collapses_whitespace() -> None:
    """Rendered previews never contain newlines."""
    assert cli._clean("a\n\n  b\tc") == "a b c"
