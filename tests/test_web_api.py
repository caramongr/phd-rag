"""The mode API and the home page (RAG-014 §19).

Every test here goes through the real RAG-013 mode service wired to the RAG-013 test doubles,
so the JSON the API returns is the pipeline's own result — including its safety fields.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.api.modes import MODE_DESCRIPTIONS, MODE_LABELS
from app.generation.models import ModelFailure
from app.modes.models import MODES, ModeName, ModeResult
from tests.modes_support import external_record
from tests.web_support import build_client, build_web_app, static_files

#: The copy the brief (§5) requires the UI to show, verbatim.
REQUIRED_DESCRIPTIONS = {
    "evidence": "Answers from scholarly literature only.",
    "writing": (
        "Drafts academic prose using literature as evidence and your own work/rules as "
        "context."
    ),
    "consistency": "Compares your existing work with literature and writing rules.",
    "supervisor": "Searches supervision and guidance material.",
    "research": "Searches your local corpus plus external academic sources.",
}


# --- the page ---------------------------------------------------------------


def test_home_page_loads() -> None:
    response = build_client().get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert 'id="mode"' in body
    assert 'id="query"' in body
    assert 'id="run"' in body


def test_home_page_loads_its_frontend_files() -> None:
    client = build_client()
    script = client.get("/app.js")
    style = client.get("/style.css")
    assert script.status_code == 200 and "javascript" in script.headers["content-type"]
    assert style.status_code == 200 and "css" in style.headers["content-type"]
    assert "app.js" in client.get("/").text


def test_the_page_asks_the_api_for_the_modes() -> None:
    """The selector is filled from the API, so the copy has one source of truth."""
    script = static_files()["app.js"]
    assert 'requestJSON("/api/modes")' in script
    assert 'requestJSON("/api/mode"' in script


# --- the mode list ----------------------------------------------------------


def test_all_five_modes_are_listed() -> None:
    payload = build_client().get("/api/modes").json()
    names = [mode["name"] for mode in payload["modes"]]
    assert names == [mode.value for mode in ModeName]
    assert len(names) == 5


def test_every_mode_carries_the_required_description() -> None:
    payload = build_client().get("/api/modes").json()
    descriptions = {mode["name"]: mode["description"] for mode in payload["modes"]}
    assert descriptions == REQUIRED_DESCRIPTIONS
    labels = {mode["name"]: mode["label"] for mode in payload["modes"]}
    assert labels == {
        "evidence": "Evidence",
        "writing": "Writing",
        "consistency": "Consistency",
        "supervisor": "Supervisor",
        "research": "Research",
    }


def test_every_mode_has_ui_copy_in_the_backend() -> None:
    """A new mode cannot be added without copy for the UI."""
    for mode in MODES:
        assert mode in MODE_LABELS and mode in MODE_DESCRIPTIONS


def test_mode_list_exposes_only_the_documented_advanced_options() -> None:
    payload = build_client().get("/api/modes").json()
    assert set(payload["advanced_options"]) == {
        "length",
        "candidate_k",
        "final_k",
        "rerank",
        "year_min",
        "year_max",
    }
    by_mode = {mode["name"]: mode["options"] for mode in payload["modes"]}
    assert "length" in by_mode["evidence"] and "length" in by_mode["writing"]
    assert "length" not in by_mode["research"]
    assert "year_min" in by_mode["research"] and "year_min" not in by_mode["consistency"]


# --- running a mode ---------------------------------------------------------


def test_post_mode_calls_the_mode_service() -> None:
    client, harness, runner, factory = build_web_app()
    response = client.post(
        "/api/mode", json={"mode": "evidence", "query": "what is ai governance"}
    )
    assert response.status_code == 200
    assert runner.build_count == 1
    assert len(factory.calls) == 1
    assert harness.retrieval.calls[0]["query"] == "what is ai governance"
    assert harness.model.calls  # the real generation path ran


def test_post_mode_runs_the_second_mode_too() -> None:
    client, harness, runner, _ = build_web_app()
    client.post("/api/mode", json={"mode": "evidence", "query": "one"})
    client.post("/api/mode", json={"mode": "writing", "query": "two"})
    assert runner.build_count == 1  # one pipeline for the whole process
    assert harness.retrieval.modes_called()[:2] == [
        "RetrievalMode.EVIDENCE",
        "RetrievalMode.EVIDENCE",
    ]


def test_post_mode_passes_the_advanced_options_through() -> None:
    client, harness, _, _ = build_web_app()
    response = client.post(
        "/api/mode",
        json={
            "mode": "evidence",
            "query": "ai governance",
            "options": {
                "length": "concise",
                "candidate_k": 12,
                "final_k": 4,
                "rerank": False,
                "year_min": 2015,
                "year_max": 2025,
            },
        },
    )
    assert response.status_code == 200
    call = harness.retrieval.calls[0]
    assert call["candidate_k"] == 12
    assert call["final_k"] == 4
    assert call["rerank"] is False
    # The answer length reaches the model as an instruction in the prompt.
    prompt = " ".join(
        message.content for message in harness.model.calls[0]["messages"]
    )
    assert "3 to 5 sentences" in prompt


def test_post_mode_response_is_the_mode_result_plus_only_a_notice() -> None:
    client = build_client()
    payload = client.post(
        "/api/mode", json={"mode": "evidence", "query": "ai governance"}
    ).json()
    expected = set(ModeResult.model_fields)
    assert expected <= set(payload)
    assert set(payload) - expected <= {"notice"}
    assert payload["mode"] == "evidence"
    assert payload["query"] == "ai governance"
    assert payload["config"]["name"] == "evidence"


# --- what each mode returns -------------------------------------------------


def test_evidence_response_renders_references() -> None:
    payload = build_client().post(
        "/api/mode", json={"mode": "evidence", "query": "what is ai governance"}
    ).json()
    assert payload["status"] == "ok"
    assert "(Chan, 2024)" in payload["validated_answer"]
    assert payload["references"]
    assert "Chan, C.K.Y. (2024)" in payload["references"][0]
    assert payload["validation_summary"]["supported"] >= 1
    assert payload["claim_validations"]
    assert payload["claim_validations"][0]["harvard_citation"]
    assert "[S1]" not in payload["validated_answer"]
    assert "notice" not in payload  # nothing to warn about


def test_writing_response_preserves_citations_and_non_citable_context() -> None:
    payload = build_client().post(
        "/api/mode", json={"mode": "writing", "query": "write about ai governance"}
    ).json()
    assert "(Chan, 2024)" in payload["validated_answer"]
    non_citable = [source for source in payload["local_sources"] if not source["role"] == "evidence"]
    assert non_citable  # the draft and the rules are present ...
    cited = {entry["source_id"] for entry in payload["citation_map"]}
    assert not (cited & {source["source_id"] for source in non_citable})
    assert any("non-citable" in warning for warning in payload["warnings"])


def test_research_response_keeps_local_and_external_separate() -> None:
    client = build_client(
        discovery_records=[
            external_record(title="New Paper", doi="10.5555/new"),
            external_record(title="Have It", doi="10.5555/old", already_in_corpus=True),
        ]
    )
    payload = client.post(
        "/api/mode", json={"mode": "research", "query": "ai governance in universities"}
    ).json()
    names = [section["name"] for section in payload["sections"]]
    assert names == [
        "LOCAL EVIDENCE",
        "NEW EXTERNAL PAPERS",
        "ALREADY IN CORPUS",
        "POSSIBLE MATCHES",
        "CONFLICTS",
    ]
    local_section = payload["sections"][0]
    assert local_section["external_indices"] == []
    assert all(
        section["local_indices"] == [] for section in payload["sections"][1:]
    )
    assert all(record["usable_as_evidence"] is False for record in payload["external_sources"])
    groups = {record["title"]: record["group"] for record in payload["external_sources"]}
    assert groups["New Paper"] == "new_external"
    assert groups["Have It"] == "already_in_corpus"
    assert payload["validated_answer"] == ""  # nothing is generated in this mode


def test_research_external_records_carry_the_card_fields() -> None:
    client = build_client(discovery_records=[external_record()])
    payload = client.post(
        "/api/mode", json={"mode": "research", "query": "ai governance"}
    ).json()
    record = payload["external_sources"][0]
    for field in (
        "title",
        "authors",
        "year",
        "venue",
        "doi",
        "providers",
        "discovery_status",
        "open_access",
        "fulltext_status",
        "abstract",
        "already_in_corpus",
    ):
        assert field in record, field


def test_consistency_response_carries_findings_by_kind() -> None:
    payload = build_client().post(
        "/api/mode", json={"mode": "consistency", "query": "how do I describe ai governance?"}
    ).json()
    assert payload["findings"]
    kinds = {finding["kind"] for finding in payload["findings"]}
    assert kinds <= {
        "terminology",
        "duplication",
        "unsupported_claim",
        "numeric_discrepancy",
        "style",
    }
    finding = payload["findings"][0]
    assert finding["summary"] and finding["method"]
    assert payload["references"] == []
    assert payload["validation_summary"] is None


def test_supervisor_response_labels_the_material() -> None:
    payload = build_client().post(
        "/api/mode", json={"mode": "supervisor", "query": "methodology guidance"}
    ).json()
    assert "[supervision]" in payload["answer"]
    assert all(source["role"] != "evidence" for source in payload["local_sources"])
    assert all(
        source["corpus_role"] in {"supervision", "rules", "guidance", "summaries", "future_chapters"}
        for source in payload["local_sources"]
    )
    assert any("never cited" in warning for warning in payload["warnings"])
    assert payload["references"] == []


# --- failures are friendly --------------------------------------------------


def test_unknown_mode_is_rejected_with_a_readable_error() -> None:
    response = build_client().post("/api/mode", json={"mode": "psychic", "query": "hi"})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["kind"] == "invalid_request"
    assert "mode" in error["detail"]


def test_empty_query_is_rejected_with_a_readable_error() -> None:
    response = build_client().post("/api/mode", json={"mode": "evidence", "query": "   "})
    assert response.status_code == 422
    assert response.json()["error"]["kind"] == "invalid_request"


def test_a_backwards_year_range_is_rejected() -> None:
    response = build_client().post(
        "/api/mode",
        json={
            "mode": "research",
            "query": "ai governance",
            "options": {"year_min": 2025, "year_max": 2015},
        },
    )
    assert response.status_code == 422
    assert "year_min" in response.json()["error"]["detail"]
    assert "later than" in response.json()["error"]["detail"]


def test_a_model_failure_becomes_a_friendly_notice() -> None:
    client, harness, _, _ = build_web_app()
    harness.model.failure = ModelFailure(
        kind="model_not_found",
        message="model 'qwen3.5:35b' not found, try pulling it first",
        provider="ollama",
        endpoint="/api/chat",
    )
    payload = client.post("/api/mode", json={"mode": "evidence", "query": "ai"}).json()
    assert payload["status"] == "model_error"
    assert payload["validated_answer"] == ""
    notice = payload["notice"]
    assert notice["level"] == "error"
    assert "not installed" in notice["title"]
    assert notice["debug"]["status"] == "model_error"


def test_no_results_is_reported_as_a_warning_not_an_error() -> None:
    client = build_client(evidence=[], discovery_records=())
    response = client.post("/api/mode", json={"mode": "evidence", "query": "nothing at all"})
    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] in {"no_results", "no_citable_context"}
    assert payload["notice"]["level"] == "warning"
    assert payload["notice"]["title"]


def test_a_provider_outage_is_explained_in_the_notice() -> None:
    client = build_client(discovery_available=False)
    payload = client.post(
        "/api/mode", json={"mode": "research", "query": "ai governance"}
    ).json()
    assert payload["status"] == "ok"  # local results are never lost
    assert payload["notice"]["level"] == "warning"
    assert "external providers" in payload["notice"]["title"]


def test_a_locked_index_is_explained_in_the_notice() -> None:
    client = build_client(evidence=[])
    payload = client.post(
        "/api/mode", json={"mode": "evidence", "query": "ai"}
    ).json()
    assert payload["status"] in {"no_results", "no_citable_context"}
    assert payload["notice"]["title"]


# --- escaping and payload fidelity -----------------------------------------


def test_user_html_arrives_as_data_and_is_never_markup() -> None:
    """The API echoes the text unchanged; the frontend writes it with textContent."""
    hostile = '<script>alert("x")</script> & <img src=x onerror=alert(1)>'
    payload = build_client().post(
        "/api/mode", json={"mode": "consistency", "query": hostile}
    ).json()
    assert payload["query"] == hostile

    script = static_files()["app.js"]
    assert "innerHTML" not in script
    assert "textContent" in script
    assert "insertAdjacentHTML" not in script
    assert "document.write" not in script


def test_the_page_never_renders_document_html_as_markup() -> None:
    page = static_files()["index.html"]
    assert "<script src=" in page
    assert "onerror=" not in page and "onclick=" not in page


def test_json_response_is_serialisable_for_every_mode() -> None:
    client = build_client(discovery_records=[external_record()])
    for mode in ModeName:
        response = client.post("/api/mode", json={"mode": mode.value, "query": "ai governance"})
        assert response.status_code == 200, mode
        json.dumps(response.json())  # raises if anything is not JSON-ready
