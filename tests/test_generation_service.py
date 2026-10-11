"""Generation-service, audit and determinism tests (RAG-011 §31 items 18-23).

The service is the pipeline itself, so these tests check the things a report has to be
able to claim: the prompt version and model are recorded, the RAG-009 audit survives,
timings are measured, and the same evidence always produces the same context ids.
"""

from __future__ import annotations

from app.generation.models import (
    GenerationMode,
    GenerationStatus,
    ModelFailure,
    OutputLength,
)
from app.generation.service import NO_CONTEXT_ANSWER, GenerationService
from tests.generation_support import (
    FakeGenerationModel,
    FakeRetrievalService,
    build_service,
    candidate,
    external_record,
)


def _service(**kwargs):
    """A service with scripted evidence and a scripted answer."""
    retrieval = FakeRetrievalService(evidence=[candidate("c1"), candidate("c2")])
    model = FakeGenerationModel(content=kwargs.pop("content", "Grounded answer [S1]."))
    return build_service(retrieval=retrieval, model=model, **kwargs), retrieval, model


# --- basic pipeline ----------------------------------------------------------


def test_answer_returns_grounded_text_with_its_sources() -> None:
    """The happy path: retrieved evidence becomes a labelled context and an answer."""
    service, _retrieval, _model = _service()

    response = service.answer("what is ai governance?")

    assert response.status is GenerationStatus.OK
    assert response.answer == "Grounded answer [S1]."
    assert response.source_map["S1"]["chunk_id"] == "c1"
    assert response.used_source_ids == ("S1",)
    assert response.unused_source_ids == ("S2",)
    assert response.mode is GenerationMode.EVIDENCE_ONLY
    assert response.context.included_chunks == 2


def test_max_output_tokens_scales_with_the_requested_length() -> None:
    """Length controls reach the model as ``num_predict``."""
    service, _retrieval, model = _service(generation_max_output_tokens=900)

    service.answer("q", output_length=OutputLength.DETAILED)

    assert model.calls[-1]["options"].max_output_tokens >= 1100


def test_explicit_max_output_tokens_overrides_the_length_control() -> None:
    """An explicit cap wins, so the caller always has the last word."""
    service, _retrieval, model = _service()

    service.answer("q", output_length=OutputLength.DETAILED, max_output_tokens=128)

    assert model.calls[-1]["options"].max_output_tokens == 128


# --- 18-19. prompt version and model recorded -------------------------------


def test_prompt_version_and_model_are_recorded_in_the_audit() -> None:
    """A generation can be reproduced: the exact prompt hash and model are stored."""
    service, _retrieval, _model = _service()

    response = service.answer("what is ai governance?")
    audit = response.generation_audit

    assert audit.prompt_version == service.prompt_version
    assert audit.prompt_version.startswith("system_v2:")
    assert audit.prompt_sha256 and len(audit.prompt_sha256) == 64
    assert audit.model == "fake:qwen"
    assert audit.provider == "ollama"
    assert audit.base_url == "http://127.0.0.1:11434"
    assert audit.temperature == 0.1
    assert audit.num_ctx > 0
    assert audit.think is False
    assert response.system_prompt and response.user_prompt


def test_prompt_hash_changes_when_the_prompt_changes() -> None:
    """The stored hash really identifies the prompt that was sent."""
    service, _retrieval, _model = _service()

    first = service.answer("what is ai governance?").generation_audit.prompt_sha256
    second = service.answer("a different question entirely").generation_audit.prompt_sha256

    assert first != second


def test_model_details_come_from_the_model_not_from_a_constant() -> None:
    """Whatever the server reports about the model is preserved in the audit."""
    service, _retrieval, _model = _service()

    details = service.answer("q").generation_audit.model_details

    assert details["parameter_size"] == "34.7B"
    assert details["quantization_level"] == "Q4_K_M"
    assert details["ollama_version"] == "0.35.1"


# --- 20. retrieval audit preserved ------------------------------------------


def test_retrieval_audit_is_preserved_in_full() -> None:
    """The RAG-009 audit travels with the generation, including drop reasons."""
    from app.retrieval.models import DroppedCandidate

    retrieval = FakeRetrievalService(
        evidence=[candidate("c1")],
        dropped=[
            DroppedCandidate(
                chunk_id="refs-1", document_id="doc-x", reason="bibliography_like"
            ),
            DroppedCandidate(
                chunk_id="refs-2", document_id="doc-y", reason="bibliography_like"
            ),
        ],
    )
    model = FakeGenerationModel(content="Answer [S1].")
    service = build_service(retrieval=retrieval, model=model)

    response = service.answer("q")
    audit = response.retrieval_audit["evidence"]

    assert audit["mode"] == "evidence"
    assert audit["collection"] == "test_collection"
    assert audit["rerank_enabled"] is True
    assert audit["exclude_bibliography"] is True
    assert audit["dropped"] == 2
    assert audit["dropped_reasons"] == {"bibliography_like": 2}
    assert audit["stats"]["total_seconds"] == 0.06
    assert audit["returned"] == 1


def test_generation_does_not_reimplement_retrieval() -> None:
    """Evidence comes from RAG-009 with the documented defaults (§4)."""
    retrieval = FakeRetrievalService(evidence=[candidate("c1")])
    model = FakeGenerationModel(content="Answer [S1].")
    service = build_service(retrieval=retrieval, model=model)

    service.answer("what is ai governance?")

    call = retrieval.calls[0]
    assert call["mode"].value == "evidence"
    assert call["rerank"] is True
    assert call["candidate_k"] is None and call["final_k"] is None  # service defaults


def test_context_mode_requests_the_extra_retrievals_it_needs() -> None:
    """Opting into own work adds a my_work retrieval; guidance adds another."""
    retrieval = FakeRetrievalService(
        evidence=[candidate("c1")], my_work=[candidate("m1", corpus_role="my_work")]
    )
    model = FakeGenerationModel(content="Answer [S1], continuing my draft [S2].")
    service = build_service(retrieval=retrieval, model=model)

    response = service.answer("what have I written?", include_my_work=True)

    modes = [call["mode"].value for call in retrieval.calls]
    assert modes == ["evidence", "my_work"]
    assert response.mode is GenerationMode.EVIDENCE_PLUS_MY_WORK
    assert response.source_map["S2"]["source_class"] == "my_work"


def test_evidence_only_mode_never_adds_own_work_retrieval() -> None:
    """The default mode makes exactly one retrieval call."""
    retrieval = FakeRetrievalService(evidence=[candidate("c1")])
    service = build_service(retrieval=retrieval, model=FakeGenerationModel())

    service.answer("what is ai governance?")

    assert [call["mode"].value for call in retrieval.calls] == ["evidence"]


# --- 21. timings recorded ---------------------------------------------------


def test_timings_and_token_counts_are_measured() -> None:
    """Retrieval, context and generation time are all recorded."""
    service, _retrieval, _model = _service()
    _service_model = _model

    response = service.answer("q")
    audit = response.generation_audit

    assert audit.retrieval_seconds >= 0.0
    assert audit.context_seconds >= 0.0
    assert audit.wall_seconds >= 0.0
    assert audit.generation_seconds == 0.5
    assert audit.generated_tokens == 20
    assert audit.tokens_per_second == 40.0
    assert audit.prompt_tokens == 100
    assert audit.prompt_tokens_estimated > 0
    assert audit.token_estimate_ratio is not None


# --- model failure handling -------------------------------------------------


def test_model_failure_is_reported_without_losing_the_evidence() -> None:
    """Ollama down: the answer is empty, the failure is typed, the evidence survives."""
    retrieval = FakeRetrievalService(evidence=[candidate("c1")])
    model = FakeGenerationModel(
        failure=ModelFailure(kind="connection_error", message="server is not running")
    )
    service = build_service(retrieval=retrieval, model=model)

    response = service.answer("q")

    assert response.status is GenerationStatus.MODEL_ERROR
    assert response.answer == ""
    assert response.errors and "connection_error" in response.errors[0]
    assert response.generation_audit.failure is not None
    assert response.generation_audit.failure.kind == "connection_error"
    assert response.source_map["S1"]["chunk_id"] == "c1"
    assert any("unaffected" in warning for warning in response.warnings)


def test_retrieval_failure_is_reported_without_crashing_generation() -> None:
    """A broken index produces a typed failure, not an exception."""
    retrieval = FakeRetrievalService(error=RuntimeError("index unavailable"))
    service = build_service(retrieval=retrieval, model=FakeGenerationModel())

    response = service.answer("q")

    assert response.status is GenerationStatus.MODEL_ERROR
    assert "index unavailable" in response.errors[0]
    assert response.retrieval_audit["evidence"]["error"].endswith("index unavailable")


def test_no_citable_evidence_skips_generation_entirely() -> None:
    """Nothing to ground on means no model call and an honest statement."""
    retrieval = FakeRetrievalService(evidence=[])
    model = FakeGenerationModel(content="I would have hallucinated [S1].")
    service = build_service(retrieval=retrieval, model=model)

    response = service.answer("what does the literature say about x?")

    assert response.status is GenerationStatus.NO_CITABLE_CONTEXT
    assert response.answer == NO_CONTEXT_ANSWER
    assert "does not provide enough support" in response.answer
    assert model.calls == []  # the model was never asked
    assert response.used_source_ids == ()


def test_insufficient_evidence_answer_is_recognised_as_such() -> None:
    """When the model says the evidence is thin, the status says so."""
    retrieval = FakeRetrievalService(evidence=[candidate("c1")])
    model = FakeGenerationModel(
        content=(
            "The retrieved evidence does not provide enough support to answer this "
            "confidently. The sources cover governance broadly, but not this question [S1]."
        )
    )
    service = build_service(retrieval=retrieval, model=model)

    response = service.answer("how many universities adopted AI policies in 2025?")

    assert response.status is GenerationStatus.INSUFFICIENT_EVIDENCE
    assert response.status.is_answer is True


def test_grounding_warnings_change_the_status_but_keep_the_answer() -> None:
    """A grounded-but-imperfect answer is returned with warnings (§14, §33)."""
    retrieval = FakeRetrievalService(evidence=[candidate("c1")])
    model = FakeGenerationModel(content="A claim with an invented marker [S9].")
    service = build_service(retrieval=retrieval, model=model)

    response = service.answer("q")

    assert response.status is GenerationStatus.GROUNDING_WARNINGS
    assert response.answer == "A claim with an invented marker [S9]."
    assert response.marker_audit.invalid_markers == ("S9",)
    assert any("invalid_source_marker" in warning for warning in response.unsupported_claim_warnings)


def test_truncated_answer_is_flagged() -> None:
    """Hitting the output limit is reported, because the answer may be incomplete."""
    retrieval = FakeRetrievalService(evidence=[candidate("c1")])
    model = FakeGenerationModel(content="Half an answer [S1]", done_reason="length")
    service = build_service(retrieval=retrieval, model=model)

    response = service.answer("q")

    assert any("output token limit" in warning for warning in response.warnings)


def test_reasoning_output_is_reported_but_not_returned_as_the_answer() -> None:
    """A thinking trace is recorded as a warning and kept out of the answer."""
    retrieval = FakeRetrievalService(evidence=[candidate("c1")])
    model = FakeGenerationModel(content="The answer [S1].", thinking="Step 1: think...")
    service = build_service(retrieval=retrieval, model=model)

    response = service.answer("q")

    assert response.answer == "The answer [S1]."
    assert any("reasoning" in warning for warning in response.warnings)


# --- evidence selection (§8) ------------------------------------------------


def test_min_rerank_score_excludes_weak_evidence_and_reports_it() -> None:
    """Weak tail evidence can be excluded before generation, with the drop recorded."""
    retrieval = FakeRetrievalService(
        evidence=[
            candidate("strong", rerank_score=0.88),
            candidate("weak", rerank_score=0.03),
        ]
    )
    model = FakeGenerationModel(content="Answer [S1].")
    service = build_service(retrieval=retrieval, model=model)

    response = service.answer("q", min_generation_rerank_score=0.2)

    assert [block.chunk_id for block in response.context.blocks] == ["strong"]
    assert response.retrieval_audit["generation_min_rerank_score"] == 0.2
    assert any("min_generation_rerank_score" in warning for warning in response.warnings)


def test_budget_drops_are_reported_to_the_caller() -> None:
    """A budget exclusion is visible, never silent (§7)."""
    retrieval = FakeRetrievalService(
        evidence=[candidate(f"c{index}", text="word " * 300) for index in range(6)]
    )
    model = FakeGenerationModel(content="Answer [S1].")
    service = build_service(retrieval=retrieval, model=model)

    response = service.answer("q", context_budget_tokens=250)

    assert response.context.excluded_chunks > 0
    assert any("context budget" in warning for warning in response.warnings)
    assert response.context.context_tokens_estimated <= 250


# --- external records (§5, §28) ---------------------------------------------


def test_external_records_are_labelled_and_kept_out_of_evidence_claims() -> None:
    """Supplied Internet results are context, not evidence, and are reported as such."""
    retrieval = FakeRetrievalService(evidence=[candidate("c1")])
    model = FakeGenerationModel(
        content="The corpus covers governance [S1]. A 2026 paper also exists [S2]."
    )
    service = build_service(retrieval=retrieval, model=model)

    response = service.answer(
        "q", external_records=[external_record(status="EXTERNAL_METADATA_VERIFIED")]
    )

    assert response.source_map["S2"]["source_class"] == "external_metadata_verified"
    assert response.source_map["S2"]["citable"] is False
    assert response.external_discovery_notes and "metadata only" in response.external_discovery_notes[0]
    assert response.retrieval_audit["external_records_supplied"] == 1


def test_verified_external_fulltext_is_citable_but_still_external() -> None:
    """Retrieved external content may be cited, and is reported as external."""
    retrieval = FakeRetrievalService(evidence=[candidate("c1")])
    model = FakeGenerationModel(content="Corpus evidence [S1]. An external survey [S2].")
    service = build_service(retrieval=retrieval, model=model)

    response = service.answer(
        "q",
        external_records=[
            external_record(
                status="EXTERNAL_FULLTEXT_VERIFIED",
                fulltext_verified=True,
                fulltext_text="The retrieved text reports an adoption survey.",
            )
        ],
    )

    assert response.source_map["S2"]["citable"] is True
    assert response.source_map["S2"]["external"] is True
    assert response.source_map["S2"]["fulltext_verified"] is True
    assert response.marker_audit.non_evidence_citations == ()


# --- 22-23. determinism -----------------------------------------------------


def test_same_evidence_produces_the_same_context_ids() -> None:
    """Two identical requests give identical ids, rendering and source maps."""
    first_retrieval = FakeRetrievalService(evidence=[candidate("c1"), candidate("c2")])
    second_retrieval = FakeRetrievalService(evidence=[candidate("c1"), candidate("c2")])

    first = build_service(
        retrieval=first_retrieval, model=FakeGenerationModel(content="A [S1].")
    ).answer("q")
    second = build_service(
        retrieval=second_retrieval, model=FakeGenerationModel(content="A [S1].")
    ).answer("q")

    assert first.context.source_ids == second.context.source_ids
    assert first.context.rendered == second.context.rendered
    assert first.source_map == second.source_map
    assert first.generation_audit.prompt_sha256 == second.generation_audit.prompt_sha256


def test_source_map_keys_are_ordered_and_complete() -> None:
    """The map covers exactly the supplied ids, so RAG-012 can trust it."""
    retrieval = FakeRetrievalService(
        evidence=[candidate("c1"), candidate("c2"), candidate("c3")]
    )
    service = build_service(retrieval=retrieval, model=FakeGenerationModel(content="A [S1]."))

    response = service.answer("q")

    assert list(response.source_map) == ["S1", "S2", "S3"]
    for source_id, metadata in response.source_map.items():
        assert metadata["source_id"] == source_id
        assert "chunk_id" in metadata and "document_id" in metadata


def test_describe_reports_the_configuration_used() -> None:
    """The service can explain itself for the report."""
    service = build_service(
        retrieval=FakeRetrievalService(), model=FakeGenerationModel(), generation_num_ctx=4096
    )

    described = service.describe()

    assert described["model"] == "fake:qwen"
    assert described["num_ctx"] == 4096
    assert described["prompt_version"].startswith("system_v2:")
    assert described["retrieval"]["exclude_bibliography"] is True
    assert described["model_details"]["parameter_size"] == "34.7B"


def test_service_closes_the_retrieval_it_created_and_not_the_one_it_was_given() -> None:
    """Lifecycle: an injected service is the caller's to close."""
    retrieval = FakeRetrievalService(evidence=[candidate("c1")])
    model = FakeGenerationModel()
    service = GenerationService(retrieval_service=retrieval, model=model)

    service.close()

    assert retrieval.calls == []  # closed without being used, and not an error
