"""Grounded generation service (RAG-011 §1, §4, §8, §20).

The pipeline is fixed:

```
question
  -> RAG-009 retrieval (EVIDENCE mode by default, reranked, bibliography guard on)
  -> generation-time evidence selection (optional min_generation_rerank_score)
  -> deterministic context package with labelled source blocks
  -> local Qwen generation through the GenerationModel abstraction
  -> marker/claim/quotation checks against exactly what was supplied
  -> structured response + retrieval audit + generation audit
```

Three rules shape the code:

* **RAG-009 is the only search system.** Evidence comes from
  :class:`~app.retrieval.service.RetrievalService`; nothing here re-implements ranking,
  filtering or the bibliography guard.
* **A model failure is not a retrieval failure.** Ollama being down, slow or missing the
  model produces a ``MODEL_ERROR`` response with the failure recorded; the retrieval
  result stays intact in ``retrieval_audit``.
* **Grounding problems are reported, never repaired.** The answer is returned exactly as
  generated, and every invalid marker, unmarked claim or unverified quotation is listed
  for RAG-012 and for manual review.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from app.config import Settings, get_settings
from app.generation import grounding, prompts
from app.generation.context import (
    ContextBuildRequest,
    ExternalSource,
    build_context,
    estimate_tokens,
)
from app.generation.models import (
    ContextPackage,
    DroppedBlock,
    GenerationAudit,
    GenerationMode,
    GenerationResponse,
    GenerationStatus,
    ModelFailure,
    OutputLength,
    SourceClass,
    SourceMarkerAudit,
)
from app.generation.ollama import (
    ChatMessage,
    ChatResult,
    GenerationModel,
    GenerationOptions,
    OllamaClient,
)
from app.retrieval.models import RetrievalMode, RetrievalResponse

#: Extra system-authored instruction lines per mode (never document text).
MY_WORK_INSTRUCTION = (
    "The my_work blocks are the author's own current writing. Use them for terminology "
    "and continuity, and avoid contradicting them, but never cite them as evidence."
)
GUIDANCE_INSTRUCTION = (
    "The guidance blocks are supervision notes and writing rules. Treat them as "
    "instructions about the project, never as evidence about the world."
)
#: The answer the service returns when there is nothing citable to ground on.
NO_CONTEXT_ANSWER = (
    "The retrieved evidence does not provide enough support to answer this confidently. "
    "No evidence-eligible passages were retrieved for this question."
)


class GenerationService:
    """Answer a question from the local corpus, with traceable sources."""

    def __init__(
        self,
        *,
        retrieval_service: Any | None = None,
        model: GenerationModel | None = None,
        settings: Settings | None = None,
        prompt_directory: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._retrieval = retrieval_service
        self._model = model
        self._prompt_directory = prompt_directory
        self._prompt_version = prompts.prompt_version(prompt_directory)

    # --- components -----------------------------------------------------------
    @property
    def retrieval(self) -> Any:
        """The RAG-009 retrieval service, built on first use."""
        if self._retrieval is None:
            from app.retrieval.factory import build_retrieval_service

            self._retrieval = build_retrieval_service(self.settings)
        return self._retrieval

    @property
    def model(self) -> GenerationModel:
        """The generation model, built on first use."""
        if self._model is None:
            self._model = OllamaClient(
                base_url=self.settings.ollama_base_url,
                model=self.settings.generation_model,
                timeout=self.settings.generation_timeout,
                keep_alive=self.settings.generation_keep_alive,
                think=self.settings.generation_think,
            )
        return self._model

    @property
    def prompt_version(self) -> str:
        """The version of the loaded prompt pair."""
        return self._prompt_version

    def close(self) -> None:
        """Close the retrieval service if this service created it."""
        if self._retrieval is not None:
            close = getattr(self._retrieval, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> "GenerationService":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # --- helpers --------------------------------------------------------------
    def _retrieve(
        self,
        question: str,
        *,
        mode: GenerationMode,
        retrieval_mode: RetrievalMode,
        candidate_k: int | None,
        final_k: int | None,
        rerank: bool,
    ) -> tuple[RetrievalResponse, RetrievalResponse | None, RetrievalResponse | None]:
        """Run the evidence retrieval plus the optional my_work/guidance retrievals."""
        evidence = self.retrieval.retrieve(
            question,
            mode=retrieval_mode,
            candidate_k=candidate_k,
            final_k=final_k,
            rerank=rerank,
        )
        my_work = None
        guidance = None
        if mode.includes_my_work:
            my_work = self.retrieval.retrieve(
                question,
                mode=RetrievalMode.MY_WORK,
                candidate_k=candidate_k,
                final_k=min(3, final_k or 3),
                rerank=rerank,
            )
        if mode.includes_guidance:
            guidance = self.retrieval.retrieve(
                question,
                mode=RetrievalMode.GUIDANCE,
                candidate_k=candidate_k,
                final_k=min(3, final_k or 3),
                rerank=rerank,
            )
        return evidence, my_work, guidance

    @staticmethod
    def _retrieval_audit(
        evidence: RetrievalResponse,
        my_work: RetrievalResponse | None,
        guidance: RetrievalResponse | None,
        *,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The RAG-009 audit, preserved whole so a generation can be reproduced (§4)."""

        def summarise(response: RetrievalResponse) -> dict[str, Any]:
            audit = response.audit
            return {
                "raw_query": audit.raw_query,
                "normalized_query": audit.normalized_query,
                "mode": audit.retrieval_mode.value,
                "candidate_k": audit.candidate_k,
                "final_k": audit.final_k,
                "rerank_enabled": audit.rerank_enabled,
                "rerank_model": audit.rerank_model,
                "rerank_revision": audit.rerank_revision,
                "embedding_model": audit.embedding_model,
                "collection": audit.collection,
                "max_chunks_per_document": audit.max_chunks_per_document,
                "max_candidate_rounds": audit.max_candidate_rounds,
                "suppress_duplicates": audit.suppress_duplicates,
                "exclude_bibliography": audit.exclude_bibliography,
                "min_dense_score": audit.min_dense_score,
                "min_rerank_score": audit.min_rerank_score,
                "timestamp": audit.timestamp,
                "returned": len(response.results),
                "dense_candidates": len(response.dense_candidates),
                "dropped": len(response.dropped),
                "dropped_reasons": _dropped_reasons(response),
                "stats": response.stats.model_dump(),
            }

        audit: dict[str, Any] = {"evidence": summarise(evidence)}
        if my_work is not None:
            audit["my_work"] = summarise(my_work)
        if guidance is not None:
            audit["guidance"] = summarise(guidance)
        if extra:
            audit.update(extra)
        return audit

    # --- API ------------------------------------------------------------------
    def answer(
        self,
        question: str,
        *,
        mode: GenerationMode | str | None = None,
        retrieval_mode: RetrievalMode | str | None = None,
        candidate_k: int | None = None,
        final_k: int | None = None,
        rerank: bool = True,
        generation_model: GenerationModel | None = None,
        include_my_work: bool = False,
        include_guidance: bool = False,
        external_records: Sequence[Any] | None = None,
        external_sources: Sequence[ExternalSource] | None = None,
        output_length: OutputLength | str = OutputLength.NORMAL,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        min_generation_rerank_score: float | None = None,
        context_budget_tokens: int | None = None,
    ) -> GenerationResponse:
        """Answer ``question`` from retrieved evidence, with source markers.

        ``external_records`` accepts RAG-010 results (or anything exposing the same
        attributes) and keeps them separate from corpus evidence: they are labelled,
        metadata-only records are not citable, and if the caller attaches retrieved full
        text to a verified record it is labelled as external content.
        """
        settings = self.settings
        mode = _coerce_mode(mode, include_my_work=include_my_work, include_guidance=include_guidance)
        retrieval_mode = _coerce_retrieval_mode(retrieval_mode)
        length = OutputLength(output_length) if not isinstance(output_length, OutputLength) else output_length
        budget = (
            context_budget_tokens
            if context_budget_tokens is not None
            else settings.generation_context_tokens
        )
        min_score = (
            min_generation_rerank_score
            if min_generation_rerank_score is not None
            else settings.generation_min_rerank_score
        )
        model = generation_model or self.model

        warnings: list[str] = []
        errors: list[str] = []

        retrieval_started = time.perf_counter()
        try:
            evidence, my_work, guidance = self._retrieve(
                question,
                mode=mode,
                retrieval_mode=retrieval_mode,
                candidate_k=candidate_k,
                final_k=final_k,
                rerank=rerank,
            )
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            return GenerationResponse(
                question=question,
                mode=mode,
                output_length=length,
                status=GenerationStatus.MODEL_ERROR,
                errors=(f"retrieval failed: {failure}",),
                retrieval_audit={"evidence": {"error": failure}},
                generation_audit=GenerationAudit(
                    model=model.name,
                    base_url=getattr(model, "base_url", ""),
                    prompt_version=self._prompt_version,
                    failure=ModelFailure(kind="retrieval_error", message=failure),
                ),
            )
        retrieval_seconds = time.perf_counter() - retrieval_started

        sources = tuple(external_sources or ()) + (
            tuple(_adapt_external(record) for record in external_records or ())
        )

        context_started = time.perf_counter()
        context = build_context(
            ContextBuildRequest(
                evidence=list(evidence.results),
                my_work=list(my_work.results) if my_work else [],
                guidance=list(guidance.results) if guidance else [],
                external=sources,
                mode=mode,
                budget_tokens=budget,
                chars_per_token=settings.generation_chars_per_token,
                prompt_version=self._prompt_version,
                min_rerank_score=min_score,
                reserved_non_evidence_tokens=(
                    settings.generation_reserved_non_evidence_tokens
                ),
                retrieval_seconds=round(retrieval_seconds, 4),
            )
        )
        context_seconds = time.perf_counter() - context_started

        retrieval_audit = self._retrieval_audit(
            evidence,
            my_work,
            guidance,
            extra={
                "generation_min_rerank_score": min_score,
                "context_budget_tokens": budget,
                "external_records_supplied": len(sources),
            },
        )

        for dropped in context.dropped:
            if dropped.reason == "below_min_rerank_score":
                warnings.append(
                    "evidence excluded by min_generation_rerank_score: "
                    f"{dropped.title or dropped.chunk_id} (score {dropped.rerank_score})"
                )
            elif dropped.reason in ("budget", "block_too_large"):
                warnings.append(
                    f"evidence excluded by the context budget ({dropped.reason}): "
                    f"{dropped.title or dropped.chunk_id}"
                )

        base = {
            "question": question,
            "mode": mode,
            "output_length": length,
            "source_map": context.source_map(),
            "context": context,
            "retrieval_audit": retrieval_audit,
        }

        # Nothing citable at all: do not ask the model to invent an answer (§14).
        if not context.citable_blocks:
            warnings.append(
                "no citable evidence was available, so no generation was attempted"
            )
            return GenerationResponse(
                **base,
                answer=NO_CONTEXT_ANSWER,
                status=GenerationStatus.NO_CITABLE_CONTEXT,
                unused_source_ids=context.source_ids,
                warnings=tuple(warnings),
                external_discovery_notes=(),
                generation_audit=GenerationAudit(
                    model=model.name,
                    base_url=getattr(model, "base_url", ""),
                    prompt_version=self._prompt_version,
                    temperature=settings.generation_temperature,
                    top_p=settings.generation_top_p,
                    num_ctx=settings.generation_num_ctx,
                    max_output_tokens=_max_output_tokens(max_output_tokens, settings),
                    think=settings.generation_think,
                    keep_alive=settings.generation_keep_alive,
                    retrieval_seconds=round(retrieval_seconds, 4),
                    context_seconds=round(context_seconds, 4),
                    prompt_tokens_estimated=context.context_tokens_estimated,
                ),
            )

        system_prompt = prompts.build_system_prompt(self._prompt_directory)
        user_prompt = prompts.build_user_prompt(
            context=context.rendered,
            question=question,
            length_instruction=length.instruction,
            extra_instructions=_extra_instructions(mode),
            directory=self._prompt_directory,
        )
        options = GenerationOptions(
            temperature=(
                temperature if temperature is not None else settings.generation_temperature
            ),
            top_p=settings.generation_top_p,
            num_ctx=settings.generation_num_ctx,
            max_output_tokens=_max_output_tokens(max_output_tokens, settings, length),
            seed=settings.generation_seed,
        )

        started = time.perf_counter()
        result = model.chat(
            [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_prompt),
            ],
            options=options,
        )
        wall_seconds = time.perf_counter() - started

        generation_audit = _generation_audit(
            result=result,
            model=model,
            settings=settings,
            prompt_version=self._prompt_version,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context=context,
            options=options,
            retrieval_seconds=retrieval_seconds,
            context_seconds=context_seconds,
            wall_seconds=wall_seconds,
        )

        if not result.ok:
            failure = result.failure or ModelFailure(kind="empty_response")
            errors.append(f"{failure.kind}: {failure.message}")
            warnings.append(
                "the answer could not be generated; the retrieved evidence is "
                "unaffected and is reported in retrieval_audit"
            )
            return GenerationResponse(
                **base,
                answer="",
                status=GenerationStatus.MODEL_ERROR,
                unused_source_ids=context.source_ids,
                warnings=tuple(warnings),
                errors=tuple(errors),
                generation_audit=generation_audit,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

        markdown_audit = grounding.audit_markers(
            result.content,
            context,
            external_metadata_only_ids=_metadata_only_ids(context),
        )
        warnings.extend(grounding.grounding_warnings(markdown_audit))
        if result.done_reason == "length":
            warnings.append(
                "the model stopped because it reached the output token limit; the answer "
                "may be incomplete"
            )
        if result.thinking:
            warnings.append(
                "the model returned reasoning text as well as an answer; only the answer "
                "is reported"
            )

        status = GenerationStatus.OK
        if markdown_audit.has_problems:
            status = GenerationStatus.GROUNDING_WARNINGS
        if _insufficient_evidence(answer=result.content):
            status = GenerationStatus.INSUFFICIENT_EVIDENCE

        used = grounding.used_source_ids(result.content, context)
        return GenerationResponse(
            **base,
            answer=result.content,
            status=status,
            used_source_ids=used,
            unused_source_ids=grounding.unused_source_ids(result.content, context),
            marker_audit=markdown_audit,
            unsupported_claim_warnings=grounding.grounding_warnings(markdown_audit),
            external_discovery_notes=grounding.external_discovery_notes(context, result.content),
            warnings=tuple(warnings),
            errors=tuple(errors),
            generation_audit=generation_audit,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    def describe(self) -> dict[str, Any]:
        """Configuration summary for the report and the CLI."""
        model = self.model
        details: dict[str, Any] = {"model": model.name}
        try:
            details.update(model.describe())
        except Exception as exc:  # pragma: no cover - defensive
            details["error"] = f"{type(exc).__name__}: {exc}"
        return {
            "provider": getattr(model, "provider", "unknown"),
            "base_url": getattr(model, "base_url", self.settings.ollama_base_url),
            "model": model.name,
            "model_details": details,
            "prompt_version": self._prompt_version,
            "temperature": self.settings.generation_temperature,
            "top_p": self.settings.generation_top_p,
            "num_ctx": self.settings.generation_num_ctx,
            "max_output_tokens": self.settings.generation_max_output_tokens,
            "context_budget_tokens": self.settings.generation_context_tokens,
            "chars_per_token": self.settings.generation_chars_per_token,
            "min_generation_rerank_score": self.settings.generation_min_rerank_score,
            "think": self.settings.generation_think,
            "seed": self.settings.generation_seed,
            "retrieval": {
                "collection": self.settings.qdrant_collection,
                "candidate_k": self.settings.retrieval_candidate_k,
                "final_k": self.settings.retrieval_final_k,
                "exclude_bibliography": self.settings.retrieval_exclude_bibliography,
            },
        }


# --- small helpers ------------------------------------------------------------


def _coerce_mode(
    mode: GenerationMode | str | None,
    *,
    include_my_work: bool,
    include_guidance: bool,
) -> GenerationMode:
    """Resolve the context mode from either the flag or the two convenience booleans."""
    if include_guidance:
        return GenerationMode.EVIDENCE_PLUS_GUIDANCE
    if include_my_work:
        return GenerationMode.EVIDENCE_PLUS_MY_WORK
    if mode is None:
        return GenerationMode.EVIDENCE_ONLY
    return mode if isinstance(mode, GenerationMode) else GenerationMode(str(mode))


def _coerce_retrieval_mode(mode: RetrievalMode | str | None) -> RetrievalMode:
    if mode is None:
        return RetrievalMode.EVIDENCE
    return mode if isinstance(mode, RetrievalMode) else RetrievalMode(str(mode))


def _extra_instructions(mode: GenerationMode) -> str:
    """Mode-specific instruction lines (system-authored, never document text)."""
    lines: list[str] = []
    if mode.includes_my_work:
        lines.append(MY_WORK_INSTRUCTION)
    if mode.includes_guidance:
        lines.append(GUIDANCE_INSTRUCTION)
    return "\n- ".join(lines)


def _max_output_tokens(
    override: int | None, settings: Settings, length: OutputLength | None = None
) -> int:
    """The output cap: an explicit override, else the configured default."""
    if override is not None:
        return override
    if length is not None:
        return max(settings.generation_max_output_tokens, length.target_tokens)
    return settings.generation_max_output_tokens


def _metadata_only_ids(context: ContextPackage) -> tuple[str, ...]:
    """Source ids that are external records with no retrieved content."""
    return tuple(
        block.source_id
        for block in context.blocks
        if block.source_class.is_metadata_only
    )


def _adapt_external(record: Any) -> ExternalSource:
    """Adapt one external record (RAG-010 ``ExternalWork`` or similar) to a source."""
    from app.generation.context import external_sources_from_records

    sources = external_sources_from_records((record,))
    if not sources:  # pragma: no cover - defensive: the adapter always returns one
        return ExternalSource()
    return sources[0]


def _dropped_reasons(response: RetrievalResponse) -> dict[str, int]:
    """Count RAG-009 drop reasons, so a generation audit shows *why* chunks were lost."""
    counts: dict[str, int] = {}
    for item in response.dropped:
        reason = getattr(item, "reason", None) or "unknown"
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _generation_audit(
    *,
    result: ChatResult,
    model: GenerationModel,
    settings: Settings,
    prompt_version: str,
    system_prompt: str,
    user_prompt: str,
    context: ContextPackage,
    options: GenerationOptions,
    retrieval_seconds: float,
    context_seconds: float,
    wall_seconds: float,
) -> GenerationAudit:
    """Build the audit record, including the prompt hash of what was actually sent."""
    import hashlib

    combined = f"{system_prompt}\n\n{user_prompt}"
    return GenerationAudit(
        provider=getattr(model, "provider", "unknown"),
        model=result.model or model.name,
        base_url=getattr(model, "base_url", ""),
        ollama_version=(result.model_details or {}).get("ollama_version"),
        prompt_version=prompt_version,
        prompt_sha256=hashlib.sha256(combined.encode("utf-8")).hexdigest(),
        system_chars=len(system_prompt),
        context_chars=len(context.rendered),
        question_chars=len(user_prompt),
        temperature=options.temperature,
        top_p=options.top_p,
        num_ctx=options.num_ctx,
        max_output_tokens=options.max_output_tokens,
        think=bool(getattr(model, "think", settings.generation_think)),
        keep_alive=getattr(model, "keep_alive", None),
        prompt_tokens=result.prompt_tokens,
        # The context appears *inside* the user prompt, so the estimate counts the two
        # prompt strings once each instead of adding the context twice.
        prompt_tokens_estimated=estimate_tokens(
            system_prompt, chars_per_token=context.chars_per_token
        )
        + estimate_tokens(user_prompt, chars_per_token=context.chars_per_token),
        generated_tokens=result.generated_tokens,
        prompt_eval_seconds=result.prompt_eval_seconds,
        generation_seconds=result.generation_seconds,
        load_seconds=result.load_seconds,
        total_seconds=result.total_seconds,
        tokens_per_second=result.tokens_per_second,
        done_reason=result.done_reason,
        retrieval_seconds=round(retrieval_seconds, 4),
        context_seconds=round(context_seconds, 4),
        wall_seconds=round(wall_seconds, 4),
        model_details=result.model_details or {},
        failure=result.failure,
    )


def _insufficient_evidence(answer: str) -> bool:
    """Whether the model said the evidence is insufficient rather than answering."""
    lowered = (answer or "").casefold()
    if not lowered.strip():
        return False
    for phrase in grounding.INSUFFICIENT_MARKERS:
        if phrase in lowered:
            return True
    return False
