"""The mode orchestration service (RAG-013 §8).

One entry point, :meth:`ModeService.run_mode`, and one rule: it *calls* the existing
services and never reimplements them.

```
EVIDENCE     GenerationService.answer  -> CitationService.validate
             (RAG-009 retrieval with the bibliography guard, RAG-011 grounding,
              RAG-012 claim validation and Harvard citations)

WRITING      GenerationService.answer(mode=evidence_plus_guidance,
                                      include_my_work, include_guidance)
             -> CitationService.validate
             (literature is evidence; your draft and the rules enter as non-citable
              context, and RAG-012 refuses to cite them)

CONSISTENCY  RAG-009 retrieval of my_work (+ literature, + rules)
             -> app.modes.consistency.compare  (deterministic findings, no model)

SUPERVISOR   RAG-009 retrieval of supervision + guidance
             -> labelled excerpts (no model, no citations)

RESEARCH     app.discovery.research.research_discover
             (RAG-009 local retrieval + RAG-010 discovery, kept separate)
```

Safety is inherited, not re-implemented: the retrieval modes come from RAG-009, the
non-citable labels and grounded generation from RAG-011, and claim validation, quoting and
Harvard citations from RAG-012. This module adds only the preset, the orchestration and the
common result shape.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from app.citations.models import CitationValidationResult
from app.citations.service import CitationService
from app.config import Settings, get_settings
from app.generation.models import (
    GenerationResponse,
    GenerationStatus,
    OutputLength,
    SourceBlock,
)
from app.generation.service import GenerationService
from app.modes.consistency import ComparisonPolicy, compare, finding_counts, render_report
from app.modes.models import (
    ConsistencyFinding,
    ExternalSourceView,
    ModeConfig,
    ModeName,
    ModeResult,
    ModeSection,
    ModeSource,
    ModeStatus,
    SourceRole,
    mode_config,
)
from app.retrieval.models import RetrievalMode, RetrievalResponse

#: Recorded in every result so the safety rules travel with the output.
POLICY_NOTE = (
    "only evidence-eligible literature supports scholarly claims; my_work, supervision, "
    "guidance and rules enter as non-citable context and are never cited; metadata-only "
    "Internet records are shown separately and are never evidence; unsupported claims are "
    "withheld by RAG-012 and never published silently"
)

#: How many characters of a chunk a result shows. The web UI's expanded source card is
#: the reason it is not smaller: this excerpt *is* the retrieved-text view.
EXCERPT_WIDTH = 400
#: How many local chunks the non-generating modes show by default.
DEFAULT_LOCAL_K = 6


def _excerpt(text: str, width: int = EXCERPT_WIDTH) -> str:
    return " ".join((text or "").split())[:width]


def _result_kwargs(common: dict[str, Any], audits: dict[str, Any]) -> dict[str, Any]:
    """The shared result fields, with this run's audit trail replacing the placeholder."""
    payload = dict(common)
    payload["audits"] = audits
    return payload


def _retrieval_mode_for_role(corpus_role: str | None, config: ModeConfig) -> RetrievalMode:
    """Which RAG-009 mode actually produced a block, for the audit trail."""
    from app.retrieval.models import GUIDANCE_ROLES

    if corpus_role == "my_work":
        return RetrievalMode.MY_WORK
    if corpus_role == "supervision":
        return RetrievalMode.SUPERVISION
    if corpus_role in GUIDANCE_ROLES:
        return RetrievalMode.GUIDANCE
    return config.retrieval_mode


def _status_for(generation: GenerationResponse) -> ModeStatus:
    """Map a RAG-011 status onto a mode status."""
    return {
        GenerationStatus.OK: ModeStatus.OK,
        GenerationStatus.GROUNDING_WARNINGS: ModeStatus.OK,
        GenerationStatus.INSUFFICIENT_EVIDENCE: ModeStatus.INSUFFICIENT_EVIDENCE,
        GenerationStatus.NO_CITABLE_CONTEXT: ModeStatus.NO_CITABLE_CONTEXT,
        GenerationStatus.MODEL_ERROR: ModeStatus.MODEL_ERROR,
    }[generation.status]


class ModeService:
    """Run one academic working mode over the existing pipeline."""

    def __init__(
        self,
        *,
        retrieval_service: Any | None = None,
        generation_service: GenerationService | None = None,
        citation_service: CitationService | None = None,
        discovery_service: Any | None = None,
        settings: Settings | None = None,
        comparison_policy: ComparisonPolicy | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._retrieval = retrieval_service
        self._generation = generation_service
        self._citations = citation_service
        self._discovery = discovery_service
        self.comparison_policy = comparison_policy or ComparisonPolicy()

    # --- components -----------------------------------------------------------
    @property
    def retrieval(self) -> Any:
        """The RAG-009 retrieval service, built on first use."""
        if self._retrieval is None:
            from app.retrieval.factory import build_retrieval_service

            self._retrieval = build_retrieval_service(self.settings)
        return self._retrieval

    @property
    def retrieval_service(self) -> Any | None:
        """The RAG-009 service once it exists, or ``None`` (this never builds it).

        Used by callers that want to report on the index (RAG-014 status) without paying
        for model loading, and without competing for the Qdrant client this service owns.
        """
        return self._retrieval

    @property
    def generation(self) -> GenerationService:
        """The RAG-011 generation service, sharing this service's retrieval."""
        if self._generation is None:
            self._generation = GenerationService(
                retrieval_service=self.retrieval, settings=self.settings
            )
        return self._generation

    @property
    def citations(self) -> CitationService:
        """The RAG-012 citation service, over the same generation service."""
        if self._citations is None:
            self._citations = CitationService(generation_service=self.generation)
        return self._citations

    @property
    def discovery(self) -> Any:
        """The RAG-010 discovery service, built on first use."""
        if self._discovery is None:
            from app.discovery.service import DiscoveryService

            self._discovery = DiscoveryService(settings=self.settings)
        return self._discovery

    def close(self) -> None:
        """Close the retrieval and generation services this service created."""
        for component in (self._citations, self._generation):
            close = getattr(component, "close", None)
            if callable(close):
                close()
        if self._retrieval is not None:
            close = getattr(self._retrieval, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> "ModeService":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # --- entry point ----------------------------------------------------------
    def run_mode(
        self,
        mode: ModeName | str,
        query: str,
        *,
        candidate_k: int | None = None,
        final_k: int | None = None,
        rerank: bool = True,
        output_length: OutputLength | str = OutputLength.NORMAL,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        min_generation_rerank_score: float | None = None,
        context_budget_tokens: int | None = None,
        local_k: int = DEFAULT_LOCAL_K,
        external_k: int = 20,
        providers: Sequence[str] | None = None,
        refresh: bool = False,
        year_min: int | None = None,
        year_max: int | None = None,
        external: bool | None = None,
        comparison_policy: ComparisonPolicy | None = None,
    ) -> ModeResult:
        """Run ``mode`` for ``query`` and return the common structured result."""
        config = mode_config(mode)
        policy = comparison_policy or self.comparison_policy
        started = time.perf_counter()
        common: dict[str, Any] = {
            "mode": config.name,
            "query": query,
            "config": config,
            "audits": {
                "mode": config.name.value,
                "purpose": config.purpose,
                "config": config.model_dump(mode="json"),
                "policy_note": POLICY_NOTE,
                "comparison_policy": policy.__dict__.copy(),
                "retrieval": {"modes": [config.retrieval_mode.value]},
            },
        }
        if config.run_generation:
            result = self._run_generation_mode(
                config,
                query,
                candidate_k=candidate_k,
                final_k=final_k,
                rerank=rerank,
                output_length=output_length,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                min_generation_rerank_score=min_generation_rerank_score,
                context_budget_tokens=context_budget_tokens,
                external_k=external_k,
                providers=providers,
                refresh=refresh,
                year_min=year_min,
                year_max=year_max,
                external=external,
                common=common,
            )
        elif config.name is ModeName.CONSISTENCY:
            result = self._run_consistency(
                config,
                query,
                candidate_k=candidate_k,
                final_k=final_k,
                rerank=rerank,
                local_k=local_k,
                policy=policy,
                common=common,
            )
        elif config.name is ModeName.SUPERVISOR:
            result = self._run_supervisor(
                config,
                query,
                candidate_k=candidate_k,
                final_k=final_k,
                rerank=rerank,
                local_k=local_k,
                common=common,
            )
        else:
            result = self._run_research(
                config,
                query,
                candidate_k=candidate_k,
                final_k=final_k,
                local_k=local_k,
                external_k=external_k,
                providers=providers,
                refresh=refresh,
                year_min=year_min,
                year_max=year_max,
                common=common,
            )
        return result.model_copy(
            update={"seconds": round(time.perf_counter() - started, 4)}
        )

    # --- generation modes: EVIDENCE and WRITING --------------------------------
    def _run_generation_mode(
        self,
        config: ModeConfig,
        query: str,
        *,
        candidate_k: int | None,
        final_k: int | None,
        rerank: bool,
        output_length: OutputLength | str,
        temperature: float | None,
        max_output_tokens: int | None,
        min_generation_rerank_score: float | None,
        context_budget_tokens: int | None,
        external_k: int,
        providers: Sequence[str] | None,
        refresh: bool,
        year_min: int | None,
        year_max: int | None,
        external: bool | None,
        common: dict[str, Any],
    ) -> ModeResult:
        """Retrieve, generate, validate and cite (RAG-009 -> RAG-011 -> RAG-012)."""
        warnings: list[str] = []
        external_records: tuple[Any, ...] = ()
        external_views: tuple[ExternalSourceView, ...] = ()
        want_external = (
            config.use_external_discovery if external is None else bool(external)
        )
        if want_external:
            try:
                external_records, external_views = self._discover(
                    query,
                    external_k=external_k,
                    providers=providers,
                    refresh=refresh,
                    year_min=year_min,
                    year_max=year_max,
                )
            except Exception as exc:  # a discovery failure never removes the answer
                warnings.append(f"external discovery failed: {type(exc).__name__}: {exc}")
        try:
            response = self.generation.answer(
                query,
                mode=config.generation_mode,
                retrieval_mode=config.retrieval_mode,
                candidate_k=candidate_k,
                final_k=final_k,
                rerank=rerank,
                include_my_work=config.include_my_work,
                include_guidance=config.include_guidance,
                external_records=external_records or None,
                output_length=output_length,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                min_generation_rerank_score=min_generation_rerank_score,
                context_budget_tokens=context_budget_tokens,
            )
        except Exception as exc:
            return ModeResult(
                status=ModeStatus.MODEL_ERROR,
                answer="",
                warnings=tuple(warnings + [f"generation failed: {type(exc).__name__}: {exc}"]),
                generation_status=GenerationStatus.MODEL_ERROR.value,
                **_result_kwargs(common, common["audits"]),
            )

        citations = self.citations.validate(response)
        local_sources = self._blocks_as_sources(response, config)
        warnings.extend(response.errors)
        warnings.extend(response.warnings)
        warnings.extend(citations.warnings)
        if config.name is ModeName.WRITING:
            warnings.append(
                "your draft and the rules entered the context as non-citable material; "
                "they are never cited"
            )
        audits = dict(common["audits"])
        audits.update(
            {
                "retrieval": response.retrieval_audit,
                "generation": {
                    "model": response.generation_audit.model,
                    "prompt_version": response.generation_audit.prompt_version,
                    "prompt_tokens": response.generation_audit.prompt_tokens,
                    "generated_tokens": response.generation_audit.generated_tokens,
                    "retrieval_seconds": response.generation_audit.retrieval_seconds,
                    "generation_seconds": response.generation_audit.wall_seconds,
                },
                "context": {
                    "included_chunks": response.context.included_chunks,
                    "retrieved_chunks": response.context.retrieved_chunks,
                    "tokens_estimated": response.context.context_tokens_estimated,
                    "mode": response.context.mode.value,
                },
                "citations": {
                    "validator": citations.validator,
                    "policy_note": citations.policy_note,
                },
                "validation_summary": citations.validation_summary.model_dump(mode="json"),
                "withheld": list(citations.unsupported_sentences),
                "marker_audit": response.marker_audit.model_dump(mode="json"),
            }
        )
        return ModeResult(
            status=_status_for(response),
            answer=response.answer,
            validated_answer=citations.validated_answer,
            references=citations.reference_list,
            reference_entries=citations.references,
            citation_map=citations.citation_map,
            validation_summary=citations.validation_summary,
            claim_validations=citations.claim_validations,
            local_sources=local_sources,
            external_sources=external_views,
            sections=self._sections_for_sources(local_sources),
            warnings=tuple(warnings),
            generation_status=response.status.value,
            **_result_kwargs(common, audits),
        )

    # --- CONSISTENCY -----------------------------------------------------------
    def _run_consistency(
        self,
        config: ModeConfig,
        query: str,
        *,
        candidate_k: int | None,
        final_k: int | None,
        rerank: bool,
        local_k: int,
        policy: ComparisonPolicy,
        common: dict[str, Any],
    ) -> ModeResult:
        """Compare the author's own work with the literature and the rules."""
        warnings: list[str] = []
        depth = final_k or local_k
        your_response = self._retrieve(
            query, RetrievalMode.MY_WORK, candidate_k=candidate_k, final_k=depth, rerank=rerank
        )
        literature_response = None
        rules_response = None
        if config.compare_with_literature:
            try:
                literature_response = self._retrieve(
                    query,
                    RetrievalMode.EVIDENCE,
                    candidate_k=candidate_k,
                    final_k=depth,
                    rerank=rerank,
                )
            except Exception as exc:
                warnings.append(f"literature retrieval failed: {type(exc).__name__}: {exc}")
            try:
                rules_response = self._retrieve(
                    query,
                    RetrievalMode.GUIDANCE,
                    candidate_k=candidate_k,
                    final_k=min(3, depth),
                    rerank=rerank,
                )
            except Exception as exc:
                warnings.append(f"rules retrieval failed: {type(exc).__name__}: {exc}")

        your_chunks = list(your_response.results)
        literature = list(literature_response.results) if literature_response else []
        rules = list(rules_response.results) if rules_response else []
        if not literature:
            warnings.append(
                "no literature was retrieved: the unsupported-claim and numeric-discrepancy "
                "checks were skipped, and the comparison rests on your own work only"
            )
        if not rules:
            warnings.append(
                "no rules/guidance chunk was retrieved for this query: the style findings rest "
                "on RAG-013's recorded writing preferences only"
            )

        findings = compare(
            your_chunks, literature=literature, rules=rules, policy=policy
        )
        local_sources = tuple(
            [self._candidate_source(c, config, RetrievalMode.MY_WORK) for c in your_chunks]
            + [self._candidate_source(c, config, RetrievalMode.EVIDENCE) for c in literature]
            + [self._candidate_source(c, config, RetrievalMode.GUIDANCE) for c in rules]
        )
        report = render_report(
            query,
            findings,
            literature_count=len(literature),
            rules_count=len(rules),
            your_count=len(your_chunks),
        )
        audits = dict(common["audits"])
        audits.update(
            {
                "retrieval": {
                    "my_work": self._retrieval_summary(your_response),
                    "literature": self._retrieval_summary(literature_response),
                    "rules": self._retrieval_summary(rules_response),
                },
                "findings": finding_counts(findings),
                "method": "deterministic: content-term containment, figures, writing preferences",
                "no_model_used": True,
            }
        )
        if not your_chunks:
            warnings.append(
                "no chunk of your own work matched this query: nothing was compared"
            )
        return ModeResult(
            status=ModeStatus.OK if (your_chunks or literature) else ModeStatus.NO_RESULTS,
            answer=report,
            local_sources=local_sources,
            sections=self._sections_for_sources(local_sources),
            findings=findings,
            warnings=tuple(warnings),
            **_result_kwargs(common, audits),
        )

    # --- SUPERVISOR ------------------------------------------------------------
    def _run_supervisor(
        self,
        config: ModeConfig,
        query: str,
        *,
        candidate_k: int | None,
        final_k: int | None,
        rerank: bool,
        local_k: int,
        common: dict[str, Any],
    ) -> ModeResult:
        """Surface supervision and guidance material, clearly labelled."""
        warnings: list[str] = [
            "supervision and guidance material is the author's own scaffolding: it is not "
            "scholarly evidence, it is never cited, and it cannot support a claim about the "
            "literature"
        ]
        depth = final_k or local_k
        responses = []
        for retrieval_mode in (RetrievalMode.SUPERVISION, RetrievalMode.GUIDANCE):
            try:
                responses.append(
                    (
                        retrieval_mode,
                        self._retrieve(
                            query,
                            retrieval_mode,
                            candidate_k=candidate_k,
                            final_k=depth,
                            rerank=rerank,
                        ),
                    )
                )
            except Exception as exc:
                warnings.append(
                    f"{retrieval_mode.value} retrieval failed: {type(exc).__name__}: {exc}"
                )
        local_sources = tuple(
            self._candidate_source(candidate, config, retrieval_mode)
            for retrieval_mode, response in responses
            for candidate in response.results
        )
        lines = [
            "SUPERVISION AND GUIDANCE MATERIAL (not scholarly evidence, never cited)",
            f"query: {query}",
            "",
        ]
        if not local_sources:
            lines.append(
                "Nothing matched this query in the supervision or guidance material."
            )
        for index, source in enumerate(local_sources, start=1):
            role = source.corpus_role or "unknown"
            lines.append(
                f"{index}. [{role}] {source.title or '(untitled)'}"
                + (f" | section {source.section_title}" if source.section_title else "")
                + (f" | pages {source.pages}" if source.pages else "")
            )
            lines.append(f"   {source.excerpt}")
        audits = dict(common["audits"])
        audits.update(
            {
                "retrieval": {
                    retrieval_mode.value: self._retrieval_summary(response)
                    for retrieval_mode, response in responses
                },
                "cited": False,
                "no_model_used": True,
            }
        )
        return ModeResult(
            status=ModeStatus.OK if local_sources else ModeStatus.NO_RESULTS,
            answer="\n".join(lines),
            local_sources=local_sources,
            sections=self._sections_for_sources(local_sources),
            warnings=tuple(warnings),
            **_result_kwargs(common, audits),
        )

    # --- RESEARCH --------------------------------------------------------------
    def _run_research(
        self,
        config: ModeConfig,
        query: str,
        *,
        candidate_k: int | None,
        final_k: int | None,
        local_k: int,
        external_k: int,
        providers: Sequence[str] | None,
        refresh: bool,
        year_min: int | None,
        year_max: int | None,
        common: dict[str, Any],
    ) -> ModeResult:
        """Local evidence and external discovery, kept strictly apart (RAG-010)."""
        from app.discovery.render import group_records
        from app.discovery.research import research_discover

        warnings: list[str] = []
        combined = research_discover(
            query,
            local_k=final_k or local_k,
            external_k=external_k,
            retrieval_service=self.retrieval,
            discovery_service=self.discovery,
            mode=config.retrieval_mode,
            candidate_k=candidate_k,
            providers=providers,
            refresh=refresh,
            year_min=year_min,
            year_max=year_max,
        )
        if combined.local.error:
            warnings.append(f"local retrieval failed: {combined.local.error}")
        if not combined.external_available:
            warnings.append(
                "no external provider answered: only the local corpus is shown"
            )
        local_sources = tuple(
            self._chunk_dict_source(chunk, config) for chunk in combined.local.chunks
        )
        already, verified, discovery_only, conflicts = group_records(combined.external.records)
        views: list[ExternalSourceView] = []
        for group_name, records in (
            ("new_external", verified),
            ("already_in_corpus", already),
            ("possible_match", discovery_only),
            ("conflict", conflicts),
        ):
            for record in records:
                views.append(self._external_view(record, group_name))
        sections = (
            ModeSection(
                name="LOCAL EVIDENCE",
                note="ranked chunks from your corpus (RAG-009)",
                local_indices=tuple(range(len(local_sources))),
            ),
            ModeSection(
                name="NEW EXTERNAL PAPERS",
                note="not in your corpus; metadata only, never evidence",
                external_indices=tuple(
                    index
                    for index, view in enumerate(views)
                    if view.group == "new_external"
                ),
            ),
            ModeSection(
                name="ALREADY IN CORPUS",
                note="the corpus already has these works",
                external_indices=tuple(
                    index
                    for index, view in enumerate(views)
                    if view.group == "already_in_corpus"
                ),
            ),
            ModeSection(
                name="POSSIBLE MATCHES",
                note="identity not verified, or the providers disagree",
                external_indices=tuple(
                    index
                    for index, view in enumerate(views)
                    if view.group == "possible_match"
                ),
            ),
            ModeSection(
                name="CONFLICTS",
                note="the providers contradict each other; never silently resolved",
                external_indices=tuple(
                    index for index, view in enumerate(views) if view.group == "conflict"
                ),
            ),
        )
        audits = dict(common["audits"])
        audits.update(
            {
                "retrieval": {
                    "local": {
                        "collection": combined.local.collection,
                        "mode": combined.local.mode,
                        "candidate_k": combined.local.candidate_k,
                        "final_k": combined.local.final_k,
                        "seconds": combined.local.seconds,
                        "error": combined.local.error,
                    }
                },
                "discovery": combined.external.audit.model_dump(mode="json"),
                "separation": "local and external results are reported separately and never merged",
                "no_model_used": True,
            }
        )
        return ModeResult(
            status=ModeStatus.OK
            if (local_sources or combined.external.records)
            else ModeStatus.NO_RESULTS,
            local_sources=local_sources,
            external_sources=tuple(views),
            sections=sections,
            warnings=tuple(warnings),
            **_result_kwargs(common, audits),
        )

    # --- helpers ---------------------------------------------------------------
    def _discover(
        self,
        query: str,
        *,
        external_k: int,
        providers: Sequence[str] | None,
        refresh: bool,
        year_min: int | None,
        year_max: int | None,
    ) -> tuple[tuple[Any, ...], tuple[ExternalSourceView, ...]]:
        """Run RAG-010 discovery and return both the records and their view."""
        result = self.discovery.discover(
            query,
            max_results=external_k,
            providers=providers,
            refresh=refresh,
            year_min=year_min,
            year_max=year_max,
        )
        records = tuple(result.records)
        views = tuple(
            self._external_view(
                record,
                "already_in_corpus"
                if getattr(record.local_match, "already_in_corpus", False)
                else "new_external",
            )
            for record in records
        )
        return records, views

    @staticmethod
    def _external_view(record: Any, group: str) -> ExternalSourceView:
        """One external record as the result presents it (never as evidence)."""
        status = getattr(record, "discovery_status", None)
        local_match = getattr(record, "local_match", None)
        return ExternalSourceView(
            group=group,
            title=getattr(record, "title", None),
            authors=tuple(getattr(record, "authors", ()) or ()),
            year=getattr(record, "year", None),
            doi=getattr(record, "doi", None),
            url=getattr(record, "url", None),
            venue=getattr(record, "venue", None),
            publisher=getattr(record, "publisher", None),
            providers=tuple(getattr(record, "providers", ()) or ()),
            abstract=getattr(record, "abstract", None),
            open_access=getattr(record, "open_access", None),
            fulltext_status=str(
                getattr(getattr(record, "fulltext_status", ""), "value", "")
                or getattr(record, "fulltext_status", "")
                or ""
            ),
            discovery_status=getattr(status, "value", str(status) if status else ""),
            already_in_corpus=bool(getattr(local_match, "already_in_corpus", False)),
            local_match_kind=str(
                getattr(getattr(local_match, "kind", ""), "value", "")
                or getattr(local_match, "kind", "")
                or ""
            ),
            local_evidence_eligible=bool(
                getattr(local_match, "local_evidence_eligible", False)
            ),
            identity_confidence=float(getattr(record, "identity_confidence", 0.0) or 0.0),
            relevance_score=float(getattr(record, "relevance_score", 0.0) or 0.0),
            fulltext_verified=bool(getattr(record, "fulltext_verified", False)),
        )

    def _retrieve(
        self,
        query: str,
        mode: RetrievalMode,
        *,
        candidate_k: int | None,
        final_k: int,
        rerank: bool,
    ) -> RetrievalResponse:
        """One RAG-009 retrieval call (the bibliography guard stays active)."""
        return self.retrieval.retrieve(
            query,
            mode=mode,
            candidate_k=candidate_k,
            final_k=final_k,
            rerank=rerank,
        )

    @staticmethod
    def _retrieval_summary(response: RetrievalResponse | None) -> dict[str, Any]:
        """A compact audit of one retrieval call."""
        if response is None:
            return {"ran": False}
        return {
            "ran": True,
            "mode": response.audit.retrieval_mode.value,
            "candidate_k": response.audit.candidate_k,
            "final_k": response.audit.final_k,
            "rerank_enabled": response.audit.rerank_enabled,
            "suppress_duplicates": response.audit.suppress_duplicates,
            "returned": len(response.results),
            "dropped": len(response.dropped),
            "collection": response.audit.collection,
        }

    def _blocks_as_sources(
        self, response: GenerationResponse, config: ModeConfig
    ) -> tuple[ModeSource, ...]:
        """The context blocks of a generation, with the role each one played."""
        sources: list[ModeSource] = []
        for block in response.context.blocks:
            if block.external or block.source_class.value.startswith("external_"):
                continue  # external records are reported as external, never as local
            sources.append(self._block_source(block, config))
        return tuple(sources)

    @staticmethod
    def _block_source(block: SourceBlock, config: ModeConfig) -> ModeSource:
        """One context block as a mode source."""
        return ModeSource(
            source_id=block.source_id,
            chunk_id=block.chunk_id,
            document_id=block.document_id,
            filename=block.filename,
            title=block.title,
            first_author=block.first_author,
            year=block.year,
            doi=block.doi,
            pages=block.pages,
            section_title=block.section_title,
            corpus_role=block.corpus_role,
            evidence_eligible=block.evidence_eligible,
            ocr_involved=block.ocr_involved,
            content_type=block.content_type,
            dense_score=block.dense_score,
            rerank_score=block.rerank_score,
            role=config.role_for(block.corpus_role),
            retrieval_mode=_retrieval_mode_for_role(block.corpus_role, config).value,
            excerpt=_excerpt(block.text),
        )

    @staticmethod
    def _candidate_source(
        candidate: Any, config: ModeConfig, retrieval_mode: RetrievalMode
    ) -> ModeSource:
        """One RAG-009 candidate as a mode source."""
        return ModeSource(
            source_id=getattr(candidate, "chunk_id", "") or "",
            chunk_id=getattr(candidate, "chunk_id", None),
            document_id=getattr(candidate, "document_id", None),
            filename=getattr(candidate, "filename", None),
            title=getattr(candidate, "title", None),
            first_author=getattr(candidate, "first_author", None),
            year=getattr(candidate, "year", None),
            doi=getattr(candidate, "doi", None),
            pages=candidate.pages if hasattr(candidate, "pages") else None,
            section_title=getattr(candidate, "section_title", None),
            corpus_role=getattr(candidate, "corpus_role", None),
            evidence_eligible=bool(getattr(candidate, "evidence_eligible", False)),
            ocr_involved=bool(getattr(candidate, "ocr_involved", False)),
            content_type=getattr(candidate, "content_type", None),
            dense_score=getattr(candidate, "dense_score", None),
            rerank_score=getattr(candidate, "rerank_score", None),
            role=config.role_for(getattr(candidate, "corpus_role", None)),
            retrieval_mode=retrieval_mode.value,
            excerpt=_excerpt(getattr(candidate, "text", "") or ""),
        )

    @staticmethod
    def _chunk_dict_source(chunk: dict[str, Any], config: ModeConfig) -> ModeSource:
        """One local chunk from RAG-010's combined result as a mode source."""
        return ModeSource(
            source_id=str(chunk.get("chunk_id") or ""),
            chunk_id=chunk.get("chunk_id"),
            document_id=chunk.get("document_id"),
            filename=chunk.get("filename"),
            title=chunk.get("title"),
            first_author=chunk.get("first_author"),
            year=chunk.get("year"),
            doi=chunk.get("doi"),
            pages=chunk.get("pages"),
            section_title=chunk.get("section_title"),
            corpus_role=chunk.get("corpus_role"),
            evidence_eligible=bool(chunk.get("evidence_eligible")),
            ocr_involved=bool(chunk.get("ocr_involved")),
            content_type=chunk.get("content_type"),
            dense_score=chunk.get("dense_score"),
            rerank_score=chunk.get("rerank_score"),
            role=config.role_for(chunk.get("corpus_role")),
            retrieval_mode=config.retrieval_mode.value,
            excerpt=_excerpt(str(chunk.get("text") or chunk.get("excerpt") or "")),
        )

    def _sections_for_sources(
        self, sources: Sequence[ModeSource]
    ) -> tuple[ModeSection, ...]:
        """Group local sources by the role they played, for the report."""
        order = [
            (SourceRole.EVIDENCE, "EVIDENCE (may support scholarly claims)"),
            (SourceRole.SUBJECT, "YOUR OWN WORK (subject of the comparison)"),
            (SourceRole.CONTEXT_ONLY, "CONTEXT ONLY (never cited)"),
            (SourceRole.INSTRUCTIONS_ONLY, "RULES / GUIDANCE (instructions only)"),
            (SourceRole.GUIDANCE_ONLY, "SUPERVISION / GUIDANCE (labelled, never cited)"),
        ]
        sections: list[ModeSection] = []
        for role, name in order:
            indices = tuple(
                index for index, source in enumerate(sources) if source.role is role
            )
            if not indices:
                continue
            sections.append(ModeSection(name=name, local_indices=indices))
        return tuple(sections)


__all__ = [
    "DEFAULT_LOCAL_K",
    "EXCERPT_WIDTH",
    "ModeService",
    "POLICY_NOTE",
]
