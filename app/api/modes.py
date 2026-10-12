"""The mode API (RAG-014 §14).

Two endpoints, and no mode logic:

* ``GET /api/modes`` — what the five modes are, in the words the UI shows. The labels and
  descriptions live here (one source of truth) so the frontend never restates them.
* ``POST /api/mode`` — run one mode through :class:`~app.web.service.ModeRunner`, which runs
  the existing RAG-013 :class:`~app.modes.service.ModeService`.

The response body is the RAG-013 result, serialised as it is: ``result.to_json()`` is the
ModeResult, so a client that knows RAG-013 needs no second schema. Exactly one extra
top-level key may appear — ``notice`` — which is transport only (a friendly sentence the UI
shows above the result, produced by :mod:`app.api.errors`). It is absent when there is
nothing to say, and carries no result data.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator, model_validator

from app.api.errors import ModeNotice, describe_result
from app.generation.models import OutputLength
from app.modes.models import MODES, ModeName
from app.web.service import ModeRunner

router = APIRouter(prefix="/api", tags=["modes"])


#: UI copy (§2, §5). One entry per mode; a test asserts every mode has one.
MODE_LABELS: dict[ModeName, str] = {
    ModeName.EVIDENCE: "Evidence",
    ModeName.WRITING: "Writing",
    ModeName.CONSISTENCY: "Consistency",
    ModeName.SUPERVISOR: "Supervisor",
    ModeName.RESEARCH: "Research",
}

MODE_DESCRIPTIONS: dict[ModeName, str] = {
    ModeName.EVIDENCE: "Answers from scholarly literature only.",
    ModeName.WRITING: (
        "Drafts academic prose using literature as evidence and your own work/rules as "
        "context."
    ),
    ModeName.CONSISTENCY: "Compares your existing work with literature and writing rules.",
    ModeName.SUPERVISOR: "Searches supervision and guidance material.",
    ModeName.RESEARCH: "Searches your local corpus plus external academic sources.",
}

#: The only advanced controls the UI offers (§10), and which modes they affect.
ADVANCED_OPTIONS: dict[str, dict[str, Any]] = {
    "length": {
        "label": "Answer length",
        "type": "select",
        "choices": [length.value for length in OutputLength],
        "default": OutputLength.NORMAL.value,
        "modes": [ModeName.EVIDENCE.value, ModeName.WRITING.value],
    },
    "candidate_k": {
        "label": "Candidates to search (candidate_k)",
        "type": "number",
        "min": 1,
        "max": 200,
        "default": None,
        "modes": [mode.value for mode in ModeName],
    },
    "final_k": {
        "label": "Chunks to consider (final_k)",
        "type": "number",
        "min": 1,
        "max": 50,
        "default": None,
        "modes": [mode.value for mode in ModeName],
    },
    "rerank": {
        "label": "Rerank results (cross-encoder)",
        "type": "boolean",
        "default": True,
        "modes": [mode.value for mode in ModeName],
    },
    "year_min": {
        "label": "Earliest year (Research)",
        "type": "number",
        "min": 1500,
        "max": 3000,
        "default": None,
        "modes": [ModeName.RESEARCH.value],
    },
    "year_max": {
        "label": "Latest year (Research)",
        "type": "number",
        "min": 1500,
        "max": 3000,
        "default": None,
        "modes": [ModeName.RESEARCH.value],
    },
}

#: The largest question the UI accepts. A paragraph or a thesis section, not a document.
MAX_QUERY_CHARS = 20_000


class ModeOptions(BaseModel):
    """The advanced controls, all optional (§10)."""

    model_config = {"extra": "forbid"}

    length: OutputLength = OutputLength.NORMAL
    candidate_k: int | None = Field(default=None, ge=1, le=200)
    final_k: int | None = Field(default=None, ge=1, le=50)
    rerank: bool = True
    year_min: int | None = Field(default=None, ge=1500, le=3000)
    year_max: int | None = Field(default=None, ge=1500, le=3000)

    @model_validator(mode="after")
    def _check_years(self) -> "ModeOptions":
        if (
            self.year_min is not None
            and self.year_max is not None
            and self.year_min > self.year_max
        ):
            raise ValueError("year_min must not be later than year_max")
        return self

    def run_kwargs(self) -> dict[str, Any]:
        """The keyword arguments for ``ModeService.run_mode`` (only what was set)."""
        kwargs: dict[str, Any] = {"rerank": self.rerank, "output_length": self.length}
        if self.candidate_k is not None:
            kwargs["candidate_k"] = self.candidate_k
        if self.final_k is not None:
            kwargs["final_k"] = self.final_k
        if self.year_min is not None:
            kwargs["year_min"] = self.year_min
        if self.year_max is not None:
            kwargs["year_max"] = self.year_max
        return kwargs


class ModeRequest(BaseModel):
    """``{"mode": "...", "query": "...", "options": {}}`` (§14)."""

    model_config = {"extra": "forbid"}

    mode: ModeName
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    options: ModeOptions = Field(default_factory=ModeOptions)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """A question, a paragraph or a thesis section — but not whitespace."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("the query is empty")
        return stripped


def get_runner(request: Request) -> ModeRunner:
    """The app's runner, created with the application."""
    runner = getattr(request.app.state, "mode_runner", None)
    if runner is None:  # pragma: no cover - defensive: create_app always sets it
        raise HTTPException(status_code=503, detail="the mode service is not configured")
    return runner


@router.get("/modes")
def list_modes() -> dict[str, Any]:
    """The modes, their UI copy, and which advanced options apply to each."""
    modes: list[dict[str, Any]] = []
    for mode, config in MODES.items():
        modes.append(
            {
                "name": mode.value,
                "label": MODE_LABELS[mode],
                "description": MODE_DESCRIPTIONS[mode],
                "purpose": config.purpose,
                "retrieval_mode": config.retrieval_mode.value,
                "output_style": config.output_style.value,
                "generates": config.run_generation,
                "validates_citations": config.run_citation_validation,
                "uses_external_discovery": config.use_external_discovery,
                "options": [
                    option for option, spec in ADVANCED_OPTIONS.items() if mode.value in spec["modes"]
                ],
            }
        )
    return {
        "modes": modes,
        "advanced_options": ADVANCED_OPTIONS,
        "max_query_chars": MAX_QUERY_CHARS,
    }


@router.post("/mode")
def run_mode(request: ModeRequest, runner: ModeRunner = Depends(get_runner)) -> dict[str, Any]:
    """Run one mode and return the RAG-013 result (plus a transport-only notice)."""
    result = runner.run(
        request.mode.value, request.query, options=request.options.run_kwargs()
    )
    payload: dict[str, Any] = result.to_json()
    notice: ModeNotice | None = describe_result(result)
    if notice is not None:
        payload["notice"] = notice.model_dump(mode="json")
    return payload


__all__ = [
    "ADVANCED_OPTIONS",
    "MAX_QUERY_CHARS",
    "MODE_DESCRIPTIONS",
    "MODE_LABELS",
    "ModeOptions",
    "ModeRequest",
    "get_runner",
    "list_modes",
    "router",
    "run_mode",
]
