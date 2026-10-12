"""Friendly failure reporting for the web API (RAG-014 §16).

The mode service already reports failures honestly and without raising: a missing model, an
unreachable Ollama, a locked Qdrant store, a provider outage and "nothing matched" all arrive
as a status plus warnings. What a *person* needs on top of that is a sentence they can act
on, and what a *developer* needs is the raw detail — so both are produced here, separately.

The mapping is server-side on purpose: it is testable without a browser, it cannot drift
between the CLI and the UI, and the frontend stays a renderer.
"""

from __future__ import annotations

from typing import Any, Mapping

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.modes.models import ModeResult, ModeStatus

#: Notice levels the UI knows how to colour.
LEVEL_INFO = "info"
LEVEL_WARNING = "warning"
LEVEL_ERROR = "error"


class ModeNotice(BaseModel):
    """A transport-only, human-readable summary of what a run needs the user to know.

    This is not part of :class:`~app.modes.models.ModeResult`: it carries no result data,
    only the message the UI shows above it.
    """

    level: str = LEVEL_INFO
    title: str = ""
    detail: str = ""
    #: Raw material for the collapsed debug panel. Never shown by default.
    debug: dict[str, Any] = {}


#: Substring → (title, detail) for the warnings RAG-009/RAG-010/RAG-011 emit.
_WARNING_MESSAGES: tuple[tuple[str, str, str], ...] = (
    (
        "no external provider answered",
        "The external providers did not answer",
        "Crossref and OpenAlex could not be reached, so only your local corpus is shown. "
        "Local results are unaffected.",
    ),
    (
        "local retrieval failed",
        "Local retrieval failed",
        "The corpus could not be searched, so no local sources are shown. Check that the "
        "Qdrant index is available and not locked by another process.",
    ),
    (
        "excluded by the context budget",
        "Some evidence did not fit the context budget",
        "The most relevant sources were used; the ones listed in the warnings were dropped "
        "because the model's context is finite. Raise the context budget in the advanced "
        "options, or ask a narrower question.",
    ),
    (
        "non-citable material",
        "Your own work and the rules were used as context",
        "They are never cited as evidence: only literature can support a claim.",
    ),
)

_STATUS_MESSAGES: dict[ModeStatus, tuple[str, str, str]] = {
    ModeStatus.MODEL_ERROR: (
        LEVEL_ERROR,
        "The model could not answer",
        "Ollama did not return an answer. Check that Ollama is running and that the "
        "configured model is installed, then run again.",
    ),
    ModeStatus.NO_CITABLE_CONTEXT: (
        LEVEL_WARNING,
        "Nothing citable was retrieved",
        "No evidence-eligible literature matched this query, so no answer was generated "
        "rather than an unsupported one. Try different words, or check the index.",
    ),
    ModeStatus.INSUFFICIENT_EVIDENCE: (
        LEVEL_WARNING,
        "The retrieved evidence was not enough to answer",
        "The model reported that the sources it was given do not cover the question. "
        "Broaden the query or widen the candidate pool in the advanced options.",
    ),
    ModeStatus.NO_RESULTS: (
        LEVEL_WARNING,
        "Nothing matched this query",
        "Neither your corpus nor the external providers returned anything. Try fewer or "
        "different words.",
    ),
}

#: Fragments that identify a model-side failure, checked against the raw error text.
_FAILURE_HINTS: tuple[tuple[str, str, str], ...] = (
    (
        "not found",
        "The configured model is not installed",
        "Ollama is reachable but does not have this model. Run `ollama pull <model>`, or "
        "start the UI with a different model.",
    ),
    (
        "could not reach",
        "Ollama is not reachable",
        "Start Ollama (`ollama serve`) and check the base URL, then run again.",
    ),
    (
        "timeout",
        "The model took too long",
        "Generation timed out. A shorter answer length or a smaller context usually helps.",
    ),
    (
        "locked",
        "The index is locked",
        "Another process (a CLI run, or a second UI) holds the local Qdrant store open. "
        "Close it and try again.",
    ),
)


def _search(
    text: str,
    table: tuple[tuple[str, str, str], ...],
    *,
    level: str,
) -> tuple[str, str, str] | None:
    lowered = (text or "").casefold()
    for needle, title, detail in table:
        if needle in lowered:
            return level, title, detail
    return None


def _failure_text(result: ModeResult) -> str:
    """Every message a result carries, flattened, for hint matching."""
    parts: list[str] = list(result.warnings)
    audits = result.audits or {}
    for key in ("failure", "error", "model_error"):
        value = audits.get(key)
        if value:
            parts.append(str(value))
    generation = audits.get("generation")
    if isinstance(generation, Mapping):
        for key in ("failure", "error", "message"):
            if generation.get(key):
                parts.append(str(generation[key]))
    return " \n".join(parts)


def describe_result(result: ModeResult) -> ModeNotice | None:
    """The notice a result deserves, or ``None`` when there is nothing to say."""
    text = _failure_text(result)
    status_note = _STATUS_MESSAGES.get(result.status)
    hint = (
        _search(text, _FAILURE_HINTS, level=LEVEL_ERROR) if text else None
    )

    if result.status is ModeStatus.MODEL_ERROR:
        level, title, detail = hint or status_note or (
            LEVEL_ERROR,
            "The model could not answer",
            "See the debug details for the raw error.",
        )
        return ModeNotice(
            level=level,
            title=title,
            detail=detail,
            debug={"status": result.status.value, "warnings": list(result.warnings)},
        )

    if status_note is not None:
        level, title, detail = status_note
        return ModeNotice(
            level=level,
            title=title,
            detail=detail,
            debug={"status": result.status.value, "warnings": list(result.warnings)},
        )

    if result.unsupported_claims:
        return ModeNotice(
            level=LEVEL_WARNING,
            title=(
                f"{len(result.unsupported_claims)} claim(s) were withheld as unsupported"
            ),
            detail=(
                "The cited source did not support them, so they are not in the validated "
                "answer. They are listed under the warnings."
            ),
            debug={"withheld": list(result.unsupported_claims)},
        )

    if text:
        found = _search(text, _WARNING_MESSAGES, level=LEVEL_WARNING)
        if found is not None:
            level, title, detail = found
            return ModeNotice(
                level=level,
                title=title,
                detail=detail,
                debug={"warnings": list(result.warnings)},
            )
    return None


def validation_error_detail(exc: RequestValidationError) -> str:
    """A one-line, human-readable summary of a rejected request."""
    problems: list[str] = []
    for error in exc.errors()[:5]:
        location = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
        message = str(error.get("msg") or "is not valid")
        problems.append(f"{location or 'request'}: {message}")
    return "; ".join(problems) or "the request could not be read"


def validation_error_fields(exc: RequestValidationError) -> list[dict[str, Any]]:
    """The per-field errors, reduced to JSON-safe data.

    Pydantic puts the raised exception itself in ``ctx`` (for a ``ValueError`` from a
    validator), which is not serialisable — and a debug panel is no use if the response
    cannot be built.
    """
    fields: list[dict[str, Any]] = []
    for error in exc.errors()[:10]:
        fields.append(
            {
                "location": [str(part) for part in error.get("loc", ())],
                "message": str(error.get("msg") or "is not valid"),
                "type": str(error.get("type") or ""),
            }
        )
    return fields


def install_error_handlers(application: FastAPI) -> None:
    """Make every failure leave the API as JSON a person can read.

    An HTML error page in a JSON UI is worse than useless: the frontend cannot show it, and
    the user sees nothing. Invalid requests get the friendly summary plus the field list;
    anything unexpected gets a generic message, with the traceback only when ``debug`` is on.
    """

    @application.exception_handler(RequestValidationError)
    async def _invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "kind": "invalid_request",
                    "title": "That request could not be read",
                    "detail": validation_error_detail(exc),
                    "fields": validation_error_fields(exc),
                }
            },
        )

    @application.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        import traceback

        from app.config import get_settings

        debug = bool(getattr(get_settings(), "debug", False))
        payload: dict[str, Any] = {
            "error": {
                "kind": "internal_error",
                "title": "Something went wrong inside the app",
                "detail": f"{type(exc).__name__}: {exc}"
                if debug
                else "The run failed before it produced a result. See the server log.",
            }
        }
        if debug:
            payload["error"]["traceback"] = traceback.format_exc()
        return JSONResponse(status_code=500, content=payload)


__all__ = [
    "LEVEL_ERROR",
    "LEVEL_INFO",
    "LEVEL_WARNING",
    "ModeNotice",
    "describe_result",
    "install_error_handlers",
    "validation_error_detail",
    "validation_error_fields",
]
