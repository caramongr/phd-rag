"""Grounded answer generation (RAG-011).

Generation answers a question from the RAG-009 evidence pipeline using a local model
through Ollama, and returns the answer together with everything needed to check it:
the labelled context, a ``source_map`` from ``S<n>`` to chunk metadata, the marker and
claim audit, and the retrieval audit.

It does not import anything, does not generate prose about itself, and never lets
external Internet metadata stand in for evidence.
"""

from __future__ import annotations

from app.generation.context import (
    ContextBuildRequest,
    ExternalSource,
    build_context,
    estimate_tokens,
    external_sources_from_records,
)
from app.generation.grounding import (
    audit_markers,
    extract_markers,
    grounding_warnings,
    split_sentences,
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
    QuotationEvidence,
    SourceBlock,
    SourceClass,
    SourceMarkerAudit,
)
from app.generation.ollama import (
    ChatMessage,
    ChatResult,
    GenerationModel,
    GenerationOptions,
    OllamaClient,
    model_summary,
)
from app.generation.prompts import build_system_prompt, build_user_prompt, prompt_version
from app.generation.service import GenerationService

__all__ = [
    "ChatMessage",
    "ChatResult",
    "ContextBuildRequest",
    "ContextPackage",
    "DroppedBlock",
    "ExternalSource",
    "GenerationAudit",
    "GenerationMode",
    "GenerationModel",
    "GenerationOptions",
    "GenerationResponse",
    "GenerationService",
    "GenerationStatus",
    "ModelFailure",
    "OllamaClient",
    "OutputLength",
    "QuotationEvidence",
    "SourceBlock",
    "SourceClass",
    "SourceMarkerAudit",
    "audit_markers",
    "build_context",
    "build_system_prompt",
    "build_user_prompt",
    "estimate_tokens",
    "external_sources_from_records",
    "extract_markers",
    "grounding_warnings",
    "model_summary",
    "prompt_version",
    "split_sentences",
]
