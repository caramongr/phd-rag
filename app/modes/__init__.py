"""Academic working modes (RAG-013).

Five presets over the pipeline that already exists — EVIDENCE, WRITING, CONSISTENCY,
SUPERVISOR and RESEARCH — each one an orchestration decision, not a new pipeline:

```python
from app.modes import ModeService

with ModeService() as modes:
    result = modes.run_mode("evidence", "What is AI governance in higher education?")
    print(result.validated_answer)
    print("\\n".join(result.references))
```

EVIDENCE and WRITING generate, validate and cite (`app/modes/service.py` calls RAG-011 and
RAG-012). CONSISTENCY compares the author's own work with the literature deterministically.
SUPERVISOR surfaces supervision material as labelled excerpts. RESEARCH runs local retrieval
and Internet discovery and reports them side by side, never merged.

The rules no mode may break are recorded in every result: only evidence-eligible literature
supports scholarly claims, the author's own work and supervision/guidance material are never
cited, metadata-only Internet records are never evidence, and a claim RAG-012 could not
verify is withheld rather than published.
"""

from __future__ import annotations

from app.modes.consistency import (
    BUILTIN_PREFERENCE_SOURCE,
    ComparisonPolicy,
    compare,
    finding_counts,
    render_report,
)
from app.modes.factory import build_mode_service
from app.modes.models import (
    CONSISTENCY_CONFIG,
    EM_DASH,
    EVIDENCE_CONFIG,
    LONG_SENTENCE_WORDS,
    MODES,
    PROMOTIONAL_TERMS,
    RESEARCH_CONFIG,
    SUPERVISOR_CONFIG,
    WRITING_CONFIG,
    WRITING_PREFERENCES,
    ConsistencyFinding,
    ConsistencyFindingKind,
    ExternalSourceView,
    ModeConfig,
    ModeName,
    ModeResult,
    ModeSection,
    ModeSource,
    ModeStatus,
    OutputStyle,
    SourceRole,
    mode_config,
)
from app.modes.service import POLICY_NOTE, ModeService

__all__ = [
    "BUILTIN_PREFERENCE_SOURCE",
    "CONSISTENCY_CONFIG",
    "ComparisonPolicy",
    "ConsistencyFinding",
    "ConsistencyFindingKind",
    "EM_DASH",
    "EVIDENCE_CONFIG",
    "ExternalSourceView",
    "LONG_SENTENCE_WORDS",
    "MODES",
    "ModeConfig",
    "ModeName",
    "ModeResult",
    "ModeSection",
    "ModeService",
    "ModeSource",
    "ModeStatus",
    "OutputStyle",
    "POLICY_NOTE",
    "PROMOTIONAL_TERMS",
    "RESEARCH_CONFIG",
    "SUPERVISOR_CONFIG",
    "SourceRole",
    "WRITING_CONFIG",
    "WRITING_PREFERENCES",
    "build_mode_service",
    "compare",
    "finding_counts",
    "mode_config",
    "render_report",
]
