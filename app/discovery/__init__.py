"""External academic discovery (RAG-010).

Discovery finds material that is *not* in the local corpus. It is deliberately
separated from retrieval in three ways:

* it never writes to the corpus, the Qdrant index, the embeddings or the manifests;
* external records are :class:`~app.discovery.models.ExternalWork` objects carrying an
  explicit provenance, confidence and verification status - never evidence;
* provider failures are reported as provider failures, never as "no results".

Typical use:

```python
from app.discovery import DiscoveryService

service = DiscoveryService()
result = service.discover("AI governance in higher education", max_results=10)
for record in result.records:
    print(record.display_title, record.year, record.local_match.kind)
```
"""

from __future__ import annotations

from app.discovery.cache import DiscoveryCache, normalize_query
from app.discovery.local import (
    LocalCorpusIndex,
    LocalDocument,
    build_index,
    local_documents_from_payloads,
    payloads_from_collection,
)
from app.discovery.models import (
    CombinedDiscovery,
    DiscoveryAudit,
    DiscoveryFilters,
    DiscoveryResult,
    DiscoveryStatus,
    ExternalWork,
    FieldConflict,
    FulltextStatus,
    LocalDiscoveryResult,
    LocalMatch,
    LocalMatchKind,
    ProviderFailure,
    ProviderQuery,
)
from app.discovery.render import (
    format_audit,
    format_combined,
    format_record,
    format_result,
    status_counts,
)
from app.discovery.service import DEFAULT_PROVIDERS, DiscoveryService


def __getattr__(name: str):
    """Resolve :func:`research_discover` lazily.

    The combined service lives in :mod:`app.discovery.research`, which also provides the
    ``python -m app.discovery.research`` entry point. Importing it eagerly here would make
    ``-m`` execute an already-imported module, so it is resolved on first use instead.
    """
    if name == "research_discover":
        from app.discovery.research import research_discover

        return research_discover
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "CombinedDiscovery",
    "DEFAULT_PROVIDERS",
    "DiscoveryAudit",
    "DiscoveryCache",
    "DiscoveryFilters",
    "DiscoveryResult",
    "DiscoveryService",
    "DiscoveryStatus",
    "ExternalWork",
    "FieldConflict",
    "FulltextStatus",
    "LocalCorpusIndex",
    "LocalDiscoveryResult",
    "LocalDocument",
    "LocalMatch",
    "LocalMatchKind",
    "ProviderFailure",
    "ProviderQuery",
    "build_index",
    "format_audit",
    "format_combined",
    "format_record",
    "format_result",
    "local_documents_from_payloads",
    "normalize_query",
    "payloads_from_collection",
    "research_discover",
    "status_counts",
]
