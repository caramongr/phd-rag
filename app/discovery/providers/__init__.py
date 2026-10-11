"""Optional provider adapters (RAG-010).

Semantic Scholar is implemented as a *clean optional* adapter: it is never required for
discovery to work, its absence or rate limiting is reported like any other provider
failure, and it is not enabled unless asked for. ``Scite`` is deliberately not
implemented because it requires credentialed access; the adapter boundary is the place
for it when credentials exist.
"""

from __future__ import annotations

from app.discovery.providers.base import AcademicDiscoveryProvider, ProviderResponse
from app.discovery.providers.crossref import CrossrefProvider
from app.discovery.providers.openalex import OpenAlexProvider
from app.discovery.providers.semanticscholar import SemanticScholarProvider

__all__ = [
    "AcademicDiscoveryProvider",
    "CrossrefProvider",
    "OpenAlexProvider",
    "ProviderResponse",
    "SemanticScholarProvider",
    "build_provider",
    "PROVIDER_FACTORIES",
]


def build_provider(
    name: str,
    *,
    http,
    cache=None,
    mailto: str | None = None,
    openalex_api_key: str | None = None,
    openalex_email: str | None = None,
    base_urls: dict[str, str] | None = None,
) -> AcademicDiscoveryProvider:
    """Build one provider by name.

    Unknown providers raise :class:`KeyError` so a typo fails loudly instead of
    silently searching nothing.
    """
    base_urls = base_urls or {}
    key = name.strip().lower()
    if key == "crossref":
        return CrossrefProvider(
            http,
            cache=cache,
            base_url=base_urls.get("crossref", "https://api.crossref.org"),
            mailto=mailto,
        )
    if key == "openalex":
        return OpenAlexProvider(
            http,
            cache=cache,
            base_url=base_urls.get("openalex", "https://api.openalex.org"),
            mailto=openalex_email or mailto,
            api_key=openalex_api_key,
        )
    if key in ("semanticscholar", "semantic-scholar", "s2"):
        return SemanticScholarProvider(
            http,
            cache=cache,
            base_url=base_urls.get("semanticscholar", "https://api.semanticscholar.org/graph/v1"),
        )
    raise KeyError(f"unknown discovery provider: {name!r}")


#: Provider names that can be built without credentials.
PROVIDER_FACTORIES: tuple[str, ...] = ("crossref", "openalex", "semanticscholar")
