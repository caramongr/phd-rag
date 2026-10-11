"""Discovery CLI (RAG-010 §24).

    python -m app.discovery.search "AI governance in higher education"
    python -m app.discovery.search "generative AI academic integrity" --year-min 2025 --new-only
    python -m app.discovery.search --doi 10.1016/j.compedu.2019.103778
    python -m app.discovery.search --title "Artificial intelligence in higher education"

Default output groups results into already-have / new-verified / discovery-only /
conflicts and never dumps raw provider JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from app.config import get_settings
from app.discovery.models import DiscoveryFilters, DiscoveryResult
from app.discovery.render import format_result
from app.discovery.service import DEFAULT_PROVIDERS, DiscoveryService
from app.metadata.external.http import HttpClient, RateLimiter, RequestsTransport


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m app.discovery.search",
        description="Search external scholarly sources without importing anything.",
    )
    parser.add_argument("query", nargs="?", help="natural-language query")
    parser.add_argument("--doi", default=None, help="look up one DOI instead of searching")
    parser.add_argument("--title", default=None, help="look up one exact title instead")
    parser.add_argument("--year-min", type=int, default=None)
    parser.add_argument("--year-max", type=int, default=None)
    parser.add_argument("--limit", type=int, default=20, help="maximum results per provider")
    parser.add_argument(
        "--provider",
        action="append",
        default=None,
        help="provider to use (repeatable); default crossref + openalex",
    )
    parser.add_argument("--author", default=None, help="author filter, where supported")
    parser.add_argument(
        "--sort",
        choices=("relevance", "recent"),
        default="relevance",
        help="provider sort order (recent = newest first)",
    )
    parser.add_argument(
        "--verify-fulltext",
        action="store_true",
        help="opt-in: fetch open-access full text to confirm identity (never imported)",
    )
    parser.add_argument("--refresh", action="store_true", help="bypass a valid cache entry")
    parser.add_argument("--no-cache", action="store_true", help="disable the cache entirely")
    parser.add_argument("--show-abstract", action="store_true")
    parser.add_argument("--new-only", action="store_true", help="hide results already in the corpus")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="do not contact providers; only compare nothing (sanity check)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    parser.add_argument("--width", type=int, default=100, help="preview width")
    parser.add_argument("--no-audit", action="store_true", help="hide the audit header")
    return parser.parse_args(argv)


def build_service(args: argparse.Namespace) -> DiscoveryService:
    """Build the discovery service from CLI arguments and settings."""
    settings = get_settings()
    cache = None
    if not args.no_cache:
        from app.discovery.cache import DiscoveryCache

        cache = DiscoveryCache(
            settings.discovery_cache_path,
            ttl_seconds=settings.discovery_cache_ttl,
            enabled=True,
        )
    http = HttpClient(
        RequestsTransport(),
        timeout=settings.discovery_request_timeout,
        max_retries=settings.discovery_max_retries,
        backoff_base=settings.discovery_backoff_base,
        rate_limiter=RateLimiter(settings.discovery_min_request_interval),
    )
    provider_names = tuple(args.provider or settings.discovery_providers or DEFAULT_PROVIDERS)
    if args.local_only:
        provider_names = ()
    return DiscoveryService(
        http=http,
        cache=cache,
        settings=settings,
        provider_names=provider_names,
    )


def run(args: argparse.Namespace, service: DiscoveryService) -> DiscoveryResult:
    """Perform the requested operation."""
    filters = DiscoveryFilters(
        year_min=args.year_min,
        year_max=args.year_max,
        author=args.author,
        sort=args.sort,
    )
    providers = tuple(args.provider or ()) or None
    if args.doi:
        result = service.lookup_doi(args.doi, providers=providers, refresh=args.refresh)
    elif args.title:
        result = service.lookup_title(args.title, providers=providers, refresh=args.refresh)
    elif args.query:
        result = service.discover(
            args.query,
            max_results=args.limit,
            providers=providers,
            filters=filters,
            refresh=args.refresh,
        )
    else:
        raise SystemExit("a query, --doi or --title is required")
    if args.verify_fulltext:
        result = DiscoveryResult(
            audit=result.audit,
            records=tuple(
                service.verify_fulltext(record) if record.fulltext_url else record
                for record in result.records
            ),
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    service = build_service(args)
    result = run(args, service)

    if args.json:
        payload = {
            "audit": result.audit.model_dump(mode="json"),
            "records": [record.model_dump(mode="json") for record in result.records],
        }
        print(json.dumps(payload, indent=2))
        return 0

    if not result.audit.providers_requested:
        print(
            "Discovery disabled for this run (no providers requested). Local RAG "
            "retrieval is unaffected.\n"
        )
    elif not result.records and not result.audit.any_provider_succeeded:
        print(
            "No external results: every requested provider failed. Local RAG retrieval "
            "is unaffected.\n"
        )
    print(
        format_result(
            result,
            show_abstract=args.show_abstract,
            new_only=args.new_only,
            width=args.width,
            show_audit=not args.no_audit,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
