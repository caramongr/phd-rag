"""Optional full-text verification with SSRF controls (RAG-010 §17-§19, §29).

``verify_fulltext(record)`` may follow a legitimate open-access URL to confirm that the
record's identity is real. It is **opt-in**, it never writes to the corpus, and it never
replaces metadata with downloaded content.

External URLs are untrusted input, so every hop is validated before it is fetched:

* only ``http``/``https``;
* ``file:``, ``ftp:``, ``data:`` and other schemes are refused;
* ``localhost``, loopback, link-local, private and reserved addresses are refused
  (both literal IPs and resolved hostnames);
* redirects are followed manually, one hop at a time, and each hop is re-validated, so
  a redirect cannot smuggle the fetch into a private network;
* a timeout, a download size cap and a content-type allowlist are enforced;
* nothing is executed, and only a hash plus a short identity excerpt are kept.

Identity is confirmed only when the fetched bytes actually carry the record's DOI or
its title; otherwise the fetch is recorded as unverified rather than assumed good.
"""

from __future__ import annotations

import hashlib
import ipaddress
import re
import socket
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Protocol
from urllib.parse import urlparse

from app.discovery.models import DiscoveryStatus, ExternalWork, FulltextStatus

#: Schemes we are willing to fetch.
ALLOWED_SCHEMES = ("http", "https")
#: Content types we accept as "content", not as an executable or a blob.
ALLOWED_CONTENT_TYPES = (
    "application/pdf",
    "text/html",
    "text/plain",
    "application/xml",
    "text/xml",
    "application/xhtml+xml",
)
#: Default hard cap on a single download (25 MiB).
DEFAULT_MAX_BYTES = 25 * 1024 * 1024
#: Maximum redirect hops followed.
DEFAULT_MAX_REDIRECTS = 5
#: How much text is inspected when confirming identity.
IDENTITY_EXCERPT_CHARS = 20000


class UnsafeUrlError(RuntimeError):
    """The URL is not something this module will fetch."""


class FetchedTooLargeError(RuntimeError):
    """The download exceeded the configured size cap."""


@dataclass(frozen=True)
class FetchResult:
    """One hop's response."""

    status_code: int
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    content: bytes = b""
    content_type: str | None = None

    @property
    def location(self) -> str | None:
        """``Location`` header for a redirect, if any."""
        for key, value in self.headers.items():
            if key.lower() == "location":
                return value
        return None


class BinaryFetcher(Protocol):
    """Fetch bytes with redirects disabled (the caller handles redirects)."""

    def fetch(
        self, url: str, *, headers: dict[str, str], timeout: float, max_bytes: int
    ) -> FetchResult: ...


class RequestsBinaryFetcher:
    """A :class:`BinaryFetcher` built on ``requests`` with streaming and a size cap."""

    def fetch(
        self, url: str, *, headers: dict[str, str], timeout: float, max_bytes: int
    ) -> FetchResult:
        import requests

        with requests.get(
            url,
            headers=headers,
            timeout=timeout,
            allow_redirects=False,
            stream=True,
        ) as response:
            status = int(response.status_code)
            response_headers = {str(k): str(v) for k, v in response.headers.items()}
            if status >= 300:
                return FetchResult(
                    status_code=status, url=url, headers=response_headers
                )
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise FetchedTooLargeError(
                        f"download exceeded {max_bytes} bytes at {url}"
                    )
                chunks.append(chunk)
            content_type = response_headers.get("Content-Type")
            return FetchResult(
                status_code=status,
                url=url,
                headers=response_headers,
                content=b"".join(chunks),
                content_type=content_type.split(";")[0].strip().lower()
                if content_type
                else None,
            )


@dataclass(frozen=True)
class VerificationResult:
    """What full-text verification concluded."""

    verified: bool
    url: str | None = None
    status_code: int | None = None
    content_type: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    fetched_at: str | None = None
    identity_evidence: str | None = None
    note: str = ""
    hops: tuple[str, ...] = ()
    truncated: bool = False
    error: str | None = None

    def apply(self, work: ExternalWork) -> ExternalWork:
        """Return ``work`` with the verification outcome attached (never replaced)."""
        provenance = tuple(
            dict.fromkeys(
                (*work.source_provenance, *(f"fulltext:{url}" for url in self.hops[-1:]))
            )
        )
        status = work.discovery_status
        if self.verified and status not in (
            DiscoveryStatus.CONFLICT,
            DiscoveryStatus.LOCAL_MATCH,
        ):
            status = DiscoveryStatus.EXTERNAL_FULLTEXT_VERIFIED
        elif self.error and status is DiscoveryStatus.EXTERNAL_DISCOVERY:
            status = DiscoveryStatus.UNAVAILABLE
        return work.model_copy(
            update={
                "discovery_status": status,
                "fulltext_verified": self.verified,
                "fulltext_sha256": self.sha256,
                "fulltext_bytes": self.size_bytes,
                "fulltext_content_type": self.content_type,
                "fulltext_fetched_at": self.fetched_at,
                "fulltext_source_url": self.url,
                "fulltext_note": self.note,
                "source_provenance": provenance,
            }
        )


def is_safe_url(url: str, *, resolve: bool = False, resolver: Callable[[str], list[str]] | None = None) -> tuple[bool, str]:
    """Whether ``url`` may be fetched, plus the reason when it may not."""
    try:
        parsed = urlparse(url)
    except ValueError as exc:  # pragma: no cover - urlparse rarely raises
        return False, f"unparseable URL: {exc}"
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        return False, f"scheme {parsed.scheme or '(missing)'} is not allowed"
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return False, "URL has no host"
    if host in ("localhost", "localhost.localdomain") or host.endswith(".localhost"):
        return False, "localhost is not allowed"
    if host.endswith(".local") or host.endswith(".internal"):
        return False, f"internal hostname {host!r} is not allowed"
    addresses: list[str] = []
    try:
        addresses.append(str(ipaddress.ip_address(host)))
    except ValueError:
        if resolve:
            try:
                addresses.extend(_resolve_host(host, resolver))
            except OSError as exc:
                return False, f"could not resolve {host!r}: {exc}"
    for address in addresses:
        try:
            parsed_ip = ipaddress.ip_address(address)
        except ValueError:
            continue
        if (
            parsed_ip.is_private
            or parsed_ip.is_loopback
            or parsed_ip.is_link_local
            or parsed_ip.is_reserved
            or parsed_ip.is_multicast
            or parsed_ip.is_unspecified
        ):
            return False, f"{host!r} resolves to non-public address {address}"
    return True, "ok"


def _resolve_host(host: str, resolver: Callable[[str], list[str]] | None) -> list[str]:
    if resolver is not None:
        return list(resolver(host))
    infos = socket.getaddrinfo(host, None)
    return [info[4][0] for info in infos]


def _extract_text(content: bytes, content_type: str | None) -> str:
    """Best-effort text for identity checking (never stored wholesale)."""
    if content_type == "application/pdf" or content[:5] == b"%PDF-":
        try:
            import pymupdf  # already a project dependency (see the parsers)

            with pymupdf.open(stream=content, filetype="pdf") as document:
                pages = []
                for index, page in enumerate(document):
                    if index >= 2:
                        break
                    pages.append(page.get_text())
                return "\n".join(pages)
        except Exception:
            return content.decode("latin-1", "ignore")
    return content.decode("utf-8", "ignore")


def identity_evidence(text: str, work: ExternalWork) -> str | None:
    """Return which identity signal the fetched text contains, or ``None``."""
    haystack = text[:IDENTITY_EXCERPT_CHARS]
    folded = haystack.casefold()
    if work.doi:
        doi = work.doi.casefold()
        if doi in folded or doi.replace("/", " ") in folded:
            return f"doi:{work.doi}"
    if work.title:
        words = [word for word in re.split(r"[^0-9A-Za-z]+", work.title.casefold()) if len(word) > 3]
        if words:
            hits = sum(1 for word in words if word in folded)
            if hits / len(words) >= 0.8:
                return f"title:{hits}/{len(words)} words"
    return None


def verify_fulltext(
    work: ExternalWork,
    *,
    fetcher: BinaryFetcher | None = None,
    timeout: float = 20.0,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    resolve: bool = True,
    resolver: Callable[[str], list[str]] | None = None,
    follow_landing_pages: bool = False,
    mailto: str | None = None,
) -> VerificationResult:
    """Fetch and verify the full text of ``work`` (opt-in).

    Only a real fetch that carries the record's DOI or 80% of its title words counts as
    verified; an HTML landing page counts only when ``follow_landing_pages`` is set,
    because a landing page is not the paper.
    """
    url = work.fulltext_url
    if not url:
        return VerificationResult(
            verified=False,
            note="no full-text URL supplied by any provider",
            error="no_fulltext_url",
        )
    if work.fulltext_status is not FulltextStatus.FULLTEXT_OPEN and not follow_landing_pages:
        return VerificationResult(
            verified=False,
            url=url,
            note="provider did not state an open full text; not fetched",
            error="not_open_access",
        )
    fetcher = fetcher or RequestsBinaryFetcher()
    agent = "phd-rag/0.1 (academic discovery; fulltext verification)"
    headers = {"User-Agent": f"{agent} mailto:{mailto}" if mailto else agent}
    hops: list[str] = []
    current = url
    for _hop in range(max_redirects + 1):
        safe, reason = is_safe_url(current, resolve=resolve, resolver=resolver)
        if not safe:
            return VerificationResult(
                verified=False,
                url=current,
                note=(
                    f"refused to follow the redirect to {current!r}: {reason}"
                    if hops
                    else f"refused to fetch: {reason}"
                ),
                error="unsafe_redirect" if hops else "unsafe_url",
                hops=tuple(hops),
            )
        hops.append(current)
        try:
            result = fetcher.fetch(current, headers=headers, timeout=timeout, max_bytes=max_bytes)
        except FetchedTooLargeError as exc:
            return VerificationResult(
                verified=False, url=current, note=str(exc), error="too_large", hops=tuple(hops)
            )
        except Exception as exc:
            return VerificationResult(
                verified=False,
                url=current,
                note=f"fetch failed: {type(exc).__name__}: {exc}",
                error="fetch_failed",
                hops=tuple(hops),
            )
        if 300 <= result.status_code < 400 and result.location:
            location = result.location
            if location.startswith("/"):
                parsed = urlparse(current)
                location = f"{parsed.scheme}://{parsed.netloc}{location}"
            current = location
            continue
        if result.status_code != 200:
            return VerificationResult(
                verified=False,
                url=current,
                status_code=result.status_code,
                note=f"HTTP {result.status_code}",
                error="http_error",
                hops=tuple(hops),
            )
        content_type = (result.content_type or "").split(";")[0].strip().lower() or None
        if content_type and not any(
            content_type.startswith(allowed) for allowed in ALLOWED_CONTENT_TYPES
        ):
            return VerificationResult(
                verified=False,
                url=current,
                status_code=result.status_code,
                content_type=content_type,
                note=f"content type {content_type!r} is not accepted",
                error="bad_content_type",
                hops=tuple(hops),
            )
        content = result.content or b""
        if len(content) > max_bytes:
            return VerificationResult(
                verified=False,
                url=current,
                status_code=result.status_code,
                note=f"download exceeded {max_bytes} bytes",
                error="too_large",
                hops=tuple(hops),
            )
        sha256 = hashlib.sha256(content).hexdigest()
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        text = _extract_text(content, content_type)
        evidence = identity_evidence(text, work)
        is_pdf = content_type == "application/pdf" or content[:5] == b"%PDF-"
        if evidence is None:
            return VerificationResult(
                verified=False,
                url=current,
                status_code=result.status_code,
                content_type=content_type,
                sha256=sha256,
                size_bytes=len(content),
                fetched_at=fetched_at,
                note="content fetched but identity could not be confirmed from its text",
                error="identity_unconfirmed",
                hops=tuple(hops),
            )
        if not is_pdf and not follow_landing_pages:
            return VerificationResult(
                verified=False,
                url=current,
                status_code=result.status_code,
                content_type=content_type,
                sha256=sha256,
                size_bytes=len(content),
                fetched_at=fetched_at,
                identity_evidence=evidence,
                note="identity found, but the URL serves a page, not full text",
                error="not_fulltext",
                hops=tuple(hops),
            )
        return VerificationResult(
            verified=True,
            url=current,
            status_code=result.status_code,
            content_type=content_type,
            sha256=sha256,
            size_bytes=len(content),
            fetched_at=fetched_at,
            identity_evidence=evidence,
            note=f"identity confirmed via {evidence}",
            hops=tuple(hops),
        )
    return VerificationResult(
        verified=False,
        url=current,
        note=f"more than {max_redirects} redirects",
        error="too_many_redirects",
        hops=tuple(hops),
    )


def verify_records(
    records: list[ExternalWork], **kwargs: Any
) -> tuple[list[ExternalWork], list[VerificationResult]]:
    """Verify several records, returning the updated records and the raw outcomes."""
    updated: list[ExternalWork] = []
    results: list[VerificationResult] = []
    for work in records:
        result = verify_fulltext(work, **kwargs)
        results.append(result)
        updated.append(result.apply(work))
    return updated, results
