"""Full-text verification and SSRF controls (RAG-010 §29, §30 items 18, 25-28).

External URLs are untrusted input. These tests prove the fetch refuses local, private,
non-http and oversize targets, and that identity is only claimed when the retrieved
bytes really carry the record's DOI or title.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.discovery.fulltext import (
    FetchedTooLargeError,
    VerificationResult,
    is_safe_url,
    verify_fulltext,
)
from app.discovery.models import DiscoveryStatus, ExternalWork, FulltextStatus
from tests.discovery_support import FakeFetcher, fetch_result


def _work(**kwargs: object) -> ExternalWork:
    """An open-access external record with a full-text URL."""
    base: dict[str, object] = {
        "title": "AI Governance in Universities",
        "authors": ("Jane Doe",),
        "year": 2024,
        "doi": "10.1234/abc",
        "fulltext_url": "https://example.org/paper.pdf",
        "fulltext_status": FulltextStatus.FULLTEXT_OPEN,
        "providers": ("openalex",),
    }
    base.update(kwargs)
    return ExternalWork(**base)  # type: ignore[arg-type]


# --- 25/26. scheme, host and private-network controls -----------------------


def test_file_scheme_is_blocked() -> None:
    """A ``file://`` URL is never fetched."""
    safe, reason = is_safe_url("file:///C:/Windows/System32/config/SAM")

    assert safe is False
    assert "scheme" in reason


def test_localhost_and_loopback_are_blocked() -> None:
    """Localhost, loopback and other local names are refused."""
    for url in (
        "http://localhost/paper.pdf",
        "http://127.0.0.1/paper.pdf",
        "http://[::1]/paper.pdf",
        "http://localhost.localdomain/x",
        "http://intranet.local/x",
    ):
        safe, _reason = is_safe_url(url)
        assert safe is False, url


def test_private_and_link_local_addresses_are_blocked() -> None:
    """Private, link-local, reserved and multicast literals are refused."""
    for url in (
        "http://10.0.0.5/paper.pdf",
        "http://192.168.1.10/paper.pdf",
        "http://172.16.4.4/paper.pdf",
        "http://169.254.169.254/latest/meta-data/",
        "http://0.0.0.0/paper.pdf",
        "http://224.0.0.1/x",
    ):
        safe, _reason = is_safe_url(url)
        assert safe is False, url


def test_non_http_schemes_are_blocked() -> None:
    """FTP, data and other schemes are refused."""
    for url in ("ftp://example.org/paper.pdf", "data:text/plain,hello", "javascript:alert(1)"):
        safe, _reason = is_safe_url(url)
        assert safe is False, url


def test_hostname_resolving_to_a_private_address_is_blocked() -> None:
    """A public-looking hostname that resolves internally is refused."""
    safe, reason = is_safe_url(
        "https://papers.example.org/x.pdf",
        resolve=True,
        resolver=lambda _host: ["192.168.0.10"],
    )
    assert safe is False
    assert "non-public" in reason


def test_public_url_is_allowed() -> None:
    """A plain public HTTPS URL is allowed."""
    safe, reason = is_safe_url(
        "https://example.org/paper.pdf", resolve=True, resolver=lambda _host: ["93.184.216.34"]
    )
    assert safe is True
    assert reason == "ok"


def test_url_without_a_host_is_blocked() -> None:
    """A URL with no host cannot be fetched."""
    safe, _reason = is_safe_url("https:///paper.pdf")
    assert safe is False


# --- 28. unsafe redirect ----------------------------------------------------


def test_redirect_to_a_private_address_is_rejected() -> None:
    """A redirect cannot smuggle the fetch into a private network."""
    fetcher = FakeFetcher(
        {
            "https://example.org/paper.pdf": fetch_result(
                url="https://example.org/paper.pdf",
                body=b"",
                status_code=302,
                headers={"Location": "http://127.0.0.1/secret"},
            )
        }
    )

    result = verify_fulltext(_work(), fetcher=fetcher, resolve=False)

    assert result.verified is False
    assert result.error == "unsafe_redirect"
    assert len(fetcher.calls) == 1


def test_redirect_to_a_public_address_is_followed_and_verified() -> None:
    """A legitimate public redirect is followed, one validated hop at a time."""
    body = b"%PDF-1.4 DOI: 10.1234/abc"
    fetcher = FakeFetcher(
        {
            "https://example.org/paper.pdf": fetch_result(
                url="https://example.org/paper.pdf",
                body=b"",
                status_code=301,
                headers={"Location": "https://publisher.example.org/paper.pdf"},
            ),
            "https://publisher.example.org/paper.pdf": fetch_result(
                url="https://publisher.example.org/paper.pdf",
                body=body,
            ),
        }
    )

    result = verify_fulltext(_work(), fetcher=fetcher, resolve=False)

    assert result.verified is True
    assert result.sha256 == hashlib.sha256(body).hexdigest()
    assert result.url == "https://publisher.example.org/paper.pdf"
    assert len(fetcher.calls) == 2


def test_redirect_loop_is_bounded() -> None:
    """A redirect cycle stops instead of spinning."""
    responses = {
        f"https://example.org/hop{index}.pdf": fetch_result(
            url=f"https://example.org/hop{index}.pdf",
            body=b"",
            status_code=302,
            headers={"Location": f"https://example.org/hop{index + 1}.pdf"},
        )
        for index in range(10)
    }
    fetcher = FakeFetcher(responses)

    result = verify_fulltext(
        _work(fulltext_url="https://example.org/hop0.pdf"),
        fetcher=fetcher,
        resolve=False,
        max_redirects=3,
    )

    assert result.verified is False
    assert result.error == "too_many_redirects"
    assert len(fetcher.calls) == 4


# --- 27. oversize download --------------------------------------------------


def test_oversize_download_is_rejected() -> None:
    """A download over the size cap is refused, not truncated silently."""
    fetcher = FakeFetcher(
        {
            "https://example.org/paper.pdf": FetchedTooLargeError(
                "download exceeded 100 bytes"
            )
        }
    )

    result = verify_fulltext(_work(), fetcher=fetcher, resolve=False, max_bytes=100)

    assert result.verified is False
    assert result.error == "too_large"


def test_unexpected_content_type_is_rejected() -> None:
    """A binary blob that is not a document is not treated as full text."""
    fetcher = FakeFetcher(
        {
            "https://example.org/paper.pdf": fetch_result(
                content_type="application/octet-stream",
                body=b"\x00\x01\x02",
            )
        }
    )

    result = verify_fulltext(_work(), fetcher=fetcher, resolve=False)

    assert result.verified is False
    assert result.error == "bad_content_type"


# --- identity: only after real content --------------------------------------


def test_verification_requires_the_doi_in_the_downloaded_text() -> None:
    """Bytes without the DOI or the title are not accepted as this paper."""
    fetcher = FakeFetcher(
        {
            "https://example.org/paper.pdf": fetch_result(
                content_type="application/pdf",
                body=b"%PDF-1.4\nSome other document entirely.",
            )
        }
    )

    result = verify_fulltext(_work(), fetcher=fetcher, resolve=False)

    assert result.verified is False
    assert result.error == "identity_unconfirmed"
    assert result.sha256 is not None  # the fetch itself is still recorded honestly


def test_html_page_with_the_identity_is_not_claimed_as_full_text() -> None:
    """A landing page carrying the title is a page, not the paper."""
    fetcher = FakeFetcher(
        {
            "https://example.org/paper.pdf": fetch_result(
                content_type="text/html",
                body=b"<html><h1>AI Governance in Universities</h1></html>",
            )
        }
    )

    result = verify_fulltext(_work(), fetcher=fetcher, resolve=False)

    assert result.verified is False
    assert result.error == "not_fulltext"
    assert result.identity_evidence is not None


def test_title_words_confirm_identity_in_a_pdf() -> None:
    """80% of the title words is enough when the DOI is absent."""
    fetcher = FakeFetcher(
        {
            "https://example.org/paper.pdf": fetch_result(
                content_type="application/pdf",
                body=b"%PDF-1.4\nAI Governance in Universities: a study",
            )
        }
    )

    result = verify_fulltext(_work(doi=None), fetcher=fetcher, resolve=False)

    assert result.verified is True
    assert result.identity_evidence is not None
    assert result.identity_evidence.startswith("title:")


def test_landing_page_only_record_is_not_fetched_by_default() -> None:
    """Without an open full text, nothing is fetched unless explicitly asked."""
    fetcher = FakeFetcher({})

    result = verify_fulltext(
        _work(fulltext_status=FulltextStatus.LANDING_PAGE_ONLY),
        fetcher=fetcher,
        resolve=False,
    )

    assert result.verified is False
    assert result.error == "not_open_access"
    assert fetcher.calls == []


def test_record_without_fulltext_url_is_reported() -> None:
    """No URL means no fetch and a plain explanation."""
    result = verify_fulltext(_work(fulltext_url=None), fetcher=FakeFetcher({}), resolve=False)

    assert result.verified is False
    assert result.error == "no_fulltext_url"


def test_verification_never_replaces_metadata() -> None:
    """The record keeps its provider metadata; only verification facts are added."""
    body = b"%PDF-1.4 DOI: 10.1234/abc"
    fetcher = FakeFetcher({"https://example.org/paper.pdf": fetch_result(body=body)})
    work = _work()

    result = verify_fulltext(work, fetcher=fetcher, resolve=False)
    updated = result.apply(work)

    assert updated.title == work.title
    assert updated.doi == work.doi
    assert updated.authors == work.authors
    assert updated.venue == work.venue
    assert updated.providers == work.providers
    assert updated.fulltext_verified is True
    assert updated.fulltext_sha256 == hashlib.sha256(body).hexdigest()
    assert updated.fulltext_source_url == "https://example.org/paper.pdf"
    assert updated.fulltext_fetched_at
    assert updated.discovery_status is DiscoveryStatus.EXTERNAL_FULLTEXT_VERIFIED


def test_failed_verification_does_not_upgrade_the_record() -> None:
    """A failed fetch leaves the metadata status alone."""
    fetcher = FakeFetcher(
        {
            "https://example.org/paper.pdf": fetch_result(
                status_code=403, body=b"", content_type="text/html"
            )
        }
    )
    work = _work(discovery_status=DiscoveryStatus.EXTERNAL_METADATA_VERIFIED)

    updated = verify_fulltext(work, fetcher=fetcher, resolve=False).apply(work)

    assert updated.fulltext_verified is False
    assert updated.discovery_status is DiscoveryStatus.EXTERNAL_METADATA_VERIFIED


def test_invalid_pdf_is_not_treated_as_verified_text() -> None:
    """A body that is not really a PDF and carries no identity signal stays unverified."""
    fetcher = FakeFetcher(
        {
            "https://example.org/paper.pdf": fetch_result(
                content_type="application/pdf", body=b"not really a pdf"
            )
        }
    )

    result = verify_fulltext(_work(), fetcher=fetcher, resolve=False)

    assert result.verified is False
    assert result.error == "identity_unconfirmed"


def test_service_verification_attaches_evidence_and_audit(tmp_path: Path) -> None:
    """The service-level opt-in verification path returns an updated record."""
    from tests.discovery_support import QueuedTransport, build_service

    service = build_service(QueuedTransport([]))
    body = b"%PDF-1.4 DOI: 10.1234/abc"
    fetcher = FakeFetcher({"https://example.org/paper.pdf": fetch_result(body=body)})

    original = _work()
    updated = service.verify_fulltext(original, fetcher=fetcher)

    assert updated.fulltext_verified is True
    assert original.fulltext_verified is False  # the input is untouched
    assert isinstance(service.last_verification, VerificationResult)
    assert service.last_verification.sha256 == hashlib.sha256(body).hexdigest()


def test_fetcher_headers_carry_the_configured_contact_identity() -> None:
    """Full-text fetches identify themselves politely too, when a contact exists."""
    fetcher = FakeFetcher(
        {"https://example.org/paper.pdf": fetch_result(body=b"%PDF-1.4 DOI 10.1234/abc")}
    )

    verify_fulltext(
        _work(), fetcher=fetcher, resolve=False, mailto="someone@example.org"
    )

    assert "mailto:someone@example.org" in fetcher.calls[0]["headers"]["User-Agent"]


def test_fetcher_headers_have_no_invented_contact_when_unconfigured() -> None:
    """Without a configured contact, no address is invented."""
    fetcher = FakeFetcher(
        {"https://example.org/paper.pdf": fetch_result(body=b"%PDF-1.4 DOI 10.1234/abc")}
    )

    verify_fulltext(_work(), fetcher=fetcher, resolve=False)

    agent = fetcher.calls[0]["headers"]["User-Agent"]
    assert agent == "phd-rag/0.1 (academic discovery; fulltext verification)"
    assert "mailto" not in agent
