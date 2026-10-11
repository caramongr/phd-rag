"""RAG-006 closure tests: hygiene for *external* metadata text.

Provider records occasionally contain JATS/XML markup, HTML entities or
double-encoded UTF-8. These tests pin the deterministic cleanup, its
conservatism (valid Unicode and ordinary titles are returned unchanged) and the
provenance that keeps the provider's original text inspectable.
"""

from __future__ import annotations

from metadata_support import (
    FakeResponse,
    FakeTransport,
    build_test_enricher,
    crossref_doi_handler,
    crossref_message,
    make_http,
)

from app.metadata.enums import Confidence, CorpusRole, MetadataSource
from app.metadata.external.clients import CrossrefClient, OpenAlexClient
from app.metadata.external.hygiene import clean_external_text, has_artefacts
from app.metadata.models import AcademicMetadata, MetadataValue


def test_jats_inline_markup_is_removed_keeping_its_text() -> None:
    assert (
        clean_external_text(
            "<scp>AI</scp> Literacy: Definition, Teaching, Evaluation and Ethical Issues"
        )
        == "AI Literacy: Definition, Teaching, Evaluation and Ethical Issues"
    )
    assert clean_external_text("Q<i>-</i>learning in the classroom") == (
        "Q-learning in the classroom"
    )
    assert clean_external_text("<title>Real Title</title>") == "Real Title"


def test_block_and_nested_markup_separates_words() -> None:
    assert clean_external_text("<jats:p>First part.</jats:p><jats:p>Second part.</jats:p>") == (
        "First part. Second part."
    )
    assert clean_external_text("<p>First</p><p>Second</p>") == "First Second"
    assert clean_external_text("The <italic>nested <bold>markup</bold></italic> case") == (
        "The nested markup case"
    )


def test_html_entities_are_decoded() -> None:
    assert clean_external_text("Sense &amp; Sensibility in &lt;b&gt;AI&lt;/b&gt; education") == (
        "Sense & Sensibility in AI education"
    )
    assert clean_external_text("Mind&#8211;body practices") == "Mind–body practices"


def test_double_encoded_utf8_is_repaired() -> None:
    assert clean_external_text("Students\u00e2\u20ac\u2122 Academic Performance") == (
        "Students\u2019 Academic Performance"
    )
    assert clean_external_text(
        "Tree-Based Ensemble Methods and Their Applications for Predicting "
        "Students\u00e2\u20ac\u2122 Academic Performance"
    ) == (
        "Tree-Based Ensemble Methods and Their Applications for Predicting "
        "Students\u2019 Academic Performance"
    )
    assert clean_external_text("Fran\u00c3\u00a7ois and caf\u00c3\u00a9") == "François and café"
    assert clean_external_text("\u00e2\u20ac\u2122quoted\u00e2\u20ac\u009d") == "\u2019quoted\u201d"


def test_valid_unicode_is_unchanged() -> None:
    for value in (
        "Café culture in universités françaises",
        "Başak Melis Öcal",
        "Explainable Artificial Intelligence (XAI): Concepts, Taxonomies – Opportunities",
        "‘Smart’ quotes and an em—dash",
        "Ângulo and Ãngela are Portuguese words",
        "注意力机制在推荐系统中的应用",
    ):
        assert clean_external_text(value) == value
        assert not has_artefacts(value)


def test_ordinary_titles_are_unchanged() -> None:
    for value in (
        "DREAMFUSION: TEXT-TO-3D USING 2D DIFFUSION",
        "Modeling Tabular data using Conditional GAN",
        "PATE-GAN: GENERATING SYNTHETIC DATA WITH DIFFERENTIAL PRIVACY GUARANTEES",
        "feduc-2021-645758 1..9",
        "5 < 10 and 20 > 3 cases",
        "DSJ_2300028 1..24",
    ):
        assert clean_external_text(value) == value
        assert not has_artefacts(value)


def test_only_whitespace_runs_are_collapsed() -> None:
    # Whitespace collapsing is pre-existing behaviour; case and punctuation stay.
    assert clean_external_text(
        "Soft Actor-Critic:  Off-Policy Maximum Entropy Deep Reinforcement Learning"
    ) == "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning"
    assert clean_external_text("  Padded Title \n") == "Padded Title"


def test_blank_values_stay_missing() -> None:
    assert clean_external_text(None) is None
    assert clean_external_text("") is None
    assert clean_external_text("   ") is None


def test_crossref_record_cleans_title_and_keeps_the_provider_text() -> None:
    transport = FakeTransport(
        crossref_doi_handler(
            crossref_message(
                doi="10.1000/hygiene",
                title="<scp>AI</scp> Literacy: Definition, Teaching, Evaluation and Ethical Issues",
                venue="Journal of <italic>Testing</italic>",
            )
        )
    )
    client = CrossrefClient(make_http(transport))

    record = client.lookup_doi("10.1000/hygiene")

    assert record is not None
    assert record.title == "AI Literacy: Definition, Teaching, Evaluation and Ethical Issues"
    assert record.raw_title == (
        "<scp>AI</scp> Literacy: Definition, Teaching, Evaluation and Ethical Issues"
    )
    assert record.normalised == ("title", "venue")
    assert record.venue == "Journal of Testing"


def test_openalex_record_cleans_title_and_identifies_itself() -> None:
    def handler(url: str, params: dict | None) -> FakeResponse:
        if "/works/" in url:
            return FakeResponse(
                200,
                {
                    "title": "Students\u00e2\u20ac\u2122 Use of <i>Generative</i> AI",
                    "doi": "https://doi.org/10.1000/openalex",
                    "publication_year": 2024,
                    "type": "article",
                    "primary_location": {
                        "source": {
                            "display_name": "Journal of Testing",
                            "host_organization_name": "Test Press",
                        }
                    },
                },
            )
        return FakeResponse(404, {})

    transport = FakeTransport(handler)
    client = OpenAlexClient(make_http(transport), mailto="researcher@example.org")

    record = client.lookup_doi("10.1000/openalex")

    assert record is not None
    assert record.title == "Students’ Use of Generative AI"
    assert record.raw_title == "Students\u00e2\u20ac\u2122 Use of <i>Generative</i> AI"
    assert record.normalised == ("title",)
    call = transport.calls[0]
    assert call["headers"]["User-Agent"].endswith("mailto:researcher@example.org")
    assert call["params"]["mailto"] == "researcher@example.org"


def test_unconfigured_contact_sends_no_invented_identity() -> None:
    transport = FakeTransport(lambda url, params: FakeResponse(404, {}))

    crossref = CrossrefClient(make_http(transport))
    openalex = OpenAlexClient(make_http(transport))
    crossref.lookup_doi("10.1000/none")
    openalex.lookup_doi("10.1000/none")

    for call in transport.calls:
        assert "mailto" not in call["headers"]["User-Agent"]
        assert "mailto" not in (call["params"] or {})


def test_merged_title_evidence_records_normalised_provider_text() -> None:
    message = crossref_message(
        doi="10.1000/merged",
        title="<scp>AI</scp> Literacy: Definition, Teaching, Evaluation and Ethical Issues",
    )
    enricher = build_test_enricher(FakeTransport(crossref_doi_handler(message)))
    metadata = AcademicMetadata(
        filename="paper.pdf",
        source_path="paper.pdf",
        extension=".pdf",
        corpus_role=CorpusRole.LITERATURE,
        doi=MetadataValue(
            value="10.1000/merged",
            source=MetadataSource.DOCUMENT_TEXT,
            confidence=Confidence.HIGH,
            evidence="first-page DOI pattern",
        ),
        candidates={
            "doi": [
                MetadataValue(
                    value="10.1000/merged",
                    source=MetadataSource.DOCUMENT_TEXT,
                    confidence=Confidence.HIGH,
                    evidence="first-page DOI pattern",
                )
            ]
        },
    )

    result = enricher.enrich(metadata)

    assert result.title is not None
    assert result.title.value == (
        "AI Literacy: Definition, Teaching, Evaluation and Ethical Issues"
    )
    assert result.title.source is MetadataSource.CROSSREF
    assert result.title.evidence is not None
    assert result.title.evidence.endswith("(provider text normalised)")

    external = [
        candidate
        for candidate in result.candidates["title"]
        if candidate.source is MetadataSource.CROSSREF
    ]
    assert len(external) == 1
