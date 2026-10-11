"""RAG-009 bibliography guard: detection, mode policy and audit trail.

The guard exists because 142 indexed chunks are reference lists whose upstream
``content_type`` is not ``references``. These tests pin both directions: a real
reference list is detected and withheld, and ordinary academic prose that cites
sources - even with DOIs and URLs - is never withheld.
"""

from __future__ import annotations

import pytest

from app.retrieval.bibliography import (
    REASON_AVAILABLE_AT,
    REASON_DENSITY,
    REASON_DOI,
    REASON_ENTRY_LINES,
    REASON_HEADING,
    REASON_NUMBERED,
    REASON_YEARS,
    classify,
    is_bibliography_like,
    measure,
)
from app.retrieval.filters import mode_excludes_bibliography
from app.retrieval.models import RetrievalMode
from app.retrieval.service import RetrievalService
from tests.retrieval_support import (
    TEST_COLLECTION,
    FakeReranker,
    QueryEmbedder,
    memory_collection,
    point,
)

# --- text fixtures ------------------------------------------------------------

HARVARD_REFERENCES = "\n".join(
    f"Author{index}, A. B. & Coauthor, C. (20{10 + index}). The title of study {index}. "
    f"Journal of Something, {index + 1}(2), 1{index}5-1{index}9. Available at: "
    f"https://doi.org/10.1016/j.journal.20{10 + index}.01.00{index} [Accessed 1 January 202{index % 10}]."
    for index in range(6)
)

APA_REFERENCE_LIST = "\n".join(
    f"Zimmerman{index}, K. L., & Park, J. (201{index}). Analysing student data {index}. "
    f"In P. Editor (Ed.), Handbook of learning analytics (pp. 4{index}-6{index}). Publisher."
    for index in range(8)
)

NUMBERED_REFERENCES = "\n".join(
    f"{index + 1}. Smith, J. and Jones, B. 199{index}. A study on adaptive tutoring. "
    f"J. Educ. Res. {index + 3}, 2{index}1-2{index}9. (doi:10.1038/s41598-02{index}-0000{index})"
    for index in range(9)
)

PROSE_WITH_CITATIONS = (
    "Artificial intelligence in higher education has grown quickly. Recent work has "
    "examined governance (Smith, 2021) and academic integrity (Jones & Lee, 2022; "
    "Taylor et al., 2023), while others report engagement effects (Wong, 2020). "
    "The literature also records challenges: staff workload, data protection and the "
    "risk that models reinforce existing inequalities. See https://doi.org/10.1016/"
    "j.compedu.2019.103778 for one systematic review, and the UNESCO (2021) guidance "
    "for institutions. Taken together, these sources describe a field that is expanding "
    "faster than its policy frameworks, which is the gap this thesis addresses."
)

PROSE_WITH_MANY_CITATIONS = (
    "The review organises findings in three groups. First, pedagogy: Smith (2019), "
    "Jones (2020) and Lee (2021) all report gains in engagement, while Baker (2019) "
    "and Chen (2022) find none. Second, governance: Adams (2020), Bennett (2021), "
    "Carter et al. (2022) and Davis (2023) describe institutional policies, and the "
    "OECD (2021) framework is widely cited. Third, assessment: Evans (2018), Frank "
    "(2019), Garcia (2020), Hughes et al. (2021) and Ibrahim (2022) discuss integrity. "
    "Across all three groups the evidence remains uneven, and none of these studies "
    "isolates the effect of the specific tool used, which limits their comparability "
    "for a thesis that must justify each claim it repeats."
)

ASSESSMENT_SECTION = (
    "Assessment design matters. Peer assessment has been studied extensively "
    "(Van den Berg, 2006) and the Assessment Design Decisions Framework (Bearman, "
    "2016) offers one approach. Design principles can be derived from these sources "
    "and then tested, which is what this chapter does in the following sections, "
    "first for the formative case and then for the summative one."
)


# --- detection ----------------------------------------------------------------

@pytest.mark.parametrize(
    ("label", "text", "expected_reason"),
    [
        ("harvard", HARVARD_REFERENCES, REASON_AVAILABLE_AT),
        ("apa", APA_REFERENCE_LIST, None),
        ("numbered", NUMBERED_REFERENCES, None),
    ],
)
def test_reference_lists_are_detected(label: str, text: str, expected_reason: str | None) -> None:
    verdict = classify(text)
    assert verdict.is_bibliography_like, label
    assert verdict.reason
    if expected_reason:
        assert verdict.reason == expected_reason


def test_numbered_reference_list_is_detected() -> None:
    verdict = classify(NUMBERED_REFERENCES)
    assert verdict.is_bibliography_like
    assert verdict.reason in (REASON_DOI, REASON_NUMBERED, REASON_YEARS, REASON_DENSITY)


def test_apa_reference_list_is_detected() -> None:
    verdict = classify(APA_REFERENCE_LIST)
    assert verdict.is_bibliography_like
    assert verdict.reason in (REASON_ENTRY_LINES, REASON_YEARS, REASON_DENSITY)


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("prose", PROSE_WITH_CITATIONS),
        ("prose-many-citations", PROSE_WITH_MANY_CITATIONS),
        ("assessment-section", ASSESSMENT_SECTION),
        ("empty", ""),
        ("short", "Smith (2020) studied adaptive learning. See doi:10.1000/xyz."),
    ],
)
def test_ordinary_academic_prose_is_not_flagged(label: str, text: str) -> None:
    """A chunk is never flagged merely because it cites several sources."""
    assert is_bibliography_like(text) is False, label


def test_a_single_reference_entry_is_not_flagged() -> None:
    """One dangling citation entry is not a reference list (conservative by design)."""
    entry = (
        "Gregor, S. & Hevner, A.R., 2013. Positioning and Presenting Design Science "
        "Research for Maximum Impact. MIS Quarterly, 37(2), pp.337-355. Available at: "
        "https://doi.org/10.25300/MISQ/2013/37.2.01."
    )
    assert is_bibliography_like(entry) is False


def test_a_header_fragment_with_links_is_not_flagged() -> None:
    header = (
        "Vol:.(1234567890) Scientific Reports | (2023) 13:8425 | "
        "https://doi.org/10.1038/s41598-023-33510-4 natureportfolio "
        "www.nature.com/scientificreports Keywords: virtual reality, higher education"
    )
    assert is_bibliography_like(header) is False


def test_orcid_and_affiliation_lists_are_not_flagged() -> None:
    """Bare URL lists are metadata, not bibliography, and must stay retrievable."""
    orcids = (
        "ORCID iDs Chiao Ling Huang https://orcid.org/0000-0001-9590-806X Yi Fang Luo "
        "https://orcid.org/0000-0002-2208-0932 Shu Ching Yang "
        "https://orcid.org/0000-0003-4105-1747 Chia Mei Lu https://orcid.org/0000-0002-"
        "1234-5678 Department of Psychology, National Cheng Kung University, Tainan, "
        "Taiwan https://www.ncku.edu.tw/index.php?Lang=en https://orcid.org/0000-0002-"
        "9876-5432 Faculty of Education, National Cheng Kung University, Tainan, Taiwan "
        "https://www.example.org/institute/en https://orcid.org/0000-0002-1111-2222"
    )
    verdict = classify(orcids)
    assert verdict.is_bibliography_like is False
    assert verdict.features.non_url_markers < 3


def test_a_long_bare_link_list_is_not_flagged() -> None:
    """A list of links (resources, datasets, ORCIDs) is not a bibliography."""
    links = "\n".join(
        f"https://example.org/resource/{index}/document-{index}.pdf" for index in range(15)
    )
    assert is_bibliography_like(links) is False


def test_a_labelled_reference_section_is_flagged() -> None:
    """The document itself calling the section 'References' is strong evidence."""
    chunk = (
        "References\n"
        "1. Aerts, H. J. W. L. et al. Decoding tumour phenotype by noninvasive imaging "
        "using a quantitative radiomics approach. Nat. Commun. 5, 4006 (2014).\n"
        "2. Lambin, P. et al. Radiomics: the bridge between medical imaging and "
        "personalized medicine. Nat. Rev. Clin. Oncol. 14, 749-762 (2017).\n"
        "3. Gillies, R. J., Kinahan, P. E. & Hricak, H. Radiomics: images are more than "
        "pictures, they are data. Radiology 278, 563-577 (2016)."
    )
    verdict = classify(chunk)
    assert verdict.is_bibliography_like is True
    assert verdict.reason == REASON_HEADING
    assert verdict.features.reference_heading is True


def test_a_prose_chunk_that_mentions_references_is_not_flagged() -> None:
    prose = (
        "The literature on adaptive learning is fragmented. Earlier references to "
        "knowledge tracing (Koedinger, 2019) describe mastery models, while later work "
        "moves towards reinforcement learning. This section reviews those references and "
        "explains why the thesis treats them as background rather than as evidence, and "
        "it sets out the criteria used to include or exclude studies in the next chapter."
    )
    assert is_bibliography_like(prose) is False


def test_a_nature_style_reference_list_on_one_line_is_flagged() -> None:
    """PDF extraction can emit a whole reference list as a single line."""
    chunk = (
        "Preprint at https://arxiv.org/abs/2001.08103 (2020). 83. Pandemic data "
        "challenges. Nat. Mach. Intell. 2, 193 (2020). 84. Son, J. et al. "
        "Privacy-preserving electrocardiogram monitoring for intelligent arrhythmia "
        "detection. Biomed. Signal Process. Control 58, 101862 (2020). 85. Kim, H. & "
        "Park, S. Deep learning for arrhythmia classification. IEEE Trans. Biomed. "
        "Eng. 67, 1155-1163 (2020). 86. Zhang, Y. et al. Wearable sensors in clinical "
        "practice. Sensors 20, 4433 (2020)."
    )
    assert is_bibliography_like(chunk) is True


def test_features_are_deterministic_and_explain_the_verdict() -> None:
    first = measure(HARVARD_REFERENCES)
    second = measure(HARVARD_REFERENCES)
    assert first == second
    assert first.available_at >= 2
    assert first.doi >= 2
    assert first.density >= 4
    assert "density=" in first.describe()
    assert classify(HARVARD_REFERENCES).features == first


# --- mode policy --------------------------------------------------------------

@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (RetrievalMode.EVIDENCE, True),
        (RetrievalMode.MY_WORK, True),
        (RetrievalMode.GENERAL, True),
        (RetrievalMode.GUIDANCE, False),
        (RetrievalMode.SUPERVISION, False),
    ],
)
def test_mode_policy(mode: RetrievalMode, expected: bool) -> None:
    """Guidance/supervision material may legitimately *be* a reference list."""
    assert mode_excludes_bibliography(mode) is expected


# --- pipeline ------------------------------------------------------------------

def bibliography_corpus() -> list[dict]:
    return [
        point(
            "bib-1",
            angle=3,
            document_id="doc-bib",
            filename="own-draft.docx",
            corpus_role="my_work",
            title="My own draft with a long reference list",
            text=HARVARD_REFERENCES,
        ),
        point(
            "prose-1",
            angle=5,
            document_id="doc-own-prose",
            filename="own-draft.docx",
            corpus_role="my_work",
            title="My own draft with a long reference list",
            chunk_index=1,
            text=PROSE_WITH_MANY_CITATIONS,
        ),
        point(
            "lit-bib",
            angle=8,
            document_id="doc-lit-bib",
            filename="paper-with-bibliography.pdf",
            title="A published paper whose last chunk is its bibliography",
            text=APA_REFERENCE_LIST,
        ),
        point(
            "lit-prose",
            angle=11,
            document_id="doc-lit-prose",
            filename="paper-with-bibliography.pdf",
            title="A published paper whose last chunk is its bibliography",
            chunk_index=1,
            text=PROSE_WITH_CITATIONS,
        ),
        point(
            "guidance-1",
            angle=13,
            document_id="doc-guidance",
            filename="Reference checklist.docx",
            corpus_role="rules",
            content_type="table",
            title="Reference checklist",
            text=HARVARD_REFERENCES,
        ),
    ]


def build(client, **kwargs) -> RetrievalService:
    return RetrievalService(
        embedder=QueryEmbedder(),
        reranker=FakeReranker(score=0.5),
        client=client,
        collection=TEST_COLLECTION,
        final_k=8,
        **kwargs,
    )


def test_candidates_carry_the_bibliography_verdict() -> None:
    service = build(memory_collection(bibliography_corpus()))
    response = service.retrieve(
        "anything", mode=RetrievalMode.GENERAL, exclude_bibliography=False
    )
    by_id = {candidate.chunk_id: candidate for candidate in response.results}
    assert by_id["bib-1"].bibliography_like is True
    assert by_id["bib-1"].bibliography_reason
    assert by_id["lit-bib"].bibliography_like is True
    assert by_id["prose-1"].bibliography_like is False
    assert by_id["prose-1"].bibliography_reason is None
    service.close()


@pytest.mark.parametrize(
    "mode", [RetrievalMode.EVIDENCE, RetrievalMode.MY_WORK, RetrievalMode.GENERAL]
)
def test_bibliography_is_withheld_by_default(mode: RetrievalMode) -> None:
    service = build(memory_collection(bibliography_corpus()))
    response = service.retrieve("anything", mode=mode)
    assert all(candidate.bibliography_like is False for candidate in response.results)
    assert response.stats.dropped_as_bibliography >= 1
    assert response.audit.exclude_bibliography is True
    dropped = [item for item in response.dropped if item.reason == "bibliography_like"]
    assert dropped and dropped[0].detail
    service.close()


def test_guidance_mode_keeps_reference_material() -> None:
    """A rules document that *is* a reference table must stay retrievable."""
    service = build(memory_collection(bibliography_corpus()))
    response = service.retrieve("reference checklist", mode=RetrievalMode.GUIDANCE)
    assert [candidate.chunk_id for candidate in response.results] == ["guidance-1"]
    assert response.results[0].bibliography_like is True
    assert response.audit.exclude_bibliography is False
    service.close()


def test_explicit_opt_in_returns_bibliography_chunks() -> None:
    """The escape hatch for citation/bibliography analysis."""
    service = build(memory_collection(bibliography_corpus()), exclude_bibliography=False)
    response = service.retrieve("anything", mode=RetrievalMode.EVIDENCE)
    assert any(candidate.bibliography_like for candidate in response.results)
    assert response.audit.exclude_bibliography is False

    per_call = service.retrieve("anything", mode=RetrievalMode.EVIDENCE, exclude_bibliography=True)
    assert all(candidate.bibliography_like is False for candidate in per_call.results)
    service.close()


def test_bibliography_guard_does_not_change_which_prose_is_returned() -> None:
    """Prose results are identical with the guard on and off."""
    client = memory_collection(bibliography_corpus())
    service = build(client)
    with_guard = service.retrieve("anything", mode=RetrievalMode.MY_WORK)
    without = service.retrieve("anything", mode=RetrievalMode.MY_WORK, exclude_bibliography=False)
    assert [c.chunk_id for c in with_guard.results] == ["prose-1"]
    # without the guard the bibliography chunk comes back - ranked first, because the
    # fake reranker scores every candidate equally and dense rank then decides
    assert [c.chunk_id for c in without.results] == ["bib-1", "prose-1"]
    service.close()


def test_bibliography_guard_is_deterministic() -> None:
    client = memory_collection(bibliography_corpus())
    service = build(client)
    first = service.retrieve("anything", mode=RetrievalMode.GENERAL)
    second = service.retrieve("anything", mode=RetrievalMode.GENERAL)
    assert first.chunk_ids == second.chunk_ids
    assert first.stats.dropped_as_bibliography == second.stats.dropped_as_bibliography
    service.close()