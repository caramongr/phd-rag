"""Manual retrieval evaluation set for the real PhD corpus (RAG-009).

Every entry is either

* **gold** - one or more documents that are unambiguously *about* the query, named
  by a title or filename substring taken from the actual corpus inventory
  (``docs/rag009-document-inventory.txt``); or
* **exploratory** - no gold document is asserted, because the corpus does not
  clearly contain one. The harness inspects those manually instead of inventing a
  target.

Gold needles are matched against the *returned* result metadata, never against
the retrieval candidates, and the harness verifies that each needle matches at
least one document in the collection - so a typo cannot silently become a
0-score "failure".

Limitations recorded deliberately:

* several works exist in the corpus as multiple physical copies with different
  file hashes, so any copy counts as a hit for its gold needle;
* a gold document that is present but whose relevant passage is not the chunk
  returned still counts as a hit (document-level, not passage-level, judgment).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.models import RetrievalMode


@dataclass(frozen=True)
class EvalQuery:
    """One evaluation query with its (optional) expected documents."""

    query: str
    mode: RetrievalMode = RetrievalMode.EVIDENCE
    category: str = "general"
    gold_titles: tuple[str, ...] = ()
    gold_filenames: tuple[str, ...] = ()
    note: str = ""

    @property
    def is_exploratory(self) -> bool:
        """Whether this query asserts no gold document."""
        return not (self.gold_titles or self.gold_filenames)

    @property
    def label(self) -> str:
        """Short identifier for reports."""
        return self.category


def _q(
    query: str,
    category: str,
    *,
    titles: tuple[str, ...] = (),
    filenames: tuple[str, ...] = (),
    mode: RetrievalMode = RetrievalMode.EVIDENCE,
    note: str = "",
) -> EvalQuery:
    return EvalQuery(
        query=query,
        mode=mode,
        category=category,
        gold_titles=tuple(title.casefold() for title in titles),
        gold_filenames=tuple(name.casefold() for name in filenames),
        note=note,
    )


#: The evaluation set. Topic coverage follows the RAG-009 brief (AI in higher
#: education, VR, adaptive learning, tutoring, engagement, experiential and
#: immersive learning, governance, ethics, integrity, generative AI, learning
#: analytics, DSR, assessment, personalization) plus non-evidence modes.
EVAL_QUERIES: tuple[EvalQuery, ...] = (
    _q(
        "virtual reality student engagement higher education",
        "vr-engagement",
        titles=("the impact of virtual reality on student engagement in higher education",),
    ),
    _q(
        "adaptive learning using artificial intelligence in e-learning literature review",
        "adaptive-learning",
        titles=("adaptive learning using artificial intelligence in e-learning",),
    ),
    _q(
        "generative AI academic integrity and learning outcomes case study",
        "academic-integrity",
        titles=("impact of generative ai in academic integrity and learning outcomes",),
    ),
    _q(
        "design science research methodology for maximum impact",
        "dsr-methodology",
        titles=("positioning and presenting design science research",),
    ),
    _q(
        "AI literacy definition teaching evaluation and ethical issues",
        "ai-literacy",
        titles=("ai literacy: definition, teaching, evaluation and ethical issues",),
    ),
    _q(
        "intelligent tutoring system pedagogical policy induction deep reinforcement learning",
        "intelligent-tutoring",
        titles=("leveraging deep reinforcement learning for pedagogical policy induction",),
    ),
    _q(
        "learning analytics to increase student success course signals",
        "learning-analytics",
        titles=("course signals at purdue",),
    ),
    _q(
        "discourses of artificial intelligence in higher education critical literature review",
        "key-concepts",
        titles=("discourses of artificial intelligence in higher education",),
    ),
    _q(
        "bibliometric study on the impact of artificial intelligence in higher education",
        "bibliometrics",
        titles=("artificial intelligence in higher education: a bibliometric study",),
    ),
    _q(
        "artificial intelligence in higher education the state of the field",
        "ai-in-he",
        titles=("artificial intelligence in higher education: the state of the field",),
    ),
    _q(
        "formative assessment and self-regulated learning feedback principles",
        "assessment",
        titles=("formative assessment and self-regulated learning",),
    ),
    _q(
        "design principles and outcomes of peer assessment in higher education",
        "peer-assessment",
        titles=("design principles and outcomes of peer assessment",),
    ),
    _q(
        "experiential learning in higher education learning styles and learning spaces",
        "experiential-learning",
        titles=("learning styles and learning spaces",),
    ),
    _q(
        "immersive virtual reality for increasing presence and empathy",
        "immersive-vr",
        titles=("immersive virtual reality for increasing presence and empathy",),
    ),
    _q(
        "haptic virtual reality immersive learning organic chemistry instruction",
        "immersive-vr",
        titles=("haptic virtual reality and immersive learning for enhanced organic chemistry",),
    ),
    _q(
        "predictive learning analytics at scale guidelines for implementation in higher education",
        "learning-analytics",
        titles=("predictive learning analytics",),
    ),
    _q(
        "ChatGPT and the end of traditional assessments in higher education",
        "generative-ai",
        titles=("bullshit spewer or the end of traditional assessments",),
    ),
    _q(
        "automating assessment and providing personalized feedback in e-learning template matching",
        "personalization",
        titles=("automating assessment and providing personalized feedback",),
    ),
    _q(
        "AI-enabled virtual reality systems for dental education",
        "vr-medical-education",
        titles=("ai-enabled virtual reality systems for dental education",),
    ),
    _q(
        "generative adversarial networks image synthesis",
        "technical-foundations",
        titles=("generative adversarial nets",),
    ),
    _q(
        "artificial intelligence for virtual reality a review",
        "technical-foundations",
        titles=("artificial intelligence for virtual reality: a review",),
    ),
    _q(
        "self-evolving generative AI tutors for personalized proactive learning",
        "intelligent-tutoring",
        titles=("self-evolving generative ai tutors",),
    ),
    # --- evidence mode, no confident gold document in this corpus ---
    _q(
        "artificial intelligence governance in higher education",
        "governance",
        note="required by the brief; no corpus document is clearly about AI governance",
    ),
    _q(
        "ethical use of artificial intelligence in universities",
        "ethics",
        note="required by the brief; ethics appears inside other papers, not as a topic document",
    ),
    _q(
        "implementation barriers to adopting artificial intelligence in universities",
        "barriers",
    ),
    _q(
        "generative AI policy for university teaching and learning",
        "governance",
    ),
    # --- other retrieval modes ---
    _q(
        "what have I written about the historical development of AI?",
        "my-work-history",
        mode=RetrievalMode.MY_WORK,
        filenames=("historical-development-24-02-2025.docx",),
        note="own draft has no extracted title, so the gold needle is the filename",
    ),
    _q(
        "my paper about AI and VR tools for next generation higher education",
        "my-work-paper",
        mode=RetrievalMode.MY_WORK,
        titles=("intelligent immersion: ai and vr tools",),
    ),
    _q(
        "what did my supervisor say about methodology?",
        "supervision-methodology",
        mode=RetrievalMode.SUPERVISION,
        note="supervision documents are unstructured forms; inspected manually",
    ),
    _q(
        "what are the rules for referencing and citation style in my thesis?",
        "guidance-referencing",
        mode=RetrievalMode.GUIDANCE,
        titles=("reference checklist",),
        note="same document under a second phrasing; the first phrasing misses it (see report)",
    ),
    _q(
        "Harvard referencing style guidelines for my thesis",
        "guidance-referencing",
        mode=RetrievalMode.GUIDANCE,
        titles=("reference checklist",),
    ),
    _q(
        "virtual reality and artificial intelligence in higher education",
        "general-mode",
        mode=RetrievalMode.GENERAL,
        titles=("a systematic review of immersive virtual reality applications for higher education",),
        note="unfiltered mode: checks role is preserved in results",
    ),
    _q(
        "personalized learning recommendation with knowledge tracing",
        "personalization",
    ),
    _q(
        "student engagement and motivation in online higher education",
        "engagement",
    ),
    _q(
        "research design framework for classifying virtual reality studies",
        "immersive-vr",
        titles=("a systematic review of immersive virtual reality applications for higher education",),
        note="the passage also exists in several physical copies of the same paper",
    ),
    _q(
        "supervisor feedback on research design and methodology",
        "supervision-methodology",
        mode=RetrievalMode.SUPERVISION,
        note="second wording for the same non-evidence mode",
    ),
)


def query_set() -> tuple[EvalQuery, ...]:
    """Return the whole evaluation set."""
    return EVAL_QUERIES


def gold_needles() -> tuple[str, ...]:
    """Every distinct gold needle used by the set (for pre-flight validation)."""
    needles: list[str] = []
    for entry in EVAL_QUERIES:
        needles.extend(entry.gold_titles)
        needles.extend(entry.gold_filenames)
    return tuple(dict.fromkeys(needles))


def counts() -> dict[str, int]:
    """Summary counts: total, gold, exploratory, per mode."""
    summary: dict[str, int] = {
        "total": len(EVAL_QUERIES),
        "gold": sum(1 for entry in EVAL_QUERIES if not entry.is_exploratory),
        "exploratory": sum(1 for entry in EVAL_QUERIES if entry.is_exploratory),
    }
    for entry in EVAL_QUERIES:
        key = f"mode_{entry.mode.value}"
        summary[key] = summary.get(key, 0) + 1
    return summary
