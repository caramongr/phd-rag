"""Corpus-role and evidence policy for chunks (RAG-007).

The role vocabulary is RAG-006's :class:`~app.metadata.enums.CorpusRole`; the
authority vocabulary is RAG-001's :class:`~app.models.enums.AuthorityLevel`.
Neither is re-invented here. What this module adds is the *policy* those roles
imply for retrieval and evidence, so a chunk can always answer "may this text be
used as scholarly evidence?".

Roles are never collapsed into one conceptual vector corpus:

===================  =========  =========  ===================  ==========================
role                 retrieval  evidence   authority            meaning
===================  =========  =========  ===================  ==========================
literature           yes        yes        primary_evidence     scholarly evidence
my_work              yes        no         high                 terminology continuity,
                                                                 consistency checking,
                                                                 never external evidence
summaries            yes        no         navigation_only      navigation to the original
supervision          yes        no         medium               supervisory decisions
rules                yes        no         instruction          instruction context
guidance             yes        no         instruction          instruction context
future_chapters      yes        no         navigation_only      planning guidance only
other                yes        no         navigation_only      unknown provenance is never
                                                                 treated as evidence
===================  =========  =========  ===================  ==========================

``future_chapters`` therefore carries the RAG-007 requirement verbatim:
NAVIGATION_ONLY + NOT_EVIDENCE, used as planning guidance. ``my_work`` is
retrievable (it is the author's own framework) but is not external evidence.

Reference lists are a *content-type* override rather than a role: a bibliography
inside a literature paper is not retrieval-eligible for normal evidence queries
regardless of the document's role.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.metadata.enums import CorpusRole
from app.models.enums import AuthorityLevel, ContentType


@dataclass(frozen=True)
class RolePolicy:
    """What a corpus role is allowed to do in retrieval and evidence."""

    retrieval_eligible: bool
    evidence_eligible: bool
    authority: AuthorityLevel
    meaning: str


ROLE_POLICIES: dict[CorpusRole, RolePolicy] = {
    CorpusRole.LITERATURE: RolePolicy(
        retrieval_eligible=True,
        evidence_eligible=True,
        authority=AuthorityLevel.PRIMARY_EVIDENCE,
        meaning="scholarly literature: retrievable and usable as evidence",
    ),
    CorpusRole.MY_WORK: RolePolicy(
        retrieval_eligible=True,
        evidence_eligible=False,
        authority=AuthorityLevel.HIGH,
        meaning=(
            "the author's own work: retrievable for terminology continuity and "
            "consistency checking, never external evidence"
        ),
    ),
    CorpusRole.SUMMARIES: RolePolicy(
        retrieval_eligible=True,
        evidence_eligible=False,
        authority=AuthorityLevel.NAVIGATION_ONLY,
        meaning="internal summaries: navigation to the original, never cited as evidence",
    ),
    CorpusRole.SUPERVISION: RolePolicy(
        retrieval_eligible=True,
        evidence_eligible=False,
        authority=AuthorityLevel.MEDIUM,
        meaning="supervision records: context for supervisory decisions, not scholarly evidence",
    ),
    CorpusRole.RULES: RolePolicy(
        retrieval_eligible=True,
        evidence_eligible=False,
        authority=AuthorityLevel.INSTRUCTION,
        meaning="rules and regulations: instruction context, not scholarly evidence",
    ),
    CorpusRole.GUIDANCE: RolePolicy(
        retrieval_eligible=True,
        evidence_eligible=False,
        authority=AuthorityLevel.INSTRUCTION,
        meaning="guidance material: instruction context, not scholarly evidence",
    ),
    CorpusRole.FUTURE_CHAPTERS: RolePolicy(
        retrieval_eligible=True,
        evidence_eligible=False,
        authority=AuthorityLevel.NAVIGATION_ONLY,
        meaning=(
            "future chapters: NAVIGATION_ONLY planning guidance, NOT_EVIDENCE - "
            "they may help locate ideas but must never support a scholarly claim"
        ),
    ),
    CorpusRole.OTHER: RolePolicy(
        retrieval_eligible=True,
        evidence_eligible=False,
        authority=AuthorityLevel.NAVIGATION_ONLY,
        meaning="unclassified provenance: retrievable, never treated as evidence",
    ),
}


def policy_for_role(role: CorpusRole | None) -> RolePolicy:
    """Return the policy for ``role`` (unknown roles are treated conservatively)."""
    if role is None:
        return ROLE_POLICIES[CorpusRole.OTHER]
    return ROLE_POLICIES.get(role, ROLE_POLICIES[CorpusRole.OTHER])


def content_type_allows_retrieval(content_type: ContentType) -> bool:
    """Whether a content type may answer normal (non-explicit) retrieval.

    Only reference lists are excluded: a bibliography entry must never be
    retrieved as if it were a substantive claim of the paper.
    """
    return content_type is not ContentType.REFERENCES


def eligibility_for(
    role: CorpusRole | None, content_type: ContentType
) -> tuple[bool, bool]:
    """Return ``(retrieval_eligible, evidence_eligible)`` for a chunk."""
    policy = policy_for_role(role)
    retrieval = policy.retrieval_eligible and content_type_allows_retrieval(content_type)
    evidence = policy.evidence_eligible and content_type_allows_retrieval(content_type)
    return retrieval, evidence
