"""Deterministic context assembly for one chat turn (RAG-015 §6, §15).

The window is large (262144 tokens on this workstation), so the useful work is not "fit
something in" but "spend it in the right order". The composition is fixed:

```
system instructions
+  conversation history
+  attached working documents (whole documents wherever they fit)
+  optional corpus material, only when it was asked for
+  the current user message
+  output headroom
```

Priority when something has to go is the same order, read from the bottom: the system
instructions and the current message are never trimmed; the most recent conversation is kept
as a contiguous window; attached documents are kept whole and are cut *at their end* only when
the budget truly runs out; corpus material is included after the documents, because the
documents are what the user is working on.

Three properties are enforced rather than assumed:

* the assembled request can never exceed the input budget (the prompt is measured after it is
  rendered and shrunk again if framing pushed it over);
* the input budget can never exceed ``num_ctx`` minus the answer headroom and the safety
  margin, so the window Ollama is asked for always has room for the answer;
* everything left out is recorded in the report — reason, label and tokens — and a truncated
  document says so inside the prompt, so the model is never told a document ends where the
  budget ended.

The estimator is :func:`app.generation.context.estimate_tokens`, the same deterministic
character/word estimate RAG-011 uses, so chat and evidence accounting can be compared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from app.chat.models import (
    AttachmentUsage,
    ChatError,
    ChatMessage,
    ChatRole,
    ContextReport,
    MessageTooLargeError,
    OmittedContextItem,
)
from app.config import Settings, get_settings
from app.generation.context import (
    context_block,
    estimate_tokens,
    evidence_block,
    render_block,
)
from app.generation.models import SourceBlock, SourceClass
from app.generation.ollama import ChatMessage as ModelMessage

#: Per-message overhead counted against the history budget (role markers, framing).
MESSAGE_OVERHEAD_TOKENS = 8
#: How much of a truncated document must survive before a paragraph boundary is preferred.
_BOUNDARY_FLOOR = 0.5
#: How many times the builder may shrink the request before giving up (it converges in one).
_MAX_SHRINK_ROUNDS = 16
#: The line placed inside a truncated document, so the model knows text was omitted.
TRUNCATION_NOTE = (
    "[the rest of this document did not fit the context window and was omitted: "
    "{tokens} tokens]"
)


@dataclass(frozen=True)
class WorkingDocument:
    """One attached document, as stored with the chat."""

    attachment_id: str
    filename: str
    text: str


@dataclass(frozen=True)
class ChatContextRequest:
    """Everything one chat turn needs, kept explicit so the budgeting stays testable."""

    system_prompt: str
    question: str
    history: Sequence[ChatMessage] = ()
    documents: Sequence[WorkingDocument] = ()
    #: A RAG-009 ``RetrievalResponse`` (or anything exposing ``results``) when corpus
    #: material is wanted for this turn.
    corpus: Any | None = None
    corpus_requested: bool = False
    settings: Settings | None = None
    prompt_directory: str | None = None


@dataclass(frozen=True)
class ChatContext:
    """The messages to send, plus how the window was spent."""

    messages: tuple[ModelMessage, ...]
    report: ContextReport
    user_prompt: str = ""


# --- corpus material ----------------------------------------------------------


def block_for_candidate(candidate: Any, source_id: str) -> SourceBlock:
    """Label one retrieved chunk with the project's own corpus-role wording.

    The labelling is the RAG-011 one, deliberately: ``my_work`` and the rules/supervision
    material must never be presented as evidence, and reusing ``evidence_block`` /
    ``context_block`` means the chat inherits that rule instead of restating it.
    """
    role = (getattr(candidate, "corpus_role", None) or "").lower()
    eligible = bool(getattr(candidate, "evidence_eligible", False))
    non_evidence = {"my_work", "supervision", "rules", "guidance"}
    if role == "literature" or (eligible and role not in non_evidence):
        return evidence_block(candidate, source_id)
    if role == "my_work":
        return context_block(candidate, source_id, SourceClass.MY_WORK)
    return context_block(candidate, source_id, SourceClass.GUIDANCE)


def corpus_blocks(corpus: Any) -> tuple[SourceBlock, ...]:
    """Label every retrieved candidate, in retrieval rank order."""
    results = list(getattr(corpus, "results", None) or [])
    return tuple(
        block_for_candidate(candidate, f"[C{index}]")
        for index, candidate in enumerate(results, start=1)
    )


def _corpus_section(blocks: Sequence[SourceBlock], include: Sequence[bool]) -> str:
    """Render the corpus section, with each block's own class header."""
    rendered = [render_block(block) for block, keep in zip(blocks, include) if keep]
    if not rendered:
        return ""
    return (
        "CORPUS MATERIAL (retrieved from your permanent PhD corpus for this message; each "
        "excerpt states its own class)\n\n" + "\n\n".join(rendered)
    )


# --- truncation ---------------------------------------------------------------


def truncate_text_to_tokens(
    text: str, tokens: int, *, chars_per_token: float = 4.0
) -> tuple[str, int, bool]:
    """Cut ``text`` to about ``tokens``, at a paragraph or sentence boundary.

    Returns ``(prefix, omitted_tokens, truncated)``. The cut is always at the *end* of the
    text: a document is never shortened from the middle, because a hole in the middle is
    invisible to the reader of the answer.
    """
    full = estimate_tokens(text, chars_per_token=chars_per_token)
    if tokens <= 0:
        return "", full, bool(text.strip())
    if full <= tokens:
        return text, 0, False

    limit = max(1, int(tokens * chars_per_token))
    prefix = text[:limit]
    cut = _boundary_cut(prefix)
    candidate = (text[:cut] if cut else prefix).rstrip()
    # The estimate is not linear in characters (words count too), so shrink until it fits.
    guard = 0
    while candidate and estimate_tokens(candidate, chars_per_token=chars_per_token) > tokens:
        candidate = candidate[: max(1, int(len(candidate) * 0.92))].rstrip()
        guard += 1
        if guard > 64:  # pragma: no cover - defensive: the loop converges
            candidate = candidate[: max(1, int(tokens * chars_per_token))]
            break
    kept = estimate_tokens(candidate, chars_per_token=chars_per_token)
    return candidate, max(0, full - kept), True


def _boundary_cut(prefix: str) -> int:
    """The best paragraph, sentence or word boundary inside ``prefix`` (or ``0``)."""
    floor = int(len(prefix) * _BOUNDARY_FLOOR)
    paragraph = prefix.rfind("\n\n")
    if paragraph >= floor:
        return paragraph
    sentence = max(
        prefix.rfind(". "), prefix.rfind(".\n"), prefix.rfind("? "), prefix.rfind("! ")
    )
    if sentence >= floor:
        return sentence + 1
    word = prefix.rfind(" ")
    if word >= floor:
        return word
    return 0


# --- the builder --------------------------------------------------------------


def build_chat_context(request: ChatContextRequest) -> ChatContext:
    """Build the messages for one chat turn under a documented budget.

    Raises:
        MessageTooLargeError: when the instructions plus the user's own message do not fit
            the window. That is the one case where nothing sensible can be sent, so it is
            reported rather than silently trimmed (the user's text is never cut).
    """
    settings = request.settings or get_settings()
    num_ctx = int(settings.chat_num_ctx)
    headroom = int(settings.chat_max_output_tokens)
    margin = int(settings.chat_context_margin_tokens)
    window_input = num_ctx - headroom - margin
    if window_input <= 0:  # pragma: no cover - a misconfiguration, not a user error
        raise ChatError(
            "the chat context window is smaller than its output headroom; check "
            "PHD_RAG_CHAT_NUM_CTX and PHD_RAG_CHAT_MAX_OUTPUT_TOKENS"
        )
    budget = min(int(settings.chat_context_tokens), window_input)
    chars_per_token = float(settings.generation_chars_per_token)

    documents = list(request.documents)
    system_tokens = estimate_tokens(request.system_prompt, chars_per_token=chars_per_token)
    question_tokens = estimate_tokens(request.question, chars_per_token=chars_per_token)
    if system_tokens + question_tokens > budget:
        raise MessageTooLargeError(
            f"This message is about {question_tokens:,} tokens, and together with the chat "
            f"instructions it does not fit the {budget:,}-token input budget (window "
            f"{num_ctx:,} tokens). Attach it as a document instead, or send a shorter message."
        )

    # 1. conversation history: newest first, contiguous, so no turn is skipped mid-thread.
    history = list(request.history)
    history_cap = int(budget * float(settings.chat_history_share))
    included_history: list[ChatMessage] = []
    used_history = 0
    for index, message in enumerate(reversed(history)):
        tokens = estimate_tokens(message.content, chars_per_token=chars_per_token)
        tokens += MESSAGE_OVERHEAD_TOKENS
        limit = history_cap if index else max(history_cap, budget - system_tokens - question_tokens)
        if used_history + tokens > limit:
            break
        included_history.append(message)
        used_history += tokens
    included_history.reverse()
    omitted_history = len(history) - len(included_history)

    # 2. attached working documents: whole documents wherever they fit, end-truncated after.
    attachment_cap = min(int(budget * float(settings.chat_attachment_share)), budget)
    allocations: list[int] = []
    available = attachment_cap
    for index, document in enumerate(documents):
        documents_left = len(documents) - index
        share = available // documents_left if documents_left else 0
        allocation = min(
            estimate_tokens(document.text, chars_per_token=chars_per_token), max(0, share)
        )
        allocations.append(allocation)
        available -= allocation

    # 3. corpus material, only when this turn asked for it (rank order, whole blocks).
    blocks = corpus_blocks(request.corpus) if request.corpus is not None else ()
    corpus_cap = min(int(budget * float(settings.chat_corpus_share)), budget)
    include: list[bool] = []
    used_corpus = 0
    for block in blocks:
        keep = used_corpus + block.token_estimate <= corpus_cap
        include.append(keep)
        if keep:
            used_corpus += block.token_estimate

    # 4. any budget left over goes back to the documents, so a big window is actually used.
    used_documents = sum(allocations)
    leftover = budget - system_tokens - question_tokens - used_history - used_corpus - used_documents
    for index, document in enumerate(documents):
        if leftover <= 0:
            break
        needed = (
            estimate_tokens(document.text, chars_per_token=chars_per_token) - allocations[index]
        )
        if needed <= 0:
            continue
        extra = min(needed, leftover)
        allocations[index] += extra
        leftover -= extra
        used_documents += extra

    def render(
        current_allocations: Sequence[int], current_include: Sequence[bool]
    ) -> tuple[str, list[AttachmentUsage]]:
        """Render the turn: the document sections and how much of each was used."""
        sections: list[str] = []
        usages: list[AttachmentUsage] = []
        for index, document in enumerate(documents, start=1):
            prefix, omitted_tokens, truncated = truncate_text_to_tokens(
                document.text,
                current_allocations[index - 1],
                chars_per_token=chars_per_token,
            )
            usages.append(
                AttachmentUsage(
                    attachment_id=document.attachment_id,
                    filename=document.filename,
                    tokens=estimate_tokens(prefix, chars_per_token=chars_per_token),
                    included_chars=len(prefix),
                    total_chars=len(document.text),
                    truncated=truncated,
                )
            )
            if not prefix.strip():
                continue
            if truncated:
                prefix = f"{prefix}\n\n{TRUNCATION_NOTE.format(tokens=omitted_tokens)}"
            sections.append(
                f"WORKING DOCUMENT {index}: {document.filename}\n"
                f"<<<WORKING_DOCUMENT\n{prefix}\nWORKING_DOCUMENT>>>"
            )
        corpus_text = _corpus_section(blocks, current_include)
        return _user_prompt(
            request.prompt_directory, request.question, sections, corpus_text
        ), usages

    # 5. measure what is actually sent and shrink until it fits, giving up corpus material
    #    first and document text second, never the instructions or the user's message.
    prompt = ""
    usages: list[AttachmentUsage] = []
    for _ in range(_MAX_SHRINK_ROUNDS):
        prompt, usages = render(allocations, include)
        sent = system_tokens + used_history + estimate_tokens(
            prompt, chars_per_token=chars_per_token
        )
        if sent <= budget:
            break
        excess = sent - budget
        freed = 0
        for index in reversed(range(len(include))):
            if freed >= excess or not include[index]:
                continue
            include[index] = False
            freed += blocks[index].token_estimate
        used_corpus = sum(block.token_estimate for block, keep in zip(blocks, include) if keep)
        shortfall = excess - freed
        for index in reversed(range(len(allocations))):
            if shortfall <= 0:
                break
            taken = min(allocations[index], shortfall)
            allocations[index] -= taken
            shortfall -= taken
        if shortfall > 0:
            raise MessageTooLargeError(
                "This request does not fit the chat context window even with no document "
                "content. Send a shorter message."
            )
    else:  # pragma: no cover - the shrink loop converges in one or two rounds
        raise ChatError("the chat context could not be made to fit the context window")

    kept_blocks = [block for block, keep in zip(blocks, include) if keep]
    used_corpus = sum(block.token_estimate for block in kept_blocks)
    used_documents = sum(usage.tokens for usage in usages)
    sent_total = system_tokens + used_history + estimate_tokens(
        prompt, chars_per_token=chars_per_token
    )

    omitted = _omissions(
        history=history,
        omitted_history=omitted_history,
        blocks=blocks,
        include=include,
        documents=documents,
        usages=usages,
        chars_per_token=chars_per_token,
    )

    messages: list[ModelMessage] = [ModelMessage(role="system", content=request.system_prompt)]
    for message in included_history:
        messages.append(
            ModelMessage(
                role=(
                    ChatRole.USER.value
                    if message.role == ChatRole.USER
                    else ChatRole.ASSISTANT.value
                ),
                content=message.content,
            )
        )
    messages.append(ModelMessage(role="user", content=prompt))

    report = ContextReport(
        num_ctx=num_ctx,
        input_budget_tokens=budget,
        output_headroom_tokens=headroom,
        system_tokens=system_tokens,
        history_tokens=used_history,
        attachment_tokens=used_documents,
        corpus_tokens=used_corpus,
        current_message_tokens=question_tokens,
        total_tokens=sent_total,
        history_messages_included=len(included_history),
        history_messages_omitted=omitted_history,
        corpus_requested=bool(request.corpus_requested),
        corpus_invoked=bool(kept_blocks),
        corpus_sources=len(kept_blocks),
        attachments=tuple(usages),
        omitted=tuple(omitted),
    )
    return ChatContext(messages=tuple(messages), report=report, user_prompt=prompt)


def _user_prompt(directory: str | None, question: str, sections: Sequence[str], corpus: str) -> str:
    """Lay out the last user message with the chat's own template."""
    from app.chat.prompts import build_user_prompt

    working_documents = ""
    if sections:
        working_documents = (
            "WORKING DOCUMENTS (files the user attached to this conversation)\n\n"
            + "\n\n".join(sections)
        )
    return build_user_prompt(
        question=question,
        working_documents=working_documents,
        corpus=corpus,
        directory=directory,
    )


def _omissions(
    *,
    history: Sequence[ChatMessage],
    omitted_history: int,
    blocks: Sequence[SourceBlock],
    include: Sequence[bool],
    documents: Sequence[WorkingDocument],
    usages: Sequence[AttachmentUsage],
    chars_per_token: float,
) -> list[OmittedContextItem]:
    """Everything that was left out, with the reason it was left out."""
    omitted: list[OmittedContextItem] = []
    if omitted_history:
        omitted.append(
            OmittedContextItem(
                kind="history",
                label=f"{omitted_history} earlier message(s)",
                reason="context limit",
                detail=(
                    "The oldest turns of this conversation no longer fit the history share of "
                    "the context window."
                ),
                tokens=sum(
                    estimate_tokens(message.content, chars_per_token=chars_per_token)
                    + MESSAGE_OVERHEAD_TOKENS
                    for message in history[:omitted_history]
                ),
            )
        )
    for block, keep in zip(blocks, include):
        if keep:
            continue
        omitted.append(
            OmittedContextItem(
                kind="corpus",
                label=block.title or block.filename or block.chunk_id or "corpus excerpt",
                reason="context limit",
                detail=(
                    "This retrieved excerpt did not fit the corpus share of the context "
                    "window; the documents you attached and your message were kept."
                ),
                tokens=block.token_estimate,
            )
        )
    for index, document in enumerate(documents):
        usage = usages[index]
        total_tokens = estimate_tokens(document.text, chars_per_token=chars_per_token)
        if usage.included_chars == 0 and document.text.strip():
            omitted.append(
                OmittedContextItem(
                    kind="attachment",
                    label=document.filename,
                    reason="context limit",
                    detail=(
                        "There was no room left for this document in this request. It stays "
                        "attached to the chat; start a new chat or remove another document to "
                        "make room."
                    ),
                    tokens=total_tokens,
                )
            )
        elif usage.truncated:
            omitted.append(
                OmittedContextItem(
                    kind="attachment",
                    label=document.filename,
                    reason="context limit",
                    detail=(
                        "The document was included from the beginning up to a paragraph "
                        "boundary; the end of it did not fit this request."
                    ),
                    tokens=max(0, total_tokens - usage.tokens),
                )
            )
        elif usage.included_chars == 0:
            omitted.append(
                OmittedContextItem(
                    kind="attachment",
                    label=document.filename,
                    reason="no parsed text",
                    detail=(
                        "This document has no extracted text, so there is nothing to send. "
                        "Check the attachment's parse status."
                    ),
                    tokens=0,
                )
            )
    return omitted


__all__ = [
    "MESSAGE_OVERHEAD_TOKENS",
    "TRUNCATION_NOTE",
    "ChatContext",
    "ChatContextRequest",
    "WorkingDocument",
    "block_for_candidate",
    "build_chat_context",
    "corpus_blocks",
    "truncate_text_to_tokens",
]
