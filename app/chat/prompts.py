"""Prompt templates for the simple chat (RAG-015 §6, §7).

The templates live in ``app/chat/templates`` as plain text, like the RAG-011 ones, so the
instructions the model receives can be read and edited without touching code. They are loaded
through :func:`app.generation.prompts.load_template`, so the version string a chat reports is
derived from the file name *and* a hash of its contents in exactly the same way as the
evidence prompts.

Two templates, and neither of them forces the RAG-011 behaviour: there is no marker
requirement, no reference list, and no citation-validation step. The chat answers the user's
instruction, with the material the user gave it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.generation.prompts import PromptTemplate, load_template

PROMPT_DIR = Path(__file__).resolve().parent / "templates"

#: Template file stems. Bumping the name is the human-visible version change.
#: ``system_v2`` adds the brevity rule: answers are short by default, which is the single
#: biggest lever on how long a turn takes (generation, not context, dominates the waiting).
SYSTEM_TEMPLATE = "system_v2"
USER_TEMPLATE = "user_v1"


@lru_cache(maxsize=8)
def system_template(directory: str | None = None) -> PromptTemplate:
    """The chat system prompt (how to behave, what the supplied material is)."""
    return load_template(SYSTEM_TEMPLATE, Path(directory) if directory else PROMPT_DIR)


@lru_cache(maxsize=8)
def user_template(directory: str | None = None) -> PromptTemplate:
    """The chat user prompt (working documents, corpus material, the user's message)."""
    return load_template(USER_TEMPLATE, Path(directory) if directory else PROMPT_DIR)


def prompt_version(directory: str | None = None) -> str:
    """The combined version of the chat prompt pair."""
    return f"{system_template(directory).version}+{user_template(directory).version}"


def build_system_prompt(directory: str | None = None) -> str:
    """The system message: fixed instructions, never trimmed."""
    return system_template(directory).text


def build_user_prompt(
    *,
    question: str,
    working_documents: str = "",
    corpus: str = "",
    directory: str | None = None,
) -> str:
    """Render the turn that carries the context and the user's message.

    The sections are built by :mod:`app.chat.context` (that is where budgets and labelling
    live); this function only lays them out, in the documented order: working documents, then
    corpus material, then the user's own message.
    """
    template = user_template(directory)
    return template.text.format(
        working_documents=_section(working_documents),
        corpus=_section(corpus),
        question=question.strip(),
    )


def _section(text: str) -> str:
    """One non-empty section, followed by a blank line."""
    clean = (text or "").strip()
    return f"{clean}\n\n" if clean else ""


__all__ = [
    "PROMPT_DIR",
    "SYSTEM_TEMPLATE",
    "USER_TEMPLATE",
    "build_system_prompt",
    "build_user_prompt",
    "prompt_version",
    "system_template",
    "user_template",
]
