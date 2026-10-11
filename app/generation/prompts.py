"""Prompt templates kept in plain text so they can be read and edited directly (§22).

Templates live in ``app/generation/templates`` as ``.txt`` files. The version recorded
in every audit is derived from the file name *and* a hash of the file contents, so an
edited template can never be reported under an unchanged version string.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "templates"

#: Template file stems. Bumping the name is the human-visible version change.
#: v1 -> v2: the prompt now requires a marker on *every* claim sentence including the
#: opening and any summary sentence, and forbids "the literature agrees" style wording
#: (RAG-011 §33 records the failure that prompted the change).
SYSTEM_TEMPLATE = "system_v2"
USER_TEMPLATE = "user_v2"


class PromptError(RuntimeError):
    """A template is missing or malformed."""


@dataclass(frozen=True)
class PromptTemplate:
    """One loaded template plus its verifiable version."""

    name: str
    text: str
    sha256: str

    @property
    def version(self) -> str:
        """``name:hash8``, e.g. ``system_v1:4a91c2ef``."""
        return f"{self.name}:{self.sha256[:8]}"


def _load(name: str, directory: Path | None = None) -> PromptTemplate:
    path = (directory or PROMPT_DIR) / f"{name}.txt"
    if not path.is_file():
        raise PromptError(f"prompt template not found: {path}")
    # Read as text so line endings are normalised before hashing: the hash then
    # identifies exactly the characters that get sent to the model, and an editor that
    # only rewrites CRLF does not look like a content change.
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise PromptError(f"prompt template is empty: {path}")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return PromptTemplate(name=name, text=text, sha256=digest)


@lru_cache(maxsize=8)
def system_template(directory: str | None = None) -> PromptTemplate:
    """The system prompt (grounding rules, untrusted-content rules, style)."""
    return _load(SYSTEM_TEMPLATE, Path(directory) if directory else None)


@lru_cache(maxsize=8)
def user_template(directory: str | None = None) -> PromptTemplate:
    """The user prompt (context, question, output requirements)."""
    return _load(USER_TEMPLATE, Path(directory) if directory else None)


def prompt_version(directory: str | None = None) -> str:
    """The combined version of the prompt pair, recorded in every audit."""
    return f"{system_template(directory).version}+{user_template(directory).version}"


def build_user_prompt(
    *,
    context: str,
    question: str,
    length_instruction: str,
    extra_instructions: str = "",
    directory: str | None = None,
) -> str:
    """Render the user message.

    ``extra_instructions`` is a small list of additional, *system-authored* lines (for
    example "you may use my_work for terminology, not as evidence"). It never contains
    text taken from a document.
    """
    template = user_template(directory)
    extras = f"- {extra_instructions.strip()}\n" if extra_instructions.strip() else ""
    return template.text.format(
        context=context.rstrip(),
        question=question.strip(),
        length_instruction=length_instruction.strip(),
        extra_instructions=extras,
    )


def build_system_prompt(directory: str | None = None) -> str:
    """The system message."""
    return system_template(directory).text
