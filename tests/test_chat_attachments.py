"""Chat attachments (RAG-015 §5, §6, §16, §17).

The properties that matter: the four formats are parsed with the project's own parsers, the
parsed text is stored with the chat (so a reopen needs no re-upload), the text reaches the
model, a document belongs to one chat only, and an attachment is never corpus material. The
rejections are tested too, because a readable error is part of the feature.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.chat.models import (
    AttachmentLimitError,
    AttachmentTextStatus,
    AttachmentTooLargeError,
    UnsupportedAttachmentError,
)
from tests.chat_support import (
    build_chat,
    docx_bytes,
    markdown_bytes,
    text_bytes,
)

CHAPTER_PARAGRAPHS = (
    "Paragraph one explains the research context of the study.",
    "Paragraph two argues that governance mechanisms shape adoption decisions.",
    "Paragraph three closes the section with the research questions.",
)


def _pdf_bytes() -> bytes:
    """A real one-page PDF with extractable text, so the PDF parser is exercised for real."""
    import fitz

    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72), "A short paper about governance and adoption in artificial intelligence."
    )
    data = document.tobytes()
    document.close()
    return data


def test_a_docx_attachment_is_parsed_and_stored(tmp_path: Path) -> None:
    service, _, settings = build_chat(tmp_path)
    chat = service.new_chat()

    attachment = service.add_attachment(
        chat.id, "chapter.docx", docx_bytes(CHAPTER_PARAGRAPHS, heading="Chapter three")
    )

    assert attachment.extension == ".docx"
    assert attachment.text_status == AttachmentTextStatus.PARSED
    assert attachment.char_count > 0
    text = service.attachment_text(chat.id, attachment.id)
    assert "governance mechanisms shape adoption decisions" in text
    # The file is stored under the chat's own directory, not in the corpus.
    stored = Path(attachment.stored_path)
    assert stored.is_file()
    assert Path(settings.chat_attachment_dir) in stored.parents
    service.close()


def test_attachments_are_isolated_from_the_permanent_corpus(tmp_path: Path) -> None:
    """§5: an attachment is never added to the corpus, and no corpus directory is touched."""
    corpus_root = tmp_path / "data"
    literature = corpus_root / "literature"
    my_work = corpus_root / "my_work"
    literature.mkdir(parents=True)
    my_work.mkdir(parents=True)
    before = sorted(p.name for p in corpus_root.rglob("*"))

    service, _, settings = build_chat(tmp_path)
    chat = service.new_chat()
    attachment = service.add_attachment(chat.id, "paper.pdf", text_bytes("not a pdf"))
    service.close()

    after = sorted(p.name for p in corpus_root.rglob("*"))
    assert before == after  # nothing was written into the corpus tree
    assert not any(str(path).startswith(str(literature)) for path in [attachment.stored_path])
    assert not any(str(path).startswith(str(my_work)) for path in [attachment.stored_path])
    assert str(settings.chat_attachment_dir) in attachment.stored_path


@pytest.mark.parametrize(
    ("name", "data"),
    [
        pytest.param(
            "notes.txt", text_bytes("A plain text note about governance."), id="txt"
        ),
        pytest.param(
            "notes.md", markdown_bytes("# Heading\n\nA markdown note about governance."), id="md"
        ),
        pytest.param(
            "chapter.docx",
            docx_bytes(CHAPTER_PARAGRAPHS, heading="Chapter three"),
            id="docx",
        ),
        pytest.param("paper.pdf", _pdf_bytes(), id="pdf"),
    ],
)
def test_every_supported_format_is_parsed(tmp_path: Path, name: str, data: bytes) -> None:
    service, _, _ = build_chat(tmp_path)
    chat = service.new_chat()

    attachment = service.add_attachment(chat.id, name, data)

    assert attachment.text_status == AttachmentTextStatus.PARSED
    assert "governance" in service.attachment_text(chat.id, attachment.id).lower()
    service.close()


def test_a_unparseable_pdf_is_kept_with_a_readable_status(tmp_path: Path) -> None:
    service, _, _ = build_chat(tmp_path)
    chat = service.new_chat()

    attachment = service.add_attachment(chat.id, "broken.pdf", b"definitely not a pdf")

    assert attachment.text_status == AttachmentTextStatus.FAILED
    assert attachment.parse_error
    assert Path(attachment.stored_path).is_file()  # the file is not thrown away
    service.close()


def test_parsed_text_survives_reopening_the_chat(tmp_path: Path) -> None:
    service, _, _ = build_chat(tmp_path)
    chat = service.new_chat()
    attachment = service.add_attachment(
        chat.id, "feedback.docx", docx_bytes(CHAPTER_PARAGRAPHS, heading="Supervisor feedback")
    )

    reopened = service.get_chat(chat.id)

    assert [item.id for item in reopened.attachments] == [attachment.id]
    assert "governance" in service.attachment_text(chat.id, attachment.id).lower()
    service.close()


def test_attachment_text_reaches_the_model_prompt(tmp_path: Path) -> None:
    """§17 example 1: the question refers to paragraph two, and paragraph two is in the prompt."""
    service, model, _ = build_chat(tmp_path)
    chat = service.new_chat("Chapter three")
    service.add_attachment(
        chat.id, "chapter.docx", docx_bytes(CHAPTER_PARAGRAPHS, heading="Chapter three")
    )

    service.send_message(chat.id, "Is governance a fit for paragraph two?")

    prompt = model.last_prompt
    assert "WORKING DOCUMENT 1: chapter.docx" in prompt
    assert "Paragraph two argues that governance mechanisms shape adoption decisions." in prompt
    assert "Is governance a fit for paragraph two?" in prompt
    service.close()


def test_an_attachment_belongs_only_to_its_own_chat(tmp_path: Path) -> None:
    service, _, _ = build_chat(tmp_path)
    chat_a = service.new_chat("A")
    chat_b = service.new_chat("B")

    attachment = service.add_attachment(chat_a.id, "notes.txt", text_bytes("only for A"))

    assert service.list_attachments(chat_b.id) == ()
    assert service.store.attachment_text(chat_b.id, attachment.id) == ""
    assert service.delete_attachment(chat_b.id, attachment.id) is None
    service.close()


def test_unsupported_extension_is_rejected_with_a_readable_error(tmp_path: Path) -> None:
    service, _, _ = build_chat(tmp_path)
    chat = service.new_chat()

    with pytest.raises(UnsupportedAttachmentError) as raised:
        service.add_attachment(chat.id, "spreadsheet.xlsx", b"whatever")

    message = raised.value.detail
    assert "xlsx" not in message.lower() or "supported" in message.lower()
    assert "PDF" in message and "DOCX" in message and "TXT" in message and "MD" in message
    assert service.list_attachments(chat.id) == ()
    service.close()


def test_oversized_attachment_is_rejected_cleanly(tmp_path: Path) -> None:
    service, _, _ = build_chat(tmp_path, chat_max_attachment_bytes=1024 * 1024)
    chat = service.new_chat()

    with pytest.raises(AttachmentTooLargeError) as raised:
        service.add_attachment(chat.id, "big.txt", b"x" * (2 * 1024 * 1024))

    detail = raised.value.detail
    assert "limit for one attachment is 1.0 MB" in detail
    assert "2.0 MB" in detail
    assert service.list_attachments(chat.id) == ()
    service.close()


def test_empty_attachment_is_rejected(tmp_path: Path) -> None:
    service, _, _ = build_chat(tmp_path)
    chat = service.new_chat()
    with pytest.raises(UnsupportedAttachmentError):
        service.add_attachment(chat.id, "empty.txt", b"")
    service.close()


def test_the_attachment_count_is_limited_per_chat(tmp_path: Path) -> None:
    service, _, _ = build_chat(tmp_path, chat_max_attachments_per_chat=2)
    chat = service.new_chat()
    service.add_attachment(chat.id, "one.txt", text_bytes("one"))
    service.add_attachment(chat.id, "two.txt", text_bytes("two"))

    with pytest.raises(AttachmentLimitError):
        service.add_attachment(chat.id, "three.txt", text_bytes("three"))
    service.close()


def test_a_hostile_filename_cannot_escape_the_attachment_directory(tmp_path: Path) -> None:
    """§16 security: the stored path is built from this app's own identifiers."""
    service, _, settings = build_chat(tmp_path)
    chat = service.new_chat()
    hostile = "..\\..\\..\\Windows\\System32\\evil.txt"

    attachment = service.add_attachment(chat.id, hostile, text_bytes("hostile name"))

    stored = Path(attachment.stored_path).resolve()
    root = Path(settings.chat_attachment_dir).resolve()
    assert root in stored.parents
    assert attachment.filename == "evil.txt"  # displayed, never used as a path
    assert not (tmp_path / "evil.txt").exists()
    service.close()


def test_deleting_a_chat_removes_its_stored_files(tmp_path: Path) -> None:
    service, _, _ = build_chat(tmp_path)
    chat = service.new_chat()
    attachment = service.add_attachment(chat.id, "notes.txt", text_bytes("notes"))
    directory = Path(attachment.stored_path).parent

    service.delete_chat(chat.id)

    assert not Path(attachment.stored_path).exists()
    assert not directory.exists()
    service.close()


def test_detaching_one_attachment_removes_only_that_file(tmp_path: Path) -> None:
    service, _, _ = build_chat(tmp_path)
    chat = service.new_chat()
    first = service.add_attachment(chat.id, "one.txt", text_bytes("one"))
    second = service.add_attachment(chat.id, "two.txt", text_bytes("two"))

    removed = service.delete_attachment(chat.id, first.id)

    assert removed is not None and removed.id == first.id
    assert not Path(first.stored_path).exists()
    assert Path(second.stored_path).is_file()
    assert [item.id for item in service.list_attachments(chat.id)] == [second.id]
    service.close()
