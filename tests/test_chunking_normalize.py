"""RAG-007 tests 1-8: normalization rules for retrieval."""

from __future__ import annotations

from app.chunking.normalize import (
    collapse_whitespace,
    dehyphenate_lines,
    detect_running_lines,
    normalize_block_text,
    normalize_inline,
    normalize_line_endings,
    normalize_unicode,
    strip_running_lines,
)


def test_crlf_and_cr_line_endings_are_normalized() -> None:
    assert normalize_line_endings("a\r\nb\rc\nd") == "a\nb\nc\nd"
    assert normalize_block_text("first line\r\nsecond line") == "first line second line"


def test_repeated_spaces_and_tabs_are_collapsed() -> None:
    assert collapse_whitespace("multiple    spaces\tand\ttabs") == "multiple spaces and tabs"
    assert normalize_inline("  Padded   title \n") == "Padded title"


def test_non_breaking_spaces_are_normalized() -> None:
    assert normalize_block_text("non\u00a0breaking\u202fspace") == "non breaking space"
    assert normalize_inline("2024\u2007results") == "2024 results"


def test_wrapped_prose_lines_are_joined() -> None:
    assert (
        normalize_block_text("The first line of a paragraph\ncontinues on the next line.")
        == "The first line of a paragraph continues on the next line."
    )
    # List items keep their own line instead of being glued to the previous one.
    assert normalize_block_text("- first item\n- second item") == "- first item\n- second item"


def test_line_wrap_hyphen_is_removed() -> None:
    assert normalize_block_text("The impact of educa-\ntion is clear.") == (
        "The impact of education is clear."
    )
    assert dehyphenate_lines(["educa-", "tion"]) == ["education"]
    # A wrapped word split on a typographic hyphen is joined too.
    assert dehyphenate_lines(["informa\u2010", "tion"]) == ["information"]


def test_genuine_compound_hyphens_are_preserved() -> None:
    assert normalize_block_text("AI-based and higher-order thinking") == (
        "AI-based and higher-order thinking"
    )
    assert normalize_block_text("real-time\ncollaboration") == "real-time collaboration"
    # A compound broken at its hyphen before an uppercase continuation keeps it.
    assert normalize_block_text("AI-\nbased methods") == "AI- based methods"
    assert dehyphenate_lines(["higher-order"]) == ["higher-order"]


def test_academic_in_text_citations_are_preserved() -> None:
    text = "Prior work (Smith, 2022) and Smith et al. (2023) agree with the review [14]."
    assert normalize_block_text(text) == text
    inline = "Another view [12, pp. 4-8] appears in the literature (Jones & Lee, 2021)."
    assert normalize_block_text(inline) == inline


def test_unicode_is_preserved_but_ligatures_are_folded() -> None:
    assert normalize_unicode("Café – Başak Melis Öcal ’quoted’ 注意力") == (
        "Café – Başak Melis Öcal ’quoted’ 注意力"
    )
    assert normalize_block_text("The de\ufb01nition of \ufb02ow") == "The definition of flow"
    assert "ﬁ" not in normalize_block_text("de\ufb01nition")


def test_running_headers_and_footers_are_removed_conservatively() -> None:
    pages = [
        (number, f"Journal of Testing\nBody text of page {number}.\n{number}")
        for number in range(1, 7)
    ]
    report = detect_running_lines(pages)
    assert "Journal of Testing" in report.lines

    stripped = strip_running_lines("Journal of Testing\nBody text of page 3.\n3", report)
    assert stripped == "Body text of page 3."

    # A single-page document never triggers running-line removal.
    assert not detect_running_lines([(1, "Journal of Testing\nBody")]).detected
    # A sentence that reappears mid-page is not treated as a running line.
    assert strip_running_lines("Body text of page 3.", report) == "Body text of page 3."
