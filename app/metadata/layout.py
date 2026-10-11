"""Layout primitives shared by the page readers and the title extractor.

A :class:`PageLine` is one text line on a page with just enough geometry to make
conservative typographic decisions (is this line prominent? is it near the top
of the page?). Font size is used here only for metadata extraction, not for
structural parsing.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class PageLine(BaseModel):
    """One text line with normalized position and dominant font size."""

    model_config = ConfigDict(extra="forbid")

    text: str
    size: float
    #: Relative vertical position of the line's top edge, 0.0 (page top) to 1.0.
    top_fraction: float
    #: Relative vertical position of the line's bottom edge, 0.0 to 1.0.
    bottom_fraction: float
    #: 0-based reading order within the page.
    index: int
    bold: bool = False


class PageLayout(BaseModel):
    """The ordered lines of a single page."""

    model_config = ConfigDict(extra="forbid")

    page_number: int
    lines: list[PageLine]
    width: float
    height: float
