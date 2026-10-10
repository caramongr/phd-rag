# RAG-003 — End-of-Task Report

**Project:** PhD Academic RAG
**Task ID:** RAG-003 — Structured DOCX + Markdown parsing (structure preserved, no chunking)
**Status:** Complete — awaiting review
**Date:** 2026-10-10

---

## 1. Files created

| File | Responsibility |
|------|----------------|
| `app/parsing/__init__.py` | Package marker |
| `app/parsing/models.py` | `ParsedDocument`, `ParsedBlock`, `ParsedTable`, `ParsedTableRow`, `ParsedTableCell`, `ParsedBlockType` |
| `app/parsing/exceptions.py` | `DocumentParseError`, `UnsupportedDocumentFormat` |
| `app/parsing/docx_parser.py` | `parse_docx(path, file_hash=None)` — order-preserving DOCX parsing |
| `app/parsing/markdown_parser.py` | `parse_markdown(path, file_hash=None)` — Markdown parsing |
| `app/parsing/parser.py` | `parse_document(path, file_hash=None)` dispatcher |
| `app/parsing/parse.py` | CLI: `python -m app.parsing.parse <file>` |
| `tests/test_docx_parser.py` | DOCX tests |
| `tests/test_markdown_parser.py` | Markdown tests |
| `tests/test_parser_dispatcher.py` | Dispatcher tests |

## 2. Files modified

None. Existing RAG-001 and RAG-002 code and tests are untouched.

## 3. Dependencies added

None. `python-docx 1.2.0` was already present in the venv. All other code uses the standard library.

## 4. Final pytest result

```
118 passed, 1 warning in 1.45s
```

Covers RAG-001 (health, models), RAG-002 (scanner), and RAG-003 (parsing). The warning is the pre-existing third-party Starlette/httpx deprecation notice.

## 5. Exact DOCX structure preserved

- **Paragraph order + table order**: preserved by walking the body XML children directly (`w:p`, `w:tbl`) rather than reading `document.paragraphs` then `document.tables`.
- **Headings**: Word styles matching `Heading <n>` (case-insensitive) → `HEADING` block with `heading_level = n`.
- **Title**: only from a paragraph styled `Title` (level 0) or, as fallback, `core_properties.title`. Never guessed from body text.
- **Paragraphs**: non-empty paragraphs → `PARAGRAPH` blocks.
- **List items**: `List*` style or Word numbering (`numPr`) → `LIST_ITEM`.
- **Quotes**: `Quote*` style → `QUOTE`.
- **Tables**: preserved separately with row/cell text (never flattened away).
- **Empty paragraphs**: skipped (no meaningless blocks). `paragraph_index` still counts every source paragraph for deterministic provenance.

## 6. Exact Markdown structure preserved

- **Headings** `#`…`######` → `HEADING` with correct level.
- **Paragraphs**: consecutive non-blank lines joined into one paragraph.
- **List items** (`-`, `*`, `+`, `1.`, `1)`) → `LIST_ITEM`, marker retained.
- **Block quotes** (`>`) grouped into a single `QUOTE` block.
- **Tables**: simple pipe-style tables with a delimiter row → `ParsedTable` (header + body rows).
- **Encoding**: UTF-8 with BOM tolerated (`utf-8-sig`); CRLF/CR normalised to LF.
- Inline emphasis and citation syntax are **not** stripped; Markdown is never rendered to HTML.

## 7. How heading hierarchy is represented

- Maintained as a `dict[int, str]` (heading level → text).
- On a heading of level *L*: all levels ≥ *L* are dropped, then *L* is set.
- Every block stores `heading_path` = the ordered list of currently active headings.
- A heading block's own `heading_path` **includes itself**.
- A `Title`-styled paragraph occupies **level 0**, so it roots the path of everything below it.

Example:

```
Heading 1: Chapter 4
Heading 2: Development of VR
Heading 3: Early immersive systems   -> path [Chapter 4, Development of VR, Early immersive systems]
Heading 2: Evidence of educational use -> path [Chapter 4, Evidence of educational use]  (H3 cleared)
```

## 8. How DOCX paragraph/table order is preserved

`docx_parser._iter_body_items` iterates `document.element.body.iterchildren()` and yields a `Paragraph` for each `w:p` and a `Table` for each `w:tbl`, using `docx.oxml.ns.qn`. This small explicit helper avoids reading all paragraphs first and all tables afterwards, so a table stays next to its subsection. No extra dependency was introduced.

## 9. How raw text is generated

- **DOCX**: blocks and rendered tables are appended in true document order and joined with `"\n\n"`. Tables appear as a readable grid (`cell | cell`), while the structured `ParsedTable` is retained separately.
- **Markdown**: `raw_text` is the decoded source **verbatim** (BOM removed, line endings normalised to LF) — the most faithful representation and ideal for later re-chunking without reparsing.
- No aggressive normalisation; only per-paragraph `strip()` and newline normalisation.

## 10. Example CLI summary

```
File: data\rules\writing_guidelines.md
Format: .md
Title: Academic Writing Guidelines
Blocks: 77
Tables: 0
Headings: 11
Characters: 5368
```

(`--show-blocks N` optionally prints one-line previews of the first N blocks. The full document is never dumped.)

## 11. Real-corpus validation results (read-only)

| File | Format | Title | Blocks | Headings | Tables | Errors |
|------|--------|-------|--------|----------|--------|--------|
| `data\rules\writing_guidelines.md` | .md | Academic Writing Guidelines | 77 | 11 | 0 | none |
| `data\rules\phd_direction_and_supervisor_plan.md` | .md | PhD Direction and Supervisor Plan | 98 | 14 | 0 | none |
| `data\guidance\future_chapters_guidance_policy.md` | .md | Future Chapters Guidance Policy | 88 | 12 | 0 | none |
| `data\my_work\current_paper\second-paper-draft-final.docx` | .docx | — | 353 | 0 | 0 | none |
| `data\future_chapters\Chapter 7.docx` | .docx | — | 423 | 0 | 2 | none |
| `data\future_chapters\chapter4.docx` | .docx | — | 136 | 0 | 1 | none |
| `data\rules\plan.docx` | .docx | — | 0 | 0 | 1 | none |

No source file was modified.

### Key finding — the real DOCX files carry no structure

Probing the three thesis DOCX files showed:

- **Every** non-empty paragraph uses style `Normal` (423/423, 136/136, 353/353).
- **No** `w:outlineLvl` on any paragraph.
- **No** `core_properties.title`.
- The only visual signal is that some paragraphs are **fully bold** (58, 2, and 46 respectively).

Consequently these DOCX files parse into flat paragraph lists with `headings = 0`. The Markdown files parse with correct structure.

## 12. Parsing limitations discovered

1. **Word headings are unusable for the real corpus** — the authoring tool that produced the thesis DOCX files did not apply heading styles or outline levels, so faithful parsing (per the brief) cannot recover chapter/section structure for them.
2. **Markdown tables are structured; DOCX tables are structured** — both supported for the simple/reliable cases. Complex merged-cell DOCX tables are preserved as cell text only (no span semantics).
3. **`plan.docx` has a table but no body text** → 0 blocks, 1 table. Correct, not an error.
4. **Markdown `raw_text` is verbatim source** (includes markup); DOCX `raw_text` is synthesised. This asymmetry is intentional (see §9).

## 13. Architectural concerns / ambiguities

1. **Structural decision requiring your call (not implemented):** Should a later task detect DOCX headings from **bold runs / font size / heuristic outline**? That is inference, so I did **not** implement it — the brief says to stop and report rather than decide. Without it, the thesis DOCX files have no recoverable heading hierarchy.
2. **Title at heading level 0**, and thus included in every `heading_path`. Confirmed as a deliberate choice; easy to change if you prefer titles excluded from the path.
3. **Interleaving representation:** `blocks` and `tables` are separate ordered lists, so a table's position *between* two paragraphs is captured in `raw_text` (and via `heading_path` / `source_locator`), not by a single global index. The suggested model had no unified order field.
4. **Folder layout clarified (correction):** `data\future_chapters\` is the **intended top-level** folder. `data\guidance\` holds a single policy file. No files were moved.

---

## 14. Final review corrections (applied)

### 14.1 Heading provenance: `EXPLICIT` vs `INFERRED`

- New `HeadingSource` enum (`explicit`, `inferred`).
- `ParsedBlock.heading_source` records how each heading was found; non-headings are `None`.
- Explicit Word `Heading N` styles and Markdown `#` → `EXPLICIT` (authoritative). The conservative DOCX fallback → `INFERRED`.

### 14.2 Conservative deterministic DOCX heading inference

Applied **only** when explicit heading metadata is absent. A paragraph qualifies only if it is **fully bold** *and* matches a strong structural pattern (`Chapter 4`, `4`, `4.1`, `4.1.2`). Rules:

- boldness alone is **never** sufficient;
- must not look like prose (> 100 chars, > 12 words, or ending in `.` / `!` / `?`);
- numbered components limited to ≤ 3 digits (rejects years such as `2024`);
- any trailing title text must start uppercase/digit and be ≤ 8 words;
- level = number of dotted components (capped at 6); `Chapter N` / `Appendix N` → level 1;
- no LLM; font sizes deliberately **not** used (unreliable here).

### 14.3 Canonical ordered representation

- Both `ParsedBlock` and `ParsedTable` now carry a `sequence_index`.
- `ParsedDocument.content_items: list[ContentItem]` is the canonical, globally-ordered stream; `ContentItem` has `sequence_index`, `item_type` (`block`/`table`) and the item itself.
- `paragraph, paragraph, table, paragraph` is preserved **exactly**. The `blocks` and `tables` lists remain for convenience only.

### 14.4 Re-parse of the real DOCX files (post-correction, read-only)

| File | Explicit headings | Inferred headings | Normal paragraphs | Tables |
|------|-------------------|-------------------|-------------------|--------|
| `second-paper-draft-final.docx` | 0 | 40 | 313 | 0 |
| `Chapter 7.docx` | 0 | 51 | 372 | 2 |
| `chapter4.docx` | 0 | 0 | 136 | 1 |

No suspicious heading classifications were observed: every inferred heading is a genuine numbered section title (e.g. `2. Materials and Methods`, `7.3.1 Mapping data collection and movement`).

### 14.5 Noteworthy classifications and limitations

- **`4.0 Foundations of Extended Reality (XR) Architectures`** in `second-paper-draft-final.docx` is inferred as **level 2** because the source numbers it `4.0` (two dotted components). Correct per the numbering rule, but the section is semantically level-1; the inconsistency is in the source numbering, not the parser.
- **Missing headings**: some real headings are not inferred because they are neither fully bold nor numbered — e.g. `3.3`, `4.4`, `5.2`, `5.3`, `5.5` in the second paper, and `7.1` / `7.2` in `Chapter 7.docx`. Inference is intentionally conservative, so recall is lower than for an explicitly-styled document.
- **`chapter4.docx` yields 0 headings**: none of its headings are bold+numbered, so it parses as flat paragraphs. Full structural recovery there would require a font-size/visual heuristic, which is out of scope and was deliberately not implemented.

---

## Scope confirmation

- No PDF parsing, Excel parsing, `.doc`/`.rtf`/`.txt` parsing, OCR, chunking, token counting, embeddings, Qdrant, Ollama, metadata enrichment, or LLM classification implemented.
- `ParsedBlock` is **not** a `Chunk`; no retrieval grouping performed.
- All parsing is strictly read-only.
- `GET /health` and RAG-002 scanning are unchanged.
- `python -m app.parsing.parse` works for both `.docx` and `.md`.
- RAG-004 has **not** been started.
