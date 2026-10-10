# RAG-002 — End-of-Task Report

**Project:** PhD Academic RAG
**Task ID:** RAG-002 — Bulk file discovery, SHA-256 hashing, folder hints, duplicate detection
**Status:** Complete — awaiting review
**Date:** 2026-10-10

---

## 1. Files created

| File | Responsibility |
|------|----------------|
| `app/models/physical_location.py` | `PhysicalLocation` model |
| `app/ingestion/__init__.py` | Package marker |
| `app/ingestion/file_hash.py` | `calculate_sha256(path)` — chunked, streaming |
| `app/ingestion/folder_hints.py` | Deterministic topic/project hints |
| `app/ingestion/models.py` | `DiscoveredDocument`, `ScanError`, `ScanResult` |
| `app/ingestion/scanner.py` | `scan_directory(root) -> ScanResult`, `SUPPORTED_EXTENSIONS`, `ScanRootError` |
| `app/ingestion/scan.py` | CLI: `python -m app.ingestion.scan <dir>` |
| `tests/test_file_hash.py` | 5 hashing tests |
| `tests/test_physical_location.py` | 5 model tests |
| `tests/test_scanner.py` | discovery / dedup / nested-hint / error tests |

## 2. Files modified

- `app/models/document.py` — added `physical_locations: list[PhysicalLocation] = Field(default_factory=list)` (existing `source_path` / `original_folders` untouched).
- `app/models/__init__.py` — re-export `PhysicalLocation`.
- `tests/test_document_model.py` — added 2 tests for `physical_locations`.

`app/main.py` and the `/health` contract were **not** touched.

## 3. Dependencies added

None. Uses only the standard library (`hashlib`, `os`, `pathlib`, `argparse`) plus existing Pydantic v2.

## 4. Final pytest result

```
68 passed, 1 warning in 0.52s
```

*(The single warning is the pre-existing third-party Starlette/httpx deprecation notice.)*

## 5. Exact supported extension list

`.pdf`, `.docx`, `.doc`, `.rtf`, `.txt`, `.md`, `.xlsx`, `.xls`

- Matching is case-insensitive (extension is lower-cased before comparison).
- Excel and legacy Office files are discovery-only; no contents are opened.

## 6. Exact duplicate-detection behavior

- SHA-256 is authoritative; files are streamed in 1 MiB blocks (never loaded fully into RAM).
- Files sharing a hash collapse into **one** `DiscoveredDocument`: `unique_documents` counts the group once; each additional file increments `duplicate_copies`.
- Every physical path is retained in `physical_locations`; folder labels are added to `original_folders` **deduplicated** (first-seen order).
- `canonical_path` / `filename` come from the **first discovered** copy; `os.walk` sorts directories and files so results are deterministic.
- Same filename + different content → two separate documents (hash differs).
- A per-file `OSError` is recorded in `ScanResult.errors` and scanning continues; an invalid/inaccessible root raises `ScanRootError`.

## 7. Exact folder-hint mappings

| Folder (case-insensitive) | Hint |
|---|---|
| `AI` | topic `artificial_intelligence` |
| `VR` | topic `virtual_reality` |
| `Teaching` | topic `teaching` |
| `Algorithms` | topic `algorithms` |
| `Second paper` | project `RelatedProject.SECOND_PAPER` |

Only `Second paper` produces a project hint — `Algorithms` never yields `SECOND_PAPER`. No other categories are invented. A document accumulates hints from all its physical locations. Hints never replace future semantic classification.

## 8. Example CLI output

```
Root scanned:         C:\Users\User\AppData\Local\Temp\phd_rag_scan_demo
Scanned files:        6
Supported files:      5
Unique documents:     4
Duplicate copies:     1
Unsupported files:    1
Errors:               0

Duplicates:
  43c6481f36fa2d4c2373ddb2c717ad92f94df02aae1601319190f75c51d2169c
    - ...\literature\AI\smith2024.pdf
    - ...\literature\Algorithms\smith2024-copy.pdf
```

## 9. Architectural concerns / ambiguities

1. **No conflicts with RAG-001.** `/health` and all RAG-001 tests still pass unchanged.
2. **Corpus folder layout corrected:** `data\future_chapters\` is a **top-level** folder (the intended architecture), not `data\guidance\future_chapters\`. `data\guidance\` holds a single policy file (`future_chapters_guidance_policy.md`). The scanner is folder-name-agnostic, so both are discovered normally with no hint (none was specified).
3. **`Document` integration is one-directional.** RAG-002 emits `DiscoveredDocument` (discovery-stage type); it does **not** yet populate the enriched `Document` (which requires `document_type`, `publication_status`, `authority`, etc.). That mapping belongs to a later task — flagged so it isn't assumed done.
4. **Minor decisions made** (non-blocking):
   - `extension` is stored lower-cased **with** the leading dot (e.g. `.pdf`).
   - `folder` label = the file's **immediate parent** directory name.
   - `os.walk` sorts entries for determinism.

   Confirm if any of these should differ.

---

## 10. Final review correction — nested folder hints

- `PhysicalLocation.folder` remains the **immediate parent** directory (unchanged).
- Deterministic hints are now evaluated against **every directory component of the file path relative to the scan root** (the scan root itself is excluded), not only the immediate parent.
- Example: `root\literature\AI\adaptive-learning\paper.pdf` → topic hint `artificial_intelligence`; folder label stays `adaptive-learning`.
- Example: `root\literature\Second paper\Algorithms\paper.pdf` → `topic_hint = algorithms`, `related_project_hint = SECOND_PAPER`.
- New tests: `test_nested_folder_hint_uses_ancestor`, `test_nested_second_paper_and_algorithms_hints`, `test_nested_hints_accumulate_from_multiple_ancestors`, `test_nested_hints_accumulate_across_duplicate_locations`.
- Implementation: `app/ingestion/scanner.py::_relative_folder_components` + loop over components.
- No other RAG-002 behavior changed.

## 11. Real corpus scan — `F:\phd-rag\data`

Command:

```
python -m app.ingestion.scan "F:\phd-rag\data"
```

Summary:

```
Root scanned:         F:\phd-rag\data
Scanned files:        501
Supported files:      500
Unique documents:     466
Duplicate copies:     34
Unsupported files:    1
Errors:               0
```

Supported files by extension:

| Extension | Count |
|---|---|
| `.pdf` | 478 |
| `.docx` | 18 |
| `.md` | 3 |
| `.xlsx` | 1 |
| `.doc` | 0 |
| `.xls` | 0 |
| `.rtf` | 0 |
| `.txt` | 0 |

Unsupported (1): `data\literature\Teaching\Student diversity in learning-1.png`

Duplicate groups: **32** (34 extra copies). Duplicates include:

- renamed identical copies (e.g. `economics-ai.pdf` == `saidakhor-Article.pdf`),
- `(1)` / `(2)` download copies,
- cross-folder copies (e.g. `AI\...` == `VR\...`, `AI\...` == `Teaching\...`).

Files by top-level corpus folder:

| Folder | Files |
|---|---|
| `literature` | 475 |
| `future_chapters` | 8 |
| `rules` | 8 |
| `my_work` | 6 |
| `supervision` | 2 |
| `guidance` | 1 |
| `summaries` | 1 |

**Roadmap implication:** the real corpus is effectively PDF + Word + a little Markdown/Excel. There are **zero** `.rtf`, `.txt`, `.doc`, and `.xls` files, so there is no need to build RTF / legacy-DOC / XLS pipelines next.

## Scope confirmation

- No content parsing, chunking, embeddings, Qdrant, Ollama, metadata enrichment, or LLM classification implemented.
- Scanning is strictly read-only: no source file is moved, renamed, edited, or deleted (verified by a test capturing mtime + bytes).
- The scanner **has now been run** against the real `F:\phd-rag\data` corpus (results in section 11).
- RAG-003 has **not** been started.
