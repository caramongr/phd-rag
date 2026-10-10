# phd-rag

PhD Academic RAG — backend for ingesting and parsing a personal academic corpus.

## Layout

| Path | Contents |
|------|----------|
| `app/` | FastAPI backend: ingestion (scanning, hashing, folder/topic hints) and parsing (PDF via PyMuPDF with a selective Tesseract OCR fallback, DOCX, Markdown) |
| `tests/` | pytest suite (the OCR engine is mocked, so the suite runs without Tesseract) |
| `RAG-00N.md` | Task briefs |
| `RAG-00N-report.md` | End-of-task reports |
| `docs/` | Reserved for project documentation |

## Setup

Requires Python 3.12+.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip install pymupdf python-docx pytesseract Pillow
.\.venv\Scripts\python.exe -m pytest -q
```

`pyproject.toml` currently declares only `fastapi` and `pydantic`; the parsing/OCR
extras above are required at runtime but not yet declared as dependencies.

The OCR fallback needs a local Tesseract installation (never downloaded
automatically, no cloud OCR). Configuration lives in `app/config.py` and is
overridable through `PHD_RAG_*` environment variables:

```
PHD_RAG_OCR_ENABLED=true
PHD_RAG_OCR_LANGUAGE=eng
PHD_RAG_OCR_DPI=300
PHD_RAG_OCR_MIN_NATIVE_CHARS=50
```

## CLI

```powershell
# Parse one document and print a summary
.\.venv\Scripts\python.exe -m app.parsing.parse "paper.pdf"

# Native extraction only
.\.venv\Scripts\python.exe -m app.parsing.parse "paper.pdf" --no-ocr

# Report local OCR availability, version and installed languages
.\.venv\Scripts\python.exe -m app.parsing.parse --ocr-info
```

## Not tracked

`data/` is excluded from git: it holds the document corpus (third-party published
PDFs plus personal drafts and supervision material), which is not redistributed.
