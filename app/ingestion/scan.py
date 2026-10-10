"""Command-line entry point for scanning the research corpus.

Usage::

    python -m app.ingestion.scan "F:\\phd-rag\\data"

The command is strictly read-only. It prints an inventory summary and, when
binary duplicates exist, the physical paths that share each hash.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from app.ingestion.scanner import ScanRootError, scan_directory


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.ingestion.scan",
        description="Read-only discovery and duplicate inventory for the corpus.",
    )
    parser.add_argument("root", type=Path, help="Directory to scan recursively.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the scanner and print a human-readable report. Returns an exit code."""
    args = _build_parser().parse_args(argv)
    root: Path = args.root

    try:
        result = scan_directory(root)
    except ScanRootError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Root scanned:         {root}")
    print(f"Scanned files:        {result.scanned_files}")
    print(f"Supported files:      {result.supported_files}")
    print(f"Unique documents:     {result.unique_documents}")
    print(f"Duplicate copies:     {result.duplicate_copies}")
    print(f"Unsupported files:    {result.unsupported_files}")
    print(f"Errors:               {len(result.errors)}")

    duplicates = [doc for doc in result.documents if len(doc.physical_locations) > 1]
    if duplicates:
        print()
        print("Duplicates:")
        for document in duplicates:
            print(f"  {document.file_hash}")
            for location in document.physical_locations:
                print(f"    - {location.path}")

    if result.errors:
        print()
        print("Errors:")
        for error in result.errors:
            print(f"  - {error.path}: {error.message}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
