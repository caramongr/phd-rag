"""Run the local UI: ``python -m app.web`` (RAG-014 §22).

Prints the exact URL to open, then serves. Binding to anything other than ``127.0.0.1``
requires the explicit ``--host`` flag, and says so when it happens.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from app.web import DEFAULT_HOST, DEFAULT_PORT, create_web_app


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the (deliberately few) command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m app.web",
        description=(
            "Serve the local PhD RAG interface (FastAPI + static HTML/JS). Localhost only "
            "unless --host is given."
        ),
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"interface to bind (default: {DEFAULT_HOST}; use 0.0.0.0 to expose it)",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"port (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--model", default=None, help="Ollama model to use instead of the configured one"
    )
    parser.add_argument(
        "--no-rerank", action="store_true", help="skip the cross-encoder reranker"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Start the server."""
    from app.web.service import ModeRunner

    args = parse_args(argv)
    runner = ModeRunner(model=args.model, rerank=not args.no_rerank)
    application = create_web_app(mode_runner=runner)
    url = f"http://{args.host}:{args.port}/"
    print("PhD RAG local interface")
    print(f"  open: {url}")
    print(f"  model: {runner.model}")
    if args.host not in ("127.0.0.1", "localhost"):
        print(f"  WARNING: bound to {args.host}; this interface is local-only by design")
    print("  stop: Ctrl+C")

    import uvicorn

    uvicorn.run(application, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(main())
