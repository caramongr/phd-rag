"""The local web UI (RAG-014).

One command starts it::

    python -m app.web

It serves the RAG-013 mode service through two JSON endpoints (``/api/modes``,
``/api/mode``) plus ``/api/status``, and a plain HTML/CSS/JavaScript page that renders the
result. There is no build step, no framework and no state beyond the mode service the process
owns.

The default binding is ``127.0.0.1``: this UI is for one person on their own machine, and it
can read the whole corpus (including the author's own drafts and supervision material), so
exposing it on a network interface has to be a deliberate act — ``--host 0.0.0.0`` — not a
default.
"""

from __future__ import annotations

from pathlib import Path

#: Where the frontend lives. Served as static files; nothing is templated server-side,
#: so no user or document text ever passes through HTML generation.
STATIC_DIR = Path(__file__).parent / "static"

#: Local-only defaults (§17).
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

__all__ = ["DEFAULT_HOST", "DEFAULT_PORT", "STATIC_DIR", "create_web_app"]


def create_web_app(*, mode_runner=None, mount_ui: bool = True):  # noqa: ANN001, ANN201
    """The FastAPI application with the UI mounted (imported lazily to avoid a cycle)."""
    from app.main import create_app

    return create_app(mode_runner=mode_runner, mount_ui=mount_ui)
