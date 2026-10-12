"""FastAPI application entry point.

Builds the application, wires configuration, and mounts the API routers. Two runners reach the
pipeline and nothing else does:

* :class:`~app.web.service.ModeRunner` owns the single RAG-013 mode service the RAG-014 mode API
  uses, so no retrieval, generation, citation or discovery logic lives in the API or frontend;
* :class:`~app.web.service.ChatRunner` owns the RAG-015 simple chat, whose corpus retrieval is
  borrowed from the mode runner so the Qdrant store is still opened once per process.

The static UI is mounted last, at ``/``, so ``/api/...`` and ``/health`` are matched by their
routers before the catch-all static handler is consulted.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.chats import install_chat_error_handlers
from app.api.chats import router as chats_router
from app.api.errors import install_error_handlers
from app.api.health import router as health_router
from app.api.modes import router as modes_router
from app.api.status import router as status_router
from app.config import Settings, get_settings
from app.web import STATIC_DIR
from app.web.service import ChatRunner, ModeRunner, make_corpus_provider


def create_app(
    *,
    settings: Settings | None = None,
    mode_runner: ModeRunner | None = None,
    chat_runner: ChatRunner | None = None,
    mount_ui: bool = True,
) -> FastAPI:
    """Construct and configure the FastAPI application.

    ``mode_runner`` and ``chat_runner`` can be injected (the ``python -m app.web`` entry point
    passes the ones configured from its command line, and the tests pass fakes); otherwise the
    application builds its own, and the services they hold are built on first use and closed
    when the server stops.
    """
    settings = settings or get_settings()
    runner = mode_runner or ModeRunner(settings=settings)
    chat = chat_runner or ChatRunner(
        settings=settings,
        model=getattr(runner, "model", None),
        corpus_provider=make_corpus_provider(runner),
    )

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=_lifespan(runner, chat),
    )
    application.state.mode_runner = runner
    application.state.chat_runner = chat
    application.state.settings = settings
    application.include_router(health_router)
    application.include_router(modes_router)
    application.include_router(status_router)
    application.include_router(chats_router)
    install_error_handlers(application)
    install_chat_error_handlers(application)
    if mount_ui:
        application.mount(
            "/", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui"
        )
    return application


def _lifespan(runner: ModeRunner, chat: ChatRunner):
    """Close the services the runners built when the application stops."""

    async def lifespan(_: FastAPI):
        yield
        runner.close()
        chat.close()

    return lifespan


app = create_app()
