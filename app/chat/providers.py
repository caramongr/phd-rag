"""Which model answers the next chat turn (RAG-016 §6, §7; FIX1).

One small registry and no architecture: the simple chat can be answered by the local Ollama
model or by one of the DeepSeek models, the user picks one per message, and the choice is stored
with the answer it produced. Everything here is *description* — what can be selected, what it is
called, and whether it is usable right now. The service does the selecting and the API does the
transport.

Three names matter throughout the codebase and are defined here once:

* ``"ollama"`` — the local model, the default, always available;
* ``"deepseek"`` — the cloud provider, available only when a key is configured on the server;
* ``"deepseek-flash"`` / ``"deepseek-v4-pro"`` — the two cloud models, configured in
  :mod:`app.config`. The first is the everyday default; the second is chosen by hand.

The privacy sentence for each provider lives here too, because it is a property of the provider
and not of the page that displays it.

Display names are a *presentation* table, not logic: a model id that is not in it is shown by
its own id, so a configured model is never hidden behind a name this file does not know.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.chat.models import UnsupportedModelError, UnsupportedProviderError
from app.config import Settings

#: The local model served by Ollama. The default, and the only provider that needs no key.
PROVIDER_LOCAL = "ollama"
#: The DeepSeek API. Selected manually for one turn; never reached automatically.
PROVIDER_DEEPSEEK = "deepseek"

#: What each provider is called when the provider itself has no better name.
PROVIDER_NAME = {PROVIDER_LOCAL: "Local model", PROVIDER_DEEPSEEK: "DeepSeek"}

#: How the selector groups the entries (RAG-016 FIX1).
PROVIDER_GROUP = {PROVIDER_LOCAL: "Local", PROVIDER_DEEPSEEK: "DeepSeek"}

#: Display names for the model ids this project ships with. Anything else keeps its id.
MODEL_DISPLAY_NAMES = {
    "qwen3.5:35b": "Qwen 3.5 35B",
    "deepseek-flash": "DeepSeek V4.1 Flash",
    "deepseek-v4-pro": "DeepSeek V4 Pro",
}

#: The two sentences the UI shows next to the selector (RAG-016 §6). Written here so the
#: backend and the page agree on the wording of the privacy statement.
LOCAL_NOTE = "🔒 Local - processing remains on this computer"
CLOUD_NOTE = "☁ Cloud - this request's selected context will be sent to DeepSeek"

#: Spellings the API accepts for each provider, so the frontend and a script can differ.
_ALIASES = {
    "": PROVIDER_LOCAL,
    "local": PROVIDER_LOCAL,
    "ollama": PROVIDER_LOCAL,
    "qwen": PROVIDER_LOCAL,
    "cloud": PROVIDER_DEEPSEEK,
    "deepseek": PROVIDER_DEEPSEEK,
    "api": PROVIDER_DEEPSEEK,
}


@dataclass(frozen=True)
class ModelOption:
    """One selectable model: the id the API takes, and the name the user sees."""

    id: str
    label: str


@dataclass(frozen=True)
class ProviderOption:
    """One selectable provider with every model it offers."""

    id: str
    label: str
    #: ``"local"`` or ``"cloud"``. The UI uses it, and so does the wording of a warning.
    kind: str
    #: The models the selector offers, in the order it offers them; the default first.
    models: tuple[ModelOption, ...]
    #: Whether the provider can answer right now (the cloud one needs a configured key).
    available: bool
    #: The privacy sentence for the selector.
    note: str
    #: What the provider's own ``/models`` answered for this process, when it could be read.
    #: Informational: it never adds, removes or disables a selectable model, so the configured
    #: models are always offered even when a listing is unavailable or out of date.
    reported_models: tuple[str, ...] = ()
    #: The group heading the selector files this provider under.
    group: str = ""
    #: The window the context builder budgets against for this provider.
    num_ctx: int = 0


def model_label(model_id: str) -> str:
    """The name to show for a model id: its display name, else the id itself."""
    clean = str(model_id or "").strip()
    return MODEL_DISPLAY_NAMES.get(clean, clean)


def normalise_provider(value: str | None, *, default: str = PROVIDER_LOCAL) -> str:
    """Turn whatever the caller sent into a provider id.

    Raises:
        UnsupportedProviderError: when the value is not a provider this build knows.
    """
    if value is None:
        return default
    key = str(value).strip().lower()
    if key not in _ALIASES:
        raise UnsupportedProviderError(value)
    return _ALIASES[key]


def is_cloud(provider_id: str) -> bool:
    """Whether this provider sends the selected context off the machine."""
    return normalise_provider(provider_id) == PROVIDER_DEEPSEEK


def local_model(settings: Settings, *, override: str | None = None) -> str:
    """The local model id: the command-line override, else the configured one."""
    return str(override or settings.generation_model)


def cloud_models(settings: Settings) -> tuple[str, ...]:
    """Every DeepSeek model id the selector offers, the default first (RAG-016 FIX1).

    The configured ids only: the default (``DEEPSEEK_MODEL``) followed by the rest
    (``DEEPSEEK_MODELS``). Duplicates are dropped, so naming the default twice changes nothing,
    and the order is stable so the selector does not move between page loads.
    """
    ordered: list[str] = []
    for model in (str(settings.deepseek_model), *settings.deepseek_models):
        clean = str(model or "").strip()
        if clean and clean not in ordered:
            ordered.append(clean)
    return tuple(ordered)


def provider_options(
    settings: Settings, *, local_model_id: str | None = None, reported: tuple[str, ...] = ()
) -> tuple[ProviderOption, ...]:
    """The selectable providers with their models, local first (it is the default)."""
    local = local_model(settings, override=local_model_id)
    cloud = cloud_models(settings)
    return (
        ProviderOption(
            id=PROVIDER_LOCAL,
            label=model_label(local),
            kind="local",
            models=(ModelOption(id=local, label=model_label(local)),),
            available=True,
            note=LOCAL_NOTE,
            group=PROVIDER_GROUP[PROVIDER_LOCAL],
            num_ctx=int(settings.chat_num_ctx),
        ),
        ProviderOption(
            id=PROVIDER_DEEPSEEK,
            label=model_label(cloud[0]) if cloud else PROVIDER_NAME[PROVIDER_DEEPSEEK],
            kind="cloud",
            models=tuple(ModelOption(id=model, label=model_label(model)) for model in cloud),
            available=bool(settings.deepseek_api_key),
            note=CLOUD_NOTE,
            reported_models=tuple(reported),
            group=PROVIDER_GROUP[PROVIDER_DEEPSEEK],
            num_ctx=int(settings.deepseek_num_ctx),
        ),
    )


def resolve_model(
    provider_id: str, settings: Settings, model: str | None, *, local_model_id: str | None = None
) -> str:
    """Which model a turn uses for a provider, refusing anything this build cannot offer.

    ``None`` means "the provider's own default", so a client that never names a model keeps
    working exactly as before.

    Raises:
        UnsupportedModelError: when the id is not one the provider offers.
    """
    if provider_id == PROVIDER_DEEPSEEK:
        offered = cloud_models(settings)
    else:
        offered = (local_model(settings, override=local_model_id),)
    if model is None or not str(model).strip():
        return offered[0]
    if str(model).strip() not in offered:
        raise UnsupportedModelError(provider_id, model, offered)
    return str(model).strip()


__all__ = [
    "CLOUD_NOTE",
    "LOCAL_NOTE",
    "MODEL_DISPLAY_NAMES",
    "PROVIDER_DEEPSEEK",
    "PROVIDER_GROUP",
    "PROVIDER_LOCAL",
    "PROVIDER_NAME",
    "ModelOption",
    "ProviderOption",
    "cloud_models",
    "is_cloud",
    "local_model",
    "model_label",
    "normalise_provider",
    "provider_options",
    "resolve_model",
]
