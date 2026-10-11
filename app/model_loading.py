"""Local model loading helpers shared by the embedder and the reranker (RAG-009).

RAG-008 loads ``BAAI/bge-m3`` and RAG-009 loads ``BAAI/bge-reranker-v2-m3``. Both
come from the same Hugging Face cache and both need the same two policies:

* the device is never silently substituted - asking for CUDA without CUDA fails;
* the revision is read from the resolved snapshot directory instead of guessed, so
  a report can always state exactly which weights produced a score.

Only *policy* lives here. Framework specifics (``SentenceTransformer``,
``CrossEncoder``) stay in the layer that owns them, so this module imports no
heavy dependency at module import time.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Iterable


class ModelLoadError(RuntimeError):
    """A local model could not be resolved or loaded."""


#: Weights and config for a sentence-transformers style checkpoint. ONNX exports
#: are excluded: the torch path is the one that gets benchmarked and reported.
DEFAULT_ALLOW_PATTERNS: tuple[str, ...] = (
    "*.json",
    "*.txt",
    "*.model",
    "*.safetensors",
    "*.bin",
    "*.py",
    "1_Pooling/*",
)
DEFAULT_IGNORE_PATTERNS: tuple[str, ...] = ("onnx/*", "imgs/*", "*.onnx", "*.onnx_data")


def library_version(distribution: str) -> str | None:
    """Installed version of ``distribution``, or ``None`` when not installed."""
    try:
        return importlib.metadata.version(distribution)
    except Exception:  # pragma: no cover - metadata is present in practice
        return None


def resolve_device(
    requested: str | None,
    *,
    error_cls: type[Exception] | None = None,
) -> str:
    """Return the device to load on, refusing an explicit unavailable device.

    ``None`` means "auto": CUDA when the installed torch can use it, otherwise CPU.
    A caller that explicitly asked for ``cuda`` on a CPU-only machine gets an error
    rather than a silent downgrade, because silently scoring on a different device
    than the one in a report is worse than failing.
    """
    import torch

    errors: type[Exception] = error_cls or ModelLoadError
    cuda_available = bool(torch.cuda.is_available())
    if requested is None:
        return "cuda" if cuda_available else "cpu"
    if requested.startswith("cuda") and not cuda_available:
        raise errors(
            "CUDA was requested but torch reports no usable CUDA device; "
            "pass --device cpu (the pipeline never silently switches device)"
        )
    return requested


def resolve_snapshot(
    model_name: str,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
    cache_dir: str | Path | None = None,
    allow_patterns: Iterable[str] | None = None,
    ignore_patterns: Iterable[str] | None = None,
    error_cls: type[Exception] | None = None,
) -> tuple[str, str | None]:
    """Resolve ``model_name`` to a local path plus the exact revision.

    A hub id is downloaded/re-used through the Hugging Face cache; the snapshot
    directory name is the commit hash, which is the revision recorded in reports.
    A local directory is used as-is (``revision=None`` until the caller overrides).
    """
    errors: type[Exception] = error_cls or ModelLoadError
    local = Path(model_name)
    if local.is_dir():
        return str(local), revision

    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(
            repo_id=model_name,
            revision=revision,
            local_files_only=local_files_only,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            allow_patterns=list(allow_patterns or DEFAULT_ALLOW_PATTERNS),
            ignore_patterns=list(
                DEFAULT_IGNORE_PATTERNS if ignore_patterns is None else ignore_patterns
            ),
        )
    except Exception as exc:
        raise errors(f"could not resolve model {model_name!r}: {exc}") from exc
    return snapshot, Path(snapshot).name
