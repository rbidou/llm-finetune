"""Filesystem layout for every artefact the pipeline generates.

Artefacts are grouped per base model so several fine-tunes can coexist::

    out/<model_id>/final_adapter/       LoRA adapter produced by training
    out/<model_id>/merged_model/        adapter merged back into the base weights
    out/<model_id>/gguf/<name>-<t>.gguf quantized model, one file per type

``<model_id>`` keeps the HuggingFace ``org/name`` form, so the organisation
becomes an intermediate directory. Identifiers are validated before they ever
reach the filesystem: they arrive from user input and from the HuggingFace API,
so they are treated as untrusted and may not escape :data:`OUT_DIR`.
"""

from __future__ import annotations

import re
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = APP_ROOT / "out"
CONFIG_DIR = APP_ROOT / "configs"
TOOLS_DIR = APP_ROOT / "tools"
LLAMA_CPP_DIR = TOOLS_DIR / "llama.cpp"

ADAPTER_SUBDIR = "final_adapter"
MERGED_SUBDIR = "merged_model"
GGUF_SUBDIR = "gguf"

# HuggingFace repository identifiers: an optional organisation, then the model
# name. Both segments must start alphanumeric, which also rules out "." and "..".
MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")


class InvalidModelIdError(ValueError):
    """Raised when a model identifier is unusable as a filesystem path."""


def adapter_dir(model_id: str) -> Path:
    """Return the directory holding the trained LoRA adapter for ``model_id``."""
    return model_output_dir(model_id) / ADAPTER_SUBDIR


def merged_dir(model_id: str) -> Path:
    """Return the directory holding the merged full-precision model."""
    return model_output_dir(model_id) / MERGED_SUBDIR


def gguf_dir(model_id: str) -> Path:
    """Return the directory holding GGUF conversions of ``model_id``."""
    return model_output_dir(model_id) / GGUF_SUBDIR


def gguf_file(model_id: str, quant_type: str) -> Path:
    """Return the GGUF path for one quantization type, e.g. ``…/Llama-3-8B-q8_0.gguf``."""
    return gguf_dir(model_id) / f"{model_name(model_id)}-{quant_type}.gguf"


def model_output_dir(model_id: str) -> Path:
    """Return the per-model artefact root, validating ``model_id`` first."""
    return OUT_DIR / validated_model_id(model_id)


def model_name(model_id: str) -> str:
    """Return the model name without its organisation prefix."""
    return validated_model_id(model_id).rsplit("/", maxsplit=1)[-1]


def validated_model_id(model_id: str) -> str:
    """Return ``model_id`` unchanged, or raise :class:`InvalidModelIdError`.

    Rejects anything that could traverse outside the output directory, plus the
    characters Windows forbids in path components.
    """
    candidate = model_id.strip()
    if not MODEL_ID_PATTERN.match(candidate):
        raise InvalidModelIdError(
            f"{model_id!r} is not a valid model identifier. Expected 'name' or "
            "'organisation/name' using letters, digits, '.', '-' and '_'."
        )
    return candidate


def ensure_dir(path: Path) -> Path:
    """Create ``path`` and its parents if needed, then return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
