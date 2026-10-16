"""Filesystem layout for every artefact the pipeline generates.

Artefacts are grouped per base model so several fine-tunes can coexist::

    out/<model_id>/final_adapter/       LoRA adapter produced by training
    out/<model_id>/merged_model/        adapter merged back into the base weights
    out/<model_id>/gguf/<name>-<t>.gguf quantized model, one file per type

``<model_id>`` keeps the HuggingFace ``org/name`` form, so the organisation
becomes an intermediate directory. Identifiers are validated before they ever
reach the filesystem: they arrive from user input and from the HuggingFace API,
so they are treated as untrusted and may not escape the output root.

The output root defaults to :data:`DEFAULT_OUT_DIR` and can be moved from the
Setup menu. The choice is stored in :data:`SETTINGS_FILE` and read on every
call rather than cached, so the CLI and the web server always agree on it.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = APP_ROOT / "out"
SETTINGS_FILE = APP_ROOT / "settings.json"
OUTPUT_DIR_KEY = "output_dir"
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


class OutputDirError(ValueError):
    """Raised when a directory cannot hold the generated artefacts."""


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
    return output_root() / validated_model_id(model_id)


def output_root() -> Path:
    """Return the directory every checkpoint, merged model and GGUF is written under.

    A missing or unreadable settings file means the default: the setting is a
    convenience, and a damaged one should not stop the application starting.
    """
    stored = _read_settings().get(OUTPUT_DIR_KEY)
    return Path(stored) if isinstance(stored, str) and stored else DEFAULT_OUT_DIR


def set_output_root(raw: str) -> Path:
    """Validate ``raw``, create it, store it as the output root and return it.

    An empty value restores the default. Relative paths are taken from the
    application directory, matching where the default lives.
    """
    target = validated_output_dir(raw) if raw.strip() else DEFAULT_OUT_DIR
    settings = {**_read_settings(), OUTPUT_DIR_KEY: str(target)}
    SETTINGS_FILE.write_text(f"{json.dumps(settings, indent=2)}\n", encoding="utf-8")
    return target


def validated_output_dir(raw: str) -> Path:
    """Return ``raw`` as an absolute, existing, writable directory, or raise.

    The directory is created if needed and a probe file is written into it:
    permission checks are unreliable on Windows, and finding out at the end of
    a two-hour training run is the expensive way to learn the answer.
    """
    candidate = Path(raw.strip().strip('"')).expanduser()
    target = (candidate if candidate.is_absolute() else APP_ROOT / candidate).resolve()
    if target.exists() and not target.is_dir():
        raise OutputDirError(f"{target} is a file, not a directory.")
    try:
        ensure_dir(target)
        with tempfile.TemporaryFile(dir=target):
            pass
    except OSError as error:
        raise OutputDirError(f"{target} cannot be written to: {error.strerror or error}") from error
    return target


def _read_settings() -> dict[str, object]:
    """Return the stored settings, or an empty mapping if there are none."""
    try:
        payload = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = {}
    return payload if isinstance(payload, dict) else {}


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
