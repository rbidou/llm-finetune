"""Filesystem layout for every artefact the pipeline generates.

Artefacts are grouped per base model so several fine-tunes can coexist::

    <model dir>/<model_id>/final_adapter/     LoRA adapter produced by training
    <model dir>/<model_id>/merged_model/      adapter merged back into the base weights
    <output dir>/<model_id>/<name>-<t>.gguf   quantized model, one file per type

The working artefacts and the finished GGUF files live under separate roots: the
model directory holds what only this application needs, the output model
directory holds what is handed to a model runner.

``<model_id>`` keeps the HuggingFace ``org/name`` form, so the organisation
becomes an intermediate directory. Identifiers are validated before they ever
reach the filesystem: they arrive from user input and from the HuggingFace API,
so they are treated as untrusted and may not escape their root.

Both roots have a default and can be moved from the Setup menu. The choice is
stored in :data:`SETTINGS_FILE` and read on every call rather than cached, so
the CLI and the web server always agree on it.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = APP_ROOT / "out"
# A Windows drive path means nothing elsewhere, so other systems get the same
# folder under the home directory.
DEFAULT_OUTPUT_MODEL_DIR = (
    Path(r"D:\Models\ParaCyberBellum") if os.name == "nt" else Path.home() / "Models" / "ParaCyberBellum"
)
SETTINGS_FILE = APP_ROOT / "settings.json"
CONFIG_DIR = APP_ROOT / "configs"
TOOLS_DIR = APP_ROOT / "tools"
LLAMA_CPP_DIR = TOOLS_DIR / "llama.cpp"

ADAPTER_SUBDIR = "final_adapter"
MERGED_SUBDIR = "merged_model"
GGUF_SUFFIX = ".gguf"

# HuggingFace repository identifiers: an optional organisation, then the model
# name. Both segments must start alphanumeric, which also rules out "." and "..".
MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")

# A quantized model's name becomes a filename, so it is one identifier segment.
GGUF_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class InvalidModelIdError(ValueError):
    """Raised when a model identifier is unusable as a filesystem path."""


class InvalidNameError(ValueError):
    """Raised when a quantized model's name is unusable as a filename."""


class DirectoryError(ValueError):
    """Raised when a directory cannot hold the generated artefacts."""


@dataclass(frozen=True)
class DirectorySetting:
    """One configurable root: where it is stored, what it holds, its default."""

    key: str
    label: str
    holds: str
    default: Path


MODEL_DIRECTORY = DirectorySetting(
    "model_dir", "Model directory", "training checkpoints, LoRA adapters and merged models", DEFAULT_MODEL_DIR
)
OUTPUT_MODEL_DIRECTORY = DirectorySetting(
    "output_model_dir", "Output model directory", "quantized GGUF files", DEFAULT_OUTPUT_MODEL_DIR
)


def adapter_dir(model_id: str) -> Path:
    """Return the directory holding the trained LoRA adapter for ``model_id``."""
    return model_output_dir(model_id) / ADAPTER_SUBDIR


def merged_dir(model_id: str) -> Path:
    """Return the directory holding the merged full-precision model."""
    return model_output_dir(model_id) / MERGED_SUBDIR


def gguf_dir(model_id: str) -> Path:
    """Return the directory holding GGUF conversions of ``model_id``."""
    return output_model_root() / validated_model_id(model_id)


def gguf_file(model_id: str, quant_type: str, name: str = "") -> Path:
    """Return the GGUF path for one quantization type, e.g. ``…/Llama-3-8B-q8_0.gguf``.

    ``name`` is what the user calls the quantized model; empty means the
    original model's name.
    """
    stem = validated_gguf_name(name) if name else model_name(model_id)
    return gguf_dir(model_id) / f"{stem}-{quant_type}{GGUF_SUFFIX}"


def validated_gguf_name(name: str) -> str:
    """Return ``name`` stripped, or raise :class:`InvalidNameError`."""
    candidate = name.strip()
    if not GGUF_NAME_PATTERN.match(candidate):
        raise InvalidNameError(
            f"{name!r} is not a valid model name. Use letters, digits, '.', '-' and '_', "
            "starting with a letter or digit."
        )
    return candidate


def find_gguf_files() -> list[Path]:
    """Every GGUF under the output model directory, newest first.

    A directory that does not exist yet -- an unplugged drive, say -- simply
    holds nothing.
    """
    return sorted(output_model_root().glob(f"**/*{GGUF_SUFFIX}"), key=lambda path: -path.stat().st_mtime)


def model_id_from_gguf(path: Path) -> str:
    """Recover the model identifier from a GGUF's folder, or ``""`` if it has none.

    Files copied in by hand need not follow the ``<model_id>/`` layout, so
    anything that is not a valid identifier is treated as unknown.
    """
    candidate = "/".join(path.parent.relative_to(output_model_root()).parts)
    return candidate if MODEL_ID_PATTERN.match(candidate) else ""


def model_output_dir(model_id: str) -> Path:
    """Return the per-model artefact root, validating ``model_id`` first."""
    return model_root() / validated_model_id(model_id)


def model_root() -> Path:
    """Return the directory checkpoints, adapters and merged models are written under."""
    return stored_directory(MODEL_DIRECTORY)


def output_model_root() -> Path:
    """Return the directory quantized GGUF files are written under."""
    return stored_directory(OUTPUT_MODEL_DIRECTORY)


def set_model_root(raw: str) -> Path:
    """Store a new model directory; see :func:`set_directory`."""
    return set_directory(MODEL_DIRECTORY, raw)


def set_output_model_root(raw: str) -> Path:
    """Store a new output model directory; see :func:`set_directory`."""
    return set_directory(OUTPUT_MODEL_DIRECTORY, raw)


def stored_directory(setting: DirectorySetting) -> Path:
    """Return the stored value of ``setting``, or its default.

    A missing or unreadable settings file means the default: the setting is a
    convenience, and a damaged one should not stop the application starting.
    """
    stored = read_settings().get(setting.key)
    return Path(stored) if isinstance(stored, str) and stored else setting.default


def set_directory(setting: DirectorySetting, raw: str) -> Path:
    """Validate ``raw``, create it, store it for ``setting`` and return it.

    An empty value restores the default without touching the disk, so the
    default may name a drive that is not plugged in yet. Relative paths are
    taken from the application directory.
    """
    target = validated_directory(raw) if raw.strip() else setting.default
    update_settings({setting.key: str(target)})
    return target


def validated_directory(raw: str) -> Path:
    """Return ``raw`` as an absolute, existing, writable directory, or raise.

    The directory is created if needed and a probe file is written into it:
    permission checks are unreliable on Windows, and finding out at the end of
    a two-hour training run is the expensive way to learn the answer.
    """
    candidate = Path(raw.strip().strip('"')).expanduser()
    target = (candidate if candidate.is_absolute() else APP_ROOT / candidate).resolve()
    if target.exists() and not target.is_dir():
        raise DirectoryError(f"{target} is a file, not a directory.")
    try:
        ensure_dir(target)
        with tempfile.TemporaryFile(dir=target):
            pass
    except OSError as error:
        raise DirectoryError(f"{target} cannot be written to: {error.strerror or error}") from error
    return target


def read_settings() -> dict[str, object]:
    """Return the stored settings, or an empty mapping if there are none."""
    try:
        payload = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = {}
    return payload if isinstance(payload, dict) else {}


def update_settings(changes: dict[str, object]) -> None:
    """Merge ``changes`` into the stored settings, keeping every other key."""
    settings = {**read_settings(), **changes}
    SETTINGS_FILE.write_text(f"{json.dumps(settings, indent=2)}\n", encoding="utf-8")


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
