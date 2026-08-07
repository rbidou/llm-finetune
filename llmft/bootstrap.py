"""Confirmation-gated installation of the libraries and binaries we do not ship.

Two things are installed: the Python dependencies, and llama.cpp. Both are
planned first and executed only after an explicit yes, so the user always sees
what is about to be downloaded and how large it is.

This module uses the standard library only. It runs against a bare virtual
environment -- if it needed ``requests`` it could not install ``requests``.

Nothing here installs LM Studio or Ollama. Those are the user's to manage; their
absence disables the matching deploy option and nothing more.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from llmft.environment import LLAMA_BIN_DIR, Capabilities, REPORTED_PACKAGES
from llmft.paths import APP_ROOT, LLAMA_CPP_DIR, TOOLS_DIR, ensure_dir
from llmft.process import ProcessError, run_streaming
from llmft.progress import ProgressSink, Stage, null_sink, report

REQUIREMENTS_FILE = APP_ROOT / "requirements.txt"
VERSION_STAMP = TOOLS_DIR / "llama-cpp-version.txt"

TORCH_INDEX_URL = "https://download.pytorch.org/whl"
LLAMA_CPP_RELEASE_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
USER_AGENT = "llm-finetune"

# llama-quantize is a CPU-only operation, so the small CPU build is always the
# right download: on Windows it is 18 MB against 250 MB for the CUDA build.
PLATFORM_ASSET_SUFFIXES: dict[tuple[str, str], str] = {
    ("Windows", "x86_64"): "bin-win-cpu-x64.zip",
    ("Windows", "arm64"): "bin-win-cpu-arm64.zip",
    ("Linux", "x86_64"): "bin-ubuntu-x64.tar.gz",
    ("Linux", "arm64"): "bin-ubuntu-arm64.tar.gz",
    ("Darwin", "x86_64"): "bin-macos-x64.tar.gz",
    ("Darwin", "arm64"): "bin-macos-arm64.tar.gz",
}

_DOWNLOAD_CHUNK_BYTES = 1 << 16
_NETWORK_TIMEOUT_SECONDS = 60

Confirm = Callable[[str], bool]


class BootstrapError(RuntimeError):
    """Raised when an installation step fails."""


@dataclass(frozen=True)
class InstallStep:
    """One command to run, with a label a front-end can display."""

    label: str
    command: tuple[str, ...]


@dataclass(frozen=True)
class DependencyPlan:
    """What needs installing, and how, before the pipeline can run."""

    missing: tuple[str, ...]
    wheel_tag: str
    steps: tuple[InstallStep, ...]

    @property
    def is_empty(self) -> bool:
        """Whether every dependency is already present."""
        return not self.steps


@dataclass(frozen=True)
class LlamaCppPlan:
    """The llama.cpp release to install, pinned to a single tag.

    The convert script and the quantize binary come from the same release on
    purpose: GGUF is a moving format, and a mismatched pair fails in ways that
    are hard to diagnose.
    """

    tag: str
    source_url: str
    binary_url: str | None
    binary_name: str | None
    binary_bytes: int | None
    needs_source: bool
    needs_binary: bool

    @property
    def is_empty(self) -> bool:
        """Whether llama.cpp is already fully installed."""
        return not self.needs_source and not self.needs_binary


def plan_dependencies(capabilities: Capabilities) -> DependencyPlan:
    """Decide which install commands are needed for the detected machine.

    torch is installed first and on its own, from the wheel index matching the
    driver; everything else then resolves against that torch rather than pulling
    a second, conflicting build.
    """
    missing = capabilities.missing(REPORTED_PACKAGES)
    wheel_tag = capabilities.wheel_tag
    steps: list[InstallStep] = []
    if "torch" in missing:
        steps.append(
            InstallStep(
                label=f"Install PyTorch ({wheel_tag})",
                command=(*_pip(), "install", "torch", "--index-url", f"{TORCH_INDEX_URL}/{wheel_tag}"),
            )
        )
    if [name for name in missing if name != "torch"]:
        steps.append(
            InstallStep(
                label="Install remaining dependencies",
                command=(*_pip(), "install", "-r", str(REQUIREMENTS_FILE)),
            )
        )
    return DependencyPlan(missing=missing, wheel_tag=wheel_tag, steps=tuple(steps))


def install_dependencies(plan: DependencyPlan, confirm: Confirm, sink: ProgressSink = null_sink) -> bool:
    """Run a dependency plan after confirmation. Returns whether it ran."""
    question = f"Install {len(plan.missing)} missing package(s) using {plan.wheel_tag} wheels?"
    approved = not plan.is_empty and confirm(question)
    if approved:
        for index, step in enumerate(plan.steps, start=1):
            report(sink, Stage.BOOTSTRAP, step.label, index - 1, len(plan.steps))
            _run_streaming(step.command, sink, step.label)
        report(sink, Stage.BOOTSTRAP, "Dependencies installed", len(plan.steps), len(plan.steps))
    return approved


def plan_llama_cpp(capabilities: Capabilities) -> LlamaCppPlan:
    """Look up the latest llama.cpp release and work out what is missing.

    An unsupported platform yields a plan with no binary: plain GGUF conversion
    still works from the source archive, and k-quants simply stay unavailable.
    """
    release = _fetch_release()
    suffix = PLATFORM_ASSET_SUFFIXES.get((platform.system(), _normalised_machine()))
    asset = _select_asset(release.get("assets", []), suffix) if suffix else None
    return LlamaCppPlan(
        tag=str(release.get("tag_name", "unknown")),
        source_url=str(release["zipball_url"]),
        binary_url=asset["browser_download_url"] if asset else None,
        binary_name=asset["name"] if asset else None,
        binary_bytes=asset.get("size") if asset else None,
        needs_source=capabilities.convert_script is None,
        needs_binary=capabilities.llama_quantize is None and asset is not None,
    )


def install_llama_cpp(plan: LlamaCppPlan, confirm: Confirm, sink: ProgressSink = null_sink) -> bool:
    """Download and unpack llama.cpp after confirmation. Returns whether it ran."""
    approved = not plan.is_empty and confirm(describe_llama_cpp_plan(plan))
    if approved:
        ensure_dir(TOOLS_DIR)
        if plan.needs_source:
            _install_source(plan, sink)
        if plan.needs_binary and plan.binary_url:
            _install_binary(plan, sink)
        VERSION_STAMP.write_text(f"{plan.tag}\n", encoding="utf-8")
        report(sink, Stage.BOOTSTRAP, f"llama.cpp {plan.tag} installed", 1, 1)
    return approved


def describe_llama_cpp_plan(plan: LlamaCppPlan) -> str:
    """Phrase a llama.cpp plan as a question, naming every download."""
    parts: list[str] = []
    if plan.needs_source:
        parts.append("llama.cpp source (GGUF conversion script; size reported during download)")
    if plan.needs_binary and plan.binary_name:
        parts.append(f"{plan.binary_name} ({_human_bytes(plan.binary_bytes)}; enables k-quants)")
    return f"Download llama.cpp {plan.tag} into {TOOLS_DIR}?\n  - " + "\n  - ".join(parts)


def _install_source(plan: LlamaCppPlan, sink: ProgressSink) -> None:
    """Fetch the release source archive and place it at ``tools/llama.cpp``."""
    with tempfile.TemporaryDirectory() as workspace:
        staging = Path(workspace)
        archive = _download(plan.source_url, staging / "llama-cpp-source.zip", sink, "llama.cpp source")
        unpacked = staging / "unpacked"
        _extract(archive, unpacked, sink)
        _replace_directory(_single_child(unpacked), LLAMA_CPP_DIR)


def _install_binary(plan: LlamaCppPlan, sink: ProgressSink) -> None:
    """Fetch the prebuilt release archive and unpack it to ``tools/llama-bin``."""
    with tempfile.TemporaryDirectory() as workspace:
        staging = Path(workspace)
        archive = _download(str(plan.binary_url), staging / str(plan.binary_name), sink, str(plan.binary_name))
        unpacked = staging / "unpacked"
        _extract(archive, unpacked, sink)
        _replace_directory(unpacked, LLAMA_BIN_DIR)
    _make_executable(LLAMA_BIN_DIR)


def _fetch_release() -> dict:
    """Read the latest release metadata from the GitHub API."""
    request = urllib.request.Request(LLAMA_CPP_RELEASE_API, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_NETWORK_TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
        raise BootstrapError(f"Could not reach the llama.cpp release API: {error}") from error
    return payload


def _select_asset(assets: list[dict], suffix: str) -> dict | None:
    """Pick the release asset whose name ends with ``suffix``.

    Matching the full suffix rather than a fragment is deliberate: it keeps
    ``bin-ubuntu-x64`` from also matching ``bin-ubuntu-vulkan-x64``.
    """
    return next((asset for asset in assets if str(asset.get("name", "")).endswith(suffix)), None)


def _download(url: str, destination: Path, sink: ProgressSink, label: str) -> Path:
    """Stream a URL to disk, reporting progress when the size is known."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_NETWORK_TIMEOUT_SECONDS) as response:
            total = int(response.headers.get("Content-Length") or 0)
            _stream_to_file(response, destination, total, sink, label)
    except (urllib.error.URLError, OSError) as error:
        raise BootstrapError(f"Download of {label} failed: {error}") from error
    return destination


def _stream_to_file(response, destination: Path, total: int, sink: ProgressSink, label: str) -> None:
    """Copy a response body to ``destination`` in chunks, emitting progress."""
    received = 0
    with destination.open("wb") as target:
        while chunk := response.read(_DOWNLOAD_CHUNK_BYTES):
            target.write(chunk)
            received += len(chunk)
            report(sink, Stage.BOOTSTRAP, f"Downloading {label} ({_human_bytes(received)})", received, total or None)


def _extract(archive: Path, destination: Path, sink: ProgressSink) -> None:
    """Unpack a ``.zip`` or ``.tar.gz`` archive, refusing unsafe member paths."""
    report(sink, Stage.BOOTSTRAP, f"Extracting {archive.name}")
    ensure_dir(destination)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(destination)
    elif archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as bundle:
            bundle.extractall(destination, filter="data")
    else:
        raise BootstrapError(f"Unsupported archive format: {archive.name}")


def _replace_directory(source: Path, destination: Path) -> None:
    """Move ``source`` onto ``destination``, discarding any previous install."""
    if destination.exists():
        shutil.rmtree(destination)
    ensure_dir(destination.parent)
    shutil.move(str(source), str(destination))


def _single_child(directory: Path) -> Path:
    """Return the sole directory inside ``directory``.

    Source archives from GitHub wrap everything in one commit-stamped folder
    whose name is not predictable, so it is discovered rather than assumed.
    """
    children = [child for child in directory.iterdir() if child.is_dir()]
    if len(children) != 1:
        raise BootstrapError(f"Expected one directory in the archive, found {len(children)}.")
    return children[0]


def _make_executable(root: Path) -> None:
    """Restore the executable bit on extracted binaries on POSIX systems."""
    if os.name != "nt":
        for candidate in root.rglob("*"):
            if candidate.is_file() and not candidate.suffix:
                candidate.chmod(candidate.stat().st_mode | 0o111)


def _run_streaming(command: tuple[str, ...], sink: ProgressSink, label: str) -> None:
    """Run a command, forwarding each output line as a progress message."""
    try:
        run_streaming(command, lambda line: report(sink, Stage.BOOTSTRAP, line), label)
    except ProcessError as error:
        raise BootstrapError(str(error)) from error


def _pip() -> tuple[str, ...]:
    """The pip invocation for the interpreter currently running."""
    return (sys.executable, "-m", "pip")


def _normalised_machine() -> str:
    """Map the many spellings of a CPU architecture onto two names."""
    machine = platform.machine().lower()
    return "arm64" if machine in {"arm64", "aarch64"} else "x86_64" if machine in {"x86_64", "amd64"} else machine


def _human_bytes(size: int | None) -> str:
    """Format a byte count for display, or report it as unknown."""
    result = "unknown size"
    if size:
        units = ("B", "KB", "MB", "GB")
        scaled = float(size)
        unit_index = 0
        while scaled >= 1024 and unit_index < len(units) - 1:
            scaled /= 1024
            unit_index += 1
        result = f"{scaled:.1f} {units[unit_index]}"
    return result
