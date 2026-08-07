"""Detection of everything the pipeline needs but does not ship.

This module is pure observation: it never installs, downloads or mutates
anything, and it imports nothing heavier than the standard library. That matters
because it runs before :mod:`llmft.bootstrap` has had a chance to install
:mod:`torch` -- the GPU is inspected through ``nvidia-smi`` rather than through
torch for exactly this reason.

Front-ends turn the result into enabled or disabled options via :func:`features`,
which pairs each capability with a reason when it is unavailable. A missing
binary must degrade the interface, never fail a run half-way through.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from llmft.paths import LLAMA_CPP_DIR, TOOLS_DIR

CONVERT_SCRIPT_NAME = "convert_hf_to_gguf.py"
LLAMA_QUANTIZE_STEM = "llama-quantize"
LLAMA_BIN_DIR = TOOLS_DIR / "llama-bin"

# Environment overrides for users who already have llama.cpp installed.
LLAMA_CPP_DIR_VAR = "LLMFT_LLAMA_CPP_DIR"
LLAMA_QUANTIZE_VAR = "LLMFT_LLAMA_QUANTIZE"

# Minimum NVIDIA driver version for each PyTorch CUDA wheel index, newest first.
CUDA_WHEEL_TAGS: tuple[tuple[float, str], ...] = ((570.0, "cu128"), (525.0, "cu126"), (450.0, "cu118"))
CPU_WHEEL_TAG = "cpu"

# bfloat16 needs Ampere (compute capability 8.0) or newer.
BF16_MIN_COMPUTE_CAPABILITY = 8.0

# Below this fraction of free memory, another application is in the way.
OCCUPIED_THRESHOLD = 0.75

TRAINING_PACKAGES = ("torch", "transformers", "peft", "trl", "datasets", "accelerate", "bitsandbytes")
MERGE_PACKAGES = ("torch", "transformers", "peft", "safetensors")
CONVERT_PACKAGES = ("torch", "transformers", "safetensors", "numpy", "sentencepiece", "protobuf")
WEB_PACKAGES = ("flask",)
CLI_PACKAGES = ("rich", "prompt_toolkit")
HUB_PACKAGES = ("huggingface_hub",)

REPORTED_PACKAGES = tuple(
    dict.fromkeys(TRAINING_PACKAGES + MERGE_PACKAGES + CONVERT_PACKAGES + HUB_PACKAGES + CLI_PACKAGES + WEB_PACKAGES)
)

_SUBPROCESS_TIMEOUT_SECONDS = 20


@dataclass(frozen=True)
class GpuInfo:
    """A single CUDA device as reported by ``nvidia-smi``."""

    name: str
    total_memory_mb: int
    driver_version: str
    compute_capability: float
    free_memory_mb: int = 0

    @property
    def supports_bf16(self) -> bool:
        """Whether the device can run bfloat16 arithmetic natively."""
        return self.compute_capability >= BF16_MIN_COMPUTE_CAPABILITY

    @property
    def occupied_memory_mb(self) -> int:
        """Memory already held by other applications."""
        return max(0, self.total_memory_mb - self.free_memory_mb)

    @property
    def is_heavily_occupied(self) -> bool:
        """Whether another application is holding most of the card.

        A game, a browser or a previous run that has not exited can leave a
        nominally large GPU with almost nothing available, so an estimate
        against total memory would approve a run that cannot start.

        A driver that does not report free memory reports zero, which is not
        evidence of occupancy and must not be treated as such.
        """
        known = self.free_memory_mb > 0
        return known and self.free_memory_mb < self.total_memory_mb * OCCUPIED_THRESHOLD


@dataclass(frozen=True)
class Feature:
    """One user-facing capability, and why it is unavailable when it is."""

    key: str
    label: str
    available: bool
    reason: str = ""


@dataclass(frozen=True)
class Capabilities:
    """Everything detected about this machine at one point in time."""

    platform_description: str
    python_version: str
    python_executable: Path
    gpu: GpuInfo | None
    packages: dict[str, str | None]
    convert_script: Path | None
    llama_quantize: Path | None
    ollama: Path | None
    lms: Path | None

    @property
    def wheel_tag(self) -> str:
        """The PyTorch wheel index matching the detected driver, or ``cpu``."""
        return cuda_wheel_tag(self.gpu)

    @property
    def has_gpu(self) -> bool:
        """Whether a usable CUDA device was found."""
        return self.gpu is not None

    def missing(self, names: tuple[str, ...]) -> tuple[str, ...]:
        """Return which of ``names`` are not installed, in the given order."""
        return tuple(name for name in names if not self.packages.get(name))


def detect() -> Capabilities:
    """Inspect the machine and return what is currently available."""
    return Capabilities(
        platform_description=platform.platform(),
        python_version=platform.python_version(),
        python_executable=Path(sys.executable),
        gpu=detect_gpu(),
        packages={name: package_version(name) for name in REPORTED_PACKAGES},
        convert_script=find_convert_script(),
        llama_quantize=find_llama_quantize(),
        ollama=find_ollama(),
        lms=find_lms(),
    )


def find_ollama() -> Path | None:
    """Locate the Ollama executable, including its default install locations."""
    return find_executable("ollama", _ollama_locations())


def find_lms() -> Path | None:
    """Locate the LM Studio CLI.

    LM Studio creates this on first launch and does not add it to ``PATH`` of
    already-running processes, so the default location must be checked too.
    Detection and deployment must use this one function: if they disagree, a
    deploy option is offered that then fails.
    """
    return find_executable("lms", _lms_locations())


def features(capabilities: Capabilities) -> tuple[Feature, ...]:
    """Describe each capability as available or not, with an actionable reason."""
    return (
        _package_feature("browse", "Model browsing", capabilities, HUB_PACKAGES),
        _training_feature(capabilities),
        _package_feature("merge", "Adapter merge", capabilities, MERGE_PACKAGES),
        _convert_feature(capabilities),
        _k_quant_feature(capabilities),
        _binary_feature("deploy_ollama", "Deploy to Ollama", capabilities.ollama, "Ollama is not installed."),
        _binary_feature("deploy_lms", "Deploy to LM Studio", capabilities.lms, "LM Studio is not installed."),
    )


def cuda_wheel_tag(gpu: GpuInfo | None) -> str:
    """Choose the PyTorch wheel index for a GPU's driver.

    Newer drivers are backward compatible, so the newest index whose minimum
    driver is satisfied is selected. Absent or unreadable GPU means CPU wheels.
    """
    result = CPU_WHEEL_TAG
    if gpu is not None:
        driver = _leading_float(gpu.driver_version)
        for minimum_driver, tag in CUDA_WHEEL_TAGS:
            if driver is not None and driver >= minimum_driver:
                result = tag
                break
    return result


def detect_gpu() -> GpuInfo | None:
    """Return the first CUDA device ``nvidia-smi`` reports, or ``None``.

    Multi-GPU machines report only the first device: training targets a single
    GPU, so the others would only make the estimate misleading.
    """
    output = _run_capture(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,compute_cap,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    return _parse_gpu(output) if output else None


def find_convert_script() -> Path | None:
    """Locate ``convert_hf_to_gguf.py``, preferring an explicit override."""
    override = os.environ.get(LLAMA_CPP_DIR_VAR)
    candidates = [Path(override) / CONVERT_SCRIPT_NAME] if override else []
    candidates.append(LLAMA_CPP_DIR / CONVERT_SCRIPT_NAME)
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def find_llama_quantize() -> Path | None:
    """Locate the ``llama-quantize`` binary.

    Checked in order: an explicit override, the copy this application installed
    (searched recursively, because the layout inside a release archive differs
    between platforms), then ``PATH``.
    """
    override = os.environ.get(LLAMA_QUANTIZE_VAR)
    name = executable_name(LLAMA_QUANTIZE_STEM)
    if override and Path(override).is_file():
        result = Path(override)
    else:
        result = _search_directory(LLAMA_BIN_DIR, name) or find_executable(LLAMA_QUANTIZE_STEM)
    return result


def find_executable(stem: str, extra_locations: tuple[Path, ...] = ()) -> Path | None:
    """Find an executable on ``PATH``, falling back to known install locations.

    The fallback matters because Windows installers frequently add themselves to
    the user's ``PATH`` without it being visible to an already-running process.
    """
    on_path = shutil.which(stem)
    result = Path(on_path) if on_path else None
    if result is None:
        result = next((location for location in extra_locations if location.is_file()), None)
    return result


def executable_name(stem: str) -> str:
    """Return the platform-correct executable filename for ``stem``."""
    return f"{stem}.exe" if os.name == "nt" else stem


def package_version(name: str) -> str | None:
    """Return an installed distribution's version without importing it.

    Reading metadata rather than importing keeps detection fast and avoids
    loading torch as a side effect of asking whether torch exists.
    """
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = None
    return version


def _training_feature(capabilities: Capabilities) -> Feature:
    """QLoRA needs both the libraries and a CUDA device for 4-bit weights."""
    missing = capabilities.missing(TRAINING_PACKAGES)
    if not capabilities.has_gpu:
        reason = "No NVIDIA GPU detected. 4-bit QLoRA training requires CUDA."
    else:
        reason = _missing_packages_reason(missing)
    return Feature("train", "QLoRA training", not reason, reason)


def _convert_feature(capabilities: Capabilities) -> Feature:
    """GGUF conversion needs llama.cpp's script plus the libraries it imports."""
    missing = capabilities.missing(CONVERT_PACKAGES)
    if capabilities.convert_script is None:
        reason = f"{CONVERT_SCRIPT_NAME} not found. Run: python run.py setup"
    else:
        reason = _missing_packages_reason(missing)
    return Feature("quantize", "GGUF conversion", not reason, reason)


def _k_quant_feature(capabilities: Capabilities) -> Feature:
    """K-quants are a second pass over an f16 GGUF, so both tools are required."""
    convert = _convert_feature(capabilities)
    if not convert.available:
        reason = convert.reason
    elif capabilities.llama_quantize is None:
        reason = f"{executable_name(LLAMA_QUANTIZE_STEM)} not found. Run: python run.py setup"
    else:
        reason = ""
    return Feature("k_quantize", "K-quant quantization", not reason, reason)


def _package_feature(key: str, label: str, capabilities: Capabilities, required: tuple[str, ...]) -> Feature:
    """Build a feature gated only on installed packages."""
    reason = _missing_packages_reason(capabilities.missing(required))
    return Feature(key, label, not reason, reason)


def _binary_feature(key: str, label: str, located: Path | None, absent_reason: str) -> Feature:
    """Build a feature gated on an external program this application must not install."""
    return Feature(key, label, located is not None, "" if located else absent_reason)


def _missing_packages_reason(missing: tuple[str, ...]) -> str:
    """Phrase a missing-package list as an instruction, or return an empty string."""
    return f"Missing packages: {', '.join(missing)}. Run: python run.py setup" if missing else ""


def _parse_gpu(output: str) -> GpuInfo | None:
    """Parse one ``nvidia-smi`` CSV row, tolerating anything unexpected."""
    lines = output.strip().splitlines()
    fields = [field.strip() for field in lines[0].split(",")] if lines else []
    result = None
    if len(fields) == 5:
        try:
            total = int(float(fields[1]))
            result = GpuInfo(
                name=fields[0],
                total_memory_mb=total,
                driver_version=fields[2],
                compute_capability=float(fields[3]),
                free_memory_mb=min(int(float(fields[4])), total),
            )
        except ValueError:
            result = None
    return result


def _search_directory(root: Path, filename: str) -> Path | None:
    """Find ``filename`` anywhere beneath ``root``."""
    return next(root.rglob(filename), None) if root.is_dir() else None


def _ollama_locations() -> tuple[Path, ...]:
    """Default Ollama install locations that are often absent from ``PATH``."""
    name = executable_name("ollama")
    return (
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / name,
        Path(os.environ.get("ProgramFiles", "")) / "Ollama" / name,
        Path("/usr/local/bin") / name,
        Path.home() / ".ollama" / "bin" / name,
    )


def _lms_locations() -> tuple[Path, ...]:
    """Default LM Studio CLI locations."""
    name = executable_name("lms")
    return (Path.home() / ".lmstudio" / "bin" / name, Path.home() / ".cache" / "lm-studio" / "bin" / name)


def _leading_float(text: str) -> float | None:
    """Read ``610.88.01`` as ``610.88``, ignoring any further components."""
    parts = text.split(".")
    try:
        value = float(".".join(parts[:2])) if len(parts) >= 2 else float(text)
    except ValueError:
        value = None
    return value


def _run_capture(command: list[str]) -> str | None:
    """Run a command and return its stdout, or ``None`` if it cannot be run.

    Detection must never raise: a missing tool is an expected outcome here.
    """
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_SECONDS,
            check=False,
        )
        result = completed.stdout.strip() if completed.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        result = None
    return result
