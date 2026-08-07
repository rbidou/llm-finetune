"""Tunable parameters, their presentation metadata, and named presets.

Every tunable is declared exactly once, carrying the label and explanation the
front-ends need. The CLI parameter editor, the GUI form and the parameter
reference in the manual are all generated from :func:`describe`, so the list is
never restated and cannot drift between them.

There is deliberately no model identifier in this module. The model is always an
explicit user choice, validated against the machine by the pre-flight estimator.
A default here would silently bind the application to particular hardware, and a
preset carrying a model identifier would reintroduce that default through the
back door -- hence :func:`load_preset` rejects one outright.
"""

from __future__ import annotations

import json
import re
from dataclasses import Field, asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any

from llmft.paths import CONFIG_DIR, ensure_dir

PRESET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]*$")
PRESET_SUFFIX = ".json"

# Rejected in presets so a model choice can never arrive as a stored default.
MODEL_KEYS = frozenset({"model", "model_id", "modelid", "base_model", "basemodel"})

ATTENTION_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_MODULES = ("gate_proj", "up_proj", "down_proj")
TARGET_MODULE_CHOICES = ATTENTION_MODULES + MLP_MODULES
BIAS_CHOICES = ("none", "all", "lora_only")

DEFAULT_QUANT_TYPE = "q8_0"


class ConfigError(ValueError):
    """Raised when a parameter value or a stored preset is unusable."""


class ParamKind(str, Enum):
    """How a tunable should be validated and rendered by a front-end."""

    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    CHOICE = "choice"
    MULTI_CHOICE = "multi_choice"


@dataclass(frozen=True)
class ParamSpec:
    """Everything a front-end needs to render and validate one tunable."""

    name: str
    label: str
    explanation: str
    kind: ParamKind
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] | None = None


def param(
    default: Any,
    *,
    label: str,
    explanation: str,
    kind: ParamKind,
    minimum: float | None = None,
    maximum: float | None = None,
    choices: tuple[str, ...] | None = None,
) -> Any:
    """Declare a tunable dataclass field that carries its own presentation metadata.

    Defaults must be immutable, which is why multi-valued parameters use tuples:
    settings groups are frozen so editing one produces a new instance.
    """
    return field(
        default=default,
        metadata={
            "label": label,
            "explanation": explanation,
            "kind": kind,
            "minimum": minimum,
            "maximum": maximum,
            "choices": choices,
        },
    )


@dataclass(frozen=True)
class Settings:
    """Base for parameter groups; validates every tunable on construction."""

    def __post_init__(self) -> None:
        validate_settings(self)


@dataclass(frozen=True)
class LoraSettings(Settings):
    """LoRA adapter geometry, passed through to :class:`peft.LoraConfig`."""

    r: int = param(
        32,
        label="LoRA rank (r)",
        explanation=(
            "Width of the low-rank update. Each targeted layer learns two small "
            "matrices of this rank instead of a full weight matrix, which is what "
            "makes training cheap. Higher values add capacity, trainable parameters "
            "and VRAM roughly linearly. 8-16 suits small, narrow datasets; 32-64 "
            "suits broader ones."
        ),
        kind=ParamKind.INT,
        minimum=1,
        maximum=256,
    )
    lora_alpha: int = param(
        64,
        label="LoRA alpha",
        explanation=(
            "Scaling applied to the adapter output: the update is multiplied by "
            "alpha / r. Because it is a ratio, raising the rank without raising "
            "alpha weakens the adapter. The common convention is alpha = 2 x r, "
            "giving a scale of 2."
        ),
        kind=ParamKind.INT,
        minimum=1,
        maximum=512,
    )
    lora_dropout: float = param(
        0.05,
        label="LoRA dropout",
        explanation=(
            "Dropout applied to the adapter input during training only. Acts as "
            "regularisation against memorising a small dataset. 0.05-0.1 is usual; "
            "0 is reasonable for large datasets or very short runs."
        ),
        kind=ParamKind.FLOAT,
        minimum=0.0,
        maximum=1.0,
    )
    target_modules: tuple[str, ...] = param(
        TARGET_MODULE_CHOICES,
        label="Target modules",
        explanation=(
            "Which projection layers receive an adapter. The attention projections "
            "(q, k, v, o) change how the model attends; the MLP projections (gate, "
            "up, down) hold most of its factual capacity. Targeting all seven gives "
            "the best quality and costs the most memory; attention-only is the "
            "cheapest useful subset."
        ),
        kind=ParamKind.MULTI_CHOICE,
        choices=TARGET_MODULE_CHOICES,
    )
    bias: str = param(
        "none",
        label="Bias handling",
        explanation=(
            "Which bias terms are trainable. 'none' keeps the adapter smallest and "
            "is almost always correct; 'lora_only' trains biases on adapted layers; "
            "'all' trains every bias and prevents a clean merge back into the base."
        ),
        kind=ParamKind.CHOICE,
        choices=BIAS_CHOICES,
    )


@dataclass(frozen=True)
class SftSettings(Settings):
    """Supervised fine-tuning schedule, passed through to :class:`trl.SFTConfig`."""

    num_train_epochs: int = param(
        15,
        label="Training epochs",
        explanation=(
            "How many times the trainer walks the whole dataset. Small "
            "instruction sets need many passes to take effect; large ones overfit "
            "quickly. If answers start being reproduced verbatim, reduce this first."
        ),
        kind=ParamKind.INT,
        minimum=1,
        maximum=1000,
    )
    per_device_train_batch_size: int = param(
        1,
        label="Batch size per device",
        explanation=(
            "Examples processed together in one forward and backward pass. This is "
            "the single largest driver of VRAM use. Leave at 1 on consumer GPUs and "
            "raise the accumulation steps instead."
        ),
        kind=ParamKind.INT,
        minimum=1,
        maximum=64,
    )
    gradient_accumulation_steps: int = param(
        2,
        label="Gradient accumulation steps",
        explanation=(
            "Number of batches accumulated before weights are updated. The effective "
            "batch size is this multiplied by the batch size per device, so it buys "
            "the stability of a larger batch at almost no extra VRAM cost."
        ),
        kind=ParamKind.INT,
        minimum=1,
        maximum=256,
    )
    learning_rate: float = param(
        5e-4,
        label="Learning rate",
        explanation=(
            "Step size for the optimiser. LoRA tolerates rates one to two orders of "
            "magnitude above full fine-tuning because only the adapters move. "
            "1e-4 to 5e-4 is the usual band; lower it if the loss oscillates."
        ),
        kind=ParamKind.FLOAT,
        minimum=1e-6,
        maximum=1e-2,
    )
    warmup_steps: int = param(
        5,
        label="Warmup steps",
        explanation=(
            "Steps spent ramping the learning rate up from zero, which avoids a "
            "destructive first update on freshly initialised adapters. A handful is "
            "enough for short runs."
        ),
        kind=ParamKind.INT,
        minimum=0,
        maximum=10000,
    )
    max_length: int = param(
        2048,
        label="Maximum sequence length",
        explanation=(
            "Token budget per training example; longer conversations are truncated. "
            "Memory grows with this value, so it is the most effective dial to turn "
            "when a run runs out of VRAM. Set it just above your longest example."
        ),
        kind=ParamKind.INT,
        minimum=128,
        maximum=131072,
    )
    bf16: bool = param(
        True,
        label="bfloat16 compute",
        explanation=(
            "Run the forward and backward passes in bfloat16. Halves activation "
            "memory against float32 and, unlike float16, needs no loss scaling. "
            "Requires an Ampere or newer GPU."
        ),
        kind=ParamKind.BOOL,
    )
    gradient_checkpointing: bool = param(
        True,
        label="Gradient checkpointing",
        explanation=(
            "Discard intermediate activations and recompute them during the backward "
            "pass. Cuts activation memory substantially at roughly 20-30% slower "
            "training. Usually the difference between fitting and not fitting."
        ),
        kind=ParamKind.BOOL,
    )


@dataclass(frozen=True)
class QuantTypeInfo:
    """One GGUF output type and the tool needed to produce it."""

    name: str
    label: str
    explanation: str
    requires_llama_quantize: bool


QUANT_TYPES: tuple[QuantTypeInfo, ...] = (
    QuantTypeInfo("f32", "F32 - full precision", "No quantization. Four bytes per weight; a reference baseline, rarely worth shipping.", False),
    QuantTypeInfo("f16", "F16 - half precision", "Two bytes per weight, effectively lossless for inference. Also the staging format for k-quants.", False),
    QuantTypeInfo("bf16", "BF16 - brain float", "Two bytes per weight with the exponent range of float32. Preferred over F16 for models trained in bfloat16.", False),
    QuantTypeInfo("q8_0", "Q8_0 - 8-bit", "Roughly one byte per weight and near-indistinguishable from F16. The default: the best quality-to-size trade for most users.", False),
    QuantTypeInfo("tq1_0", "TQ1_0 - ternary", "Extreme compression for ternary-trained models. Produces unusable output on ordinary models.", False),
    QuantTypeInfo("tq2_0", "TQ2_0 - ternary", "As TQ1_0, slightly larger and more accurate. Only for models trained ternary.", False),
    QuantTypeInfo("q2_K", "Q2_K - 2-bit k-quant", "Smallest usable k-quant. Noticeable quality loss; for when memory is the binding constraint.", True),
    QuantTypeInfo("q3_K_S", "Q3_K_S - 3-bit small", "Aggressive 3-bit packing with uniform block sizes.", True),
    QuantTypeInfo("q3_K_M", "Q3_K_M - 3-bit medium", "3-bit with more precision spent on attention weights. Better than Q3_K_S at a small size cost.", True),
    QuantTypeInfo("q3_K_L", "Q3_K_L - 3-bit large", "3-bit with the highest-impact tensors kept larger. Approaches Q4_K_S quality.", True),
    QuantTypeInfo("q4_K_S", "Q4_K_S - 4-bit small", "Compact 4-bit k-quant. A good floor for quality-sensitive work.", True),
    QuantTypeInfo("q4_K_M", "Q4_K_M - 4-bit medium", "The community default for local inference: about half the size of Q8_0 with modest quality loss.", True),
    QuantTypeInfo("q5_K_S", "Q5_K_S - 5-bit small", "5-bit k-quant; quality close to Q8_0 at a meaningfully smaller size.", True),
    QuantTypeInfo("q5_K_M", "Q5_K_M - 5-bit medium", "5-bit with extra precision on sensitive tensors. Very close to lossless in practice.", True),
    QuantTypeInfo("q6_K", "Q6_K - 6-bit", "Practically indistinguishable from Q8_0 while noticeably smaller.", True),
)

QUANT_TYPE_NAMES: tuple[str, ...] = tuple(info.name for info in QUANT_TYPES)


@dataclass(frozen=True)
class QuantSettings(Settings):
    """Which GGUF type to produce. Not part of a preset: quantization is a
    standalone operation that can be re-run against any previously merged model."""

    quant_type: str = param(
        DEFAULT_QUANT_TYPE,
        label="Quantization type",
        explanation=(
            "Output format of the GGUF file. Types marked as k-quants are produced "
            "by a second pass with llama-quantize and are unavailable until that "
            "binary is installed."
        ),
        kind=ParamKind.CHOICE,
        choices=QUANT_TYPE_NAMES,
    )


@dataclass(frozen=True)
class TrainingPreset:
    """A named, reusable set of training parameters. Never holds a model choice."""

    name: str
    lora: LoraSettings = field(default_factory=LoraSettings)
    sft: SftSettings = field(default_factory=SftSettings)


def describe(settings: Settings | type[Settings]) -> tuple[ParamSpec, ...]:
    """Return the presentation metadata for every tunable in a settings group.

    Accepts either a class or an instance, so front-ends can build a form before
    any values exist.
    """
    return tuple(_spec_from_field(source) for source in fields(settings) if "label" in source.metadata)


def quant_type_info(name: str) -> QuantTypeInfo:
    """Return the descriptor for one quantization type."""
    matches = [info for info in QUANT_TYPES if info.name == name]
    if not matches:
        raise ConfigError(f"Unknown quantization type {name!r}. Expected one of: {', '.join(QUANT_TYPE_NAMES)}.")
    return matches[0]


def available_quant_types(has_llama_quantize: bool) -> tuple[QuantTypeInfo, ...]:
    """Return the quantization types usable right now.

    Front-ends call this with the detected capability so k-quants disappear
    rather than failing part-way through a run.
    """
    return tuple(info for info in QUANT_TYPES if has_llama_quantize or not info.requires_llama_quantize)


def save_preset(preset: TrainingPreset, directory: Path | None = None) -> Path:
    """Write ``preset`` as JSON and return the file it was written to."""
    target = ensure_dir(directory or CONFIG_DIR) / f"{validated_preset_name(preset.name)}{PRESET_SUFFIX}"
    payload = {"name": preset.name, "lora": asdict(preset.lora), "sft": asdict(preset.sft)}
    target.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")
    return target


def load_preset(name: str, directory: Path | None = None) -> TrainingPreset:
    """Read a stored preset, rejecting unknown, missing or model-bearing keys."""
    source = (directory or CONFIG_DIR) / f"{validated_preset_name(name)}{PRESET_SUFFIX}"
    if not source.is_file():
        raise ConfigError(f"No preset named {name!r} in {source.parent}.")
    payload = _read_preset_json(source)
    _reject_model_keys(payload, "preset")
    return TrainingPreset(
        name=str(payload.get("name", name)),
        lora=_settings_from_dict(LoraSettings, payload.get("lora"), "lora"),
        sft=_settings_from_dict(SftSettings, payload.get("sft"), "sft"),
    )


def list_presets(directory: Path | None = None) -> tuple[str, ...]:
    """Return the names of every stored preset, alphabetically."""
    source_dir = directory or CONFIG_DIR
    names = () if not source_dir.is_dir() else tuple(sorted(path.stem for path in source_dir.glob(f"*{PRESET_SUFFIX}")))
    return names


def validated_preset_name(name: str) -> str:
    """Return ``name`` unchanged, or raise if it is unusable as a filename."""
    candidate = name.strip()
    if not PRESET_NAME_PATTERN.match(candidate):
        raise ConfigError(
            f"{name!r} is not a valid preset name. Use letters, digits, spaces, "
            "'.', '-' and '_', starting with a letter or digit."
        )
    return candidate


def validate_settings(settings: Settings) -> None:
    """Check every tunable in ``settings`` against its declared constraints."""
    for spec in describe(settings):
        _validate_value(spec, getattr(settings, spec.name))


def _spec_from_field(source: Field) -> ParamSpec:
    """Lift one dataclass field and its metadata into a :class:`ParamSpec`."""
    metadata = source.metadata
    return ParamSpec(
        name=source.name,
        label=metadata["label"],
        explanation=metadata["explanation"],
        kind=metadata["kind"],
        default=source.default,
        minimum=metadata["minimum"],
        maximum=metadata["maximum"],
        choices=metadata["choices"],
    )


def _validate_value(spec: ParamSpec, value: Any) -> None:
    """Raise :class:`ConfigError` if ``value`` does not satisfy ``spec``."""
    if spec.kind is ParamKind.BOOL:
        _require(isinstance(value, bool), spec, value, "a true/false value")
    elif spec.kind is ParamKind.INT:
        _require(isinstance(value, int) and not isinstance(value, bool), spec, value, "a whole number")
        _check_range(spec, value)
    elif spec.kind is ParamKind.FLOAT:
        _require(isinstance(value, (int, float)) and not isinstance(value, bool), spec, value, "a number")
        _check_range(spec, value)
    elif spec.kind is ParamKind.CHOICE:
        _require(value in (spec.choices or ()), spec, value, f"one of: {', '.join(spec.choices or ())}")
    elif spec.kind is ParamKind.MULTI_CHOICE:
        _check_multi_choice(spec, value)


def _check_range(spec: ParamSpec, value: float) -> None:
    """Raise if a numeric value falls outside the declared bounds."""
    if spec.minimum is not None and value < spec.minimum:
        raise ConfigError(f"{spec.label}: {value} is below the minimum of {spec.minimum}.")
    if spec.maximum is not None and value > spec.maximum:
        raise ConfigError(f"{spec.label}: {value} is above the maximum of {spec.maximum}.")


def _check_multi_choice(spec: ParamSpec, value: Any) -> None:
    """Raise unless ``value`` is a non-empty selection drawn from the choices."""
    allowed = spec.choices or ()
    _require(isinstance(value, tuple), spec, value, "a tuple of options")
    _require(bool(value), spec, value, "at least one option")
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ConfigError(f"{spec.label}: unknown option(s) {', '.join(unknown)}. Expected any of: {', '.join(allowed)}.")


def _require(condition: bool, spec: ParamSpec, value: Any, expected: str) -> None:
    """Raise a uniformly worded :class:`ConfigError` when ``condition`` fails."""
    if not condition:
        raise ConfigError(f"{spec.label}: expected {expected}, got {value!r}.")


def _read_preset_json(source: Path) -> dict[str, Any]:
    """Parse a preset file into a mapping, or raise with the file named."""
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigError(f"Preset {source} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ConfigError(f"Preset {source} must contain a JSON object.")
    return payload


def _settings_from_dict(settings_cls: type[Settings], payload: Any, section: str) -> Settings:
    """Rebuild a settings group from stored JSON, strictly.

    Unknown and missing keys are both errors: a preset is machine-written, so a
    mismatch means the file is stale rather than merely incomplete, and silently
    filling the gap with a default would hide that.
    """
    if not isinstance(payload, dict):
        raise ConfigError(f"Preset section '{section}' must be a JSON object.")
    _reject_model_keys(payload, section)
    specs = {spec.name: spec for spec in describe(settings_cls)}
    unknown = sorted(set(payload) - set(specs))
    if unknown:
        raise ConfigError(f"Preset section '{section}' has unknown parameter(s): {', '.join(unknown)}.")
    missing = sorted(set(specs) - set(payload))
    if missing:
        raise ConfigError(f"Preset section '{section}' is missing parameter(s): {', '.join(missing)}.")
    return settings_cls(**{name: _coerce(specs[name], value) for name, value in payload.items()})


def _coerce(spec: ParamSpec, value: Any) -> Any:
    """Restore the in-memory type JSON cannot represent, leaving the rest alone."""
    return tuple(value) if spec.kind is ParamKind.MULTI_CHOICE and isinstance(value, list) else value


def _reject_model_keys(payload: dict[str, Any], section: str) -> None:
    """Refuse any stored model identifier.

    Presets carry hyperparameters only. Allowing one to name a model would make
    loading it equivalent to having a default model, which the application does
    not have by design.
    """
    found = sorted(key for key in payload if key.lower() in MODEL_KEYS)
    if found:
        raise ConfigError(
            f"Preset section '{section}' names a model ({', '.join(found)}). Presets store "
            "training parameters only -- the model is always chosen explicitly for each run."
        )
