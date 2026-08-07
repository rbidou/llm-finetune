"""Pre-flight memory estimates, and the gate that stops a run from OOM-ing.

Because there is no default model, the user picks one freely -- and on a 6 GB
laptop most of the interesting choices do not fit. This module is what makes
that free choice safe: it predicts peak memory *before* anything is downloaded
and, when a run cannot fit, says by how much and what to change.

The estimates are deliberately conservative and the breakdown is always
reported, because the point is to inform a decision rather than to be exactly
right. Every assumption is a named constant below.

One term is worth singling out. The logits tensor is
``batch x sequence x vocabulary``, and for a large vocabulary it dominates
everything else: at 2048 tokens with Qwen's 152k vocabulary it alone approaches
2 GB. It is the most common cause of an OOM that looks inexplicable from the
model size, so it appears in the breakdown by name.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from llmft.config import ATTENTION_MODULES, LoraSettings, SftSettings
from llmft.environment import GpuInfo

# 4-bit NF4 weights, plus the double-quantized scale constants that accompany them.
NF4_BYTES_PER_PARAM = 0.55
BF16_BYTES_PER_PARAM = 2

# Trainable adapter weights carry a gradient and two fp32 AdamW moments.
ADAPTER_BYTES_PER_PARAM = BF16_BYTES_PER_PARAM + BF16_BYTES_PER_PARAM + 8

# With gradient checkpointing only one tensor per layer is kept; without it the
# attention and MLP intermediates are retained as well.
CHECKPOINTED_TENSORS_PER_LAYER = 1
RETAINED_TENSORS_PER_LAYER = 10

# Logits are produced in bf16 then upcast to fp32 for the cross-entropy loss,
# and the gradient of the upcast copy is live at the same time.
LOGITS_BYTES_PER_ELEMENT = 6

# CUDA context, cuBLAS workspaces and allocator fragmentation.
CUDA_OVERHEAD_BYTES = 800 * 1024**2

# A GPU driving a display never has all of its memory available.
USABLE_VRAM_FRACTION = 0.90

# Merging holds the base model in bf16 and writes shards from it.
MERGE_OVERHEAD_FACTOR = 1.25
MERGE_BASE_OVERHEAD_BYTES = 1024**3

# Input and output dimensions of each adapted projection, as config keys.
MODULE_DIMENSIONS: dict[str, tuple[str, str]] = {
    "q_proj": ("hidden", "hidden"),
    "k_proj": ("hidden", "hidden"),
    "v_proj": ("hidden", "hidden"),
    "o_proj": ("hidden", "hidden"),
    "gate_proj": ("hidden", "intermediate"),
    "up_proj": ("hidden", "intermediate"),
    "down_proj": ("intermediate", "hidden"),
}

_SEQUENCE_LENGTH_LADDER = (4096, 2048, 1024, 512, 256)
_RANK_LADDER = (16, 8)

# How many partial savings to list when no single change is sufficient.
MAX_PARTIAL_REMEDIES = 4


@dataclass(frozen=True)
class ModelShape:
    """The dimensions of a model, as the estimate needs them."""

    model_id: str
    parameter_count: int
    num_layers: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int

    @classmethod
    def from_config(cls, model_id: str, config: dict[str, Any], parameter_count: int | None = None) -> "ModelShape":
        """Build a shape from a HuggingFace ``config.json``.

        ``parameter_count`` comes from the safetensors index when available; it
        is derived from the dimensions otherwise, which is close enough for a
        memory estimate.
        """
        text_config = config.get("text_config") or config
        layers = int(text_config.get("num_hidden_layers") or 0)
        hidden = int(text_config.get("hidden_size") or 0)
        intermediate = int(text_config.get("intermediate_size") or hidden * 4)
        vocab = int(text_config.get("vocab_size") or 0)
        if not (layers and hidden and vocab):
            raise ValueError(f"config.json for {model_id} is missing the dimensions needed to size a run")
        derived = _derive_parameter_count(layers, hidden, intermediate, vocab)
        return cls(
            model_id=model_id,
            parameter_count=int(parameter_count or derived),
            num_layers=layers,
            hidden_size=hidden,
            intermediate_size=intermediate,
            vocab_size=vocab,
        )


@dataclass(frozen=True)
class MemoryEstimate:
    """Predicted peak memory for each stage of a run."""

    training_bytes: int
    merge_ram_bytes: int
    merged_disk_bytes: int
    components: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class Verdict:
    """Whether a run is expected to fit, and what to do when it is not."""

    fits: bool
    required_bytes: int
    available_bytes: int
    headline: str
    remedies: tuple[str, ...] = ()

    @property
    def shortfall_bytes(self) -> int:
        """How much memory the run is short of, or zero when it fits."""
        return max(0, self.required_bytes - self.available_bytes)


def check_fit(shape: ModelShape, lora: LoraSettings, sft: SftSettings, gpu: GpuInfo | None) -> Verdict:
    """Decide whether a training run can proceed on this GPU.

    This is a gate rather than a hint. A run predicted not to fit is refused,
    with the numbers and a list of changes that would make it fit; the caller
    may still override, but only deliberately.
    """
    estimate = estimate_training(shape, lora, sft)
    if gpu is None:
        return Verdict(
            fits=False,
            required_bytes=estimate.training_bytes,
            available_bytes=0,
            headline="No CUDA GPU detected. 4-bit QLoRA training requires one.",
            remedies=("Install an NVIDIA GPU driver, or run the merge and quantize stages only.",),
        )

    available = usable_vram_bytes(gpu)
    fits = estimate.training_bytes <= available
    headline = (
        f"Estimated peak {format_bytes(estimate.training_bytes)} of VRAM for {shape.model_id}; "
        f"{format_bytes(available)} usable of {gpu.total_memory_mb} MiB on {gpu.name}."
    )
    remedies = () if fits else suggest_remedies(shape, lora, sft, available)
    if not fits and gpu.is_heavily_occupied:
        remedies = (
            f"Close other GPU applications: {gpu.occupied_memory_mb} MiB of "
            f"{gpu.total_memory_mb} MiB is already held by something else.",
            *remedies,
        )
    return Verdict(fits, estimate.training_bytes, available, headline, remedies)


def estimate_training(shape: ModelShape, lora: LoraSettings, sft: SftSettings) -> MemoryEstimate:
    """Break peak training memory down by contributor."""
    batch = sft.per_device_train_batch_size
    tokens = batch * sft.max_length
    retained = CHECKPOINTED_TENSORS_PER_LAYER if sft.gradient_checkpointing else RETAINED_TENSORS_PER_LAYER

    base_weights = int(shape.parameter_count * NF4_BYTES_PER_PARAM)
    adapter = int(adapter_parameter_count(shape, lora) * ADAPTER_BYTES_PER_PARAM)
    activations = int(tokens * shape.hidden_size * BF16_BYTES_PER_PARAM * shape.num_layers * retained)
    logits = int(tokens * shape.vocab_size * LOGITS_BYTES_PER_ELEMENT)

    components = (
        ("Base weights (4-bit)", base_weights),
        ("Adapters, gradients, optimizer", adapter),
        ("Activations", activations),
        ("Logits and loss", logits),
        ("CUDA overhead", CUDA_OVERHEAD_BYTES),
    )
    merged = shape.parameter_count * BF16_BYTES_PER_PARAM
    return MemoryEstimate(
        training_bytes=sum(size for _, size in components),
        merge_ram_bytes=int(merged * MERGE_OVERHEAD_FACTOR) + MERGE_BASE_OVERHEAD_BYTES,
        merged_disk_bytes=merged,
        components=components,
    )


def adapter_parameter_count(shape: ModelShape, lora: LoraSettings) -> int:
    """Count the trainable parameters the chosen LoRA configuration adds."""
    dimensions = {"hidden": shape.hidden_size, "intermediate": shape.intermediate_size}
    per_layer = sum(
        lora.r * (dimensions[MODULE_DIMENSIONS[module][0]] + dimensions[MODULE_DIMENSIONS[module][1]])
        for module in lora.target_modules
        if module in MODULE_DIMENSIONS
    )
    return per_layer * shape.num_layers


def suggest_remedies(shape: ModelShape, lora: LoraSettings, sft: SftSettings, budget: int) -> tuple[str, ...]:
    """Propose setting changes that would bring a run within ``budget``.

    Each candidate is re-estimated rather than guessed at, so a suggestion is
    only offered when it genuinely helps, and the saving quoted is the real one.
    """
    current = estimate_training(shape, lora, sft).training_bytes
    evaluated = [
        _remedy(shape, budget, current, label, new_lora, new_sft)
        for label, new_lora, new_sft in _candidates(lora, sft)
    ]
    scored = [outcome for outcome in evaluated if outcome is not None]
    sufficient = tuple(text for is_enough, text in scored if is_enough)
    if sufficient:
        return sufficient
    partial = tuple(text for _, text in scored[:MAX_PARTIAL_REMEDIES])
    return ("No single change is enough; combine several, or choose a smaller model.", *partial)


def usable_vram_bytes(gpu: GpuInfo) -> int:
    """Memory realistically available for a run right now.

    Measured against *free* memory rather than total, because a card is rarely
    idle: a game, a browser or an earlier run that has not exited can leave a
    6 GB GPU with a couple of hundred megabytes available. Budgeting against
    total memory would approve a run that cannot possibly start.

    Older drivers that do not report free memory fall back to a fraction of the
    total, which is the best available guess.
    """
    reported_free = gpu.free_memory_mb or int(gpu.total_memory_mb * USABLE_VRAM_FRACTION)
    return int(min(reported_free, gpu.total_memory_mb * USABLE_VRAM_FRACTION) * 1024**2)


def format_bytes(size: int) -> str:
    """Format a byte count for a person reading a warning."""
    gigabytes = size / 1024**3
    return f"{gigabytes:.2f} GB" if gigabytes >= 1 else f"{size / 1024**2:.0f} MB"


def _candidates(lora: LoraSettings, sft: SftSettings) -> tuple[tuple[str, LoraSettings, SftSettings], ...]:
    """Enumerate the changes worth trying, cheapest in quality cost first."""
    options: list[tuple[str, LoraSettings, SftSettings]] = []
    if not sft.gradient_checkpointing:
        options.append(("Enable gradient checkpointing", lora, replace(sft, gradient_checkpointing=True)))
    if sft.per_device_train_batch_size > 1:
        options.append(("Set batch size per device to 1", lora, replace(sft, per_device_train_batch_size=1)))
    options.extend(
        (f"Reduce maximum sequence length to {length}", lora, replace(sft, max_length=length))
        for length in _SEQUENCE_LENGTH_LADDER
        if length < sft.max_length
    )
    if tuple(sorted(lora.target_modules)) != tuple(sorted(ATTENTION_MODULES)):
        options.append(("Target attention projections only", replace(lora, target_modules=ATTENTION_MODULES), sft))
    options.extend(
        (f"Reduce LoRA rank to {rank}", replace(lora, r=rank, lora_alpha=rank * 2), sft)
        for rank in _RANK_LADDER
        if rank < lora.r
    )
    return tuple(options)


def _remedy(
    shape: ModelShape,
    budget: int,
    current: int,
    label: str,
    new_lora: LoraSettings,
    new_sft: SftSettings,
) -> tuple[bool, str] | None:
    """Evaluate one candidate change, or return ``None`` if it saves nothing."""
    projected = estimate_training(shape, new_lora, new_sft).training_bytes
    saved = current - projected
    is_enough = projected <= budget
    outcome = "fits" if is_enough else f"still {format_bytes(projected - budget)} short"
    return (is_enough, f"{label}: saves {format_bytes(saved)} ({outcome})") if saved > 0 else None


def _derive_parameter_count(layers: int, hidden: int, intermediate: int, vocab: int) -> int:
    """Approximate a parameter count from a model's dimensions.

    Covers the embedding and output head, the four attention projections and the
    three MLP projections. Used only when the safetensors index is unavailable.
    """
    embeddings = 2 * vocab * hidden
    attention = 4 * hidden * hidden
    mlp = 3 * hidden * intermediate
    return embeddings + layers * (attention + mlp)
