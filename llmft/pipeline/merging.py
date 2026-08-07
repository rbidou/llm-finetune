"""Merging a trained adapter back into the base weights.

A LoRA adapter is a pair of small matrices per targeted layer. Merging computes
``W + (B @ A) * (alpha / r)`` for each of them and folds the result into the base
weight, producing an ordinary model with no adapter and no inference overhead --
and, importantly, one that GGUF conversion can read.

This stage runs on the CPU in bfloat16, so it is bounded by system RAM rather
than VRAM: roughly two bytes per parameter, about 15 GB for a 7B model. The base
model must be loaded unquantized, because merging into 4-bit weights would
discard most of what was trained.
"""

from __future__ import annotations

from pathlib import Path

from llmft.paths import adapter_dir, ensure_dir, merged_dir
from llmft.progress import CapturedOutput, ProgressSink, Stage, null_sink, quiet, report

# Bounds peak memory while writing: the whole model is never serialised at once.
MAX_SHARD_SIZE = "2GB"

TOTAL_PHASES = 4
FAILURE_CONTEXT_LINES = 15


class MergeError(RuntimeError):
    """Raised when an adapter cannot be merged into its base model."""


def merge(
    model_id: str,
    adapter: Path | None = None,
    destination: Path | None = None,
    sink: ProgressSink = null_sink,
) -> Path:
    """Merge an adapter into the base weights and return the saved model directory.

    Defaults to the adapter and output locations this application uses, so the
    usual call is just ``merge(model_id)``.
    """
    source = adapter or adapter_dir(model_id)
    target = destination or merged_dir(model_id)
    if not source.is_dir():
        raise MergeError(f"No adapter at {source}. Train one first, or point at an existing adapter.")

    with quiet() as captured:
        try:
            _run(model_id, source, target, sink)
        except Exception as error:  # noqa: BLE001 - reported with the library output
            raise MergeError(_failure_message(error, captured)) from error
    report(sink, Stage.MERGE, f"Merged model saved to {target}", TOTAL_PHASES, TOTAL_PHASES)
    return target


def _run(model_id: str, source: Path, target: Path, sink: ProgressSink) -> None:
    """Perform the merge, inside the silenced context."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    report(sink, Stage.MERGE, f"Loading {model_id} at full precision", 0, TOTAL_PHASES)
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)

    report(sink, Stage.MERGE, "Applying adapter", 1, TOTAL_PHASES)
    adapted = PeftModel.from_pretrained(base, str(source))

    report(sink, Stage.MERGE, "Folding adapter into base weights", 2, TOTAL_PHASES)
    merged = adapted.merge_and_unload()

    report(sink, Stage.MERGE, f"Writing model to {target}", 3, TOTAL_PHASES)
    ensure_dir(target)
    merged.save_pretrained(str(target), max_shard_size=MAX_SHARD_SIZE)
    _save_tokenizer(model_id, source, target)


def _save_tokenizer(model_id: str, source: Path, target: Path) -> None:
    """Copy the tokenizer next to the merged weights.

    The adapter's own tokenizer is preferred: training may have added a padding
    token, and a merged model whose tokenizer disagrees with its weights will
    produce subtly wrong output.
    """
    from transformers import AutoTokenizer

    origin = source if (source / "tokenizer_config.json").is_file() else model_id
    AutoTokenizer.from_pretrained(str(origin)).save_pretrained(str(target))


def _failure_message(error: Exception, captured: CapturedOutput) -> str:
    """Combine the exception with the library output that led up to it."""
    context = captured.tail(FAILURE_CONTEXT_LINES)
    detail = f"\n\nLast output before the failure:\n{context}" if context.strip() else ""
    return f"Merge failed: {type(error).__name__}: {error}{detail}"
