"""Converting a merged model to GGUF, and compressing it further.

There are two routes, and which one applies depends on the type requested.

``f32``, ``f16``, ``bf16``, ``q8_0``, ``tq1_0`` and ``tq2_0`` are produced in one
step by llama.cpp's ``convert_hf_to_gguf.py``.

The k-quants are not: they are a second pass performed by ``llama-quantize`` over
an f16 GGUF. So a k-quant is convert-then-quantize, and the intermediate f16 is
written to the same directory under its own name. That is deliberate -- ``f16``
is itself an offered output type, so the staging file *is* the f16 artefact, and
a later request for a different k-quant reuses it instead of re-converting.

This stage runs independently of training: it takes any merged model directory,
so previously produced models can be re-quantized at will.
"""

from __future__ import annotations

import re
from pathlib import Path

from llmft.config import DEFAULT_QUANT_TYPE, quant_type_info
from llmft.environment import find_convert_script, find_llama_quantize
from llmft.paths import InvalidModelIdError, InvalidNameError, ensure_dir, gguf_file, merged_dir, model_name
from llmft.process import ProcessError, run_streaming
from llmft.progress import ProgressSink, Stage, null_sink, report

# Staging format for the k-quant route: full precision in half the bytes.
STAGING_TYPE = "f16"

# "Writing:  47%|####7     | 127M/269M" -- the converter's tqdm bar.
CONVERT_PROGRESS = re.compile(r"Writing:\s+(\d+)%")

# "[   3/ 272] blk.0.attn_k.weight - ..." -- llama-quantize, one line per tensor.
QUANTIZE_PROGRESS = re.compile(r"^\[\s*(\d+)\s*/\s*(\d+)\s*\]")


class QuantizationError(RuntimeError):
    """Raised when a model cannot be converted or quantized."""


def quantize(
    model_id: str,
    quant_type: str = DEFAULT_QUANT_TYPE,
    source: Path | None = None,
    destination: Path | None = None,
    sink: ProgressSink = null_sink,
    name: str = "",
) -> Path:
    """Produce a GGUF file and return its path.

    ``source`` defaults to this application's merged model for ``model_id``, so
    the usual call is ``quantize(model_id)`` or ``quantize(model_id, "q4_K_M")``.
    ``name`` names the quantized model -- its file and the name recorded inside
    it -- and defaults to the original model's name.
    """
    info = quant_type_info(quant_type)
    try:
        gguf_name = name.strip() or model_name(model_id)
        target = destination or gguf_file(model_id, quant_type, gguf_name)
    except (InvalidModelIdError, InvalidNameError) as error:
        raise QuantizationError(str(error)) from error
    merged = source or merged_dir(model_id)
    if not merged.is_dir():
        raise QuantizationError(f"No merged model at {merged}. Train and merge one first.")
    try:
        ensure_dir(target.parent)
    except OSError as error:
        raise QuantizationError(
            f"Cannot create {target.parent}: {error.strerror or error}. "
            "Choose another output model directory in Setup."
        ) from error

    if info.requires_llama_quantize:
        staging = _convert(gguf_name, merged, gguf_file(model_id, STAGING_TYPE, gguf_name), STAGING_TYPE, sink)
        _run_llama_quantize(staging, target, quant_type, sink)
    else:
        _convert(gguf_name, merged, target, quant_type, sink)

    report(sink, Stage.QUANTIZE, f"Wrote {target.name} ({_megabytes(target)} MB)", 1, 1)
    return target


def _convert(gguf_name: str, merged: Path, target: Path, outtype: str, sink: ProgressSink) -> Path:
    """Run ``convert_hf_to_gguf.py``, reusing an existing staging file."""
    if target.is_file():
        report(sink, Stage.QUANTIZE, f"Reusing {target.name}")
        return target

    script = find_convert_script()
    if script is None:
        raise QuantizationError("convert_hf_to_gguf.py is not installed. Run: python run.py setup")

    report(sink, Stage.QUANTIZE, f"Converting to GGUF {outtype}", 0, 100)
    command = (
        _interpreter(),
        str(script),
        str(merged),
        "--outfile",
        str(target),
        "--outtype",
        outtype,
        "--model-name",
        gguf_name,
    )
    _run(command, f"GGUF conversion to {outtype}", sink, _convert_reporter(sink, outtype))
    return target


def _run_llama_quantize(staging: Path, target: Path, quant_type: str, sink: ProgressSink) -> None:
    """Run the second pass that produces a k-quant from the f16 GGUF."""
    binary = find_llama_quantize()
    if binary is None:
        raise QuantizationError(
            f"{quant_type} needs llama-quantize, which is not installed. Run: python run.py setup"
        )
    report(sink, Stage.QUANTIZE, f"Quantizing to {quant_type}", 0, 100)
    command = (str(binary), str(staging), str(target), quant_type)
    _run(command, f"Quantization to {quant_type}", sink, _quantize_reporter(sink, quant_type))


def _convert_reporter(sink: ProgressSink, outtype: str):
    """Translate the converter's tqdm percentage into progress events."""

    def handle(line: str) -> None:
        match = CONVERT_PROGRESS.search(line)
        if match:
            report(sink, Stage.QUANTIZE, f"Writing GGUF {outtype}", int(match.group(1)), 100)

    return handle


def _quantize_reporter(sink: ProgressSink, quant_type: str):
    """Translate llama-quantize's per-tensor counter into progress events."""

    def handle(line: str) -> None:
        match = QUANTIZE_PROGRESS.match(line)
        if match:
            current, total = int(match.group(1)), int(match.group(2))
            report(sink, Stage.QUANTIZE, f"Quantizing to {quant_type}: tensor {current}/{total}", current, total)

    return handle


def _run(command: tuple[str, ...], label: str, sink: ProgressSink, handle) -> None:
    """Run one external step, converting failures into a stage error."""
    try:
        run_streaming(command, handle, label)
    except ProcessError as error:
        raise QuantizationError(str(error)) from error


def _interpreter() -> str:
    """The Python that runs the converter.

    The converter imports its sibling ``conversion`` package and relies on
    ``sys.path[0]`` being its own directory, which happens only when it is
    invoked by file path. It must never be run with ``-m``.
    """
    import sys

    return sys.executable


def _megabytes(path: Path) -> int:
    """Size of a produced file, for the completion message."""
    return round(path.stat().st_size / 1e6) if path.is_file() else 0
