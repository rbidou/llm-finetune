"""JSON endpoints over the engine.

Every route is a thin adapter: it converts JSON into the engine's own types,
calls exactly the function the CLI calls, and converts the result back. No
decision about training, quantizing or deploying is made here.

Long stages return a job identifier immediately and report through
``/api/jobs/stream``; a request cannot be held open for the twenty minutes a
fine-tune takes.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from flask import Blueprint, Response, current_app, jsonify, request

from llmft.config import (
    ConfigError,
    LoraSettings,
    QuantSettings,
    SftSettings,
    TrainingPreset,
    available_quant_types,
    describe,
    list_presets,
    load_preset,
    save_preset,
)
from llmft.dataset import DatasetReport, validate_dataset
from llmft.environment import detect, features
from llmft.estimator import ModelShape, check_fit, format_bytes
from llmft.hub import HubError, fetch_config, fetch_parameter_count, local_models, search_models
from llmft.paths import ADAPTER_SUBDIR, GGUF_SUBDIR, MERGED_SUBDIR, OUT_DIR, ensure_dir, merged_dir, model_name
from llmft.pipeline import (
    available_targets,
    deploy_to_lm_studio,
    deploy_to_ollama,
    ollama_name,
    quantize,
    suggested_name,
    train_and_merge,
)
from llmft.progress import ProgressEvent
from llmft.web.jobs import Job, JobBusy

# The specification asks the dropdown for the top twenty.
SEARCH_LIMIT = 20
UPLOAD_DIR = Path("temp") / "uploads"

api = Blueprint("api", __name__, url_prefix="/api")


# --- environment -----------------------------------------------------------


@api.get("/capabilities")
def get_capabilities() -> Response:
    """What this machine can do, and what it cannot do and why."""
    capabilities = detect()
    gpu = capabilities.gpu
    return jsonify(
        {
            "features": [asdict(item) for item in features(capabilities)],
            "targets": [asdict(item) for item in available_targets(capabilities)],
            "quantTypes": [asdict(item) for item in available_quant_types(capabilities.llama_quantize is not None)],
            "defaultQuantType": QuantSettings().quant_type,
            "gpu": _gpu_payload(gpu) if gpu else None,
        }
    )


def _gpu_payload(gpu: Any) -> dict[str, Any]:
    """Describe the GPU, including how much of it something else is using."""
    return {
        "name": gpu.name,
        "totalMemoryMb": gpu.total_memory_mb,
        "freeMemoryMb": gpu.free_memory_mb,
        "occupiedMemoryMb": gpu.occupied_memory_mb,
        "heavilyOccupied": gpu.is_heavily_occupied,
        "supportsBf16": gpu.supports_bf16,
    }


# --- models ----------------------------------------------------------------


@api.get("/models/search")
def get_model_search() -> Response:
    """Models whose name starts with the query, best first."""
    query = request.args.get("q", "")
    try:
        found = search_models(query, limit=SEARCH_LIMIT)
    except HubError as error:
        return _failure(str(error), 502)
    return jsonify(
        [
            {
                "modelId": item.model_id,
                "author": item.author,
                "downloads": item.downloads,
                "trendingScore": item.trending_score,
                "isLocal": item.is_local,
                "knownPublisher": item.is_known_publisher,
            }
            for item in found
        ]
    )


@api.get("/models/local")
def get_local_models() -> Response:
    """Cached models that can actually be fine-tuned."""
    try:
        cached = [item for item in local_models() if item.is_causal_lm]
    except HubError as error:
        return _failure(str(error), 500)
    return jsonify(
        [
            {"modelId": item.model_id, "description": item.description, "sizeOnDisk": item.size_on_disk}
            for item in cached
        ]
    )


@api.get("/models/merged")
def get_merged_models() -> Response:
    """Models that have been merged and are ready to quantize."""
    found = sorted(_model_id_from_artefact(path) for path in OUT_DIR.glob(f"**/{MERGED_SUBDIR}"))
    return jsonify([{"modelId": name, "sizeOnDisk": _directory_size(merged_dir(name))} for name in found])


@api.get("/models/gguf")
def get_gguf_files() -> Response:
    """Every GGUF this application has produced, newest first."""
    found = sorted(OUT_DIR.glob(f"**/{GGUF_SUBDIR}/*.gguf"), key=lambda path: -path.stat().st_mtime)
    return jsonify(
        [
            {
                "path": str(path),
                "name": path.name,
                "sizeOnDisk": path.stat().st_size,
                "modelId": _model_id_from_artefact(path),
                "quantType": _quant_type_from_gguf(path, _model_id_from_artefact(path)),
            }
            for path in found
        ]
    )


# --- dataset ---------------------------------------------------------------


@api.post("/dataset")
def post_dataset() -> Response:
    """Validate an uploaded dataset, reporting every problem at once."""
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return _failure("No file was uploaded.", 400)
    target = ensure_dir(UPLOAD_DIR) / Path(uploaded.filename).name
    uploaded.save(target)
    return jsonify(_dataset_payload(validate_dataset(target)))


def _dataset_payload(report: DatasetReport) -> dict[str, Any]:
    """Describe a validated dataset for the browser."""
    return {
        "path": str(report.path),
        "valid": report.is_valid,
        "summary": report.summary() if report.is_valid else "",
        "recordCount": report.record_count,
        "issues": [{"location": issue.location, "problem": issue.problem} for issue in report.issues],
        "truncated": report.truncated,
    }


# --- parameters and presets ------------------------------------------------


@api.get("/parameters")
def get_parameters() -> Response:
    """Every tunable, with the label and explanation the form needs.

    This is the same metadata the CLI editor and the manual are built from, so
    the three can never describe a parameter differently.
    """
    return jsonify(
        {
            "lora": [_spec_payload(spec) for spec in describe(LoraSettings)],
            "sft": [_spec_payload(spec) for spec in describe(SftSettings)],
        }
    )


def _spec_payload(spec: Any) -> dict[str, Any]:
    """Describe one tunable for the form."""
    return {
        "name": spec.name,
        "label": spec.label,
        "explanation": spec.explanation,
        "kind": spec.kind.value,
        "default": list(spec.default) if isinstance(spec.default, tuple) else spec.default,
        "minimum": spec.minimum,
        "maximum": spec.maximum,
        "choices": list(spec.choices) if spec.choices else None,
    }


@api.get("/presets")
def get_presets() -> Response:
    """The names of every stored preset."""
    return jsonify(list(list_presets()))


@api.get("/presets/<name>")
def get_preset(name: str) -> Response:
    """One stored preset's parameters."""
    try:
        preset = load_preset(name)
    except ConfigError as error:
        return _failure(str(error), 404)
    return jsonify({"name": preset.name, "lora": asdict(preset.lora), "sft": asdict(preset.sft)})


@api.post("/presets")
def post_preset() -> Response:
    """Store the submitted parameters under a name."""
    payload = request.get_json(silent=True) or {}
    try:
        settings = _settings_from(payload)
        target = save_preset(TrainingPreset(name=str(payload.get("name", "")), lora=settings[0], sft=settings[1]))
    except ConfigError as error:
        return _failure(str(error), 400)
    return jsonify({"path": str(target)})


# --- estimate --------------------------------------------------------------


@api.post("/estimate")
def post_estimate() -> Response:
    """Predict peak memory for a run before anything is downloaded."""
    payload = request.get_json(silent=True) or {}
    model_id = str(payload.get("modelId", ""))
    try:
        lora, sft = _settings_from(payload)
        shape = ModelShape.from_config(model_id, fetch_config(model_id), fetch_parameter_count(model_id))
    except (ConfigError, HubError, ValueError) as error:
        return _failure(str(error), 400)

    verdict = check_fit(shape, lora, sft, detect().gpu)
    estimate = _estimate_payload(shape, verdict)
    return jsonify(estimate)


def _estimate_payload(shape: ModelShape, verdict: Any) -> dict[str, Any]:
    """Describe a pre-flight verdict for the browser."""
    return {
        "modelId": shape.model_id,
        "parameterCount": shape.parameter_count,
        "vocabSize": shape.vocab_size,
        "fits": verdict.fits,
        "headline": verdict.headline,
        "requiredBytes": verdict.required_bytes,
        "availableBytes": verdict.available_bytes,
        "shortfall": format_bytes(verdict.shortfall_bytes) if verdict.shortfall_bytes else "",
        "remedies": list(verdict.remedies),
    }


# --- jobs ------------------------------------------------------------------


@api.post("/jobs/train")
def post_train() -> Response:
    """Start a fine-tune, which merges automatically when it succeeds."""
    payload = request.get_json(silent=True) or {}
    dataset = Path(str(payload.get("dataset", "")))
    if not dataset.is_file():
        return _failure(f"No dataset at {dataset}.", 400)
    try:
        lora, sft = _settings_from(payload)
    except ConfigError as error:
        return _failure(str(error), 400)
    model_id = str(payload.get("modelId", ""))
    return _start("train", lambda sink: train_and_merge(model_id, dataset, lora, sft, sink=sink))


@api.post("/jobs/quantize")
def post_quantize() -> Response:
    """Start a quantization of an already merged model."""
    payload = request.get_json(silent=True) or {}
    model_id = str(payload.get("modelId", ""))
    quant_type = str(payload.get("quantType", QuantSettings().quant_type))
    return _start("quantize", lambda sink: quantize(model_id, quant_type, sink=sink))


@api.post("/jobs/deploy")
def post_deploy() -> Response:
    """Start a deployment to a local model runner."""
    payload = request.get_json(silent=True) or {}
    gguf = Path(str(payload.get("path", "")))
    if not gguf.is_file():
        return _failure(f"No GGUF file at {gguf}.", 400)
    runner = str(payload.get("runner", ""))
    model_id = str(payload.get("modelId", ""))
    name = str(payload.get("name", "")) or _default_name(gguf, model_id, str(payload.get("quantType", "")))
    if runner == "ollama":
        return _start("deploy", lambda sink: deploy_to_ollama(gguf, name, sink=sink))
    return _start("deploy", lambda sink: deploy_to_lm_studio(gguf, model_id or None, sink=sink))


@api.get("/jobs/current")
def get_current_job() -> Response:
    """The running or most recently finished stage."""
    job = current_app.extensions["llmft_jobs"].current
    return jsonify(_job_payload(job) if job else None)


@api.get("/jobs/stream")
def get_job_stream() -> Response:
    """Stream a stage's progress as server-sent events."""
    job = current_app.extensions["llmft_jobs"].current
    if job is None:
        return _failure("No job is running.", 404)
    return Response(_events(job), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})


def _events(job: Job) -> Iterator[str]:
    """Yield every event the browser has not seen, then the final state.

    Polling the job rather than waiting on a condition keeps this simple and
    lets the loop notice a finished stage even if it reported nothing at all.
    """
    import time

    cursor = 0
    while True:
        pending, cursor = job.since(cursor)
        for event in pending:
            yield _sse("progress", _event_payload(event))
        if job.finished:
            yield _sse("done", _job_payload(job))
            return
        time.sleep(0.2)


def _sse(name: str, payload: dict[str, Any]) -> str:
    """Format one server-sent event."""
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


def _event_payload(event: ProgressEvent) -> dict[str, Any]:
    """Describe one progress event for the browser."""
    return {
        "stage": event.stage.value,
        "message": event.message,
        "current": event.current,
        "total": event.total,
        "fraction": event.fraction,
    }


def _job_payload(job: Job) -> dict[str, Any]:
    """Describe a job's state for the browser."""
    latest = job.latest()
    return {
        "id": job.identifier,
        "kind": job.kind,
        "state": job.state.value,
        "result": job.result,
        "error": job.error,
        "finished": job.finished,
        "latest": _event_payload(latest) if latest else None,
    }


def _start(kind: str, work: Any) -> Response:
    """Hand a stage to the runner, refusing if one is already going."""
    try:
        job = current_app.extensions["llmft_jobs"].start(kind, work)
    except JobBusy as error:
        return _failure(str(error), 409)
    return jsonify(_job_payload(job))


# --- shared ----------------------------------------------------------------


def _settings_from(payload: dict[str, Any]) -> tuple[LoraSettings, SftSettings]:
    """Build validated settings from submitted JSON.

    Validation lives in the dataclasses, so a value the CLI would reject is
    rejected here too, with the same message.
    """
    lora = dict(payload.get("lora") or {})
    if "target_modules" in lora:
        lora["target_modules"] = tuple(lora["target_modules"])
    return LoraSettings(**lora), SftSettings(**dict(payload.get("sft") or {}))


def _default_name(gguf: Path, model_id: str, quant_type: str) -> str:
    """Propose a runner name, matching what the CLI would offer."""
    return suggested_name(model_id, quant_type) if model_id and quant_type else ollama_name(gguf.stem)


def _model_id_from_artefact(path: Path) -> str:
    """Recover the model identifier from an artefact path under ``out/``."""
    parts = path.relative_to(OUT_DIR).parts
    boundary = next(
        (index for index, part in enumerate(parts) if part in {MERGED_SUBDIR, GGUF_SUBDIR, ADAPTER_SUBDIR}),
        len(parts),
    )
    return "/".join(parts[:boundary])


def _quant_type_from_gguf(path: Path, model_id: str) -> str:
    """Read the quantization type back out of a produced filename."""
    prefix = f"{model_name(model_id)}-" if model_id else ""
    return path.stem[len(prefix):] if prefix and path.stem.startswith(prefix) else ""


def _directory_size(path: Path) -> int:
    """Total size of everything under a directory."""
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.is_dir() else 0


def _failure(message: str, status: int) -> Response:
    """Return an error the browser can display verbatim."""
    response = jsonify({"error": message})
    response.status_code = status
    return response
