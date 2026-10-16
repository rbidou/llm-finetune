"""JSON endpoints over the engine.

Every route is a thin adapter: it converts JSON into the engine's own types,
calls exactly the function the CLI calls, and converts the result back. No
decision about training, quantizing or deploying is made here.

All of them answer with the standard envelope from :mod:`llmft.web.responses`,
built once at the end of the function from a result tuple initialised at the
top. The single exception is ``/api/jobs/stream``, which is a server-sent event
stream rather than a JSON document; its refusal path still uses the envelope.

Long stages return a job identifier immediately and report through that stream;
a request cannot be held open for the twenty minutes a fine-tune takes.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from flask import Blueprint, Response, current_app, request

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
from llmft.folders import FolderError, FolderListing, list_folder
from llmft.estimator import ModelShape, check_fit, format_bytes
from llmft.hub import HubError, fetch_config, fetch_parameter_count, local_models, search_models
from llmft.paths import (
    ADAPTER_SUBDIR,
    DEFAULT_OUT_DIR,
    GGUF_SUBDIR,
    MERGED_SUBDIR,
    OutputDirError,
    ensure_dir,
    merged_dir,
    model_name,
    output_root,
    set_output_root,
)
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
from llmft.server_address import DEFAULT_ADDRESS, ServerAddressError, server_address, set_server_address
from llmft.web.jobs import Job, JobBusy
from llmft.web.responses import check_json_data, make_api_response

# The specification asks the dropdown for the top twenty.
SEARCH_LIMIT = 20
UPLOAD_DIR = Path("temp") / "uploads"

# A chat dataset is text, so this is generous; it exists to stop a mistaken
# upload of a model file. Enforced in the browser too, and by MAX_CONTENT_LENGTH.
MAX_DATASET_SIZE = 32 * 1024 * 1024

HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404
HTTP_CONFLICT = 409
HTTP_UNPROCESSABLE = 422
HTTP_BAD_GATEWAY = 502

api = Blueprint("api", __name__, url_prefix="/api")


# --- environment -----------------------------------------------------------


@api.get("/capabilities")
def api_capabilities() -> dict[str, Any]:
    """What this machine can do, and what it cannot do and why."""
    capabilities = detect()
    gpu = capabilities.gpu
    data = {
        "features": [asdict(item) for item in features(capabilities)],
        "targets": [asdict(item) for item in available_targets(capabilities)],
        "quantTypes": [asdict(item) for item in available_quant_types(capabilities.llama_quantize is not None)],
        "defaultQuantType": QuantSettings().quant_type,
        "gpu": _gpu_payload(gpu) if gpu else None,
    }

    return make_api_response(data=data)


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
def api_models_search() -> dict[str, Any] | tuple[dict[str, Any], int]:
    """Models whose name starts with the query, best first."""
    (result, message, data, status_code, reflect) = (True, "", {}, 200, True)

    try:
        found = search_models(request.args.get("q", ""), limit=SEARCH_LIMIT)
    except HubError as error:
        result = False
        message = str(error)
        status_code = HTTP_BAD_GATEWAY
    else:
        data = {"models": [_search_result_payload(item) for item in found]}

    return make_api_response(result, message, data, status_code, reflect)


def _search_result_payload(item: Any) -> dict[str, Any]:
    """Describe one search hit for the autocomplete list."""
    return {
        "modelId": item.model_id,
        "author": item.author,
        "downloads": item.downloads,
        "trendingScore": item.trending_score,
        "isLocal": item.is_local,
        "knownPublisher": item.is_known_publisher,
    }


@api.get("/models/local")
def api_models_local() -> dict[str, Any] | tuple[dict[str, Any], int]:
    """Cached models that can actually be fine-tuned."""
    (result, message, data, status_code, reflect) = (True, "", {}, 200, True)

    try:
        cached = [item for item in local_models() if item.is_causal_lm]
    except HubError as error:
        result = False
        message = str(error)
        status_code = HTTP_UNPROCESSABLE
    else:
        data = {
            "models": [
                {"modelId": item.model_id, "description": item.description, "sizeOnDisk": item.size_on_disk}
                for item in cached
            ]
        }

    return make_api_response(result, message, data, status_code, reflect)


@api.get("/models/merged")
def api_models_merged() -> dict[str, Any]:
    """Models that have been merged and are ready to quantize."""
    found = sorted(_model_id_from_artefact(path) for path in output_root().glob(f"**/{MERGED_SUBDIR}"))
    data = {"models": [{"modelId": name, "sizeOnDisk": _directory_size(merged_dir(name))} for name in found]}

    return make_api_response(data=data)


@api.get("/models/gguf")
def api_models_gguf() -> dict[str, Any]:
    """Every GGUF this application has produced, newest first."""
    found = sorted(output_root().glob(f"**/{GGUF_SUBDIR}/*.gguf"), key=lambda path: -path.stat().st_mtime)
    data = {"files": [_gguf_payload(path) for path in found]}

    return make_api_response(data=data)


def _gguf_payload(path: Path) -> dict[str, Any]:
    """Describe one produced GGUF file."""
    model_id = _model_id_from_artefact(path)
    return {
        "path": str(path),
        "name": path.name,
        "sizeOnDisk": path.stat().st_size,
        "modelId": model_id,
        "quantType": _quant_type_from_gguf(path, model_id),
    }


# --- settings --------------------------------------------------------------


@api.get("/settings")
def api_settings() -> dict[str, Any]:
    """The application settings, with the defaults they fall back to."""
    return make_api_response(data=_settings_payload())


@api.post("/settings")
@check_json_data
def api_settings_save(json_data: dict[str, Any]) -> dict[str, Any] | tuple[dict[str, Any], int]:
    """Store a new model directory; an empty one restores the default.

    Refused while a stage runs: it would write its first artefacts in one
    directory and its last in another.
    """
    (result, message, data, status_code, reflect) = (True, "", {}, 200, True)

    job = current_app.extensions["llmft_jobs"].current

    if job is not None and not job.finished:
        result = False
        message = f"A {job.kind} job is running. Change the model directory once it has finished."
        status_code = HTTP_CONFLICT
    else:
        try:
            set_output_root(str(json_data.get("outputDir", "")))
        except OutputDirError as error:
            result = False
            message = str(error)
            status_code = HTTP_UNPROCESSABLE
        else:
            data = _settings_payload()

    return make_api_response(result, message, data, status_code, reflect)


@api.post("/settings/server")
@check_json_data
def api_settings_server(json_data: dict[str, Any]) -> dict[str, Any] | tuple[dict[str, Any], int]:
    """Store the address the web server listens on; empty values restore the defaults.

    Not refused while a stage runs: the address only applies at the next start.
    """
    (result, message, data, status_code, reflect) = (True, "", {}, 200, True)

    try:
        set_server_address(str(json_data.get("host", "")), str(json_data.get("port", "")))
    except ServerAddressError as error:
        result = False
        message = str(error)
        status_code = HTTP_UNPROCESSABLE
    else:
        data = _settings_payload()

    return make_api_response(result, message, data, status_code, reflect)


@api.get("/folders")
def api_folders() -> dict[str, Any] | tuple[dict[str, Any], int]:
    """The subfolders of a directory, for the Setup page's directory browser.

    Without a path, the browser opens on the current model directory.
    """
    (result, message, data, status_code, reflect) = (True, "", {}, 200, True)

    try:
        listing = list_folder(request.args.get("path", "") or str(output_root()))
    except FolderError as error:
        result = False
        message = str(error)
        status_code = HTTP_UNPROCESSABLE
    else:
        data = _folders_payload(listing)

    return make_api_response(result, message, data, status_code, reflect)


def _folders_payload(listing: FolderListing) -> dict[str, Any]:
    """Describe one folder listing for the directory browser."""
    return {
        "path": str(listing.path),
        "parent": str(listing.parent) if listing.parent else None,
        "folders": [{"name": folder.name, "path": str(folder)} for folder in listing.folders],
        "roots": [str(root) for root in listing.roots],
    }


def _settings_payload() -> dict[str, Any]:
    """Describe the current settings for the Setup page."""
    current = output_root()
    address = server_address()
    return {
        "outputDir": str(current),
        "defaultOutputDir": str(DEFAULT_OUT_DIR),
        "isDefault": current == DEFAULT_OUT_DIR,
        "server": {
            "host": address.host,
            "port": address.port,
            "url": address.url,
            "defaultHost": DEFAULT_ADDRESS.host,
            "defaultPort": DEFAULT_ADDRESS.port,
            "isDefault": address.is_default,
            "isLoopback": address.is_loopback,
        },
    }


# --- dataset ---------------------------------------------------------------


@api.post("/dataset")
def api_dataset() -> dict[str, Any] | tuple[dict[str, Any], int]:
    """Validate an uploaded dataset, reporting every problem at once.

    Multipart, so no JSON body decorator: there is nothing to parse.
    """
    (result, message, data, status_code, reflect) = (True, "", {}, 200, True)

    uploaded = request.files.get("file")

    if uploaded is None or not uploaded.filename:
        result = False
        message = "No file was uploaded."
        status_code = HTTP_BAD_REQUEST
    else:
        target = ensure_dir(UPLOAD_DIR) / Path(uploaded.filename).name
        uploaded.save(target)
        if target.stat().st_size > MAX_DATASET_SIZE:
            target.unlink()
            result = False
            message = f"The dataset exceeds the {MAX_DATASET_SIZE // (1024 * 1024)} MB limit."
            status_code = HTTP_UNPROCESSABLE
        else:
            data = _dataset_payload(validate_dataset(target))

    return make_api_response(result, message, data, status_code, reflect)


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
def api_parameters() -> dict[str, Any]:
    """Every tunable, with the label and explanation the form needs.

    This is the same metadata the CLI editor and the manual are built from, so
    the three can never describe a parameter differently.
    """
    data = {
        "lora": [_spec_payload(spec) for spec in describe(LoraSettings)],
        "sft": [_spec_payload(spec) for spec in describe(SftSettings)],
    }

    return make_api_response(data=data)


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
def api_presets() -> dict[str, Any]:
    """The names of every stored preset."""
    return make_api_response(data={"presets": list(list_presets())})


@api.get("/presets/<name>")
def api_preset(name: str) -> dict[str, Any] | tuple[dict[str, Any], int]:
    """One stored preset's parameters."""
    (result, message, data, status_code, reflect) = (True, "", {}, 200, True)

    try:
        preset = load_preset(name)
    except ConfigError as error:
        result = False
        message = str(error)
        status_code = HTTP_NOT_FOUND
    else:
        data = {"name": preset.name, "lora": asdict(preset.lora), "sft": asdict(preset.sft)}

    return make_api_response(result, message, data, status_code, reflect)


@api.post("/presets")
@check_json_data
def api_preset_save(json_data: dict[str, Any]) -> dict[str, Any] | tuple[dict[str, Any], int]:
    """Store the submitted parameters under a name."""
    (result, message, data, status_code, reflect) = (True, "", {}, 200, True)

    try:
        lora, sft = _settings_from(json_data)
        target = save_preset(TrainingPreset(name=str(json_data.get("name", "")), lora=lora, sft=sft))
    except ConfigError as error:
        result = False
        message = str(error)
        status_code = HTTP_UNPROCESSABLE
    else:
        data = {"path": str(target)}

    return make_api_response(result, message, data, status_code, reflect)


# --- estimate --------------------------------------------------------------


@api.post("/estimate")
@check_json_data
def api_estimate(json_data: dict[str, Any]) -> dict[str, Any] | tuple[dict[str, Any], int]:
    """Predict peak memory for a run before anything is downloaded."""
    (result, message, data, status_code, reflect) = (True, "", {}, 200, True)

    model_id = str(json_data.get("modelId", ""))

    try:
        lora, sft = _settings_from(json_data)
        shape = ModelShape.from_config(model_id, fetch_config(model_id), fetch_parameter_count(model_id))
    except (ConfigError, HubError, ValueError) as error:
        result = False
        message = str(error)
        status_code = HTTP_UNPROCESSABLE
    else:
        data = _estimate_payload(shape, check_fit(shape, lora, sft, detect().gpu))

    return make_api_response(result, message, data, status_code, reflect)


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
@check_json_data
def api_jobs_train(json_data: dict[str, Any]) -> dict[str, Any] | tuple[dict[str, Any], int]:
    """Start a fine-tune, which merges automatically when it succeeds."""
    (result, message, data, status_code, reflect) = (True, "", {}, 200, True)

    dataset = Path(str(json_data.get("dataset", "")))

    if not dataset.is_file():
        result = False
        message = f"No dataset at {dataset}."
        status_code = HTTP_UNPROCESSABLE
    else:
        try:
            lora, sft = _settings_from(json_data)
        except ConfigError as error:
            result = False
            message = str(error)
            status_code = HTTP_UNPROCESSABLE
        else:
            model_id = str(json_data.get("modelId", ""))
            (result, message, data, status_code) = _start(
                "train", lambda sink: train_and_merge(model_id, dataset, lora, sft, sink=sink)
            )

    return make_api_response(result, message, data, status_code, reflect)


@api.post("/jobs/quantize")
@check_json_data
def api_jobs_quantize(json_data: dict[str, Any]) -> dict[str, Any] | tuple[dict[str, Any], int]:
    """Start a quantization of an already merged model."""
    model_id = str(json_data.get("modelId", ""))
    quant_type = str(json_data.get("quantType", QuantSettings().quant_type))

    (result, message, data, status_code) = _start("quantize", lambda sink: quantize(model_id, quant_type, sink=sink))

    return make_api_response(result, message, data, status_code, True)


@api.post("/jobs/deploy")
@check_json_data
def api_jobs_deploy(json_data: dict[str, Any]) -> dict[str, Any] | tuple[dict[str, Any], int]:
    """Start a deployment to a local model runner."""
    (result, message, data, status_code, reflect) = (True, "", {}, 200, True)

    gguf = Path(str(json_data.get("path", "")))

    if not gguf.is_file():
        result = False
        message = f"No GGUF file at {gguf}."
        status_code = HTTP_UNPROCESSABLE
    else:
        model_id = str(json_data.get("modelId", ""))
        name = str(json_data.get("name", "")) or _default_name(gguf, model_id, str(json_data.get("quantType", "")))
        work = (
            (lambda sink: deploy_to_ollama(gguf, name, sink=sink))
            if str(json_data.get("runner", "")) == "ollama"
            else (lambda sink: deploy_to_lm_studio(gguf, model_id or None, sink=sink))
        )
        (result, message, data, status_code) = _start("deploy", work)

    return make_api_response(result, message, data, status_code, reflect)


@api.get("/jobs/current")
def api_jobs_current() -> dict[str, Any]:
    """The running or most recently finished stage, or nothing at all."""
    job = current_app.extensions["llmft_jobs"].current

    return make_api_response(data={"job": _job_payload(job) if job else None})


@api.get("/jobs/stream")
def api_jobs_stream() -> Response | tuple[dict[str, Any], int]:
    """Stream a stage's progress as server-sent events.

    Not an envelope: this is a stream the browser reads with ``EventSource``,
    and each event carries its own JSON. The refusal path is an envelope.
    """
    job = current_app.extensions["llmft_jobs"].current
    result: Response | tuple[dict[str, Any], int]

    if job is None:
        result = make_api_response(False, "No job is running.", {}, HTTP_NOT_FOUND, True)
    else:
        result = Response(_events(job), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})

    return result


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


def _start(kind: str, work: Any) -> tuple[bool, str, dict[str, Any], int]:
    """Hand a stage to the runner, refusing if one is already going.

    Args:
        kind: The stage name, used in the job identifier.
        work: A callable taking a progress sink.

    Returns:
        Tuple of (result, message, data, status_code).
    """
    try:
        job = current_app.extensions["llmft_jobs"].start(kind, work)
    except JobBusy as error:
        outcome = (False, str(error), {}, HTTP_CONFLICT)
    else:
        outcome = (True, "", {"job": _job_payload(job)}, 200)

    return outcome


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
    """Recover the model identifier from an artefact path under the output root."""
    parts = path.relative_to(output_root()).parts
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
