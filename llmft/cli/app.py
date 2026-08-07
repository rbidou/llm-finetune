"""The interactive CLI: the steps of each flow, and the movement between them.

Each flow is an explicit sequence of steps rather than nested calls. That is
what makes ``back`` mean what it says: the runner simply decrements an index and
re-runs the previous step, with the answers gathered so far kept in a
:class:`FlowState`. Nested calls could only ever unwind the whole flow.

``home`` and ``exit`` are not handled here at all -- they raise past every step
to the main loop, which is the only place that decides what they mean.

Capabilities are detected before each action rather than once per session,
because the user may install Ollama or accept the llama.cpp download while the
application is open.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Sequence

from rich.table import Table
from rich.text import Text

from llmft.cli.completer import model_completer
from llmft.cli.ui import Choice, ExitRequested, GoBack, GoHome, Ui
from llmft.config import (
    ConfigError,
    LoraSettings,
    ParamKind,
    ParamSpec,
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
from llmft.environment import Capabilities, detect, features
from llmft.estimator import ModelShape, Verdict, check_fit, format_bytes
from llmft.hub import HubError, fetch_config, fetch_parameter_count, local_models
from llmft.paths import ADAPTER_SUBDIR, GGUF_SUBDIR, MERGED_SUBDIR, OUT_DIR, merged_dir, model_name
from llmft.pipeline import (
    DeploymentError,
    MergeError,
    QuantizationError,
    TrainingError,
    available_targets,
    deploy_to_lm_studio,
    deploy_to_ollama,
    ollama_name,
    quantize,
    suggested_name,
    train_and_merge,
)

HOME = "Home"
FINETUNE = "Fine-Tune"
QUANTIZE = "Quantize"
DEPLOY = "Deploy"
ENVIRONMENT = "Environment"

MAX_ISSUES_SHOWN = 12
CONTINUE_PROMPT = "Press Enter to continue"

STAGE_ERRORS = (TrainingError, MergeError, QuantizationError, DeploymentError)


@dataclass
class FlowState:
    """Answers gathered so far, kept across steps so ``back`` loses nothing."""

    local_only: bool = False
    model_id: str = ""
    dataset: Path | None = None
    lora: LoraSettings = field(default_factory=LoraSettings)
    sft: SftSettings = field(default_factory=SftSettings)
    quant_type: str = ""
    gguf: Path | None = None
    runner: str = ""


Step = Callable[[Ui, FlowState, Capabilities], None]


def run_cli() -> int:
    """Run the interactive session until the user leaves. Returns an exit code."""
    ui = Ui()
    while True:
        try:
            _one_action(ui)
        except ExitRequested:
            ui.goto("Finished")
            return 0
        except (GoBack, GoHome):
            # Both mean "show the main menu again": there is nowhere above it to
            # step back to, and home is defined as returning here.
            continue


def _one_action(ui: Ui) -> None:
    """Offer the main menu and run whatever the user picked."""
    capabilities = detect()
    ui.goto(HOME)
    choice = ui.choose(_main_menu(capabilities))
    if choice.key == "quit":
        raise ExitRequested
    try:
        _FLOWS[choice.key](ui, capabilities)
    except STAGE_ERRORS as error:
        _acknowledge(ui, str(error))
    except KeyboardInterrupt:
        _acknowledge(ui, "Interrupted.")


def _run_steps(ui: Ui, steps: Sequence[Step], state: FlowState, capabilities: Capabilities) -> None:
    """Walk a flow's steps, letting ``back`` return to the previous one.

    Stepping back from the first step leaves the flow. ``home`` and ``exit`` are
    not caught: they are meant to unwind further than this.
    """
    index = 0
    while index < len(steps):
        try:
            steps[index](ui, state, capabilities)
        except GoBack:
            index -= 1
            if index < 0:
                return
            continue
        index += 1


# --- fine-tune -------------------------------------------------------------


def run_finetune(ui: Ui, capabilities: Capabilities) -> None:
    """Choose a model and dataset, tune the parameters, then train and merge."""
    steps = (_step_model_source, _step_model, _step_dataset, _step_parameters, _step_estimate, _step_train)
    _run_steps(ui, steps, FlowState(), capabilities)


def _step_model_source(ui: Ui, state: FlowState, capabilities: Capabilities) -> None:
    """Local cache, or a search of HuggingFace."""
    ui.goto(FINETUNE, "Model Source")
    choice = ui.choose(
        [
            Choice("local", "Only models already downloaded", "no network needed"),
            Choice("hub", "Search HuggingFace", "type to see suggestions"),
        ]
    )
    state.local_only = choice.key == "local"


def _step_model(ui: Ui, state: FlowState, capabilities: Capabilities) -> None:
    """Pick the base model."""
    ui.goto(FINETUNE, "Model Source", "Model Selection")
    state.model_id = _select_local_model(ui) if state.local_only else _select_hub_model(ui)


def _select_local_model(ui: Ui) -> str:
    """List cached models that can actually be fine-tuned."""
    try:
        cached = [model for model in local_models() if model.is_causal_lm]
    except HubError as error:
        ui.error(str(error))
        _acknowledge(ui, "")
        raise GoBack from error
    if not cached:
        ui.warn("No fine-tunable models are cached. Go back and search HuggingFace instead.")
        _acknowledge(ui, "")
        raise GoBack
    return ui.choose([Choice(item.model_id, item.model_id, item.description) for item in cached]).key


def _select_hub_model(ui: Ui) -> str:
    """Prompt for a model identifier with live suggestions."""
    if ui.supports_completion:
        ui.hint("Suggestions appear after two characters. Only original models are listed.")
    else:
        ui.hint("This terminal does not support suggestions. Type the full identifier, e.g. Qwen/Qwen3-0.6B.")
    model_id = ui.ask("Model", completer=model_completer() if ui.supports_completion else None)
    if not model_id:
        raise GoBack
    try:
        fetch_config(model_id)
    except HubError as error:
        ui.error(str(error))
        _acknowledge(ui, "")
        raise GoBack from error
    return model_id


def _step_dataset(ui: Ui, state: FlowState, capabilities: Capabilities) -> None:
    """Read a dataset path and validate it, reporting every problem found."""
    ui.goto(FINETUNE, "Model Selection", "Dataset")
    while True:
        answer = ui.ask("Training dataset (JSON file)")
        if not answer:
            raise GoBack
        report = validate_dataset(Path(answer.strip('"')))
        if report.is_valid:
            ui.success(f"Dataset accepted: {report.summary()}")
            state.dataset = report.path
            return
        _show_issues(ui, report)


def _show_issues(ui: Ui, report: DatasetReport) -> None:
    """Replace the panel with what is wrong with this file."""
    ui.show(Text.from_markup(f"[red]{report.path} cannot be used:[/]"))
    for issue in report.issues[:MAX_ISSUES_SHOWN]:
        ui.note(f"  {issue}")
    hidden = len(report.issues) - MAX_ISSUES_SHOWN
    if hidden > 0:
        ui.hint(f"  ... and {hidden} more.")
    if report.truncated:
        ui.hint("  More problems were found than are listed; fix these first.")


def _step_parameters(ui: Ui, state: FlowState, capabilities: Capabilities) -> None:
    """Show the parameters and let the user change, save or load them."""
    ui.goto(FINETUNE, "Dataset", "Parameters")
    while True:
        action = ui.choose(
            [
                Choice("continue", "Continue with these settings"),
                Choice("edit", "Change a value"),
                Choice("load", "Load a saved preset", enabled=bool(list_presets()), reason="No presets saved yet."),
                Choice("save", "Save these as a preset"),
            ],
            _parameter_table(ui, state.lora, state.sft),
            Text.from_markup("[dim]Choose 'Change a value' to see what a parameter does before editing it.[/]"),
        )
        if action.key == "continue":
            return
        if action.key == "edit":
            state.lora, state.sft = _edit_one(ui, state.lora, state.sft)
        elif action.key == "load":
            state.lora, state.sft = _load_preset(ui, state.lora, state.sft)
        else:
            _save_preset(ui, state.lora, state.sft)


def _parameter_table(ui: Ui, lora: LoraSettings, sft: SftSettings) -> Table:
    """Build the table of every tunable with its value and permitted range.

    Explanations are deliberately not in this table. Thirteen paragraphs wrapped
    into a terminal column is unreadable, and the specification asks for the
    explanatory text in the GUI. Here the full explanation is shown for the one
    parameter being changed, at the moment it is changed.
    """
    rows = [
        (str(number), spec.label, _format_value(getattr(group, spec.name)), _describe_range(spec))
        for number, (spec, group) in enumerate(_all_specs(lora, sft), start=1)
    ]
    return ui.table(("#", "Parameter", "Value", "Allowed"), rows, title="LoRA and training parameters")


def _edit_one(ui: Ui, lora: LoraSettings, sft: SftSettings) -> tuple[LoraSettings, SftSettings]:
    """Change a single parameter, keeping the previous value if it is rejected."""
    specs = _all_specs(lora, sft)
    answer = ui.ask(f"Which parameter (1-{len(specs)})")
    if not answer.isdigit() or not 1 <= int(answer) <= len(specs):
        return lora, sft

    spec, group = specs[int(answer) - 1]
    ui.hint(f"{spec.label}: {spec.explanation}")
    if spec.choices:
        ui.hint(f"Options: {', '.join(spec.choices)}")
    raw = ui.ask(spec.label, default=_format_value(getattr(group, spec.name)))
    try:
        updated = replace(group, **{spec.name: _parse_value(spec, raw)})
    except (ConfigError, ValueError) as error:
        ui.error(str(error))
        _acknowledge(ui, "")
        return lora, sft
    return (updated, sft) if isinstance(updated, LoraSettings) else (lora, updated)


def _load_preset(ui: Ui, lora: LoraSettings, sft: SftSettings) -> tuple[LoraSettings, SftSettings]:
    """Replace the current settings with a stored preset."""
    picked = ui.choose([Choice(name, name) for name in list_presets()])
    try:
        preset = load_preset(picked.key)
    except ConfigError as error:
        ui.error(str(error))
        _acknowledge(ui, "")
        return lora, sft
    return preset.lora, preset.sft


def _save_preset(ui: Ui, lora: LoraSettings, sft: SftSettings) -> None:
    """Store the current settings under a name."""
    name = ui.ask("Preset name")
    if not name:
        return
    try:
        target = save_preset(TrainingPreset(name=name, lora=lora, sft=sft))
    except ConfigError as error:
        ui.error(str(error))
        _acknowledge(ui, "")
        return
    ui.success(f"Saved to {target}")
    _acknowledge(ui, "")


def _step_estimate(ui: Ui, state: FlowState, capabilities: Capabilities) -> None:
    """Run the pre-flight estimate and let the user act on it.

    A run predicted not to fit is refused by default. Overriding is possible but
    has to be chosen: the estimate is conservative, not infallible.
    """
    ui.goto(FINETUNE, "Parameters", "Pre-flight Estimate")
    try:
        shape = ModelShape.from_config(
            state.model_id, fetch_config(state.model_id), fetch_parameter_count(state.model_id)
        )
    except (HubError, ValueError) as error:
        ui.error(f"Could not size {state.model_id}: {error}")
        _acknowledge(ui, "")
        raise GoBack from error

    verdict = check_fit(shape, state.lora, state.sft, capabilities.gpu)
    _show_verdict(ui, shape, verdict)
    if verdict.fits:
        if not ui.confirm("Start training?", default=True):
            raise GoBack
    elif not ui.confirm("The estimate says this will not fit. Try anyway?", default=False):
        raise GoBack


def _show_verdict(ui: Ui, shape: ModelShape, verdict: Verdict) -> None:
    """Report the memory estimate and, when blocked, what would help."""
    ui.note(f"{shape.model_id}: {shape.parameter_count / 1e9:.2f}B parameters, vocabulary {shape.vocab_size:,}")
    (ui.success if verdict.fits else ui.error)(verdict.headline)
    if not verdict.fits:
        ui.note(f"Short by {format_bytes(verdict.shortfall_bytes)}.")
        for remedy in verdict.remedies:
            ui.note(f"  - {remedy}")


def _step_train(ui: Ui, state: FlowState, capabilities: Capabilities) -> None:
    """Train the adapter and merge it back into the base weights."""
    with ui.progress(FINETUNE, "Pre-flight Estimate", "Training") as sink:
        merged = train_and_merge(state.model_id, state.dataset, state.lora, state.sft, sink=sink)
    ui.success(f"Merged model written to {merged}")
    if ui.confirm("Quantize it now?", default=True):
        _run_steps(ui, (_step_quant_type, _step_quantize), replace(state, quant_type=""), capabilities)


# --- quantize --------------------------------------------------------------


def run_quantize(ui: Ui, capabilities: Capabilities) -> None:
    """Quantize any merged model, including one from an earlier session."""
    _run_steps(ui, (_step_pick_merged, _step_quant_type, _step_quantize), FlowState(), capabilities)


def _step_pick_merged(ui: Ui, state: FlowState, capabilities: Capabilities) -> None:
    """List models that have been merged and are ready to quantize."""
    ui.goto(QUANTIZE, "Model Selection")
    found = sorted(_model_id_from_artefact(path) for path in OUT_DIR.glob(f"**/{MERGED_SUBDIR}"))
    if not found:
        ui.warn(f"No merged models under {OUT_DIR}. Fine-tune one first.")
        _acknowledge(ui, "")
        raise GoBack
    state.model_id = ui.choose([Choice(name, name, _size_of(merged_dir(name))) for name in found]).key


def _step_quant_type(ui: Ui, state: FlowState, capabilities: Capabilities) -> None:
    """Choose the GGUF output type, offering only what is installed."""
    ui.goto(QUANTIZE, "Model Selection", "Quantization Type")
    offered = available_quant_types(capabilities.llama_quantize is not None)
    ui.hint(f"Default is {QuantSettings().quant_type}.")
    state.quant_type = ui.choose([Choice(item.name, item.label, item.explanation) for item in offered]).key


def _step_quantize(ui: Ui, state: FlowState, capabilities: Capabilities) -> None:
    """Produce the GGUF, then offer to deploy it."""
    with ui.progress(QUANTIZE, "Quantization Type", state.quant_type) as sink:
        state.gguf = quantize(state.model_id, state.quant_type, sink=sink)
    ui.success(f"Wrote {state.gguf}")
    if ui.confirm("Deploy it now?", default=False):
        _run_steps(ui, (_step_pick_runner, _step_deploy), state, capabilities)


# --- deploy ----------------------------------------------------------------


def run_deploy(ui: Ui, capabilities: Capabilities) -> None:
    """Hand an existing GGUF to a local model runner."""
    _run_steps(ui, (_step_pick_gguf, _step_pick_runner, _step_deploy), FlowState(), capabilities)


def _step_pick_gguf(ui: Ui, state: FlowState, capabilities: Capabilities) -> None:
    """List every GGUF this application has produced."""
    ui.goto(DEPLOY, "Model Selection")
    found = _find_gguf_files()
    if not found:
        ui.warn(f"No GGUF files under {OUT_DIR}. Quantize a model first.")
        _acknowledge(ui, "")
        raise GoBack
    picked = ui.choose([Choice(str(path), path.name, _size_of(path)) for path in found])
    state.gguf = Path(picked.key)
    state.model_id = _model_id_from_artefact(state.gguf)
    state.quant_type = _quant_type_from_gguf(state.gguf, state.model_id)


def _step_pick_runner(ui: Ui, state: FlowState, capabilities: Capabilities) -> None:
    """Choose between the runners that are actually installed."""
    ui.goto(DEPLOY, "Model Selection", "Target")
    targets = available_targets(capabilities)
    state.runner = ui.choose(
        [Choice(item.key, item.label, enabled=item.available, reason=item.reason) for item in targets]
    ).key


def _step_deploy(ui: Ui, state: FlowState, capabilities: Capabilities) -> None:
    """Register the model with the chosen runner.

    Everything that needs an answer is asked before the progress bar starts. A
    prompt issued underneath a live display is painted over: the caller waits on
    input for a question nobody can see, which reads exactly like a hang.
    """
    gguf = state.gguf
    name = ""
    if state.runner == "ollama":
        ui.goto(DEPLOY, "Target", "Name")
        default = (
            suggested_name(state.model_id, state.quant_type)
            if state.model_id and state.quant_type
            else ollama_name(gguf.stem)
        )
        name = ui.ask("Name in Ollama", default=default) or default

    with ui.progress(DEPLOY, "Target", gguf.name) as sink:
        if state.runner == "ollama":
            tag = deploy_to_ollama(gguf, name, sink=sink)
        else:
            deploy_to_lm_studio(gguf, state.model_id or None, sink=sink)
            tag = ""

    # Printed after the bar closes: a progress description is truncated to the
    # terminal width, and this is the one line the user needs to act on.
    ui.success(f"Deployed as {tag}. Run it with: ollama run {tag}" if tag else "Imported into LM Studio.")
    _acknowledge(ui, "")


# --- environment -----------------------------------------------------------


def show_environment(ui: Ui, capabilities: Capabilities) -> None:
    """Show what this machine can currently do."""
    ui.goto(ENVIRONMENT)
    gpu = capabilities.gpu
    if gpu is None:
        ui.warn("No CUDA GPU detected. Training is unavailable; merging and quantizing still work.")
    else:
        ui.note(f"{gpu.name}: {gpu.free_memory_mb} MiB free of {gpu.total_memory_mb} MiB")
        if gpu.is_heavily_occupied:
            ui.warn(f"{gpu.occupied_memory_mb} MiB is held by another application.")
    rows = [(item.label, "yes" if item.available else "no", item.reason) for item in features(capabilities)]
    ui.append(ui.table(("Operation", "Available", "Reason"), rows))
    _acknowledge(ui, "")


_FLOWS: dict[str, Callable[[Ui, Capabilities], None]] = {
    "finetune": run_finetune,
    "quantize": run_quantize,
    "deploy": run_deploy,
    "check": show_environment,
}


def _main_menu(capabilities: Capabilities) -> list[Choice]:
    """Build the top-level menu, disabling what this machine cannot do."""
    gates = {item.key: item for item in features(capabilities)}
    train, quant = gates["train"], gates["quantize"]
    can_deploy = any(item.available for item in available_targets(capabilities))
    return [
        Choice("finetune", "Fine-tune a model", "train, then merge", train.available, train.reason),
        Choice("quantize", "Quantize a merged model", "to GGUF", quant.available, quant.reason),
        Choice("deploy", "Deploy a quantized model", "", can_deploy, "No local model runner is installed."),
        Choice("check", "Show environment details"),
        Choice("quit", "Quit"),
    ]


def _acknowledge(ui: Ui, message: str) -> None:
    """Show a message, if any, and wait for the user to take it in."""
    if message:
        ui.error(message)
    ui.read(f"{CONTINUE_PROMPT}: ")


def _find_gguf_files() -> list[Path]:
    """Every GGUF this application has produced, newest first."""
    return sorted(OUT_DIR.glob(f"**/{GGUF_SUBDIR}/*.gguf"), key=lambda path: -path.stat().st_mtime)


def _model_id_from_artefact(path: Path) -> str:
    """Recover the model identifier from an artefact path under ``out/``.

    Artefacts live at ``out/<model_id>/<kind>/...`` and a model identifier may
    or may not carry an organisation, so the identifier is whatever sits between
    the output root and the artefact directory rather than a fixed depth.
    """
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


def _all_specs(lora: LoraSettings, sft: SftSettings) -> list[tuple[ParamSpec, object]]:
    """Pair every tunable with the settings group that holds it."""
    return [(spec, lora) for spec in describe(lora)] + [(spec, sft) for spec in describe(sft)]


def _describe_range(spec: ParamSpec) -> str:
    """Summarise what a parameter will accept, compactly."""
    if spec.choices:
        return ", ".join(spec.choices) if spec.kind is ParamKind.CHOICE else f"any of {len(spec.choices)}"
    if spec.kind is ParamKind.BOOL:
        return "true / false"
    if spec.minimum is not None and spec.maximum is not None:
        return f"{_format_number(spec.minimum)} to {_format_number(spec.maximum)}"
    return ""


def _format_number(value: float) -> str:
    """Render a bound without scientific-notation noise for whole numbers."""
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _parse_value(spec: ParamSpec, raw: str):
    """Turn typed text into the type a parameter expects."""
    text = raw.strip()
    if spec.kind is ParamKind.INT:
        return int(text)
    if spec.kind is ParamKind.FLOAT:
        return float(text)
    if spec.kind is ParamKind.BOOL:
        return text.lower() in {"true", "yes", "y", "1", "on"}
    if spec.kind is ParamKind.MULTI_CHOICE:
        return tuple(part.strip() for part in text.split(",") if part.strip())
    return text


def _format_value(value: object) -> str:
    """Render a parameter value the same way it is typed back in."""
    return ", ".join(value) if isinstance(value, tuple) else str(value)


def _size_of(path: Path) -> str:
    """Human-readable size of a file or directory."""
    if path.is_file():
        total = path.stat().st_size
    else:
        total = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return f"{total / 1e9:.2f} GB" if total >= 1e9 else f"{total / 1e6:.0f} MB"
