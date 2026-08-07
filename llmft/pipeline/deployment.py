"""Handing a finished GGUF to a local model runner.

Ollama and LM Studio are the user's own installations. This application never
installs either of them; when one is absent its option is simply unavailable.

Deployment is independent of everything before it: any GGUF file can be
deployed, whether it was produced minutes ago or by an earlier session.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from llmft.environment import Capabilities, find_lms, find_ollama
from llmft.process import ProcessError, run_streaming
from llmft.progress import ProgressSink, Stage, null_sink, report

OLLAMA = "ollama"
LM_STUDIO = "lms"

# One lookup per runner, shared with capability detection so the two agree.
RUNNERS = {OLLAMA: (find_ollama, "Ollama"), LM_STUDIO: (find_lms, "LM Studio")}

# Ollama accepts lowercase names of these characters, optionally with a :tag.
OLLAMA_NAME_ALLOWED = re.compile(r"[^a-z0-9._-]+")


class DeploymentError(RuntimeError):
    """Raised when a model cannot be handed to a local runner."""


@dataclass(frozen=True)
class DeployTarget:
    """One runner the user could deploy to, and whether they can right now."""

    key: str
    label: str
    available: bool
    reason: str = ""


def available_targets(capabilities: Capabilities) -> tuple[DeployTarget, ...]:
    """Describe every deployment option, enabled or not."""
    return (
        DeployTarget(OLLAMA, "Ollama", capabilities.ollama is not None, "Ollama is not installed."),
        DeployTarget(LM_STUDIO, "LM Studio", capabilities.lms is not None, "LM Studio is not installed."),
    )


def deploy_to_ollama(gguf: Path, name: str, sink: ProgressSink = null_sink) -> str:
    """Register a GGUF with Ollama under ``name`` and return the name used.

    The Modelfile carries only a ``FROM`` line. llama.cpp writes the chat
    template into the GGUF during conversion and Ollama reads it from there, so
    restating a template here could only introduce a disagreement between the
    two.
    """
    binary = _require(OLLAMA)
    _require_gguf(gguf)
    tag = ollama_name(name)

    report(sink, Stage.DEPLOY, f"Registering {tag} with Ollama", 0, 1)
    with tempfile.TemporaryDirectory() as workspace:
        modelfile = Path(workspace) / "Modelfile"
        modelfile.write_text(f"FROM {gguf.resolve()}\n", encoding="utf-8")
        _run((str(binary), "create", tag, "-f", str(modelfile)), f"ollama create {tag}", sink)

    # A fact, not an instruction: the front-end composes "how to run it" from
    # the returned tag, where it can be shown in full rather than truncated to
    # the width of a progress bar.
    report(sink, Stage.DEPLOY, f"Registered {tag} with Ollama", 1, 1)
    return tag


def deploy_to_lm_studio(
    gguf: Path,
    repository: str | None = None,
    sink: ProgressSink = null_sink,
    dry_run: bool = False,
) -> Path:
    """Import a GGUF into LM Studio's model library and return the original path.

    Three flags are not optional here.

    ``--copy`` matters most: ``lms import`` **moves** the file by default, which
    would take the GGUF out of this application's output directory and break
    every later use of it, including a second deployment to Ollama. The original
    stays put and LM Studio gets its own copy.

    ``--yes`` is needed because the command is otherwise interactive, and a
    pipeline stage has no console to answer with.

    ``--user-repo`` gives LM Studio the publisher and model name explicitly.
    Left to itself it guesses them from the filename, which for a name like
    ``SmolLM2-135M-Instruct-q4_K_M.gguf`` files the model under something
    arbitrary.
    """
    binary = _require(LM_STUDIO)
    _require_gguf(gguf)

    report(sink, Stage.DEPLOY, f"Importing {gguf.name} into LM Studio", 0, 1)
    command = [str(binary), "import", str(gguf.resolve()), "--yes", "--copy"]
    if repository:
        command += ["--user-repo", lm_studio_repository(repository)]
    if dry_run:
        command.append("--dry-run")
    _run(tuple(command), f"lms import {gguf.name}", sink)

    report(sink, Stage.DEPLOY, f"Imported {gguf.name} into LM Studio", 1, 1)
    return gguf


def lm_studio_repository(model_id: str) -> str:
    """Return a ``user/repo`` pair for LM Studio's library layout.

    A model identifier already has that shape. One without an organisation is
    filed under the application's own name rather than left to a guess.
    """
    organisation, separator, name = model_id.partition("/")
    return f"{organisation}/{name}" if separator and name else f"llm-finetune/{organisation}"


def ollama_name(name: str) -> str:
    """Convert a model name into something Ollama will accept.

    Ollama wants lowercase and a restricted character set. A ``:tag`` suffix is
    preserved, so ``SmolLM2-135M:q4_K_M`` becomes ``smollm2-135m:q4_k_m``.
    """
    head, separator, tag = name.lower().partition(":")
    cleaned = OLLAMA_NAME_ALLOWED.sub("-", head).strip("-")
    suffix = OLLAMA_NAME_ALLOWED.sub("-", tag).strip("-") if separator else ""
    if not cleaned:
        raise DeploymentError(f"{name!r} does not contain any character Ollama accepts in a model name.")
    return f"{cleaned}:{suffix}" if suffix else cleaned


def suggested_name(model_id: str, quant_type: str) -> str:
    """Propose a runner name from the model and its quantization."""
    return ollama_name(f"{model_id.rsplit('/', maxsplit=1)[-1]}:{quant_type}")


def _require(key: str) -> Path:
    """Locate a runner's executable, or explain that it is absent.

    Uses the same lookup as :func:`llmft.environment.detect`, so a runner can
    never be offered by the interface and then turn out to be missing here.
    """
    finder, label = RUNNERS[key]
    binary = finder()
    if binary is None:
        raise DeploymentError(
            f"{label} is not installed, so this model cannot be deployed to it. "
            f"Install {label} yourself and run 'python run.py check' again."
        )
    return binary


def _require_gguf(gguf: Path) -> None:
    """Refuse to deploy something that is not there."""
    if not gguf.is_file():
        raise DeploymentError(f"No GGUF file at {gguf}. Quantize a model first.")


def _run(command: tuple[str, ...], label: str, sink: ProgressSink) -> None:
    """Run a runner command, converting failures into a stage error."""
    try:
        run_streaming(command, lambda line: report(sink, Stage.DEPLOY, line), label)
    except ProcessError as error:
        raise DeploymentError(str(error)) from error
