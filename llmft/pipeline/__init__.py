"""The pipeline stages: train, merge, quantize and deploy.

Every stage is headless. Stages take their inputs as arguments, report through a
:class:`~llmft.progress.ProgressSink`, and return the path they produced. None of
them prompt, print, or decide policy -- a front-end does that.

Modules here import torch and transformers *inside* their functions rather than
at module scope, so importing :mod:`llmft.pipeline` stays cheap and the bootstrap
can still install those libraries on a machine that does not yet have them.

Module names are gerunds (``training``, ``merging``) while the functions they
export are verbs (``train``, ``merge``). Naming both the same would leave the
re-exported function shadowing its own submodule on the package, which is a
quietly confusing thing to debug.
"""

from __future__ import annotations

from pathlib import Path

from llmft.config import LoraSettings, SftSettings
from llmft.pipeline.deployment import (
    DeploymentError,
    DeployTarget,
    available_targets,
    deploy_to_lm_studio,
    deploy_to_ollama,
    name_for_gguf,
    ollama_name,
    suggested_name,
)
from llmft.pipeline.merging import MergeError, merge
from llmft.pipeline.quantization import QuantizationError, quantize
from llmft.pipeline.training import TrainingError, train
from llmft.progress import ProgressSink, null_sink

__all__ = [
    "DeployTarget",
    "DeploymentError",
    "MergeError",
    "QuantizationError",
    "TrainingError",
    "available_targets",
    "deploy_to_lm_studio",
    "deploy_to_ollama",
    "merge",
    "name_for_gguf",
    "ollama_name",
    "quantize",
    "suggested_name",
    "train",
    "train_and_merge",
]


def train_and_merge(
    model_id: str,
    dataset_path: Path,
    lora: LoraSettings,
    sft: SftSettings,
    sink: ProgressSink = null_sink,
) -> Path:
    """Train an adapter and merge it back into the base weights.

    The merge follows a successful training run automatically, as specified, so
    the usual path leaves the user with a standalone model rather than an
    adapter they then have to combine by hand.

    Returns the merged model directory.
    """
    adapter = train(model_id, dataset_path, lora, sft, sink=sink)
    return merge(model_id, adapter=adapter, sink=sink)
