"""LLM Finetune — a simple, fast and cheap QLoRA fine-tuning pipeline.

The package is split into a headless engine and two interchangeable front-ends:

* ``llmft.config``      typed parameter groups and named presets
* ``llmft.paths``       filesystem layout for generated artefacts
* ``llmft.environment`` capability detection (GPU, binaries, runtimes)
* ``llmft.bootstrap``   confirmation-gated installation of libraries and binaries
* ``llmft.pipeline``    train, merge, quantize and deploy stages
* ``llmft.cli``         interactive terminal front-end
* ``llmft.web``         Flask front-end

Engine modules never write to the terminal. Every stage reports through a
``ProgressSink`` so the CLI and the web layer can render identical progress
without the pipeline knowing which one is attached.

Nothing in this package imports :mod:`torch` at module scope: the bootstrap has
to be able to install it before the pipeline is first loaded.
"""

__version__ = "1.0.0"
