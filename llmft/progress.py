"""Progress reporting shared by every stage of the pipeline.

Engine code never writes to the terminal. Each stage emits :class:`ProgressEvent`
values to a sink supplied by whichever front-end is attached, so the CLI can draw
a progress bar and the web layer can stream the same events to a browser without
the stage knowing the difference.

A sink is just a callable, which keeps stages trivial to test: pass a
:class:`CollectingSink` and assert on what was reported.
"""

from __future__ import annotations

import contextlib
import io
import os
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator

# Set before huggingface_hub, transformers, datasets or bitsandbytes are
# imported, because each reads its verbosity from the environment at import
# time. Setting them afterwards silences some messages but not all.
QUIET_ENVIRONMENT = {
    # Downloads report through this application's own progress bar instead.
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    # "your machine does not support symlinks" on Windows without Developer Mode.
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    # "You are sending unauthenticated requests to the HF Hub".
    "HF_HUB_VERBOSITY": "error",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "DATASETS_VERBOSITY": "error",
    "TOKENIZERS_PARALLELISM": "false",
    "BITSANDBYTES_NOWELCOME": "1",
}


class Stage(str, Enum):
    """The long-running operations a front-end may need to render."""

    BOOTSTRAP = "bootstrap"
    TRAIN = "train"
    MERGE = "merge"
    QUANTIZE = "quantize"
    DEPLOY = "deploy"


@dataclass(frozen=True)
class ProgressEvent:
    """One observation about a running stage.

    ``current`` and ``total`` are optional: some steps can report a precise
    fraction, others only know that they are still working.
    """

    stage: Stage
    message: str
    current: int | None = None
    total: int | None = None

    @property
    def fraction(self) -> float | None:
        """Completion in the range 0..1, or ``None`` when the total is unknown."""
        measurable = self.current is not None and self.total is not None and self.total > 0
        return min(1.0, max(0.0, self.current / self.total)) if measurable else None


ProgressSink = Callable[[ProgressEvent], None]


def report(
    sink: ProgressSink,
    stage: Stage,
    message: str,
    current: int | None = None,
    total: int | None = None,
) -> None:
    """Emit one event, so call sites stay readable at the point of work."""
    sink(ProgressEvent(stage=stage, message=message, current=current, total=total))


def null_sink(event: ProgressEvent) -> None:
    """Discard progress. The default for callers that render nothing."""


@dataclass
class CapturedOutput:
    """Whatever the silenced libraries tried to print."""

    buffer: io.StringIO

    @property
    def text(self) -> str:
        """Everything captured so far."""
        return self.buffer.getvalue()

    def tail(self, lines: int = 20) -> str:
        """The last few lines, for attaching to an error message."""
        return "\n".join(self.text.splitlines()[-lines:])


def silence_libraries() -> None:
    """Stop third-party libraries writing to the terminal at all.

    Call this before anything else, from the entry point: these settings are
    read when the libraries are first imported, so applying them later silences
    some output but not all of it.

    This covers the messages that appear *outside* a pipeline stage -- browsing
    models and sizing a run both reach the network, and neither is wrapped in
    :func:`quiet`. Downloads still show progress, but through this application's
    own bar rather than a library's.
    """
    os.environ.update(QUIET_ENVIRONMENT)
    warnings.filterwarnings("ignore", category=UserWarning, module=r"huggingface_hub.*")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    _silence_loaded_libraries()


@contextlib.contextmanager
def quiet() -> Iterator[CapturedOutput]:
    """Silence the machine-learning stack for the duration of a stage.

    The specification asks for a progress bar and no other terminal output, but
    transformers, datasets, trl and bitsandbytes are all talkative. This turns
    their logging down, disables their progress bars, and redirects whatever
    still escapes into a buffer.

    Output is *captured*, not discarded. Swallowing a stack trace to keep a
    progress bar tidy would trade a small annoyance for a much worse one, so a
    failing stage can quote :meth:`CapturedOutput.tail` back to the user.

    Note for front-ends: this replaces ``sys.stdout`` while active, so a renderer
    that draws a progress bar must hold the real stream (``sys.__stdout__``),
    captured before the stage begins. Redirection is Python-level only; output
    written by native extensions is not intercepted.
    """
    captured = CapturedOutput(io.StringIO())
    previous = {key: os.environ.get(key) for key in QUIET_ENVIRONMENT}
    os.environ.update(QUIET_ENVIRONMENT)
    try:
        with contextlib.redirect_stdout(captured.buffer), contextlib.redirect_stderr(captured.buffer):
            _silence_loaded_libraries()
            yield captured
    finally:
        _restore_environment(previous)


def _silence_loaded_libraries() -> None:
    """Turn down every logger that writes during a run, if it is loaded."""
    with contextlib.suppress(ImportError):
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
    with contextlib.suppress(ImportError):
        import datasets

        datasets.disable_progress_bars()
        datasets.logging.set_verbosity_error()
    with contextlib.suppress(ImportError):
        from huggingface_hub.utils import logging as hub_logging

        hub_logging.set_verbosity_error()


def _restore_environment(previous: dict[str, str | None]) -> None:
    """Put the environment back exactly as it was."""
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


class CollectingSink:
    """Keeps every event it receives.

    Used by the tests, and by the web job runner as the backing store a browser
    polls or streams from.
    """

    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def __call__(self, event: ProgressEvent) -> None:
        self.events.append(event)

    @property
    def messages(self) -> tuple[str, ...]:
        """Every message reported so far, in order."""
        return tuple(event.message for event in self.events)

    def last(self) -> ProgressEvent | None:
        """The most recent event, or ``None`` if nothing has been reported."""
        return self.events[-1] if self.events else None
