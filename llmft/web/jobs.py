"""Running one pipeline stage at a time on behalf of a browser.

A stage is a long, blocking call that reports through a
:class:`~llmft.progress.ProgressSink`. A web request cannot hold it open, so the
stage runs in a worker thread and its events accumulate here for the browser to
read at its own pace.

Only one stage runs at a time, deliberately. Two concurrent fine-tunes would
compete for the same GPU and both fail, and the second would be far harder to
explain than a refusal.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock, Thread
from typing import Any, Callable

from llmft.progress import ProgressEvent, ProgressSink

Work = Callable[[ProgressSink], Any]


class JobState(str, Enum):
    """Where a stage has got to."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobBusy(RuntimeError):
    """Raised when a stage is asked for while another is still running."""


@dataclass
class Job:
    """One run of one pipeline stage, and everything it has reported."""

    identifier: str
    kind: str
    state: JobState = JobState.RUNNING
    result: str = ""
    error: str = ""
    events: list[ProgressEvent] = field(default_factory=list)
    lock: Lock = field(default_factory=Lock, repr=False)

    @property
    def finished(self) -> bool:
        """Whether the stage has stopped, successfully or not."""
        return self.state is not JobState.RUNNING

    def record(self, event: ProgressEvent) -> None:
        """Store one progress event."""
        with self.lock:
            self.events.append(event)

    def since(self, cursor: int) -> tuple[list[ProgressEvent], int]:
        """Return events the browser has not seen, and the new cursor."""
        with self.lock:
            pending = self.events[cursor:]
            return pending, cursor + len(pending)

    def latest(self) -> ProgressEvent | None:
        """The most recent event, for a client that just connected."""
        with self.lock:
            return self.events[-1] if self.events else None


class JobRunner:
    """Owns the single stage that may be running at any moment."""

    def __init__(self) -> None:
        self._current: Job | None = None
        self._counter = itertools.count(1)
        self._lock = Lock()

    @property
    def current(self) -> Job | None:
        """The running or most recently finished stage."""
        return self._current

    def start(self, kind: str, work: Work) -> Job:
        """Begin a stage in the background and return the job tracking it."""
        with self._lock:
            if self._current is not None and not self._current.finished:
                raise JobBusy(f"A {self._current.kind} job is already running. Wait for it to finish.")
            job = Job(identifier=f"{kind}-{next(self._counter)}", kind=kind)
            self._current = job
        Thread(target=self._run, args=(job, work), daemon=True, name=f"llmft-{job.identifier}").start()
        return job

    def _run(self, job: Job, work: Work) -> None:
        """Perform the work, recording how it ended.

        Every exception is captured onto the job rather than raised: this runs
        on a thread nobody is waiting on, so an escaping error would vanish and
        leave the browser watching a stage that never finishes.
        """
        try:
            outcome = work(job.record)
        except Exception as error:  # noqa: BLE001 - reported to the browser instead
            job.error = str(error) or error.__class__.__name__
            job.state = JobState.FAILED
        else:
            job.result = str(outcome) if outcome is not None else ""
            job.state = JobState.SUCCEEDED
