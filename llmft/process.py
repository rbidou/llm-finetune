"""Running external commands and reporting what they print.

Shared by the bootstrap and the quantization stage, which both shell out to
tools that report progress as they work.

The one subtlety is line splitting. pip, the GGUF converter and llama-quantize
all draw progress bars that overwrite a single line with a carriage return
rather than emitting new lines, so splitting on ``\\n`` alone would yield one
enormous line at the end and no progress at all. Output is therefore split on
either character.

Standard library only: the bootstrap uses this before anything is installed.
"""

from __future__ import annotations

import re
import subprocess
from typing import Callable, Iterator

LINE_BREAK = re.compile(r"[\r\n]")


class ProcessError(RuntimeError):
    """Raised when an external command fails or cannot be started."""


def run_streaming(command: tuple[str, ...], on_line: Callable[[str], None], label: str) -> None:
    """Run ``command``, passing each line it writes to ``on_line``.

    stderr is merged into stdout because these tools log to both and the
    interleaving matters when something goes wrong. A non-zero exit raises
    :class:`ProcessError` quoting the last of the output, which is almost always
    where the explanation is.
    """
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as error:
        raise ProcessError(f"{label} could not be started: {error}") from error

    recent: list[str] = []
    with process:
        for line in _iter_lines(process.stdout):
            recent = [*recent[-4:], line]
            on_line(line)
    if process.returncode != 0:
        tail = "\n".join(recent)
        raise ProcessError(f"{label} failed with exit code {process.returncode}.\n{tail}")


def _iter_lines(stream) -> Iterator[str]:
    """Yield lines from a stream, treating a carriage return as a break.

    Reads one character at a time so a progress bar is reported as it is drawn
    rather than after the process exits. These tools emit only a few kilobytes
    of text, so the cost is irrelevant next to the work they are doing.
    """
    buffer = ""
    while character := stream.read(1):
        if LINE_BREAK.match(character):
            if buffer.strip():
                yield buffer.strip()
            buffer = ""
        else:
            buffer += character
    if buffer.strip():
        yield buffer.strip()
