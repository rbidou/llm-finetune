"""The interactive terminal front-end.

Split so the parts stay testable in isolation:

* ``ui``        terminal input and output, including progress rendering
* ``completer`` HuggingFace search as you type
* ``app``       the menus and the flow between them

Everything here is presentation. The engine decides nothing about how it is
displayed, and this package decides nothing about how a model is trained.
"""

from __future__ import annotations

from llmft.cli.app import run_cli

__all__ = ["run_cli"]
