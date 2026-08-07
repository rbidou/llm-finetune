"""The Flask front-end.

The browser is a second renderer for the same engine the CLI drives. Nothing in
here trains, merges or quantizes anything: it starts a pipeline stage in a worker
thread and streams the progress events that stage was already emitting.

* ``jobs``   one stage at a time, with its events kept for the browser to read
* ``api``    JSON endpoints over the engine
* ``server`` the application factory
"""

from __future__ import annotations

from llmft.web.server import DEFAULT_HOST, DEFAULT_PORT, create_app, serve

__all__ = ["DEFAULT_HOST", "DEFAULT_PORT", "create_app", "serve"]
