"""The Flask application factory.

Threaded on purpose: a stage runs in a worker thread while the browser holds a
server-sent-events connection open to watch it, so a single-threaded server
would deadlock on the first fine-tune.
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, abort, render_template, send_file

from llmft import __version__
from llmft.web.api import api
from llmft.web.jobs import JobRunner

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000

MANUAL_PDF = Path(__file__).resolve().parent.parent.parent / "docs" / "llm-finetune-manual.pdf"
DOCUMENTATION_TEMPLATE = "documentation.html"


def create_app() -> Flask:
    """Build the application, with one job runner shared by every request."""
    app = Flask(__name__)
    # Jinja caches templates for the process lifetime unless told otherwise, so
    # without this an edited page keeps serving the old markup until a restart.
    app.jinja_env.auto_reload = True
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.extensions["llmft_jobs"] = JobRunner()
    app.register_blueprint(api)

    @app.get("/")
    def index() -> str:
        """The single page the whole interface lives on."""
        return render_template("index.html", version=__version__)

    @app.get("/documentation")
    def documentation() -> str:
        """The online subset of the manual, built by ``docs/build_docs.py``.

        A fragment rather than a page: the browser drops it into the
        documentation tab, so it shares the interface's own layout.
        """
        return render_template(DOCUMENTATION_TEMPLATE)

    @app.get("/manual.pdf")
    def download_manual():
        """The full manual, including the parts the online subset leaves out."""
        if not MANUAL_PDF.is_file():
            abort(404, "The manual has not been built. Run: python docs/build_docs.py")
        return send_file(MANUAL_PDF, as_attachment=True, download_name=MANUAL_PDF.name)

    return app


def serve(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> int:
    """Run the development server. Returns a process exit code.

    Bound to the loopback address by default: this exposes local model files and
    starts processes, and is not built to face a network.
    """
    create_app().run(host=host, port=port, threaded=True, debug=False)
    return 0
