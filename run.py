#!/usr/bin/env python
"""Entry point for LLM Finetune.

    python run.py           start the interactive fine-tuning session
    python run.py web       serve the browser interface
    python run.py check     report what this machine can currently do
    python run.py setup     install the missing libraries and binaries

Two rules shape this file. Engine modules are imported inside the command
functions, never at module scope, because ``setup`` has to be able to install
torch before anything imports it. And nothing here uses ``rich``: on a bare
virtual environment it is not installed yet, so the bootstrap path can only rely
on the standard library.
"""

from __future__ import annotations

import argparse
import sys

LABEL_WIDTH = 22


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to a command. Returns a process exit code."""
    from llmft.progress import silence_libraries

    # Before any library is imported: each reads its verbosity at import time,
    # and this application draws its own progress bars.
    silence_libraries()

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    commands = {"run": handle_run, "web": handle_web, "check": handle_check, "setup": handle_setup}
    return commands[arguments.command]()


def handle_run() -> int:
    """Start the interactive session, refusing if it cannot work yet."""
    from llmft.environment import detect

    missing = detect().missing(("rich", "prompt_toolkit", "huggingface_hub"))
    if missing:
        print(f"The interactive session needs: {', '.join(missing)}.\nRun: python run.py setup")
        return 1

    from llmft.cli import run_cli

    return run_cli()


def handle_web() -> int:
    """Serve the browser interface, refusing if it cannot work yet."""
    from llmft.environment import detect

    missing = detect().missing(("flask", "huggingface_hub"))
    if missing:
        print(f"The web interface needs: {', '.join(missing)}.\nRun: python run.py setup")
        return 1

    from llmft.web import DEFAULT_HOST, DEFAULT_PORT, serve

    print(f"LLM Finetune is at http://{DEFAULT_HOST}:{DEFAULT_PORT}  (Ctrl+C to stop)")
    return serve()


def handle_check() -> int:
    """Print a capability report. Exit code 0 always: this is an observation."""
    from llmft.environment import detect, features

    capabilities = detect()
    _print_system(capabilities)
    _print_packages(capabilities)
    _print_tools(capabilities)
    _print_features(features(capabilities))
    return 0


def handle_setup() -> int:
    """Plan and, once confirmed, perform every installation step."""
    from llmft import bootstrap
    from llmft.environment import detect

    capabilities = detect()
    result = 0
    try:
        _run_dependency_setup(bootstrap, capabilities)
        _run_llama_cpp_setup(bootstrap, capabilities)
    except bootstrap.BootstrapError as error:
        print(f"\nSetup failed: {error}", file=sys.stderr)
        result = 1
    else:
        print("\nSetup complete. Run 'python run.py check' to confirm.")
    return result


def _run_dependency_setup(bootstrap, capabilities) -> None:
    """Install Python dependencies, reporting what was already satisfied."""
    plan = bootstrap.plan_dependencies(capabilities)
    _print_heading("Python dependencies")
    if plan.is_empty:
        print("  Everything is already installed.")
    else:
        print(f"  Missing: {', '.join(plan.missing)}")
        print(f"  PyTorch wheels: {plan.wheel_tag}")
        for step in plan.steps:
            print(f"    $ {' '.join(step.command)}")
        bootstrap.install_dependencies(plan, _confirm, _print_progress)


def _run_llama_cpp_setup(bootstrap, capabilities) -> None:
    """Install llama.cpp, keeping quantization optional if it is declined."""
    _print_heading("llama.cpp (GGUF conversion and k-quants)")
    plan = bootstrap.plan_llama_cpp(capabilities)
    if plan.is_empty:
        print("  Already installed.")
    else:
        bootstrap.install_llama_cpp(plan, _confirm, _print_progress)


def _print_system(capabilities) -> None:
    """Show platform, interpreter and GPU details."""
    from llmft import __version__

    _print_heading(f"LLM Finetune {__version__} - environment check")
    _print_row("Platform", capabilities.platform_description)
    _print_row("Python", f"{capabilities.python_version} ({capabilities.python_executable})")
    gpu = capabilities.gpu
    if gpu is None:
        _print_row("GPU", "none detected (nvidia-smi unavailable)")
    else:
        _print_row("GPU", f"{gpu.name}, {gpu.total_memory_mb} MiB")
        _print_row("Driver", f"{gpu.driver_version}  ->  {capabilities.wheel_tag} wheels")
        supported = "supported" if gpu.supports_bf16 else "NOT supported, disable bf16"
        _print_row("bfloat16", f"{supported} (compute capability {gpu.compute_capability})")


def _print_packages(capabilities) -> None:
    """List every tracked package and its installed version."""
    _print_heading("Packages")
    for name, version in capabilities.packages.items():
        _print_row(name, version or "not installed")


def _print_tools(capabilities) -> None:
    """List the external programs the pipeline shells out to."""
    _print_heading("External tools")
    _print_row("llama.cpp convert", str(capabilities.convert_script or "not installed"))
    _print_row("llama-quantize", str(capabilities.llama_quantize or "not installed"))
    _print_row("ollama", str(capabilities.ollama or "not installed"))
    _print_row("lms (LM Studio)", str(capabilities.lms or "not installed"))


def _print_features(all_features) -> None:
    """Show which operations are available, and why the others are not."""
    _print_heading("Available operations")
    for feature in all_features:
        marker = "x" if feature.available else " "
        detail = "" if feature.available else f"  {feature.reason}"
        print(f"  [{marker}] {feature.label:<{LABEL_WIDTH}}{detail}")


def _print_progress(event) -> None:
    """Render a bootstrap progress event on a single rewritten line."""
    fraction = event.fraction
    percentage = f" {fraction * 100:5.1f}%" if fraction is not None else ""
    sys.stdout.write(f"\r  {event.message[:90]:<98}{percentage}")
    sys.stdout.flush()
    if fraction == 1.0:
        sys.stdout.write("\n")


def _confirm(question: str) -> bool:
    """Ask for explicit approval before anything is downloaded or installed."""
    print(f"\n{question}")
    answer = input("Proceed? [y/N] ").strip().lower()
    approved = answer in {"y", "yes"}
    if not approved:
        print("  Skipped.")
    return approved


def _print_heading(title: str) -> None:
    """Print a section heading."""
    print(f"\n{title}\n{'-' * len(title)}")


def _print_row(label: str, value: str) -> None:
    """Print one aligned label/value pair."""
    print(f"  {label:<{LABEL_WIDTH}}{value}")


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(prog="run.py", description="QLoRA fine-tuning for open-weight models.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("run", help="start the interactive session (the default)")
    subparsers.add_parser("web", help="serve the browser interface")
    subparsers.add_parser("check", help="report what this machine can currently do")
    subparsers.add_parser("setup", help="install missing libraries and binaries")
    parser.set_defaults(command="run")
    return parser


if __name__ == "__main__":
    sys.exit(main())
