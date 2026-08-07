"""Model discovery: searching HuggingFace, and listing what is already cached.

The hard part here is "only original models, no fine-tuned or altered versions".
HuggingFace has no authoritative base-model flag, so :func:`exclusion_reason`
applies a heuristic. Its shape is driven by one observation:

    Almost every official instruction-tuned model declares a ``base_model``.
    Qwen2.5-Coder-7B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct,
    SmolLM2-135M-Instruct and gemma-2-9b-it all do.

So "declares a base model" cannot mean "is a derivative" -- that rule would hide
exactly the models people most want to fine-tune. What separates an official
variant from someone else's alteration is *who published it*: a model family's
own instruct release is published by the author of its base, while a community
re-tune, merge or quantization is published by somebody else. That, plus the
format and quantization markers, is the rule.

The heuristic is a pure function over primitives so it can be tested against
recorded metadata without a network call.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Sequence

from llmft.progress import ProgressSink, Stage, null_sink, report

SEARCH_SORT = "trending_score"

# Over-fetch, because the prefix filter and the heuristic both discard results
# and the API can only sort and limit before either has been applied.
SEARCH_OVERFETCH = 8
MAX_API_LIMIT = 200

# Publishers of original base models. Used only for ordering, never to exclude:
# a good model from an unlisted organisation must still be reachable.
KNOWN_PUBLISHERS = frozenset(
    {
        "qwen", "meta-llama", "mistralai", "google", "microsoft", "deepseek-ai",
        "huggingfacetb", "ibm-granite", "allenai", "tiiuae", "eleutherai", "bigcode",
        "nvidia", "apple", "openai", "stabilityai", "01-ai", "thudm", "internlm",
        "coherelabs", "cohereforai", "zai-org", "moonshotai", "baidu", "liquidai",
        "arcee-ai", "jetbrains", "salesforce", "facebook", "bytedance-seed", "kyutai",
        "openbmb", "rwkv", "state-spaces", "xai-org", "inclusionai", "servicenow-ai",
    }
)

# Repackagings of some other model: never fine-tunable, whatever else is present.
QUANTIZED_TAGS = frozenset(
    {
        "gguf", "ggml", "awq", "gptq", "exl2", "exl3", "mlx", "peft", "lora",
        "adapter", "quantization", "quantized", "bitsandbytes",
        "2-bit", "3-bit", "4-bit", "5-bit", "6-bit", "8-bit", "nvfp4", "fp8", "int4", "int8",
    }
)

# Formats that official repositories often publish *alongside* real weights, as a
# convenience. SmolLM2-135M-Instruct ships ONNX next to its safetensors; an
# onnx-community mirror of the same model ships only ONNX. These exclude a
# repository only when no safetensors weights accompany them.
EXPORT_TAGS = frozenset({"onnx", "openvino", "tensorrt", "transformers.js", "coreml", "tflite"})

# Presence of this tag means the repository holds weights transformers can load.
# It is the discriminator between "also exports ONNX" and "is only an ONNX port".
WEIGHTS_TAG = "safetensors"

# Tags describing a structural or behavioural alteration of a published model.
ALTERATION_TAGS = frozenset({"merge", "mergekit", "abliterated", "uncensored", "heretic", "frankenmerge"})

# Explicit derivative relationships, when a card bothers to declare one.
DERIVATIVE_RELATIONS = frozenset({"quantized", "adapter", "merge"})

# Quantization and packaging markers that appear in repository names.
DERIVATIVE_NAME_PATTERN = re.compile(
    r"(gguf|ggml|awq|gptq|exl[23]|mlx|nvfp4|bnb|[2-8]-?bit|int[48]|fp8|w[48]a[48]|"
    r"abliterated|uncensored|mergekit|-lora\b|-adapter\b)",
    re.IGNORECASE,
)


class HubError(RuntimeError):
    """Raised when HuggingFace cannot be reached or a model cannot be read."""


@dataclass(frozen=True)
class ModelCandidate:
    """One model offered to the user by the picker."""

    model_id: str
    author: str
    trending_score: float
    downloads: int
    likes: int
    is_local: bool

    @property
    def name(self) -> str:
        """The repository name without its organisation prefix."""
        return self.model_id.rsplit("/", maxsplit=1)[-1]

    @property
    def is_known_publisher(self) -> bool:
        """Whether the author is a recognised publisher of base models."""
        return self.author.lower() in KNOWN_PUBLISHERS


@dataclass(frozen=True)
class LocalModel:
    """A model already present in the HuggingFace cache."""

    model_id: str
    size_on_disk: int
    path: Path
    description: str
    architecture: str = ""

    @property
    def is_causal_lm(self) -> bool:
        """Whether this model can be fine-tuned by this application.

        A HuggingFace cache accumulates every kind of model. Offering a
        summarization encoder-decoder as a fine-tuning target would only produce
        a failure several minutes into a run.
        """
        return self.architecture.endswith("ForCausalLM")


def search_models(query: str, limit: int = 20, api: Any | None = None) -> tuple[ModelCandidate, ...]:
    """Return original models whose name starts with ``query``, best first.

    Ordering puts recognised publishers ahead of everyone else, then falls back
    to trending score and download count.
    """
    cleaned = query.strip()
    if not cleaned:
        return ()
    infos = _list_models(cleaned, min(limit * SEARCH_OVERFETCH, MAX_API_LIMIT), api)
    local = local_model_ids()
    candidates = [
        _to_candidate(info, local)
        for info in infos
        if matches_prefix(str(info.id), cleaned) and _is_original(info)
    ]
    return tuple(sorted(candidates, key=_ranking_key)[:limit])


def local_models() -> tuple[LocalModel, ...]:
    """Return every model in the HuggingFace cache, largest first."""
    from huggingface_hub import scan_cache_dir

    try:
        cache = scan_cache_dir()
    except Exception as error:  # noqa: BLE001 - the cache may be absent or unreadable
        raise HubError(f"Could not read the HuggingFace cache: {error}") from error
    found = [_to_local_model(repo) for repo in cache.repos if repo.repo_type == "model"]
    return tuple(sorted(found, key=lambda model: -model.size_on_disk))


def local_model_ids() -> frozenset[str]:
    """Return the identifiers of every cached model, for the "available" marker."""
    try:
        identifiers = frozenset(model.model_id for model in local_models())
    except HubError:
        identifiers = frozenset()
    return identifiers


def matches_prefix(model_id: str, query: str) -> bool:
    """Whether ``model_id`` starts with ``query``, as the specification requires.

    The HuggingFace ``search`` parameter matches substrings, so the prefix rule
    is applied here. A query without a slash may match either the full
    identifier or the repository name, because users type "Qwen3" far more often
    than "Qwen/Qwen3".
    """
    wanted = query.strip().lower()
    full = model_id.lower()
    name = full.rsplit("/", maxsplit=1)[-1]
    return full.startswith(wanted) if "/" in wanted else (full.startswith(wanted) or name.startswith(wanted))


def exclusion_reason(
    model_id: str,
    tags: Sequence[str] = (),
    base_model: Any = None,
    relation: str | None = None,
) -> str | None:
    """Explain why a model is not an original, or return ``None`` if it is.

    Returning the reason rather than a boolean keeps the rule debuggable: the
    picker can say *why* a repository was hidden.
    """
    lowered = {str(tag).lower() for tag in tags or ()}
    author = _author_of(model_id)
    name = model_id.rsplit("/", maxsplit=1)[-1]

    if relation and str(relation).lower() in DERIVATIVE_RELATIONS:
        return f"declared as a {relation} of another model"
    if quantized_tags := lowered & QUANTIZED_TAGS:
        return f"repackaged format ({', '.join(sorted(quantized_tags))})"
    if (export_tags := lowered & EXPORT_TAGS) and WEIGHTS_TAG not in lowered:
        return f"export-only repository ({', '.join(sorted(export_tags))})"
    if alteration_tags := lowered & ALTERATION_TAGS:
        return f"altered model ({', '.join(sorted(alteration_tags))})"
    if marker := DERIVATIVE_NAME_PATTERN.search(name):
        return f"name marks a derivative ({marker.group(0)})"
    foreign = [owner for owner in _base_authors(base_model) if owner != author.lower()]
    if foreign:
        return f"fine-tune of {foreign[0]}'s model"
    return None


def is_original_model(
    model_id: str,
    tags: Sequence[str] = (),
    base_model: Any = None,
    relation: str | None = None,
) -> bool:
    """Whether a model looks like an original rather than an altered version."""
    return exclusion_reason(model_id, tags, base_model, relation) is None


def download_model(model_id: str, sink: ProgressSink = null_sink) -> Path:
    """Fetch a model's files, reporting one aggregated progress figure.

    Done explicitly, before the model is loaded, because on a first run the
    download is by far the longest part and transformers would otherwise do it
    silently inside ``from_pretrained``. Library progress bars are disabled
    application-wide, so this is what the user actually sees.

    Only the files transformers needs are fetched. A repository often carries
    both ``.safetensors`` and ``.bin`` copies of the same weights, and pulling
    both would double a multi-gigabyte download for nothing.
    """
    from huggingface_hub import snapshot_download

    reporter = _DownloadReporter(sink, model_id)
    try:
        location = snapshot_download(
            model_id,
            allow_patterns=_needed_files(model_id),
            tqdm_class=reporter.tqdm_class(),
        )
    except Exception as error:  # noqa: BLE001 - network, auth and disk failures alike
        raise HubError(f"Could not download {model_id}: {error}") from error

    # Transferred bytes can fall short of the total when files are already
    # cached or deduplicated, which would leave the bar stopped part-way
    # through a download that has in fact finished.
    report(sink, Stage.TRAIN, f"Fetched {model_id}", 1, 1)
    return Path(location)


def _needed_files(model_id: str, api: Any | None = None) -> list[str]:
    """Choose which files to fetch, preferring safetensors weights."""
    client = api or _api()
    try:
        present = list(client.list_repo_files(model_id))
    except Exception:  # noqa: BLE001 - fall back to fetching everything
        return ["*"]
    weights = "*.safetensors" if any(name.endswith(".safetensors") for name in present) else "*.bin"
    return ["*.json", "*.txt", "*.model", "*.jinja", weights]


class _DownloadReporter:
    """Turns the download's internal counters into one progress figure.

    ``snapshot_download`` creates several progress bars measured in bytes: one
    for the network transfer, one for reconstructing files on disk, and more per
    file. Summing them would report roughly twice the real download size, so
    only the transfer bar is followed, identified by its description.

    Its total starts at zero and grows as files are discovered, so the total is
    read from the bar at each update rather than captured once at the start.
    """

    TRANSFER_PREFIX = "downloading"

    def __init__(self, sink: ProgressSink, model_id: str) -> None:
        self.sink = sink
        self.model_id = model_id
        self.completed = 0
        self.lock = Lock()

    def tqdm_class(self) -> type:
        """Build the tqdm replacement that ``snapshot_download`` should use."""
        from tqdm import tqdm as base_tqdm

        reporter = self

        class _ReportingTqdm(base_tqdm):
            """A tqdm that draws nothing and forwards its counter instead.

            Whether to follow this bar is decided from the arguments, not the
            instance: a disabled tqdm returns early from its constructor and
            never sets ``unit``. It does set ``total``, which the library then
            keeps updating, so that one is read from the instance.
            """

            def __init__(self, *args, **kwargs) -> None:
                description = str(kwargs.get("desc") or "").lower()
                self.is_transfer_bar = description.startswith(reporter.TRANSFER_PREFIX)
                kwargs["disable"] = True
                super().__init__(*args, **kwargs)

            def update(self, n: int = 1) -> bool | None:
                displayed = super().update(n)
                if self.is_transfer_bar:
                    reporter._advance(n or 0, self.total)
                return displayed

        return _ReportingTqdm

    def _advance(self, amount: int, total: int | None) -> None:
        """Record bytes received and report the running total."""
        with self.lock:
            self.completed += amount
            completed = self.completed
        report(
            self.sink,
            Stage.TRAIN,
            f"Downloading {self.model_id} ({_human_gigabytes(completed)} of {_human_gigabytes(total or 0)})",
            completed,
            total or None,
        )


@lru_cache(maxsize=64)
def fetch_config(model_id: str) -> dict[str, Any]:
    """Download and parse a model's ``config.json``.

    Cached because the estimator, the trainer and the picker all want it, and it
    is the one file that must be read before anything large is downloaded.
    """
    from huggingface_hub import hf_hub_download

    try:
        location = hf_hub_download(repo_id=model_id, filename="config.json")
        payload = json.loads(Path(location).read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001 - network, auth and parse failures alike
        raise HubError(f"Could not read config.json for {model_id}: {error}") from error
    return payload


def fetch_parameter_count(model_id: str, api: Any | None = None) -> int | None:
    """Return the model's parameter count as reported by the safetensors index."""
    client = api or _api()
    try:
        info = client.model_info(model_id, expand=["safetensors"])
    except Exception as error:  # noqa: BLE001 - absent, gated or unreachable
        raise HubError(f"Could not read model metadata for {model_id}: {error}") from error
    safetensors = getattr(info, "safetensors", None)
    return getattr(safetensors, "total", None) if safetensors else None


def _list_models(query: str, limit: int, api: Any | None) -> list[Any]:
    """Call the search API, translating any failure into :class:`HubError`."""
    client = api or _api()
    try:
        results = list(client.list_models(search=query, sort=SEARCH_SORT, limit=limit, cardData=True))
    except Exception as error:  # noqa: BLE001 - offline, rate-limited or API change
        raise HubError(f"HuggingFace search failed: {error}") from error
    return results


def _api() -> Any:
    """Build a HuggingFace API client."""
    from huggingface_hub import HfApi

    return HfApi()


def _is_original(info: Any) -> bool:
    """Apply the heuristic to one search result."""
    card = getattr(info, "card_data", None)
    return is_original_model(
        str(info.id),
        tags=getattr(info, "tags", None) or (),
        base_model=_card_value(card, "base_model"),
        relation=_card_value(card, "base_model_relation"),
    )


def _card_value(card: Any, key: str) -> Any:
    """Read one key from card metadata, which may be absent or a plain mapping."""
    return card.get(key) if card is not None and hasattr(card, "get") else None


def _to_candidate(info: Any, local: frozenset[str]) -> ModelCandidate:
    """Adapt a search result into the picker's own type."""
    model_id = str(info.id)
    return ModelCandidate(
        model_id=model_id,
        author=_author_of(model_id),
        trending_score=float(getattr(info, "trending_score", 0) or 0),
        downloads=int(getattr(info, "downloads", 0) or 0),
        likes=int(getattr(info, "likes", 0) or 0),
        is_local=model_id in local,
    )


def _to_local_model(repo: Any) -> LocalModel:
    """Adapt a cache entry, describing it from its config where possible."""
    config = _newest_local_config(repo)
    architecture = str((config.get("architectures") or [config.get("model_type") or "unknown"])[0])
    return LocalModel(
        model_id=repo.repo_id,
        size_on_disk=repo.size_on_disk,
        path=repo.repo_path,
        description=_describe_cached(config, architecture, repo.size_on_disk),
        architecture=architecture,
    )


def _describe_cached(config: dict[str, Any], architecture: str, size_on_disk: int) -> str:
    """Summarise a cached model for the local picker."""
    layers = config.get("num_hidden_layers")
    hidden = config.get("hidden_size")
    shape = f", {layers} layers, hidden {hidden}" if layers and hidden else ""
    return f"{architecture}{shape}, {_human_gigabytes(size_on_disk)} on disk"


def _newest_local_config(repo: Any) -> dict[str, Any]:
    """Read ``config.json`` from the most recently modified cached revision."""
    revisions = sorted(repo.revisions, key=lambda revision: revision.last_modified or 0, reverse=True)
    payload: dict[str, Any] = {}
    for revision in revisions:
        candidate = Path(revision.snapshot_path) / "config.json"
        if candidate.is_file():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            break
    return payload


def _ranking_key(candidate: ModelCandidate) -> tuple[int, float, int]:
    """Order recognised publishers first, then by trend, then by downloads."""
    return (0 if candidate.is_known_publisher else 1, -candidate.trending_score, -candidate.downloads)


def _base_authors(base_model: Any) -> tuple[str, ...]:
    """Return the owning organisations declared in ``base_model``.

    The field is a string on some cards and a list on others, so both are
    accepted. Entries without an organisation are ignored: they carry no
    ownership information to compare against.
    """
    entries = base_model if isinstance(base_model, (list, tuple)) else [base_model] if base_model else []
    return tuple(str(entry).split("/", maxsplit=1)[0].lower() for entry in entries if "/" in str(entry))


def _author_of(model_id: str) -> str:
    """Return the organisation that published a model, or an empty string."""
    return model_id.split("/", maxsplit=1)[0] if "/" in model_id else ""


def _human_gigabytes(size: int) -> str:
    """Format a byte count in gigabytes."""
    return f"{size / 1e9:.1f} GB"
