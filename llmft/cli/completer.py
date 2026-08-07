"""HuggingFace model search as the user types.

The specification asks for suggestions that update on every input change. Taken
literally that is a network request per keystroke, so three things temper it:
a minimum query length, a cache keyed on the query, and a worker thread so the
prompt never blocks on the network.

Results already have the original-model heuristic and the prefix rule applied by
:mod:`llmft.hub`; this module only presents them.
"""

from __future__ import annotations

from functools import lru_cache

from prompt_toolkit.completion import Completer, Completion, ThreadedCompleter

from llmft.hub import HubError, ModelCandidate, search_models

# Below this, a query matches so much that the suggestions are noise.
MINIMUM_QUERY_LENGTH = 2
SUGGESTION_LIMIT = 20
CACHE_SIZE = 128


def model_completer() -> Completer:
    """Build the completer used by the model prompt.

    Wrapped in a :class:`ThreadedCompleter` so a slow or unreachable network
    leaves the user still able to type.
    """
    return ThreadedCompleter(ModelCompleter())


class ModelCompleter(Completer):
    """Suggests original HuggingFace models whose name starts with the input."""

    def get_completions(self, document, complete_event):
        """Yield one completion per matching model, best first."""
        query = document.text_before_cursor.strip()
        if len(query) < MINIMUM_QUERY_LENGTH:
            return
        for candidate in _cached_search(query):
            yield Completion(
                candidate.model_id,
                start_position=-len(document.text_before_cursor),
                display=candidate.model_id,
                display_meta=_describe(candidate),
            )


def _describe(candidate: ModelCandidate) -> str:
    """Summarise a candidate for the dropdown's right-hand column."""
    marks = ["cached"] if candidate.is_local else []
    if candidate.is_known_publisher:
        marks.append("known publisher")
    marks.append(f"{candidate.downloads:,} downloads")
    return " | ".join(marks)


@lru_cache(maxsize=CACHE_SIZE)
def _cached_search(query: str) -> tuple[ModelCandidate, ...]:
    """Search, remembering the answer.

    Typing "Qwen3-8" issues one request per new prefix rather than repeating
    earlier ones, and backspacing costs nothing. A failed search yields no
    suggestions instead of an error: the user can still type an identifier in
    full, and the model is validated before anything is downloaded.
    """
    try:
        results = search_models(query, limit=SUGGESTION_LIMIT)
    except HubError:
        results = ()
    return results
