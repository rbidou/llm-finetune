"""Validation of the training dataset.

The expected format is a JSON array of conversations::

    [{"messages": [{"role": "user", "content": "..."}, ...]}, ...]

Validation reports *every* problem it finds rather than stopping at the first,
and each one carries the exact JSON path to the offending value. Fixing a
hand-written dataset one error per run is miserable; a single list of
``record[12].messages[3]: ...`` lines is not.

The statistics gathered alongside are not decoration: the longest conversation
is what tells the user whether their sequence length is set sensibly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KNOWN_ROLES = ("system", "user", "assistant", "tool")
MESSAGES_KEY = "messages"
ROLE_KEY = "role"
CONTENT_KEY = "content"

# Enough to fix a broken file in one pass without printing a novel.
MAX_REPORTED_ISSUES = 50


@dataclass(frozen=True)
class DatasetIssue:
    """One validation failure, located precisely within the file."""

    location: str
    problem: str

    def __str__(self) -> str:
        return f"{self.location}: {self.problem}"


@dataclass(frozen=True)
class DatasetReport:
    """The outcome of validating a dataset file."""

    path: Path
    record_count: int = 0
    message_count: int = 0
    role_counts: dict[str, int] = field(default_factory=dict)
    longest_record_chars: int = 0
    issues: tuple[DatasetIssue, ...] = ()
    truncated: bool = False

    @property
    def is_valid(self) -> bool:
        """Whether the file can be used for training."""
        return not self.issues

    def summary(self) -> str:
        """A one-line description of what the file contains."""
        roles = ", ".join(f"{count} {role}" for role, count in sorted(self.role_counts.items()))
        return (
            f"{self.record_count} conversations, {self.message_count} messages "
            f"({roles}); longest conversation {self.longest_record_chars} characters"
        )


def validate_dataset(path: Path) -> DatasetReport:
    """Check a dataset file and report everything wrong with it."""
    payload, failure = _read_json(path)
    if failure is not None:
        return DatasetReport(path=path, issues=(failure,))
    if not isinstance(payload, list):
        problem = f"the file must contain a JSON array of conversations, found {_type_name(payload)}"
        return DatasetReport(path=path, issues=(DatasetIssue("file", problem),))
    if not payload:
        return DatasetReport(path=path, issues=(DatasetIssue("file", "the dataset is empty"),))
    return _validate_records(path, payload)


def _validate_records(path: Path, records: list[Any]) -> DatasetReport:
    """Walk every conversation, accumulating issues and statistics."""
    issues: list[DatasetIssue] = []
    role_counts: dict[str, int] = {}
    message_count = 0
    longest = 0

    for index, record in enumerate(records):
        location = f"record[{index}]"
        messages = _record_messages(record, location, issues)
        if messages is None:
            continue
        message_count += len(messages)
        longest = max(longest, len(json.dumps(record, ensure_ascii=False)))
        for position, message in enumerate(messages):
            role = _validate_message(message, f"{location}.{MESSAGES_KEY}[{position}]", issues)
            if role:
                role_counts[role] = role_counts.get(role, 0) + 1
        if not any(_role_of(message) == "assistant" for message in messages):
            issues.append(DatasetIssue(location, "no assistant message, so this conversation teaches nothing"))

    return DatasetReport(
        path=path,
        record_count=len(records),
        message_count=message_count,
        role_counts=role_counts,
        longest_record_chars=longest,
        issues=tuple(issues[:MAX_REPORTED_ISSUES]),
        truncated=len(issues) > MAX_REPORTED_ISSUES,
    )


def _record_messages(record: Any, location: str, issues: list[DatasetIssue]) -> list[Any] | None:
    """Extract and check a record's message list, or record why it is unusable."""
    result: list[Any] | None = None
    if not isinstance(record, dict):
        issues.append(DatasetIssue(location, f"expected an object, found {_type_name(record)}"))
    elif MESSAGES_KEY not in record:
        issues.append(DatasetIssue(location, f"missing key '{MESSAGES_KEY}'"))
    elif not isinstance(record[MESSAGES_KEY], list):
        issues.append(
            DatasetIssue(location, f"'{MESSAGES_KEY}' must be an array, found {_type_name(record[MESSAGES_KEY])}")
        )
    elif not record[MESSAGES_KEY]:
        issues.append(DatasetIssue(location, f"'{MESSAGES_KEY}' is empty"))
    else:
        result = record[MESSAGES_KEY]
    return result


def _validate_message(message: Any, location: str, issues: list[DatasetIssue]) -> str | None:
    """Check one message and return its role when the message is usable."""
    if not isinstance(message, dict):
        issues.append(DatasetIssue(location, f"expected an object, found {_type_name(message)}"))
        return None

    role = message.get(ROLE_KEY)
    content = message.get(CONTENT_KEY)
    result: str | None = None

    if ROLE_KEY not in message:
        issues.append(DatasetIssue(location, f"missing key '{ROLE_KEY}'"))
    elif not isinstance(role, str) or not role.strip():
        issues.append(DatasetIssue(location, f"'{ROLE_KEY}' must be a non-empty string, found {role!r}"))
    elif role not in KNOWN_ROLES:
        issues.append(DatasetIssue(location, f"unknown role {role!r}; expected one of: {', '.join(KNOWN_ROLES)}"))
    else:
        result = role

    if CONTENT_KEY not in message:
        issues.append(DatasetIssue(location, f"missing key '{CONTENT_KEY}'"))
    elif not isinstance(content, str):
        issues.append(DatasetIssue(location, f"'{CONTENT_KEY}' must be a string, found {_type_name(content)}"))
    elif not content.strip():
        issues.append(DatasetIssue(location, f"'{CONTENT_KEY}' is empty"))

    return result


def _read_json(path: Path) -> tuple[Any, DatasetIssue | None]:
    """Load the file, converting any read or parse failure into an issue."""
    payload: Any = None
    failure: DatasetIssue | None = None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        failure = DatasetIssue("file", f"{path} does not exist")
    except OSError as error:
        failure = DatasetIssue("file", f"{path} could not be read: {error}")
    except UnicodeDecodeError as error:
        failure = DatasetIssue("file", f"{path} is not valid UTF-8: {error}")
    except json.JSONDecodeError as error:
        failure = DatasetIssue(f"line {error.lineno}, column {error.colno}", f"invalid JSON: {error.msg}")
    return payload, failure


def _role_of(message: Any) -> str | None:
    """Read a message's role defensively, for use during validation."""
    return message.get(ROLE_KEY) if isinstance(message, dict) else None


def _type_name(value: Any) -> str:
    """Name a JSON value's type the way a user would recognise it."""
    names = {dict: "an object", list: "an array", str: "a string", bool: "a boolean", type(None): "null"}
    return names.get(type(value), f"a {type(value).__name__}")
