"""Folder listings for the directory browser on the Setup page.

A browser cannot hand the server the absolute path of a folder picked with a
native dialog -- that is deliberate on the browser's part -- so the browser is
built here instead: the server lists folders on this machine and the page walks
them. Only folder names are returned, never file contents.
"""

from __future__ import annotations

import os
import string
from dataclasses import dataclass
from pathlib import Path

# FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM, as reported by os.stat on Windows.
WINDOWS_HIDDEN = 0x2 | 0x4


class FolderError(ValueError):
    """Raised when a folder cannot be listed."""


@dataclass(frozen=True)
class FolderListing:
    """One folder's subfolders, plus what is needed to move around from it."""

    path: Path
    parent: Path | None
    folders: tuple[Path, ...]
    roots: tuple[Path, ...]


def list_folder(raw: str) -> FolderListing:
    """List the subfolders of ``raw``, or of its nearest existing ancestor.

    Starting from an ancestor lets the browser open on a directory that has been
    typed but not created yet. Hidden folders are left out, and folders this
    user may not read are skipped rather than failing the whole listing.
    """
    path = _nearest_existing(Path(raw.strip().strip('"')).expanduser().resolve())
    try:
        entries = sorted(os.scandir(path), key=lambda entry: entry.name.lower())
    except OSError as error:
        raise FolderError(f"{path} cannot be read: {error.strerror or error}") from error
    folders = tuple(Path(entry.path) for entry in entries if _is_visible_folder(entry))
    return FolderListing(path, path.parent if path.parent != path else None, folders, filesystem_roots())


def filesystem_roots() -> tuple[Path, ...]:
    """Every drive on Windows; the single root elsewhere."""
    candidates = (Path(f"{letter}:\\") for letter in string.ascii_uppercase) if os.name == "nt" else (Path("/"),)
    return tuple(root for root in candidates if root.exists())


def _nearest_existing(path: Path) -> Path:
    """Return ``path`` if it is a directory, else its closest directory ancestor.

    A path on a drive that is not there at all -- the default output directory
    on a machine without a ``D:`` drive -- opens on the home folder instead.
    """
    return next((candidate for candidate in (path, *path.parents) if candidate.is_dir()), Path.home())


def _is_visible_folder(entry: os.DirEntry) -> bool:
    """True for a non-hidden directory.

    Windows marks its own hidden folders by attribute rather than by name, and
    those -- ``System Volume Information`` among them -- refuse to be opened.
    """
    try:
        attributes = getattr(entry.stat(), "st_file_attributes", 0)
        return entry.is_dir() and not entry.name.startswith((".", "$")) and not attributes & WINDOWS_HIDDEN
    except OSError:
        return False
