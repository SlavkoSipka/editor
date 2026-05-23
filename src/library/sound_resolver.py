"""Resolve the playable local path for a sound, fetching from R2 if needed.

Sound metadata stored in match plans keeps relative paths like
``data/library/198340.mp3`` or ``data/library/sonniss/foo.mp3``. The resolver
maps those to:

* ``local`` mode (default) → the path under ``LIBRARY_DIR`` (or absolute as-is).
* ``r2`` mode             → the LRU cache, downloading on miss.

Callers should treat a ``None`` return as "sound unavailable" and skip cleanly
rather than failing the whole job.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from src.config import LIBRARY_DIR, PROJECT_ROOT, STORAGE_MODE
from src.utils.logger import get_logger

logger = get_logger("sound_resolver")

_LIBRARY_PREFIXES = ("data/library/", "data/library\\")


def _stored_path(match: dict[str, Any]) -> str:
    return str(match.get("sound_path") or match.get("local_path") or "").strip()


def _rel_to_library(stored: str) -> str | None:
    """Strip the ``data/library/`` prefix to get a path relative to LIBRARY_DIR.

    Returns ``None`` if the stored path isn't shaped like a library asset.
    """
    s = stored.replace("\\", "/")
    for prefix in ("data/library/",):
        if s.startswith(prefix):
            return s[len(prefix):]
    if s.startswith("library/"):
        return s[len("library/"):]
    return None


def _r2_key_for(stored: str) -> str | None:
    """Map a stored path to its R2 key (under the ``library/`` prefix)."""
    rel = _rel_to_library(stored)
    if rel is None:
        return None
    return f"library/{rel}"


def _local_path_for(stored: str) -> Path | None:
    """Resolve a stored path to a real on-disk location (no fetching)."""
    if not stored:
        return None
    p = Path(stored)
    if p.is_absolute():
        return p if p.exists() else None

    rel = _rel_to_library(stored)
    if rel is not None:
        candidate = LIBRARY_DIR / rel
        if candidate.exists():
            return candidate

    candidate = PROJECT_ROOT / stored
    if candidate.exists():
        return candidate

    fallback = LIBRARY_DIR / Path(stored).name
    return fallback if fallback.exists() else None


def resolve_sound_path_with_status(
    match: dict[str, Any],
) -> tuple[Path | None, str, str | None]:
    """Like :func:`resolve_sound_path` but also returns a status + optional note.

    Status values:

    * ``"local"``  — file found on local disk (no fetch needed)
    * ``"cached"`` — already in the R2 LRU cache from a previous fetch
    * ``"fetched"`` — downloaded from R2 just now
    * ``"missing_path"`` — match has no usable ``sound_path``
    * ``"no_r2_key"``    — stored path doesn't look like a library asset
    * ``"failed"``       — local lookup miss + R2 fetch failed (or local-only miss)
    """
    stored = _stored_path(match)
    if not stored:
        return None, "missing_path", "no sound_path / local_path in match"

    if STORAGE_MODE != "r2":
        path = _local_path_for(stored)
        if path is None:
            logger.warning("Sound not found locally: %s", stored)
            return None, "failed", f"local file missing: {stored}"
        return path, "local", None

    path = _local_path_for(stored)
    if path is not None:
        return path, "local", None

    from src.library.r2_fetch import _cache_path_for, fetch_sound

    r2_key = _r2_key_for(stored)
    if r2_key is None:
        logger.warning("Cannot derive R2 key from stored path: %s", stored)
        return None, "no_r2_key", f"unrecognised path: {stored}"

    cache_path = _cache_path_for(r2_key)
    already_cached = cache_path.exists() and cache_path.stat().st_size > 0

    path = fetch_sound(r2_key)
    if path is None:
        return None, "failed", f"R2 fetch failed: {r2_key}"
    return path, ("cached" if already_cached else "fetched"), None


def resolve_sound_path(match: dict[str, Any]) -> Path | None:
    """Return a usable local file path for the sound referenced by ``match``.

    Pure local lookup in local mode; falls back to R2 fetch in r2 mode.
    Returns ``None`` if the sound can't be obtained.
    """
    path, _status, _note = resolve_sound_path_with_status(match)
    return path


def _iter_match_stored_paths(match_plan: dict[str, Any]) -> Iterable[str]:
    for m in match_plan.get("matches") or []:
        stored = _stored_path(m)
        if stored:
            yield stored
    music = match_plan.get("music") or {}
    if isinstance(music, dict):
        stored = str(music.get("sound_path") or "").strip()
        if stored:
            yield stored


def prefetch_match_plan_sounds(match_plan: dict[str, Any]) -> None:
    """In R2 mode, parallel-fetch every sound the plan references.

    No-op in local mode. Failures are tolerated — the mixer skips missing
    sounds with a warning.
    """
    if STORAGE_MODE != "r2":
        return

    keys: list[str] = []
    for stored in _iter_match_stored_paths(match_plan):
        if _local_path_for(stored) is not None:
            continue
        key = _r2_key_for(stored)
        if key:
            keys.append(key)

    if not keys:
        return

    from src.library.r2_fetch import fetch_many

    fetch_many(keys)
