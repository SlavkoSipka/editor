"""Fetch individual sound files from Cloudflare R2 on demand with a local LRU cache.

Used in R2 storage mode (Railway Hobby): the full library lives on R2, only
recently-used sounds occupy the small persistent volume. Falls back gracefully
if R2 is unreachable — callers treat a missing return as "skip this sound".
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore.config import Config

from src.config import SOUND_CACHE_DIR, SOUND_CACHE_MAX_MB
from src.utils.logger import get_logger

logger = get_logger("r2_fetch")

_EVICT_TARGET_RATIO = 0.85
_DEFAULT_BUCKET = "ai-sfx-library"

_cache_lock = threading.Lock()
_client = None
_client_lock = threading.Lock()


def _get_client():
    global _client
    with _client_lock:
        if _client is None:
            endpoint = os.environ.get("R2_ENDPOINT_URL")
            access_key = os.environ.get("R2_ACCESS_KEY_ID")
            secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
            if not all([endpoint, access_key, secret_key]):
                raise RuntimeError(
                    "R2 credentials missing (need R2_ENDPOINT_URL, "
                    "R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY)"
                )
            _client = boto3.client(
                "s3",
                endpoint_url=endpoint.rstrip("/"),
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                config=Config(
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "standard"},
                ),
                region_name="auto",
            )
        return _client


def _bucket() -> str:
    return os.environ.get("R2_BUCKET", _DEFAULT_BUCKET)


def _cache_size_mb() -> float:
    total = 0
    for f in SOUND_CACHE_DIR.glob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total / 1024 / 1024


def _evict_if_needed() -> None:
    """LRU eviction by access time until cache is well under the limit."""
    size = _cache_size_mb()
    if size <= SOUND_CACHE_MAX_MB:
        return
    files = sorted(
        (f for f in SOUND_CACHE_DIR.glob("*") if f.is_file()),
        key=lambda f: f.stat().st_atime,
    )
    target = SOUND_CACHE_MAX_MB * _EVICT_TARGET_RATIO
    for f in files:
        if _cache_size_mb() <= target:
            break
        try:
            f.unlink()
        except OSError:
            pass
    logger.info("Sound cache evicted to %.0f MB (limit %d MB)",
                _cache_size_mb(), SOUND_CACHE_MAX_MB)


def _cache_path_for(r2_key: str) -> Path:
    """Flatten an R2 key to a single cache filename (preserves extension)."""
    return SOUND_CACHE_DIR / r2_key.replace("/", "_")


def fetch_sound(r2_key: str, local_only_check: bool = False) -> Path | None:
    """Return a local Path for the given R2 key, downloading if needed.

    ``r2_key`` is an object key like ``library/198340.mp3`` or
    ``library/sonniss/foo.mp3``. Cached files are reused; access time is
    refreshed so frequently-used sounds stick around through LRU eviction.

    Returns ``None`` if the fetch fails (caller should skip the sound and warn).
    """
    SOUND_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = _cache_path_for(r2_key)

    if cached.exists() and cached.stat().st_size > 0:
        try:
            cached.touch()
        except OSError:
            pass
        return cached

    if local_only_check:
        return None

    try:
        with _cache_lock:
            if cached.exists() and cached.stat().st_size > 0:
                return cached
            client = _get_client()
            tmp = cached.with_suffix(cached.suffix + ".tmp")
            client.download_file(_bucket(), r2_key, str(tmp))
            tmp.rename(cached)
        _evict_if_needed()
        return cached
    except Exception as exc:
        logger.warning("Failed to fetch %s from R2: %s", r2_key, exc)
        return None


def fetch_many(r2_keys: list[str], max_workers: int = 8) -> dict[str, Path]:
    """Parallel pre-fetch; returns ``{r2_key: local_path}`` for successes."""
    if not r2_keys:
        return {}

    unique = list(dict.fromkeys(r2_keys))
    result: dict[str, Path] = {}
    lock = threading.Lock()

    def _one(key: str) -> None:
        path = fetch_sound(key)
        if path is not None:
            with lock:
                result[key] = path

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(_one, unique))

    logger.info("Pre-fetched %d/%d sounds from R2", len(result), len(unique))
    return result


def fetch_object(r2_key: str, dest: Path) -> Path | None:
    """Download an arbitrary R2 object to a specific destination path.

    Unlike ``fetch_sound``, this does NOT go through the LRU cache — used for
    bootstrap files like ``library/manifest.json`` that must live at a known
    location. Returns ``dest`` on success, ``None`` on failure.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        client = _get_client()
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        client.download_file(_bucket(), r2_key, str(tmp))
        tmp.rename(dest)
        return dest
    except Exception as exc:
        logger.warning("Failed to fetch object %s -> %s: %s", r2_key, dest, exc)
        return None
