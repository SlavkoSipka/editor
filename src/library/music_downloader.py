from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from src.config import LIBRARY_DIR
from src.library.freesound_client import search_sounds
from src.utils.logger import get_logger

logger = get_logger("music_downloader")

MUSIC_QUERIES: list[str] = [
    # Cinematic / dramatic
    "cinematic underscore", "epic music loop", "cinematic music", "trailer music",
    "dramatic underscore", "tension music", "suspense music", "dark cinematic",
    # Energetic / upbeat
    "upbeat music loop", "energetic music", "corporate upbeat", "motivational music",
    "electronic loop", "trap loop", "hip hop loop", "edm loop",
    # Ambient / mysterious
    "ambient music", "atmospheric music", "drone music", "ambient pad",
    "evolving texture", "soundscape", "ethereal music",
    # Playful / light
    "happy music loop", "playful music", "ukulele loop", "whistle music", "bouncy music",
    # Luxury / elegant
    "elegant music", "minimal music", "soft piano loop", "lofi loop", "smooth jazz loop",
    # Generic
    "background music", "music underscore", "instrumental loop",
]

MUSIC_DIR: Path = LIBRARY_DIR / "music"
_PAGE_SIZE = 50
_PAGES = (1, 2)
_MIN_DURATION_SEC = 15.0
_MAX_DURATION_SEC = 120.0
_DOWNLOAD_TIMEOUT_SEC = 120
_MAX_DOWNLOAD_WORKERS = 5


def _download_preview(url: str, dest: Path) -> bool:
    if dest.is_file() and dest.stat().st_size > 0:
        return False
    tmp = dest.parent / (dest.name + ".tmp")
    with requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT_SEC) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)
    return True


def _collect_candidates(queries: list[str]) -> dict[int, dict[str, Any]]:
    catalog: dict[int, dict[str, Any]] = {}

    for query in queries:
        for page in _PAGES:
            try:
                response = search_sounds(
                    query,
                    page=page,
                    page_size=_PAGE_SIZE,
                    cc0_only=True,
                    min_duration=_MIN_DURATION_SEC,
                    max_duration=_MAX_DURATION_SEC,
                )
            except RuntimeError as exc:
                logger.error("Music search failed for %r page %d: %s", query, page, exc)
                continue

            results = response.get("results", []) or []
            logger.info(
                "Searching for: %r page %d (found %d results)", query, page, len(results),
            )

            for r in results:
                sid = r.get("id")
                if sid is None:
                    continue
                if sid in catalog:
                    if query not in catalog[sid]["queries_matched"]:
                        catalog[sid]["queries_matched"].append(query)
                    continue

                preview_url = (r.get("previews") or {}).get("preview-hq-mp3")
                if not preview_url:
                    continue

                catalog[sid] = {
                    "id": sid,
                    "name": r.get("name", ""),
                    "description": r.get("description", ""),
                    "tags": r.get("tags", []) or [],
                    "duration": float(r.get("duration", 0.0) or 0.0),
                    "queries_matched": [query],
                    "freesound_url": f"https://freesound.org/s/{sid}/",
                    "license": "Creative Commons 0",
                    "username": r.get("username", ""),
                    "is_music": True,
                    "_preview_url": preview_url,
                    "_avg_rating": float(r.get("avg_rating", 0.0) or 0.0),
                    "_num_downloads": int(r.get("num_downloads", 0) or 0),
                }

    return catalog


def _truncate_to_target(
    catalog: dict[int, dict[str, Any]], target_count: int,
) -> dict[int, dict[str, Any]]:
    if not target_count or len(catalog) <= target_count:
        return catalog
    ranked = sorted(
        catalog.values(),
        key=lambda x: (x["_avg_rating"], x["_num_downloads"]),
        reverse=True,
    )[:target_count]
    logger.info("Truncated music catalog from %d to top %d", len(catalog), target_count)
    return {item["id"]: item for item in ranked}


def _download_all(catalog: dict[int, dict[str, Any]]) -> tuple[int, int, int]:
    new_count = 0
    skip_count = 0
    fail_count = 0

    def _task(meta: dict[str, Any]) -> tuple[int, bool]:
        local = MUSIC_DIR / f"{meta['id']}.mp3"
        was_new = _download_preview(meta["_preview_url"], local)
        return meta["id"], was_new

    with ThreadPoolExecutor(max_workers=_MAX_DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(_task, meta): meta for meta in catalog.values()}
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Music", unit="track",
        ):
            meta = futures[fut]
            try:
                _, was_new = fut.result()
                if was_new:
                    new_count += 1
                else:
                    skip_count += 1
            except Exception as exc:
                logger.error("Failed to download music %d: %s", meta["id"], exc)
                fail_count += 1

    return new_count, skip_count, fail_count


def _write_manifest(catalog: dict[int, dict[str, Any]]) -> tuple[Path, int, int]:
    entries: list[dict[str, Any]] = []
    total_size = 0
    for meta in catalog.values():
        local = MUSIC_DIR / f"{meta['id']}.mp3"
        if not local.is_file() or local.stat().st_size == 0:
            continue
        total_size += local.stat().st_size
        entry = {k: v for k, v in meta.items() if not k.startswith("_")}
        entry["local_path"] = f"data/library/music/{meta['id']}.mp3"
        entries.append(entry)

    manifest_path = MUSIC_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(entries, indent=2))
    return manifest_path, len(entries), total_size


def download_music_library(
    target_count: int = 1500,
    queries: list[str] | None = None,
) -> Path:
    """Download CC0 instrumental music tracks from Freesound into data/library/music/."""
    queries = list(queries) if queries is not None else list(MUSIC_QUERIES)
    MUSIC_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Phase 1: searching Freesound for music across %d queries", len(queries))
    catalog = _collect_candidates(queries)
    logger.info("Collected %d unique candidate tracks", len(catalog))

    catalog = _truncate_to_target(catalog, target_count)

    logger.info("Phase 2: downloading %d preview MP3s", len(catalog))
    new_count, skip_count, fail_count = _download_all(catalog)
    logger.info(
        "Music downloaded %d/%d, skipped %d, failed %d",
        new_count, len(catalog), skip_count, fail_count,
    )

    manifest_path, manifest_count, total_size = _write_manifest(catalog)
    size_mb = total_size / (1024 * 1024)
    logger.info(
        "Music library complete: %d tracks, total size %.1f MB, manifest saved to %s",
        manifest_count, size_mb, manifest_path,
    )
    return manifest_path


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a curated CC0 instrumental music library from Freesound.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=1500,
        help="Maximum number of unique music tracks to keep.",
    )
    parser.add_argument(
        "--limit-queries",
        type=int,
        default=0,
        help="If > 0, only run the first N music queries (useful for smoke tests).",
    )
    args = parser.parse_args()

    queries = (
        MUSIC_QUERIES if args.limit_queries <= 0 else MUSIC_QUERIES[: args.limit_queries]
    )
    manifest_path = download_music_library(target_count=args.target_count, queries=queries)
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    _main()
