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

logger = get_logger("downloader")

SEARCH_QUERIES: list[str] = [
    # === IMPACTS / HITS / BOOMS ===
    "impact", "hit", "thud", "bass drop", "boom", "low boom", "deep impact",
    "cinematic impact", "trailer hit", "epic impact", "sub drop",
    "punch impact", "metal hit", "wood impact", "stone impact",
    "explosion impact", "drum hit", "kick drum", "snare hit",

    # === WHOOSHES / TRANSITIONS ===
    "whoosh", "swoosh", "swipe", "sweep", "transition",
    "cinematic whoosh", "trailer whoosh", "fast whoosh", "slow whoosh",
    "high whoosh", "low whoosh", "deep whoosh", "air whoosh",
    "metal whoosh", "wind whoosh", "magical whoosh",
    "video transition", "scene transition", "camera whoosh",

    # === RISERS / SWELLS / BUILD-UPS ===
    "riser", "swell", "build up", "tension riser",
    "cinematic riser", "trailer riser", "tonal riser",
    "noise riser", "synth riser", "uplifter", "downlifter",
    "reverse swell", "reverse cymbal",

    # === LOGO STINGS / REVEALS ===
    "logo sting", "logo reveal", "logo intro", "brand sting",
    "stinger", "ident", "intro sting", "outro sting",
    "shimmer reveal", "sparkle reveal", "magical reveal",
    "corporate logo", "epic logo", "cinematic logo",

    # === GLITCH / DIGITAL ===
    "glitch", "digital glitch", "data glitch", "static",
    "digital error", "digital noise", "bit crush", "static noise",
    "digital transition", "tech glitch", "circuit", "interference",
    "data corruption", "vhs glitch", "scan line", "digital riser",

    # === UI / CLICKS / TAPS ===
    "click", "tap", "button", "ui", "interface",
    "notification", "alert", "beep", "blip", "pop",
    "ui hit", "ui select", "ui confirm", "ui error",
    "menu click", "select click", "confirm click",
    "touchscreen", "screen tap", "swipe ui", "tech click",
    "futuristic ui", "sci fi ui", "hud", "computer ui",

    # === TECH / SCI-FI ===
    "sci fi", "futuristic", "robotic", "mechanical",
    "tech hit", "tech impact", "tech transition",
    "laser", "energy", "spark", "electric zap",
    "robot beep", "computer beep", "data transfer",

    # === FOOTSTEPS ===
    "footstep", "footsteps", "footsteps wood", "footsteps concrete",
    "footsteps grass", "footsteps gravel", "footsteps metal",
    "footsteps stairs", "footsteps snow", "footsteps dirt",
    "boot footstep", "shoe footstep", "running footsteps",
    "single footstep", "heavy footstep", "soft footstep",

    # === DOORS ===
    "door open", "door close", "door slam", "door creak",
    "door knock", "door handle", "door lock", "door bell",
    "wooden door", "metal door", "sliding door", "garage door",

    # === AMBIENCE / ROOM TONE ===
    "city ambience", "office ambience", "nature ambience",
    "wind", "rain", "forest ambience", "ocean ambience",
    "indoor ambience", "outdoor ambience", "room tone",
    "cafe ambience", "restaurant ambience", "park ambience",
    "subway ambience", "street ambience", "crowd ambience",
    "library ambience", "warehouse ambience", "factory ambience",
    "studio ambience", "quiet room", "empty room",

    # === MECHANICAL / INDUSTRIAL ===
    "machine", "motor", "engine", "gear",
    "factory machine", "industrial machine", "machine hum",
    "sewing machine", "drill", "saw", "hammer",
    "machine click", "machine clank", "mechanical click",
    "latch", "buckle", "snap mechanism", "metal mechanism",

    # === HUMAN / BODY ===
    "breathing", "heartbeat", "footstep run", "jump",
    "clap", "snap fingers", "applause", "crowd cheer",
    "voice whisper", "voice shout", "voice gasp",

    # === CINEMATIC / DRAMATIC ===
    "cinematic hit", "cinematic boom", "cinematic stinger",
    "drone tense", "drone dark", "drone deep", "drone evolving",
    "tension drone", "horror drone", "ambient pad",
    "orchestra hit", "epic stinger", "movie trailer hit",

    # === GLASS / METAL / OBJECTS ===
    "glass break", "glass shatter", "glass clink",
    "metal clang", "metal scrape", "metal drop", "metal rattle",
    "paper", "paper crumple", "paper turn",
    "fabric", "cloth rustle", "leather",
    "water drop", "water splash", "water pour",
    "ice", "fire crackle", "match strike",

    # === TOOLS / CRAFT ===
    "scissors", "scissors cut", "snip",
    "knife cut", "blade", "cutter",
    "stapler", "tape rip", "pen click",
    "keyboard", "keyboard typing", "key press",
    "mouse click", "scroll wheel",

    # === WHOOSH SUB-CATEGORIES (for variety) ===
    "stylized whoosh", "designed whoosh", "processed whoosh",
    "whoosh up", "whoosh down", "whoosh by",
    "whoosh in", "whoosh out", "whoosh reverse",

    # === MUSIC HITS / ORCHESTRAL ===
    "orchestral hit", "brass hit", "string hit",
    "piano hit", "synth hit", "bell hit",
    "chime", "shimmer", "magical shimmer", "sparkle",
]

_DOWNLOAD_TIMEOUT_SEC = 60
_MAX_DOWNLOAD_WORKERS = 5
_PAGE_SIZE = 150
_MIN_AVG_RATING = 3.0
_MIN_NUM_DOWNLOADS = 50


def _download_preview(url: str, dest: Path) -> bool:
    """Download a preview MP3 to `dest`. Returns True if newly downloaded, False if it already existed."""
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


def _passes_quality_filter(avg_rating: float, num_downloads: int) -> bool:
    # Lenient on purpose: we want category breadth, and a sound that's been
    # downloaded a lot is at least useful even if unrated.
    return avg_rating >= _MIN_AVG_RATING or num_downloads >= _MIN_NUM_DOWNLOADS


def _collect_candidates(
    queries: list[str],
    pages_per_query: int = 3,
) -> dict[int, dict[str, Any]]:
    catalog: dict[int, dict[str, Any]] = {}

    for query in queries:
        for page in range(1, pages_per_query + 1):
            try:
                response = search_sounds(
                    query, page=page, page_size=_PAGE_SIZE, cc0_only=True,
                )
            except RuntimeError as exc:
                logger.error("Search failed for %r page %d: %s", query, page, exc)
                break

            results = response.get("results", []) or []
            logger.info(
                "Searching for: %r page %d (found %d results, catalog=%d)",
                query, page, len(results), len(catalog),
            )

            if not results:
                break

            for r in results:
                sid = r.get("id")
                if sid is None:
                    continue
                if sid in catalog:
                    if query not in catalog[sid]["queries_matched"]:
                        catalog[sid]["queries_matched"].append(query)
                    continue

                avg_rating = float(r.get("avg_rating", 0.0) or 0.0)
                num_downloads = int(r.get("num_downloads", 0) or 0)
                if not _passes_quality_filter(avg_rating, num_downloads):
                    continue

                preview_url = (r.get("previews") or {}).get("preview-hq-mp3")
                if not preview_url:
                    logger.warning("Sound %d has no preview-hq-mp3 URL; skipping", sid)
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
                    "source": "freesound",
                    "_preview_url": preview_url,
                    "_avg_rating": avg_rating,
                    "_num_downloads": num_downloads,
                }

            # Last page was partial → no point asking for the next one.
            if len(results) < _PAGE_SIZE:
                break

    return catalog


def _truncate_to_target(
    catalog: dict[int, dict[str, Any]],
    target_count: int,
) -> dict[int, dict[str, Any]]:
    if not target_count or len(catalog) <= target_count:
        return catalog
    ranked = sorted(
        catalog.values(),
        key=lambda x: (x["_avg_rating"], x["_num_downloads"]),
        reverse=True,
    )[:target_count]
    logger.info("Truncated catalog from %d to top %d by rating", len(catalog), target_count)
    return {item["id"]: item for item in ranked}


def _download_all(catalog: dict[int, dict[str, Any]]) -> tuple[int, int, int]:
    new_count = 0
    skip_count = 0
    fail_count = 0

    def _task(meta: dict[str, Any]) -> tuple[int, bool]:
        local = LIBRARY_DIR / f"{meta['id']}.mp3"
        was_new = _download_preview(meta["_preview_url"], local)
        return meta["id"], was_new

    with ThreadPoolExecutor(max_workers=_MAX_DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(_task, meta): meta for meta in catalog.values()}
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Downloading", unit="snd",
        ):
            meta = futures[fut]
            try:
                _, was_new = fut.result()
                if was_new:
                    new_count += 1
                else:
                    skip_count += 1
            except Exception as exc:
                logger.error("Failed to download %d: %s", meta["id"], exc)
                fail_count += 1

    return new_count, skip_count, fail_count


def _load_existing_non_freesound_entries() -> list[dict[str, Any]]:
    """Preserve manifest entries from other sources (e.g. Sonniss) on re-run."""
    manifest_path = LIBRARY_DIR / "manifest.json"
    if not manifest_path.is_file():
        return []
    try:
        existing = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return []
    return [
        entry for entry in existing
        if isinstance(entry, dict) and entry.get("source") and entry["source"] != "freesound"
    ]


def _write_manifest(catalog: dict[int, dict[str, Any]]) -> tuple[Path, int, int]:
    manifest_entries: list[dict[str, Any]] = list(_load_existing_non_freesound_entries())
    total_size = 0
    for meta in catalog.values():
        local = LIBRARY_DIR / f"{meta['id']}.mp3"
        if not local.is_file() or local.stat().st_size == 0:
            continue
        total_size += local.stat().st_size
        entry = {k: v for k, v in meta.items() if not k.startswith("_")}
        entry["local_path"] = f"data/library/{meta['id']}.mp3"
        manifest_entries.append(entry)

    manifest_path = LIBRARY_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_entries, indent=2))
    return manifest_path, len(manifest_entries), total_size


def download_library(
    target_count: int = 40000,
    queries: list[str] | None = None,
    pages_per_query: int = 3,
) -> Path:
    """Download CC0 sounds from Freesound covering common ad SFX categories.
    Returns the path to manifest.json with all sound metadata."""
    queries = list(queries) if queries is not None else list(SEARCH_QUERIES)
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Phase 1: searching Freesound across %d queries (up to %d pages each)",
        len(queries), pages_per_query,
    )
    catalog = _collect_candidates(queries, pages_per_query=pages_per_query)
    logger.info(
        "Collected %d unique candidate sounds across %d queries",
        len(catalog), len(queries),
    )

    catalog = _truncate_to_target(catalog, target_count)

    logger.info("Phase 2: downloading %d preview MP3s", len(catalog))
    new_count, skip_count, fail_count = _download_all(catalog)
    logger.info(
        "Downloaded %d/%d, skipped %d (already existed), failed %d",
        new_count, len(catalog), skip_count, fail_count,
    )

    manifest_path, manifest_count, total_size = _write_manifest(catalog)
    size_mb = total_size / (1024 * 1024)
    logger.info(
        "Library complete: %d sounds, total size %.1f MB, manifest saved to %s",
        manifest_count, size_mb, manifest_path,
    )
    return manifest_path


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a curated CC0 SFX library from Freesound.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=40000,
        help="Maximum number of unique sounds to download.",
    )
    parser.add_argument(
        "--limit-queries",
        type=int,
        default=0,
        help="If > 0, only run the first N search queries (useful for smoke tests).",
    )
    parser.add_argument(
        "--pages-per-query",
        type=int,
        default=3,
        help="Result pages (150 sounds each) to pull per query. Lower for quick re-tests.",
    )
    args = parser.parse_args()

    queries = (
        SEARCH_QUERIES if args.limit_queries <= 0 else SEARCH_QUERIES[: args.limit_queries]
    )
    manifest_path = download_library(
        target_count=args.target_count,
        queries=queries,
        pages_per_query=max(1, args.pages_per_query),
    )
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    _main()
