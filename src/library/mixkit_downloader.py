"""Mixkit sound-effects downloader.

Scrapes Mixkit's PUBLIC free sound-effect category pages and downloads MP3s
into ``data/library/mixkit/`` together with a per-source manifest.

License notes
-------------
Mixkit's free license (https://mixkit.co/license/) allows commercial use of
their free sound effects without attribution. This module is a batched version
of what you'd do clicking "free download" manually. We do NOT bypass any
paywalls, scrape any personal data, or scrape private endpoints. If Mixkit's
HTML changes and scraping breaks, we log a clear warning and return whatever
was collected; the rest of the pipeline keeps working.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.config import LIBRARY_DIR
from src.utils.logger import get_logger

logger = get_logger("mixkit_downloader")

MIXKIT_TARGET_DIR: Path = LIBRARY_DIR / "mixkit"
_MIXKIT_BASE = "https://mixkit.co"
_PAGE_DELAY_SEC = 2.0
_DOWNLOAD_TIMEOUT_SEC = 60
_MAX_DOWNLOAD_WORKERS = 4
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}
_TOKEN_RE = re.compile(r"[^A-Za-z0-9]+")
_MP3_URL_RE = re.compile(r"https?://[^\s\"'\\]+?\.mp3(?:\?[^\s\"'\\]*)?")

MIXKIT_CATEGORIES: list[str] = [
    "transitions", "notification", "click", "swoosh", "whoosh",
    "pop", "cartoon", "game-sound", "cinematic", "impact",
    "horror", "footstep", "percussion", "comic", "glitch",
    "explosion", "achievement", "alert", "ambience",
    "applause", "bell", "button", "camera", "cellphone",
    "computer", "crowd", "drone", "logo", "magic",
    "movie", "musical-stinger", "scratch", "ui",
]


def _stable_id(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:6], "big") % (10**9)


def _name_tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.split(text) if len(t) >= 2]


def _name_from_url(mp3_url: str, fallback: str) -> str:
    last = mp3_url.split("/")[-1].split("?")[0]
    stem = last.rsplit(".", 1)[0]
    stem = re.sub(r"[-_]+", " ", stem).strip()
    return stem or fallback


def search_mixkit_category(category: str) -> list[dict[str, Any]]:
    """Scrape a Mixkit category page. Best-effort; returns [] if HTML breaks."""
    page_url = f"{_MIXKIT_BASE}/free-sound-effects/{category}/"
    try:
        resp = requests.get(page_url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Mixkit category %r failed: %s", category, exc)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for audio in soup.find_all("audio"):
        for attr in ("src", "data-audio-url"):
            src = (audio.get(attr) or "").strip()
            if src and ".mp3" in src:
                url = src if src.startswith("http") else f"{_MIXKIT_BASE}{src}"
                if url in seen:
                    continue
                seen.add(url)
                results.append({
                    "name": _name_from_url(url, category.replace("-", " ")),
                    "mp3_url": url,
                    "page_url": page_url,
                    "category": category,
                })

    for source in soup.find_all("source"):
        src = (source.get("src") or "").strip()
        if src and ".mp3" in src and src not in seen:
            url = src if src.startswith("http") else f"{_MIXKIT_BASE}{src}"
            seen.add(url)
            results.append({
                "name": _name_from_url(url, category.replace("-", " ")),
                "mp3_url": url,
                "page_url": page_url,
                "category": category,
            })

    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip()
        if href.endswith(".mp3") and href not in seen:
            url = href if href.startswith("http") else f"{_MIXKIT_BASE}{href}"
            if url in seen:
                continue
            seen.add(url)
            results.append({
                "name": _name_from_url(url, category.replace("-", " ")),
                "mp3_url": url,
                "page_url": page_url,
                "category": category,
            })

    for script in soup.find_all("script"):
        blob = script.string or ""
        if ".mp3" not in blob:
            continue
        for url in _MP3_URL_RE.findall(blob):
            if "mixkit" not in url or url in seen:
                continue
            seen.add(url)
            results.append({
                "name": _name_from_url(url, category.replace("-", " ")),
                "mp3_url": url,
                "page_url": page_url,
                "category": category,
            })

    logger.info("Mixkit %s → %d sounds", category, len(results))
    return results


def _entry_from_scrape(item: dict[str, Any]) -> dict[str, Any]:
    url = str(item["mp3_url"])
    sid = _stable_id(url)
    category = str(item.get("category", ""))
    name = str(item.get("name") or _name_from_url(url, category))
    tags = list(dict.fromkeys(_name_tokens(category) + _name_tokens(name)))
    return {
        "id": sid,
        "name": name,
        "description": f"mixkit {category.replace('-', ' ')}".strip(),
        "tags": tags,
        "duration": None,
        "queries_matched": [category] if category else [],
        "source": "mixkit",
        "license": "Mixkit Free License",
        "local_path": f"data/library/mixkit/{sid}.mp3",
        "page_url": item.get("page_url"),
        "_mp3_url": url,
    }


def _download_mp3(url: str, dest: Path) -> bool:
    if dest.is_file() and dest.stat().st_size > 0:
        return False
    tmp = dest.parent / (dest.name + ".tmp")
    with requests.get(
        url, headers=_HEADERS, stream=True, timeout=_DOWNLOAD_TIMEOUT_SEC,
    ) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)
    return True


def _download_all(
    catalog: dict[int, dict[str, Any]],
) -> tuple[int, int, int]:
    new_count = 0
    skip_count = 0
    fail_count = 0

    def _task(meta: dict[str, Any]) -> tuple[int, bool]:
        local = MIXKIT_TARGET_DIR / f"{meta['id']}.mp3"
        was_new = _download_mp3(meta["_mp3_url"], local)
        return meta["id"], was_new

    with ThreadPoolExecutor(max_workers=_MAX_DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(_task, m): m for m in catalog.values()}
        for fut in tqdm(
            as_completed(futures), total=len(futures),
            desc="Mixkit", unit="snd",
        ):
            meta = futures[fut]
            try:
                _, was_new = fut.result()
                if was_new:
                    new_count += 1
                else:
                    skip_count += 1
            except Exception as exc:
                logger.error("Failed to download mixkit %s: %s", meta.get("_mp3_url"), exc)
                fail_count += 1

    return new_count, skip_count, fail_count


def _merge_category_metadata(
    catalog: dict[int, dict[str, Any]],
    entry: dict[str, Any],
) -> None:
    existing = catalog.get(entry["id"])
    if existing is None:
        catalog[entry["id"]] = entry
        return
    for c in entry.get("queries_matched", []):
        if c and c not in existing["queries_matched"]:
            existing["queries_matched"].append(c)
    existing["tags"] = list(dict.fromkeys(existing.get("tags", []) + entry.get("tags", [])))


def _write_manifest(catalog: dict[int, dict[str, Any]]) -> tuple[Path, int, int]:
    entries: list[dict[str, Any]] = []
    total_size = 0
    for meta in catalog.values():
        local = MIXKIT_TARGET_DIR / f"{meta['id']}.mp3"
        if not local.is_file() or local.stat().st_size == 0:
            continue
        total_size += local.stat().st_size
        entries.append({k: v for k, v in meta.items() if not k.startswith("_")})

    manifest_path = MIXKIT_TARGET_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(entries, indent=2))
    return manifest_path, len(entries), total_size


def download_mixkit_library(
    categories: list[str] | None = None,
) -> Path:
    """Scrape all Mixkit categories, download MP3s, build the Mixkit manifest."""
    categories = list(categories) if categories is not None else list(MIXKIT_CATEGORIES)
    MIXKIT_TARGET_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Phase 1: scraping Mixkit across %d categories", len(categories))

    catalog: dict[int, dict[str, Any]] = {}
    for category in tqdm(categories, desc="Mixkit categories", unit="cat"):
        try:
            scraped = search_mixkit_category(category)
        except Exception as exc:
            logger.warning("Unexpected error scraping %r: %s", category, exc)
            scraped = []
        for item in scraped:
            _merge_category_metadata(catalog, _entry_from_scrape(item))
        time.sleep(_PAGE_DELAY_SEC)

    if not catalog:
        logger.warning(
            "Mixkit scraping returned no results. Their HTML may have changed. "
            "Skipping Mixkit."
        )
        manifest_path = MIXKIT_TARGET_DIR / "manifest.json"
        if not manifest_path.is_file():
            manifest_path.write_text("[]")
        return manifest_path

    logger.info("Collected %d unique Mixkit candidates", len(catalog))

    logger.info("Phase 2: downloading %d MP3s", len(catalog))
    new_count, skip_count, fail_count = _download_all(catalog)
    logger.info(
        "Mixkit downloaded %d/%d, skipped %d (already existed), failed %d",
        new_count, len(catalog), skip_count, fail_count,
    )

    manifest_path, manifest_count, total_size = _write_manifest(catalog)
    size_mb = total_size / (1024 * 1024)
    logger.info(
        "Mixkit library complete: %d sounds, %.1f MB, manifest saved to %s",
        manifest_count, size_mb, manifest_path,
    )
    return manifest_path


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Mixkit free sound effects into the local library.",
    )
    parser.add_argument(
        "--limit-categories", type=int, default=0,
        help="If > 0, only scrape the first N categories (smoke test).",
    )
    args = parser.parse_args()

    cats = (
        MIXKIT_CATEGORIES if args.limit_categories <= 0
        else MIXKIT_CATEGORIES[: args.limit_categories]
    )
    manifest_path = download_mixkit_library(categories=cats)
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    _main()
