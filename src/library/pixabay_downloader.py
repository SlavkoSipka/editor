"""Pixabay sound-effects downloader.

Scrapes Pixabay's PUBLIC sound-effects search pages and downloads MP3 previews
into ``data/library/pixabay/`` together with a per-source manifest.

License notes
-------------
Pixabay's content license (https://pixabay.com/service/license-summary/) allows
commercial use without attribution, including downloading audio files. This
module is a batched version of what you'd do clicking "download" manually. It
does NOT bypass paywalls, scrape personal data, or scrape any private endpoint.
If Pixabay's HTML changes and scraping breaks, we log a clear warning and
return what we collected; the rest of the pipeline keeps working.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.config import DATA_DIR, LIBRARY_DIR
from src.utils.logger import get_logger

logger = get_logger("pixabay_downloader")

PIXABAY_TARGET_DIR: Path = LIBRARY_DIR / "pixabay"
_PIXABAY_BASE = "https://pixabay.com"
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

PIXABAY_QUERIES: list[str] = [
    # Viral / meme sounds (the bread and butter)
    "vine boom", "boom meme", "bruh", "discord notification",
    "iphone notification", "iphone keyboard", "iphone camera",
    "tiktok transition", "tiktok pop", "tiktok beat drop",
    "air horn", "sad violin", "ironic violin", "ave maria",
    "anime hit", "anime sting", "anime wow",
    "tssk tssk", "click pop", "ear rape",
    "movie tense violin", "horror sting", "jumpscare",

    # UGC editing essentials
    "swoosh transition", "pop transition", "zoom transition",
    "punch in", "snap zoom", "vlog whoosh",
    "subscribe sound", "like button", "bell notification",
    "reveal pop", "magic reveal", "sparkle pop",
    "freeze frame sting", "record scratch", "vinyl scratch",

    # Comedic / playful
    "boing", "cartoon hit", "cartoon slip", "cartoon fall",
    "comedic", "funny sound", "silly sound",
    "duck quack", "rubber ducky", "squeak",
    "wow kid", "boo crowd", "applause clap",

    # Drops / hits
    "beat drop", "bass drop", "trap drop", "808 hit",
    "kick punch", "snare clap", "vocal chop",
    "riser drop", "drum fill",

    # Modern UI / app
    "app notification", "message ping", "swipe phone",
    "phone unlock", "phone lock", "camera shutter",
    "scan beep", "qr scan", "barcode beep",

    # Money / business / luxury
    "cash register", "coin drop", "money sound",
    "champagne pop", "elegant chime", "luxury sting",

    # Food / lifestyle (huge UGC category)
    "food sizzle", "drink pour", "ice clink",
    "bottle open", "can open", "pop top",

    # Body / interaction
    "head turn whoosh", "punch face", "slap face",
    "thud body", "heartbeat fast",

    # Tech / digital
    "computer error", "loading bar", "level up",
    "achievement sound", "coin pickup game",
    "8 bit", "retro game", "arcade",
]


def _stable_id(text: str) -> int:
    # Deterministic, fits in a positive 31-bit-ish range to play nice with Qdrant.
    import hashlib
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:6], "big") % (10**9)


def _query_tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.split(text) if len(t) >= 2]


def _name_from_url(mp3_url: str, fallback: str) -> str:
    try:
        last = mp3_url.split("/")[-1].split("?")[0]
        stem = last.rsplit(".", 1)[0]
        stem = re.sub(r"[-_]+", " ", stem).strip()
        if stem:
            return stem
    except Exception:
        pass
    return fallback


def search_pixabay(query: str, max_pages: int = 2) -> list[dict[str, Any]]:
    """Scrape Pixabay search results pages. Returns list of metadata dicts.

    Each dict contains: ``name``, ``mp3_url``, ``page_url``, ``query_matched``.
    Best-effort: if the page can't be parsed, returns whatever was collected.
    """
    encoded = query.replace(" ", "%20")
    results: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for page in range(1, max_pages + 1):
        page_url = f"{_PIXABAY_BASE}/sound-effects/search/{encoded}/?pagi={page}"
        try:
            resp = requests.get(page_url, headers=_HEADERS, timeout=15)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("Pixabay search failed for %r page %d: %s", query, page, exc)
            break

        page_results: list[dict[str, Any]] = []
        soup = BeautifulSoup(resp.text, "html.parser")

        for audio in soup.find_all("audio"):
            src = (audio.get("src") or "").strip()
            if not src or ".mp3" not in src:
                continue
            url = src if src.startswith("http") else f"{_PIXABAY_BASE}{src}"
            if url in seen_urls:
                continue
            seen_urls.add(url)
            parent = audio.find_parent()
            name = (
                audio.get("data-name")
                or (parent.get("data-title") if parent else None)
                or _name_from_url(url, query)
            )
            page_results.append({
                "name": str(name),
                "mp3_url": url,
                "page_url": page_url,
                "query_matched": query,
            })

        for script in soup.find_all("script"):
            blob = script.string or ""
            if ".mp3" not in blob:
                continue
            for url in _MP3_URL_RE.findall(blob):
                if "pixabay" not in url:
                    continue
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                page_results.append({
                    "name": _name_from_url(url, query),
                    "mp3_url": url,
                    "page_url": page_url,
                    "query_matched": query,
                })

        logger.info(
            "Pixabay %r page %d → %d new sounds (running total %d)",
            query, page, len(page_results), len(results) + len(page_results),
        )

        if not page_results:
            break
        results.extend(page_results)
        time.sleep(_PAGE_DELAY_SEC)

    return results


def _entry_from_scrape(item: dict[str, Any]) -> dict[str, Any]:
    url = str(item["mp3_url"])
    sid = _stable_id(url)
    query = str(item.get("query_matched", ""))
    name = str(item.get("name") or _name_from_url(url, query) or "pixabay sound")
    tags = list(dict.fromkeys(_query_tokens(query) + _query_tokens(name)))
    return {
        "id": sid,
        "name": name,
        "description": f"{query} (pixabay)".strip(),
        "tags": tags,
        "duration": None,
        "queries_matched": [query] if query else [],
        "source": "pixabay",
        "license": "Pixabay License",
        "local_path": f"data/library/pixabay/{sid}.mp3",
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
        local = PIXABAY_TARGET_DIR / f"{meta['id']}.mp3"
        was_new = _download_mp3(meta["_mp3_url"], local)
        return meta["id"], was_new

    with ThreadPoolExecutor(max_workers=_MAX_DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(_task, m): m for m in catalog.values()}
        for fut in tqdm(
            as_completed(futures), total=len(futures),
            desc="Pixabay", unit="snd",
        ):
            meta = futures[fut]
            try:
                _, was_new = fut.result()
                if was_new:
                    new_count += 1
                else:
                    skip_count += 1
            except Exception as exc:
                logger.error("Failed to download pixabay %s: %s", meta.get("_mp3_url"), exc)
                fail_count += 1

    return new_count, skip_count, fail_count


def _merge_query_metadata(
    catalog: dict[int, dict[str, Any]],
    entry: dict[str, Any],
) -> None:
    existing = catalog.get(entry["id"])
    if existing is None:
        catalog[entry["id"]] = entry
        return
    for q in entry.get("queries_matched", []):
        if q and q not in existing["queries_matched"]:
            existing["queries_matched"].append(q)
    merged_tags = list(dict.fromkeys(existing.get("tags", []) + entry.get("tags", [])))
    existing["tags"] = merged_tags


def _write_manifest(catalog: dict[int, dict[str, Any]]) -> tuple[Path, int, int]:
    entries: list[dict[str, Any]] = []
    total_size = 0
    for meta in catalog.values():
        local = PIXABAY_TARGET_DIR / f"{meta['id']}.mp3"
        if not local.is_file() or local.stat().st_size == 0:
            continue
        total_size += local.stat().st_size
        entries.append({k: v for k, v in meta.items() if not k.startswith("_")})

    manifest_path = PIXABAY_TARGET_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(entries, indent=2))
    return manifest_path, len(entries), total_size


def download_pixabay_library(
    queries: list[str] | None = None,
    max_pages: int = 2,
) -> Path:
    """Scrape all queries, download MP3s, build the Pixabay manifest."""
    queries = list(queries) if queries is not None else list(PIXABAY_QUERIES)
    PIXABAY_TARGET_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Phase 1: scraping Pixabay across %d queries (up to %d pages each)",
        len(queries), max_pages,
    )

    catalog: dict[int, dict[str, Any]] = {}
    for query in tqdm(queries, desc="Pixabay queries", unit="q"):
        try:
            scraped = search_pixabay(query, max_pages=max_pages)
        except Exception as exc:
            logger.warning("Unexpected error scraping %r: %s", query, exc)
            continue
        for item in scraped:
            _merge_query_metadata(catalog, _entry_from_scrape(item))

    if not catalog:
        logger.warning(
            "Pixabay scraping returned no results. Their HTML may have changed. "
            "Skipping Pixabay."
        )
        manifest_path = PIXABAY_TARGET_DIR / "manifest.json"
        if not manifest_path.is_file():
            manifest_path.write_text("[]")
        return manifest_path

    logger.info("Collected %d unique Pixabay candidates", len(catalog))

    logger.info("Phase 2: downloading %d MP3s", len(catalog))
    new_count, skip_count, fail_count = _download_all(catalog)
    logger.info(
        "Pixabay downloaded %d/%d, skipped %d (already existed), failed %d",
        new_count, len(catalog), skip_count, fail_count,
    )

    manifest_path, manifest_count, total_size = _write_manifest(catalog)
    size_mb = total_size / (1024 * 1024)
    logger.info(
        "Pixabay library complete: %d sounds, %.1f MB, manifest saved to %s",
        manifest_count, size_mb, manifest_path,
    )
    return manifest_path


def import_from_url_list(csv_path: Path) -> int:
    """Fallback for power users: import a CSV with columns ``url,name,tags``.

    ``tags`` is a ``|``-separated string. Skips rows whose URL doesn't end in
    ``.mp3`` or which can't be downloaded. Returns count of new MP3s written.
    """
    if not csv_path.is_file():
        logger.warning("CSV not found: %s", csv_path)
        return 0

    PIXABAY_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    catalog: dict[int, dict[str, Any]] = {}

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = (row.get("url") or "").strip()
            if not url or ".mp3" not in url:
                continue
            name = (row.get("name") or _name_from_url(url, "pixabay")).strip()
            tags_raw = (row.get("tags") or "").strip()
            tags = [t.strip().lower() for t in tags_raw.split("|") if t.strip()]
            sid = _stable_id(url)
            catalog[sid] = {
                "id": sid,
                "name": name,
                "description": f"{name} (manual import)",
                "tags": tags or _query_tokens(name),
                "duration": None,
                "queries_matched": ["manual"],
                "source": "pixabay",
                "license": "Pixabay License",
                "local_path": f"data/library/pixabay/{sid}.mp3",
                "page_url": None,
                "_mp3_url": url,
            }

    if not catalog:
        logger.info("No usable rows in %s", csv_path)
        return 0

    logger.info("Importing %d sounds from %s", len(catalog), csv_path)
    new_count, _, fail_count = _download_all(catalog)

    existing: list[dict[str, Any]] = []
    manifest_path = PIXABAY_TARGET_DIR / "manifest.json"
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            existing = []
    by_id: dict[int, dict[str, Any]] = {e["id"]: e for e in existing if isinstance(e, dict)}
    for meta in catalog.values():
        local = PIXABAY_TARGET_DIR / f"{meta['id']}.mp3"
        if not local.is_file() or local.stat().st_size == 0:
            continue
        by_id[meta["id"]] = {k: v for k, v in meta.items() if not k.startswith("_")}
    manifest_path.write_text(json.dumps(list(by_id.values()), indent=2))

    logger.info(
        "Manual import: downloaded %d new, failed %d; manifest has %d entries",
        new_count, fail_count, len(by_id),
    )
    return new_count


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Pixabay sound effects into the local CC0 library.",
    )
    parser.add_argument(
        "--max-pages", type=int, default=2,
        help="Result pages to scrape per query.",
    )
    parser.add_argument(
        "--limit-queries", type=int, default=0,
        help="If > 0, only run the first N queries (smoke test).",
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Optional CSV (url,name,tags) for manual imports; bypasses scraping.",
    )
    args = parser.parse_args()

    if args.csv is not None:
        csv_path = args.csv if args.csv.is_absolute() else (DATA_DIR / args.csv)
        count = import_from_url_list(csv_path)
        print(f"Imported {count} sounds from {csv_path}")
        return

    queries = (
        PIXABAY_QUERIES if args.limit_queries <= 0
        else PIXABAY_QUERIES[: args.limit_queries]
    )
    manifest_path = download_pixabay_library(
        queries=queries, max_pages=max(1, args.max_pages),
    )
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    _main()
