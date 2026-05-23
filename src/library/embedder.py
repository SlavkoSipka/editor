from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from src.config import EMBEDDINGS_DIR, LIBRARY_DIR
from src.library.tier_classifier import classify_tiers, tier_distribution
from src.utils.logger import get_logger

logger = get_logger("embedder")

_MODEL_NAME = "all-MiniLM-L6-v2"
_MAX_TEXT_CHARS = 500
_BATCH_SIZE = 64
_LOG_EVERY = 2500
_UGC_SOURCES: frozenset[str] = frozenset({"pixabay", "mixkit"})

MANIFEST_PATHS: list[Path] = [
    LIBRARY_DIR / "manifest.json",
    LIBRARY_DIR / "sonniss" / "manifest.json",
    LIBRARY_DIR / "pixabay" / "manifest.json",
    LIBRARY_DIR / "mixkit" / "manifest.json",
]

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading sentence-transformer model: %s", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def build_embedding_text(sound_metadata: dict[str, Any]) -> str:
    name = str(sound_metadata.get("name", "") or "")
    description = str(sound_metadata.get("description", "") or "")
    tags = sound_metadata.get("tags", []) or []
    tags_str = ", ".join(str(t) for t in tags)
    source = str(sound_metadata.get("source", "") or "")
    ugc_marker = " [ugc viral]" if source in _UGC_SOURCES else ""
    parts = [name, description, f"Tags: {tags_str}.{ugc_marker}"]
    if source:
        parts.append(f"Source: {source}.")
    text = ". ".join(p for p in parts if p)
    text = " ".join(text.lower().split())
    return text[:_MAX_TEXT_CHARS]


def load_all_manifests(
    paths: list[Path] | None = None,
) -> list[dict[str, Any]]:
    """Load every known manifest, dedup by ``id``, return merged entries.

    Missing manifests are skipped with an info log. Later manifests overwrite
    earlier ones on id collision so that, e.g., a per-source pixabay entry
    wins over the same id in the main manifest (which shouldn't happen in
    practice, but keeps merge semantics predictable).
    """
    paths = paths if paths is not None else MANIFEST_PATHS
    merged: dict[Any, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            logger.info("Skipping missing manifest: %s", path)
            continue
        try:
            entries = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            logger.warning("Manifest %s is not valid JSON: %s", path, exc)
            continue
        if not isinstance(entries, list):
            logger.warning("Manifest %s is not a JSON list; skipping", path)
            continue
        kept = 0
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("id") is None:
                continue
            merged[entry["id"]] = entry
            kept += 1
        logger.info("Loaded %d entries from %s", kept, path.name)
    return list(merged.values())


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    arr = model.encode(
        texts,
        batch_size=_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return [vec.tolist() for vec in arr]


def _embed_with_progress(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    total = len(texts)
    out: list[list[float]] = []
    next_log = _LOG_EVERY
    step = max(_BATCH_SIZE, 256)
    for start in range(0, total, step):
        chunk = texts[start : start + step]
        arr = model.encode(
            chunk,
            batch_size=_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        out.extend(vec.tolist() for vec in arr)
        done = len(out)
        if done >= next_log or done == total:
            pct = (done / total) * 100 if total else 100.0
            logger.info("Embedded %d/%d sounds (%.0f%%)...", done, total, pct)
            while done >= next_log:
                next_log += _LOG_EVERY
    return out


def embed_library(
    manifest_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Embed the merged library across all known manifests.

    ``manifest_path`` is accepted for backwards compatibility — if provided,
    only that single manifest is loaded; otherwise all known sources are
    merged via :func:`load_all_manifests`.
    """
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = EMBEDDINGS_DIR / "embeddings.json"
    if cache_path.is_file():
        logger.info("Loading cached embeddings from %s", cache_path)
        cached: list[dict[str, Any]] = json.loads(cache_path.read_text())
        retagged = 0
        for s in cached:
            payload = s.get("payload") or {}
            if "tiers" not in payload:
                payload["tiers"] = classify_tiers(payload)
                s["payload"] = payload
                retagged += 1
        if retagged:
            cache_path.write_text(json.dumps(cached))
            logger.info(
                "Re-tagged %d cached entries with tiers (cache pre-dates tiers); "
                "cache updated.",
                retagged,
            )
        return cached

    if manifest_path is not None:
        valid = load_all_manifests([manifest_path])
    else:
        valid = load_all_manifests()

    if not valid:
        raise RuntimeError(
            "No manifest entries found. Run `python -m src.library.downloader` "
            "(and optionally pixabay/mixkit/sonniss importers) first."
        )

    sources: dict[str, int] = {}
    for item in valid:
        src = str(item.get("source", "freesound") or "freesound")
        sources[src] = sources.get(src, 0) + 1
    logger.info(
        "Embedding %d sounds with %s (sources: %s)",
        len(valid), _MODEL_NAME,
        ", ".join(f"{k}={v}" for k, v in sorted(sources.items())),
    )

    for item in valid:
        item["tiers"] = classify_tiers(item)

    dist = tier_distribution(valid)
    logger.info(
        "Tier distribution (overlapping): %s",
        ", ".join(f"{k}={v}" for k, v in dist.items()),
    )

    texts = [build_embedding_text(item) for item in valid]
    vectors = _embed_with_progress(texts)

    sounds: list[dict[str, Any]] = [
        {"id": item["id"], "vector": vector, "payload": item}
        for item, vector in zip(valid, vectors)
    ]

    cache_path.write_text(json.dumps(sounds))
    logger.info("Cached %d embeddings to %s", len(sounds), cache_path)
    return sounds


def _main() -> None:
    import argparse

    from src.library.vector_store import (
        COLLECTION_NAME,
        create_collection,
        get_qdrant_client,
        index_sounds,
        search_by_text,
    )

    parser = argparse.ArgumentParser(
        description="Embed the library and index it into Qdrant.",
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help=(
            "Refuse to compute embeddings from manifests; require the cached "
            "embeddings.json (used in R2 / small-volume mode where audio files "
            "live on R2 and never touch the disk)."
        ),
    )
    args = parser.parse_args()

    if args.from_cache:
        cache_path = EMBEDDINGS_DIR / "embeddings.json"
        if not cache_path.is_file():
            raise RuntimeError(
                f"--from-cache requires {cache_path} but it does not exist. "
                "Pre-compute embeddings locally and upload them to R2 first."
            )

    sounds = embed_library()
    print(f"Embedded {len(sounds)} sounds")

    client = get_qdrant_client()
    create_collection(client, COLLECTION_NAME)
    indexed = index_sounds(client, COLLECTION_NAME, sounds)
    print(f"Indexed {indexed} sounds in collection {COLLECTION_NAME!r}")

    payloads = [s.get("payload") or {} for s in sounds]
    dist = tier_distribution(payloads)
    print("\nTier distribution (a sound can belong to multiple tiers):")
    for tier, count in dist.items():
        print(f"  {tier:<10} {count:>7}")

    print("\nTest query: 'door slam'")
    results = search_by_text(client, COLLECTION_NAME, "door slam", top_k=3)
    for i, r in enumerate(results, start=1):
        name = r["payload"].get("name", "")
        print(f"  {i}. id={r['id']} score={r['score']:.3f}  {name}")

    print(f"\n{len(sounds)} sounds indexed in collection {COLLECTION_NAME!r}")


if __name__ == "__main__":
    _main()
