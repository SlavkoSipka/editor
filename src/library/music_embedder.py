from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import EMBEDDINGS_DIR, LIBRARY_DIR
from src.library.embedder import embed_texts
from src.library.vector_store import (
    create_collection,
    get_qdrant_client,
    index_sounds,
    search_by_text,
)
from src.utils.logger import get_logger

logger = get_logger("music_embedder")

MUSIC_COLLECTION_NAME = "music_library"
_MUSIC_MANIFEST_PATH = LIBRARY_DIR / "music" / "manifest.json"
_MUSIC_CACHE_PATH = EMBEDDINGS_DIR / "music_embeddings.json"
_MAX_TEXT_CHARS = 500

_MOOD_KEYWORDS: dict[str, tuple[str, ...]] = {
    "dramatic": ("epic", "cinematic", "dramatic", "trailer", "tension", "suspense", "dark"),
    "energetic": (
        "upbeat", "energetic", "fast", "edm", "electronic", "trap", "hip hop",
        "motivational", "corporate",
    ),
    "mysterious": (
        "ambient", "atmospheric", "drone", "ethereal", "soundscape", "evolving",
        "texture", "pad",
    ),
    "playful": ("playful", "bouncy", "fun", "ukulele", "whistle", "happy"),
    "luxury": ("elegant", "minimal", "smooth", "lofi", "jazz", "soft piano", "piano"),
}


def infer_music_mood(name: str, description: str, tags: list[str]) -> str:
    text = f"{name} {description} {' '.join(str(t) for t in tags)}".lower()
    scores = {
        mood: sum(1 for kw in kws if kw in text)
        for mood, kws in _MOOD_KEYWORDS.items()
    }
    best_mood, best_score = max(scores.items(), key=lambda x: x[1])
    return best_mood if best_score > 0 else "neutral"


def build_music_text(item: dict[str, Any]) -> str:
    name = str(item.get("name", "") or "")
    description = str(item.get("description", "") or "")
    tags = item.get("tags", []) or []
    mood = item.get("inferred_mood") or infer_music_mood(name, description, tags)
    tags_str = ", ".join(str(t) for t in tags)
    text = f"{name}. {description}. Tags: {tags_str}. Mood: {mood}."
    text = " ".join(text.lower().split())
    return text[:_MAX_TEXT_CHARS]


def embed_music_library(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Music manifest not found: {manifest_path}. "
            f"Run `python -m src.library.music_downloader` first."
        )

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    if _MUSIC_CACHE_PATH.is_file():
        logger.info("Loading cached music embeddings from %s", _MUSIC_CACHE_PATH)
        return json.loads(_MUSIC_CACHE_PATH.read_text())

    manifest: list[dict[str, Any]] = json.loads(manifest_path.read_text())
    if not manifest:
        raise RuntimeError(f"Music manifest is empty: {manifest_path}")

    enriched: list[dict[str, Any]] = []
    for item in manifest:
        item = dict(item)
        item["inferred_mood"] = infer_music_mood(
            str(item.get("name", "") or ""),
            str(item.get("description", "") or ""),
            item.get("tags", []) or [],
        )
        enriched.append(item)

    texts = [build_music_text(item) for item in enriched]
    logger.info("Embedding %d music tracks", len(texts))
    vectors = embed_texts(texts)

    sounds: list[dict[str, Any]] = [
        {"id": item["id"], "vector": vector, "payload": item}
        for item, vector in zip(enriched, vectors)
    ]

    _MUSIC_CACHE_PATH.write_text(json.dumps(sounds))
    logger.info("Cached %d music embeddings to %s", len(sounds), _MUSIC_CACHE_PATH)
    return sounds


def _main() -> None:
    sounds = embed_music_library(_MUSIC_MANIFEST_PATH)
    print(f"Embedded {len(sounds)} music tracks")

    client = get_qdrant_client()
    create_collection(client, MUSIC_COLLECTION_NAME)
    indexed = index_sounds(client, MUSIC_COLLECTION_NAME, sounds)
    print(f"Indexed {indexed} tracks in collection {MUSIC_COLLECTION_NAME!r}")

    mood_counts: dict[str, int] = {}
    for s in sounds:
        m = str(s["payload"].get("inferred_mood", "neutral"))
        mood_counts[m] = mood_counts.get(m, 0) + 1
    print("\nMood distribution:")
    for mood, count in sorted(mood_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {mood:<12} {count}")

    print("\nTest query: 'dramatic cinematic underscore'")
    results = search_by_text(
        client, MUSIC_COLLECTION_NAME, "dramatic cinematic underscore", top_k=3,
    )
    for i, r in enumerate(results, start=1):
        name = r["payload"].get("name", "")
        mood = r["payload"].get("inferred_mood", "?")
        print(f"  {i}. id={r['id']} score={r['score']:.3f} mood={mood}  {name}")


if __name__ == "__main__":
    _main()
