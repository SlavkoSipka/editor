from __future__ import annotations

from collections import Counter
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.library.embedder import embed_texts
from src.library.music_embedder import MUSIC_COLLECTION_NAME
from src.presets import get_preset
from src.utils.logger import get_logger

logger = get_logger("music_matcher")

_MUSIC_SEARCH_TOP_K = 30
_TOP_MOODS_FOR_QUERY = 3


def _collection_exists(client: QdrantClient, name: str) -> bool:
    try:
        existing = {c.name for c in client.get_collections().collections}
    except Exception as exc:
        logger.warning("Failed to list Qdrant collections: %s", exc)
        return False
    return name in existing


def _aggregate_scene_moods(analysis: dict[str, Any]) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for scene in analysis.get("scenes", []) or []:
        mood = str(scene.get("mood", "") or "").strip().lower()
        if mood and mood != "unknown":
            counter[mood] += 1
    return counter.most_common()


def _aggregate_environments(analysis: dict[str, Any]) -> list[str]:
    counter: Counter[str] = Counter()
    for scene in analysis.get("scenes", []) or []:
        env = str(scene.get("environment", "") or "").strip().lower()
        if env and env != "unknown":
            counter[env] += 1
    return [env for env, _ in counter.most_common(2)]


def _build_music_query(
    preset_mood: str,
    scene_moods: list[tuple[str, int]],
    environments: list[str],
) -> str:
    parts: list[str] = [f"{preset_mood} music underscore"]
    for mood, _ in scene_moods[:_TOP_MOODS_FOR_QUERY]:
        parts.append(mood)
    for env in environments:
        parts.append(env.replace("_", " "))
    return " ".join(parts)


def _search_music(
    client: QdrantClient,
    query_vector: list[float],
    inferred_mood: str | None,
    top_k: int,
) -> list[dict[str, Any]]:
    qdrant_filter = (
        Filter(must=[FieldCondition(
            key="inferred_mood",
            match=MatchValue(value=inferred_mood),
        )])
        if inferred_mood else None
    )
    hits = client.search(
        collection_name=MUSIC_COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=qdrant_filter,
        limit=top_k,
    )
    return [
        {"id": h.id, "score": float(h.score), "payload": h.payload or {}}
        for h in hits
    ]


def _pick_best_for_duration(
    candidates: list[dict[str, Any]], video_duration_sec: float,
) -> dict[str, Any]:
    """Prefer tracks whose duration is >= video duration to avoid heavy looping."""
    if not candidates:
        raise ValueError("No candidates to pick from")
    long_enough = [
        c for c in candidates
        if float(c["payload"].get("duration") or 0.0) >= video_duration_sec
    ]
    pool = long_enough if long_enough else candidates
    return pool[0]


def select_music(
    analysis: dict[str, Any],
    preset_name: str,
    qdrant_client: QdrantClient,
) -> dict[str, Any] | None:
    """Pick a single background music track for the whole video. Returns None if
    the music_library collection is missing or empty."""
    if not _collection_exists(qdrant_client, MUSIC_COLLECTION_NAME):
        logger.warning(
            "Music collection %r not found in Qdrant. "
            "Run `python -m src.library.music_downloader` then "
            "`python -m src.library.music_embedder`. Skipping music.",
            MUSIC_COLLECTION_NAME,
        )
        return None

    preset = get_preset(preset_name)
    preferred_mood = preset.music_mood_preference

    scene_moods = _aggregate_scene_moods(analysis)
    environments = _aggregate_environments(analysis)
    query_text = _build_music_query(preferred_mood, scene_moods, environments)
    video_duration_sec = float(analysis.get("duration_sec", 0.0))

    logger.info("Music query: %r (preferred mood: %s)", query_text, preferred_mood)
    query_vector = embed_texts([query_text])[0]

    # First pass: filter by preferred mood.
    results = _search_music(
        qdrant_client, query_vector, preferred_mood, _MUSIC_SEARCH_TOP_K,
    )
    used_mood = preferred_mood

    # Second pass: if the mood filter starves us, broaden.
    if not results:
        logger.warning(
            "No music candidates for mood %r; broadening search", preferred_mood,
        )
        results = _search_music(
            qdrant_client, query_vector, None, _MUSIC_SEARCH_TOP_K,
        )
        used_mood = "any"

    if not results:
        logger.warning("Music collection returned no candidates at all; skipping music")
        return None

    chosen = _pick_best_for_duration(results, video_duration_sec)
    payload = chosen["payload"]

    dominant_label = (
        f"{scene_moods[0][1]} {scene_moods[0][0]} scenes"
        if scene_moods else "no clear scene mood"
    )
    reason = (
        f"{preset.name} preset (music_mood_preference={preferred_mood}, "
        f"matched={used_mood}) + {dominant_label}"
    )

    return {
        "sound_id": chosen["id"],
        "sound_name": payload.get("name", ""),
        "sound_path": payload.get("local_path", ""),
        "sound_duration": float(payload.get("duration") or 0.0),
        "selected_mood": str(payload.get("inferred_mood", used_mood)),
        "match_score": float(chosen["score"]),
        "reason": reason,
    }
