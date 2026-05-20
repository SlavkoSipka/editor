"""Tier classification for library sounds.

Each sound carries zero or more tiers in its Qdrant payload, used by the
matcher to route searches based on the active preset and action type. Tiers
overlap on purpose — a Sonniss "cinematic whoosh" is both cinematic and foley
and shows up for either filter.
"""

from __future__ import annotations

from typing import Any

TIER_KEYWORDS: dict[str, dict[str, list[str]]] = {
    "ugc": {
        "sources": ["pixabay", "mixkit"],
        "keywords": [
            "tiktok", "viral", "meme", "vine", "bruh", "discord",
            "iphone notification", "tssk", "pop transition",
            "punch in", "snap zoom", "freeze frame", "record scratch",
            "ironic violin", "sad violin", "anime", "wow kid",
            "boing", "cartoon", "comedic", "funny",
            "subscribe", "like button", "notification ping",
            "app notification", "swipe phone", "8 bit", "retro game",
            "vlog", "youtube", "instagram",
        ],
    },
    "cinematic": {
        "sources": [],
        "keywords": [
            "cinematic", "trailer", "epic", "dramatic",
            "riser", "swell", "build up", "uplifter", "downlifter",
            "boom", "impact deep", "sub drop", "bass drop",
            "orchestral", "stinger", "ident",
            "logo sting", "logo intro", "brand reveal",
            "movie", "film", "theatrical",
            "tension", "suspense", "horror sting",
            "shimmer reveal", "magical reveal",
            "tonal riser", "noise riser",
        ],
    },
    "foley": {
        "sources": [],
        "keywords": [
            "footstep", "footsteps", "walking", "step",
            "door", "knock", "latch",
            "scissors", "knife", "cut", "snip",
            "fabric", "cloth", "paper",
            "glass", "metal clang", "wood", "ceramic",
            "water", "rain", "wind",
            "machine", "motor", "engine", "mechanical",
            "click", "button", "switch",
            "drink", "pour", "ice", "bottle",
        ],
    },
    "ambient": {
        "sources": [],
        "keywords": [
            "ambience", "ambient", "room tone", "atmosphere",
            "loop", "loopable", "background",
            "city ambience", "office", "nature", "forest",
            "rain background", "wind background",
            "indoor", "outdoor", "drone", "pad",
        ],
    },
}

TIER_ORDER: tuple[str, ...] = ("ugc", "cinematic", "foley", "ambient")


def classify_tiers(sound: dict[str, Any]) -> list[str]:
    """Return the list of tiers a sound belongs to.

    A sound can match multiple tiers. An empty result means "generic" — the
    sound is still indexed but only retrievable via the global (unfiltered)
    pool.
    """
    name = str(sound.get("name") or "").lower()
    description = str(sound.get("description") or "").lower()
    tags = " ".join(str(t) for t in (sound.get("tags") or [])).lower()
    source = str(sound.get("source") or "freesound").lower()
    haystack = f"{name} {description} {tags}"

    tiers: list[str] = []
    for tier in TIER_ORDER:
        rules = TIER_KEYWORDS[tier]
        if source in rules.get("sources", []):
            tiers.append(tier)
            continue
        if any(kw in haystack for kw in rules["keywords"]):
            tiers.append(tier)

    return tiers


def tier_distribution(sounds: list[dict[str, Any]]) -> dict[str, int]:
    """Count how many sounds each tier covers. ``untagged`` = no tiers."""
    counts: dict[str, int] = {t: 0 for t in TIER_ORDER}
    counts["untagged"] = 0
    for s in sounds:
        tiers = s.get("tiers") or []
        if not tiers:
            counts["untagged"] += 1
            continue
        for t in tiers:
            if t in counts:
                counts[t] += 1
    return counts
