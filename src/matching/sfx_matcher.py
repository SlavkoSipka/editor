from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from src.analysis.gemini_client import analyze_video
from src.config import TEMP_DIR
from src.library.vector_store import (
    COLLECTION_NAME,
    count_by_tier,
    get_qdrant_client,
    search_by_text,
)
from src.matching.music_matcher import select_music
from src.matching.preset_reranker import rerank_candidates
from src.matching.recipe_builder import build_recipe_layers
from src.matching.recipes import moment_type_for_action
from src.matching.selectivity import apply_selectivity
from src.matching.timing_adjuster import snap_to_onsets
from src.matching.verifier import research_dropped_anchors, verify_matches
from src.preprocessing.onset_detector import detect_onsets_from_video
from src.preprocessing.scene_detector import Scene, detect_scenes
from src.presets import PRESETS, Preset, get_preset
from src.utils.logger import get_logger

logger = get_logger("sfx_matcher")

SFX_MIN_SCORE = 0.4
SFX_MIN_SCORE_TIER_FILTERED = 0.35  # lower bar when results came from a tier filter
AMBIENT_MIN_SCORE = 0.3
MIN_SCORE_THRESHOLD = AMBIENT_MIN_SCORE  # legacy alias
AMBIENT_MIN_DURATION_SEC = 3.0
_AMBIENT_SEARCH_MULTIPLIER = 4
_SFX_SEARCH_MULTIPLIER = 3
_PRIMARY_ACCEPT_SCORE = 0.4
_FALLBACK_ACCEPT_SCORE = 0.35

# Action-type → tier override. Some action types always want a specific tier
# regardless of preset (e.g., logo reveals always benefit from cinematic stings).
_ACTION_TIER_OVERRIDES: dict[str, list[str]] = {
    "cinematic": ["logo", "reveal_sting", "drop_moment", "epic"],
    "ugc": ["punchline", "freeze_frame", "punch_in", "zoom_meme"],
    "foley": ["footstep", "door", "scissors", "machine"],
    "ambient": ["ambient"],
}

# Anti-repetition / density tuning.
MAX_SFX_PER_SCENE = 3
SAME_TYPE_MIN_GAP_SEC = 0.6
OVERUSED_THRESHOLD = 2
OVERUSED_PENALTY = 0.10
TOP_K_CANDIDATES = 5

# Hard filters: candidate's name + tags must NOT contain any of these tokens.
_AMBIENT_BLACKLIST = (
    "footstep", "footsteps", "walk", "walking",
    "door open", "door close", "door slam", "door creak",
    "click", "tap", "button",
    "knock", "bell", "alarm",
    "voice", "speech", "speak", "talk",
    "snap", "clap", "punch", "hit", "impact",
    "synthesized",
)
_TAP_CLICK_TYPE_KEYWORDS = ("tap", "click", "screen")
_TAP_CLICK_BLACKLIST = ("snap", "finger snap")
_MACHINE_TYPE_KEYWORDS = ("machine", "motor", "stitch")
_MACHINE_BLACKLIST = ("food", "processor", "blender", "kitchen")

# action_type → at least one of these tokens must appear in candidate's
# name/description/tags. Otherwise we soft-fall-back to the unfiltered set.
ACTION_CATEGORY_WHITELIST: dict[str, list[str]] = {
    # Transitions / whooshes
    "transition": ["whoosh", "swoosh", "swipe", "transition", "riser", "sweep"],
    "whoosh": ["whoosh", "swoosh", "swipe", "transition", "sweep", "air"],
    "swoosh": ["whoosh", "swoosh", "swipe", "transition", "sweep"],
    # Impacts / hits
    "impact": ["impact", "hit", "boom", "thud", "slam", "punch"],
    "hit": ["impact", "hit", "boom", "thud", "slam", "punch"],
    "punch": ["punch", "impact", "hit", "thud"],
    # UI / clicks / taps
    "click": ["click", "tap", "button", "ui", "switch", "key"],
    "tap": ["tap", "click", "touch", "ui", "button", "screen"],
    "screen_tap": ["tap", "click", "touch", "screen", "ui", "interface"],
    "ui_click": ["click", "tap", "ui", "button", "interface"],
    "mouse_click": ["click", "mouse", "button"],
    "screen_transition": ["whoosh", "transition", "swipe", "sweep", "swoosh"],
    "graphic_reveal": ["whoosh", "transition", "reveal", "riser", "swell"],
    # Footsteps
    "footstep": ["footstep", "foot", "step", "walk", "boot", "shoe"],
    "footsteps": ["footstep", "foot", "step", "walk", "boot", "shoe"],
    # Doors
    "door": ["door"],
    # Logo / brand reveal
    "logo": ["logo", "sting", "stinger", "hit", "impact", "riser", "swell", "cinematic"],
    "logo_appear": ["logo", "sting", "stinger", "hit", "impact", "swell"],
    "logo_settle": ["logo", "sting", "settle", "hit", "swell"],
    "logo_reveal": ["logo", "sting", "stinger", "hit", "impact", "riser", "reveal"],
    # Glitch / digital
    "glitch": ["glitch", "digital", "data", "static", "noise", "error"],
    "digital_reveal": ["glitch", "digital", "riser", "data"],
    "digital_transition": ["glitch", "digital", "data", "transition"],
    # Machine / mechanical
    "machine": ["machine", "motor", "engine", "mechanical", "hum", "whir", "factory"],
    "machine_start": ["motor", "engine", "machine", "startup", "whir", "rev"],
    "machine_stop": ["motor", "engine", "shutdown", "stop", "power down", "off"],
    "machine_stitch": ["sewing", "stitch", "needle", "machine", "punch"],
    "machine_stitching": ["sewing", "stitch", "needle", "machine"],
    "needle_stitch": ["sewing", "stitch", "needle", "punch", "click"],
    # Cutting / scissors
    "scissors": ["scissors", "snip", "cut"],
    "scissors_cut": ["scissors", "snip", "cut", "fabric"],
    "cut": ["cut", "snip", "scissors", "knife"],
    # Latch / mechanism / clip
    "latch": ["latch", "click", "mechanism", "snap", "lock", "clip"],
    "latch_click": ["latch", "click", "mechanism", "snap", "lock"],
    "hoop_placement": ["place", "set", "wood", "soft", "thud"],
    # Light / reveals
    "light_reveal": ["whoosh", "swell", "riser", "shimmer", "reveal", "magic"],
    "reveal": ["whoosh", "riser", "swell", "reveal", "shimmer"],
}

# Per-intensity allowed duration band (seconds) for SFX candidates.
_INTENSITY_DURATION_LIMITS: dict[str, tuple[float, float]] = {
    "sharp": (0.05, 2.5),
    "medium": (0.10, 5.0),
    "soft": (0.10, 8.0),
}


def get_whitelist_for_action(action_type: str) -> list[str] | None:
    """Find the best matching whitelist for an action_type. None if no rule applies."""
    action_lower = (action_type or "").lower()
    if not action_lower:
        return None
    if action_lower in ACTION_CATEGORY_WHITELIST:
        return ACTION_CATEGORY_WHITELIST[action_lower]
    for key, keywords in ACTION_CATEGORY_WHITELIST.items():
        if key in action_lower:
            return keywords
    return None


def _build_query(action: dict[str, Any]) -> str:
    parts: list[str] = [
        str(action.get("description", "") or ""),
        f"intensity {action.get('intensity', 'medium')}",
        f"type {str(action.get('type', '') or '').replace('_', ' ')}",
    ]
    env = action.get("_scene_environment")
    if env:
        parts.append(str(env).replace("_", " "))
    return ". ".join(p.strip() for p in parts if p and p.strip())


def _payload_text(payload: dict[str, Any]) -> str:
    name = str(payload.get("name", "") or "").lower()
    tags = " ".join(str(t) for t in (payload.get("tags") or [])).lower()
    return f"{name} {tags}"


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(tok in text for tok in tokens)


def _apply_hard_filters(
    raw: list[dict[str, Any]], action: dict[str, Any],
) -> list[dict[str, Any]]:
    layer = action.get("layer")
    action_type = str(action.get("type", "") or "").lower()

    if layer == "ambient":
        return [c for c in raw if not _contains_any(_payload_text(c["payload"]), _AMBIENT_BLACKLIST)]

    if _contains_any(action_type, _TAP_CLICK_TYPE_KEYWORDS):
        raw = [c for c in raw if not _contains_any(_payload_text(c["payload"]), _TAP_CLICK_BLACKLIST)]

    if _contains_any(action_type, _MACHINE_TYPE_KEYWORDS):
        raw = [c for c in raw if not _contains_any(_payload_text(c["payload"]), _MACHINE_BLACKLIST)]

    return raw


def _apply_usage_penalty(
    candidates: list[dict[str, Any]],
    used_sound_counts: dict[int, int],
) -> list[dict[str, Any]]:
    for c in candidates:
        usage = used_sound_counts.get(c["id"], 0)
        if usage >= OVERUSED_THRESHOLD:
            c["reranked_score"] = float(c.get("reranked_score", c["score"])) - OVERUSED_PENALTY
            existing = c.get("rerank_reason", "") or ""
            tag = f"-{OVERUSED_PENALTY:.2f} over-used (x{usage})"
            c["rerank_reason"] = f"{existing}; {tag}".lstrip("; ")
    candidates.sort(key=lambda x: x["reranked_score"], reverse=True)
    return candidates


def _apply_whitelist_filter(
    candidates: list[dict[str, Any]],
    action: dict[str, Any],
    stats: dict[str, int] | None,
) -> list[dict[str, Any]]:
    whitelist = get_whitelist_for_action(str(action.get("type", "") or ""))
    if not whitelist:
        return candidates
    filtered: list[dict[str, Any]] = []
    for c in candidates:
        payload = c.get("payload") or {}
        text = (
            f"{payload.get('name', '') or ''} "
            f"{payload.get('description', '') or ''} "
            f"{' '.join(str(t) for t in (payload.get('tags') or []))}"
        ).lower()
        if any(kw in text for kw in whitelist):
            filtered.append(c)
    if filtered:
        return filtered
    if stats is not None:
        stats["whitelist_fallback"] = stats.get("whitelist_fallback", 0) + 1
    return candidates


def _apply_intensity_duration_filter(
    candidates: list[dict[str, Any]],
    action: dict[str, Any],
    stats: dict[str, int] | None,
) -> list[dict[str, Any]]:
    intensity = str(action.get("intensity", "medium") or "medium").lower()
    min_dur, max_dur = _INTENSITY_DURATION_LIMITS.get(intensity, (0.10, 5.0))
    filtered = [
        c for c in candidates
        if min_dur <= float(c["payload"].get("duration") or 0.0) <= max_dur
    ]
    if filtered:
        return filtered
    if stats is not None:
        stats["duration_fallback"] = stats.get("duration_fallback", 0) + 1
    return candidates


def get_tier_override_for_action(action_type: str) -> list[str] | None:
    """Some action types always want a specific tier regardless of preset.
    Returns ``None`` if no override applies."""
    action_lower = (action_type or "").lower()
    if not action_lower:
        return None
    for tier, action_keywords in _ACTION_TIER_OVERRIDES.items():
        if any(kw in action_lower for kw in action_keywords):
            return [tier]
    return None


def _available_tiers(
    tiers: list[str] | None,
    tier_counts: dict[str, int] | None,
) -> list[str]:
    if not tiers:
        return []
    if tier_counts is None:
        return list(tiers)
    return [t for t in tiers if tier_counts.get(t, 0) > 0]


def normalize_scores_by_tier(
    candidates: list[dict[str, Any]],
    tier_counts: dict[str, int] | None,
) -> list[dict[str, Any]]:
    """Boost candidates from smaller tiers slightly so they compete fairly.
    Smallest tier gets ~+0.10, largest tier gets 0. Mutates and returns."""
    if not candidates or not tier_counts:
        return candidates
    sizes = {t: c for t, c in tier_counts.items() if c > 0}
    if not sizes:
        return candidates
    largest = max(sizes.values())
    for c in candidates:
        tiers = (c.get("payload") or {}).get("tiers") or []
        relevant = [sizes[t] for t in tiers if t in sizes]
        if not relevant:
            continue
        primary_size = min(relevant)
        bump = 0.10 * (1.0 - primary_size / largest)
        c["score"] = float(c.get("score", 0.0)) + bump
    return candidates


def _merge_candidates(
    primary: list[dict[str, Any]],
    extra: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Dedup by id; keep the entry with the higher score on collision. Preserves
    the ``_routing_path`` / ``_tier_filter_used`` already stamped on each entry."""
    by_id: dict[Any, dict[str, Any]] = {c["id"]: c for c in primary}
    for c in extra:
        existing = by_id.get(c["id"])
        if existing is None or c.get("score", 0.0) > existing.get("score", 0.0):
            by_id[c["id"]] = c
    merged = list(by_id.values())
    merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return merged


def _stamp(
    candidates: list[dict[str, Any]],
    routing_path: str,
    tier_filter: list[str] | None,
) -> list[dict[str, Any]]:
    for c in candidates:
        c["_routing_path"] = routing_path
        c["_tier_filter_used"] = list(tier_filter) if tier_filter else []
    return candidates


def _resolve_tier_routing(
    action: dict[str, Any],
    preset: Preset,
    tier_counts: dict[str, int] | None,
    stats: dict[str, int] | None,
) -> tuple[list[str], list[str], str | None]:
    """Return (primary_filter, fallback_filter, action_override_tier)."""
    if action.get("layer") == "ambient":
        ambient_filter = _available_tiers(["ambient"], tier_counts)
        return ambient_filter, [], None

    override = get_tier_override_for_action(str(action.get("type", "") or ""))
    if override:
        override_avail = _available_tiers(override, tier_counts)
        if override_avail:
            if stats is not None:
                stats["action_override_used"] = stats.get("action_override_used", 0) + 1
            return override_avail, _available_tiers(preset.fallback_tiers, tier_counts), override[0]

    primary = _available_tiers(preset.primary_tiers, tier_counts)
    fallback = _available_tiers(preset.fallback_tiers, tier_counts)
    if not primary and preset.primary_tiers:
        # Every requested primary tier is empty → promote fallback to primary.
        primary = fallback
        fallback = []
    return primary, fallback, None


def _search_raw(
    qdrant_client: QdrantClient,
    collection_name: str,
    query: str,
    search_k: int,
    tier_filter: list[str] | None,
) -> list[dict[str, Any]]:
    return search_by_text(
        qdrant_client, collection_name, query, top_k=search_k, tier_filter=tier_filter,
    )


def match_action_to_sound(
    action: dict[str, Any],
    qdrant_client: QdrantClient,
    preset: Preset,
    used_sound_counts: dict[int, int] | None = None,
    stats: dict[str, int] | None = None,
    collection_name: str = COLLECTION_NAME,
    top_k: int = TOP_K_CANDIDATES,
    tier_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Search Qdrant with tier-aware routing, hard-filter, layer-aware re-rank,
    apply usage penalty, return up to ``top_k`` candidates.

    Routing order:
      1. Per-action tier override (e.g., ``logo_reveal`` → ``cinematic``).
      2. Preset ``primary_tiers``.
      3. Preset ``fallback_tiers`` (merged with primary).
      4. Global pool, no tier filter (merged in).

    Each returned candidate carries ``_routing_path`` ("primary"/"fallback"/
    "global"/"action_override"/"ambient") and ``_tier_filter_used`` for
    downstream debugging."""
    if used_sound_counts is None:
        used_sound_counts = {}

    query = _build_query(action)
    if not query:
        return []

    is_ambient = action.get("layer") == "ambient"
    multiplier = _AMBIENT_SEARCH_MULTIPLIER if is_ambient else _SFX_SEARCH_MULTIPLIER
    search_k = top_k * multiplier

    primary_filter, fallback_filter, action_override_tier = _resolve_tier_routing(
        action, preset, tier_counts, stats,
    )
    primary_label = "ambient" if is_ambient else (
        "action_override" if action_override_tier else "primary"
    )

    raw: list[dict[str, Any]] = []
    if primary_filter:
        raw = _stamp(
            _search_raw(qdrant_client, collection_name, query, search_k, primary_filter),
            primary_label, primary_filter,
        )

    used_fallback = False
    if not is_ambient:
        primary_best = max((r["score"] for r in raw), default=0.0)
        if primary_best < _PRIMARY_ACCEPT_SCORE and fallback_filter:
            extra = _stamp(
                _search_raw(qdrant_client, collection_name, query, search_k, fallback_filter),
                "fallback", fallback_filter,
            )
            raw = _merge_candidates(raw, extra)
            used_fallback = True
            if stats is not None:
                stats["fallback_tier_used"] = stats.get("fallback_tier_used", 0) + 1

        merged_best = max((r["score"] for r in raw), default=0.0)
        if merged_best < _FALLBACK_ACCEPT_SCORE or not raw:
            extra = _stamp(
                _search_raw(qdrant_client, collection_name, query, search_k, None),
                "global", None,
            )
            raw = _merge_candidates(raw, extra)
            if stats is not None:
                stats["global_pool_used"] = stats.get("global_pool_used", 0) + 1
    elif not raw:
        # Ambient with no ambient tier indexed — fall back to global pool.
        raw = _stamp(
            _search_raw(qdrant_client, collection_name, query, search_k, None),
            "global", None,
        )

    if is_ambient:
        long_enough = [
            r for r in raw
            if float(r["payload"].get("duration") or 0.0) >= AMBIENT_MIN_DURATION_SEC
        ]
        if long_enough:
            raw = long_enough

    raw = _apply_hard_filters(raw, action)

    if not is_ambient:
        raw = _apply_whitelist_filter(raw, action, stats)
        raw = _apply_intensity_duration_filter(raw, action, stats)

    raw = normalize_scores_by_tier(raw, tier_counts)

    def _threshold_for(c: dict[str, Any]) -> float:
        if is_ambient:
            return AMBIENT_MIN_SCORE
        if c.get("_routing_path") in ("primary", "action_override", "fallback"):
            return SFX_MIN_SCORE_TIER_FILTERED
        return SFX_MIN_SCORE

    candidates = [r for r in raw if r["score"] >= _threshold_for(r)]
    if not candidates:
        if stats is not None and not is_ambient:
            stats["min_score_rejected"] = stats.get("min_score_rejected", 0) + 1
        return []

    layer = "ambient" if is_ambient else "sfx"
    reranked = rerank_candidates(candidates, preset, layer=layer)
    reranked = _apply_usage_penalty(reranked, used_sound_counts)
    _ = used_fallback  # kept for readability; stats already track it
    return reranked[:top_k]


def _candidate_to_match_payload(c: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    payload = c.get("payload") or {}
    return {
        "sound_id": c["id"],
        "score": c["score"],
        "reranked_score": c.get("reranked_score", c["score"]),
        "rerank_reason": c.get("rerank_reason", ""),
        "local_path": payload.get("local_path", ""),
        "name": payload.get("name", ""),
        "duration": float(payload.get("duration") or 0.0),
        "tiers": list(payload.get("tiers") or []),
        "routing_path": c.get("_routing_path", "global"),
        "tier_filter_used": list(c.get("_tier_filter_used") or []),
        "action": action,
    }


def _matched_tier_for(best: dict[str, Any]) -> str:
    """Tier label to surface on a match: intersection of the chosen sound's
    tiers and the filter that was applied, otherwise ``global``."""
    filter_used = best.get("tier_filter_used") or []
    tiers = best.get("tiers") or []
    if filter_used:
        for t in filter_used:
            if t in tiers:
                return t
        return filter_used[0]
    return "global"


def _build_match_entry(
    scene_index: int,
    scene_start: float,
    scene_end: float,
    absolute_ts: float,
    action_type: str,
    description: str,
    intensity: str,
    confidence: float,
    layer: str,
    best: dict[str, Any],
    gemini_sound_value: float | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "scene_index": scene_index,
        "scene_start_sec": scene_start,
        "scene_end_sec": scene_end,
        "absolute_timestamp": absolute_ts,
        "action_type": action_type,
        "action_description": description,
        "intensity": intensity,
        "confidence": confidence,
        "sound_id": best["sound_id"],
        "sound_name": best["name"],
        "sound_path": best["local_path"],
        "sound_duration": best["duration"],
        "match_score": best["score"],
        "reranked_score": best.get("reranked_score", best["score"]),
        "rerank_reason": best.get("rerank_reason", ""),
        "layer": layer,
        "matched_tier": _matched_tier_for(best),
        "routing_path": best.get("routing_path", "global"),
        "tier_filter_used": list(best.get("tier_filter_used") or []),
    }
    if gemini_sound_value is not None:
        entry["gemini_sound_value"] = float(gemini_sound_value)
    return entry


def _enforce_same_type_gap(
    scene_sfx: list[dict[str, Any]],
    unmatched: list[dict[str, Any]],
    used_sound_counts: dict[int, int],
    scene_index: int,
) -> list[dict[str, Any]]:
    """Within a scene, drop near-duplicates of the same action_type that fire
    too close together. Prefer the higher-confidence one."""
    keep: list[dict[str, Any]] = []
    last_for_type: dict[str, dict[str, Any]] = {}
    for m in sorted(scene_sfx, key=lambda x: x["absolute_timestamp"]):
        t = str(m["action_type"])
        prev = last_for_type.get(t)
        if prev is None:
            keep.append(m)
            last_for_type[t] = m
            continue
        gap = m["absolute_timestamp"] - prev["absolute_timestamp"]
        if gap >= SAME_TYPE_MIN_GAP_SEC:
            keep.append(m)
            last_for_type[t] = m
            continue
        # Too close to the previous match of the same type — keep the higher-confidence one.
        if m["confidence"] > prev["confidence"]:
            keep.remove(prev)
            unmatched.append({
                "scene_index": scene_index,
                "action_type": prev["action_type"],
                "reason": f"same-type gap < {SAME_TYPE_MIN_GAP_SEC:.1f}s",
            })
            used_sound_counts[prev["sound_id"]] = max(
                0, used_sound_counts.get(prev["sound_id"], 0) - 1,
            )
            keep.append(m)
            last_for_type[t] = m
        else:
            unmatched.append({
                "scene_index": scene_index,
                "action_type": m["action_type"],
                "reason": f"same-type gap < {SAME_TYPE_MIN_GAP_SEC:.1f}s",
            })
            used_sound_counts[m["sound_id"]] = max(
                0, used_sound_counts.get(m["sound_id"], 0) - 1,
            )
    return keep


def _enforce_density_cap(
    scene_sfx: list[dict[str, Any]],
    unmatched: list[dict[str, Any]],
    used_sound_counts: dict[int, int],
    scene_index: int,
) -> list[dict[str, Any]]:
    if len(scene_sfx) <= MAX_SFX_PER_SCENE:
        return scene_sfx
    by_confidence = sorted(scene_sfx, key=lambda m: m["confidence"], reverse=True)
    kept = by_confidence[:MAX_SFX_PER_SCENE]
    dropped = by_confidence[MAX_SFX_PER_SCENE:]
    for d in dropped:
        unmatched.append({
            "scene_index": scene_index,
            "action_type": d["action_type"],
            "reason": f"scene density limit ({MAX_SFX_PER_SCENE} SFX max)",
        })
        used_sound_counts[d["sound_id"]] = max(
            0, used_sound_counts.get(d["sound_id"], 0) - 1,
        )
    return sorted(kept, key=lambda m: m["absolute_timestamp"])


def _apply_director_constraints(
    analysis: dict[str, Any],
    scene_times: dict[int, tuple[float, float]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Drop SFX actions that fall inside ``silent_regions`` and cap the total
    SFX count to ``global_constraints.max_sfx_count`` (keep highest confidence).

    Returns ``(filtered_analysis, director_stats, dropped_records)``.
    ``dropped_records`` are pre-built unmatched entries the caller appends to
    its unmatched list."""
    strategy = analysis.get("strategy") or {}
    silent_regions = strategy.get("silent_regions") or []
    max_total = int((strategy.get("global_constraints") or {}).get("max_sfx_count") or 0)
    if not strategy or (not silent_regions and max_total <= 0):
        return analysis, {"silent_dropped": 0, "budget_dropped": 0,
                          "sfx_in_budget": 0, "max_sfx_count": max_total}, []

    flat: list[tuple[int, float, dict[str, Any]]] = []
    for scene in analysis.get("scenes") or []:
        si = int(scene.get("scene_index", -1))
        scene_start = scene_times.get(si, (0.0, 0.0))[0]
        for action in scene.get("actions") or []:
            ts_in_scene = float(action.get("timestamp_in_scene") or 0.0)
            abs_ts = scene_start + ts_in_scene
            flat.append((si, abs_ts, action))

    dropped_records: list[dict[str, Any]] = []
    surviving: list[tuple[int, float, dict[str, Any]]] = []
    silent_dropped = 0
    for si, abs_ts, action in flat:
        hit = next(
            (sr for sr in silent_regions
             if float(sr.get("start_sec") or 0.0) <= abs_ts <= float(sr.get("end_sec") or 0.0)),
            None,
        )
        if hit is not None:
            silent_dropped += 1
            dropped_records.append({
                "scene_index": si,
                "action_type": str(action.get("type", "")),
                "reason": f"silent region {float(hit.get('start_sec') or 0):.1f}\u2013"
                          f"{float(hit.get('end_sec') or 0):.1f}s ({hit.get('reason','')})",
            })
            continue
        surviving.append((si, abs_ts, action))

    budget_dropped = 0
    if max_total > 0 and len(surviving) > max_total:
        ranked = sorted(
            surviving,
            key=lambda x: float(x[2].get("confidence") or 0.0),
            reverse=True,
        )
        kept = set(id(x[2]) for x in ranked[:max_total])
        for si, _abs_ts, action in ranked[max_total:]:
            budget_dropped += 1
            dropped_records.append({
                "scene_index": si,
                "action_type": str(action.get("type", "")),
                "reason": f"director SFX budget exceeded (max {max_total})",
            })
        surviving = [(si, ts, a) for (si, ts, a) in surviving if id(a) in kept]

    kept_ids = {id(a) for _si, _ts, a in surviving}

    filtered_scenes: list[dict[str, Any]] = []
    for scene in analysis.get("scenes") or []:
        new_scene = dict(scene)
        new_scene["actions"] = [
            a for a in (scene.get("actions") or []) if id(a) in kept_ids
        ]
        filtered_scenes.append(new_scene)

    filtered_analysis = dict(analysis)
    filtered_analysis["scenes"] = filtered_scenes

    return (
        filtered_analysis,
        {
            "silent_dropped": silent_dropped,
            "budget_dropped": budget_dropped,
            "sfx_in_budget": len(surviving),
            "max_sfx_count": max_total,
        },
        dropped_records,
    )


def match_analysis(
    analysis: dict[str, Any],
    qdrant_client: QdrantClient,
    preset_name: str = "dramatic",
    scenes: list[Scene] | None = None,
    density: float | None = None,
    verify: bool = True,
) -> dict[str, Any]:
    """Resolve every Gemini-detected action and per-scene ambience to a concrete sound."""
    preset = get_preset(preset_name)
    if density is None:
        density = preset.density
    if scenes is None:
        video_path = Path(analysis.get("video_path", ""))
        if not video_path.is_file():
            raise FileNotFoundError(
                f"Cannot re-detect scenes: analysis['video_path'] = {video_path!r} not found."
            )
        scenes = detect_scenes(video_path)
    scene_times: dict[int, tuple[float, float]] = {
        s.index: (s.start_sec, s.end_sec) for s in scenes
    }

    matches: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    used_sound_counts: dict[int, int] = {}
    last_scene_sfx_ids: set[int] = set()
    last_scene_ambient_id: int | None = None
    stats: dict[str, Any] = {
        "whitelist_fallback": 0,
        "duration_fallback": 0,
        "min_score_rejected": 0,
        "action_override_used": 0,
        "fallback_tier_used": 0,
        "global_pool_used": 0,
    }

    analysis, director_stats, dropped_records = _apply_director_constraints(
        analysis, scene_times,
    )
    unmatched.extend(dropped_records)
    if director_stats["silent_dropped"] or director_stats["budget_dropped"]:
        logger.info(
            "Director constraints: %d action(s) dropped for silent regions, "
            "%d for budget cap (max=%d)",
            director_stats["silent_dropped"],
            director_stats["budget_dropped"],
            director_stats["max_sfx_count"],
        )
    stats["director"] = director_stats

    try:
        tier_counts = count_by_tier(qdrant_client)
    except Exception as exc:
        logger.warning("Tier count lookup failed (%s); routing falls back to global.", exc)
        tier_counts = {}
    missing = [t for t, c in tier_counts.items() if c == 0]
    if missing:
        logger.warning(
            "Tiers with no indexed sounds: %s. Presets requesting these tiers "
            "will use their fallback / global pool.",
            ", ".join(sorted(missing)),
        )
    logger.info(
        "Tier counts in collection: %s",
        ", ".join(f"{k}={v}" for k, v in sorted(tier_counts.items())) or "(unknown)",
    )

    for scene_resp in analysis.get("scenes", []) or []:
        scene_index = int(scene_resp.get("scene_index", -1))
        if scene_index not in scene_times:
            logger.warning(
                "Scene index %d present in analysis but not in re-detected scenes; skipping",
                scene_index,
            )
            continue
        scene_start, scene_end = scene_times[scene_index]
        scene_environment = str(scene_resp.get("environment", "") or "")

        # ---- SFX matching for this scene ----
        scene_sfx_matches: list[dict[str, Any]] = []
        for action in scene_resp.get("actions", []) or []:
            ts_in_scene = float(action.get("timestamp_in_scene") or 0.0)
            absolute_ts = scene_start + ts_in_scene
            sfx_action = {
                **action,
                "layer": "sfx",
                "_scene_environment": scene_environment,
            }
            candidates = match_action_to_sound(
                sfx_action, qdrant_client, preset, used_sound_counts, stats,
                tier_counts=tier_counts,
            )
            if not candidates:
                unmatched.append({
                    "scene_index": scene_index,
                    "action_type": action.get("type", ""),
                    "reason": f"no candidates above {SFX_MIN_SCORE} score",
                })
                continue
            # Avoid reusing the same sound from the previous scene if there's an alternative.
            chosen = next(
                (c for c in candidates if c["id"] not in last_scene_sfx_ids),
                candidates[0],
            )
            best = _candidate_to_match_payload(chosen, sfx_action)
            sound_value = action.get("sound_value")
            scene_sfx_matches.append(_build_match_entry(
                scene_index=scene_index,
                scene_start=scene_start,
                scene_end=scene_end,
                absolute_ts=absolute_ts,
                action_type=str(action.get("type", "")),
                description=str(action.get("description", "")),
                intensity=str(action.get("intensity", "medium")),
                confidence=float(action.get("confidence") or 0.0),
                layer="sfx",
                best=best,
                gemini_sound_value=(
                    float(sound_value) if sound_value is not None else None
                ),
            ))
            used_sound_counts[chosen["id"]] = used_sound_counts.get(chosen["id"], 0) + 1

        # Density rules.
        scene_sfx_matches = _enforce_same_type_gap(
            scene_sfx_matches, unmatched, used_sound_counts, scene_index,
        )
        scene_sfx_matches = _enforce_density_cap(
            scene_sfx_matches, unmatched, used_sound_counts, scene_index,
        )
        matches.extend(scene_sfx_matches)
        last_scene_sfx_ids = {m["sound_id"] for m in scene_sfx_matches}

        # ---- Ambient layer ----
        ambient_desc = scene_resp.get("ambient_suggestion")
        if ambient_desc:
            ambient_action: dict[str, Any] = {
                "type": "ambient",
                "description": str(ambient_desc),
                "layer": "ambient",
                "_scene_environment": scene_environment,
                "intensity": "soft",
            }
            ambient_candidates = match_action_to_sound(
                ambient_action, qdrant_client, preset, used_sound_counts, stats,
                tier_counts=tier_counts,
            )
            if not ambient_candidates:
                unmatched.append({
                    "scene_index": scene_index,
                    "action_type": "ambient",
                    "reason": f"no candidates above {AMBIENT_MIN_SCORE} score",
                })
            else:
                chosen = ambient_candidates[0]
                # Force variety vs. the previous scene's ambience.
                if (
                    last_scene_ambient_id is not None
                    and chosen["id"] == last_scene_ambient_id
                    and len(ambient_candidates) > 1
                ):
                    chosen = ambient_candidates[1]
                best = _candidate_to_match_payload(chosen, ambient_action)
                matches.append(_build_match_entry(
                    scene_index=scene_index,
                    scene_start=scene_start,
                    scene_end=scene_end,
                    absolute_ts=scene_start,
                    action_type="ambient",
                    description=str(ambient_desc),
                    intensity="soft",
                    confidence=1.0,
                    layer="ambient",
                    best=best,
                ))
                used_sound_counts[chosen["id"]] = used_sound_counts.get(chosen["id"], 0) + 1
                last_scene_ambient_id = int(chosen["id"])

    matches.sort(key=lambda m: m["absolute_timestamp"])

    routing_breakdown: dict[str, int] = {}
    routing_by_tier: dict[str, int] = {}
    for m in matches:
        if m.get("layer") == "ambient":
            continue
        path = str(m.get("routing_path") or "global")
        routing_breakdown[path] = routing_breakdown.get(path, 0) + 1
        tier = str(m.get("matched_tier") or "global")
        routing_by_tier[tier] = routing_by_tier.get(tier, 0) + 1
    stats["routing_breakdown"] = routing_breakdown
    stats["routing_by_tier"] = routing_by_tier
    stats["tier_counts"] = tier_counts

    strategy = analysis.get("strategy") or {}
    anchor_moments = strategy.get("anchor_moments") or []
    anchor_timestamps = [
        float(am.get("timestamp_sec") or 0.0) for am in anchor_moments
    ]
    recipes_info: list[dict[str, Any]] = []
    if anchor_timestamps or any(
        moment_type_for_action(str(m.get("action_type") or "")) for m in matches
    ):
        recipe_matches: list[dict[str, Any]] = []
        replaced_ids: set[int] = set()
        for m in matches:
            if m.get("layer") != "sfx":
                continue
            ts = float(m.get("absolute_timestamp") or 0.0)
            is_anchor = (
                any(abs(ts - at) < 0.8 for at in anchor_timestamps)
                if anchor_timestamps else False
            )
            if not is_anchor:
                continue
            layers = build_recipe_layers(m, preset_name, qdrant_client)
            if layers:
                recipe_matches.extend(layers)
                replaced_ids.add(id(m))
                recipes_info.append({
                    "anchor_timestamp": ts,
                    "action_type": m.get("action_type"),
                    "recipe_name": layers[0].get("recipe_name"),
                    "layer_count": len(layers),
                })
        if recipe_matches:
            matches = [m for m in matches if id(m) not in replaced_ids] + recipe_matches
            matches.sort(key=lambda x: float(x.get("absolute_timestamp") or 0.0))
    stats["recipes"] = {
        "applied": len(recipes_info),
        "layer_total": sum(int(r.get("layer_count") or 0) for r in recipes_info),
        "details": recipes_info,
    }

    music = select_music(analysis, preset_name, qdrant_client)

    match_plan: dict[str, Any] = {
        "video_path": analysis.get("video_path", ""),
        "preset": analysis.get("preset", ""),
        "duration_sec": analysis.get("duration_sec", 0.0),
        "music": music,
        "matches": matches,
        "unmatched_actions": unmatched,
        "match_stats": stats,
    }

    # Verification pass: drop concretely-wrong picks BEFORE density selection so
    # the density fill works with verified sounds only. Fails open on any error.
    if verify:
        match_plan = verify_matches(match_plan, analysis)
        match_plan = research_dropped_anchors(
            match_plan, analysis, qdrant_client, preset_name,
        )
        stats["verification"] = match_plan.get("verification_stats", {})

    energy_curve = strategy.get("energy_curve") or []
    match_plan = apply_selectivity(
        match_plan, strategy, energy_curve,
        density=density,
        duration_sec=analysis.get("duration_sec"),
    )
    stats["selectivity"] = match_plan.get("selectivity_stats", {})

    return match_plan


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Match Gemini-detected actions to concrete sounds via Qdrant.",
    )
    parser.add_argument("video_path", type=Path, help="Path to input video file.")
    parser.add_argument(
        "--preset",
        default="tiktok_viral",
        choices=list(PRESETS.keys()),
        help="Mood preset.",
    )
    parser.add_argument(
        "--use-cached-analysis",
        action="store_true",
        help="Require a cached analysis.json; fail if missing instead of calling Gemini.",
    )
    args = parser.parse_args()

    if not args.video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {args.video_path}")

    analysis_path = TEMP_DIR / args.video_path.stem / "analysis.json"
    if analysis_path.is_file():
        logger.info("Loading cached analysis from %s", analysis_path)
        analysis = json.loads(analysis_path.read_text())
    elif args.use_cached_analysis:
        raise RuntimeError(
            f"--use-cached-analysis specified but no cache at {analysis_path}. "
            f"Run `python -m src.analysis.gemini_client {args.video_path} "
            f"--preset {args.preset}` first."
        )
    else:
        logger.info("No cached analysis; running Gemini pipeline")
        analysis = analyze_video(args.video_path, preset=args.preset)

    client = get_qdrant_client()
    plan = match_analysis(analysis, client, preset_name=args.preset)

    onsets = detect_onsets_from_video(args.video_path)
    plan = snap_to_onsets(plan, onsets)

    output_dir = TEMP_DIR / args.video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "match_plan.json"
    out_path.write_text(json.dumps(plan, indent=2))

    matches = plan["matches"]
    unmatched = plan["unmatched_actions"]
    sfx_matches = [m for m in matches if m["layer"] == "sfx"]
    ambient_matches = [m for m in matches if m["layer"] == "ambient"]
    total_sfx_actions = len(sfx_matches) + len(
        [u for u in unmatched if u["action_type"] != "ambient"]
    )
    n_scenes = len(analysis.get("scenes", []))
    avg_score = (
        sum(m["match_score"] for m in matches) / len(matches) if matches else 0.0
    )

    print(
        f"Matched {len(sfx_matches)}/{total_sfx_actions} actions across {n_scenes} scenes"
    )
    print(f"{len(ambient_matches)} ambient layers")
    print(f"Avg match score: {avg_score:.2f}")

    snap_stats = plan.get("snap_stats", {})
    if snap_stats:
        print(
            f"Snap stats: {snap_stats['snapped']}/{snap_stats['total_sfx']} "
            f"SFX snapped to onsets (avg distance: {snap_stats['avg_distance_ms']:.0f}ms)"
        )

    match_stats = plan.get("match_stats", {})
    if match_stats:
        print(
            f"Filter stats: whitelist fall-backs {match_stats.get('whitelist_fallback', 0)}, "
            f"duration fall-backs {match_stats.get('duration_fallback', 0)}, "
            f"min-score 0.4 rejects {match_stats.get('min_score_rejected', 0)}"
        )

    if unmatched:
        print(f"Unmatched: {len(unmatched)}")
        for u in unmatched:
            print(f"  - scene {u['scene_index']}: {u['action_type']} ({u['reason']})")

    print("\nFirst 3 matches:")
    for m in matches[:3]:
        print(
            f"  [{m['absolute_timestamp']:.2f}s] {m['action_type']:<22} "
            f"-> {m['sound_name']}"
        )
        print(
            f"     score={m['match_score']:.2f}, reranked={m['reranked_score']:.2f}, "
            f"layer={m['layer']}, {m['rerank_reason']}"
        )
        if m.get("snapped") is True:
            print(
                f"     timing: {m['original_timestamp']:.3f}s -> "
                f"{m['absolute_timestamp']:.3f}s (snapped {m['snap_distance_ms']:+.0f}ms)"
            )
        elif m.get("snapped") is False:
            print(
                f"     timing: {m['original_timestamp']:.3f}s "
                f"(not snapped — no onset within window)"
            )

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    _main()
