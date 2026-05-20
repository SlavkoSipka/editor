from __future__ import annotations

import bisect
from typing import Any

from src.config import ONSET_SYNC_WINDOW_MS
from src.utils.logger import get_logger

logger = get_logger("timing_adjuster")


_TRANSITION_TYPE_KEYWORDS = ("transition",)
_TRANSITION_EXACT_TYPES = {"scene_transition", "video_transition"}


def _is_transition(action_type: str) -> bool:
    if not action_type:
        return False
    lower = action_type.lower()
    if lower in _TRANSITION_EXACT_TYPES:
        return True
    return any(kw in lower for kw in _TRANSITION_TYPE_KEYWORDS)


def _nearest_onset(onsets: list[float], ts: float) -> float | None:
    if not onsets:
        return None
    idx = bisect.bisect_left(onsets, ts)
    candidates: list[float] = []
    if idx < len(onsets):
        candidates.append(onsets[idx])
    if idx > 0:
        candidates.append(onsets[idx - 1])
    return min(candidates, key=lambda o: abs(o - ts))


def snap_to_onsets(
    match_plan: dict[str, Any],
    onsets: list[float],
    window_ms: int = ONSET_SYNC_WINDOW_MS,
) -> dict[str, Any]:
    """Snap each SFX match's `absolute_timestamp` to the nearest librosa onset
    within ±window_ms. Ambient layers are left untouched."""
    sorted_onsets = sorted(float(o) for o in onsets)

    total_sfx = 0
    snapped_count = 0
    snap_distances_ms: list[float] = []

    for match in match_plan.get("matches", []) or []:
        if match.get("layer") != "sfx":
            continue
        total_sfx += 1

        original_ts = float(match["absolute_timestamp"])
        match["original_timestamp"] = original_ts
        match["snapped"] = False
        match["snap_distance_ms"] = None

        # Visual transitions should fire at Gemini's estimated cut point, not at
        # an unrelated audio transient.
        if _is_transition(str(match.get("action_type", ""))):
            continue

        nearest = _nearest_onset(sorted_onsets, original_ts)
        if nearest is None:
            continue

        distance_ms = (nearest - original_ts) * 1000.0
        if abs(distance_ms) <= window_ms:
            match["absolute_timestamp"] = float(nearest)
            match["snapped"] = True
            match["snap_distance_ms"] = float(distance_ms)
            snapped_count += 1
            snap_distances_ms.append(abs(distance_ms))

    avg_distance_ms = (
        sum(snap_distances_ms) / len(snap_distances_ms) if snap_distances_ms else 0.0
    )
    match_plan["snap_stats"] = {
        "total_sfx": total_sfx,
        "snapped": snapped_count,
        "avg_distance_ms": avg_distance_ms,
    }
    logger.info(
        "Snapped %d/%d SFX to onsets (avg distance %.1fms, window %dms)",
        snapped_count, total_sfx, avg_distance_ms, window_ms,
    )
    return match_plan


_HIGH_IMPACT_KEYWORDS = (
    "transition", "drop", "logo", "reveal", "hit", "boom",
    "whoosh", "sting", "impact", "riser",
)
_DROP_WINDOW_SEC = 1.0
_BEAT_SNAP_WINDOW_MS = 400


def _deserves_beat_snap(
    match: dict[str, Any],
    strategy: dict[str, Any] | None,
) -> tuple[bool, bool, bool, bool]:
    """Return ``(deserves, is_hook, is_outro, is_drop)`` based on the strategy
    and the action's own metadata."""
    abs_ts = float(match.get("absolute_timestamp") or 0.0)
    action_type = str(match.get("action_type") or "").lower()
    intensity = str(match.get("intensity") or "medium").lower()

    hook_end = 0.0
    outro_start = float("inf")
    drops: list[float] = []
    if strategy:
        hook_end = float((strategy.get("hook") or {}).get("end_sec") or 0.0)
        outro_start = float((strategy.get("outro") or {}).get("start_sec")
                            or float("inf"))
        drops = [
            float(d.get("timestamp_sec") or 0.0)
            for d in (strategy.get("drops") or [])
        ]

    is_hook = abs_ts < hook_end
    is_outro = abs_ts >= outro_start
    is_drop = any(abs(abs_ts - d) < _DROP_WINDOW_SEC for d in drops)
    is_high_impact = any(kw in action_type for kw in _HIGH_IMPACT_KEYWORDS)
    is_sharp = intensity == "sharp"

    deserves = is_hook or is_outro or is_drop or (is_high_impact and is_sharp)
    return deserves, is_hook, is_outro, is_drop


def snap_to_beats(
    match_plan: dict[str, Any],
    beats: list[float],
    strong_beats: list[float] | None = None,
    strategy: dict[str, Any] | None = None,
    window_ms: int = _BEAT_SNAP_WINDOW_MS,
) -> dict[str, Any]:
    """Snap high-impact SFX to the nearest music beat (drops/outro prefer
    *strong* beats). Onset-snapped SFX are kept as-is — onset alignment to the
    actual visual event is more precise than beat alignment.

    Modifies ``match_plan`` in place. Each beat-snapped match gains:
    - ``beat_snapped`` = True
    - ``beat_snap_distance_ms`` (signed, ms)
    - ``snapped_to`` = ``"strong_beat"`` or ``"beat"``
    - ``original_timestamp_before_beat`` (pre-snap absolute timestamp)"""
    sorted_beats = sorted(float(b) for b in (beats or []))
    sorted_strong = sorted(float(b) for b in (strong_beats or []))

    if not sorted_beats:
        match_plan["beat_snap_stats"] = {
            "total_snapped": 0,
            "to_strong": 0,
            "to_regular": 0,
            "window_ms": window_ms,
            "skipped_reason": "no_beats",
        }
        logger.info("No beats available; skipping beat snap.")
        return match_plan

    snapped = 0
    strong_count = 0
    regular_count = 0
    distances_ms: list[float] = []

    for match in match_plan.get("matches", []) or []:
        if match.get("layer") != "sfx":
            continue
        if match.get("snapped"):
            continue  # onset alignment wins

        deserves, _is_hook, is_outro, is_drop = _deserves_beat_snap(match, strategy)
        if not deserves:
            continue

        abs_ts = float(match["absolute_timestamp"])
        use_strong = (is_drop or is_outro) and sorted_strong
        candidate_beats = sorted_strong if use_strong else sorted_beats

        nearest = min(candidate_beats, key=lambda b: abs(b - abs_ts))
        distance_ms = (nearest - abs_ts) * 1000.0
        if abs(distance_ms) > window_ms:
            continue

        match["original_timestamp_before_beat"] = abs_ts
        match["absolute_timestamp"] = float(nearest)
        match["beat_snapped"] = True
        match["beat_snap_distance_ms"] = float(distance_ms)
        match["snapped_to"] = "strong_beat" if use_strong else "beat"
        snapped += 1
        distances_ms.append(abs(distance_ms))
        if use_strong:
            strong_count += 1
        else:
            regular_count += 1

    avg_distance_ms = sum(distances_ms) / len(distances_ms) if distances_ms else 0.0
    match_plan["beat_snap_stats"] = {
        "total_snapped": snapped,
        "to_strong": strong_count,
        "to_regular": regular_count,
        "window_ms": window_ms,
        "avg_distance_ms": avg_distance_ms,
    }
    logger.info(
        "Beat-snapped %d high-impact SFX (%d strong, %d regular; "
        "avg distance %.1fms, window %dms)",
        snapped, strong_count, regular_count, avg_distance_ms, window_ms,
    )
    return match_plan
