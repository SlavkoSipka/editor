"""Selectivity engine — prune SFX to a curated, high-value set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.utils.logger import get_logger

logger = get_logger("selectivity")


# ─────────────────────────────────────────────────────────────
# TUNING CONSTANTS
# ─────────────────────────────────────────────────────────────

# Action types that almost ALWAYS add value (anchors)
HIGH_VALUE_TYPES = {
    "transition", "scene_transition", "video_transition", "whoosh",
    "drop", "impact", "logo", "logo_reveal", "logo_appear", "logo_settle",
    "reveal", "graphic_reveal", "digital_reveal", "hook",
    "punchline", "freeze_frame", "beat_drop",
}

# Action types that are MEDIUM value — depends on context
MEDIUM_VALUE_TYPES = {
    "click", "tap", "screen_tap", "ui_click", "button",
    "machine_start", "machine_stop", "latch_click",
    "scissors_cut", "cut", "snip", "glass", "metal",
    "notification", "swipe",
}

# Action types that are LOW value — usually skippable
LOW_VALUE_TYPES = {
    "footstep", "footsteps", "footstep_wood", "footstep_rock", "footstep_concrete",
    "clothes_rustle", "fabric_rustle", "rustle", "body_movement",
    "subtle_motion", "ambient_motion", "breathing", "background",
}

# Tier thresholds (looser — too-strict thresholds left videos almost SFX-less)
ANCHOR_THRESHOLD = 0.65
ACCENT_THRESHOLD = 0.38

# Minimum gap between consecutive SFX (seconds)
MIN_GAP_SEC = 0.6

# Protect a window around anchors from accent SFX (tightened — 0.8s was eating
# too many otherwise-good accents)
ANCHOR_PROTECT_SEC = 0.4

# When everything would be skipped, rescue the strongest survivor above this score
RESCUE_FLOOR = 0.35

# Minimum-density rescue: target roughly 1 SFX per this many seconds, with at
# least MIN_TARGET_FLOOR survivors (never exceeds strategy.max_sfx_count).
MIN_TARGET_SECONDS_PER_SFX = 7
MIN_TARGET_FLOOR = 4
MIN_TARGET_RESCUE_FLOOR_SCORE = 0.30

# A genuinely bad library match cannot become an anchor (better no anchor than
# blasting a wrong sound at the viewer).
WEAK_MATCH_THRESHOLD = 0.40
WEAK_MATCH_ANCHOR_CAP = 0.55


# ─────────────────────────────────────────────────────────────
# DATA TYPES
# ─────────────────────────────────────────────────────────────


@dataclass
class ScoredAction:
    action: dict          # the original match dict
    value_score: float    # 0.0 - 1.0, our estimate of sound design value
    tier: str             # "anchor" | "accent" | "skip"
    reasons: list[str]    # human-readable explanation


# ─────────────────────────────────────────────────────────────
# VALUE SCORING
# ─────────────────────────────────────────────────────────────


def score_action_value(
    action: dict,
    strategy: dict,
    music_energy_at: float = 0.5,
    same_type_index: int = 0,
) -> ScoredAction:
    """Estimate the sound-design VALUE of a single SFX action."""
    reasons: list[str] = []
    score = 0.5

    action_type = str(action.get("action_type", action.get("type", "")) or "").lower()
    intensity = str(action.get("intensity", "medium") or "medium").lower()
    confidence = float(action.get("confidence") or 0.5)
    match_score = float(action.get("reranked_score", action.get("match_score", 0.5)) or 0.5)
    abs_ts = float(action.get("absolute_timestamp") or 0.0)

    # 0. Blend Gemini's own sound_value rating (the model is asked to rate
    # how much a sound here genuinely helps).
    gemini_value = action.get("gemini_sound_value")
    if gemini_value is None:
        gemini_value = action.get("sound_value")
    if gemini_value is not None:
        try:
            gv = float(gemini_value)
        except (TypeError, ValueError):
            gv = None
        if gv is not None:
            score = 0.75 * score + 0.25 * gv
            reasons.append(f"gemini sound_value {gv:.2f}")

    # 1. Base value from action type category
    if any(t in action_type for t in HIGH_VALUE_TYPES):
        score += 0.30
        reasons.append("high-value type")
    elif any(t in action_type for t in MEDIUM_VALUE_TYPES):
        score += 0.05
        reasons.append("medium-value type")
    elif any(t in action_type for t in LOW_VALUE_TYPES):
        score -= 0.25
        reasons.append("low-value type")

    # 2. Anchor moment bonus
    for am in strategy.get("anchor_moments") or []:
        if abs(abs_ts - float(am.get("timestamp_sec", -999) or -999)) < 0.8:
            score += 0.35
            reasons.append(f"anchor moment ({am.get('type','')})")
            break

    # 3. Hook / drop / outro position bonus
    hook = strategy.get("hook") or {}
    if float(hook.get("start_sec", 0) or 0) <= abs_ts <= float(hook.get("end_sec", 0) or 0):
        score += 0.20
        reasons.append("in hook region")

    for drop in strategy.get("drops") or []:
        if abs(abs_ts - float(drop.get("timestamp_sec", -999) or -999)) < 1.0:
            score += 0.25
            reasons.append("in drop moment")
            break

    outro = strategy.get("outro") or {}
    if float(outro.get("start_sec", 1e9) or 1e9) <= abs_ts <= float(outro.get("end_sec", 1e9) or 1e9):
        if any(t in action_type for t in ("logo", "reveal", "sting", "impact")):
            score += 0.20
            reasons.append("outro logo moment")

    # 4. Match quality factor
    if match_score < 0.42:
        score -= 0.15
        reasons.append(f"weak library match ({match_score:.2f})")
    elif match_score > 0.6:
        score += 0.10
        reasons.append("strong library match")

    # 5. Intensity factor
    if intensity == "sharp":
        score += 0.08
    elif intensity == "soft":
        score -= 0.10
        reasons.append("soft intensity (less impactful)")

    # 6. Music masking penalty
    if music_energy_at > 0.7 and not any(t in action_type for t in HIGH_VALUE_TYPES):
        score -= 0.15
        reasons.append("music already energetic here (SFX would be masked)")

    # 7. Repetition penalty (diminishing returns)
    if same_type_index > 0:
        penalty = min(0.35, 0.15 * same_type_index)
        score -= penalty
        reasons.append(f"repeat #{same_type_index + 1} of this type (-{penalty:.2f})")

    # 8. Low detection confidence
    if confidence < 0.6:
        score -= 0.10
        reasons.append(f"low detection confidence ({confidence:.2f})")

    score = max(0.0, min(1.0, score))

    # HARD GATE: a sound with a genuinely bad library match cannot be an anchor,
    # no matter how important the moment. A bad sound in a key moment is worse
    # than no sound.
    if match_score < WEAK_MATCH_THRESHOLD:
        if score > WEAK_MATCH_ANCHOR_CAP:
            score = WEAK_MATCH_ANCHOR_CAP
            reasons.append("match too weak to be an anchor — capped")

    if score >= ANCHOR_THRESHOLD:
        tier = "anchor"
    elif score >= ACCENT_THRESHOLD:
        tier = "accent"
    else:
        tier = "skip"

    return ScoredAction(action=action, value_score=score, tier=tier, reasons=reasons)


# ─────────────────────────────────────────────────────────────
# PRUNING
# ─────────────────────────────────────────────────────────────


def _prune_record(sa: ScoredAction, reason: str) -> dict[str, Any]:
    return {
        "action_type": sa.action.get("action_type"),
        "timestamp": sa.action.get("absolute_timestamp"),
        "value_score": round(sa.value_score, 2),
        "reason": reason,
    }


def apply_selectivity(
    match_plan: dict,
    strategy: dict,
    music_energy_curve: list | None = None,
) -> dict:
    """Score every SFX match, then prune to a curated set."""
    matches = match_plan.get("matches", []) or []
    sfx_all = [m for m in matches if m.get("layer") == "sfx"]
    ambient = [m for m in matches if m.get("layer") == "ambient"]

    # Recipe layers are part of a designed anchor unit — never score or prune
    # them individually. They count as ONE group toward the global cap.
    recipe_sfx = [m for m in sfx_all if m.get("recipe_name")]
    sfx = [m for m in sfx_all if not m.get("recipe_name")]

    recipe_group_keys: list[tuple[str, float]] = []
    for m in recipe_sfx:
        key = (
            str(m.get("recipe_name") or ""),
            round(float(m.get("scene_start_sec") or 0.0), 2),
        )
        if key not in recipe_group_keys:
            recipe_group_keys.append(key)
    recipe_group_count = len(recipe_group_keys)

    if not sfx and not recipe_sfx:
        match_plan["selectivity_stats"] = {
            "input_sfx": 0, "kept_sfx": 0, "pruned_sfx": 0,
            "anchors": 0, "accents": 0,
            "recipe_groups": 0, "recipe_layers": 0,
        }
        match_plan["pruned_actions"] = match_plan.get("pruned_actions", [])
        return match_plan

    sfx.sort(key=lambda m: float(m.get("absolute_timestamp") or 0.0))

    type_seen: dict[str, int] = {}
    scored: list[ScoredAction] = []
    for m in sfx:
        atype = str(m.get("action_type", m.get("type", "")) or "").lower()
        base_type = atype.split("_")[0] if "_" in atype else atype
        idx = type_seen.get(base_type, 0)
        type_seen[base_type] = idx + 1

        music_energy = _energy_at(
            music_energy_curve, float(m.get("absolute_timestamp") or 0.0),
        )
        scored.append(score_action_value(m, strategy, music_energy, idx))

    pruned: list[dict[str, Any]] = []
    survivors: list[ScoredAction] = []
    for sa in scored:
        if sa.tier == "skip":
            pruned.append(_prune_record(
                sa,
                "below value threshold: " + ", ".join(sa.reasons),
            ))
        else:
            survivors.append(sa)

    if not survivors:
        # All SFX scored as skip — rescue the single best one if above floor.
        best = max(scored, key=lambda s: s.value_score)
        if best.value_score >= RESCUE_FLOOR:
            pruned = [p for p in pruned if p["timestamp"] != best.action.get("absolute_timestamp")]
            best.tier = "accent"
            best.reasons.append("rescued (would otherwise be SFX-less)")
            survivors = [best]
            logger.warning(
                "Selectivity: every SFX scored as skip — rescued strongest "
                "(value=%.2f, type=%s).",
                best.value_score, best.action.get("action_type"),
            )

    max_count = int(
        ((strategy.get("global_constraints") or {}).get("max_sfx_count") or 8),
    )
    # Recipe groups are always-kept; they count as 1 toward the cap each, so
    # non-recipe survivors share what's left.
    non_recipe_cap = max(0, max_count - recipe_group_count)
    if len(survivors) > non_recipe_cap:
        survivors.sort(
            key=lambda sa: (sa.tier == "anchor", sa.value_score), reverse=True,
        )
        kept = survivors[:non_recipe_cap]
        for sa in survivors[non_recipe_cap:]:
            pruned.append(_prune_record(sa, f"over global cap of {max_count}"))
        survivors = kept

    survivors.sort(key=lambda sa: float(sa.action.get("absolute_timestamp") or 0.0))
    spaced: list[ScoredAction] = []
    for sa in survivors:
        ts = float(sa.action.get("absolute_timestamp") or 0.0)
        if spaced:
            prev = spaced[-1]
            prev_ts = float(prev.action.get("absolute_timestamp") or 0.0)
            gap = ts - prev_ts
            if gap < MIN_GAP_SEC:
                if sa.value_score > prev.value_score and prev.tier != "anchor":
                    pruned.append(_prune_record(
                        prev, f"too close to higher-value SFX (gap {gap:.2f}s)",
                    ))
                    spaced[-1] = sa
                else:
                    pruned.append(_prune_record(
                        sa, f"too close to previous SFX (gap {gap:.2f}s)",
                    ))
                continue
        spaced.append(sa)

    anchor_times = [
        float(sa.action.get("absolute_timestamp") or 0.0)
        for sa in spaced if sa.tier == "anchor"
    ]
    final: list[ScoredAction] = []
    for sa in spaced:
        if sa.tier == "accent":
            ts = float(sa.action.get("absolute_timestamp") or 0.0)
            too_close = any(
                0 < abs(ts - at) < ANCHOR_PROTECT_SEC for at in anchor_times
            )
            if too_close:
                pruned.append(_prune_record(
                    sa, "accent too close to an anchor — protecting anchor's impact",
                ))
                continue
        final.append(sa)

    duration = float(
        strategy.get("duration_sec")
        or match_plan.get("duration_sec")
        or 30.0
    )
    min_target = max(MIN_TARGET_FLOOR, int(duration / MIN_TARGET_SECONDS_PER_SFX))
    min_target = min(min_target, max_count)
    # Recipe groups count as 1 each toward the density target.
    effective_density = len(final) + recipe_group_count

    if effective_density < min_target:
        rescuable = sorted(
            [
                sa for sa in scored
                if sa.tier == "skip" and sa.value_score >= MIN_TARGET_RESCUE_FLOOR_SCORE
            ],
            key=lambda sa: sa.value_score,
            reverse=True,
        )
        needed = min_target - effective_density
        for sa in rescuable:
            if needed <= 0:
                break
            ts = float(sa.action.get("absolute_timestamp") or 0.0)
            if not all(
                abs(ts - float(f.action.get("absolute_timestamp") or 0.0)) >= MIN_GAP_SEC
                for f in final
            ):
                continue
            sa.tier = "accent"
            sa.reasons.append("rescued for min density")
            final.append(sa)
            needed -= 1
            logger.info(
                "Rescued SFX at %.2fs (value %.2f) to meet min density",
                ts, sa.value_score,
            )
        final.sort(key=lambda sa: float(sa.action.get("absolute_timestamp") or 0.0))

        rescued_timestamps = {
            float(sa.action.get("absolute_timestamp") or 0.0)
            for sa in final if "rescued for min density" in sa.reasons
        }
        if rescued_timestamps:
            pruned = [
                p for p in pruned
                if float(p.get("timestamp") or 0.0) not in rescued_timestamps
            ]

    for sa in final:
        sa.action["value_score"] = round(sa.value_score, 2)
        sa.action["value_tier"] = sa.tier
        sa.action["value_reasons"] = sa.reasons

    # Recipe layers are always retained as anchors (already marked).
    for m in recipe_sfx:
        m.setdefault("value_tier", "anchor")
        m.setdefault("value_score", 1.0)
        m.setdefault("value_reasons", ["recipe layer"])

    kept_matches = [sa.action for sa in final] + recipe_sfx
    match_plan["matches"] = sorted(
        kept_matches + ambient,
        key=lambda m: float(m.get("absolute_timestamp") or 0.0),
    )
    existing_pruned = list(match_plan.get("pruned_actions") or [])
    match_plan["pruned_actions"] = existing_pruned + pruned
    match_plan["selectivity_stats"] = {
        "input_sfx": len(sfx_all),
        "kept_sfx": len(kept_matches),
        "pruned_sfx": len(pruned),
        "recipe_groups": recipe_group_count,
        "recipe_layers": len(recipe_sfx),
        "anchors": sum(1 for sa in final if sa.tier == "anchor") + len(recipe_sfx),
        "accents": sum(1 for sa in final if sa.tier == "accent"),
    }

    logger.info(
        "Selectivity: %d SFX in → %d kept (%d anchors, %d accents, "
        "%d recipe layers across %d group(s)), %d pruned",
        len(sfx_all), len(kept_matches),
        match_plan["selectivity_stats"]["anchors"],
        match_plan["selectivity_stats"]["accents"],
        len(recipe_sfx), recipe_group_count,
        len(pruned),
    )
    return match_plan


def _energy_at(curve: list | None, timestamp: float) -> float:
    """Interpolate music energy at a timestamp. Returns 0.5 if no curve."""
    if not curve:
        return 0.5
    first = curve[0]
    last = curve[-1]
    if timestamp <= float(first.get("timestamp_sec", 0) or 0):
        return float(first.get("level", 0.5) or 0.5)
    if timestamp >= float(last.get("timestamp_sec", 0) or 0):
        return float(last.get("level", 0.5) or 0.5)
    for i in range(len(curve) - 1):
        a, b = curve[i], curve[i + 1]
        a_ts = float(a.get("timestamp_sec", 0) or 0)
        b_ts = float(b.get("timestamp_sec", 0) or 0)
        if a_ts <= timestamp <= b_ts and b_ts > a_ts:
            ratio = (timestamp - a_ts) / (b_ts - a_ts)
            a_lvl = float(a.get("level", 0.5) or 0.5)
            b_lvl = float(b.get("level", 0.5) or 0.5)
            return a_lvl + ratio * (b_lvl - a_lvl)
    return 0.5
