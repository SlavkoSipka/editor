"""Central density control — turns a density value into a target sound count.

This replaces the old "cut below thresholds" pruning with a single
"fill up to a target" model. All density tuning lives here.
"""

from __future__ import annotations

from typing import Any

from src.utils.logger import get_logger

logger = get_logger("density")

# Rescue floor: a 'skip'-tier action must beat this to be pulled in when we're
# still under target (don't rescue genuinely terrible matches).
RESCUE_MIN_VALUE = 0.25

# Seconds-per-SFX at the extremes of the density slider.
SEC_PER_SFX_SPARSE = 11.0   # density 0.0
SEC_PER_SFX_DENSE = 3.5     # density 1.0
SEC_PER_SFX_HARD_FLOOR = 2.5


def _ts(scored_action: Any) -> float:
    return float(scored_action.action.get("absolute_timestamp", 0) or 0.0)


def compute_target_sound_count(
    duration_sec: float,
    density: float,            # 0.0 - 1.0
    energy_avg: float = 0.5,   # average energy from director's curve
) -> dict:
    """Compute the target number of SFX for a video.

    Returns a dict with ``target`` (ideal), ``min`` (hard floor), ``max``
    (hard ceiling) and ``sec_per_sfx`` for logging.

    - Base rate: at density=0.5, aim for ~1 SFX per 6s.
    - density scales this from ~1 per 11s (0.0) to ~1 per 3.5s (1.0).
    - energy nudges it slightly (high-energy videos tolerate a few more).
    """
    duration_sec = max(3.0, float(duration_sec or 0.0))
    density = max(0.0, min(1.0, float(density)))

    span = SEC_PER_SFX_SPARSE - SEC_PER_SFX_DENSE
    sec_per_sfx = SEC_PER_SFX_SPARSE - (density * span)
    # Energy nudge: high energy shortens the interval up to ~15%.
    sec_per_sfx *= (1.0 - (energy_avg - 0.5) * 0.3)
    sec_per_sfx = max(SEC_PER_SFX_HARD_FLOOR, sec_per_sfx)

    target = round(duration_sec / sec_per_sfx)
    floor = max(2, round(target * 0.6))
    ceiling = max(floor, round(target * 1.4))

    result = {
        "target": target,
        "min": floor,
        "max": ceiling,
        "sec_per_sfx": round(sec_per_sfx, 1),
    }
    logger.info(
        "Density target: %d SFX (min %d, max %d) for %.0fs @ density %.2f "
        "(~1 per %.1fs)",
        target, floor, ceiling, duration_sec, density, sec_per_sfx,
    )
    return result


def fill_to_target(
    scored_actions: list,      # list of ScoredAction, any order
    target: dict,              # output of compute_target_sound_count
    min_gap_sec: float = 0.5,
    anchor_protect_sec: float = 0.35,
) -> tuple[list, list]:
    """Select sounds by FILLING UP to the target count, highest value first,
    respecting spacing only when it doesn't push us below target.

    Returns ``(kept, pruned)`` lists of ScoredAction.

    Strategy:
      1. Always keep all 'anchor' tier actions (they earned it).
      2. Add 'accent' actions by descending value until we reach target.
      3. If still short, pull in 'skip' actions above RESCUE_MIN_VALUE.
      4. Cap to max.
      5. Spacing pass: drop the lower-value of a too-close pair, but never
         below min. Two anchors may sit close together (deliberate layering).
      6. Final safety: if below min, restore the best pruned actions.
    """
    anchors = [sa for sa in scored_actions if sa.tier == "anchor"]
    accents = [sa for sa in scored_actions if sa.tier == "accent"]
    rest = [sa for sa in scored_actions if sa.tier == "skip"]

    target_n = max(0, int(target.get("target", 0)))
    min_n = max(0, int(target.get("min", 0)))
    max_n = max(min_n, int(target.get("max", target_n)))

    # 1 + 2: anchors always, then accents by value up to target.
    kept = list(anchors)
    for sa in sorted(accents, key=lambda s: s.value_score, reverse=True):
        if len(kept) >= target_n:
            break
        kept.append(sa)

    # 3: rescue from 'skip' if still under target.
    if len(kept) < target_n:
        for sa in sorted(rest, key=lambda s: s.value_score, reverse=True):
            if len(kept) >= target_n:
                break
            if sa.value_score >= RESCUE_MIN_VALUE:
                kept.append(sa)

    # 4: cap to max (keep anchors + highest-value accents).
    if len(kept) > max_n:
        kept.sort(key=lambda s: (s.tier == "anchor", s.value_score), reverse=True)
        kept = kept[:max_n]

    # 5: spacing pass.
    kept.sort(key=_ts)
    spaced: list = []
    pruned: list = []
    for sa in kept:
        if not spaced:
            spaced.append(sa)
            continue
        prev = spaced[-1]
        gap = _ts(sa) - _ts(prev)
        both_anchors = sa.tier == "anchor" and prev.tier == "anchor"
        min_required = 0.0 if both_anchors else min_gap_sec
        if gap >= min_required:
            spaced.append(sa)
            continue
        # Too close. Only resolve by dropping if we can stay >= min.
        if len(spaced) < min_n:
            spaced.append(sa)  # can't afford to drop — relax spacing
            continue
        if prev.tier == "anchor" and sa.tier != "anchor":
            pruned.append(sa)
        elif sa.tier == "anchor" and prev.tier != "anchor":
            pruned.append(prev)
            spaced[-1] = sa
        elif sa.value_score > prev.value_score:
            pruned.append(prev)
            spaced[-1] = sa
        else:
            pruned.append(sa)

    kept = spaced

    # 6: final safety — restore best pruned until we reach min.
    if len(kept) < min_n and pruned:
        pruned.sort(key=lambda s: s.value_score, reverse=True)
        while len(kept) < min_n and pruned:
            kept.append(pruned.pop(0))
        kept.sort(key=_ts)

    kept_ids = {id(sa) for sa in kept}
    all_pruned = [sa for sa in scored_actions if id(sa) not in kept_ids]

    logger.info(
        "Fill-to-target: kept %d (target %d, min %d), pruned %d",
        len(kept), target_n, min_n, len(all_pruned),
    )
    _ = anchor_protect_sec  # reserved for future anchor-window logic
    return kept, all_pruned
