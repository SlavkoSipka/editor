"""Objective quality metrics computed from a match plan + strategy."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass


@dataclass
class EvalMetrics:
    anchor_moments_total: int = 0
    anchor_moments_covered: int = 0
    anchor_coverage_pct: float = 0.0

    avg_match_score: float = 0.0
    weak_matches_count: int = 0

    total_sfx: int = 0
    sfx_per_minute: float = 0.0
    density_ok: bool = True

    whitelist_violations: int = 0
    speech_collision_count: int = 0
    silent_region_violations: int = 0

    unique_sounds: int = 0
    most_repeated_count: int = 0
    repetition_ratio: float = 0.0

    recipes_applied: int = 0

    objective_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(
    match_plan: dict, strategy: dict, duration_sec: float,
) -> EvalMetrics:
    """Compute objective metrics from a finished match plan."""
    m = EvalMetrics()

    matches = match_plan.get("matches") or []
    sfx = [x for x in matches if x.get("layer") == "sfx"]

    anchors = strategy.get("anchor_moments") or []
    m.anchor_moments_total = len(anchors)
    covered = 0
    for am in anchors:
        at = float(am.get("timestamp_sec") or -999)
        if any(
            abs(float(s.get("absolute_timestamp") or -1) - at) < 1.0
            for s in sfx
        ):
            covered += 1
    m.anchor_moments_covered = covered
    m.anchor_coverage_pct = (
        round(covered / len(anchors) * 100, 1) if anchors else 100.0
    )

    scores = [
        float(s.get("reranked_score", s.get("match_score", 0)) or 0)
        for s in sfx
    ]
    m.avg_match_score = round(sum(scores) / len(scores), 3) if scores else 0.0
    m.weak_matches_count = sum(1 for s in scores if s < 0.42)

    m.total_sfx = len(sfx)
    m.sfx_per_minute = (
        round(len(sfx) / (duration_sec / 60), 1) if duration_sec else 0.0
    )
    m.density_ok = (
        3 <= m.sfx_per_minute <= 20 if duration_sec > 5 else True
    )

    m.whitelist_violations = int(
        (match_plan.get("match_stats") or {}).get("whitelist_fallback") or 0
    )

    speech_regions = [
        (float(p[0]), float(p[1]))
        for p in (strategy.get("speech_regions") or [])
        if len(p) >= 2
    ]
    for s in sfx:
        ts = float(s.get("absolute_timestamp") or 0)
        in_speech = any(a <= ts <= b for a, b in speech_regions)
        if (
            in_speech
            and s.get("intensity") == "sharp"
            and s.get("value_tier") == "anchor"
        ):
            m.speech_collision_count += 1

    silent = strategy.get("silent_regions") or []
    for s in sfx:
        ts = float(s.get("absolute_timestamp") or 0)
        if any(
            float(sr.get("start_sec") or 0) <= ts <= float(sr.get("end_sec") or 0)
            for sr in silent
        ):
            m.silent_region_violations += 1

    sound_ids = [s.get("sound_id") for s in sfx]
    m.unique_sounds = len(set(sound_ids))
    if sound_ids:
        counts = Counter(sound_ids)
        m.most_repeated_count = max(counts.values())
        m.repetition_ratio = round(m.unique_sounds / len(sound_ids), 2)

    m.recipes_applied = len({
        s.get("recipe_name") for s in sfx if s.get("recipe_name")
    })

    score = 100.0
    score -= (100 - m.anchor_coverage_pct) * 0.30
    score -= m.weak_matches_count * 3
    if not m.density_ok:
        score -= 15
    score -= m.whitelist_violations * 2
    score -= m.speech_collision_count * 8
    score -= m.silent_region_violations * 5
    if m.repetition_ratio < 0.6 and len(sound_ids) > 3:
        score -= 10
    score += min(m.recipes_applied * 3, 10)

    m.objective_score = round(max(0, min(100, score)), 1)
    return m
