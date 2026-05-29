"""Persist user feedback to disk as JSONL (one record per job).

Lives on the Railway volume at ``$DATA_DIR/feedback/feedback.jsonl`` so it
survives restarts. Append-only; aggregation reads the whole file.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

from api.models import JobFeedback
from src.config import DATA_DIR

FEEDBACK_DIR = DATA_DIR / "feedback"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_FILE = FEEDBACK_DIR / "feedback.jsonl"


def save_feedback(fb: JobFeedback) -> None:
    fb.submitted_at = datetime.now(timezone.utc).isoformat()
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(fb.model_dump()) + "\n")


def load_all_feedback() -> list[dict]:
    if not FEEDBACK_FILE.exists():
        return []
    out: list[dict] = []
    with open(FEEDBACK_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def aggregate_feedback() -> dict:
    """Aggregate feedback into patterns useful for improving the system."""
    records = load_all_feedback()
    if not records:
        return {"total_jobs": 0}

    by_action_rating: dict[str, Counter] = defaultdict(Counter)
    by_tier_rating: dict[str, Counter] = defaultdict(Counter)
    by_sound_rating: dict[str, Counter] = defaultdict(Counter)
    score_buckets: dict[str, Counter] = defaultdict(Counter)
    density_feedback: Counter = Counter()
    overall_ratings: list[int] = []
    notes: list[dict] = []
    missing_count = 0

    for r in records:
        if r.get("overall_rating"):
            overall_ratings.append(int(r["overall_rating"]))
        if r.get("density_feedback"):
            density_feedback[r["density_feedback"]] += 1
        if r.get("overall_note"):
            notes.append({"preset": r.get("preset"), "note": r["overall_note"]})
        missing_count += len(r.get("missing_sounds", []) or [])

        for sf in r.get("sound_feedback", []) or []:
            rating = sf.get("rating")
            if sf.get("action_type"):
                by_action_rating[sf["action_type"]][rating] += 1
            if sf.get("matched_tier"):
                by_tier_rating[sf["matched_tier"]][rating] += 1
            if sf.get("sound_name"):
                by_sound_rating[sf["sound_name"]][rating] += 1
            score = sf.get("match_score")
            if score is not None:
                low = int(float(score) * 10) / 10
                bucket = f"{low:.1f}-{low + 0.1:.1f}"
                score_buckets[bucket][rating] += 1

    worst_actions = sorted(
        by_action_rating.items(),
        key=lambda kv: kv[1].get("wrong", 0) + kv[1].get("unnecessary", 0),
        reverse=True,
    )[:15]
    worst_sounds = sorted(
        by_sound_rating.items(),
        key=lambda kv: kv[1].get("wrong", 0),
        reverse=True,
    )[:20]

    return {
        "total_jobs": len(records),
        "avg_overall_rating": (
            round(sum(overall_ratings) / len(overall_ratings), 2)
            if overall_ratings else None
        ),
        "density_feedback": dict(density_feedback),
        "missing_sounds_total": missing_count,
        "ratings_by_action_type": {k: dict(v) for k, v in by_action_rating.items()},
        "ratings_by_tier": {k: dict(v) for k, v in by_tier_rating.items()},
        "ratings_by_score_bucket": {
            k: dict(v) for k, v in sorted(score_buckets.items())
        },
        "worst_action_types": [
            {"action": k, "ratings": dict(v)} for k, v in worst_actions
        ],
        "worst_sounds": [{"sound": k, "ratings": dict(v)} for k, v in worst_sounds],
        "notes": notes,
    }
