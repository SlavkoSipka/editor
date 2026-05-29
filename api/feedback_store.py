"""Persist feedback to Supabase. Falls back to local JSONL if Supabase is down.

Supabase keeps feedback durable across redeploys and is reused later for auth +
subscriptions. When SUPABASE_* env vars are missing or the API call fails, we
append to a JSONL file on the volume so feedback is never lost.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

from api.models import JobFeedback
from api.supabase_client import get_supabase
from src.config import DATA_DIR
from src.utils.logger import get_logger

logger = get_logger("feedback_store")

FEEDBACK_DIR = DATA_DIR / "feedback"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
FALLBACK_FILE = FEEDBACK_DIR / "feedback_fallback.jsonl"


def save_feedback(fb: JobFeedback) -> dict:
    """Save feedback to Supabase. Returns {ok, id} or falls back to JSONL."""
    sb = get_supabase()
    if sb is None:
        return _save_fallback(fb)

    try:
        parent = {
            "job_id": fb.job_id,
            "reviewer": fb.reviewer,
            "preset": fb.preset,
            "density": fb.density,
            "overall_rating": fb.overall_rating,
            "density_feedback": fb.density_feedback,
            "overall_note": fb.overall_note,
            "status": "pending",
        }
        res = sb.table("job_feedback").insert(parent).execute()
        feedback_id = res.data[0]["id"]

        if fb.sound_feedback:
            rows = [{
                "feedback_id": feedback_id,
                "sound_index": s.sound_index,
                "sound_id": s.sound_id,
                "sound_name": s.sound_name,
                "action_type": s.action_type,
                "timestamp": s.timestamp,
                "matched_tier": s.matched_tier,
                "match_score": s.match_score,
                "rating": s.rating,
            } for s in fb.sound_feedback]
            sb.table("sound_feedback").insert(rows).execute()

        if fb.missing_sounds:
            rows = [{
                "feedback_id": feedback_id,
                "timestamp": m.timestamp,
                "note": m.note,
            } for m in fb.missing_sounds]
            sb.table("missing_sounds").insert(rows).execute()

        logger.info("Saved feedback %s to Supabase", feedback_id)
        return {"ok": True, "id": feedback_id}
    except Exception as exc:
        logger.error("Supabase save failed (%s); using fallback", exc)
        return _save_fallback(fb)


def _save_fallback(fb: JobFeedback) -> dict:
    data = fb.model_dump()
    data["submitted_at"] = datetime.now(timezone.utc).isoformat()
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    with open(FALLBACK_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")
    return {"ok": True, "id": None, "fallback": True}


def load_all_feedback() -> list[dict]:
    """Raw fallback records (used by /feedback/export when running on JSONL)."""
    if not FALLBACK_FILE.exists():
        return []
    out: list[dict] = []
    with open(FALLBACK_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def list_feedback(
    status: str | None = None, reviewer: str | None = None,
) -> list[dict]:
    """List feedback parent rows with optional filters."""
    sb = get_supabase()
    if sb is None:
        return []
    q = sb.table("job_feedback").select("*").order("created_at", desc=True)
    if status:
        q = q.eq("status", status)
    if reviewer:
        q = q.eq("reviewer", reviewer)
    return q.execute().data


def get_feedback_detail(feedback_id: str) -> dict:
    """Get a feedback row with its sound + missing children."""
    sb = get_supabase()
    if sb is None:
        return {}
    parent = sb.table("job_feedback").select("*").eq("id", feedback_id).execute().data
    sounds = (
        sb.table("sound_feedback").select("*").eq("feedback_id", feedback_id).execute().data
    )
    missing = (
        sb.table("missing_sounds").select("*").eq("feedback_id", feedback_id).execute().data
    )
    return {
        "feedback": parent[0] if parent else None,
        "sounds": sounds,
        "missing": missing,
    }


def update_status(
    feedback_id: str, status: str, dev_note: str | None = None,
) -> dict:
    """Mark a feedback as applied/pending/ignored, with an optional dev note."""
    sb = get_supabase()
    if sb is None:
        return {"ok": False, "reason": "supabase unavailable"}
    update: dict = {"status": status}
    if dev_note is not None:
        update["dev_note"] = dev_note
    sb.table("job_feedback").update(update).eq("id", feedback_id).execute()
    return {"ok": True}


def aggregate_feedback() -> dict:
    """Aggregate patterns across all feedback (queries Supabase)."""
    sb = get_supabase()
    if sb is None:
        return {"total_jobs": 0, "supabase": False}

    parents = sb.table("job_feedback").select("*").execute().data
    sounds = sb.table("sound_feedback").select("*").execute().data
    missing = sb.table("missing_sounds").select("*").execute().data

    by_action: dict[str, Counter] = defaultdict(Counter)
    by_tier: dict[str, Counter] = defaultdict(Counter)
    by_sound: dict[str, Counter] = defaultdict(Counter)
    by_score: dict[str, Counter] = defaultdict(Counter)
    for s in sounds:
        r = s.get("rating")
        if s.get("action_type"):
            by_action[s["action_type"]][r] += 1
        if s.get("matched_tier"):
            by_tier[s["matched_tier"]][r] += 1
        if s.get("sound_name"):
            by_sound[s["sound_name"]][r] += 1
        sc = s.get("match_score")
        if sc is not None:
            low = int(float(sc) * 10) / 10
            b = f"{low:.1f}-{low + 0.1:.1f}"
            by_score[b][r] += 1

    ratings = [p["overall_rating"] for p in parents if p.get("overall_rating")]
    density_fb = Counter(
        p["density_feedback"] for p in parents if p.get("density_feedback")
    )

    worst_actions = sorted(
        by_action.items(),
        key=lambda kv: kv[1].get("wrong", 0) + kv[1].get("unnecessary", 0),
        reverse=True,
    )[:15]
    worst_sounds = sorted(
        by_sound.items(), key=lambda kv: kv[1].get("wrong", 0), reverse=True,
    )[:20]

    return {
        "total_jobs": len(parents),
        "by_status": dict(Counter(p.get("status", "pending") for p in parents)),
        "by_reviewer": dict(Counter(p.get("reviewer", "unknown") for p in parents)),
        "avg_overall_rating": (
            round(sum(ratings) / len(ratings), 2) if ratings else None
        ),
        "density_feedback": dict(density_fb),
        "missing_sounds_total": len(missing),
        "ratings_by_action_type": {k: dict(v) for k, v in by_action.items()},
        "ratings_by_tier": {k: dict(v) for k, v in by_tier.items()},
        "ratings_by_score_bucket": {
            k: dict(v) for k, v in sorted(by_score.items())
        },
        "worst_action_types": [
            {"action": k, "ratings": dict(v)} for k, v in worst_actions
        ],
        "worst_sounds": [{"sound": k, "ratings": dict(v)} for k, v in worst_sounds],
        "pending_notes": [
            {
                "id": p["id"], "preset": p.get("preset"),
                "reviewer": p.get("reviewer"), "note": p.get("overall_note"),
                "status": p.get("status"),
            }
            for p in parents if p.get("overall_note")
        ],
    }
