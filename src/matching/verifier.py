"""Verify chosen sounds actually fit their scenes, using one batched Gemini call.

Embedding similarity often produces matches that are semantically close but
concretely wrong (a sword "schwing" on a sewing scene). This pass reviews the
final picks in context with a single, cheap, text-only Gemini call and drops
the ones that are clearly wrong. It fails OPEN — any error keeps all sounds.
"""

from __future__ import annotations

import json
from typing import Any

from src.config import GEMINI_API_KEY
from src.utils.logger import get_logger

logger = get_logger("verifier")

_MODEL_NAME = "gemini-2.5-flash"


VERIFY_SYSTEM_PROMPT = """You are a sound design quality controller. You are given
a list of moments from a short video, each with: what happens in the scene, and the
sound effect that was automatically chosen for it.

For each item, judge whether the chosen sound genuinely makes sense for that moment.

A sound is WRONG when:
- It implies an object/action not present (e.g. a sword "schwing" on a sewing scene,
  a car engine in a kitchen)
- It's the wrong category (e.g. a notification beep for a physical impact)
- It would confuse or distract the viewer rather than enhance the moment

A sound is FINE when:
- It plausibly matches what's happening, OR
- It's a generic transition/whoosh/impact used at a cut or emphasis point (these are
  stylistic and almost always acceptable)

Be reasonable, not paranoid. Generic stylistic sounds (whoosh, pop, riser, impact,
swoosh) at transitions or emphasis beats are FINE even if not literal. Only DROP
sounds that are clearly, concretely wrong for the content.

Return STRICT JSON only."""


VERIFY_USER_TEMPLATE = """Video style: {style}

Review each moment. For each, return keep=true if the sound fits or is an acceptable
stylistic choice, keep=false only if it is concretely wrong. If keep=false, suggest a
better short search query.

Moments:
{items}

Return JSON exactly:
{{
  "verdicts": [
    {{"id": <int>, "keep": <bool>, "reason": "<short>", "better_query": "<string or null>"}}
  ]
}}"""


def _scene_desc_lookup(analysis: dict) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for s in analysis.get("scenes", []) or []:
        idx = int(s.get("scene_index", -1))
        lookup[idx] = str(s.get("scene_description", "") or "")
    return lookup


def verify_matches(match_plan: dict, analysis: dict, dry_run: bool = False) -> dict:
    """Review all SFX matches with one Gemini call. Mark wrong ones.

    Adds to each SFX match: ``verified`` (bool), ``verify_reason`` (str), and
    ``better_query`` (str|None) when dropped. Sounds judged keep=false move to
    ``match_plan['verification_dropped']``. Ambient, music, and recipe layers
    are NOT verified.
    """
    matches = match_plan.get("matches", []) or []
    sfx = [m for m in matches if m.get("layer") == "sfx" and not m.get("recipe_name")]
    others = [m for m in matches if m.get("layer") != "sfx" or m.get("recipe_name")]

    if not sfx:
        match_plan["verification_stats"] = {"checked": 0, "dropped": 0}
        return match_plan

    scene_desc = _scene_desc_lookup(analysis)

    items_lines: list[str] = []
    for i, m in enumerate(sfx):
        sidx = int(m.get("scene_index", m.get("_scene_index", -1)) or -1)
        desc = scene_desc.get(sidx) or m.get("action_description") or "unknown scene"
        sound = m.get("sound_name", "?")
        atype = m.get("action_type", "?")
        items_lines.append(
            f'{i}. scene="{str(desc)[:120]}" | moment_type="{atype}" '
            f'| chosen_sound="{sound}"'
        )
    items_str = "\n".join(items_lines)

    if dry_run:
        for m in sfx:
            m["verified"] = True
        match_plan["verification_stats"] = {
            "checked": len(sfx), "dropped": 0, "dry_run": True,
        }
        return match_plan

    style = (analysis.get("strategy") or {}).get("style") or "short-form video"

    try:
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            _MODEL_NAME, system_instruction=VERIFY_SYSTEM_PROMPT,
        )
        prompt = VERIFY_USER_TEMPLATE.format(style=style, items=items_str)
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
        )
        verdicts = json.loads(response.text).get("verdicts", [])
    except Exception as exc:  # fail open — never empty the video
        logger.warning("Verification failed (%s); keeping all sounds", exc)
        for m in sfx:
            m["verified"] = True
        match_plan["verification_stats"] = {
            "checked": len(sfx), "dropped": 0, "error": str(exc),
        }
        return match_plan

    verdict_by_id = {
        int(v["id"]): v for v in verdicts
        if isinstance(v, dict) and "id" in v
    }
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for i, m in enumerate(sfx):
        v = verdict_by_id.get(i)
        if v is None or v.get("keep", True):
            m["verified"] = True
            if v is not None:
                m["verify_reason"] = v.get("reason", "")
            kept.append(m)
        else:
            m["verified"] = False
            m["verify_reason"] = v.get("reason", "wrong for scene")
            m["better_query"] = v.get("better_query")
            dropped.append(m)

    match_plan["matches"] = sorted(
        kept + others, key=lambda x: float(x.get("absolute_timestamp", 0) or 0.0),
    )
    match_plan["verification_dropped"] = [
        {
            "timestamp": m.get("absolute_timestamp"),
            "action_type": m.get("action_type"),
            "sound_name": m.get("sound_name"),
            "reason": m.get("verify_reason"),
            "better_query": m.get("better_query"),
        }
        for m in dropped
    ]
    match_plan["verification_stats"] = {
        "checked": len(sfx),
        "dropped": len(dropped),
        "kept": len(kept),
    }
    logger.info(
        "Verification: checked %d, dropped %d wrong sound(s)",
        len(sfx), len(dropped),
    )
    return match_plan


def research_dropped_anchors(
    match_plan: dict,
    analysis: dict,
    qdrant_client: Any,
    preset_name: str,
) -> dict:
    """For dropped sounds that were at anchor moments, re-search the library with
    Gemini's suggested ``better_query`` and add a good replacement back so we
    don't leave a hole at a key beat."""
    from src.library.embedder import embed_texts
    from src.library.vector_store import COLLECTION_NAME, search_sounds

    strategy = analysis.get("strategy") or {}
    anchor_ts = [
        float(am.get("timestamp_sec", -999) or -999)
        for am in strategy.get("anchor_moments", []) or []
    ]

    dropped = match_plan.get("verification_dropped", []) or []
    re_added: list[dict[str, Any]] = []

    for d in dropped:
        ts = float(d.get("timestamp") or 0.0)
        is_anchor = any(abs(ts - at) < 0.8 for at in anchor_ts)
        better_query = d.get("better_query")
        if not is_anchor or not better_query:
            continue

        try:
            query_vec = embed_texts([str(better_query)])[0]
            candidates = search_sounds(
                qdrant_client, COLLECTION_NAME, query_vec, top_k=5,
            )
        except Exception as exc:
            logger.warning("Re-search failed for %r: %s", better_query, exc)
            continue

        if candidates and float(candidates[0].get("score", 0.0)) >= 0.45:
            best = candidates[0]
            payload = best.get("payload") or {}
            new_match = {
                "scene_index": -1,
                "absolute_timestamp": ts,
                "action_type": d.get("action_type", "anchor"),
                "action_description": str(better_query),
                "intensity": "sharp",
                "confidence": 0.8,
                "sound_id": best["id"],
                "sound_name": payload.get("name", "?"),
                "sound_path": payload.get("local_path", ""),
                "sound_duration": float(payload.get("duration") or 1.0),
                "match_score": float(best["score"]),
                "reranked_score": float(best["score"]),
                "layer": "sfx",
                "value_tier": "anchor",
                "verified": True,
                "verify_reason": "re-searched after verification drop",
            }
            match_plan.setdefault("matches", []).append(new_match)
            re_added.append(new_match)

    if re_added:
        match_plan["matches"].sort(
            key=lambda x: float(x.get("absolute_timestamp", 0) or 0.0),
        )
        logger.info("Re-searched and re-added %d anchor sound(s)", len(re_added))
    match_plan.setdefault("verification_stats", {})["re_added"] = len(re_added)
    return match_plan
