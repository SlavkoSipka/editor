from __future__ import annotations

from typing import Any

from src.presets import Preset

_KEYWORD_DELTA = 0.05
_MAX_BOOST = 0.30
_MAX_PENALTY = 0.30
_REASON_KEYWORDS_PREVIEW = 3


def _candidate_text(payload: dict[str, Any]) -> str:
    name = str(payload.get("name", "") or "")
    description = str(payload.get("description", "") or "")
    tags = payload.get("tags") or []
    tags_str = " ".join(str(t) for t in tags)
    return f"{name} {description} {tags_str}".lower()


def _matched_keywords(text: str, keywords: list[str]) -> list[str]:
    return [kw for kw in keywords if kw.lower() in text]


def rerank_candidates(
    candidates: list[dict[str, Any]],
    preset: Preset,
    layer: str = "sfx",
) -> list[dict[str, Any]]:
    """Re-rank Qdrant candidates using preset boost/penalty keywords.
    For `layer == "ambient"`, penalty keywords are skipped (an "ambient" preset
    penalty word would otherwise unfairly punish the ambient layer category itself).
    Adds `reranked_score` and `rerank_reason` fields. Returns a new list."""
    out: list[dict[str, Any]] = []
    apply_penalties = layer != "ambient"
    for c in candidates:
        payload = c.get("payload") or {}
        text = _candidate_text(payload)

        boosts = _matched_keywords(text, preset.boost_keywords)
        penalties = (
            _matched_keywords(text, preset.penalty_keywords) if apply_penalties else []
        )

        boost_adj = min(_MAX_BOOST, _KEYWORD_DELTA * len(boosts))
        penalty_adj = min(_MAX_PENALTY, _KEYWORD_DELTA * len(penalties))
        keyword_adj = boost_adj - penalty_adj

        original_score = float(c.get("score", 0.0) or 0.0)
        reranked_score = original_score + (keyword_adj * preset.rerank_strength)

        reason_parts: list[str] = []
        if boosts:
            preview = ", ".join(boosts[:_REASON_KEYWORDS_PREVIEW])
            reason_parts.append(f"+{len(boosts)} boost matches: {preview}")
        if penalties:
            preview = ", ".join(penalties[:_REASON_KEYWORDS_PREVIEW])
            reason_parts.append(f"-{len(penalties)} penalty: {preview}")
        reason = "; ".join(reason_parts) if reason_parts else "no preset keyword matches"

        new_c = dict(c)
        new_c["reranked_score"] = reranked_score
        new_c["rerank_reason"] = reason
        out.append(new_c)

    out.sort(key=lambda x: x["reranked_score"], reverse=True)
    return out
