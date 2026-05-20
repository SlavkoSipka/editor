"""Director pass — one Gemini call over the entire video that produces a
global Sound Design Strategy. The strategy then steers per-scene analysis and
the matcher (e.g. silent regions get no SFX, global SFX budget gets enforced).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import google.generativeai as genai
from PIL import Image

from src.config import FIXTURES_DIR, GEMINI_API_KEY
from src.preprocessing.scene_detector import Scene
from src.utils.logger import get_logger

logger = get_logger("director")

_MODEL_NAME = "gemini-2.5-flash"
_MAX_DIRECTOR_FRAMES = 15
_RETRY_BACKOFFS_SEC = (1.0, 2.0)
_FIXTURE_PATH = FIXTURES_DIR / "director_response_sample.json"


DIRECTOR_SYSTEM_PROMPT = """You are a senior sound designer analyzing a short-form video (social media ad, vlog, or promotional content) to create a SOUND DESIGN STRATEGY before any actual sound placement.

Your job is to look at the entire video as a whole — not scene-by-scene — and identify:

1. **Style/genre** of the video (UGC TikTok, polished brand, cinematic ad, vlog, product showcase, comedy, dramatic, etc.)
2. **Energy curve** — how energy evolves from start to finish
3. **Hook moment** — the first 1-3 seconds that grab attention; needs maximum impact
4. **Drop moments** — climactic reveals, product appearances, "wow" moments where SFX must hit hard
5. **Silent moments** — places where ABSENCE of SFX creates more impact than presence (dramatic pause, before-the-drop tension, breathing room)
6. **Outro / CTA** — the logo/text/call-to-action ending; usually wants cinematic sting then silence
7. **Continuous regions** — scenes that flow together with same environment (don't restart ambient for every cut)
8. **Overall vibe** — one phrase capturing the feeling (playful, dramatic, mysterious, energetic, luxurious)
9. **Recommended preset** — based on what you see, suggest one of: tiktok_viral, viral_ad, vlog_casual, dramatic, energetic, mysterious, playful, luxury
10. **Anchor moments** — the 3-5 MOST IMPORTANT sound moments that MUST be perfect

CRITICAL PRINCIPLES:
- Less is more. Modern short-form ads use 3-8 SFX moments total, not 20.
- Silence is a tool. Identify it deliberately.
- The hook (first 1-3s) is the single most important sound moment in any social video.
- The outro should always feel resolved, not just stop.
- Sounds should respect the energy curve — don't put dramatic boom in a calm part.

You will return STRICTLY VALID JSON. No prose, no markdown."""


DIRECTOR_USER_PROMPT_TEMPLATE = """Here are {n_frames} frames sampled evenly from a {duration:.1f}-second video. Frame 1 is at the start, Frame {n_frames} is at the end.

Speech detection summary:
- {speech_summary}

The user selected preset: "{user_preset}" — you may agree, or recommend a different one if the content clearly doesn't fit.

Scene boundaries (in seconds): {scene_boundaries}

Audio onset count (rough indication of how many natural audio events exist): {onset_count}

IMPORTANT: If the video contains significant speech (someone talking to camera), your strategy MUST set:
- "music_dominance": "subtle"
- "preferred_density": "sparse"
- Avoid placing SFX inside speech moments (mark those windows as "silent_regions" when appropriate)
- Music must never compete with the spoken voice

Return strategy JSON with this EXACT schema:

{{
  "style": "string (e.g. 'TikTok UGC ad', 'Polished brand commercial', 'Vlog')",
  "vibe": "one short phrase (e.g. 'playful upbeat', 'tense dramatic')",
  "recommended_preset": "tiktok_viral|viral_ad|vlog_casual|dramatic|energetic|mysterious|playful|luxury",
  "override_user_preset": false,
  "override_reason": "string or null — explain only if recommended differs from user choice in a meaningful way",

  "energy_curve": [
    {{"timestamp_sec": 0.0, "level": 0.3, "label": "intriguing intro"}},
    {{"timestamp_sec": 5.0, "level": 0.5, "label": "build up"}},
    {{"timestamp_sec": 12.0, "level": 1.0, "label": "drop / reveal"}},
    {{"timestamp_sec": 20.0, "level": 0.7, "label": "sustain"}},
    {{"timestamp_sec": 28.0, "level": 0.4, "label": "outro / cta"}}
  ],

  "hook": {{
    "start_sec": 0.0,
    "end_sec": 2.5,
    "intent": "string (what mood/feeling the hook should evoke)",
    "sfx_directive": "string (concrete instruction, e.g. 'sharp riser building into transition pop on cut at 2.4s')"
  }},

  "drops": [
    {{"timestamp_sec": 12.0, "intent": "product reveal", "sfx_directive": "deep impact + shimmer rise"}}
  ],

  "silent_regions": [
    {{"start_sec": 10.5, "end_sec": 12.0, "reason": "tension before drop — only music, no SFX"}}
  ],

  "outro": {{
    "start_sec": 26.0,
    "end_sec": 30.0,
    "intent": "resolved cinematic close with logo reveal",
    "sfx_directive": "single logo sting with long tail, then silence except music"
  }},

  "continuous_regions": [
    {{"start_sec": 3.5, "end_sec": 8.0, "label": "exterior mountain — single continuous ambient"}}
  ],

  "anchor_moments": [
    {{"timestamp_sec": 0.0, "type": "hook_start", "description": "video opens — first frame must have anchor sound"}},
    {{"timestamp_sec": 12.0, "type": "drop", "description": "product reveal needs major impact"}},
    {{"timestamp_sec": 27.5, "type": "logo_sting", "description": "logo appears, single cinematic hit"}}
  ],

  "global_constraints": {{
    "max_sfx_count": 8,
    "preferred_density": "sparse|moderate|dense",
    "music_dominance": "subtle|balanced|prominent",
    "sfx_style": "ugc|cinematic|foley|hybrid"
  }}
}}

Return JSON only. No commentary."""


_REQUIRED_STRATEGY_KEYS = (
    "style", "vibe", "recommended_preset", "energy_curve",
    "hook", "anchor_moments", "global_constraints",
)


_director_model: genai.GenerativeModel | None = None


def _get_model() -> genai.GenerativeModel:
    global _director_model
    if _director_model is None:
        genai.configure(api_key=GEMINI_API_KEY)
        _director_model = genai.GenerativeModel(
            _MODEL_NAME,
            system_instruction=DIRECTOR_SYSTEM_PROMPT,
        )
    return _director_model


def _validate_strategy(strategy: dict[str, Any]) -> None:
    missing = [k for k in _REQUIRED_STRATEGY_KEYS if k not in strategy]
    if missing:
        raise ValueError(f"Director strategy missing keys: {missing}")


def _minimal_default_strategy(
    user_preset: str,
    duration_sec: float,
    scenes: list[Scene],
) -> dict[str, Any]:
    """Conservative fallback when the director call fails or is skipped.

    Keeps the pipeline running without imposing any silent regions, anchor
    moments, or budget cuts."""
    _ = scenes  # accepted for symmetry with the real director call
    duration_sec = float(duration_sec or 0.0)
    return {
        "style": "unknown",
        "vibe": "neutral",
        "recommended_preset": user_preset,
        "override_user_preset": False,
        "override_reason": None,
        "energy_curve": [
            {"timestamp_sec": 0.0, "level": 0.4, "label": "start"},
            {"timestamp_sec": duration_sec / 2 if duration_sec else 0.0,
             "level": 0.7, "label": "middle"},
            {"timestamp_sec": duration_sec, "level": 0.5, "label": "end"},
        ],
        "hook": {
            "start_sec": 0.0,
            "end_sec": min(3.0, duration_sec) if duration_sec else 3.0,
            "intent": "open the video",
            "sfx_directive": "",
        },
        "drops": [],
        "silent_regions": [],
        "outro": {
            "start_sec": max(0.0, duration_sec - 3.0),
            "end_sec": duration_sec,
            "intent": "close",
            "sfx_directive": "",
        },
        "continuous_regions": [],
        "anchor_moments": [],
        "global_constraints": {
            "max_sfx_count": 999,
            "preferred_density": "moderate",
            "music_dominance": "balanced",
            "sfx_style": "hybrid",
        },
        "_fallback": True,
    }


def _sample_frames(frame_paths: list[Path]) -> list[Path]:
    if len(frame_paths) <= _MAX_DIRECTOR_FRAMES:
        return list(frame_paths)
    step = len(frame_paths) / _MAX_DIRECTOR_FRAMES
    return [frame_paths[int(i * step)] for i in range(_MAX_DIRECTOR_FRAMES)]


def _scene_boundaries_str(scenes: list[Scene]) -> str:
    if not scenes:
        return "(no scenes detected)"
    snippet = ", ".join(f"{s.start_sec:.1f}\u2192{s.end_sec:.1f}" for s in scenes[:20])
    if len(scenes) > 20:
        snippet += f" ... ({len(scenes)} total)"
    return snippet


def _speech_summary_for_prompt(
    speech_regions: list[tuple[float, float]] | None,
    duration_sec: float,
) -> str:
    if not speech_regions:
        return "Little to no speech detected \u2014 music and SFX can be more prominent."
    cov = 0.0
    if duration_sec > 0:
        cov = min(1.0, sum(float(e) - float(s) for s, e in speech_regions) / duration_sec)
    preview = ", ".join(
        f"{float(s):.1f}\u2013{float(e):.1f}s" for s, e in speech_regions[:3]
    )
    if len(speech_regions) > 3:
        preview += ", ..."
    if cov > 0.4:
        return (
            f"HEAVY speech ({cov * 100:.0f}% of video). Person is talking "
            f"throughout \u2014 speech protection critical. Regions: {preview}"
        )
    if cov > 0.15:
        return (
            f"Moderate speech ({cov * 100:.0f}% of video). Regions: {preview}"
        )
    return (
        f"Light speech ({cov * 100:.0f}% of video). Regions: {preview}"
    )


def run_director_pass(
    video_path: Path,
    frame_paths: list[Path],
    scenes: list[Scene],
    onsets: list[float],
    user_preset: str,
    duration_sec: float,
    dry_run: bool = False,
    speech_regions: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Run the global director analysis. Returns the strategy dict.

    Uses up to ~15 evenly-sampled frames so the model sees the whole video
    without paying for dense frames. Falls back to a minimal default on
    repeated failure so the pipeline keeps running."""
    _ = video_path  # currently unused; kept for future per-video customisation
    if dry_run:
        if _FIXTURE_PATH.is_file():
            logger.info("Director dry-run: loading fixture %s", _FIXTURE_PATH)
            strategy = json.loads(_FIXTURE_PATH.read_text())
            strategy.setdefault("duration_sec", duration_sec)
            if speech_regions is not None:
                strategy["speech_regions"] = [
                    [float(s), float(e)] for s, e in speech_regions
                ]
            return strategy
        logger.warning(
            "Director dry-run requested but no fixture at %s; using default.",
            _FIXTURE_PATH,
        )
        strategy = _minimal_default_strategy(user_preset, duration_sec, scenes)
        if speech_regions is not None:
            strategy["speech_regions"] = [
                [float(s), float(e)] for s, e in speech_regions
            ]
        return strategy

    sampled = _sample_frames(frame_paths)
    if not sampled:
        logger.warning("Director: no frames available; using minimal default.")
        strategy = _minimal_default_strategy(user_preset, duration_sec, scenes)
        if speech_regions is not None:
            strategy["speech_regions"] = [
                [float(s), float(e)] for s, e in speech_regions
            ]
        return strategy

    prompt = DIRECTOR_USER_PROMPT_TEMPLATE.format(
        n_frames=len(sampled),
        duration=duration_sec,
        scene_boundaries=_scene_boundaries_str(scenes),
        onset_count=len(onsets),
        user_preset=user_preset,
        speech_summary=_speech_summary_for_prompt(speech_regions, duration_sec),
    )

    images = [Image.open(p) for p in sampled]
    request_parts: list[Any] = [prompt, *images]

    model = _get_model()
    logger.info(
        "Director: analyzing video as a whole (%d frames, %d scenes, %.1fs)",
        len(sampled), len(scenes), duration_sec,
    )

    last_exc: Exception | None = None
    for attempt in range(len(_RETRY_BACKOFFS_SEC) + 1):
        try:
            response = model.generate_content(
                request_parts,
                generation_config={"response_mime_type": "application/json"},
            )
            strategy = json.loads(response.text)
            _validate_strategy(strategy)
            strategy["duration_sec"] = duration_sec
            if speech_regions is not None:
                strategy["speech_regions"] = [
                    [float(s), float(e)] for s, e in speech_regions
                ]
            return strategy
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Director attempt %d/%d failed: %s",
                attempt + 1, len(_RETRY_BACKOFFS_SEC) + 1, exc,
            )
            if attempt < len(_RETRY_BACKOFFS_SEC):
                time.sleep(_RETRY_BACKOFFS_SEC[attempt])

    logger.error("Director failed after retries (%s); using minimal default.", last_exc)
    fallback = _minimal_default_strategy(user_preset, duration_sec, scenes)
    fallback["_director_error"] = str(last_exc) if last_exc else "unknown"
    if speech_regions is not None:
        fallback["speech_regions"] = [
            [float(s), float(e)] for s, e in speech_regions
        ]
    return fallback


def _interpolate_energy(curve: list[dict[str, Any]], t: float) -> float:
    if not curve:
        return 0.5
    pts = sorted(
        ((float(p.get("timestamp_sec") or 0.0), float(p.get("level") or 0.0)) for p in curve),
        key=lambda x: x[0],
    )
    if t <= pts[0][0]:
        return pts[0][1]
    if t >= pts[-1][0]:
        return pts[-1][1]
    for (a_t, a_l), (b_t, b_l) in zip(pts, pts[1:]):
        if a_t <= t <= b_t and b_t > a_t:
            ratio = (t - a_t) / (b_t - a_t)
            return a_l + ratio * (b_l - a_l)
    return 0.5


def scene_context_from_strategy(
    scene: Scene,
    strategy: dict[str, Any],
    total_scenes: int,
) -> dict[str, Any]:
    """Derive per-scene context (role, energy, budget, silence/anchor notes)
    from the global strategy."""
    if not strategy:
        return {
            "scene_role": "middle",
            "energy_level": 0.5,
            "max_sfx_count": 999,
            "scene_sfx_budget": 999,
            "silence_note": "",
            "anchor_note": "",
            "scene_specific_directive": "",
        }

    t_start = float(scene.start_sec)
    t_end = float(scene.end_sec)
    duration = float(strategy.get("duration_sec") or 0.0)

    hook = strategy.get("hook") or {}
    outro = strategy.get("outro") or {}
    drops = strategy.get("drops") or []

    hook_end = float(hook.get("end_sec") or 0.0)
    outro_start = float(outro.get("start_sec") or (duration or 0.0))

    if t_start < hook_end:
        role = "hook"
    elif t_start >= outro_start > 0:
        role = "outro"
    elif any(t_start <= float(d.get("timestamp_sec") or 0.0) <= t_end for d in drops):
        role = "drop"
    else:
        rel = (t_start / duration) if duration else 0.5
        role = "build" if rel < 0.5 else "sustain"

    energy = _interpolate_energy(strategy.get("energy_curve") or [], t_start)

    constraints = strategy.get("global_constraints") or {}
    max_total = int(constraints.get("max_sfx_count") or 999)
    scene_budget = max(1, max_total // max(1, total_scenes))
    if role in ("hook", "drop", "outro"):
        scene_budget = min(3, scene_budget + 1)

    silence_note = ""
    for sr in strategy.get("silent_regions") or []:
        if t_start >= float(sr.get("start_sec") or 0.0) and t_end <= float(sr.get("end_sec") or 0.0):
            silence_note = (
                "IMPORTANT: This scene is in a SILENT REGION. Suggest 0 SFX "
                f"unless absolutely critical. Reason: {sr.get('reason','')}"
            )
            scene_budget = 0
            break

    anchor_note = ""
    for am in strategy.get("anchor_moments") or []:
        am_t = float(am.get("timestamp_sec") or 0.0)
        if t_start <= am_t <= t_end:
            anchor_note = (
                f"IMPORTANT: An ANCHOR MOMENT occurs here at {am_t:.1f}s \u2014 "
                f"'{am.get('description','')}'. The SFX here MUST be excellent."
            )
            break

    directive = ""
    if role == "hook":
        directive = f"Hook directive: {hook.get('sfx_directive','')}"
    elif role == "outro":
        directive = f"Outro directive: {outro.get('sfx_directive','')}"
    elif role == "drop":
        matching = next(
            (d for d in drops
             if t_start <= float(d.get("timestamp_sec") or 0.0) <= t_end),
            None,
        )
        if matching:
            directive = f"Drop directive: {matching.get('sfx_directive','')}"

    return {
        "scene_role": role,
        "energy_level": round(energy, 2),
        "max_sfx_count": max_total,
        "scene_sfx_budget": scene_budget,
        "silence_note": silence_note,
        "anchor_note": anchor_note,
        "scene_specific_directive": directive,
    }


def strategy_summary_lines(strategy: dict[str, Any]) -> list[str]:
    """Compact, human-readable summary of a strategy for CLI output."""
    if not strategy:
        return []
    lines: list[str] = []
    style = strategy.get("style") or "?"
    vibe = strategy.get("vibe") or "?"
    rec = strategy.get("recommended_preset") or "?"
    override = strategy.get("override_user_preset")
    override_reason = strategy.get("override_reason") or ""

    lines.append(f"Style: {style}")
    lines.append(f"Vibe:  {vibe}")
    lines.append(
        f"Recommended preset: {rec}"
        + (" \u2014 override suggested" if override else "")
        + (f" ({override_reason})" if override and override_reason else "")
    )

    hook = strategy.get("hook") or {}
    if hook:
        lines.append(
            f"Hook:  {hook.get('start_sec', 0.0):.1f}\u2013{hook.get('end_sec', 0.0):.1f}s "
            f"\u2014 \"{hook.get('sfx_directive') or hook.get('intent') or ''}\""
        )
    drops = strategy.get("drops") or []
    if drops:
        drop_ts = ", ".join(f"{float(d.get('timestamp_sec') or 0.0):.1f}s" for d in drops)
        lines.append(f"Drops: {len(drops)} (at {drop_ts})")
    silent = strategy.get("silent_regions") or []
    if silent:
        sr_str = ", ".join(
            f"{float(s.get('start_sec') or 0):.1f}\u2013{float(s.get('end_sec') or 0):.1f}s"
            for s in silent
        )
        lines.append(f"Silent regions: {len(silent)} ({sr_str})")
    anchors = strategy.get("anchor_moments") or []
    if anchors:
        lines.append(f"Anchor moments: {len(anchors)}")
    constraints = strategy.get("global_constraints") or {}
    max_total = constraints.get("max_sfx_count")
    if max_total is not None:
        lines.append(f"Max SFX budget: {max_total}")
    sp = strategy.get("speech_regions") or []
    if sp:
        dur = float(strategy.get("duration_sec") or 0.0)
        cov = 0.0
        if dur > 0:
            cov = min(1.0, sum(float(e) - float(s) for s, e in sp) / dur)
        lines.append(f"Speech: {len(sp)} region(s), {cov * 100:.0f}% coverage")
    if strategy.get("_fallback"):
        lines.append("(using fallback strategy \u2014 director call skipped or failed)")
    return lines
