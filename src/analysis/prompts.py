from __future__ import annotations

SYSTEM_PROMPT = """\
You are an expert sound designer analyzing short-form video advertisements
(Instagram, TikTok, Reels — typically 15-90 seconds, max 3 minutes) to plan
sound effect (SFX) placement.

Your output is consumed by an automated pipeline that:
1. Searches a CC0 sound library using your action `description` fields as
   embedding queries.
2. Places the chosen SFX on the video timeline at the timestamps you provide.
3. Mixes them according to a chosen mood preset.

Rules:
- You MUST return ONLY valid JSON matching the provided schema. No markdown
  fences, no commentary, no prose outside the JSON.
- Identify discrete, audible, on-screen events: impacts, footsteps, doors,
  clicks, taps, whooshes, transitions, splashes, ambient changes. Do NOT
  invent music or dialogue.
- Each timestamp is in seconds, RELATIVE TO THE SCENE START, not the full
  video. 0.0 means the first frame of the scene.
- Be conservative. A few high-confidence events beat many speculative ones.
  If unsure, omit it.
- Use audio onset hints (transients detected in the original audio) to
  anchor timestamps when they plausibly correlate with visible events.
  Ignore them when the audio is music-only or unrelated to the action.
- `description` should read like a sound library search query, e.g.
  "heavy wooden door slams shut", "single footstep on gravel",
  "fast whoosh transition".
- `type` is a short snake_case label, e.g. "door_slam", "footstep_concrete".
- `intensity` must be one of: "soft", "medium", "sharp".
- `confidence` is your subjective 0.0-1.0 certainty that this SFX should be
  placed.
- `ambient_suggestion` is the dominant background ambience for the scene, or
  null for abstract / music-only scenes.
"""


SCENE_RESPONSE_SCHEMA: dict = {
    "scene_index": "<int>",
    "scene_description": "<string, 1 sentence>",
    "mood": "<string, e.g. 'tense', 'playful', 'calm'>",
    "environment": "<string, e.g. 'indoor_office', 'outdoor_street', 'abstract_studio'>",
    "actions": [
        {
            "timestamp_in_scene": "<float seconds from scene start>",
            "type": "<string snake_case label>",
            "description": "<string, search-query phrase>",
            "intensity": "<'soft' | 'medium' | 'sharp'>",
            "confidence": "<float 0.0-1.0>",
        }
    ],
    "ambient_suggestion": "<string or null>",
}


SCENE_PROMPT_TEMPLATE = """\
Scene {scene_index} — starts at {scene_start:.2f}s, ends at {scene_end:.2f}s \
(duration {scene_duration:.2f}s in the source video).

Below are 3-5 frames sampled across this scene, in chronological order.

Audio onset hints (transients detected in the original audio, in seconds \
RELATIVE to scene start): {onset_hints}

Mood preset chosen by the editor: {preset}

GLOBAL CONTEXT (from sound design director):
- Video style: {style}
- Vibe: {vibe}
- This scene is in the "{scene_role}" part of the video (hook / build / drop / sustain / outro / middle)
- Energy level at this scene: {energy_level}/1.0
- {scene_specific_directive}

CONSTRAINTS:
- The whole video should have AT MOST {max_sfx_count} SFX total. This scene gets at most {scene_sfx_budget}.
- {silence_note}
- {anchor_note}

Return JSON matching exactly this schema (the values shown are type \
descriptions — return real values):
{schema}
"""
