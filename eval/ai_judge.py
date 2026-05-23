"""AI judge — uses Gemini to subjectively score the final video's sound design."""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.config import GEMINI_API_KEY
from src.utils.logger import get_logger

logger = get_logger("ai_judge")


JUDGE_SYSTEM_PROMPT = """You are an expert sound designer and critic evaluating
the automatically-generated sound design of a short-form video ad.

You will be shown a video that already has AI-generated sound effects, ambient,
and music mixed in. Judge ONLY the sound design quality — not the video footage.

Score these dimensions, each 1-10:
1. timing — do sounds hit at the right moments?
2. relevance — do sounds match what's on screen?
3. balance — are volume levels good? speech clear? nothing too loud/quiet?
4. restraint — is it tasteful, or cluttered with unnecessary sounds?
5. impact — do key moments (hook, drop, logo) land well?
6. cohesion — does it feel like one designed piece, or random clips?

Be honest and critical. A 5 is "mediocre", 7 is "good", 9 is "excellent".

Return STRICT JSON only."""


JUDGE_USER_PROMPT = """Evaluate this video's sound design. Return JSON:

{
  "timing": <1-10>,
  "relevance": <1-10>,
  "balance": <1-10>,
  "restraint": <1-10>,
  "impact": <1-10>,
  "cohesion": <1-10>,
  "overall": <1-10>,
  "biggest_strength": "<short>",
  "biggest_weakness": "<short>",
  "one_fix": "<the single most impactful change to improve it>"
}"""


def judge_video(video_path: Path) -> dict | None:
    """Send the final video to Gemini and get a sound design assessment."""
    try:
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            system_instruction=JUDGE_SYSTEM_PROMPT,
        )

        logger.info("AI judge: uploading %s...", video_path.name)
        uploaded = genai.upload_file(str(video_path))

        while uploaded.state.name == "PROCESSING":
            time.sleep(2)
            uploaded = genai.get_file(uploaded.name)

        response = model.generate_content(
            [JUDGE_USER_PROMPT, uploaded],
            generation_config={"response_mime_type": "application/json"},
        )
        judgment = json.loads(response.text)
        logger.info("AI judge: overall %s/10", judgment.get("overall"))

        try:
            genai.delete_file(uploaded.name)
        except Exception:
            pass

        return judgment
    except Exception as exc:
        logger.warning("AI judge failed: %s", exc)
        return None
