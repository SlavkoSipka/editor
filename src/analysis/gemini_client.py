from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

import google.generativeai as genai
import numpy as np
from PIL import Image

from src.analysis.director import (
    run_director_pass,
    scene_context_from_strategy,
)
from src.analysis.prompts import (
    SCENE_PROMPT_TEMPLATE,
    SCENE_RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
)
from src.config import FIXTURES_DIR, FRAME_RATE, GEMINI_API_KEY, TEMP_DIR
from src.preprocessing.audio_extractor import extract_audio
from src.preprocessing.frame_extractor import extract_frames
from src.preprocessing.onset_detector import detect_onsets
from src.preprocessing.scene_detector import Scene, detect_scenes
from src.preprocessing.speech_detector import detect_speech_regions
from src.utils.logger import get_logger

logger = get_logger("gemini_client")

_MODEL_NAME = "gemini-2.5-flash"
_MAX_FRAMES_PER_SCENE = 5
_MIN_FRAMES_PER_SCENE = 3
_RETRY_BACKOFFS_SEC = (1.0, 2.0)
_REQUIRED_KEYS = {"scene_index", "mood", "environment", "actions"}
_FIXTURE_PATH = FIXTURES_DIR / "scene_response_sample.json"

_model: genai.GenerativeModel | None = None


def _get_model() -> genai.GenerativeModel:
    global _model
    if _model is None:
        genai.configure(api_key=GEMINI_API_KEY)
        _model = genai.GenerativeModel(
            _MODEL_NAME,
            system_instruction=SYSTEM_PROMPT,
        )
    return _model


def _select_frames(frame_paths: list[Path]) -> list[Path]:
    if len(frame_paths) <= _MAX_FRAMES_PER_SCENE:
        return list(frame_paths)
    indices = np.linspace(0, len(frame_paths) - 1, _MAX_FRAMES_PER_SCENE, dtype=int)
    return [frame_paths[i] for i in indices]


def _sample_scene_frames(
    scene_frames: list[Path],
    fps: int = FRAME_RATE,
    max_frames: int = _MAX_FRAMES_PER_SCENE,
) -> tuple[list[Path], list[float]]:
    """Pick up to ``max_frames`` from ``scene_frames`` (already sliced for this
    scene at ``fps``) and return (paths, scene-relative timestamps in seconds)."""
    if not scene_frames:
        return [], []
    if len(scene_frames) <= max_frames:
        sampled = list(scene_frames)
        local_indices = list(range(len(scene_frames)))
    else:
        idx = np.linspace(0, len(scene_frames) - 1, max_frames, dtype=int)
        sampled = [scene_frames[int(i)] for i in idx]
        local_indices = [int(i) for i in idx]
    frame_times = [max(0.0, i / float(fps)) for i in local_indices]
    return sampled, frame_times


def _format_onset_hints(onsets: list[float]) -> str:
    if not onsets:
        return "none detected"
    return "[" + ", ".join(f"{t:.2f}" for t in onsets) + "]"


def _scene_has_speech(
    scene: Scene, speech_regions: list[tuple[float, float]] | None,
) -> bool:
    if not speech_regions:
        return False
    return any(
        not (float(e) < scene.start_sec or float(s) > scene.end_sec)
        for s, e in speech_regions
    )


def _normalize_action(action: dict[str, Any]) -> dict[str, Any]:
    """Backfill defaults for fields the v2 prompt added so old cached analyses
    keep working."""
    if "sound_value" not in action or action.get("sound_value") is None:
        action["sound_value"] = 0.6
    return action


def _fallback_response(scene: Scene) -> dict[str, Any]:
    return {
        "scene_index": scene.index,
        "scene_description": "",
        "mood": "unknown",
        "environment": "unknown",
        "actions": [],
        "ambient_suggestion": None,
    }


def _load_fixture(scene_index: int) -> dict[str, Any]:
    if not _FIXTURE_PATH.is_file():
        raise FileNotFoundError(
            f"Dry-run fixture missing: {_FIXTURE_PATH}. "
            f"Create one matching SCENE_RESPONSE_SCHEMA."
        )
    data = json.loads(_FIXTURE_PATH.read_text())
    data = deepcopy(data)
    data["scene_index"] = scene_index
    return data


def analyze_scene(
    scene: Scene,
    frame_paths: list[Path],
    onset_hints: list[float],
    preset: str,
    dry_run: bool = False,
    strategy: dict[str, Any] | None = None,
    total_scenes: int = 1,
) -> dict[str, Any]:
    if dry_run:
        logger.info("Dry-run: loading fixture for scene %d", scene.index)
        return _load_fixture(scene.index)

    sampled, frame_times = _sample_scene_frames(
        frame_paths, fps=FRAME_RATE, max_frames=_MAX_FRAMES_PER_SCENE,
    )
    if len(sampled) < _MIN_FRAMES_PER_SCENE and len(frame_paths) > 0:
        logger.warning(
            "Scene %d only has %d frames (< %d preferred); proceeding anyway",
            scene.index, len(sampled), _MIN_FRAMES_PER_SCENE,
        )
    if not sampled:
        logger.error("Scene %d has no frames; returning fallback", scene.index)
        return _fallback_response(scene)

    strategy = strategy or {}
    scene_ctx = scene_context_from_strategy(scene, strategy, total_scenes)
    frame_ts_str = ", ".join(
        f"frame {i + 1} = {t:.2f}s" for i, t in enumerate(frame_times)
    )

    has_speech = _scene_has_speech(scene, strategy.get("speech_regions"))
    speech_note = (
        "NOTE: This scene contains SPEECH. Do not add SFX over spoken words "
        "unless it's a hard cut or deliberate punctuation."
        if has_speech else ""
    )

    prompt_text = SCENE_PROMPT_TEMPLATE.format(
        scene_index=scene.index,
        scene_start=scene.start_sec,
        scene_end=scene.end_sec,
        scene_duration=scene.duration_sec,
        scene_role=scene_ctx["scene_role"],
        energy_level=scene_ctx["energy_level"],
        style=strategy.get("style") or "UGC short-form ad",
        vibe=strategy.get("vibe") or "neutral",
        n_frames=len(sampled),
        frame_timestamps=frame_ts_str,
        onset_hints=_format_onset_hints(onset_hints),
        speech_note=speech_note,
        scene_specific_directive=scene_ctx["scene_specific_directive"] or "(none)",
        scene_sfx_budget=scene_ctx["scene_sfx_budget"],
        silence_note=scene_ctx["silence_note"] or "",
        anchor_note=scene_ctx["anchor_note"] or "",
    )

    model = _get_model()
    images = [Image.open(p) for p in sampled]
    request_parts: list[Any] = [prompt_text, *images]

    last_exc: Exception | None = None
    for attempt in range(len(_RETRY_BACKOFFS_SEC) + 1):
        try:
            response = model.generate_content(
                request_parts,
                generation_config={"response_mime_type": "application/json"},
            )
            data = json.loads(response.text)
            missing = _REQUIRED_KEYS - set(data.keys())
            if missing:
                raise ValueError(f"Response missing required keys: {missing}")
            data["scene_index"] = scene.index
            for action in data.get("actions") or []:
                if isinstance(action, dict):
                    _normalize_action(action)
            return data
        except Exception as exc:
            last_exc = exc
            logger.error(
                "Gemini attempt %d/%d failed for scene %d: %s",
                attempt + 1, len(_RETRY_BACKOFFS_SEC) + 1, scene.index, exc,
            )
            if attempt < len(_RETRY_BACKOFFS_SEC):
                time.sleep(_RETRY_BACKOFFS_SEC[attempt])

    logger.error(
        "All retries exhausted for scene %d (last error: %s); using fallback",
        scene.index, last_exc,
    )
    return _fallback_response(scene)


def _slice_frames_for_scene(
    all_frames: list[Path],
    scene: Scene,
    fps: int,
) -> list[Path]:
    start_idx = max(0, int(scene.start_sec * fps))
    end_idx = max(start_idx + 1, int(round(scene.end_sec * fps)))
    end_idx = min(end_idx, len(all_frames))
    sliced = all_frames[start_idx:end_idx]
    if not sliced and all_frames:
        sliced = [all_frames[min(start_idx, len(all_frames) - 1)]]
    return sliced


def prepare_for_analysis(video_path: Path) -> dict[str, Any]:
    """Extract frames, detect scenes, detect onsets + speech, compute duration.

    Idempotent / cache-friendly: returns a dict with the keys
    ``frames`` (list[Path]), ``scenes`` (list[Scene]), ``onsets`` (list[float]),
    ``speech_regions`` (list[(float, float)]), ``audio_path`` (Path | None),
    ``duration_sec`` (float)."""
    all_frames = extract_frames(video_path, fps=FRAME_RATE)
    scenes = detect_scenes(video_path)

    onsets: list[float] = []
    speech_regions: list[tuple[float, float]] = []
    audio_path: Path | None = None
    try:
        audio_path = extract_audio(video_path)
        onsets = detect_onsets(audio_path)
        try:
            speech_regions = detect_speech_regions(audio_path)
        except Exception as exc:
            logger.warning("Speech detection failed: %s", exc)
    except ValueError as exc:
        logger.warning("No audio track for onset detection: %s", exc)

    duration_sec = scenes[-1].end_sec if scenes else 0.0
    return {
        "frames": all_frames,
        "scenes": scenes,
        "onsets": onsets,
        "speech_regions": speech_regions,
        "audio_path": audio_path,
        "duration_sec": duration_sec,
    }


def analyze_scenes_parallel(
    scenes: list[Scene],
    all_frames: list[Path],
    onsets: list[float],
    preset: str,
    strategy: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Per-scene Gemini analysis, one parallel batch. Strategy (when provided)
    is injected into each scene's prompt as global context."""
    scene_payloads: list[tuple[Scene, list[Path], list[float]]] = []
    for scene in scenes:
        scene_frames = _slice_frames_for_scene(all_frames, scene, FRAME_RATE)
        scene_onsets = [
            o - scene.start_sec for o in onsets
            if scene.start_sec <= o < scene.end_sec
        ]
        scene_payloads.append((scene, scene_frames, scene_onsets))

    results: list[dict[str, Any] | None] = [None] * len(scenes)
    total = len(scenes)

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {
            pool.submit(
                analyze_scene, scene, frames, hints, preset, dry_run, strategy, total,
            ): scene
            for scene, frames, hints in scene_payloads
        }
        for done_count, fut in enumerate(as_completed(futures), start=1):
            scene = futures[fut]
            t0 = time.time()
            results[scene.index] = fut.result()
            logger.info(
                "Analyzed scene %d/%d (took %.2fs)",
                done_count, total, time.time() - t0,
            )

    return [r for r in results if r is not None]


def analyze_video(
    video_path: Path,
    preset: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    logger.info("Analyzing %s (preset=%s, dry_run=%s)", video_path, preset, dry_run)

    prep = prepare_for_analysis(video_path)
    scenes: list[Scene] = prep["scenes"]
    all_frames: list[Path] = prep["frames"]
    onsets: list[float] = prep["onsets"]
    speech_regions: list[tuple[float, float]] = prep["speech_regions"]
    duration_sec: float = prep["duration_sec"]

    output_dir = TEMP_DIR / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    strategy = run_director_pass(
        video_path=video_path,
        frame_paths=all_frames,
        scenes=scenes,
        onsets=onsets,
        user_preset=preset,
        duration_sec=duration_sec,
        dry_run=dry_run,
        speech_regions=speech_regions,
    )
    (output_dir / "strategy.json").write_text(json.dumps(strategy, indent=2))

    if strategy.get("override_user_preset"):
        logger.info(
            "Director suggests preset override: %s -> %s (reason: %s)",
            preset, strategy.get("recommended_preset"), strategy.get("override_reason"),
        )

    scene_results = analyze_scenes_parallel(
        scenes, all_frames, onsets, preset, strategy=strategy, dry_run=dry_run,
    )

    aggregated: dict[str, Any] = {
        "video_path": str(video_path),
        "preset": preset,
        "duration_sec": duration_sec,
        "strategy": strategy,
        "scenes": scene_results,
    }

    output_path = output_dir / "analysis.json"
    output_path.write_text(json.dumps(aggregated, indent=2))
    logger.info("Saved analysis to %s", output_path)

    return aggregated


def _summarize(result: dict[str, Any]) -> str:
    scenes = result.get("scenes", [])
    total_actions = 0
    confidences: list[float] = []
    for scene in scenes:
        actions = scene.get("actions", [])
        total_actions += len(actions)
        for a in actions:
            c = a.get("confidence")
            if isinstance(c, (int, float)):
                confidences.append(float(c))
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return (
        f"Scenes: {len(scenes)} | "
        f"Actions: {total_actions} | "
        f"Avg confidence: {avg_conf:.2f}"
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a video with Gemini and emit an SFX plan JSON.")
    parser.add_argument("video_path", type=Path, help="Path to input video file.")
    from src.presets import PRESETS
    parser.add_argument(
        "--preset",
        default="tiktok_viral",
        choices=list(PRESETS.keys()),
        help="Mood preset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip Gemini API calls; load fixture instead.",
    )
    args = parser.parse_args()

    result = analyze_video(args.video_path, preset=args.preset, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
    print()
    print(_summarize(result))


if __name__ == "__main__":
    _main()
