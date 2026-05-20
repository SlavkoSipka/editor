from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

from src.utils.logger import get_logger

logger = get_logger("scene_detector")


@dataclass
class Scene:
    index: int
    start_sec: float
    end_sec: float
    duration_sec: float


def detect_scenes(
    video_path: Path,
    threshold: float = 27.0,
    min_scene_len_sec: float = 0.5,
) -> list[Scene]:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    raw = scene_manager.get_scene_list()

    total_duration_sec = float(video.duration.get_seconds())

    if not raw:
        scene = Scene(0, 0.0, total_duration_sec, total_duration_sec)
        logger.info("Detected 1 scene (no cuts) in %s", video_path)
        return [scene]

    raw_pairs: list[tuple[float, float]] = [
        (float(start.get_seconds()), float(end.get_seconds())) for start, end in raw
    ]

    # Merge scenes shorter than min_scene_len_sec into a neighbor so the
    # final list never contains tiny micro-cuts.
    merged: list[list[float]] = []
    for start, end in raw_pairs:
        if not merged:
            merged.append([start, end])
            continue
        prev = merged[-1]
        prev_dur = prev[1] - prev[0]
        cur_dur = end - start
        if cur_dur < min_scene_len_sec:
            prev[1] = end
        elif prev_dur < min_scene_len_sec:
            prev[1] = end
        else:
            merged.append([start, end])

    scenes = [
        Scene(idx, start, end, end - start)
        for idx, (start, end) in enumerate(merged)
    ]

    logger.info("Detected %d scenes in %s", len(scenes), video_path)
    return scenes


def _main() -> None:
    parser = argparse.ArgumentParser(description="Detect scene cuts in a video using PySceneDetect.")
    parser.add_argument("video_path", type=Path, help="Path to input video file.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=27.0,
        help="ContentDetector threshold (lower = more sensitive). Default 27.0.",
    )
    args = parser.parse_args()

    scenes = detect_scenes(args.video_path, threshold=args.threshold)
    for scene in scenes:
        print(
            f"Scene {scene.index}: {scene.start_sec:.2f}s - {scene.end_sec:.2f}s "
            f"(duration: {scene.duration_sec:.2f}s)"
        )
    print(f"Total: {len(scenes)} scenes")


if __name__ == "__main__":
    _main()
