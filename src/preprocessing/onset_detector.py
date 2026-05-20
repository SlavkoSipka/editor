from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np

from src.preprocessing.audio_extractor import extract_audio
from src.utils.logger import get_logger

logger = get_logger("onset_detector")

_SILENCE_RMS_THRESHOLD = 0.001
_MIN_AUDIO_DURATION_SEC = 0.5


def detect_onsets(
    audio_path: Path,
    sensitivity: float = 0.5,
) -> list[float]:
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info("Detecting onsets in %s", audio_path)

    y, sr = librosa.load(str(audio_path), sr=None, mono=True)

    duration_sec = float(len(y)) / float(sr) if sr else 0.0
    if duration_sec < _MIN_AUDIO_DURATION_SEC:
        logger.warning(
            "Audio too short for onset detection (%.3fs < %.2fs); returning no onsets",
            duration_sec, _MIN_AUDIO_DURATION_SEC,
        )
        return []

    rms_mean = float(np.mean(librosa.feature.rms(y=y)))
    if rms_mean < _SILENCE_RMS_THRESHOLD:
        logger.warning(
            "Audio appears silent (mean RMS %.6f); returning no onsets", rms_mean,
        )
        return []

    # Onset detection is best-effort: librosa internals occasionally choke on
    # unusual audio (extreme dynamic range, near-DC content). Failure here
    # must not break the pipeline — Gemini timestamps still work without it.
    try:
        onsets = librosa.onset.onset_detect(
            y=y,
            sr=sr,
            units="time",
            delta=sensitivity,
        )
    except Exception as exc:
        logger.warning("librosa onset detection failed: %s; returning no onsets", exc)
        return []

    timestamps = sorted(float(t) for t in onsets)
    if not timestamps:
        logger.info("No onsets detected (silent or music-only audio)")
    else:
        logger.info("Found %d onsets", len(timestamps))
    return timestamps


def detect_onsets_from_video(video_path: Path) -> list[float]:
    """Extract audio from video, then detect onsets. Returns [] if video has no audio."""
    try:
        audio_path = extract_audio(video_path)
    except ValueError as exc:
        logger.warning("Skipping onset detection: %s", exc)
        return []
    return detect_onsets(audio_path)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Detect audio onset timestamps in a video.")
    parser.add_argument("video_path", type=Path, help="Path to input video file.")
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=0.5,
        help="Onset detection delta (lower = more onsets). Default 0.5.",
    )
    args = parser.parse_args()

    try:
        audio_path = extract_audio(args.video_path)
    except ValueError as exc:
        print(f"No onsets detected — {exc}.")
        return

    onsets = detect_onsets(audio_path, sensitivity=args.sensitivity)
    if not onsets:
        print("No onsets detected — audio may be silent, music-only, or very quiet.")
        return

    for i, t in enumerate(onsets, start=1):
        print(f"Onset {i}: {t:.2f}s")
    print(f"Total: {len(onsets)} onsets")


if __name__ == "__main__":
    _main()
