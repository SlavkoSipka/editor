"""Beat detection on the selected music track.

Wraps ``librosa.beat.beat_track`` and exposes small helpers for the matcher
to align "anchor" SFX (hook, drops, logo sting) to the music grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("beat_detector")

_SR = 22050
_MIN_AUDIO_SEC = 2.0


def detect_beats(
    music_path: Path | str,
    duration_limit_sec: float | None = None,
) -> tuple[float, list[float]]:
    """Detect tempo (BPM) and beat timestamps in a music track.

    Returns ``(tempo_bpm, beat_times_sec)``. On failure or for tracks without a
    detectable pulse (drones, pads, ambiences) returns ``(0.0, [])``."""
    music_path = Path(music_path)
    if not music_path.is_file():
        logger.warning("Music file not found: %s", music_path)
        return 0.0, []

    try:
        y, sr = librosa.load(
            music_path, sr=_SR, mono=True, duration=duration_limit_sec,
        )
    except Exception as exc:
        logger.warning("Could not load music for beat detection: %s", exc)
        return 0.0, []

    if len(y) < sr * _MIN_AUDIO_SEC:
        logger.warning("Music too short for reliable beat detection")
        return 0.0, []

    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    except Exception as exc:
        logger.warning("Beat detection failed: %s", exc)
        return 0.0, []

    if isinstance(tempo, np.ndarray):
        tempo_bpm = float(tempo[0]) if tempo.size > 0 else 0.0
    else:
        tempo_bpm = float(tempo)

    beat_times = [float(t) for t in beat_times]
    logger.info("Detected tempo: %.1f BPM, %d beats", tempo_bpm, len(beat_times))
    return tempo_bpm, beat_times


def find_nearest_beat(
    timestamp: float, beats: list[float],
) -> tuple[float, float]:
    """Return ``(nearest_beat_time_sec, distance_ms)``. Distance is signed."""
    if not beats:
        return timestamp, float("inf")
    nearest = min(beats, key=lambda b: abs(b - timestamp))
    return float(nearest), (float(nearest) - timestamp) * 1000.0


def find_strong_beats(beats: list[float], every_n: int = 4) -> list[float]:
    """Return every Nth beat (assumed downbeats). For 4/4 with ``every_n=4``
    this gives the "1"-beats."""
    if every_n <= 0:
        return list(beats)
    return list(beats[::every_n])


def _main() -> None:
    parser = argparse.ArgumentParser(description="Detect beats in an audio file.")
    parser.add_argument("music_path", type=Path)
    parser.add_argument("--duration", type=float, default=None,
                        help="Limit analysis to first N seconds.")
    args = parser.parse_args()
    tempo, beats = detect_beats(args.music_path, duration_limit_sec=args.duration)
    print(f"Tempo: {tempo:.1f} BPM")
    preview = ", ".join(f"{b:.2f}s" for b in beats[:10])
    print(f"First 10 beats: [{preview}]")
    print(f"Total beats: {len(beats)}")
    if len(beats) >= 2:
        avg_interval = (beats[-1] - beats[0]) / (len(beats) - 1)
        if avg_interval > 0:
            print(
                f"Avg beat interval: {avg_interval * 1000:.0f}ms "
                f"(= {60.0 / avg_interval:.1f} BPM)"
            )


if __name__ == "__main__":
    _main()
