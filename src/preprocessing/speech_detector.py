"""Detect speech regions in audio using energy + spectral analysis.

Lightweight heuristic (no models) — combines RMS energy, spectral centroid,
and spectral rolloff. Good enough to drive ducking and director context for
short-form video where the question is just "is someone talking here?"."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("speech_detector")


_FRAME_DURATION_SEC = 0.025
_HOP_DURATION_SEC = 0.010
_SR = 16000

_RMS_THRESHOLD = 0.2
_CENTROID_MIN_HZ = 500.0
_CENTROID_MAX_HZ = 3500.0
_ROLLOFF_MAX_HZ = 6000.0


def detect_speech_regions(
    audio_path: Path | str,
    min_region_sec: float = 0.3,
    merge_gap_sec: float = 0.3,
) -> list[tuple[float, float]]:
    """Find time regions where speech is likely present.

    Returns a list of ``(start_sec, end_sec)`` tuples.

    Algorithm:
    1. Resample to 16 kHz mono.
    2. Compute frame-level RMS energy + spectral centroid + rolloff.
    3. Mark frames where energy is non-trivial and the spectrum sits in the
       typical human-voice band.
    4. Merge regions closer than ``merge_gap_sec``.
    5. Drop regions shorter than ``min_region_sec``."""
    audio_path = Path(audio_path)
    try:
        y, sr = librosa.load(audio_path, sr=_SR, mono=True)
    except Exception as exc:
        logger.warning("Could not load audio for speech detection: %s", exc)
        return []

    if len(y) < sr * 0.5:
        return []

    frame_length = int(sr * _FRAME_DURATION_SEC)
    hop_length = int(sr * _HOP_DURATION_SEC)

    rms = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop_length,
    )[0]
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=hop_length,
    )[0]
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, hop_length=hop_length, roll_percent=0.85,
    )[0]

    rms_ref = float(np.percentile(rms, 95)) + 1e-9
    rms_norm = np.clip(rms / rms_ref, 0.0, 1.0)

    energy_ok = rms_norm > _RMS_THRESHOLD
    centroid_ok = (centroid >= _CENTROID_MIN_HZ) & (centroid <= _CENTROID_MAX_HZ)
    rolloff_ok = rolloff <= _ROLLOFF_MAX_HZ

    n = min(len(energy_ok), len(centroid_ok), len(rolloff_ok))
    speech_mask = energy_ok[:n] & centroid_ok[:n] & rolloff_ok[:n]
    times = librosa.frames_to_time(
        np.arange(n), sr=sr, hop_length=hop_length,
    )

    regions: list[tuple[float, float]] = []
    in_region = False
    start_t = 0.0
    last_t = float(times[-1]) if n else 0.0
    for t, is_speech in zip(times, speech_mask):
        t = float(t)
        if is_speech and not in_region:
            start_t = t
            in_region = True
        elif not is_speech and in_region:
            regions.append((start_t, t))
            in_region = False
    if in_region:
        regions.append((start_t, last_t))

    merged: list[tuple[float, float]] = []
    for s, e in regions:
        if merged and s - merged[-1][1] < merge_gap_sec:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    final = [(s, e) for s, e in merged if (e - s) >= min_region_sec]

    total = sum(e - s for s, e in final)
    logger.info(
        "Detected %d speech region(s), total %.1fs", len(final), total,
    )
    return final


def is_in_speech_region(
    timestamp: float,
    regions: list[tuple[float, float]],
    padding_sec: float = 0.1,
) -> bool:
    """Whether ``timestamp`` falls inside any region (with symmetric padding)."""
    for s, e in regions:
        if (float(s) - padding_sec) <= timestamp <= (float(e) + padding_sec):
            return True
    return False


def coverage_ratio(
    regions: list[tuple[float, float]],
    total_duration_sec: float,
) -> float:
    """Fraction of the video covered by speech, in ``[0, 1]``."""
    if total_duration_sec <= 0:
        return 0.0
    speech_time = sum(float(e) - float(s) for s, e in regions)
    return min(1.0, speech_time / total_duration_sec)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect speech regions in an audio file.",
    )
    parser.add_argument("audio_path", type=Path)
    args = parser.parse_args()
    regions = detect_speech_regions(args.audio_path)
    for i, (s, e) in enumerate(regions, start=1):
        print(f"  Region {i}: {s:.2f}s \u2014 {e:.2f}s (duration {e - s:.2f}s)")
    print(f"\nTotal: {len(regions)} region(s)")


if __name__ == "__main__":
    _main()
