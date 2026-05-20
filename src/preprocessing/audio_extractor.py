from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import librosa

from src.config import AUDIO_SAMPLE_RATE, TEMP_DIR
from src.utils.logger import get_logger

logger = get_logger("audio_extractor")


def _has_audio_stream(video_path: Path) -> bool:
    """Probe video with ffmpeg to detect whether an audio stream is present."""
    result = subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-hide_banner"],
        capture_output=True,
        text=True,
    )
    # ffmpeg with no output args exits non-zero but prints stream info to stderr.
    return "Audio:" in result.stderr


def extract_audio(
    video_path: Path,
    output_path: Path | None = None,
    sample_rate: int = AUDIO_SAMPLE_RATE,
) -> Path:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not _has_audio_stream(video_path):
        raise ValueError("Video has no audio track")

    if output_path is None:
        output_path = TEMP_DIR / video_path.stem / "audio.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting audio from %s", video_path)

    cmd: list[str] = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", str(sample_rate),
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed (exit {result.returncode}) extracting audio "
            f"from {video_path}:\n{result.stderr}"
        )

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise ValueError("Video has no audio track")

    duration_sec = float(librosa.get_duration(path=str(output_path)))
    logger.info("Extracted audio to %s (%.2fs)", output_path, duration_sec)
    return output_path


def _main() -> None:
    parser = argparse.ArgumentParser(description="Extract mono WAV audio from a video via FFmpeg.")
    parser.add_argument("video_path", type=Path, help="Path to input video file.")
    args = parser.parse_args()

    output = extract_audio(args.video_path)
    duration_sec = float(librosa.get_duration(path=str(output)))
    print(f"Wrote {output} ({duration_sec:.2f}s)")


if __name__ == "__main__":
    _main()
