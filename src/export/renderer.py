from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from src.config import TEMP_DIR
from src.utils.logger import get_logger

logger = get_logger("renderer")


def _video_duration_sec(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return 0.0
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def render_final_video(
    original_video_path: Path,
    mixed_audio_path: Path,
    output_path: Path,
) -> Path:
    if not original_video_path.is_file():
        raise FileNotFoundError(f"Original video not found: {original_video_path}")
    if not mixed_audio_path.is_file():
        raise FileNotFoundError(f"Mixed audio not found: {mixed_audio_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Rendering final video: %s + %s -> %s",
        original_video_path, mixed_audio_path, output_path,
    )

    cmd: list[str] = [
        "ffmpeg",
        "-y",
        "-i", str(original_video_path),
        "-i", str(mixed_audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed (exit {result.returncode}) rendering "
            f"{output_path}:\n{result.stderr}"
        )

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"FFmpeg reported success but output is missing or empty: {output_path}"
        )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Final video rendered: %s (%.1f MB)", output_path, size_mb)
    return output_path


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Mux the original video with the mixed audio into a final MP4.",
    )
    parser.add_argument("video_path", type=Path, help="Path to input video file.")
    parser.add_argument(
        "--preset",
        default="dramatic",
        choices=["dramatic", "energetic", "mysterious", "playful", "luxury"],
        help="Mood preset (informational only; mixing already happened).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit output path. Defaults to data/temp/{video_name}/final.mp4.",
    )
    args = parser.parse_args()

    if not args.video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {args.video_path}")

    video_dir = TEMP_DIR / args.video_path.stem
    mixed_audio_path = video_dir / "mixed_audio.wav"

    if not mixed_audio_path.is_file():
        raise RuntimeError(
            f"Mixed audio not found at {mixed_audio_path}. "
            f"Run `python -m src.mixing.mixer {args.video_path} "
            f"--preset {args.preset}` first."
        )

    output_path = args.output if args.output is not None else video_dir / "final.mp4"

    render_final_video(args.video_path, mixed_audio_path, output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    duration_sec = _video_duration_sec(output_path)

    print("\u2713 Final video rendered")
    print(f"  Input video:  {args.video_path}")
    print(f"  Mixed audio:  {mixed_audio_path}")
    print(f"  Output:       {output_path}")
    print(f"  Size:         {size_mb:.1f} MB")
    print(f"  Duration:     {duration_sec:.2f}s")
    print()
    print(f"  Play it: open {output_path}")


if __name__ == "__main__":
    _main()
