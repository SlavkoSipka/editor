from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from src.config import FRAME_RATE, TEMP_DIR
from src.utils.logger import get_logger

logger = get_logger("frame_extractor")


def extract_frames(
    video_path: Path,
    output_dir: Path | None = None,
    fps: int = FRAME_RATE,
) -> list[Path]:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_dir is None:
        output_dir = TEMP_DIR / video_path.stem / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = output_dir / "frame_%04d.jpg"

    logger.info("Extracting frames from %s at %d fps", video_path, fps)

    cmd: list[str] = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        str(output_pattern),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed (exit {result.returncode}) extracting frames "
            f"from {video_path}:\n{result.stderr}"
        )

    frames = sorted(output_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError(
            f"FFmpeg produced 0 frames for {video_path}. "
            f"Video may be corrupt or unreadable."
        )

    logger.info("Extracted %d frames to %s", len(frames), output_dir)
    return frames


def _main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from a video via FFmpeg.")
    parser.add_argument("video_path", type=Path, help="Path to input video file.")
    parser.add_argument("--fps", type=int, default=FRAME_RATE, help="Sampling frame rate.")
    args = parser.parse_args()

    frames = extract_frames(args.video_path, fps=args.fps)
    print(f"Extracted {len(frames)} frames to {frames[0].parent}")
    print("First 3:")
    for frame in frames[:3]:
        print(f"  {frame}")
    print(f"Last: {frames[-1]}")


if __name__ == "__main__":
    _main()
