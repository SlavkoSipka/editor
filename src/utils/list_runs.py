"""List all output versions for a video.

Usage:
    python -m src.utils.list_runs path/to/video.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import TEMP_DIR


def _format_row(v: dict) -> str:
    ver = f"v{int(v.get('version') or 0):03d}"
    preset = str(v.get("preset", "?"))[:18]
    sfx = str(v.get("sfx_count", "?"))
    music = (v.get("music_track") or "—")
    music = str(music)[:33]
    when = str(v.get("timestamp", ""))[:19].replace("T", " ")
    return f"{ver:<5} {preset:<20} {sfx:<5} {music:<35} {when:<20}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", help="Path to the original video file.")
    args = parser.parse_args()

    video_name = Path(args.video_path).stem
    outputs_dir = TEMP_DIR / video_name / "outputs"
    index_path = outputs_dir / "index.json"

    if not index_path.is_file():
        print(f"No runs found for {video_name} (looked in {outputs_dir})")
        return 1

    try:
        index = json.loads(index_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"Could not read {index_path}: {exc}")
        return 1

    versions = index.get("versions") or []
    count = int(index.get("count") or len(versions))

    print(f"\n{video_name} \u2014 {count} run(s):\n")
    header = f"{'Ver':<5} {'Preset':<20} {'SFX':<5} {'Music':<35} {'When':<20}"
    print(header)
    print("\u2500" * len(header))
    for v in versions:
        print(_format_row(v))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
