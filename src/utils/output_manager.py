"""Versioned output management for per-video pipeline runs.

Each `process.py` invocation writes to ``data/temp/{video_name}/outputs/`` with
a unique versioned filename, a sidecar JSON of run metadata, an aggregate
``index.json`` of all runs, and a ``latest.mp4`` copy of the most recent run.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

_PRESET_SAFE_RE = re.compile(r"[^a-z0-9_]+")
_VERSION_RE = re.compile(r"^v(\d+)(?:_.*)?$")


def _sanitize_preset(preset_name: str) -> str:
    safe = (preset_name or "preset").strip().lower().replace(" ", "_")
    safe = _PRESET_SAFE_RE.sub("_", safe).strip("_")
    return safe or "preset"


def get_next_version_path(
    work_dir: Path,
    preset_name: str,
) -> tuple[Path, int]:
    """Return ``(output_path, version_number)`` for the next run.

    Scans ``{work_dir}/outputs/v*.mp4`` for the highest existing version,
    increments by 1, and builds a filename like ``v007_tiktok_viral.mp4``.
    """
    outputs_dir = work_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    max_version = 0
    for f in outputs_dir.glob("v*.mp4"):
        m = _VERSION_RE.match(f.stem)
        if not m:
            continue
        try:
            max_version = max(max_version, int(m.group(1)))
        except ValueError:
            continue

    next_version = max_version + 1
    filename = f"v{next_version:03d}_{_sanitize_preset(preset_name)}.mp4"
    return outputs_dir / filename, next_version


def write_version_metadata(
    output_path: Path,
    version: int,
    preset_name: str,
    match_plan: dict[str, Any],
    pipeline_stats: dict[str, Any],
    project_zip: Path | None = None,
    timeline_formats: list[str] | None = None,
) -> Path:
    """Write a sidecar JSON file capturing the settings + summary of this run."""
    matches = match_plan.get("matches") or []
    sfx_count = sum(1 for m in matches if m.get("layer") == "sfx")
    ambient_count = sum(1 for m in matches if m.get("layer") == "ambient")
    music = match_plan.get("music") or {}

    metadata = {
        "version": version,
        "preset": preset_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_file": output_path.name,
        "video_path": match_plan.get("video_path"),
        "duration_sec": match_plan.get("duration_sec"),
        "sfx_count": sfx_count,
        "ambient_count": ambient_count,
        "music_track": music.get("sound_name") if music else None,
        "music_mood": music.get("selected_mood") if music else None,
        "project_zip": project_zip.name if project_zip is not None else None,
        "timeline_formats": list(timeline_formats or []),
        "pipeline_stats": pipeline_stats,
    }

    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(metadata, indent=2))
    return json_path


def update_outputs_index(work_dir: Path) -> Path | None:
    """Rebuild ``outputs/index.json`` from every ``v*.json`` sidecar.

    Sorted by version descending. Returns the index path, or ``None`` if the
    outputs directory doesn't exist yet.
    """
    outputs_dir = work_dir / "outputs"
    if not outputs_dir.is_dir():
        return None

    versions: list[dict[str, Any]] = []
    for json_file in sorted(outputs_dir.glob("v*.json")):
        try:
            meta = json.loads(json_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(meta, dict):
            versions.append(meta)

    versions.sort(key=lambda v: int(v.get("version") or 0), reverse=True)

    index_path = outputs_dir / "index.json"
    index_path.write_text(
        json.dumps({"versions": versions, "count": len(versions)}, indent=2)
    )
    return index_path


def update_latest_link(output_path: Path) -> Path:
    """Refresh ``outputs/latest.mp4`` to a copy of the most recent run.

    Uses a file copy (not a symlink) for cross-platform safety on macOS/Windows.
    """
    latest = output_path.parent / "latest.mp4"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    shutil.copy2(output_path, latest)
    return latest
