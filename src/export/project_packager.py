"""Package the final project as a ZIP for editor import.

Bundles the rendered MP4, source video + audio, all referenced library assets,
the timeline files (OTIO / FCPXML / EDL / XML), and a README explaining how
to import into Premiere, Resolve, or Final Cut."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger("project_packager")


def _readme_text(
    preset_name: str,
    version: int,
    match_plan: dict[str, Any],
    timeline_files: dict[str, Path],
) -> str:
    matches = match_plan.get("matches") or []
    sfx_count = sum(1 for m in matches if m.get("layer") == "sfx")
    ambient_count = sum(1 for m in matches if m.get("layer") == "ambient")
    music_name = (match_plan.get("music") or {}).get("sound_name") or "(none)"
    formats_list = "\n".join(f"  - timeline{p.suffix}" for p in timeline_files.values())

    return (
        f"AI Sound Effects \u2014 Project Export v{version:03d}\n"
        "================================================================\n"
        f"Preset:     {preset_name}\n"
        f"SFX count:  {sfx_count}\n"
        f"Ambient:    {ambient_count}\n"
        f"Music:      {music_name}\n\n"
        "CONTENTS\n"
        "--------\n"
        "  - final.mp4           The rendered video with all SFX baked in\n"
        "  - original.mp4        The source video (relink to this when importing)\n"
        "  - original_audio.wav  Extracted source audio\n"
        "  - match_plan.json     Full metadata about every SFX decision\n"
        f"{formats_list}\n"
        "  - assets/             All referenced audio files (music, ambient, SFX)\n\n"
        "HOW TO IMPORT\n"
        "=============\n\n"
        "* DaVinci Resolve\n"
        "  File > Import > Timeline > choose timeline.fcpxml or timeline.otio.\n"
        "  Resolve will ask you to relink media if paths don't match \u2014 point\n"
        "  to the assets/ folder.\n\n"
        "* Adobe Premiere Pro\n"
        "  File > Import > choose timeline.xml or timeline.fcpxml.\n"
        "  If Premiere can't open .otio, install the OTIO Premiere panel.\n\n"
        "* Final Cut Pro\n"
        "  File > Import > XML > choose timeline.fcpxml.\n\n"
        "* Other (universal)\n"
        "  Any NLE supporting OpenTimelineIO can read timeline.otio.\n\n"
        "TROUBLESHOOTING\n"
        "===============\n"
        "- If audio doesn't play, the editor needs to relink to the assets/ folder.\n"
        "- All sound files are MP3; some pro editors may want WAV \u2014 convert with\n"
        "  ffmpeg if needed.\n"
        "- Tracks are named A1-Axx with descriptive labels for easy identification.\n"
    )


def _safe_copy(src: Path, dst: Path) -> bool:
    if not src.is_file():
        logger.warning("Asset missing, skipping: %s", src)
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dst)
        return True
    except Exception as exc:
        logger.warning("Copy failed (%s \u2192 %s): %s", src, dst, exc)
        return False


def package_project(
    work_dir: Path,
    output_path: Path,
    version: int,
    preset_name: str,
    match_plan: dict[str, Any],
    timeline_files: dict[str, Path],
    original_video_path: Path,
    original_audio_path: Path,
) -> Path:
    """Bundle the project into a ZIP for editor import.

    Returns the path to the created ZIP under ``work_dir/outputs/``."""
    project_name = f"project_v{version:03d}_{preset_name}"
    staging_root = work_dir / "staging"
    staging = staging_root / project_name
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging.mkdir(parents=True)

    _safe_copy(output_path, staging / "final.mp4")
    _safe_copy(original_video_path, staging / "original.mp4")
    _safe_copy(original_audio_path, staging / "original_audio.wav")

    for src in timeline_files.values():
        _safe_copy(src, staging / src.name)

    (staging / "match_plan.json").write_text(json.dumps(match_plan, indent=2))

    assets_dir = staging / "assets"
    music_dir = assets_dir / "music"
    ambient_dir = assets_dir / "ambient"
    sfx_dir = assets_dir / "sfx"
    for d in (music_dir, ambient_dir, sfx_dir):
        d.mkdir(parents=True, exist_ok=True)

    music = match_plan.get("music") or None
    if music and music.get("sound_path"):
        music_src = Path(music["sound_path"])
        _safe_copy(music_src, music_dir / music_src.name)

    seen_ids: set[Any] = set()
    for match in match_plan.get("matches") or []:
        sid = match.get("sound_id")
        if sid is None or sid in seen_ids:
            continue
        seen_ids.add(sid)
        src = Path(match.get("sound_path") or "")
        if not src.is_file():
            logger.warning("Match asset missing, skipping: %s", src)
            continue
        subdir = ambient_dir if match.get("layer") == "ambient" else sfx_dir
        _safe_copy(src, subdir / src.name)

    (staging / "README.txt").write_text(
        _readme_text(preset_name, version, match_plan, timeline_files),
    )

    outputs_dir = work_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    zip_path = outputs_dir / f"{project_name}.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in staging.rglob("*"):
            if f.is_file():
                arcname = f.relative_to(staging.parent)
                zf.write(f, arcname=arcname)

    shutil.rmtree(staging_root)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    logger.info("Project packaged: %s (%.1f MB)", zip_path, size_mb)
    return zip_path
