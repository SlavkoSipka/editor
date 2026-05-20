"""Timeline export to OTIO / FCPXML / EDL formats.

Builds an OpenTimelineIO timeline from a match plan, with each SFX, ambient,
and music clip on its own labeled track so editors can move/mute/replace
individual sounds in Premiere, Resolve, or Final Cut."""

from __future__ import annotations

from pathlib import Path

import opentimelineio as otio

from src.utils.logger import get_logger

logger = get_logger("timeline_exporter")


_FRAMERATE = 24

TRACK_VIDEO = "V1 - Source Video"
TRACK_ORIGINAL_AUDIO = "A1 - Original Audio"
TRACK_MUSIC = "A2 - Music Bed"


def _rt(seconds: float) -> otio.opentime.RationalTime:
    return otio.opentime.RationalTime(round(float(seconds) * _FRAMERATE), _FRAMERATE)


def _rt_range(start_sec: float, duration_sec: float) -> otio.opentime.TimeRange:
    duration_sec = max(0.001, float(duration_sec))
    return otio.opentime.TimeRange(
        start_time=_rt(max(0.0, float(start_sec))),
        duration=_rt(duration_sec),
    )


def _make_media_ref(
    media_path: Path,
    media_duration_sec: float,
) -> otio.schema.ExternalReference:
    return otio.schema.ExternalReference(
        target_url=Path(media_path).resolve().as_uri(),
        available_range=_rt_range(0.0, media_duration_sec),
    )


def _make_clip(
    media_path: Path,
    media_in_sec: float,
    duration_sec: float,
    clip_name: str,
    media_total_duration_sec: float | None = None,
) -> otio.schema.Clip:
    """Create a clip that plays ``[media_in_sec, media_in_sec + duration_sec]``
    of ``media_path``. The available range is the full media duration so
    editors can extend the clip in either direction."""
    total = float(media_total_duration_sec) if media_total_duration_sec else (
        max(duration_sec, media_in_sec + duration_sec)
    )
    media_ref = _make_media_ref(media_path, total)
    return otio.schema.Clip(
        name=clip_name,
        media_reference=media_ref,
        source_range=_rt_range(media_in_sec, duration_sec),
    )


def _append_at(
    track: otio.schema.Track,
    clip: otio.schema.Clip,
    timeline_offset_sec: float,
) -> None:
    """Insert a leading Gap so the clip starts at ``timeline_offset_sec``."""
    if timeline_offset_sec > 0:
        gap = otio.schema.Gap(
            name="(silence)",
            source_range=_rt_range(0.0, timeline_offset_sec),
        )
        track.append(gap)
    track.append(clip)


def _short_label(match: dict) -> str:
    action = str(match.get("action_type") or "sound")
    return action.replace("_", " ")[:24]


def build_timeline(
    video_path: Path,
    original_audio_path: Path,
    match_plan: dict,
    video_duration_sec: float,
    project_name: str = "AI SFX Export",
) -> otio.schema.Timeline:
    """Build an OTIO Timeline from the match plan.

    Layout:
      V1   Source video
      A1   Original audio
      A2   Music bed (if present)
      A3+  One track per ambient
      then One track per SFX, sorted by time
    """
    timeline = otio.schema.Timeline(name=project_name)
    timeline.global_start_time = _rt(0.0)

    v_track = otio.schema.Track(
        name=TRACK_VIDEO, kind=otio.schema.TrackKind.Video,
    )
    v_track.append(_make_clip(
        media_path=video_path,
        media_in_sec=0.0,
        duration_sec=video_duration_sec,
        clip_name=f"Source: {video_path.name}",
        media_total_duration_sec=video_duration_sec,
    ))
    timeline.tracks.append(v_track)

    a1_track = otio.schema.Track(
        name=TRACK_ORIGINAL_AUDIO, kind=otio.schema.TrackKind.Audio,
    )
    a1_track.append(_make_clip(
        media_path=original_audio_path,
        media_in_sec=0.0,
        duration_sec=video_duration_sec,
        clip_name="Original Audio",
        media_total_duration_sec=video_duration_sec,
    ))
    timeline.tracks.append(a1_track)

    music = match_plan.get("music") or None
    if music and music.get("sound_path"):
        m_track = otio.schema.Track(
            name=TRACK_MUSIC, kind=otio.schema.TrackKind.Audio,
        )
        music_dur = float(music.get("sound_duration") or video_duration_sec)
        m_track.append(_make_clip(
            media_path=Path(music["sound_path"]),
            media_in_sec=0.0,
            duration_sec=video_duration_sec,
            clip_name=f"Music: {music.get('sound_name', 'music')}",
            media_total_duration_sec=max(music_dur, video_duration_sec),
        ))
        timeline.tracks.append(m_track)

    matches = match_plan.get("matches") or []
    ambient_matches = [m for m in matches if m.get("layer") == "ambient"]
    sfx_matches = sorted(
        (m for m in matches if m.get("layer") == "sfx"),
        key=lambda m: float(m.get("absolute_timestamp") or 0.0),
    )

    next_track_idx = 3 if (music and music.get("sound_path")) else 2

    for i, amb in enumerate(ambient_matches, start=1):
        scene_start = float(amb.get("scene_start_sec") or 0.0)
        scene_end = float(amb.get("scene_end_sec") or (scene_start + 5.0))
        scene_duration = max(0.5, scene_end - scene_start)
        track_label = f"A{next_track_idx} - Ambient {i} ({_short_label(amb)})"
        track = otio.schema.Track(
            name=track_label, kind=otio.schema.TrackKind.Audio,
        )
        clip = _make_clip(
            media_path=Path(amb.get("sound_path") or ""),
            media_in_sec=0.0,
            duration_sec=scene_duration,
            clip_name=f"Ambient: {Path(amb.get('sound_name') or 'ambient').stem}",
            media_total_duration_sec=float(amb.get("sound_duration") or scene_duration),
        )
        _append_at(track, clip, scene_start)
        timeline.tracks.append(track)
        next_track_idx += 1

    for i, sfx in enumerate(sfx_matches, start=1):
        abs_ts = float(sfx.get("absolute_timestamp") or 0.0)
        sound_dur = float(sfx.get("sound_duration") or 1.0)
        track_label = f"A{next_track_idx} - SFX {i} ({_short_label(sfx)})"
        track = otio.schema.Track(
            name=track_label, kind=otio.schema.TrackKind.Audio,
        )
        clip = _make_clip(
            media_path=Path(sfx.get("sound_path") or ""),
            media_in_sec=0.0,
            duration_sec=sound_dur,
            clip_name=f"SFX: {Path(sfx.get('sound_name') or 'sfx').stem} @ {abs_ts:.2f}s",
            media_total_duration_sec=sound_dur,
        )
        _append_at(track, clip, abs_ts)
        timeline.tracks.append(track)
        next_track_idx += 1

    return timeline


def export_all_formats(
    timeline: otio.schema.Timeline,
    output_dir: Path,
    base_name: str,
) -> dict[str, Path]:
    """Export to every available NLE format. Missing adapters are logged and
    skipped \u2014 the pipeline never fails just because Premiere XML isn't
    installed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    available = set(otio.adapters.available_adapter_names())
    results: dict[str, Path] = {}

    targets: list[tuple[str, str, str]] = [
        ("otio",     "otio_json",                   ".otio"),
        ("fcpxml",   "otio_fcpx_xml_lite_adapter",  ".fcpxml"),
        ("premiere", "premiere_xml",                ".xml"),
        ("edl",      "cmx_3600",                    ".edl"),
    ]

    for label, adapter_name, ext in targets:
        if adapter_name not in available:
            logger.warning(
                "%s adapter not installed (%s); skipping.",
                label, adapter_name,
            )
            continue
        out_path = output_dir / f"{base_name}{ext}"
        try:
            otio.adapters.write_to_file(timeline, str(out_path), adapter_name=adapter_name)
            results[label] = out_path
            logger.info("Exported %s: %s", label.upper(), out_path.name)
        except Exception as exc:
            logger.warning("%s export failed: %s", label.upper(), exc)
    return results
