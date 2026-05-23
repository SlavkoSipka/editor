from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyloudnorm
from pydub import AudioSegment

from src.config import PROJECT_ROOT, TEMP_DIR
from src.preprocessing.speech_detector import (
    coverage_ratio,
    detect_speech_regions,
    is_in_speech_region,
)
from src.presets import Preset, get_preset
from src.utils.logger import get_logger

logger = get_logger("mixer")

_TARGET_LUFS = -14.0
_MAX_NORMALIZATION_GAIN_DB = 20.0
_MAX_SFX_DURATION_MS = 30_000
_SFX_TAIL_OVERLAP_MS = 200
_SFX_FADE_OUT_MS = 150
_AMBIENT_FADE_MS = 200
_MUSIC_FADE_IN_MS = 1500
_MUSIC_FADE_OUT_MS = 2000
_MAX_MUSIC_DURATION_MS = 10 * 60 * 1000  # 10 min cap, plenty for a 3-min ad

_SFX_PEAK_TARGET_DBFS = -6.0
_SFX_LOUDNESS_TARGET_DBFS = -18.0
_SFX_FADE_IN_MS = 15
_INTENSITY_GAIN_DB: dict[str, float] = {
    "sharp": 0.0,
    "medium": -2.0,
    "soft": -4.0,
}
_TIER_GAIN_DB: dict[str, float] = {
    "anchor": 0.0,
    "accent": -3.5,
}
_TIER_GAIN_DEFAULT_DB = -2.0
SFX_MAX_DBFS_IN_MIX = -8.0
_DUCK_DURATION_SEC = 0.5
_DUCK_GAIN_DB = -3.0
_DUCK_FADE_MS = 80
_TRUE_PEAK_CEILING_DBFS = -1.0

SPEECH_MUSIC_DUCK_DB = -10.0
SPEECH_AMBIENT_DUCK_DB = -6.0
SPEECH_SFX_DUCK_DB_LIGHT = -5.0
SPEECH_SFX_DUCK_DB_HEAVY = -9.0
SPEECH_SFX_DUCK_HEAVY_COVERAGE = 0.4
SPEECH_PROTECTION_FADE_MS = 200
SPEECH_REGION_PADDING_SEC = 0.1
MUSIC_MAX_DB_WITH_SPEECH = -16.0
MUSIC_MAX_DB_WITHOUT_SPEECH = -10.0
ORIGINAL_MIN_DB_WITH_SPEECH = -2.0
SPEECH_COVERAGE_THRESHOLD = 0.15
OPENING_WINDOW_SEC = 1.5
OPENING_EARLY_SPEECH_WINDOW_SEC = 2.0
OPENING_SFX_OVER_SPEECH_DUCK_DB = -6.0


def get_sfx_speech_duck(speech_coverage: float) -> float:
    """More speech coverage = duck SFX harder so the voice always wins."""
    if speech_coverage > SPEECH_SFX_DUCK_HEAVY_COVERAGE:
        return SPEECH_SFX_DUCK_DB_HEAVY
    return SPEECH_SFX_DUCK_DB_LIGHT


def load_sfx_safely(
    path: Path, max_duration_ms: int = _MAX_SFX_DURATION_MS,
) -> AudioSegment | None:
    """Load an SFX file. Returns None if missing or corrupt. Truncates if too long."""
    if not path.is_file() or path.stat().st_size == 0:
        logger.warning("SFX file missing or empty: %s", path)
        return None
    try:
        seg = AudioSegment.from_file(path)
    except Exception as exc:
        logger.warning("Failed to load SFX %s: %s", path, exc)
        return None
    if len(seg) > max_duration_ms:
        seg = seg[:max_duration_ms]
    return seg


def normalize_loudness(
    audio_segment: AudioSegment, target_lufs: float = _TARGET_LUFS,
) -> AudioSegment:
    """Normalize a pydub AudioSegment to target integrated LUFS via pyloudnorm."""
    sr = audio_segment.frame_rate
    samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
    samples /= float(2 ** (audio_segment.sample_width * 8 - 1))
    if audio_segment.channels == 2:
        samples = samples.reshape(-1, 2)

    try:
        measured_lufs = pyloudnorm.Meter(sr).integrated_loudness(samples)
    except ValueError as exc:
        logger.warning("LUFS measurement failed (%s); skipping normalization", exc)
        return audio_segment

    if not np.isfinite(measured_lufs):
        logger.warning(
            "Audio is effectively silent (LUFS=%s); skipping normalization",
            measured_lufs,
        )
        return audio_segment

    gain_delta_db = float(target_lufs - measured_lufs)
    if gain_delta_db > _MAX_NORMALIZATION_GAIN_DB:
        logger.warning(
            "LUFS gain capped: requested %+.2fdB, applying %+.2fdB",
            gain_delta_db, _MAX_NORMALIZATION_GAIN_DB,
        )
        gain_delta_db = _MAX_NORMALIZATION_GAIN_DB

    logger.info(
        "LUFS normalization: measured %.2f, target %.2f, applying %+.2fdB",
        measured_lufs, target_lufs, gain_delta_db,
    )
    return audio_segment + gain_delta_db


def _resolve_sound_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p


def normalize_sfx_to_peak(
    audio: AudioSegment, target_peak_dbfs: float = _SFX_PEAK_TARGET_DBFS,
) -> AudioSegment:
    """Normalize so the peak sits at `target_peak_dbfs`.
    Caps gain to [-24, +18] dB to avoid silly amplification on near-silent files."""
    if audio.dBFS == float("-inf"):
        return audio
    current_peak = audio.max_dBFS
    if current_peak == float("-inf"):
        return audio
    gain_needed = target_peak_dbfs - current_peak
    gain_needed = max(min(gain_needed, 18.0), -24.0)
    return audio + gain_needed


def normalize_sfx_loudness(
    audio: AudioSegment, target_dbfs: float = _SFX_LOUDNESS_TARGET_DBFS,
) -> AudioSegment:
    """Normalize an SFX by its RMS (average) loudness, not peak.

    Gives consistent PERCEIVED loudness across different sounds. Caps gain to
    avoid blowing up near-silent files, and pulls the peak back below -1 dBFS
    if normalization pushed it into clipping territory."""
    if audio.dBFS == float("-inf"):
        return audio
    gain_needed = target_dbfs - audio.dBFS
    gain_needed = max(min(gain_needed, 20.0), -25.0)
    adjusted = audio + gain_needed
    if adjusted.max_dBFS > -1.0:
        adjusted = adjusted + (-1.0 - adjusted.max_dBFS)
    return adjusted


def hard_limit(
    audio: AudioSegment, ceiling_dbfs: float = _TRUE_PEAK_CEILING_DBFS,
) -> AudioSegment:
    """Brick-wall limiter: pulls down overall gain so true peak ≤ ceiling.
    Crude (no look-ahead) but adequate for a -1 dBTP safety margin on the master."""
    peak = audio.max_dBFS
    if peak == float("-inf") or peak <= ceiling_dbfs:
        return audio
    return audio + (ceiling_dbfs - peak)


def _merge_overlapping_duck_events(
    events: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    if not events:
        return []
    events_sorted = sorted(events, key=lambda e: e[0])
    merged: list[tuple[float, float, float]] = [events_sorted[0]]
    for start, dur, db in events_sorted[1:]:
        prev_start, prev_dur, prev_db = merged[-1]
        prev_end = prev_start + prev_dur
        if start <= prev_end:
            new_end = max(prev_end, start + dur)
            new_db = min(prev_db, db)  # stronger duck wins
            merged[-1] = (prev_start, new_end - prev_start, new_db)
        else:
            merged.append((start, dur, db))
    return merged


def apply_ducking_envelope(
    music_segment: AudioSegment,
    duck_events: list[tuple[float, float, float]],
    fade_ms: int = _DUCK_FADE_MS,
) -> AudioSegment:
    """Per-region volume reduction on a music bed so transients punch through.
    Each event = (start_sec, duration_sec, gain_db_reduction). Short fades in/out
    of each duck region prevent zipper-noise clicks."""
    if not duck_events or len(music_segment) == 0:
        return music_segment

    events = _merge_overlapping_duck_events(duck_events)
    total_ms = len(music_segment)
    result = AudioSegment.empty()
    cursor_ms = 0

    for start_sec, duration_sec, duck_db in events:
        start_ms = int(start_sec * 1000)
        end_ms = start_ms + int(duration_sec * 1000)
        start_ms = max(cursor_ms, min(start_ms, total_ms))
        end_ms = max(start_ms, min(end_ms, total_ms))

        if start_ms > cursor_ms:
            result += music_segment[cursor_ms:start_ms]
        if end_ms > start_ms:
            ducked = music_segment[start_ms:end_ms] + duck_db
            fade = min(fade_ms, len(ducked) // 2)
            if fade > 0:
                ducked = ducked.fade_in(fade).fade_out(fade)
            result += ducked
        cursor_ms = end_ms

    if cursor_ms < total_ms:
        result += music_segment[cursor_ms:]
    return result


def _speech_duck_events(
    speech_regions: list[tuple[float, float]],
    duck_db: float,
) -> list[tuple[float, float, float]]:
    return [
        (float(s), max(0.0, float(e) - float(s)), duck_db)
        for s, e in speech_regions
    ]


def apply_speech_ducking_to_layer(
    layer: AudioSegment,
    speech_regions: list[tuple[float, float]],
    duck_db: float,
    fade_ms: int = SPEECH_PROTECTION_FADE_MS,
) -> AudioSegment:
    """Apply per-speech-region volume reduction to ``layer`` (in global time).

    The layer must be sized to the full mix length so global timestamps map
    correctly. Returns the layer unmodified when there are no regions."""
    if not speech_regions or len(layer) == 0:
        return layer
    return apply_ducking_envelope(
        layer, _speech_duck_events(speech_regions, duck_db), fade_ms=fade_ms,
    )


def _overlay_sfx(
    master: AudioSegment,
    seg: AudioSegment,
    match: dict[str, Any],
    preset: Preset,
    total_duration_ms: int,
    speech_regions: list[tuple[float, float]] | None = None,
    speech_coverage: float = 0.0,
    has_speech: bool = False,
    speech_stats: dict[str, Any] | None = None,
) -> tuple[AudioSegment, bool]:
    """Process & overlay a single SFX in the order described in Step 23.5:
    1. normalize_sfx_loudness  →  2. intensity gain  →  3. tier gain  →
    4. preset SFX gain         →  5. speech duck     →  6. opening cap  →
    7. SFX_MAX_DBFS_IN_MIX cap →  8. fade in/out     →  9. scene trim   →
    10. overlay."""
    abs_ts = float(match.get("absolute_timestamp") or 0.0)

    seg = normalize_sfx_loudness(seg)

    recipe_volume_db = match.get("recipe_layer_volume_db")
    if recipe_volume_db is None:
        intensity = str(match.get("intensity", "medium") or "medium").lower()
        seg = seg + _INTENSITY_GAIN_DB.get(intensity, _INTENSITY_GAIN_DB["medium"])
    else:
        # Recipe layers are pre-balanced; use the layer volume instead of
        # the generic intensity gain.
        seg = seg + float(recipe_volume_db)

    value_tier = str(match.get("value_tier", "accent") or "accent").lower()
    seg = seg + _TIER_GAIN_DB.get(value_tier, _TIER_GAIN_DEFAULT_DB)

    seg = seg + preset.sfx_volume_db

    if has_speech and speech_regions and is_in_speech_region(
        abs_ts, speech_regions, padding_sec=SPEECH_REGION_PADDING_SEC,
    ):
        seg = seg + get_sfx_speech_duck(speech_coverage)
        if speech_stats is not None:
            speech_stats["sfx_ducked_count"] = (
                int(speech_stats.get("sfx_ducked_count", 0)) + 1
            )

    if abs_ts < OPENING_WINDOW_SEC and speech_regions:
        speech_in_opening = any(
            float(s) < OPENING_EARLY_SPEECH_WINDOW_SEC
            for s, _e in speech_regions
        )
        if speech_in_opening:
            seg = seg + OPENING_SFX_OVER_SPEECH_DUCK_DB
            logger.info(
                "Opening SFX at %.2fs reduced %+.1fdB (early speech present)",
                abs_ts, OPENING_SFX_OVER_SPEECH_DUCK_DB,
            )

    if seg.dBFS != float("-inf") and seg.dBFS > SFX_MAX_DBFS_IN_MIX:
        seg = seg + (SFX_MAX_DBFS_IN_MIX - seg.dBFS)

    if len(seg) > _SFX_FADE_IN_MS:
        seg = seg.fade_in(_SFX_FADE_IN_MS)

    position_ms = max(0, int(abs_ts * 1000))
    if position_ms >= total_duration_ms:
        logger.warning(
            "SFX position %dms >= mix duration %dms; skipping (%s)",
            position_ms, total_duration_ms, match.get("sound_name", ""),
        )
        return master, False

    scene_start_ms = int(float(match.get("scene_start_sec", 0.0)) * 1000)
    scene_end_ms = int(float(match.get("scene_end_sec", 0.0)) * 1000)
    scene_duration_ms = max(0, scene_end_ms - scene_start_ms)
    time_in_scene_ms = max(0, position_ms - scene_start_ms)
    max_sfx_duration_ms = scene_duration_ms - time_in_scene_ms + _SFX_TAIL_OVERLAP_MS
    if max_sfx_duration_ms > 0 and len(seg) > max_sfx_duration_ms:
        seg = seg[:max_sfx_duration_ms]

    if len(seg) > _SFX_FADE_OUT_MS:
        seg = seg.fade_out(_SFX_FADE_OUT_MS)

    return master.overlay(seg, position=position_ms), True


def _overlay_ambient(
    master: AudioSegment,
    seg: AudioSegment,
    match: dict[str, Any],
    preset: Preset,
    total_duration_ms: int,
) -> tuple[AudioSegment, bool]:
    if len(seg) == 0:
        return master, False
    seg = seg + preset.ambient_volume_db
    scene_start_ms = max(0, int(float(match["scene_start_sec"]) * 1000))
    scene_end_ms = int(float(match["scene_end_sec"]) * 1000)
    scene_end_ms = min(scene_end_ms, total_duration_ms)
    scene_duration_ms = max(0, scene_end_ms - scene_start_ms)
    if scene_duration_ms <= 0:
        return master, False

    loops_needed = scene_duration_ms // len(seg) + 1
    looped = (seg * loops_needed)[:scene_duration_ms]

    fade_ms = min(_AMBIENT_FADE_MS, scene_duration_ms // 3)
    if fade_ms > 0:
        looped = looped.fade_in(fade_ms).fade_out(fade_ms)

    return master.overlay(looped, position=scene_start_ms), True


def _build_duck_events(match_plan: dict[str, Any]) -> list[tuple[float, float, float]]:
    return [
        (
            float(m["absolute_timestamp"]),
            _DUCK_DURATION_SEC,
            _DUCK_GAIN_DB,
        )
        for m in match_plan.get("matches", []) or []
        if m.get("layer") == "sfx" and m.get("intensity") == "sharp"
    ]


def _build_music_bed(
    match_plan: dict[str, Any],
    preset: Preset,
    total_duration_ms: int,
    speech_regions: list[tuple[float, float]] | None = None,
    has_significant_speech: bool = False,
) -> tuple[AudioSegment | None, float]:
    """Build the music bed at its effective dB level after applying the
    speech-aware hard ceiling. Also applies sharp-SFX ducking and speech
    ducking. Returns ``(music_bed_or_none, applied_volume_db)``."""
    music_meta = match_plan.get("music")
    if not music_meta:
        return None, 0.0
    music_path = _resolve_sound_path(music_meta.get("sound_path", ""))
    music = load_sfx_safely(music_path, max_duration_ms=_MAX_MUSIC_DURATION_MS)
    if music is None or len(music) == 0:
        logger.warning("Music track unavailable: %s", music_path)
        return None, 0.0

    effective_volume_db = float(preset.music_volume_db)
    ceiling = (
        MUSIC_MAX_DB_WITH_SPEECH if has_significant_speech
        else MUSIC_MAX_DB_WITHOUT_SPEECH
    )
    if effective_volume_db > ceiling:
        logger.info(
            "Capping music volume %+.1fdB \u2192 %+.1fdB (%s)",
            effective_volume_db, ceiling,
            "speech protection" if has_significant_speech
            else "no-speech ceiling",
        )
        effective_volume_db = ceiling

    music = music + effective_volume_db
    fade_in = min(_MUSIC_FADE_IN_MS, len(music) // 2)
    fade_out = min(_MUSIC_FADE_OUT_MS, len(music) // 2)
    music = music.fade_in(fade_in).fade_out(fade_out)

    if len(music) < total_duration_ms:
        loops_needed = (total_duration_ms // len(music)) + 1
        music = music * loops_needed
    music = music[:total_duration_ms]

    duck_events = _build_duck_events(match_plan)
    if duck_events:
        music = apply_ducking_envelope(music, duck_events)
        logger.info(
            "Applied %d sharp-SFX ducking events to music bed", len(duck_events),
        )

    if has_significant_speech and speech_regions:
        music = apply_speech_ducking_to_layer(
            music, speech_regions, duck_db=SPEECH_MUSIC_DUCK_DB,
        )
        logger.info(
            "Applied speech ducking to music (%d region(s) @ %+.1fdB)",
            len(speech_regions), SPEECH_MUSIC_DUCK_DB,
        )

    return music, effective_volume_db


def _resolve_speech_regions(
    match_plan: dict[str, Any],
    original_audio_path: Path,
) -> list[tuple[float, float]]:
    """Prefer pre-computed regions from the caller; re-detect only if absent.
    JSON round-trips turn tuples into lists, so normalize back to tuples."""
    raw = match_plan.get("speech_regions")
    if raw:
        return [(float(r[0]), float(r[1])) for r in raw if len(r) >= 2]
    try:
        return detect_speech_regions(original_audio_path)
    except Exception as exc:
        logger.warning("Speech detection failed (%s); proceeding without it.", exc)
        return []


def mix_audio(
    match_plan: dict[str, Any],
    original_audio_path: Path,
    output_path: Path,
    preset: Preset,
) -> Path:
    """Mix music bed + speech-protected ducked layers + SFX + ambient from the
    match plan, LUFS-normalize, and write a WAV to output_path."""
    if not original_audio_path.is_file():
        raise FileNotFoundError(f"Original audio not found: {original_audio_path}")

    original = AudioSegment.from_file(original_audio_path)
    total_duration_ms = len(original)
    total_duration_sec = total_duration_ms / 1000.0
    master = AudioSegment.silent(duration=total_duration_ms)

    speech_regions = _resolve_speech_regions(match_plan, original_audio_path)
    speech_cov = coverage_ratio(speech_regions, total_duration_sec)
    has_speech = speech_cov >= SPEECH_COVERAGE_THRESHOLD
    music_ceiling_db = (
        MUSIC_MAX_DB_WITH_SPEECH if has_speech else MUSIC_MAX_DB_WITHOUT_SPEECH
    )
    logger.info(
        "Speech: %d region(s), %.0f%% coverage \u2014 protection %s",
        len(speech_regions), speech_cov * 100,
        "ON" if has_speech else "off",
    )

    sfx_duck_db = get_sfx_speech_duck(speech_cov) if has_speech else 0.0
    speech_stats: dict[str, Any] = {
        "regions": len(speech_regions),
        "coverage_pct": round(speech_cov * 100, 1),
        "protection_enabled": has_speech,
        "music_ceiling_db": music_ceiling_db,
        "music_duck_db": SPEECH_MUSIC_DUCK_DB if has_speech else 0.0,
        "ambient_duck_db": SPEECH_AMBIENT_DUCK_DB if has_speech else 0.0,
        "sfx_duck_db": sfx_duck_db,
        "sfx_ducked_count": 0,
    }

    music_bed, music_db_used = _build_music_bed(
        match_plan, preset, total_duration_ms,
        speech_regions=speech_regions,
        has_significant_speech=has_speech,
    )
    if music_bed is not None:
        master = master.overlay(music_bed, position=0)
        logger.info(
            "Mixed music bed: %s (%.1fs @ %+.1fdB)",
            match_plan["music"].get("sound_name", ""),
            len(music_bed) / 1000.0,
            music_db_used,
        )

    ambient_layer = AudioSegment.silent(duration=total_duration_ms)
    sfx_count = 0
    ambient_count = 0
    skipped = 0

    matches = sorted(
        match_plan.get("matches", []) or [],
        key=lambda m: float(m.get("absolute_timestamp", 0.0)),
    )

    for match in matches:
        sound_path = _resolve_sound_path(match.get("sound_path", ""))
        seg = load_sfx_safely(sound_path)
        if seg is None:
            skipped += 1
            continue

        layer = match.get("layer")
        if layer == "ambient":
            ambient_layer, ok = _overlay_ambient(
                ambient_layer, seg, match, preset, total_duration_ms,
            )
            if ok:
                ambient_count += 1
            else:
                skipped += 1
        else:
            master, ok = _overlay_sfx(
                master, seg, match, preset, total_duration_ms,
                speech_regions=speech_regions,
                speech_coverage=speech_cov,
                has_speech=has_speech,
                speech_stats=speech_stats,
            )
            if ok:
                sfx_count += 1
            else:
                skipped += 1

    if has_speech and ambient_count > 0:
        ambient_layer = apply_speech_ducking_to_layer(
            ambient_layer, speech_regions, duck_db=SPEECH_AMBIENT_DUCK_DB,
        )
        logger.info(
            "Applied speech ducking to ambient layer (%d region(s) @ %+.1fdB)",
            len(speech_regions), SPEECH_AMBIENT_DUCK_DB,
        )
    if ambient_count > 0:
        master = master.overlay(ambient_layer, position=0)

    orig_db = float(preset.original_audio_volume_db)
    if has_speech and orig_db < ORIGINAL_MIN_DB_WITH_SPEECH:
        logger.info(
            "Clamping original audio %+.1fdB \u2192 %+.1fdB (speech protection)",
            orig_db, ORIGINAL_MIN_DB_WITH_SPEECH,
        )
        orig_db = ORIGINAL_MIN_DB_WITH_SPEECH
    original_adjusted = original + orig_db
    master = master.overlay(original_adjusted, position=0)

    master = normalize_loudness(master, target_lufs=_TARGET_LUFS)
    master = hard_limit(master, ceiling_dbfs=_TRUE_PEAK_CEILING_DBFS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    master.export(output_path, format="wav")
    logger.info(
        "Mixed %d SFX + %d ambient (skipped %d, %d SFX ducked) -> %s",
        sfx_count, ambient_count, skipped,
        speech_stats["sfx_ducked_count"], output_path,
    )

    match_plan["mix_stats"] = {
        "speech": speech_stats,
        "sfx_overlaid": sfx_count,
        "ambient_overlaid": ambient_count,
        "skipped": skipped,
        "music_volume_db": music_db_used if music_bed is not None else None,
        "original_volume_db": orig_db,
    }
    return output_path


def _main() -> None:
    from src.presets import PRESETS
    parser = argparse.ArgumentParser(
        description="Mix the match-plan SFX onto the original audio.",
    )
    parser.add_argument("video_path", type=Path, help="Path to input video file.")
    parser.add_argument(
        "--preset",
        default="tiktok_viral",
        choices=list(PRESETS.keys()),
        help="Mood preset.",
    )
    args = parser.parse_args()

    video_dir = TEMP_DIR / args.video_path.stem
    match_plan_path = video_dir / "match_plan.json"
    audio_path = video_dir / "audio.wav"

    if not match_plan_path.is_file():
        raise RuntimeError(
            f"Match plan not found at {match_plan_path}. "
            f"Run `python -m src.matching.sfx_matcher {args.video_path} "
            f"--preset {args.preset} --use-cached-analysis` first."
        )
    if not audio_path.is_file():
        raise RuntimeError(
            f"Original audio not found at {audio_path}. "
            f"Run `python -m src.preprocessing.audio_extractor {args.video_path}` first."
        )

    plan = json.loads(match_plan_path.read_text())
    preset = get_preset(args.preset)

    output_path = video_dir / "mixed_audio.wav"
    mix_audio(plan, audio_path, output_path, preset)

    sfx_n = sum(1 for m in plan.get("matches", []) if m.get("layer") == "sfx")
    ambient_n = sum(1 for m in plan.get("matches", []) if m.get("layer") == "ambient")
    duration_sec = float(plan.get("duration_sec", 0.0))

    print(f"Mixed {sfx_n} SFX layers + {ambient_n} ambient layers")
    print(f"Total duration: {duration_sec:.2f}s")
    print(f"Output: {output_path}")
    print(f"\nPlay it with: open {output_path}")


if __name__ == "__main__":
    _main()
