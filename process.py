from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import librosa

from src.analysis.beat_detector import detect_beats, find_strong_beats
from src.analysis.director import (
    _minimal_default_strategy as _default_strategy,
    run_director_pass,
    strategy_summary_lines,
)
from src.analysis.gemini_client import (
    analyze_scenes_parallel,
    prepare_for_analysis,
)
from src.config import LIBRARY_DIR, MAX_VIDEO_LENGTH_SEC, TEMP_DIR, ensure_dirs
from src.export.project_packager import package_project
from src.export.renderer import render_final_video
from src.export.timeline_exporter import build_timeline, export_all_formats
from src.library.vector_store import get_qdrant_client
from src.matching.sfx_matcher import match_analysis
from src.matching.timing_adjuster import snap_to_beats, snap_to_onsets
from src.mixing.mixer import mix_audio
from src.preprocessing.audio_extractor import extract_audio
from src.preprocessing.onset_detector import detect_onsets_from_video
from src.preprocessing.speech_detector import detect_speech_regions
from src.presets import PRESETS, get_preset
from src.utils.ffmpeg_check import check_ffmpeg
from src.utils.logger import get_logger
from src.utils.output_manager import (
    get_next_version_path,
    update_latest_link,
    update_outputs_index,
    write_version_metadata,
)

logger = get_logger("process")

_BAR = "\u2501" * 47
_TOTAL_STEPS = 10


def _print_header(video_path: Path, preset_name: str) -> None:
    preset = get_preset(preset_name)
    print(_BAR)
    print("AI Sound Effects Generator")
    print(_BAR)
    print(f"Video:  {video_path}")
    print(f"Preset: {preset.name}")
    print(_BAR)
    print()


def _step_start(step: int, msg: str) -> float:
    print(f"[{step}/{_TOTAL_STEPS}] {msg}", flush=True)
    return time.time()


def _step_done(t0: float, summary: str = "") -> None:
    elapsed = time.time() - t0
    suffix = f" \u2014 {summary}" if summary else ""
    print(f"     \u2713 Done in {elapsed:.1f}s{suffix}")
    print()


def _validate_video(video_path: Path) -> float:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if video_path.stat().st_size == 0:
        raise ValueError(f"Video file is empty: {video_path}")
    duration = float(librosa.get_duration(path=str(video_path)))
    if duration > MAX_VIDEO_LENGTH_SEC:
        raise ValueError(
            f"Video too long: {duration:.1f}s > {MAX_VIDEO_LENGTH_SEC}s. "
            f"Phase 0 prototype only supports clips up to {MAX_VIDEO_LENGTH_SEC}s."
        )
    return duration


def _preflight_checks() -> None:
    manifest_path = LIBRARY_DIR / "manifest.json"
    if not manifest_path.is_file():
        print(
            "Library not built. Run:\n"
            "  python -m src.library.downloader && python -m src.library.embedder"
        )
        sys.exit(1)

    try:
        get_qdrant_client()
    except RuntimeError as exc:
        print(str(exc))
        sys.exit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Sound Effects Generator — end-to-end pipeline.",
    )
    parser.add_argument("video_path", type=Path, help="Path to input MP4 file.")
    parser.add_argument(
        "--preset",
        default="tiktok_viral",
        choices=list(PRESETS.keys()),
        help="Mood preset. Default is tiktok_viral — most short-form ads want UGC.",
    )
    parser.add_argument(
        "--use-cached-analysis",
        action="store_true",
        help="Skip Gemini if a cached analysis.json exists for this video.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Explicit output path. When omitted, the pipeline writes a versioned "
            "file under data/temp/{video_name}/outputs/ "
            "(e.g. v007_tiktok_viral.mp4) and updates latest.mp4 + index.json."
        ),
    )
    parser.add_argument(
        "--verbose-selectivity",
        action="store_true",
        help="Print value scores and reasons for every SFX (kept and pruned).",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=None,
        help="Override sound density 0.0-1.0 (default: the preset's density).",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the Gemini match-verification pass (keeps all matched sounds).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    ensure_dirs()
    check_ffmpeg()
    _validate_video(args.video_path)
    _preflight_checks()

    work_dir = TEMP_DIR / args.video_path.stem
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.output is not None:
        output_path = args.output
        version: int | None = None
    else:
        output_path, version = get_next_version_path(work_dir, args.preset)

    _print_header(args.video_path, args.preset)

    pipeline_t0 = time.time()
    current_step = "init"

    try:
        analysis_path = work_dir / "analysis.json"
        strategy_path = work_dir / "strategy.json"
        prep: dict | None = None

        if args.use_cached_analysis and analysis_path.is_file():
            current_step = "[1/7] Loading cached director strategy"
            t0 = _step_start(1, "Loading cached director strategy...")
            analysis = json.loads(analysis_path.read_text())
            strategy = analysis.get("strategy")
            if not strategy:
                strategy = _default_strategy(
                    args.preset,
                    float(analysis.get("duration_sec") or 0.0),
                    [],
                )
                analysis["strategy"] = strategy
                print(
                    "     (cached analysis has no strategy \u2014 using minimal default)"
                )
            _step_done(
                t0,
                f"style: {strategy.get('style','?')} | vibe: {strategy.get('vibe','?')}",
            )
            for line in strategy_summary_lines(strategy):
                print(f"     {line}")
            print()

            current_step = "[2/7] Loading cached per-scene analysis"
            t0 = _step_start(2, "Loading cached per-scene analysis...")
            n_scenes = len(analysis.get("scenes", []))
            n_actions = sum(
                len(s.get("actions") or []) for s in analysis.get("scenes", [])
            )
            _step_done(t0, f"{n_scenes} scenes, {n_actions} actions")
        else:
            current_step = "[1/7] Running Director pass"
            t0 = _step_start(1, "Running Director pass...")
            prep = prepare_for_analysis(args.video_path)
            strategy = run_director_pass(
                video_path=args.video_path,
                frame_paths=prep["frames"],
                scenes=prep["scenes"],
                onsets=prep["onsets"],
                user_preset=args.preset,
                duration_sec=prep["duration_sec"],
                dry_run=False,
                speech_regions=prep["speech_regions"],
            )
            strategy_path.write_text(json.dumps(strategy, indent=2))
            _step_done(
                t0,
                f"style: {strategy.get('style','?')} | vibe: {strategy.get('vibe','?')}",
            )
            for line in strategy_summary_lines(strategy):
                print(f"     {line}")
            print()

            current_step = "[2/7] Per-scene analysis (with director context)"
            t0 = _step_start(2, "Per-scene analysis (with director context)...")
            scene_results = analyze_scenes_parallel(
                prep["scenes"], prep["frames"], prep["onsets"],
                args.preset, strategy=strategy, dry_run=False,
            )
            analysis = {
                "video_path": str(args.video_path),
                "preset": args.preset,
                "duration_sec": prep["duration_sec"],
                "strategy": strategy,
                "scenes": scene_results,
            }
            analysis_path.write_text(json.dumps(analysis, indent=2))
            n_actions = sum(len(s.get("actions") or []) for s in scene_results)
            _step_done(t0, f"{len(scene_results)} scenes, {n_actions} actions detected")

        current_step = "[3/7] Detecting audio onsets"
        t0 = _step_start(3, "Detecting audio onsets...")
        onsets_path = work_dir / "onsets.json"
        if prep is not None and prep.get("onsets"):
            onsets = list(prep["onsets"])
            onsets_path.write_text(json.dumps(onsets))
        elif onsets_path.is_file():
            onsets = json.loads(onsets_path.read_text())
        else:
            onsets = detect_onsets_from_video(args.video_path)
            onsets_path.write_text(json.dumps(onsets))
        _step_done(t0, f"{len(onsets)} onsets found")

        current_step = "[4/7] Matching SFX from library"
        t0 = _step_start(4, "Matching SFX from library...")
        client = get_qdrant_client()
        match_plan = match_analysis(
            analysis, client, preset_name=args.preset, density=args.density,
            verify=not args.no_verify,
        )
        sfx_count = sum(1 for m in match_plan["matches"] if m["layer"] == "sfx")
        ambient_count = sum(
            1 for m in match_plan["matches"] if m["layer"] == "ambient"
        )
        unmatched_sfx = sum(
            1 for u in match_plan["unmatched_actions"] if u["action_type"] != "ambient"
        )
        total_sfx_actions = sfx_count + unmatched_sfx
        match_plan_path = work_dir / "match_plan.json"
        match_plan_path.write_text(json.dumps(match_plan, indent=2))
        _step_done(
            t0,
            f"{sfx_count}/{total_sfx_actions} actions matched, "
            f"{ambient_count} ambient layers",
        )

        recipes_stat = (match_plan.get("match_stats") or {}).get("recipes") or {}
        recipes_applied = int(recipes_stat.get("applied") or 0)
        if recipes_applied:
            print(
                f"     Recipes: {recipes_applied} anchor moment(s) → "
                f"{int(recipes_stat.get('layer_total') or 0)} layer(s)"
            )
            for r in recipes_stat.get("details") or []:
                print(
                    f"       \u00b7 {r.get('action_type', '?')} @ "
                    f"{float(r.get('anchor_timestamp') or 0):.1f}s — "
                    f"{r.get('recipe_name', '?')} "
                    f"({int(r.get('layer_count') or 0)} layers)"
                )

        vstats = (match_plan.get("match_stats") or {}).get("verification") or {}
        if vstats and vstats.get("checked"):
            print(
                f"     Verification: checked {vstats.get('checked', 0)}, "
                f"dropped {vstats.get('dropped', 0)} wrong"
                + (
                    f", re-added {vstats.get('re_added', 0)}"
                    if vstats.get("re_added") else ""
                )
            )
            for d in match_plan.get("verification_dropped") or []:
                print(
                    f"       \u2717 {str(d.get('sound_name', '?'))} @ "
                    f"{float(d.get('timestamp') or 0):.1f}s "
                    f"({d.get('action_type', '?')}) — {d.get('reason', '')}"
                )

        sel = match_plan.get("selectivity_stats") or {}
        if sel:
            print(
                f"     Selectivity: {sel.get('input_sfx', 0)} SFX → "
                f"{sel.get('kept_sfx', 0)} kept "
                f"({sel.get('anchors', 0)} anchors, {sel.get('accents', 0)} accents, "
                f"{sel.get('recipe_layers', 0)} recipe layers in "
                f"{sel.get('recipe_groups', 0)} group(s)), "
                f"{sel.get('pruned_sfx', 0)} pruned"
            )

        if args.verbose_selectivity:
            print()
            print("     SELECTIVITY DETAIL")
            print("     " + "\u2500" * 60)
            for m in match_plan["matches"]:
                if m.get("layer") != "sfx":
                    continue
                tier = m.get("value_tier", "?")
                vs = m.get("value_score", 0.0)
                reasons = ", ".join(m.get("value_reasons") or [])
                print(
                    f"     KEPT ({tier:<6}) {float(m.get('absolute_timestamp') or 0):>5.1f}s  "
                    f"{str(m.get('action_type', '')):<22} value={vs:.2f}  [{reasons}]"
                )
            for p in match_plan.get("pruned_actions") or []:
                ts = float(p.get("timestamp") or 0.0)
                print(
                    f"     PRUNED         {ts:>5.1f}s  "
                    f"{str(p.get('action_type', '')):<22} value={float(p.get('value_score') or 0):.2f}  "
                    f"[{p.get('reason', '')}]"
                )
            print()

        director_stats = match_plan.get("match_stats", {}).get("director") or {}
        if director_stats.get("silent_dropped") or director_stats.get("budget_dropped"):
            print(
                f"     Director: dropped {director_stats.get('silent_dropped', 0)} "
                f"(silent regions) + {director_stats.get('budget_dropped', 0)} "
                f"(budget cap, max {director_stats.get('max_sfx_count', 0)})"
            )

        routing_breakdown = (
            match_plan.get("match_stats", {}).get("routing_breakdown") or {}
        )
        routing_by_tier = (
            match_plan.get("match_stats", {}).get("routing_by_tier") or {}
        )
        if routing_breakdown:
            print("     Routing breakdown (SFX only):")
            label_map = {
                "primary": "Primary tier",
                "action_override": "Action override",
                "fallback": "Fallback tier",
                "global": "Global pool",
            }
            for path, count in sorted(
                routing_breakdown.items(), key=lambda x: -x[1],
            ):
                label = label_map.get(path, path)
                tiers_for_path = sorted({
                    str(m.get("matched_tier"))
                    for m in match_plan["matches"]
                    if m.get("layer") == "sfx" and m.get("routing_path") == path
                })
                tier_hint = (
                    " (" + ", ".join(tiers_for_path) + ")"
                    if tiers_for_path and path != "global"
                    else ""
                )
                print(f"       {label}{tier_hint}: {count}")
            print()
        _ = routing_by_tier

        current_step = "[5/10] Detecting music beats"
        t0 = _step_start(5, "Detecting music beats...")
        music_meta = match_plan.get("music") or {}
        beats: list[float] = []
        strong_beats: list[float] = []
        tempo_bpm = 0.0
        if music_meta.get("sound_path"):
            music_path = Path(music_meta["sound_path"])
            duration_limit = float(analysis.get("duration_sec") or 0.0) or None
            tempo_bpm, beats = detect_beats(
                music_path, duration_limit_sec=duration_limit,
            )
            strong_beats = find_strong_beats(beats, every_n=4)
            match_plan["music_beats"] = {
                "tempo_bpm": tempo_bpm,
                "beat_count": len(beats),
                "first_10_beats_sec": beats[:10],
            }
            _step_done(
                t0,
                f"tempo {tempo_bpm:.1f} BPM, {len(beats)} beats "
                f"({len(strong_beats)} strong)",
            )
        else:
            match_plan["music_beats"] = {
                "tempo_bpm": 0.0, "beat_count": 0, "first_10_beats_sec": [],
            }
            _step_done(t0, "no music selected \u2014 skipping beat sync")

        current_step = "[6/10] Snapping SFX (onsets + beats)"
        t0 = _step_start(6, "Snapping SFX (onsets + beats)...")
        match_plan = snap_to_onsets(match_plan, onsets)
        strategy_for_snap = analysis.get("strategy") if isinstance(analysis, dict) else None
        match_plan = snap_to_beats(
            match_plan, beats, strong_beats, strategy=strategy_for_snap,
        )
        match_plan_path.write_text(json.dumps(match_plan, indent=2))
        snap_stats = match_plan.get("snap_stats", {})
        beat_stats = match_plan.get("beat_snap_stats", {})
        total_sfx = snap_stats.get("total_sfx", 0)
        onset_snapped = snap_stats.get("snapped", 0)
        beat_snapped = beat_stats.get("total_snapped", 0)
        unsnapped = max(0, total_sfx - onset_snapped - beat_snapped)
        _step_done(
            t0,
            f"{onset_snapped} onset, {beat_snapped} beat "
            f"({beat_stats.get('to_strong', 0)} strong), "
            f"{unsnapped} unsnapped",
        )

        current_step = "[7/10] Mixing audio layers"
        t0 = _step_start(7, "Mixing audio layers...")
        original_audio_path = work_dir / "audio.wav"
        if not original_audio_path.is_file():
            original_audio_path = extract_audio(args.video_path)
        speech_regions = (analysis.get("strategy") or {}).get("speech_regions") or []
        if not speech_regions:
            try:
                speech_regions = [
                    [float(s), float(e)]
                    for s, e in detect_speech_regions(original_audio_path)
                ]
            except Exception as exc:
                logger.warning("Speech detection failed: %s", exc)
                speech_regions = []
        match_plan["speech_regions"] = speech_regions
        mixed_audio_path = work_dir / "mixed_audio.wav"
        preset = get_preset(args.preset)
        mix_audio(match_plan, original_audio_path, mixed_audio_path, preset)
        mix_stats = match_plan.get("mix_stats") or {}
        sp_stats = mix_stats.get("speech") or {}
        if sp_stats.get("protection_enabled"):
            _step_done(
                t0,
                f"speech protection ON \u2014 ducked {sp_stats.get('sfx_ducked_count', 0)} "
                f"SFX, music cap {sp_stats.get('music_ceiling_db', 0):.0f}dB",
            )
            print(
                f"     Speech: {sp_stats.get('regions', 0)} region(s), "
                f"{sp_stats.get('coverage_pct', 0):.0f}% coverage"
            )
            print(
                f"     \u2192 Music ducked {sp_stats.get('music_duck_db', 0):.0f}dB, "
                f"ambient {sp_stats.get('ambient_duck_db', 0):.0f}dB, "
                f"SFX {sp_stats.get('sfx_duck_db', 0):.0f}dB during speech"
            )
            print()
        else:
            _step_done(t0, "final loudness -14.0 LUFS")
            if sp_stats:
                print(
                    f"     Speech: {sp_stats.get('regions', 0)} region(s), "
                    f"{sp_stats.get('coverage_pct', 0):.0f}% coverage \u2014 protection off"
                )
                print()

        current_step = "[8/10] Rendering final video"
        render_label = (
            f"Rendering version v{version:03d}_{args.preset}..."
            if version is not None
            else "Rendering final video..."
        )
        t0 = _step_start(8, render_label)
        render_final_video(args.video_path, mixed_audio_path, output_path)
        _step_done(t0)

        current_step = "[9/10] Building editable timeline"
        t0 = _step_start(9, "Building editable timeline...")
        timeline_base = (
            f"v{version:03d}_{args.preset}"
            if version is not None
            else output_path.stem
        )
        timeline_dir = work_dir / "timeline_exports"
        timeline_files: dict[str, Path] = {}
        try:
            timeline = build_timeline(
                video_path=args.video_path,
                original_audio_path=original_audio_path,
                match_plan=match_plan,
                video_duration_sec=float(analysis.get("duration_sec") or 0.0),
                project_name=timeline_base,
            )
            timeline_files = export_all_formats(
                timeline, timeline_dir, base_name=timeline_base,
            )
        except Exception as exc:
            logger.warning("Timeline export failed: %s", exc)
        _step_done(
            t0,
            f"{len(timeline_files)} format(s): "
            f"{', '.join(timeline_files.keys()) or 'none'}",
        )

        current_step = "[10/10] Packaging project for editor import"
        t0 = _step_start(10, "Packaging project for editor import...")
        zip_path: Path | None = None
        if version is not None and timeline_files:
            try:
                zip_path = package_project(
                    work_dir=work_dir,
                    output_path=output_path,
                    version=version,
                    preset_name=args.preset,
                    match_plan=match_plan,
                    timeline_files=timeline_files,
                    original_video_path=args.video_path,
                    original_audio_path=original_audio_path,
                )
            except Exception as exc:
                logger.warning("Project packaging failed: %s", exc)
        if zip_path is not None:
            size_mb = zip_path.stat().st_size / (1024 * 1024)
            _step_done(t0, f"{zip_path.name} ({size_mb:.1f} MB)")
        elif version is None:
            _step_done(t0, "skipped (explicit --output, no versioned package)")
        else:
            _step_done(t0, "skipped (no timeline available)")

        total_elapsed = time.time() - pipeline_t0

        latest_path: Path | None = None
        index_path: Path | None = None
        if version is not None:
            pipeline_stats = {
                "total_seconds": round(total_elapsed, 2),
                "scene_count": len(analysis.get("scenes", [])),
                "snap_stats": match_plan.get("snap_stats", {}),
                "match_stats": match_plan.get("match_stats", {}),
            }
            write_version_metadata(
                output_path, version, args.preset, match_plan, pipeline_stats,
                project_zip=zip_path,
                timeline_formats=list(timeline_files.keys()),
            )
            index_path = update_outputs_index(work_dir)
            latest_path = update_latest_link(output_path)

        print(_BAR)
        print(f"Total time: {total_elapsed:.1f}s")
        print()
        music = match_plan.get("music")
        if music:
            print(
                f"Music:  {music['sound_name']} "
                f"({music['sound_duration']:.0f}s, mood: {music['selected_mood']})"
            )
        else:
            print("Music:  none (skipped)")

        ms = match_plan.get("match_stats", {})
        if ms:
            print(
                f"Filters: whitelist fall-back {ms.get('whitelist_fallback', 0)}, "
                f"duration fall-back {ms.get('duration_fallback', 0)}, "
                f"min-score 0.4 reject {ms.get('min_score_rejected', 0)}"
            )

        print()
        print(f"Final video:    {output_path}")
        if zip_path is not None:
            print(f"Editor project: {zip_path}")
        if latest_path is not None:
            print(f"Latest:         {latest_path}")
        if index_path is not None:
            print(f"Index:          {index_path}")
        print()
        print(f"Play video:     open \"{output_path}\"")
        if latest_path is not None:
            print(f"Play latest:    open \"{latest_path}\"")
        if zip_path is not None:
            print(f"Open project:   open \"{zip_path}\"")
            print()
            print(
                f"Import {zip_path.name} into Premiere/Resolve/Final Cut to edit "
                "individual sounds."
            )
        print(_BAR)
        return 0

    except Exception as exc:
        print()
        print(f"[FAILED] {current_step}")
        print(f"Error: {exc}")
        print()
        traceback.print_exc()
        if work_dir.exists():
            files = sorted(p for p in work_dir.glob("*") if p.is_file())
            if files:
                print("\nIntermediate files created:")
                for f in files:
                    print(f"  - {f}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
