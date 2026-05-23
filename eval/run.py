"""Run the evaluation suite over the test video set."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from src.analysis.gemini_client import analyze_video
from src.config import TEMP_DIR
from src.library.vector_store import get_qdrant_client
from src.matching.sfx_matcher import match_analysis
from src.utils.logger import get_logger

from eval.ai_judge import judge_video
from eval.metrics import compute_metrics
from eval.test_set import EVAL_VIDEOS_DIR, available_eval_videos

logger = get_logger("eval")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_eval(use_ai_judge: bool = False) -> dict:
    """Process every eval video, compute metrics, optionally AI-judge.

    Returns a results dict and saves it to eval/results/."""
    videos = available_eval_videos()
    if not videos:
        logger.error(
            "No eval videos found in %s. Add videos and register them in eval/test_set.py.",
            EVAL_VIDEOS_DIR,
        )
        return {}

    logger.info(
        "Running eval on %d video(s) (ai_judge=%s)", len(videos), use_ai_judge,
    )

    qdrant = get_qdrant_client()
    results: list[dict] = []

    for ev in videos:
        video_path = EVAL_VIDEOS_DIR / ev.filename
        logger.info("=== Evaluating: %s ===", ev.label)
        t0 = time.time()

        try:
            analysis = analyze_video(
                video_path, preset=ev.recommended_preset, dry_run=False,
            )
            match_plan = match_analysis(
                analysis, qdrant, preset_name=ev.recommended_preset,
            )
            strategy = analysis.get("strategy") or {}
            duration = float(analysis.get("duration_sec") or 0.0)

            metrics = compute_metrics(match_plan, strategy, duration)

            sanity = {
                "sfx_in_expected_range": (
                    ev.min_expected_sfx
                    <= metrics.total_sfx
                    <= ev.max_expected_sfx
                ),
                "expected_min": ev.min_expected_sfx,
                "expected_max": ev.max_expected_sfx,
                "actual_sfx": metrics.total_sfx,
            }

            entry: dict = {
                "video": ev.filename,
                "label": ev.label,
                "preset": ev.recommended_preset,
                "elapsed_sec": round(time.time() - t0, 1),
                "metrics": metrics.to_dict(),
                "sanity": sanity,
                "detected_style": strategy.get("style"),
                "detected_vibe": strategy.get("vibe"),
            }

            if use_ai_judge:
                final_video = _render_for_judge(
                    video_path, match_plan, analysis, ev.recommended_preset,
                )
                if final_video:
                    judgment = judge_video(final_video)
                    if judgment:
                        entry["ai_judgment"] = judgment

            results.append(entry)
            logger.info("  objective_score: %.1f/100", metrics.objective_score)

        except Exception as exc:
            logger.error("  FAILED: %s", exc)
            results.append({
                "video": ev.filename,
                "label": ev.label,
                "error": str(exc),
            })

    valid = [r for r in results if "metrics" in r]
    avg_objective = (
        round(sum(r["metrics"]["objective_score"] for r in valid) / len(valid), 1)
        if valid else 0.0
    )
    ai_scores = [
        r["ai_judgment"]["overall"]
        for r in results
        if r.get("ai_judgment")
        and "overall" in r.get("ai_judgment", {})
    ]
    avg_ai = round(sum(ai_scores) / len(ai_scores), 1) if ai_scores else None

    summary = {
        "timestamp": datetime.now().isoformat(),
        "videos_evaluated": len(valid),
        "videos_failed": len(results) - len(valid),
        "avg_objective_score": avg_objective,
        "avg_ai_overall": avg_ai,
        "results": results,
    }

    out_path = RESULTS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(summary, indent=2))

    _print_report(summary, out_path)
    return summary


def _render_for_judge(
    video_path: Path,
    match_plan: dict,
    analysis: dict,
    preset: str,
) -> Path | None:
    """Render a video so the AI judge can evaluate it. Returns path or None."""
    try:
        from src.export.renderer import render_final_video
        from src.matching.timing_adjuster import snap_to_onsets
        from src.mixing.mixer import mix_audio
        from src.preprocessing.audio_extractor import extract_audio
        from src.preprocessing.onset_detector import detect_onsets_from_video
        from src.presets import get_preset

        onsets = detect_onsets_from_video(video_path)
        match_plan = snap_to_onsets(match_plan, onsets)
        work_dir = TEMP_DIR / video_path.stem
        work_dir.mkdir(parents=True, exist_ok=True)
        original_audio = work_dir / "audio.wav"
        if not original_audio.exists():
            extract_audio(video_path, original_audio)
        mixed = work_dir / "eval_mixed.wav"
        mix_audio(match_plan, original_audio, mixed, get_preset(preset))
        final = work_dir / "eval_final.mp4"
        render_final_video(video_path, mixed, final)
        return final
    except Exception as exc:
        logger.warning("Render for judge failed: %s", exc)
        return None


def _print_report(summary: dict, out_path: Path) -> None:
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(
        f"Videos: {summary['videos_evaluated']} ok, "
        f"{summary['videos_failed']} failed"
    )
    print(f"Avg objective score: {summary['avg_objective_score']}/100")
    if summary.get("avg_ai_overall") is not None:
        print(f"Avg AI judge score:  {summary['avg_ai_overall']}/10")
    print("-" * 60)
    for r in summary["results"]:
        if "metrics" in r:
            m = r["metrics"]
            mark = "\u2713" if r["sanity"]["sfx_in_expected_range"] else "\u2717"
            line = (
                f"  {mark} {r['label'][:35]:35} "
                f"score={m['objective_score']:5.1f}  "
                f"sfx={m['total_sfx']:2}  "
                f"coverage={m['anchor_coverage_pct']:.0f}%"
            )
            if r.get("ai_judgment"):
                line += f"  ai={r['ai_judgment'].get('overall','?')}/10"
            print(line)
        else:
            print(
                f"  \u2717 {r['label'][:35]:35} FAILED: "
                f"{(r.get('error') or '')[:40]}"
            )
    print("=" * 60)
    print(f"Full results: {out_path}\n")


def compare_runs(path_a: Path, path_b: Path) -> None:
    """Compare two eval result files and show the delta."""
    a = json.loads(path_a.read_text())
    b = json.loads(path_b.read_text())

    print(f"\nComparing:\n  A: {path_a.name} (avg {a['avg_objective_score']})")
    print(f"  B: {path_b.name} (avg {b['avg_objective_score']})")
    delta = float(b["avg_objective_score"]) - float(a["avg_objective_score"])
    arrow = "\u2191" if delta > 0 else "\u2193" if delta < 0 else "="
    print(f"\n  Overall: {arrow} {delta:+.1f} points")

    by_video_a = {r.get("video"): r for r in a.get("results", [])}
    by_video_b = {r.get("video"): r for r in b.get("results", [])}
    common = sorted(set(by_video_a) & set(by_video_b))
    if common:
        print("\n  Per-video delta:")
        for v in common:
            ra, rb = by_video_a[v], by_video_b[v]
            if "metrics" not in ra or "metrics" not in rb:
                continue
            d = rb["metrics"]["objective_score"] - ra["metrics"]["objective_score"]
            mark = "\u2191" if d > 0 else "\u2193" if d < 0 else "="
            print(
                f"    {mark} {ra['label'][:30]:30} "
                f"{ra['metrics']['objective_score']:5.1f} \u2192 "
                f"{rb['metrics']['objective_score']:5.1f} "
                f"({d:+.1f})"
            )
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the AI SFX evaluation suite.",
    )
    parser.add_argument(
        "--ai-judge", action="store_true",
        help="Also run the Gemini AI judge (renders + uses tokens).",
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("RUN_A", "RUN_B"),
        help="Compare two result JSON files instead of running eval.",
    )
    args = parser.parse_args()

    if args.compare:
        compare_runs(Path(args.compare[0]), Path(args.compare[1]))
    else:
        run_eval(use_ai_judge=args.ai_judge)
