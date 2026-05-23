"""Job store and background processing (thread-based MVP)."""
from __future__ import annotations

import shutil
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.analysis.beat_detector import detect_beats, find_strong_beats
from src.analysis.gemini_client import analyze_video
from src.config import LIBRARY_DIR, PROJECT_ROOT, STORAGE_MODE, TEMP_DIR
from src.export.project_packager import package_project
from src.export.renderer import render_final_video
from src.export.timeline_exporter import build_timeline, export_all_formats
from src.library.vector_store import get_qdrant_client
from src.matching.sfx_matcher import match_analysis
from src.matching.timing_adjuster import snap_to_beats, snap_to_onsets
from src.mixing.mixer import mix_audio
from src.preprocessing.audio_extractor import extract_audio
from src.preprocessing.onset_detector import detect_onsets_from_video
from src.presets import get_preset
from src.utils.logger import get_logger

from api.models import JobDiagnostics, JobInfo, JobStatus, JobStep, SoundDiagnostic
from api.storage import store_result

logger = get_logger("api_jobs")

_JOBS: dict[str, JobInfo] = {}
_JOBS_LOCK = threading.Lock()


def _resolve_media_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p


def create_job(preset: str) -> JobInfo:
    job_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc)
    job = JobInfo(
        job_id=job_id,
        status=JobStatus.pending,
        preset=preset,
        created_at=now,
        updated_at=now,
        progress_pct=0,
        steps=[],
    )
    with _JOBS_LOCK:
        _JOBS[job_id] = job
    return job


def get_job(job_id: str) -> Optional[JobInfo]:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def list_jobs(limit: int = 50) -> list[JobInfo]:
    with _JOBS_LOCK:
        all_jobs = sorted(_JOBS.values(), key=lambda j: j.created_at, reverse=True)
        return all_jobs[:limit]


def _update_job(job_id: str, **changes: Any) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
        for k, v in changes.items():
            setattr(job, k, v)
        job.updated_at = datetime.now(timezone.utc)


def _add_step(job_id: str, name: str, detail: Optional[str] = None) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
        if job.steps and job.steps[-1].completed_at is None:
            job.steps[-1].completed_at = datetime.now(timezone.utc)
        job.steps.append(
            JobStep(name=name, started_at=datetime.now(timezone.utc), detail=detail),
        )
        job.current_step = name
        job.updated_at = datetime.now(timezone.utc)


def _close_steps(job_id: str) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
        if job.steps and job.steps[-1].completed_at is None:
            job.steps[-1].completed_at = datetime.now(timezone.utc)


def _collect_analysis_diag(diag: JobDiagnostics, analysis: dict[str, Any]) -> None:
    try:
        strategy = analysis.get("strategy") or {}
        diag.director_style = strategy.get("style")
        diag.director_vibe = strategy.get("vibe")
        diag.director_recommended_preset = strategy.get("recommended_preset")
        constraints = strategy.get("global_constraints") or {}
        diag.director_max_sfx = constraints.get("max_sfx_count")
        diag.anchor_moments = len(strategy.get("anchor_moments") or [])
        scenes = analysis.get("scenes") or []
        diag.scenes_detected = len(scenes)
        diag.actions_suggested = sum(len(s.get("actions") or []) for s in scenes)
        speech_regions = strategy.get("speech_regions") or []
        diag.speech_regions = len(speech_regions)
    except Exception as exc:
        diag.warnings.append(f"diag analysis failed: {exc}")


def _collect_match_plan_diag(diag: JobDiagnostics, match_plan: dict[str, Any]) -> None:
    try:
        matches = match_plan.get("matches") or []
        diag.sfx_matched = sum(1 for m in matches if m.get("layer") == "sfx")
        selectivity = match_plan.get("selectivity_stats") or {}
        diag.sfx_after_selectivity = int(
            selectivity.get("kept_sfx", diag.sfx_matched),
        )
        music = match_plan.get("music") or {}
        if music:
            diag.music_found = True
            diag.music_name = music.get("sound_name")
    except Exception as exc:
        diag.warnings.append(f"diag match-plan failed: {exc}")


def _collect_mix_diag(diag: JobDiagnostics, match_plan: dict[str, Any]) -> None:
    try:
        fetch_stats = match_plan.get("fetch_stats") or {}
        diag.r2_fetch_attempts = int(fetch_stats.get("attempts", 0) or 0)
        diag.r2_fetch_failures = int(fetch_stats.get("failures", 0) or 0)
        diag.music_fetch_status = fetch_stats.get("music_status")

        mix_stats = match_plan.get("mix_stats") or {}
        speech = mix_stats.get("speech") or {}
        diag.speech_coverage_pct = float(speech.get("coverage_pct", 0.0) or 0.0)
        if not diag.speech_regions:
            diag.speech_regions = int(speech.get("regions", 0) or 0)

        diag.sounds = []
        for m in match_plan.get("matches") or []:
            try:
                diag.sounds.append(SoundDiagnostic(
                    timestamp=float(m.get("absolute_timestamp") or 0.0),
                    action_type=str(m.get("action_type") or "?"),
                    sound_name=str(m.get("sound_name") or "?"),
                    sound_id=str(m.get("sound_id")) if m.get("sound_id") is not None else None,
                    matched_tier=m.get("matched_tier"),
                    match_score=(
                        float(m["reranked_score"]) if m.get("reranked_score") is not None
                        else (float(m["match_score"]) if m.get("match_score") is not None else None)
                    ),
                    value_tier=m.get("value_tier"),
                    layer=str(m.get("layer") or "sfx"),
                    fetch_status=str(m.get("fetch_status") or "unknown"),
                    included_in_mix=bool(m.get("included_in_mix", True)),
                    note=m.get("fetch_note"),
                ))
            except Exception as exc:
                diag.warnings.append(f"diag sound entry failed: {exc}")

        music = match_plan.get("music") or {}
        if music:
            try:
                diag.sounds.append(SoundDiagnostic(
                    timestamp=0.0,
                    action_type="music",
                    sound_name=str(music.get("sound_name") or "?"),
                    sound_id=str(music.get("sound_id")) if music.get("sound_id") is not None else None,
                    matched_tier=music.get("matched_tier"),
                    match_score=(
                        float(music["reranked_score"]) if music.get("reranked_score") is not None
                        else (float(music["match_score"]) if music.get("match_score") is not None else None)
                    ),
                    layer="music",
                    fetch_status=str(music.get("fetch_status") or "unknown"),
                    included_in_mix=bool(music.get("included_in_mix", True)),
                    note=music.get("fetch_note"),
                ))
            except Exception as exc:
                diag.warnings.append(f"diag music entry failed: {exc}")

        diag.sfx_in_final_mix = sum(
            1 for s in diag.sounds if s.layer == "sfx" and s.included_in_mix
        )
        diag.ambient_count = sum(1 for s in diag.sounds if s.layer == "ambient")
    except Exception as exc:
        diag.warnings.append(f"diag mix failed: {exc}")


def _safe_update_diag(job_id: str, diag: JobDiagnostics) -> None:
    try:
        _update_job(job_id, diagnostics=diag)
    except Exception as exc:
        logger.warning("Failed to attach diagnostics to job %s: %s", job_id, exc)


def run_pipeline(job_id: str, video_path: Path, preset: str, base_url: str) -> None:
    """Run the full pipeline in a background thread."""
    diag = JobDiagnostics(storage_mode=STORAGE_MODE)
    work_dir: Path | None = None
    try:
        _update_job(job_id, status=JobStatus.processing, progress_pct=5)

        _add_step(job_id, "Analyzing video (Director + scenes)")
        analysis = analyze_video(video_path, preset=preset)
        _collect_analysis_diag(diag, analysis)
        _safe_update_diag(job_id, diag)
        _update_job(job_id, progress_pct=40)

        _add_step(job_id, "Detecting onsets")
        onsets = detect_onsets_from_video(video_path)
        _update_job(job_id, progress_pct=50)

        _add_step(job_id, "Matching SFX from library")
        qdrant = get_qdrant_client()
        match_plan = match_analysis(analysis, qdrant, preset_name=preset)
        _collect_match_plan_diag(diag, match_plan)
        _safe_update_diag(job_id, diag)
        _update_job(job_id, progress_pct=65)

        _add_step(job_id, "Snapping timing")
        match_plan = snap_to_onsets(match_plan, onsets)

        if match_plan.get("music"):
            music_path = _resolve_media_path(str(match_plan["music"]["sound_path"]))
            tempo, beats = detect_beats(
                music_path,
                duration_limit_sec=float(analysis.get("duration_sec") or 0.0) or None,
            )
            strong = find_strong_beats(beats, every_n=4)
            match_plan = snap_to_beats(
                match_plan,
                beats,
                strong,
                strategy=analysis.get("strategy"),
            )
        _update_job(job_id, progress_pct=72)

        _add_step(job_id, "Mixing audio layers")
        work_dir = TEMP_DIR / video_path.stem
        work_dir.mkdir(parents=True, exist_ok=True)
        original_audio = work_dir / "audio.wav"
        if not original_audio.is_file():
            try:
                extract_audio(video_path, original_audio)
            except ValueError as exc:
                raise RuntimeError(
                    "Video has no audio track; cannot mix or extract speech/onsets.",
                ) from exc
        mixed_audio = work_dir / "mixed_audio.wav"
        preset_obj = get_preset(preset)
        mix_audio(match_plan, original_audio, mixed_audio, preset_obj)
        _collect_mix_diag(diag, match_plan)
        _safe_update_diag(job_id, diag)
        _update_job(job_id, progress_pct=85)

        _add_step(job_id, "Rendering final MP4")
        outputs_dir = work_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        final_mp4 = outputs_dir / f"api_{job_id}.mp4"
        render_final_video(video_path, mixed_audio, final_mp4)
        _update_job(job_id, progress_pct=92)

        _add_step(job_id, "Exporting timeline + packaging")
        timeline = build_timeline(
            video_path=video_path,
            original_audio_path=original_audio,
            match_plan=match_plan,
            video_duration_sec=float(analysis.get("duration_sec") or 0.0),
            project_name=f"api_{job_id}",
        )
        timeline_dir = work_dir / "timeline_exports"
        timeline_files = export_all_formats(
            timeline, timeline_dir, base_name=f"api_{job_id}",
        )

        zip_path = package_project(
            work_dir=work_dir,
            output_path=final_mp4,
            version=0,
            preset_name=preset,
            match_plan=match_plan,
            timeline_files=timeline_files,
            original_video_path=video_path,
            original_audio_path=original_audio,
        )

        store_result(job_id, final_mp4, "final.mp4")
        store_result(job_id, zip_path, "project.zip")

        _close_steps(job_id)

        sfx_count = sum(
            1 for m in match_plan.get("matches", []) if m.get("layer") == "sfx"
        )
        music_meta = match_plan.get("music") or {}
        music_name = music_meta.get("sound_name") if music_meta else None

        base = base_url.rstrip("/")
        _update_job(
            job_id,
            status=JobStatus.done,
            progress_pct=100,
            current_step=None,
            video_url=f"{base}/jobs/{job_id}/download/final.mp4",
            project_zip_url=f"{base}/jobs/{job_id}/download/project.zip",
            duration_sec=analysis.get("duration_sec"),
            sfx_count=sfx_count,
            music_track=music_name,
            diagnostics=diag,
        )
        logger.info("Job %s done", job_id)

    except Exception as exc:
        logger.error(
            "Job %s failed: %s\n%s",
            job_id,
            exc,
            traceback.format_exc(),
        )
        diag.warnings.append(f"Pipeline error: {exc}")
        _close_steps(job_id)
        _update_job(job_id, status=JobStatus.failed, error=str(exc), diagnostics=diag)
    finally:
        # Always clean up the per-job temp + the original upload — final.mp4 +
        # project.zip have already been copied into api_results/ by then. On a
        # 5 GB volume this is what keeps the disk from filling up after a few
        # jobs (uploads ~50–500 MB each, work_dir easily 200+ MB).
        try:
            if work_dir is not None and work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
        except Exception as cleanup_exc:
            logger.warning("temp cleanup failed for %s: %s", job_id, cleanup_exc)
        try:
            if video_path.is_file():
                video_path.unlink(missing_ok=True)
        except Exception as cleanup_exc:
            logger.warning("upload cleanup failed for %s: %s", job_id, cleanup_exc)
