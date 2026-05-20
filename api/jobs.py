"""Job store and background processing (thread-based MVP)."""
from __future__ import annotations

import traceback
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.analysis.beat_detector import detect_beats, find_strong_beats
from src.analysis.gemini_client import analyze_video
from src.config import LIBRARY_DIR, PROJECT_ROOT, TEMP_DIR
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

from api.models import JobInfo, JobStatus, JobStep
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


def run_pipeline(job_id: str, video_path: Path, preset: str, base_url: str) -> None:
    """Run the full pipeline in a background thread."""
    try:
        _update_job(job_id, status=JobStatus.processing, progress_pct=5)

        _add_step(job_id, "Analyzing video (Director + scenes)")
        analysis = analyze_video(video_path, preset=preset)
        _update_job(job_id, progress_pct=40)

        _add_step(job_id, "Detecting onsets")
        onsets = detect_onsets_from_video(video_path)
        _update_job(job_id, progress_pct=50)

        _add_step(job_id, "Matching SFX from library")
        qdrant = get_qdrant_client()
        match_plan = match_analysis(analysis, qdrant, preset_name=preset)
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
        )
        logger.info("Job %s done", job_id)

    except Exception as exc:
        logger.error(
            "Job %s failed: %s\n%s",
            job_id,
            exc,
            traceback.format_exc(),
        )
        _close_steps(job_id)
        _update_job(job_id, status=JobStatus.failed, error=str(exc))
