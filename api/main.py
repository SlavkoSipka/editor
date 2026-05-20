"""FastAPI app entrypoint."""
from __future__ import annotations

import os
import threading
from pathlib import Path

import librosa
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.config import LIBRARY_DIR, MAX_VIDEO_LENGTH_SEC, ensure_dirs
from src.library.vector_store import get_qdrant_client
from src.presets import PRESETS
from src.utils.ffmpeg_check import check_ffmpeg

from api.jobs import create_job, get_job, list_jobs, run_pipeline
from api.models import HealthResponse, JobInfo, JobStatus, PresetInfo
from api.storage import get_result_path, save_upload

_ALLOWED_DOWNLOADS = frozenset({"final.mp4", "project.zip"})

_raw_origins = os.environ.get("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [
    o.strip() for o in _raw_origins.split(",") if o.strip()
] or ["*"]

app = FastAPI(title="AI Sound Effects API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    ensure_dirs()
    check_ffmpeg()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    manifest_ok = (LIBRARY_DIR / "manifest.json").is_file()
    qdrant_ok = False
    try:
        client = get_qdrant_client()
        client.get_collections()
        qdrant_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        pipeline_ready=True,
        library_loaded=manifest_ok,
        qdrant_connected=qdrant_ok,
    )


@app.get("/presets", response_model=list[PresetInfo])
def presets() -> list[PresetInfo]:
    return [
        PresetInfo(id=k, name=p.name, description=p.description)
        for k, p in PRESETS.items()
    ]


@app.post("/jobs", response_model=JobInfo)
async def submit_job(
    request: Request,
    video: UploadFile = File(...),
    preset: str = Form(...),
) -> JobInfo:
    if preset not in PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset: {preset}")

    contents = await video.read()
    if len(contents) > 500 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Video file too large (max 500 MB)")
    if len(contents) < 1024:
        raise HTTPException(status_code=400, detail="Video file empty")

    saved_path, _ = save_upload(contents, video.filename or "upload.mp4")

    try:
        duration = float(librosa.get_duration(path=str(saved_path)))
    except Exception as exc:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not read video: {exc}") from exc

    if duration > MAX_VIDEO_LENGTH_SEC:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"Video too long: {duration:.1f}s > {MAX_VIDEO_LENGTH_SEC}s",
        )

    job = create_job(preset)
    base_url = str(request.base_url).rstrip("/")
    thread = threading.Thread(
        target=run_pipeline,
        args=(job.job_id, saved_path, preset, base_url),
        daemon=True,
    )
    thread.start()
    return job


@app.get("/jobs/{job_id}", response_model=JobInfo)
def get_job_status(job_id: str) -> JobInfo:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/jobs", response_model=list[JobInfo])
def list_recent_jobs(limit: int = 20) -> list[JobInfo]:
    return list_jobs(limit=limit)


@app.get("/jobs/{job_id}/download/{filename}")
def download_result(job_id: str, filename: str) -> FileResponse:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.done:
        raise HTTPException(
            status_code=409,
            detail=f"Job not done (status: {job.status})",
        )
    if filename not in _ALLOWED_DOWNLOADS:
        raise HTTPException(status_code=404, detail="Unknown file")

    path = get_result_path(job_id, filename)
    if not path:
        raise HTTPException(status_code=404, detail="Result file not found")

    media_type = "video/mp4" if filename.endswith(".mp4") else "application/zip"
    return FileResponse(path, media_type=media_type, filename=filename)
