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


@app.get("/debug/collections")
def debug_collections() -> dict:
    """Show all Qdrant collections and their point counts. Diagnostics only."""
    try:
        client = get_qdrant_client()
        collections = client.get_collections().collections
        result: list[dict] = []
        for col in collections:
            try:
                info = client.get_collection(col.name)
                vectors_cfg = getattr(info.config.params, "vectors", None)
                vector_size = getattr(vectors_cfg, "size", None)
                result.append({
                    "name": col.name,
                    "points_count": info.points_count,
                    "status": str(info.status),
                    "vector_size": vector_size,
                })
            except Exception as exc:
                result.append({"name": col.name, "error": str(exc)})
        return {
            "qdrant_connected": True,
            "collection_count": len(result),
            "collections": result,
        }
    except Exception as exc:
        return {"qdrant_connected": False, "error": str(exc)}


@app.get("/debug/library")
def debug_library() -> dict:
    """Show storage mode, R2 config presence, and local cache stats."""
    try:
        from src.config import DATA_DIR, STORAGE_MODE

        cache_dir = DATA_DIR / "sound_cache"
        cache_files = list(cache_dir.glob("*")) if cache_dir.exists() else []
        cache_size_mb = sum(
            f.stat().st_size for f in cache_files if f.is_file()
        ) / 1024 / 1024

        manifest_path = LIBRARY_DIR / "manifest.json"
        music_manifest_path = LIBRARY_DIR / "music" / "manifest.json"

        return {
            "storage_mode": STORAGE_MODE,
            "r2_endpoint_set": bool(os.environ.get("R2_ENDPOINT_URL")),
            "r2_bucket": os.environ.get("R2_BUCKET", "(not set)"),
            "sound_cache_files": len(cache_files),
            "sound_cache_size_mb": round(cache_size_mb, 1),
            "data_dir": str(DATA_DIR),
            "library_dir": str(LIBRARY_DIR),
            "sfx_manifest_exists": manifest_path.is_file(),
            "music_manifest_exists": music_manifest_path.is_file(),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@app.get("/debug/music-test")
def debug_music_test() -> dict:
    """Try a sample music search to see if the music_library collection works."""
    try:
        from src.library.music_embedder import MUSIC_COLLECTION_NAME
        from src.library.vector_store import search_by_text

        client = get_qdrant_client()
        collections = [c.name for c in client.get_collections().collections]
        if MUSIC_COLLECTION_NAME not in collections:
            return {
                "ok": False,
                "reason": f"{MUSIC_COLLECTION_NAME} collection does not exist in Qdrant",
                "available_collections": collections,
            }

        info = client.get_collection(MUSIC_COLLECTION_NAME)
        if info.points_count == 0:
            return {
                "ok": False,
                "reason": f"{MUSIC_COLLECTION_NAME} collection exists but has 0 points",
            }

        try:
            results = search_by_text(
                client,
                MUSIC_COLLECTION_NAME,
                "energetic upbeat background music",
                top_k=3,
            )
            return {
                "ok": True,
                "music_points": info.points_count,
                "sample_search_results": [
                    {
                        "id": r.get("id"),
                        "name": (r.get("payload") or {}).get("name")
                            or (r.get("payload") or {}).get("sound_name"),
                        "score": round(float(r.get("score", 0.0)), 3),
                    }
                    for r in results
                ],
            }
        except Exception as exc:
            return {
                "ok": False,
                "reason": f"music search failed: {exc}",
                "music_points": info.points_count,
            }
    except Exception as exc:
        return {"ok": False, "reason": f"debug failed: {exc}"}


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
