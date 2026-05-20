"""Local file storage. Easy to swap for S3/R2 later."""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from src.config import DATA_DIR

UPLOADS_DIR = DATA_DIR / "api_uploads"
RESULTS_DIR = DATA_DIR / "api_results"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_upload(file_bytes: bytes, original_filename: str) -> tuple[Path, str]:
    """Save uploaded file and return (path, unique_filename)."""
    safe_ext = Path(original_filename).suffix.lower() or ".mp4"
    unique_name = f"{uuid.uuid4().hex}{safe_ext}"
    dest = UPLOADS_DIR / unique_name
    dest.write_bytes(file_bytes)
    return dest, unique_name


def store_result(job_id: str, source_path: Path, result_name: str) -> Path:
    """Copy a finished file into the results directory under the job_id."""
    job_dir = RESULTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    dest = job_dir / result_name
    shutil.copy2(source_path, dest)
    return dest


def get_result_path(job_id: str, filename: str) -> Path | None:
    """Return the full path to a result file, or None if missing."""
    p = (RESULTS_DIR / job_id / filename).resolve()
    job_root = (RESULTS_DIR / job_id).resolve()
    try:
        p.relative_to(job_root)
    except ValueError:
        return None
    return p if p.is_file() else None
