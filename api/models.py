from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    done = "done"
    failed = "failed"


class JobStep(BaseModel):
    name: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    detail: Optional[str] = None


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    preset: str
    created_at: datetime
    updated_at: datetime
    progress_pct: int = 0
    current_step: Optional[str] = None
    steps: list[JobStep] = Field(default_factory=list)
    error: Optional[str] = None
    video_url: Optional[str] = None
    project_zip_url: Optional[str] = None
    duration_sec: Optional[float] = None
    sfx_count: Optional[int] = None
    music_track: Optional[str] = None


class PresetInfo(BaseModel):
    id: str
    name: str
    description: str


class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    library_loaded: bool
    qdrant_connected: bool
