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


class SoundDiagnostic(BaseModel):
    timestamp: float = 0.0
    action_type: str = "?"
    sound_name: str = "?"
    sound_id: Optional[str] = None
    matched_tier: Optional[str] = None
    match_score: Optional[float] = None
    value_tier: Optional[str] = None
    layer: str = "sfx"
    fetch_status: str = "unknown"
    included_in_mix: bool = True
    note: Optional[str] = None


class JobDiagnostics(BaseModel):
    director_style: Optional[str] = None
    director_vibe: Optional[str] = None
    director_recommended_preset: Optional[str] = None
    director_max_sfx: Optional[int] = None
    anchor_moments: int = 0

    scenes_detected: int = 0
    actions_suggested: int = 0
    sfx_matched: int = 0
    sfx_after_selectivity: int = 0
    sfx_in_final_mix: int = 0
    ambient_count: int = 0

    music_found: bool = False
    music_name: Optional[str] = None
    music_fetch_status: Optional[str] = None

    storage_mode: str = "unknown"
    r2_fetch_attempts: int = 0
    r2_fetch_failures: int = 0

    speech_regions: int = 0
    speech_coverage_pct: float = 0.0

    verification_checked: int = 0
    verification_dropped: int = 0
    verification_dropped_detail: list[str] = Field(default_factory=list)

    sounds: list[SoundDiagnostic] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


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
    diagnostics: Optional[JobDiagnostics] = None


class PresetInfo(BaseModel):
    id: str
    name: str
    description: str
    density: float = 0.5


class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    library_loaded: bool
    qdrant_connected: bool


class SoundFeedback(BaseModel):
    sound_index: int
    sound_id: Optional[str] = None
    sound_name: Optional[str] = None
    action_type: Optional[str] = None
    timestamp: float
    matched_tier: Optional[str] = None
    match_score: Optional[float] = None
    rating: str  # "good" | "wrong" | "unnecessary"


class MissingSoundMark(BaseModel):
    timestamp: float
    note: Optional[str] = None


class JobFeedback(BaseModel):
    job_id: str
    preset: str
    density: Optional[float] = None
    overall_rating: Optional[int] = None  # 1-5
    density_feedback: Optional[str] = None  # "too_low" | "right" | "too_high"
    overall_note: Optional[str] = None
    sound_feedback: list[SoundFeedback] = Field(default_factory=list)
    missing_sounds: list[MissingSoundMark] = Field(default_factory=list)
    submitted_at: Optional[str] = None
