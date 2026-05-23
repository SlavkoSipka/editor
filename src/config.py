from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

FRAME_RATE: int = 2
MAX_VIDEO_LENGTH_SEC: int = 180
AUDIO_SAMPLE_RATE: int = 44100
ONSET_SYNC_WINDOW_MS: int = 350

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
# Persistent volume on Railway (and other hosts); default stays repo ./data locally.
DATA_DIR: Path = Path(os.environ.get("DATA_DIR", str(PROJECT_ROOT / "data")))
LIBRARY_DIR: Path = DATA_DIR / "library"
EMBEDDINGS_DIR: Path = DATA_DIR / "embeddings"
TEMP_DIR: Path = DATA_DIR / "temp"
LOGS_DIR: Path = DATA_DIR / "logs"
FIXTURES_DIR: Path = PROJECT_ROOT / "tests" / "fixtures"

# Storage mode for sound files:
#   "local" — full library on disk (dev machines, baked images)
#   "r2"    — sounds fetched on-demand from Cloudflare R2 into SOUND_CACHE_DIR
#             (used on Railway Hobby where the 5 GB volume can't hold the full library)
STORAGE_MODE: str = os.environ.get("STORAGE_MODE", "local").strip().lower() or "local"
SOUND_CACHE_DIR: Path = DATA_DIR / "sound_cache"
SOUND_CACHE_MAX_MB: int = int(os.environ.get("SOUND_CACHE_MAX_MB", "1500"))


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value or value.startswith("your_"):
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Copy .env.example to .env and set a real value."
        )
    return value


GEMINI_API_KEY: str = _require_env("GEMINI_API_KEY")
FREESOUND_API_KEY: str = _require_env("FREESOUND_API_KEY")


def ensure_dirs() -> None:
    for directory in (LIBRARY_DIR, EMBEDDINGS_DIR, TEMP_DIR, LOGS_DIR, SOUND_CACHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
