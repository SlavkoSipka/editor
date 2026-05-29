"""Supabase client wrapper for feedback storage."""

from __future__ import annotations

import os
from functools import lru_cache

from src.utils.logger import get_logger

logger = get_logger("supabase")


@lru_cache(maxsize=1)
def get_supabase():
    """Return a cached Supabase client, or None if not configured."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        logger.warning(
            "Supabase not configured (SUPABASE_URL / SUPABASE_SERVICE_KEY missing)"
        )
        return None
    try:
        from supabase import create_client

        return create_client(url, key)
    except Exception as exc:
        logger.error("Failed to init Supabase: %s", exc)
        return None


def is_available() -> bool:
    return get_supabase() is not None
