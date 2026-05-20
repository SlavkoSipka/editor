from __future__ import annotations

import time
from typing import Any

import requests

from src.config import FREESOUND_API_KEY
from src.utils.logger import get_logger

logger = get_logger("freesound_client")

_BASE_URL = "https://freesound.org/apiv2"
_API_FIELDS = (
    "id,name,description,tags,duration,filesize,previews,license,"
    "download,username,avg_rating,num_downloads"
)
_RATE_LIMIT_DELAY_SEC = 1.1


def search_sounds(
    query: str,
    page: int = 1,
    page_size: int = 150,
    cc0_only: bool = True,
    min_duration: float = 0.2,
    max_duration: float = 15.0,
) -> dict[str, Any]:
    filters: list[str] = [f"duration:[{min_duration} TO {max_duration}]"]
    if cc0_only:
        filters.insert(0, 'license:"Creative Commons 0"')

    params = {
        "query": query,
        "filter": " ".join(filters),
        "fields": _API_FIELDS,
        "sort": "rating_desc",
        "page": page,
        "page_size": page_size,
        "token": FREESOUND_API_KEY,
    }

    response = requests.get(f"{_BASE_URL}/search/text/", params=params, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(
            f"Freesound search failed (HTTP {response.status_code}) for query={query!r}: "
            f"{response.text}"
        )
    # Stay comfortably under the 60 req/min API limit.
    time.sleep(_RATE_LIMIT_DELAY_SEC)
    return response.json()


def get_sound_details(sound_id: int) -> dict[str, Any]:
    params = {"token": FREESOUND_API_KEY, "fields": _API_FIELDS}
    response = requests.get(
        f"{_BASE_URL}/sounds/{sound_id}/", params=params, timeout=30,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Freesound sound details failed (HTTP {response.status_code}) "
            f"for id={sound_id}: {response.text}"
        )
    time.sleep(_RATE_LIMIT_DELAY_SEC)
    return response.json()
