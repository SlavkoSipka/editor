from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PointStruct,
    VectorParams,
)

from src.utils.logger import get_logger

logger = get_logger("vector_store")

import os
import time

COLLECTION_NAME = "sfx_library"
VECTOR_SIZE = 384
_HOST = os.environ.get("QDRANT_HOST", "localhost")
_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
_CONNECT_RETRIES = int(os.environ.get("QDRANT_CONNECT_RETRIES", "5"))
_CONNECT_BACKOFF_SEC = float(os.environ.get("QDRANT_CONNECT_BACKOFF_SEC", "1.5"))
_BATCH_SIZE = 100

_DOCKER_HINT = (
    "Qdrant not reachable at {host}:{port}. On Railway: ensure Qdrant boots in "
    "the entrypoint (check /tmp/qdrant.log) or set QDRANT_RESET=1 once to wipe "
    "corrupt collection data. Locally: docker run -p 6333:6333 "
    "-v $(pwd)/data/embeddings:/qdrant/storage qdrant/qdrant"
)


def get_qdrant_client() -> QdrantClient:
    """Connect to Qdrant with a small retry/backoff so single-container deploys
    (Qdrant + API in the same image) don't fail the first request after boot."""
    last_exc: Exception | None = None
    for attempt in range(1, _CONNECT_RETRIES + 1):
        client = QdrantClient(host=_HOST, port=_PORT, timeout=10.0)
        try:
            client.get_collections()
            return client
        except Exception as exc:
            last_exc = exc
            if attempt < _CONNECT_RETRIES:
                logger.info(
                    "Qdrant not ready (attempt %d/%d): %s; retrying in %.1fs",
                    attempt, _CONNECT_RETRIES, exc, _CONNECT_BACKOFF_SEC,
                )
                time.sleep(_CONNECT_BACKOFF_SEC)
    raise RuntimeError(
        _DOCKER_HINT.format(host=_HOST, port=_PORT)
    ) from last_exc


def create_collection(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
) -> None:
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    logger.info(
        "Created/recreated collection %r (size=%d, distance=Cosine)",
        collection_name, VECTOR_SIZE,
    )


def index_sounds(
    client: QdrantClient,
    collection_name: str,
    sounds: list[dict[str, Any]],
) -> int:
    total = 0
    for i in range(0, len(sounds), _BATCH_SIZE):
        chunk = sounds[i : i + _BATCH_SIZE]
        points = [
            PointStruct(id=s["id"], vector=s["vector"], payload=s["payload"])
            for s in chunk
        ]
        client.upsert(collection_name=collection_name, points=points)
        total += len(points)
        logger.info("Upserted %d/%d points", total, len(sounds))
    return total


def _build_tier_filter(tier_filter: list[str] | None) -> Filter | None:
    if not tier_filter:
        return None
    return Filter(
        should=[FieldCondition(key="tiers", match=MatchAny(any=list(tier_filter)))]
    )


def search_sounds(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    top_k: int = 5,
    tier_filter: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Search by vector. When ``tier_filter`` is provided, returned points must
    have AT LEAST ONE matching tier in their ``tiers`` payload field."""
    hits = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        query_filter=_build_tier_filter(tier_filter),
    )
    return [
        {"id": h.id, "score": float(h.score), "payload": h.payload or {}}
        for h in hits
    ]


def search_by_text(
    client: QdrantClient,
    collection_name: str,
    query_text: str,
    top_k: int = 5,
    tier_filter: list[str] | None = None,
) -> list[dict[str, Any]]:
    # Lazy import: avoids loading the sentence-transformers model when callers
    # only need raw vector search.
    from src.library.embedder import embed_texts

    vector = embed_texts([query_text])[0]
    return search_sounds(
        client, collection_name, vector, top_k=top_k, tier_filter=tier_filter,
    )


def count_by_tier(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    tiers: tuple[str, ...] = ("ugc", "cinematic", "foley", "ambient"),
) -> dict[str, int]:
    """Return how many indexed points carry each tier. Used by the matcher to
    skip tier filters that would resolve to an empty pool."""
    counts: dict[str, int] = {}
    for tier in tiers:
        try:
            result = client.count(
                collection_name=collection_name,
                count_filter=Filter(
                    must=[FieldCondition(key="tiers", match=MatchAny(any=[tier]))],
                ),
                exact=False,
            )
            counts[tier] = int(result.count)
        except Exception as exc:
            logger.warning("Tier count for %r failed: %s", tier, exc)
            counts[tier] = 0
    return counts
