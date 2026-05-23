"""Build layered SFX from a recipe for an anchor moment."""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient

from src.library.embedder import embed_texts
from src.library.vector_store import COLLECTION_NAME, search_sounds
from src.matching.recipes import Recipe, RecipeLayer, get_recipe, moment_type_for_action
from src.utils.logger import get_logger

logger = get_logger("recipe_builder")

RECIPE_LAYER_MIN_SCORE = 0.32
RECIPE_SEARCH_TOP_K = 8


def _layer_match_entry(
    anchor_action: dict[str, Any],
    layer: RecipeLayer,
    candidate: dict[str, Any],
    recipe: Recipe,
    layer_ts: float,
) -> dict[str, Any]:
    action_type = str(anchor_action.get("action_type", anchor_action.get("type", "")))
    payload = candidate.get("payload") or {}
    return {
        "scene_index": int(anchor_action.get("scene_index", -1)),
        "scene_start_sec": float(anchor_action.get("scene_start_sec", layer_ts)),
        "scene_end_sec": float(anchor_action.get("scene_end_sec", layer_ts + 5)),
        "absolute_timestamp": layer_ts,
        "action_type": f"{action_type}__recipe_{layer.role}",
        "action_description": layer.search_query,
        "intensity": "sharp" if layer.role == "impact" else "medium",
        "confidence": float(anchor_action.get("confidence") or 0.9),
        "sound_id": candidate["id"],
        "sound_name": payload.get("name", "unknown"),
        "sound_path": payload.get("local_path", ""),
        "sound_duration": float(payload.get("duration") or 1.0),
        "match_score": float(candidate["score"]),
        "reranked_score": float(candidate["score"]),
        "rerank_reason": "",
        "layer": "sfx",
        "matched_tier": (layer.tier_filter[0] if layer.tier_filter else "global"),
        "routing_path": "recipe",
        "tier_filter_used": list(layer.tier_filter),
        "recipe_name": recipe.name,
        "recipe_moment_type": anchor_action.get("_recipe_moment_type", ""),
        "recipe_role": layer.role,
        "recipe_layer_volume_db": float(layer.volume_db),
        "value_tier": "anchor",
    }


def build_recipe_layers(
    anchor_action: dict[str, Any],
    preset_name: str,
    qdrant_client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
) -> list[dict[str, Any]] | None:
    """Build all layers of a recipe for an anchor action.

    Returns a list of match dicts (one per recipe layer), or None if no recipe
    applies / the required layers can't be matched well enough."""
    action_type = str(anchor_action.get("action_type", anchor_action.get("type", "")))
    moment = moment_type_for_action(action_type)
    if not moment:
        return None

    recipe = get_recipe(moment, preset_name)
    if not recipe:
        return None

    anchor_ts = float(anchor_action.get("absolute_timestamp") or 0.0)
    anchor_action["_recipe_moment_type"] = moment

    layer_matches: list[dict[str, Any]] = []
    for layer in recipe.layers:
        try:
            query_vec = embed_texts([layer.search_query])[0]
        except Exception as exc:
            logger.warning(
                "Recipe '%s': embedding failed for layer '%s' (%s) — %s falling back",
                recipe.name, layer.role,
                "optional" if layer.optional else "required", exc,
            )
            if layer.optional:
                continue
            return None

        candidates = search_sounds(
            qdrant_client, collection_name, query_vec,
            top_k=RECIPE_SEARCH_TOP_K, tier_filter=layer.tier_filter or None,
        )
        candidates = [
            c for c in candidates
            if float((c.get("payload") or {}).get("duration") or 0.0)
            <= layer.duration_limit_sec
        ]
        if not candidates:
            if layer.optional:
                logger.info(
                    "Recipe '%s': skipping optional layer '%s' (no match)",
                    recipe.name, layer.role,
                )
                continue
            logger.info(
                "Recipe '%s': required layer '%s' unmatched — falling back",
                recipe.name, layer.role,
            )
            return None

        best = candidates[0]
        if float(best["score"]) < RECIPE_LAYER_MIN_SCORE:
            if layer.optional:
                logger.info(
                    "Recipe '%s': optional layer '%s' below score floor (%.2f); skipping",
                    recipe.name, layer.role, float(best["score"]),
                )
                continue
            logger.info(
                "Recipe '%s': required layer '%s' match too weak (%.2f) — falling back",
                recipe.name, layer.role, float(best["score"]),
            )
            return None

        layer_ts = max(0.0, anchor_ts + layer.offset_sec)
        layer_matches.append(
            _layer_match_entry(anchor_action, layer, best, recipe, layer_ts)
        )

    if not layer_matches:
        return None

    logger.info(
        "Recipe '%s': built %d layer(s) at %.2fs",
        recipe.name, len(layer_matches), anchor_ts,
    )
    return layer_matches
