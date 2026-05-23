"""Sound Recipes — layered sound design templates for anchor moments."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.utils.logger import get_logger

logger = get_logger("recipes")


@dataclass
class RecipeLayer:
    """One layer within a recipe."""

    role: str                  # 'anticipation' | 'impact' | 'body' | 'tail' | 'accent'
    search_query: str          # what to search the library for
    tier_filter: list[str]     # which library tiers to pull from
    offset_sec: float          # timing relative to anchor (negative = before)
    volume_db: float           # relative volume for this layer
    duration_limit_sec: float  # max length for this layer's sound
    optional: bool = False     # if True, recipe still works without this layer


@dataclass
class Recipe:
    """A layered sound design template for a moment type."""

    name: str
    description: str
    layers: list[RecipeLayer] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# RECIPE DEFINITIONS
# ─────────────────────────────────────────────────────────────
# Keyed by "{moment_type}__{preset_family}".

RECIPES: dict[str, Recipe] = {
    "logo_reveal__ugc": Recipe(
        name="UGC Logo Reveal",
        description="Punchy logo hit with a quick riser and short shimmer",
        layers=[
            RecipeLayer(
                "anticipation", "fast riser whoosh build up",
                ["ugc", "cinematic"], offset_sec=-0.45, volume_db=-6.0,
                duration_limit_sec=0.6, optional=True,
            ),
            RecipeLayer(
                "impact", "punchy impact hit logo sting",
                ["ugc", "cinematic"], offset_sec=0.0, volume_db=0.0,
                duration_limit_sec=1.5,
            ),
            RecipeLayer(
                "body", "sparkle shimmer pop", ["ugc"],
                offset_sec=0.05, volume_db=-7.0, duration_limit_sec=1.0,
                optional=True,
            ),
        ],
    ),
    "logo_reveal__cinematic": Recipe(
        name="Cinematic Logo Reveal",
        description="Deep impact with long reverb tail and orchestral shimmer",
        layers=[
            RecipeLayer(
                "anticipation", "cinematic riser tension build",
                ["cinematic"], offset_sec=-0.8, volume_db=-5.0,
                duration_limit_sec=1.0, optional=True,
            ),
            RecipeLayer(
                "impact", "deep cinematic boom impact",
                ["cinematic"], offset_sec=0.0, volume_db=0.0,
                duration_limit_sec=2.5,
            ),
            RecipeLayer(
                "body", "metallic shimmer swell glass",
                ["cinematic"], offset_sec=0.0, volume_db=-8.0,
                duration_limit_sec=2.0, optional=True,
            ),
            RecipeLayer(
                "tail", "long reverb tail ambience",
                ["cinematic", "ambient"], offset_sec=0.1, volume_db=-12.0,
                duration_limit_sec=4.0, optional=True,
            ),
        ],
    ),

    "hook__ugc": Recipe(
        name="UGC Hook",
        description="Attention-grabbing opener — quick whoosh into a tight pop",
        layers=[
            RecipeLayer(
                "impact", "sharp pop transition hit", ["ugc"],
                offset_sec=0.0, volume_db=-2.0, duration_limit_sec=1.0,
            ),
            RecipeLayer(
                "body", "quick whoosh sweep", ["ugc"],
                offset_sec=-0.15, volume_db=-8.0, duration_limit_sec=0.5,
                optional=True,
            ),
        ],
    ),
    "hook__cinematic": Recipe(
        name="Cinematic Hook",
        description="Tense atmospheric opener with a low swell",
        layers=[
            RecipeLayer(
                "anticipation", "low drone swell tension",
                ["cinematic", "ambient"], offset_sec=-0.3, volume_db=-8.0,
                duration_limit_sec=1.5, optional=True,
            ),
            RecipeLayer(
                "impact", "subtle cinematic hit", ["cinematic"],
                offset_sec=0.0, volume_db=-4.0, duration_limit_sec=1.5,
            ),
        ],
    ),

    "drop__ugc": Recipe(
        name="UGC Drop",
        description="Hard-hitting beat drop moment",
        layers=[
            RecipeLayer(
                "anticipation", "riser build up tension",
                ["ugc", "cinematic"], offset_sec=-0.6, volume_db=-4.0,
                duration_limit_sec=0.8, optional=True,
            ),
            RecipeLayer(
                "impact", "bass drop deep impact boom",
                ["ugc", "cinematic"], offset_sec=0.0, volume_db=0.0,
                duration_limit_sec=2.0,
            ),
            RecipeLayer(
                "body", "punchy sub hit", ["ugc"],
                offset_sec=0.0, volume_db=-6.0, duration_limit_sec=1.0,
                optional=True,
            ),
        ],
    ),
    "drop__cinematic": Recipe(
        name="Cinematic Drop",
        description="Epic orchestral impact with sustain",
        layers=[
            RecipeLayer(
                "anticipation", "cinematic tension riser orchestral",
                ["cinematic"], offset_sec=-1.0, volume_db=-3.0,
                duration_limit_sec=1.2, optional=True,
            ),
            RecipeLayer(
                "impact", "epic cinematic boom impact",
                ["cinematic"], offset_sec=0.0, volume_db=0.0,
                duration_limit_sec=3.0,
            ),
            RecipeLayer(
                "tail", "reverb tail rumble",
                ["cinematic", "ambient"], offset_sec=0.1, volume_db=-10.0,
                duration_limit_sec=4.0, optional=True,
            ),
        ],
    ),

    "transition__ugc": Recipe(
        name="UGC Transition",
        description="Snappy whoosh transition",
        layers=[
            RecipeLayer(
                "impact", "fast whoosh swoosh transition", ["ugc"],
                offset_sec=0.0, volume_db=-3.0, duration_limit_sec=1.0,
            ),
        ],
    ),
    "transition__cinematic": Recipe(
        name="Cinematic Transition",
        description="Deep whoosh with low-end body",
        layers=[
            RecipeLayer(
                "impact", "cinematic whoosh deep transition", ["cinematic"],
                offset_sec=0.0, volume_db=-2.0, duration_limit_sec=2.0,
            ),
            RecipeLayer(
                "body", "low rumble sweep", ["cinematic"],
                offset_sec=0.0, volume_db=-10.0, duration_limit_sec=1.5,
                optional=True,
            ),
        ],
    ),
}


UGC_PRESETS = {"tiktok_viral", "vlog_casual", "viral_ad", "playful", "energetic"}
CINEMATIC_PRESETS = {"dramatic", "mysterious", "luxury"}


def preset_family(preset_name: str) -> str:
    """Map a preset to 'ugc' or 'cinematic'."""
    if preset_name in CINEMATIC_PRESETS:
        return "cinematic"
    return "ugc"


def moment_type_for_action(action_type: str) -> str | None:
    """Determine which recipe moment type (if any) applies to an action."""
    if not action_type:
        return None
    at = action_type.lower()
    if "logo" in at:
        return "logo_reveal"
    if "hook" in at:
        return "hook"
    if any(k in at for k in ("drop", "beat_drop", "climax")):
        return "drop"
    if any(k in at for k in (
        "scene_transition", "video_transition", "transition", "whoosh",
    )):
        return "transition"
    return None


def get_recipe(moment_type: str, preset_name: str) -> Recipe | None:
    """Look up a recipe for a moment type + preset. Returns None if not found."""
    return RECIPES.get(f"{moment_type}__{preset_family(preset_name)}")
