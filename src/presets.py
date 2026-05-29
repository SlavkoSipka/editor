from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Preset:
    name: str
    description: str
    boost_keywords: list[str] = field(default_factory=list)
    penalty_keywords: list[str] = field(default_factory=list)
    sfx_volume_db: float = 0.0
    ambient_volume_db: float = -8.0
    original_audio_volume_db: float = -3.0
    music_volume_db: float = -18.0
    music_mood_preference: str = "neutral"
    reverb_amount: float = 0.0
    rerank_strength: float = 0.5
    # Sound density: 0.0 = minimal/sparse SFX, 1.0 = busy/packed.
    density: float = 0.5
    # Tier routing — empty lists mean "no filter" (search the global pool).
    primary_tiers: list[str] = field(default_factory=list)
    fallback_tiers: list[str] = field(default_factory=list)


PRESETS: dict[str, Preset] = {
    "dramatic": Preset(
        name="Dramatic",
        description="Deep impacts, long reverb tails, low-end emphasis, sparse and impactful",
        boost_keywords=[
            "deep", "low", "boom", "impact", "sub", "bass", "cinematic",
            "reverb", "tail", "epic", "heavy", "dark",
        ],
        penalty_keywords=[
            "cartoon", "comedic", "playful", "bright", "tinny", "high pitch", "happy",
        ],
        sfx_volume_db=-2.0,
        ambient_volume_db=-10.0,
        original_audio_volume_db=-4.0,
        music_volume_db=-16.0,
        music_mood_preference="dramatic",
        reverb_amount=0.4,
        rerank_strength=0.6,
        density=0.40,
        primary_tiers=["cinematic", "foley"],
        fallback_tiers=["ugc"],
    ),
    "energetic": Preset(
        name="Energetic",
        description="Sharp percussive transitions, whooshes, fast pace, bright frequencies",
        boost_keywords=[
            "sharp", "fast", "snappy", "whoosh", "swoosh", "transition", "impact",
            "punchy", "bright", "tight", "crisp", "trailer",
        ],
        penalty_keywords=["slow", "drone", "ambient", "soft", "muffled", "dull"],
        sfx_volume_db=-1.0,
        ambient_volume_db=-12.0,
        original_audio_volume_db=-3.0,
        music_volume_db=-14.0,
        music_mood_preference="energetic",
        reverb_amount=0.15,
        rerank_strength=0.6,
        density=0.75,
        primary_tiers=["ugc", "cinematic"],
        fallback_tiers=["foley"],
    ),
    "mysterious": Preset(
        name="Mysterious",
        description="Ambient drones, reverse swells, subtle textures, moderate reverb",
        boost_keywords=[
            "drone", "ambient", "reverse", "swell", "tense", "dark", "ethereal",
            "atmospheric", "tonal", "evolving", "haunting",
        ],
        penalty_keywords=["bright", "happy", "cartoon", "comedic", "snappy", "sharp"],
        sfx_volume_db=-4.0,
        ambient_volume_db=-6.0,
        original_audio_volume_db=-3.0,
        music_volume_db=-15.0,
        music_mood_preference="mysterious",
        reverb_amount=0.6,
        rerank_strength=0.5,
        density=0.35,
        primary_tiers=["cinematic", "ambient"],
        fallback_tiers=["foley"],
    ),
    "playful": Preset(
        name="Playful",
        description="Bouncy cartoon-ish hits, plucks, whistles, light reverb, comedic timing",
        boost_keywords=[
            "cartoon", "comedic", "bouncy", "pluck", "whistle", "boing", "pop",
            "playful", "light", "fun", "cute", "bubble",
        ],
        penalty_keywords=["dark", "ominous", "heavy", "scary", "tense", "horror", "boom"],
        sfx_volume_db=-2.0,
        ambient_volume_db=-10.0,
        original_audio_volume_db=-2.0,
        music_volume_db=-16.0,
        music_mood_preference="playful",
        reverb_amount=0.1,
        rerank_strength=0.6,
        density=0.70,
        primary_tiers=["ugc"],
        fallback_tiers=["foley"],
    ),
    "luxury": Preset(
        name="Luxury",
        description="Clean minimal SFX, soft impacts, subtle ambient, polished and restrained",
        boost_keywords=[
            "clean", "smooth", "polished", "soft", "subtle", "premium", "elegant",
            "minimal", "refined", "silk",
        ],
        penalty_keywords=["harsh", "noisy", "distorted", "rough", "messy", "cartoon", "cheap"],
        sfx_volume_db=-4.0,
        ambient_volume_db=-12.0,
        original_audio_volume_db=-2.0,
        music_volume_db=-18.0,
        music_mood_preference="luxury",
        reverb_amount=0.25,
        rerank_strength=0.5,
        density=0.30,
        primary_tiers=["cinematic"],
        fallback_tiers=["foley", "ugc"],
    ),
    "tiktok_viral": Preset(
        name="TikTok Viral",
        description="High-energy UGC style with viral meme sounds, sharp transitions, beat-driven SFX",
        boost_keywords=["pop", "tiktok", "viral", "snappy", "sharp", "punchy", "drop"],
        penalty_keywords=["slow", "orchestral", "classical", "ambient drone"],
        sfx_volume_db=0.0,
        ambient_volume_db=-14.0,
        original_audio_volume_db=-3.0,
        music_volume_db=-12.0,
        music_mood_preference="energetic",
        reverb_amount=0.1,
        rerank_strength=0.7,
        density=0.85,
        primary_tiers=["ugc"],
        fallback_tiers=["cinematic"],
    ),
    "vlog_casual": Preset(
        name="Vlog Casual",
        description="Light, friendly UGC sounds — vlog whooshes, soft pops, notification chirps",
        boost_keywords=["pop", "whoosh", "notification", "swipe", "light", "soft"],
        penalty_keywords=["dramatic", "epic", "horror", "tense"],
        sfx_volume_db=-2.0,
        ambient_volume_db=-12.0,
        original_audio_volume_db=-2.0,
        music_volume_db=-16.0,
        music_mood_preference="playful",
        reverb_amount=0.15,
        rerank_strength=0.6,
        density=0.50,
        primary_tiers=["ugc"],
        fallback_tiers=["foley"],
    ),
    "viral_ad": Preset(
        name="Viral Ad",
        description="UGC-style with cinematic punch — viral SFX + cinematic logo reveal",
        boost_keywords=["punchy", "viral", "tiktok", "drop", "impact", "cinematic"],
        penalty_keywords=["slow", "drone", "ambient pad"],
        sfx_volume_db=-1.0,
        ambient_volume_db=-12.0,
        original_audio_volume_db=-3.0,
        music_volume_db=-13.0,
        music_mood_preference="energetic",
        reverb_amount=0.2,
        rerank_strength=0.7,
        density=0.70,
        primary_tiers=["ugc", "cinematic"],
        fallback_tiers=["foley"],
    ),
}


def get_preset(name: str) -> Preset:
    """Look up a preset by name (case-insensitive)."""
    key = (name or "").strip().lower()
    if key not in PRESETS:
        valid = ", ".join(sorted(PRESETS.keys()))
        raise KeyError(f"Unknown preset {name!r}. Valid presets: {valid}")
    return PRESETS[key]
