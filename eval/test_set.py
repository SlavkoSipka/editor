"""Definition of the evaluation test video set."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import PROJECT_ROOT

EVAL_VIDEOS_DIR = PROJECT_ROOT / "eval" / "videos"


@dataclass
class EvalVideo:
    """One test video with expectations."""

    filename: str
    label: str
    expected_style: str
    recommended_preset: str
    min_expected_sfx: int
    max_expected_sfx: int
    has_speech: bool
    notes: str = ""


EVAL_SET: list[EvalVideo] = [
    EvalVideo(
        filename="test.mp4",
        label="Embroidery brand ad (existing test video)",
        expected_style="UGC product ad",
        recommended_preset="tiktok_viral",
        min_expected_sfx=4,
        max_expected_sfx=12,
        has_speech=True,
        notes="The original development test video",
    ),
    # Add more here as the test set grows. Files must live in eval/videos/.
    # EvalVideo(
    #     filename="fashion_reel.mp4",
    #     label="Fashion try-on reel",
    #     expected_style="UGC fashion",
    #     recommended_preset="vlog_casual",
    #     min_expected_sfx=3,
    #     max_expected_sfx=10,
    #     has_speech=False,
    # ),
]


def available_eval_videos() -> list[EvalVideo]:
    """Return only eval videos whose files actually exist on disk."""
    return [v for v in EVAL_SET if (EVAL_VIDEOS_DIR / v.filename).exists()]
