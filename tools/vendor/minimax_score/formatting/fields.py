"""Shared field lists and task normalization for base/ref output formats.

Lives in ``formatting`` (not ``scoring``) so both scoring and postprocessing
can depend on it without import cycles: scoring -> formatting is the allowed
direction.
"""

from __future__ import annotations

BASE_FIELDS = [
    "integrated_multimodal_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
]
REF_FIELDS = [
    "subject_definitions:",
    "summary:",
    "retention_analysis:",
    "detailed_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
]


def normalize_task(task: str) -> str:
    t = (task or "T2VA").strip()
    if t in {"T2VA", "I2VA", "FL2VA", "L2VA"}:
        return t
    if t.startswith("full-reference"):
        return "ref"
    if any(
        x in t
        for x in (
            "reference_generation",
            "keyframe",
            "video_editing",
            "video_continuation",
            "audio_",
        )
    ):
        return "ref"
    return t


def is_base(task: str) -> bool:
    return normalize_task(task) in {"T2VA", "I2VA", "FL2VA", "L2VA"}
