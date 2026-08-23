"""Detect system-prompt instruction leakage in generations.

A small student can paraphrase system rules into the scene body. Detection is
used by the format scorer (and teacher validation) so those outputs fail
honestly. Content is *not* auto-rewritten here — fix via training + systems.
"""

from __future__ import annotations

import re

# Phrases that only appear when the model is narrating the *format rules*
# rather than describing the video.
_LEAK_RES: list[re.Pattern[str]] = [
    re.compile(r"(?i)\brequired alignment instruction\b"),
    re.compile(r"(?i)\bbegin(?:s|ning)? with the required\b"),
    re.compile(r"(?i)\bintegrated_multimodal_description begins\b"),
    re.compile(r"(?i)\bdetailed_description begins\b"),
    re.compile(r"(?i)\bfills in a full story\b"),
    re.compile(r"(?i)\bfrom the referenced picture\b"),
    re.compile(r"(?i)\bthree core fields\b"),
    re.compile(r"(?i)\bas the first line,? then one blank line\b"),
    re.compile(r"(?i)\bwrites? the body in english\b"),
    re.compile(r"(?i)\bdo not invent timestamps\b"),
    re.compile(r"(?i)\boutput rules?\b"),
]


def has_instruction_leak(text: str) -> bool:
    """True if any known instruction-leak phrase appears in the generation."""
    t = text or ""
    return any(rx.search(t) for rx in _LEAK_RES)
