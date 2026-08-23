"""Deterministic repair of shot timestamps in generated descriptions.

The guide fixes the timestamp grammar completely: [Shot 1] carries no
timestamp, every later shot must open with "At MM:SS.mmm," and times must
increase strictly inside the clip duration. When a generation drops or
garbles a timestamp, the product layer can rebuild it mechanically —
existing valid stamps are kept, missing ones are interpolated evenly
between their known neighbours (0.0 at the start, the duration at the end).

Only the description section is touched, and only its *sequential* shot
markers ([Shot 1] then [Shot 2] then ...) count as section starts; shot
references elsewhere (e.g. retention_analysis "appears in [Shot 2]") are
left alone.
"""

from __future__ import annotations

import re

SHOT_RE = re.compile(r"\[Shot (\d+)\]\s*")
STAMP_RE = re.compile(r"^At (\d{2}):(\d{2}\.\d{3}),?\s*")

_DESC_HEADERS = ("detailed_description:", "integrated_multimodal_description:")
_NEXT_HEADER = "overall_soundscape:"


def _fmt(seconds: float) -> str:
    m = int(seconds // 60)
    return f"{m:02d}:{seconds - m * 60:06.3f}"


def _repair_section(section: str, duration: float) -> str:
    # Sequential shot-section markers only: [Shot 1], then [Shot 2], ...
    marks: list[re.Match] = []
    expected = 1
    for m in SHOT_RE.finditer(section):
        if int(m.group(1)) == expected:
            marks.append(m)
            expected += 1
    if len(marks) < 2:
        return section

    stamps: list[float | None] = []
    spans: list[tuple[int, int]] = []
    for m in marks:
        sm = STAMP_RE.match(section[m.end():])
        if sm:
            stamps.append(int(sm.group(1)) * 60 + float(sm.group(2)))
            spans.append((m.end(), m.end() + sm.end()))
        else:
            stamps.append(None)
            spans.append((m.end(), m.end()))

    # Shot 1 implicitly owns 0.0; keep valid increasing stamps, interpolate
    # anything missing or out of order toward the next trusted value.
    times: list[float] = [0.0] * len(stamps)
    prev = 0.0
    i = 1
    while i < len(stamps):
        s = stamps[i]
        if s is not None and prev < s < duration:
            times[i] = s
            prev = s
            i += 1
            continue
        j = i + 1
        nxt = duration
        while j < len(stamps):
            sj = stamps[j]
            if sj is not None and prev < sj < duration:
                nxt = sj
                break
            j += 1
        gaps = (j - i) + 1
        for k in range(i, j):
            times[k] = prev + (nxt - prev) * (k - i + 1) / gaps
        prev = times[j - 1]
        i = j

    out = section
    for idx in range(len(marks) - 1, 0, -1):
        start, end = spans[idx]
        if stamps[idx] is not None and abs(stamps[idx] - times[idx]) < 0.0005:
            continue
        out = out[:start] + f"At {_fmt(times[idx])}, " + out[end:]
    # Shot 1 must not carry a timestamp.
    s0, e0 = spans[0]
    if stamps[0] is not None:
        out = out[:s0] + out[e0:]
    return out


def enforce_shot_timestamps(text: str, duration: float) -> str:
    """Ensure every sequential [Shot N>=2] opens with a valid increasing stamp."""
    if duration <= 0:
        return text

    for header in _DESC_HEADERS:
        h = text.find(header)
        if h < 0:
            continue
        start = h + len(header)
        end = text.find(_NEXT_HEADER, start)
        if end < 0:
            end = len(text)
        section = text[start:end]
        return text[:start] + _repair_section(section, duration) + text[end:]
    return text
