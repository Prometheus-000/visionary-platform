"""Generation postprocessing: trim chat markers, section loops, structural fixes.

Structural repairs only (deterministic guide rules) — no invented scene prose.
"""

from __future__ import annotations

import re

from .fields import BASE_FIELDS, REF_FIELDS, is_base, normalize_task
from .instructions import enforce_instruction_line
from .timestamps import enforce_shot_timestamps


_CITED_SHOTS_RE = re.compile(r"\s*\(appears in ([^)]*\[Shot \d+\][^)]*)\)")
_SHOT_TOKEN_RE = re.compile(r"\[Shot (\d+)\]")
_DESC_HEADERS = ("detailed_description:", "integrated_multimodal_description:")


def _description_body_span(text: str, header: str) -> tuple[int, int] | None:
    h = text.find(header)
    if h < 0:
        return None
    start = h + len(header)
    end = text.find("overall_soundscape:", start)
    if end < 0:
        end = len(text)
    return start, end


def ensure_trailing_audio_fields(text: str, *, base: bool) -> str:
    """Append missing overall_soundscape / non_diegetic_music with N/A.

    Models sometimes EOS right after the description (seen on long FL2VA).
    Headers are required by the guide; N/A is the guide-legal empty value
    for music and total silence, so this is structural — not scene invention.
    """
    t = text.rstrip()
    # Only repair when a description section exists (otherwise empty/garbage).
    if "integrated_multimodal_description:" not in t and "detailed_description:" not in t:
        return t
    sep = " " if base else "\n"
    # Append missing fields in guide order after whatever we already have.
    if "overall_soundscape:" not in t:
        t = t + "\n\noverall_soundscape:" + sep + "N/A"
    if "non_diegetic_music:" not in t:
        t = t + "\n\nnon_diegetic_music:" + sep + "N/A"
    return t


def ensure_shot1_header(text: str) -> str:
    """If a description body has content but no [Shot 1], prepend the marker.

    Guide requires an opening shot section; 350M models sometimes write only
    prose under detailed_description. Injecting the marker is structural —
    it does not invent visual content.
    """
    t = text
    for header in _DESC_HEADERS:
        span = _description_body_span(t, header)
        if span is None:
            continue
        start, end = span
        body = t[start:end]
        if not body.strip() or "[Shot 1]" in body:
            continue
        t = t[:start] + " [Shot 1] " + body.lstrip() + t[end:]
        break
    return t


def _align_retention_citations(t: str) -> str:
    """Drop retention citations of shots that have no section in the description.

    The guide only allows retention_analysis to cite shot numbers that exist
    in detailed_description. Phantom [Shot 2] citations trip the whole-text
    timestamp scorer even when the description is legitimately single-shot.
    """
    if "retention_analysis:" not in t or "detailed_description:" not in t:
        return t
    desc = t.split("detailed_description:", 1)[1]
    real = set(
        re.findall(r"\[Shot (\d+)\]", desc.split("overall_soundscape:", 1)[0])
    )

    head, rest = t.split("retention_analysis:", 1)
    if "detailed_description:" not in rest:
        # Sections out of order (DD before retention) — leave text unchanged;
        # the format scorer will penalize the ordering itself.
        return t
    ret, desc_part = rest.split("detailed_description:", 1)

    if not real:
        # No shot sections in DD — strip all shot tokens from retention so
        # format scoring is not poisoned by phantom multi-shot claims.
        ret2 = _CITED_SHOTS_RE.sub("", ret)
        ret2 = _SHOT_TOKEN_RE.sub("", ret2)
        return head + "retention_analysis:" + ret2 + "detailed_description:" + desc_part

    def fix_appears(m: re.Match) -> str:
        cited = re.findall(r"\[Shot (\d+)\]", m.group(1))
        kept = [n for n in cited if n in real]
        if not kept:
            return ""
        if kept == cited:
            return m.group(0)
        return " (appears in " + ", ".join(f"[Shot {n}]" for n in kept) + ")"

    ret2 = _CITED_SHOTS_RE.sub(fix_appears, ret)

    def bare_shot(m: re.Match) -> str:
        return m.group(0) if m.group(1) in real else ""

    ret2 = _SHOT_TOKEN_RE.sub(bare_shot, ret2)
    return head + "retention_analysis:" + ret2 + "detailed_description:" + desc_part


def postprocess_generation(text: str, task: str, duration: float | None = None) -> str:
    t = (text or "").strip()
    # Reasoning-model leakage: keep only the answer after a closed think block,
    # and drop a dangling opener if the block never closed.
    if "</think>" in t:
        t = t.rsplit("</think>", 1)[1].strip()
    if t.startswith("<think>"):
        t = t[len("<think>") :].lstrip()
    for stop in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        if stop in t:
            t = t.split(stop, 1)[0].strip()

    task_n = normalize_task(task)
    fields = BASE_FIELDS if is_base(task_n) else REF_FIELDS

    # Field headers are fixed lowercase tokens; models occasionally emit
    # sentence-cased variants ("Overall_soundscape:") after a paragraph break.
    for f in fields:
        t = re.sub(rf"(?im)^[ \t]*{re.escape(f)}", f, t)

    # If a field header appears twice, keep only the first complete document.
    for f in fields:
        first = t.find(f)
        if first < 0:
            continue
        second = t.find(f, first + len(f))
        if second > 0:
            t = t[:second].rstrip()
            break

    # Trim anything after the first paragraph of non_diegetic_music.
    if "non_diegetic_music:" in t:
        head, tail = t.split("non_diegetic_music:", 1)
        music_body = tail.strip().split("\n\n")[0].strip()
        for f in fields:
            if f in music_body:
                music_body = music_body.split(f, 1)[0].strip()
        sep = " " if is_base(task_n) else "\n"
        t = head + "non_diegetic_music:" + sep + music_body

    # The alignment instruction is deterministic boilerplate; rebuild it
    # rather than trusting the model to reproduce it verbatim.
    if duration is not None and is_base(task_n):
        t = enforce_instruction_line(task_n, t.strip(), duration)

    # Structural shot markers before timestamp repair / retention cleanup.
    t = ensure_shot1_header(t)

    # Timestamp grammar is fully specified by the guide; repair missing or
    # non-increasing "At MM:SS.mmm" stamps by interpolation.
    if duration is not None:
        t = enforce_shot_timestamps(t, duration)

    if not is_base(task_n):
        t = _align_retention_citations(t)

    # Fill trailing sound/music headers when the model stops after the body
    # (common on long keyframe FL2VA). Uses N/A — not invented ambience.
    t = ensure_trailing_audio_fields(t, base=is_base(task_n))

    return t.strip()
