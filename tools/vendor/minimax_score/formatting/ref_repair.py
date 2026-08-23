"""Mechanical repairs for full-reference gold that drifted from the ref guide.

Two known teacher drifts (gold_04..06 era):

1. retention_analysis lines shaped "<Subject 1> fully_preserved" — missing the
   guide's "<Label> (appears in [Shot ...]): marker - explanation" form. The
   repair keeps the label + marker and derives the explanation from that
   label's own subject_definitions entry (content-derived, not templated).

2. video_editing summaries missing the mandated opener sentence
   "The target video is an edited version of <Video 1>." right after the
   [task type] prefix.
"""

from __future__ import annotations

import re

BARE_RET_LINE = re.compile(
    r"^(?P<label><(?:Subject|Picture|Video|Audio) \d+>)\s+"
    r"(?P<marker>fully_preserved|partially_preserved|attribute_transfer|"
    r"weak_reference|fully_copy|partially_copy|reference)\s*$"
)

_MARKER_PHRASE = {
    "fully_preserved": "{desc} is retained in the target video",
    "partially_preserved": "{desc} is used with some defined characteristics changed",
    "attribute_transfer": "characteristics of {desc} are transferred to the target subject",
    "weak_reference": "only broad similarity to {desc} is retained",
    "fully_copy": "the audio of {desc} is reused as the target video's final audio track",
    "partially_copy": "part of the audio timeline of {desc} is copied into the target video",
    "reference": "only the audible characteristics of {desc} are referenced without copying the signal",
}


def _label_description(label: str, subject_definitions: str) -> str:
    """First clause of the label's definition line, e.g. 'the young woman ...'."""
    for line in subject_definitions.splitlines():
        line = line.strip()
        if line.startswith(label):
            rest = line[len(label):].lstrip()
            rest = re.sub(r"^(is|are)\s+", "", rest)
            clause = re.split(r"[.;]", rest, 1)[0].strip().rstrip(",")
            if clause:
                return clause
    return "this reference"


def _shots_for_label(label: str, detailed_description: str) -> list[int]:
    """Shot numbers whose section mentions the label."""
    parts = re.split(r"(\[Shot (\d+)\])", detailed_description)
    shots: list[int] = []
    current = None
    for chunk in parts:
        m = re.match(r"\[Shot (\d+)\]$", chunk or "")
        if m:
            current = int(m.group(1))
        elif current is not None and label in (chunk or ""):
            if current not in shots:
                shots.append(current)
    return shots


def repair_retention_lines(output: str) -> str:
    if "retention_analysis:" not in output:
        return output
    head, rest = output.split("retention_analysis:", 1)
    if "detailed_description:" in rest:
        ret_block, tail = rest.split("detailed_description:", 1)
        tail = "detailed_description:" + tail
    else:
        ret_block, tail = rest, ""
    subj_defs = head.split("subject_definitions:", 1)[-1]
    detailed = tail

    fixed_lines = []
    for line in ret_block.split("\n"):
        m = BARE_RET_LINE.match(line.strip())
        if not m:
            fixed_lines.append(line)
            continue
        label, marker = m.group("label"), m.group("marker")
        desc = _label_description(label, subj_defs)
        phrase = _MARKER_PHRASE[marker].format(desc=desc)
        appears = ""
        if label.startswith("<Subject"):
            shots = _shots_for_label(label, detailed)
            if shots:
                appears = " (appears in " + ", ".join(f"[Shot {n}]" for n in shots) + ")"
        fixed_lines.append(f"{label}{appears}: {marker} - {phrase}.")

    return head + "retention_analysis:" + "\n".join(fixed_lines) + tail


VIDEO_EDIT_OPENER = "The target video is an edited version of <Video 1>."


def repair_video_editing_opener(output: str, task: str) -> str:
    if not task.startswith("video_editing"):
        return output
    if "an edited version of <Video 1>" in output:
        return output
    m = re.search(r"(summary:\s*\n?\s*\[[^\]]+\])\s*", output)
    if not m:
        return output
    return output[: m.end(1)] + f" {VIDEO_EDIT_OPENER}" + output[m.end(1):]
