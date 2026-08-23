"""Canonical alignment-instruction lines from the official writing guides.

VIDEO_PROMPT_WRITING_GUIDE_base_en.md fixes these lines verbatim (only the
final shot number N and the duration mark S.SS vary). Teacher batches that
paraphrase them create mixed supervision and destabilize the student's first
line, so packaging/validation must enforce the exact forms:

  I2VA : For the target video, at 0.00 seconds into the target video,
         <Picture 1> (from [Shot 1]) is fully referenced.
  FL2VA: How the reference pictures align with the target video — Picture 1
         (from Shot 1) aligns with the 0.00-second mark of the target video;
         Picture 2 (from Shot N) aligns with the S.SS-second mark of the
         target video.
  L2VA : How the reference pictures align with the target video —
         <Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the
         target video.

Note the asymmetry from the guide: FL2VA uses bare "Picture 1 (from Shot 1)"
while L2VA uses bracketed "<Picture 1> (from [Shot N])".
"""

from __future__ import annotations

import re

I2VA_INSTRUCTION = (
    "For the target video, at 0.00 seconds into the target video, "
    "<Picture 1> (from [Shot 1]) is fully referenced."
)

FL2VA_RE = re.compile(
    r"^How the reference pictures align with the target video — "
    r"Picture 1 \(from Shot 1\) aligns with the 0\.00-second mark of the target video; "
    r"Picture 2 \(from Shot \d+\) aligns with the \d+\.\d{2}-second mark of the target video\.$"
)
L2VA_RE = re.compile(
    r"^How the reference pictures align with the target video — "
    r"<Picture 1> \(from \[Shot \d+\]\) aligns with the \d+\.\d{2}-second mark of the target video\.$"
)


def fl2va_instruction(final_shot: int, duration: float) -> str:
    return (
        "How the reference pictures align with the target video — "
        "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
        f"Picture 2 (from Shot {final_shot}) aligns with the {duration:.2f}-second mark "
        "of the target video."
    )


def l2va_instruction(final_shot: int, duration: float) -> str:
    return (
        "How the reference pictures align with the target video — "
        f"<Picture 1> (from [Shot {final_shot}]) aligns with the {duration:.2f}-second "
        "mark of the target video."
    )


def is_canonical_instruction(task: str, first_line: str) -> bool:
    line = first_line.strip()
    if task == "T2VA":
        return line.startswith("integrated_multimodal_description:")
    if task == "I2VA":
        return line == I2VA_INSTRUCTION
    if task == "FL2VA":
        return bool(FL2VA_RE.match(line))
    if task == "L2VA":
        return bool(L2VA_RE.match(line))
    return True


def final_shot_number(body: str) -> int:
    shots = [int(n) for n in re.findall(r"\[Shot (\d+)\]", body)]
    return max(shots) if shots else 1


def canonical_instruction(task: str, body: str, duration: float) -> str | None:
    """The exact guide line for a base task, or None for T2VA."""
    if task == "I2VA":
        return I2VA_INSTRUCTION
    if task == "FL2VA":
        return fl2va_instruction(final_shot_number(body), duration)
    if task == "L2VA":
        return l2va_instruction(final_shot_number(body), duration)
    return None


def enforce_instruction_line(task: str, text: str, duration: float) -> str:
    """Deterministically (re)build the alignment instruction on a generation.

    A 350M student is unreliable at reproducing long fixed boilerplate (it may
    omit the line, blend two task lines, or invent a Picture 2). Since the
    line is fully determined by task + duration + final shot number, the
    runtime constructs it: keep the body from the first core field onward and
    prepend the canonical line.
    """
    if task not in {"I2VA", "FL2VA", "L2VA"}:
        return text
    idx = text.find("integrated_multimodal_description:")
    if idx < 0:
        return text  # no recognizable body; nothing safe to repair
    body = text[idx:].strip()
    line = canonical_instruction(task, body, duration)
    return f"{line}\n\n{body}"


def repair_base_instruction(task: str, output: str, duration: float) -> str:
    """Replace a paraphrased/missing alignment instruction with the guide line.

    The instruction is fully deterministic given task, duration, and the final
    shot number, so this is a safe mechanical repair (no content invented).
    """
    if task not in {"I2VA", "FL2VA", "L2VA"}:
        return output

    text = output.strip()
    if text.startswith("integrated_multimodal_description:"):
        body = text  # instruction missing entirely
    else:
        parts = text.split("\n", 1)
        body = parts[1].strip() if len(parts) > 1 else ""
        if is_canonical_instruction(task, parts[0]):
            return output

    line = canonical_instruction(task, body, duration)
    return f"{line}\n\n{body}"
