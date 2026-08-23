"""MiniMax structure scoring for base + full-reference rewrites.

pass = every critical check true. ``score`` is the fraction of critical checks
passed (soft signal; a high mean score with low pass rate means outputs are
one brittle check away from passing).

Changes vs the v1 scorer:
  - ``has_reference_label`` accepts any of <Subject N> / <Picture N> /
    <Video N> / <Audio N>. The v1 ``has_subject_label`` required <Subject N>
    and wrongly failed e.g. video continuations that only track <Video 1>.
  - every [Shot N] with N >= 2 must carry an "At MM:SS.mmm" timestamp
    (v1 only checked Shot 2).
"""

from __future__ import annotations

import re
from typing import Any

from ..formatting.fields import (  # noqa: F401  (re-exported for compat)
    BASE_FIELDS,
    REF_FIELDS,
    is_base,
    normalize_task,
)
from ..formatting.instructions import FL2VA_RE, I2VA_INSTRUCTION, L2VA_RE
from ..formatting.meta_leak import has_instruction_leak

REFERENCE_LABEL_RE = re.compile(r"<(?:Subject|Picture|Video|Audio)\s+\d+>")


def field_order_ok(text: str, fields: list[str]) -> bool:
    positions = []
    for f in fields:
        p = text.find(f)
        if p < 0:
            return False
        positions.append(p)
    return positions == sorted(positions)


def count_section_repeats(text: str, fields: list[str]) -> int:
    return sum(1 for f in fields if text.count(f) > 1)


def shots_have_timestamps(text: str) -> bool:
    """Every shot number N >= 2 needs a timestamped header "[Shot N] At MM:SS.mmm".

    Checked per shot number, not per occurrence: sections like
    retention_analysis legitimately cross-reference shots as "([Shot 1], [Shot 2])"
    without timestamps.
    """
    shot_numbers = {int(n) for n in re.findall(r"\[Shot (\d+)\]", text)}
    for n in shot_numbers:
        if n < 2:
            continue
        if not re.search(rf"\[Shot {n}\]\s+At\s+\d{{2}}:\d{{2}}\.\d{{3}}", text):
            return False
    return True


def score_format(text: str, task: str) -> dict[str, Any]:
    """Return pass/fail + checklist for one generation."""
    text = (text or "").strip()
    task_n = normalize_task(task)
    checks: dict[str, bool] = {}
    notes: list[str] = []

    if not text:
        return {
            "pass": False,
            "score": 0.0,
            "task": task_n,
            "checks": {"nonempty": False},
            "notes": ["empty generation"],
        }

    checks["nonempty"] = True
    checks["has_shot1"] = "[Shot 1]" in text
    # MiniMax H3 prompt field hard limit: 7,000 characters incl. whitespace.
    checks["within_h3_char_limit"] = len(text) <= 7000
    # Reject system-rule paraphrases dumped into the scene body. Structural
    # field checks alone green-lit this failure mode (T2VA pass with meta text).
    checks["no_instruction_leak"] = not has_instruction_leak(text)

    if is_base(task_n):
        fields = BASE_FIELDS
        for f in fields:
            checks[f"field:{f}"] = f in text
        checks["field_order"] = field_order_ok(text, fields)
        checks["no_heavy_loop"] = count_section_repeats(text, fields) == 0

        # Alignment instructions are fixed verbatim by the writing guide
        # (only shot number / duration mark vary), so check the exact line.
        first_line = text.split("\n", 1)[0].strip()
        if task_n == "I2VA":
            checks["i2va_instruction"] = first_line == I2VA_INSTRUCTION
        if task_n == "FL2VA":
            checks["fl2va_alignment"] = bool(FL2VA_RE.match(first_line))
        if task_n == "L2VA":
            checks["l2va_alignment"] = bool(L2VA_RE.match(first_line))

        checks["has_shot2_optional"] = "[Shot 2]" in text
        checks["shots_have_timestamps"] = shots_have_timestamps(text)
        # Shot 1 must not carry a timestamp in base style
        checks["shot1_no_bogus_timestamp"] = not bool(
            re.search(r"\[Shot 1\]\s+At\s+\d{2}:", text)
        )

    else:
        fields = REF_FIELDS
        for f in fields:
            checks[f"field:{f}"] = f in text
        checks["field_order"] = field_order_ok(text, fields)
        checks["no_heavy_loop"] = count_section_repeats(text, fields) == 0

        if "summary:" in text:
            sum_part = text.split("summary:", 1)[1]
            if "retention_analysis:" in sum_part:
                sum_part = sum_part.split("retention_analysis:", 1)[0]
            checks["summary_task_prefix"] = bool(re.search(r"\[[^\]]+\]", sum_part))
        else:
            checks["summary_task_prefix"] = False

        checks["has_reference_label"] = bool(REFERENCE_LABEL_RE.search(text))
        checks["has_shot2_optional"] = "[Shot 2]" in text
        checks["shots_have_timestamps"] = shots_have_timestamps(text)

    critical = [k for k in checks if not k.endswith("_optional")]
    n_ok = sum(1 for k in critical if checks.get(k))
    score = n_ok / max(len(critical), 1)
    passed = all(checks.get(k, False) for k in critical)

    if not checks.get("no_heavy_loop", True):
        notes.append("section headers repeated (possible loop)")
    if not checks.get("has_shot1", True):
        notes.append("missing [Shot 1]")
    if not checks.get("no_instruction_leak", True):
        notes.append("instruction/rule language leaked into scene body")

    return {
        "pass": passed,
        "score": round(score, 3),
        "task": task_n,
        "checks": checks,
        "notes": notes,
        "char_len": len(text),
    }


def default_max_new_tokens(task: str) -> int:
    # The MiniMax H3 prompt field caps at 7,000 characters (~1,800-2,300
    # tokens); budgets leave headroom so deep briefs are never truncated.
    task_n = normalize_task(task)
    if task_n in {"T2VA", "I2VA", "L2VA", "FL2VA"}:
        return 1200
    return 2048
