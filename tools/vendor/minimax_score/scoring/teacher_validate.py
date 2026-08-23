"""Structural validation for teacher SFT records (base + full-ref).

Stricter than :mod:`format_score` (which scores model generations): teacher
gold must additionally keep retention lines for every defined label and
balanced dialogue tags. A gold row that fails here must be fixed by the
teacher, never auto-repaired from templates.
"""

from __future__ import annotations

import re

from ..formatting.instructions import is_canonical_instruction
from ..formatting.meta_leak import has_instruction_leak
from ..formatting.ref_repair import BARE_RET_LINE

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


def validate_record(rec: dict) -> list[str]:
    errs: list[str] = []
    msgs = rec.get("messages") or []
    if len(msgs) != 3:
        return ["messages must be [system, user, assistant]"]
    for i, role in enumerate(("system", "user", "assistant")):
        if msgs[i].get("role") != role:
            errs.append(f"messages[{i}] role should be {role}")
        if not (msgs[i].get("content") or "").strip():
            errs.append(f"messages[{i}] empty content")

    out = msgs[2]["content"] if len(msgs) > 2 else ""
    mode = rec.get("mode")
    task = rec.get("task") or ""

    if mode == "base":
        positions = []
        for f in BASE_FIELDS:
            p = out.find(f)
            if p < 0:
                errs.append(f"missing {f}")
            else:
                positions.append(p)
        if len(positions) == len(BASE_FIELDS) and positions != sorted(positions):
            errs.append("base fields out of order")
        if task in {"I2VA", "FL2VA", "L2VA"}:
            first_line = out.strip().split("\n", 1)[0].strip()
            if not is_canonical_instruction(task, first_line):
                errs.append(
                    f"{task} alignment instruction is not the canonical guide line"
                )
        if "[Shot 1]" not in out:
            errs.append("missing [Shot 1]")
    elif mode == "ref":
        positions = []
        for f in REF_FIELDS:
            p = out.find(f)
            if p < 0:
                errs.append(f"missing {f}")
            else:
                positions.append(p)
        if len(positions) == len(REF_FIELDS) and positions != sorted(positions):
            errs.append("ref fields out of order")
        if "summary:" in out:
            sum_part = out.split("summary:", 1)[1].split("retention_analysis:", 1)[0]
            if not re.search(r"\[[^\]]+\]", sum_part):
                errs.append("summary missing [task type] prefix")
        if "[Shot 1]" not in out:
            errs.append("missing [Shot 1]")
        if all(f in out for f in ("subject_definitions:", "summary:", "retention_analysis:")):
            def_block = out.split("subject_definitions:", 1)[1].split("summary:", 1)[0]
            ret_block = out.split("retention_analysis:", 1)[1].split(
                "detailed_description:", 1
            )[0]
            for line in def_block.splitlines():
                m = re.match(r"<(Subject|Picture|Video|Audio) \d+>", line.strip())
                if m:
                    lab = m.group(0)
                    if not any(rl.strip().startswith(lab) for rl in ret_block.splitlines()):
                        errs.append(f"retention missing line for {lab}")
            for rl in ret_block.splitlines():
                if BARE_RET_LINE.match(rl.strip()):
                    errs.append(
                        f"bare retention line (needs ': marker - explanation'): {rl.strip()}"
                    )
    else:
        errs.append("mode must be base or ref")

    if out.count("<d>") != out.count("</d>"):
        errs.append("unbalanced <d> tags")
    if "[Shot 2]" in out and not re.search(r"At \d{2}:\d{2}\.\d{3}", out):
        errs.append("multi-shot missing At MM:SS.mmm timestamp")
    if has_instruction_leak(out):
        errs.append("instruction/rule language leaked into assistant body")

    return errs
