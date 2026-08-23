"""User/system message construction and parsing.

The user envelope is the contract between training data, inference, and eval:

    Task: <label>
    Duration: <D.DD>s
    Assets:
    - <asset or "(none)">

    User prompt:
    <original rough prompt>

Task labels:
  - base tasks use the task name directly (T2VA / I2VA / FL2VA / L2VA)
  - ref tasks use the fine-grained form "video_editing (full-reference rewrite)"
    so the 350M student can condition on the subtype. The old corpus used the
    generic "full-reference rewrite" label; ``legacy=True`` reproduces it,
    which is needed when evaluating checkpoints trained on the old corpus.
"""

from __future__ import annotations

import html
import re

from ..paths import SYSTEM_BASE_FILE, SYSTEM_BASE_TASK_FILES, SYSTEM_REF_FILE

BASE_TASKS = {"T2VA", "I2VA", "FL2VA", "L2VA"}
REF_TASKS = {
    "reference_generation",
    "reference_generation+audio_reference",
    "keyframe_completion",
    "video_editing",
    "video_editing+audio_reuse",
    "video_continuation",
    "video_continuation+audio_reference",
}

LEGACY_REF_LABEL = "full-reference rewrite"


def is_base_task(task: str) -> bool:
    return task in BASE_TASKS


def task_label(task: str, *, legacy: bool = False) -> str:
    if task in BASE_TASKS:
        return task
    if legacy:
        return LEGACY_REF_LABEL
    return f"{task} ({LEGACY_REF_LABEL})"


def build_user_message(
    task: str,
    duration: float,
    prompt: str,
    assets: list[str] | None = None,
    *,
    legacy: bool = False,
) -> str:
    asset_block = "\n".join(f"- {a}" for a in assets) if assets else "- (none)"
    return (
        f"Task: {task_label(task, legacy=legacy)}\n"
        f"Duration: {float(duration):.2f}s\n"
        f"Assets:\n{asset_block}\n\n"
        f"User prompt:\n{html.unescape(prompt).strip()}"
    )


def load_system(task: str) -> str:
    """Load the system prompt for a task.

    Base tasks use a task-specific file when present (T2VA has no alignment
    rules; I2VA/FL2VA/L2VA only their own line) to cut cross-task instruction
    leakage. Falls back to ``system_base.txt`` / ``system_ref.txt``.
    """
    if task in BASE_TASKS:
        path = SYSTEM_BASE_TASK_FILES.get(task, SYSTEM_BASE_FILE)
        if not path.is_file():
            path = SYSTEM_BASE_FILE
    else:
        path = SYSTEM_REF_FILE
    return path.read_text(encoding="utf-8").strip()


def parse_user_blob(user_content: str) -> dict:
    """Extract task, duration, assets, original prompt from a packaged user message.

    Handles both the fine-grained label ("video_editing (full-reference rewrite)")
    and the legacy generic label ("full-reference rewrite").
    """
    task = "T2VA"
    duration = 6.0
    assets: list[str] = []
    prompt = user_content

    m = re.search(r"Task:\s*(.+)", user_content)
    if m:
        raw = m.group(1).strip()
        if raw == LEGACY_REF_LABEL:
            task = "full-reference"
        else:
            fm = re.match(r"(.+?)\s*\(full-reference rewrite\)$", raw)
            task = fm.group(1).strip() if fm else raw
    m = re.search(r"Duration:\s*([0-9.]+)", user_content)
    if m:
        duration = float(m.group(1))
    if "Assets:" in user_content and "User prompt:" in user_content:
        asset_block = user_content.split("Assets:", 1)[1].split("User prompt:", 1)[0]
        for line in asset_block.splitlines():
            line = line.strip()
            if line.startswith("- ") and line not in {"- (none)", "- none"}:
                assets.append(line[2:].strip())
        prompt = user_content.split("User prompt:", 1)[1].strip()

    return {
        "task": task,
        "duration": duration,
        "assets": assets or None,
        "prompt": prompt,
    }
