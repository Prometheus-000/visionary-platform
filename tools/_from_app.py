"""
Pull plain-Python pieces out of app.py without importing it.

Shared by `smoke_prompt.py` and `preview_ui.py`, which both need the real thing
rather than a copy of it: a compiler checked against a reimplementation checks
the reimplementation, and a UI preview built against a hand-written vocabulary
is a preview of a palette that does not exist.

Importing app.py is what this avoids, for the reason `preview_ui.py` already
records — it pulls in modal and builds image definitions at module scope, so it
wants credentials and a network to answer a question about a string. The AST is
already on disk, and a module assembled from a subset of its top-level
statements runs on a laptop with no torch, no modal and no credentials.

The subset is named, never pattern-matched. A rename upstream should fail here
loudly rather than quietly hand back one function fewer.
"""

import ast
import hashlib
import json
import math
import re
import time
import typing
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app.py"


def _named(node: ast.stmt) -> str:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.AnnAssign):
        return getattr(node.target, "id", "")
    if isinstance(node, ast.Assign):
        return getattr(node.targets[0], "id", "")
    return ""


def pull(names: set[str]) -> dict:
    """Execute just those top-level definitions, and return the namespace."""
    tree = ast.parse(APP.read_text())
    body = [n for n in tree.body if _named(n) in names]
    missing = names - {_named(n) for n in body}
    if missing:
        raise SystemExit(f"not in app.py any more: {', '.join(sorted(missing))}")
    # `Any` because the annotations are evaluated: app.py deliberately does not
    # use `from __future__ import annotations` — see the note at the top of it.
    # The rest are app.py's own module-level imports, seeded rather than pulled:
    # an `import` statement is not a named definition, and a subset that had to
    # list them would be a subset that breaks when a function starts using one
    # it did not before. PIL is deliberately absent — the pulled code imports it
    # inside the functions that need it, so a caller with no Pillow can still
    # pull them and only pays when it calls one.
    ns: dict = {"Any": typing.Any, "Path": Path, "json": json,
                "hashlib": hashlib, "math": math, "re": re, "time": time}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(APP), "exec"), ns)
    return ns


# Everything the shot compiler is built out of, in one place because both
# callers want all of it.
SHOT = {
    "H3_ALIGN", "H3_LANGUAGES", "SHOT_VALUE_MAX", "SHOT_REF_ROLES",
    "SHOT_VOCAB", "SHOT_ITEMS",
    "_validate_shot", "_validate_ref_roles", "_h3_task",
    "_shot_phrases", "_shot_text", "_shot_sentence", "_close", "_oneline",
    "_shot_join",
    "_first_sentence", "_shot_audio", "_shot_body",
    "_compile_h3_prompt", "_compile_image_prompt", "_compile_wan_prompt",
    # The scene. Pulled with the rest of the compiler rather than as its own
    # set, because `_compile_h3_prompt` delegates to it — a subset that had one
    # and not the other would be a compiler that raises `NameError` on the only
    # input this feature exists for.
    "MAX_CAST", "MAX_SCENE_SHOTS", "MAX_SCENE_LINE",
    "H3_RETENTION", "H3_AUDIO_RETENTION", "H3_TASK_TYPES", "H3_VIDEO_ROLES",
    "H3_AUDIO_ROLES", "H3_AUDIO_NOUN", "H3_SOURCES", "MAX_H3_PROMPT",
    # Blocking. Pulled with the compiler because `_h3_shot_text` calls into
    # it — the rule the MODULES comment records, one edit or a NameError.
    "STAGE_SENSOR_W", "STAGE_SENSOR_H", "STAGE_LENS", "STAGE_FIGURE_H",
    "STAGE_EYE_H", "STAGE_EYE_RATIO", "STAGE_MAX_MARKS", "STAGE_SIZE",
    "STAGE_ANGLE",
    "STAGE_FACING", "STAGE_NEAR", "STAGE_OWN_BODY", "_stage_norm", "_stage_band", "_stage_fov",
    "_stage_dims", "_stage_see", "_stage_where", "_stage_pills", "_stage_move",
    "_stage_others", "_stage_lead", "_stage_read", "_stage_end",
    "_stage_phrase", "_stage_arc",
    "_stage_move_note", "_stage_move_sentence", "STAGE_VERB",
    "_stage_clauses", "_validate_stage", "_stage_boxes", "STAGE_FIGURE_W",
    "H3_CAST_KINDS", "H3_SLOT_MEDIA", "H3_SOUNDSCAPE_DEFAULT", "_H3_HANDLE",
    "_shot_groups", "_h3_clock", "_validate_scene", "_h3_handle",
    "_h3_handles", "_h3_subjects", "_h3_label", "_h3_speakers",
    "_h3_resolve", "_h3_task_types", "_h3_shot_text", "_compile_h3_scene",
    "_h3_list", "_h3_asset", "_h3_cap", "_h3_across",
}

# The captioner's two menus, for the same reason the vocabulary is here: the
# labels and the notes are what the preview draws, and a hand-written copy of
# them is a picker whose options `/api/caption` would reject by name.
CAPTION = {
    "CAPTION_MODELS", "CAPTION_PRESETS",
    "DEFAULT_CAPTION_MODEL", "DEFAULT_CAPTION_PRESET",
}

# The rewrite operations, for `preview_ui.py`'s `/api/state` stub. Pulled rather
# than transcribed for the reason the shot vocabulary is: a stub that omits them
# renders no button at all, so the one surface this feature has is invisible in
# the preview that exists to make the front end developable without a GPU.
# `KREA_EXPANSION` comes too, and its absence is the one failure this file has:
# the subset executes top-level statements in order, so `REWRITE_OPS` naming a
# constant nobody pulled raises `NameError` from inside app.py — which reads as
# a broken preview rather than an incomplete pull.
REWRITE = {"REWRITE_OPS", "KREA_EXPANSION"}

# The motion panel's sections and its parser, for the same two callers the
# rewrite set has: `preview_ui.py` serves `motion_groups` and stubs
# `/api/motion` through the real parser, and `smoke_rewrite.py` asserts the
# instruction budget and the parse cases. `_THINK` and `_oneline` ride along
# because `_parse_motion` calls them — the rule the MODULES comment records:
# a constant a pulled function names has to be pulled in the same edit.
MOTION = {
    "MOTION_GROUPS", "MOTION_MAX_PER", "MOTION_PHRASE_MAX", "MOTION_TOKENS",
    "_MOTION_HEAD_IMAGE", "_MOTION_HEAD_TEXT", "_MOTION_LINE",
    "_motion_instruction", "_parse_motion", "_THINK", "_oneline",
}

# What each video model can be asked for. Pulled rather than transcribed for
# the reason every other set here is, and because this one has already drifted
# once: `preview_ui.py` hand-wrote `supports` and went on rendering H3 with no
# LoRA button for a release after the backend grew one. A stub that omits a
# control is a preview of a control that does not exist.
#
# `requires` and `defaults` name six other constants, and they come too — the
# rule the REWRITE set records: a constant a pulled definition names has to be
# pulled in the same edit, or app.py raises NameError from inside the subset.
VIDEO = {
    "VIDEO_MODELS", "VIDEO_MODEL_KEYS", "VIDEO_REF_MODEL_KEYS",
    "WAN_MODEL_KEYS", "WAN_DEFAULT_STEPS", "WAN_DEFAULT_CFG",
    "WAN_DEFAULT_SHIFT",
}

# The trainer's three menus and its dial defaults, for the same reason again:
# the session form builds itself out of these, and a hand-written copy here
# would be a form offering an optimizer `/api/sessions` rejects by name — or
# worse, one it accepts and the GPU container dies on.
TRAINER = {
    "TRAIN_OPTIMIZERS", "LR_SCHEDULERS", "TIMESTEP_SAMPLINGS", "TRAIN_DEFAULTS",
}

# The storyline validator, for `preview_ui.py`'s `/api/parse` stub. The stub
# invents which words are the model's — it has no model — but it must not invent
# the *offsets*, because those are the thing the mirror is aimed by and a
# hand-written pair would let the preview underline correctly while the shipped
# code underlined three characters left.
MODULES = {
    "MODULE_ROLES", "MAX_MODULES", "MODULE_TEXT_MAX", "MAX_MODULE_DEPTH",
    "MAX_SPANS", "_flat", "_oneline", "_spans_to_text", "_validate_modules",
    "_prominence", "_module_words",
    # The join, because `/api/parse` answers with the prose its elements add up
    # to and the page puts *that* in the box. Without these the stub raises a
    # NameError from inside the real compiler — which reads as a broken preview
    # rather than an incomplete pull, and is the whole failure mode this
    # AST-based lift has: a function arrives without what it calls.
    #
    # **Twice in one session, so it is worth the rule rather than the anecdote:
    # a module-level constant added beside a pulled function has to be added
    # here in the same edit.** `_CONTINUES` and `_ORIGIN_JOINS` were both born
    # that way, and the second one cost a GPU run — ten Sandbox parses that all
    # died on `name '_CONTINUES' is not defined`, minutes after the model had
    # loaded and answered. The name in the traceback is the fix; what makes it
    # expensive is that nothing fails until something rents a card.
    "_module_texts", "_module_clause", "_CONTINUES",
    # The trust checks, so the preview refuses what the deployment refuses.
    "_document_trust", "_trusted_modules", "_derived_from", "_ORIGIN_JOINS",
    "_derived_runs", "_walk_document", "_preserved",
    "_merge_document", "_swap_element",
}

# Duplicate grouping, whole. `smoke_dupes.py` checks the real hash against real
# re-encodes, and `preview_ui.py` groups the preview's own fixtures with it —
# a stub that hand-wrote its groups would be a preview of an arrangement the
# server never produces, which is the one thing this file exists to prevent.
DUPES = {
    "IMAGE_EXTS", "THUMB_DIR", "FINGERPRINT_FILE", "FINGERPRINT_VERSION",
    "SCAN_BUDGET_S",
    "DUPLICATE_MATCH", "SIMILAR_MATCH", "CROP_SHARES", "CROP_MATCH",
    "_FORMAT_RANK",
    "_upright", "_dataset_images", "_caption_of",
    "_dhash", "_phash", "_sharpness", "_crop_variants",
    "_fingerprint", "_fingerprints", "_link", "_components",
    "_keep_rank", "_keep_reason", "_duplicate_groups",
}
