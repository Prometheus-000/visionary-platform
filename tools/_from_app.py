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
import typing
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app.py"


def _named(node: ast.stmt) -> str:
    if isinstance(node, ast.FunctionDef):
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
    ns: dict = {"Any": typing.Any}
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
}
