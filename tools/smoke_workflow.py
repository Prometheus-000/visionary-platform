"""
The Playground's pure halves, on a laptop: the graph validator and the toggle.

    python3 tools/smoke_workflow.py

Pulled out of app.py by AST like every other smoke, because a test against a
reimplementation tests the reimplementation. No Modal, no torch, no network —
`_validate_playground_graph` and `_apply_workflow` are arithmetic, which is
exactly why they can gate a request.

What is asserted is the *contract*, message text included where the message is
the feature: every refusal here is one a person hits from the editor or the
model menu, and a refusal that stops naming the fix is a regression even when
the return value is right.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _from_app import pull  # noqa: E402

ns = pull({"_validate_playground_graph", "_apply_workflow",
           "_workflow_name", "_WORKFLOW_NAME_RE"})
validate = ns["_validate_playground_graph"]
apply_wf = ns["_apply_workflow"]
wf_name = ns["_workflow_name"]

fails: list[str] = []


def refuses(fn, *args, saying: str = "", why: str = "") -> None:
    try:
        fn(*args)
    except ValueError as exc:
        if saying and saying not in str(exc):
            fails.append(f"{why}: refused, but the message lost "
                         f"{saying!r}: {exc}")
        return
    fails.append(f"{why}: accepted")


GOOD = {
    "1": {"class_type": "CLIPTextEncode",
          "inputs": {"text": "hello", "clip": ["2", 0]}},
    "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "x"}},
}

# ── the validator ──────────────────────────────────────────────────────────
if validate(dict(GOOD)) != GOOD:
    fails.append("a valid graph does not come back unchanged")

refuses(validate, [], saying="non-empty object",
        why="a list is not a graph")
refuses(validate, {"1": {"inputs": {}}}, saying="UI-format",
        why="a node without class_type is the UI export, and the error "
            "must say so")
refuses(validate, {"1": {"class_type": "X",
                         "inputs": {"a": ["9", 0]}}},
        saying="not in the graph", why="a link to a missing node")
refuses(validate, {"1": {"class_type": "X",
                         "inputs": {"a": [1, 2, 3]}}},
        saying="neither a link nor a value", why="a malformed link")
refuses(validate, {"1": {"class_type": "Fancy", "inputs": {}}},
        {"Other"}, saying="Unknown node type",
        why="a class_type the catalogue lacks")
# No catalogue, no catalogue check — a fresh install must not refuse
# every run until one is harvested.
validate({"1": {"class_type": "Fancy", "inputs": {}}}, None)

# ── names ──────────────────────────────────────────────────────────────────
for bad in ("", "a/b", "../up", ".hidden", "x" * 70):
    refuses(wf_name, bad, why=f"workflow name {bad!r}")
if wf_name("My workflow v2.1") != "My workflow v2.1":
    fails.append("an ordinary name was mangled")

# ── the toggle's substitution ──────────────────────────────────────────────
default = {
    "pos": {"class_type": "CLIPTextEncode",
            "inputs": {"text": "compiled prompt", "clip": ["clip", 0]}},
    "sample": {"class_type": "KSampler",
               "inputs": {"seed": 42, "steps": 8, "model": ["dit", 0]}},
    "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "x"}},
    "dit": {"class_type": "UNETLoader", "inputs": {"unet_name": "y"}},
}
# The user kept pos and sample, replaced the sampler's steps, added a node.
workflow = {
    "pos": {"class_type": "CLIPTextEncode",
            "inputs": {"text": "stale saved prompt", "clip": ["clip", 0]}},
    "sample": {"class_type": "KSampler",
               "inputs": {"seed": 7, "steps": 30, "model": ["up", 0]}},
    "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "x"}},
    "dit": {"class_type": "UNETLoader", "inputs": {"unet_name": "y"}},
    "up": {"class_type": "Upscaler", "inputs": {"model": ["dit", 0],
                                                "scale": 2.0}},
}
out = apply_wf(default, workflow, {"up": {"scale": 4.0}})

if out["pos"]["inputs"]["text"] != "compiled prompt":
    fails.append("the console's prompt did not reach the inherited encoder")
if out["sample"]["inputs"]["seed"] != 42 or out["sample"]["inputs"]["steps"] != 8:
    fails.append("the console's sampler values did not reach the "
                 "inherited sampler")
if out["sample"]["inputs"]["model"] != ["up", 0]:
    fails.append("substitution rewired the user's graph — links must be "
                 "the workflow's own")
if out["up"]["inputs"]["scale"] != 4.0:
    fails.append("an exposed extra did not land")
if workflow["sample"]["inputs"]["seed"] != 7:
    fails.append("_apply_workflow mutated its input — the caller's copy "
                 "stopped being theirs")

refuses(apply_wf, default,
        {"z": {"class_type": "Nothing", "inputs": {}}},
        saying="inherited nothing", why="a workflow with no inherited node")
refuses(apply_wf, default, workflow, {"nope": {"x": 1}},
        saying="not in the workflow", why="an extra naming a missing node")
refuses(apply_wf, default, workflow, {"up": {"nope": 1}},
        saying="not an input", why="an extra naming a missing input")
refuses(apply_wf, default, workflow, {"up": {"model": 1}},
        saying="wired", why="an extra aimed at a wired input")

for f in fails:
    print(f"  FAIL  {f}")
print(f"\n{'PASS' if not fails else str(len(fails)) + ' FAILED'}")
raise SystemExit(1 if fails else 0)
