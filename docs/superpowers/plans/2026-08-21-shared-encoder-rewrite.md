# Shared-Encoder Rewrite — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the prompt rewrite off its own copy of Qwen3-VL-4B and onto the `CLIP` object ComfyUI already holds to condition Krea 2 renders, and turn Enhance from a one-shot that silently drifts on a second press into a reroll that always reads the sentence the person typed.

**Architecture:** `comfy_nodes/visionary_rewrite` stops loading weights and takes a `("CLIP",)` input, driving ComfyUI's own first-party `BaseGenerate` KV-cached decode loop through the public `comfy.sd.CLIP.generate` / `.decode`. One `_rewrite_graph()` builder in `app.py` emits a two-node graph whose `CLIPLoader` is produced by the same function `_krea2_graph` uses, so the render and the rewrite resolve to one cached `CLIP` and one `ModelPatcher`. On the page, a kept `rewriteFrom` string replaces "whatever is in the box" as the rewrite's input, and the decode switches to sampling so a second press is a genuine reroll.

**Tech Stack:** Python 3.11, Modal, ComfyUI pinned at `COMFY_SHA`, React 19 + TypeScript + Zustand + Vite, Playwright for UI checks.

**Spec:** [`docs/design-notes/one-encoder-not-two.md`](../../design-notes/one-encoder-not-two.md) — read it before Task 1. This plan argues from it and does not restate its reasoning.

## Global Constraints

- **No `from __future__ import annotations` in `app.py`.** It breaks FastAPI's `get_type_hints()`. See the note at the top of the file.
- **Comments explain why, not what.** Every non-obvious line earns its comment by naming the failure that produced it. If a comment could be deleted without losing a fact, delete it.
- **Commit straight to `main`.** No branch first, no branch-and-merge.
- **Python is `/opt/homebrew/bin/python3.11`.** The `python3` on `PATH` has no `modal`; a `ModuleNotFoundError` there is the wrong interpreter, not a missing package.
- **Tests are standalone scripts, not pytest.** The convention is a module-level `def check() -> int` returning a failure count, printing `  ok  ` / `  FAIL` lines, summed by `def main() -> int`. Run as `python3.11 tools/smoke_x.py`.
- **Do not deploy between Task 5 and Task 6.** They are two commits and one gate — see the banner on Task 5.
- **The instruction, `REWRITE_OPS`, `_clean_rewrite` and `_rewrite_tokens` are untouched.** Prose in and prose out are unchanged; that is what makes the parity check meaningful.
- Sampling constants, once introduced in Task 6, are `temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, repetition_penalty=1.0, presence_penalty=0.5`. These are Qwen3's own non-thinking guidance plus a presence penalty for the adjective echo `CLAUDE.md` records. They are proposed values; if the author disagrees, change them here before Task 6 rather than during it.

---

## File Structure

| File | Responsibility after this plan |
|---|---|
| `comfy_nodes/visionary_rewrite/__init__.py` | Compose a chat string, drive a supplied `CLIP`, return text. No weights, no `transformers`. ~70 lines. |
| `app.py` | `_krea2_clip_node()` — the one `CLIPLoader` literal. `_rewrite_graph()` — the one rewrite graph. `_rewrite_backend`/`/api/rewrite` gain a seed. `comfy_image` loses the `transformers` bake. |
| `tools/smoke_rewrite.py` | Gains `graph()` and `chat()` checks; docstring corrected. Still offline, still no GPU. |
| `tools/smoke_graphs.py` | Gains two rewrite variants and `VisionaryRewrite` in the node-presence loop. |
| `tools/smoke_tokens.py` | **New.** `modal run`. Asserts ComfyUI's bundled tokenizer produces the ids we expect for the composed template. |
| `tools/tokens-golden.json` | **New.** Committed fixture: the expected id lists, captured once while `transformers` is still in the image. |
| `tools/preview_ui.py` | **New stub** for `/api/rewrite`, so the button the preview already renders can be pressed. |
| `tools/ui-checks/check_rewrite.py` | **New.** Drives the reroll: press twice, assert the second press read the prose and that Undo lands on the typed sentence. |
| `web/src/store.ts` | `rewriteFrom` and the re-basing rule; `applyRewrite` stops overwriting the undo slot. |
| `web/src/console/Rewrite.tsx` | Sends `rewriteFrom`, not the box. |
| `CLAUDE.md` | The "second copy" section rewritten as claim-then-what-retired-it. |

---

## Task 1: The node stops loading weights

**Files:**
- Modify: `comfy_nodes/visionary_rewrite/__init__.py` (whole file)
- Test: `tools/smoke_rewrite.py` (new `chat()` check)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: module-level `chat(instruction: str, prose: str, image: bool = False) -> str`, `VISION_BLOCK: str`, `MAX_FRAME_SIDE: int`, and a `VisionaryRewrite` node whose `INPUT_TYPES` required keys are `clip`, `prose`, `instruction`, `max_tokens` and whose optional keys are `image_b64`, `warm_only`. Task 2 builds a graph against exactly those names; Task 4 imports `chat`.

- [ ] **Step 1: Write the failing test**

Add to `tools/smoke_rewrite.py`, above `def main()`:

```python
def chat() -> int:
    """
    The chat string the node composes, byte for byte.

    Composed here rather than asked of a tokenizer because **no template on the
    Krea 2 path has a system turn**: `Qwen3VLTokenizer.llama_template` opens at
    `<|im_start|>user` and `KREA2_TEMPLATE` is the fixed descriptor prompt the
    DiT conditions on. The instruction has nowhere to go through either. The
    trailing empty think block is what suppresses reasoning — without it the
    model's deliberation lands in the prompt box as the prompt.
    """
    print("chat template")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from comfy_nodes.visionary_rewrite import VISION_BLOCK, chat as compose

    bad = 0
    want = ("<|im_start|>system\nRewrite it.<|im_end|>\n"
            "<|im_start|>user\nempty diner, 3am<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    got = compose("Rewrite it.", "empty diner, 3am")
    ok = got == want
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  the text turn is system + user + no-think")
    if not ok:
        print(f"        want: {want!r}")
        print(f"        got : {got!r}")

    # The vision block opens the *user* turn, before the prose. A bare string
    # never emits the placeholder tokens, and the failure is a model that
    # silently answers without looking at the picture.
    got = compose("Propose motion.", "what moves here", image=True)
    ok = f"<|im_start|>user\n{VISION_BLOCK}what moves here<|im_end|>" in got
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  a frame opens the user turn")

    # Importable with no torch, no PIL and no ComfyUI on the path — which is
    # what lets this run on a laptop and what the module-scope imports have to
    # stay small enough to allow.
    ok = "empty diner" in compose("x", "empty diner")
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  the module imports with no heavy deps")
    return bad
```

Add `from pathlib import Path` to the imports at the top of `tools/smoke_rewrite.py` if it is not already there (it is — `Path` is imported for `_from_app`), and change `main()` to:

```python
def main() -> int:
    bad = ops() + clean() + refusals() + wiring() + motion() + chat()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3.11 tools/smoke_rewrite.py`
Expected: FAIL — `ImportError: cannot import name 'chat' from 'comfy_nodes.visionary_rewrite'`

- [ ] **Step 3: Rewrite the node**

Replace the whole of `comfy_nodes/visionary_rewrite/__init__.py` with:

```python
"""
Rewrite a prompt on the model that is already going to read it.

Krea 2's text encoder is **Qwen3-VL-4B in bf16** — a decoder language model, not
a T5 or a CLIP — and ComfyUI holds it for the whole session because every render
goes through it. So the model that turns "empty diner, 3am" into a prompt is the
same object that will parse the result, and a rewriter which *is* the encoder
writes in the dialect the encoder reads.

**This used to load a second copy, and the reason it no longer does is that the
reason was retired upstream.** The old note here said driving generation on
ComfyUI's object meant reaching past its public surface and writing a KV-cached
decode loop by hand. At COMFY_SHA that loop is `comfy/text_encoders/llama.py`'s
`BaseGenerate`, first-party and public: a pre-allocated cache, a full sampler,
and `BaseQwen3.logits` handling the tied-embedding case by name — which is the
same fact this file used to discover for itself and graft into a state dict.
`comfy.sd.CLIP.generate` and `.decode` are the way in. See
`docs/design-notes/one-encoder-not-two.md`.

So this node loads nothing. It composes a chat string, hands it to whichever
`CLIP` the graph wired in, and returns what came back.
"""

import base64
import io

# The third bound on a frame, and the one that binds. The page shrinks to 1536
# before sending and `/api/motion` refuses a payload past 8 MB; this is
# `_fit_reference`'s rule applied to the vision path — the browser's cap is an
# optimisation and the server's is the contract.
MAX_FRAME_SIDE = 1536

# Opens the *user* turn, before the prose. A bare string never emits the
# placeholder tokens and the failure is a model that answers without looking.
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"

# Composed here rather than asked of the tokenizer, because **no template on
# this path has a system turn**: `Qwen3VLTokenizer.llama_template` opens at
# `<|im_start|>user` and `KREA2_TEMPLATE` is the fixed descriptor prompt the DiT
# conditions on, so the instruction has nowhere to go through either. The
# trailing empty think block is the Qwen3 convention that suppresses reasoning —
# `Krea2Tokenizer` defaults `thinking=True`, which omits it, correct for
# conditioning and wrong here. `skip_template=True` is the documented way past
# both, and `Qwen3VLTokenizer` sets it for any text opening `<|im_start|>` anyway.
_TEMPLATE = (
    "<|im_start|>system\n{instruction}<|im_end|>\n"
    "<|im_start|>user\n{vision}{prose}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def chat(instruction: str, prose: str, image: bool = False) -> str:
    """The exact string handed to `clip.tokenize`. Pure, so it is testable off-GPU."""
    return _TEMPLATE.format(instruction=instruction, prose=prose,
                            vision=VISION_BLOCK if image else "")


def _frame(image_b64: str):
    """base64 -> ComfyUI's IMAGE layout: (1, H, W, 3), float, 0-1."""
    # Imported here rather than at module scope so `chat` is importable on a
    # laptop with no torch — which is what makes the template checkable in
    # `smoke_rewrite.py` without a GPU or a container.
    import numpy as np
    import torch
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    except Exception as exc:
        raise RuntimeError(
            f"image_b64 did not decode to an image: {exc}") from exc
    w, h = img.size
    if max(w, h) > MAX_FRAME_SIDE:
        scale = MAX_FRAME_SIDE / max(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                         Image.LANCZOS)
    return torch.from_numpy(
        np.asarray(img).astype("float32") / 255.0)[None, ...]


class VisionaryRewrite:
    """prose + instruction (+ an optional frame) -> prose, on the graph's CLIP."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # The whole change. Whatever CLIPLoader the graph names is what
                # this drives, so the renders and the rewrites share one object.
                "clip": ("CLIP",),
                "prose": ("STRING", {"multiline": True, "default": ""}),
                "instruction": ("STRING", {"multiline": True, "default": ""}),
                "max_tokens": ("INT", {"default": 420, "min": 16, "max": 2048}),
            },
            # Base64 as a STRING rather than an IMAGE socket, deliberately: the
            # caller holds base64 (it is what the routes carry), the graph is
            # already an HTTP POST so the bytes ride the same channel as
            # everything else, and an IMAGE socket would mean staging a file and
            # a LoadImage for a picture that already arrived.
            "optional": {
                "image_b64": ("STRING", {"multiline": True, "default": ""}),
                # The container's warm-up knock, and it samples one token rather
                # than none: there is no loader left to call on its own, so the
                # only way to make ComfyUI's own load happen is to ask for work.
                "warm_only": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "visionary"
    # The answer has to come back out of ComfyUI, and an output node's `ui` dict
    # is the one channel that reaches `/history/{prompt_id}` — which `run_text`
    # already polls. Upstream's own `TextGenerate` is not an OUTPUT_NODE, which
    # is one of the two reasons this node exists rather than that one.
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Never cached. Two identical requests are two deliberate presses, and
        # ComfyUI returning the first one's output for the second would look
        # exactly like a model that ignored the button.
        return float("nan")

    def run(self, clip, prose, instruction, max_tokens, image_b64="",
            warm_only=False):
        if warm_only:
            prose, instruction, image_b64, max_tokens = " ", "", "", 1

        img = _frame(image_b64) if image_b64 else None
        text = chat(instruction, prose or " ", image=img is not None)
        # `min_length=1` and `skip_template=True` are what upstream's
        # `TextGenerate.execute` passes; the template is ours because the system
        # turn has nowhere else to go.
        tokens = clip.tokenize(text, image=img, skip_template=True, min_length=1)
        ids = clip.generate(tokens, do_sample=False, max_length=int(max_tokens))
        said = clip.decode(ids)
        # Cleaning stays in app.py — `_clean_rewrite` is the one implementation
        # and it is already what every backend's output goes through.
        return {"ui": {"text": [said]}, "result": (said,)}


NODE_CLASS_MAPPINGS = {"VisionaryRewrite": VisionaryRewrite}
NODE_DISPLAY_NAME_MAPPINGS = {"VisionaryRewrite": "Visionary Rewrite"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
           "VISION_BLOCK", "MAX_FRAME_SIDE", "chat"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3.11 tools/smoke_rewrite.py`
Expected: PASS, with three new `ok` lines under `chat template`.

- [ ] **Step 5: Commit**

```bash
git add comfy_nodes/visionary_rewrite/__init__.py tools/smoke_rewrite.py
git commit -m "Drive the CLIP the graph wired in, rather than a second copy of it"
```

---

## Task 2: One CLIPLoader literal, one rewrite graph

**Files:**
- Modify: `app.py` — add `_krea2_clip_node()` and `_rewrite_graph()`; change `_krea2_graph` (~line 5370-5380), `ImageGenerator.rewrite` (~5709), `VideoGenerator.rewrite` (~9040), `_warm_rewrite` (~5638)
- Test: `tools/smoke_rewrite.py` (new `graph()` check)

**Interfaces:**
- Consumes: the node's input names from Task 1 (`clip`, `prose`, `instruction`, `max_tokens`, `image_b64`, `warm_only`).
- Produces: `_krea2_clip_node() -> dict[str, Any]` and `_rewrite_graph(prose: str, instruction: str, max_tokens: int, image_b64: str = "", warm_only: bool = False) -> dict[str, Any]`. Task 3 imports `_rewrite_graph`; Task 6 adds a `seed` parameter to it.

- [ ] **Step 1: Write the failing test**

Add to `tools/smoke_rewrite.py`, above `def main()`:

```python
def graph() -> int:
    """
    The rewrite graph and the render graph name the same encoder.

    **This is the whole mechanism and it fails silently.** ComfyUI keys its
    execution cache on the input signature, so identical `CLIPLoader` inputs are
    one cached `CLIP` and one `ModelPatcher` — and one differing character is two
    resident copies of 8.9 GiB with no error, no log line and no symptom beyond
    a card that is fuller than it should be.
    """
    print("graph")
    bad = 0
    g = G2["_rewrite_graph"]("empty diner, 3am", "Rewrite it.", 200)

    ok = g["clip"] == G2["_krea2_clip_node"]()
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  the loader is the render graph's own")

    ok = g["rw"]["inputs"]["clip"] == ["clip", 0]
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  the node is wired to it")

    ok = g["clip"]["inputs"]["type"] == "krea2"
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  the loader asks for the krea2 tap")

    # The literal has one home. A second one anywhere in app.py is the drift
    # above, re-introduced by somebody inlining what looks like three obvious
    # strings — which is exactly how it would arrive.
    src = (Path(__file__).resolve().parent.parent / "app.py").read_text()
    n = src.count('"type": "krea2"')
    ok = n == 1
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  one CLIPLoader literal in app.py "
          f"(found {n})")

    ok = G2["_rewrite_graph"]("", "", 1, warm_only=True)["rw"]["inputs"]["warm_only"]
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  the warm-up rides the same graph")
    return bad
```

At the top of `tools/smoke_rewrite.py`, beside the existing `G = pull({...})`, add a second pull — kept separate because these two need `MODEL_CATALOGUE`, which the first set does not:

```python
# The graph builders, pulled separately because they need MODEL_CATALOGUE and
# the rewrite plumbing above does not. `_from_app` names its subset rather than
# pattern-matching it, so a rename upstream fails here loudly.
G2 = pull({"MODEL_CATALOGUE", "MODELS", "LORAS", "WORKSPACE",
           "_krea2_clip_node", "_rewrite_graph"})
```

Add `graph()` to `main()`:

```python
def main() -> int:
    bad = ops() + clean() + refusals() + wiring() + motion() + chat() + graph()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3.11 tools/smoke_rewrite.py`
Expected: FAIL — `SystemExit: not in app.py any more: _krea2_clip_node, _rewrite_graph`

- [ ] **Step 3: Add the two builders to `app.py`**

Insert immediately above `def _krea2_graph(` (currently ~line 5324):

```python
def _krea2_clip_node() -> dict[str, Any]:
    """
    The CLIPLoader the render graph and the rewrite graph both emit.

    **One function rather than two literals, because the drift is invisible.**
    ComfyUI keys its execution cache on the input signature, so identical inputs
    resolve to one cached `CLIP` object and one `ModelPatcher` — which is what
    makes the rewrite and the render genuinely share 8.9 GiB rather than each
    hold their own. A single differing character — a `device`, a name spelled
    from somewhere else — is two copies, with no error and no log line.
    """
    return {
        "class_type": "CLIPLoader",
        # type="krea2" is what selects Krea2Tokenizer and the Qwen3-VL hidden
        # state stack.
        "inputs": {"clip_name": MODEL_CATALOGUE["text_encoder"]["dest"].name,
                   "type": "krea2", "device": "default"},
    }


def _rewrite_graph(prose: str, instruction: str, max_tokens: int,
                   image_b64: str = "", warm_only: bool = False
                   ) -> dict[str, Any]:
    """
    Two nodes: the encoder the renders use, and the thing that talks to it.

    One builder with three callers — both generators and the warm-up knock —
    rather than the dict written out where it is needed, which is what makes it
    reachable from `smoke_graphs.py`. A graph shape never built is a node name
    never checked.
    """
    return {
        "clip": _krea2_clip_node(),
        "rw": {"class_type": "VisionaryRewrite",
               "inputs": {"clip": ["clip", 0], "prose": prose,
                          "instruction": instruction,
                          "max_tokens": int(max_tokens),
                          "image_b64": image_b64,
                          "warm_only": warm_only}},
    }
```

- [ ] **Step 4: Point `_krea2_graph` at the helper**

In `_krea2_graph`, delete the line `te = MODEL_CATALOGUE["text_encoder"]["dest"].name` (~5370) and replace the `"clip": {...}` entry (~5379-5380) with:

```python
        "clip": _krea2_clip_node(),
```

Delete the two comment lines above it that explain `type="krea2"` — that explanation now lives on the helper, and a fact with two homes is a fact that goes stale in one of them.

- [ ] **Step 5: Point both `rewrite()` methods and the warm-up at the builder**

In `ImageGenerator.rewrite` (~5709) replace the graph literal with:

```python
        return {"text": self._comfy.run_text(
            _rewrite_graph(prose, instruction, int(max_tokens),
                           image_b64=image_b64))}
```

In `VideoGenerator.rewrite` (~9040), the identical replacement.

In `_warm_rewrite` (~5642), replace the `comfy.run_text({...})` call with:

```python
            comfy.run_text(_rewrite_graph("", "", 1, warm_only=True),
                           timeout=600.0)
```

Then update `_warm_rewrite`'s docstring paragraph that reads "~8 GB read off a *network* volume plus a CPU construction the node prices at forty seconds" — it is no longer true. Replace that paragraph with:

```python
# **What the knock now heats is the copy the renders use.** It used to warm a
# second model no render would ever touch, while the render's own encoder stayed
# cold — so the warm-up and the first generation each paid for their own 8.9 GB.
# The graph names the same CLIPLoader `_krea2_graph` does, so one load serves
# both and the first render is warmer for it.
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `python3.11 tools/smoke_rewrite.py`
Expected: PASS, five new `ok` lines under `graph`.

- [ ] **Step 7: Confirm nothing else builds that literal**

Run: `grep -n 'VisionaryRewrite' app.py`
Expected: exactly three hits — the two `require_nodes` calls and the one inside `_rewrite_graph`. If a fourth appears, a call site was missed.

- [ ] **Step 8: Commit**

```bash
git add app.py tools/smoke_rewrite.py
git commit -m "Build the rewrite graph once, off the render graph's own loader"
```

---

## Task 3: The rewrite graph joins the CPU-container smoke test

**Files:**
- Modify: `tools/smoke_graphs.py` — the `from app import` block (~line 68), `_variants()` (~line 190), the node-presence loop (~line 437)

**Interfaces:**
- Consumes: `_rewrite_graph` from Task 2.
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Add the import**

In the `from app import (...)` block, add `_rewrite_graph,` in alphabetical position (after `_krea2_graph,`).

- [ ] **Step 2: Add the variants**

At the end of `_variants()`, immediately before `return out`:

```python
    # ── the rewrite ───────────────────────────────────────────────────────
    #
    # Not a render, and here for the reason every other row is: this is the one
    # graph whose node we own *and* whose `CLIP` input is a type ComfyUI
    # validates. Both branches, because the vision one is the motion path and a
    # branch never built is an input name never checked.
    out += [
        ("rewrite text", _rewrite_graph("empty diner, 3am", "Rewrite it.", 200)),
        ("rewrite vision", _rewrite_graph("what moves here", "Propose motion.",
                                          200, image_b64="Zm9v")),
        ("rewrite warm-up", _rewrite_graph("", "", 1, warm_only=True)),
    ]
```

- [ ] **Step 3: Name the node in the presence loop**

Change the loop at ~line 437 from:

```python
        for node in (KREA2_REGIONAL_NODE, "VisionaryBoxes"):
```

to:

```python
        for node in (KREA2_REGIONAL_NODE, "VisionaryBoxes", "VisionaryRewrite",
                     "VisionaryFreeRegional"):
```

- [ ] **Step 4: Run it**

Run: `modal run tools/smoke_graphs.py`
Expected: PASS, with `rewrite text`, `rewrite vision` and `rewrite warm-up` among the checked graphs and `VisionaryRewrite` present in `/object_info`.

If `VisionaryRewrite` is reported missing, the node raised on import — read the ComfyUI startup output the script prints and fix Task 1 before continuing.

- [ ] **Step 5: Commit**

```bash
git add tools/smoke_graphs.py
git commit -m "Check the rewrite graph on a CPU container, like every other graph"
```

---

## Task 4: Token-id parity, then drop the transformers bake

**Files:**
- Create: `tools/smoke_tokens.py`, `tools/tokens-golden.json`
- Modify: `app.py` — remove the `AutoConfig`/`AutoProcessor`/`AutoTokenizer` bake from `comfy_image` (~line 536-541)

**Interfaces:**
- Consumes: `chat` from Task 1.
- Produces: nothing later tasks depend on.

**Order matters here.** The golden fixture is captured while `transformers` is still baked into `comfy_image`, and the bake is removed in the same task once the fixture exists. Do not reorder the steps.

- [ ] **Step 1: Write the tool**

Create `tools/smoke_tokens.py`:

```python
"""
Does ComfyUI's tokenizer see the same string we think we are sending?

    modal run tools/smoke_tokens.py            # check against the fixture
    modal run tools/smoke_tokens.py --capture  # re-freeze it (commit the diff)

**The gate for the shared-encoder move, and the only exactness available.** The
old node tokenized with HuggingFace's `Qwen/Qwen3-VL-4B-Instruct`; the new one
goes through ComfyUI's bundled `qwen25_tokenizer`. If those disagree the model
is handed a different string than the one we composed, and the sharpest way for
that to show up is `<think>` — a Qwen3 addition, in a Qwen2.5 vocabulary. If it
tokenizes as plain text the suppression block stops suppressing, and what lands
in the prompt box is the model's reasoning about the prompt, as the prompt:
fluent, plausible at a glance, and on its way to the DiT.

**Why a fixture and not a live comparison.** `comfy_image` no longer carries
transformers — the node does not import it — so there is no HF tokenizer in this
container to compare against. `--capture` cross-checks the two while both are
available and writes the ids down; the standing check compares ComfyUI's
tokenizer to those ids, which is also what catches a `COMFY_SHA` bump changing
the bundled vocabulary.

**What this deliberately does not assert: that the prose comes out the same.**
Greedy decode is deterministic given identical logits, and two implementations
do not produce identical logits — a different attention kernel and a
pre-allocated KV cache written in place are mathematically equivalent and not
bitwise equal in bf16, so one near-tie flips argmax and the rest of a 400-token
decode diverges. That is arithmetic, not a defect. Whether the prose is *better*
is `tools/prompt_ab.py`.
"""

import json
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import comfy_image  # noqa: E402

smoke = modal.App("visionary-smoke-tokens")
smoke_image = comfy_image.add_local_python_source("app")

GOLDEN = Path(__file__).resolve().parent / "tokens-golden.json"

# The corpus is the fragments this feature exists for, plus the two shapes that
# stress the template rather than the prose: an empty prose (the warm-up knock)
# and a frame (the motion path).
CORPUS = [
    ("fragment", "Rewrite it.", "empty diner, 3am", False),
    ("correction", "Rewrite it.", "night. no, late afternoon", False),
    ("hedge", "Rewrite it.", "something like a courtroom but colder", False),
    ("recollection", "Rewrite it.",
     "We were walking through a regular forest and there was this colossal "
     "stone hand bursting out of the dirt. It felt ancient and terrifying.",
     False),
    ("quoted", "Rewrite it.", 'a sign reading "NO EXIT" above the door', False),
    ("warm-up", "", " ", False),
    ("vision", "Propose motion.", "what moves here", True),
]


def _ids(compose, tokenizer, row):
    _, instruction, prose, image = row
    text = compose(instruction, prose, image=image)
    tokens = tokenizer.tokenize_with_weights(text)
    pairs = next(iter(tokens.values()))[0]
    # The tokenizer returns (id, weight) pairs; an image row would carry a dict
    # in slot 0, which cannot appear here because no picture is attached.
    return [int(t[0]) for t in pairs]


@smoke.function(image=smoke_image, cpu=2.0, timeout=20 * 60)
def run(capture: bool = False) -> dict:
    sys.path.insert(0, "/root/comfy/ComfyUI")
    import comfy.text_encoders.krea2

    from comfy_nodes.visionary_rewrite import chat

    tok = comfy.text_encoders.krea2.Krea2Tokenizer()
    got = {row[0]: _ids(chat, tok, row) for row in CORPUS}

    if not capture:
        return {"ids": got}

    # Cross-check against HuggingFace while it is still in the image. Every row
    # must agree, or the fixture would freeze a disagreement.
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    mismatch = {}
    for row in CORPUS:
        text = chat(row[1], row[2], image=row[3])
        want = hf(text, add_special_tokens=False)["input_ids"]
        if want != got[row[0]]:
            mismatch[row[0]] = {"hf": want, "comfy": got[row[0]]}
    return {"ids": got, "mismatch": mismatch}


@smoke.local_entrypoint()
def main(capture: bool = False):
    out = run.remote(capture=capture)
    if capture:
        bad = out.get("mismatch") or {}
        for name, d in bad.items():
            print(f"  FAIL  {name}: ComfyUI and HuggingFace disagree")
            print(f"        hf   : {d['hf'][:24]}…")
            print(f"        comfy: {d['comfy'][:24]}…")
        if bad:
            raise SystemExit(
                f"\n{len(bad)} rows disagree — the fixture was NOT written. "
                "The composed template does not survive ComfyUI's tokenizer, "
                "which is the one thing this move had to be true.")
        GOLDEN.write_text(json.dumps(out["ids"], indent=2) + "\n")
        print(f"\nPASS — captured {len(out['ids'])} rows to {GOLDEN.name}")
        return

    want = json.loads(GOLDEN.read_text())
    bad = 0
    for name in want:
        ok = want[name] == out["ids"].get(name)
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}")
    print(f"\n{'PASS' if not bad else str(bad) + ' FAILED'}")
    if bad:
        raise SystemExit(1)
```

- [ ] **Step 2: Capture the fixture**

Run: `modal run tools/smoke_tokens.py --capture`
Expected: `PASS — captured 7 rows to tokens-golden.json`

**If it reports rows disagreeing, stop.** That is the gate failing, and it means the composed template does not survive ComfyUI's tokenizer. Read the ids either side of the first divergence, identify the token, and bring the finding back before changing anything — the fix might be the template, and it might be that this whole move needs a different approach to `<think>`.

- [ ] **Step 3: Run the standing check**

Run: `modal run tools/smoke_tokens.py`
Expected: PASS, seven `ok` lines.

- [ ] **Step 4: Drop the transformers bake**

In `app.py`'s `comfy_image` definition, delete the `.run_commands(...)` block that begins `"python -c \"from transformers import AutoConfig, AutoProcessor, "` (~line 536-541) together with the four comment paragraphs immediately above it that explain why it was baked (they start at `# \`visionary_rewrite\` loads the text encoder's *weights* from the local`). The node no longer imports `transformers` at all.

Leave `trainer_image`'s own bake (~line 368) alone — `krea2_encoder` still fetches a tokenizer by repo id.

- [ ] **Step 5: Confirm the image still builds and the node still imports**

Run: `modal run tools/smoke_graphs.py`
Expected: PASS, and `VisionaryRewrite` still present in `/object_info`. This is the check that the bake removal did not take the node with it.

- [ ] **Step 6: Commit**

```bash
git add tools/smoke_tokens.py tools/tokens-golden.json app.py
git commit -m "Freeze the token ids, then stop baking transformers into comfy_image"
```

---

## Task 5: Enhance rerolls from your prose

> **Ships with Task 6.** Between this task and the next, a second press returns byte-identical text and the note line says "found nothing to change," which is a true sentence about a misleading situation. Commit both, deploy once.

**Files:**
- Modify: `web/src/store.ts` (~line 261-276 for the interface, ~404-427 for the implementation), `web/src/console/Rewrite.tsx`
- Modify: `tools/preview_ui.py` — add an `/api/rewrite` stub beside `/api/motion` (~line 1077)
- Create: `tools/ui-checks/check_rewrite.py`

**Interfaces:**
- Consumes: nothing from earlier tasks — this half is client-side.
- Produces: store fields `rewriteFrom: string | null`, `rewriteLast: string | null`, and `proseForRewrite(): string`. Task 6 does not touch them.

- [ ] **Step 1: Add the preview stub**

`preview_ui.py` serves `rewrite_ops` in `/api/state` but has no `/api/rewrite`, so the button the preview renders cannot be pressed — the same fault as a missing menu, one level down. Add immediately before `if path == "/api/motion":`:

```python
        # The rewrite, stubbed so the button the state route already advertises
        # can actually be pressed. It answers with a *marked* expansion rather
        # than prose, because what the checks need to see is which string the
        # press was a reading of — a plausible rewrite would make press 2
        # indistinguishable from press 1 having been re-read.
        if path == "/api/rewrite":
            try:
                p = json.loads(body or b"{}")
            except json.JSONDecodeError:
                p = {}
            prose = str(p.get("prose") or "")
            if not prose:
                return self.reply({"ok": True, "text": "", "op": p.get("op")})
            self._rw = getattr(self, "_rw", 0) + 1
            time.sleep(0.4)
            return self.reply({"ok": True, "op": p.get("op"),
                               "text": f"[{self._rw}] {prose}, lit from one side"})
```

Note `self._rw` is per-request-handler in `http.server`, so make it a module-level counter instead — replace `self._rw = getattr(self, "_rw", 0) + 1` with a `global` on a module-level `REWRITES = 0` declared beside `STATE`:

```python
# How many rewrites this preview has served. The count rides in the answer so a
# check can tell "pressed again" from "pressed once", which is the whole thing
# the reroll rules are about.
REWRITES = 0
```

and in the handler:

```python
            global REWRITES
            REWRITES += 1
            time.sleep(0.4)
            return self.reply({"ok": True, "op": p.get("op"),
                               "text": f"[{REWRITES}] {prose}, lit from one side"})
```

- [ ] **Step 2: Write the failing UI check**

Create `tools/ui-checks/check_rewrite.py`:

```python
"""
Enhance is a reroll, and a reroll does not drift.

    /opt/homebrew/bin/python3.11 tools/preview_ui.py 8791 &
    python3.11 tools/ui-checks/check_rewrite.py

Three rules, and every one of them was broken while looking fine:

  * **Every press reads the prose, not the box.** `Rewrite.tsx` sent
    `stripLoras(s.prompt)`, so press 2 rewrote press 1's output and rule 7 of
    KREA_EXPANSION turned it into a polish of a polish. Nothing errors; you just
    end up four removes from what you meant.
  * **Undo lands on your sentence.** `applyRewrite` overwrote the undo slot
    every press, so ⌘Z from press 2 restored press 1 and the typed sentence was
    gone. `undoDoc`'s "there is only ever one parse write to take back" is true
    of the parse, which fires once, and false of a button.
  * **A hand edit re-bases.** If you edit the rewritten prompt and press again,
    the box *is* the intent now. Same rule `motion.base` carries.

The preview's `/api/rewrite` stub numbers its answers, which is what lets a check
tell a re-read from a re-rewrite: press 2 on the same prose comes back
`[2] <the original>`, and press 2 on the box comes back `[2] [1] <the original>`.
"""
import sys
import time

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8791"
fails = []

TYPED = "empty diner, 3am"


def box(page):
    return page.eval_on_selector("#prompt", "el => el.value")


def press(page, label="Enhance"):
    page.click(f"button:has-text('{label}')")
    page.wait_for_function(
        "() => !document.querySelector('button[aria-busy=\"true\"]')",
        timeout=15000)
    time.sleep(0.1)


with sync_playwright() as pw:
    b = pw.chromium.launch()
    page = b.new_page(viewport={"width": 1512, "height": 982})
    page.goto(URL, wait_until="networkidle")

    page.click("#prompt")
    page.fill("#prompt", TYPED)
    press(page)
    first = box(page)
    if first != f"[1] {TYPED}, lit from one side":
        fails.append(f"press 1 wrote {first!r}")

    press(page)
    second = box(page)
    # The rule: press 2 is a reading of TYPED, not of press 1's answer.
    if second != f"[2] {TYPED}, lit from one side":
        fails.append(f"press 2 drifted — read the box, not the prose: {second!r}")

    press(page)
    third = box(page)
    if third != f"[3] {TYPED}, lit from one side":
        fails.append(f"press 3 drifted: {third!r}")

    # Undo, from three presses deep, lands on what was typed.
    page.click("button:has-text('Undo')")
    time.sleep(0.2)
    if box(page) != TYPED:
        fails.append(f"undo landed on {box(page)!r}, not the typed sentence")

    # A hand edit re-bases: the box is the intent now, so the next press reads it.
    page.fill("#prompt", TYPED)
    press(page)
    page.fill("#prompt", "a courtroom, colder")
    press(page)
    if box(page) != "[5] a courtroom, colder, lit from one side":
        fails.append(f"a hand edit did not re-base: {box(page)!r}")

    b.close()

for f in fails:
    print(f"  FAIL  {f}")
print(f"\n{'PASS' if not fails else str(len(fails)) + ' FAILED'}")
raise SystemExit(1 if fails else 0)
```

- [ ] **Step 3: Run it to verify it fails**

```bash
/opt/homebrew/bin/python3.11 tools/preview_ui.py 8791 &
cd web && npm run build && cd ..
python3.11 tools/ui-checks/check_rewrite.py
```

Expected: FAIL on `press 2 drifted` — the current code sends the box.

- [ ] **Step 4: Add the store fields**

In `web/src/store.ts`, in the interface beside `applyRewrite` (~line 276):

```ts
  /** The string every rewrite press is a reading of.
   *
   *  **Not the box.** `Rewrite.tsx` used to send whatever was in the field, so
   *  press 2 rewrote press 1's output — and `KREA_EXPANSION`'s rule 7 ("if the
   *  user's prompt is already detailed, lightly polish rather than heavily
   *  expanding") made that a polish of a polish. Four presses is four removes
   *  from what was meant, with nothing having gone wrong. Kept, so every press
   *  is an independent interpretation and no press is further from intent than
   *  the first. */
  rewriteFrom: string | null
  /** What the last press wrote, so a hand edit can be told from another press.
   *  The box no longer equalling this is the re-base signal — the same rule
   *  `motion.base` carries, for the same reason: once you have edited the
   *  answer, the answer is what you meant. */
  rewriteLast: string | null
  /** The prose a press should send, and the only thing that decides it. */
  proseForRewrite: () => string
```

In the implementation, beside `applyRewrite` (~line 420):

```ts
  rewriteFrom: null,
  rewriteLast: null,
  // Read at press time rather than tracked on every keystroke: a keystroke
  // handler that maintained this would be a second place for the re-base rule
  // to live, and the two would disagree the first time something else wrote the
  // box — which `toggleMotion` and `+ LoRA` both do.
  proseForRewrite: () => {
    const s = useStore.getState()
    return s.rewriteFrom !== null && s.prompt === s.rewriteLast
      ? s.rewriteFrom
      : s.prompt
  },
  applyRewrite: (text) =>
    set((s) => {
      // An unchanged answer is a real answer — Enhance returns its input when
      // it finds nothing to fix. Recording an undo for a write that did not
      // happen would arm ⌘Z to do nothing visible.
      if (!text || text === s.prompt) return {}
      const rebasing = s.rewriteFrom === null || s.prompt !== s.rewriteLast
      return {
        prompt: text,
        doc: null,
        rewriteFrom: rebasing ? s.prompt : s.rewriteFrom,
        rewriteLast: text,
        // **The slot holds your sentence, never the previous rewrite.** Only a
        // re-base arms it; a second press onto a box this feature wrote is not
        // a new thing to take back, it is the same one answered again.
        docUndo: rebasing ? { prompt: s.prompt, doc: s.doc } : s.docUndo,
      }
    }),
```

And in `undoDoc` (~line 416), clear the pair so a press after an undo re-bases:

```ts
  undoDoc: () => set((s) => (s.docUndo
    ? { ...s.docUndo, docUndo: null, rewriteFrom: null, rewriteLast: null }
    : {})),
```

- [ ] **Step 5: Send the prose from `Rewrite.tsx`**

In `web/src/console/Rewrite.tsx`, replace:

```ts
  const prose = stripLoras(s.prompt).trim()
```

with:

```ts
  // Whether the button is *available* is about the box — an empty field has
  // nothing to act on. What gets *sent* is `proseForRewrite()`, read at press
  // time, because those are two different questions and conflating them is how
  // this drifted: the box is the affordance, the kept prose is the input.
  const prose = stripLoras(s.prompt).trim()
```

and inside `run`, change the request line from `{ prose, op, kind: s.kind }` to:

```ts
      const r = await rewrite({ prose: stripLoras(s.proseForRewrite()).trim(),
                                op, kind: s.kind })
```

- [ ] **Step 6: Typecheck, rebuild and run the check**

```bash
cd web && npm run typecheck && npm run build && cd ..
python3.11 tools/ui-checks/check_rewrite.py
```

Expected: PASS.

- [ ] **Step 7: Confirm nothing else regressed**

Run: `python3.11 tools/ui-checks/baseline.py check`
Expected: PASS. If the console-budget or clause probes moved, the change touched something it should not have — read the diff before re-baselining.

- [ ] **Step 8: Commit**

```bash
git add web/src/store.ts web/src/console/Rewrite.tsx tools/preview_ui.py tools/ui-checks/check_rewrite.py
git commit -m "Read the sentence you typed on every press, not the last answer"
```

---

## Task 6: The decode samples, so a reroll is a reroll

> **Ships with Task 5.** See that task's banner.

**Files:**
- Modify: `comfy_nodes/visionary_rewrite/__init__.py` — a `seed` input, `do_sample=True`, the sampling constants
- Modify: `app.py` — `_rewrite_graph` takes a seed; `_rewrite_backend`, both `rewrite()` methods, `/api/rewrite` and `/api/motion` plumb one
- Modify: `tools/smoke_rewrite.py` — extend `graph()`

**Interfaces:**
- Consumes: `_rewrite_graph` from Task 2, whose signature gains `seed: int = 0` as its last parameter.
- Produces: `_rewrite_backend(prose, instruction, max_tokens, kind="image", seed=0)`.

- [ ] **Step 1: Extend the failing test**

In `tools/smoke_rewrite.py`'s `graph()`, before `return bad`:

```python
    ok = G2["_rewrite_graph"]("x", "y", 200, seed=7)["rw"]["inputs"]["seed"] == 7
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  the seed reaches the node")

    # Repeatability is the harness's requirement, never the product's — every
    # measurement in tools/ pins a seed and every press draws one.
    ok = G2["_rewrite_graph"]("x", "y", 200)["rw"]["inputs"]["seed"] == 0
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  an unpinned seed defaults to 0")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python3.11 tools/smoke_rewrite.py`
Expected: FAIL — `KeyError: 'seed'`

- [ ] **Step 3: Add sampling to the node**

In `comfy_nodes/visionary_rewrite/__init__.py`, add above the class:

```python
# **Sampling, not greedy, and the two halves are one decision.** With every
# press reading the same kept prose (see `proseForRewrite` on the page), a
# greedy decode returns byte-identical text on press two — and the page would
# report "found nothing to change" about a button that did exactly what it did
# the first time. Reroll-from-prose is what makes sampling necessary, and
# sampling is what makes reroll-from-prose mean anything.
#
# Qwen3's own guidance for non-thinking mode. ComfyUI's `TextGenerate` defaults
# are Gemma-shaped (0.7 / 0.95 / 64 / 0.05) and are the wrong family. The
# presence penalty is for a symptom CLAUDE.md already records: the red armchair
# coming back as "a pristine, bright red velvet armchair, velvet, bright red,
# pristine".
SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0,
                repetition_penalty=1.0, presence_penalty=0.5)
```

Add to `INPUT_TYPES` under `"required"`, after `max_tokens`:

```python
                # Drawn per press by the route, pinned by the harness. See
                # `_rewrite_backend`.
                "seed": ("INT", {"default": 0, "min": 0,
                                 "max": 0xffffffffffffffff}),
```

Change `run`'s signature to `def run(self, clip, prose, instruction, max_tokens, seed=0, image_b64="", warm_only=False):` and the generate call to:

```python
        ids = clip.generate(tokens, do_sample=not warm_only,
                            max_length=int(max_tokens), seed=int(seed),
                            **SAMPLING)
```

`do_sample=not warm_only` because the knock is loading a model, not answering anybody, and greedy is one less thing for it to do.

- [ ] **Step 4: Plumb the seed through `app.py`**

`_rewrite_graph` gains `seed: int = 0` as its last parameter and `"seed": int(seed)` in the node's inputs.

Both `rewrite()` methods gain `seed: int = 0` and pass it through:

```python
    def rewrite(self, prose: str, instruction: str,
                max_tokens: int = 420, image_b64: str = "",
                seed: int = 0) -> dict[str, Any]:
```

`_rewrite_backend` gains it and forwards:

```python
def _rewrite_backend(prose: str, instruction: str, max_tokens: int,
                     kind: str = "image", seed: int = 0) -> str:
    """Prose in, prose out, wherever the model happens to live."""
    if REWRITE_BACKEND == "comfy":
        said = _rewrite_generator(kind).rewrite.remote(
            prose, instruction, max_tokens, seed=seed)
```

In `/api/rewrite`, draw one per press, above the `try:`:

```python
        # **Drawn here rather than in the node, so the harness can pin it.**
        # Repeatability was never a product requirement — it was a measurement
        # one, and `smoke_parse.py`, `prompt_ab.py` and the judge all pass a
        # fixed seed. A press wants a different answer; a measurement wants the
        # same one, and only one of those is the default.
        seed = int(payload.get("seed") or random.getrandbits(63))
```

then pass `seed` as the last argument to `_rewrite_backend`.

Confirm `import random` is present at the top of `app.py`; add it if not.

In `/api/motion`, pass a drawn seed the same way — the motion suggestions are a press too, and re-pressing should propose differently:

```python
            said = _rewrite_generator("video").rewrite.remote(
                prose or "(nothing was typed — ground every proposal in the image)",
                instruction, MOTION_TOKENS, image_b64=image,
                seed=int(payload.get("seed") or random.getrandbits(63)))
```

- [ ] **Step 5: Run the offline test**

Run: `python3.11 tools/smoke_rewrite.py`
Expected: PASS, two new `ok` lines under `graph`.

- [ ] **Step 6: Run the graph check**

Run: `modal run tools/smoke_graphs.py`
Expected: PASS — this is what catches `seed` being a required input the builder forgot on some branch.

- [ ] **Step 7: Commit**

```bash
git add comfy_nodes/visionary_rewrite/__init__.py app.py tools/smoke_rewrite.py
git commit -m "Sample the decode and draw a seed per press, so a reroll rerolls"
```

---

## Task 7: Deploy, and measure what reading cannot answer

**Files:** none modified. This task is the verification the plan exists to earn.

- [ ] **Step 1: Deploy**

Run: `modal deploy app.py`
Expected: builds and deploys. If the image build fails, see the memory note — Modal's build logs interleave and "Runner terminated" is a decoy; build one image in a Sandbox to find which.

- [ ] **Step 2: Confirm one encoder, not two**

Open a session, press Generate once, then press Enhance. In the Modal logs for the image container, count text-encoder load lines.

Expected: **one**. Two means the `CLIPLoader` inputs drifted despite Task 2's helper, or ComfyUI's `RAMPressureCache` evicted the loader output under RAM pressure. If it is the second, the fix is `--cache-lru` in `_Comfy.start`'s argument list — but confirm which before reaching for it, because the two have different remedies.

- [ ] **Step 3: Confirm the handoff does not stall**

Render, Enhance, render again. Time the second render's `loading` phase against the first's.

Expected: comparable. A second render that pays a full checkpoint reload is the eviction case above.

- [ ] **Step 4: Confirm the reroll**

Type `empty diner, 3am`. Press Enhance three times, reading each answer.

Expected: three different prompts, each a plausible reading of *that fragment* — not of the previous answer. Then press Undo: the box reads `empty diner, 3am`.

- [ ] **Step 5: Confirm the vision path**

On a video session with a first frame attached, open the motion panel.

Expected: suggestions grounded in the picture. This is the one path where the frame reaches the tower, and it is the one Task 1 rewrote most.

- [ ] **Step 6: Measure whether the prose is better**

Run: `python3.11 tools/prompt_ab.py` against the fragments in `tools/rewrite-dump.jsonl`, then `python3.11 tools/judge_renders.py` on the output.

This is the only measurement here that is not a proxy, and it is the one that answers what none of the checks can: whether ComfyUI's sampler, ComfyUI's tokenizer and sampling-instead-of-greedy left the pictures as good. Report the numbers rather than a verdict — a small loss might be the sampling constants rather than the move.

- [ ] **Step 7: Commit the findings**

Add what Step 6 measured to `docs/design-notes/one-encoder-not-two.md` under a new "What it measured" heading, in the file's own shape: the claim, then the number.

```bash
git add docs/design-notes/one-encoder-not-two.md
git commit -m "Record what the shared encoder measured"
```

---

## Task 8: Correct the record

**Files:**
- Modify: `CLAUDE.md` — the "A second copy in the same container, not the resident instance" paragraph and the two beneath it
- Modify: `tools/smoke_rewrite.py` — the docstring paragraph beginning "**Run this after any `COMFY_SHA` bump.**"

- [ ] **Step 1: Rewrite the CLAUDE.md section**

Replace the paragraph beginning `**A second copy in the same container, not the resident instance.**` and the two that follow it (`**And the OOM cascade...**` stays; the two on the vision tower and meta-device init go, because the code they describe is gone) with:

```markdown
**It runs on the resident instance, and getting there was a retraction.** This
section used to argue the opposite: reuse meant reaching into `comfy.sd.CLIP` at
a pinned `COMFY_SHA` and hand-writing a KV-cached decode loop, because ComfyUI's
wrapper runs a single forward for conditioning and has no cache. That was true
when it was written and is not true at the SHA we pin. `comfy/text_encoders/
llama.py` carries `BaseGenerate` — a pre-allocated cache, a full sampler — and
`BaseQwen3.logits` handles the tied-embedding case by name, annotating
`Qwen3VL_4BConfig` with `lm_head: bool = False  # 4B ties word embeddings`, which
is the fact the old node discovered for itself and grafted into a state dict.
`CLIP.generate` and `CLIP.decode` are public on `comfy.sd`, and
`comfy_extras/nodes_textgen.py` ships a `TextGenerate` node doing exactly this.
Upstream had already built the thing the objection was about.

So the node loads nothing. It takes a `CLIP` input, and `_rewrite_graph` emits
the same `CLIPLoader` `_krea2_graph` does — through one `_krea2_clip_node()`
rather than two literals, because ComfyUI keys its execution cache on the input
signature and one differing character is two resident copies of 8.9 GiB with no
error and no log line. What that bought: ~8.9 GiB on the image container, an
allocation `unload_all_models()` could not see becoming one `_reclaim()` can
reach, ~200 lines and a `transformers` dependency deleted, and a warm-up that
heats the copy the *renders* use instead of a second one they never touch.

The full account, including the H3 generation tail that is Phase 2 and why Wan
gets nothing, is `docs/design-notes/one-encoder-not-two.md`.
```

- [ ] **Step 2: Correct the smoke test's docstring**

Replace the paragraph:

```
**Run this after any `COMFY_SHA` bump.** `comfy_image` takes its transformers
from ComfyUI's own `requirements.txt`, unpinned, and `visionary_rewrite` loads
the encoder against a pinned config — this is the check that the two have not
drifted apart in a way the cleaner would hide.
```

with:

```
**Run this after any `COMFY_SHA` bump, and `smoke_tokens.py` with it.** The
rewrite no longer loads weights — it drives the `CLIP` the graph wires in — so
three upstream names are load-bearing where none used to be: `BaseGenerate`,
`BaseQwen3.logits` and `CLIP.generate`. This checks the graph and the template;
`smoke_tokens.py` checks that ComfyUI's bundled tokenizer still produces the ids
the template was frozen against.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md tools/smoke_rewrite.py
git commit -m "Record that the second copy is gone, and what retired it"
```

---

## Self-Review

**Spec coverage.** Every section of `one-encoder-not-two.md` maps to a task: "The node" → Task 1; "The graph, and why the loader line is copied exactly" → Task 2; "The video container" → Task 2 (both `rewrite()` methods take the same builder); "Why the render/rewrite handoff does not stall" → Task 7 Steps 2-3, which measure rather than assume; "What else changes" → Tasks 3, 4 and 8; "Enhance becomes a reroll" → Tasks 5 and 6; the risk table's four rows → Task 4 (token ids), Task 7 Step 4 (structural), Task 7 Step 6 (quality). **Phase 2 is deliberately absent** — it is sequenced after and is its own plan.

**One gap, named rather than papered over.** The spec's risk row "fluent garbage — growth ratio inside the measured band" has no task. It is not implemented because the band comes from a single measurement in `CLAUDE.md` and the memory note `ask-before-hard-wiring-a-rule` applies directly: it would be a threshold nobody agreed to. Task 7 Step 6 covers the same failure by reading, which is the honest instrument for it. If the author wants it as a check, it belongs in `smoke_rewrite.py` as a **warning**, not a failure, and needs its own conversation first.

**Type consistency.** `chat(instruction, prose, image=False)` is defined in Task 1 and used in Task 4 with the same signature. `_rewrite_graph(prose, instruction, max_tokens, image_b64="", warm_only=False)` is defined in Task 2, used in Task 3, and gains `seed=0` in Task 6 — Task 3's call sites use only the first three positionally and stay valid. `_krea2_clip_node()` takes no arguments in both its definition and its two call sites. `proseForRewrite()` is declared and implemented in Task 5 and called in Task 5 only.

**Placeholders.** None. Every code step carries the actual code; every test step carries the actual command and the expected output.
