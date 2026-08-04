"""
Smoke test for the video graphs — every variant, on CPU, with no weights.

    modal run tools/smoke_video.py

Checks that every graph `_h3_graph` and `_wan_graph` can emit is structurally
valid against the ComfyUI build pinned in `video_image`: every `class_type`
exists, every input key is a real input on that node, and every link points at
a node in the same graph.

Why this is worth its own tool. The graphs are written as Python dicts against
Comfy's templates, so the failure they invite is a name: a node renamed
upstream, an input that moved, a slot that shifted. ComfyUI answers that with a
validation error *after* the container is up — which for video means an H100
cold start behind up to 28.6 GB of weights. The names can be checked against
`/object_info` in about a minute on a CPU container instead, and `/object_info`
does not care whether a single weight is on the volume.

So this runs ComfyUI with `--cpu` and never loads a model. Which also sets the
boundary of what it proves:

- It DOES catch a wrong node name, a wrong or misspelled input, a dangling
  link, and a required input the builder forgot.
- It DOES catch a wrong value in a fixed combo — `CLIPLoader.type` having to be
  "minimax" or "wan", a sampler or scheduler this deployment offers that
  ComfyUI does not have, `ref_image_size`, the two-expert noise flags. Those
  lists are compiled in, so they are populated with no weights present.
- It does NOT catch a filename that is not on the volume. Loader nodes take a
  combo too, but one whose options are the files ComfyUI can see, and with no
  weights present every one of those lists is empty. Emptiness is what
  separates the two cases, and it is what this skips on: `_require_models()`
  covers the file combos, and it covers them before the GPU is rented.
- It does not run a sampler, so it says nothing about whether a clip looks
  right. The two-expert handover flags in particular (`add_noise`,
  `return_with_leftover_noise`) are structurally valid whatever they are set
  to; getting them wrong produces a washed-out clip, not an error, and only a
  real run will show it.

A pass here means the graphs are wired to nodes that exist. It does not mean
the video is good.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import (  # noqa: E402
    COMFY,
    COMFY_PORT,
    VIDEO_MODELS,
    _h3_canvas,
    _h3_frames,
    _h3_graph,
    _wan_canvas,
    _wan_frames,
    _wan_graph,
    video_image,
)

smoke = modal.App("visionary-smoke-video")

# `modal run tools/smoke_video.py` mounts tools/, not the repo root, so app.py
# has to be added explicitly — the graph builders come from there.
smoke_image = video_image.add_local_python_source("app")

# ComfyUI's frontend appends this to seed widgets; it is not something an API
# graph supplies, and some nodes still list it as required. Excluding it by
# name keeps the missing-required check from crying wolf on every sampler.
FRONTEND_ONLY = {"control_after_generate"}


def _variants() -> list[tuple[str, dict]]:
    """
    Every shape the two builders can produce, named for the failure it covers.

    One graph per branch, not per parameter: the point is to reach every
    `class_type` the code can emit — the keyframe nodes, the first/last node,
    the reference node, the two-expert pair, the 5B single expert — because a
    node name is what goes stale, and a branch never built is a name never
    checked.
    """
    out: list[tuple[str, dict]] = []
    seed, steps = 1, 4

    # ── MiniMax-H3 ────────────────────────────────────────────────────────
    w, h = _h3_canvas("16:9", "draft")
    n = _h3_frames(5)
    base = dict(prompt="a smoke test", width=w, height=h, frames=n,
                seed=seed, steps=steps, sampler="res_multistep", scheduler="simple")
    out += [
        ("h3 t2v", _h3_graph(**base)),
        ("h3 i2v (first)", _h3_graph(**base, first_frame="a.png")),
        ("h3 first+last", _h3_graph(**base, first_frame="a.png", last_frame="b.png")),
        # The reference branch loads a different transformer and a different
        # conditioning node, and its inputs are the autogrow keys — the one
        # place in either builder where the input name is computed.
        ("h3 ref2va", _h3_graph(**base, references=["r0.png", "r1.png"],
                                ref_videos=["v0.mp4"], ref_size="match")),
    ]

    # ── Wan 2.2 ───────────────────────────────────────────────────────────
    stack = [{"name": "hi.safetensors", "unet": 1.0, "expert": "high"},
             {"name": "lo.safetensors", "unet": 1.0, "expert": "low"},
             {"name": "both.safetensors", "unet": 0.8, "expert": "both"}]
    for fam in ("14b", "5b"):
        d = VIDEO_MODELS["wan14b" if fam == "14b" else "wan5b"]["defaults"]
        w, h = _wan_canvas("16:9", "draft", fam)
        n = _wan_frames(2, fam)
        base = dict(family=fam, prompt="a smoke test", negative_prompt="blurry",
                    width=w, height=h, frames=n, seed=seed, steps=steps,
                    cfg=d["cfg"], shift=d["shift"], switch_at=steps // 2,
                    sampler="euler", scheduler="simple")
        out += [
            (f"wan {fam} t2v", _wan_graph(**base)),
            (f"wan {fam} t2v + loras", _wan_graph(**base, loras=stack)),
            (f"wan {fam} i2v", _wan_graph(**base, first_frame="a.png")),
        ]
        # Only the A14B pair advertises a last frame; the 5B has no node for it
        # and the API refuses one, so building it here would test a request the
        # product does not make.
        if VIDEO_MODELS["wan14b"]["supports"]["last_frame"] and fam == "14b":
            out += [
                (f"wan {fam} first+last",
                 _wan_graph(**base, first_frame="a.png", last_frame="b.png")),
                # Last alone still has to reach WanFirstLastFrameToVideo — the
                # i2v node has no end_image input at all.
                (f"wan {fam} last only", _wan_graph(**base, last_frame="b.png")),
            ]
    return out


# Inputs whose options are whatever this container can see, rather than a menu
# compiled into the node. Their contents are a property of the volume, and this
# runs without one — `vae_name` here answers `["pixel_space"]` and `image`
# answers `["example.png"]`, so checking a value against them would fail every
# real filename for the wrong reason.
#
# Named rather than detected: the tempting rule is "skip the empty lists", and
# it is wrong in both directions — these two are non-empty, and `LoadVideo.file`
# is empty. A list is what the code can be honest about.
VOLUME_BACKED = {"vae_name", "unet_name", "clip_name", "lora_name", "image", "file"}


def _combo(spec: Any) -> list | None:
    """
    The option list of a combo input, or None if this input is not one.

    Two encodings are live in this ComfyUI at once, and the video graphs touch
    both: the legacy `[[...options...], {...}]` that VAELoader and
    `CLIPLoader.type` still use, and the V3 `["COMBO", {"options": [...]}]` that
    `sampler_name`, `scheduler` and `ref_image_size` moved to. Reading only the
    first form is not a partial check — it silently reports every V3 menu as
    empty, which reads as "this sampler does not exist" for samplers that do.
    """
    if not isinstance(spec, list) or not spec:
        return None
    if isinstance(spec[0], list):
        return spec[0]
    if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1].get("options")
    # COMFY_DYNAMICCOMBO_V3 (SaveVideo.codec) nests its options as dicts with
    # their own sub-inputs. Its keys are checkable, its shape is not this one.
    return None


def _check_menus(schema: dict) -> list[str]:
    """
    Every sampler and scheduler `VIDEO_MODELS` offers, against what exists.

    The graph variants only build one of each, but the page renders the whole
    list — so an entry that ComfyUI dropped is a menu item that raises on
    selection, hours after this passed. Checked against the sampler node each
    family actually uses, because the two are not guaranteed to agree.
    """
    bad: list[str] = []
    for key, spec in VIDEO_MODELS.items():
        # H3 picks its sampler through KSamplerSelect and its scheduler through
        # BasicScheduler; Wan takes both as widgets on KSamplerAdvanced.
        where = {"sampler": ("KSamplerSelect", "sampler_name"),
                 "scheduler": ("BasicScheduler", "scheduler")} if key == "h3" else {
                 "sampler": ("KSamplerAdvanced", "sampler_name"),
                 "scheduler": ("KSamplerAdvanced", "scheduler")}
        for field, (cls, inp) in where.items():
            options = _combo(
                (schema.get(cls, {}).get("input", {}).get("required", {}) or {}).get(inp)
            ) or []
            for offered in spec[f"{field}s"]:
                if offered not in options:
                    bad.append(f"VIDEO_MODELS[{key!r}] offers {field} {offered!r}, "
                               f"which {cls}.{inp} does not have")
    return bad


def _check(name: str, graph: dict, schema: dict) -> list[str]:
    """Structural errors in one graph, as lines ready to print."""
    bad: list[str] = []
    for node_id, node in graph.items():
        cls = node["class_type"]
        spec = schema.get(cls)
        if spec is None:
            bad.append(f"{name}: node {node_id!r} — no such node type {cls!r}")
            continue

        req = spec.get("input", {}).get("required", {}) or {}
        opt = spec.get("input", {}).get("optional", {}) or {}
        known = set(req) | set(opt)

        for key, val in (node.get("inputs") or {}).items():
            options = None if key in VOLUME_BACKED else _combo(req.get(key)
                                                               or opt.get(key))
            if options and not isinstance(val, list) and val not in options:
                bad.append(f"{name}: {cls}.{key} = {val!r} is not one of "
                           f"{', '.join(map(str, options))}")
            # Autogrow inputs ("ref_images.ref_image_0") are named by their
            # group, which is what the schema carries — so the group is the
            # part worth checking, and a typo in it is the bug that makes
            # ComfyUI accept the graph and silently ignore the reference.
            probe = key.split(".", 1)[0] if "." in key else key
            if probe not in known:
                bad.append(f"{name}: {cls}.{key} — not an input on this node "
                           f"(has: {', '.join(sorted(known)) or 'none'})")
            # A link is [node_id, slot]; the target has to be in this graph.
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                if val[0] not in graph:
                    bad.append(f"{name}: {cls}.{key} links to {val[0]!r}, "
                               "which is not in the graph")

        supplied = {k.split(".", 1)[0] for k in (node.get("inputs") or {})}
        for key in req:
            if key in FRONTEND_ONLY or key in supplied:
                continue
            bad.append(f"{name}: {cls}.{key} — required input not supplied")
    return bad


@smoke.function(image=smoke_image, cpu=4.0, timeout=1800)
def main() -> None:
    import urllib.error
    import urllib.request

    # No extra_model_paths.yaml and no volume: /object_info describes the nodes
    # ComfyUI has compiled in, which is a property of the pin in video_image,
    # not of anything on our storage.
    (COMFY / "input").mkdir(parents=True, exist_ok=True)
    (COMFY / "output").mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        ["python", "main.py", "--listen", "127.0.0.1", "--port", str(COMFY_PORT),
         "--disable-auto-launch", "--disable-metadata", "--cpu"],
        cwd=str(COMFY), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    try:
        deadline = time.time() + 600
        while True:
            if proc.poll() is not None:
                out = (proc.stdout.read() if proc.stdout else "") or ""
                raise SystemExit("ComfyUI exited during startup:\n" + out[-4000:])
            if time.time() > deadline:
                raise SystemExit("ComfyUI did not become ready in 10 minutes.")
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{COMFY_PORT}/system_stats", timeout=2).read()
                break
            except (urllib.error.URLError, OSError):
                time.sleep(1.0)
        print("[smoke] ComfyUI ready (cpu, no models)", flush=True)

        with urllib.request.urlopen(
            f"http://127.0.0.1:{COMFY_PORT}/object_info", timeout=120
        ) as r:
            schema = json.loads(r.read())
        print(f"[smoke] {len(schema)} node types available", flush=True)

        failures: list[str] = []
        for name, graph in _variants():
            bad = _check(name, graph, schema)
            nodes = len(graph)
            print(f"  {'FAIL' if bad else ' ok '}  {name}  ({nodes} nodes)", flush=True)
            failures += bad

        menus = _check_menus(schema)
        print(f"  {'FAIL' if menus else ' ok '}  the sampler and scheduler menus",
              flush=True)
        failures += menus
    finally:
        proc.terminate()

    if failures:
        print("\n" + "\n".join("  " + f for f in failures), flush=True)
        raise SystemExit(f"\n{len(failures)} structural problem(s) in the video graphs.")
    print("\nAll video graphs are wired to nodes that exist.", flush=True)
