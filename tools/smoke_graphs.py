"""
Smoke test for every graph this app builds — all variants, on CPU, no weights.

    modal run tools/smoke_graphs.py

Checks that every graph `_krea2_graph` and `_h3_graph` can emit is
structurally valid against the ComfyUI build pinned in `comfy_image`: every
`class_type` exists, every input key is a real input on that node, and every
link points at a node in the same graph.

This was `smoke_video.py`, and covered video only because images ran on a
vendored Forge with a smoke test of its own that imported the backend. There is
one backend now, so there is one of these. The Krea 2 variants also make it the
check for the custom node pack: `/object_info` lists a node only if it
imported, so a pack that raises on import fails here on a CPU container rather
than at the first regional render on a warm H100.

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
  "minimax" or "krea2", a sampler or scheduler this deployment offers that
  ComfyUI does not have, `ref_image_size`. Those lists are compiled in, so they
  are populated with no weights present.
- It DOES catch a model-sampling node whose baked-in multiplier disagrees with
  the checkpoint's own config. That one is checked against
  `comfy.supported_models` rather than against a name, because it is the shape
  of failure names cannot reach: `ModelSamplingSD3` and `ModelSamplingAuraFlow`
  differ only in a number neither node exposes, both validate, and sending
  Krea 2 the wrong one returns coloured noise at the right step count with no
  error anywhere. See `_check_model_sampling`.
- It does NOT catch a filename that is not on the volume. Loader nodes take a
  combo too, but one whose options are the files ComfyUI can see, and with no
  weights present every one of those lists is empty. Emptiness is what
  separates the two cases, and it is what this skips on: `_require_models()`
  covers the file combos, and it covers them before the GPU is rented.
- It does not run a sampler, so it says nothing about whether a clip looks
  right. Sampling flags are structurally valid whatever they are set to;
  getting one wrong produces a washed-out clip, not an error, and only a real
  run will show it.

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
    KREA2_REGIONAL_NODE,
    SAMPLERS,
    SCHEDULERS,
    VIDEO_MODELS,
    _h3_canvas,
    _h3_frames,
    _h3_graph,
    _krea2_graph,
    comfy_image,
)

smoke = modal.App("visionary-smoke-graphs")

# `modal run tools/smoke_graphs.py` mounts tools/, not the repo root, so app.py
# has to be added explicitly — the graph builders come from there.
smoke_image = comfy_image.add_local_python_source("app")

# ComfyUI's frontend appends this to seed widgets; it is not something an API
# graph supplies, and some nodes still list it as required. Excluding it by
# name keeps the missing-required check from crying wolf on every sampler.
FRONTEND_ONLY = {"control_after_generate"}


def _variants() -> list[tuple[str, dict]]:
    """
    Every shape the two builders can produce, named for the failure it covers.

    One graph per branch, not per parameter: the point is to reach every
    `class_type` the code can emit — the keyframe nodes, the first/last node,
    the reference node — because a node name is what goes stale, and a branch
    never built is a name never checked.
    """
    out: list[tuple[str, dict]] = []
    seed, steps = 1, 4

    # ── Krea 2 images ─────────────────────────────────────────────────────
    #
    # Three branches, because three is how many shapes _krea2_graph has and a
    # branch never built is a name never checked. The regional pair is the one
    # that matters most here: it is the only place the app names a node it does
    # not own, and `_edit_lora_name` picks a combo value that has to be a member
    # of a list the node computes from the volume — which is empty here, so this
    # also pins the empty-volume fallback.
    krea = dict(model="turbo", prompt="a smoke test", negative_prompt="blurry",
                width=1216, height=832, batch_size=1, seed=seed, steps=steps,
                cfg=1.0, shift=1.15, sampler="euler", scheduler="simple")
    stack_img = [{"name": "style.safetensors", "unet": 0.8, "text_encoder": 0.7}]
    boxes = [
        {"x": 0.0, "y": 0.0, "width": 0.5, "height": 1.0,
         "prompt": "a man", "lora": "a.safetensors", "strength": 1.35},
        {"x": 0.5, "y": 0.0, "width": 0.5, "height": 1.0,
         "prompt": "a woman", "lora": "b.safetensors", "strength": 1.3},
    ]
    # A mold on one box and not the other, and the second box carries no LoRA at
    # all — the ref-only region, which is a character from a photograph with no
    # training run behind it. Both halves are worth building: `ref_image` has to
    # reach regions_json in box order, and a row whose `lora` is "None" while
    # its `ref_image` is set is the row most likely to be filtered out by
    # accident somewhere between the form and the graph.
    molded = [
        {**boxes[0], "ref_image": "gen1-region0.png"},
        {"x": 0.5, "y": 0.0, "width": 0.5, "height": 1.0,
         "prompt": "a woman", "lora": "None", "strength": 1.0,
         "ref_image": "gen1-region1.png"},
    ]
    out += [
        ("krea2 plain", _krea2_graph(**krea, loras=stack_img, regions=[])),
        ("krea2 regional", _krea2_graph(**krea, loras=stack_img, regions=boxes,
                                        region_weight=1.1)),
        ("krea2 regional + molds", _krea2_graph(**krea, loras=[], regions=molded)),
        ("krea2 krea2edit", _krea2_graph(**krea, loras=[], regions=boxes,
                                         scene="s.png", outfit="o.png")),
        # The free-role sockets and the style engine each name nodes no other
        # branch reaches — the object plates ride extra_ref_3/4 on V12, and
        # style is a different pack entirely (K2ST_*). Both were live features
        # before they were smoke variants, which is exactly the gap this file's
        # docstring warns about.
        ("krea2 krea2edit + objects", _krea2_graph(
            **krea, loras=[], regions=boxes, scene="s.png",
            objects=[{"image": "obj1.png", "note": "a motorcycle she leans against"},
                     {"image": "obj2.png", "note": "a paper lantern overhead"}])),
        # The conjured shape: a plate over zero drawn boxes rides a derived
        # full-canvas row the job manufactures from the run's first LoRA chip.
        # Beside it, the rest of the new edit surface in one graph — a
        # person-role plate (noteless on purpose: the node writes its own
        # clause), a row anchored through V9's per-row portrait flag, and the
        # two exposed numbers. One variant because they share every node name;
        # what goes stale here is an input key, not a class_type.
        ("krea2 krea2edit conjured + person + anchor", _krea2_graph(
            **krea, loras=[], scene="s.png",
            regions=[{"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0,
                      "prompt": "", "lora": "a.safetensors", "strength": 1.0,
                      "derived": True},
                     {**boxes[1], "anchor": True}],
            objects=[{"image": "p1.png", "note": "", "role": "person"}],
            edit_strength=0.5, compose_seed=7)),
        ("krea2 style reference", _krea2_graph(
            **krea, loras=[], regions=[], style_refs=["style.jpg"],
            style_strength=0.8)),
    ]

    # ── MiniMax-H3 ────────────────────────────────────────────────────────
    w, h = _h3_canvas("16:9", "draft")
    n = _h3_frames(5)
    base = dict(prompt="a smoke test", width=w, height=h, frames=n,
                seed=seed, steps=steps, sampler="res_multistep", scheduler="simple")
    h3_stack = [{"name": "one.safetensors", "unet": 0.9},
                {"name": "two.safetensors", "unet": 0.6}]
    out += [
        ("h3 t2v", _h3_graph(**base)),
        ("h3 i2v (first)", _h3_graph(**base, first_frame="a.png")),
        ("h3 first+last", _h3_graph(**base, first_frame="a.png", last_frame="b.png")),
        # The reference branch loads a different transformer and a different
        # conditioning node, and its inputs are the autogrow keys — the one
        # place in either builder where the input name is computed.
        ("h3 ref2va", _h3_graph(**base, references=["r0.png", "r1.png"],
                                ref_videos=["v0.mp4"], ref_size="match")),
        # H3 has one expert, so the stack is one chain rather than two — and
        # both the guider and the scheduler have to be moved onto its end, which
        # is the link this case exists to check. A LoRA reaching the sampler
        # while the schedule still reads the bare DiT validates perfectly.
        ("h3 t2v + loras", _h3_graph(**base, loras=h3_stack)),
        ("h3 ref2va + loras", _h3_graph(**base, references=["r0.png"],
                                        ref_size="match", loras=h3_stack)),
        # The shift node exists to be moved off the model's own 12.0/3.0 for a
        # distilled LoRA, and it has to sit *after* the stack. Absent otherwise,
        # so the plain cases above are also the check that it stays absent.
        ("h3 + loras + shift", _h3_graph(**base, loras=h3_stack,
                                         shift_video=5.0, shift_audio=2.0)),
        ("h3 + shift only", _h3_graph(**base, shift_video=5.0)),
        # `<Audio N>` is its own autogrow group, and a scene whose only
        # reference is a voice still has to reach the reference transformer.
        ("h3 ref2va + audio", _h3_graph(**base, references=["r0.png"],
                                        ref_audios=["a0.wav"], ref_size="match")),
        ("h3 audio only", _h3_graph(**base, ref_audios=["a0.wav"],
                                    ref_size="match")),
        # Motion continuation names four nodes from its own pack and rewires
        # the guider and the video muxer — the save half rides every take, the
        # load half only a continued one, so both need building.
        ("h3 t2v + save context", _h3_graph(**base, save_context_as="vidsmoke")),
        ("h3 continued", _h3_graph(**base, save_context_as="vidsmoke2",
                                   load_context_from="vidsmoke")),
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
    Every sampler and scheduler the page offers, against what exists.

    The graph variants only build one of each, but the page renders the whole
    list — so an entry that ComfyUI dropped is a menu item that raises on
    selection, hours after this passed. Checked against the sampler node each
    family actually uses, because the two are not guaranteed to agree.

    This is the check the image side most needed and never had: SAMPLERS and
    SCHEDULERS were Forge's labels ("Euler a", "Automatic") and every one of
    them is a value KSampler rejects. Nothing would have caught that until a
    render failed, because the old smoke test imported a backend instead of
    validating a graph.
    """
    bad: list[str] = []
    for field, offered_list in (("sampler", SAMPLERS), ("scheduler", SCHEDULERS)):
        inp = "sampler_name" if field == "sampler" else "scheduler"
        options = _combo(
            (schema.get("KSampler", {}).get("input", {}).get("required", {}) or {}).get(inp)
        ) or []
        for offered in offered_list:
            if offered not in options:
                bad.append(f"{field.upper()}S offers {offered!r}, which "
                           f"KSampler.{inp} does not have")

    for key, spec in VIDEO_MODELS.items():
        # H3 picks its sampler through KSamplerSelect and its scheduler through
        # BasicScheduler. This was a branch while a second family took both as
        # widgets on KSamplerAdvanced, and the loop is kept because the menus
        # are still per model — a family that samples differently changes where
        # its names are validated, not whether they are.
        where = {"sampler": ("KSamplerSelect", "sampler_name"),
                 "scheduler": ("BasicScheduler", "scheduler")}
        for field, (cls, inp) in where.items():
            options = _combo(
                (schema.get(cls, {}).get("input", {}).get("required", {}) or {}).get(inp)
            ) or []
            for offered in spec[f"{field}s"]:
                if offered not in options:
                    bad.append(f"VIDEO_MODELS[{key!r}] offers {field} {offered!r}, "
                               f"which {cls}.{inp} does not have")
    return bad


# Which multiplier each model-sampling node bakes in. Both build the same
# ModelSamplingDiscreteFlow and differ only here: `ModelSamplingAuraFlow.
# patch_aura` is one line, `return self.patch(model, shift, multiplier=1.0)`,
# against `patch`'s own default of 1000.
SAMPLING_MULTIPLIER = {"ModelSamplingSD3": 1000, "ModelSamplingAuraFlow": 1.0}

# The model config each family's graph is sampling. Named here because the
# check below has to read `sampling_settings` off the right one, and nothing in
# a graph says which checkpoint it is for. Krea 2 is the one that asks for 1.0;
# most video configs declare no multiplier at all and fall through to
# ModelSamplingDiscreteFlow's own default of 1000, which is what
# ModelSamplingSD3 bakes in — which is exactly how the wrong node got copied.
GRAPH_MODEL_CLASS = {"krea2": "Krea2"}


def _check_model_sampling(_schema: dict) -> list[str]:
    """
    The check that only ever fails silently: a valid node with the wrong number.

    `timestep(sigma) = sigma * multiplier` is what the DiT is handed, and model
    configs disagree about that multiplier — Krea 2 asks for 1.0 where most
    video families inherit 1000. Both nodes exist, both take a `shift`, both
    validate, and the graph runs to completion at the right step count and the
    right speed. Send Krea 2 the 1000 and every render is coloured noise, on
    every sampler, with and without LoRAs, and nothing anywhere raises. That
    happened, by copying the node off a neighbouring video graph where it was
    correct — a graph since deleted, which is why this check now has one family
    in it and is still worth every line.

    Nothing about that is reachable by checking names, which is what the rest of
    this file does — so it is checked against the model config itself rather
    than against a number written down twice.
    """
    # ComfyUI is a git clone at /opt/comfyui, not an installed package, so
    # `comfy` is importable only with that directory on the path — which is why
    # the server is started with `cwd=COMFY`. Every other check here asks the
    # running server over HTTP and so never needed the import; this one reads
    # the model config directly, and was the only line in the file that had
    # never executed.
    if str(COMFY) not in sys.path:
        sys.path.insert(0, str(COMFY))
    # And it has to be told it is on CPU *before* that import, because reaching
    # `supported_models` drags in `comfy.model_management`, which resolves a
    # torch device at module scope and dies with "Found no NVIDIA driver" on
    # this container. The server process is spared that by `--cpu`; nothing
    # parses argv on this path, so the same flag is set on the same `args`
    # object model_management reads.
    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    import comfy.supported_models

    bad: list[str] = []
    for graph_name, graph in _variants():
        family = "krea2" if graph_name.startswith("krea2") else None
        if family is None:          # H3 sets no shift node at all
            continue
        cfg = getattr(comfy.supported_models, GRAPH_MODEL_CLASS[family], None)
        if cfg is None:
            bad.append(f"{graph_name}: no model config named "
                       f"{GRAPH_MODEL_CLASS[family]!r} in comfy.supported_models — "
                       "it was renamed upstream, so this check is now blind")
            continue
        want = cfg.sampling_settings.get("multiplier", 1000)
        for node_id, node in graph.items():
            got = SAMPLING_MULTIPLIER.get(node["class_type"])
            if got is None:
                continue
            if got != want:
                bad.append(
                    f"{graph_name}: {node['class_type']} bakes in "
                    f"multiplier={got}, but {GRAPH_MODEL_CLASS[family]}'s config "
                    f"asks for {want}. The DiT gets timesteps {got / want:g}x "
                    f"too large and the render comes out as noise with no error. "
                    f"Use {'ModelSamplingAuraFlow' if want == 1.0 else 'ModelSamplingSD3'}.")
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
    # ComfyUI has compiled in, which is a property of the pins in comfy_image,
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
        # Named before the graphs are walked, because "no such node type" for a
        # custom node has one cause worth separating from a typo: the pack
        # raised on import and ComfyUI carried on without it. The traceback is
        # in the startup output either way, and saying which of the two it is
        # decides whether you look at CLIFF_SHA or at the graph builder.
        for node in (KREA2_REGIONAL_NODE, "VisionaryBoxes"):
            if node not in schema:
                failures.append(
                    f"custom node {node!r} did not register — it failed to "
                    "import, and its traceback is in the startup output above")
        print(f"  {'FAIL' if failures else ' ok '}  custom nodes", flush=True)
        for name, graph in _variants():
            bad = _check(name, graph, schema)
            nodes = len(graph)
            print(f"  {'FAIL' if bad else ' ok '}  {name}  ({nodes} nodes)", flush=True)
            failures += bad

        menus = _check_menus(schema)
        print(f"  {'FAIL' if menus else ' ok '}  the sampler and scheduler menus",
              flush=True)
        failures += menus

        sampling = _check_model_sampling(schema)
        print(f"  {'FAIL' if sampling else ' ok '}  the model-sampling multipliers",
              flush=True)
        failures += sampling
    finally:
        proc.terminate()

    if failures:
        print("\n" + "\n".join("  " + f for f in failures), flush=True)
        raise SystemExit(f"\n{len(failures)} structural problem(s) in the video graphs.")
    print("\nAll video graphs are wired to nodes that exist.", flush=True)


# Not `main()`. `main` is a `modal.Function` after decoration, so calling it
# here raises "'Function' object is not callable", and `main.local()` would get
# further only to spawn ComfyUI from /opt/comfyui — a path in `comfy_image`,
# not on your laptop. There is no local form of this check to fall back to.
#
# The guard is here because the failure without it was silence: run under an
# interpreter that has modal and the module imports, defines every function,
# prints nothing and exits 0. A check that reports success by not running is
# worse than no check, and this file is the only thing asserting that the
# graphs name nodes that exist.
if __name__ == "__main__":
    raise SystemExit(
        "This is a Modal function, not a local script: it drives the ComfyUI "
        f"in comfy_image at {COMFY}, which exists in the container and not "
        "here.\n\n    modal run tools/smoke_graphs.py\n")
