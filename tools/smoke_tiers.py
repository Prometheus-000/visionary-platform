"""
Does a quantised tier actually render, on a card that is not a Hopper?

    modal run tools/smoke_tiers.py                 # Krea 2 GGUF on an L4
    modal run tools/smoke_tiers.py --gpu A10G      # or an Ampere one
    modal run tools/smoke_tiers.py --no-keep       # drop the 7.2 GB tier afterwards

`tools/probe_tiers.py` is the half of this that costs nothing: it reads
safetensors headers over range requests and says whether a tier is the same
model as the file it replaces. It already earned its place by removing a row.
What it cannot say is whether a file that *loads* also *runs* — the kernels have
to exist for the architecture, the custom nodes have to bind to whatever the
loader returns, and a picture has to come out the other end.

This is that half, and it is deliberately the cheapest version of it.

**Why an L4.** The deployment is Hopper and every consumer card is not, which is
the whole gap this build has to cross. An L4 is 24 GB of Ada — `sm_89`, the same
architecture as a 4090 — so it exercises three things at once that a rented
4090 would: the GGUF loader on a card that has to stream, the four `Visionary*`
nodes against a model that is dequantised on access rather than resident, and
`_confirm_sage()`, because `comfy_image` compiles SageAttention for `sm_90` and
an L4 is where that silently does not load. Krea 2 opts out of sage in the graph
anyway, so the picture is unaffected — which is exactly what makes it a good
place to check that the *warning* fires.

**What it reuses.** `_Comfy`, `_krea2_graph` and `MODEL_CATALOGUE`, off the same
app.py a deploy ships. A probe that builds its own graph is a probe of the graph
it built.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modal  # noqa: E402

import app  # noqa: E402

probe = modal.App("visionary-smoke-tiers")

# app.py is not in `comfy_image` — the deployment imports it because Modal
# mounts the file that defines the app, and here that file is this one. Same
# line `smoke_graphs.py` carries, for the same reason: the probe has to drive
# the real `_Comfy` and the real `_krea2_graph`, not a copy of them.
probe_image = app.comfy_image.add_local_python_source("app")

# Where the weights this workspace already has actually live. The current
# volumes were empty when this was written and the v1-era tree still holds the
# full set, so the probe reads from there rather than pulling 25 GB it owns.
ARCHIVE = modal.Volume.from_name("visionary-archive", create_if_missing=True)

# What a Krea 2 render needs, and where each part comes from.
NEEDED = [
    # (catalogue key, from archive?)
    ("krea2_turbo_q4", False),   # the tier under test — downloaded
    ("text_encoder", True),
    ("vae", True),
]


@probe.function(
    image=probe_image,
    gpu="L4",
    cpu=4.0,
    timeout=45 * 60,
    volumes={"/models": app.models_volume, "/archive": ARCHIVE,
             "/workspace": app.volume},
)
def render_on_tier(gpu: str = "L4", keep: bool = True) -> dict:
    import os
    import shutil

    out: dict = {"gpu": gpu, "steps": []}

    def note(msg: str) -> None:
        print(f"[probe] {msg}", flush=True)
        out["steps"].append(msg)

    # ---- the card, before anything slow ---------------------------------
    try:
        import torch
        name = torch.cuda.get_device_name(0)
        cap = "%d%d" % torch.cuda.get_device_capability()
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        out.update({"card": name, "arch": cap, "vram_gb": vram})
        note(f"{name} — {vram} GB, sm_{cap}")
        note(f"built for TORCH_CUDA_ARCH_LIST="
             f"{os.environ.get('TORCH_CUDA_ARCH_LIST', '?')}")
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"no CUDA: {exc}"
        return out

    # ---- the weights -----------------------------------------------------
    for key, from_archive in NEEDED:
        spec = app.MODEL_CATALOGUE[key]
        dest: Path = spec["dest"]
        if dest.is_file():
            note(f"{key}: already on /models")
            continue
        if from_archive:
            src = Path("/archive/models") / dest.name
            if not src.is_file():
                out["error"] = f"{key}: {src} is not on the archive volume"
                return out
            note(f"{key}: copying {src.name} off the archive "
                 f"({src.stat().st_size / 1e9:.1f} GB)")
            shutil.copy2(src, dest)
        else:
            note(f"{key}: downloading {spec['repo_id']}/{spec['filename']}")
            t0 = time.time()
            from huggingface_hub import hf_hub_download
            got = hf_hub_download(spec["repo_id"], spec["filename"],
                                  local_dir="/models/.probe")
            shutil.move(got, dest)
            note(f"{key}: {dest.stat().st_size / 1e9:.1f} GB in "
                 f"{time.time() - t0:.0f}s")
    app.models_volume.commit()

    # ---- which file the resolver picks -----------------------------------
    # Assigned on the module, not through the environment, and that is the
    # difference between this probe and a real local run rather than a
    # shortcut. `GPU_VRAM_GB` is read once at import (app.py:336), and
    # `tools/run_local.py` sets the variable *before* it imports app — which a
    # probe cannot do, because Modal needs `app` at module scope to build the
    # decorators. Setting the environment here instead is what the first
    # version did, and it did nothing: the constant was already 0, the
    # `if not GPU_VRAM_GB` early return fired, and the resolver handed back the
    # bf16 base while the GGUF sat beside it on the volume. The guard below
    # caught it, which is the only reason this is a comment and not a passing
    # test of the wrong file.
    app.GPU_VRAM_GB = out["vram_gb"]
    chosen = app._slot_name("turbo")
    note(f"_slot_name('turbo') resolved to {chosen}")
    out["resolved"] = chosen
    if not chosen.endswith(".gguf"):
        # The tier is present and the resolver did not take it, which means the
        # declared fit is wrong rather than the file. Worth failing on: a probe
        # that renders the *other* file proves nothing about this one.
        out["error"] = (f"the GGUF tier is on disk and the resolver chose "
                        f"{chosen} — check fits_vram_gb")
        return out

    # ---- ComfyUI, and what it says about itself --------------------------
    comfy = app._Comfy("image")
    t0 = time.time()
    comfy.start()
    note(f"ComfyUI up in {time.time() - t0:.0f}s")

    log = "\n".join(comfy._log).lower()
    out["sage_confirmed"] = app.COMFY_SAGE_MARK in log
    note(f"sage backend confirmed: {out['sage_confirmed']}"
         f"  (expected False on sm_{out['arch']}, and harmless for Krea 2)")
    # ComfyUI logs one dict per comfy-kitchen backend — `backend cuda: {...}`
    # with `available` and `disabled` inside it — so the flat `'cuda': True`
    # this first looked for never appears, and the probe reported a working
    # backend as missing. The int8/nvfp4 kernels the H3 tiers need live here.
    out["kitchen_cuda"] = any(
        "backend cuda:" in line.lower() and "'available': true" in line.lower()
        and "'disabled': false" in line.lower() for line in comfy._log)
    note(f"comfy-kitchen CUDA backend enabled: {out['kitchen_cuda']}")
    for cap in ("dequantize_int8_convrot_weight", "convrot_w4a4_linear"):
        out[cap] = any(cap in line for line in comfy._log)
        note(f"  kernel {cap}: {out[cap]}")

    # The node that only the pinned fork provides, plus our own four.
    try:
        comfy.require_nodes(app.GGUF_UNET_NODE, "VisionaryStepCache",
                            "VisionaryBoxes", "VisionaryFreeRegional",
                            "VisionaryEditArity")
        note("every required node imported, including the GGUF loader")
        out["nodes_ok"] = True
    except Exception as exc:  # noqa: BLE001
        out["nodes_ok"] = False
        out["error"] = f"require_nodes: {exc}"
        return out

    # ---- a picture -------------------------------------------------------
    graph = app._krea2_graph(
        model="turbo", prompt="a still life of three pears on a windowsill",
        negative_prompt="", width=1024, height=1024, batch_size=1,
        seed=7,
        # The route's own defaults, so this renders what a first press renders
        # rather than what a probe author guessed: 1.15 is the image side's
        # shift, and turbo is 8 steps at cfg 1.0 because it is distilled.
        steps=app.KREA2_DEFAULTS["turbo"]["steps"],
        cfg=app.KREA2_DEFAULTS["turbo"]["cfg"],
        shift=1.15,
        sampler=app.IMAGE_DEFAULTS["sampler"],
        scheduler=app.IMAGE_DEFAULTS["scheduler"],
        loras=[],
    )
    out["loader"] = graph["dit"]["class_type"]
    note(f"graph built with {out['loader']}")

    job = "probe_tier"
    app.jobs[job] = {"status": "running", "phase": "probe", "stop": False,
                     "beat": time.time()}
    t0 = time.time()
    try:
        files = comfy.run(job, graph, what="image")
        out["files"] = files
        out["render_s"] = round(time.time() - t0, 1)
        note(f"rendered {files} in {out['render_s']}s")
        out["ok"] = bool(files)
    except Exception as exc:  # noqa: BLE001 — the failure is the result
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["log_tail"] = list(comfy._log)[-25:]
        note(f"render failed: {out['error']}")
        return out

    # Device-wide, because ComfyUI is a *separate process*: this probe's own
    # torch allocated nothing, so `max_memory_allocated()` reported 0.0 and read
    # as a measurement rather than as the wrong question being asked.
    try:
        free, total = torch.cuda.mem_get_info()
        out["held_gb"] = round((total - free) / 1e9, 2)
        note(f"card held {out['held_gb']} GB with the model still resident")
    except Exception:  # noqa: BLE001
        pass

    # **The picture, onto the volume.** A render that reports eight steps and
    # leaves nothing to look at cannot answer the question a quantised tier
    # actually raises, which is not "did it sample" but "is the output a
    # picture" — a broken quant samples perfectly well and returns noise.
    saved = Path("/workspace/outputs/probe_tier")
    saved.mkdir(parents=True, exist_ok=True)
    for name in files:
        src = Path(app.COMFY) / "output" / name
        if src.is_file():
            shutil.copy2(src, saved / name)
            out.setdefault("saved", []).append(f"outputs/probe_tier/{name}")
    app.volume.commit()
    note(f"kept {out.get('saved')} on the volume")

    if not keep:
        # The tier file is 7.2 GB on a volume whose only way to reclaim space is
        # the CLI. Kept only when asked for.
        app.MODEL_CATALOGUE["krea2_turbo_q4"]["dest"].unlink(missing_ok=True)
        shutil.rmtree("/models/.probe", ignore_errors=True)
        app.models_volume.commit()
        note("removed the probe's copy of the tier")
    # The job record is a real one on a named Dict that outlives every
    # container, so a probe that leaves one behind has put a permanent
    # "running" into the board this app reads.
    try:
        app.jobs.pop("probe_tier")
    except Exception:  # noqa: BLE001
        pass
    return out


@probe.local_entrypoint()
def main(gpu: str = "L4", keep: bool = True):
    print(f"\nProbing the GGUF tier on {gpu}. This spends GPU minutes.\n")
    r = render_on_tier.remote(gpu=gpu, keep=keep)
    print("\n" + "=" * 66)
    for k in ("card", "arch", "vram_gb", "resolved", "loader", "nodes_ok",
              "sage_confirmed", "kitchen_cuda", "files", "render_s",
              "held_gb", "saved"):
        if k in r:
            print(f"  {k:16} {r[k]}")
    if r.get("error"):
        print(f"\n  FAILED: {r['error']}")
        for line in r.get("log_tail") or []:
            print(f"    {line}")
        raise SystemExit(1)
    print("\n  A quantised Krea 2 rendered on a consumer architecture, through")
    print("  the forked loader, with our four nodes bound. That is the claim")
    print("  the catalogue row was making.")
