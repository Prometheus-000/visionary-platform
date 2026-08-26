"""
Does a step cache pay for itself on a real take? Act one was CacheDiT; act
two is TeaCache, at the production shape act one taught us to insist on.

    modal run tools/ab_cache.py::smoke     # CPU: do the packs even import
    modal run tools/ab_cache.py::main      # H100: stock vs teacache, timed

Two arms, one warm container: the H3 graph exactly as `_h3_graph` builds it,
and the same graph with `CacheDiT_MiniMax_H3_Advanced_Optimizer` spliced in
front of the guider at the pack's own H3 preset (F8/B0, threshold 0.12,
3-step warmup). Same seed on both timed runs, 20 steps — the full-quality
path, which is the only population step-skipping serves; distilled runs have
nothing to skip.

The verdict is two numbers and two files: wall-clock per arm, and the takes
themselves for eyes and ears — the audio stream rides the same cached blocks,
so listen, don't just look.

The pack is cloned at container start, pinned by SHA — the ab_style pattern:
the deployed app does not grow a node pack for an experiment. cache-dit is
pinned to the release measured here; it imports lazily inside the node, so a
broken dependency surfaces at the node call, and the CPU smoke exists to find
that for cents before an H100 finds it for dollars.

TeaCache's physics dodge what sank CacheDiT: its skip test is one rel-L1
over the *latent* — a few million elements, not 768p hidden states — and it
wraps apply_model rather than patching blocks in place, so computed steps
cost stock price and nothing persists on the resident model. The trade sits
on the other side: a skip reuses the whole previous output, cruder per skip
than DBCache's partial recompute. Known pack defects, priced in: no H3
calibration despite the README (raw rel-L1, "polynomial is Phase-2"), and
node-output caching makes a second identical execution reuse spent state —
harmless here, one execution per arm; a 3-line fix if it ever ships.

POSTSCRIPT (act one) — the 1.40x this file measured was real and did not survive
production shape: at 768p on 8-10s takes the wrapper inflated computed
steps ~50% (TaylorSeer calibrator on by default, boundary clones, residual
bookkeeping — all scaling with tensor size while skips did not), and the
pack was removed. docs/decisions.md § "CacheDiT lasted one day" is the full
account. If you are re-running this harness, run it at production shape.
"""

import base64
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import (  # noqa: E402
    COMFY,
    _Comfy,
    _h3_frames,
    _h3_graph,
    comfy_image,
    models_volume,
    volume,
)

CACHEDIT_REPO = "https://github.com/Jasonzzt/ComfyUI-CacheDiT"
CACHEDIT_SHA = "1d92bbd86ec59aa6223fe2368849b7413a1acb93"  # 2026-08-04, H3 support
CACHEDIT_PIP = "cache-dit==1.5.0"
H3_NODE = "CacheDiT_MiniMax_H3_Advanced_Optimizer"

TEACACHE_REPO = "https://github.com/Icyoung/ComfyUI-MiniMaxH3-TeaCache"
TEACACHE_SHA = "4cbb50d69c73a19a5d6ec42c5aec1989d5a04b6f"  # 2026-08-04, v0.1
TEACACHE_NODE = "MiniMaxH3TeaCache"

ab = modal.App("visionary-ab-cache")
ab_image = comfy_image.add_local_python_source("app")

# Production shape, not the draft shape act one was measured at — 1344x768
# is the canvas H3 was trained on and the tier the regression bit; the frame
# count is a real 8-second take. Act one's numbers came from 960x544x124.
STEPS, SEED = 20, 42
WIDTH, HEIGHT = 1344, 768
FRAMES = _h3_frames(8.0)
PROMPT = ("a street musician playing accordion under an awning while rain "
          "falls, passers-by hurrying past with umbrellas")


def _install_pack() -> None:
    import subprocess
    subprocess.run(
        f"pip install {CACHEDIT_PIP}"
        f" && git clone {CACHEDIT_REPO} {COMFY}/custom_nodes/cachedit"
        f" && cd {COMFY}/custom_nodes/cachedit && git checkout {CACHEDIT_SHA}"
        f" && git clone {TEACACHE_REPO} {COMFY}/custom_nodes/teacache"
        f" && cd {COMFY}/custom_nodes/teacache && git checkout {TEACACHE_SHA}",
        shell=True, check=True, capture_output=True)


def _graph(seed: int) -> dict:
    return _h3_graph(prompt=PROMPT, width=WIDTH, height=HEIGHT, frames=FRAMES,
                     seed=seed, steps=STEPS, sampler="res_multistep",
                     scheduler="simple")


def _cached_variant(graph: dict) -> dict:
    """Splice the H3 node between the model chain and its two consumers."""
    import copy
    out = copy.deepcopy(graph)
    src = out["guider"]["inputs"]["model"]
    out["cachedit"] = {"class_type": H3_NODE,
                      "inputs": {"model": src, "enable": True,
                                 "fn_blocks": 8, "bn_blocks": 0,
                                 "residual_diff_threshold": 0.12,
                                 "warmup_steps": 3, "print_summary": True}}
    out["guider"]["inputs"]["model"] = ["cachedit", 0]
    out["sigmas"]["inputs"]["model"] = ["cachedit", 0]
    return out


def _teacache_variant(graph: dict) -> dict:
    """The same splice, TeaCache's node, at its own benchmarked settings —
    an engine is judged as shipped. total_steps is the node's only view of
    the schedule; it must agree with KSampler's or the end guard drifts."""
    import copy
    out = copy.deepcopy(graph)
    src = out["guider"]["inputs"]["model"]
    out["teacache"] = {"class_type": TEACACHE_NODE,
                       "inputs": {"model": src, "rel_l1_thresh": 0.15,
                                  "start_step": 2, "end_step": -2,
                                  "total_steps": STEPS}}
    out["guider"]["inputs"]["model"] = ["teacache", 0]
    out["sigmas"]["inputs"]["model"] = ["teacache", 0]
    return out


@ab.function(image=ab_image, cpu=4.0, timeout=20 * 60)
def smoke_remote() -> str:
    """Three failure classes, priced at CPU rates: cache-dit's dependency
    tree breaking on install, its import breaking against this image's
    torch/hub versions, and the pack failing ComfyUI's node import — the
    smoke_graphs boot pattern, because /object_info lists a node only if its
    pack imported."""
    import importlib
    import json
    import subprocess
    import urllib.request

    _install_pack()
    importlib.import_module("cache_dit")

    (COMFY / "input").mkdir(parents=True, exist_ok=True)
    (COMFY / "output").mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["python", "main.py", "--listen", "127.0.0.1", "--port", "8199",
         "--disable-auto-launch", "--disable-metadata", "--cpu"],
        cwd=str(COMFY), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1)
    try:
        deadline = time.time() + 600
        while True:
            if proc.poll() is not None:
                tail = (proc.stdout.read() if proc.stdout else "") or ""
                raise SystemExit("ComfyUI exited during startup:\n" + tail[-4000:])
            if time.time() > deadline:
                raise SystemExit("ComfyUI did not become ready in 10 minutes.")
            try:
                urllib.request.urlopen(
                    "http://127.0.0.1:8199/system_stats", timeout=2).read()
                break
            except OSError:
                time.sleep(1.0)
        with urllib.request.urlopen(
                "http://127.0.0.1:8199/object_info", timeout=120) as r:
            schema = json.loads(r.read())
        missing = [n for n in (H3_NODE, TEACACHE_NODE) if n not in schema]
        if missing:
            raise SystemExit(
                f"{missing} missing from /object_info — that pack raised on "
                f"import; the traceback is in the startup log above.")
        return f"cache_dit imports; {H3_NODE} and {TEACACHE_NODE} registered"
    finally:
        proc.terminate()


@ab.local_entrypoint()
def smoke():
    print(smoke_remote.remote())


@ab.function(image=ab_image, gpu="H100", timeout=60 * 60,
             volumes={"/workspace": volume, "/models": models_volume})
def render() -> dict:
    _install_pack()
    comfy = _Comfy("video")
    comfy.start()
    comfy.require_nodes(H3_NODE, TEACACHE_NODE)

    # Weights resident before either clock starts, on a seed neither timed
    # run reuses — an identical graph would come back as a cache hit.
    comfy.run("warm", _graph(SEED - 1), what="video")

    out: dict[str, object] = {}
    files: dict[str, bytes] = {}
    for arm, graph in (("stock", _graph(SEED)),
                       ("teacache", _teacache_variant(_graph(SEED)))):
        t0 = time.perf_counter()
        names = comfy.run(arm, graph, what="video")
        out[f"{arm}_s"] = round(time.perf_counter() - t0, 1)
        files[arm] = (COMFY / "output" / names[0]).read_bytes()
    out["files"] = {k: base64.b64encode(v).decode() for k, v in files.items()}
    return out


@ab.local_entrypoint()
def main(out: str = "."):
    r = render.remote()
    for arm, b64 in r["files"].items():
        p = Path(out) / f"ab-cache-{arm}.mp4"
        p.write_bytes(base64.b64decode(b64))
        print(f"saved {p}")
    base = r["stock_s"]
    for arm in [k[:-2] for k in r if k.endswith("_s") and k != "stock_s"]:
        t = r[arm + "_s"]
        print(f"\nstock        {base}s")
        print(f"{arm:12s} {t}s   ({base / t:.2f}x, saves {base - t:.1f}s per take)")
