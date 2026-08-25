"""
Does CacheDiT's H3 DBCache pay for itself on a real take?

    modal run tools/ab_cache.py::smoke     # CPU: does the pack even import
    modal run tools/ab_cache.py::main      # H100: the two takes, timed

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
    _h3_graph,
    comfy_image,
    models_volume,
    volume,
)

CACHEDIT_REPO = "https://github.com/Jasonzzt/ComfyUI-CacheDiT"
CACHEDIT_SHA = "1d92bbd86ec59aa6223fe2368849b7413a1acb93"  # 2026-08-04, H3 support
CACHEDIT_PIP = "cache-dit==1.5.0"
H3_NODE = "CacheDiT_MiniMax_H3_Advanced_Optimizer"

ab = modal.App("visionary-ab-cache")
ab_image = comfy_image.add_local_python_source("app")

WIDTH, HEIGHT, FRAMES, STEPS, SEED = 960, 544, 124, 20, 42
PROMPT = ("a street musician playing accordion under an awning while rain "
          "falls, passers-by hurrying past with umbrellas")


def _install_pack() -> None:
    import subprocess
    subprocess.run(
        f"pip install {CACHEDIT_PIP}"
        f" && git clone {CACHEDIT_REPO} {COMFY}/custom_nodes/cachedit"
        f" && cd {COMFY}/custom_nodes/cachedit && git checkout {CACHEDIT_SHA}",
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
        if H3_NODE not in schema:
            raise SystemExit(
                f"{H3_NODE} missing from /object_info — the pack raised on "
                f"import; the traceback is in the startup log above.")
        return f"cache_dit imports; {H3_NODE} registered"
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
    comfy.require_nodes(H3_NODE)

    # Weights resident before either clock starts, on a seed neither timed
    # run reuses — an identical graph would come back as a cache hit.
    comfy.run("warm", _graph(SEED - 1), what="video")

    out: dict[str, object] = {}
    files: dict[str, bytes] = {}
    for arm, graph in (("stock", _graph(SEED)),
                       ("cachedit", _cached_variant(_graph(SEED)))):
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
    s, c = r["stock_s"], r["cachedit_s"]
    print(f"\nstock     {s:7.1f}s")
    print(f"cachedit  {c:7.1f}s   ({s / c:.2f}x, saves {s - c:.1f}s per take)")
