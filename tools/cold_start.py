"""
Where a cold H3 take's time actually goes.

    modal run tools/cold_start.py
    modal run tools/cold_start.py --steps 20 --frames 124

One cold container, real image, real volume, real weights, and a clock on
each phase of the wait a session's first take pays:

  spinup — modal scheduling + container start (local clock minus remote clock)
  boot   — _Comfy.start(): ComfyUI imports, node registration, /object_info up
  load   — first run minus warm run: the checkpoint's trip volume -> VRAM
  sample — the warm run: what the same take costs once resident

Plus raw volume throughput, single-stream and 8-way parallel, read from the
ref DiT — a file the t2v graph never touches, so measuring it does not warm
the cache under the file the load phase is about to read. If `load` is the
bill, the parallel number says whether the volume has headroom a smarter
reader could take, or whether it is the pipe itself.

The fix this file exists to price is not chosen yet — snapshots, enter-time
residency, a prefetching reader. The numbers choose it; that is the point.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import (  # noqa: E402
    MODEL_CATALOGUE,
    _Comfy,
    _h3_graph,
    _prefetch_weights,
    comfy_image,
    models_volume,
    volume,
)

cs = modal.App("visionary-cold-start")
cs_image = comfy_image.add_local_python_source("app")

READ_GB = 6  # per throughput probe; enough to swamp startup effects


def _read_slice(path: Path, offset: int, length: int) -> int:
    got = 0
    with open(path, "rb", buffering=0) as f:
        f.seek(offset)
        while got < length:
            chunk = f.read(min(64 << 20, length - got))
            if not chunk:
                break
            got += len(chunk)
    return got


def _throughput(path: Path, offset: int, workers: int) -> float:
    """GB/s over READ_GB starting at offset, split across workers."""
    total = READ_GB << 30
    per = total // workers
    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda i: _read_slice(path, offset + i * per, per),
                      range(workers)))
    return total / (1 << 30) / (time.perf_counter() - t0)


# cpu=4.0 to match the generator classes exactly — the first run of the
# prefetch arm left it at Modal's default sliver, and eight reader threads on
# an eighth of a core starved the ComfyUI boot they were meant to hide behind.
# A harness that measures the app must rent what the app rents.
@cs.function(image=cs_image, gpu="H100", cpu=4.0, timeout=45 * 60,
             volumes={"/workspace": volume, "/models": models_volume})
def measure(steps: int, frames: int, prefetch: bool = False) -> dict:
    t_start = time.time()
    out: dict[str, float] = {"remote_epoch": t_start}

    # The same call, in the same place, as the generators' enter hooks —
    # this measures the app's own behaviour, not a simulation of it.
    stop = _prefetch_weights(
        [MODEL_CATALOGUE[k]["dest"]
         for k in ("h3_dit", "h3_te", "h3_vae", "h3_audio_vae",
                   "h3_ref_dit")]) if prefetch else None

    t0 = time.perf_counter()
    comfy = _Comfy("video")
    comfy.start()
    out["boot_s"] = round(time.perf_counter() - t0, 1)

    # Throughput probes on the ref DiT, sequential first, distinct slices —
    # the second probe must not re-read bytes the first left in any cache.
    # Skipped under prefetch: the probes read the file the prefetch is
    # warming, and each would corrupt the other's number.
    ref = MODEL_CATALOGUE["h3_ref_dit"]["dest"]
    if prefetch:
        out["read_probe_s"] = 0.0
    elif ref.exists() and ref.stat().st_size > (2 * READ_GB << 30):
        t0 = time.perf_counter()
        out["read_1way_gbps"] = round(_throughput(ref, 0, 1), 2)
        out["read_8way_gbps"] = round(_throughput(ref, READ_GB << 30, 8), 2)
        out["read_probe_s"] = round(time.perf_counter() - t0, 1)
    else:
        out["read_probe_s"] = 0.0

    def graph(seed: int) -> dict:
        return _h3_graph(prompt="a paper boat drifting across a puddle",
                         width=960, height=544, frames=frames, seed=seed,
                         steps=steps, sampler="res_multistep",
                         scheduler="simple")

    # Different seeds, or the second run is a ComfyUI cache hit rather than a
    # warm sample — the exact lie ab_sage.py told once.
    if stop is not None:
        stop.set()  # what generate() does on job arrival, mirrored
    t0 = time.perf_counter()
    comfy.run("cold", graph(41), what="video")
    out["first_run_s"] = round(time.perf_counter() - t0, 1)

    t0 = time.perf_counter()
    comfy.run("warm", graph(42), what="video")
    out["warm_run_s"] = round(time.perf_counter() - t0, 1)
    if out["warm_run_s"] < 5.0:
        raise RuntimeError(
            f"warm run answered in {out['warm_run_s']}s — that is ComfyUI's "
            f"cache, not a render; the seed did not miss.")

    out["load_s"] = round(out["first_run_s"] - out["warm_run_s"], 1)
    dit = MODEL_CATALOGUE["h3_dit"]["dest"]
    if dit.exists():
        out["dit_gb"] = round(dit.stat().st_size / (1 << 30), 1)
    return out


@cs.local_entrypoint()
def main(steps: int = 4, frames: int = 124, prefetch: bool = False):
    t_local = time.time()
    r = measure.remote(steps, frames, prefetch)
    r["spinup_s"] = round(r.pop("remote_epoch") - t_local, 1)

    print(f"\n== cold H3 take, {steps} steps, {frames} frames ==")
    print(f"spinup   {r['spinup_s']:7.1f}s   modal scheduling + container start")
    print(f"boot     {r['boot_s']:7.1f}s   ComfyUI to /object_info")
    print(f"load     {r['load_s']:7.1f}s   weights -> VRAM "
          f"({r.get('dit_gb', '?')} GB DiT)")
    print(f"sample   {r['warm_run_s']:7.1f}s   warm render at {steps} steps "
          f"(~{r['warm_run_s'] / max(steps, 1):.1f}s/step)")
    if "read_1way_gbps" in r:
        print(f"volume   {r['read_1way_gbps']} GB/s single stream, "
              f"{r['read_8way_gbps']} GB/s 8-way "
              f"(probe cost {r['read_probe_s']}s)")
    total = r["spinup_s"] + r["boot_s"] + r["first_run_s"]
    print(f"total    {total:7.1f}s   spinup + boot + first run")


# ---------------------------------------------------------------------------
# Volume v2 probe — is the pipe the fix?
#
# The baseline above measured the v1 volume at 0.08 GB/s single-stream and
# 0.1 GB/s 8-way: parallelism buys nothing, so the mount is the cap and no
# smarter reader on our side can beat it. Modal's v2 volumes claim up to
# 2.5 GB/s. This seeds one real weight file onto a v2 volume and reads it
# back from a fresh container — the same cold-worker read a cold start pays.

models_v2 = models_volume


@cs.function(image=cs_image, cpu=8.0, timeout=60 * 60,
             volumes={"/workspace": volume, "/v2": models_v2})
def seed_v2() -> float:
    """Copy the ref DiT onto the v2 volume once; returns GB copied."""
    import shutil
    src = MODEL_CATALOGUE["h3_ref_dit"]["dest"]
    dst = Path("/v2") / src.name
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copyfile(src, dst)
        models_v2.commit()
    return round(dst.stat().st_size / (1 << 30), 1)


@cs.function(image=cs_image, cpu=8.0, timeout=30 * 60,
             volumes={"/v2": models_v2})
def probe_v2() -> dict:
    ref = Path("/v2") / MODEL_CATALOGUE["h3_ref_dit"]["dest"].name
    return {
        "read_1way_gbps": round(_throughput(ref, 0, 1), 2),
        "read_8way_gbps": round(_throughput(ref, READ_GB << 30, 8), 2),
    }


@cs.local_entrypoint()
def v2():
    gb = seed_v2.remote()
    print(f"seeded {gb} GB onto visionary-models (v2)")
    r = probe_v2.remote()
    print(f"v2 volume: {r['read_1way_gbps']} GB/s single stream, "
          f"{r['read_8way_gbps']} GB/s 8-way")
    print("v1 measured 0.08 / 0.1 — anything at 1 GB/s or better makes the "
          "weight split the cold-start fix.")
