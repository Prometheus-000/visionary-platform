"""
What Krea 2 pays for leaving SageAttention, and what it stops paying in pictures.

    modal run tools/ab_sage.py
    modal run tools/ab_sage.py --prompt "..." --seed 1234

`_krea2_graph` puts a `ModelAttentionBackend` node first on the model chain, so
every Krea 2 render is on PyTorch attention while `--use-sage-attention` stays
argv for H3. That was decided on a mechanism and a pile of upstream reports —
the sm90 kernel quantizes V to FP8 with one scale per channel and runs with
both of its outlier mitigations off — and a mechanism is not a measurement.
This is the measurement: same seed, same prompt, one warm container, the graph
as built and the graph with the node cut out.

The timing is the point that needed a number. The blotches are the point that
needs eyes, and they are intermittent, so a clean pair here proves nothing on
its own — pass the prompt and seed of a render that actually blotched and look
at the two files.
"""

import base64
import copy
import statistics
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import (  # noqa: E402
    COMFY,
    IMAGE_DEFAULTS,
    KREA2_DEFAULTS,
    _Comfy,
    _krea2_graph,
    comfy_image,
    volume,
)

ab = modal.App("visionary-ab-sage")
ab_image = comfy_image.add_local_python_source("app")

WIDTH, HEIGHT, SEED, REPEATS = 1024, 1024, 42, 2
MODEL = "turbo"

# Three sentences rather than one, because the kernel's error is data-dependent
# — a per-channel FP8 scale is set by whatever the largest activation in that
# channel turns out to be, and what sets it is the prompt. A single prompt
# measures a single draw.
PROMPTS = [
    "a woman walking a small dog in a city park, overcast afternoon",
    "a cluttered watchmaker's bench under a bright lamp, close on the tools",
    "a lone figure at the end of a long neon-lit corridor at night",
]


def _sage_variant(graph: dict) -> dict:
    """
    The same graph with the opt-out cut out — what this used to build.

    Deep, not shallow. A shallow copy shares the node dicts, so rewiring the
    links here rewired them in the arm this is supposed to be compared against:
    both arms ran the same graph, ComfyUI pruned the orphaned node before
    hashing, and the second arm came back as a 0.00s cache hit that looked
    like a 30x speedup.
    """
    out = {k: copy.deepcopy(v) for k, v in graph.items() if k != "attn"}
    for node in out.values():
        for key, val in node["inputs"].items():
            if isinstance(val, list) and len(val) == 2 and val[0] == "attn":
                node["inputs"][key] = ["dit", val[1]]
    return out


@ab.function(image=ab_image, gpu="H100", timeout=30 * 60,
             volumes={"/workspace": volume})
def compare(prompts: list[str], seed: int) -> dict:
    comfy = _Comfy("image")
    comfy.start()

    def build(prompt: str, seed: int) -> dict:
        return _krea2_graph(
            model=MODEL, prompt=prompt, negative_prompt="",
            width=WIDTH, height=HEIGHT, batch_size=1, seed=seed,
            steps=KREA2_DEFAULTS[MODEL]["steps"], cfg=KREA2_DEFAULTS[MODEL]["cfg"],
            shift=1.15, sampler=IMAGE_DEFAULTS["sampler"],
            scheduler=IMAGE_DEFAULTS["scheduler"], loras=[],
        )

    # One render before the clock starts. The first one pays a ~35 GB load and
    # a torch autotune, and averaging that into either arm would say more about
    # which arm went first than about attention. Its own seed, so it does not
    # leave a cache entry the first timed render would land on.
    comfy.run("warm", build(prompts[0], seed - 1), what="image")

    times: dict[str, list[float]] = {"pytorch": [], "sage": []}
    images: dict[str, bytes] = {}
    for i, prompt in enumerate(prompts):
        for rep in range(REPEATS):
            # A repeat needs its own seed or it is not a repeat: ComfyUI caches
            # by graph signature, so a second run of an identical graph returns
            # the first one's files in 0.2s. Both arms share the seed within a
            # repeat, which is what keeps the two pictures comparable.
            base = build(prompt, seed + rep)
            arms = {"pytorch": base, "sage": _sage_variant(base)}
            # Alternated per repeat, so a container that drifts warmer or
            # cooler across the run does not hand the drift to one arm.
            order = ("pytorch", "sage") if rep % 2 == 0 else ("sage", "pytorch")
            for arm in order:
                t0 = time.perf_counter()
                names = comfy.run(f"{arm}-{i}-{rep}", arms[arm], what="image")
                dt = time.perf_counter() - t0
                # A cached render answers in ~0.2s and looks like a win. It is
                # how this harness lied the first time it ran, so it raises
                # here rather than averaging a cache hit into a median.
                if dt < 1.0:
                    raise RuntimeError(
                        f"{arm} p{i} rep{rep} answered in {dt:.2f}s — that is "
                        f"ComfyUI's cache, not a render. The two arms are "
                        f"hashing the same, or the seed did not move.")
                times[arm].append(dt)
                if rep == 0:
                    images[f"{i}-{arm}"] = (COMFY / "output" / names[0]).read_bytes()
                print(f"[ab-sage] p{i} {arm} rep{rep} {dt:.2f}s", flush=True)

    return {
        "times": times,
        "images": {k: base64.b64encode(v).decode() for k, v in images.items()},
    }


@ab.local_entrypoint()
def main(prompt: str = "", seed: int = SEED, out: str = "."):
    prompts = [prompt] if prompt else PROMPTS
    res = compare.remote(prompts, seed)

    out_dir = Path(out)
    for name, b64 in res["images"].items():
        p = out_dir / f"ab-sage-{name}.png"
        p.write_bytes(base64.b64decode(b64))
        print(f"saved {p}")

    py, sg = res["times"]["pytorch"], res["times"]["sage"]
    mp, ms = statistics.median(py), statistics.median(sg)
    print(f"\npytorch attention: median {mp:.2f}s  {[round(t, 2) for t in py]}")
    print(f"sage attention:    median {ms:.2f}s  {[round(t, 2) for t in sg]}")
    print(f"cost of the switch: {mp - ms:+.2f}s per render "
          f"({(mp / ms - 1) * 100:+.1f}%)")
