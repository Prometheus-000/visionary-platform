"""
A/B the two style engines on one seed, one reference, live weights.

    modal run tools/ab_style.py --style-jpg /path/to/reference.jpg

Left corner: the ostris reference-LoRA path exactly as `_krea2_graph` builds
it. Right corner: nkxx188's training-free K/V injection, wired exactly as its
shipped workflow wires it — its own sampler, its own zeroed negative, no shift
node — because an engine is judged as shipped, not squeezed into the other
one's harness.

The question this answers is the one the live 0.5-strength render raised:
the ostris path leaks reference *content* (the bridge, the suit), and the
K/V pack's whole claim is style without leakage. A claim like that is a
render, not an argument — same seed, same prompt, look at the pictures.

The pack is cloned into a derived image here and nowhere else: the deployed
app does not grow a node pack for an experiment. If this engine wins it gets
the CLIFF treatment — a pin in app.py and a require_nodes line — and this
file stays as the harness that decided it.
"""

import base64
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import (  # noqa: E402
    COMFY,
    MODEL_CATALOGUE,
    _Comfy,
    _krea2_graph,
    comfy_image,
    volume,
)

OSTRIS_REPO = "https://github.com/ostris/ComfyUI-Krea2-Ostris-Edit"
OSTRIS_SHA = "7756566160c4a1b24bb1bd9f0ff3ced1a83d7547"
# The weight the ostris path loads — still on the volume from the era it was
# the engine; the harness names it directly since the catalogue no longer does.
OSTRIS_LORA = "krea2_style_reference.safetensors"

K2ST_REPO = "https://github.com/nkxx188/ComfyUI-Krea2-StyleTransfer"
K2ST_SHA = "b30d495ab7e5626a2effc72a071430297643b718"  # 2026-07-21


def _ostris_graph(prompt: str, ref_name: str, seed: int, strength: float) -> dict:
    """The retired app branch, rebuilt here so strength can be swept."""
    dit = MODEL_CATALOGUE["turbo"]["dest"].name
    te = MODEL_CATALOGUE["text_encoder"]["dest"].name
    vae = MODEL_CATALOGUE["vae"]["dest"].name
    return {
        "dit": {"class_type": "UNETLoader",
                "inputs": {"unet_name": dit, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": te, "type": "krea2", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "latent": {"class_type": "EmptyHunyuanLatentVideo",
                   "inputs": {"width": WIDTH, "height": HEIGHT,
                              "length": 1, "batch_size": 1}},
        "stylelora": {"class_type": "LoraLoader",
                      "inputs": {"model": ["dit", 0], "clip": ["clip", 0],
                                 "lora_name": OSTRIS_LORA,
                                 "strength_model": strength,
                                 "strength_clip": strength}},
        "shift": {"class_type": "ModelSamplingAuraFlow",
                  "inputs": {"model": ["stylelora", 0], "shift": 1.15}},
        "refimg": {"class_type": "LoadImage", "inputs": {"image": ref_name}},
        "pos": {"class_type": "TextEncodeKrea2OstrisEdit",
                "inputs": {"clip": ["stylelora", 1], "prompt": prompt,
                           "vae": ["vae", 0], "image1": ["refimg", 0]}},
        "neg": {"class_type": "TextEncodeKrea2OstrisEdit",
                "inputs": {"clip": ["stylelora", 1], "prompt": ""}},
        "patch": {"class_type": "Krea2OstrisEditModelPatch",
                  "inputs": {"model": ["shift", 0], "kv_cache": False}},
        "sample": {"class_type": "KSampler",
                   "inputs": {"model": ["patch", 0], "positive": ["pos", 0],
                              "negative": ["neg", 0], "latent_image": ["latent", 0],
                              "seed": seed, "steps": STEPS, "cfg": 1.0,
                              "sampler_name": "er_sde",
                              "scheduler": "sgm_uniform", "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode",
                   "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["decode", 0], "filename_prefix": "ab_ost"}},
    }

ab = modal.App("visionary-ab-style")
# comfy_image ends in local-dir mounts, and Modal refuses run_commands after a
# mount — so the pack is cloned at container start instead, pinned by the same
# SHA a build layer would pin. Slower per run, wrong for production, fine for
# a harness that exists to decide whether production wants it at all.
ab_image = comfy_image.add_local_python_source("app")

WIDTH, HEIGHT, STEPS, SEED = 896, 1152, 8, 42


def _kv_graph(prompt: str, ref_name: str, seed: int) -> dict:
    """The pack's own workflow, translated to API format — every widget spelled."""
    dit = MODEL_CATALOGUE["turbo"]["dest"].name
    te = MODEL_CATALOGUE["text_encoder"]["dest"].name
    vae = MODEL_CATALOGUE["vae"]["dest"].name
    return {
        "dit": {"class_type": "UNETLoader",
                "inputs": {"unet_name": dit, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": te, "type": "krea2", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "latent": {"class_type": "EmptyHunyuanLatentVideo",
                   "inputs": {"width": WIDTH, "height": HEIGHT,
                              "length": 1, "batch_size": 1}},
        "refimg": {"class_type": "LoadImage", "inputs": {"image": ref_name}},
        "styleref": {"class_type": "Krea2StyleReference",
                     "inputs": {"vae": ["vae", 0], "target_latent": ["latent", 0],
                                "reference_image": ["refimg", 0],
                                "fit": "crop", "upscale_method": "lanczos"}},
        "pos": {"class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["clip", 0]}},
        # Their recommended route: the negative is the positive zeroed, not an
        # empty encode — Turbo is CFG-free and this is what their workflow ships.
        "neg": {"class_type": "ConditioningZeroOut",
                "inputs": {"conditioning": ["pos", 0]}},
        "st": {"class_type": "Krea2StyleTransfer",
               "inputs": {"model": ["dit", 0],
                          "reference_latent": ["styleref", 0],
                          "ref_conditioning": ["pos", 0],
                          "mode": "recommended",
                          # The node's own defaults, written out per the
                          # optional-inputs rule; `recommended` governs.
                          "style_strength": 1.0,
                          "value_adain_strength": 0.65,
                          "ref_value_mix": 1.0,
                          "ref_k_strength": 1.06,
                          "rf_mode": "flowturbo_pc",
                          "gamma": 0.5,
                          "beta": 2.5,
                          "high_scale_start": 1.04,
                          "high_scale_end": 0.0,
                          "low_scale_start": 1.0,
                          "low_scale_end": 1.10,
                          "adain_strength": 0.85,
                          "blocks": "7-27"}},
        "sample": {"class_type": "KSampler",
                   "inputs": {"model": ["st", 0], "positive": ["pos", 0],
                              "negative": ["neg", 0], "latent_image": ["latent", 0],
                              "seed": seed, "steps": STEPS, "cfg": 1.0,
                              "sampler_name": "euler_ancestral",
                              "scheduler": "simple", "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode",
                   "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["decode", 0], "filename_prefix": "ab_kv"}},
    }


@ab.function(image=ab_image, gpu="H100", timeout=30 * 60,
             volumes={"/workspace": volume})
def sweep_ostris(style_b64: str, prompt: str,
                 strengths: list[float]) -> dict[str, bytes]:
    """Does more LoRA strength make the ostris transfer better, or leak more?"""
    import subprocess
    for repo, sha, name in ((OSTRIS_REPO, OSTRIS_SHA, "krea2_ostris_edit"),):
        subprocess.run(
            f"git clone {repo} {COMFY}/custom_nodes/{name}"
            f" && cd {COMFY}/custom_nodes/{name} && git checkout {sha}",
            shell=True, check=True, capture_output=True)
    comfy = _Comfy("image")
    comfy.start()
    ref = comfy.stage("absweep", style_b64, "ref", ext="jpg")
    out: dict[str, bytes] = {}
    for st in strengths:
        names = comfy.run(f"sw-{st}", _ostris_graph(prompt, ref, SEED, st),
                          what="image")
        out[f"ostris-s{st}"] = (COMFY / "output" / names[0]).read_bytes()
        print(f"[sweep] strength {st} done", flush=True)
    return out


@ab.local_entrypoint()
def sweep(style_jpg: str):
    style_b64 = base64.b64encode(Path(style_jpg).read_bytes()).decode()
    results = sweep_ostris.remote(
        style_b64, "a woman walking a small dog in a city park",
        [1.0, 1.3, 1.6])
    out_dir = Path(style_jpg).parent
    for name, data in results.items():
        p = out_dir / f"ab-{name}.png"
        p.write_bytes(data)
        print(f"saved {p}")


@ab.function(image=ab_image, gpu="H100", timeout=30 * 60,
             volumes={"/workspace": volume})
def render_pairs(style_b64: str, prompts: list[str]) -> dict[str, bytes]:
    import subprocess
    for repo, sha, name in ((K2ST_REPO, K2ST_SHA, "krea2_styletransfer"),
                            (OSTRIS_REPO, OSTRIS_SHA, "krea2_ostris_edit")):
        subprocess.run(
            f"git clone {repo} {COMFY}/custom_nodes/{name}"
            f" && cd {COMFY}/custom_nodes/{name} && git checkout {sha}",
            shell=True, check=True, capture_output=True)
    comfy = _Comfy("image")
    comfy.start()
    ref = comfy.stage("abstyle", style_b64, "ref", ext="jpg")
    out: dict[str, bytes] = {}
    for i, prompt in enumerate(prompts):
        g = _krea2_graph(
            model="turbo", prompt=prompt, negative_prompt="",
            width=WIDTH, height=HEIGHT, batch_size=1, seed=SEED, steps=STEPS,
            cfg=1.0, shift=1.15, sampler="er_sde", scheduler="sgm_uniform",
            loras=[], style_refs=[ref], style_strength=1.0,
        )
        for tag, graph in (("ostris", g), ("kv", _kv_graph(prompt, ref, SEED))):
            names = comfy.run(f"ab-{tag}-{i}", graph, what="image")
            out[f"{tag}-p{i}"] = (COMFY / "output" / names[0]).read_bytes()
            print(f"[ab] {tag} p{i} done", flush=True)
    return out


@ab.local_entrypoint()
def main(style_jpg: str):
    style_b64 = base64.b64encode(Path(style_jpg).read_bytes()).decode()
    prompts = [
        "a woman walking a small dog in a city park",
        "a chef plating a dish in a restaurant kitchen",
    ]
    results = render_pairs.remote(style_b64, prompts)
    out_dir = Path(style_jpg).parent
    for name, data in results.items():
        p = out_dir / f"ab-{name}.png"
        p.write_bytes(data)
        print(f"saved {p}")
