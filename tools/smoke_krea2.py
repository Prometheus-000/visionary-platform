"""
Smoke test for the Forge-backed Krea 2 pipeline.

    modal run tools/smoke_krea2.py                 # imports only, on a T4
    modal run tools/smoke_krea2.py --gpu           # load the model and render

The import check is the cheap one and catches most of what breaks after a forge
sync: a missing dependency, a `modules` shim that no longer covers what
`backend/` reaches for, a vendor patch that got lost. It still needs a GPU —
`backend/memory_management.py` calls `torch.cuda.current_device()` at module
scope, so there is no CPU-only path into the backend — but a T4 is enough since
no weights are touched.
"""

import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import (  # noqa: E402
    FORGE,
    LORAS,
    TE_PATH,
    TURBO_PATH,
    VAE_PATH,
    GPU,
    inference_image,
    volume,
)

smoke = modal.App("visionary-smoke")

# `modal run tools/smoke_krea2.py` mounts tools/, not the repo root, so app.py
# has to be added explicitly — this module reads its paths and image from there.
smoke_image = inference_image.add_local_python_source("app")


@smoke.function(image=smoke_image, gpu="T4", cpu=2.0, timeout=600)
def check_imports() -> dict:
    """No weights — just prove the vendored tree imports and is still patched."""
    sys.path.insert(0, str(FORGE))

    import krea2
    from backend.diffusion_engine.krea import Krea2
    from backend.nn.krea import SingleStreamDiT
    from backend.loader import forge_loader  # noqa: F401
    from huggingface_guess import model_list

    import inspect

    source = inspect.getsource(SingleStreamDiT.forward)
    if "krea2_regional" not in source:
        raise AssertionError("backend/nn/krea.py lost its regional vendor patch")

    hf_config = Path(FORGE) / "backend" / "huggingface" / "krea" / "Krea-2-Raw" / "model_index.json"
    if not hf_config.is_file():
        raise AssertionError(f"missing Krea 2 component configs at {hf_config}")

    return {
        "samplers": krea2.sampler_names(),
        "schedulers": krea2.scheduler_names(),
        "engine": Krea2.__name__,
        "guess": model_list.Krea2.huggingface_repo,
        "vendor_patch": True,
    }


@smoke.function(
    image=smoke_image, gpu=GPU, cpu=2.0, timeout=45 * 60,
    volumes={"/workspace": volume},
)
def render(prompt: str, lora: str = "", steps: int = 8) -> dict:
    """Load Krea 2 Turbo off the volume and render one small image."""
    sys.path.insert(0, str(FORGE))

    import time

    from krea2 import GenerateRequest, Krea2Pipeline, LoraSpec, columns

    volume.reload()

    started = time.time()
    pipe = Krea2Pipeline(
        dit_path=TURBO_PATH, vae_path=VAE_PATH, text_encoder_path=TE_PATH, key="turbo"
    )
    load_s = round(time.time() - started, 1)

    specs = []
    if lora:
        candidates = sorted(LORAS.rglob("*.safetensors")) if lora == "any" else [Path(lora)]
        if candidates:
            specs = [LoraSpec(str(candidates[0]), unet=0.8)]

    started = time.time()
    images = pipe.generate(GenerateRequest(
        prompt=prompt, width=768, height=768, steps=steps, seed=1234, loras=specs,
    ))
    first_s = round(time.time() - started, 1)

    # Second call proves the model stayed warm and that restacking is cheap.
    started = time.time()
    pipe.generate(GenerateRequest(prompt=prompt, width=768, height=768, steps=steps, seed=1235, loras=specs))
    second_s = round(time.time() - started, 1)

    out = Path("/workspace/outputs/smoke")
    out.mkdir(parents=True, exist_ok=True)
    images[0].save(out / "smoke.png")

    result = {
        "load_s": load_s, "first_s": first_s, "warm_s": second_s,
        "size": images[0].size, "report": pipe.last_report,
    }

    # Stacking: same LoRA twice at different strengths. Not a useful image, but
    # it proves patches compose onto clones instead of overwriting each other.
    if specs:
        stacked = [LoraSpec(specs[0].path, unet=0.5, text_encoder=0.2),
                   LoraSpec(specs[0].path, unet=0.3, text_encoder=0.1)]
        pipe.generate(GenerateRequest(prompt=prompt, width=512, height=512, steps=4, seed=7, loras=stacked))
        result["stack"] = pipe.last_report["loras"]

    # Regional: two vertical strips with different prompts. The assertion that
    # matters is that the attention bias is built and sampled through at all.
    started = time.time()
    regional = pipe.generate(GenerateRequest(
        prompt="a wide landscape photograph",
        regions=columns(["a snowy pine forest on the left", "a red desert canyon on the right"]),
        width=768, height=512, steps=steps, seed=99,
    ))
    regional[0].save(out / "smoke_regional.png")
    result["regional_s"] = round(time.time() - started, 1)
    result["regional"] = pipe.last_report["regions"]

    volume.commit()
    return result


@smoke.local_entrypoint()
def main(gpu: bool = False, prompt: str = "a red fox in a snowy forest, photograph", lora: str = ""):
    print(check_imports.remote())
    if gpu:
        print(render.remote(prompt, lora))
