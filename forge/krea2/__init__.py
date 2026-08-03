"""
Krea 2 inference on sd-webui-forge-classic's backend, headless.

    from krea2 import Krea2Pipeline, GenerateRequest, LoraSpec

    pipe = Krea2Pipeline(dit_path=..., vae_path=..., text_encoder_path=...)
    images = pipe.generate(GenerateRequest(
        prompt="a photo of sks dog",
        loras=[LoraSpec("/workspace/models/Lora/dog/dog.safetensors", unet=0.8)],
    ))

No Gradio, no `launch.py`, no webui bootstrap — see `bootstrap.py` for the only
setup step, and `../VENDOR.md` for what was copied from Forge and what changed.
"""

from .pipeline import (
    GenerateRequest,
    Krea2Pipeline,
    LoraSpec,
    Region,
    columns,
    rows,
    unload_all,
)
from .sampling import (
    DEFAULT_SAMPLER,
    DEFAULT_SCHEDULER,
    sampler_names,
    scheduler_names,
)

__all__ = [
    "Krea2Pipeline",
    "GenerateRequest",
    "LoraSpec",
    "Region",
    "columns",
    "rows",
    "unload_all",
    "sampler_names",
    "scheduler_names",
    "DEFAULT_SAMPLER",
    "DEFAULT_SCHEDULER",
]
