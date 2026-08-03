"""
Krea 2 text-to-image on Forge's backend, with no webui around it.

Replaces the previous `musubi_tuner/krea2_generate_image.py` subprocess. That
script reloaded ~35 GB of weights from the volume on every single image and
offered one LoRA at one strength. This keeps the model resident between
requests and gives you Forge's LoRA stacking, sampler and schedule choice.

What is reused unchanged from Forge:

  `backend.loader.forge_loader`  weight loading, dtype selection, quant detection
  `backend.diffusion_engine.krea.Krea2`  the Krea 2 engine (Qwen3-VL text path)
  `backend.nn.krea.SingleStreamDiT`      the optimised DiT
  `backend.memory_management`            offloading, partial loads, VRAM budget
  `backend.sampling.sampling_function`   CFG batching and the cond/uncond split
  `k_diffusion.sampling`                 the samplers

What is written here is only the driver that `modules/processing.py` would
otherwise provide: noise, conditioning, the sampling loop, and VAE decode.
"""

import logging
import os
import time
from dataclasses import dataclass, field

from . import bootstrap

bootstrap.install()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from backend import memory_management  # noqa: E402
from backend.args import dynamic_args  # noqa: E402
from backend.loader import forge_loader  # noqa: E402
from backend.logging import setup_logger  # noqa: E402
from backend.sampling.condition import ConditionCrossAttn  # noqa: E402
from backend.sampling.sampling_function import (  # noqa: E402
    sampling_cleanup,
    sampling_function_inner,
    sampling_prepare,
)
from modules.prompt_parser import SdConditioning  # noqa: E402

from . import loras, sampling  # noqa: E402
from .regional import Region, RegionalMask, columns, rows  # noqa: E402

logger = logging.getLogger("krea2")
setup_logger(logger)

__all__ = ["Krea2Pipeline", "GenerateRequest", "LoraSpec", "Region", "columns", "rows", "unload_all"]

LoraSpec = loras.LoraSpec

# Krea 2's VAE is the Qwen/Wan 16-channel one at 8x, and the DiT carries a
# frame axis it immediately squeezes. Latents are therefore (B, 16, 1, H/8, W/8).
LATENT_DOWNSCALE = 8
LATENT_FRAMES = 1


def unload_all() -> None:
    """
    Evict everything Forge has on the GPU.

    Call before constructing a second `Krea2Pipeline`. Dropping the Python
    reference is not enough: `memory_management` keeps its own list of loaded
    models, so the outgoing checkpoint's ~24 GB stays resident and the incoming
    one OOMs on a 40 GB card.
    """
    memory_management.unload_all_models()
    memory_management.soft_empty_cache(force=True)


class _StrictCrossAttn(ConditionCrossAttn):
    """
    Only batch cond and uncond together when their token counts match exactly.

    Upstream `ConditionCrossAttn.concat` pads shorter conditionings by repeating
    them with a 3-argument `Tensor.repeat`. Krea 2's conditioning is 4-D
    (batch, tokens, 12 tap layers, 2560), so that call raises whenever the
    positive and negative prompts tokenise to different lengths with a small
    LCM ratio. Refusing the mismatched concat costs one extra forward pass at
    CFG > 1 and removes the crash. At CFG == 1 — the Turbo default — Forge drops
    the uncond pass entirely, so this costs nothing at all.
    """

    def can_concat(self, other) -> bool:
        return self.cond.shape == other.cond.shape


@dataclass
class GenerateRequest:
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int | None = None
    cfg_scale: float | None = None
    seed: int | None = None
    batch_size: int = 1
    sampler: str = sampling.DEFAULT_SAMPLER
    scheduler: str = sampling.DEFAULT_SCHEDULER
    shift: float = 1.15
    loras: list = field(default_factory=list)
    regions: list = field(default_factory=list)
    region_weight: float = 1.0
    """Weight applied to the base prompt inside regional mode."""


class Krea2Pipeline:
    """
    A loaded Krea 2 checkpoint.

    Construct once per process and call `generate` repeatedly; the DiT, VAE and
    text encoder stay resident and `memory_management` handles moving them on
    and off the GPU. `key` identifies which checkpoint is loaded so a caller can
    decide whether to swap.
    """

    def __init__(self, *, dit_path, vae_path, text_encoder_path, key: str = ""):
        self.key = key or str(dit_path)
        self.dit_path = str(dit_path)

        for label, path in (("DiT", dit_path), ("VAE", vae_path), ("text encoder", text_encoder_path)):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Krea 2 {label} not found: {path}")

        started = time.time()
        logger.info("Loading Krea 2 from %s", dit_path)
        self.sd_model = forge_loader(str(dit_path), additional_state_dicts=[str(vae_path), str(text_encoder_path)])
        logger.info("Loaded in %.1fs", time.time() - started)

        self._lora_key: tuple = ()
        self._lora_report: list[dict] = []
        self.last_report: dict = {}

    # region conditioning

    def _encode(self, texts: list[str], *, negative: bool) -> torch.Tensor:
        """Run the Qwen3-VL text path and stack to (B, tokens, 12, 2560)."""
        conditioning = SdConditioning(texts, is_negative_prompt=negative)
        zs = self.sd_model.get_learned_conditioning(conditioning)
        return torch.stack(zs)

    @staticmethod
    def _as_cond(tensor: torch.Tensor) -> list[dict]:
        return [{"cross_attn": tensor, "model_conds": {"c_crossattn": _StrictCrossAttn(tensor)}}]

    def _build_conditioning(self, req: GenerateRequest):
        """
        Returns (cond, uncond, regional_mask).

        Plain mode encodes one positive prompt. Regional mode encodes the base
        prompt plus each region separately and concatenates them along the token
        axis, recording the span lengths so `RegionalMask` can address them.
        """
        if not req.regions:
            cond = self._encode([req.prompt], negative=False)
            return cond, self._encode([req.negative_prompt], negative=True), None

        base_len = 0
        parts: list[torch.Tensor] = []

        if req.prompt.strip():
            base = self._encode([req.prompt], negative=False)
            base_len = base.shape[1]
            parts.append(base)

        lengths: list[int] = []
        for region in req.regions:
            encoded = self._encode([region.prompt], negative=False)
            lengths.append(encoded.shape[1])
            parts.append(encoded)

        cond = torch.cat(parts, dim=1)
        mask = RegionalMask(lengths, list(req.regions), base_length=base_len, base_weight=req.region_weight)
        logger.info("Regional: %d regions, %d text tokens total", len(req.regions), cond.shape[1])

        return cond, self._encode([req.negative_prompt], negative=True), mask

    # region noise

    @staticmethod
    def _noise(shape, seeds: list[int], device) -> torch.Tensor:
        """
        Per-image CPU-seeded noise, matching webui's default `randn_source=CPU`.

        Seeding on CPU rather than CUDA is what makes a seed reproduce across
        different GPUs, which matters when the same seed is shared between a
        local run and this one.
        """
        out = []
        for seed in seeds:
            generator = torch.Generator(device="cpu").manual_seed(int(seed) & 0xFFFFFFFF)
            out.append(torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu"))
        return torch.stack(out).to(device)

    # region sampling

    @torch.inference_mode()
    def generate(self, req: GenerateRequest, *, progress=None) -> list[Image.Image]:
        model = self.sd_model
        unet_model = model.forge_objects.unet.model

        # Krea 2 is txt2img here; no reference latents should survive a request.
        dynamic_args.ref_latents.clear()

        steps = int(req.steps or self._default_steps())
        cfg = float(self._default_cfg() if req.cfg_scale is None else req.cfg_scale)
        batch = max(1, int(req.batch_size))
        base_seed = int(time.time() * 1000) & 0x7FFFFFFF if req.seed is None else int(req.seed)
        seeds = [base_seed + i for i in range(batch)]

        # Width/height must land on the patch grid: 8x VAE, then patch 2 in the DiT.
        width = max(64, int(req.width) // 16 * 16)
        height = max(64, int(req.height) // 16 * 16)

        lora_report = self._sync_loras(req.loras)

        # The predictor is shared state on a warm model; set it every request so
        # one call's shift cannot leak into the next.
        unet_model.predictor.set_parameters(shift=float(req.shift))

        cond, uncond, regional = self._build_conditioning(req)
        cond = cond.repeat(batch, *([1] * (cond.ndim - 1)))
        uncond = uncond.repeat(batch, *([1] * (uncond.ndim - 1)))

        device = memory_management.get_torch_device()
        latent_channels = model.forge_objects.vae.latent_channels
        shape = (latent_channels, LATENT_FRAMES, height // LATENT_DOWNSCALE, width // LATENT_DOWNSCALE)
        noise = self._noise(shape, seeds, device)

        sigmas = sampling.get_sigmas(
            sd_model=model, steps=steps, sampler_name=req.sampler,
            scheduler_name=req.scheduler, device=device,
        )

        unet = model.forge_objects.unet
        sampling_prepare(unet, x=noise)

        try:
            x = unet_model.predictor.noise_scaling(sigmas[0], noise, torch.zeros_like(noise), max_denoise=False)

            model_options = dict(unet.model_options)
            transformer_options = dict(model_options.get("transformer_options", {}))
            transformer_options["sampling_sigmas"] = sigmas
            if regional is not None:
                transformer_options["krea2_regional"] = regional
            model_options["transformer_options"] = transformer_options

            denoiser = _CFGDenoiser(
                model=unet_model,
                cond=self._as_cond(cond),
                uncond=self._as_cond(uncond),
                cond_scale=cfg,
                model_options=model_options,
                seed=seeds[0],
                total_steps=len(sigmas) - 1,
                progress=progress,
            )

            latent = sampling.sample(
                denoiser=denoiser, x=x, sigmas=sigmas,
                sampler_name=req.sampler, extra_args={}, seed=seeds[0],
            )
        finally:
            sampling_cleanup(unet)

        images = self._decode(latent)
        self.last_report = {
            "seeds": seeds, "steps": steps, "cfg_scale": cfg, "shift": req.shift,
            "sampler": req.sampler, "scheduler": req.scheduler,
            "width": width, "height": height, "loras": lora_report,
            "regions": [r.to_dict() for r in req.regions],
        }
        return images

    def _decode(self, latent: torch.Tensor) -> list[Image.Image]:
        # decode_first_stage returns (B, frames, 3, H, W) in [-1, 1] for the
        # Wan-style VAE; one frame for a still image.
        decoded = self.sd_model.decode_first_stage(latent.to(torch.float32))
        if decoded.ndim == 5:
            decoded = decoded[:, 0]

        decoded = decoded.float().mul_(0.5).add_(0.5).clamp_(0.0, 1.0)
        arrays = (decoded.movedim(1, -1).cpu().numpy() * 255.0).round().astype(np.uint8)
        return [Image.fromarray(a) for a in arrays]

    # region helpers

    def _sync_loras(self, specs: list) -> list[dict]:
        """Restack only when the request differs from what is already patched."""
        key = tuple(s.key() for s in specs)
        if key == self._lora_key:
            return getattr(self, "_lora_report", [])
        report = loras.apply_stack(self.sd_model, specs)
        self._lora_key = key
        self._lora_report = report
        return report

    def _default_steps(self) -> int:
        return 8 if "turbo" in os.path.basename(self.dit_path).lower() else 28

    def _default_cfg(self) -> float:
        # Turbo is distilled; CFG > 1 washes it out. 5.5 for RAW keeps parity
        # with the --guidance_scale the musubi path used.
        return 1.0 if "turbo" in os.path.basename(self.dit_path).lower() else 5.5


class _CFGDenoiser(torch.nn.Module):
    """
    k-diffusion's `model(x, sigma)` interface on top of Forge's CFG machinery.

    This is the headless equivalent of `modules/sd_samplers_cfg_denoiser.py`,
    which is ~300 lines because it also handles hires fix, inpainting masks,
    prompt scheduling, AND/composable prompts, and the live preview.
    """

    def __init__(self, *, model, cond, uncond, cond_scale, model_options, seed, total_steps, progress=None):
        super().__init__()
        self.inner_model = model
        self.cond = cond
        self.uncond = uncond
        self.cond_scale = cond_scale
        self.model_options = model_options
        self.seed = seed
        self.total_steps = total_steps
        self.progress = progress
        self.step = 0

    def forward(self, x, sigma, **kwargs):
        denoised = sampling_function_inner(
            self.inner_model, x, sigma, self.uncond, self.cond,
            self.cond_scale, self.model_options, self.seed,
        )
        self.step += 1
        if self.progress is not None:
            self.progress(self.step, self.total_steps)
        return denoised
