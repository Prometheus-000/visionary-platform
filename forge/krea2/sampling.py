"""
Samplers and sigma schedules, wired straight to Forge's own implementations.

`modules/sd_samplers_kdiffusion.py` upstream is mostly plumbing between Gradio
state and `k_diffusion.sampling`; the useful parts are the sampler table and the
`get_sigmas` logic. Both are reproduced here against the vendored
`modules/sd_schedulers.py`, so a schedule picked in this UI is the same schedule
Forge would build for the same name.
"""

import inspect

from . import bootstrap

bootstrap.install()

import k_diffusion  # noqa: E402
import torch  # noqa: E402
from k_diffusion.external import ForgeScheduleLinker  # noqa: E402

from modules import sd_schedulers  # noqa: E402

# (label, k_diffusion function name, options) — a subset of Forge's table.
# Dropped: the `sd_samplers_extra` entries (Restart, UniPC), which live in a
# module that reaches into `modules.shared.state` for the progress bar.
SAMPLERS: dict[str, tuple[str, dict]] = {
    "Euler": ("sample_euler", {}),
    "Euler a": ("sample_euler_ancestral", {}),
    "DPM++ 2M": ("sample_dpmpp_2m", {"scheduler": "karras"}),
    "DPM++ 2M SDE": ("sample_dpmpp_2m_sde", {"scheduler": "exponential", "brownian_noise": True}),
    "DPM++ 3M SDE": ("sample_dpmpp_3m_sde", {"scheduler": "exponential", "discard_next_to_last_sigma": True, "brownian_noise": True}),
    "DPM++ SDE": ("sample_dpmpp_sde", {"scheduler": "karras", "second_order": True, "brownian_noise": True}),
    "DPM++ 2s a RF": ("sample_dpmpp_2s_ancestral_RF", {}),
    "Heun": ("sample_heun", {"second_order": True}),
    "LMS": ("sample_lms", {}),
    "LCM": ("sample_lcm", {}),
    "ER SDE": ("sample_er_sde", {}),
    "Res Multistep": ("sample_res_multistep", {}),
}

DEFAULT_SAMPLER = "Euler"

# Krea 2 is not a legacy webui model, so Forge's "Automatic" resolves to Normal
# unless the sampler itself names a schedule (see sd_samplers_kdiffusion.get_sigmas).
DEFAULT_SCHEDULER = "Automatic"

# "Flux2" is excluded: its function signature takes width/height rather than an
# inner model, and it exists for Flux.2 Klein's resolution-aware mu.
SCHEDULERS: list[str] = [s.label for s in sd_schedulers.schedulers if s.label != "Flux2"]


def sampler_names() -> list[str]:
    return list(SAMPLERS)


def scheduler_names() -> list[str]:
    return list(SCHEDULERS)


def _resolve(sampler_name: str) -> tuple[callable, dict]:
    funcname, options = SAMPLERS[sampler_name]
    return getattr(k_diffusion.sampling, funcname), options


def get_sigmas(
    *,
    sd_model,
    steps: int,
    sampler_name: str,
    scheduler_name: str,
    device: torch.device,
) -> torch.Tensor:
    """Build the sigma schedule exactly the way Forge's KDiffusionSampler does."""
    _, options = _resolve(sampler_name)

    discard = options.get("discard_next_to_last_sigma", False)
    if discard:
        steps += 1

    if scheduler_name in (None, "", "Automatic"):
        # Forge: sampler-declared schedule first, then "Normal" for non-legacy models.
        scheduler_name = options.get("scheduler") or "Normal"

    scheduler = sd_schedulers.schedulers_map.get(scheduler_name)

    linker = ForgeScheduleLinker(sd_model.forge_objects.unet.model.predictor)
    # FlowMatchEulerDiscrete reaches back through the linker for the shift value.
    linker.inner_model = sd_model

    if scheduler is None or scheduler.function is None:
        sigmas = linker.get_sigmas(steps)
    else:
        kwargs = {
            "sigma_min": float(linker.sigmas[0]),
            "sigma_max": float(linker.sigmas[-1]),
        }
        if scheduler.need_inner_model:
            kwargs["inner_model"] = linker

        params = inspect.signature(scheduler.function).parameters
        if "device" in params:
            kwargs["device"] = device

        sigmas = scheduler.function(n=steps, **kwargs)

    sigmas = torch.as_tensor(sigmas, dtype=torch.float32).to(device)

    if discard:
        sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])

    return sigmas


def sample(
    *,
    denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    sampler_name: str,
    extra_args: dict,
    seed: int,
    callback=None,
) -> torch.Tensor:
    """Run one k-diffusion sampler over `sigmas`."""
    func, options = _resolve(sampler_name)

    kwargs: dict = {}
    params = inspect.signature(func).parameters

    if "sigmas" in params:
        kwargs["sigmas"] = sigmas
    if "n" in params:
        kwargs["n"] = len(sigmas) - 1
    if "sigma_min" in params:
        kwargs["sigma_min"] = float(sigmas[sigmas > 0].min())
        kwargs["sigma_max"] = float(sigmas.max())
    if options.get("solver_type") == "heun":
        kwargs["solver_type"] = "heun"

    if options.get("brownian_noise"):
        # The SDE samplers need a seeded Brownian tree or successive runs with
        # the same seed diverge.
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        kwargs["noise_sampler"] = k_diffusion.sampling.BrownianTreeNoiseSampler(
            x, sigma_min, sigma_max, seed=seed
        )

    return func(denoiser, x, extra_args=extra_args, disable=True, callback=callback, **kwargs)
