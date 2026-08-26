"""
TeaCache for H3, as our own sixty lines rather than a pinned pack.

The mechanism is Liu et al. 2024 (arXiv:2411.19108), first proven against
this app by Icyoung/ComfyUI-MiniMaxH3-TeaCache in tools/ab_cache.py — 220.0s
to 97.4s on a 20-step 1344x768 take, with computed steps at exactly stock
price. Adjacent denoising steps produce nearly identical outputs, so when the
*input* latent has barely moved since the last real forward, the last real
output is returned instead of running 33B parameters to recompute it. The
skip test is one rel-L1 over the latent — a few million elements — which is
why this survives the shape that killed the block-level cache the day
before: overhead that scales with hidden-state size loses at 768p, and this
has none. See docs/decisions.md, "CacheDiT lasted one day".

Written first-party for one reason beyond size: the pack's state lives in
its node execute, and ComfyUI caches node outputs — a second take with the
same settings reuses the patched MODEL together with the *spent* state, and
its step counter, never reset, disables caching silently from take two on.
The state here keys on the sampler's own sigma instead: sampling walks
sigma downward, so a sigma at or above the last one seen is a new run, and
the state resets itself. No reliance on when ComfyUI chooses to re-execute
the node, which is exactly the thing the bug proved unreliable.

The wrapper sits on `set_model_unet_function_wrapper` — apply_model level,
a ModelPatcher clone. Nothing touches model internals, so nothing persists
on the resident model of a warm container: dropping the node from a graph
is the whole uninstall. That, too, is a lesson with a receipt.
"""

import logging

import torch


class _State:
    __slots__ = ("last_sigma", "step", "prev_input", "prev_output",
                 "accumulated", "reused")

    def __init__(self):
        self.reset()

    def reset(self):
        self.last_sigma = None
        self.step = -1
        self.prev_input = None
        self.prev_output = None
        self.accumulated = 0.0
        self.reused = 0


class VisionaryStepCache:
    """Reuse the previous step's output while the latent has barely moved."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            # The defaults are the measured configuration, not a
            # preference: tools/ab_cache.py, 2.26x, 12 of 20 steps reused,
            # judged on the takes. This threshold is the speed/fidelity
            # dial — 0.25 was considered as "top of the fidelity range" and
            # walked back to the number that was actually measured. Turning
            # it up means re-measuring, which the harness makes cheap.
            "rel_l1_thresh": ("FLOAT", {"default": 0.15, "min": 0.0,
                                        "max": 0.5, "step": 0.01}),
            "start_step": ("INT", {"default": 2, "min": 0, "max": 100}),
            "final_steps": ("INT", {"default": 2, "min": 0, "max": 20}),
            "total_steps": ("INT", {"default": 20, "min": 1, "max": 200}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "visionary"

    def patch(self, model, rel_l1_thresh, start_step, final_steps, total_steps):
        state = _State()

        def wrapper(apply_model, args):
            x, timestep = args["input"], args["timestep"]
            sigma = float(timestep.max())

            # Sampling only ever walks sigma down; equal-or-up is a new run.
            if state.last_sigma is None or sigma >= state.last_sigma:
                if state.reused:
                    logging.info("[step-cache] reused %d of %d steps last run",
                                 state.reused, state.step + 1)
                state.reset()
            if sigma != state.last_sigma:
                state.step += 1
                state.last_sigma = sigma

            reusable = (state.prev_output is not None
                        and state.step >= start_step
                        and state.step < total_steps - final_steps)
            if reusable:
                prev = state.prev_input
                delta = ((x - prev).abs().mean()
                         / prev.abs().mean().clamp(min=1e-8)).item()
                state.accumulated += delta
                if state.accumulated < rel_l1_thresh:
                    state.reused += 1
                    return state.prev_output
                state.accumulated = 0.0

            out = apply_model(x, timestep, **args["c"])
            state.prev_input = x.detach()
            state.prev_output = out.detach() if torch.is_tensor(out) else out
            return out

        patched = model.clone()
        patched.set_model_unet_function_wrapper(wrapper)
        return (patched,)


NODE_CLASS_MAPPINGS = {"VisionaryStepCache": VisionaryStepCache}
NODE_DISPLAY_NAME_MAPPINGS = {"VisionaryStepCache": "Visionary Step Cache"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
