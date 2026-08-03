"""
Multiple LoRAs with independent strengths, the way Forge does it.

Forge's own path is `extensions-builtin/sd_forge_lora/networks.py`, which is
half key-matching and half Gradio extra-networks bookkeeping (disk scanning,
aliases, hashes, infotext). Only the first half matters here, so
`load_lora_for_models` is ported and the rest is dropped.

What is preserved from Forge, because it is what makes stacking behave:

  * Each LoRA is a *patch on a clone* of the previous patcher, so N LoRAs
    compose into one patch list rather than overwriting each other.
  * `strength_model` (UNet) and `strength_clip` (text encoder) are separate
    knobs per LoRA — the two numbers Forge shows you.
  * Patches are lazy. `add_patches` only records them; the merge into the
    weights happens when `memory_management` loads the model, so restacking
    between generations costs nothing until sampling starts.
  * Key matching goes through `model_lora_keys_unet`, which understands the
    `lora_unet_*` naming that musubi-tuner writes as well as the diffusers and
    lycoris spellings.
"""

import logging
import os

from . import bootstrap

bootstrap.install()

import torch  # noqa: E402

from backend.logging import setup_logger  # noqa: E402
from backend.patcher.lora import (  # noqa: E402
    load_lora,
    model_lora_keys_clip,
    model_lora_keys_unet,
)
from backend.state_dict import state_dict_prefix_replace  # noqa: E402
from backend.utils import load_torch_file  # noqa: E402

# Named "lora" so Forge's own handler highlights the UNet/CLIP keywords, and so
# these lines sit alongside the loader's in the Modal log. Without setup_logger
# the root logger swallows everything below WARNING, which hides exactly the
# key-match counts you need when a LoRA quietly does nothing.
logger = logging.getLogger("lora")
setup_logger(logger)


class LoraSpec:
    """One entry in the stack: a file plus its two strengths."""

    __slots__ = ("path", "unet", "text_encoder")

    def __init__(self, path: str, unet: float = 1.0, text_encoder: float | None = None):
        self.path = str(path)
        self.unet = float(unet)
        # Forge defaults the TE weight to the UNet weight when only one is given.
        self.text_encoder = float(unet if text_encoder is None else text_encoder)

    @property
    def name(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0]

    def key(self) -> tuple:
        return (self.path, self.unet, self.text_encoder)

    def __repr__(self) -> str:
        return f"<Lora {self.name} unet={self.unet} te={self.text_encoder}>"


def read_lora(path: str) -> dict[str, torch.Tensor]:
    sd = load_torch_file(path, safe_load=True)
    # Some trainers emit a doubled underscore after the prefix; Forge normalises
    # it before key matching and so do we.
    if any(k.startswith("lora_unet__") for k in sd):
        sd = state_dict_prefix_replace(sd, {"lora_unet__": "lora_unet_"})
    return sd


def build_key_maps(unet, clip) -> tuple[dict, dict]:
    """
    Map LoRA key names onto model weight names, once per model.

    Both upstream helpers are declared `def f(model, key_map={})` — a shared
    mutable default that accumulates every model ever passed to it. A fresh dict
    is handed in explicitly so a warm container cannot leak one model's key map
    into another's, and so the map does not grow without bound across requests.
    """
    unet_keys = model_lora_keys_unet(unet.model, {}) if unet is not None else {}
    clip_keys = model_lora_keys_clip(clip.cond_stage_model, {}) if clip is not None else {}
    return unet_keys, clip_keys


def apply_one(unet, clip, lora_sd: dict, spec: LoraSpec, unet_keys: dict, clip_keys: dict) -> tuple:
    """
    Patch one LoRA onto (unet, clip); returns (unet, clip, report).

    Ported from `load_lora_for_models`. Returns the originals untouched if the
    file does not match the model, so a stray LoRA in the stack degrades to a
    no-op with a warning instead of corrupting the run.

    The report counts each side separately. musubi-tuner's `lora_krea2` wraps
    only the DiT's Linear layers, so its files land 264 UNet keys and zero text
    encoder keys — the TE strength is inert for them, and a caller that shows
    both numbers should say so rather than implying the second one did work.
    """
    report = {"name": spec.name, "unet": spec.unet, "text_encoder": spec.text_encoder,
              "unet_keys": 0, "te_keys": 0, "applied": False}

    if not lora_sd:
        logger.warning("[LoRA] %s is empty", spec.name)
        return unet, clip, {**report, "reason": "empty file"}

    lora_unet, unmatched = load_lora(lora_sd, unet_keys)
    lora_clip, unmatched = load_lora(unmatched, clip_keys)

    if len(unmatched) / len(lora_sd) > 0.5:
        logger.warning(
            "[LoRA] %s does not match Krea 2 — %d/%d keys unmatched, skipping",
            spec.name, len(unmatched), len(lora_sd),
        )
        return unet, clip, {**report, "reason": "not a Krea 2 LoRA"}

    if unet is not None and lora_unet:
        patched = unet.clone()
        loaded = patched.add_patches(filename=spec.path, patches=lora_unet, strength_patch=spec.unet)
        skipped = [k for k in lora_unet if k not in loaded]
        if len(skipped) / len(lora_unet) > 0.25:
            logger.warning("[LoRA] %s UNet mismatch: %d skipped of %d", spec.name, len(skipped), len(lora_unet))
        else:
            logger.info("[LoRA] %s -> UNet, %d keys @ %s", spec.name, len(loaded), spec.unet)
            unet = patched
            report["unet_keys"] = len(loaded)

    if clip is not None and lora_clip:
        patched = clip.clone()
        loaded = patched.add_patches(filename=spec.path, patches=lora_clip, strength_patch=spec.text_encoder)
        skipped = [k for k in lora_clip if k not in loaded]
        if len(skipped) / len(lora_clip) > 0.25:
            logger.warning("[LoRA] %s TE mismatch: %d skipped of %d", spec.name, len(skipped), len(lora_clip))
        else:
            logger.info("[LoRA] %s -> TE, %d keys @ %s", spec.name, len(loaded), spec.text_encoder)
            clip = patched
            report["te_keys"] = len(loaded)

    report["applied"] = bool(report["unet_keys"] or report["te_keys"])
    if not report["applied"]:
        report["reason"] = "no matching keys"
    return unet, clip, report


def apply_stack(sd_model, specs: list[LoraSpec]) -> list[dict]:
    """
    Rebuild `sd_model.forge_objects` with `specs` applied, in order.

    Always restarts from `forge_objects_original` so a generation never inherits
    the previous request's stack — the failure mode that makes strengths drift
    upward across a session.

    Returns one report dict per spec for the caller to surface.
    """
    sd_model.forge_objects.unet = sd_model.forge_objects_original.unet
    sd_model.forge_objects.clip = sd_model.forge_objects_original.clip

    unet = sd_model.forge_objects.unet
    clip = sd_model.forge_objects.clip
    report: list[dict] = []

    if specs:
        unet_keys, clip_keys = build_key_maps(unet, clip)

    for spec in specs:
        if spec.unet == 0.0 and spec.text_encoder == 0.0:
            report.append({"name": spec.name, "applied": False, "reason": "strength 0"})
            continue
        try:
            lora_sd = read_lora(spec.path)
        except Exception as exc:
            logger.error("[LoRA] cannot read %s: %s", spec.path, exc)
            report.append({"name": spec.name, "applied": False, "reason": str(exc)})
            continue

        unet, clip, entry = apply_one(unet, clip, lora_sd, spec, unet_keys, clip_keys)
        del lora_sd
        report.append(entry)

    sd_model.forge_objects.unet = unet
    sd_model.forge_objects.clip = clip
    sd_model.forge_objects_after_applying_lora = sd_model.forge_objects.shallow_copy()
    sd_model.current_lora_hash = str([s.key() for s in specs])

    return report
