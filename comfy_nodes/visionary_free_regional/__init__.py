"""
Give the card back after a regional render, which the node pack never does.

`Krea2RegionalMultiLoRAV12` uploads every region's LoRA to the device in
`_RegionalSession._prepare()` and writes the copies **into the same dicts it
keeps**, beside the CPU originals:

    d["down_d"] = d["down"].to(dev, cdt)
    d["up_d"]   = d["up"].to(dev, cdt) * d["scale"]
    self._masks_d = [m.to(dev, cdt) for m in masks]

Nothing ever removes them. The pack's `run()` has a `finally` and it only
unhooks the forward hooks; there is no `free`, no `clear`, no `empty_cache`, and
`_prepare` is guarded by `if "down_d" in d: continue`, so the copies are a cache
whose eviction was never written.

They outlive the render because the session is held by the wrapper closure on
the patcher, and the patcher is the node's returned MODEL — which ComfyUI's
**execution cache** holds until something resets it. `unload_all_models()`
cannot see it, which is why `_reclaim()` in app.py has to fall all the way back
to `/free` and drop the 24 GB checkpoint along with it.

So the symptom is not one graph too big. It is a few in a row, none reproducible
alone: every regional render leaves its LoRAs on the card and the next one
starts with less room than the last.

**This is a node rather than a patch to the pack**, because CLIFF_SHA's whole
claim is that nothing in it is patched — an install, not a vendor, with no
VENDOR.md to keep in sync. Everything below reaches the session through
attributes the pack and ComfyUI already expose.

**Ordering is bought with the latent, not with a hope.** The node takes LATENT
and returns it, so ComfyUI cannot schedule it until the sampler has produced
one, and VAEDecode cannot run until this has. A node with no such edge would be
free to run first, and freeing before sampling is a rebuild rather than a leak
fix.
"""

import gc
import logging

import torch


def _sessions(model):
    """
    Every regional session reachable from this patcher.

    **Found by shape rather than by name.** The obvious lookup is
    `model.wrappers[WrappersMP.DIFFUSION_MODEL]["krea2_regional_multilora"]`,
    and it would break twice over: the pack falls back to the string
    `"diffusion_model"` when `comfy.patcher_extension` has no enum, and it
    falls back again to `add_wrapper` — which takes no key at all — on a
    ComfyUI without `add_wrapper_with_key`. Two pinned SHAs and four
    combinations. Duck-typing on `region_loras` matches all of them and
    matches nothing else in the tree.
    """
    found, seen = [], set()

    def consider(obj):
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))
        if hasattr(obj, "region_loras") and hasattr(obj, "_masks_d"):
            found.append(obj)

    def walk(value, depth=0):
        if depth > 4:
            return
        # Every value on the way down, not only the ones inside closures. The
        # first version tested only cell contents and so found a session held
        # by a wrapper and missed one parked straight in a dict — which is the
        # arrangement `model_options` uses, and it would have failed silently
        # as "no regional session on this model".
        consider(value)
        # The session is a free variable of the wrapper closure, which is what
        # `add_wrapper*` stores. Cells are the only way in.
        for cell in getattr(value, "__closure__", None) or ():
            try:
                walk(cell.cell_contents, depth + 1)
            except ValueError:
                # An empty cell, which happens on a recursive closure.
                continue
        if isinstance(value, dict):
            for v in value.values():
                walk(v, depth + 1)
        elif isinstance(value, (list, tuple, set)):
            for v in value:
                walk(v, depth + 1)

    walk(getattr(model, "wrappers", None))
    # Older packs put the session straight into model_options rather than into
    # a wrapper, so this covers the arrangement the docstring above describes
    # as historical without needing to know which one shipped.
    walk(getattr(model, "model_options", None))
    return found


def _release(session) -> int:
    """Drop the device copies, keep the CPU originals. Returns tensors freed."""
    freed = 0
    for region in getattr(session, "region_loras", None) or ():
        if not isinstance(region, dict):
            continue
        for entry in region.values():
            if not isinstance(entry, dict):
                continue
            # `_prepare` writes exactly these four and reads them back through
            # the `"down_d" in d` guard, so clearing them is what makes the next
            # render rebuild rather than reuse a tensor that is no longer there.
            for key in ("down_d", "up_d", "w1_d", "w2_d"):
                if entry.pop(key, None) is not None:
                    freed += 1
    if getattr(session, "_masks_d", None):
        freed += len(session._masks_d)
    session._masks_d = []
    session._full_mask_cache = {}
    # The rebuild flag, and the reason this is safe to run on a session that
    # will be used again: `run()` calls `_prepare` whenever this is false.
    session._prepared = False
    return freed


class VisionaryFreeRegional:
    """Free a regional session's device tensors once its render is done."""

    CATEGORY = "visionary"
    FUNCTION = "free"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",), "model": ("MODEL",)}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Never cached. The whole job is a side effect, and a cached call is a
        # render whose LoRAs stayed on the card — the exact bug this exists for,
        # reintroduced by ComfyUI helpfully skipping the cleanup.
        return float("nan")

    def free(self, latent, model):
        total, sessions = 0, _sessions(model)
        for session in sessions:
            try:
                total += _release(session)
            except Exception as exc:
                # Never fail a finished render over tidying. The picture is
                # already made; the worst case is the leak this was meant to
                # fix, which is where we started.
                logging.warning("[VisionaryFreeRegional] release failed: %s", exc)
        if sessions:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            free_b, total_b = torch.cuda.mem_get_info()
            logging.info(
                "[VisionaryFreeRegional] released %d tensors from %d session(s); "
                "vram now %.1f of %.1f GiB free",
                total, len(sessions), free_b / 2**30, total_b / 2**30)
        else:
            # Worth saying out loud rather than passing silently: it means the
            # duck-typed search above stopped matching, which is what a pack
            # bump would look like, and the leak would otherwise return with no
            # symptom until somebody re-measured headroom.
            logging.info("[VisionaryFreeRegional] no regional session on this "
                         "model — nothing to release")
        return (latent,)


NODE_CLASS_MAPPINGS = {"VisionaryFreeRegional": VisionaryFreeRegional}
NODE_DISPLAY_NAME_MAPPINGS = {"VisionaryFreeRegional": "Visionary Free Regional"}
