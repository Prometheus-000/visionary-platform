"""
Re-arm the pack's edit wrappers against a host that grew a positional arg.

ComfyUI c9602625 (2026-07-18) inserted `ref_latents` into the Krea 2 forward,
between `attention_mask` and `transformer_options`:

    .execute(x, timesteps, context, attention_mask, ref_latents,
             transformer_options, **kwargs)

Seven positionals now reach every DIFFUSION_MODEL wrapper. The pack's regional
*session* wrappers are `(executor, *args, **kwargs)` and never noticed — which
is why plain regional renders kept working and hid this. Its two *edit*
wrappers (`_arm_edit`, `_arm_edit_official` in V7, inherited by V9/V12) spell
the old parameter list out, so every scene/outfit compose dies in step 0 with

    wrapper() takes from 4 to 6 positional arguments but 7 were given

— observed live on job gen202608241342316807. The change predates both COMFY
pins this app has carried (the 2026-08-03 one included), so there is no SHA to
roll back to, and the pack's HEAD (CLIFF_SHA, 2026-08-01) has no fix to pull.

**This is a node rather than a patch**, for the same reason
`visionary_free_regional` is: CLIFF_SHA's whole claim is that nothing in it is
patched — an install, not a vendor. It sits between V12's MODEL output and the
sampler, and rewraps in place.

**Found by shape rather than by key.** The pack registers under
WRAPPER_KEY_V7/V9/V12 with two fallbacks for older hosts (`add_wrapper_with_key`
-> `add_wrapper`, which takes no key at all), so matching key names would break
on exactly the installs the fallbacks exist for. What identifies a wrapper that
needs help is its *signature*: no `*args`, and too few positional slots for the
host's seven. That test also excuses itself the day the pack modernises.

The dropped `ref_latents` is the host's new reference-image channel for the
ostris/identity-edit ref LoRAs — a feature no graph in app.py wires, so it is
always None here. If a future graph ever wires both, dropping it silently would
be the sin this codebase exists to avoid, hence the log line.
"""

import inspect
import logging


def _positional_capacity(fn):
    """How many positionals fn can take, or None when *args makes it moot."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        # A signature we cannot read is one we must not rewrite.
        return None
    if any(p.kind == p.VAR_POSITIONAL for p in params):
        return None
    return sum(1 for p in params
               if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))


def _adapt(wrapper):
    """Wrap a pre-ref_latents wrapper so the host's seven positionals fit."""
    if getattr(wrapper, "_visionary_arity", False):
        return wrapper  # already adapted — a cached model re-swept, not a bug
    cap = _positional_capacity(wrapper)
    # executor + (x, timesteps, context, attention_mask, transformer_options)
    # is the signature both edit wrappers spell. Anything else — *args, or a
    # wrapper already written for seven — is left exactly as it is.
    if cap != 6:
        return wrapper

    def adapted(executor, *args, **kwargs):
        if len(args) == 6:
            if args[4] is not None:
                logging.warning(
                    "[visionary_edit_arity] dropping non-None ref_latents on "
                    "the way into a pre-c9602625 wrapper — the regional edit "
                    "path cannot carry it."
                )
            args = args[:4] + args[5:]
        return wrapper(executor, *args, **kwargs)

    adapted._visionary_arity = True
    return adapted


class VisionaryEditArity:
    """MODEL -> MODEL, with fixed-arity diffusion-model wrappers adapted."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL", {
            "tooltip": "The regional node's output; its edit wrappers are "
                       "adapted to the host's post-ref_latents call.",
        })}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "run"
    CATEGORY = "visionary"

    def run(self, model):
        # The edit wrappers land in model_options["transformer_options"] at
        # node run time — that is the dict the pack writes with its own `to =
        # patched.model_options.setdefault(...)`. Patcher-level wrappers are
        # merged in later, at sample time, but those are the *args-safe session
        # wrappers, so this one dict is the whole surface.
        wrappers = (model.model_options
                    .get("transformer_options", {})
                    .get("wrappers", {})
                    .get("diffusion_model", None))
        if isinstance(wrappers, dict):
            for key, entry in wrappers.items():
                if isinstance(entry, list):
                    wrappers[key] = [_adapt(w) for w in entry]
        elif isinstance(wrappers, list):
            # The no-key fallback stores a bare list on hosts without
            # add_wrapper_with_key.
            wrappers[:] = [_adapt(w) for w in wrappers]
        return (model,)


NODE_CLASS_MAPPINGS = {"VisionaryEditArity": VisionaryEditArity}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VisionaryEditArity": "Visionary — edit wrapper arity",
}
