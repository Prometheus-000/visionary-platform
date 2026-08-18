"""
Does the leak fix actually reach the tensors, and only those?

`VisionaryFreeRegional` finds the regional session by **shape** rather than by
name, because the pack stores it four different ways depending on which
fallbacks fire in `comfy.patcher_extension` and `ModelPatcher`. A search by
duck-typing is only worth having if it is exercised against every arrangement,
and the first version passed the wrapper case and silently missed the
`model_options` one — which is not a crash, it is the leak coming back with a
log line saying nothing is wrong.

CPU-only, no ComfyUI, no GPU. `torch` is stubbed because the node imports it
for `empty_cache` and nothing here samples anything.
"""

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.modules.setdefault("torch", types.SimpleNamespace(
    cuda=types.SimpleNamespace(is_available=lambda: False,
                               empty_cache=lambda: None,
                               mem_get_info=lambda: (0, 0))))
spec = importlib.util.spec_from_file_location(
    "vfr", ROOT / "comfy_nodes" / "visionary_free_regional" / "__init__.py")
vfr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vfr)


class Session:
    """The pack's `_RegionalSession`, reduced to the attributes this touches."""

    def __init__(self, loras):
        self.region_loras = loras
        self._masks_d = ["m0", "m1"]
        self._full_mask_cache = {"k": 1}
        self._prepared = True


class Patcher:
    def __init__(self, wrappers=None, model_options=None):
        self.wrappers = wrappers or {}
        self.model_options = model_options or {}


def loras():
    """One plain LoRA, one LoKr, one empty region — the three `_prepare` sees."""
    return [
        {"blk.0": {"down": "cpu", "up": "cpu", "scale": 1.0,
                   "down_d": "GPU", "up_d": "GPU"}},
        {"blk.1": {"w1": "cpu", "w2": "cpu", "scale": 1.0,
                   "w1_d": "GPU", "w2_d": "GPU"}},
        {},
    ]


def wrapped(session):
    """`session` as a closure cell, which is how `add_wrapper_with_key` holds it."""
    def wrapper(executor, *args, **kwargs):
        return session.run(executor, *args, **kwargs)
    return wrapper


def main() -> int:
    bad = 0

    def check(name, got):
        nonlocal bad
        bad += not got
        print(f"  {'ok  ' if got else 'FAIL'}  {name}")

    # Every arrangement the pack can leave the session in.
    s = Session(loras())
    check("found through add_wrapper_with_key (enum + key)",
          vfr._sessions(Patcher({"diffusion_model":
                                 {"krea2_regional_multilora": [wrapped(s)]}})) == [s])
    s = Session(loras())
    check("found through add_wrapper (no key)",
          vfr._sessions(Patcher({"diffusion_model": [wrapped(s)]})) == [s])
    s = Session(loras())
    check("found in model_options, not a wrapper at all",
          vfr._sessions(Patcher(model_options={"transformer_options":
                                               {"sess": s}})) == [s])
    check("a model with no regional session finds nothing",
          vfr._sessions(Patcher()) == [])

    # And that it frees the right half.
    ls = loras()
    s = Session(ls)
    freed = vfr._release(s)
    check("released 4 lora copies and 2 masks", freed == 6)
    check("no device copy survives",
          not [k for r in ls for e in r.values() for k in e if k.endswith("_d")])
    check("every cpu original survives",
          all(("down" in e and "up" in e) or ("w1" in e and "w2" in e)
              for r in ls if r for e in r.values()))
    check("scale survives, so the rebuild is identical",
          all(e["scale"] == 1.0 for r in ls if r for e in r.values()))
    check("_prepared is false, so run() rebuilds", s._prepared is False)
    check("masks and mask cache cleared",
          s._masks_d == [] and s._full_mask_cache == {})
    check("a second release is a no-op", vfr._release(s) == 0)

    # The node itself: it must pass the latent through untouched, and must not
    # raise on a model it does not recognise — a finished render is not worth
    # failing over tidying.
    node = vfr.VisionaryFreeRegional()
    latent = {"samples": object()}
    check("latent passes through by identity",
          node.free(latent, Patcher(model_options={"transformer_options":
                                                   {"sess": Session(loras())}}))[0] is latent)
    check("an unrecognised model is not an error",
          node.free(latent, Patcher())[0] is latent)
    check("never cached — IS_CHANGED is always new",
          vfr.VisionaryFreeRegional.IS_CHANGED() != vfr.VisionaryFreeRegional.IS_CHANGED())

    print(f"\n  {'all good' if not bad else f'{bad} failed'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
