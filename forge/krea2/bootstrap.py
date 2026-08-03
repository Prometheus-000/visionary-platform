"""
Put the vendored Forge tree on `sys.path` before anything imports `backend`.

Forge's own entrypoint does this in `modules_forge/initialization.py`; there is
no entrypoint here, so it happens explicitly. `install()` is idempotent and must
run before the first `import backend` — every module in this package calls it at
the top rather than relying on import order.

Note that importing `backend` at all requires a visible CUDA device:
`backend/memory_management.py` calls `torch.cuda.current_device()` at module
scope to size its VRAM budget. There is no CPU-only path into this package, so
anything that merely wants the sampler/scheduler names should read them from
`app.SAMPLERS` / `app.SCHEDULERS` rather than importing `krea2`.
"""

import os
import sys

# .../forge  — the directory holding backend/, modules/, modules_forge/
FORGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# huggingface_guess, comfy, k_diffusion and gguf are imported by absolute name
# (`from huggingface_guess import model_list`), so their container goes on the
# path too rather than being reachable as modules_forge.packages.*.
PACKAGES = os.path.join(FORGE_ROOT, "modules_forge", "packages")


def install() -> None:
    for path in (FORGE_ROOT, PACKAGES):
        if path not in sys.path:
            sys.path.insert(0, path)
