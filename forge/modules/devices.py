"""
Device/dtype constants, resolved from Forge's own memory manager.

Upstream `modules/devices.py` is a much larger file that also owns autocast,
NaN checks and CPU fallbacks. `backend/` only imports `device` from it; `dtype`
is added here because the regional-prompt code wants the same inference dtype
Forge would pick.
"""

import torch

from backend import memory_management

device: torch.device = memory_management.get_torch_device()

dtype_vae: torch.dtype = memory_management.vae_dtype()
dtype_unet: torch.dtype = memory_management.unet_dtype()
dtype_inference: torch.dtype = memory_management.inference_cast(dtype_unet, device)
dtype: torch.dtype = torch.float32 if dtype_unet is torch.float32 else torch.float16

cpu: torch.device = torch.device("cpu")


def get_optimal_device() -> torch.device:
    return device


def torch_gc() -> None:
    memory_management.soft_empty_cache()
