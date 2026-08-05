"""
Visionary — standalone Krea 2 LoRA trainer on Modal.

One file. One command. No Vercel, no npm, no Modal Secrets, no access keys.
The UI is served by Modal itself, so `modal deploy app.py` gives you a single
URL that is the whole application.

    modal deploy app.py

Training runs on musubi-tuner. Inference runs on sd-webui-forge-classic (neo),
vendored into forge/ — see forge/VENDOR.md. The two are deliberately separate
images: forge wants newer transformers/diffusers than musubi pins, and the
generation side gets Forge's LoRA stacking, samplers and schedules for free.

Storage is ours, not borrowed. The volume is created on first deploy and the
layout is flat and self-describing — nothing here mirrors a checkout of another
project, so there is no directory that only makes sense to somebody who has read
Forge's source.

    $VISIONARY_VOLUME (default "visionary")  ->  /workspace
      models/krea2-raw.safetensors        Krea 2 RAW DiT   (training)
      models/krea2-turbo.safetensors      Krea 2 Turbo DiT (inference)
      models/qwen-image-vae.safetensors
      models/qwen3vl-4b-bf16.safetensors
      loras/{folder}/{name}.safetensors   trained output, any nesting
      outputs/{job}/                      generated images
      datasets/{job}/, uploads/{job}/     working dirs
      .cache/                             HF staging, never read directly

Set VISIONARY_VOLUME to run a second copy (staging, a different account) against
its own storage.

Nothing downloads on its own — pick what you want on the Models tab.
"""

# NOTE: deliberately NO `from __future__ import annotations`.
#
# It turns every annotation into a string, and FastAPI resolves those with
# get_type_hints() against the *module* globals. The routes live inside web()
# and import Request locally, so "Request" was unresolvable — FastAPI then fell
# back to treating `request` as a required query parameter and answered every
# upload with 422 {"loc":["query","request"],"msg":"Field required"}.
# Only /api/upload was affected; every other route annotates `payload: dict`,
# and `dict` resolves from builtins either way.
#
# All images run Python 3.10+, so `X | Y` and `list[...]` work natively without it.

import base64
import io
import json
import os
import re
import shutil
import subprocess
import time
import zipfile
from collections import deque
from pathlib import Path
from typing import Any

import modal

APP_NAME = "visionary"

# Storage this app owns. create_if_missing because a fresh account or a new
# VISIONARY_VOLUME should just work — the cost is that a typo'd name silently
# yields an empty volume rather than an error, so _require_models() prints the
# resolved name and what it actually found instead of a bare "not downloaded".
VOLUME_NAME = os.environ.get("VISIONARY_VOLUME", "visionary")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Downloaded model weights live on their own volume, never on the data volume.
# Qwen3-VL-8B is ~17 GB, and with the HuggingFace cache pointed at /workspace
# every commit after a caption run had to snapshot those 17 GB — the job wrote
# all 80 captions in eight minutes and then sat in commit() for another fifteen.
# Separating them keeps a caption commit proportional to the captions.
hf_cache = modal.Volume.from_name("visionary-hf-cache", create_if_missing=True)
HF_CACHE = Path("/hf")

# Live job state (progress, stop flags) and the saved HF token. Dicts rather
# than Secrets so there is no CLI setup step — paste the key into the UI.
jobs = modal.Dict.from_name("visionary-jobs", create_if_missing=True)
config = modal.Dict.from_name("visionary-config", create_if_missing=True)

# Mount point is an internal detail; the layout under it is the contract.
# Weights are addressed by exact path, never scanned, so models/ is flat with
# descriptive filenames rather than the per-architecture directories a webui
# needs in order to populate its dropdowns.
WORKSPACE = Path("/workspace")
MODELS = WORKSPACE / "models"
LORAS = WORKSPACE / "loras"
# A dataset is a named folder of images with .txt captions beside them — the
# same thing musubi reads, so the sidecars stay the source of truth and nothing
# we build here is required to get a dataset back out. Datasets outlive the
# training runs that use them; WORK is the per-run scratch copy musubi resizes
# and caches into, and is disposable.
DATASETS = WORKSPACE / "datasets"
WORK = WORKSPACE / "work"
OUTPUTS = WORKSPACE / "outputs"
STAGING = WORKSPACE / ".cache" / "hf-staging"

# Removed images land here rather than being unlinked. Non-destructive by
# default: a mis-click during a cull should cost a file move, not the file.
TRASH_DIR = ".trash"
THUMB_DIR = ".thumbs"
MUSUBI = Path("/opt/musubi-tuner")

# The vendored sd-webui-forge-classic backend that inference runs on. Lives next
# to this file so `modal deploy` from anywhere still finds it.
FORGE_DIR = str(Path(__file__).parent / "forge")
FORGE = Path("/opt/forge")

GPU = "A100-40GB"  # measured 29.26 GiB peak at 1024px, rank 32 — ~11 GiB spare

# Video is its own GPU class. The H3 stack is 42.5 GB of weights before any
# activations, so it does not share a card with training or Krea 2 — and an
# A100-40GB cannot hold it at all.
VIDEO_GPU = os.environ.get("VISIONARY_VIDEO_GPU", "H100")

# Cards the UI may ask for, per feature.
#
# Modal's Cls.with_options() returns a variant that autoscales independently of
# the base configuration — which is exactly the point (one class, any card) and
# exactly the cost (a card you have not used recently has no warm container, so
# picking it pays a cold start). The UI confirms before switching for that
# reason, and requests for the default are sent to the base class rather than
# through with_options so the common path keeps its warm container.
#
# Video is Hopper-only on purpose: SageAttention is compiled for sm_90 in
# video_image. B200 is sm_100 and would load the model fine and then fall back
# off the fast kernels — the failure this list exists to prevent. Adding it
# means changing TORCH_CUDA_ARCH_LIST and forcing an image rebuild.
IMAGE_GPUS = ("A100-40GB", "A100-80GB", "H100")
VIDEO_GPUS = ("H100", "H200")

# ComfyUI is the video inference backend, pinned by commit rather than vendored.
#
# Vendoring earned its place for Krea 2 (forge/VENDOR.md) because that path
# needed a patch. This one does not: we drive ComfyUI, we do not modify it, so
# a SHA in the image definition is both smaller and more honest than a copy of
# the tree. Updating is a one-line change; `git log` upstream is the changelog.
#
# Why ComfyUI at all, when diffusers has a MiniMax-H3 pipeline: the diffusers
# integration runs the released bf16 weights, 123.6 GB across the transformer
# and the Qwen3-VL-32B conditioner. ComfyUI runs Comfy's repackage — modulation
# weights pruned into a lookup table, int8-convrot weights, and their own
# kernels — for 42.5 GB and int8 tensor-core matmuls instead of bf16 ones. On
# one card that is the difference between offloading every request and holding
# the model resident, on top of roughly 2x on the denoise loop itself.
COMFY_SHA = "16e3f3034f2bba1fff6c70cbd759339778555cd6"  # 2026-08-03, H3 VAE fix
COMFY = Path("/opt/comfyui")
COMFY_PORT = 8188

app = modal.App(APP_NAME)


# --------------------------------------------------------------------------
# Images — every dependency baked in, nothing installed at runtime
# --------------------------------------------------------------------------

# Plain `fastapi`, not `fastapi[standard]`: Modal serves the ASGI app itself, so
# the bundled uvicorn/typer/rich/httpx/jinja2/email-validator are all dead
# weight. Only FastAPI, Request, HTMLResponse and JSONResponse are used, which
# are core. python-multipart is listed explicitly because the upload form needs
# it and plain fastapi does not pull it.
web_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi==0.115.6",
    "python-multipart==0.0.20",
    "pillow==11.1.0",
    "huggingface_hub[hf_transfer]==0.27.1",
)

trainer_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.10"
    )
    # git for the build-time clone; libgl1/libglib2.0-0 are opencv-python's
    # runtime libs (a musubi dependency). No unzip — extraction is pure Python.
    .apt_install("git", "libgl1", "libglib2.0-0")
    # torch first from the cu124 index so musubi's install sees it satisfied and
    # does not pull a default-CUDA wheel over the top.
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/kohya-ss/musubi-tuner /opt/musubi-tuner",
        # Brings accelerate, transformers, bitsandbytes (for adamw8bit),
        # voluptuous and toml at musubi's own pins. Deliberately not re-pinned
        # here — pinning them separately fights those versions.
        "cd /opt/musubi-tuner && pip install -e .",
    )
    .pip_install("hf_transfer==0.1.9")
    # krea2_encoder loads TE weights from the local safetensors but still fetches
    # the *tokenizer* by repo id at runtime. Bake it so a paid GPU run never
    # waits on HuggingFace.
    .run_commands(
        "python -c \"from transformers import AutoTokenizer, Qwen2TokenizerFast; "
        "r='Qwen/Qwen3-VL-4B-Instruct'; "
        "AutoTokenizer.from_pretrained(r); Qwen2TokenizerFast.from_pretrained(r)\""
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1"})
)

# Inference runs on sd-webui-forge-classic's backend, not musubi — see
# forge/VENDOR.md. Deliberately a separate image from the trainer: forge wants
# newer transformers/diffusers than musubi pins, and coupling them means every
# forge sync risks breaking training.
#
# Python 3.11 because comfy-kitchen ships cp311 manylinux wheels and numpy 2.x
# requires >=3.11. CUDA 12.8 for the torch 2.8 wheels; comfy-kitchen disables
# its own CUDA backend below CUDA 13 and falls back to plain torch ops, which is
# fine — Krea 2 only uses it for fused RoPE.
inference_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "accelerate==1.10.1",
        "comfy-kitchen==0.2.26",
        "diffusers==0.35.1",
        "einops==0.8.1",
        "huggingface_hub==0.34.4",
        "numpy==2.2.6",
        "pillow==11.3.0",
        "psutil==7.0.0",
        "pyyaml==6.0.2",
        "rich==14.1.0",
        "safetensors==0.6.2",
        "scipy==1.16.1",          # sd_schedulers' Beta schedule
        "torchsde==0.2.6",        # the SDE samplers' Brownian tree
        "transformers==4.56.1",
        "tqdm==4.67.1",
    )
    # The Qwen3-VL tokenizer is loaded from backend/huggingface/krea/.../tokenizer,
    # so unlike the trainer this image needs no HuggingFace round-trip at runtime.
    .env({"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(FORGE_DIR, remote_path="/opt/forge")
)


# Captioning is a third image for the same reason there is a second one:
# Qwen3VLForConditionalGeneration landed in transformers 4.57, which is newer
# than musubi's pins and newer than the 4.56.1 forge is verified against.
# Pinning one transformers across all three would mean every captioner bump
# re-litigates both training and inference.
caption_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "accelerate==1.10.1",
        # Explicit, not the huggingface_hub[hf_transfer] extra — the extra does
        # not reliably pull the package, and with the env var set below its
        # absence surfaces as "Can't load the configuration of <repo>", which
        # reads like a wrong or gated repo id rather than a missing dependency.
        "hf_transfer==0.1.9",
        "huggingface_hub==0.35.3",
        "pillow==11.3.0",
        "transformers==4.57.1",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1",
          "TOKENIZERS_PARALLELISM": "false"})
)


# Video: ComfyUI on a CUDA 13 torch wheel.
#
# The CUDA version is the whole point of this image, not an incidental pin.
# ComfyUI's quant_ops.py reads torch.version.cuda and disables comfy-kitchen's
# CUDA backend below 13, falling back to plain torch ops — which silently
# throws away exactly the int8-convrot and nvfp4 kernels the H3 repackage is
# quantized for. A cu128 build would load the same weights, produce the same
# pictures, and run at roughly half the speed with no error to explain it.
# That is the failure this pin exists to prevent, so if you move this wheel,
# check `Comfy-Kitchen ... {'cuda': True}` is still in the container's logs.
#
# A -devel base, not debian_slim, because SageAttention below compiles CUDA
# kernels from source and needs nvcc. The torch wheels carry a CUDA *runtime*,
# never a compiler, so a slim base gets all the way to the SageAttention build
# before failing on a missing nvcc.
video_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04", add_python="3.12"
    )
    # clang and libomp are for the SageAttention build, not for CUDA: Modal's
    # add_python ships a clang-built interpreter, so sysconfig hands distutils
    # `clang++` as the linker for the extension. nvcc compiles the kernels
    # fine with build-essential alone and then the final link fails.
    .apt_install("git", "libgl1", "libglib2.0-0", "ffmpeg",
                 "build-essential", "clang", "libomp-dev")
    .pip_install(
        "torch==2.9.1",
        "torchvision==0.24.1",
        "torchaudio==2.9.1",
        index_url="https://download.pytorch.org/whl/cu130",
    )
    .run_commands(
        f"git clone https://github.com/Comfy-Org/ComfyUI {COMFY}",
        f"cd {COMFY} && git checkout {COMFY_SHA}",
        # --no-deps on torch is not available here, so requirements.txt is
        # installed as-is; it lists torch unpinned, which pip leaves satisfied
        # by the cu130 wheel above rather than replacing with a default-CUDA one.
        f"cd {COMFY} && pip install -r requirements.txt",
    )
    # SageAttention, not FlashAttention.
    #
    # Attention is the right lever — H3 runs full self-attention over video,
    # text and audio rows in one packed sequence, so it is where a 33B dense
    # model spends its time. But ComfyUI reaches FlashAttention through
    # `from flash_attn import flash_attn_func`, which is FA2's package. The
    # only build PyTorch publishes for CUDA 13 is `flash_attn_3`, a different
    # module name that --use-flash-attention will not find: you would pay the
    # install and silently get no attention backend at all. SageAttention is
    # what ComfyUI's own H3 documentation names, and it is worth roughly 2x.
    #
    # TORCH_CUDA_ARCH_LIST must match VIDEO_GPU. 9.0 is Hopper (H100/H200);
    # move to "10.0" for B200 and force a rebuild, or the kernels will not load.
    .env({"TORCH_CUDA_ARCH_LIST": "9.0", "MAX_JOBS": "8"})
    # --no-build-isolation is required (the build imports the torch installed
    # above rather than a fresh one), but it also means pip supplies nothing:
    # without `wheel` the build dies on `invalid command 'bdist_wheel'`, and
    # without `ninja` torch's cpp_extension silently falls back to distutils
    # and compiles the CUDA kernels single-threaded.
    .pip_install("wheel", "setuptools", "packaging", "ninja")
    .run_commands(
        "git clone --depth 1 https://github.com/thu-ml/SageAttention /tmp/sage",
        "cd /tmp/sage && pip install . --no-build-isolation",
        "rm -rf /tmp/sage",
    )
    # pillow only, and deliberately no huggingface_hub.
    #
    # This used to also install huggingface_hub[hf_transfer]==0.35.3, copied
    # from the images that actually download weights. Here it was worse than
    # useless: it ran *after* ComfyUI's requirements.txt and pinned the hub
    # back to the 0.x line, while the transformers those requirements bring
    # imports `is_offline_mode` from it — a symbol that only exists in 1.x. So
    # `import execution` died in main.py and ComfyUI never opened a port, which
    # surfaces as `_wait_ready()` raising "ComfyUI exited during startup" with
    # a traceback about a symbol nobody here asked for.
    #
    # Nothing in this container has any business talking to HuggingFace anyway:
    # weights arrive on the volume via `_download_weight`, which runs on
    # web_image on CPU. Leaving the pin out means ComfyUI's own resolution is
    # the only thing deciding the hub version, which is the only way it can be
    # right. `tools/smoke_video.py` is what catches it if this regresses.
    .pip_install("pillow==11.3.0")
    .env({"PYTHONUNBUFFERED": "1"})
)


# --------------------------------------------------------------------------
# Model catalogue
#
# Every repo/filename verified against the HuggingFace API, not copied from
# docs. Two things worth remembering:
#   * docs/krea2.md puts the VAE in Comfy-Org/Qwen-Image-Edit_ComfyUI — that
#     path 404s. It lives in Comfy-Org/Qwen-Image_ComfyUI (no "-Edit").
#   * The text encoder must be bf16, NOT fp8_scaled. musubi builds a plain
#     Qwen3VL model and errors on unexpected keys; the ComfyUI fp8 file carries
#     ~504 extra weight_scale/comfy_quant tensors and is rejected outright.
# --------------------------------------------------------------------------

MODEL_CATALOGUE: dict[str, dict[str, Any]] = {
    "raw": {
        "label": "Krea 2 RAW",
        "note": "DiT for training",
        "family": "Krea 2 — images",
        "repo_id": "krea/Krea-2-Raw",
        "filename": "raw.safetensors",
        "dest": MODELS / "krea2-raw.safetensors",
        "gated": True,
        "approx_gb": 26.3,
    },
    "turbo": {
        "label": "Krea 2 Turbo",
        "note": "DiT for generating — 8 steps",
        "family": "Krea 2 — images",
        "repo_id": "krea/Krea-2-Turbo",
        "filename": "turbo.safetensors",
        "dest": MODELS / "krea2-turbo.safetensors",
        "gated": True,
        "approx_gb": 26.3,
    },
    "vae": {
        "label": "Qwen Image VAE",
        "note": "Required for both",
        "family": "Krea 2 — images",
        "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
        "filename": "split_files/vae/qwen_image_vae.safetensors",
        "dest": MODELS / "qwen-image-vae.safetensors",
        "gated": False,
        "approx_gb": 0.25,
    },
    "text_encoder": {
        "label": "Qwen3-VL 4B",
        "note": "Text encoder, bf16",
        "family": "Krea 2 — images",
        "repo_id": "Comfy-Org/Qwen3-VL",
        "filename": "text_encoders/qwen3vl_4b_bf16.safetensors",
        "dest": MODELS / "qwen3vl-4b-bf16.safetensors",
        "gated": False,
        "approx_gb": 8.9,
    },
    # ── MiniMax-H3 video ───────────────────────────────────────────────────
    #
    # Comfy-Org's repackage, not MiniMaxAI's release. The released checkpoint
    # is 123.6 GB in bf16; this is the same model with the modulation weights
    # (~40% of parameters, and a function of timestep and modality only, never
    # of a token) pruned into a lookup table, then quantized. 42.5 GB total.
    #
    # `fl2va` is one checkpoint covering both tasks: text-to-video when no
    # keyframe is given, image-to-video when one is. There is no second file to
    # download for i2v, and no second code path to write. `ref2va` — up to 9
    # images / 3 videos / 3 audio clips as semantic references — is a separate
    # 21 GB transformer and is deliberately not here; it is its own feature,
    # not a checkbox on this one.
    "h3_dit": {
        "label": "MiniMax-H3",
        "note": "Video DiT, int8 — t2v and i2v",
        "family": "MiniMax-H3 — video with sound",
        "repo_id": "Comfy-Org/MiniMax-H3",
        "filename": "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "dest": MODELS / "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "gated": False,
        "approx_gb": 21.0,
    },
    # The second half of H3. Same VAEs, same conditioner, different transformer:
    # this one takes an ordered list of references — up to 9 images, 3 videos,
    # 3 audio clips, 12 total — and the order is semantic, because it both
    # labels them in the prompt (<Picture 1>, <Video 1>, <Audio 1>) and advances
    # the shared audio/video rotary clock. Reordering the same references is a
    # different request, not a cosmetic change.
    #
    # Unlike a first frame, a reference does not bind the geometry: references
    # are encoded at their own resolution and the canvas stays whatever you ask
    # for. That is what makes "animate this generated image" a reference job
    # rather than a keyframe job when you want a new camera on the same subject.
    "h3_ref_dit": {
        "label": "MiniMax-H3 Reference",
        "note": "Video DiT, int8 — reference-to-video",
        "family": "MiniMax-H3 — video with sound",
        "repo_id": "Comfy-Org/MiniMax-H3",
        "filename": "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "dest": MODELS / "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "gated": False,
        "approx_gb": 21.0,
    },
    "h3_te": {
        "label": "Qwen3-VL 32B",
        "note": "H3 text encoder, nvfp4",
        "family": "MiniMax-H3 — video with sound",
        "repo_id": "Comfy-Org/MiniMax-H3",
        "filename": "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "dest": MODELS / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "gated": False,
        "approx_gb": 15.7,
    },
    "h3_vae": {
        "label": "H3 Video VAE",
        "note": "Required for video",
        "family": "MiniMax-H3 — video with sound",
        "repo_id": "Comfy-Org/MiniMax-H3",
        "filename": "vae/minimax_h3_video_vae_fp16.safetensors",
        "dest": MODELS / "minimax_h3_video_vae_fp16.safetensors",
        "gated": False,
        "approx_gb": 5.2,
    },
    "h3_audio_vae": {
        "label": "H3 Audio VAE",
        "note": "Native stereo soundtrack",
        "family": "MiniMax-H3 — video with sound",
        "repo_id": "Comfy-Org/MiniMax-H3",
        "filename": "vae/minimax_h3_audio_vae_fp32.safetensors",
        "dest": MODELS / "minimax_h3_audio_vae_fp32.safetensors",
        "gated": False,
        "approx_gb": 0.6,
    },
    # ── Wan 2.2 video ──────────────────────────────────────────────────────
    #
    # Comfy-Org's repackage again, and for the same reason the H3 entries use
    # one: these are the split single-file weights ComfyUI's loaders take, not
    # the diffusers layout Wan-AI publishes.
    #
    # The A14B models are a mixture of two experts, and it is a *storage*
    # mixture, not a runtime one: high-noise denoises the early steps, low-noise
    # the rest, and they are separate 14 GB checkpoints. So a 14B task is always
    # two files, never one, and the pair is what the run needs — half of it
    # downloaded is not a model that runs at half quality, it is a model that
    # does not run. `fp8_scaled` rather than the 28.6 GB fp16: two experts plus
    # the encoder is 35 GB this way against 68 GB, which is the difference
    # between holding both resident on one H100 and swapping every switch.
    "wan_t2v_high": {
        "label": "Wan 2.2 T2V · high noise",
        "note": "14B fp8 — early steps",
        "family": "Wan 2.2 — video",
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "filename": "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
        "dest": MODELS / "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
        "gated": False,
        "approx_gb": 14.3,
    },
    "wan_t2v_low": {
        "label": "Wan 2.2 T2V · low noise",
        "note": "14B fp8 — late steps",
        "family": "Wan 2.2 — video",
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "filename": "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
        "dest": MODELS / "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
        "gated": False,
        "approx_gb": 14.3,
    },
    "wan_i2v_high": {
        "label": "Wan 2.2 I2V · high noise",
        "note": "14B fp8 — early steps",
        "family": "Wan 2.2 — video",
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "filename": "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        "dest": MODELS / "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        "gated": False,
        "approx_gb": 14.3,
    },
    "wan_i2v_low": {
        "label": "Wan 2.2 I2V · low noise",
        "note": "14B fp8 — late steps",
        "family": "Wan 2.2 — video",
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "filename": "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        "dest": MODELS / "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        "gated": False,
        "approx_gb": 14.3,
    },
    # The 5B is not a smaller A14B — it is a single dense model with its own
    # VAE at 16x spatial compression (48 latent channels against 16), which is
    # why it needs `wan2.2_vae` and the 14B pair needs `wan_2.1_vae`. One file,
    # one expert, no switch step: it is the whole Wan stack for 10 GB, and it
    # covers text-to-video and image-to-video from the same checkpoint.
    "wan_ti2v_5b": {
        "label": "Wan 2.2 TI2V 5B",
        "note": "Single 5B DiT — t2v and i2v, 24 fps",
        "family": "Wan 2.2 — video",
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "filename": "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
        "dest": MODELS / "wan2.2_ti2v_5B_fp16.safetensors",
        "gated": False,
        "approx_gb": 10.0,
    },
    "wan_te": {
        "label": "umT5-XXL",
        "note": "Wan text encoder, fp8 scaled",
        "family": "Wan 2.2 — video",
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "filename": "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "dest": MODELS / "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "gated": False,
        "approx_gb": 6.7,
    },
    "wan_vae": {
        "label": "Wan 2.1 VAE",
        "note": "Required by the 14B pair",
        "family": "Wan 2.2 — video",
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "filename": "split_files/vae/wan_2.1_vae.safetensors",
        "dest": MODELS / "wan_2.1_vae.safetensors",
        "gated": False,
        "approx_gb": 0.25,
    },
    "wan_vae_22": {
        "label": "Wan 2.2 VAE",
        "note": "Required by the 5B",
        "family": "Wan 2.2 — video",
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "filename": "split_files/vae/wan2.2_vae.safetensors",
        "dest": MODELS / "wan2.2_vae.safetensors",
        "gated": False,
        "approx_gb": 1.4,
    },
    # ── Wan 2.2 speed LoRAs ────────────────────────────────────────────────
    #
    # These land in loras/, not models/, because that is what they are: they
    # show up in the LoRA picker beside anything you trained, and nothing
    # special-cases them. They are here rather than left to a manual upload
    # because they are the 14B path's draft tier — 4 steps instead of 20, at
    # CFG 1, which is a bigger lever than resolution and the reason the LoRA
    # stack below is worth having on day one.
    #
    # Paired per expert, and the pair is not interchangeable: the high-noise
    # LoRA on the low-noise expert is a silent quality loss, not an error. The
    # folder holds both files so the picker shows them as `high` and `low` of
    # one thing.
    "wan_speed_t2v_high": {
        "label": "Wan 2.2 T2V speed · high",
        "note": "LightX2V 4-step LoRA",
        "family": "Wan 2.2 speed LoRAs",
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "filename": "split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
        "dest": LORAS / "wan22-speed-t2v" / "high.safetensors",
        "gated": False,
        "approx_gb": 1.2,
    },
    "wan_speed_t2v_low": {
        "label": "Wan 2.2 T2V speed · low",
        "note": "LightX2V 4-step LoRA",
        "family": "Wan 2.2 speed LoRAs",
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "filename": "split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors",
        "dest": LORAS / "wan22-speed-t2v" / "low.safetensors",
        "gated": False,
        "approx_gb": 1.2,
    },
    "wan_speed_i2v_high": {
        "label": "Wan 2.2 I2V speed · high",
        "note": "LightX2V 4-step LoRA",
        "family": "Wan 2.2 speed LoRAs",
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "filename": "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
        "dest": LORAS / "wan22-speed-i2v" / "high.safetensors",
        "gated": False,
        "approx_gb": 1.2,
    },
    "wan_speed_i2v_low": {
        "label": "Wan 2.2 I2V speed · low",
        "note": "LightX2V 4-step LoRA",
        "family": "Wan 2.2 speed LoRAs",
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "filename": "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
        "dest": LORAS / "wan22-speed-i2v" / "low.safetensors",
        "gated": False,
        "approx_gb": 1.2,
    },
}

# The video weights keep their upstream filenames, unlike every other entry
# above. ComfyUI addresses models by basename inside a search path, so a
# renamed file would have to be renamed again in every graph that names it —
# and the names Comfy-Org ships already say the quantization, which is the one
# thing you need to read off a video checkpoint at a glance.
VIDEO_MODEL_KEYS = ("h3_dit", "h3_te", "h3_vae", "h3_audio_vae")
# ref2va shares everything but the transformer, so it is one extra 21 GB file
# on top of the base set rather than a second stack.
VIDEO_REF_MODEL_KEYS = ("h3_ref_dit", "h3_te", "h3_vae", "h3_audio_vae")

# What each Wan run needs, keyed by (family, task). Written out per task rather
# than as "the Wan models" because the t2v and i2v 14B pairs are 57 GB together
# and there is no reason to make a text-to-video run wait on the i2v weights —
# _require_models() names only the pair the requested run actually loads.
WAN_MODEL_KEYS: dict[tuple[str, str], tuple[str, ...]] = {
    ("14b", "t2v"): ("wan_t2v_high", "wan_t2v_low", "wan_te", "wan_vae"),
    ("14b", "i2v"): ("wan_i2v_high", "wan_i2v_low", "wan_te", "wan_vae"),
    ("5b", "t2v"): ("wan_ti2v_5b", "wan_te", "wan_vae_22"),
    ("5b", "i2v"): ("wan_ti2v_5b", "wan_te", "wan_vae_22"),
}

RAW_PATH = MODEL_CATALOGUE["raw"]["dest"]
TURBO_PATH = MODEL_CATALOGUE["turbo"]["dest"]
VAE_PATH = MODEL_CATALOGUE["vae"]["dest"]
TE_PATH = MODEL_CATALOGUE["text_encoder"]["dest"]

# Qwen3-VL-8B-Instruct, not a booru tagger and not JoyCaption.
#
# Krea 2 reads its prompt through Qwen3-VL-4B, a language model that parses
# grammar — subordinate clauses, spatial prepositions, and the binding between
# an adjective and the noun it modifies. Tag lists cannot express binding at
# all: "red, blue, dress, jacket" leaves the model to guess which colour
# belongs to which garment, and with two people in frame that guess is where
# attribute bleed comes from. A sentence resolves it by construction.
#
# The second reason is symmetry. You prompt in prose, so the captions should be
# prose — any gap between how the dataset describes an image and how you
# describe the one you want is a gap the model has to guess across. Captioning
# with the same model family the text encoder comes from closes that gap
# further than a differently-trained captioner can.
CAPTION_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif"}
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_CAPTION_CHARS = 1024
THUMB_PX = 320

# All prose. The booru-tag style that used to live here is gone rather than
# hidden: it existed to match CLIP's 77-token bag of words, and emitting it for
# a grammar-parsing encoder would be actively worse than the default.
CAPTION_STYLES = {
    "descriptive": (
        "Describe this image in plain, factual prose. Name the subject and what it is "
        "doing, then its appearance, clothing, pose, setting, lighting and style. "
        "Attach every adjective to the noun it belongs to, so it is unambiguous which "
        "garment or object each colour and material describes. Do not speculate about "
        "anything you cannot see. Do not begin with 'This image' or 'The photo'."
    ),
    "casual": (
        "Describe this image in natural, conversational prose, the way you would "
        "describe a photo to someone who cannot see it. Cover what is happening, how "
        "it looks and the overall mood, keeping each adjective clearly attached to the "
        "thing it describes. Do not begin with 'This image' or 'The photo'."
    ),
}
CAPTION_LENGTHS = {
    "short": " Keep it to one dense sentence.",
    "medium": " Keep it to two or three sentences.",
    "long": " Be thorough, four or more sentences.",
}


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _publish(job_id: str, /, **fields: Any) -> None:
    """
    Merge progress fields into the job record the UI polls.

    `job_id` is positional-only for a reason. Every completion path builds a
    result dict that carries its own "job_id" (it is the function's return
    value as well as the record) and then calls `_publish(job_id, **res)`. With
    a normal parameter that is `TypeError: got multiple values for argument
    'job_id'`, raised *after* the images are already on the volume and outside
    the try block that would have marked the job failed — so the file lands,
    the status stays "running" forever, and the UI polls a job that will never
    answer. Positional-only makes the collision impossible instead of relying
    on every caller remembering to pop the key.
    """
    try:
        cur = jobs.get(job_id) or {}
        cur.update(fields)
        jobs[job_id] = cur
    except Exception as exc:  # progress must never take the job down
        print(f"[progress] {job_id}: {exc}")


def _stop_requested(job_id: str) -> bool:
    try:
        return bool((jobs.get(job_id) or {}).get("stop"))
    except Exception:
        return False


def _hf_token() -> str | None:
    """The token pasted into the UI, stored in a Modal Dict. No Secrets needed."""
    try:
        tok = (config.get("hf_token") or "").strip()
        return tok or None
    except Exception:
        return None


# tqdm writes progress with \r; text mode uses universal newlines so each update
# arrives as its own line. This pulls step counts out of musubi's "steps:" bar.
TQDM_RE = re.compile(
    r"(?P<pct>\d+)%\|[^|]*\|\s*(?P<step>\d+)/(?P<total>\d+)"
    r"(?:\s*\[(?P<el>[\d:]+)<(?P<eta>[\d:?]+),\s*(?P<rate>[\d.]+)(?P<unit>s/it|it/s))?"
)
LOSS_RE = re.compile(r"avg_loss=(?P<loss>[\d.]+)")
EPOCH_RE = re.compile(r"epoch (?P<cur>\d+)/(?P<total>\d+)")


def _run(cmd: list[str], label: str, job_id: str, log: deque[str]) -> None:
    """
    Run a subprocess, streaming output to Modal logs, parsing progress for the
    UI, and honouring a cooperative stop request between lines.
    """
    print(f"\n===== {label} =====\n$ {' '.join(cmd)}", flush=True)
    _publish(job_id, phase=label, step=0, total_steps=0, percent=0)

    env = {**os.environ, "PYTHONPATH": str(MUSUBI / "src")}
    proc = subprocess.Popen(
        cmd, cwd=str(MUSUBI), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert proc.stdout is not None

    stopped = False
    last_push = 0.0
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            print(line, flush=True)
            log.append(line)

        m = TQDM_RE.search(line)
        if m and time.time() - last_push > 1.0:
            last_push = time.time()
            fields: dict[str, Any] = {
                "phase": label,
                "step": int(m.group("step")),
                "total_steps": int(m.group("total")),
                "percent": int(m.group("pct")),
            }
            if m.group("eta"):
                fields["eta"] = m.group("eta")
                fields["rate"] = f"{m.group('rate')}{m.group('unit')}"
            if (lm := LOSS_RE.search(line)):
                fields["loss"] = float(lm.group("loss"))
            if (em := EPOCH_RE.search(line)):
                fields["epoch"] = int(em.group("cur"))
                fields["total_epochs"] = int(em.group("total"))
            _publish(job_id, **fields)

        if not stopped and _stop_requested(job_id):
            # Cooperative stop. musubi has no SIGINT handler, so terminating is
            # the only lever — periodic checkpoints are what make it survivable.
            stopped = True
            print(f"[{job_id}] stop requested — terminating {label}", flush=True)
            _publish(job_id, phase=f"stopping {label}")
            proc.terminate()

    code = proc.wait()
    if stopped:
        raise StopRequested(label)
    if code != 0:
        raise RuntimeError(f"{label} failed (exit {code}).\n" + "\n".join(list(log)[-25:]))


class StopRequested(Exception):
    """Raised when the user pressed Stop; not a failure."""


def _safe_extract_zip(zip_path: Path, dest: Path) -> int:
    """Extract images from a zip, flattened. Rebuilds names from the basename so
    `../` members and absolute paths cannot escape (zip-slip)."""
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir() or "__MACOSX" in member.filename:
                continue
            name = Path(member.filename).name
            if not name or name.startswith("."):
                continue
            suffix = Path(name).suffix.lower()
            if suffix not in IMAGE_EXTS and suffix != ".txt":
                continue
            with zf.open(member) as src, open(dest / name, "wb") as out:
                shutil.copyfileobj(src, out)
            if suffix in IMAGE_EXTS:
                count += 1
    return count


def _require_models(*keys: str) -> None:
    """
    Assert the given models are on the volume, and if not, say what IS there.

    "Not downloaded" is useless when the file looks present in the UI. The
    listing distinguishes the three real causes at a glance: an empty volume
    (wrong Modal profile), a filename or case mismatch, or a partial download.
    """
    sizes = _sizes_on_disk(MODEL_CATALOGUE[k]["dest"] for k in keys)
    missing = [MODEL_CATALOGUE[k] for k in keys
               if not sizes[MODEL_CATALOGUE[k]["dest"]]]
    if not missing:
        return

    lines = [f"Missing: {', '.join(s['label'] for s in missing)}", ""]
    for spec in missing:
        lines.append(f"  expected: {spec['dest']}")
    lines.append("")
    lines.append("Actually on the volume:")
    for d in sorted({MODEL_CATALOGUE[k]["dest"].parent for k in keys}):
        if not d.is_dir():
            lines.append(f"  {d}/  (directory does not exist)")
            continue
        entries = sorted(p.name for p in d.iterdir() if not p.name.startswith("."))
        if entries:
            for name in entries[:12]:
                size = (d / name).stat().st_size / 1e9
                lines.append(f"  {d}/{name}  ({size:.2f} GB)")
            if len(entries) > 12:
                lines.append(f"  … and {len(entries) - 12} more")
        else:
            lines.append(f"  {d}/  (empty)")
    lines += [
        "",
        f"Resolved volume: {VOLUME_NAME!r} (override with VISIONARY_VOLUME). "
        "If those directories are empty it is a new or wrong volume, not a "
        "failed download — check `modal profile current` and the Models tab.",
    ]
    raise RuntimeError("\n".join(lines))


def _sizes_on_disk(dests: Any) -> dict[Path, int]:
    """
    {dest: size in bytes} for a set of weight paths, 0 when absent.

    One readdir per directory, deliberately, rather than a stat per file.

    A container that asked for a weight *before* it was downloaded could keep
    answering "not there" long after it landed: the failed lookup is cached
    below us, and `volume.reload()` brings the volume forward without
    invalidating a name we have already asked about. That is exactly why Krea 2
    was always reported correctly and the video weights were not — the Krea
    files were on the volume before any of these containers started, so nothing
    negative was ever cached for them, while Wan and H3 were downloaded into a
    deployment that had already been asked whether they were there. The symptom
    was being told to download 18 GB that were sitting on the volume.

    Reading the parent directory sidesteps it: readdir reports what the
    directory holds now, so a file that has appeared is a file we see. It is
    also fewer syscalls than statting twenty-one paths one at a time.
    """
    listings: dict[Path, dict[str, int]] = {}
    out: dict[Path, int] = {}
    for dest in dests:
        entries = listings.get(dest.parent)
        if entries is None:
            try:
                entries = {e.name: e.stat().st_size
                           for e in os.scandir(dest.parent) if e.is_file()}
            except (FileNotFoundError, NotADirectoryError):
                entries = {}
            listings[dest.parent] = entries
        out[dest] = entries.get(dest.name, 0)
    return out


def _model_status() -> list[dict[str, Any]]:
    sizes = _sizes_on_disk(spec["dest"] for spec in MODEL_CATALOGUE.values())
    out = []
    for key, spec in MODEL_CATALOGUE.items():
        dest: Path = spec["dest"]
        present = sizes[dest] > 0
        out.append(
            {
                "key": key,
                "label": spec["label"],
                "note": spec["note"],
                # Grouping the sheet, and the order groups appear in is the
                # order they appear in the catalogue — one source of truth
                # rather than a second list that drifts when a model is added.
                "family": spec["family"],
                "repo_id": spec["repo_id"],
                "gated": spec["gated"],
                "approx_gb": spec["approx_gb"],
                "present": present,
                "size_gb": round(sizes[dest] / 1e9, 2) if present else 0,
                "path": str(dest),
            }
        )
    return out


# --------------------------------------------------------------------------
# Downloads — CPU only.
#
# Deliberately not on the GPU function: pulling 26 GB while an A100 idles is
# money burned for nothing. This runs on plain CPU at a fraction of the cost.
# --------------------------------------------------------------------------


@app.function(image=web_image, cpu=2.0, timeout=4 * 60 * 60, volumes={"/workspace": volume})
def download_job(key: str) -> dict[str, Any]:
    job_id = f"dl_{key}"
    jobs[job_id] = {
        "status": "running",
        "phase": f"Downloading {MODEL_CATALOGUE[key]['label']}",
        "percent": 0,
    }
    return _download_weight(key, job_id)


def _download_weight(key: str, job_id: str) -> dict[str, Any]:
    """Fetch one weight to its exact destination. Shared by single and bulk downloads."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import (
        EntryNotFoundError, GatedRepoError, RepositoryNotFoundError,
    )

    spec = MODEL_CATALOGUE[key]
    dest: Path = spec["dest"]

    volume.reload()
    if dest.exists() and dest.stat().st_size > 0:
        res = {"status": "completed", "key": key, "note": "already present"}
        _publish(job_id, **res)
        return res

    token = _hf_token()
    if spec["gated"] and not token:
        err = (
            f"{spec['label']} is a gated repo. Paste your HuggingFace token on the "
            f"Models tab, and accept the licence at https://huggingface.co/{spec['repo_id']}"
        )
        _publish(job_id, status="failed", error=err)
        raise RuntimeError(err)

    dest.parent.mkdir(parents=True, exist_ok=True)
    STAGING.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f"[download] {spec['label']}: {spec['repo_id']}/{spec['filename']}")

    try:
        staged = hf_hub_download(
            repo_id=spec["repo_id"],
            filename=spec["filename"],
            local_dir=str(STAGING),
            token=token,
        )
    except GatedRepoError:
        err = (
            f"Access to {spec['repo_id']} was refused. Accept the licence at "
            f"https://huggingface.co/{spec['repo_id']} using the same account "
            "that issued this token."
        )
        _publish(job_id, status="failed", error=err)
        raise RuntimeError(err)
    except RepositoryNotFoundError:
        err = f"Repo {spec['repo_id']} not found, or the token cannot see it."
        _publish(job_id, status="failed", error=err)
        raise RuntimeError(err)
    except EntryNotFoundError:
        err = f"{spec['filename']} is missing from {spec['repo_id']}."
        _publish(job_id, status="failed", error=err)
        raise RuntimeError(err)

    # Same filesystem, so this is an instant rename rather than a 26 GB copy.
    shutil.move(staged, dest)
    volume.commit()

    res = {
        "status": "completed",
        "key": key,
        "size_gb": round(dest.stat().st_size / 1e9, 2),
        "duration_s": round(time.time() - started, 1),
    }
    _publish(job_id, **res)
    print(f"[download] {spec['label']}: {res['size_gb']} GB in {res['duration_s']}s")
    return res


@app.function(image=web_image, cpu=2.0, timeout=6 * 60 * 60, volumes={"/workspace": volume})
def download_missing_job(keys: list[str]) -> dict[str, Any]:
    """
    Fetch every missing weight in one container, sequentially.

    Sequential rather than four parallel containers: these are large files
    sharing one uplink, so running them at once mostly splits the same bandwidth
    while multiplying container cost — and it gives the UI a single job to
    follow instead of four independent ones.
    """
    job_id = "dl_all"
    done, failed = [], []
    for i, key in enumerate(keys, 1):
        label = MODEL_CATALOGUE[key]["label"]
        _publish(
            job_id,
            status="running",
            phase=f"{label} ({i} of {len(keys)})",
            index=i,
            total=len(keys),
            percent=round((i - 1) / len(keys) * 100),
        )
        try:
            _download_weight(key, job_id)
            done.append(key)
        except Exception as exc:
            # One gated repo must not abandon the rest of the queue.
            print(f"[download-all] {label} failed: {exc}")
            failed.append({"key": key, "label": label, "error": str(exc)})

    res: dict[str, Any] = {
        "status": "completed" if not failed else "failed",
        "downloaded": done,
        "failed": failed,
        "percent": 100,
    }
    if failed:
        res["error"] = "; ".join(f["error"] for f in failed)
    _publish(job_id, **res)
    return res


# --------------------------------------------------------------------------
# Captioning
# --------------------------------------------------------------------------


def _caption_images(
    image_dir: Path, trigger_word: str, job_id: str,
    style: str, length: str, overwrite: bool,
) -> int:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    every = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    todo = [
        p for p in every
        if overwrite
        or not p.with_suffix(".txt").exists()
        or not p.with_suffix(".txt").read_text().strip()
    ]
    if not todo:
        return 0

    print(f"[caption] {len(todo)}/{len(every)} images")
    _publish(job_id, phase="caption", step=0, total_steps=len(todo), percent=0)

    cache_dir = str(HF_CACHE)
    # Cap the vision tower's token budget. Qwen3-VL scales patches with input
    # resolution, so a 4000px training image would otherwise spend thousands of
    # tokens on detail that never reaches the caption — slow, and no better.
    processor = AutoProcessor.from_pretrained(
        CAPTION_MODEL, cache_dir=cache_dir,
        min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        CAPTION_MODEL, dtype=torch.bfloat16, device_map="cuda:0", cache_dir=cache_dir,
    )
    model.eval()
    # Persist the downloaded weights now, on their own volume, so the next cold
    # start reuses them and the dataset commit below stays small.
    try:
        hf_cache.commit()
    except Exception as exc:
        print(f"[caption] hf cache commit skipped: {exc}")

    instruction = CAPTION_STYLES.get(style, CAPTION_STYLES["descriptive"])
    instruction += CAPTION_LENGTHS.get(length, CAPTION_LENGTHS["medium"])

    written = 0
    for i, img_path in enumerate(todo, 1):
        if _stop_requested(job_id):
            print("[caption] stop requested")
            break
        try:
            image = Image.open(img_path).convert("RGB")
            # Qwen3-VL takes the image as a content part, not an inline <image>
            # token, and apply_chat_template does the placeholder expansion.
            convo = [{
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}],
            }]
            inputs = processor.apply_chat_template(
                convo, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to("cuda:0")

            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=320, do_sample=True,
                    temperature=0.6, top_p=0.9,
                )[0][inputs["input_ids"].shape[1]:]
            caption = processor.decode(out, skip_special_tokens=True).strip()

            # Strip stock preambles so the trigger word stays at the very front,
            # which is where it does its work.
            for junk in ("This image shows ", "The image shows ", "This image depicts "):
                if caption.startswith(junk):
                    caption = caption[len(junk):]
                    caption = caption[:1].upper() + caption[1:]
                    break

            final = f"{trigger_word}, {caption}" if trigger_word else caption
            img_path.with_suffix(".txt").write_text(final[:MAX_CAPTION_CHARS])
            written += 1
        except Exception as exc:
            print(f"[caption] {img_path.name} failed: {exc}")

        _publish(job_id, phase="caption", step=i, total_steps=len(todo),
                 percent=round(i / len(todo) * 100))

    del model
    torch.cuda.empty_cache()
    volume.commit()
    return written


@app.function(
    # A100 rather than the training GPU: Qwen3-VL-8B in bf16 is ~17 GB, so this
    # does not need the headroom a rank-32 Krea 2 run does.
    image=caption_image, gpu="A100-40GB", cpu=2.0, timeout=2 * 60 * 60,
    volumes={"/workspace": volume, str(HF_CACHE): hf_cache},
)
def caption_job(
    job_id: str, dataset: str, trigger_word: str = "", style: str = "descriptive",
    length: str = "medium", overwrite: bool = False,
) -> dict[str, Any]:
    jobs[job_id] = {"status": "running", "phase": "caption", "stop": False}
    volume.reload()
    src = _dataset_dir(dataset)
    if not src.is_dir():
        raise RuntimeError(f"No dataset named {dataset!r}.")

    started = time.time()
    written = _caption_images(src, trigger_word.strip(), job_id, style, length, overwrite)
    res = {
        "status": "completed", "job_id": job_id, "dataset": dataset,
        "captioned": written, "duration_s": round(time.time() - started, 1),
    }
    _publish(job_id, **res)
    return res


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


@app.function(
    image=trainer_image, gpu=GPU, cpu=4.0, timeout=6 * 60 * 60,
    volumes={"/workspace": volume},
)
def train_job(
    job_id: str, dataset: str, lora_name: str, trigger_word: str,
    resolution: int = 1024, batch_size: int = 1, num_repeats: int = 1,
    network_dim: int = 32, network_alpha: int = 32, learning_rate: float = 1e-4,
    max_train_epochs: int = 30, save_every_n_epochs: int = 1,
    discrete_flow_shift: float = 2.5, seed: int = 42,
    fp8: bool = False, blocks_to_swap: int = 0,
) -> dict[str, Any]:
    if not NAME_RE.match(job_id) or not NAME_RE.match(lora_name):
        raise ValueError("job_id and lora_name must be 1-64 chars of [A-Za-z0-9_-].")

    started = time.time()
    log: deque[str] = deque(maxlen=400)
    jobs[job_id] = {"status": "running", "phase": "starting", "stop": False}
    volume.reload()

    _require_models("raw", "vae", "text_encoder")

    src = _dataset_dir(dataset)
    if not src.is_dir():
        raise RuntimeError(f"No dataset named {dataset!r}.")

    # Copy into per-run scratch rather than training out of the dataset: musubi
    # writes latent caches beside the images and rewrites missing .txt files, and
    # a dataset that survives its training runs must not accumulate either.
    work = WORK / job_id
    image_dir, cache_dir = work / "images", work / "cache"
    image_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.suffix.lower() in IMAGE_EXTS or item.suffix.lower() == ".txt":
            shutil.copy2(item, image_dir / item.name)

    images = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        raise RuntimeError("No images to train on.")

    # Krea 2 trains from caption files; a missing .txt is a hard error inside the
    # cache step, so an uncaptioned image gets the bare trigger word.
    for img in images:
        txt = img.with_suffix(".txt")
        if not txt.exists() or not txt.read_text().strip():
            txt.write_text(trigger_word)

    # The keys here are image_directory / cache_directory. musubi validates this
    # with a voluptuous Schema that does NOT allow extra keys, so the shorter
    # image_dir / cache_dir spellings raise MultipleInvalid rather than being
    # ignored — the most common way this config fails.
    toml_path = work / "dataset.toml"
    toml_path.write_text(
        f"""[general]
resolution = [{resolution}, {resolution}]
caption_extension = ".txt"
batch_size = {batch_size}
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "{image_dir}"
cache_directory = "{cache_dir}"
num_repeats = {num_repeats}
"""
    )
    volume.commit()

    out_dir = LORAS / lora_name
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        _run(
            ["python", "src/musubi_tuner/krea2_cache_latents.py",
             "--dataset_config", str(toml_path), "--vae", str(VAE_PATH)],
            "cache latents", job_id, log,
        )
        _run(
            ["python", "src/musubi_tuner/krea2_cache_text_encoder_outputs.py",
             "--dataset_config", str(toml_path), "--text_encoder", str(TE_PATH),
             "--batch_size", "1"],
            "cache text encoder", job_id, log,
        )
        volume.commit()

        mem: list[str] = []
        if fp8:
            # musubi rejects --fp8_base without --fp8_scaled (plain fp8 casts the
            # norms and breaks the model), so these always travel together.
            mem += ["--fp8_base", "--fp8_scaled"]
        if blocks_to_swap > 0:
            mem += ["--blocks_to_swap", str(blocks_to_swap)]

        _run(
            [
                # Explicit flags rather than `accelerate config`, which is
                # interactive and cannot run in a detached container.
                "accelerate", "launch",
                "--num_processes", "1", "--num_machines", "1",
                "--mixed_precision", "bf16", "--dynamo_backend", "no",
                "--num_cpu_threads_per_process", "1",
                "src/musubi_tuner/krea2_train_network.py",
                # --dit is the RAW DiT, --vae is the Qwen VAE. Swapping these two
                # is the classic mistake; they are different models.
                "--dit", str(RAW_PATH),
                "--vae", str(VAE_PATH),
                "--dataset_config", str(toml_path),
                "--sdpa", "--mixed_precision", "bf16",
                "--timestep_sampling", "shift", "--weighting_scheme", "none",
                "--discrete_flow_shift", str(discrete_flow_shift),
                "--optimizer_type", "adamw8bit",
                "--learning_rate", str(learning_rate),
                "--gradient_checkpointing",
                "--max_data_loader_n_workers", "2", "--persistent_data_loader_workers",
                "--network_module", "networks.lora_krea2",
                "--network_dim", str(network_dim), "--network_alpha", str(network_alpha),
                "--max_train_epochs", str(max_train_epochs),
                "--save_every_n_epochs", str(save_every_n_epochs),
                "--seed", str(seed),
                "--output_dir", str(out_dir),
                # No extension — musubi appends .safetensors itself, and passing
                # it produces name.safetensors.safetensors.
                "--output_name", lora_name,
                *mem,
            ],
            "train", job_id, log,
        )
        status = "completed"
    except StopRequested:
        status = "stopped"

    produced = sorted(p.name for p in out_dir.glob("*.safetensors"))
    (out_dir / "visionary.json").write_text(
        json.dumps(
            {"job_id": job_id, "dataset": dataset, "lora_name": lora_name,
             "trigger_word": trigger_word,
             "images": len(images), "status": status, "files": produced,
             "hyperparams": {
                 "resolution": resolution, "network_dim": network_dim,
                 "network_alpha": network_alpha, "learning_rate": learning_rate,
                 "max_train_epochs": max_train_epochs, "seed": seed,
             }},
            indent=2,
        )
    )
    volume.commit()

    res = {
        "status": status, "job_id": job_id, "dataset": dataset,
        "lora_name": lora_name,
        "trigger_word": trigger_word, "images": len(images),
        "output_dir": str(out_dir), "files": produced,
        "duration_s": round(time.time() - started, 1),
        "log_tail": list(log)[-30:],
    }
    if status == "stopped":
        res["note"] = (
            "Stopped. Checkpoints saved up to the last completed epoch are kept."
            if produced else "Stopped before the first checkpoint — nothing saved."
        )
    _publish(job_id, **res)
    return res


# --------------------------------------------------------------------------
# Generation
#
# Backed by sd-webui-forge-classic (neo), not musubi. musubi's
# krea2_generate_image.py is a one-shot CLI: it reloaded ~35 GB of weights for
# every image and took a single LoRA at a single strength. This is a Modal Cls,
# so the checkpoint stays resident between requests, and LoRAs stack the way
# they do in Forge — any number of them, each with its own UNet and text-encoder
# weight. See forge/VENDOR.md and forge/krea2/.
# --------------------------------------------------------------------------


# Mirrors krea2.sampler_names() / scheduler_names(). Duplicated rather than
# imported because /api/state runs in the CPU web image, and importing the forge
# backend needs a CUDA device — see forge/krea2/bootstrap.py. tools/smoke_krea2.py
# prints the authoritative lists if these ever need re-checking.
SAMPLERS = [
    "Euler", "Euler a", "DPM++ 2M", "DPM++ 2M SDE", "DPM++ 3M SDE",
    "DPM++ SDE", "DPM++ 2s a RF", "Heun", "LMS", "LCM", "ER SDE", "Res Multistep",
]
SCHEDULERS = [
    "Automatic", "Karras", "Exponential", "Polyexponential", "Normal", "Simple",
    "Uniform", "SGM Uniform", "Linear Quadratic", "KL Optimal", "DDIM",
    "Align Your Steps", "Beta", "Turbo", "Bong Tangent", "FlowMatchEulerDiscrete",
]
MAX_LORAS = 6


# --------------------------------------------------------------------------
# Datasets
#
# Named, reusable, and independent of any training run — caption once, train a
# rank sweep from it. The directory is the whole model: images plus .txt
# sidecars, which is exactly what musubi consumes, so there is no database to
# fall out of sync with the files and no export step.
# --------------------------------------------------------------------------


def _dataset_dir(name: str) -> Path:
    if not NAME_RE.match(name or ""):
        raise ValueError("Dataset names are 1-64 chars of letters, numbers, _ or -.")
    return DATASETS / name


def _dataset_images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS) if d.is_dir() else []


def _caption_of(img: Path) -> str:
    txt = img.with_suffix(".txt")
    try:
        return txt.read_text().strip() if txt.is_file() else ""
    except OSError:
        return ""


def _dataset_stats(d: Path) -> dict[str, Any]:
    images = _dataset_images(d)
    captioned = sum(1 for p in images if _caption_of(p))
    meta = d / "dataset.json"
    info: dict[str, Any] = {}
    if meta.is_file():
        try:
            info = json.loads(meta.read_text())
        except (OSError, json.JSONDecodeError):
            info = {}
    return {
        "name": d.name,
        "count": len(images),
        "captioned": captioned,
        "uncaptioned": len(images) - captioned,
        "trigger_word": str(info.get("trigger_word") or ""),
        "modified": max((p.stat().st_mtime for p in images), default=0.0),
        "cover": images[0].name if images else None,
    }


def _write_dataset_meta(d: Path, **fields: Any) -> dict[str, Any]:
    meta = d / "dataset.json"
    info: dict[str, Any] = {}
    if meta.is_file():
        try:
            info = json.loads(meta.read_text())
        except (OSError, json.JSONDecodeError):
            info = {}
    info.update({k: v for k, v in fields.items() if v is not None})
    meta.write_text(json.dumps(info, indent=2))
    return info


# Clause boundaries. Humans write captions as comma-delimited clauses, so this
# is what a hand-edited caption divides into.
_CLAUSE_RE = re.compile(r"[,.;:!?\n]+")


def _caption_insight(d: Path, trigger: str = "", top: int = 24) -> dict[str, Any]:
    """
    What is this dataset accidentally teaching the model?

    The tag-frequency histogram this replaces counted booru tags, which only
    made sense while the text encoder was CLIP — a 77-token bag of words. Krea 2
    reads through Qwen3-VL, which parses grammar, so captions are prose and the
    unit that carries the same signal is the recurring *phrase*: if most
    captions open "a woman standing in", the model learns that as surely as it
    would learn an over-weighted tag. Tags cannot even express the failure they
    were used to detect — "red, blue, dress, jacket" does not say which colour
    binds to which garment, and that ambiguity is where attribute bleed starts.

    Everything here is derived from the .txt files on demand. Nothing is cached,
    because a dataset is a few hundred captions and a stale panel would be worse
    than a recomputed one.
    """
    images = _dataset_images(d)
    captions = [(p.name, _caption_of(p)) for p in images]
    non_empty = [(n, c) for n, c in captions if c]

    trigger = (trigger or "").strip()
    if not trigger:
        try:
            trigger = str(json.loads((d / "dataset.json").read_text()).get("trigger_word") or "")
        except (OSError, json.JSONDecodeError):
            trigger = ""

    # Substring, not token match: triggers are often deliberately unwordlike
    # ("ohwx_style"), and a word-boundary test would miss them inside prose.
    with_trigger = [n for n, c in captions if trigger and trigger.lower() in c.lower()]

    lengths = sorted(len(c.split()) for _, c in non_empty)
    median = lengths[len(lengths) // 2] if lengths else 0
    # Short captions are weak signal; flag them relative to this dataset rather
    # than an absolute cutoff, since a style set and a character set differ.
    floor = max(4, median // 3)
    thin = [n for n, c in non_empty if len(c.split()) < floor]

    # Comma-delimited clauses, counted whole.
    #
    # This watches hand-written and hand-edited captions, not generated ones.
    # A VLM varies its phrasing, so counting sub-phrases of its output just
    # splits one idea across rows — "she wears" and "she is wearing" are the
    # same statement and neither count means anything alone. People are the
    # ones who make caption mistakes, and people write in clauses: they paste a
    # description across a batch, they leave tag-era fragments in a prose set,
    # they duplicate a caption while editing. A whole clause repeating verbatim
    # is copy-paste, not coincidence, which is why the comma is the right unit
    # here even though it was the wrong one for n-grams.
    #
    # On clean generated captions this panel stays quiet. That is the correct
    # reading, not a failure to find anything.
    def _clauses(text: str) -> list[str]:
        out = []
        for raw in _CLAUSE_RE.split(text):
            c = " ".join(raw.split()).strip().lower()
            if c and (not trigger or c != trigger.lower()):
                out.append(c)
        return out

    # Whole captions that are byte-identical after normalising — almost always
    # a paste that was never edited.
    whole: dict[str, list[str]] = {}
    for name, caption in non_empty:
        whole.setdefault(" ".join(caption.split()).strip().lower(), []).append(name)
    duplicates = sorted(
        ({"caption": text[:180], "images": names, "count": len(names)}
         for text, names in whole.items() if len(names) > 1),
        key=lambda r: -r["count"],
    )

    clause_use: dict[str, set[str]] = {}
    for name, caption in non_empty:
        for c in set(_clauses(caption)):
            clause_use.setdefault(c, set()).add(name)

    total = len(non_empty) or 1
    repeated_clauses = sorted(
        ({"phrase": c, "count": len(names), "share": round(len(names) / total, 3),
          "words": len(c.split())}
         for c, names in clause_use.items() if len(names) > 1),
        key=lambda r: (-r["count"], -r["words"], r["phrase"]),
    )[:top]

    # Tag-era leftovers: a caption of many short comma fragments in a set that
    # is otherwise prose. Krea 2 reads grammar, so a fragment list is a real
    # mistake now rather than a style choice.
    tag_style = []
    for name, caption in non_empty:
        cl = _clauses(caption)
        if len(cl) >= 4 and sum(len(c.split()) for c in cl) / len(cl) <= 2.5:
            tag_style.append(name)

    return {
        "images": len(images),
        "captioned": len(non_empty),
        "uncaptioned": len(images) - len(non_empty),
        "trigger_word": trigger,
        "with_trigger": len(with_trigger),
        "missing_trigger": [n for n, c in captions if trigger and trigger.lower() not in c.lower()],
        "median_words": median,
        "thin": thin,
        "duplicates": duplicates,
        "tag_style": tag_style,
        "phrases": repeated_clauses,
    }


def _on_gpu(cls: Any, requested: Any, allowed: tuple[str, ...], default: str) -> Any:
    """
    Resolve a UI GPU choice to a Cls to spawn from.

    Returns the base class untouched when the choice is the default, so the
    ordinary request keeps hitting the container that is already warm; only a
    genuine switch pays for a variant. An unknown card falls back to the default
    rather than raising — a stale tab asking for a card that has been removed
    from the list should still get its picture.
    """
    gpu = str(requested or default)
    if gpu not in allowed or gpu == default:
        return cls
    return cls.with_options(gpu=gpu)


OUTPUT_META = "visionary.json"
MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp",
               ".mp4": "video/mp4"}
OUTPUT_FILE_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}\.(png|jpg|webp|mp4)$")


def _infotext(
    *,
    prompt: str,
    negative_prompt: str = "",
    model: str = "",
    seed: Any = None,
    report: dict[str, Any] | None = None,
) -> str:
    """
    The generation settings as an A1111 `parameters` string.

    Written into the PNG itself, not just the sidecar, because the two answer
    different questions. The sidecar keeps the gallery working; this keeps the
    settings attached to the file after it has been dragged out of the browser,
    dropped into Discord, or found in a folder a year later. It is also the
    format every existing tool already reads — PNG Info tabs, ComfyUI, the
    civitai uploader — so "how did I make this" stays answerable outside here.

    Shape is A1111's, deliberately, down to the details that look arbitrary:
    the prompt is the first line bare, `Negative prompt:` is a whole line and
    is omitted rather than left empty, and everything else is one comma-joined
    `Key: value` line. Values containing a comma are quoted, which is how the
    parsers on the other end know the comma is not a field separator.
    """
    report = report or {}

    def field(value: Any) -> str:
        text = str(value)
        return f'"{text}"' if ("," in text or ":" in text) else text

    lines = [prompt.strip()]
    if negative_prompt.strip():
        lines.append(f"Negative prompt: {negative_prompt.strip()}")

    pairs: list[str] = []

    def add(key: str, value: Any) -> None:
        if value not in (None, "", []):
            pairs.append(f"{key}: {field(value)}")

    add("Steps", report.get("steps"))
    add("Sampler", report.get("sampler"))
    add("Schedule type", report.get("scheduler"))
    add("CFG scale", report.get("cfg_scale"))
    add("Seed", seed)
    if report.get("width") and report.get("height"):
        add("Size", f"{report['width']}x{report['height']}")
    add("Model", model)
    add("Shift", report.get("shift"))

    # A LoRA stack is the part most worth recovering and the part a bare
    # filename loses, so name and strength both go in.
    loras = [
        f"{l.get('name') or Path(str(l.get('path') or '')).stem}:{l.get('unet', 1.0)}"
        for l in (report.get("loras") or [])
        if l.get("applied", True)
    ]
    add("Loras", ", ".join(loras))

    regions = report.get("regions") or []
    if regions:
        add("Regions", len(regions))
        add("Region prompts", " | ".join(str(r.get("prompt", "")) for r in regions))

    lines.append(", ".join(pairs))
    return "\n".join(lines)


def _write_output_meta(out_dir: Path, **fields: Any) -> None:
    """
    Describe a result beside the result, in the same shape loras/ already uses.

    The job Dict is live state, not a record: it is polled during a run and
    means nothing afterwards. Everything you would want a week later — the
    prompt, the seed, the model — has to live next to the file or it is gone,
    which is why the gallery reads the volume rather than replaying job ids.
    Non-fatal: an unwritable sidecar must not lose you the image it describes.
    """
    try:
        (out_dir / OUTPUT_META).write_text(json.dumps(fields, indent=2))
    except OSError as exc:
        print(f"[meta] {out_dir.name}: {exc}")


def _gallery(limit: int = 200) -> list[dict[str, Any]]:
    """
    Every output folder on the volume, newest first.

    Keyed by what is on disk, not by a job id the browser happened to keep:
    a reload, a redeploy, or a job whose record expired all leave the work
    reachable. A folder with no sidecar still lists — older results predate
    the metadata and are not less real for it.
    """
    if not OUTPUTS.is_dir():
        return []

    out: list[dict[str, Any]] = []
    for d in OUTPUTS.iterdir():
        if not d.is_dir() or d.name.startswith("."):
            continue
        files = sorted(
            (p for p in d.iterdir()
             if p.is_file() and p.suffix.lower() in MEDIA_TYPES),
            key=lambda p: p.name,
        )
        if not files:
            continue
        meta: dict[str, Any] = {}
        try:
            meta = json.loads((d / OUTPUT_META).read_text())
        except (OSError, json.JSONDecodeError):
            pass
        out.append({
            "job_id": d.name,
            "kind": "video" if files[0].suffix.lower() == ".mp4" else "image",
            "files": [p.name for p in files],
            "modified": max(p.stat().st_mtime for p in files),
            **meta,
        })

    out.sort(key=lambda r: r["modified"], reverse=True)
    return out[:limit]


def _lora_path(raw_path: Any) -> Path:
    """
    Resolve one LoRA path and confine it to loras/.

    `resolve()` before the check, so a crafted `../../` cannot read a
    checkpoint — or anything else — off the volume. Shared by the image and
    video stacks because the confinement rule is a property of the storage
    layout, not of which model is about to load the file.

    The two failures are reported separately because they have nothing to do
    with each other: a path outside loras/ is a malformed request, and a path
    inside it that is not there is a LoRA deleted since the page loaded — which
    is the one a stale tab actually hits, and which "must be a file under
    loras/" sends you to debug in entirely the wrong place.
    """
    path = Path(str(raw_path or "")).resolve()
    if LORAS.resolve() not in path.parents:
        raise ValueError(f"LoRA must be under loras/: {str(raw_path)!r}")
    # Via the parent listing, not is_file(), for the same reason the weight
    # catalogue is — see _sizes_on_disk. A LoRA trained minutes ago into a
    # container that had already asked for that name would otherwise be
    # rejected as deleted.
    if not _sizes_on_disk([path])[path]:
        raise ValueError(
            f"No such LoRA: {path.relative_to(LORAS.resolve()).as_posix()}. "
            "It may have been deleted — reload to refresh the list."
        )
    return path


def _validate_loras(raw: Any) -> list[dict[str, Any]]:
    """
    Validate the LoRA stack coming off the wire. Pure stdlib, no torch.

    Anything malformed raises rather than being silently dropped; a LoRA that
    quietly does not load looks exactly like a LoRA with no effect.

    Deliberately importable from the CPU web container, so /api/generate can
    reject a bad stack in milliseconds. Validating only inside the GPU job meant
    a malformed path still paid a ~30 s cold start and ~35 GB of weight loading
    before failing, and surfaced as a dead job rather than a form error.
    """
    if not raw:
        return []
    if isinstance(raw, (str, Path)):
        raw = [{"path": str(raw)}]

    out: list[dict[str, Any]] = []
    for entry in list(raw)[:MAX_LORAS]:
        if isinstance(entry, (str, Path)):
            entry = {"path": str(entry)}
        path = _lora_path(entry.get("path"))

        try:
            unet = float(entry.get("unet", entry.get("weight", 1.0)))
            te = entry.get("text_encoder", entry.get("te"))
            te = None if te in (None, "") else float(te)
        except (TypeError, ValueError):
            raise ValueError(f"LoRA strengths must be numbers: {path.name}")

        out.append({"path": str(path), "unet": unet, "text_encoder": te})
    return out


def _lora_specs(raw: Any) -> list:
    """GPU-side wrapper. Re-validates: this is the container that opens the file."""
    from krea2 import LoraSpec

    return [LoraSpec(e["path"], unet=e["unet"], text_encoder=e["text_encoder"])
            for e in _validate_loras(raw)]


def _regions(raw: Any) -> list:
    """Regional prompting rectangles, in normalised 0..1 canvas coordinates."""
    from krea2 import Region

    if not raw:
        return []
    out = []
    for entry in list(raw)[:8]:
        prompt = str(entry.get("prompt") or "").strip()
        if not prompt:
            continue
        out.append(Region(
            prompt,
            x=float(entry.get("x", 0.0)), y=float(entry.get("y", 0.0)),
            width=float(entry.get("width", 1.0)), height=float(entry.get("height", 1.0)),
            weight=float(entry.get("weight", 1.0)),
        ))
    return out


@app.cls(
    image=inference_image, gpu=GPU, cpu=2.0, timeout=60 * 60,
    volumes={"/workspace": volume},
    # One container: the checkpoint is ~35 GB across DiT/VAE/TE, so a second
    # replica costs a full cold load rather than sharing the warm one.
    max_containers=1,
    scaledown_window=10 * 60,
)
@modal.concurrent(max_inputs=1)  # one GPU, one sampling loop
class Generator:
    """Holds a loaded Krea 2 checkpoint across requests."""

    @modal.enter()
    def setup(self):
        import sys

        sys.path.insert(0, str(FORGE))
        self._pipelines: dict[str, Any] = {}

    def _pipeline(self, model: str):
        from krea2 import Krea2Pipeline, unload_all

        if model not in self._pipelines:
            # Only one checkpoint fits on the card, so switching between Turbo
            # and RAW evicts the outgoing one first — dropping the reference
            # alone leaves its weights resident and the new load OOMs.
            if self._pipelines:
                self._pipelines.clear()
                unload_all()
            _require_models(model, "vae", "text_encoder")
            self._pipelines[model] = Krea2Pipeline(
                dit_path=TURBO_PATH if model == "turbo" else RAW_PATH,
                vae_path=VAE_PATH,
                text_encoder_path=TE_PATH,
                key=model,
            )
        return self._pipelines[model]

    @modal.method()
    def generate(self, job_id: str, params: dict[str, Any]) -> dict[str, Any]:
        from krea2 import GenerateRequest

        started = time.time()
        jobs[job_id] = {"status": "running", "phase": "loading", "stop": False}
        volume.reload()

        model = "turbo" if str(params.get("model") or "turbo") != "raw" else "raw"

        try:
            pipe = self._pipeline(model)
            _publish(job_id, phase="generate", step=0, percent=0)

            def progress(step: int, total: int) -> None:
                _publish(job_id, phase="generate", step=step, total_steps=total,
                         percent=int(step * 100 / max(1, total)))
                # Stop between steps rather than mid-forward: the model stays
                # loaded and the container survives for the next request, which
                # a killed process would not.
                if _stop_requested(job_id):
                    raise StopRequested("generate")

            req = GenerateRequest(
                prompt=str(params.get("prompt") or ""),
                negative_prompt=str(params.get("negative_prompt") or ""),
                width=int(params.get("width") or 1024),
                height=int(params.get("height") or 1024),
                steps=params.get("steps") or None,
                cfg_scale=params.get("cfg_scale"),
                seed=params.get("seed"),
                batch_size=max(1, min(4, int(params.get("num_images") or 1))),
                sampler=str(params.get("sampler") or "Euler"),
                scheduler=str(params.get("scheduler") or "Automatic"),
                shift=float(params.get("shift") or 1.15),
                loras=_lora_specs(params.get("loras")),
                regions=_regions(params.get("regions")),
                region_weight=float(params.get("region_weight") or 1.0),
            )
            if not req.prompt.strip() and not req.regions:
                raise ValueError("A prompt is required.")

            images = pipe.generate(req, progress=progress)
        except StopRequested:
            res = {"status": "stopped", "job_id": job_id, "files": [],
                   "duration_s": round(time.time() - started, 1)}
            _publish(job_id, **res)
            return res
        except Exception as exc:
            _publish(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
            raise

        from PIL import PngImagePlugin

        out_dir = OUTPUTS / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%H%M%S")
        report = getattr(pipe, "last_report", {})
        seeds = report.get("seeds") or []
        names = []
        for i, image in enumerate(images):
            name = f"{stamp}_{i:02d}.png"
            # Per-image seed, not the batch's first: in a batch of four each
            # image has its own, and a metadata block that reports the same
            # seed for all four cannot reproduce three of them.
            info = PngImagePlugin.PngInfo()
            info.add_text("parameters", _infotext(
                prompt=str(params.get("prompt") or ""),
                negative_prompt=str(params.get("negative_prompt") or ""),
                model=model,
                seed=seeds[i] if i < len(seeds) else None,
                report=report,
            ))
            image.save(out_dir / name, pnginfo=info)
            names.append(name)
        _write_output_meta(
            out_dir, kind="image", job_id=job_id, model=model,
            prompt=str(params.get("prompt") or ""),
            negative_prompt=str(params.get("negative_prompt") or ""),
            created=time.time(), **report,
        )
        volume.commit()

        # Only filenames go into the job record. The PNGs themselves are served
        # by /api/outputs/{job_id} straight off the volume — a 1024px base64
        # image is megabytes, and this dict is polled every few seconds.
        res = {
            "status": "completed", "job_id": job_id, "files": names,
            "model": model, "output_dir": str(out_dir),
            "duration_s": round(time.time() - started, 1),
            **report,
        }
        _publish(job_id, **res)
        return res


# --------------------------------------------------------------------------
# Video — MiniMax-H3 through ComfyUI
#
# H3 denoises video and audio as one packed sequence: a single transformer
# call steps both, and the soundtrack is not a second pass. That is why the
# result of this feature is an mp4 with sound rather than a silent clip, and
# why the prompt is worth writing to include what you want to *hear*.
#
# The checkpoint is guidance-distilled — no CFG, no negative prompt, one
# forward pass per step. Any UI that offers those is offering knobs the model
# does not read.
# --------------------------------------------------------------------------

# 24 fps is the model's native rate; nothing here is configurable because
# nothing about H3 is trained at another one.
H3_FPS = 24

# The video VAE decodes in blocks of 17 frames after a leading 5, so a frame
# count is only valid on the 17n+5 grid. This mirrors the Math Expression node
# in Comfy's own template rather than inventing a second rounding rule.
H3_FRAME_STEP = 17
H3_FRAME_BASE = 5

# Trained range. 124 frames is ~5.2 s, 345 is ~14.4 s — the next grid point up
# (362) is 15.083 s, past the 15 s the model was trained to, so it is excluded
# rather than offered and quietly disappointing.
H3_MIN_FRAMES = 124
H3_MAX_FRAMES = 345

# H3's canvas is a short edge, not a resolution: the aspect ratio picks the
# long edge. 768 is what it was trained at; 544 is the draft tier, which is
# roughly 2.3x faster per step and is the single biggest speed lever there is —
# far bigger than step count, and it costs detail rather than coherence.
H3_TIERS = {"full": 768, "draft": 544}

# ref2va's limits: 9 images, 3 videos of 2-15 s, 3 standalone audio clips, and
# 12 across all types. Standalone audio is not wired up; a video's own
# soundtrack rides along with it and does not count against the audio budget
# here because it is packed as that video's, not as its own <Audio n>.
MAX_H3_REFS = 9
MAX_H3_REF_VIDEOS = 3
MAX_H3_REF_TOTAL = 12

# Reference tokens ride through every sampling step, so their size is a per-step
# cost, not a one-off encode. "match" scales each reference to the generation's
# pixel area; "max" uses the 2048px short edge the reference pipeline was built
# for and buys identity fidelity at several times the time.
H3_REF_SIZES = ("match", "max")

# Shared by every video model, because an aspect ratio is a property of the
# shot and not of the checkpoint. What differs per model is the short edge and
# the alignment, which is what _canvas() takes.
VIDEO_ASPECTS = {
    "21:9": (21, 9), "16:9": (16, 9), "4:3": (4, 3),
    "1:1": (1, 1), "3:4": (3, 4), "9:16": (9, 16),
}


def _snap_frames(seconds: float, *, fps: int, step: int, base: int,
                 lo: int, hi: int) -> int:
    """
    Seconds at a model's fps, snapped up onto the frame grid its VAE decodes.

    Every video VAE here has one: a leading `base` frames and then blocks of
    `step`. Off-grid counts are not a rounding nicety — they are a decode error
    or a silently truncated clip, so the snap is up and the range is clamped to
    what the model was trained for rather than to what it will accept.
    """
    raw = max(lo, round(float(seconds) * fps))
    snapped = raw + (base - raw % step) % step
    return min(hi, snapped)


def _canvas(aspect: str, *, short: int, align: int) -> tuple[int, int]:
    """
    (width, height) for an aspect ratio at a given short edge.

    Floors the long edge to the alignment rather than rounding: 16:9 at 768
    is 1344x768 that way, which is the canvas H3 was trained on and the one
    Comfy's resolution table lists. Rounding gives 1376 and a subtly off-ratio
    frame.
    """
    if aspect not in VIDEO_ASPECTS:
        raise ValueError(f"Unknown aspect {aspect!r}. One of: {', '.join(VIDEO_ASPECTS)}")
    num, den = VIDEO_ASPECTS[aspect]
    long = (short * max(num, den) // min(num, den)) // align * align
    return (long, short) if num >= den else (short, long)


def _fit_canvas(image_path: Path, *, short: int, align: int) -> tuple[int, int]:
    """
    The canvas a keyframe implies: its own ratio at the tier's short edge.

    A first frame anchors the geometry of the clip, so the aspect picker stops
    being the thing that decides it — the alternative is cropping or stretching
    a frame the user chose, silently. Read from the file on disk rather than
    from the request, because that is the pixels the model will actually see.
    """
    from PIL import Image

    with Image.open(image_path) as im:
        src_w, src_h = im.size
    if src_w >= src_h:
        return (src_w * short // src_h) // align * align, short
    return short, (src_h * short // src_w) // align * align


def _h3_frames(seconds: float) -> int:
    """Seconds at 24 fps, snapped up to the next 17n+5 the VAE can decode."""
    return _snap_frames(seconds, fps=H3_FPS, step=H3_FRAME_STEP, base=H3_FRAME_BASE,
                        lo=H3_MIN_FRAMES, hi=H3_MAX_FRAMES)


def _h3_canvas(aspect: str, tier: str) -> tuple[int, int]:
    """(width, height) for an aspect ratio at an H3 tier's short edge."""
    if tier not in H3_TIERS:
        raise ValueError(f"Unknown tier {tier!r}. One of: {', '.join(H3_TIERS)}")
    return _canvas(aspect, short=H3_TIERS[tier], align=32)


def _h3_graph(
    *,
    prompt: str,
    width: int,
    height: int,
    frames: int,
    seed: int,
    steps: int,
    sampler: str,
    scheduler: str,
    first_frame: str | None = None,
    last_frame: str | None = None,
    references: list[str] | None = None,
    ref_videos: list[str] | None = None,
    ref_size: str = "match",
) -> dict[str, Any]:
    """
    Build ComfyUI's API-format graph for one H3 clip.

    Wiring is taken from Comfy's own `video_minimax_h3_i2v` template, which is
    the same graph for t2v — `MiniMaxH3ImageToVideo` covers both and switches on
    whether a keyframe is connected. Written out as a dict here rather than
    shipped as workflow JSON so that a wrong node name is a Python error next to
    the code that caused it, and so nothing in this repo has to be re-exported
    from a GUI when a parameter changes.

    Note both decoders read SamplerCustomAdvanced slot 0 (`output`), not slot 1
    (`denoised_output`) — video and audio come out of the same latent.
    """
    ref_mode = bool(references or ref_videos)
    dit = MODEL_CATALOGUE["h3_ref_dit" if ref_mode else "h3_dit"]["dest"].name
    te = MODEL_CATALOGUE["h3_te"]["dest"].name
    vae = MODEL_CATALOGUE["h3_vae"]["dest"].name
    avae = MODEL_CATALOGUE["h3_audio_vae"]["dest"].name

    cond: dict[str, Any] = {
        "clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt,
        "width": width, "height": height, "length": frames,
    }
    if ref_mode:
        # The reference node also encodes audio references, so it takes the
        # audio VAE directly — the keyframe node never needs it.
        cond["audio_vae"] = ["avae", 0]
        cond["ref_image_size"] = ref_size
    graph: dict[str, Any] = {
        "dit": {"class_type": "UNETLoader",
                "inputs": {"unet_name": dit, "weight_dtype": "default"}},
        # type="minimax" selects H3's conditioner handling: the hidden state is
        # read from partway up Qwen3-VL, not from its last layer.
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": te, "type": "minimax", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": avae}},
        "cond": {"class_type": "MiniMaxH3ReferenceToVideo" if ref_mode
                 else "MiniMaxH3ImageToVideo", "inputs": cond},
        # BasicGuider, not CFGGuider: there is no negative branch to weigh.
        "guider": {"class_type": "BasicGuider",
                   "inputs": {"model": ["dit", 0], "conditioning": ["cond", 0]}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler}},
        "sigmas": {"class_type": "BasicScheduler",
                   "inputs": {"model": ["dit", 0], "scheduler": scheduler,
                              "steps": steps, "denoise": 1.0}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sample": {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["noise", 0], "guider": ["guider", 0],
                              "sampler": ["sampler", 0], "sigmas": ["sigmas", 0],
                              "latent_image": ["cond", 1]}},
        "frames": {"class_type": "VAEDecode",
                   "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "audio": {"class_type": "VAEDecodeAudio",
                  "inputs": {"samples": ["sample", 0], "vae": ["avae", 0]}},
        "video": {"class_type": "CreateVideo",
                  "inputs": {"images": ["frames", 0], "audio": ["audio", 0],
                             "fps": H3_FPS}},
        "save": {"class_type": "SaveVideo",
                 "inputs": {"video": ["video", 0], "filename_prefix": "visionary",
                            "format": "auto", "codec": "auto"}},
    }

    # A keyframe is stretched to the canvas when it is the first frame (it
    # anchors the geometry) and cover-cropped when it is the last — that is
    # MiniMaxH3ImageToVideo's own behaviour, not something to reimplement here.
    if first_frame:
        graph["first"] = {"class_type": "LoadImage", "inputs": {"image": first_frame}}
        cond["first_frame"] = ["first", 0]
    if last_frame:
        graph["last"] = {"class_type": "LoadImage", "inputs": {"image": last_frame}}
        cond["last_frame"] = ["last", 0]

    # Reference inputs are namespaced by their autogrow group and indexed from
    # ZERO, while the prompt tag for the same reference counts from one. So
    # `ref_images.ref_image_0` is the thing the prompt calls <Picture 1>.
    #
    # That off-by-one is not ours to correct: both halves are the model's, and
    # the node's own docstring documents the 1-based side ("ordinals are 1-based
    # per type") while the schema carries the 0-based side. Verified against
    # node 136 of Comfy's video_minimax_h3_r2v template, which is the only place
    # the two appear side by side. Get it wrong and ComfyUI accepts the graph
    # and ignores the reference.
    for i, name in enumerate(references or []):
        graph[f"refimg{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        cond[f"ref_images.ref_image_{i}"] = [f"refimg{i}", 0]

    # A video reference conditions on motion, and on its soundtrack when it has
    # one. The node takes frames rather than a VIDEO, so the file is decomposed
    # first — and the audio output goes to the *same-numbered* audio slot, which
    # is how the model knows that soundtrack belongs to that clip rather than
    # being a standalone <Audio n>.
    for i, name in enumerate(ref_videos or []):
        graph[f"refvid{i}"] = {"class_type": "LoadVideo", "inputs": {"file": name}}
        graph[f"refvidparts{i}"] = {"class_type": "GetVideoComponents",
                                    "inputs": {"video": [f"refvid{i}", 0]}}
        cond[f"ref_videos.ref_video_{i}"] = [f"refvidparts{i}", 0]
        cond[f"ref_video_audios.ref_video_audio_{i}"] = [f"refvidparts{i}", 1]
    return graph


# --------------------------------------------------------------------------
# Video — Wan 2.2 through the same ComfyUI
#
# Same container, same warm process, same job/status/stop contract. Wan is a
# second family in the existing video path rather than a second video path,
# which is the whole reason the backend was a driven ComfyUI rather than ported
# model code: adding a family is a graph and a table, not an image.
#
# What Wan is *for*, next to H3: it takes LoRAs. H3's repackage is int8-convrot
# quantized and has no LoRA ecosystem to speak of; Wan 2.2 has both, and phase 4
# trains against it. It also reads CFG and a negative prompt, which H3 —
# guidance-distilled — does not. Silent video is the cost.
# --------------------------------------------------------------------------

# Two families under one name. `14b` is the A14B mixture of experts, two 14 GB
# transformers split by noise level; `5b` is the single dense TI2V checkpoint
# with its own higher-compression VAE. They differ in fps, latent stride and
# frame budget, so nothing below is a shared constant by accident.
WAN_FPS = {"14b": 16, "5b": 24}

# Latent stride. The 14B pair uses the 2.1 VAE at 8x spatial, so 16-aligned
# pixels; the 5B's VAE is 16x, so 32-aligned. Feed either an unaligned canvas
# and the encode silently crops.
WAN_ALIGN = {"14b": 16, "5b": 32}

# Both families decode 4n+1 frames — a leading frame then blocks of four.
WAN_FRAME_STEP = 4
WAN_FRAME_BASE = 1
WAN_MIN_FRAMES = 17

# Trained length, not maximum accepted length. Wan 2.2 is a five-second model
# in both families: 81 at 16 fps and 121 at 24 fps are both 5.04 s. Longer runs
# and then loops or drifts, so it is excluded rather than offered.
WAN_MAX_FRAMES = {"14b": 81, "5b": 121}

# Short edge per tier, the same shape as H3_TIERS. 720/480 are Wan's own two
# resolution tiers; the 5B is trained at 704 and reuses 480 as its draft.
WAN_TIERS = {
    "14b": {"full": 720, "draft": 480},
    "5b": {"full": 704, "draft": 480},
}

# Wan reads CFG, so these are real defaults rather than placeholders. 3.5 for
# the 14B pair and 5.0 for the 5B are what Comfy's own templates ship, and the
# shift is the 8.0 both model configs already carry — ModelSamplingSD3 is in
# the graph so that a speed LoRA (which wants ~5.0) has somewhere to say so.
WAN_DEFAULT_CFG = {"14b": 3.5, "5b": 5.0}
WAN_DEFAULT_SHIFT = 8.0
WAN_DEFAULT_STEPS = {"14b": 20, "5b": 30}

# Which expert a LoRA row patches. The A14B pair is two checkpoints, so a LoRA
# has to say — and it matters: the LightX2V pair ships one file per expert, and
# crossing them is a silent quality loss rather than an error. "both" is the
# default because a LoRA trained on the whole model is the common case, and it
# is what the 5B always does since it has no second expert to choose.
WAN_EXPERTS = ("both", "high", "low")


# What the composer is allowed to ask for, per video model.
#
# Served to the page rather than written into it, for the reason the GPU list
# already is: which controls a model reads is a property of the model, and a
# copy in the HTML is a second source of truth that drifts the first time one
# of them changes. It is also the honest way to present two families that
# genuinely differ — H3 is guidance-distilled and takes no CFG and no negative
# prompt, Wan takes both; Wan takes LoRAs, H3's int8 repackage has no LoRA
# ecosystem to offer. A control that is present but ignored is worse than one
# that is absent, so absent is what the model says it wants.
VIDEO_MODELS: dict[str, dict[str, Any]] = {
    "h3": {
        "label": "MiniMax-H3",
        "note": "Sound and picture in one pass",
        "requires": {"fl2va": VIDEO_MODEL_KEYS, "ref2va": VIDEO_REF_MODEL_KEYS},
        "tiers": {"full": "768p", "draft": "544p draft"},
        "lengths": [5, 6, 8, 10, 12, 14],
        "samplers": ["res_multistep", "euler", "dpmpp_2m"],
        "schedulers": ["simple", "normal", "beta"],
        "defaults": {"steps": 20, "sampler": "res_multistep", "scheduler": "simple",
                     "tier": "full", "seconds": 5},
        "supports": {"loras": False, "experts": False, "cfg": False,
                     "negative": False, "references": True, "last_frame": True,
                     "audio": True},
    },
    "wan14b": {
        "label": "Wan 2.2 A14B",
        "note": "Two experts · silent",
        "requires": {"t2v": WAN_MODEL_KEYS[("14b", "t2v")],
                     "i2v": WAN_MODEL_KEYS[("14b", "i2v")]},
        "tiers": {"full": "720p", "draft": "480p draft"},
        "lengths": [2, 3, 4, 5],
        "samplers": ["euler", "uni_pc", "dpmpp_2m", "res_multistep"],
        "schedulers": ["simple", "normal", "beta"],
        "defaults": {"steps": WAN_DEFAULT_STEPS["14b"], "cfg": WAN_DEFAULT_CFG["14b"],
                     "shift": WAN_DEFAULT_SHIFT, "sampler": "euler",
                     "scheduler": "simple", "tier": "full", "seconds": 5},
        "supports": {"loras": True, "experts": True, "cfg": True,
                     "negative": True, "references": False, "last_frame": True,
                     "audio": False},
    },
    "wan5b": {
        "label": "Wan 2.2 TI2V 5B",
        "note": "24 fps · silent",
        "requires": {"t2v": WAN_MODEL_KEYS[("5b", "t2v")],
                     "i2v": WAN_MODEL_KEYS[("5b", "i2v")]},
        "tiers": {"full": "704p", "draft": "480p draft"},
        "lengths": [2, 3, 4, 5],
        "samplers": ["euler", "uni_pc", "dpmpp_2m", "res_multistep"],
        "schedulers": ["simple", "normal", "beta"],
        "defaults": {"steps": WAN_DEFAULT_STEPS["5b"], "cfg": WAN_DEFAULT_CFG["5b"],
                     "shift": WAN_DEFAULT_SHIFT, "sampler": "euler",
                     "scheduler": "simple", "tier": "full", "seconds": 5},
        "supports": {"loras": True, "experts": False, "cfg": True,
                     "negative": True, "references": False, "last_frame": False,
                     "audio": False},
    },
}


def _video_model_status() -> list[dict[str, Any]]:
    """
    Every video model with what is actually on the volume for each of its tasks.

    Per task, not per model, because a 14B t2v run and a 14B i2v run load
    different 28.6 GB pairs — reporting one "Wan 2.2: missing" would send you to
    download 57 GB when the run you are composing needs half of it.
    """
    sizes = _sizes_on_disk(
        MODEL_CATALOGUE[k]["dest"]
        for spec in VIDEO_MODELS.values()
        for keys in spec["requires"].values()
        for k in keys
    )
    out = []
    for key, spec in VIDEO_MODELS.items():
        tasks = {}
        for task, keys in spec["requires"].items():
            missing = [MODEL_CATALOGUE[k]["label"] for k in keys
                       if not sizes[MODEL_CATALOGUE[k]["dest"]]]
            tasks[task] = {"ready": not missing, "missing": missing}
        out.append({
            "key": key, "label": spec["label"], "note": spec["note"],
            "tiers": spec["tiers"], "lengths": spec["lengths"],
            "samplers": spec["samplers"], "schedulers": spec["schedulers"],
            "defaults": spec["defaults"], "supports": spec["supports"],
            "tasks": tasks,
            "ready": any(t["ready"] for t in tasks.values()),
        })
    return out


def _wan_frames(seconds: float, family: str) -> int:
    """Seconds at the family's fps, snapped up onto its 4n+1 grid."""
    return _snap_frames(
        seconds, fps=WAN_FPS[family], step=WAN_FRAME_STEP, base=WAN_FRAME_BASE,
        lo=WAN_MIN_FRAMES, hi=WAN_MAX_FRAMES[family],
    )


def _wan_canvas(aspect: str, tier: str, family: str) -> tuple[int, int]:
    """(width, height) for an aspect ratio at a Wan tier's short edge."""
    if family not in WAN_TIERS:
        raise ValueError(f"Unknown Wan family {family!r}. One of: {', '.join(WAN_TIERS)}")
    if tier not in WAN_TIERS[family]:
        raise ValueError(
            f"Unknown tier {tier!r}. One of: {', '.join(WAN_TIERS[family])}"
        )
    return _canvas(aspect, short=WAN_TIERS[family][tier], align=WAN_ALIGN[family])


def _wan_task(first_frame: Any, last_frame: Any) -> str:
    """
    Which Wan task a request is, read off what was attached rather than asked.

    The same choice H3 makes, and for the same reason: text-to-video and
    image-to-video are not two things you pick between, they are what you get
    depending on whether you gave the clip a frame to start on. The difference
    is that on the 14B they are genuinely different weights, so this also
    decides which 28.6 GB pair loads.
    """
    return "i2v" if (first_frame or last_frame) else "t2v"


def _validate_video_loras(raw: Any) -> list[dict[str, Any]]:
    """
    Validate the video LoRA stack, and resolve each path to the name ComfyUI
    addresses it by.

    ComfyUI's LoraLoaderModelOnly takes a *combo* — a filename relative to the
    loras search path, validated against a directory listing — not a path. So a
    LoRA that exists on the volume but resolves outside loras/ fails inside the
    graph as "Value not in list", which reads like a corrupt request rather
    than a wrong path. Converting here means the failure is a form error naming
    the file, on CPU, before an H100 is warm.

    Same confinement as the image stack: `resolve()` before the check, so a
    crafted `../../` cannot name a checkpoint. No text-encoder weight — umT5 is
    loaded through CLIPLoader and LoraLoaderModelOnly patches the DiT only.
    """
    if not raw:
        return []

    out: list[dict[str, Any]] = []
    for entry in list(raw)[:MAX_LORAS]:
        if isinstance(entry, (str, Path)):
            entry = {"path": str(entry)}
        path = _lora_path(entry.get("path"))

        try:
            strength = float(entry.get("unet", entry.get("weight", 1.0)))
        except (TypeError, ValueError):
            raise ValueError(f"LoRA strength must be a number: {path.name}")

        expert = str(entry.get("expert") or "both")
        if expert not in WAN_EXPERTS:
            raise ValueError(
                f"{path.name}: expert must be one of {', '.join(WAN_EXPERTS)}"
            )

        out.append({
            "path": str(path),
            # Forward slashes, relative to loras/ — os.walk on Linux produces
            # exactly this, and it is what the combo is validated against.
            "name": path.relative_to(LORAS.resolve()).as_posix(),
            "unet": strength,
            "expert": expert,
        })
    return out


def _wan_graph(
    *,
    family: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    frames: int,
    seed: int,
    steps: int,
    cfg: float,
    shift: float,
    switch_at: int,
    sampler: str,
    scheduler: str,
    loras: list[dict[str, Any]] | None = None,
    first_frame: str | None = None,
    last_frame: str | None = None,
) -> dict[str, Any]:
    """
    Build ComfyUI's API-format graph for one Wan 2.2 clip.

    Wiring follows Comfy's `video_wan2_2_14B_{t2v,i2v}` and `video_wan2_2_5B_ti2v`
    templates. Written as a dict here for the same reason `_h3_graph` is: a
    wrong node name is a Python error next to the code that caused it, not a
    workflow JSON that has to be re-exported from a GUI when a parameter moves.

    The 14B path is two KSamplerAdvanced calls, not one sampler with a switch.
    That is the only way to express the mixture: the high-noise expert runs
    steps 0..switch_at and hands the *unfinished* latent over
    (`return_with_leftover_noise`), and the low-noise expert picks it up without
    re-noising it (`add_noise: disable`). Get either flag wrong and the graph
    still runs — it just produces a clip that is subtly washed out or doubly
    noised, with nothing in the log to say so.
    """
    task = _wan_task(first_frame, last_frame)
    keys = WAN_MODEL_KEYS[(family, task)]
    te = MODEL_CATALOGUE["wan_te"]["dest"].name
    vae = MODEL_CATALOGUE["wan_vae_22" if family == "5b" else "wan_vae"]["dest"].name
    fps = WAN_FPS[family]

    graph: dict[str, Any] = {
        # type="wan" selects umT5's tokenizer and Wan's conditioning layout.
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": te, "type": "wan", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "pos": {"class_type": "CLIPTextEncode",
                "inputs": {"clip": ["clip", 0], "text": prompt}},
        # Unlike H3 there is a negative branch, because Wan is not
        # guidance-distilled — an empty one is still a real, weighed branch.
        "neg": {"class_type": "CLIPTextEncode",
                "inputs": {"clip": ["clip", 0], "text": negative_prompt}},
    }

    def expert(tag: str, key: str) -> str:
        """
        Loader → LoRA chain → shift. Returns the node the sampler reads.

        `tag` is "main" for the single-expert 5B, and that is not cosmetic: it
        means no row can be filtered out by an expert that does not exist here.
        A LoRA the user added and the graph quietly skipped is indistinguishable
        from a LoRA with no effect, so on the 5B every row applies and the API
        is what refuses a per-expert row it cannot honour.
        """
        loader = f"dit_{tag}"
        graph[loader] = {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": MODEL_CATALOGUE[key]["dest"].name,
                       "weight_dtype": "default"},
        }
        src = loader
        # Stacked in the order given: each loader patches the model the
        # previous one returned, so the order shown in the UI is the order
        # they apply, the same as the image side.
        for i, lora in enumerate(loras or []):
            if tag != "main" and lora["expert"] not in ("both", tag):
                continue
            node = f"lora_{tag}_{i}"
            graph[node] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {"model": [src, 0], "lora_name": lora["name"],
                           "strength_model": lora["unet"]},
            }
            src = node
        # After the LoRAs, not before: a speed LoRA changes the schedule it
        # wants, and the shift has to be the last word on the sampling curve.
        graph[f"shift_{tag}"] = {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": [src, 0], "shift": shift},
        }
        return f"shift_{tag}"

    if family == "5b":
        model = expert("main", keys[0])
        graph["latent"] = {
            "class_type": "Wan22ImageToVideoLatent",
            "inputs": {"vae": ["vae", 0], "width": width, "height": height,
                       "length": frames, "batch_size": 1},
        }
        if first_frame:
            graph["first"] = {"class_type": "LoadImage",
                              "inputs": {"image": first_frame}}
            graph["latent"]["inputs"]["start_image"] = ["first", 0]
        pos, neg, latent = ["pos", 0], ["neg", 0], ["latent", 0]
        stages = [(model, 0, steps, "enable", "disable")]
    else:
        high, low = expert("high", keys[0]), expert("low", keys[1])
        if first_frame and last_frame:
            # Both ends given is its own node, not the i2v node with an extra
            # input: the mask it builds pins the tail frames as well as the
            # head, and WanImageToVideo has no way to express that.
            cond: dict[str, Any] = {"class_type": "WanFirstLastFrameToVideo"}
        elif first_frame or last_frame:
            cond = {"class_type": "WanImageToVideo"}
        else:
            cond = {}

        if cond:
            graph["cond"] = cond
            cond["inputs"] = {
                "positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
                "width": width, "height": height, "length": frames, "batch_size": 1,
            }
            if first_frame:
                graph["first"] = {"class_type": "LoadImage",
                                  "inputs": {"image": first_frame}}
                cond["inputs"]["start_image"] = ["first", 0]
            if last_frame:
                graph["last"] = {"class_type": "LoadImage",
                                 "inputs": {"image": last_frame}}
                # A last frame alone still needs the first-last node; the i2v
                # node has no end_image input at all.
                if not first_frame:
                    cond["class_type"] = "WanFirstLastFrameToVideo"
                cond["inputs"]["end_image"] = ["last", 0]
            # The node rewrites both conditioning branches with the encoded
            # keyframe, so the sampler must read its outputs and not the raw
            # text encodes — wiring `pos` straight through is a graph that runs
            # and ignores the image.
            pos, neg, latent = ["cond", 0], ["cond", 1], ["cond", 2]
        else:
            graph["latent"] = {
                "class_type": "EmptyHunyuanLatentVideo",
                "inputs": {"width": width, "height": height,
                           "length": frames, "batch_size": 1},
            }
            pos, neg, latent = ["pos", 0], ["neg", 0], ["latent", 0]

        stages = [
            (high, 0, switch_at, "enable", "enable"),
            (low, switch_at, steps, "disable", "disable"),
        ]

    prev = latent
    for i, (model, start, end, add_noise, leftover) in enumerate(stages):
        node = f"sample{i}"
        graph[node] = {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": [model, 0], "add_noise": add_noise, "noise_seed": seed,
                "steps": steps, "cfg": cfg, "sampler_name": sampler,
                "scheduler": scheduler, "positive": pos, "negative": neg,
                "latent_image": prev, "start_at_step": start, "end_at_step": end,
                "return_with_leftover_noise": leftover,
            },
        }
        prev = [node, 0]

    graph["frames"] = {"class_type": "VAEDecode",
                       "inputs": {"samples": prev, "vae": ["vae", 0]}}
    # No audio input: Wan is silent. CreateVideo takes it optionally, so the
    # difference between the two families is one absent key rather than a
    # second save path.
    graph["video"] = {"class_type": "CreateVideo",
                      "inputs": {"images": ["frames", 0], "fps": fps}}
    graph["save"] = {"class_type": "SaveVideo",
                     "inputs": {"video": ["video", 0], "filename_prefix": "visionary",
                                "format": "auto", "codec": "auto"}}
    return graph


@app.cls(
    image=video_image, gpu=VIDEO_GPU, cpu=4.0, timeout=60 * 60,
    volumes={"/workspace": volume},
    max_containers=1,
    # Longer than the image side's 10 min: 42.5 GB is a slow thing to reload,
    # and video is worked in takes — you watch one clip, then adjust and go again.
    scaledown_window=15 * 60,
)
@modal.concurrent(max_inputs=1)
class VideoGenerator:
    """
    Holds a warm ComfyUI process across requests.

    ComfyUI runs as a local server inside this container and is spoken to over
    127.0.0.1 — it is never exposed, and the only client is the code below.
    Running the real thing rather than porting its model code is what keeps the
    int8-convrot kernels, the dynamic offloader and every upstream H3 fix on
    our side of the line instead of in a fork we would own.
    """

    @modal.enter()
    def setup(self):
        import threading

        # Point ComfyUI at our flat models/ directory instead of moving weights
        # into the per-type tree it expects. The volume layout is the contract;
        # ComfyUI adapts to it. Every type maps to the same folder, so all four
        # files are visible to whichever loader asks for them.
        #
        # loras/ is the exception and keeps its nesting, because that nesting is
        # meaningful — one folder per trained LoRA, checkpoints beside the final
        # weights. ComfyUI walks it recursively and names a file by its path
        # relative to here, which is exactly what _validate_video_loras() emits.
        (COMFY / "extra_model_paths.yaml").write_text(
            "visionary:\n"
            f"  base_path: {WORKSPACE}/\n"
            "  diffusion_models: models/\n"
            "  text_encoders: models/\n"
            "  clip: models/\n"
            "  vae: models/\n"
            "  loras: loras/\n"
        )

        # A fresh clone ships input/ and output/, but a changed --base-directory
        # or a pruned image would not, and the failure would surface as a
        # LoadImage error about a file we thought we had just written.
        (COMFY / "input").mkdir(parents=True, exist_ok=True)
        (COMFY / "output").mkdir(parents=True, exist_ok=True)

        self._job_id: str | None = None
        self._log: deque[str] = deque(maxlen=200)
        self._proc = subprocess.Popen(
            ["python", "main.py", "--listen", "127.0.0.1", "--port", str(COMFY_PORT),
             "--disable-auto-launch", "--disable-metadata",
             # Safe here in a way it would not be on the image side: the Krea 2
             # path masks attention for regional prompting, and sageattention
             # asserts `mask is None` and falls back per call. H3 never passes a
             # mask, so this image gets the fast kernel and the inference image
             # deliberately still does not. See forge/krea2/regional.py.
             "--use-sage-attention"],
            cwd=str(COMFY),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        threading.Thread(target=self._drain, daemon=True).start()
        self._wait_ready()

    def _drain(self) -> None:
        """
        Mirror ComfyUI's output to Modal's logs, and lift step progress out of it.

        ComfyUI prints a tqdm bar for the sampling loop, which TQDM_RE already
        parses for musubi — the same shape, so the same regex. Reading progress
        off stdout rather than opening ComfyUI's websocket keeps this to one
        connection and one dependency-free thread.
        """
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            print(line, flush=True)
            self._log.append(line)
            m = TQDM_RE.search(line)
            if m and self._job_id:
                fields: dict[str, Any] = {
                    "phase": "generate",
                    "step": int(m.group("step")),
                    "total_steps": int(m.group("total")),
                    "percent": int(m.group("pct")),
                }
                if m.group("eta"):
                    fields["eta"] = m.group("eta")
                    fields["rate"] = f"{m.group('rate')}{m.group('unit')}"
                _publish(self._job_id, **fields)

    def _wait_ready(self, timeout: float = 300.0) -> None:
        import urllib.error
        import urllib.request

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    "ComfyUI exited during startup.\n" + "\n".join(self._log)
                )
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{COMFY_PORT}/system_stats", timeout=2
                ).read()
                print("[video] ComfyUI ready", flush=True)
                return
            except (urllib.error.URLError, OSError):
                time.sleep(1.0)
        raise RuntimeError("ComfyUI did not become ready.\n" + "\n".join(self._log))

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        import urllib.request

        req = urllib.request.Request(
            f"http://127.0.0.1:{COMFY_PORT}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        return json.loads(urllib.request.urlopen(req, timeout=30).read() or b"{}")

    def _get(self, path: str) -> Any:
        import urllib.request

        with urllib.request.urlopen(
            f"http://127.0.0.1:{COMFY_PORT}{path}", timeout=30
        ) as r:
            return json.loads(r.read() or b"{}")

    @modal.method()
    def generate(self, job_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """
        Run one clip, whichever family it is.

        Everything outside the graph is the same job for H3 and for Wan — the
        record, the staging directory, the stop check, the file that lands on
        the volume and the sidecar beside it. Only the graph and the numbers
        that describe it are per-family, so only those are dispatched. That is
        the same contract the image side uses, extended rather than duplicated.
        """
        started = time.time()
        jobs[job_id] = {"status": "running", "phase": "loading", "stop": False}
        volume.reload()

        model = str(params.get("model") or "h3")

        try:
            # Inside the try, not above it: a missing weight raised out here
            # would leave the record saying "running" forever, and the UI
            # polling a job that is never going to answer.
            #
            # Every input image lands in ComfyUI's input directory under the job
            # id, so a LoadImage node can name it and two takes cannot collide
            # on a filename.
            def stage(blob: str, slot: str, ext: str = "png") -> str:
                name = f"{job_id}-{slot}.{ext}"
                (COMFY / "input" / name).write_bytes(base64.b64decode(blob))
                return name

            plan = (self._plan_h3 if model == "h3" else self._plan_wan)(params, stage)
            graph, info = plan["graph"], plan["info"]

            self._job_id = job_id
            _publish(job_id, phase="generate", step=0,
                     total_steps=info["steps"], percent=0, **info)

            prompt_id = self._post("/prompt", {"prompt": graph})["prompt_id"]
            out_name = self._await(job_id, prompt_id)
        except StopRequested:
            res = {"status": "stopped", "job_id": job_id, "files": [],
                   "duration_s": round(time.time() - started, 1)}
            _publish(job_id, **res)
            return res
        except Exception as exc:
            _publish(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self._job_id = None

        out_dir = OUTPUTS / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        name = f"{time.strftime('%H%M%S')}.mp4"
        shutil.copyfile(COMFY / "output" / out_name, out_dir / name)
        _write_output_meta(
            out_dir, kind="video", job_id=job_id, model=model,
            prompt=params["prompt"], created=time.time(), **info, **plan["meta"],
        )
        volume.commit()

        res = {
            "status": "completed", "job_id": job_id, "files": [name],
            "output_dir": str(out_dir), "model": model,
            "duration_s": round(time.time() - started, 1), **info,
        }
        _publish(job_id, **res)
        return res

    @staticmethod
    def _plan_h3(params: dict[str, Any], stage: Any) -> dict[str, Any]:
        """Graph and shot description for a MiniMax-H3 take."""
        refs_b64 = list(params.get("references") or [])[:MAX_H3_REFS]
        vids_b64 = list(params.get("ref_videos") or [])[:MAX_H3_REF_VIDEOS]
        _require_models(*(VIDEO_REF_MODEL_KEYS if (refs_b64 or vids_b64)
                          else VIDEO_MODEL_KEYS))

        width, height = _h3_canvas(params["aspect"], params["tier"])
        frames = _h3_frames(params["seconds"])
        seed = params.get("seed")
        seed = int(seed) if seed is not None else int.from_bytes(os.urandom(4), "big")
        steps = int(params["steps"])

        references = [stage(b, f"refimg{i}") for i, b in enumerate(refs_b64)]
        # LoadVideo lists the input directory and filters it by content
        # type, so the extension is load-bearing, not cosmetic.
        ref_videos = [stage(b, f"refvid{i}", "mp4") for i, b in enumerate(vids_b64)]
        keyframes: dict[str, str] = {}
        if not (references or ref_videos):
            for slot in ("first_frame", "last_frame"):
                if params.get(slot):
                    keyframes[slot] = stage(params[slot], slot)

        # A first keyframe anchors the geometry, so the canvas follows the
        # image rather than the aspect picker — cropping a source frame the
        # user chose is not ours to do silently. References are the opposite
        # case: they are encoded at their own resolution and bind nothing,
        # so the canvas stays whatever was asked for.
        if "first_frame" in keyframes:
            width, height = _fit_canvas(
                COMFY / "input" / keyframes["first_frame"],
                short=H3_TIERS[params["tier"]], align=32,
            )

        graph = _h3_graph(
            prompt=params["prompt"], width=width, height=height, frames=frames,
            seed=seed, steps=steps,
            sampler=params["sampler"], scheduler=params["scheduler"],
            references=references, ref_videos=ref_videos,
            ref_size=params.get("ref_size") or "match",
            **keyframes,
        )
        return {
            "graph": graph,
            "info": {"width": width, "height": height, "frames": frames,
                     "seconds": round(frames / H3_FPS, 2), "fps": H3_FPS,
                     "seed": seed, "steps": steps},
            "meta": {"mode": "ref2va" if (references or ref_videos) else "fl2va",
                     "sampler": params["sampler"], "scheduler": params["scheduler"],
                     "references": len(references), "ref_videos": len(ref_videos)},
        }

    @staticmethod
    def _plan_wan(params: dict[str, Any], stage: Any) -> dict[str, Any]:
        """Graph and shot description for a Wan 2.2 take."""
        family = "5b" if str(params.get("model")) == "wan5b" else "14b"

        keyframes: dict[str, str] = {}
        for slot in ("first_frame", "last_frame"):
            # The 5B latent node takes a start image only. Dropping a last
            # frame silently would be a control that looks live and is ignored,
            # so the UI hides it for this family and this is the backstop.
            if params.get(slot) and not (family == "5b" and slot == "last_frame"):
                keyframes[slot] = stage(params[slot], slot)

        task = _wan_task(keyframes.get("first_frame"), keyframes.get("last_frame"))
        _require_models(*WAN_MODEL_KEYS[(family, task)])

        width, height = _wan_canvas(params["aspect"], params["tier"], family)
        frames = _wan_frames(params["seconds"], family)
        seed = params.get("seed")
        seed = int(seed) if seed is not None else int.from_bytes(os.urandom(4), "big")
        steps = int(params["steps"])
        # Half the run each is Comfy's own split and the only defensible default
        # without measuring the sigma boundary per checkpoint. Clamped into the
        # interior: a switch at 0 or at `steps` is a one-expert run wearing the
        # cost of loading two.
        #
        # `is None`, not a falsy test: an explicit 0 is a real request (give the
        # low-noise expert everything), and `or` would silently answer it with
        # the default instead.
        asked = params.get("switch_at")
        switch_at = steps // 2 if asked is None else max(1, min(steps - 1, int(asked)))

        if "first_frame" in keyframes:
            width, height = _fit_canvas(
                COMFY / "input" / keyframes["first_frame"],
                short=WAN_TIERS[family][params["tier"]], align=WAN_ALIGN[family],
            )

        loras = _validate_video_loras(params.get("loras"))
        graph = _wan_graph(
            family=family, prompt=params["prompt"],
            negative_prompt=str(params.get("negative_prompt") or ""),
            width=width, height=height, frames=frames, seed=seed, steps=steps,
            cfg=float(params["cfg"]), shift=float(params["shift"]),
            switch_at=switch_at, sampler=params["sampler"],
            scheduler=params["scheduler"], loras=loras, **keyframes,
        )
        return {
            "graph": graph,
            "info": {"width": width, "height": height, "frames": frames,
                     "seconds": round(frames / WAN_FPS[family], 2),
                     "fps": WAN_FPS[family], "seed": seed, "steps": steps},
            "meta": {"mode": task, "family": family,
                     "cfg_scale": float(params["cfg"]),
                     "shift": float(params["shift"]),
                     # Only meaningful for the 14B pair, but recorded either way:
                     # a sidecar that omits it for the 5B is a sidecar you have
                     # to know the family to read.
                     "switch_at": switch_at if family == "14b" else None,
                     "sampler": params["sampler"], "scheduler": params["scheduler"],
                     "negative_prompt": str(params.get("negative_prompt") or ""),
                     "loras": [{"name": l["name"], "unet": l["unet"],
                                "expert": l["expert"]} for l in loras]},
        }

    def _await(self, job_id: str, prompt_id: str) -> str:
        """Poll until the graph finishes; return the saved file's path in output/."""
        while True:
            if _stop_requested(job_id):
                # ComfyUI unwinds the sampler itself and stays warm, which a
                # killed process would not — the 42.5 GB stays loaded for the
                # next take.
                self._post("/interrupt", {})
                raise StopRequested("generate")

            entry = (self._get(f"/history/{prompt_id}") or {}).get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error" or not status.get("completed", True):
                    raise RuntimeError(self._why_failed(status))
                for out in entry.get("outputs", {}).values():
                    # "images" is the one that actually fires: SaveVideo returns
                    # ui.PreviewVideo, whose as_dict() is {"images": [...],
                    # "animated": (True,)} — a video is reported through the
                    # image channel, flagged rather than separately named. The
                    # other two are what the older save nodes emit and cost a
                    # tuple to keep, and getting this wrong is expensive in a
                    # specific way: the clip renders, saves, and is then thrown
                    # away by a job that says it saved nothing.
                    for key in ("images", "videos", "gifs"):
                        for item in out.get(key) or []:
                            name = item.get("filename")
                            if not name:
                                continue
                            # Honour the subfolder rather than assuming the flat
                            # case our filename_prefix happens to produce: the
                            # prefix is split on a path separator, so the day one
                            # gains a slash this stops silently copying the
                            # wrong path.
                            return str(Path(item.get("subfolder") or "") / name)
                raise RuntimeError(
                    "ComfyUI reported success but saved no video.\n"
                    + "\n".join(list(self._log)[-25:])
                )

            if self._proc.poll() is not None:
                raise RuntimeError(
                    "ComfyUI exited mid-generation.\n" + "\n".join(list(self._log)[-25:])
                )
            time.sleep(1.5)

    @staticmethod
    def _why_failed(status: dict[str, Any]) -> str:
        """
        Turn ComfyUI's execution_error message into one line worth reading.

        The raw record nests the useful part — node type and exception — under
        a list of (event, payload) pairs, and a bare "status_str: error" sends
        you to the container logs for something the record already knows.
        """
        for event, payload in status.get("messages") or []:
            if event == "execution_error":
                node = payload.get("node_type") or payload.get("node_id")
                return f"{node}: {payload.get('exception_message') or 'failed'}"
        return "ComfyUI reported an error with no detail."


# --------------------------------------------------------------------------
# Web app — UI + API on a single URL
#
# The routes below are `def`, not `async def`, and that is deliberate: FastAPI
# runs a sync handler in a threadpool, so everything blocking in it stays off
# the event loop. Nearly every route here blocks twice over — a Modal Dict or
# volume call, and real filesystem work (a directory walk, a PIL thumbnail, a
# file read off the volume). `async def` served by 20 concurrent inputs meant
# one slow thumbnail stalled every other request in the container, and Modal
# said so on each one: "A blocking Modal interface is being used in an async
# context", once per poll of /api/status, which the UI hits every two seconds
# for the length of a training run.
#
# Awaiting the `.aio()` variants would have silenced that warning without
# fixing it — the Dict call is the part Modal can see, not the part that costs
# the most. `/api/upload` is the one exception and stays async, because it
# awaits the multipart stream itself; its one Modal call is `.aio()`d in place.
# --------------------------------------------------------------------------


@app.function(
    image=web_image, cpu=1.0, timeout=900, volumes={"/workspace": volume},
    max_containers=1,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

    api = FastAPI()

    @api.get("/", response_class=HTMLResponse)
    def index() -> str:
        return UI_HTML

    @api.get("/api/where")
    def where() -> dict[str, Any]:
        """
        What this deployment can actually see on the volume.

        Cheap CPU check for when the Models tab and a GPU job disagree — the
        usual cause is the app resolving a different volume than the one
        holding the weights, so the resolved name is part of the answer.
        """
        volume.reload()
        tree: dict[str, Any] = {}
        for d in (MODELS, LORAS, DATASETS, OUTPUTS):
            if d.is_dir():
                tree[str(d)] = sorted(
                    f"{p.name} ({p.stat().st_size / 1e9:.2f} GB)" if p.is_file() else f"{p.name}/"
                    for p in d.iterdir() if not p.name.startswith(".")
                )[:25]
            else:
                tree[str(d)] = "(directory does not exist)"
        return {"volume": VOLUME_NAME, "mounted_at": str(WORKSPACE), "contents": tree}

    @api.get("/api/state")
    def state() -> dict[str, Any]:
        volume.reload()
        # Two shapes, because the volume really holds two.
        #
        # Training writes loras/{name}/, so a folder is a LoRA and its epoch
        # checkpoints are that LoRA's versions. But anything that arrives some
        # other way — migrated off an older volume, uploaded by hand, pulled
        # down as a speed LoRA — is a bare file at the top level, and the
        # folders-only walk skipped it silently. Four real LoRAs sat on the
        # volume being invisible to the picker, which reads as "training never
        # produced anything" rather than "this listing has an opinion about
        # directory layout". A file that a run can load is a file the picker
        # has to offer.
        loras = []
        if LORAS.is_dir():
            for d in sorted(LORAS.iterdir()):
                if d.is_dir():
                    final = d / f"{d.name}.safetensors"
                    ckpts = sorted(
                        (p for p in d.glob("*.safetensors") if p != final),
                        key=lambda p: p.stat().st_mtime, reverse=True,
                    )
                    files = ([final] if final.exists() else []) + ckpts
                    if not files:
                        continue
                    trigger = ""
                    meta = d / "visionary.json"
                    if meta.exists():
                        try:
                            trigger = json.loads(meta.read_text()).get("trigger_word", "")
                        except Exception:
                            pass
                    loras.append({
                        "name": d.name, "trigger_word": trigger,
                        "path": str(files[0]),
                        "files": [{"name": f.name, "path": str(f)} for f in files],
                    })
                elif d.suffix == ".safetensors":
                    # No sidecar to read a trigger word out of, and no epochs to
                    # choose between — one file, one entry, named for itself.
                    loras.append({
                        "name": d.stem, "trigger_word": "",
                        "path": str(d),
                        "files": [{"name": d.name, "path": str(d)}],
                    })
        loras.sort(key=lambda l: l["name"].lower())
        return {
            "models": _model_status(),
            "loras": loras,
            "hf_token_set": bool(_hf_token()),
            "samplers": SAMPLERS,
            "schedulers": SCHEDULERS,
            "max_loras": MAX_LORAS,
            # Served rather than hardcoded in the page: the allowed cards are a
            # property of what the images were compiled for (see VIDEO_GPUS), so
            # a copy in the HTML would be a second source of truth that drifts
            # silently the first time one of them changes.
            "gpus": {
                "image": {"options": list(IMAGE_GPUS), "default": GPU},
                "video": {"options": list(VIDEO_GPUS), "default": VIDEO_GPU},
            },
            "max_refs": MAX_H3_REFS,
            "max_ref_videos": MAX_H3_REF_VIDEOS,
            # Same reason as gpus: which controls each video model reads, and
            # what is on the volume for each of its tasks, are properties of
            # the deployment. The composer builds itself from this.
            "video_models": _video_model_status(),
            "wan_experts": list(WAN_EXPERTS),
        }

    @api.post("/api/token")
    def set_token(payload: dict) -> dict[str, Any]:
        token = str(payload.get("hf_token") or "").strip()
        config["hf_token"] = token
        return {"ok": True, "hf_token_set": bool(token)}

    @api.post("/api/download")
    def download(payload: dict) -> dict[str, Any]:
        key = str(payload.get("key") or "")
        if key not in MODEL_CATALOGUE:
            return {"error": f"Unknown model: {key}"}
        download_job.spawn(key)
        return {"ok": True, "job_id": f"dl_{key}"}

    @api.post("/api/download-missing")
    def download_missing(payload: dict) -> dict[str, Any]:
        """
        Save the token (if one was supplied) and queue every missing weight.

        Taking the token in the same call is deliberate: pasting a key and then
        having to press Save before Download is a step that exists for no reason.
        """
        token = str(payload.get("hf_token") or "").strip()
        if token:
            config["hf_token"] = token

        volume.reload()
        missing = [m["key"] for m in _model_status() if not m["present"]]
        if not missing:
            return {"ok": True, "job_id": None, "missing": [], "note": "Everything is already here."}

        gated = [k for k in missing if MODEL_CATALOGUE[k]["gated"]]
        if gated and not _hf_token():
            return {
                "error": "A HuggingFace token is required for "
                + ", ".join(MODEL_CATALOGUE[k]["label"] for k in gated)
                + ". Paste one in the field above."
            }

        download_missing_job.spawn(missing)
        return {"ok": True, "job_id": "dl_all", "missing": missing}

    @api.post("/api/upload")
    async def upload(request: Request) -> JSONResponse:
        """
        Stage images. Pass an existing job_id to add to that dataset instead of
        starting a new one, so more files can be dropped in at any point.
        """
        try:
            return await _do_upload(request)
        except Exception as exc:
            # Surface the real reason in the UI instead of an opaque 500. A
            # missing python-multipart, a full volume or a permissions problem
            # all look identical otherwise.
            import traceback

            traceback.print_exc()
            return JSONResponse(
                {"error": f"{type(exc).__name__}: {exc}"}, status_code=500
            )

    async def _do_upload(request: Request) -> JSONResponse:
        form = await request.form()

        # Uploads always target a named dataset. Appending must never be able to
        # delete one that already exists, so track whether this call created it.
        # `dataset` is deliberately not reused as a loop variable below — an
        # earlier version named both this and the per-file basename `name`, so
        # the response reported the last uploaded filename as the dataset.
        dataset = str(form.get("dataset") or "").strip()
        if not dataset:
            return JSONResponse({"error": "A dataset name is required."}, 400)
        try:
            raw = _dataset_dir(dataset)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, 400)
        appending = raw.is_dir()
        raw.mkdir(parents=True, exist_ok=True)

        count, zips = 0, []
        for up in form.getlist("files"):
            filename = getattr(up, "filename", None)
            if not filename:
                continue
            basename = Path(filename).name
            suffix = Path(basename).suffix.lower()
            if suffix not in IMAGE_EXTS and suffix not in {".zip", ".txt"}:
                continue
            target = raw / basename
            with open(target, "wb") as out:
                while chunk := await up.read(1024 * 1024):
                    out.write(chunk)
            if suffix == ".zip":
                zips.append(target)
            elif suffix in IMAGE_EXTS:
                count += 1

        for z in zips:
            try:
                count += _safe_extract_zip(z, raw)
            except zipfile.BadZipFile:
                return JSONResponse({"error": f"{z.name} is not a valid zip."}, 400)
            finally:
                z.unlink(missing_ok=True)

        if not count:
            # Only bin the directory if this call made it — an append that
            # happens to contain no images must leave the dataset alone.
            if not appending:
                shutil.rmtree(raw, ignore_errors=True)
            return JSONResponse({"error": "No images found in the upload."}, 400)

        # `.aio()` rather than the blocking call every other route uses: this
        # is the one handler that has to stay `async def`, because it awaits
        # the multipart stream. A blocking commit here stalls the event loop
        # for the whole web container, not just this request.
        await volume.commit.aio()
        # Same-named files overwrite rather than duplicate, so re-dropping the
        # same folder is idempotent instead of doubling the dataset.
        return JSONResponse({"dataset": dataset, "added": count, **_dataset_stats(raw)})

    def _dataset_or_error(name: str):
        """Resolve a dataset name, returning (dir, None) or (None, error dict)."""
        try:
            d = _dataset_dir(name)
        except ValueError as exc:
            return None, {"error": str(exc)}
        if not d.is_dir():
            return None, {"error": f"No dataset named {name!r}."}
        return d, None

    @api.get("/api/datasets")
    def list_datasets() -> dict[str, Any]:
        volume.reload()
        DATASETS.mkdir(parents=True, exist_ok=True)
        out = [
            _dataset_stats(d) for d in sorted(DATASETS.iterdir())
            if d.is_dir() and not d.name.startswith(".")
        ]
        out.sort(key=lambda r: -r["modified"])
        return {"datasets": out}

    @api.post("/api/datasets")
    def create_dataset(payload: dict) -> dict[str, Any]:
        name = str(payload.get("name") or "").strip()
        try:
            d = _dataset_dir(name)
        except ValueError as exc:
            return {"error": str(exc)}
        if d.exists():
            return {"error": f"A dataset named {name!r} already exists."}
        volume.reload()
        d.mkdir(parents=True)
        _write_dataset_meta(d, trigger_word=str(payload.get("trigger_word") or ""))
        volume.commit()
        return {"ok": True, **_dataset_stats(d)}

    @api.get("/api/datasets/{name}")
    def dataset_detail(name: str) -> dict[str, Any]:
        """
        Image metadata only — thumbnails come from /api/thumb one at a time.

        The previous version inlined every thumbnail as base64 in this response,
        which put a 200-image dataset at ~6.6 MB before a single tile rendered
        and rebuilt every thumbnail on every load.
        """
        volume.reload()
        d, err = _dataset_or_error(name)
        if err:
            return err

        from PIL import Image

        items = []
        for img in _dataset_images(d):
            try:
                st = img.stat()
            except OSError:
                continue
            # Pixel dimensions alongside filesize: together they are what
            # actually informs a keep/cut call. PIL parses the header only, so
            # this is a small read per file rather than a decode.
            w = h = None
            try:
                with Image.open(img) as im:
                    w, h = im.size
            except Exception:
                pass
            items.append({
                "name": img.name,
                "caption": _caption_of(img),
                "bytes": st.st_size,
                "width": w, "height": h,
                "mtime": st.st_mtime,
            })
        return {**_dataset_stats(d), "images": items}

    @api.post("/api/datasets/{name}/meta")
    def dataset_meta(name: str, payload: dict) -> dict[str, Any]:
        volume.reload()
        d, err = _dataset_or_error(name)
        if err:
            return err
        _write_dataset_meta(d, trigger_word=str(payload.get("trigger_word") or ""))
        volume.commit()
        return {"ok": True, **_dataset_stats(d)}

    @api.post("/api/datasets/{name}/delete")
    def delete_dataset(name: str) -> dict[str, Any]:
        volume.reload()
        d, err = _dataset_or_error(name)
        if err:
            return err
        shutil.rmtree(d, ignore_errors=True)
        volume.commit()
        return {"ok": True}

    @api.get("/api/thumb/{name}/{filename}")
    def thumb(name: str, filename: str):
        """
        One thumbnail, cached beside the dataset, served with a long max-age.

        Cached by mtime, so re-editing a caption never re-encodes the image and
        replacing an image does invalidate it.
        """
        from fastapi.responses import Response
        from PIL import Image

        d, err = _dataset_or_error(name)
        if err:
            return JSONResponse(err, status_code=404)
        img = d / Path(filename).name  # basename only — no directory escape
        if img.suffix.lower() not in IMAGE_EXTS or not img.is_file():
            return JSONResponse({"error": "Image not found."}, status_code=404)

        thumbs = d / THUMB_DIR
        thumbs.mkdir(exist_ok=True)
        cached = thumbs / (img.stem + ".jpg")
        try:
            if not cached.exists() or cached.stat().st_mtime < img.stat().st_mtime:
                with Image.open(img) as im:
                    im = im.convert("RGB")
                    im.thumbnail((THUMB_PX, THUMB_PX), Image.LANCZOS)
                    buf = io.BytesIO()
                    im.save(buf, "JPEG", quality=78, optimize=True)
                    cached.write_bytes(buf.getvalue())
                # Deliberately no volume.commit() here. Thumbnails are derived
                # data — if a container dies before the write is durable, the
                # next request regenerates one. Committing per thumbnail turned
                # a grid of 700 tiles into 700 volume commits.
            data = cached.read_bytes()
        except Exception as exc:
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

        return Response(content=data, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})

    @api.get("/api/image/{name}/{filename}")
    def full_image(name: str, filename: str):
        """The original file, for the full-size viewer. Never the thumbnail."""
        from fastapi.responses import Response

        d, err = _dataset_or_error(name)
        if err:
            return JSONResponse(err, status_code=404)
        img = d / Path(filename).name
        if img.suffix.lower() not in IMAGE_EXTS or not img.is_file():
            return JSONResponse({"error": "Image not found."}, status_code=404)
        mime = {"png": "image/png", "webp": "image/webp"}.get(
            img.suffix.lower().lstrip("."), "image/jpeg")
        return Response(content=img.read_bytes(), media_type=mime,
                        headers={"Cache-Control": "public, max-age=86400"})

    @api.post("/api/datasets/{name}/caption")
    def save_caption(name: str, payload: dict) -> dict[str, Any]:
        """One caption, saved on blur. Bulk save was how edits went missing."""
        volume.reload()
        d, err = _dataset_or_error(name)
        if err:
            return err
        img = d / Path(str(payload.get("image") or "")).name
        if img.suffix.lower() not in IMAGE_EXTS or not img.is_file():
            return {"error": "Image not found."}
        img.with_suffix(".txt").write_text(
            str(payload.get("caption") or "").strip()[:MAX_CAPTION_CHARS])
        volume.commit()
        return {"ok": True}

    @api.post("/api/datasets/{name}/remove")
    def remove_image(name: str, payload: dict) -> dict[str, Any]:
        """
        Move an image and its caption into the dataset's .trash/.

        Non-destructive by default: a mis-click during a cull costs a file move,
        not the file. Nothing surfaces .trash/ yet — it is there so the bytes
        still exist when someone asks for undo.
        """
        volume.reload()
        d, err = _dataset_or_error(name)
        if err:
            return err
        img = d / Path(str(payload.get("image") or "")).name
        if img.suffix.lower() not in IMAGE_EXTS or not img.is_file():
            return {"error": "Image not found."}

        trash = d / TRASH_DIR
        trash.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d%H%M%S")
        for part in (img, img.with_suffix(".txt")):
            if part.is_file():
                part.rename(trash / f"{stamp}-{part.name}")
        (d / THUMB_DIR / (img.stem + ".jpg")).unlink(missing_ok=True)

        volume.commit()
        return {"ok": True, **_dataset_stats(d)}

    @api.get("/api/datasets/{name}/insight")
    def dataset_insight(name: str, trigger: str = "") -> dict[str, Any]:
        volume.reload()
        d, err = _dataset_or_error(name)
        if err:
            return err
        return _caption_insight(d, trigger)

    @api.post("/api/datasets/{name}/prepend-trigger")
    def prepend_trigger(name: str, payload: dict) -> dict[str, Any]:
        """
        Put the trigger word at the front of every caption that lacks it.

        For imported datasets: your own .txt files are used verbatim, so a
        caption without the trigger word trains a LoRA the trigger cannot
        summon. This fixes that without discarding the text.

        Idempotent by design — the test is `startswith`, not `in`. A substring
        test would false-positive on short triggers (a "cat" LoRA would skip
        "a cat sitting"), and running this twice must never double the prefix.
        """
        trigger = str(payload.get("trigger_word") or "").strip()
        if not trigger:
            return {"error": "A trigger word is required."}

        volume.reload()
        d, err = _dataset_or_error(name)
        if err:
            return err

        changed = 0
        for img in _dataset_images(d):
            txt = img.with_suffix(".txt")
            cur = txt.read_text().strip() if txt.exists() else ""
            if not cur:
                new = trigger
            elif cur.startswith(trigger):
                continue
            else:
                new = f"{trigger}, {cur}"
            txt.write_text(new[:MAX_CAPTION_CHARS])
            changed += 1

        _write_dataset_meta(d, trigger_word=trigger)
        volume.commit()
        return {"ok": True, "changed": changed}

    @api.post("/api/caption")
    def caption(payload: dict) -> dict[str, Any]:
        name = str(payload.get("dataset") or "")
        d, err = _dataset_or_error(name)
        if err:
            return err
        trigger = str(payload.get("trigger_word") or "")
        if trigger:
            _write_dataset_meta(d, trigger_word=trigger)
            volume.commit()
        job_id = f"cap{time.strftime('%Y%m%d%H%M%S')}{os.urandom(2).hex()}"
        caption_job.spawn(
            job_id=job_id, dataset=name, trigger_word=trigger,
            style=str(payload.get("style") or "descriptive"),
            length=str(payload.get("length") or "medium"),
            overwrite=bool(payload.get("overwrite")),
        )
        return {"ok": True, "job_id": job_id}

    @api.post("/api/train")
    def train(payload: dict) -> dict[str, Any]:
        dataset = str(payload.get("dataset") or "")
        lora_name = str(payload.get("lora_name") or "").strip()
        trigger = str(payload.get("trigger_word") or "").strip()
        d, err = _dataset_or_error(dataset)
        if err:
            return err
        if not _dataset_images(d):
            return {"error": f"{dataset!r} has no images."}
        if not NAME_RE.match(lora_name):
            return {"error": "LoRA name: letters, digits, - and _ only."}
        if not trigger:
            return {"error": "A trigger word is required."}
        job_id = f"tr{time.strftime('%Y%m%d%H%M%S')}{os.urandom(2).hex()}"

        def num(k, d, cast):
            try:
                v = payload.get(k)
                return cast(v) if v not in (None, "") else d
            except (TypeError, ValueError):
                return d

        train_job.spawn(
            job_id=job_id, dataset=dataset, lora_name=lora_name, trigger_word=trigger,
            resolution=num("resolution", 1024, int),
            batch_size=num("batch_size", 1, int),
            num_repeats=num("num_repeats", 1, int),
            network_dim=num("network_dim", 32, int),
            network_alpha=num("network_alpha", 32, int),
            learning_rate=num("learning_rate", 1e-4, float),
            max_train_epochs=num("max_train_epochs", 30, int),
            save_every_n_epochs=num("save_every_n_epochs", 1, int),
            seed=num("seed", 42, int),
            fp8=bool(payload.get("fp8")),
            blocks_to_swap=num("blocks_to_swap", 0, int),
        )
        return {"ok": True, "job_id": job_id}

    @api.post("/api/generate")
    def generate(payload: dict) -> dict[str, Any]:
        prompt = str(payload.get("prompt") or "").strip()
        regions = payload.get("regions") or []
        if not prompt and not regions:
            return {"error": "A prompt is required."}

        def num(k, d, cast):
            try:
                v = payload.get(k)
                return cast(v) if v not in (None, "") else d
            except (TypeError, ValueError):
                return d

        # `lora_path`/`lora_multiplier` are the pre-stack shape of this request;
        # accepted so an older client keeps working against the new backend.
        stack = payload.get("loras")
        if not stack and payload.get("lora_path"):
            stack = [{"path": payload["lora_path"], "unet": num("lora_multiplier", 1.0, float)}]

        # Reject here rather than in the job: a bad path is a form error, and
        # spawning would cost a cold A100 before discovering it.
        volume.reload()
        try:
            stack = _validate_loras(stack)
        except ValueError as exc:
            return {"error": str(exc)}

        job_id = f"gen{time.strftime('%Y%m%d%H%M%S')}{os.urandom(2).hex()}"
        runner = _on_gpu(Generator, payload.get("gpu"), IMAGE_GPUS, GPU)
        runner().generate.spawn(job_id=job_id, params={
            "prompt": prompt,
            "negative_prompt": str(payload.get("negative_prompt") or ""),
            "model": str(payload.get("model") or "turbo"),
            "loras": stack,
            "regions": regions,
            "region_weight": num("region_weight", 1.0, float),
            "width": num("width", 1024, int),
            "height": num("height", 1024, int),
            "num_images": max(1, min(4, num("num_images", 1, int))),
            "steps": num("steps", None, int),
            "cfg_scale": num("cfg_scale", None, float),
            "seed": num("seed", None, int),
            "sampler": str(payload.get("sampler") or "Euler"),
            "scheduler": str(payload.get("scheduler") or "Automatic"),
            "shift": num("shift", 1.15, float),
        })
        return {"ok": True, "job_id": job_id}

    @api.post("/api/video")
    def video(payload: dict) -> dict[str, Any]:
        """
        Queue one clip, on whichever video model was asked for.

        Which *task* runs is never asked for — text-to-video, image-to-video and
        first/last are read off what was attached, the same way the image side
        never asks whether you meant a batch. What the client picks is the
        model, because that is the thing it cannot infer.

        Everything rejectable is rejected here, on CPU: a bad aspect, a LoRA
        outside loras/, a missing weight. Discovering any of them inside the job
        costs a cold H100 and tens of gigabytes of loading first, and surfaces
        as a dead job rather than a form error.
        """
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            return {"error": "A prompt is required."}

        model = str(payload.get("model") or "h3")
        spec = VIDEO_MODELS.get(model)
        if spec is None:
            return {"error": f"Unknown video model {model!r}. "
                             f"One of: {', '.join(VIDEO_MODELS)}"}
        supports = spec["supports"]

        def num(k, d, cast):
            try:
                v = payload.get(k)
                return cast(v) if v not in (None, "") else d
            except (TypeError, ValueError):
                return d

        first = payload.get("first_frame") or None
        last = payload.get("last_frame") or None
        if last and not supports["last_frame"]:
            return {"error": f"{spec['label']} takes a first frame only."}

        refs = [r for r in (payload.get("references") or []) if r][:MAX_H3_REFS]
        vids = [v for v in (payload.get("ref_videos") or []) if v][:MAX_H3_REF_VIDEOS]
        if (refs or vids) and not supports["references"]:
            return {"error": f"{spec['label']} does not take references."}
        if len(refs) + len(vids) > MAX_H3_REF_TOTAL:
            return {"error": f"{MAX_H3_REF_TOTAL} references in total is the "
                             f"model's limit ({len(refs)} images + {len(vids)} videos)."}

        volume.reload()
        try:
            stack = _validate_video_loras(payload.get("loras")) if supports["loras"] else []
        except ValueError as exc:
            return {"error": str(exc)}
        if stack and not supports["experts"]:
            # One model, so there is no expert to target. Refusing beats
            # applying it anyway (a high-noise speed LoRA on a dense checkpoint
            # is a quality loss you would never be told about) and beats
            # dropping it (a row that does nothing looks like a row that did).
            crossed = [l["name"] for l in stack if l["expert"] != "both"]
            if crossed:
                return {"error": f"{spec['label']} has one expert, so it cannot "
                                 f"target high or low noise: {', '.join(crossed)}."}

        aspect = str(payload.get("aspect") or "16:9")
        tier = str(payload.get("tier") or spec["defaults"]["tier"])
        try:
            if model == "h3":
                task = "ref2va" if (refs or vids) else "fl2va"
                _h3_canvas(aspect, tier)  # raises with the valid set named
            else:
                task = _wan_task(first, last)
                _wan_canvas(aspect, tier, "5b" if model == "wan5b" else "14b")
        except ValueError as exc:
            return {"error": str(exc)}

        # Named per task, so "download 57 GB" is never the answer to a run that
        # needs 28.6 of it.
        sizes = _sizes_on_disk(MODEL_CATALOGUE[k]["dest"]
                               for k in spec["requires"][task])
        missing = [MODEL_CATALOGUE[k]["label"] for k in spec["requires"][task]
                   if not sizes[MODEL_CATALOGUE[k]["dest"]]]
        if missing:
            return {"error": f"Not downloaded: {', '.join(missing)}. "
                             "Get them under Settings."}

        ref_size = str(payload.get("ref_size") or "match")
        if ref_size not in H3_REF_SIZES:
            return {"error": f"ref_size must be one of: {', '.join(H3_REF_SIZES)}"}

        d = spec["defaults"]
        job_id = f"vid{time.strftime('%Y%m%d%H%M%S')}{os.urandom(2).hex()}"
        runner = _on_gpu(VideoGenerator, payload.get("gpu"), VIDEO_GPUS, VIDEO_GPU)
        runner().generate.spawn(job_id=job_id, params={
            "model": model,
            "prompt": prompt,
            # Dropped rather than passed through for a model that cannot read
            # it: a negative prompt that reaches a guidance-distilled checkpoint
            # is not applied, and a sidecar that records one is a sidecar that
            # lies about how the clip was made.
            "negative_prompt": (str(payload.get("negative_prompt") or "")
                                if supports["negative"] else ""),
            "aspect": aspect,
            "tier": tier,
            "seconds": num("seconds", float(d["seconds"]), float),
            "steps": max(1, min(60, num("steps", d["steps"], int))),
            "cfg": num("cfg", d.get("cfg", 1.0), float),
            "shift": num("shift", d.get("shift", WAN_DEFAULT_SHIFT), float),
            "switch_at": num("switch_at", None, int),
            "seed": num("seed", None, int),
            "sampler": str(payload.get("sampler") or d["sampler"]),
            "scheduler": str(payload.get("scheduler") or d["scheduler"]),
            "loras": stack,
            # References and keyframes are alternatives, not a stack: on H3 they
            # load different transformers. The job takes both fields and the
            # generator ignores the keyframes when references are present, so a
            # client that sends both gets the reference run it asked for rather
            # than a validation error about a combination it cannot express.
            "references": refs,
            "ref_videos": vids,
            "ref_size": ref_size,
            "first_frame": first,
            "last_frame": last,
        })
        return {"ok": True, "job_id": job_id, "model": model, "mode": task}

    @api.get("/api/gallery")
    def gallery() -> dict[str, Any]:
        """Everything on the volume, newest first — no job id required."""
        volume.reload()
        return {"items": _gallery()}

    @api.get("/api/file/{job_id}/{name}")
    def output_file(job_id: str, name: str):
        """
        Stream one result off the volume, image or video.

        Deliberately not base64 in a JSON body the way /api/outputs does it:
        inlining is what made a gallery impossible, since a page of stills or a
        clip with its soundtrack is tens of megabytes of JSON before anything
        renders, and a <video> cannot seek until all of it has arrived.

        Matching the whole filename, not its stem: `Path("../../x").stem` is
        "x", which passes NAME_RE while the joined path still escapes outputs/.
        The separators have to be visible to the regex to be rejected.
        """
        if not NAME_RE.match(job_id) or not OUTPUT_FILE_RE.match(name):
            return JSONResponse({"error": "Invalid name."}, status_code=400)

        # Optimistic: look first, reload only on a miss.
        #
        # Reloading on every request was both slow and actively wrong here. A
        # gallery is a grid, so opening it fires a dozen of these at once, and
        # this container serves 20 concurrently — concurrent reloads of the same
        # volume returned a 500 for one of two simultaneous requests in testing.
        # A file that is already visible needs no reload at all; only a file
        # written by the GPU container since this one last synced does, and that
        # is exactly the miss case.
        path = OUTPUTS / job_id / name
        if not path.is_file():
            volume.reload()
            if not path.is_file():
                return JSONResponse({"error": "Not found."}, status_code=404)
        return FileResponse(
            str(path),
            media_type=MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"),
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @api.post("/api/outputs/{job_id}/delete")
    def delete_output(job_id: str) -> dict[str, Any]:
        """Cull a result. Moves to outputs/.trash/, the way datasets do."""
        if not NAME_RE.match(job_id):
            return {"error": "Invalid job_id."}
        volume.reload()
        d = OUTPUTS / job_id
        if not d.is_dir():
            return {"error": "Not found."}
        trash = OUTPUTS / TRASH_DIR
        trash.mkdir(parents=True, exist_ok=True)
        shutil.move(str(d), str(trash / f"{job_id}-{int(time.time())}"))
        volume.commit()
        return {"ok": True}

    @api.get("/api/outputs/{job_id}")
    def outputs(job_id: str) -> dict[str, Any]:
        """Serve generated PNGs off the volume, keeping them out of the job dict."""
        if not NAME_RE.match(job_id):
            return {"error": "Invalid job_id."}
        volume.reload()
        d = OUTPUTS / job_id
        if not d.is_dir():
            return {"images": []}
        return {
            "images": [
                {"name": p.name,
                 "data": "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()}
                for p in sorted(d.glob("*.png"), key=lambda p: p.stat().st_mtime)
            ]
        }

    @api.get("/api/status/{job_id}")
    def status(job_id: str) -> dict[str, Any]:
        try:
            return jobs.get(job_id) or {"status": "unknown"}
        except Exception as exc:
            return {"status": "unknown", "error": str(exc)}

    @api.post("/api/stop/{job_id}")
    def stop(job_id: str) -> dict[str, Any]:
        cur = jobs.get(job_id) or {}
        cur["stop"] = True
        jobs[job_id] = cur
        return {"ok": True}

    return api


# --------------------------------------------------------------------------
# CLI — train from a local folder, no browser
#
#   modal run app.py --images-dir ./photos --lora-name my_style
#   modal run app.py --images-dir ./photos --lora-name my_style --caption
# --------------------------------------------------------------------------


@app.function(image=web_image, cpu=2.0, timeout=1800, volumes={"/workspace": volume})
def stage_upload(dataset: str, files: list[tuple[str, bytes]]) -> int:
    """Land locally-read files on the volume. Browser uploads use /api/upload."""
    raw = _dataset_dir(dataset)
    raw.mkdir(parents=True, exist_ok=True)
    for name, blob in files:
        (raw / Path(name).name).write_bytes(blob)  # basename only
    volume.commit()
    return len(files)


@app.local_entrypoint()
def main(
    images_dir: str,
    lora_name: str,
    trigger_word: str = "",
    dataset_name: str = "",
    caption: bool = False,
    max_train_epochs: int = 30,
    network_dim: int = 32,
    resolution: int = 1024,
) -> None:
    """Train straight from a local folder. Runs synchronously and prints the result."""
    src = Path(images_dir).expanduser()
    if not src.is_dir():
        raise SystemExit(f"Not a directory: {src}")

    payload = [
        (p.name, p.read_bytes())
        for p in sorted(src.iterdir())
        if p.suffix.lower() in IMAGE_EXTS or p.suffix.lower() == ".txt"
    ]
    n_images = sum(1 for n, _ in payload if Path(n).suffix.lower() in IMAGE_EXTS)
    if not n_images:
        raise SystemExit(f"No images in {src}")

    trigger = trigger_word or lora_name
    # The CLI names the dataset after the LoRA so a run started here is visible
    # and reusable on the Datasets tab rather than being a one-shot upload.
    dataset = dataset_name or lora_name
    job_id = f"cli_{time.strftime('%Y%m%d_%H%M%S')}"

    print(f"Uploading {n_images} images to dataset {dataset!r}…")
    stage_upload.remote(dataset, payload)

    if caption:
        print("Captioning…")
        print(json.dumps(
            caption_job.remote(job_id=f"{job_id}_cap", dataset=dataset, trigger_word=trigger),
            indent=2,
        ))

    print(json.dumps(
        train_job.remote(
            job_id=job_id,
            dataset=dataset,
            lora_name=lora_name,
            trigger_word=trigger,
            max_train_epochs=max_train_epochs,
            network_dim=network_dim,
            network_alpha=network_dim,
            resolution=resolution,
        ),
        indent=2,
    ))


# --------------------------------------------------------------------------
# UI — one page, no build step, no dependencies
# --------------------------------------------------------------------------

UI_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Visionary</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#000;--panel:rgba(255,255,255,.04);--line:rgba(255,255,255,.10);--fg:#f5f5f5;
      --mut:#8a8a8a;--dim:#5a5a5a;--drawer:320px}
body{background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif;
     -webkit-font-smoothing:antialiased;height:100dvh;display:flex;flex-direction:column;overflow:hidden}
svg{width:100%;height:100%;display:block}

/* Chrome ---------------------------------------------------------------- */
/* There is no navigation rail, and no two-up switcher up here either.
   Generate and Train are not peers: one is where the machine is used, the
   other is where it is changed, and a 50/50 toggle asserts a balance that
   does not exist. So Generate is simply the page — it has no nav item because
   it is not a place you go — and Train is a door on the right: one slot,
   labelled with where it leads rather than with where you are. */
.top{flex:0 0 56px;display:flex;align-items:center;gap:14px;padding:0 14px 0 18px;
     border-bottom:1px solid rgba(255,255,255,.07)}
.brand{border:0;background:none;color:var(--fg);font:600 15px/1 inherit;letter-spacing:-.01em;
  padding:8px 2px;cursor:pointer}
.grow{flex:1;min-width:0}

/* The door, and the reason it is worth a permanent slot: a training run lasts
   hours and you are meant to leave and keep generating while it goes. So the
   way back to it is also the readout on it — progress on the control that
   takes you there, rather than in a place you have to go to look. */
.door{display:inline-flex;align-items:center;gap:8px;border:1px solid var(--line);
  background:rgba(255,255,255,.04);color:#ddd;border-radius:999px;padding:7px 14px 7px 11px;
  font:500 13px/1 inherit;cursor:pointer;transition:background .12s,color .12s,border-color .12s}
.door:hover{background:rgba(255,255,255,.1);color:var(--fg);border-color:rgba(255,255,255,.22)}
.door svg{width:15px;height:15px;flex:none}
.door .ring circle{fill:none;stroke-width:3}
.door .ring .bg{stroke:rgba(255,255,255,.2)}
.door .ring .fg{stroke:currentColor;stroke-linecap:round;
  transform:rotate(-90deg);transform-origin:50% 50%;transition:stroke-dashoffset .5s}
.door.live{color:#fff;border-color:rgba(255,255,255,.3)}
.sep{width:1px;height:22px;flex:none;background:rgba(255,255,255,.12)}

/* The one black-on-white switcher left in the product: Training and Datasets,
   which is the only pair in here that really is two equal halves of one thing. */
.switch{display:inline-flex;background:rgba(255,255,255,.06);border-radius:11px;padding:3px;gap:2px}
.switch button{border:0;background:none;color:var(--mut);padding:6px 15px;border-radius:9px;
  font:500 13px/1.35 inherit;cursor:pointer;transition:color .12s,background .12s}
.switch button:hover{color:#ddd}
.switch button.on{background:#fff;color:#000}
.ico{width:34px;height:34px;flex:none;border:0;border-radius:10px;background:none;color:var(--mut);
  cursor:pointer;padding:8px;transition:background .12s,color .12s}
.ico:hover{background:rgba(255,255,255,.08);color:var(--fg)}
.ico.on{color:var(--fg);background:rgba(255,255,255,.1)}

.views{flex:1;min-height:0;position:relative}
.view{position:absolute;inset:0;display:flex;min-width:0}
.view.scroll{display:block;overflow:auto}
.hide{display:none !important}

/* Generate -------------------------------------------------------------- */
/* The canvas gets the width; the console gets the height.
   A settings rail down the left costs the picture 384px of the one dimension
   it cannot get back — an image is wide, a screen is wide, and a column of
   dropdowns is neither. Vertical is the cheap axis: the console is a bar under
   the canvas that grows downward when it has more to say and collapses when it
   does not, so on a bare prompt the canvas has the whole room. */
.stage{flex:1;min-width:0;display:flex;flex-direction:column}
.canvas{flex:1;min-height:0;overflow:auto;padding:22px 28px;display:flex;flex-direction:column}
/* Capped, so a console with everything open can never push the canvas out of
   the frame — past the cap it scrolls itself instead. */
.console{flex:none;max-height:54dvh;overflow:auto;padding:13px 28px 15px;
  border-top:1px solid rgba(255,255,255,.07);background:rgba(255,255,255,.012)}
.crow{display:flex;align-items:flex-end;gap:10px}
.crow .field{flex:1;min-width:0}
.crow button.b{flex:none;height:44px;padding:0 26px}
.drawer{flex:0 0 var(--drawer);min-width:0;border-left:1px solid rgba(255,255,255,.07);overflow:auto}
.drawer-in{width:var(--drawer);padding:12px 14px 40px}
/* Collapsed by flex-basis rather than display:none so the canvas reflows
   smoothly; the inner column keeps its width so nothing re-wraps on the way. */
.studio.nodrawer .drawer{flex-basis:0;border-left:0;overflow:hidden}
.drawer{transition:flex-basis .18s ease}
.drawer-head{display:flex;align-items:center;gap:8px;margin-bottom:10px;min-height:28px}

/* Controls in a row, not a form. Each one shows its own value — "Krea 2
   Turbo", "16:9", "720p", "5s" — so a label above it would only repeat what
   the control already says. The two whose value means nothing on its own get
   an icon instead of a word: a count and a seed. */
.opts{display:flex;flex-wrap:wrap;align-items:center;gap:7px;margin-top:9px}
.opt{display:inline-flex;align-items:center;gap:5px;height:36px;padding:0 5px 0 9px;
  background:rgba(255,255,255,.05);border:1px solid var(--line);border-radius:11px}
.opt:focus-within{border-color:rgba(255,255,255,.28)}
.opt>svg{width:14px;height:14px;flex:none;color:var(--dim)}
.opt select,.opt input{width:auto;border:0;background:none;padding:0 2px;height:34px;border-radius:8px}
.opt input{width:76px}
.opt.wide{flex:1;min-width:220px}
.opt.wide input,.opt.wide textarea{flex:1;min-width:0;width:auto}
.vr{width:1px;height:20px;flex:none;background:var(--line);margin:0 4px}
/* Icon-only controls in the strip: same box, square, no value to show. */
.opt.ib{padding:0;width:36px;justify-content:center;cursor:pointer;color:var(--mut);
  transition:background .12s,color .12s}
.opt.ib:hover{background:rgba(255,255,255,.1);color:var(--fg)}
.opt.ib.on{background:rgba(255,255,255,.14);color:var(--fg);border-color:rgba(255,255,255,.24)}
.opt.ib>svg{width:16px;height:16px;color:inherit}
.adv{margin-top:9px;padding-top:10px;border-top:1px solid var(--line)}

/* The prompt field. The textarea and the Image/Video chip are one bordered
   box, and the prompt itself is shared by both — switching mid-sentence keeps
   the sentence. Image and video are not two workspaces to navigate between;
   they are two things the same sentence can become, so the choice belongs
   inside the field you are already typing in, at the smallest size that still
   reads. Everything below the field is options for that choice. */
.field{border:1px solid var(--line);background:rgba(255,255,255,.05);border-radius:13px;padding:2px 2px 0}
.field:focus-within{border-color:rgba(255,255,255,.28)}
.field textarea{border:0;background:none;border-radius:0;padding:9px 10px 2px}
.field .bar2{display:flex;align-items:center;gap:8px;padding:2px 5px 5px}
.kinds{display:inline-flex;gap:2px;background:rgba(255,255,255,.05);border-radius:999px;padding:2px}
.kinds button{display:inline-flex;align-items:center;gap:5px;border:0;background:none;color:var(--mut);
  border-radius:999px;padding:4px 10px 4px 8px;font:500 12px/1 inherit;cursor:pointer;
  transition:background .12s,color .12s}
.kinds button svg{width:13px;height:13px;flex:none}
.kinds button:hover{color:#ddd}
.kinds button.on{background:rgba(255,255,255,.14);color:var(--fg)}

.sec{border-top:1px solid var(--line);margin-top:14px;padding-top:14px}
.sec:first-child{border-top:0;margin-top:0;padding-top:0}
.sec>label:first-child{margin-bottom:8px}
.f2{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.f3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}

h1{font-size:19px;font-weight:600;margin-bottom:3px;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:13px;margin-bottom:22px}
.card{border:1px solid var(--line);background:rgba(255,255,255,.02);border-radius:16px;padding:16px;margin-bottom:12px}
.row{display:flex;align-items:center;gap:14px}
label{display:block;font-size:12px;color:var(--mut);margin-bottom:6px}
input,textarea,select{width:100%;background:rgba(255,255,255,.05);border:1px solid var(--line);
  border-radius:11px;padding:10px 12px;color:var(--fg);font:inherit;outline:none}
input:focus,textarea:focus,select:focus{border-color:rgba(255,255,255,.28)}
input:disabled,select:disabled{opacity:.45}
textarea{resize:vertical}
button.b{background:#fff;color:#000;border:0;border-radius:12px;padding:11px 18px;font:600 14px/1 inherit;cursor:pointer}
button.b:disabled{background:rgba(255,255,255,.2);color:rgba(0,0,0,.4);cursor:not-allowed}
button.s{background:rgba(255,255,255,.07);color:var(--fg);border:1px solid var(--line);
  border-radius:11px;padding:9px 15px;font:500 13px/1 inherit;cursor:pointer}
button.s:hover{background:rgba(255,255,255,.12)}
button.s:disabled{opacity:.4;cursor:not-allowed}
button.t{background:none;border:0;color:var(--mut);font:500 12px/1 inherit;cursor:pointer;padding:6px 2px}
button.t:hover{color:var(--fg)}
.pill{display:inline-flex;align-items:center;gap:6px;background:rgba(255,255,255,.06);border:0;
  border-radius:999px;padding:7px 13px;color:#ddd;font:13px inherit;cursor:pointer}
.pill.on{background:#fff;color:#000}
.bar{height:3px;background:rgba(255,255,255,.1);border-radius:99px;overflow:hidden;margin-top:10px}
.bar>i{display:block;height:100%;background:#fff;border-radius:99px;transition:width .4s}
.muted{color:var(--dim);font-size:12px}
.ok{color:#4ade80}.warn{color:#fbbf24}.err{color:#f87171}
.err-box{border:1px solid rgba(248,113,113,.25);background:rgba(248,113,113,.1);color:#fca5a5;
  border-radius:12px;padding:11px 14px;font-size:13px;margin-bottom:12px}
code{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:#bbb}

/* Canvas ---------------------------------------------------------------- */
/* Not `.empty` — `.ds-cover.empty` already owned that name, and a bare `.empty`
   carrying min-height:62vh landed on the dataset placeholder and inflated every
   card in its grid row to 606px. */
.blank{flex:1;min-height:200px;display:grid;place-items:center;color:var(--dim)}
/* `.blank`, not `.empty` — the placeholder is `.blank`, so this rule matched
   nothing and the global `svg{width:100%}` inflated the glyph to the width of
   the canvas. A selector that misses is invisible in the CSS and enormous on
   the page. */
.blank .glyph{width:44px;height:44px;margin:0 auto 14px;opacity:.3}
/* The figure hugs the image rather than the column: a portrait still in a 1fr
   cell would otherwise sit inside letterbox bars it does not have, and the
   border would describe the grid instead of the picture. --shot-h is set per
   batch size so four images are a contact sheet you can see at once, not a
   scroll. */
.shots{display:grid;gap:16px;margin:0 auto}
.shot{position:relative;width:fit-content;max-width:100%;justify-self:center;
  border-radius:16px;overflow:hidden;border:1px solid var(--line);background:rgba(255,255,255,.02)}
.shot img{display:block;max-width:100%;max-height:var(--shot-h,none);width:auto;height:auto}
.shot .acts{position:absolute;right:10px;bottom:10px;display:flex;gap:6px;opacity:0;transition:opacity .12s}
.shot:hover .acts{opacity:1}
#vid-out video{width:100%;max-width:1180px;margin:0 auto;display:block;border-radius:16px;
  border:1px solid var(--line);background:#000}

/* Cards ----------------------------------------------------------------- */
/* One card, three homes: the drawer, the full gallery, the dataset index.
   4/3 with contain, so nothing on the volume is shown as a crop it is not. */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(232px,1fr));gap:14px}
.drawer .grid{grid-template-columns:1fr;gap:10px}
.gal{border:1px solid var(--line);background:rgba(255,255,255,.02);border-radius:14px;overflow:hidden;position:relative}
.gal:hover{border-color:rgba(255,255,255,.2)}
.gal .media{aspect-ratio:4/3;background:rgba(255,255,255,.045);display:block;width:100%;object-fit:contain;cursor:zoom-in}
/* Hover cluster on the media: the two verbs worth one click. Everything else
   is one more click away in the overflow, which is the correct price for it. */
.gal .quick{position:absolute;top:8px;right:8px;display:flex;gap:6px;opacity:0;transition:opacity .12s}
.gal:hover .quick,.gal .quick:focus-within{opacity:1}
.gal .quick button{width:28px;height:28px;flex:none;border:0;border-radius:999px;background:rgba(0,0,0,.66);
  color:#eee;padding:7px;cursor:pointer;backdrop-filter:blur(8px)}
.gal .quick button:hover{background:rgba(0,0,0,.9)}
/* The footer is the label. A 24px photo or play glyph says what this is in
   less space and less noise than the word "image" ever did. */
.foot{display:flex;align-items:center;gap:8px;padding:6px 8px 6px 10px;border-top:1px solid var(--line);min-height:40px}
.foot .kind{width:24px;height:24px;flex:none;color:var(--dim);padding:2px}
.foot .when{font-size:11px;color:var(--dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.foot .more{width:28px;height:28px;flex:none;border:0;border-radius:8px;background:none;color:var(--mut);
  padding:5px;cursor:pointer}
.foot .more:hover{background:rgba(255,255,255,.09);color:var(--fg)}

/* Overflow menu --------------------------------------------------------- */
.menu{position:fixed;z-index:80;min-width:196px;padding:5px;border-radius:13px;
  border:1px solid rgba(255,255,255,.14);background:#111;box-shadow:0 18px 48px rgba(0,0,0,.6)}
.menu button{display:block;width:100%;text-align:left;border:0;background:none;color:#e6e6e6;
  font:13px/1 inherit;padding:9px 11px;border-radius:9px;cursor:pointer}
.menu button:hover{background:rgba(255,255,255,.09)}
.menu button.danger{color:#f87171}
.menu button.danger:hover{background:rgba(248,113,113,.14)}
.menu hr{border:0;border-top:1px solid rgba(255,255,255,.1);margin:5px 7px}

/* Sheets: settings, metadata, lightbox ---------------------------------- */
.scrim{position:fixed;inset:0;background:rgba(0,0,0,.72);z-index:60;display:grid;place-items:center;padding:32px}
.sheet{width:100%;max-width:720px;max-height:100%;overflow:auto;background:#0b0b0b;
  border:1px solid var(--line);border-radius:20px;padding:22px 24px 26px}
.sheet-head{display:flex;align-items:flex-start;gap:14px;margin-bottom:18px}
/* Model families. A rule above each heading rather than a box around each
   group: the cards already have edges, and nesting borders inside borders is
   two frames for one idea. */
.fam{margin-bottom:22px}
.fam+.fam{border-top:1px solid var(--line);padding-top:18px}
.fam-head{display:flex;align-items:baseline;gap:10px;margin:0 2px 10px}
.fam-head .muted{margin-left:auto;font-size:12px}
.kv{display:grid;grid-template-columns:132px 1fr;gap:5px 14px;font-size:13px}
.kv dt{color:var(--mut)}
.kv dd{color:#e8e8e8;word-break:break-word}
.lb{position:fixed;inset:0;background:rgba(0,0,0,.93);z-index:70;display:grid;place-items:center;padding:34px}
.lb img,.lb video{max-width:100%;max-height:100%;object-fit:contain}
.lb .x{position:absolute;top:16px;right:20px;width:34px;height:34px;border:0;background:none;color:#bbb;padding:7px;cursor:pointer}

/* Datasets -------------------------------------------------------------- */
/* flex-column, not the default: a <button> centres its content vertically, so
   once one card in the row is taller than another the covers float in the
   middle of their cards with black above them. */
.ds-card{border:1px solid var(--line);background:rgba(255,255,255,.02);border-radius:14px;overflow:hidden;
  cursor:pointer;text-align:left;padding:0;color:inherit;font:inherit;
  display:flex;flex-direction:column;align-items:stretch}
.ds-card:hover{border-color:rgba(255,255,255,.24)}
.ds-cover{aspect-ratio:4/3;background:rgba(255,255,255,.045);display:block;width:100%;
  flex:none;object-fit:contain}
.ds-cover.empty{display:grid;place-items:center;color:var(--dim);font-size:22px}
.ds-meta{padding:11px 13px}
.ds-meta b{font-size:13px;font-weight:600}
.tiles{display:grid;gap:10px}
.tile{border:1px solid var(--line);border-radius:12px;overflow:hidden;background:rgba(255,255,255,.02);
  display:flex;flex-direction:column}
.tile.sel{border-color:rgba(255,255,255,.5)}
/* A true square cell with the whole image inside it, Bridge-style — you cannot
   judge a crop you cannot see. The image is absolutely positioned: as a normal
   flow child with height:100% it establishes the container's height itself, so
   aspect-ratio never applies and the cell silently takes the image's ratio.
   The cell is also lifted off the page background, because letterbox bars the
   same colour as the page read as a cropped photo rather than a contained one. */
.tile .ph{position:relative;background:rgba(255,255,255,.045);aspect-ratio:1}
.tile .ph img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;display:block;cursor:zoom-in}
.tile .dim{position:absolute;left:6px;bottom:6px;font-size:10px;color:#ddd;background:rgba(0,0,0,.62);
  padding:2px 6px;border-radius:5px;pointer-events:none}
.tile .rm{position:absolute;top:6px;right:6px;width:24px;height:24px;border:0;border-radius:50%;
  background:rgba(0,0,0,.62);color:#eee;cursor:pointer;font-size:13px;line-height:1;opacity:0;transition:opacity .12s}
.tile:hover .rm{opacity:1}
.tile textarea{border:0;border-top:1px solid var(--line);border-radius:0;background:none;resize:none;
  font-size:12px;padding:8px 9px;min-height:66px;line-height:1.45}
.tile textarea.dirty{background:rgba(56,189,248,.08)}
.tile.thin textarea{border-top-color:rgba(245,158,11,.5)}
.tile.notrig textarea{border-top-color:rgba(239,68,68,.55)}
/* Insight panel: bars are the readout, numbers confirm them. */
.ins{position:sticky;top:0}
.ins .stat{display:flex;align-items:baseline;gap:8px;margin-bottom:8px}
.ins .stat b{font-size:19px;font-weight:600;letter-spacing:-.01em}
.ins .stat span{font-size:12px;color:var(--mut)}
.meter{height:5px;border-radius:3px;background:rgba(255,255,255,.09);overflow:hidden;margin:2px 0 14px}
.meter i{display:block;height:100%;background:#34d399}
.meter i.warn{background:#f59e0b}
.meter i.bad{background:#ef4444}
.ph-row{display:flex;align-items:center;gap:8px;margin-bottom:3px;cursor:pointer;border:0;background:none;
  padding:2px 0;width:100%;color:inherit;font:inherit;text-align:left}
.ph-row:hover .ph-bar{outline:1px solid rgba(255,255,255,.3)}
.ph-bar{position:relative;flex:1;min-width:0;height:20px;border-radius:5px;background:rgba(255,255,255,.06);overflow:hidden}
/* Proportional fill only. An earlier version turned these red above 60% share,
   which read as "this is wrong" — but a phrase recurring is information about
   the set, not an error. A feature that is present and *not* captioned is the
   actual hazard, since it has nowhere to attach except the trigger. Red is
   reserved for trigger coverage, where a gap really is a defect. */
.ph-bar i{position:absolute;inset:0;width:var(--w);background:rgba(255,255,255,.11)}
.ph-bar i.hot{background:rgba(255,255,255,.2)}
/* Only the two human-error rows are coloured, because only they are defects. */
.ph-bar i.bad{background:rgba(239,68,68,.3)}
.ph-bar i.warn{background:rgba(245,158,11,.28)}
.ph-bar span{position:absolute;left:7px;top:0;line-height:20px;font-size:11px;color:#e8e8e8;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;right:7px}
.ph-n{font-size:11px;color:var(--mut);width:26px;text-align:right;flex:none}
.drop{border:1px dashed rgba(255,255,255,.2);border-radius:16px;text-align:center;cursor:pointer}
.drop.hot{border-color:rgba(255,255,255,.45);background:rgba(255,255,255,.05)}
.drop>span{display:block;padding:22px 12px;color:var(--dim);font-size:12px}
.drop img{display:block;width:100%;max-height:150px;object-fit:contain;border-radius:15px}
/* The strip's version: a 36px square that is a dashed outline when empty and
   the frame itself once filled. The thumbnail is the label — a filled first
   tile says "image-to-video" more directly than the words do. */
.drop.mini{width:36px;height:36px;flex:none;padding:0;border-radius:10px;overflow:hidden;
  background:rgba(255,255,255,.03);display:grid;place-items:center;color:var(--dim)}
.drop.mini:hover{border-color:rgba(255,255,255,.4);color:var(--fg)}
.drop.mini>span{padding:0;display:grid;place-items:center;width:16px;height:16px}
.drop.mini img{width:100%;height:100%;max-height:none;object-fit:cover;border-radius:0}
.drop.mini.set{border-style:solid;border-color:rgba(255,255,255,.28)}
.pad{padding:26px}

/* Reference chips. Numbered, because the number is the <Picture n> the prompt
   refers to — it is data, not decoration. */
.ref{position:relative;width:64px;height:64px;border-radius:11px;overflow:hidden;border:1px solid var(--line);
  background:rgba(255,255,255,.04)}
.ref img,.ref video{width:100%;height:100%;object-fit:cover;display:block}
.ref b{position:absolute;left:4px;bottom:3px;font-size:10px;font-weight:600;color:#fff;
  background:rgba(0,0,0,.7);padding:1px 5px;border-radius:4px}
.ref button{position:absolute;top:3px;right:3px;width:19px;height:19px;border:0;border-radius:50%;
  background:rgba(0,0,0,.66);color:#eee;font-size:11px;line-height:1;cursor:pointer}
/* Stack rows wrap to two lines rather than compressing. The rail is 384px and
   a LoRA path squeezed into 110px of select is a control you cannot read. */
.stack-row,.region{margin-bottom:12px}
.stack-row .nums{display:flex;gap:6px;margin-top:6px}
.num{width:62px;text-align:center;padding:8px 4px}
.stack-row button,.region button{padding:8px 10px;flex:none}
.region .cells{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:6px;margin-top:6px}
.region .cells input{padding:8px 2px;text-align:center;font-size:12px}
.wrap{display:flex;flex-wrap:wrap;gap:8px;align-items:center}

/* Train ----------------------------------------------------------------- */
.hold{padding:24px 28px 72px}
.grid2{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}

@media(max-width:1180px){:root{--drawer:284px}}
@media(max-width:980px){
  body{overflow:auto;height:auto}
  .views{position:static}
  .view{position:static;flex-direction:column}
  .canvas{overflow:visible}
  .console{max-height:none;padding:13px 16px 15px}
  .drawer{flex:none;border-left:0;border-top:1px solid rgba(255,255,255,.07)}
  .drawer-in{width:auto}
  .drawer .grid{grid-template-columns:repeat(auto-fill,minmax(200px,1fr))}
  .grid2{grid-template-columns:1fr}
}
</style></head><body>

<header class="top">
  <!-- The wordmark is the way home, which is the only reason Generate needs no
       nav item: you are already there, or you are one click from it. -->
  <button class="brand" id="go-home">Visionary</button>
  <!-- Train's two halves. In Generate this slot is empty — Generate's own
       switch lives down in the composer, beside the prompt it applies to. -->
  <div class="switch hide" id="train-tabs">
    <button data-tab="run" class="on">Training</button>
    <button data-tab="data">Datasets</button>
  </div>
  <span class="grow"></span>
  <button class="door" id="door"></button>
  <span class="sep"></span>
  <button class="ico on" id="t-drawer" title="Recent work"></button>
  <button class="ico" id="t-settings" title="Settings"></button>
</header>

<div class="views">

<!-- ============================ GENERATE ============================ -->
<!-- Canvas first, console under it. Image and video are one place: which one
     you get is a property of what you are making, not an address you navigate
     to, so it sits inside the prompt field rather than in the chrome. -->
<div class="view studio" id="v-generate">
 <div class="stage">
  <div class="canvas" id="canvas">
    <!-- No copy. An empty frame above a focused prompt field is already the
         whole instruction, and a sentence telling you to type is a sentence
         that will be read on every visit forever to be useful once. -->
    <div id="canvas-empty" class="blank"><div class="glyph" id="canvas-glyph"></div></div>
    <div id="gen-out" class="shots hide"></div>
    <p class="muted" id="gen-meta" style="margin:12px 2px"></p>
    <div id="vid-out" class="hide"></div>
    <p class="muted" id="vid-meta" style="margin:12px 2px"></p>
  </div>

  <div class="console">
    <!-- One prompt for both. It is the same sentence either way, and losing it
         to a mode switch is the fastest way to make two things that should
         feel like one feel like two. -->
    <div id="gen-err"></div>
    <div id="vid-err"></div>
    <div class="crow">
      <div class="field">
        <textarea id="prompt" rows="2" placeholder="Describe an image…"></textarea>
        <div class="bar2">
          <div class="kinds" id="kinds">
            <button data-kind="image" class="on">Image</button>
            <button data-kind="video">Video</button>
          </div>
        </div>
      </div>
      <button class="b" id="go-gen">Generate</button>
      <button class="b hide" id="go-vid">Generate</button>
    </div>
    <div id="gen-prog" class="hide" style="margin-top:9px"><div class="bar"><i style="width:0%"></i></div><p class="muted" style="margin-top:6px"></p></div>
    <div id="vid-prog" class="hide" style="margin-top:9px"><div class="bar"><i style="width:0%"></i></div><p class="muted" style="margin-top:6px"></p></div>
    <p class="muted warn" id="gen-note" style="margin:8px 2px 0"></p>
    <p class="muted warn" id="vid-note" style="margin:8px 2px 0"></p>

    <!-- IMAGE -->
    <div id="c-image">
      <div class="opts">
        <div class="opt"><select id="g-model"></select></div>
        <div class="opt"><select id="g-aspect">
          <option value="1024x1024">1:1</option><option value="1152x896">4:3</option>
          <option value="1216x832">3:2</option><option value="1344x768">16:9</option>
          <option value="832x1216">2:3</option><option value="768x1344">9:16</option>
        </select></div>
        <!-- A bare "1" and a bare number box say nothing on their own, so these
             two are the only controls in the strip that get an icon. -->
        <div class="opt" data-ico="copies"><select id="g-n"><option>1</option><option>2</option><option>3</option><option>4</option></select></div>
        <div class="opt" data-ico="dice"><input id="g-seed" placeholder="random" inputmode="numeric"></div>
        <div class="opt"><select id="g-gpu"></select></div>
        <span class="vr"></span>
        <!-- Rows are added by hand; order is the order they patch in. -->
        <button class="s" id="add-lora" style="height:36px;padding:0 13px">+ LoRA</button>
        <span class="grow"></span>
        <span class="muted" id="gen-model-line"></span>
        <button class="opt ib" id="toggle-adv" data-ico="sliders" title="Advanced"></button>
      </div>
      <div id="lora-stack" style="margin-top:9px"></div>

      <div id="gen-adv" class="hide adv">
        <textarea id="g-neg" rows="2" placeholder="Negative prompt"></textarea>
        <div class="opts">
          <div class="opt"><select id="g-sampler"></select></div>
          <div class="opt"><select id="g-scheduler"></select></div>
          <div class="opt" data-ico="steps"><input id="g-steps" placeholder="auto" inputmode="numeric"></div>
          <div class="opt" data-ico="cfg"><input id="g-cfg" placeholder="CFG" inputmode="decimal"></div>
          <div class="opt" data-ico="shift"><input id="g-shift" placeholder="shift 1.15" inputmode="decimal"></div>
          <span class="vr"></span>
          <label class="row" style="gap:7px;margin:0;color:#ddd;font-size:13px">
            <input type="checkbox" id="g-regional" style="width:auto"> Regional</label>
          <select id="g-region-dir" class="hide" style="width:auto;height:36px;padding:0 10px"><option value="columns">Columns</option><option value="rows">Rows</option></select>
          <button class="s hide" id="add-region" style="height:36px;padding:0 13px">+ Region</button>
        </div>
        <div id="region-stack" class="hide" style="margin-top:9px"></div>
      </div>
    </div>

    <!-- VIDEO -->
    <div id="c-video" class="hide">
      <!-- On H3 the shared prompt describes picture and sound together, which
           it denoises from the same sequence — so what you do not describe
           hearing, it invents. The placeholder says so for those models and
           not for the silent ones, rather than asking every model for audio
           only some of them render. -->
      <div class="opts">
        <!-- The model comes first because it decides what the rest of this
             strip even offers: LoRAs, CFG and a negative prompt on Wan;
             references and a soundtrack on H3. -->
        <div class="opt"><select id="v-model"></select></div>
        <div class="opt"><select id="v-aspect">
          <option value="21:9">21:9</option><option value="16:9" selected>16:9</option>
          <option value="4:3">4:3</option><option value="1:1">1:1</option>
          <option value="3:4">3:4</option><option value="9:16">9:16</option>
        </select></div>
        <div class="opt"><select id="v-tier"></select></div>
        <div class="opt"><select id="v-seconds"></select></div>
        <div class="opt" data-ico="dice"><input id="v-seed" placeholder="random" inputmode="numeric"></div>
        <div class="opt"><select id="v-gpu"></select></div>
        <span class="vr"></span>
        <!-- Keyframes are what make this image-to-video; with neither, the
             same checkpoint runs text-to-video. Two tiles side by side, in the
             order they play — a filled first tile is the whole explanation. -->
        <button class="drop mini" id="v-drop-first" title="First frame">
          <img id="v-thumb-first" class="hide" alt=""><span id="v-hint-first"></span>
        </button>
        <button class="drop mini hide" id="v-drop-last" title="Last frame">
          <img id="v-thumb-last" class="hide" alt=""><span id="v-hint-last"></span>
        </button>
        <!-- Wan only. Same idea as the image side's stack, plus the one thing
             the A14B pair forces: which expert a row patches. -->
        <button class="s hide" id="v-add-lora" style="height:36px;padding:0 13px">+ LoRA</button>
        <span class="grow"></span>
        <span class="muted" id="v-model-line"></span>
        <button class="opt ib" id="v-toggle-adv" data-ico="sliders" title="Advanced"></button>
      </div>
      <div id="v-lora-stack" style="margin-top:9px"></div>

      <!-- References are the other way to condition a clip, and they exclude
           keyframes because they load a different transformer. H3 only — Wan
           has no reference checkpoint, so the row is not there at all. The
           chips carry their own <Picture n> labels, which is the part the
           prompt actually refers to and the only part worth spelling out. -->
      <div class="opts" id="v-ref-sec">
        <span class="wrap" id="v-refs"></span>
        <button class="drop mini" id="v-add-ref" title="Add image reference"></button>
        <button class="drop mini" id="v-add-vid" title="Add video reference"></button>
        <div class="opt" id="v-ref-size-wrap"><select id="v-ref-size">
          <option value="match">match canvas</option><option value="max">max detail</option>
        </select></div>
        <span class="muted" id="v-ref-max" hidden>9</span><span class="muted" id="v-vid-max" hidden>3</span>
      </div>

      <div id="vid-adv" class="hide adv">
        <!-- Shown only for the models that read them. H3 is guidance-distilled,
             so on H3 a negative prompt and a CFG dial would be controls the
             model never looks at. -->
        <textarea id="v-neg" rows="2" class="hide" placeholder="Negative prompt"></textarea>
        <div class="opts">
          <div class="opt"><select id="v-sampler"></select></div>
          <div class="opt"><select id="v-scheduler"></select></div>
          <div class="opt" data-ico="steps"><input id="v-steps" inputmode="numeric"></div>
          <div class="opt hide" id="v-cfg-wrap" data-ico="cfg"><input id="v-cfg" placeholder="CFG" inputmode="decimal"></div>
          <div class="opt hide" id="v-shift-wrap" data-ico="shift"><input id="v-shift" placeholder="shift" inputmode="decimal"></div>
          <!-- Only the A14B pair has a handover to place. -->
          <div class="opt hide" id="v-switch-wrap" data-ico="handover"><input id="v-switch" placeholder="switch at" inputmode="numeric"></div>
        </div>
      </div>
    </div>
  </div>
 </div>

  <!-- The gallery lives beside the canvas, because the thing you made an hour
       ago is raw material for the thing you are making now — not a destination
       you leave the studio to visit. -->
  <aside class="drawer" id="drawer">
    <div class="drawer-in">
      <div class="drawer-head">
        <span class="grow"></span>
        <button class="ico" id="gal-expand" title="Open gallery"></button>
      </div>
      <div id="drawer-grid" class="grid"></div>
      <p class="muted" id="drawer-empty" style="margin-top:6px"></p>
    </div>
  </aside>

  <!-- Full gallery: the same drawer, given the whole room. Exits back to the
       canvas rather than to a nav item, because that is where you came from. -->
  <div id="gal-full" class="hide" style="position:absolute;inset:0;background:var(--bg);z-index:20;overflow:auto;padding:18px 28px 72px">
    <div class="row" style="gap:10px;margin-bottom:18px;flex-wrap:wrap">
      <button class="ico" id="gal-back" title="Back to canvas"></button>
      <span class="grow"></span>
      <button class="pill on" data-filter="all">All</button>
      <button class="pill" data-filter="image">Images</button>
      <button class="pill" data-filter="video">Video</button>
      <button class="ico" id="gal-refresh" title="Refresh"></button>
    </div>
    <div id="gal-grid" class="grid"></div>
    <p class="muted" id="gal-empty"></p>
  </div>
</div>

<!-- ============================== TRAIN ============================== -->
<div class="view scroll hide" id="v-train">
  <!-- TRAINING -->
  <div class="hold" id="t-run">
    <div id="train-err"></div>
    <div id="step-build" style="max-width:860px">
      <div class="card">
        <label>Dataset</label>
        <select id="t-dataset"></select>
        <p class="muted" id="t-dsinfo" style="margin-top:8px"></p>
      </div>
      <div id="dataset" class="hide">
        <!-- Name and trigger sit here, next to the action that uses them. -->
        <div class="card grid2">
          <div><label>LoRA name</label><input id="lname" placeholder="my_style" spellcheck="false"></div>
          <div><label>Trigger word</label><input id="ltrig" placeholder="ohwx_style" spellcheck="false"></div>
        </div>
        <details class="card"><summary class="muted" style="cursor:pointer">Advanced · Krea 2 RAW, rank 32, bf16</summary>
          <div class="grid2" style="margin-top:14px">
            <div><label>Rank (dim)</label><input id="a-dim" type="number" value="32"></div>
            <div><label>Alpha</label><input id="a-alpha" type="number" value="32"></div>
            <div><label>Epochs</label><input id="a-epochs" type="number" value="30"></div>
            <div><label>Learning rate</label><input id="a-lr" type="number" step="0.00001" value="0.0001"></div>
            <div><label>Resolution</label><input id="a-res" type="number" step="64" value="1024"></div>
            <div><label>Repeats</label><input id="a-rep" type="number" value="1"></div>
            <div><label>Batch size</label><input id="a-bs" type="number" value="1"></div>
            <div><label>Seed</label><input id="a-seed" type="number" value="42"></div>
          </div>
        </details>
        <button class="b" id="go-train" disabled style="width:100%">Start training</button>
        <p class="muted" id="train-hint" style="text-align:center;margin-top:8px;height:16px"></p>
      </div>
    </div>

    <div id="step-run" class="hide" style="max-width:860px">
      <div class="card">
        <div class="row"><b id="run-phase" class="grow">Starting…</b><span class="muted" id="run-pct"></span></div>
        <div class="bar"><i id="run-bar" style="width:0%"></i></div>
        <p class="muted" id="run-meta" style="margin-top:9px"></p>
        <div class="row" style="margin-top:14px"><button class="s" id="do-stop">Stop &amp; keep checkpoints</button></div>
      </div>
      <div class="card hide" id="run-done"></div>
    </div>
  </div>

  <!-- DATASETS -->
  <div class="hold hide" id="t-data">
    <!-- List. Replaced wholesale by the editor rather than stacked, so there is
         never a scroll position to lose track of. -->
    <div id="ds-index">
      <div id="ds-err"></div>
      <div class="row" style="gap:8px;margin-bottom:18px;max-width:520px">
        <input id="ds-new" class="grow" placeholder="New dataset name" spellcheck="false">
        <button class="s" id="ds-create">Create</button>
      </div>
      <div class="grid" id="ds-list"></div>
      <p class="muted" id="ds-empty"></p>
    </div>

    <!-- Editor -->
    <div id="ds-editor" class="hide">
      <div class="row" style="margin-bottom:14px">
        <button class="ico" id="ds-back" title="All datasets"></button>
        <b id="ds-title" style="font-size:15px"></b>
        <span class="grow"></span>
        <span class="muted" id="ds-count"></span>
      </div>
      <div id="ds-edit-err"></div>

      <div class="grid2" style="grid-template-columns:minmax(0,1fr) 320px;align-items:start;gap:20px">
        <div style="min-width:0">
          <!-- A short bar, not a hero. Uploading happens once per dataset and
               the contact sheet below it is the thing you came to look at. -->
          <div class="drop" id="drop">
            <div class="row" style="justify-content:center;gap:9px;padding:15px">
              <span id="drop-title" class="muted">Images or a .zip</span>
              <span class="muted" id="drop-sub"></span>
            </div>
            <input type="file" id="files" multiple accept="image/*,.zip,.txt" class="hide">
            <div id="up-prog" class="hide" style="padding:0 15px 13px"><div class="bar"><i style="width:0%"></i></div></div>
          </div>
          <div class="row" style="margin:14px 2px 10px;gap:8px;flex-wrap:wrap">
            <button class="s" id="f-all" title="Show every image">All</button>
            <button class="s" id="f-uncap" title="Only images with no caption">Uncaptioned</button>
            <button class="s" id="f-notrig" title="Only captions missing the trigger word">No trigger</button>
            <span class="grow"></span>
            <button class="s" id="dens-down" title="Smaller tiles">−</button>
            <button class="s" id="dens-up" title="Larger tiles">+</button>
          </div>
          <div class="tiles" id="tiles"></div>
        </div>

        <div class="ins">
          <div class="card">
            <label style="margin-bottom:10px">Trigger word</label>
            <div class="row" style="gap:8px;margin-bottom:14px">
              <input id="ds-trig" class="grow" placeholder="ohwx_style" spellcheck="false">
              <button class="s" id="do-prepend" title="Put the trigger word at the front of every caption that lacks it">Fix</button>
            </div>
            <div id="ins-body"></div>
          </div>
          <div class="card">
            <label>Auto-caption</label>
            <div class="f2" style="margin-bottom:10px">
              <select id="cap-style"><option value="descriptive">Descriptive</option><option value="casual">Casual</option></select>
              <select id="cap-len"><option value="short">Short</option><option value="medium" selected>Medium</option><option value="long">Long</option></select>
            </div>
            <label class="row" style="gap:7px;color:#ddd;margin-bottom:12px"><input type="checkbox" id="cap-over" style="width:auto"> Replace existing</label>
            <button class="s" id="do-caption" style="width:100%">Caption</button>
            <div id="cap-prog" class="hide"><div class="bar"><i style="width:0%"></i></div><p class="muted" style="margin-top:7px"></p></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

</div>

<!-- ============================ SETTINGS ============================= -->
<!-- Models are plumbing: chosen once, then never thought about again. That is
     what a gear is for, and it is why this is not a place you can be lost in. -->
<div id="settings" class="scrim hide">
  <div class="sheet">
    <div class="sheet-head">
      <h1 class="grow">Settings</h1>
      <button class="ico" id="settings-x"></button>
    </div>
    <div class="card">
      <label>HuggingFace token</label>
      <div class="row">
        <input id="tok" type="password" class="grow" placeholder="hf_…" autocomplete="off">
        <button class="s" id="tok-save">Save</button>
        <button class="b" id="dl-all" style="padding:9px 16px;font-size:13px">Download missing</button>
      </div>
      <p class="muted" style="margin-top:8px">
        Needed for Krea 2 RAW and Turbo, which are gated. Accept the licence at
        huggingface.co/krea/Krea-2-Raw with the same account. <span id="tok-state"></span>
      </p>
      <div id="dl-all-prog" class="hide"><div class="bar"><i style="width:0%"></i></div><p class="muted" style="margin-top:7px"></p></div>
    </div>
    <div id="models"></div>
  </div>
</div>

<script>
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const api=async(p,o)=>{const r=await fetch(p,o);return r.json()};
const post=(p,b)=>api(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;');
const errInto=(sel,msg)=>{ $(sel).innerHTML = msg ? '<div class="err-box">'+esc(msg)+'</div>' : ''; };
let poll=null;

// Icons are inline because a sprite sheet or an icon font would be a second
// asset for a single-file app to serve, and there are six of them.
const ICON={
  photo:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4.5" width="18" height="15" rx="3.5"/><circle cx="8.6" cy="10" r="1.5"/><path d="M3.6 17.4l4.2-4.2a1.9 1.9 0 0 1 2.7 0l3.3 3.3"/><path d="M13.9 14.6l1.6-1.6a1.9 1.9 0 0 1 2.7 0l2.2 2.2"/></svg>',
  play:'<svg viewBox="0 0 24 24" fill="currentColor"><path d="M9 5.6a1.8 1.8 0 0 0-2.8 1.5v9.8A1.8 1.8 0 0 0 9 18.4l7.9-4.9a1.8 1.8 0 0 0 0-3z"/></svg>',
  download:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4v10"/><path d="M8 11l4 4 4-4"/><path d="M5 19h14"/></svg>',
  close:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg>',
  more:'<svg viewBox="0 0 24 24" fill="currentColor"><circle cx="5" cy="12" r="1.7"/><circle cx="12" cy="12" r="1.7"/><circle cx="19" cy="12" r="1.7"/></svg>',
  gear:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3.2"/><path d="M19.2 14.4a1.5 1.5 0 0 0 .3 1.65l.06.06a1.8 1.8 0 1 1-2.55 2.55l-.06-.06a1.5 1.5 0 0 0-1.65-.3 1.5 1.5 0 0 0-.9 1.37V20a1.8 1.8 0 1 1-3.6 0v-.1a1.5 1.5 0 0 0-.98-1.37 1.5 1.5 0 0 0-1.65.3l-.06.06A1.8 1.8 0 1 1 5.55 16.3l.06-.06a1.5 1.5 0 0 0 .3-1.65 1.5 1.5 0 0 0-1.37-.9H4a1.8 1.8 0 1 1 0-3.6h.1a1.5 1.5 0 0 0 1.37-.98 1.5 1.5 0 0 0-.3-1.65l-.06-.06A1.8 1.8 0 1 1 7.66 4.85l.06.06a1.5 1.5 0 0 0 1.65.3H9.5a1.5 1.5 0 0 0 .9-1.37V4a1.8 1.8 0 1 1 3.6 0v.1a1.5 1.5 0 0 0 .9 1.37 1.5 1.5 0 0 0 1.65-.3l.06-.06a1.8 1.8 0 1 1 2.55 2.55l-.06.06a1.5 1.5 0 0 0-.3 1.65v.08a1.5 1.5 0 0 0 1.37.9H20a1.8 1.8 0 1 1 0 3.6h-.1a1.5 1.5 0 0 0-1.37.9z"/></svg>',
  panel:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4.5" width="18" height="15" rx="3"/><path d="M15 4.5v15"/></svg>',
  train:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"><path d="M4 19V11"/><path d="M10 19V5"/><path d="M16 19v-5"/><path d="M22 19V8"/></svg>',
  back:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 5l-7 7 7 7"/></svg>',
  // Strip icons. Only for the controls whose own value is not self-describing:
  // a bare "2" and a bare number box could be anything.
  copies:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"><rect x="8.5" y="3.5" width="12" height="12" rx="2.5"/><path d="M15.5 20.5h-10a2 2 0 0 1-2-2v-10"/></svg>',
  dice:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"><rect x="3.5" y="3.5" width="17" height="17" rx="4"/><circle cx="8.5" cy="8.5" r="1.3" fill="currentColor" stroke="none"/><circle cx="15.5" cy="15.5" r="1.3" fill="currentColor" stroke="none"/><circle cx="12" cy="12" r="1.3" fill="currentColor" stroke="none"/></svg>',
  sliders:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M4 7h10M18 7h2M4 17h4M12 17h8"/><circle cx="16" cy="7" r="2"/><circle cx="10" cy="17" r="2"/></svg>',
  steps:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 20h4v-5h5v-5h5V5h4"/></svg>',
  cfg:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><path d="M12 3v2M12 19v2M4.2 7.5l1.7 1M18.1 15.5l1.7 1M4.2 16.5l1.7-1M18.1 8.5l1.7-1"/><circle cx="12" cy="12" r="4"/></svg>',
  shift:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 16c4 0 5-8 9-8s5 8 9 8"/></svg>',
  handover:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9h7l3 6h8"/><path d="M17 11l3-2-3-2"/></svg>',
  refresh:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M20 11a8 8 0 1 0-.7 4.3"/><path d="M20 4.5V11h-6.5"/></svg>',
  expand:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 4.5H4.5v5"/><path d="M14.5 19.5h5v-5"/><path d="M4.5 4.5l6 6"/><path d="M19.5 19.5l-6-6"/></svg>',
  first:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><rect x="3.5" y="5.5" width="17" height="13" rx="2.5"/><rect x="3.5" y="5.5" width="5" height="13" rx="2.5" fill="currentColor" stroke="none" opacity=".85"/></svg>',
  last:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><rect x="3.5" y="5.5" width="17" height="13" rx="2.5"/><rect x="15.5" y="5.5" width="5" height="13" rx="2.5" fill="currentColor" stroke="none" opacity=".85"/></svg>',
  plus:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"><path d="M12 6v12M6 12h12"/></svg>',
  film:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><rect x="3.5" y="5.5" width="17" height="13" rx="2.5"/><path d="M8 5.5v13M16 5.5v13"/></svg>',
};
// Every control that asked for an icon instead of a label gets it here, so the
// markup names the idea ("dice") and only this table knows the path data.
$$('[data-ico]').forEach(el=>el.insertAdjacentHTML('afterbegin',ICON[el.dataset.ico]));
$('#v-add-ref').innerHTML='<span>'+ICON.photo+'</span>';
$('#v-add-vid').innerHTML='<span>'+ICON.film+'</span>';
$('#gal-back').innerHTML=ICON.back;
$('#ds-back').innerHTML=ICON.back;
$('#gal-refresh').innerHTML=ICON.refresh;
$('#gal-expand').innerHTML=ICON.expand;
$('#t-settings').innerHTML=ICON.gear;
$('#t-drawer').innerHTML=ICON.panel;
$('#settings-x').innerHTML=ICON.close;

// ==================== SHELL ====================
// Generate is the page, not a destination — so it has no nav item, and the
// wordmark is how you get back to it. Train is the one door, on the right.
let mode='generate', kind='image', trainPct=null;

function setMode(m){
  mode=m;
  $('#v-generate').classList.toggle('hide',m!=='generate');
  $('#v-train').classList.toggle('hide',m!=='train');
  $('#train-tabs').classList.toggle('hide',m!=='train');
  // The drawer toggle is a Generate control; it has nothing to say in Train.
  $('#t-drawer').classList.toggle('hide',m!=='generate');
  drawDoor();
  if(m==='train') loadDatasets();
  if(m==='generate') loadGallery();
}

// The door names where it goes, never where you are — so there is one button
// instead of two, and no moment where both look equally selectable. In
// Generate it doubles as the training readout: a run outlives the visit that
// started it, and the control that takes you back is the honest place to say
// how far along it is.
function drawDoor(){
  const d=$('#door');
  if(mode==='train'){ d.className='door'; d.innerHTML=ICON.back+'Generate'; return }
  if(trainPct==null){ d.className='door'; d.innerHTML=ICON.train+'Train'; return }
  const c=2*Math.PI*6;
  d.className='door live';
  d.innerHTML='<svg class="ring" viewBox="0 0 16 16">'
    +'<circle class="bg" cx="8" cy="8" r="6"/>'
    +`<circle class="fg" cx="8" cy="8" r="6" stroke-dasharray="${c.toFixed(2)}" `
    +`stroke-dashoffset="${(c*(1-trainPct/100)).toFixed(2)}"/></svg>Training ${trainPct}%`;
}
$('#door').onclick=()=>setMode(mode==='train'?'generate':'train');
$('#go-home').onclick=()=>{ closeGallery(); setMode('generate') };

// Image or video is a property of the sentence, not an address. The prompt,
// the canvas and the gallery are shared; only the options below the field
// change, so switching mid-thought costs nothing you typed.
function setKind(k){
  kind=k;
  $$('#kinds button').forEach(b=>b.classList.toggle('on',b.dataset.kind===k));
  $('#c-image').classList.toggle('hide',k!=='image');
  $('#c-video').classList.toggle('hide',k!=='video');
  $('#gen-note').classList.toggle('hide',k!=='image');
  $('#vid-note').classList.toggle('hide',k!=='video');
  $('#gen-err').classList.toggle('hide',k!=='image');
  $('#vid-err').classList.toggle('hide',k!=='video');
  $('#go-gen').classList.toggle('hide',k!=='image');
  $('#go-vid').classList.toggle('hide',k!=='video');
  syncPromptHint();
  syncCanvasView();
}
$$('#kinds button').forEach(b=>{
  b.insertAdjacentHTML('afterbegin',b.dataset.kind==='image'?ICON.photo:ICON.play);
  b.onclick=()=>setKind(b.dataset.kind);
});

// One placeholder, three states: it has to name the kind, and on the H3
// checkpoints it also has to ask for the soundtrack, which is denoised from
// the same sequence and invented when you leave it out.
function syncPromptHint(){
  if(kind==='image'){ $('#prompt').placeholder='Describe an image…'; return }
  const sup=(typeof videoModel==='function'&&videoModel()||{supports:{}}).supports;
  $('#prompt').placeholder = sup.audio
    ? 'Describe the shot, the motion — and the audio: dialogue, effects, music…'
    : 'Describe the shot and the motion…';
}

function syncCanvasView(){
  const img=kind==='image';
  const n=(img?$('#gen-out'):$('#vid-out')).children.length;
  $('#gen-out').classList.toggle('hide',!img||!$('#gen-out').children.length);
  $('#vid-out').classList.toggle('hide',img||!$('#vid-out').children.length);
  $('#gen-meta').classList.toggle('hide',!img);
  $('#vid-meta').classList.toggle('hide',img);
  $('#canvas-empty').classList.toggle('hide',!!n);
  $('#canvas-glyph').innerHTML=img?ICON.photo:ICON.play;
}

$('#t-drawer').onclick=()=>{
  const off=$('#v-generate').classList.toggle('nodrawer');
  $('#t-drawer').classList.toggle('on',!off);
};
$('#t-settings').onclick=()=>{ $('#settings').classList.remove('hide'); loadState(); };
$('#settings-x').onclick=()=>$('#settings').classList.add('hide');
$('#settings').onclick=e=>{ if(e.target.id==='settings') $('#settings').classList.add('hide') };

$$('#train-tabs button').forEach(b=>b.onclick=()=>{
  $$('#train-tabs button').forEach(x=>x.classList.toggle('on',x===b));
  $('#t-run').classList.toggle('hide',b.dataset.tab!=='run');
  $('#t-data').classList.toggle('hide',b.dataset.tab!=='data');
  loadDatasets();
});

document.addEventListener('keydown',e=>{
  if(e.key!=='Escape') return;
  if(!$('#settings').classList.contains('hide')) $('#settings').classList.add('hide');
  closeMenu();
});

// ---------- overflow menu ----------
// One floating menu, moved and refilled. Rendering a menu inside every card
// would put a hundred hidden subtrees in a grid that is already a hundred
// images, and only one of them can ever be open.
let menuEl=null;
function closeMenu(){ if(menuEl){ menuEl.remove(); menuEl=null } }
document.addEventListener('mousedown',e=>{ if(menuEl&&!menuEl.contains(e.target)) closeMenu() },true);
window.addEventListener('resize',closeMenu);
document.addEventListener('scroll',closeMenu,true);

function openMenu(btn,items){
  closeMenu();
  const m=document.createElement('div'); m.className='menu';
  m.innerHTML=items.map((it,i)=>it.sep?'<hr>':
    `<button data-i="${i}"${it.danger?' class="danger"':''}>${esc(it.label)}</button>`).join('');
  document.body.appendChild(m);
  const r=btn.getBoundingClientRect(), w=m.offsetWidth, h=m.offsetHeight;
  m.style.left=Math.max(8,Math.min(r.right-w,innerWidth-w-8))+'px';
  m.style.top=(r.bottom+h+10>innerHeight ? r.top-h-6 : r.bottom+6)+'px';
  m.querySelectorAll('button').forEach(b=>b.onclick=()=>{ const it=items[+b.dataset.i]; closeMenu(); it.run() });
  menuEl=m;
}

// ---------- sheets ----------
function sheet(html){
  const el=document.createElement('div'); el.className='scrim';
  el.innerHTML=`<div class="sheet">${html}</div>`;
  const close=()=>{ el.remove(); document.removeEventListener('keydown',onKey) };
  const onKey=e=>{ if(e.key==='Escape') close() };
  el.onclick=e=>{ if(e.target===el) close() };
  document.addEventListener('keydown',onKey);
  document.body.appendChild(el);
  el.querySelectorAll('[data-close]').forEach(b=>b.onclick=close);
  return el;
}

function lightbox(src,video){
  const el=document.createElement('div'); el.className='lb';
  el.innerHTML=`<button class="x">${ICON.close}</button>`+
    (video?`<video src="${src}" controls autoplay loop playsinline></video>`:`<img src="${src}" alt="">`);
  const close=()=>{ el.remove(); document.removeEventListener('keydown',onKey) };
  const onKey=e=>{ if(e.key==='Escape') close() };
  el.onclick=e=>{ if(e.target===el||e.target.closest('.x')) close() };
  document.addEventListener('keydown',onKey);
  document.body.appendChild(el);
}

// ==================== MODELS (settings) ====================
async function loadState(){
  const s=await api('/api/state');
  $('#tok-state').innerHTML = s.hf_token_set?'<span class="ok">Token saved.</span>':'<span class="warn">No token saved.</span>';
  // Name the cost up front: how many are missing and how many GB that is.
  const miss=s.models.filter(m=>!m.present);
  const gb=miss.reduce((a,m)=>a+m.approx_gb,0);
  const dlAll=$('#dl-all');
  dlAll.disabled=!miss.length;
  dlAll.textContent=miss.length?`Download ${miss.length} missing · ${gb.toFixed(1)} GB`:'All models present';
  // Grouped by family, in catalogue order. Twenty-odd flat cards is a wall you
  // scroll rather than a list you read, and the groups are the unit you
  // actually decide in: you want the Wan stack or you do not.
  const fams=[];
  s.models.forEach(m=>{
    const g=fams.find(f=>f.name===m.family);
    (g||fams[fams.push({name:m.family,items:[]})-1]).items.push(m);
  });
  $('#models').innerHTML = fams.map(f=>{
    const left=f.items.filter(m=>!m.present);
    const size=left.reduce((a,m)=>a+m.approx_gb,0);
    return `
    <div class="fam">
      <div class="fam-head">
        <b>${esc(f.name)}</b>
        <span class="muted">${left.length?`${left.length} missing · ${size.toFixed(1)} GB`:'complete'}</span>
      </div>
      ${f.items.map(m=>`
      <div class="card row">
        <div class="grow">
          <b>${m.label}</b> <span class="muted">${m.note}</span>
          ${m.gated?'<span class="warn" style="font-size:12px"> · gated</span>':''}
          <div class="muted" style="margin-top:3px"><code>${m.repo_id}</code></div>
        </div>
        <div style="text-align:right">
          ${m.present
            ? `<span class="ok">✓ ${m.size_gb} GB</span>`
            : `<button class="s" data-dl="${m.key}">Download ${m.approx_gb} GB</button>`}
          <div class="muted dl-state" id="dl-${m.key}"></div>
        </div>
      </div>`).join('')}
    </div>`;
  }).join('');
  $$('[data-dl]').forEach(b=>b.onclick=()=>startDownload(b.dataset.dl,b));

  // Options for every LoRA row, existing and future. Rebuilt on each poll so a
  // freshly trained LoRA appears without a reload; current picks are kept.
  window.MAX_LORAS=s.max_loras||6;
  window.WAN_EXPERTS=s.wan_experts||['both','high','low'];
  loraOpts=s.loras.map(l=>l.files.map(f=>
    `<option value="${f.path}" data-t="${l.trigger_word||''}">${l.name} · ${f.name}</option>`).join('')).join('');
  ['#lora-stack','#v-lora-stack'].forEach(sel=>
    $$(sel+' [data-f=path]').forEach(el=>{const v=el.value; el.innerHTML=loraOpts; el.value=v;}));
  $('#add-lora').disabled=!s.loras.length;
  $('#v-add-lora').disabled=!s.loras.length;

  if(s.samplers&&!$('#g-sampler').options.length){
    $('#g-sampler').innerHTML=s.samplers.map(x=>`<option>${x}</option>`).join('');
    $('#g-scheduler').innerHTML=s.schedulers.map(x=>`<option>${x}</option>`).join('');
  }

  // Built once. Rebuilding on every poll would reset a card the user picked
  // between two polls, and the list itself only changes on redeploy.
  if(s.gpus&&!$('#g-gpu').options.length){
    wireGpu('#g-gpu', s.gpus.image);
    wireGpu('#v-gpu', s.gpus.video);
  }
  if(s.max_refs) $('#v-ref-max').textContent=s.max_refs;
  if(s.max_ref_videos) $('#v-vid-max').textContent=s.max_ref_videos;

  // The video composer builds itself from what the deployment says each model
  // takes, so a model that reads no CFG has no CFG box rather than a dead one.
  // Kept across polls: only the picker's own labels are rewritten, and
  // syncVideoModel() re-runs against whatever is still selected.
  vidModels=s.video_models||[];
  const vs=$('#v-model'), vprev=vs.value;
  vs.innerHTML=vidModels.map(m=>
    `<option value="${m.key}" ${m.ready?'':'disabled'}>${esc(m.label)}${m.ready?'':' — missing'}</option>`).join('');
  const vavail=vidModels.filter(m=>m.ready).map(m=>m.key);
  vs.value = vavail.includes(vprev) ? vprev : (vavail[0]||(vidModels[0]||{}).key||'');
  syncVideoModel();

  // Model picker reflects the volume rather than being hardcoded — otherwise it
  // claims both models are available even when neither is downloaded.
  const ms=$('#g-model'), prev=ms.value;
  const pick=s.models.filter(m=>m.key==='turbo'||m.key==='raw');
  ms.innerHTML=pick.map(m=>
    `<option value="${m.key}" ${m.present?'':'disabled'}>${m.label}${m.present?'':' — missing'}</option>`).join('');
  const avail=pick.filter(m=>m.present).map(m=>m.key);
  ms.value = avail.includes(prev) ? prev : (avail[0]||'');
  const missing=pick.filter(m=>!m.present).length===pick.length;
  $('#go-gen').disabled=missing;
  // The gear is the only route to the fix, so the note names it.
  $('#gen-note').textContent = missing
    ? 'No DiT on the volume — download Krea 2 Turbo under Settings.'
    : (s.models.find(m=>m.key==='vae')?.present && s.models.find(m=>m.key==='text_encoder')?.present
        ? '' : 'The VAE and text encoder are also required.');
  syncModelLine();
}
$('#tok-save').onclick=async()=>{
  const r=await post('/api/token',{hf_token:$('#tok').value});
  $('#tok').value=''; $('#tok-state').innerHTML=r.hf_token_set?'<span class="ok">Token saved.</span>':'<span class="warn">Cleared.</span>';
};
$('#dl-all').onclick=async()=>{
  const b=$('#dl-all'); b.disabled=true;
  const box=$('#dl-all-prog'), bar=box.querySelector('i'), msg=box.querySelector('p');
  box.classList.remove('hide'); bar.style.width='0%'; msg.textContent='Starting…';
  // Token rides along, so pasting a key and pressing Download is one action.
  const r=await post('/api/download-missing',{hf_token:$('#tok').value});
  $('#tok').value='';
  if(r.error){ msg.innerHTML='<span class="err">'+r.error+'</span>'; b.disabled=false; return }
  if(!r.job_id){ msg.textContent=r.note||'Nothing missing.'; b.disabled=false; loadState(); return }
  const t=setInterval(async()=>{
    const s=await api('/api/status/dl_all');
    bar.style.width=(s.percent||0)+'%';
    msg.textContent=s.phase||'Downloading…';
    if(s.status==='completed'){ clearInterval(t); bar.style.width='100%';
      msg.innerHTML='<span class="ok">All models downloaded.</span>'; b.disabled=false; loadState(); }
    else if(s.status==='failed'){ clearInterval(t);
      msg.innerHTML='<span class="err">'+(s.error||'Download failed')+'</span>'; b.disabled=false; loadState(); }
  },3000);
};
async function startDownload(key,btn){
  btn.disabled=true; const el=$('#dl-'+key); el.textContent='Starting…';
  await post('/api/download',{key});
  const t=setInterval(async()=>{
    const s=await api('/api/status/dl_'+key);
    if(s.status==='completed'){clearInterval(t);el.innerHTML='<span class="ok">Done</span>';loadState();}
    else if(s.status==='failed'){clearInterval(t);el.innerHTML='<span class="err">'+(s.error||'Failed')+'</span>';btn.disabled=false;}
    else el.textContent=s.phase||'Downloading…';
  },3000);
}

// ==================== DATASETS ====================
// A dataset is a named folder; the editor is a view onto it. Nothing here is
// tied to a training run, which is the whole point of the section.
let dsName=null, dsImages=[], dsInsight=null, dsFilter='all', dsDensity=2, capPoll=null;

async function loadDatasets(){
  const r=await api('/api/datasets');
  if(r.error){ errInto('#ds-err',r.error); return }
  const list=r.datasets||[];
  $('#ds-list').innerHTML=list.map(d=>`
    <button class="ds-card" data-open="${esc(d.name)}">
      ${d.cover
        ? `<img class="ds-cover" loading="lazy" src="/api/thumb/${encodeURIComponent(d.name)}/${encodeURIComponent(d.cover)}" alt="">`
        : '<div class="ds-cover empty">▤</div>'}
      <div class="ds-meta">
        <b>${esc(d.name)}</b>
        <div class="muted" style="margin-top:3px;font-size:12px">
          ${d.count} image${d.count===1?'':'s'}${d.uncaptioned?` · ${d.uncaptioned} uncaptioned`:''}
        </div>
      </div>
    </button>`).join('');
  $('#ds-empty').textContent = list.length ? '' : 'No datasets yet. Name one above to start.';
  $$('#ds-list [data-open]').forEach(b=>b.onclick=()=>openDataset(b.dataset.open));
  fillTrainDatasets(list);
}

$('#ds-create').onclick=async()=>{
  const name=$('#ds-new').value.trim();
  if(!name) return;
  const r=await post('/api/datasets',{name});
  if(r.error){ errInto('#ds-err',r.error); return }
  errInto('#ds-err',''); $('#ds-new').value='';
  await loadDatasets(); openDataset(name);
};
$('#ds-new').onkeydown=e=>{ if(e.key==='Enter') $('#ds-create').click() };
$('#ds-back').onclick=()=>{ dsName=null; $('#ds-editor').classList.add('hide');
  $('#ds-index').classList.remove('hide'); loadDatasets(); };

async function openDataset(name){
  dsName=name; dsFilter='all';
  $('#ds-index').classList.add('hide');
  $('#ds-editor').classList.remove('hide');
  $('#ds-title').textContent=name;
  errInto('#ds-edit-err','');
  await loadTiles();
}

async function loadTiles(){
  if(!dsName) return;
  const d=await api('/api/datasets/'+encodeURIComponent(dsName));
  if(d.error){ errInto('#ds-edit-err',d.error); return }
  dsImages=d.images||[];
  if(!$('#ds-trig').value) $('#ds-trig').value=d.trigger_word||'';
  await loadInsight();
  renderTiles();
}

async function loadInsight(){
  const t=$('#ds-trig').value.trim();
  const r=await api('/api/datasets/'+encodeURIComponent(dsName)+'/insight?trigger='+encodeURIComponent(t));
  dsInsight = r.error ? null : r;
  renderInsight();
}

function tileFlags(img){
  const cap=(img.caption||'').trim();
  const trig=$('#ds-trig').value.trim();
  const noTrig = !!trig && !cap.toLowerCase().includes(trig.toLowerCase());
  const thin = !!dsInsight && dsInsight.thin.includes(img.name);
  return {cap, noTrig, thin};
}

function visibleImages(){
  return dsImages.filter(i=>{
    const f=tileFlags(i);
    if(dsFilter==='uncap') return !f.cap;
    if(dsFilter==='notrig') return f.noTrig;
    return true;
  });
}

function renderTiles(){
  // Wider than it was, because the editor now has the whole window: the point
  // of a contact sheet is seeing the set at once, and eight across does that.
  const cols=[10,8,6,5,3][Math.max(0,Math.min(4,dsDensity))];
  $('#tiles').style.gridTemplateColumns=`repeat(${cols},minmax(0,1fr))`;
  const vis=visibleImages();
  $('#tiles').innerHTML=vis.map(i=>{
    const f=tileFlags(i);
    const cls=['tile', f.thin?'thin':'', f.noTrig?'notrig':''].filter(Boolean).join(' ');
    const sz = i.bytes>=1048576 ? (i.bytes/1048576).toFixed(1)+' MB' : Math.round(i.bytes/1024)+' KB';
    const px = i.width ? `${i.width}×${i.height} · ` : '';
    return `<div class="${cls}" data-tile="${esc(i.name)}">
      <div class="ph">
        <img loading="lazy" src="/api/thumb/${encodeURIComponent(dsName)}/${encodeURIComponent(i.name)}"
             alt="" data-full="${esc(i.name)}">
        <div class="dim">${px}${sz}</div>
        <button class="rm" data-rm="${esc(i.name)}" title="Move to .trash">×</button>
      </div>
      <textarea data-n="${esc(i.name)}" placeholder="No caption" spellcheck="false">${esc(i.caption)}</textarea>
    </div>`;
  }).join('');

  // Autosave per caption on blur, with the pending state visible meanwhile.
  $$('#tiles textarea').forEach(t=>{
    t.oninput=()=>t.classList.toggle('dirty', t.value!==t.defaultValue);
    t.onblur=()=>saveCaption(t);
    t.onkeydown=e=>{ if(e.key==='Enter'&&!e.shiftKey){ e.preventDefault(); t.blur() } };
  });
  $$('#tiles [data-rm]').forEach(b=>b.onclick=()=>removeImage(b));
  $$('#tiles [data-full]').forEach(im=>im.onclick=()=>
    lightbox('/api/image/'+encodeURIComponent(dsName)+'/'+encodeURIComponent(im.dataset.full)));

  const capd=dsImages.filter(i=>(i.caption||'').trim()).length;
  const shown = vis.length===dsImages.length ? '' : ` · showing ${vis.length}`;
  $('#ds-count').textContent=`${dsImages.length} image${dsImages.length===1?'':'s'} · ${capd} captioned${shown}`;
  ['all','uncap','notrig'].forEach(k=>{
    const b=$('#f-'+k);
    if(b) b.style.borderColor = dsFilter===k ? 'rgba(255,255,255,.45)' : '';
  });
  $('#drop-title').textContent = dsImages.length ? 'More images' : 'Images or a .zip';
}

async function saveCaption(t){
  const val=t.value.trim();
  const rec=dsImages.find(i=>i.name===t.dataset.n);
  if(rec && rec.caption===val){ t.classList.remove('dirty'); return }
  const r=await post('/api/datasets/'+encodeURIComponent(dsName)+'/caption',
    {image:t.dataset.n, caption:val});
  if(r.error){ errInto('#ds-edit-err',r.error); return }
  if(rec) rec.caption=val;
  t.defaultValue=val; t.classList.remove('dirty');
  loadInsight();
}

async function removeImage(b){
  b.disabled=true;
  const r=await post('/api/datasets/'+encodeURIComponent(dsName)+'/remove',{image:b.dataset.rm});
  if(r.error){ errInto('#ds-edit-err',r.error); b.disabled=false; return }
  dsImages=dsImages.filter(i=>i.name!==b.dataset.rm);
  renderTiles(); loadInsight();
}

// ---------- insight ----------
// The prose answer to "what is this dataset teaching the model?" — trigger
// coverage first because a caption without it trains a LoRA you cannot summon.
function renderInsight(){
  const d=dsInsight;
  if(!d){ $('#ins-body').innerHTML=''; return }
  const pct=(n,t)=>t? Math.round(n/t*100) : 0;
  const bar=(n,t,good)=>{
    const p=pct(n,t);
    const cls = p>=100 ? '' : (good ? (p>=80?'warn':'bad') : (p>=50?'warn':'bad'));
    return `<div class="meter"><i class="${cls}" style="width:${p}%"></i></div>`;
  };
  let h='';
  // Trigger coverage is a ratio over captions, so it says nothing until there
  // are captions — "0/0 have the trigger" reads as a failure rather than a
  // not-yet.
  if(d.trigger_word && d.captioned){
    h+=`<div class="stat"><b>${d.with_trigger}/${d.captioned}</b><span>have the trigger</span></div>`
     + bar(d.with_trigger, d.captioned, true);
  }
  h+=`<div class="stat"><b>${d.captioned}/${d.images}</b><span>captioned</span></div>`
   + bar(d.captioned, d.images, true);
  if(d.median_words) h+=`<p class="muted" style="margin:-6px 0 14px;font-size:12px">median ${d.median_words} words`
   + (d.thin.length?` · ${d.thin.length} thin`:'')+`</p>`;

  // Human-error checks first. These are defects; the clause list below is not.
  const dupImgs=(d.duplicates||[]).flatMap(x=>x.images);
  if(dupImgs.length){
    h+=`<button class="ph-row" data-names="${esc(dupImgs.join('|'))}" style="margin-bottom:6px">
      <span class="ph-bar"><i class="bad" style="--w:100%"></i><span>${dupImgs.length} identical captions</span></span>
    </button>`;
  }
  if((d.tag_style||[]).length){
    h+=`<button class="ph-row" data-names="${esc(d.tag_style.join('|'))}" style="margin-bottom:6px">
      <span class="ph-bar"><i class="warn" style="--w:100%"></i><span>${d.tag_style.length} written as tags</span></span>
    </button>`;
  }

  if(d.phrases && d.phrases.length){
    const max=d.phrases[0].count||1;
    h+=`<label style="margin-top:4px">Repeated clauses</label>`;
    h+=d.phrases.map(p=>{
      const w=Math.round(p.count/max*100);
      const hot=p.share>=0.6?' hot':'';
      return `<button class="ph-row" data-phrase="${esc(p.phrase)}">
        <span class="ph-bar"><i class="${hot.trim()}" style="--w:${w}%"></i><span>${esc(p.phrase)}</span></span>
        <span class="ph-n">${p.count}</span>
      </button>`;
    }).join('');
    h+=`<p class="muted" style="margin-top:8px;font-size:11px">Click to see where it repeats.</p>`;
  }
  if(!h) h='<p class="muted" style="font-size:12px">Nothing to flag.</p>';
  $('#ins-body').innerHTML=h;
  $$('#ins-body [data-names]').forEach(b=>b.onclick=()=>{
    isolate(b.dataset.names.split('|'), b.querySelector('span span').textContent);
  });
  $$('#ins-body [data-phrase]').forEach(b=>b.onclick=()=>{
    const ph=b.dataset.phrase.toLowerCase();
    const hits=dsImages.filter(i=>(i.caption||'').toLowerCase().includes(ph)).map(i=>i.name);
    isolate(hits, '“'+b.dataset.phrase+'”');
  });
}

// Show only the named images. Any filter button clears it.
function isolate(names, label){
  dsFilter='all';
  renderTiles();
  $$('#tiles [data-tile]').forEach(t=>{
    const hit=names.includes(t.dataset.tile);
    t.classList.toggle('sel', hit);
    t.style.display = hit ? '' : 'none';
  });
  $('#ds-count').textContent=`${names.length} of ${dsImages.length} · ${label}`;
}

$('#f-all').onclick=()=>{ dsFilter='all'; renderTiles() };
$('#f-uncap').onclick=()=>{ dsFilter='uncap'; renderTiles() };
$('#f-notrig').onclick=()=>{ dsFilter='notrig'; renderTiles() };
$('#dens-up').onclick=()=>{ dsDensity=Math.min(4,dsDensity+1); renderTiles() };
$('#dens-down').onclick=()=>{ dsDensity=Math.max(0,dsDensity-1); renderTiles() };
$('#ds-trig').oninput=()=>{ renderTiles() };
$('#ds-trig').onblur=async()=>{
  if(!dsName) return;
  await post('/api/datasets/'+encodeURIComponent(dsName)+'/meta',{trigger_word:$('#ds-trig').value.trim()});
  loadInsight();
};

$('#do-prepend').onclick=async()=>{
  const trig=$('#ds-trig').value.trim();
  if(!trig){ errInto('#ds-edit-err','Set a trigger word first.'); return }
  const b=$('#do-prepend'); b.disabled=true; const was=b.textContent;
  const r=await post('/api/datasets/'+encodeURIComponent(dsName)+'/prepend-trigger',{trigger_word:trig});
  if(r.error){ errInto('#ds-edit-err',r.error) }
  else{ b.textContent=r.changed?`${r.changed}`:'ok'; await loadTiles() }
  setTimeout(()=>{ b.textContent=was; b.disabled=false },1600);
};

$('#do-caption').onclick=async()=>{
  const btn=$('#do-caption'); btn.disabled=true;
  const box=$('#cap-prog'); box.classList.remove('hide');
  errInto('#ds-edit-err','');
  const r=await post('/api/caption',{dataset:dsName,trigger_word:$('#ds-trig').value.trim(),
    style:$('#cap-style').value,length:$('#cap-len').value,overwrite:$('#cap-over').checked});
  if(r.error){ errInto('#ds-edit-err',r.error); btn.disabled=false; box.classList.add('hide'); return }
  clearInterval(capPoll);
  capPoll=setInterval(async()=>{
    const s=await api('/api/status/'+r.job_id);
    box.querySelector('i').style.width=(s.percent||0)+'%';
    box.querySelector('p').textContent=s.step?`Captioning ${s.step}/${s.total_steps}`:'Loading captioner…';
    // Refresh mid-run so captions land visibly rather than all at the end.
    if(s.step&&s.step%5===0) loadTiles();
    if(s.status==='completed'){ clearInterval(capPoll); box.classList.add('hide');
      btn.disabled=false; loadTiles(); }
    else if(s.status==='failed'){ clearInterval(capPoll); box.classList.add('hide');
      btn.disabled=false; errInto('#ds-edit-err',s.error||'Captioning failed'); }
  },2500);
};

// ---------- upload (fires immediately on drop; no prerequisites) ----------
const drop=$('#drop'), fin=$('#files');
let uploading=false;
drop.onclick=()=>{ if(!uploading) fin.click() };
drop.ondragover=e=>{e.preventDefault();drop.classList.add('hot')};
drop.ondragleave=()=>drop.classList.remove('hot');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('hot');upload(e.dataTransfer.files)};
fin.onchange=()=>{ upload(fin.files); fin.value=''; };   // reset so the same file can be re-picked

function upload(list){
  const keep=[...list].filter(f=>/\.(png|jpe?g|webp|bmp|avif|zip|txt)$/i.test(f.name));
  if(!keep.length||uploading||!dsName) return;
  uploading=true; errInto('#ds-edit-err','');
  const box=$('#up-prog'); box.classList.remove('hide'); const bar=box.querySelector('i');
  $('#drop-title').textContent='Uploading…';
  $('#drop-sub').textContent=`${keep.length} file${keep.length>1?'s':''}`;

  const fd=new FormData();
  keep.forEach(f=>fd.append('files',f,f.name));
  fd.append('dataset',dsName);

  // XHR, not fetch — fetch reports no upload progress.
  const x=new XMLHttpRequest();
  x.open('POST','/api/upload');
  x.upload.onprogress=e=>{ if(e.lengthComputable) bar.style.width=Math.round(e.loaded/e.total*100)+'%' };
  x.onload=async()=>{
    uploading=false; box.classList.add('hide'); bar.style.width='0%';
    $('#drop-sub').textContent='';
    let r={}; try{ r=JSON.parse(x.responseText) }catch{}
    if(r.error||x.status>=400){
      const detail = r.error || (x.responseText||'').slice(0,300) || 'no response body';
      errInto('#ds-edit-err','Upload failed ('+x.status+'): '+detail);
      renderTiles(); return;
    }
    await loadTiles();
  };
  x.onerror=()=>{ uploading=false; box.classList.add('hide');
    $('#drop-sub').textContent='';
    errInto('#ds-edit-err','Network error during upload.'); };
  x.send(fd);
}

// ==================== TRAIN ====================
// Train no longer owns a dataset; it picks one.
const show=s=>['build','run'].forEach(x=>$('#step-'+x).classList.toggle('hide',x!==s));
let trainDatasets=[];
function fillTrainDatasets(list){
  trainDatasets=list;
  const sel=$('#t-dataset'), cur=sel.value;
  sel.innerHTML='<option value="">Choose a dataset…</option>'+
    list.map(d=>`<option value="${esc(d.name)}"${d.count?'':' disabled'}>${esc(d.name)} · ${d.count} image${d.count===1?'':'s'}${d.count?'':' (empty)'}</option>`).join('');
  if(cur) sel.value=cur;
  syncTrainDataset();
}
function syncTrainDataset(){
  const d=trainDatasets.find(x=>x.name===$('#t-dataset').value);
  $('#dataset').classList.toggle('hide',!d);
  if(!d){ $('#t-dsinfo').textContent=''; checkTrainReady(); return }
  const warn = d.uncaptioned ? ` · ${d.uncaptioned} uncaptioned` : '';
  $('#t-dsinfo').textContent=`${d.count} image${d.count===1?'':'s'}${warn}`;
  if(d.trigger_word && !$('#ltrig').value) $('#ltrig').value=d.trigger_word;
  if(!$('#lname').value) $('#lname').value=d.name;
  checkTrainReady();
}
$('#t-dataset').onchange=syncTrainDataset;

function checkTrainReady(){
  const ok=$('#t-dataset').value&&$('#lname').value.trim()&&$('#ltrig').value.trim();
  $('#go-train').disabled=!ok;
  $('#train-hint').textContent = !$('#t-dataset').value ? '' :
    (!$('#lname').value.trim()||!$('#ltrig').value.trim()) ? 'Name it and set a trigger word to train' : '';
}
document.addEventListener('input',e=>{ if(e.target.id==='lname'||e.target.id==='ltrig') checkTrainReady() });

let trainJob=null;
$('#go-train').onclick=async()=>{
  $('#train-err').innerHTML='';
  const r=await post('/api/train',{dataset:$('#t-dataset').value,lora_name:$('#lname').value.trim(),
    trigger_word:$('#ltrig').value.trim(),network_dim:$('#a-dim').value,network_alpha:$('#a-alpha').value,
    max_train_epochs:$('#a-epochs').value,learning_rate:$('#a-lr').value,resolution:$('#a-res').value,
    num_repeats:$('#a-rep').value,batch_size:$('#a-bs').value,seed:$('#a-seed').value});
  if(r.error){$('#train-err').innerHTML='<div class="err-box">'+r.error+'</div>';return}
  trainJob=r.job_id;
  show('run'); $('#run-done').classList.add('hide');
  poll=setInterval(pollTrain,3000); pollTrain();
};
async function pollTrain(){
  const s=await api('/api/status/'+trainJob);
  $('#run-phase').textContent=s.phase||'Working…';
  $('#run-pct').textContent=s.percent!=null?s.percent+'%':'';
  $('#run-bar').style.width=(s.percent||0)+'%';
  trainPct=s.percent||0; drawDoor();
  const bits=[];
  if(s.step)bits.push(`step ${s.step}/${s.total_steps}`);
  if(s.epoch)bits.push(`epoch ${s.epoch}/${s.total_epochs}`);
  if(s.rate)bits.push(s.rate);
  if(s.eta)bits.push('ETA '+s.eta);
  if(s.loss!=null)bits.push('loss '+s.loss.toFixed(4));
  $('#run-meta').textContent=bits.join(' · ');
  if(s.status==='completed'||s.status==='stopped'){
    clearInterval(poll);
    $('#run-phase').textContent=s.status==='stopped'?'Stopped':'Done';
    $('#run-bar').style.width='100%';
    // The door goes back to being a door. A finished run left at 100% would
    // read as one still going, and the result is on the Train side anyway.
    trainPct=null; drawDoor();
    $('#run-done').classList.remove('hide');
    $('#run-done').innerHTML=`<b>${s.status==='stopped'?'Stopped':'Training complete'}</b>
      <p class="muted" style="margin-top:7px">${s.note||''}</p>
      <p class="muted" style="margin-top:7px"><code>${s.output_dir||''}</code></p>
      <p class="muted" style="margin-top:5px">${(s.files||[]).length} checkpoint(s) · ${Math.round((s.duration_s||0)/60)} min</p>`;
    loadState();
  } else if(s.status==='failed'){
    clearInterval(poll);
    trainPct=null; drawDoor();
    $('#train-err').innerHTML='<div class="err-box">'+(s.error||'Training failed')+'</div>';
  }
}
$('#do-stop').onclick=async()=>{ $('#do-stop').disabled=true; await post('/api/stop/'+trainJob); };

// ==================== IMAGE ====================
// What the model implies, not what it is called — its name is already in the
// select two controls to the left, and printing it twice is the sentence
// telling you what you can see.
function syncModelLine(){
  const v=$('#g-model').value;
  $('#gen-model-line').textContent = !v ? 'No model downloaded'
    : v==='turbo' ? '8 steps · CFG 1.0' : '28 steps · CFG 5.5';
}
$('#g-model').onchange=syncModelLine;
$('#toggle-adv').onclick=()=>{
  $('#toggle-adv').classList.toggle('on',!$('#gen-adv').classList.toggle('hide'));
};

// ---------- LoRA stack ----------
// One row per LoRA. Two weights each, the way Forge splits them: the first
// patches the DiT, the second the text encoder. Order matters — LoRAs patch in
// the order shown, so the arrows are authority, not decoration.
let loraOpts='';
function loraRow(sel,unet,te){
  const row=document.createElement('div');
  row.className='stack-row'; row.dataset.lora='1';
  row.innerHTML=`
    <select data-f="path">${loraOpts}</select>
    <div class="nums">
      <input class="num" data-f="unet" inputmode="decimal" value="${unet??1}" title="UNet weight">
      <input class="num" data-f="te" inputmode="decimal" value="${te??1}" title="Text encoder weight">
      <span class="grow"></span>
      <button class="s" data-f="up" title="Move up">↑</button>
      <button class="s" data-f="down" title="Move down">↓</button>
      <button class="s" data-f="rm" title="Remove">✕</button>
    </div>`;
  if(sel) row.querySelector('[data-f=path]').value=sel;
  const q=f=>row.querySelector('[data-f='+f+']');
  q('rm').onclick=()=>row.remove();
  q('up').onclick=()=>row.previousElementSibling&&row.parentNode.insertBefore(row,row.previousElementSibling);
  q('down').onclick=()=>row.nextElementSibling&&row.parentNode.insertBefore(row.nextElementSibling,row);
  q('path').onchange=()=>{
    const t=q('path').selectedOptions[0]?.dataset.t;
    if(t&&!$('#prompt').value.includes(t))
      $('#prompt').value=(t+', '+$('#prompt').value).trim().replace(/,\s*$/,'');
  };
  return row;
}
$('#add-lora').onclick=()=>{
  const stack=$('#lora-stack');
  if(stack.children.length>=(window.MAX_LORAS||6)) return;
  stack.appendChild(loraRow());
};
function readLoras(){
  return $$('#lora-stack [data-lora]').map(r=>({
    path:r.querySelector('[data-f=path]').value,
    unet:parseFloat(r.querySelector('[data-f=unet]').value)||0,
    text_encoder:parseFloat(r.querySelector('[data-f=te]').value)||0,
  })).filter(l=>l.path);
}

// ---------- regions ----------
// Rectangles in normalised canvas coordinates. Columns/Rows lay them out
// automatically; the x/y/w/h fields are there when a strip is not what you want.
function regionRow(prompt){
  const row=document.createElement('div');
  row.className='region'; row.dataset.region='1';
  row.innerHTML=`
    <div class="row" style="gap:6px">
      <input class="grow" data-f="prompt" placeholder="Region prompt" value="${esc(prompt||'')}">
      <button class="s" data-f="rm" title="Remove">✕</button>
    </div>
    <div class="cells">
      <input data-f="x" inputmode="decimal" placeholder="x" title="x">
      <input data-f="y" inputmode="decimal" placeholder="y" title="y">
      <input data-f="width" inputmode="decimal" placeholder="w" title="width">
      <input data-f="height" inputmode="decimal" placeholder="h" title="height">
      <input data-f="weight" inputmode="decimal" value="1" title="weight">
    </div>`;
  row.querySelector('[data-f=rm]').onclick=()=>{row.remove();autoLayout()};
  return row;
}
function autoLayout(){
  // Blank x/y/w/h means "let the direction decide"; a typed value is kept.
  const rows=$$('#region-stack [data-region]'), n=rows.length, cols=$('#g-region-dir').value==='columns';
  rows.forEach((r,i)=>{
    const set=(f,v)=>{const el=r.querySelector('[data-f='+f+']'); if(!el.dataset.touched) el.value=v};
    set('x',cols?(i/n).toFixed(3):'0'); set('y',cols?'0':(i/n).toFixed(3));
    set('width',cols?(1/n).toFixed(3):'1'); set('height',cols?'1':(1/n).toFixed(3));
  });
}
document.addEventListener('input',e=>{
  if(e.target.closest('[data-region]')&&['x','y','width','height'].includes(e.target.dataset.f))
    e.target.dataset.touched='1';
});
$('#add-region').onclick=()=>{ $('#region-stack').appendChild(regionRow()); autoLayout(); };
$('#g-region-dir').onchange=autoLayout;
$('#g-regional').onchange=()=>{
  const on=$('#g-regional').checked;
  ['#region-stack','#add-region','#g-region-dir'].forEach(s=>$(s).classList.toggle('hide',!on));
  if(on&&!$$('#region-stack [data-region]').length){
    $('#region-stack').appendChild(regionRow());
    $('#region-stack').appendChild(regionRow());
    autoLayout();
  }
};
function readRegions(){
  if(!$('#g-regional').checked) return [];
  return $$('#region-stack [data-region]').map(r=>{
    const g=f=>r.querySelector('[data-f='+f+']').value;
    return {prompt:g('prompt'),x:parseFloat(g('x'))||0,y:parseFloat(g('y'))||0,
            width:parseFloat(g('width'))||1,height:parseFloat(g('height'))||1,
            weight:parseFloat(g('weight'))||1};
  }).filter(r=>r.prompt.trim());
}

// One image gets the room to be looked at; a batch gets two columns, because
// four stills side by side are thumbnails and you cannot judge a thumbnail.
function layoutShots(n){
  const g=$('#gen-out');
  g.style.gridTemplateColumns = n<=1 ? 'minmax(0,1fr)' : 'repeat(2,minmax(0,1fr))';
  g.style.maxWidth = n<=1 ? '920px' : '1320px';
  // Measured off the canvas, not the viewport: the console under it grows and
  // shrinks with what is open, so a dvh sum would be wrong the moment anyone
  // opened Advanced — the stills would run under the bar instead of fitting
  // above it. A batch has to fit to be compared; a single still gets it all.
  //
  // Every subtrahend is measured rather than guessed. clientHeight includes the
  // canvas padding, and the caption below the grid is a real element with a
  // real height; a fixed fudge factor for the two of them was 42px short, which
  // put the caption under the console on exactly the shot you would screenshot.
  const box=$('#canvas'), cs=getComputedStyle(box);
  const cap=$(kind==='image'?'#gen-meta':'#vid-meta');
  const h=box.clientHeight-parseFloat(cs.paddingTop)-parseFloat(cs.paddingBottom)
          -(cap.offsetHeight||0)-12;
  g.style.setProperty('--shot-h', (n<=2 ? h : h/2-16)+'px');
}
// The canvas changes height whenever the console does, so the fit is recomputed
// rather than set once at generation time.
new ResizeObserver(()=>{
  const n=$('#gen-out').children.length;
  if(n) layoutShots(n);
}).observe(document.querySelector('#canvas'));

$('#go-gen').onclick=async()=>{
  const p=$('#prompt').value.trim(), regions=readRegions();
  if(!p&&!regions.length)return;
  $('#gen-err').innerHTML=''; $('#gen-meta').textContent='';
  const btn=$('#go-gen'); btn.disabled=true;
  const box=$('#gen-prog'); box.classList.remove('hide'); box.querySelector('p').textContent='Queued…';
  const [w,h]=$('#g-aspect').value.split('x');
  const r=await post('/api/generate',{
    prompt:p, negative_prompt:$('#g-neg').value, model:$('#g-model').value,
    loras:readLoras(), regions,
    width:w, height:h, num_images:$('#g-n').value, seed:$('#g-seed').value,
    sampler:$('#g-sampler').value, scheduler:$('#g-scheduler').value,
    steps:$('#g-steps').value, cfg_scale:$('#g-cfg').value, shift:$('#g-shift').value,
    gpu:$('#g-gpu').value,
  });
  if(r.error){$('#gen-err').innerHTML='<div class="err-box">'+r.error+'</div>';btn.disabled=false;box.classList.add('hide');return}
  const t=setInterval(async()=>{
    const s=await api('/api/status/'+r.job_id);
    box.querySelector('i').style.width=(s.percent||0)+'%';
    box.querySelector('p').textContent=s.step?`Step ${s.step}/${s.total_steps}`:(s.phase||'Working…');
    if(s.status==='completed'){
      clearInterval(t); btn.disabled=false; box.classList.add('hide');
      const out=await api('/api/outputs/'+r.job_id);
      // Each still carries its own way into video. Two, because they are
      // genuinely different jobs: a first frame is the shot the clip starts on,
      // a reference is a subject the clip is about.
      const imgs=out.images||[];
      layoutShots(imgs.length);
      $('#gen-out').innerHTML=imgs.map((i,n)=>
        `<figure class="shot"><img src="${i.data}" alt="">`+
        `<span class="acts"><button class="s" data-n="${n}" data-as="first">Animate</button>`+
        `<button class="s" data-n="${n}" data-as="reference">As reference</button></span></figure>`).join('')
        || '<p class="muted">Saved to '+(s.output_dir||'')+'</p>';
      $$('#gen-out .acts button').forEach(b=>b.onclick=()=>
        toVideo(imgs[+b.dataset.n].data.split(',')[1], b.dataset.as));
      // Surface which LoRAs actually matched — a stack that silently no-ops
      // looks identical to a stack that had no effect.
      const skipped=(s.loras||[]).filter(l=>!l.applied);
      $('#gen-meta').textContent=[
        (s.seeds||[]).join(', ')&&('seed '+(s.seeds||[]).join(', ')),
        s.sampler&&`${s.sampler} · ${s.steps} steps · CFG ${s.cfg_scale}`,
        s.duration_s&&`${s.duration_s}s`,
        skipped.length&&('not applied: '+skipped.map(l=>l.name+(l.reason?` (${l.reason})`:'')).join(', ')),
      ].filter(Boolean).join(' · ');
      syncCanvasView(); loadGallery();
    } else if(s.status==='stopped'){
      clearInterval(t); btn.disabled=false; box.classList.add('hide');
    } else if(s.status==='failed'){
      clearInterval(t); btn.disabled=false; box.classList.add('hide');
      $('#gen-err').innerHTML='<div class="err-box">'+(s.error||'Generation failed')+'</div>';
    }
  },2000);
};

// ---------- gpu ----------
// One warning, once per card, and only when you actually change it. Switching
// starts a container that does not exist yet — on the video side that is 42.5 GB
// of weights, so the cost is worth a sentence before it is spent rather than a
// progress bar that sits still for minutes afterwards.
function wireGpu(sel, spec){
  const el=$(sel);
  el.innerHTML=spec.options.map(g=>
    `<option value="${g}"${g===spec.default?' selected':''}>${g}</option>`).join('');
  let last=spec.default;
  el.onchange=()=>{
    if(el.value===last) return;
    if(el.value===spec.default ||
       confirm(`Switch to ${el.value}?\n\nThis card has no warm container, so the next run pays a cold start while the model loads. Runs after it are warm.`)){
      last=el.value;
    } else {
      el.value=last;
    }
  };
}

// ==================== VIDEO ====================
// Keyframes and references are held here as bare base64, the shape /api/video
// wants, so the submit handler never has to go back to the file for them.
const keyframe={first:null,last:null};
let refs=[], refVids=[];

// ---------- the model, and what it makes the panel look like ----------
// Two video families that genuinely differ: H3 carries a soundtrack and takes
// references but no LoRAs and no guidance; Wan takes LoRAs, CFG and a negative
// prompt but is silent. Rather than one panel with half its controls quietly
// inert, the composer is rebuilt from what the chosen model says it reads.
let vidModels=[];
const videoModel=()=>vidModels.find(m=>m.key===$('#v-model').value)||null;

// A value you chose is yours and survives a model change; a value that is only
// the previous model's default is not. Without the distinction, switching from
// H3 to Wan kept res_multistep — a sampler nobody picked, quietly overriding
// the euler that Wan's own templates use — because it happened to be in both
// lists. Same idiom as the region fields below.
// Assigning .value fires no change event, so only a real interaction lands
// here — reuse() is the one other thing that sets the flag, deliberately.
document.addEventListener('change',e=>{
  if(e.target.matches('#v-tier,#v-seconds,#v-sampler,#v-scheduler'))
    e.target.dataset.touched='1';
});
function fillSelect(sel,opts,value){
  const el=$(sel), prev=el.dataset.touched?el.value:null;
  el.innerHTML=opts.map(o=>`<option value="${o[0]}">${esc(o[1])}</option>`).join('');
  const keys=opts.map(o=>String(o[0]));
  el.value = keys.includes(prev) ? prev : (keys.includes(String(value))?String(value):keys[0]);
}

function syncVideoModel(){
  const m=videoModel();
  if(!m) return;
  const sup=m.supports, d=m.defaults;

  fillSelect('#v-tier', Object.entries(m.tiers), d.tier);
  fillSelect('#v-seconds', m.lengths.map(n=>[n,n+'s']), d.seconds);
  fillSelect('#v-sampler', m.samplers.map(x=>[x,x]), d.sampler);
  fillSelect('#v-scheduler', m.schedulers.map(x=>[x,x]), d.scheduler);
  // Placeholders, not values: an empty box means "the model's default", which
  // stays right when the default changes under it.
  $('#v-steps').placeholder=String(d.steps);
  $('#v-cfg').placeholder=String(d.cfg??'');
  $('#v-shift').placeholder=String(d.shift??'');

  $('#v-add-lora').classList.toggle('hide',!sup.loras);
  $('#v-ref-sec').classList.toggle('hide',!sup.references);
  $('#v-neg').classList.toggle('hide',!sup.negative);
  $('#v-cfg-wrap').classList.toggle('hide',!sup.cfg);
  $('#v-shift-wrap').classList.toggle('hide',!sup.cfg);
  $('#v-switch-wrap').classList.toggle('hide',!sup.experts);
  $('#v-drop-last').classList.toggle('hide',!sup.last_frame);
  $$('#v-lora-stack [data-f=expert]').forEach(el=>el.classList.toggle('hide',!sup.experts));

  // A model that cannot take what is already attached would fail at submit.
  // Dropping it here, where the section it came from is visibly gone, is the
  // only version of this that does not look like the request lost it.
  if(!sup.references&&(refs.length||refVids.length)){ refs=[]; refVids=[]; }
  if(!sup.last_frame&&keyframe.last){
    clearFrame('last');
  }
  if(kind==='video') syncPromptHint();

  // What this model needs that is not on the volume, per task — so a t2v run
  // is never told to download the 28.6 GB i2v pair it will not load.
  const task = sup.references&&(refs.length||refVids.length) ? 'ref2va'
             : (m.key==='h3' ? 'fl2va' : (keyframe.first||keyframe.last?'i2v':'t2v'));
  const t=m.tasks[task]||{ready:true,missing:[]};
  $('#go-vid').disabled=!t.ready;
  $('#v-model-line').textContent = t.ready ? m.note
    : 'Not downloaded: '+t.missing.join(', ')+' — get them under Settings.';
  drawRefs();
}
$('#v-model').onchange=syncVideoModel;

// ---------- video LoRA stack ----------
// The image stack's two weights do not carry over: ComfyUI's model-only loader
// patches the DiT and umT5 is loaded separately, so there is one strength. What
// replaces the second field is the expert, which the A14B pair forces — it is
// two checkpoints, and a row has to say which one it patches.
function vidLoraRow(sel,unet,expert){
  const sup=(videoModel()||{supports:{}}).supports;
  const row=document.createElement('div');
  row.className='stack-row'; row.dataset.lora='1';
  row.innerHTML=`
    <select data-f="path">${loraOpts}</select>
    <div class="nums">
      <input class="num" data-f="unet" inputmode="decimal" value="${unet??1}" title="Strength">
      <select data-f="expert" class="${sup.experts?'':'hide'}" title="Which expert this patches">
        ${(window.WAN_EXPERTS||['both','high','low']).map(e=>
          `<option value="${e}">${e==='both'?'both experts':e+' noise'}</option>`).join('')}
      </select>
      <span class="grow"></span>
      <button class="s" data-f="up" title="Move up">↑</button>
      <button class="s" data-f="down" title="Move down">↓</button>
      <button class="s" data-f="rm" title="Remove">✕</button>
    </div>`;
  const q=f=>row.querySelector('[data-f='+f+']');
  if(sel) q('path').value=sel;
  if(expert) q('expert').value=expert;
  q('rm').onclick=()=>row.remove();
  q('up').onclick=()=>row.previousElementSibling&&row.parentNode.insertBefore(row,row.previousElementSibling);
  q('down').onclick=()=>row.nextElementSibling&&row.parentNode.insertBefore(row.nextElementSibling,row);
  // The paired speed LoRAs are named `high` and `low` inside one folder, so the
  // file already says which expert it belongs to. Reading it beats making you
  // set the same fact twice and beats the silent quality loss of crossing them.
  q('path').onchange=()=>{
    const n=(q('path').value.split('/').pop()||'').toLowerCase();
    if(n.startsWith('high')||n.includes('high_noise')) q('expert').value='high';
    else if(n.startsWith('low')||n.includes('low_noise')) q('expert').value='low';
  };
  return row;
}
$('#v-add-lora').onclick=()=>{
  const stack=$('#v-lora-stack');
  if(stack.children.length>=(window.MAX_LORAS||6)) return;
  stack.appendChild(vidLoraRow());
};
function readVidLoras(){
  if(!(videoModel()||{supports:{}}).supports.loras) return [];
  return $$('#v-lora-stack [data-lora]').map(r=>({
    path:r.querySelector('[data-f=path]').value,
    unet:parseFloat(r.querySelector('[data-f=unet]').value)||0,
    expert:r.querySelector('[data-f=expert]').value||'both',
  })).filter(l=>l.path);
}

// Reading a File to base64 is needed by four different entry points — the two
// keyframe drops, the reference picker, and the hand-off from finished work —
// so it lives here rather than four times over.
const toB64=f=>new Promise(res=>{
  const r=new FileReader();
  r.onload=()=>res(r.result.split(',')[1]);
  r.readAsDataURL(f);
});

function drawRefs(){
  // Images are labelled with the <Picture n> the prompt will use, videos with
  // <Video n> — the label is the thing you type, so it is worth showing.
  const img=refs.map((b,i)=>
    `<div class="ref"><img src="data:image/png;base64,${b}" alt="">`+
    `<b>P${i+1}</b><button data-k="img" data-i="${i}" title="Remove">×</button></div>`).join('');
  const vid=refVids.map((b,i)=>
    // Same media fragment as the gallery card, and for the same reason: a
    // reference tile with no frame painted is an unlabelled black square,
    // which defeats the point of showing the reference you attached.
    `<div class="ref"><video src="data:video/mp4;base64,${b}#t=0.04" muted></video>`+
    `<b>V${i+1}</b><button data-k="vid" data-i="${i}" title="Remove">×</button></div>`).join('');
  $('#v-refs').innerHTML=img+vid;
  $$('#v-refs button').forEach(b=>b.onclick=()=>{
    (b.dataset.k==='img'?refs:refVids).splice(+b.dataset.i,1); drawRefs();
  });
  // On H3 references and keyframes load different transformers, so saying
  // which one is going to run beats letting the two sections look
  // simultaneously active. On Wan the same sentence has a different job: a
  // first frame is what makes it an i2v run, on a different 28.6 GB pair.
  const n=refs.length+refVids.length;
  const sup=(videoModel()||{supports:{}}).supports;
  $('#vid-note').textContent = n
    ? `${n} reference${n>1?'s':''} — keyframes are ignored for this run.`
    : (keyframe.first
        ? (sup.references?'':'Image-to-video. ')+'Canvas follows the first frame’s aspect ratio.'
        : '');
}

function pickRefs(kindOf){
  const isImg=kindOf==='img';
  const bucket=isImg?refs:refVids;
  const max=+$(isImg?'#v-ref-max':'#v-vid-max').textContent;
  if(bucket.length>=max){
    alert(`${max} ${isImg?'image':'video'} references is the model's limit.`); return;
  }
  const input=document.createElement('input');
  input.type='file'; input.accept=isImg?'image/*':'video/*'; input.multiple=true;
  input.onchange=async e=>{
    for(const f of [...e.target.files].slice(0,max-bucket.length)) bucket.push(await toB64(f));
    drawRefs();
  };
  input.click();
}
$('#v-add-ref').onclick=()=>pickRefs('img');
$('#v-add-vid').onclick=()=>pickRefs('vid');

function wireDrop(slot){
  const box=$('#v-drop-'+slot), img=$('#v-thumb-'+slot), hint=$('#v-hint-'+slot);
  hint.innerHTML=ICON[slot];
  const input=document.createElement('input');
  input.type='file'; input.accept='image/*'; input.className='hide';
  box.appendChild(input);

  const take=async f=>{
    if(!f||!f.type.startsWith('image/'))return;
    setFrame(slot, await toB64(f));
  };
  box.onclick=()=>input.click();
  input.onchange=e=>take(e.target.files[0]);
  box.ondragover=e=>{e.preventDefault();box.classList.add('hot')};
  box.ondragleave=()=>box.classList.remove('hot');
  box.ondrop=e=>{e.preventDefault();box.classList.remove('hot');take(e.dataTransfer.files[0])};
}
function clearFrame(slot){
  keyframe[slot]=null;
  $('#v-thumb-'+slot).classList.add('hide');
  $('#v-hint-'+slot).classList.remove('hide');
  $('#v-drop-'+slot).classList.remove('set');
}
function setFrame(slot,b64){
  keyframe[slot]=b64;
  $('#v-thumb-'+slot).src='data:image/png;base64,'+b64;
  $('#v-thumb-'+slot).classList.remove('hide');
  $('#v-hint-'+slot).classList.add('hide');
  $('#v-drop-'+slot).classList.add('set');
  if(slot==='first') syncFrameCanvas();
}
wireDrop('first'); wireDrop('last');

// A first frame anchors the geometry — the canvas follows the image, so the
// aspect picker stops being the thing that decides it. Disabling it and saying
// why beats leaving a control that looks live and is quietly ignored.
//
// It also changes which weights the run needs on Wan (t2v and i2v are separate
// checkpoints), which is why this re-runs the whole model sync rather than just
// redrawing the references.
function syncFrameCanvas(){
  $('#v-aspect').disabled=!!keyframe.first;
  syncVideoModel();
}

// The hand-off. A still you just made becomes the thing the next clip animates,
// without a download and a re-upload — which is the whole point of image and
// video sharing one workspace.
function toVideo(b64, as){
  if(as==='reference'||as==='refvideo'){
    // Only one family has a reference checkpoint. Moving to it is the useful
    // reading of the button — the alternative is accepting the image and then
    // dropping it the moment the panel redraws.
    if(!(videoModel()||{supports:{}}).supports.references){
      const m=vidModels.find(x=>x.supports.references&&x.ready);
      if(!m){alert('References need MiniMax-H3 — download it under Settings.');return}
      $('#v-model').value=m.key; syncVideoModel();
    }
    const img=as==='reference';
    const max=+$(img?'#v-ref-max':'#v-vid-max').textContent;
    const bucket=img?refs:refVids;
    if(bucket.length>=max){alert(`${max} references is the model's limit.`);return}
    bucket.push(b64);
  } else {
    setFrame('first', b64);
  }
  syncFrameCanvas();
  setKind('video');
  closeGallery();
  $('#prompt').focus();
}

$('#v-toggle-adv').onclick=()=>{
  $('#v-toggle-adv').classList.toggle('on',!$('#vid-adv').classList.toggle('hide'));
};

$('#go-vid').onclick=async()=>{
  const p=$('#prompt').value.trim();
  if(!p)return;
  $('#vid-err').innerHTML=''; $('#vid-meta').textContent='';
  const btn=$('#go-vid'); btn.disabled=true;
  const box=$('#vid-prog'); box.classList.remove('hide');
  box.querySelector('i').style.width='0%';
  box.querySelector('p').textContent='Queued…';

  const r=await post('/api/video',{
    model:$('#v-model').value,
    prompt:p, negative_prompt:$('#v-neg').value,
    aspect:$('#v-aspect').value, tier:$('#v-tier').value,
    seconds:$('#v-seconds').value, steps:$('#v-steps').value, seed:$('#v-seed').value,
    cfg:$('#v-cfg').value, shift:$('#v-shift').value, switch_at:$('#v-switch').value,
    sampler:$('#v-sampler').value, scheduler:$('#v-scheduler').value,
    loras:readVidLoras(),
    first_frame:keyframe.first, last_frame:keyframe.last,
    references:refs, ref_videos:refVids,
    ref_size:$('#v-ref-size').value, gpu:$('#v-gpu').value,
  });
  if(r.error){
    $('#vid-err').innerHTML='<div class="err-box">'+r.error+'</div>';
    btn.disabled=false; box.classList.add('hide'); return;
  }

  const t=setInterval(async()=>{
    const s=await api('/api/status/'+r.job_id);
    box.querySelector('i').style.width=(s.percent||0)+'%';
    // The first minutes are 42.5 GB loading onto the card, with no step count
    // to show yet. Naming that beats a bar that sits at zero looking stuck.
    box.querySelector('p').textContent = s.step
      ? `Step ${s.step}/${s.total_steps}${s.eta?' · '+s.eta+' left':''}`
      : (s.phase==='loading'?'Loading the model…':(s.phase||'Working…'));
    // syncVideoModel() rather than a bare re-enable: the button's state is
    // "this model can run", and a run finishing is not a reason to override it.
    if(s.status==='completed'){
      clearInterval(t); syncVideoModel(); box.classList.add('hide');
      const f=(s.files||[])[0];
      $('#vid-out').innerHTML=f
        ? `<video controls autoplay loop playsinline src="/api/file/${r.job_id}/${f}"></video>`
        : '<p class="muted">Saved to '+(s.output_dir||'')+'</p>';
      const stack=readVidLoras();
      $('#vid-meta').textContent=[
        s.width&&`${s.width}×${s.height}`,
        s.seconds&&`${s.seconds}s · ${s.frames} frames · ${s.fps} fps`,
        s.seed!=null&&`seed ${s.seed}`,
        s.steps&&`${s.steps} steps`,
        stack.length&&`${stack.length} LoRA${stack.length>1?'s':''}`,
        s.duration_s&&`${s.duration_s}s`,
      ].filter(Boolean).join(' · ');
      syncCanvasView(); loadGallery();
    } else if(s.status==='stopped'){
      clearInterval(t); syncVideoModel(); box.classList.add('hide');
    } else if(s.status==='failed'){
      clearInterval(t); syncVideoModel(); box.classList.add('hide');
      $('#vid-err').innerHTML='<div class="err-box">'+(s.error||'Generation failed')+'</div>';
    }
  },2000);
};

// ==================== GALLERY ====================
// Reads the volume, not a job id. Anything generated is here after a reload,
// a redeploy, or a job record that expired — the folder is the record.
//
// A card shows the work and nothing else. Prompts here run past two hundred
// tokens and two clamped lines of one are neither readable nor identifiable;
// the only time the settings matter is when you want to run them again, and
// that is what the overflow menu is for.
let galItems=[], galFilter='all';

const ago=t=>{
  if(!t) return '';
  const s=Math.max(0,Date.now()/1000-t);
  if(s<90) return 'just now';
  if(s<5400) return Math.round(s/60)+'m ago';
  if(s<172800) return Math.round(s/3600)+'h ago';
  return Math.round(s/86400)+'d ago';
};

async function loadGallery(){
  const r=await api('/api/gallery');
  galItems=r.items||[];
  drawDrawer();
  if(!$('#gal-full').classList.contains('hide')) drawGallery();
}

function galCard(it,i){
  const src=`/api/file/${it.job_id}/${it.files[0]}`;
  // A poster frame would be a second request per card, so the video element
  // loads metadata only. But metadata is dimensions and duration, not a
  // picture: with nothing to paint, every clip in the gallery was a black
  // rectangle you had to open to identify. `#t=` is the media fragment that
  // fixes it — the browser seeks to that time and paints the frame there,
  // out of the bytes it already has, so the card costs no extra request.
  // Not 0: seeking to exactly zero is not required to decode a frame, and
  // some browsers leave the canvas blank.
  const media=it.kind==='video'
    ? `<video class="media" src="${src}#t=0.04" preload="metadata" muted playsinline data-open></video>`
    : `<img class="media" src="${src}" alt="" loading="lazy" data-open>`;
  const n=it.files.length>1?` · ${it.files.length}`:'';
  return `<div class="gal" data-i="${i}">${media}
    <div class="quick">
      <button data-act="download" title="Download">${ICON.download}</button>
      <button data-act="del" title="Move to trash">${ICON.close}</button>
    </div>
    <div class="foot">
      <span class="kind">${it.kind==='video'?ICON.play:ICON.photo}</span>
      <span class="when">${ago(it.created||it.modified)}${n}</span>
      <span class="grow"></span>
      <button class="more" data-act="menu" title="More">${ICON.more}</button>
    </div>
  </div>`;
}

function wireCards(root,rows){
  root.querySelectorAll('.gal').forEach(card=>{
    const it=rows[+card.dataset.i];
    card.querySelector('[data-open]').onclick=()=>
      lightbox(`/api/file/${it.job_id}/${it.files[0]}`, it.kind==='video');
    card.querySelectorAll('[data-act]').forEach(b=>b.onclick=e=>{
      e.stopPropagation();
      if(b.dataset.act==='download') return download(it);
      if(b.dataset.act==='del') return remove(it);
      openMenu(b, menuFor(it));
    });
  });
}

function drawDrawer(){
  const rows=galItems.slice(0,24);
  $('#drawer-grid').innerHTML=rows.map(galCard).join('');
  $('#drawer-empty').textContent=rows.length?'':'Nothing generated yet.';
  wireCards($('#drawer-grid'),rows);
}

function drawGallery(){
  const rows=galItems.filter(i=>galFilter==='all'||i.kind===galFilter);
  $('#gal-empty').textContent=rows.length?'':
    (galItems.length?'Nothing of that kind yet.':'Nothing generated yet.');
  $('#gal-grid').innerHTML=rows.map(galCard).join('');
  wireCards($('#gal-grid'),rows);
}

function openGallery(){ $('#gal-full').classList.remove('hide'); drawGallery() }
function closeGallery(){ $('#gal-full').classList.add('hide') }
$('#gal-expand').onclick=openGallery;
$('#gal-back').onclick=closeGallery;
$('#gal-refresh').onclick=loadGallery;
$$('#gal-full [data-filter]').forEach(b=>b.onclick=()=>{
  galFilter=b.dataset.filter;
  $$('#gal-full [data-filter]').forEach(x=>x.classList.toggle('on',x===b));
  drawGallery();
});

// ---------- card actions ----------
function download(it){
  it.files.forEach(f=>{
    const a=document.createElement('a');
    a.href=`/api/file/${it.job_id}/${f}`; a.download=f;
    document.body.appendChild(a); a.click(); a.remove();
  });
}

async function remove(it){
  if(!confirm('Move this result to the trash?')) return;
  await post(`/api/outputs/${it.job_id}/delete`);
  loadGallery();
}

// Fetch the bytes rather than reusing a data URL: the card is a streamed
// <img src>, so the base64 the video side needs does not exist client-side.
async function handoff(it,as){
  const blob=await (await fetch(`/api/file/${it.job_id}/${it.files[0]}`)).blob();
  toVideo(await toB64(blob), as);
}

function menuFor(it){
  const m=[{label:'Reuse prompt & settings',run:()=>reuse(it)}];
  if(it.kind==='image'){
    m.push({label:'Animate from this frame',run:()=>handoff(it,'first')});
    m.push({label:'Use as reference',run:()=>handoff(it,'reference')});
  } else {
    m.push({label:'Use as video reference',run:()=>handoff(it,'refvideo')});
  }
  m.push({sep:true});
  m.push({label:'View metadata',run:()=>metaSheet(it)});
  m.push({label:it.files.length>1?`Download ${it.files.length} files`:'Download',run:()=>download(it)});
  m.push({label:'Delete',danger:true,run:()=>remove(it)});
  return m;
}

// The point of keeping the sidecar: everything below reproduces the result, so
// putting it back in the composer is a read of a file, not a re-typed prompt.
function reuse(it){
  closeGallery();
  const set=(sel,v)=>{ if(v!=null&&v!=='') $(sel).value=v };
  if(it.kind==='image'){
    setKind('image');
    set('#prompt',it.prompt); set('#g-neg',it.negative_prompt);
    if(it.model) $('#g-model').value=it.model;
    const size=`${it.width}x${it.height}`;
    if([...$('#g-aspect').options].some(o=>o.value===size)) $('#g-aspect').value=size;
    set('#g-seed',(it.seeds||[])[0]);
    set('#g-sampler',it.sampler); set('#g-scheduler',it.scheduler);
    set('#g-steps',it.steps); set('#g-cfg',it.cfg_scale); set('#g-shift',it.shift);
    // The stack is reported by name, not path, so rows are matched against the
    // filename each option carries. A LoRA since deleted simply does not return.
    const stack=$('#lora-stack'); stack.innerHTML='';
    (it.loras||[]).forEach(l=>{
      const row=loraRow(); stack.appendChild(row);
      const sel=row.querySelector('[data-f=path]');
      const hit=[...sel.options].find(o=>
        o.value.split('/').pop().replace(/\.safetensors$/i,'')===l.name);
      if(hit) sel.value=hit.value; else row.remove();
      if(hit){
        row.querySelector('[data-f=unet]').value=l.unet??1;
        row.querySelector('[data-f=te]').value=l.text_encoder??1;
      }
    });
    syncModelLine();
    if(it.negative_prompt||it.steps) $('#gen-adv').classList.remove('hide');
    $('#prompt').focus();
  } else {
    setKind('video');
    // The model first, and re-sync before anything else: it decides which
    // controls exist, so restoring a CFG into a panel that has not been rebuilt
    // yet writes to a box the next redraw hides.
    if(it.model&&[...$('#v-model').options].some(o=>o.value===it.model&&!o.disabled)){
      $('#v-model').value=it.model;
    }
    syncVideoModel();
    set('#prompt',it.prompt); set('#v-neg',it.negative_prompt);
    set('#v-seed',it.seed); set('#v-steps',it.steps);
    set('#v-cfg',it.cfg_scale); set('#v-shift',it.shift); set('#v-switch',it.switch_at);
    // Marked as chosen, not defaulted: restoring a clip's settings and then
    // attaching a frame must not quietly hand them back to the model's defaults.
    const pick=(sel,v)=>{
      const el=$(sel);
      if(v!=null&&v!==''&&[...el.options].some(o=>o.value===String(v))){
        el.value=String(v); el.dataset.touched='1';
      }
    };
    pick('#v-sampler',it.sampler); pick('#v-scheduler',it.scheduler);
    if(it.seconds) pick('#v-seconds',Math.round(it.seconds));
    if(it.width&&it.height&&!keyframe.first){
      const r=it.width/it.height;
      const near=[...$('#v-aspect').options].map(o=>{
        const [a,b]=o.value.split(':').map(Number);
        return {v:o.value,d:Math.abs(a/b-r)};
      }).sort((x,y)=>x.d-y.d)[0];
      if(near) $('#v-aspect').value=near.v;
    }
    // Matched on the full name under loras/, not the stem: `high.safetensors`
    // is the filename of both speed pairs, so a stem match would restore the
    // t2v LoRA into an i2v run without a word about it.
    const stack=$('#v-lora-stack'); stack.innerHTML='';
    (it.loras||[]).forEach(l=>{
      const row=vidLoraRow(); stack.appendChild(row);
      const sel=row.querySelector('[data-f=path]');
      const hit=[...sel.options].find(o=>o.value.endsWith('/'+l.name));
      if(!hit){ row.remove(); return }
      sel.value=hit.value;
      row.querySelector('[data-f=unet]').value=l.unet??1;
      row.querySelector('[data-f=expert]').value=l.expert||'both';
    });
    if(it.negative_prompt||it.steps||it.cfg_scale) $('#vid-adv').classList.remove('hide');
    $('#prompt').focus();
  }
}

function metaSheet(it){
  const rows=[
    ['Kind',it.kind], ['Model',it.model], ['Job',it.job_id],
    ['Size',it.width&&`${it.width}×${it.height}`],
    ['Seed',(it.seeds||[]).join(', ')||it.seed],
    ['Steps',it.steps], ['Sampler',it.sampler], ['Scheduler',it.scheduler],
    ['CFG',it.cfg_scale], ['Shift',it.shift],
    ['Expert switch',it.switch_at&&`step ${it.switch_at}`],
    ['Length',it.seconds&&`${it.seconds}s · ${it.frames} frames · ${it.fps} fps`],
    ['References',it.references||it.ref_videos ? `${it.references||0} image, ${it.ref_videos||0} video` : ''],
    // `expert` only exists on the video stack, `applied` only on the image one.
    ['LoRAs',(it.loras||[]).map(l=>`${l.name} @ ${l.unet}`
      +(l.expert&&l.expert!=='both'?` (${l.expert} noise)`:'')
      +(l.applied===false?' (not applied)':'')).join(', ')],
    ['Regions',(it.regions||[]).map(r=>r.prompt).join(' | ')],
    ['Files',it.files.join(', ')],
    ['Created',it.created?new Date(it.created*1000).toLocaleString():''],
  ].filter(r=>r[1]!==undefined&&r[1]!==null&&r[1]!=='');
  const el=sheet(`
    <div class="sheet-head">
      <h1 class="grow">Metadata</h1>
      <button class="ico" data-close>${ICON.close}</button>
    </div>
    <label>Prompt</label>
    <textarea id="m-prompt" rows="5" readonly>${esc(it.prompt||'')}</textarea>
    ${it.negative_prompt?`<label style="margin-top:12px">Negative</label>
      <textarea rows="2" readonly>${esc(it.negative_prompt)}</textarea>`:''}
    <dl class="kv" style="margin-top:18px">
      ${rows.map(r=>`<dt>${esc(r[0])}</dt><dd>${esc(r[1])}</dd>`).join('')}
    </dl>
    <div class="row" style="gap:8px;margin-top:20px">
      <button class="s" id="m-copy">Copy prompt</button>
      <button class="s" id="m-reuse">Reuse settings</button>
      <span class="grow"></span>
      <button class="s" data-close>Close</button>
    </div>`);
  el.querySelector('#m-copy').onclick=async e=>{
    await navigator.clipboard.writeText(it.prompt||'');
    e.target.textContent='Copied';
  };
  el.querySelector('#m-reuse').onclick=()=>{ el.remove(); reuse(it) };
}

setKind('image');
setMode('generate');
loadState();
loadDatasets();
</script></body></html>
"""
