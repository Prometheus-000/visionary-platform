"""
Visionary — standalone Krea 2 LoRA trainer on Modal.

One file. One command. No Vercel, no npm, no Modal Secrets, no access keys.
The UI is served by Modal itself, so `modal deploy app.py` gives you a single
URL that is the whole application.

    modal deploy app.py

Training runs on musubi-tuner. Inference — images and video both — runs on
ComfyUI, driven over its API rather than ported, and pinned by commit. They are
deliberately separate images: ComfyUI wants newer transformers than musubi
pins, and coupling them would mean every ComfyUI bump re-litigates training.

Storage is ours, not borrowed. The volume is created on first deploy and the
layout is flat and self-describing — nothing here mirrors a checkout of another
project, so there is no directory that only makes sense to somebody who has
read a backend's source.

    $VISIONARY_VOLUME (default "visionary")  ->  /workspace
      models/krea2-raw.safetensors        Krea 2 RAW DiT   (training)
      models/krea2-turbo.safetensors      Krea 2 Turbo DiT (inference)
      models/qwen-image-vae.safetensors
      models/qwen3vl-4b-bf16.safetensors
      loras/{folder}/{name}.safetensors   trained output, any nesting
      outputs/{job}/                      generated images
      datasets/{name}/                    sets you saved
      drafts/{name}/                      sets you have not saved yet
      .cache/                             HF staging, never read directly

Set VISIONARY_VOLUME to run a second copy (staging, a different account) against
its own storage.

Nothing downloads on its own — pick what you want under the gear.
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
import threading
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
# A set you have not kept yet. Identical folder shape to a dataset — images
# with .txt sidecars — so captioning, the contact sheet and the trainer never
# learn which root a set came from, and saving one is a move rather than a
# conversion. The split is the whole meaning of "saved": what is under
# datasets/ is your library, and nothing else is promised to survive.
DRAFTS = WORKSPACE / "drafts"
WORK = WORKSPACE / "work"
OUTPUTS = WORKSPACE / "outputs"
STAGING = WORKSPACE / ".cache" / "hf-staging"

# Deletions used to move here instead of unlinking. The name survives only so
# `_drop_legacy_trash` can clear what earlier versions left on the volume; no
# code writes into it any more.
LEGACY_TRASH_DIR = ".trash"
THUMB_DIR = ".thumbs"
MUSUBI = Path("/opt/musubi-tuner")

# Both inference paths are Hopper now, and the reason is the image rather than
# the model: SageAttention is compiled for sm_90 in comfy_image, so a card of
# any other architecture loads the weights, finds no kernel, and silently runs
# on the slow path. Krea 2 used to have an A100-40GB of its own, sized against a
# measured 29.26 GiB peak at 1024px — that headroom does not survive V12, whose
# own regression notes record a 30.27 GiB block-mask build and a 17.88 GiB dense
# score tensor at the sequence lengths reference frames produce.
GPU = os.environ.get("VISIONARY_IMAGE_GPU", "H100")

# Video is still its own GPU class, and still for the original reason: the H3
# stack is 42.5 GB of weights before any activations, so it does not share a
# card with training or Krea 2. Sharing the *image* is new; sharing the
# container is not on the table while both hold a checkpoint resident.
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
# Both lists are Hopper-only on purpose: SageAttention is compiled for sm_90 in
# comfy_image. B200 is sm_100 and would load the model fine and then fall back
# off the fast kernels — the failure this list exists to prevent. Adding it
# means changing TORCH_CUDA_ARCH_LIST and forcing an image rebuild. The A100
# entries that used to be here went with the same rebuild: sm_80 is not sm_90.
IMAGE_GPUS = ("H100", "H200")
VIDEO_GPUS = ("H100", "H200")

# ComfyUI is the inference backend for both images and video, pinned by commit
# rather than vendored.
#
# It was the video backend first, and Krea 2 ran on a vendored Forge next door.
# What ended that split was regional prompting: it was the one feature Forge
# could do and ComfyUI could not, because Forge Couple's cross-attention design
# cannot reach a single-stream DiT and reaching it needed a patch to
# backend/nn/krea.py. CLIFF_SHA below is a node pack that does the same job
# through ComfyUI's own hooks and does it better — it masks each LoRA's
# activation delta to its box, so there is no pathway left for one character's
# identity to reach another's, where masking attention only makes it unlikely.
# With the one blocker gone, keeping a second backend, a second image and a
# second GPU class bought nothing.
#
# Why ComfyUI at all, when diffusers has a MiniMax-H3 pipeline: the diffusers
# integration runs the released bf16 weights, 123.6 GB across the transformer
# and the Qwen3-VL-32B conditioner. ComfyUI runs Comfy's repackage — modulation
# weights pruned into a lookup table, int8-convrot weights, and their own
# kernels — for 42.5 GB and int8 tensor-core matmuls instead of bf16 ones. On
# one card that is the difference between offloading every request and holding
# the model resident, on top of roughly 2x on the denoise loop itself.
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

# Regional multi-character LoRA for Krea 2, by a commit rather than a branch —
# the pack was pushed to twice in the week this landed, and a floating ref means
# the graph builder below can stop matching the node it builds for.
#
# Its whole surface on ComfyUI is public: it wraps the diffusion model through
# comfy.patcher_extension, swaps attention through the transformer_options
# `optimized_attention_override` hook, and loads LoRAs with
# comfy.sd.load_lora_for_models. All three exist at COMFY_SHA, which is what
# makes this an install rather than a vendor — nothing here is patched, so
# forge/VENDOR.md has no successor.
#
# One node out of the pack's six is wired up: V12. It subclasses V9, so V9's
# engine ships whether or not it is reachable, and exposing both would mean two
# sets of semantics on the page for one capability.
CLIFF_SHA = "a33a0e487fbaaa6b64a19ad6e26e707695723b75"  # 2026-08-01
CLIFF_REPO = (
    "https://github.com/CliffNodes/"
    "Krea2-Multi-Character-Lora-Node-with-bounding-box-Scene-and-Outfit-Edit"
)

# The V12 class id, spelled once. It is what /object_info is checked against at
# startup and what the graph names, and those two have to agree or the check is
# theatre.
KREA2_REGIONAL_NODE = "Krea2RegionalMultiLoRAV12"

# Our own node next to the pack's — one shim, see comfy_nodes/visionary_boxes.
COMFY_NODES_DIR = str(Path(__file__).parent / "comfy_nodes")

app = modal.App(APP_NAME)


# --------------------------------------------------------------------------
# Images — every dependency baked in, nothing installed at runtime
# --------------------------------------------------------------------------

# Plain `fastapi`, not `fastapi[standard]`: Modal serves the ASGI app itself, so
# the bundled uvicorn/typer/rich/httpx/jinja2/email-validator are all dead
# weight. Only FastAPI, Request, HTMLResponse and JSONResponse are used, which
# are core. python-multipart is listed explicitly because the upload form needs
# it and plain fastapi does not pull it.
web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi==0.115.6",
        "python-multipart==0.0.20",
        "pillow==11.1.0",
        "huggingface_hub[hf_transfer]==0.27.1",
        # Named as well as pulled by the extra above, matching caption_image:
        # the extra has been observed not to install it, and with the env var
        # below set its absence surfaces as a confusing load error rather than
        # as a missing dependency.
        "hf_transfer==0.1.9",
        # Google Drive, for LoRAs that were never on HuggingFace. Kept in this
        # image rather than its own because it conflicts with nothing here —
        # gdown pins only requests and beautifulsoup4 — and a separate image
        # would mean a second build for a 200 kB pure-python package. The rule
        # is to split images when pins fight, not when responsibilities differ.
        "gdown==5.2.0",
    )
    # The single biggest number in this file.
    #
    # This was deliberately left unset for a long time, on the grounds that
    # hf_transfer has no progress hook and a weaker resume story — the two
    # properties that once turned a stalled download into a four-hour silence.
    # Both halves of that were measured on this image against a 21 GB file
    # rather than reasoned about, and they did not survive it:
    #
    #   plain requests    30.6 MB/s     hf_transfer   243.8 MB/s
    #
    # 8x, and it is the whole gap between "over 200 MB/s with the hf CLI" and a
    # platform that took a working day to fetch its own weights. The progress
    # objection was already dead: `_staged_bytes` sums the tree, which is
    # indifferent to which backend wrote it, and is exactly how those numbers
    # were taken. The resume objection is real and confirmed — killed mid-file,
    # plain requests restarts at 0.29 of 0.29 GB and hf_transfer at 0.00 of
    # 5.09 — but it is now an objection to something that costs less than it
    # saves: at this speed the largest weight in the catalogue lands in under
    # two minutes, which is shorter than DOWNLOAD_STALL_S itself. Resume was
    # machinery for a fifteen-minute download, and there is no longer one.
    #
    # Falling back to the plain backend on a retry was measured too and does
    # work — it picks up hf_transfer's bytes rather than starting over — but it
    # is deliberately not done: it buys a resume for a two-minute transfer at
    # the price of an 8x slower one, and a second backend on the failure path
    # is a second set of behaviour that only ever runs when things are already
    # going wrong. Retries stay on hf_transfer and start over.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
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

# Captioning is a separate image for the same reason the others are:
# Qwen3VLForConditionalGeneration landed in transformers 4.57, which is newer
# than musubi's pins and newer than the version ComfyUI's own requirements
# resolve to. Pinning one transformers across all three would mean every
# captioner bump re-litigates both training and inference.
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


# Inference: ComfyUI on a CUDA 13 torch wheel. Images and video both.
#
# One image, two GPU classes. They stay separate classes because each holds a
# checkpoint resident and `max_containers=1` is per class — merging them would
# make an image request wait behind a ten-minute clip. They share the *build*
# because there is nothing per-family in it: the same ComfyUI, the same torch,
# the same attention kernels, and one node pack that only the image side loads.
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
comfy_image = (
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
    # TORCH_CUDA_ARCH_LIST must match every card in IMAGE_GPUS and VIDEO_GPUS.
    # 9.0 is Hopper (H100/H200); move to "10.0" for B200 and force a rebuild, or
    # the kernels will not load. This is also what keeps Krea 2 off the A100 it
    # used to run on — one image means one architecture, and sm_80 is not sm_90.
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
    # right. `tools/smoke_graphs.py` is what catches it if this regresses.
    .pip_install("pillow==11.3.0")
    # Regional multi-character LoRA for Krea 2. Cloned, not vendored, for the
    # reason given at CLIFF_SHA: nothing in it is patched.
    #
    # Cloned at a *full* SHA and then checked out, rather than --depth 1 on a
    # branch, because --depth 1 cannot reach a commit that is no longer the tip
    # — which is precisely the case a pin exists to survive. The clone is 356 kB
    # of Python either way.
    #
    # torch and safetensors are its only declared dependencies and both are
    # already here, so there is deliberately no pip install of its
    # requirements: running one would resolve torch again against a CUDA
    # default and undo the cu130 wheel three steps up. `ultralytics` is left
    # out with the Detailer node it belongs to.
    .run_commands(
        f"git clone {CLIFF_REPO} {COMFY}/custom_nodes/krea2_regional",
        f"cd {COMFY}/custom_nodes/krea2_regional && git checkout {CLIFF_SHA}",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    # Last, because add_local_dir invalidates nothing above it — the shim is
    # the file most likely to change and rebuilding SageAttention to edit
    # twenty lines of Python would be its own kind of failure.
    #
    # Mounted at the package itself, not at custom_nodes/. The default
    # (copy=False) attaches this as a mount at container startup rather than an
    # image layer, so pointing it one level up would overlay the directory the
    # clone above writes into and the pack would vanish at runtime while the
    # build log still showed it being cloned.
    .add_local_dir(
        f"{COMFY_NODES_DIR}/visionary_boxes",
        remote_path=f"{COMFY}/custom_nodes/visionary_boxes",
    )
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
    # The one weight that is optional and still belongs here.
    #
    # It lands in loras/ rather than models/ because that is what it is — the
    # regional node takes it through the same `folder_paths` lookup as a
    # character LoRA, and putting it anywhere else would mean teaching the
    # node's combo about a second directory. The cost is that it appears in the
    # LoRA picker like any other file, which is honest: you can write
    # `<lora:krea2_identity_edit_v1_2:1>` and it will do something.
    #
    # Only the scene and outfit transfer path loads it. Every other render
    # names it in V12's required `edit_lora` slot and never opens it — see
    # `_edit_lora_name` for why that slot cannot simply be left empty.
    #
    # The filename is load-bearing: it is what the node pack's own fallback
    # spells, so KREA2_EDIT_LORA matches it exactly and the two must move
    # together.
    "krea2_edit": {
        "label": "Krea 2 Identity Edit",
        "note": "Optional — scene and outfit transfer",
        "family": "Krea 2 — images",
        "repo_id": "conradlocke/krea2-identity-edit",
        "filename": "krea2_identity_edit_v1_2.safetensors",
        "dest": LORAS / "krea2_identity_edit_v1_2.safetensors",
        "gated": False,
        "approx_gb": 1.8,
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

# Krea's own style LoRAs — nine repos, one 0.47 GB file each, ungated.
#
# Built by loop rather than written out nine times because they differ in two
# fields and agree on the other six; the alternative is ninety lines where the
# only thing worth reading is a name and a trigger. This is the one place in
# the catalogue where that is true — every other entry differs in repo, path,
# size and family at once, and a loop over those would hide more than it saved.
#
# Each lands as a *loose file at the top of loras/*, not in a shared folder.
# That is the storage contract doing real work: a folder is one LoRA and the
# files in it are that LoRA's epochs, so `loras/krea-styles/` holding nine
# unrelated styles would collapse to a single picker row offering nine
# "epochs" of a LoRA that does not exist. Loose, each is its own row and its
# own `<lora:darkbrush:1>`.
#
# They are here because regional multi-character LoRA is the hardest thing
# this app does to *show*. Two character LoRAs in two boxes produce a picture
# of two people, and nothing in that picture distinguishes "each LoRA was
# masked to its rectangle" from "the model drew two people". Two styles do:
# ink wash on the left, motion blur on the right and a hard seam between them
# is the activation delta being zeroed outside the box, which is the actual
# claim. A first-party, ungated set that anyone can download is what makes
# that demonstrable on someone else's install rather than only on this one.
KREA_STYLE_LORAS: dict[str, str] = {
    "darkbrush": "monochrome ink wash style",
    "retroanime": "Purple retro anime style",
    "vintagetarot": "vintage tarot style",
    "sunsetblur": "ethereal motion blur style",
    "softwatercolor": "Art Deco watercolor style",
    "neondrip": "Textured abstract style",
    "dotmatrix": "Monochrome stippling style",
    "kidsdrawing": "naive expressive sketch style",
    "rainywindow": "rainy window style",
}
MODEL_CATALOGUE.update({
    f"krea_style_{name}": {
        "label": name,
        # The trigger, not a description of the look. It is the thing you have
        # to type for the weight to do anything, and the catalogue card is the
        # only place on the page it appears before you have downloaded it.
        "note": trigger,
        "family": "Krea 2 style LoRAs",
        "repo_id": f"krea/Krea-2-LoRA-{name}",
        "filename": f"{name}.safetensors",
        "dest": LORAS / f"{name}.safetensors",
        "gated": False,
        "approx_gb": 0.47,
    }
    for name, trigger in KREA_STYLE_LORAS.items()
})

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

def _catalogue_lora_roots() -> dict[str, str]:
    """
    {listing row -> family} for every catalogue weight that lands in loras/.

    Keyed on the row rather than on the dest, because those are not the same
    thing: the identity-edit weight is a loose file at the top of loras/ and each
    Wan speed pair is a folder holding two files, and it is the folder that one
    line of the LoRA listing stands for.

    What reads it is the delete confirmation. Deleting a LoRA you trained costs
    however many hours the run took and there is nothing behind the unlink;
    deleting one of these costs a download, and the dialog is the only place that
    difference can be said *before* the fact rather than discovered after.
    """
    out: dict[str, str] = {}
    for spec in MODEL_CATALOGUE.values():
        dest: Path = spec["dest"]
        if LORAS not in dest.parents:
            continue
        root = dest
        while root.parent != LORAS:
            root = root.parent
        out[str(root)] = spec["family"]
    return out


CATALOGUE_LORA_ROOTS = _catalogue_lora_roots()

# The three weights musubi is handed on the command line. Turbo used to have an
# alias here too, for the Forge pipeline that took absolute paths; ComfyUI's
# loaders take a basename inside a search path, so the graph builders read
# `MODEL_CATALOGUE[...]["dest"].name` and nothing needs a fourth constant.
RAW_PATH = MODEL_CATALOGUE["raw"]["dest"]
VAE_PATH = MODEL_CATALOGUE["vae"]["dest"]
TE_PATH = MODEL_CATALOGUE["text_encoder"]["dest"]

# Qwen3-VL, not a booru tagger and not JoyCaption.
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
#
# Which *checkpoint* of it is a choice, because a refusal is not an error here.
# The stock instruct model declines on photographs of real people often enough
# to matter — "I can't identify or describe individuals in images" — and on a
# character set that is every image. What arrives is not an exception: it is a
# fluent, well-formed sentence that goes straight into a `.txt` sidecar and then
# into a training run, so the failure surfaces as a LoRA that learned to say it
# cannot describe someone. The abliterated repackage has the refusal direction
# removed and is otherwise the same architecture and the same loader, so it
# costs a repo id rather than a second code path. `_looks_like_refusal()` is the
# other half: a caption that declines is never written, whichever model wrote it.
CAPTION_MODELS: dict[str, dict[str, str]] = {
    "qwen3vl": {
        "repo": "Qwen/Qwen3-VL-8B-Instruct",
        "label": "Qwen3-VL 8B",
        "note": "The text encoder's own family, stock.",
    },
    "qwen3vl-uncensored": {
        "repo": "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2",
        "label": "Qwen3-VL 8B uncensored",
        # The size is in the note because this is the one control on the page
        # that can start a 17 GB download without saying so: the captioner is
        # pulled into the HF cache on first use, not chosen under the gear.
        "note": "Same weights, refusal removed. First run pulls ~17 GB.",
    },
}
DEFAULT_CAPTION_MODEL = "qwen3vl"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif"}
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_CAPTION_CHARS = 1024
THUMB_PX = 320

# What every caption obeys regardless of preset. Factored out rather than
# repeated five times because these are the sentences the *parser* below
# depends on — a preamble the model was never told to drop is a preamble
# `_caption_images` has to strip by prefix, and that list only ever grows.
CAPTION_RULES = (
    " Write continuous prose in plain declarative English: no list, no bullet "
    "points, no markdown, no headings, no labels. Do not open with 'This image', "
    "'The photo', 'Here is' or any other preamble. Do not editorialise — no "
    "'stunning', 'conveying a sense of', 'evoking a mood of'. Do not speculate "
    "about anything you cannot see. Output the caption itself and nothing else."
)

# The instruction is the product, not the model.
#
# A preset is a training intent, and the intent decides the one thing that
# matters: *what to leave out*. A caption teaches the model that whatever it
# names is free to vary, and whatever it never names belongs to the trigger
# word — so a character set that describes the jaw and the eye colour has spent
# its trigger on a face the captions already supply, and a style set that says
# "oil painting, thick impasto" has handed the model a phrase to hang the look
# on instead of the token you are training. Same rule, inverted per preset.
#
# They are here rather than in the page for the reason `SHOT_VOCAB` is: the page
# should send `character`, not four hundred words of instruction it could edit
# into something the run cannot reproduce. What the page shows is the label and
# the note.
CAPTION_PRESETS: dict[str, dict[str, str]] = {
    "general": {
        "label": "General",
        "note": "Everything in frame, each adjective bound to its noun.",
        "instruction": (
            "Describe this image in plain, factual prose. Name the subject and what "
            "it is doing, then its appearance, clothing, pose, setting, lighting and "
            "style. Attach every adjective to the noun it belongs to, so it is "
            "unambiguous which garment or object each colour and material describes."
        ),
    },
    "character": {
        "label": "Character",
        "note": "Describes everything except the face. Identity is the trigger's job.",
        "instruction": (
            "Write a training caption for this photograph of a recurring person. "
            "Describe only what changes from shot to shot: pose and body position, "
            "gaze direction, facial expression, shot type (close-up, medium, full "
            "body), camera angle, clothing and accessories, hair when it is styled "
            "differently, the setting behind them, the quality and direction of the "
            "light, and anyone or anything else in frame. "
            "Refer to the subject with a plain class noun — the woman, the man, the "
            "person — and never invent a name. "
            "Never describe permanent identity: face shape, eye colour, nose, jaw, "
            "skin tone, age, ethnicity, build, or how attractive they are. Those are "
            "constant across the set and naming them teaches the model they are free "
            "to vary. "
            "Do name anything you would want to remove later: a watermark, text "
            "overlay, motion blur, harsh on-camera flash, a hand at the edge of "
            "frame, a cluttered background. Anything named can be prompted away; "
            "anything unnamed is baked into the character."
        ),
    },
    "style": {
        "label": "Style",
        "note": "Describes the content, never the look. The look is the trigger.",
        "instruction": (
            "Write a training caption for an image in a recurring visual style. "
            "Describe what the image is *of*: the subject, what it is doing, the "
            "composition and framing, where things sit relative to each other, and "
            "the setting. "
            "Say nothing about how it is rendered — do not name the medium, the "
            "brushwork, line quality, palette, grain, colour grade, era, artist, or "
            "any word for the look itself such as cinematic, painterly, anime or "
            "retro. Those are the style you are training, and a caption that names "
            "them gives the model a phrase to hang the look on instead of the trigger "
            "word. "
            "Do name anything incidental you would want to prompt away later, such as "
            "a watermark, signature, border or caption text."
        ),
    },
    "concept": {
        "label": "Concept",
        "note": "For an object, garment or pose — describes the context around it.",
        "instruction": (
            "Write a training caption for an image of a recurring object, garment, "
            "pose or effect. "
            "Describe everything around it: the scene, who is holding or wearing it, "
            "the angle it is seen from, its scale relative to the frame, what else is "
            "present, the lighting and the background. "
            "Refer to the concept itself with the shortest plain noun that fits, and "
            "say nothing about what makes it distinctive — its shape, markings, "
            "colour scheme, materials or construction. Those are constant across the "
            "set and belong to the trigger word. "
            "Do name anything incidental you would want to prompt away later."
        ),
    },
    "casual": {
        "label": "Casual",
        "note": "Conversational prose, none of the dataset rules applied.",
        "instruction": (
            "Describe this image in natural, conversational prose, the way you would "
            "describe a photo to someone who cannot see it. Cover what is happening, "
            "how it looks and the overall mood, keeping each adjective clearly "
            "attached to the thing it describes."
        ),
    },
}
DEFAULT_CAPTION_PRESET = "general"

CAPTION_LENGTHS = {
    "short": " Keep it to one dense sentence.",
    "medium": " Keep it to two or three sentences.",
    "long": " Be thorough, four or more sentences.",
}

# Anchored at the start, like `prepend_trigger`'s `startswith` and for the same
# reason: "I cannot" halfway through a caption is a sentence about the picture
# ("a sign reads I CANNOT"), while a caption that *opens* this way is a model
# talking about itself. A substring test would throw away real captions.
REFUSAL_RE = re.compile(
    r"^\W*(?:i(?:'|’)?m sorry|i am sorry|sorry[,.]|i (?:can(?:'|’)?t|cannot|won(?:'|’)?t)\b"
    r"|i(?:'|’)?m (?:not able|unable)|i am (?:not able|unable)|unable to\b"
    r"|as an ai\b|i (?:don(?:'|’)?t|do not) (?:have the ability|feel comfortable))",
    re.I,
)


def _looks_like_refusal(caption: str) -> bool:
    """A decline is a well-formed sentence, so nothing downstream would catch it."""
    return bool(REFUSAL_RE.match(caption.strip()))


def _caption_instruction(preset: str, length: str, trigger_word: str) -> str:
    """
    One prompt out of the preset, the length and the trigger word.

    The trigger clause is added here rather than written into each preset
    because it is a fact about *this run* — the token is prepended in Python
    once the caption comes back, so the model has to be told both that the
    subject has a name and that writing it would double it.
    """
    spec = CAPTION_PRESETS.get(preset) or CAPTION_PRESETS[DEFAULT_CAPTION_PRESET]
    out = spec["instruction"]
    if trigger_word:
        out += (
            f" Every caption in this set is prefixed with the trigger word "
            f"'{trigger_word}' automatically, so never write '{trigger_word}' yourself."
        )
    out += CAPTION_LENGTHS.get(length, CAPTION_LENGTHS["medium"])
    return out + CAPTION_RULES


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
        # Locked, because two threads write this record and both of them do
        # get-update-put against a *network* Dict — a window wide enough to
        # lose a write on most runs rather than rarely. The job thread
        # publishes phases and the terminal result; `_Comfy._drain` publishes
        # step counts as ComfyUI's tqdm line scrolls past. Interleaved, the
        # drain reads {step: 6}, the job thread reads {step: 6} and writes
        # {step: 6, phase: ...} over the drain's {step: 7}, and the bar walks
        # backwards — which is what "the progress bar goes nuts" was, on the
        # server, where no amount of client-side care could reach it.
        #
        # The same interleaving on the last line is worse than cosmetic. A tqdm
        # write that read the record before the job finished puts the whole
        # stale dict back, `status: "running"` included, over a `completed`
        # that was already there — and the page then polls a finished job until
        # someone reloads it. Rare, and the one that costs a result.
        with _PUBLISH_LOCK:
            cur = jobs.get(job_id) or {}
            cur.update(fields)
            # When this record last spoke. A status is a claim about a container
            # that may no longer exist — the Dict is named and outlives every
            # container, app and deploy that writes to it — so the claim is only
            # worth what its age says. See `_download_alive`.
            cur["beat"] = time.time()
            jobs[job_id] = cur
    except Exception as exc:  # progress must never take the job down
        print(f"[progress] {job_id}: {exc}")


# Process-local, which is all it needs to be: a job record is written by the
# one container running that job, and `max_containers=1` on the GPU classes
# means there is no second writer to coordinate with.
_PUBLISH_LOCK = threading.Lock()


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
                # Same normalisation the loose-file path does. A zip is the more
                # likely carrier of camera originals, not the less.
                _upright_inplace(dest / name)
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
        "failed download — check `modal profile current` and Settings.",
    ]
    raise RuntimeError("\n".join(lines))


def _reload_volume() -> bool:
    """
    Bring the volume forward. Returns False if it had to be skipped.

    Modal refuses a reload while anything on the volume is still open, and a
    container that has loaded a checkpoint always has something open: safetensors
    maps the weights straight off /workspace and the mapping outlives the file
    descriptor, so the kernel is still holding /workspace long after the loader
    returned. The first `generate` in a fresh container reloaded fine because
    nothing was loaded yet, and every request after it raised — one image per
    container, then `ConflictError: there are open files preventing the
    operation` for the rest of the container's life.

    Raising there is the wrong trade. A reload is a *freshness* step: it exists
    so a LoRA trained ten minutes ago is visible to a warm container. Skipping it
    costs a stale listing until that container scales down; raising cost every
    generation after the first. Everything the job is about to open was already
    validated by the web container, which holds nothing open and reloads on
    every request, so the reload here is the second look, not the only one.

    Only the open-files conflict is absorbed. Any other reload failure is still
    an error, because it says something about the volume rather than about what
    this container happens to be holding.

    Serialised, because two of these at once is not two reloads — it is one
    reload and one 500. `/api/file` has always noted that concurrent reloads of
    the same volume returned an error for one of two simultaneous requests, and
    the canvas now asks for a batch of stills at once, all of which miss on the
    first look at a brand-new job. The lock turns that from a race into a queue.

    And a queue of twelve identical reloads is the wrong end of that trade,
    which is the half this used to be missing. A gallery is a grid and a canvas
    is a batch, so the misses arrive together and every one of them wanted the
    same thing: a view newer than the moment it asked. One reload gives that to
    all of them. Twelve gives it to all of them too, eleven of them twice over,
    while the browser's connection budget sits inside this function and the
    `<video>` waiting on a range request behind it stutters.

    So the sequence number is read *before* queueing: anyone who finds it moved
    by the time they hold the lock has been overtaken by a reload that began
    after they asked, which is the freshness they came for. Anyone who started
    theirs earlier than we asked does not count, so the check is `>` and a
    caller who arrives mid-reload still runs its own.

    That is one reload per in-flight window rather than one overall, and the
    difference is the whole correctness argument: a reload that began before
    you asked cannot be evidence that the file the GPU container wrote after
    that is visible. A burst of twelve costs two — one for whoever asked before
    it started, one for whoever asked during it — and never twelve. Measured on
    a 0.25s reload: 3.00s of queue becomes 0.51s.
    """
    global _RELOAD_SEQ
    asked_at = _RELOAD_SEQ
    with _RELOAD_LOCK:
        if _RELOAD_SEQ > asked_at:
            return _RELOAD_OK
        # Bumped before the call, not after: it marks when a reload *began*,
        # which is what the comparison above is asking about.
        _RELOAD_SEQ += 1
        return _remember_reload()


def _remember_reload() -> bool:
    """Reload and record the answer, so a coalesced caller can return it too."""
    global _RELOAD_OK
    _RELOAD_OK = _reload_volume_locked()
    return _RELOAD_OK


_RELOAD_LOCK = threading.Lock()
# Reloads begun, and what the last one answered. `_RELOAD_OK` matters because
# the skip path is not "assume it worked" — a reload that was skipped for open
# files returns False, and a caller riding on it has to hear the same thing or
# the coalescing quietly upgrades a stale view into a fresh one.
_RELOAD_SEQ = 0
_RELOAD_OK = True


def _reload_volume_locked() -> bool:
    try:
        volume.reload()
        return True
    except RuntimeError as exc:
        if "open files" not in str(exc):
            raise
        # Not silent: a stale view is a real cause of "that LoRA is not there",
        # and this line is the only thing that distinguishes it from a typo.
        print(f"[volume] reload skipped, weights still mapped ({exc})", flush=True)
        return False


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
    # Merged, not assigned. The route seeds this record before spawning, and a
    # Cancel pressed during the cold start writes `stop` into it — a plain
    # assignment here would erase that request seconds before the transfer it
    # was meant to stop begins. `stop` is deliberately not re-set: absent reads
    # as False, so there is nothing to clobber.
    _publish(job_id, status="running", percent=0,
             phase=f"Downloading {MODEL_CATALOGUE[key]['label']}")
    try:
        return _download_weight(key, job_id)
    except StopRequested:
        return {"status": "stopped", "key": key}


# How long a transfer may make no progress at all before it is called dead, and
# how many times it may resume. Both exist because of one failure: Krea 2 Turbo
# stopping at 4 GB of 17 and the job staying "running" — no error, no log line,
# no byte count — until the four-hour timeout collected it. A download that can
# hang is survivable; a download that can hang *silently* costs you the four
# hours before you learn anything, and it can do it again the next day.
DOWNLOAD_STALL_S = 240
DOWNLOAD_TRIES = 5

# How long a record may claim to be running without saying so again before it is
# read as a corpse. `_watch_download` publishes every 5s whether or not bytes
# moved, so a live transfer — even a stalled one — beats continuously; only a
# container that no longer exists goes quiet. Generous against the other end of
# the window: `.spawn()` returns before the container starts, and a cold start
# plus `_reload_volume()` plus the first `done.wait(5)` has to fit inside this.
DOWNLOAD_DEAD_S = 120


def _download_alive(job_id: str) -> bool:
    """
    Whether a job record claiming "running" belongs to a container still alive.

    `jobs` is a *named* Modal Dict, so it outlives the container, the app, the
    deploy and the image rebuild. A container killed mid-transfer — `modal app
    stop`, a redeploy, a preemption — never reaches any of its own terminal
    paths, so its record stays "running" for good. Trusting that field alone
    turned the concurrency guard below into a permanent lockout: three ghosts
    left over from one stopped app refused every download that came after,
    and rebuilding the image could not clear them because the Dict is not part
    of the image.

    So a status is only believed as far as its `beat`. A stale one is rewritten
    to failed here rather than merely ignored, because the UI polls this same
    record: leaving it would clear the lock while still showing a download that
    is "running" and will never finish or fail. Records written before `beat`
    existed have none, which dates them as older than any live job — which is
    exactly what they are.
    """
    rec = jobs.get(job_id) or {}
    if rec.get("status") != "running":
        return False
    if time.time() - float(rec.get("beat") or 0.0) <= DOWNLOAD_DEAD_S:
        return True
    _publish(
        job_id,
        status="failed",
        error=(f"The container running this download went away without finishing "
               f"— stopped, redeployed or preempted. It last reported "
               f"{rec.get('downloaded_gb', 0)} GB. Start it again when you are "
               f"ready; it downloads the file from the beginning."),
    )
    print(f"[download] {job_id}: stale record cleared (no beat for "
          f"{int(time.time() - float(rec.get('beat') or 0.0))}s)")
    return False


# Which download job is the live one. A pointer rather than a scan, because the
# scan was two bugs: `jobs` is a network Dict, so walking `dl_{key}` across the
# whole catalogue is 20-odd sequential round trips, and it made `/api/download`
# take seven seconds to answer. A button that does nothing for seven seconds is
# a button you press again — which is exactly how the first report of this
# arrived, and the second press is the thing the check existed to prevent.
DL_ACTIVE = "dl_active"


def _family_job_id(family: str) -> str:
    """
    A stable job id for one family's queue, derived from its name.

    Derived rather than passed in by the page: the id is what Cancel and every
    status poll address, so it has to survive a reload, and a position in a list
    does not — adding a model to the catalogue would silently repoint it at a
    different family. The page never builds one; it uses whatever `job_id` the
    route hands back.
    """
    return "dl_fam_" + (re.sub(r"[^a-z0-9]+", "-", family.lower()).strip("-")[:48] or "x")


def _active_download() -> str | None:
    """
    The job id of the weight download running now, or None.

    Two reads whatever the catalogue grows to: the pointer, then the liveness
    of the one record it names. A pointer left behind by a container that died
    is handled by `_download_alive` rather than by trusting the pointer, so the
    two can never disagree for longer than DOWNLOAD_DEAD_S.
    """
    job_id = (jobs.get(DL_ACTIVE) or {}).get("job_id")
    if not job_id:
        return None
    if _download_alive(job_id):
        return job_id
    jobs[DL_ACTIVE] = {}
    return None


def _staged_bytes(root: Path) -> int:
    """
    Bytes on disk under the staging dir, incomplete parts included.

    Polled rather than hooked because `hf_hub_download` offers no progress
    callback, and the shape of what it writes — `.incomplete` parts, blobs,
    per-version subdirectories — has changed across releases. Summing the tree
    is indifferent to all of that, and to whether hf_transfer or the plain
    requests backend is doing the writing.
    """
    total = 0
    for p in root.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            # A part file renamed out from under the walk is normal, not an error.
            continue
    return total


def _watch_download(
    root: Path, label: str, job_id: str, done: "threading.Event",
    expect_gb: float | None = None, note: str = "",
    pct_base: float = 0.0, pct_span: float = 100.0,
) -> dict[str, Any]:
    """
    Publish transfer progress until `done` is set, and record a stall or a
    stop request.

    Returns the live state dict so the caller can read `stalled`, `stopped`
    and `bytes` after giving up — the numbers are the whole point, since "it
    stopped" and "it stopped at 4.1 GB of 17.2 after 6 minutes of nothing"
    send you to completely different places.

    `expect_gb` is None when the size is not known ahead of time, which is the
    normal case for anything off Google Drive. The byte count and the stall
    detection are what actually matter and neither needs a total; only the
    percentage does, and a percentage invented from a guess is worth less than
    no percentage at all.
    """
    n0 = _staged_bytes(root)
    state = {"bytes": n0, "moved_at": time.time(), "stalled": False, "stopped": False}
    expect = int((expect_gb or 0) * 1e9)
    logged = 0.0
    # Movement in either direction, not a high-water mark. The tree can shrink:
    # hf_transfer truncates a partial rather than ranging over it, so the first
    # thing a restarted attempt does is make this number smaller. Against a
    # high-water mark that reads as "no new bytes" for as long as it takes to
    # climb back — which is a live transfer at full speed being timed out for a
    # stall it is not having. A byte count that changed is a container doing
    # something, whichever way it went.
    # Windowed, not cumulative-since-this-attempt-started: a resumed attempt
    # begins with several GB already on disk from before, so dividing the
    # total by the few seconds this attempt has been alive reports a number
    # with nothing to do with the network — high at first, then decaying
    # toward the real rate as the denominator catches up. That decay reads
    # exactly like a slowing transfer even when the transfer itself is
    # steady, which is what sent a fresh, healthy 3-way concurrent pull
    # through the logs looking like it was dying. Tracking only what arrived
    # since the last poll reports what is actually happening right now.
    prev_n, prev_t = n0, state["moved_at"]

    while not done.wait(5):
        now = time.time()
        n = _staged_bytes(root)
        if n != state["bytes"]:
            state["bytes"], state["moved_at"] = n, now
        elif now - state["moved_at"] > DOWNLOAD_STALL_S:
            state["stalled"] = True
            return state

        if _stop_requested(job_id):
            state["stopped"] = True
            return state

        gb = n / 1e9
        # Floored at zero for the same shrinking tree. A truncation is not a
        # transfer running backwards at 820 MB/s, which is what the raw delta
        # published — one window of "0.0 MB/s" is the honest reading of a
        # window that delivered nothing.
        rate = max(0.0, n - prev_n) / max(1.0, now - prev_t) / 1e6
        prev_n, prev_t = n, now
        size = f"{gb:.1f} of {expect_gb:.1f} GB" if expect else f"{gb:.1f} GB"
        fields: dict[str, Any] = {
            "phase": " · ".join(x for x in (label, note, size) if x),
            "downloaded_gb": round(gb, 2),
            "mb_s": round(rate, 1),
        }
        if expect:
            # Mapped into this file's slice of the run, so the bar on a
            # four-model pull crosses the window once instead of snapping back
            # to zero four times. Capped just short of the slice's end: 100%
            # belongs to the completion path, not to a byte count that is still
            # an estimate against approx_gb.
            fields["percent"] = min(99, int(pct_base + pct_span * min(1.0, n / expect)))
        _publish(job_id, **fields)
        # Every 30s to the container log too. The UI record is what you watch
        # live; the log is what you still have tomorrow when you are asking why
        # last night's pull died.
        if now - logged > 30:
            logged = now
            stuck = int(now - state["moved_at"])
            print(f"[download] {label}: {gb:.2f} GB  {rate:.1f} MB/s"
                  + (f"  (no new bytes for {stuck}s)" if stuck > 20 else ""))
    return state


def _download_weight(
    key: str, job_id: str, note: str = "",
    pct_base: float = 0.0, pct_span: float = 100.0,
) -> dict[str, Any]:
    """
    Fetch one weight to its exact destination. Shared by single and bulk downloads.

    `note` is the queue position ("2 of 4") when this is one of a run. It is
    threaded down to the watcher rather than published once before the transfer
    starts, because the watcher rewrites `phase` every five seconds and would
    otherwise erase the only thing telling you how much of the queue is left.
    """
    import threading

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import (
        EntryNotFoundError, GatedRepoError, RepositoryNotFoundError,
    )

    spec = MODEL_CATALOGUE[key]
    dest: Path = spec["dest"]

    _reload_volume()
    if dest.exists() and dest.stat().st_size > 0:
        res = {"status": "completed", "key": key, "note": "already present"}
        _publish(job_id, **res)
        return res

    token = _hf_token()
    if spec["gated"] and not token:
        err = (
            f"{spec['label']} is a gated repo. Paste your HuggingFace token under the "
            f"gear, and accept the licence at https://huggingface.co/{spec['repo_id']}"
        )
        _publish(job_id, status="failed", error=err)
        raise RuntimeError(err)

    dest.parent.mkdir(parents=True, exist_ok=True)
    STAGING.mkdir(parents=True, exist_ok=True)
    started = time.time()
    label = spec["label"]
    print(f"[download] {label}: {spec['repo_id']}/{spec['filename']}")

    # The four permanent failures, hoisted out of the retry loop: a gated repo,
    # a wrong repo id and a wrong filename are answers, not weather. Retrying
    # them four more times only delays the sentence that tells you what to fix.
    def fatal(exc: Exception) -> str | None:
        if isinstance(exc, GatedRepoError):
            return (f"Access to {spec['repo_id']} was refused. Accept the licence at "
                    f"https://huggingface.co/{spec['repo_id']} using the same account "
                    "that issued this token.")
        if isinstance(exc, RepositoryNotFoundError):
            return f"Repo {spec['repo_id']} not found, or the token cannot see it."
        if isinstance(exc, EntryNotFoundError):
            return f"{spec['filename']} is missing from {spec['repo_id']}."
        return None

    # Takes its sink, its flag and its directory as arguments rather than
    # closing over them. An abandoned attempt's thread is still alive and still
    # going to finish eventually; with a closure it would write its stale result
    # into the *next* attempt's dict and set the next attempt's event — handing
    # the retry the answer to the question before it. The directory is on that
    # list for the same reason and was the one that got away: a straggler kept
    # writing into the shared staging path, so its bytes were counted as the
    # next attempt's progress. Every attempt gets its own of all three, so a
    # straggler lands somewhere nobody is reading.
    def pull(sink: dict[str, Any], flag: "threading.Event", where: Path) -> None:
        try:
            sink["path"] = hf_hub_download(
                repo_id=spec["repo_id"],
                filename=spec["filename"],
                local_dir=str(where),
                token=token,
            )
        except Exception as exc:  # re-raised on the calling thread below
            sink["error"] = exc
        finally:
            flag.set()

    # Nothing carries over between attempts.
    #
    # Keeping the partial used to be the entire point — the plain backend ranged
    # over it, so a resume cost the bytes since the stall rather than the 4 GB
    # already down. hf_transfer discards it regardless, so what is left is not a
    # head start: it is dead weight on a volume whose only other way to reclaim
    # space is the Modal CLI, and a false floor under every number published
    # here. `_staged_bytes` sums a tree, so 9.6 GB of abandoned Krea 2 RAW
    # counted as bytes the next attempt had downloaded — which is where
    # "882 MB/s" five seconds into a transfer came from, and then a rate of
    # -820 MB/s when hf_transfer truncated it. Measuring from an empty directory
    # is what makes the rate the rate.
    shutil.rmtree(STAGING, ignore_errors=True)
    STAGING.mkdir(parents=True, exist_ok=True)

    staged: str | None = None
    last_err = ""
    for attempt in range(1, DOWNLOAD_TRIES + 1):
        stage = STAGING / f"attempt-{attempt}"
        stage.mkdir(parents=True, exist_ok=True)

        # The download runs in a worker while this thread watches the bytes
        # land. It has to be this way round: hf_hub_download is a blocking call
        # with no callback and no cancellation, so the only way to put a bound
        # on it is to stop waiting for it. The thread is a daemon and is
        # abandoned on a stall or a stop request — the container is torn down
        # when this function returns, and a leaked socket for those few
        # seconds is a far smaller cost than four hours of a job that will
        # never finish, or one nobody wanted running anymore.
        result: dict[str, Any] = {}
        done = threading.Event()
        threading.Thread(target=pull, args=(result, done, stage), daemon=True).start()
        state = _watch_download(stage, label, job_id, done, spec["approx_gb"],
                                note, pct_base, pct_span)

        if state.get("stopped"):
            print(f"[download] {label}: stop requested at "
                  f"{state['bytes'] / 1e9:.2f} GB — attempt {attempt} abandoned")
            _publish(job_id, status="stopped",
                     phase=f"{label} · cancelled at {state['bytes'] / 1e9:.2f} GB")
            raise StopRequested(label)

        if state["stalled"]:
            last_err = (f"stalled at {state['bytes'] / 1e9:.2f} GB with no new bytes "
                        f"for {DOWNLOAD_STALL_S}s")
            print(f"[download] {label}: {last_err} — attempt {attempt} abandoned")
            _publish(job_id, phase=f"{label} · stalled, restarting ({attempt} of {DOWNLOAD_TRIES})")
            # "Restarting", not "resuming": this image runs hf_transfer, which
            # was measured discarding a 5.09 GB partial rather than ranging over
            # it. That is the accepted cost of the 8x — see web_image — and it
            # is what keeps DOWNLOAD_STALL_S where it is rather than tuning it
            # down to match a two-minute transfer. A stall detector that fires
            # early used to cost the bytes since the stall; it now costs all of
            # them, so it stays conservative.
            continue

        exc = result.get("error")
        if exc is not None:
            msg = fatal(exc)
            if msg:
                _publish(job_id, status="failed", error=msg)
                raise RuntimeError(msg)
            last_err = f"{type(exc).__name__}: {exc}"
            print(f"[download] {label}: attempt {attempt} failed — {last_err}")
            _publish(job_id, phase=f"{label} · retrying ({attempt} of {DOWNLOAD_TRIES})")
            continue

        staged = result.get("path")
        break

    if not staged:
        # Names the file, the byte count and the number of attempts, because
        # those three separate a dead uplink from a wrong path from a repo that
        # is simply refusing to serve this one file.
        err = (f"{label} did not finish after {DOWNLOAD_TRIES} attempts — {last_err}. "
               f"Each attempt restarts the file, so this is {DOWNLOAD_TRIES} full "
               f"tries that got nowhere rather than {DOWNLOAD_TRIES} nudges at a "
               f"partial one: suspect the repo or the uplink, not the last mile.")
        _publish(job_id, status="failed", error=err)
        raise RuntimeError(err)

    # Same filesystem, so this is an instant rename rather than a 26 GB copy.
    shutil.move(staged, dest)
    # Then the attempt directories, including any a straggler is still filling.
    # Nothing here is a resume any more, so anything left is pure occupancy on a
    # volume that needs the Modal CLI to reclaim space — and this ran once with
    # 9.6 GB of a download nobody was waiting for still sitting in it.
    shutil.rmtree(STAGING, ignore_errors=True)
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
def download_missing_job(keys: list[str], job_id: str = "dl_all") -> dict[str, Any]:
    """
    Fetch a list of missing weights in one container, sequentially.

    Sequential rather than four parallel containers: these are large files
    sharing one uplink, so running them at once mostly splits the same bandwidth
    while multiplying container cost — and it gives the UI a single job to
    follow instead of four independent ones.

    `job_id` is a parameter because "every missing weight" and "this family's
    missing weights" are the same walk over a different list. A second function
    for the second one would be a second copy of the queue, the failure
    accounting and the stop path, to no end — which is why one family's button
    reaches this and not something new.
    """
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
            _download_weight(key, job_id, note=f"{i} of {len(keys)}",
                             pct_base=(i - 1) * 100 / len(keys),
                             pct_span=100 / len(keys))
            done.append(key)
        except StopRequested:
            # The whole queue stops, not just this key — cancelling "download
            # all" means getting the uplink back, not skipping ahead to the
            # next multi-GB file.
            print(f"[download-all] stop requested — {len(done)} of {len(keys)} done")
            res = {"status": "stopped", "downloaded": done, "remaining": keys[i - 1:]}
            _publish(job_id, **res)
            return res
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
# Google Drive
#
# Most LoRAs worth having were never published to HuggingFace — they are a link
# someone sent you. This is the same job/status/stop contract the weight
# downloads use, the same watcher, the same staging-then-move: what is different
# is only where the bytes come from, which is the smallest a new capability
# should be.
# --------------------------------------------------------------------------

GDRIVE_JOB = "dl_gdrive"


def _is_drive_folder(url: str) -> bool:
    """A folder link needs gdown's folder API; a file link needs its file API."""
    return "/folders/" in url or "folderview" in url


@app.function(image=web_image, cpu=2.0, timeout=4 * 60 * 60, volumes={"/workspace": volume})
def gdrive_job(url: str, folder: str) -> dict[str, Any]:
    """
    Pull one file or one folder off Google Drive into loras/.

    Staged first, then moved. Downloading straight into loras/ would put a
    half-written .safetensors in front of the picker — and the picker globs the
    directory live, so it would be offered, chosen, and fail inside a warm GPU
    container with a torch deserialization error thirty seconds into a run.
    Staging keeps a partial download invisible until it is a whole file.
    """
    import threading

    import gdown

    job_id = GDRIVE_JOB
    jobs[job_id] = {"status": "running", "phase": "Starting…", "percent": 0,
                    "stop": False, "beat": time.time()}

    if folder and not NAME_RE.match(folder):
        err = f"Folder name must be 1-64 chars of [A-Za-z0-9_-]: {folder!r}"
        _publish(job_id, status="failed", error=err)
        return {"status": "failed", "error": err}

    _reload_volume()
    stage = WORK / f"gdrive-{int(time.time())}"
    if stage.exists():
        shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f"[gdrive] {url} -> loras/{folder or ''}")

    result: dict[str, Any] = {}
    done = threading.Event()

    def pull() -> None:
        try:
            if _is_drive_folder(url):
                gdown.download_folder(url, output=str(stage), quiet=True,
                                      use_cookies=False)
            else:
                # fuzzy, so a pasted browser URL works as well as a bare id —
                # which is the form the link actually arrives in.
                gdown.download(url, output=str(stage) + "/", quiet=True, fuzzy=True)
        except Exception as exc:
            result["error"] = exc
        finally:
            done.set()

    threading.Thread(target=pull, daemon=True).start()
    state = _watch_download(stage, "Google Drive", job_id, done)

    if state.get("stopped"):
        shutil.rmtree(stage, ignore_errors=True)
        _publish(job_id, status="stopped")
        return {"status": "stopped"}

    if state["stalled"]:
        shutil.rmtree(stage, ignore_errors=True)
        err = (f"Stalled at {state['bytes'] / 1e9:.2f} GB with no new bytes for "
               f"{DOWNLOAD_STALL_S}s. Drive throttles large files and refuses "
               "ones without link sharing — check the link opens in a private window.")
        _publish(job_id, status="failed", error=err)
        return {"status": "failed", "error": err}

    if result.get("error") is not None:
        shutil.rmtree(stage, ignore_errors=True)
        exc = result["error"]
        # gdown's own failure for a private file is an HTML page, not an
        # exception with a useful message, so the likely cause is named here
        # rather than left to be inferred from a parse error.
        err = (f"{type(exc).__name__}: {exc}. If the file is not shared with "
               "'Anyone with the link', Drive serves a sign-in page instead of "
               "the file and there is nothing to download.")
        _publish(job_id, status="failed", error=err)
        return {"status": "failed", "error": err}

    # Only weights. A Drive folder usually carries a preview grid and a readme
    # too, and copying those onto the volume would put files in loras/ that the
    # picker has to keep stepping over.
    found = sorted(p for p in stage.rglob("*") if p.is_file())
    weights = [p for p in found if p.suffix.lower() == ".safetensors"]
    skipped = [p.name for p in found if p not in weights]
    if not weights:
        shutil.rmtree(stage, ignore_errors=True)
        err = ("No .safetensors in that download"
               + (f" — found {', '.join(skipped[:6])}." if skipped else "."))
        _publish(job_id, status="failed", error=err)
        return {"status": "failed", "error": err}

    # No folder given means the top level of loras/, where the listing already
    # treats a bare file as its own entry named for itself. A folder given means
    # loras/{folder}/, where the listing treats the files as versions of one
    # LoRA — which is right for a matched pair and wrong for a bag of unrelated
    # ones, so it stays a choice rather than a default.
    dest_dir = (LORAS / folder) if folder else LORAS
    dest_dir.mkdir(parents=True, exist_ok=True)
    landed = []
    for p in weights:
        target = dest_dir / p.name
        if target.exists():
            target.unlink()
        shutil.move(str(p), target)
        landed.append(target.name)

    shutil.rmtree(stage, ignore_errors=True)
    volume.commit()

    res = {
        "status": "completed",
        "percent": 100,
        "files": landed,
        "skipped": skipped,
        "folder": folder,
        "size_gb": round(sum((dest_dir / n).stat().st_size for n in landed) / 1e9, 2),
        "duration_s": round(time.time() - started, 1),
    }
    _publish(job_id, **res)
    print(f"[gdrive] {len(landed)} file(s), {res['size_gb']} GB in {res['duration_s']}s")
    return res


# --------------------------------------------------------------------------
# Captioning
# --------------------------------------------------------------------------


def _caption_images(
    image_dir: Path, trigger_word: str, job_id: str,
    preset: str, length: str, overwrite: bool, model_key: str,
) -> tuple[int, int]:
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
        return 0, 0

    spec = CAPTION_MODELS.get(model_key) or CAPTION_MODELS[DEFAULT_CAPTION_MODEL]
    repo = spec["repo"]
    print(f"[caption] {len(todo)}/{len(every)} images · {preset} · {repo}")
    _publish(job_id, phase="caption", step=0, total_steps=len(todo), percent=0)

    cache_dir = str(HF_CACHE)
    # Cap the vision tower's token budget. Qwen3-VL scales patches with input
    # resolution, so a 4000px training image would otherwise spend thousands of
    # tokens on detail that never reaches the caption — slow, and no better.
    processor = AutoProcessor.from_pretrained(
        repo, cache_dir=cache_dir,
        min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        repo, dtype=torch.bfloat16, device_map="cuda:0", cache_dir=cache_dir,
    )
    model.eval()
    # Persist the downloaded weights now, on their own volume, so the next cold
    # start reuses them and the dataset commit below stays small.
    try:
        hf_cache.commit()
    except Exception as exc:
        print(f"[caption] hf cache commit skipped: {exc}")

    instruction = _caption_instruction(preset, length, trigger_word)

    written = refused = 0
    for i, img_path in enumerate(todo, 1):
        if _stop_requested(job_id):
            print("[caption] stop requested")
            break
        try:
            image = _upright(Image.open(img_path)).convert("RGB")
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

            # A decline is fluent prose, so it would pass every check below it
            # and land in a sidecar the trainer reads. Leaving the file alone
            # keeps the image in the Uncaptioned filter, which is where it can
            # be found and re-run against the other captioner.
            if _looks_like_refusal(caption):
                print(f"[caption] {img_path.name} refused: {caption[:80]}")
                refused += 1
            else:
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
    return written, refused


@app.function(
    # A100 rather than the training GPU: Qwen3-VL-8B in bf16 is ~17 GB, so this
    # does not need the headroom a rank-32 Krea 2 run does.
    image=caption_image, gpu="A100-40GB", cpu=2.0, timeout=2 * 60 * 60,
    volumes={"/workspace": volume, str(HF_CACHE): hf_cache},
)
def caption_job(
    job_id: str, dataset: str, trigger_word: str = "",
    preset: str = DEFAULT_CAPTION_PRESET, length: str = "medium",
    overwrite: bool = False, model: str = DEFAULT_CAPTION_MODEL,
) -> dict[str, Any]:
    spec = CAPTION_MODELS.get(model) or CAPTION_MODELS[DEFAULT_CAPTION_MODEL]
    # The label rides the record because the first minute of this job is a cold
    # start that may also be a 17 GB pull, and "Loading captioner…" for twenty
    # minutes is the same UI state as a hang. Naming which one is loading is the
    # difference between that and "it is fetching the uncensored weights".
    jobs[job_id] = {"status": "running", "phase": "caption", "stop": False,
                    "model": model, "model_label": spec["label"]}
    _reload_volume()
    src = _dataset_dir(dataset)
    if not src.is_dir():
        raise RuntimeError(f"No dataset named {dataset!r}.")

    started = time.time()
    written, refused = _caption_images(
        src, trigger_word.strip(), job_id, preset, length, overwrite, model)
    res = {
        "status": "completed", "job_id": job_id, "dataset": dataset,
        "captioned": written, "refused": refused, "preset": preset,
        "model": model, "model_label": spec["label"],
        "duration_s": round(time.time() - started, 1),
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
    _reload_volume()

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
    # Orientation is baked in on the way into scratch, not left to the trainer:
    # musubi opens with PIL and PIL does not rotate, so an EXIF-rotated portrait
    # would be measured landscape, bucketed landscape and trained sideways. The
    # scratch copy is the right place for it — the dataset on the volume keeps
    # its original bytes, and every run gets upright pixels without the user
    # having to know the tag exists.
    rotated = 0
    for item in src.iterdir():
        if item.suffix.lower() in IMAGE_EXTS:
            rotated += _upright_copy(item, image_dir / item.name)
        elif item.suffix.lower() == ".txt":
            shutil.copy2(item, image_dir / item.name)

    images = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        raise RuntimeError("No images to train on.")
    if rotated:
        log.append(f"[prepare] rotated {rotated} image(s) upright from EXIF")

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
# Backed by ComfyUI, the same backend video runs on. musubi's
# krea2_generate_image.py is a one-shot CLI: it reloaded ~35 GB of weights for
# every image and took a single LoRA at a single strength. This is a Modal Cls,
# so the checkpoint stays resident between requests, and LoRAs stack — any
# number of them, each with its own UNet and text-encoder weight.
#
# This ran on a vendored sd-webui-forge-classic until regional prompting stopped
# being a reason to keep it; see the note at CLIFF_SHA.
# --------------------------------------------------------------------------


# ComfyUI's own names, not Forge's.
#
# These are values sent into a graph, not labels — `KSampler` validates
# `sampler_name` against `comfy.samplers.KSAMPLER_NAMES` and rejects anything
# else as "Value not in list", so "DPM++ 2M" is not a spelling of `dpmpp_2m`
# here, it is a rejected request. The old list was Forge's and every entry in it
# was wrong the moment the backend changed.
#
# A subset, not the full 40-odd, chosen the same way VIDEO_MODELS chooses:
# ancestral variants reintroduce noise each step, which fights an 8-step
# distilled Turbo, and the cfg_pp family is for models that take CFG. The full
# list is `python -c "import comfy.samplers; print(...)"` inside the container
# if one is ever wanted back.
#
# `er_sde` is the exception to the no-SDE rule and the reason the rule is worded
# about ancestral samplers rather than stochastic ones: it is an ER-SDE solver,
# so the noise it adds is scheduled by the solver rather than injected on top of
# the step, and it holds together at Turbo's step count where `euler_a` does not.
SAMPLERS = [
    "er_sde", "euler", "res_multistep", "dpmpp_2m", "heun", "ddpm", "lcm",
    "deis", "ipndm",
]
# Krea 2 is flow-matching, so these are the schedules that mean something on a
# sigma curve that starts at 1. "Automatic" is gone with Forge: it was Forge
# picking a schedule per sampler, and ComfyUI has no such concept — `simple` is
# the default the model config implies and what the video path already sends.
SCHEDULERS = ["simple", "normal", "beta", "sgm_uniform", "karras", "exponential",
              "linear_quadratic", "kl_optimal", "ddim_uniform"]

# What an image render uses when nothing says otherwise. Separate from
# KREA2_DEFAULTS because that dict is per-checkpoint and these are not: the
# sampler and the schedule are properties of a flow-matching curve, and Turbo
# and RAW want the same pair. Served to the page as well as applied here, so the
# menu opens on the value the backend would have picked anyway — two places
# spelling the same default is how they drift.
IMAGE_DEFAULTS = {"sampler": "er_sde", "scheduler": "sgm_uniform"}
MAX_LORAS = 6

# Per-checkpoint defaults, which used to live inside the Forge pipeline and be
# reported back in `last_report`. They are here now because the graph builder
# has to write a number into KSampler — ComfyUI has no notion of "auto", and a
# steps field left empty has to become something before the graph is valid.
#
# Turbo is guidance-distilled: CFG 1.0 is not a low setting, it is the absence
# of a negative branch, and raising it on Turbo burns contrast rather than
# adding adherence.
KREA2_DEFAULTS = {
    "turbo": {"steps": 8, "cfg": 1.0},
    "raw": {"steps": 28, "cfg": 5.5},
}


# --------------------------------------------------------------------------
# ComfyUI — one process per container, driven over 127.0.0.1
#
# Shared by both GPU classes. It was written for video and stayed there while
# images ran on Forge; when images moved, copying it would have meant two
# implementations of "wait for a graph and find what it saved" whose bugs are
# fixed one at a time. The families differ in the graph they build and nothing
# else, which is the same line the job contract already draws.
# --------------------------------------------------------------------------

# How long a dropped connection is allowed to mean "still starting to die".
# ComfyUI's sockets reset the moment the process is signalled, and it is not
# reaped for a beat after that, so a `poll()` taken at the reset reports the
# corpse alive and the diagnosis goes to the wrong branch.
COMFY_DEATH_GRACE_S = 5.0
# Consecutive dropped polls against a process that is demonstrably alive before
# the run is given up on. One is a blip; a server resetting every connection
# forever is a hang, and a poll loop that tolerates it without bound is worse
# than one that fails with the log tail in hand.
COMFY_RESET_TRIES = 5

# ComfyUI's own wording for an out-of-memory failure, and the only marker worth
# matching on. The exception underneath is `torch.OutOfMemoryError` down one
# path and a bare `Allocation on device` from the allocator down another, and
# the node blamed is whichever happened to ask for the last block — KSampler on
# the image side, but only because that is where the sampling loop lives. The
# tip is a constant in `execution.py`, so it is the stable half.
COMFY_OOM_MARK = "ran out of memory on your GPU"


class _Comfy:
    """
    A warm ComfyUI, and the four things anyone needs from it.

    ComfyUI runs as a local server inside the container and is spoken to over
    127.0.0.1 — it is never exposed, and the only client is this class. Running
    the real thing rather than porting its model code is what keeps the
    int8-convrot kernels, the dynamic offloader, Krea 2 support and every
    upstream fix on our side of the line instead of in a fork we would own.

    Not a Modal Cls itself, deliberately: `@modal.enter` and `@modal.method`
    belong to the classes Modal instantiates, and a base class carrying them is
    a question about Modal's decorator inheritance that nothing here needs to
    ask. Each GPU class owns one of these and calls `start()` from its own
    enter hook.
    """

    def __init__(self, tag: str):
        # Prefixes every line this container mirrors, so two GPU classes
        # printing the same ComfyUI format are tellable apart in Modal's
        # stream, where they interleave. See `_drain`.
        self.tag = tag
        self.job_id: str | None = None
        self._log: deque[str] = deque(maxlen=200)
        self._proc: subprocess.Popen | None = None

    def start(self) -> None:
        import threading

        # Point ComfyUI at our flat models/ directory instead of moving weights
        # into the per-type tree it expects. The volume layout is the contract;
        # ComfyUI adapts to it. Every type maps to the same folder, so every
        # file is visible to whichever loader asks for it.
        #
        # loras/ is the exception and keeps its nesting, because that nesting is
        # meaningful — one folder per trained LoRA, checkpoints beside the final
        # weights. ComfyUI walks it recursively and names a file by its path
        # relative to here, which is exactly what the LoRA validators emit.
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

        self._proc = subprocess.Popen(
            ["python", "main.py", "--listen", "127.0.0.1", "--port", str(COMFY_PORT),
             "--disable-auto-launch", "--disable-metadata",
             # Safe on both paths, for different reasons. H3 and Wan never pass
             # an attention mask. Krea 2 does, but only through the regional
             # node, and that node installs itself as
             # `optimized_attention_override` and runs its own FlexAttention
             # kernel for exactly the masked case — sageattention is what it
             # delegates *unmasked* blocks to. The one call that still reaches
             # sage with a mask is ComfyUI's own fallback-per-call path, which
             # is a slower call and not a wrong picture.
             "--use-sage-attention",
             # Not a tuning choice — the alternative is broken. ComfyUI's
             # `text_encoder_dtype()` ends in a bare `return torch.float16`
             # with no device or model test, so every text encoder is stored
             # fp16 unless a flag says otherwise, and the startup log duly
             # reported `dtype: torch.float16` for a file we download in bf16.
             #
             # Most models survive that because they read one normalised final
             # hidden state. Krea 2 does not: `comfy/text_encoders/krea2.py`
             # taps twelve RAW intermediate layers of Qwen3-VL-4B — 2 through
             # 35 — and concatenates them into a (B, seq, 12*2560) conditioning
             # tensor. Qwen's deep layers carry very large activations, and in
             # fp16 those lose most of their precision or leave the range
             # outright. That is conditioning the DiT cannot denoise against,
             # which is the noise every Krea 2 render produced from the day the
             # image path moved off Forge — Forge held this encoder in bf16.
             #
             # Process-wide, because CLIPLoader takes no dtype and the three
             # families share one ComfyUI. That is fine for the other two:
             # umT5 and a quantised Qwen3-VL-32B both prefer bf16 to fp16.
             "--bf16-text-enc"],
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
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            # Tagged on the way to Modal's logs, because there are two of these
            # processes on two GPU classes printing the same format, and an OOM
            # traceback that names KSampler, Krea2 and a CUDA device says
            # nothing about which. Untagged, the only way to tell an image
            # container's log from a video one was to recognise the model file
            # names in it. The deque stays clean: it is spliced into errors that
            # are already shown under the panel they belong to.
            print(f"[{self.tag}] {line}", flush=True)
            self._log.append(line)
            m = TQDM_RE.search(line)
            if m and self.job_id:
                fields: dict[str, Any] = {
                    "phase": "generate",
                    "step": int(m.group("step")),
                    "total_steps": int(m.group("total")),
                    "percent": int(m.group("pct")),
                }
                if m.group("eta"):
                    fields["eta"] = m.group("eta")
                    fields["rate"] = f"{m.group('rate')}{m.group('unit')}"
                _publish(self.job_id, **fields)

    def _wait_ready(self, timeout: float = 300.0) -> None:
        import urllib.error
        import urllib.request

        assert self._proc is not None
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
                print(f"[{self.tag}] ComfyUI ready", flush=True)
                return
            except (urllib.error.URLError, OSError):
                time.sleep(1.0)
        raise RuntimeError("ComfyUI did not become ready.\n" + "\n".join(self._log))

    def require_nodes(self, *class_types: str) -> None:
        """
        Fail at startup if a custom node did not register, naming it.

        ComfyUI logs an import traceback for a custom node that raises and then
        starts anyway without it. Left alone, the first symptom is a queued
        graph rejected for an unknown class_type — which reads like our graph
        builder naming a node wrong, minutes into a warm GPU, when the real
        fault was an import error printed during startup and scrolled past. The
        log tail goes in the message because that traceback is the answer and
        it is already in hand.
        """
        known = self.get("/object_info") or {}
        missing = [c for c in class_types if c not in known]
        if missing:
            raise RuntimeError(
                f"ComfyUI started without {', '.join(missing)}. A custom node "
                f"failed to import; its traceback is above.\n"
                + "\n".join(list(self._log)[-40:])
            )

    def _died(self) -> RuntimeError:
        """The one thing worth saying about a ComfyUI that is no longer there."""
        code = self._proc.poll() if self._proc else None
        return RuntimeError(
            f"ComfyUI exited mid-generation (exit code {code}). Its last output "
            "is below; a CUDA error or an OOM there is the cause. If there is no "
            "Python traceback at all, the GPU faulted under it — Modal's "
            "[gpu-health] Xid line in the container log is the record of that, "
            "and the run is worth retrying on a fresh container.\n"
            + "\n".join(list(self._log)[-25:])
        )

    def _check_alive(self) -> None:
        """
        Ask whether ComfyUI is still there before blaming the socket.

        A GPU that faults — Xid 31, an MMU fault, is the one that has actually
        happened here — takes the process with it, and every request in flight
        resets. `_await` already knew what to say about a dead ComfyUI, log tail
        and all, but that check sat *downstream* of the poll that could not
        survive to reach it: what reached the job record was urllib's
        `ConnectionResetError`, which names no CUDA error, carries no log and
        does not mention the GPU. Every caller asks this first now.

        `wait` rather than `poll`, for COMFY_DEATH_GRACE_S: see the constant.
        """
        if self._proc is None:
            return
        try:
            self._proc.wait(timeout=COMFY_DEATH_GRACE_S)
        except subprocess.TimeoutExpired:
            return
        raise self._died()

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            f"http://127.0.0.1:{COMFY_PORT}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            return json.loads(urllib.request.urlopen(req, timeout=30).read() or b"{}")
        except urllib.error.HTTPError:
            # A status code is ComfyUI answering: the transport is fine and the
            # graph is what it is complaining about. Nothing to diagnose here.
            raise
        except OSError:
            self._check_alive()
            raise

    def get(self, path: str) -> Any:
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{COMFY_PORT}{path}", timeout=30
            ) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError:
            raise
        except OSError:
            self._check_alive()
            raise

    def _revive(self) -> None:
        """
        Replace a ComfyUI that died under the last take, before starting this one.

        `@modal.enter` runs once per container, so the process it starts is the
        only one the container will ever have — and `max_containers=1` means a
        dead one is not a degraded install, it is the whole platform refusing
        every render until the scaledown window expires ten or fifteen minutes
        later. The GPU faults that kill it (Xid 31 is an illegal address in the
        kernel, not a dead card) leave the device perfectly able to run the next
        graph, so what stood between the user and a working render was nothing
        but a process nobody restarted.

        The old log goes with the old process: its tail was already delivered as
        the last job's error, and keeping it would attach a stale traceback to
        whatever fails next.
        """
        if self._proc is None or self._proc.poll() is None:
            return
        print(f"[{self.tag}] ComfyUI died (exit code {self._proc.poll()}); "
              "starting a fresh one", flush=True)
        self._log.clear()
        self.start()

    def _note_headroom(self) -> None:
        """
        Print what is left on the card, before the graph is queued.

        The one fact an out-of-memory failure never carries is whether the card
        was already full when the job started. Krea 2 is 24 GB of an 80 GB card,
        so a regional render that dies in step 0 is either a graph too big for
        55 GB of headroom or a graph handed 5 GB by whatever ran before it, and
        those want opposite fixes. ComfyUI prints its own memory summary only
        once it has already failed, which is the wrong end: by then every number
        describes the wreck rather than what it started with.

        Best effort, and deliberately not on the failure path — the honest
        "after" reading is the next run's "before", once ComfyUI's worker has
        acted on `_reclaim`'s flag. A number taken a millisecond after asking
        would mostly measure the asking.
        """
        try:
            dev = ((self.get("/system_stats") or {}).get("devices") or [{}])[0]
            free, total = dev.get("vram_free") or 0, dev.get("vram_total") or 0
        except Exception:
            return
        if total:
            print(f"[{self.tag}] vram before run: {free / 2**30:.1f} of "
                  f"{total / 2**30:.1f} GiB free", flush=True)

    def _reclaim(self) -> None:
        """
        Hand the card back after an out-of-memory failure, before the next take.

        `_revive()` covers a ComfyUI that died. This is the other half of the
        same idea: one that is alive and has nothing left to allocate, which is
        a state the process does not leave on its own.

        ComfyUI answers its own OOM with `unload_all_models()`, and on this
        install that is not enough. The regional node moves every region's LoRA
        onto the device in `_prepare()` and stores the copies on the patcher it
        returns, so they are held by the *execution cache* — which model
        management cannot see and `unload_all_models()` does not touch. They
        come back when the cached node output is dropped, and `/free` is the
        only thing that drops it (`e.reset()` in ComfyUI's prompt worker).

        Which is the shape the failure actually had: not one configuration too
        big, but a few in a row, none reproducible on its own. A run that ran
        out of memory left behind the thing it ran out of memory on, and the
        next one started with less room than the last.

        The bill is a checkpoint reload on the following take, charged only to a
        job that has already failed — against `max_containers=1`, where the
        alternative is every render refused until the scaledown window expires.
        Best effort: a container too far gone to answer this is one `_revive()`
        replaces anyway.
        """
        try:
            # `free_memory` implies `unload_models` upstream — the flag defaults
            # to it — so this is both halves, and there is no way to reset the
            # node cache without also dropping the checkpoint. That is the whole
            # reason this is not simply done after every job.
            self.post("/free", {"free_memory": True})
        except Exception as exc:
            print(f"[{self.tag}] could not reclaim after OOM: {exc}", flush=True)
            return
        print(f"[{self.tag}] out of memory — asked ComfyUI to unload its models "
              "and drop the node cache; the next take reloads the checkpoint",
              flush=True)

    def stage(self, job_id: str, blob: str, slot: str, ext: str = "png") -> str:
        """
        Drop one uploaded file where a LoadImage node can name it.

        Under the job id, so two takes cannot collide on a filename — the
        second would otherwise silently reuse the first's frame.
        """
        name = f"{job_id}-{slot}.{ext}"
        (COMFY / "input" / name).write_bytes(base64.b64decode(blob))
        return name

    def run(self, job_id: str, graph: dict[str, Any], *, what: str) -> list[str]:
        """
        Queue one graph and return what it saved, relative to output/.

        `what` is the noun for the "saved nothing" error, which is the one
        failure here that is neither an exception nor a picture: the graph ran,
        reported success, and the run is about to be thrown away by a job that
        says it produced no files.
        """
        self._revive()
        self.job_id = job_id
        try:
            self._note_headroom()
            prompt_id = self.post("/prompt", {"prompt": graph})["prompt_id"]
            return self._await(job_id, prompt_id, what)
        except RuntimeError as exc:
            # StopRequested is not a RuntimeError, so a cancelled take does not
            # come through here and does not pay for a reload it never earned.
            if COMFY_OOM_MARK in str(exc):
                self._reclaim()
            raise
        finally:
            self.job_id = None

    def _await(self, job_id: str, prompt_id: str, what: str) -> list[str]:
        assert self._proc is not None
        drops = 0
        while True:
            if _stop_requested(job_id):
                # ComfyUI unwinds the sampler itself and stays warm, which a
                # killed process would not — the weights stay loaded for the
                # next take.
                self.post("/interrupt", {})
                raise StopRequested("generate")

            try:
                entry = (self.get(f"/history/{prompt_id}") or {}).get(prompt_id)
            except OSError as exc:
                # get() has already established the process is alive, so this
                # is one request dropped by a server that is still running —
                # the next poll normally finds it recovered, and failing a
                # forty-minute clip over a single socket would be the wrong
                # trade. Bounded: see COMFY_RESET_TRIES.
                drops += 1
                if drops > COMFY_RESET_TRIES:
                    raise RuntimeError(
                        f"ComfyUI is running but dropped {drops} polls in a row "
                        f"({type(exc).__name__}: {exc}).\n"
                        + "\n".join(list(self._log)[-25:])
                    ) from exc
                time.sleep(1.5)
                continue
            drops = 0

            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error" or not status.get("completed", True):
                    raise RuntimeError(self._why_failed(status))
                names: list[str] = []
                for out in entry.get("outputs", {}).values():
                    # "images" is the one that actually fires for video too:
                    # SaveVideo returns ui.PreviewVideo, whose as_dict() is
                    # {"images": [...], "animated": (True,)} — a video is
                    # reported through the image channel, flagged rather than
                    # separately named. The other two are what the older save
                    # nodes emit and cost a tuple to keep.
                    for key in ("images", "videos", "gifs"):
                        for item in out.get(key) or []:
                            name = item.get("filename")
                            if not name:
                                continue
                            # Honour the subfolder rather than assuming the flat
                            # case our filename_prefix happens to produce: the
                            # prefix is split on a path separator, so the day one
                            # gains a slash this stops silently reading the
                            # wrong path.
                            names.append(str(Path(item.get("subfolder") or "") / name))
                if names:
                    # Sorted because a batch arrives in whatever order the save
                    # node reported it, and the gallery numbers what it is
                    # given — an unsorted batch of four renumbers itself
                    # between polls of the same finished job.
                    return sorted(names)
                raise RuntimeError(
                    f"ComfyUI reported success but saved no {what}.\n"
                    + "\n".join(list(self._log)[-25:])
                )

            if self._proc.poll() is not None:
                raise self._died()
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
# Datasets
#
# Named, reusable, and independent of any training run — caption once, train a
# rank sweep from it. The directory is the whole model: images plus .txt
# sidecars, which is exactly what musubi consumes, so there is no database to
# fall out of sync with the files and no export step.
#
# Keeping one is a choice you make afterwards. Dropping images makes a draft
# under drafts/, which trains and captions exactly like a saved set and lasts as
# long as the window that made it; saving moves the folder into datasets/. Most
# sets are dropped once to answer one question, and making every one of those a
# permanent, named entry in the library taxes the common case to serve the rare
# one.
# --------------------------------------------------------------------------


# A draft belongs to the window that made it. "When I close the app" has no
# server-side event here — the web container scales to zero on Modal's schedule,
# not yours, and a cold start is not something you did — so a page that has
# stopped saying it is open is the only honest signal there is. The grace period
# is long enough that a laptop asleep through a coffee break is still open.
DRAFT_GRACE_S = 15 * 60
SESSIONS_DIR = ".sessions"


def _check_name(name: str) -> str:
    if not NAME_RE.match(name or ""):
        raise ValueError("Set names are 1-64 chars of letters, numbers, _ or -.")
    return name


def _dataset_dir(name: str) -> Path:
    """
    Where the set called `name` lives: saved under datasets/, else drafts/.

    A name is unique across both roots — creating and saving each refuse one the
    other root already holds — so this never has to choose between two folders.
    An unused name resolving into drafts/ is what makes dropping images produce
    a draft without a second create path to keep in step with this one.
    """
    _check_name(name)
    saved = DATASETS / name
    return saved if saved.is_dir() else DRAFTS / name


def _name_taken(name: str) -> bool:
    return (DATASETS / name).exists() or (DRAFTS / name).exists()


def _touch_session(sid: str) -> None:
    """Record that a window is still open. The mtime is the entire payload."""
    if not NAME_RE.match(sid or ""):
        return
    marker = DRAFTS / SESSIONS_DIR / sid
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("")


def _upright(im):
    """
    Apply an image's EXIF orientation to its pixels.

    Every consumer here reads pixels with PIL, and PIL does not rotate on open —
    browsers do. So a phone photo stored landscape with an orientation tag looked
    right in the page and was sideways to everything that mattered: the captioner
    described a rotated scene, and musubi's loader is a bare
    `Image.open(...).convert("RGB")`, so the bucket was chosen from the stored
    dimensions and the model trained on the rotation. A 3024x4032 portrait went
    into a landscape bucket.

    Returns the image unchanged when there is no orientation tag, which is the
    common case and costs nothing.
    """
    from PIL import ImageOps

    try:
        return ImageOps.exif_transpose(im) or im
    except Exception:
        # A truncated or malformed EXIF block is not a reason to lose the image.
        return im


def _upright_inplace(path: Path) -> bool:
    """
    Rewrite one uploaded image with its EXIF rotation baked into the pixels.

    The consumers each handle orientation now, but this is the only fix at the
    root: "rotate" in Finder or Photos usually writes the orientation tag and
    leaves the pixels alone, so the file that lands here is sideways to anything
    that does not read EXIF — and that is most things, including the trainer.
    Normalising once on arrival means nothing downstream has to know the tag
    exists, which is the same reason the captions are `.txt` sidecars: the
    storage layout is the contract.

    Writes via a temp file and `replace`, so a failure mid-encode cannot leave a
    half-written image where a whole one was. Untagged files — nearly all of
    them — are not touched at all.
    """
    from PIL import Image

    tmp = path.with_name(path.name + ".upright")
    try:
        with Image.open(path) as im:
            if (im.getexif() or {}).get(274, 1) in (0, 1):
                return False
            upright = _upright(im)
            if path.suffix.lower() in {".jpg", ".jpeg"}:
                upright.convert("RGB").save(tmp, "JPEG", quality=95, subsampling=0)
            else:
                upright.save(tmp)
        tmp.replace(path)
        return True
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        print(f"[upload] could not normalise orientation for {path.name}: {exc}")
        return False


def _upright_copy(src: Path, dst: Path) -> bool:
    """
    Copy one training image, baking in its EXIF rotation. True if it rotated.

    Only re-encodes when the tag says to. Anything else is `copy2`, so a set of
    200 already-upright images pays one header read each and keeps its bytes
    exactly — the originals on the volume are never touched either way, because
    this only ever writes into the per-run scratch copy.
    """
    from PIL import Image

    try:
        with Image.open(src) as im:
            if (im.getexif() or {}).get(274, 1) in (0, 1):
                raise ValueError("no rotation")
            upright = _upright(im)
            if dst.suffix.lower() in {".jpg", ".jpeg"}:
                upright.save(dst, quality=95, subsampling=0)
            else:
                upright.save(dst)
            return True
    except Exception:
        shutil.copy2(src, dst)
        return False


def _drop_legacy_trash(root: Path) -> None:
    """
    Clear a `.trash/` an earlier version of this file left behind.

    Deletion is now unlinking everywhere, so these folders are not a safety net
    any more — they are a second copy of everything already thrown away, sitting
    on a volume whose only other way to reclaim space is the Modal CLI. Called
    from the delete paths rather than from a listing or a timer, so it happens
    where a deletion was already asked for and never as a surprise.
    """
    victim = root / LEGACY_TRASH_DIR
    if victim.is_dir():
        shutil.rmtree(victim, ignore_errors=True)


def _tree_bytes(root: Path) -> int:
    """
    Every byte a delete of `root` would reclaim, whether it is a file or a folder.

    Walked rather than summed off the listing that calls it. A trained LoRA's
    folder is flat and holds only checkpoints and `visionary.json`, so the two
    agree on everything this app writes — but the number's whole job is to be
    what the confirm dialog says is going, and a dialog that undersells the blast
    radius is the failure the .trash removal was answering. A folder that arrived
    some other way, with a preview subdirectory in it, is exactly the case where
    "sum the files I was already going to list" quietly reports less than it
    unlinks.
    """
    try:
        if root.is_file():
            return root.stat().st_size
    except OSError:
        return 0
    total = 0
    for dirpath, _, names in os.walk(root):
        for name in names:
            try:
                total += (Path(dirpath) / name).stat().st_size
            except OSError:
                pass
    return total


def _sweep_drafts() -> int:
    """
    Retire drafts whose window closed. Returns how many went.

    Unlinked, like every other deletion here. This is the one that nobody asks
    for by name, so it is the one where the grace period is doing all the work:
    `DRAFT_GRACE_S` of silence from the session, and the folder's own mtime
    counted as a heartbeat so an upload still writing cannot be swept out from
    under itself. Liveness is per session and not per container, so a second tab
    open on the same app keeps its own drafts and does not reap the first one's.
    """
    if not DRAFTS.is_dir():
        return 0
    now, swept = time.time(), 0
    for d in sorted(DRAFTS.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        sid = ""
        try:
            sid = str(json.loads((d / "dataset.json").read_text()).get("session") or "")
        except (OSError, json.JSONDecodeError):
            pass
        seen = 0.0
        if NAME_RE.match(sid or ""):
            try:
                seen = (DRAFTS / SESSIONS_DIR / sid).stat().st_mtime
            except OSError:
                seen = 0.0
        # The folder's own mtime counts as a heartbeat, so a draft created
        # seconds ago by a page whose first ping has not landed — or one made by
        # a caller that never sends a session at all — is not swept out from
        # under the upload that is still writing into it.
        try:
            seen = max(seen, d.stat().st_mtime)
        except OSError:
            continue
        if now - seen < DRAFT_GRACE_S:
            continue
        shutil.rmtree(d, ignore_errors=True)
        swept += 1

    # This is the app's periodic housekeeping pass, which makes it the one place
    # a one-time migration can run without waiting for the user to delete
    # something in each root. Both are no-ops on a volume that never had a
    # `.trash/`, and on one that did they run once and then cost a stat.
    _drop_legacy_trash(DRAFTS)
    _drop_legacy_trash(DATASETS)

    # Markers outlive the drafts they kept alive; a day is well past the point
    # where one can still be protecting anything.
    for marker in (DRAFTS / SESSIONS_DIR).glob("*"):
        try:
            if now - marker.stat().st_mtime > 86400:
                marker.unlink()
        except OSError:
            pass
    return swept


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
        # Which parent it sits under, not a field in dataset.json: the flag and
        # the folder cannot then disagree about whether the set is kept.
        "saved": d.parent == DATASETS,
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

    # A regional render is not reproducible from its prompts alone — which LoRA
    # sat in which box is the whole result — so each region is written as
    # "lora@strength: text" in box order, and box order is what pairs a row
    # with its rectangle.
    regions = report.get("regions") or []
    if regions:
        add("Regions", len(regions))
        add("Region prompts", " | ".join(
            ((f"{r['lora']}@{r.get('strength', 1.0)}: " if r.get("lora") not in
              (None, "", "None") else "") + str(r.get("prompt", "")))
            for r in regions
        ))
        add("Region weight", report.get("region_weight"))
        # Only when some box had one. A mold changes the likeness more than any
        # number here does, and "why does this one look so much more like her"
        # is unanswerable from a sidecar that does not mention the photograph.
        molds = sum(1 for r in regions if r.get("ref"))
        if molds:
            add("Region refs", molds)
        # The caption goes in as a field rather than as the infotext's first
        # line, because that line is the prompt and A1111's format says so —
        # a reader pasting this back expects to get the sentence they wrote,
        # not the machine's expansion of it.
        add("Caption", report.get("caption"))

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

        # `name` is the same conversion _validate_video_loras makes and for the
        # same reason: LoraLoader takes a combo validated against a directory
        # listing, not a path, so a path reaching the graph fails inside a warm
        # GPU as "Value not in list" rather than here as a form error.
        out.append({
            "path": str(path),
            "name": path.relative_to(LORAS.resolve()).as_posix(),
            "unet": unet,
            # Krea 2's text encoder takes LoRA weights, unlike Wan's — so this
            # stack keeps the second number the video one has no use for.
            # Defaulted to the UNet weight here rather than in the client, so
            # today's default is not frozen into every prompt ever saved.
            "text_encoder": unet if te is None else te,
        })
    return out


# What `_edit_lora_choices()` in the node pack falls back to when the volume
# holds no LoRAs at all. Read off the catalogue rather than spelled again,
# because the combo we send has to be a member of the list that function
# returns — and when the volume is empty that list is exactly `[this]`, so the
# two names have to be the same string or the empty-volume case fails
# validation on a value we chose ourselves.
KREA2_EDIT_LORA = MODEL_CATALOGUE["krea2_edit"]["dest"].name

# Rectangles per render. V12 pairs box i with row i of regions_json and the
# attention partition is built per box, so this is a real cost rather than a
# tidy limit — eight subjects in one 1024px frame is already ~128px of canvas
# each, below which a face has no pixels to be recognisable in.
MAX_REGIONS = 8


def _validate_regions(raw: Any) -> list[dict[str, Any]]:
    """
    Validate the regional stack: one box, one optional LoRA, one description,
    one optional reference photo.

    Rows are kept index-aligned with the boxes and never filtered, because V9's
    `_pair_boxes` matches box i to row i by ORIGINAL index — dropping an empty
    row here would hand every later row the rectangle belonging to the one
    before it, which is a picture with the right faces in the wrong places and
    no error anywhere.

    Pure stdlib and importable from the web container, same as the LoRA stack:
    a region naming a deleted LoRA should be a form error in milliseconds, not
    a dead job after a cold H100. That is also why `ref` is only type-checked
    here and never decoded — a base64 blob is the one field in this payload
    whose validation would cost real time, and the decode happens on the GPU
    side where the bytes are going anyway.
    """
    if not raw:
        return []

    out: list[dict[str, Any]] = []
    for i, entry in enumerate(list(raw)[:MAX_REGIONS]):
        if not isinstance(entry, dict):
            raise ValueError(f"Region {i + 1} is malformed.")
        try:
            box = {k: float(entry.get(k, d)) for k, d in
                   (("x", 0.0), ("y", 0.0), ("width", 1.0), ("height", 1.0))}
        except (TypeError, ValueError):
            raise ValueError(f"Region {i + 1} has a non-numeric box.")
        if box["width"] <= 0 or box["height"] <= 0:
            raise ValueError(f"Region {i + 1} has no area.")
        # Clamped, not rejected: a box dragged off the edge of the canvas is a
        # gesture with an obvious meaning, and refusing it would be pedantry.
        # Coordinates staying inside 0..1 also keeps `_coerce_bbox_norm` from
        # reading them as pixels, which it does for anything above 1.0.
        box["x"] = min(max(box["x"], 0.0), 1.0)
        box["y"] = min(max(box["y"], 0.0), 1.0)
        box["width"] = min(box["width"], 1.0 - box["x"])
        box["height"] = min(box["height"], 1.0 - box["y"])

        # `loras` arriving from the page, `lora`/`strength` arriving back from
        # ourselves. This function runs twice on the same rows — once in
        # /api/generate so a box naming a deleted LoRA is a form error in
        # milliseconds, and once inside the job — and it renames the field it
        # validates. So the second pass looked for a `loras` key its own output
        # does not have, took the empty branch below, and rewrote every box to
        # "None" at strength 1.0.
        #
        # Nothing showed it. A box's other fields keep their names across the
        # rename, so text and geometry survived the round trip and only the
        # identity died: the caption still placed each subject, the boxes still
        # drew, and the node got a regions_json in which no row had a LoRA —
        # which it answers by returning the model unpatched. A render that looks
        # routed and contains no LoRA at all is the failure this cost.
        #
        # `_validate_loras` is idempotent already: it reads `path` and `unet`,
        # which is what it writes. This makes its caller match rather than
        # dropping one of the two call sites, because both are load-bearing —
        # the first is the fast rejection, the second is what the graph and the
        # sidecar are built from.
        raw_loras = entry.get("loras")
        if raw_loras is None and entry.get("lora") not in (None, "", "None"):
            strength = entry.get("strength")
            raw_loras = [{"path": str(LORAS / str(entry["lora"])),
                          "unet": 1.0 if strength is None else strength}]
        lora = _validate_loras(raw_loras)
        if len(lora) > 1:
            # One LoRA per box is the node's shape, not ours to paper over: a
            # region row carries a single `lora` name. Silently keeping the
            # first would make the second token look applied.
            raise ValueError(
                f"Region {i + 1} names {len(lora)} LoRAs; a region takes one. "
                "Put the others in the main prompt to apply them to the whole "
                "canvas."
            )

        # The region's own reference photo — V9's `regions_json.ref_image`, a
        # latent mold pulling this box toward that face. It is not an
        # `extra_ref_*` plate, so it does NOT switch the run onto krea2edit and
        # does NOT need the identity-edit LoRA: molds are armed on the plain
        # single-pass path too. Anything that gates this the way the plates are
        # gated is hiding a feature that has no such cost.
        ref = entry.get("ref")
        if ref is not None and not isinstance(ref, str):
            raise ValueError(f"Region {i + 1} has a malformed reference photo.")

        out.append({
            **box,
            "prompt": str(entry.get("prompt") or "").strip(),
            "lora": lora[0]["name"] if lora else "None",
            "strength": lora[0]["unet"] if lora else 1.0,
            "ref": ref or "",
        })
    return out


# Where a box sits and how big it is, said in words. The vocabulary is mirrored
# verbatim from the node pack's own `_compile_unified_plan`, and that is the
# whole point of it: the plate path builds these sentences itself from the same
# rectangles, so inventing our own phrasing would mean dropping a scene photo
# silently re-described every box. "middle left side" on one path and "on the
# left" on the other is two different prompts for one rectangle.
def _box_horizontal(cx: float) -> str:
    if cx < 0.20:
        return "far-left side"
    if cx < 0.40:
        return "left side"
    if cx < 0.60:
        return "center"
    if cx < 0.80:
        return "right side"
    return "far-right side"


def _box_vertical(cy: float) -> str:
    if cy < 0.25:
        return "top"
    if cy < 0.45:
        return "upper portion"
    if cy < 0.65:
        return "middle"
    if cy < 0.82:
        return "lower portion"
    return "bottom"


def _box_framing(height: float) -> str:
    if height >= 0.70:
        return "a large prominent near-frame-height foreground subject"
    if height >= 0.45:
        return "a prominent medium-to-large subject"
    if height >= 0.25:
        return ("a medium-distance subject standing several steps from the "
                "camera, full body visible")
    return ("a small distant background figure far from the camera, whole body "
            "occupying only a small part of the frame")


# "a man" -> "one single man only". The node pack's own rewrite, mirrored for
# the same reason the vocabulary is: it is free anti-duplication phrasing, and
# a box that gets it on one path and not the other is a box that renders two
# people on one path and not the other.
_ONE_SUBJECT_RE = re.compile(r"^(?:a|an|one)\s+(man|woman|person)\b", re.I)

# Spelled rather than numeric, because the rest of the caption is prose and
# Qwen3-VL is reading it as prose. Indexed by count, so it stops at MAX_REGIONS.
_COUNT_WORDS = ("", "one", "two", "three", "four", "five", "six", "seven", "eight")


def _compose_caption(prompt: str, regions: list[dict[str, Any]]) -> str:
    """
    Turn the prompt and the boxes into one caption the text encoder can read.

    Without this the rectangles do almost nothing. On the plain regional path
    the node masks each LoRA's *weights* to its box, but there is no attention
    routing and no attraction field — those live in `_apply_edit_mode`, which
    only runs when a reference plate is attached. So nothing tells the model to
    put a person in the box at all: it renders whatever the prompt describes,
    wherever it likes, and the masks then apply one identity to a rectangle
    that may contain none of that person's face. Saying where each subject goes
    is what makes a box mean something on this path.

    That is also why every box gets a clause, not just the ones with words. A
    box holding only a LoRA is still the instruction "this character, here",
    and "here" is the half the token cannot say. A box holding only words gets
    placed too — no mask behind it, so it is a soft placement rather than a
    guaranteed one, but soft is what regional prompting without an identity is.

    Qwen3-VL is an instruction-tuned VLM rather than a bag-of-tokens encoder,
    which is the reason this works at all and the reason it is prose.
    """
    parts: list[str] = []
    base = (prompt or "").strip()
    if base:
        parts.append(base.rstrip(".") + ".")

    clauses: list[str] = []
    for region in regions:
        described = (region.get("prompt") or "").strip()
        has_identity = (region.get("lora") not in (None, "", "None")
                        or bool(region.get("ref_image")))
        if not described and not has_identity:
            continue
        # A box with an identity and no direction is still someone standing
        # there; the LoRA or the photo says who, so the words only have to say
        # that a single person is present.
        described = described or "a person"
        described = _ONE_SUBJECT_RE.sub(r"one single \1 only", described)
        cx = float(region["x"]) + float(region["width"]) / 2.0
        cy = float(region["y"]) + float(region["height"]) / 2.0
        clauses.append(
            f"In the {_box_vertical(cy)} {_box_horizontal(cx)}, "
            f"{described.rstrip('.')}, as {_box_framing(float(region['height']))}."
        )

    if not clauses:
        return base

    parts.extend(clauses)
    if len(clauses) > 1:
        # The failure this prevents is the classic one: ask for two people in
        # two places and get four. The node adds its own version of this
        # sentence on the plate path and does not reach this one.
        #
        # The count is spelled out because the per-clause guard cannot be
        # relied on here. That one rides on `_ONE_SUBJECT_RE`, which keys on a
        # leading article — and a box directed with a rare trigger token
        # ("alxcn, in a denim jacket") has no article to key on, so exactly the
        # descriptions a trained character uses are the ones that get no
        # singular. Naming the total covers every box however it is written.
        total = _COUNT_WORDS[len(clauses)] if len(clauses) < len(_COUNT_WORDS) \
            else str(len(clauses))
        parts.append(f"Exactly {total} distinct subjects in the frame, one in "
                     "each position described above and no others. Do not "
                     "duplicate any subject.")
    return " ".join(parts)


def _edit_lora_name(available: list[str]) -> str:
    """
    Pick a value for V12's `edit_lora` that its combo will actually accept.

    The input is required and validated against `folder_paths.get_filename_list
    ("loras")`, so any name not on the volume is rejected before the node runs —
    including, awkwardly, the identity-edit LoRA's own filename when it has not
    been downloaded. The weights are only ever *loaded* on the krea2edit path,
    so on every other render this is a required field whose value is unused,
    and the only wrong answer is one the combo rejects.

    Hence: the real file if it is there, otherwise any LoRA at all, otherwise
    the name the node itself falls back to when the volume is empty. Requests
    that genuinely need the edit path call `_require_models("krea2_edit")`,
    which is the same check every other weight gets and produces the same
    listing when it fails.
    """
    if KREA2_EDIT_LORA in available:
        return KREA2_EDIT_LORA
    return available[0] if available else KREA2_EDIT_LORA


def _lora_names_on_disk() -> list[str]:
    """Every LoRA the way ComfyUI's combo spells it — path relative to loras/."""
    if not LORAS.is_dir():
        return []
    return sorted(
        p.relative_to(LORAS).as_posix() for p in LORAS.rglob("*.safetensors")
    )


def _krea2_graph(
    *,
    model: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    batch_size: int,
    seed: int,
    steps: int,
    cfg: float,
    shift: float,
    sampler: str,
    scheduler: str,
    loras: list[dict[str, Any]],
    regions: list[dict[str, Any]] | None = None,
    scene: str | None = None,
    outfit: str | None = None,
    region_weight: float = 1.0,
) -> dict[str, Any]:
    """
    Build ComfyUI's API-format graph for one Krea 2 render.

    Written out as a dict here rather than shipped as workflow JSON for the
    same reason the video graphs are: a wrong node name should be a Python
    error next to the code that caused it, and nothing in this repo should have
    to be re-exported from a GUI when a parameter changes. The node pack's
    example workflow is a UI export that also pulls in ComfyUI-KJNodes for a
    box builder and a resolution picker — a canvas needs those, an API caller
    needs two integers and a list.

    Three shapes come out of here, and which one you get is read off what was
    attached rather than asked for:

      * no regions            — loaders, LoRA chain, KSampler. The plain path.
      * regions               — the same, plus V12 between the model and the
                                sampler, masking each region's LoRA to its box.
      * regions + a reference — V12's krea2edit path: the scene is regenerated
                                around the boxes and the identity-edit LoRA is
                                loaded.

    The middle one is the feature this backend was swapped for. It needs no
    reference image and no edit LoRA: V12 falls through to V9's likeness engine,
    which masks each LoRA's activation delta to its rectangle and samples once.
    """
    dit = MODEL_CATALOGUE["turbo" if model == "turbo" else "raw"]["dest"].name
    te = MODEL_CATALOGUE["text_encoder"]["dest"].name
    vae = MODEL_CATALOGUE["vae"]["dest"].name

    graph: dict[str, Any] = {
        "dit": {"class_type": "UNETLoader",
                "inputs": {"unet_name": dit, "weight_dtype": "default"}},
        # type="krea2" is what selects Krea2Tokenizer and the Qwen3-VL hidden
        # state Krea 2 reads. The file is the bf16 text encoder — the fp8_scaled
        # one carries ~504 extra weight_scale tensors and is rejected outright.
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": te, "type": "krea2", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        # Krea 2's latent format is `Wan21`: 16 channels, and
        # `latent_dimensions = 3`, so the sampler works on [B, 16, T, H/8, W/8].
        # `length: 1` gives ((1-1)//4)+1 = 1 frame — the same node the Wan path
        # builds its latents with, asked for a single frame.
        #
        # `EmptySD3LatentImage` also works and is what the node pack's own
        # example workflow uses: `common_ksampler` runs every latent through
        # `fix_empty_latent_channels`, which does `if latent_dimensions == 3 and
        # ndim == 4: unsqueeze(2)` — unconditionally, not just for empty ones.
        # Its 4-D output therefore arrives at the sampler as the identical
        # 5-D tensor. This node is preferred only because it states the shape
        # rather than relying on that fix-up; if you swap them back, nothing
        # changes. It is emphatically *not* what made this path render noise —
        # that was the model-sampling multiplier, see the `shift` node below.
        "latent": {"class_type": "EmptyHunyuanLatentVideo",
                   "inputs": {"width": width, "height": height,
                              "length": 1, "batch_size": batch_size}},
    }

    # Prompt-level LoRAs are the whole canvas; a region's own LoRA is applied
    # by V12 inside its box and never appears in this chain. Both weights are
    # carried because Krea 2's text encoder takes LoRA patches — the video
    # side's model-only loader has no such second number.
    model_src, clip_src = ["dit", 0], ["clip", 0]
    for i, lora in enumerate(loras):
        tag = f"lora{i}"
        graph[tag] = {
            "class_type": "LoraLoader",
            "inputs": {"model": model_src, "clip": clip_src,
                       "lora_name": lora["name"],
                       "strength_model": lora["unet"],
                       "strength_clip": lora["text_encoder"]},
        }
        model_src, clip_src = [tag, 0], [tag, 1]

    # Shift last on the model chain, before anything wraps it: it is the final
    # word on the sampling curve, and Krea 2's config already carries 1.15 —
    # this node is here so the UI can move it, not to introduce it.
    #
    # AuraFlow, not SD3, and the difference is not cosmetic. Both build the same
    # ModelSamplingDiscreteFlow; they disagree on `multiplier`, which is the
    # factor in `timestep(sigma) = sigma * multiplier` — the number the DiT is
    # actually handed. ModelSamplingSD3 hardcodes 1000. Krea 2's model config
    # asks for 1.0, and ModelSamplingAuraFlow is exactly ModelSamplingSD3 with
    # multiplier=1.0 (`patch_aura` is a one-line call into `patch`).
    #
    # Sending the SD3 one gives the model timestep ~535 where it expects ~0.53,
    # every step, so nothing denoises and the render completes at the right step
    # count, the right speed, with no error anywhere, and returns coloured
    # noise. The Wan path two sections down *does* want ModelSamplingSD3 — its
    # config takes the 1000 default — which is exactly why this looked like a
    # line worth copying. Check the model config, not the neighbouring graph.
    graph["shift"] = {"class_type": "ModelSamplingAuraFlow",
                      "inputs": {"model": model_src, "shift": shift}}
    model_src = ["shift", 0]

    if regions:
        # Boxes travel as JSON through a node of ours rather than as a literal
        # list, because ComfyUI reads any 2-element list as a [node_id, slot]
        # link — and two boxes is the commonest case this feature has. See
        # comfy_nodes/visionary_boxes.
        graph["boxes"] = {
            "class_type": "VisionaryBoxes",
            "inputs": {"boxes_json": json.dumps([
                {"x": r["x"], "y": r["y"],
                 "width": r["width"], "height": r["height"]} for r in regions
            ])},
        }
        # Index-aligned with the boxes, disabled rows included — V12 pairs box i
        # with row i by original index.
        #
        # `ref_image` is a filename in ComfyUI's input folder, which is exactly
        # what `_Comfy.stage()` returns — the caller has already staged each
        # region's photo and written the name back onto the row. Empty is the
        # common case and means "identity comes from the LoRA alone".
        regions_json = json.dumps([
            {"lora": r["lora"], "strength": r["strength"], "enable": True,
             "ref_image": r.get("ref_image", ""), "prompt": r["prompt"],
             "portrait": False}
            for r in regions
        ])

        v12: dict[str, Any] = {
            "model": model_src, "clip": clip_src, "vae": ["vae", 0],
            "bboxes": ["boxes", 0],
            "edit_lora": _edit_lora_name(_lora_names_on_disk()),
            "canvas_width": width, "canvas_height": height,
            "regions_json": regions_json,
            "prompt": prompt, "negative_prompt": negative_prompt,
            # base_strength multiplies every region's own strength. The page
            # spells it "Region weight" and it is the one global knob over a
            # stack of per-region numbers.
            "base_strength": region_weight,
            # Documented as LEAVE AT 0 by the node itself: anything above zero
            # blends every LoRA across the whole canvas, which is the identity
            # bleeding this feature exists to prevent. Not exposed.
            "blend_override": 0.0,
            # Defaults from the node, written out rather than omitted. They are
            # optional inputs, so leaving them off would work — but then the
            # graph stops recording what it ran at, and a render is only
            # reproducible from a sidecar that names every number.
            "seam_feather": 0.08,
            "ref_strength": 0.30,
            "ref_start_percent": 0.0,
            "ref_end_percent": 0.60,
            "ref_feather": 0.06,
            # The four below are the same rule extended to inputs it had missed,
            # and two of them were not the values we thought we were getting.
            # V9 declares its defaults twice — once in INPUT_TYPES for the
            # canvas widget, once in `run()`'s signature — and for an optional
            # input the graph omits, ComfyUI uses the signature. The two
            # disagree:
            #
            #   edit_lora_strength  INPUT_TYPES 0.7   run() 1.0
            #   ref_max_side        INPUT_TYPES 0     run() 1024
            #
            # So every plate render so far has run the identity-edit LoRA at
            # 1.0, which the node's own tooltip warns gives "mottled,
            # crumpled-looking texture" because V9 runs it and the character
            # LoRA in the same forward and the deltas add; and has downscaled
            # every reference to 1024, which its tooltip says "costs likeness
            # for speed". Both are the declared defaults now, spelled here so
            # the disagreement can never be silent again.
            "edit_lora_strength": 0.7,
            "compose_steps": 10,
            "compose_seed": 0,
            "ref_max_side": 0,
            # What each plate socket *is*, declared positionally: entry 1 is
            # extra_ref_1, entry 2 is extra_ref_2. Always both, whichever
            # sockets are wired, because roles are matched to sockets and the
            # unwired ones are dropped afterwards — a list compacted to fit
            # would slide the outfit's role onto the scene the moment only one
            # plate is present.
            #
            # Left at its `[]` default, every socket is `auto`, and that is
            # wrong twice. `_reference_clause` returns "" for auto with no
            # note, so the outfit was encoded into the attention sequence with
            # nothing in the prompt referring to it — krea2edit is
            # instruction-driven, so an unreferenced frame does close to
            # nothing, which reads as "outfit transfer does not work" rather
            # than as a missing sentence. And `as_scene` is
            # `role == "scene" or (auto and full-canvas box)`, so an outfit
            # dropped with no scene beside it became the canvas: the whole
            # frame replaced by a photograph of a jacket.
            #
            # The note's tail is load-bearing. With a bare noun and a subject
            # in the shot, the object branch falls back to `holding {clause}`,
            # which is not what wearing something means. The frame number is
            # deliberately absent — the node writes "the second reference"
            # itself, because it is the only thing that knows how many plates
            # ended up wired.
            "refs_json": json.dumps([
                {"role": "scene"},
                {"role": "object", "note": "outfit, worn by the subject"},
            ]),
            # A *force* flag, not the switch. V9 computes
            # `use_edit = bool(extras) or use_krea2edit or force_edit_mode`, so
            # a wired extra_ref_* already turns the edit path on by itself and
            # this only exists to reach krea2edit with no plate at all. False is
            # correct on every path we build.
            "use_krea2edit": False,
            # Auto-portrait renders a bare LoRA into a portrait and feeds it
            # back as a reference frame. Off: it costs an extra render plus a
            # model reload per region, and with two bare LoRAs it changes the
            # engine to a four-pass sequential path where only the last
            # character gets a live reference — a control whose cost is
            # measured in passes is not a default.
            "auto_portrait": False,
            "portrait_preview": False,
        }

        # A reference plate is what turns on krea2edit, so this is also what
        # decides whether the identity-edit LoRA is needed at all.
        if scene:
            graph["scene"] = {"class_type": "LoadImage", "inputs": {"image": scene}}
            v12["extra_ref_1"] = ["scene", 0]
            v12["extra_box_1"] = "0,0,1,1"
        if outfit:
            graph["outfit"] = {"class_type": "LoadImage", "inputs": {"image": outfit}}
            v12["extra_ref_2"] = ["outfit", 0]
            # Full canvas, which is what makes this an extra reference *frame*
            # rather than a paste into a sub-box — the node switches on exactly
            # that, and a sub-box here would hard-paste an outfit photo into the
            # picture instead of transferring the garment.
            v12["extra_box_2"] = "0,0,1,1"

        graph["v12"] = {"class_type": KREA2_REGIONAL_NODE, "inputs": v12}
        sampler_model, positive, negative = ["v12", 0], ["v12", 1], ["v12", 2]
    else:
        graph["pos"] = {"class_type": "CLIPTextEncode",
                        "inputs": {"text": prompt, "clip": clip_src}}
        graph["neg"] = {"class_type": "CLIPTextEncode",
                        "inputs": {"text": negative_prompt, "clip": clip_src}}
        sampler_model, positive, negative = model_src, ["pos", 0], ["neg", 0]

    graph["sample"] = {
        "class_type": "KSampler",
        "inputs": {"model": sampler_model, "seed": seed, "steps": steps,
                   "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler,
                   "positive": positive, "negative": negative,
                   "latent_image": ["latent", 0], "denoise": 1.0},
    }
    graph["decode"] = {"class_type": "VAEDecode",
                       "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}
    graph["save"] = {"class_type": "SaveImage",
                     "inputs": {"images": ["decode", 0],
                                "filename_prefix": "visionary"}}
    return graph


@app.cls(
    image=comfy_image, gpu=GPU, cpu=4.0, timeout=60 * 60,
    volumes={"/workspace": volume},
    # One container: the checkpoint is ~35 GB across DiT/VAE/TE, so a second
    # replica costs a full cold load rather than sharing the warm one.
    max_containers=1,
    scaledown_window=10 * 60,
)
@modal.concurrent(max_inputs=1)  # one GPU, one sampling loop
class ImageGenerator:
    """Holds a warm ComfyUI process, loaded with a Krea 2 checkpoint."""

    @modal.enter()
    def setup(self):
        self._comfy = _Comfy("image")
        self._comfy.start()
        # Checked once at startup rather than per request. A custom node that
        # fails to import leaves ComfyUI running happily without it, and the
        # first symptom would otherwise be a queued graph rejected for an
        # unknown class_type — which reads as our graph builder naming a node
        # wrong, minutes into a warm GPU, when the traceback that explains it
        # scrolled past during startup.
        self._comfy.require_nodes(KREA2_REGIONAL_NODE, "VisionaryBoxes")

    @modal.method()
    def generate(self, job_id: str, params: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        jobs[job_id] = {"status": "running", "phase": "loading", "stop": False}
        _reload_volume()

        model = "turbo" if str(params.get("model") or "turbo") != "raw" else "raw"

        try:
            # Inside the try, not above it: a missing weight raised out here
            # would leave the record saying "running" forever, and the UI
            # polling a job that is never going to answer.
            _require_models(model, "vae", "text_encoder")

            regions = _validate_regions(params.get("regions"))
            plates = {}
            for slot in ("scene", "outfit"):
                if params.get(slot):
                    if not regions:
                        # The plates are inputs to V12 and V12 is only in the
                        # graph when there are boxes. Silently ignoring one
                        # would be a dropped reference image with a normal
                        # picture to show for it.
                        raise ValueError(
                            f"A {slot} reference needs at least one region — "
                            "the scene is composed around the boxes."
                        )
                    _require_models("krea2_edit")
                    plates[slot] = self._comfy.stage(job_id, params[slot], slot)

            # Each region's own photo, staged the same way the plates are and
            # deliberately without their two gates: a mold is not an
            # `extra_ref_*`, so it neither turns on krea2edit nor needs the
            # identity-edit weight. The staged name goes back onto the row,
            # which is what `_krea2_graph` reads into `regions_json`.
            for i, region in enumerate(regions):
                if region["ref"]:
                    region["ref_image"] = self._comfy.stage(
                        job_id, region["ref"], f"region{i}")

            # Once, and reused by both the graph and the sidecar. Validating
            # again after the render would re-stat the volume, so a LoRA deleted
            # during a run could fail the job that already produced the picture.
            loras = _validate_loras(params.get("loras"))
            shift = float(params.get("shift") or 1.15)
            sampler = str(params.get("sampler") or IMAGE_DEFAULTS["sampler"])
            scheduler = str(params.get("scheduler") or IMAGE_DEFAULTS["scheduler"])
            # Only None and "" are unset. `or 1.0` read a typed 0 as absent and
            # rewrote it to 1.0, so /api/generate recorded the zero the page
            # sent while the render used something else — a value replaced
            # rather than honoured or refused, which is the shape of the bug
            # that let a box lose its LoRA name. It matters because
            # `base_strength` multiplies every region's own strength, so 0 is a
            # real state with a real meaning — every regional LoRA off — and it
            # has to reach the node intact rather than be rounded up on the way.
            raw_weight = params.get("region_weight")
            region_weight = 1.0 if raw_weight in (None, "") else float(raw_weight)

            seed = params.get("seed")
            seed = int(seed) if seed is not None else int.from_bytes(os.urandom(4), "big")
            steps = int(params.get("steps") or KREA2_DEFAULTS[model]["steps"])
            cfg = float(params.get("cfg_scale")
                        if params.get("cfg_scale") is not None
                        else KREA2_DEFAULTS[model]["cfg"])
            batch = max(1, min(4, int(params.get("num_images") or 1)))
            width, height = int(params.get("width") or 1024), int(params.get("height") or 1024)

            # What the encoder reads is not what was typed once there are
            # boxes: each one contributes a clause saying where its subject
            # stands. Composed here rather than in `_krea2_graph` so the record
            # below can keep both — the sentence you wrote, which is what
            # `reuse` puts back in the field, and the caption that actually ran,
            # which is the only thing that explains the picture.
            typed = str(params.get("prompt") or "")
            caption = _compose_caption(typed, regions) if regions else typed

            graph = _krea2_graph(
                model=model,
                prompt=caption,
                negative_prompt=str(params.get("negative_prompt") or ""),
                width=width, height=height, batch_size=batch,
                seed=seed, steps=steps, cfg=cfg,
                shift=shift, sampler=sampler, scheduler=scheduler,
                loras=loras, regions=regions, region_weight=region_weight,
                **plates,
            )

            info = {"width": width, "height": height, "seed": seed, "steps": steps}
            _publish(job_id, phase="generate", step=0, total_steps=steps,
                     percent=0, **info)
            out_names = self._comfy.run(job_id, graph, what="image")
        except StopRequested:
            res = {"status": "stopped", "job_id": job_id, "files": [],
                   "duration_s": round(time.time() - started, 1)}
            _publish(job_id, **res)
            return res
        except Exception as exc:
            _publish(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
            raise

        from PIL import Image, PngImagePlugin

        out_dir = OUTPUTS / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%H%M%S")
        report = {
            "sampler": sampler, "scheduler": scheduler,
            "cfg_scale": cfg, "shift": shift,
            "loras": [{"name": l["name"], "unet": l["unet"],
                       "text_encoder": l["text_encoder"]} for l in loras],
            # Boxes as a 4-list in the order the row's fields read, so `reuse`
            # can put them straight back into x/y/w/h without a schema.
            #
            # `ref` is a bool, never the photo. This dict is the job record and
            # the job record is polled every 400ms — eight base64 photographs in
            # something on that loop is the exact failure the "keep the polled
            # thing small" rule exists for. The staged file is gone with the
            # container anyway, so there is nothing here a caller could reuse.
            "regions": [{"box": [r["x"], r["y"], r["width"], r["height"]],
                         "lora": r["lora"], "strength": r["strength"],
                         "prompt": r["prompt"],
                         "ref": bool(r.get("ref_image"))} for r in regions],
            "region_weight": region_weight,
            # Only when it differs from what was typed. A caption nobody wrote
            # is the one thing about a regional render that cannot be worked
            # out from the fields, and "why is he on the right" has no answer
            # without it — but repeating the prompt back on every plain render
            # would be the record describing itself.
            **({"caption": caption} if caption != typed else {}),
            # Which engine ran, because the three are not the same picture and
            # the numbers beside them do not say which one produced it.
            "mode": ("krea2edit" if plates else "regional") if regions else "plain",
            **info,
        }
        names = []
        for i, src in enumerate(out_names):
            name = f"{stamp}_{i:02d}.png"
            # Re-encoded rather than copied, because the parameters block has to
            # go in and ComfyUI ran with --disable-metadata. The alternative is
            # a PNG whose settings live only in a job record that expires,
            # which is the thing every existing tool — PNG Info tabs, ComfyUI,
            # the gallery — reads a file to avoid.
            with Image.open(COMFY / "output" / src) as im:
                png = PngImagePlugin.PngInfo()
                png.add_text("parameters", _infotext(
                    prompt=str(params.get("prompt") or ""),
                    negative_prompt=str(params.get("negative_prompt") or ""),
                    model=model,
                    # Per-image seed, not the batch's first: ComfyUI advances
                    # the noise seed per latent in a batch, so a metadata block
                    # reporting the same seed for all four cannot reproduce
                    # three of them.
                    seed=seed + i,
                    report=report,
                ))
                im.save(out_dir / name, pnginfo=png)
            names.append(name)

        _write_output_meta(
            out_dir, kind="image", job_id=job_id, model=model,
            prompt=str(params.get("prompt") or ""),
            negative_prompt=str(params.get("negative_prompt") or ""),
            created=time.time(), **_shot_meta(params), **report,
        )
        volume.commit()

        # Only filenames go into the job record. The PNGs themselves are served
        # by /api/file/{job_id}/{name} straight off the volume — a 1024px base64
        # image is megabytes, and this dict is polled several times a second.
        # `files` is also what the canvas builds its <img> tags from, so the
        # completed record is the last round trip a run needs.
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

# The longest side a reference is allowed to arrive at, which is what makes "max
# detail" a control with a range rather than a bill the camera writes.
#
# The node's "max" scale is `min(1.0, 2048 / min(w, h))` — a floor under the
# short edge and no cap at all on the area. A 4032x3024 straight off a phone
# therefore resolves to 2720x2048, which is 21,760 latent tokens against the
# 3,996 "match" would have given the same file, and those tokens ride every
# step. Nine of them add 196k to the 149k a 5 s 16:9 clip already carries, on a
# card that is holding 42.5 GB of weights before any of it, so the run died in
# step 0 with `Allocation on device` and ComfyUI's canned advice about a batch
# size this path does not have. Every other lever here — tier, seconds, aspect,
# steps — is a control with a range. This was the one whose price was set by
# whichever file you happened to drop, and the page could not have told you.
#
# 1536 is REF_MAX, the number the image side already caps its reference photos
# at. Reused rather than tuned, because it is the same decision arriving from
# the payload side, and because it leaves "max" meaning something: 6,912 tokens
# against match's 3,996 for that phone photo, 9,216 at the square worst case, so
# nine references cost at most 83k tokens where they used to cost 196k.
H3_REF_MAX_SIDE = 1536

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


def _fit_reference(path: Path) -> None:
    """
    Shrink one staged reference to H3_REF_MAX_SIDE, in place.

    Bounding the file is what bounds the run: under 2048 on the short edge the
    node's "max" scale is exactly 1.0, so the pixels written here are the pixels
    the DiT sees, and only one place decided how big the picture is. That is the
    same reason `ref_max_side` is pinned to 0 on the image side — two things
    resizing the same photograph is two things to check when it comes out soft.

    Applied to both modes. "match" is already bounded in tokens, but not in what
    PIL and the VAE encoder chew through on the way there, and one rule is
    easier to hold than one rule per mode.

    Rewrites only when it actually resizes — and bakes the EXIF rotation in when
    it does, because ComfyUI's LoadImage is the reader that applies the tag here
    and a resized copy saved without one would reach the DiT sideways. Same trap
    `_upright_inplace` exists for, reached from the other end: not a reader that
    forgets the tag, but a writer that drops it.
    """
    from PIL import Image

    tmp = path.with_name(path.name + ".fit")
    try:
        with Image.open(path) as im:
            src = _upright(im)
            w, h = src.size
            if max(w, h) <= H3_REF_MAX_SIDE:
                return
            scale = H3_REF_MAX_SIDE / max(w, h)
            size = (max(1, round(w * scale)), max(1, round(h * scale)))
            src.resize(size, Image.LANCZOS).save(tmp, "PNG")
        tmp.replace(path)
        print(f"[video] reference {path.name}: {w}x{h} -> {size[0]}x{size[1]}",
              flush=True)
    except Exception as exc:
        # A reference that cannot be read is not one that can be measured
        # either, and LoadImage is about to fail on it with a better message
        # than anything guessable from here.
        tmp.unlink(missing_ok=True)
        print(f"[video] could not cap reference {path.name}: {exc}", flush=True)


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


# --------------------------------------------------------------------------
# The shot palette
#
# H3 does not read a paragraph. It reads a document with named fields, and
# MiniMax publishes the format in the model repo — VIDEO_PROMPT_WRITING_GUIDE
# _base_en.md and _ref_en.md. The composer offered one textarea for it, which is
# the whole reason a first-time user has to guess where camera direction goes,
# whether tone belongs in the sentence, and what a comma does. It is not a
# grammar anyone could infer; it is a schema, and the app can simply emit it.
#
# No LLM. A closed vocabulary is a table, the structure is a form, and the
# user's own sentence is the one part that was never the problem. An LLM here
# would be a GPU cold start or an external key to do work a dict does.
#
# Two rules this whole section is built on:
#
# - **A pill is a word you did not have to guess.** Anything with a closed
#   vocabulary — camera, framing, lens, light, tone, action, foley, score —
#   is a pill, never a second box to type in. The prompt field keeps only what
#   nothing else can say: who is in the shot and what happens.
# - **No pills, no document.** With nothing chosen the compiler returns the
#   typed text byte-for-byte, so this is strictly additive: every prompt that
#   worked yesterday is still exactly what reaches the encoder. The document
#   only appears once you have said something that needs one.
# --------------------------------------------------------------------------

# Verbatim from the base guide. These are a contract with the checkpoint rather
# than phrasing we chose, so they are quoted exactly and not tidied — including
# the guide's own inconsistency, which is worth knowing about before someone
# "fixes" it: i2va and l2va bracket their labels (`<Picture 1>`, `[Shot 1]`)
# and fl2va does not. Both spellings are the guide's, in the same file.
#
# `S.SS` is the clip length the composer already knows, so it is filled rather
# than left literal — a placeholder that reaches the encoder is a placeholder
# the encoder conditions on.
H3_ALIGN = {
    "t2va": "",
    "i2va": ("For the target video, at 0.00 seconds into the target video, "
             "<Picture 1> (from [Shot 1]) is fully referenced."),
    "fl2va": ("How the reference pictures align with the target video — "
              "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the "
              "target video; Picture 2 (from Shot N) aligns with the "
              "{s}-second mark of the target video."),
    "l2va": ("How the reference pictures align with the target video — "
             "<Picture 1> (from [Shot N]) aligns with the {s}-second mark of "
             "the target video."),
}

# The eleven the model card calls stable support. Deliberately not free text:
# the tag is a thing you cannot know to write, which is the entire reason
# dialogue is a pill instead of something you type into the prompt. The guide
# itself only ever demonstrates [English] and never enumerates the rest, so the
# list comes from the model card's "System Overview" — if that changes, this is
# the line to change with it.
H3_LANGUAGES = ["English", "Chinese", "Spanish", "French", "German", "Italian",
                "Portuguese", "Russian", "Japanese", "Korean", "Arabic"]

# A valued pill holds a line of dialogue or a foley description, and neither is
# a place to accept unbounded text: the whole document is one conditioner
# input, and a pill is not the field to discover that in.
SHOT_VALUE_MAX = 400

# Reference roles. A chip's role is what finally makes "do not describe your
# reference image" enforceable rather than advice — there is now somewhere for
# that description to go which is not the prompt field, and it is one click.
#
# `noun` builds the guide's own subject_definitions construction, whose shape is
# `<Subject N> is the {noun} in <Picture M>`; `retain` builds the sentence in
# retention_analysis that says which features of it must survive.
SHOT_REF_ROLES = {
    "identity": {"label": "Identity",
                 "noun": "person", "retain": "facial structure, hair and build"},
    "wardrobe": {"label": "Wardrobe",
                 "noun": "wardrobe", "retain": "garments, their cut and colour"},
    "location": {"label": "Location",
                 "noun": "location", "retain": "architecture, materials and layout"},
    "style": {"label": "Style",
              "noun": "visual style", "retain": "palette, contrast and grain"},
    "prop": {"label": "Prop",
             "noun": "prop", "retain": "shape, material and markings"},
    "action": {"label": "Action",
               "noun": "action", "retain": "the motion and its timing"},
}

# The vocabulary itself.
#
# `phrase` is the exact wording that gets written, so a pill teaches the
# phrasing it produces rather than standing in for it — "slow push-in", "stable
# rear tracking shot". Camera phrases in particular carry the guide's three
# required dimensions (motion type + amplitude + speed) because a bare "push in"
# is the half of a camera instruction H3 reads worst.
#
# Per group:
#   pick   "one" replaces within the group, "many" stacks. The guide is explicit
#          that a clip gets one camera move unless the timing is spelled out,
#          and framing and angle are single-valued by physics.
#   join   "list" folds into one comma-separated sentence; "sentence" stands
#          alone. This is where the punctuation question is answered once, in
#          code, instead of being guessed per prompt.
#   slot   position relative to the typed sentence, which sits at 0. Camera goes
#          last because the guide's rule is to describe the move after the thing
#          it is moving around. Groups that share a slot share a sentence, which
#          is why framing, angle, light and tone are all at -10: they are four
#          groups on the palette because that is how you pick them, and one
#          clause in the document because that is how a shot is described.
#   field  which of H3's three audio/visual fields it compiles into.
#   image  whether Krea 2 reads it at all. Camera, action and both audio groups
#          do not exist on the image side.
#   needs  a `supports` key the video model must carry. Wan is silent, so its
#          sound and score pills dim rather than disappear — the established
#          rule from the keyframes/references row. An item may carry its own,
#          which then stands in for the group's: see `say.dialogue`.
#   glyph  the CSS class that animates the shared tile skeleton. Not a drawing:
#          forty bespoke SVGs would drift apart, one skeleton plus a class per
#          move does not.
SHOT_VOCAB: list[dict[str, Any]] = [
    {"key": "framing", "label": "Framing", "pick": "one", "join": "list",
     "slot": -10, "field": "visual", "image": True, "needs": None, "items": [
        {"key": "xwide", "label": "extreme wide", "glyph": "fr-xw",
         "phrase": "an extreme wide shot"},
        {"key": "wide", "label": "wide", "glyph": "fr-w",
         "phrase": "a wide shot"},
        {"key": "medium", "label": "medium", "glyph": "fr-m",
         "phrase": "a medium shot"},
        {"key": "mcu", "label": "medium close-up", "glyph": "fr-mcu",
         "phrase": "a medium close-up"},
        {"key": "cu", "label": "close-up", "glyph": "fr-cu",
         "phrase": "a close-up"},
        {"key": "xcu", "label": "extreme close-up", "glyph": "fr-xcu",
         "phrase": "an extreme close-up"},
        {"key": "ots", "label": "over-the-shoulder", "glyph": "fr-ots",
         "phrase": "an over-the-shoulder shot"},
        {"key": "pov", "label": "POV", "glyph": "fr-pov",
         "phrase": "a first-person point-of-view shot"},
    ]},
    {"key": "angle", "label": "Angle", "pick": "one", "join": "list",
     "slot": -10, "field": "visual", "image": True, "needs": None, "items": [
        {"key": "eye", "label": "eye level", "glyph": "an-eye",
         "phrase": "shot at eye level"},
        {"key": "low", "label": "low", "glyph": "an-low",
         "phrase": "shot from a low angle"},
        {"key": "high", "label": "high", "glyph": "an-high",
         "phrase": "shot from a high angle"},
        {"key": "bird", "label": "bird's eye", "glyph": "an-bird",
         "phrase": "shot from directly overhead, a bird's-eye view"},
        {"key": "worm", "label": "worm's eye", "glyph": "an-worm",
         "phrase": "shot from ground level looking up, a worm's-eye view"},
        {"key": "dutch", "label": "Dutch", "glyph": "an-dutch",
         "phrase": "shot on a canted Dutch angle"},
    ]},
    {"key": "light", "label": "Light", "pick": "many", "join": "list",
     "slot": -10, "field": "visual", "image": True, "needs": None, "items": [
        {"key": "window", "label": "window light", "glyph": "li-window",
         "phrase": "lit by soft daylight from a window"},
        {"key": "golden", "label": "golden hour", "glyph": "li-golden",
         "phrase": "lit by low golden-hour sun"},
        {"key": "overcast", "label": "overcast", "glyph": "li-overcast",
         "phrase": "lit by flat overcast daylight"},
        {"key": "hardsun", "label": "hard sun", "glyph": "li-hardsun",
         "phrase": "lit by hard direct sunlight with sharp shadows"},
        {"key": "neon", "label": "neon", "glyph": "li-neon",
         "phrase": "lit by coloured neon"},
        {"key": "candle", "label": "candlelight", "glyph": "li-candle",
         "phrase": "lit by warm, flickering candlelight"},
        {"key": "practical", "label": "practicals", "glyph": "li-practical",
         "phrase": "lit by practical lamps visible in the frame"},
        {"key": "silhouette", "label": "silhouette", "glyph": "li-silhouette",
         "phrase": "backlit so the subject reads as a silhouette"},
        {"key": "top", "label": "top light", "glyph": "li-top",
         "phrase": "lit from directly above"},
    ]},
    {"key": "tone", "label": "Tone", "pick": "many", "join": "list",
     "slot": -10, "field": "visual", "image": True, "needs": None, "items": [
        {"key": "doc", "label": "documentary", "glyph": "to-doc",
         "phrase": "shot with documentary realism"},
        {"key": "noir", "label": "noir", "glyph": "to-noir",
         "phrase": "in high-contrast film noir"},
        {"key": "s16", "label": "16mm", "glyph": "to-s16",
         "phrase": "on grainy 16mm film"},
        {"key": "anamorphic", "label": "anamorphic", "glyph": "to-anamorphic",
         "phrase": "on anamorphic widescreen lenses"},
        {"key": "highkey", "label": "high-key", "glyph": "to-highkey",
         "phrase": "high-key and bright"},
        {"key": "desat", "label": "desaturated", "glyph": "to-desat",
         "phrase": "desaturated, close to monochrome"},
        {"key": "contrast", "label": "high contrast", "glyph": "to-contrast",
         "phrase": "in high contrast with deep blacks"},
    ]},
    {"key": "action", "label": "Action", "pick": "many", "join": "sentence",
     "slot": 10, "field": "visual", "image": False, "needs": None, "items": [
        {"key": "fight", "label": "fight", "glyph": "ac-fight",
         "phrase": "They fight, trading fast, tightly choreographed blows."},
        {"key": "kiss", "label": "kiss", "glyph": "ac-kiss",
         "phrase": "They move together and kiss."},
        {"key": "talk", "label": "conversation", "glyph": "ac-talk",
         "phrase": "They talk to each other, taking turns to speak."},
        {"key": "walktalk", "label": "walk and talk", "glyph": "ac-walktalk",
         "phrase": "They walk side by side, talking as they go."},
        {"key": "embrace", "label": "embrace", "glyph": "ac-embrace",
         "phrase": "They embrace and hold each other."},
        {"key": "chase", "label": "chase", "glyph": "ac-chase",
         "phrase": "One runs and the other chases, closing the distance."},
        {"key": "reveal", "label": "reveal", "glyph": "ac-reveal",
         "phrase": "The subject is revealed as the frame clears."},
        {"key": "handoff", "label": "hand-off", "glyph": "ac-handoff",
         "phrase": "One hands an object to the other."},
        {"key": "turn", "label": "turn to camera", "glyph": "ac-turn",
         "phrase": "The subject turns to face the camera."},
        {"key": "laugh", "label": "laugh", "glyph": "ac-laugh",
         "phrase": "They laugh together."},
    ]},
    # Valued, and the reason the rail can hold a line of dialogue at all: a
    # closed vocabulary cannot contain "Take the morning with you", so choosing
    # the pill reveals a place to write and nothing before that.
    {"key": "say", "label": "Speech & text", "pick": "many", "join": "sentence",
     "slot": 20, "field": "visual", "image": False, "needs": None, "items": [
        # The one item that needs what its group does not. On-screen text is a
        # picture of words and any video model can render it; a spoken line is
        # audio, and on a silent family `<d>[English] …</d>` is H3's syntax
        # arriving at umT5, which reads it as literal angle brackets rather
        # than ignoring it. So `needs` is per item as well as per group.
        {"key": "dialogue", "label": "dialogue", "glyph": "sa-dialogue",
         "valued": "dialogue", "phrase": "", "needs": "audio",
         "hint": "What they say — kept word for word"},
        {"key": "screen", "label": "on-screen text", "glyph": "sa-screen",
         "valued": "text", "phrase": "",
         "hint": "The exact words on screen"},
    ]},
    {"key": "camera", "label": "Camera", "pick": "one", "join": "sentence",
     "slot": 40, "field": "visual", "image": False, "needs": None, "items": [
        {"key": "pushin", "label": "push in", "glyph": "ca-push",
         "phrase": "The camera pushes in slowly, a small and steady move."},
        {"key": "pullout", "label": "pull out", "glyph": "ca-pull",
         "phrase": "The camera pulls out slowly, a small and steady move."},
        {"key": "panl", "label": "pan left", "glyph": "ca-panl",
         "phrase": "The camera pans left at a moderate speed, a medium-amplitude move."},
        {"key": "panr", "label": "pan right", "glyph": "ca-panr",
         "phrase": "The camera pans right at a moderate speed, a medium-amplitude move."},
        {"key": "tiltu", "label": "tilt up", "glyph": "ca-tiltu",
         "phrase": "The camera tilts up slowly, a small move."},
        {"key": "tiltd", "label": "tilt down", "glyph": "ca-tiltd",
         "phrase": "The camera tilts down slowly, a small move."},
        {"key": "truckl", "label": "truck left", "glyph": "ca-truckl",
         "phrase": "The camera trucks left at a steady, moderate speed."},
        {"key": "truckr", "label": "truck right", "glyph": "ca-truckr",
         "phrase": "The camera trucks right at a steady, moderate speed."},
        {"key": "pedu", "label": "pedestal up", "glyph": "ca-pedu",
         "phrase": "The camera pedestals up slowly, a small move."},
        {"key": "pedd", "label": "pedestal down", "glyph": "ca-pedd",
         "phrase": "The camera pedestals down slowly, a small move."},
        {"key": "orbit", "label": "orbit", "glyph": "ca-orbit",
         "phrase": "The camera orbits the subject at a slow, steady speed."},
        {"key": "arc", "label": "arc", "glyph": "ca-arc",
         "phrase": "The camera arcs around the subject in a slow, wide move."},
        {"key": "craneu", "label": "crane up", "glyph": "ca-craneu",
         "phrase": "The camera cranes up in a large, slow move."},
        {"key": "craned", "label": "crane down", "glyph": "ca-craned",
         "phrase": "The camera cranes down in a large, slow move."},
        {"key": "trackside", "label": "track side", "glyph": "ca-trackside",
         "phrase": "A stable side-tracking shot keeps pace with the subject."},
        {"key": "trackrear", "label": "track rear", "glyph": "ca-trackrear",
         "phrase": "A stable rear tracking shot follows the subject from behind."},
        {"key": "handheld", "label": "handheld", "glyph": "ca-handheld",
         "phrase": "The camera is handheld, with small, constant, organic movement."},
        {"key": "whip", "label": "whip pan", "glyph": "ca-whip",
         "phrase": "The camera whip-pans, a large and fast move."},
        {"key": "rack", "label": "rack focus", "glyph": "ca-rack",
         "phrase": "The focus racks from the foreground to the subject."},
        {"key": "zoom", "label": "zoom in", "glyph": "ca-zoom",
         "phrase": "The lens zooms in slowly, a small and steady move."},
        {"key": "static", "label": "locked off", "glyph": "ca-static",
         "phrase": "The camera is locked off and does not move."},
    ]},
    {"key": "sound", "label": "Sound", "pick": "many", "join": "list",
     "slot": 0, "field": "sound", "image": False, "needs": "audio", "items": [
        {"key": "roomtone", "label": "room tone", "glyph": "so-room",
         "phrase": "quiet room tone"},
        {"key": "footsteps", "label": "footsteps", "glyph": "so-steps",
         "phrase": "footsteps on the floor"},
        {"key": "wind", "label": "wind", "glyph": "so-wind",
         "phrase": "wind moving through the space"},
        {"key": "rain", "label": "rain", "glyph": "so-rain",
         "phrase": "rain falling steadily"},
        {"key": "traffic", "label": "traffic", "glyph": "so-traffic",
         "phrase": "distant traffic"},
        {"key": "crowd", "label": "crowd", "glyph": "so-crowd",
         "phrase": "a low crowd murmur"},
        {"key": "breathing", "label": "breathing", "glyph": "so-breath",
         "phrase": "audible breathing"},
        {"key": "cloth", "label": "cloth", "glyph": "so-cloth",
         "phrase": "cloth rustling with every movement"},
        {"key": "water", "label": "water", "glyph": "so-water",
         "phrase": "running water"},
        {"key": "fire", "label": "fire", "glyph": "so-fire",
         "phrase": "a fire crackling"},
        {"key": "other", "label": "other", "glyph": "so-other",
         "valued": "text", "phrase": "",
         "hint": "e.g. ice tapping the side of a crystal glass"},
    ]},
    {"key": "score", "label": "Score", "pick": "many", "join": "list",
     "slot": 0, "field": "score", "image": False, "needs": "audio", "items": [
        # First in its own group, because it is the one worth the whole feature:
        # H3 invents a soundtrack for every clip, and it does that because
        # nothing has ever told it not to. `non_diegetic_music: N/A` is the
        # guide's own value for "no score", and it is also what any document
        # with no score pill emits — this pill exists so that "silent" is a
        # thing you can choose on its own, without stacking four other pills to
        # get a document built.
        {"key": "silent", "label": "no score", "glyph": "sc-silent",
         "solo": True, "phrase": ""},
        {"key": "piano", "label": "solo piano", "glyph": "sc-piano",
         "phrase": "solo piano"},
        {"key": "strings", "label": "strings", "glyph": "sc-strings",
         "phrase": "sustained strings"},
        {"key": "synth", "label": "synth pad", "glyph": "sc-synth",
         "phrase": "a synth pad"},
        {"key": "perc", "label": "percussion", "glyph": "sc-perc",
         "phrase": "percussion"},
        {"key": "guitar", "label": "guitar", "glyph": "sc-guitar",
         "phrase": "acoustic guitar"},
        {"key": "slow", "label": "slow", "glyph": "sc-slow",
         "phrase": "at a slow tempo"},
        {"key": "mid", "label": "mid tempo", "glyph": "sc-mid",
         "phrase": "at a mid tempo"},
        {"key": "driving", "label": "driving", "glyph": "sc-driving",
         "phrase": "at a driving tempo"},
        {"key": "swelling", "label": "swelling", "glyph": "sc-swell",
         "phrase": "swelling through the clip"},
        {"key": "steady", "label": "steady", "glyph": "sc-steady",
         "phrase": "holding a steady level"},
        {"key": "fading", "label": "fading", "glyph": "sc-fade",
         "phrase": "fading out towards the end"},
        {"key": "other", "label": "other", "glyph": "sc-other",
         "valued": "text", "phrase": "",
         "hint": "e.g. a lone cello, barely there"},
    ]},
]

# `group.item`, not `item` — `static`, `other` and `low` each occur in more than
# one group, and a flat key space would have made two different pills the same
# pill. The compound key is also what makes a payload readable in a sidecar a
# year later without this table in front of you.
SHOT_ITEMS: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    f"{g['key']}.{it['key']}": (g, it) for g in SHOT_VOCAB for it in g["items"]
}


def _validate_shot(raw: Any) -> list[dict[str, Any]]:
    """
    Normalise the pill rail into what the compiler takes, or say what is wrong.

    Importable from the CPU web container for the same reason `_validate_loras`
    is: an unknown pill key is a form error in milliseconds, not a cold H100
    discovering it after 42 GB of weights are resident.

    Unknown keys are rejected rather than dropped. A pill silently ignored is
    the failure this whole feature exists to remove — the user picked a word and
    the model never saw it, which is indistinguishable from the model ignoring
    the word.
    """
    if not raw:
        return []
    if isinstance(raw, dict):          # {"pills": [...]} as well as a bare list
        raw = raw.get("pills") or []

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in list(raw)[:len(SHOT_ITEMS)]:
        if isinstance(entry, str):
            entry = {"key": entry}
        if not isinstance(entry, dict):
            raise ValueError(f"Not a shot pill: {entry!r}")
        key = str(entry.get("key") or "")
        if key not in SHOT_ITEMS:
            raise ValueError(f"No such shot pill: {key!r}")
        if key in seen:
            continue
        seen.add(key)
        item = SHOT_ITEMS[key][1]

        pill: dict[str, Any] = {"key": key}
        if item.get("valued"):
            # Verbatim, punctuation included — the guide is explicit that what
            # is inside <d>…</d> is preserved as written, and sentence-casing or
            # re-punctuating a line of dialogue is exactly the silent corruption
            # that rule is about.
            #
            # Whitespace is the one exception, and it is not a softening of the
            # rule but a requirement of the format: the document is one field
            # per line, so a pasted line break inside a value ends the field and
            # turns the rest of the sentence into what looks like another one.
            # Collapsing runs of whitespace keeps every word and every mark and
            # cannot produce that. The field is an <input>, so this only ever
            # fires on a paste or a client that is not this page.
            pill["value"] = _oneline(str(entry.get("value") or ""))[:SHOT_VALUE_MAX]
        if item.get("valued") == "dialogue":
            lang = str(entry.get("lang") or "English")
            if lang not in H3_LANGUAGES:
                raise ValueError(f"No such dialogue language: {lang!r}. "
                                 f"One of: {', '.join(H3_LANGUAGES)}")
            pill["lang"] = lang
        out.append(pill)

    # One camera move, one framing, one angle — the guide's rule, and the page
    # enforces it by replacing rather than stacking. Enforced again here because
    # a stale tab is a client that can send anything, and because the rule that
    # matters is not "the page prevents it" but "the encoder never sees two".
    #
    # Last wins throughout, which is what clicking a second tile does on screen:
    # a `pick:"one"` group keeps only its newest, and a `solo` item (no score)
    # and the rest of its group evict each other in whichever direction the
    # clicks arrived.
    kept: dict[str, dict[str, Any]] = {}
    for pill in out:
        group, item = SHOT_ITEMS[pill["key"]]
        drop = group["pick"] == "one" or item.get("solo")
        kept = {k: v for k, v in kept.items()
                if SHOT_ITEMS[k][0]["key"] != group["key"]
                or not (drop or SHOT_ITEMS[k][1].get("solo"))}
        kept[pill["key"]] = pill
    return list(kept.values())


def _validate_ref_roles(raw: Any, count: int) -> list[str]:
    """One role per image reference, positional, blank where none was chosen."""
    roles = [str(r or "") for r in (list(raw) if raw else [])][:count]
    for r in roles:
        if r and r not in SHOT_REF_ROLES:
            raise ValueError(f"No such reference role: {r!r}. "
                             f"One of: {', '.join(SHOT_REF_ROLES)}")
    return roles + [""] * (count - len(roles))


def _h3_task(first_frame: Any, last_frame: Any,
             references: Any, ref_videos: Any) -> str:
    """
    Which of the guide's four tasks a request is.

    Deliberately a second, finer read than the one `/api/video` makes. That one
    collapses to `ref2va` or `fl2va`, which is exactly right for *which
    checkpoint loads* — first-only, last-only and both are the same weights —
    and too coarse for *which alignment instruction*, where they are three
    different sentences and getting it wrong tells the model a picture sits at a
    timestamp it does not.
    """
    if references or ref_videos:
        return "ref2va"
    if first_frame and last_frame:
        return "fl2va"
    if first_frame:
        return "i2va"
    if last_frame:
        return "l2va"
    return "t2va"


def _shot_phrases(pills: list[dict[str, Any]], *, side: str,
                  audio: bool = True) -> dict[tuple[int, str, str], list[str]]:
    """
    Fold the chosen pills into (slot, join) buckets, in vocabulary order.

    Vocabulary order, not click order, is the entire point: clause position is
    the one thing you cannot get wrong by hand any more. Two pills from the same
    group keep the order the table lists them in, so "close-up, low angle" never
    comes back as "low angle, close-up" because of the order they were clicked.

    `audio=False` is the silent families. It drops by `needs` rather than by
    field, because dialogue is the case that breaks the simpler rule: it lands
    in the *visual* description and is still audio, and `<d>[English] …</d>`
    reaching umT5 is a pair of angle brackets in the prompt rather than a line
    anybody says.
    """
    chosen = {p["key"]: p for p in pills}
    buckets: dict[tuple[int, str, str], list[str]] = {}
    for group in SHOT_VOCAB:
        if side == "image" and not group["image"]:
            continue
        for item in group["items"]:
            if not audio and (item.get("needs") or group["needs"]) == "audio":
                continue
            key = f"{group['key']}.{item['key']}"
            pill = chosen.get(key)
            if pill is None:
                continue
            text = _shot_text(group, item, pill)
            if not text:
                continue
            buckets.setdefault(
                (group["slot"], group["join"], group["field"]), []).append(text)
    return buckets


def _shot_text(group: dict[str, Any], item: dict[str, Any],
               pill: dict[str, Any]) -> str:
    """One pill's contribution, which for a valued pill is whatever was typed."""
    kind = item.get("valued")
    if not kind:
        return item["phrase"]
    value = pill.get("value") or ""
    # An empty valued pill compiles to nothing and is never a validation error.
    # It is a decision you have started and not finished, which is a state the
    # rail can simply show by being visibly empty.
    if not value:
        return ""
    if kind == "dialogue":
        # (S1) unconditionally: a second speaker is only worth having once there
        # are shots to cut between, and multi-shot prompting is not in this pass.
        return (f"<Subject 1> (S1) says: "
                f"<d>[{pill.get('lang') or 'English'}] {value}</d>")
    if group["key"] == "say":
        return (f'On-screen text reads "{value}", rendered exactly as written '
                f'and not translated.')
    return value


def _shot_sentence(parts: list[str]) -> str:
    """A comma list, capitalised once and closed once."""
    if not parts:
        return ""
    body = ", ".join(parts)
    return body[0].upper() + body[1:] + "."


def _oneline(text: str) -> str:
    """
    Runs of whitespace collapsed to single spaces, and nothing else touched.

    The document is one field per line, so any newline that reaches it ends a
    field early and leaves the rest of the sentence looking like the start of
    another. The prompt box is a textarea and a two-line prompt is a thing this
    page deliberately supports — see the clause-reordering chord — so this is
    not a rare paste, it is the ordinary case.
    """
    return " ".join(text.split())


def _close(text: str) -> str:
    """Someone's sentence, closed if they did not close it — and not otherwise."""
    text = text.strip()
    return text + "." if text and text[-1] not in ".!?…\"'" else text


def _shot_join(parts: list[str]) -> str:
    """
    Sentences, joined so a lowercase fragment does not follow a full stop.

    The image side is where this bites: people type "a portrait of k3nan", not
    "A portrait of k3nan", and "A medium close-up. a portrait of k3nan." reads
    as a bug. The obvious fix — capitalise it — is the one thing that must not
    happen here. `k3nan` typed as a plain trigger word is a token the text
    encoder distinguishes from `K3nan`, and silently upper-casing the first
    character of someone's prompt would weaken the LoRA they trained. So no
    character of the user's text is touched; only the separator in front of it
    is chosen, and the preceding clause's full stop softens to a comma.
    """
    out = ""
    for part in [p for p in parts if p]:
        if not out:
            out = part
        elif part[:1].islower() and out.endswith("."):
            out = out[:-1] + ", " + part
        else:
            out += " " + part
    return out


def _shot_body(typed: str, buckets: dict[tuple[int, str, str], list[str]],
               *, field: str = "visual") -> str:
    """
    The typed sentence with its pills folded in around it, in slot order.

    The typed text is closed with a full stop if it does not close itself, and
    otherwise left alone: it is the one part of the document the user wrote, and
    rewriting someone's sentence is not something a compiler gets to do.
    """
    typed = _close(_oneline(typed))
    out: list[str] = []
    for slot, join, fld in sorted(k for k in buckets if k[2] == field):
        if slot >= 0 and typed:
            out.append(typed)
            typed = ""
        parts = buckets[(slot, join, fld)]
        out.append(_shot_sentence(parts) if join == "list" else " ".join(parts))
    if typed:
        out.append(typed)
    return _shot_join(out)


def _first_sentence(text: str) -> str:
    """Up to the first sentence end, for the one field that wants a précis."""
    for i, ch in enumerate(text):
        if ch in ".!?" and (i + 1 >= len(text) or text[i + 1] == " "):
            return text[:i + 1]
    return text


def _shot_audio(buckets: dict[tuple[int, str, str], list[str]],
                field: str) -> str:
    """The soundscape or the score, as one sentence over its pills."""
    parts = [p for k in sorted(k for k in buckets if k[2] == field)
             for p in buckets[k]]
    if not parts:
        return ""
    lead = "The soundtrack is " if field == "score" else "The scene carries "
    return lead + ", ".join(parts) + "."


def _compile_h3_prompt(*, typed: str, pills: list[dict[str, Any]],
                       task: str, seconds: float,
                       roles: list[str] | None = None) -> str:
    """
    The document H3 actually reads, assembled from a sentence and some pills.

    Returns `typed` untouched when there is nothing to compile — no pills, no
    reference roles. That is not a shortcut, it is the contract: this feature
    must not change what a prompt written before it meant, and a bare sentence
    wrapped in field labels is a different input to the encoder than the bare
    sentence.
    """
    roles = [r for r in (roles or [])]
    if not pills and not any(roles):
        return typed
    buckets = _shot_phrases(pills, side="video")

    lines: list[str] = []
    if task == "ref2va":
        # The six-field reference form. `summary` and `retention_analysis` are
        # derived rather than invented: the summary is the description's own
        # first sentence, and retention is read straight off the chip roles,
        # which is the only place in this app that knows what each picture was
        # attached *for*.
        subjects, retain = [], []
        for i, role in enumerate(roles):
            spec = SHOT_REF_ROLES.get(role)
            if not spec:
                continue
            n = len(subjects) + 1
            subjects.append(f"<Subject {n}> is the {spec['noun']} in "
                            f"<Picture {i + 1}>.")
            retain.append(f"<Subject {n}> must retain its {spec['retain']}.")
        if not subjects:
            # Roles are optional and references are not. With none set, the
            # honest thing to say is the one thing attaching a picture always
            # means, rather than leaving a labelled field empty.
            n = len(roles)
            picture = ", ".join(f"<Picture {i + 1}>" for i in range(n))
            subjects = [f"The reference pictures are {picture}."] if n else []
            retain = [f"Retain the appearance of {picture}."] if n else []
        body = _shot_body(typed, buckets)
        # The typed line is the summary — it is the one sentence in the document
        # that says what happens. Taking the description's first sentence
        # instead, which is what this did first, summarised a shot as "A
        # close-up, shot from a low angle": true, and about the lens rather than
        # the scene. The first sentence is the fallback for a prompt built
        # entirely out of pills.
        lines += [
            f"subject_definitions: {' '.join(subjects)}",
            f"summary: {_first_sentence(_close(typed)) or _first_sentence(body)}",
            f"retention_analysis: {' '.join(retain)}",
            f"detailed_description: {body}",
        ]
    else:
        align = H3_ALIGN[task].format(s=f"{float(seconds):.2f}")
        if align:
            lines.append(align)
        lines.append(f"integrated_multimodal_description: "
                     f"{_shot_body(typed, buckets)}")

    lines.append(f"overall_soundscape: "
                 f"{_shot_audio(buckets, 'sound') or 'N/A'}")
    # The default, and the line worth the whole feature: with no score pill the
    # document says there is no score, which is the one thing free prose could
    # never say and the reason every clip came back scored.
    lines.append(f"non_diegetic_music: "
                 f"{_shot_audio(buckets, 'score') or 'N/A'}")
    return "\n".join(lines)


def _compile_image_prompt(typed: str, pills: list[dict[str, Any]]) -> str:
    """
    The same pills as prose, because Krea 2 has no document to fill in.

    Camera, action and the two audio groups never reach here at all — they are
    filtered by `image` in the vocabulary rather than dropped, so a pill the
    image side does not read is dim on the palette rather than silently ignored.
    """
    if not pills:
        return typed
    return _shot_body(typed, _shot_phrases(pills, side="image"))


def _shot_meta(params: dict[str, Any]) -> dict[str, Any]:
    """
    What to put in the sidecar beside `prompt`, which is only ever what ran.

    Only when the compiler did something, which is exactly when the typed text
    and the compiled prompt differ. A sidecar that gains `prompt_typed` equal to
    `prompt` and `shot: []` on every plain run is two fields of noise on the
    file that has to still make sense in a year.
    """
    typed = str(params.get("prompt_typed") or "")
    if not typed or typed == params.get("prompt"):
        return {}
    out: dict[str, Any] = {"prompt_typed": typed, "shot": params.get("shot") or []}
    roles = params.get("ref_roles") or []
    if any(roles):
        # The whole positional list, blanks included: a role's index is the
        # <Picture n> it belongs to, so compacting it would renumber the
        # subjects on the way back in.
        out["ref_roles"] = list(roles)
    return out


def _compile_wan_prompt(typed: str, pills: list[dict[str, Any]]) -> str:
    """
    The same pills as prose again, for the family that reads no document.

    Wan is the middle case and it is worth naming rather than folding into one
    of the other two: it takes the camera and action pills the image side has no
    use for, and it is silent, so the sound and score pills are dropped here the
    same way a negative prompt is dropped for H3 — a sidecar that records an
    input the model never read is a sidecar that lies about how the clip was
    made. Dropped by `needs`, not by field: dialogue is the case that breaks the
    simpler rule, landing in the visual description and still being audio.
    """
    if not pills:
        return typed
    return _shot_body(typed, _shot_phrases(pills, side="video", audio=False))


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
    image=comfy_image, gpu=VIDEO_GPU, cpu=4.0, timeout=60 * 60,
    volumes={"/workspace": volume},
    max_containers=1,
    # Longer than the image side's 10 min: 42.5 GB is a slow thing to reload,
    # and video is worked in takes — you watch one clip, then adjust and go again.
    scaledown_window=15 * 60,
)
@modal.concurrent(max_inputs=1)
class VideoGenerator:
    """Holds a warm ComfyUI process, loaded with a video checkpoint."""

    @modal.enter()
    def setup(self):
        self._comfy = _Comfy("video")
        self._comfy.start()

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
        _reload_volume()

        model = str(params.get("model") or "h3")

        try:
            # Inside the try, not above it: a missing weight raised out here
            # would leave the record saying "running" forever, and the UI
            # polling a job that is never going to answer.
            def stage(blob: str, slot: str, ext: str = "png") -> str:
                return self._comfy.stage(job_id, blob, slot, ext)

            plan = (self._plan_h3 if model == "h3" else self._plan_wan)(params, stage)
            graph, info = plan["graph"], plan["info"]

            _publish(job_id, phase="generate", step=0,
                     total_steps=info["steps"], percent=0, **info)

            out_names = self._comfy.run(job_id, graph, what="video")
        except StopRequested:
            res = {"status": "stopped", "job_id": job_id, "files": [],
                   "duration_s": round(time.time() - started, 1)}
            _publish(job_id, **res)
            return res
        except Exception as exc:
            _publish(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
            raise

        out_dir = OUTPUTS / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        name = f"{time.strftime('%H%M%S')}.mp4"
        # One clip per graph, so the first is the only one. Taking [0] rather
        # than asserting length: a save node that ever emitted a poster frame
        # beside the mp4 should cost the poster, not the take.
        shutil.copyfile(COMFY / "output" / out_names[0], out_dir / name)
        _write_output_meta(
            out_dir, kind="video", job_id=job_id, model=model,
            prompt=params["prompt"], created=time.time(),
            **_shot_meta(params), **info, **plan["meta"],
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

        # Capped on arrival, not asked for smaller at the node — see
        # H3_REF_MAX_SIDE. The gallery's "Use as reference" hand-off sends a
        # finished render at whatever canvas it was made on, so this has to sit
        # behind every route in rather than in the file picker alone.
        references = []
        for i, blob in enumerate(refs_b64):
            references.append(stage(blob, f"refimg{i}"))
            _fit_reference(COMFY / "input" / references[-1])
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
        # case: they are encoded at their own size, up to H3_REF_MAX_SIDE, and
        # bind nothing, so the canvas stays whatever was asked for.
        if "first_frame" in keyframes:
            width, height = _fit_canvas(
                COMFY / "input" / keyframes["first_frame"],
                short=H3_TIERS[params["tier"]], align=32,
            )

        ref_size = params.get("ref_size") or "match"
        graph = _h3_graph(
            prompt=params["prompt"], width=width, height=height, frames=frames,
            seed=seed, steps=steps,
            sampler=params["sampler"], scheduler=params["scheduler"],
            references=references, ref_videos=ref_videos, ref_size=ref_size,
            **keyframes,
        )
        meta = {"mode": "ref2va" if (references or ref_videos) else "fl2va",
                "sampler": params["sampler"], "scheduler": params["scheduler"],
                "references": len(references), "ref_videos": len(ref_videos)}
        # Only where it meant something. It is the one input on this path that
        # changes what a take costs without changing anything the sidecar
        # already records — two takes at the same canvas, frames and steps can
        # be minutes apart on this alone, and nothing said which was which.
        if references:
            meta["ref_size"] = ref_size
        return {
            "graph": graph,
            "info": {"width": width, "height": height, "frames": frames,
                     "seconds": round(frames / H3_FPS, 2), "fps": H3_FPS,
                     "seed": seed, "steps": steps},
            "meta": meta,
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

        Cheap CPU check for when Settings and a GPU job disagree — the
        usual cause is the app resolving a different volume than the one
        holding the weights, so the resolved name is part of the answer.
        """
        _reload_volume()
        tree: dict[str, Any] = {}
        for d in (MODELS, LORAS, DATASETS, DRAFTS, OUTPUTS):
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
        _reload_volume()
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
                        # `root` is served rather than left for the page to
                        # rebuild from a file path, for the same reason the LoRA
                        # index derives `rel` by splitting on `/loras/` instead
                        # of joining two labels: the layout allows any nesting
                        # under a folder, so `dirname(files[0])` is the folder
                        # for a flat training output and one level too deep for
                        # anything else. It is what Delete addresses, and a
                        # delete that addresses the wrong directory is the one
                        # kind of bug this file cannot take back.
                        "root": str(d),
                        "bytes": _tree_bytes(d),
                        "catalogue": CATALOGUE_LORA_ROOTS.get(str(d), ""),
                        "files": [{"name": f.name, "path": str(f)} for f in files],
                    })
                elif d.suffix == ".safetensors":
                    # No sidecar to read a trigger word out of, and no epochs to
                    # choose between — one file, one entry, named for itself.
                    loras.append({
                        "name": d.stem, "trigger_word": "",
                        "path": str(d),
                        "root": str(d),
                        "bytes": _tree_bytes(d),
                        "catalogue": CATALOGUE_LORA_ROOTS.get(str(d), ""),
                        "files": [{"name": d.name, "path": str(d)}],
                    })
        loras.sort(key=lambda l: l["name"].lower())
        return {
            "models": _model_status(),
            "loras": loras,
            "hf_token_set": bool(_hf_token()),
            "samplers": SAMPLERS,
            "schedulers": SCHEDULERS,
            # Which of those two menus opens selected. The video side already
            # carries its defaults per model in VIDEO_MODELS; this is the image
            # side's one row of the same thing.
            "image_defaults": IMAGE_DEFAULTS,
            "max_loras": MAX_LORAS,
            "max_regions": MAX_REGIONS,
            "krea2_defaults": KREA2_DEFAULTS,
            # Whether the scene/outfit controls are live. Same rule VIDEO_MODELS
            # follows: a control that is present but ignored is worse than one
            # that is absent, and without this weight those two drops render a
            # picture that quietly has nothing to do with the photo.
            "edit_lora": bool(
                _sizes_on_disk([MODEL_CATALOGUE["krea2_edit"]["dest"]])[
                    MODEL_CATALOGUE["krea2_edit"]["dest"]]
            ),
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
            # The shot palette builds itself from these, for the same reason
            # the composer builds itself from `video_models`: a copy of the
            # vocabulary in UI_HTML would be a second source of truth, and the
            # first pill added on one side and not the other would compile to
            # "No such shot pill" against the page that offered it.
            "shot_vocab": SHOT_VOCAB,
            "shot_langs": H3_LANGUAGES,
            "shot_roles": [dict(spec, key=k) for k, spec in SHOT_REF_ROLES.items()],
            # Label and note only. The instruction itself stays on the server for
            # the reason the shot vocabulary's phrasing does: what the page sends
            # is a key, so the run is reproducible from the job record rather
            # than from whatever text happened to be in a field.
            "caption_presets": [
                {"key": k, "label": p["label"], "note": p["note"]}
                for k, p in CAPTION_PRESETS.items()
            ],
            "caption_models": [
                {"key": k, "label": m["label"], "note": m["note"]}
                for k, m in CAPTION_MODELS.items()
            ],
            "caption_defaults": {"preset": DEFAULT_CAPTION_PRESET,
                                 "model": DEFAULT_CAPTION_MODEL},
        }

    @api.post("/api/loras/delete")
    def delete_lora(payload: dict) -> dict[str, Any]:
        """
        Delete one LoRA — the folder with its epochs in it, or the loose file.

        The unit is the row `state()` lists, which is the unit the storage layout
        already says a LoRA is: a folder is one, and so is a bare file. An epoch
        inside a folder is deliberately not addressable here. It is a real thing
        to want — twenty checkpoints of one run is most of what fills this volume
        — but it is a second verb with a second confirmation, and offering it
        through the same route as "delete this LoRA" would mean one request whose
        blast radius is a file or a training run depending on how deep the path
        goes. That is the argument for the guard below.

        `parent != LORAS` is stricter than `_lora_path`'s confinement and is
        strict on purpose. It rejects every `../` escape, and it also rejects
        `loras/{name}/{epoch}.safetensors` — a real file, under loras/, that this
        route must not take on its own.
        """
        _reload_volume()
        raw = str(payload.get("path") or "")
        root = Path(raw).resolve()
        if not raw or root.parent != LORAS.resolve():
            return {"error": f"Not a LoRA: {raw!r}"}

        if root.is_dir():
            shutil.rmtree(root, ignore_errors=True)
        elif root.suffix == ".safetensors" and root.is_file():
            root.unlink()
        else:
            # The stale-tab case, and the one worth naming: a second window
            # deleted it, or a training run was renamed out from under this list.
            return {"error": f"No LoRA named {root.name!r} on the volume — "
                             "reopen Settings to refresh the list."}

        _drop_legacy_trash(LORAS)
        volume.commit()
        return {"ok": True}

    @api.post("/api/token")
    def set_token(payload: dict) -> dict[str, Any]:
        token = str(payload.get("hf_token") or "").strip()
        config["hf_token"] = token
        return {"ok": True, "hf_token_set": bool(token)}

    @api.post("/api/download")
    def download(payload: dict) -> dict[str, Any]:
        """
        Start one weight downloading, or report the one already going.

        Idempotent, and never an error for being busy. Pressing Download twice
        is not a mistake to be corrected — it is what anyone does when the first
        press appears to do nothing — so the second press returns the job the
        first one started rather than a red message the page has to find room
        for. `started` is the only difference between the two, and it exists so
        the caller can tell "this is yours, it is running" from "something else
        holds the line".

        Downloads stay one-at-a-time: they share an uplink, and two at once is
        not two downloads, it is the same bandwidth cut in half plus a second
        container to pay for.
        """
        key = str(payload.get("key") or "")
        if key not in MODEL_CATALOGUE:
            return {"error": f"Unknown model: {key}"}

        job_id = f"dl_{key}"
        busy = _active_download()
        if busy:
            rec = jobs.get(busy) or {}
            return {
                "ok": True, "started": False, "job_id": busy,
                "mine": busy == job_id,
                "busy_with": rec.get("phase") or busy,
            }

        # Seeded here, synchronously, rather than on the container's first line:
        # `.spawn()` returns before the container starts, and a second press
        # landing in that gap would read `_active_download()` as empty and start
        # a competitor — the gap being precisely when a second press happens.
        jobs[job_id] = {
            "status": "running",
            "phase": f"Downloading {MODEL_CATALOGUE[key]['label']}",
            "percent": 0,
            "stop": False,
            # The clock starts at the spawn, not at the container's first
            # publish, so the gap a cold start opens is covered by the same
            # liveness rule as everything after it.
            "beat": time.time(),
        }
        jobs[DL_ACTIVE] = {"job_id": job_id}
        download_job.spawn(key)
        return {"ok": True, "started": True, "job_id": job_id, "mine": True}

    @api.post("/api/gdrive")
    def gdrive(payload: dict) -> dict[str, Any]:
        """
        Queue a Google Drive pull into loras/.

        Rejected here rather than in the job for the same reason a bad LoRA path
        is: an empty box and a malformed folder name are form errors, and
        discovering either inside the job costs a container start before saying
        so. What this route cannot check is whether the link is shared — only
        Drive knows that, and it answers with an HTML sign-in page rather than
        an error, which is why the job names that case explicitly.
        """
        url = str(payload.get("url") or "").strip()
        folder = str(payload.get("folder") or "").strip()
        if not url:
            return {"error": "Paste a Google Drive link or file id."}
        if folder and not NAME_RE.match(folder):
            return {"error": "Folder name must be 1-64 chars of [A-Za-z0-9_-]."}
        gdrive_job.spawn(url, folder)
        return {"ok": True, "job_id": GDRIVE_JOB}

    @api.post("/api/download-missing")
    def download_missing(payload: dict) -> dict[str, Any]:
        """
        Save the token (if one was supplied) and queue missing weights.

        `family` scopes it to one group from the catalogue; without it the queue
        is everything missing. One route rather than two because the only
        difference is which keys go in the list — the queue, the sequencing, the
        stop and the failure accounting are identical, and a second endpoint
        would be a second copy of all four.

        Taking the token in the same call is deliberate: pasting a key and then
        having to press Save before Download is a step that exists for no reason.
        """
        token = str(payload.get("hf_token") or "").strip()
        if token:
            config["hf_token"] = token

        family = str(payload.get("family") or "").strip()
        job_id = _family_job_id(family) if family else "dl_all"

        # Same rule as `/api/download`: already-running is a state, not an error.
        busy = _active_download()
        if busy:
            rec = jobs.get(busy) or {}
            return {"ok": True, "started": False, "job_id": busy,
                    "mine": busy == job_id,
                    "busy_with": rec.get("phase") or busy}

        _reload_volume()
        models = _model_status()
        if family:
            models = [m for m in models if m["family"] == family]
            if not models:
                return {"error": f"No such family: {family}"}
        missing = [m["key"] for m in models if not m["present"]]
        if not missing:
            return {"ok": True, "job_id": None, "missing": [], "note": "Everything is already here."}

        gated = [k for k in missing if MODEL_CATALOGUE[k]["gated"]]
        if gated and not _hf_token():
            return {
                "error": "A HuggingFace token is required for "
                + ", ".join(MODEL_CATALOGUE[k]["label"] for k in gated)
                + ". Paste one in the field above."
            }

        jobs[job_id] = {"status": "running", "phase": "Starting…", "percent": 0,
                        "stop": False, "beat": time.time()}
        jobs[DL_ACTIVE] = {"job_id": job_id}
        download_missing_job.spawn(missing, job_id)
        return {"ok": True, "started": True, "job_id": job_id, "mine": True,
                "missing": missing}

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

        # Uploads always target a named set — an existing one by name, which is
        # how a second batch lands in the set you already have, otherwise a new
        # draft. Appending must never be able to delete one that already exists,
        # so track whether this call created it. `dataset` is deliberately not
        # reused as a loop variable below — an earlier version named both this
        # and the per-file basename `name`, so the response reported the last
        # uploaded filename as the dataset.
        dataset = str(form.get("dataset") or "").strip()
        if not dataset:
            return JSONResponse({"error": "A dataset name is required."}, 400)
        try:
            raw = _dataset_dir(dataset)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, 400)
        appending = raw.is_dir()
        raw.mkdir(parents=True, exist_ok=True)
        if not appending:
            # Stamp the window before the first byte is written: an upload that
            # takes longer than the grace period would otherwise be writing into
            # a folder the sweep considers ownerless.
            _write_dataset_meta(raw, session=str(form.get("session") or ""))

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
                _upright_inplace(target)
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

    @api.post("/api/session")
    def session_ping(payload: dict) -> dict[str, Any]:
        """
        The page saying it is still open, and the only thing keeping a draft
        alive. Sweeping here as well as on the listing means a window left open
        on Generate still clears out the drafts of the one you closed.
        """
        _reload_volume()
        DRAFTS.mkdir(parents=True, exist_ok=True)
        _touch_session(str(payload.get("session") or ""))
        swept = _sweep_drafts()
        volume.commit()
        return {"ok": True, "swept": swept}

    @api.get("/api/datasets")
    def list_datasets() -> dict[str, Any]:
        _reload_volume()
        DATASETS.mkdir(parents=True, exist_ok=True)
        DRAFTS.mkdir(parents=True, exist_ok=True)
        if _sweep_drafts():
            volume.commit()
        out = [
            _dataset_stats(d)
            for root in (DATASETS, DRAFTS)
            for d in sorted(root.iterdir())
            if d.is_dir() and not d.name.startswith(".")
        ]
        out.sort(key=lambda r: -r["modified"])
        return {"datasets": out}

    @api.post("/api/datasets")
    def create_dataset(payload: dict) -> dict[str, Any]:
        """
        New sets start as drafts. Pass `saved` to make one in the library
        directly; the page does not, because naming a set is a decision worth
        having images in front of you for.
        """
        name = str(payload.get("name") or "").strip()
        try:
            _check_name(name)
        except ValueError as exc:
            return {"error": str(exc)}
        _reload_volume()
        if _name_taken(name):
            return {"error": f"A set named {name!r} already exists."}
        d = (DATASETS if payload.get("saved") else DRAFTS) / name
        d.mkdir(parents=True)
        _write_dataset_meta(
            d,
            trigger_word=str(payload.get("trigger_word") or ""),
            session=str(payload.get("session") or ""),
        )
        volume.commit()
        return {"ok": True, **_dataset_stats(d)}

    @api.post("/api/datasets/{name}/save")
    def save_dataset(name: str, payload: dict) -> dict[str, Any]:
        """
        Keep a draft: move it into datasets/, under the name you give it here.

        A move and not a copy, because the draft was already the real thing —
        the only difference it ever had was which parent it sat under, so
        nothing has to be rebuilt and the images are never on the volume twice.
        """
        _reload_volume()
        d, err = _dataset_or_error(name)
        if err:
            return err
        if d.parent == DATASETS:
            return {"error": f"{name!r} is already saved."}
        new = str(payload.get("name") or name).strip() or name
        try:
            _check_name(new)
        except ValueError as exc:
            return {"error": str(exc)}
        if new != name and _name_taken(new):
            return {"error": f"A set named {new!r} already exists."}
        DATASETS.mkdir(parents=True, exist_ok=True)
        target = DATASETS / new
        shutil.move(str(d), str(target))
        # Drop the session: a saved set has no window it belongs to, and leaving
        # a stale id on it would be a fact that stops being true.
        _write_dataset_meta(target, session="")
        volume.commit()
        return {"ok": True, **_dataset_stats(target)}

    @api.get("/api/datasets/{name}")
    def dataset_detail(name: str) -> dict[str, Any]:
        """
        Image metadata only — thumbnails come from /api/thumb one at a time.

        The previous version inlined every thumbnail as base64 in this response,
        which put a 200-image dataset at ~6.6 MB before a single tile rendered
        and rebuilt every thumbnail on every load.
        """
        _reload_volume()
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
                    # The size after orientation, which is the size the browser
                    # draws and the size the bucketer will see. Reporting the
                    # stored one labelled a portrait photo "4032×3024".
                    w, h = _upright(im).size
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
        _reload_volume()
        d, err = _dataset_or_error(name)
        if err:
            return err
        _write_dataset_meta(d, trigger_word=str(payload.get("trigger_word") or ""))
        volume.commit()
        return {"ok": True, **_dataset_stats(d)}

    @api.post("/api/datasets/{name}/delete")
    def delete_dataset(name: str) -> dict[str, Any]:
        """Delete a set and everything in it. Unlinked, not recoverable."""
        _reload_volume()
        d, err = _dataset_or_error(name)
        if err:
            return err
        shutil.rmtree(d, ignore_errors=True)
        _drop_legacy_trash(d.parent)
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
                    # Upright before thumbnailing: browsers rotate the original
                    # from EXIF and PIL does not, so without this the tile and
                    # the full-screen view of the same file disagreed by 90°.
                    im = _upright(im).convert("RGB")
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
        _reload_volume()
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
        """Delete an image, its caption and its thumbnail. Not recoverable."""
        _reload_volume()
        d, err = _dataset_or_error(name)
        if err:
            return err
        img = d / Path(str(payload.get("image") or "")).name
        if img.suffix.lower() not in IMAGE_EXTS or not img.is_file():
            return {"error": "Image not found."}

        for part in (img, img.with_suffix(".txt")):
            part.unlink(missing_ok=True)
        (d / THUMB_DIR / (img.stem + ".jpg")).unlink(missing_ok=True)
        _drop_legacy_trash(d)

        volume.commit()
        return {"ok": True, **_dataset_stats(d)}

    @api.get("/api/datasets/{name}/insight")
    def dataset_insight(name: str, trigger: str = "") -> dict[str, Any]:
        _reload_volume()
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

        _reload_volume()
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

        # Named rather than defaulted. The page builds both menus out of the
        # tables `/api/state` serves, so a key that is not in them is the two
        # sides having drifted — and the cost of guessing is a cold GPU
        # container that captions eighty images with the wrong instruction, or
        # downloads 17 GB of the wrong checkpoint. Same argument as
        # `_validate_loras()`: this is a form error in milliseconds.
        preset = str(payload.get("preset") or DEFAULT_CAPTION_PRESET)
        if preset not in CAPTION_PRESETS:
            return {"error": f"No caption preset {preset!r}. "
                             f"One of: {', '.join(CAPTION_PRESETS)}"}
        model = str(payload.get("model") or DEFAULT_CAPTION_MODEL)
        if model not in CAPTION_MODELS:
            return {"error": f"No captioner {model!r}. "
                             f"One of: {', '.join(CAPTION_MODELS)}"}

        job_id = f"cap{time.strftime('%Y%m%d%H%M%S')}{os.urandom(2).hex()}"
        caption_job.spawn(
            job_id=job_id, dataset=name, trigger_word=trigger,
            preset=preset, model=model,
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

    @api.post("/api/compile")
    def compile_prompt(payload: dict) -> dict[str, Any]:
        """
        What the encoder would be given, without renting anything to find out.

        A take is two to three minutes, so before this every question about the
        format — where does the camera direction go, does the score line appear,
        did the dialogue survive its commas — was answered at that price. It is
        the same compiler the run uses, called from the same web container: a
        preview with its own implementation is a preview that can disagree with
        what runs, which is worse than no preview at all.

        No volume reload and no base64. This is re-fetched on every pill and
        every keystroke, and what the compiler needs from a reference is only
        that there is one — the pictures themselves would make a route polled
        four times a second carry megabytes.
        """
        typed = str(payload.get("prompt") or "").strip()
        try:
            shot = _validate_shot(payload.get("shot"))
            n_refs = max(0, min(MAX_H3_REFS, int(payload.get("references") or 0)))
            n_vids = max(0, min(MAX_H3_REF_VIDEOS, int(payload.get("ref_videos") or 0)))
            roles = _validate_ref_roles(payload.get("ref_roles"), n_refs)
        except (TypeError, ValueError) as exc:
            return {"error": str(exc)}

        if str(payload.get("kind") or "video") == "image":
            return {"prompt": _compile_image_prompt(typed, shot)}
        if str(payload.get("model") or "h3") != "h3":
            return {"prompt": _compile_wan_prompt(typed, shot)}

        d = VIDEO_MODELS["h3"]["defaults"]
        try:
            seconds = float(payload.get("seconds") or d["seconds"])
        except (TypeError, ValueError):
            seconds = float(d["seconds"])
        return {"prompt": _compile_h3_prompt(
            typed=typed, pills=shot, seconds=seconds, roles=roles,
            task=_h3_task(payload.get("first_frame"), payload.get("last_frame"),
                          n_refs, n_vids),
        )}

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
        # spawning would cost a cold H100 before discovering it. Regions are
        # validated on the same trip and for the same reason — a region naming
        # a LoRA deleted since the page loaded is the failure a stale tab
        # actually hits.
        _reload_volume()
        try:
            stack = _validate_loras(stack)
            regions = _validate_regions(regions)
            shot = _validate_shot(payload.get("shot"))
        except ValueError as exc:
            return {"error": str(exc)}

        # The plates are inputs to the regional node, so they mean nothing
        # without boxes. Caught here rather than in the job because the answer
        # is "draw a box", which is a thing to say while the reference image is
        # still on screen.
        for slot in ("scene", "outfit"):
            if payload.get(slot) and not regions:
                return {"error": f"A {slot} reference needs at least one region — "
                                 "the scene is composed around the boxes."}

        job_id = f"gen{time.strftime('%Y%m%d%H%M%S')}{os.urandom(2).hex()}"
        runner = _on_gpu(ImageGenerator, payload.get("gpu"), IMAGE_GPUS, GPU)
        runner().generate.spawn(job_id=job_id, params={
            # Prose, not a document: Krea 2 has no fields to fill in, so the
            # same pills append their clauses to the sentence in the same order.
            # The rail is shared with the video side and the vocabulary decides
            # what crosses — camera, action, foley and score never reach here.
            "prompt": _compile_image_prompt(prompt, shot),
            "prompt_typed": prompt,
            "shot": shot,
            "negative_prompt": str(payload.get("negative_prompt") or ""),
            "model": str(payload.get("model") or "turbo"),
            "loras": stack,
            "regions": regions,
            "region_weight": num("region_weight", 1.0, float),
            # Base64, the same shape /api/video already takes its keyframes in.
            "scene": payload.get("scene"),
            "outfit": payload.get("outfit"),
            "width": num("width", 1024, int),
            "height": num("height", 1024, int),
            "num_images": max(1, min(4, num("num_images", 1, int))),
            "steps": num("steps", None, int),
            "cfg_scale": num("cfg_scale", None, float),
            "seed": num("seed", None, int),
            "sampler": str(payload.get("sampler") or IMAGE_DEFAULTS["sampler"]),
            "scheduler": str(payload.get("scheduler") or IMAGE_DEFAULTS["scheduler"]),
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

        # The pill rail, checked before anything is rented. A pill the backend
        # does not know is a stale tab, and the answer to it is a form error
        # naming the key — not a clause quietly missing from the document, which
        # is indistinguishable from the model having ignored the word.
        try:
            shot = _validate_shot(payload.get("shot"))
            roles = _validate_ref_roles(payload.get("ref_roles"), len(refs))
        except ValueError as exc:
            return {"error": str(exc)}

        _reload_volume()
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
        seconds = num("seconds", float(d["seconds"]), float)

        # Compiled here rather than in the client, so there is one
        # implementation of the format, an unknown pill is a form error, and the
        # sidecar records exactly what the encoder was given.
        #
        # The task read here is finer than the one above. That one is right for
        # which checkpoint loads — first-only, last-only and both are the same
        # weights — and too coarse for the alignment instruction, where they are
        # three different sentences about where a picture sits in time.
        if model == "h3":
            compiled = _compile_h3_prompt(
                typed=prompt, pills=shot, seconds=seconds, roles=roles,
                task=_h3_task(first, last, refs, vids),
            )
        else:
            compiled = _compile_wan_prompt(prompt, shot)

        job_id = f"vid{time.strftime('%Y%m%d%H%M%S')}{os.urandom(2).hex()}"
        runner = _on_gpu(VideoGenerator, payload.get("gpu"), VIDEO_GPUS, VIDEO_GPU)
        runner().generate.spawn(job_id=job_id, params={
            "model": model,
            # Both travel. `prompt` is what runs and is the only one the graph
            # sees; the other two are what you chose, and they exist so a
            # gallery card can show a sentence instead of a six-field document
            # and so Reuse puts the pills back rather than the output of them.
            "prompt": compiled,
            "prompt_typed": prompt,
            "shot": shot,
            "ref_roles": roles,
            # Dropped rather than passed through for a model that cannot read
            # it: a negative prompt that reaches a guidance-distilled checkpoint
            # is not applied, and a sidecar that records one is a sidecar that
            # lies about how the clip was made.
            "negative_prompt": (str(payload.get("negative_prompt") or "")
                                if supports["negative"] else ""),
            "aspect": aspect,
            "tier": tier,
            "seconds": seconds,
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
        _reload_volume()
        return {"items": _gallery()}

    @api.get("/api/file/{job_id}/{name}")
    def output_file(job_id: str, name: str):
        """
        Stream one result off the volume, image or video.

        Deliberately not base64 in a JSON body: inlining is what made a gallery
        impossible, since a page of stills or a clip with its soundtrack is tens
        of megabytes of JSON before anything renders, and a <video> cannot seek
        until all of it has arrived. The route that did it that way is gone —
        this is now the only way a result's bytes reach the page, from the canvas
        the moment a run finishes through to the gallery.

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
            _reload_volume()
            if not path.is_file():
                return JSONResponse({"error": "Not found."}, status_code=404)
        return FileResponse(
            str(path),
            media_type=MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"),
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @api.post("/api/outputs/{job_id}/delete")
    def delete_output(job_id: str) -> dict[str, Any]:
        """Delete a result and its files. Unlinked, not recoverable."""
        if not NAME_RE.match(job_id):
            return {"error": "Invalid job_id."}
        _reload_volume()
        d = OUTPUTS / job_id
        if not d.is_dir():
            return {"error": "Not found."}
        shutil.rmtree(d, ignore_errors=True)
        _drop_legacy_trash(OUTPUTS)
        volume.commit()
        return {"ok": True}

    @api.post("/api/outputs/purge")
    def purge_outputs(payload: dict) -> dict[str, Any]:
        """
        Delete many results in one request.

        Two guards, because deletion here does not go anywhere first.

        `confirm` has to be in the body, so a bare POST to a guessed URL cannot
        fire it. And the caller names the folders rather than describing them:
        re-deriving the set here from a filter would delete whatever matched at
        request time, which is not the set the user was shown a count of and
        agreed to. A run that finished during the confirm dialog would go
        without ever having been on screen. The list is the agreement.
        """
        if payload.get("confirm") != "delete":
            return {"error": "Unconfirmed."}

        job_ids = payload.get("job_ids")
        if not isinstance(job_ids, list) or not job_ids:
            return {"error": "Nothing to delete."}
        if any(not isinstance(j, str) or not NAME_RE.match(j) for j in job_ids):
            return {"error": "Invalid job_id."}

        _reload_volume()
        removed = 0
        for job_id in dict.fromkeys(job_ids):
            d = OUTPUTS / job_id
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
                removed += 1

        _drop_legacy_trash(OUTPUTS)
        volume.commit()
        return {"ok": True, "removed": removed}

    # There was a GET /api/outputs/{job_id} here that returned every PNG of a run
    # base64'd into one JSON body. It is gone rather than left unused: /api/file
    # already serves the same bytes, streamed and cacheable, and it is what the
    # gallery, the drawer and now the canvas all use. Two routes for one job,
    # where the second one is strictly slower, is the shape a future change
    # picks the wrong half of.

    # Job ids whose completion this container has already reacted to. Bounded by
    # replacing the set rather than growing it: nothing here needs history, only
    # "have I already done the one-time thing for this job".
    _warmed: set = set()

    @api.get("/api/status/{job_id}")
    def status(job_id: str) -> dict[str, Any]:
        try:
            rec = jobs.get(job_id) or {"status": "unknown"}
        except Exception as exc:
            return {"status": "unknown", "error": str(exc)}

        # The poll that first sees "completed" is the last thing to happen before
        # the client asks for the pixels, which makes it the free place to pull
        # the volume forward. Without it the first /api/file of every run misses,
        # reloads, and pays that latency in front of the image the user is
        # waiting on — with it, the reload overlaps the client's own round trip
        # and every still is served off a warm view.
        if rec.get("status") == "completed" and job_id not in _warmed:
            if len(_warmed) > 256:
                _warmed.clear()
            _warmed.add(job_id)
            try:
                _reload_volume()
            except Exception as exc:
                print(f"[status] warm reload failed for {job_id}: {exc}", flush=True)
        return rec

    @api.post("/api/stop/{job_id}")
    def stop(job_id: str) -> dict[str, Any]:
        cur = jobs.get(job_id) or {}
        cur["stop"] = True
        jobs[job_id] = cur
        return {"ok": True}

    return api


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
      --mut:#8a8a8a;--dim:#5a5a5a;--drawer:320px;--head:56px}
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
.top{flex:0 0 var(--head);display:flex;align-items:center;gap:14px;padding:0 14px 0 18px;
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
.canvas{position:relative;flex:1;min-height:0;overflow:auto;padding:22px 28px;
  display:flex;flex-direction:column}
/* Capped, so a console with everything open can never push the canvas out of
   the frame — past the cap it scrolls itself instead. */
.console{flex:none;max-height:54dvh;overflow:auto;padding:13px 28px 15px;
  border-top:1px solid rgba(255,255,255,.07);background:rgba(255,255,255,.012)}
/* Generate lives in the bar with the controls it acts on, not beside the
   prompt — so the prompt gets the console's full width, which is the thing
   that actually benefits from it. Sized to the row: same height and radius as
   every control next to it, still white, because it is still the one control
   on the page that spends money. */
.opts button.b{flex:none;height:36px;padding:0 20px;border-radius:11px;font:600 13px/1 inherit}
.opts .actions{display:flex;align-items:center;gap:7px;flex:none;margin-left:auto}
.drawer{flex:0 0 var(--drawer);min-width:0;border-left:1px solid rgba(255,255,255,.07);overflow:auto}
.drawer-in{width:var(--drawer);padding:12px 14px 40px}
/* Collapsed by flex-basis rather than display:none so the canvas reflows
   smoothly; the inner column keeps its width so nothing re-wraps on the way. */
.studio.nodrawer .drawer{flex-basis:0;border-left:0;overflow:hidden}
#v-train .drawer{flex-basis:var(--drawer)}
/* The sheet's own toolbar sticks for the same reason the console is pinned:
   the filters are how you find the six bad captions in eighty images, and
   scrolling to look for them is exactly when they must not scroll away. */
/* The open contact sheet takes a drop too, so a second batch lands in the set
   you are looking at instead of starting another one. */
#ds-sheet.hot{outline:1px dashed rgba(255,255,255,.45);outline-offset:8px;border-radius:16px}
#ds-sheet .sheet-bar{position:sticky;top:-22px;z-index:6;margin:-22px -28px 12px;
  padding:22px 28px 10px;background:var(--bg)}
#ds-list{gap:8px}
#ds-list .ds-card{flex-direction:row;align-items:center;gap:11px;padding:8px}
#ds-list .ds-cover{width:56px;height:56px;flex:none;aspect-ratio:auto;border-radius:9px;object-fit:cover}
#ds-list .ds-meta{padding:0;min-width:0}
#ds-list .ds-meta b{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.drawer{transition:flex-basis .18s ease}
.drawer-head{display:flex;align-items:center;gap:8px;margin-bottom:10px;min-height:28px}

/* Controls in a row, not a form. A control that shows its own value — "Krea 2
   Turbo", "16:9", "720p", "5s" — still gets no label, because one would only
   repeat what the control already says.
   A number does not show its own value. "32" is a rank, an alpha, a step
   count or a seed with equal plausibility, and the icons that used to stand in
   for those words failed the only test that matters: someone who has trained
   these models for five years had to hover every numeric field to find out
   what it was. An icon is a rebus for a word you already know; it cannot teach
   you which hyperparameter you are looking at. So every numeric field is
   named, and the tooltip is promoted from repeating the name to saying what
   the number does. */
.opts{display:flex;flex-wrap:wrap;align-items:center;gap:7px;margin-top:9px}
.opt{display:inline-flex;align-items:center;gap:5px;height:36px;padding:0 5px 0 9px;
  background:rgba(255,255,255,.05);border:1px solid var(--line);border-radius:11px}
.opt:focus-within{border-color:rgba(255,255,255,.28)}

/* Unbounded ---------------------------------------------------------------
   Ten controls in a row, each in its own box, is ten boxes competing with the
   picture above them. The chrome is not what makes a control a control — the
   value is, and every one of these already shows its own. So the box is spent
   only when it is doing something: pointing at what the pointer is over, or
   showing that a popover is open from here.

   Scoped to the console. `.opt` is also used in Train, which is a form you fill
   in rather than a row you scan, and a form without field edges is a worse
   form. The region bar is included because it is the same console.

   The line: a control whose value *is* its label can lose its box; a free-text
   field cannot, because an empty one has nothing to show and no edge to aim at.
   Hence the `:has(input[type=text])`-shaped exceptions below rather than a
   blanket rule — #r-prompt and the size boxes keep their containers. */
#c-image>.opts .opt,#c-video>.opts .opt,#region-bar .opt{
  background:none;border-color:transparent}
#c-image>.opts .opt:hover,#c-video>.opts .opt:hover,#region-bar .opt:hover,
#c-image>.opts .opt:focus-within,#c-video>.opts .opt:focus-within,#region-bar .opt:focus-within{
  background:rgba(255,255,255,.06);border-color:var(--line)}
/* The exception, and the reason it is by content rather than by id: anything
   holding a text box you type into keeps its edges wherever it appears. */
#region-bar .opt.wide,#c-image>.opts .opt.wide,#c-video>.opts .opt.wide{
  background:rgba(255,255,255,.05);border-color:var(--line)}
/* Icon buttons and the value-bearing pills follow the same rule. `.on` is a
   popover open from this control, which is the one state that must read while
   the pointer is somewhere else entirely — inside the popover. */
#c-image>.opts .opt.ib.on,#c-video>.opts .opt.ib.on,#region-bar .opt.ib.on{
  background:rgba(255,255,255,.14);border-color:rgba(255,255,255,.24)}
.opt>svg{width:14px;height:14px;flex:none;color:var(--dim)}
/* `.lead`, not `.lb` — `.lb` is the lightbox, `position:fixed;inset:0`, and
   every label in the strip quietly became a full-screen black overlay. Same
   class of bug as `.blank` above: a selector that collides is invisible in the
   markup and total on the page. */
.opt>.lead,.drop.mini>.lead{font-size:11.5px;line-height:1;color:var(--dim);flex:none;
  white-space:nowrap;cursor:default;-webkit-user-select:none;user-select:none}
.opt select,.opt input{width:auto;border:0;background:none;padding:0 2px;height:34px;border-radius:8px}
/* The native select chrome was never switched off. On a desktop the macOS arrows
   pass for a chevron; at 42px on a phone they render as a stepper — a control
   that looks like it increments something, next to controls that do. One
   chevron, drawn once, so every select on the page says the same thing. */
.opt select{appearance:none;-webkit-appearance:none;padding-right:17px;
  background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 12 12' fill='none' stroke='%238a8a8a' stroke-width='1.4' stroke-linecap='round'><path d='M3 4.5 6 7.5 9 4.5'/></svg>");
  background-repeat:no-repeat;background-position:right 1px center;background-size:12px}
/* Phone. Ten controls wrapping into four ragged rows is what the strip does
   here, and three of the ten are the ones nobody touches twice: the GPU is set
   once and confirms a cold start when it changes, the seed is a thing you reuse
   off a result rather than type, and a batch count is a decision the Generate
   button could carry. They are still reachable — GPU under the gear, seed and
   steps behind Sampling — they are simply not worth a row on a 390px screen.
   
   This is the promote/demote rule with the screen as the forcing function: the
   controls that survive are the ones a render actually varies by. */
@media (max-width:640px){
    #c-image .opts,#c-video .opts{gap:6px}
}
.opt input{width:76px}
/* Named numerics do not need the width an unlabelled one did: the label
   carries the meaning, so the box only has to hold two or three digits. */
.opt.n input{width:52px}
.opt.mid input{width:136px}
.opt.wide{flex:1;min-width:220px}
.opt.wide input,.opt.wide textarea{flex:1;min-width:0;width:auto}
.vr{width:1px;height:20px;flex:none;background:var(--line);margin:0 4px}
/* Icon-only controls in the strip: same box, square, no value to show. */
.opt.ib{padding:0;width:36px;justify-content:center;cursor:pointer;color:var(--mut);
  transition:background .12s,color .12s}
.opt.ib:hover{background:rgba(255,255,255,.1);color:var(--fg)}
.opt.ib.on{background:rgba(255,255,255,.14);color:var(--fg);border-color:rgba(255,255,255,.24)}
".opt.ib>svg{width:16px;height:16px;color:inherit}
/* How many boxes are armed, on the button, because the boxes themselves are
   off the picture now. A regional render and a plain one are otherwise
   identical on screen until the result comes back. */
.opt.ib.counted::after{content:attr(data-count);margin-left:6px;font:600 10.5px/1 inherit;
  color:inherit;opacity:.85;font-variant-numeric:tabular-nums}
/* An icon button that arms a whole subsystem carries its name as well. :has()
   rather than a modifier class so the shell cannot disagree with whether
   data-lb is actually there — the two would drift, and the failure is a label
   clipped to a 36px box, which is what happened to `.drop.mini`. */
.opt.ib:has(>.lead){width:auto;padding:0 11px 0 9px;gap:7px}
/* order:2 puts the word after the glyph, matching the Train door in the header
   and `.drop.mini`; label() inserts afterbegin, so without it the word lands in
   front. color:inherit so the label brightens with the button on hover and .on
   instead of staying at --dim while the icon lights up. */
.opt.ib>.lead{order:2;color:inherit}
.adv{margin-top:9px;padding-top:10px;border-top:1px solid var(--line)}
/* Not an inline style, which is what this was: an empty <p> is zero-height but
   still spends its margin, and the :empty rule that reclaims it cannot outrank
   a style attribute. A line that has nothing to say most of the time should
   cost the canvas nothing most of the time. */
#lora-note{margin:7px 2px 0}
#lora-note:empty{margin:0}

/* The prompt field. The textarea and the Image/Video chip are one bordered
   box, and the prompt itself is shared by both — switching mid-sentence keeps
   the sentence. Image and video are not two workspaces to navigate between;
   they are two things the same sentence can become, so the choice belongs
   inside the field you are already typing in, at the smallest size that still
   reads. Everything below the field is options for that choice. */
.field{position:relative;border:1px solid var(--line);background:rgba(255,255,255,.05);
  border-radius:13px;padding:2px 2px 0}
.field:focus-within{border-color:rgba(255,255,255,.28)}
/* One row, and it grows into what you type. It was two rows fixed, with the
   native resize grip in the corner — which is a control that asks you to do by
   hand, every session, the one thing the box can measure for itself. Two rows
   is also wrong in both directions at once: a line and a half of empty box for
   the short prompt that is most of them, and a scrollbar for the long one.
   resize:none rather than resize:vertical from the base rule, because with
   autoGrow() driving the height a drag would be overwritten by the next
   keystroke — a grip that silently undoes itself is worse than no grip. */
.field textarea{border:0;background:none;border-radius:0;padding:9px 10px 2px;
  resize:none;overflow-y:auto}
.field .bar2{display:flex;align-items:center;gap:8px;padding:2px 5px 5px}
/* The negative prompt, which used to be a permanent two-row box at the top of
   Advanced — on Krea 2 Turbo, whose CFG is 1.0, that is a control the sampler
   cannot read sitting above every control it can. Not hidden behind the model's
   name either: it is the same sentence field in a different sign, so it is a
   mode on the field rather than a second box under it, and it costs the console
   nothing until you switch into it.

   Text, not a chip: a chip reads as a thing to press among other things to
   press, and this is a corner marker for a mode you are usually not in. It
   sits in the corner the resize grip just vacated.

   It names the field you are looking at — "positive", then "negative" — and
   not the one a click would take you to. A tag that reads as an instruction
   has to be decoded every time it is seen: "negative" over an empty box is
   equally readable as "this box is the negative" and "press to go there", and
   those are opposite facts. A label that states the current state is never
   ambiguous, and the click is discovered once. */
.neg-t{position:absolute;top:6px;right:8px;z-index:2;border:0;background:none;padding:3px 4px;
  color:var(--dim);font:500 10.5px/1 inherit;letter-spacing:.02em;cursor:pointer;
  border-radius:7px;-webkit-user-select:none;user-select:none}
.neg-t:hover{color:var(--fg);background:rgba(255,255,255,.07)}
.field.on-neg .neg-t{color:#fca5a5}
/* Reserved only while the toggle is there, so a model that reads no negative
   prompt gets the full width back rather than a permanent gutter for a control
   it is not showing. */
.field.has-neg textarea{padding-right:62px}
/* Something is written on the other side. Without it the negative is invisible
   from the positive — you would be looking at a prompt that renders differently
   than it reads, with nothing on screen saying why. */
.neg-t::after{content:'';display:inline-block;width:4px;height:4px;margin-left:5px;
  border-radius:50%;background:#f87171;vertical-align:1px;opacity:0;transition:opacity .12s}
.neg-t.filled::after{opacity:1}
.field.on-neg .neg-t::after{opacity:0}
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
/* Red and spelled out, because this one is a whole gallery at once. The per-card
   × is the same kind of action at a scale a mis-click can survive; this is the
   scale where the confirm dialog is the only thing between you and the volume. */
.pill.danger{background:rgba(248,113,113,.12);color:#f87171}
.pill.danger:hover{background:rgba(248,113,113,.2)}
.pill.danger[disabled]{opacity:.35;cursor:default;background:rgba(248,113,113,.12)}
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
/* zoom-in, because it does: the still on the canvas opens the same lightbox the
   gallery card does. A result you have to send to the gallery to look at
   properly is a result the canvas is only pretending to show you. */
.shot img{display:block;max-width:100%;max-height:var(--shot-h,none);width:auto;height:auto;cursor:zoom-in}
/* Quiet at rest rather than absent. These two are the whole "a still flows into
   a clip without a round trip through the filesystem" claim, and at opacity:0
   they were a feature you had to already know about to find — nothing on the
   canvas suggested there was anything under the pointer. A third of an opacity
   is enough to read as "something is here" from across the picture and not
   enough to compete with it. :focus-within because `.gal .quick` has had it all
   along and these did not, so the keyboard could reach a control it could never
   see. */
/* Under the picture, not on it. These were two filled buttons floating over the
   bottom-right corner, which is furniture on the one surface this layout exists
   to keep clear — and they were the loudest thing in a dark frame. As words
   below the corner they are the same register as the "8 steps · CFG 1.0"
   summary: present, quiet, and discovered the first time anyone's pointer
   crosses a render. ACTS_H below is the height this reserves, and layoutShots
   subtracts it, or the last row's words would sit under the caption. */
/* bottom, not top:100% — a percentage top resolves against the padded box,
   so the 26px reserved for these pushed them 26px further down instead of
   making room, and they landed on the caption. */
.shot .acts{position:absolute;left:1px;bottom:3px;display:flex;gap:8px;opacity:0;
  transition:opacity .12s}
.shot:hover .acts,.shot .acts:focus-within{opacity:1}
.shot{padding-bottom:26px}
.shot .acts button{width:20px;height:20px;flex:none;border:0;background:none;padding:0;
  border-radius:5px;color:var(--dim);cursor:pointer}
.shot .acts button:hover{color:var(--fg);background:none}
.shot .acts button svg{width:100%;height:100%;display:block}
/* Lit, unlike its two neighbours, because it is the only one of the three
   that restores something rather than starting something. */
#vid-out{position:relative}
#vid-out video{width:100%;max-width:1180px;margin:0 auto;display:block;border-radius:16px;
  border:1px solid var(--line);background:#000}
/* A button rather than a click on the video: the video already owns its own
   click, which is play/pause, and taking that away to open a viewer would
   trade a control people use constantly for one they use occasionally. */
#vid-out .zoom{position:absolute;top:10px;right:10px;width:30px;height:30px;border:0;border-radius:999px;
  background:rgba(0,0,0,.66);color:#eee;padding:7px;cursor:pointer;backdrop-filter:blur(8px);
  opacity:0;transition:opacity .12s}
#vid-out:hover .zoom{opacity:1}

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
/* Scrolls rather than growing: this is also the LoRA picker, and a volume with
   forty of them would otherwise open a menu taller than the window. */
.menu{position:fixed;z-index:80;min-width:196px;max-width:min(420px,92vw);
  max-height:min(56vh,430px);overflow:auto;padding:5px;border-radius:13px;
  border:1px solid rgba(255,255,255,.14);background:#111;box-shadow:0 18px 48px rgba(0,0,0,.6)}
.menu button{display:block;width:100%;text-align:left;border:0;background:none;color:#e6e6e6;
  font:13px/1 inherit;padding:9px 11px;border-radius:9px;cursor:pointer;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.menu button:hover{background:rgba(255,255,255,.09)}
.menu button.danger{color:#f87171}
.menu button.danger:hover{background:rgba(248,113,113,.14)}
/* A ticked item is one already in the prompt, and choosing it again takes it
   out. The gutter is reserved on every button in such a menu rather than only
   the ticked ones, so the labels do not shift sideways as items toggle. */
.menu.checks button{padding-left:28px}
.menu button.on::before{content:"";position:absolute;left:12px;top:50%;margin-top:-5px;
  width:5px;height:9px;border:solid currentColor;border-width:0 1.5px 1.5px 0;transform:rotate(45deg)}
.menu.checks button{position:relative}
.menu hr{border:0;border-top:1px solid rgba(255,255,255,.1);margin:5px 7px}

/* The shot palette ------------------------------------------------------
   A popover of tiles, sharing openMenu's element lifecycle. Grouped, because
   the groups are the question — you are choosing a shot size, then a move —
   and a flat grid of eighty tiles is a vocabulary list rather than a palette. */
.pal{position:fixed;z-index:80;width:min(576px,94vw);max-height:min(62vh,520px);
  overflow:auto;padding:13px;border-radius:16px;border:1px solid rgba(255,255,255,.14);
  background:#111;box-shadow:0 18px 48px rgba(0,0,0,.6)}
/* Longhands, not the `font:` shorthand. `font:600 10.5px/1 inherit` is
   invalid — `inherit` is a CSS-wide keyword and cannot stand in for the family
   the shorthand requires — so the whole declaration is dropped and the element
   silently keeps the browser default. It is spelled that way elsewhere in this
   stylesheet and gets away with it because 13px is what those elements wanted
   anyway; at 9.5px it is the difference between a label and a clipped label. */
.pal h4{display:flex;align-items:baseline;gap:8px;margin:0 0 8px 3px;
  font-size:10.5px;font-weight:600;line-height:1;
  letter-spacing:.1em;text-transform:uppercase;color:var(--mut)}
/* Why a whole group is out of play. The pills themselves dim, which says that
   they are unavailable; only the heading can say what would make them
   available, and that is a sentence about the model rather than about them. */
.pal h4 i{font-size:10px;font-weight:400;font-style:normal;line-height:1;
  letter-spacing:.02em;text-transform:none;color:var(--dim)}
.pal section+section{margin-top:15px;padding-top:13px;border-top:1px solid rgba(255,255,255,.08)}
.pal .tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(86px,1fr));gap:5px}
.pal .tl{display:flex;flex-direction:column;align-items:center;gap:6px;padding:8px 3px 7px;
  border:1px solid transparent;border-radius:11px;background:none;color:#ddd;cursor:pointer;
  font-size:11.5px;line-height:1.25;text-align:center}
.pal .tl:hover{background:rgba(255,255,255,.07);border-color:rgba(255,255,255,.13)}
.pal .tl.on{background:rgba(255,255,255,.15);border-color:rgba(255,255,255,.3);color:#fff}
.pal .tl.off{opacity:.26;cursor:default}

/* One skeleton, eighty behaviours -----------------------------------------
   Every tile is the same nine shapes — a frame, a horizon, its posts, two
   subjects, an object, a light wedge, bars, grain — and what makes a push-in
   look like a push-in is a class on the <svg> plus a keyframe rule. Eighty
   bespoke drawings is the version that drifts: the frame ends up 1px in thirty
   of them and 1.5 in the rest, and nobody ever sees all eighty side by side to
   notice. It is also the version nobody finishes.

   The posts are what make lateral motion visible at all. A horizontal line
   translated horizontally is a horizontal line, so panning without them is a
   perfectly correct animation of nothing happening; they are spaced exactly
   20 apart so a 20px stream loops seamlessly.

   Only an open palette animates. It is removed from the DOM on close, the same
   reason openMenu is one element that moves, and the copies of these glyphs
   that live on the pill rail are frozen — see `.spill .gl` below. So the page
   at rest is running nothing. */
.gl{width:44px;height:30px;display:block;color:#eee;overflow:hidden}
.gl *{transform-box:view-box;transform-origin:50% 50%}
.gl .fr{fill:none;stroke:currentColor;stroke-width:1;opacity:.42}
.gl .hz{stroke:currentColor;stroke-width:1;opacity:.28}
.gl .tk path{stroke:currentColor;stroke-width:1;opacity:.24}
.gl .s1,.gl .s2,.gl .ob,.gl .lt,.gl .gr circle,.gl .bu{fill:currentColor}
.gl .s2,.gl .ob,.gl .pv,.gl .bu,.gl .tx{display:none}
.gl .pv,.gl .tx{stroke:currentColor;stroke-width:1.4;stroke-linecap:round;fill:none}
.gl .bu{opacity:.75}
.gl .lt,.gl .gr{opacity:0}
.gl .bars rect{fill:#000;opacity:0}

/* Framing. The entire vocabulary of shot size is how much of the frame a
   person takes up, so that is literally the only thing that varies. */
.gl-fr-xw .s1{transform:scale(.3) translateY(14px)}
.gl-fr-w .s1{transform:scale(.58) translateY(7px)}
.gl-fr-mcu .s1{transform:scale(1.5) translateY(2px)}
.gl-fr-cu .s1{transform:scale(2.4) translateY(3px)}
.gl-fr-xcu .s1{transform:scale(4) translateY(1px)}
.gl-fr-ots .s2{display:block;transform:scale(2.9) translate(-5px,3px);opacity:.4}
.gl-fr-ots .s1{transform:scale(1.35) translateX(4px)}
.gl-fr-pov .s1{display:none}
.gl-fr-pov .pv{display:block}

/* Angle, told by where the horizon sits. Looking up puts it low in the frame
   and looking down puts it high, which is the fact the words name. */
.gl-an-low .hz,.gl-an-low .tk{transform:translateY(5px)}
.gl-an-low .s1{transform:scale(1.2) translateY(-2px)}
.gl-an-high .hz,.gl-an-high .tk{transform:translateY(-9px)}
.gl-an-high .s1{transform:scale(.85) translateY(3px)}
.gl-an-bird .hz,.gl-an-bird .tk{opacity:0}
.gl-an-bird .s1{transform:scaleY(.66)}
.gl-an-bird .s2{display:block;opacity:.18;transform:scale(2.8)}
.gl-an-worm .hz,.gl-an-worm .tk{transform:translateY(8px)}
.gl-an-worm .s1{transform:scale(1,1.5) translateY(-3px)}
.gl-an-dutch .wo{transform:rotate(-13deg)}

/* Light, as one wedge moved around the frame. */
.gl-li-window .lt{opacity:.2}
.gl-li-golden .lt{opacity:.26;transform:translate(19px,5px) rotate(30deg) scaleY(.85)}
.gl-li-overcast .lt{opacity:.1;transform:scaleX(5)}
.gl-li-hardsun .lt{opacity:.44;transform:scaleX(.5) translateX(-11px)}
.gl-li-neon .lt{opacity:.32;transform:scaleX(.36) translateX(-26px)}
.gl-li-neon .hz{opacity:.85;stroke-dasharray:4 3}
.gl-li-candle .lt{opacity:.3;transform:rotate(180deg) scale(.5) translateY(12px)}
.gl-li-practical .s2{display:block;transform:scale(.3) translate(40px,-16px)}
.gl-li-practical .lt{opacity:.15;transform:translateX(19px) scale(.75)}
.gl-li-silhouette .lt{opacity:.5;transform:scaleX(6)}
.gl-li-silhouette .s1{fill:#0d0d0d}
.gl-li-top .lt{opacity:.24;transform:translateY(-9px) scale(1.8,.5)}

/* Tone. The one group where the tile is a treatment rather than a geometry. */
.gl-to-doc .s2{display:block;transform:scale(.24) translate(60px,-42px)}
.gl-to-doc .hz{opacity:.46}
.gl-to-noir .bars{transform:rotate(-26deg) scale(1.6,.42)}
.gl-to-noir .bars rect{opacity:.72}
.gl-to-noir .lt{opacity:.3;transform:scaleX(.45) translateX(-13px)}
.gl-to-s16 .gr{opacity:.6}
.gl-to-s16 .hz,.gl-to-s16 .tk path{opacity:.16}
.gl-to-anamorphic .bars rect{opacity:.95}
.gl-to-anamorphic .lt{opacity:.3;transform:scale(3,.12) translateY(-52px)}
.gl-to-highkey .lt{opacity:.4;transform:scaleX(6)}
.gl-to-highkey .s1{opacity:.45}
.gl-to-desat .s1{opacity:.36}
.gl-to-desat .hz,.gl-to-desat .tk path{opacity:.13}
.gl-to-contrast .s1{fill:#0d0d0d}
.gl-to-contrast .lt{opacity:.62;transform:scaleX(1.15) translateX(-6px)}

/* Speech and on-screen text. */
.gl-sa-dialogue .bu{display:block}
.gl-sa-dialogue .s1{transform:scale(1.15) translate(-7px,2px)}
.gl-sa-screen .tx{display:block}
.gl-sa-screen .s1{opacity:.3;transform:scale(1.3) translateY(-3px)}

/* Camera. Three of these pairs are the distinctions a word alone never makes,
   and the tile is the only place the app can make them: a dolly changes the
   relationship between subject and background and a zoom does not, so the
   push-in tile scales the subject faster than the horizon and the zoom tile
   scales both together. Truck against pan is the same argument sideways. */
.gl-ca-push .s1{animation:gScale 2.4s ease-in-out infinite alternate}
.gl-ca-push .tk{animation:gCreep 2.4s ease-in-out infinite alternate}
.gl-ca-pull .s1{animation:gScale 2.4s ease-in-out infinite alternate-reverse}
.gl-ca-pull .tk{animation:gCreep 2.4s ease-in-out infinite alternate-reverse}
.gl-ca-zoom .wo{animation:gScale 2.4s ease-in-out infinite alternate}
.gl-ca-panl,.gl-ca-truckl,.gl-ca-orbit{--d:-1}
.gl-ca-panr,.gl-ca-truckr,.gl-ca-arc{--d:1}
.gl-ca-panl .wo,.gl-ca-panr .wo{animation:gPan 2.6s ease-in-out infinite alternate}
.gl-ca-truckl .tk,.gl-ca-truckr .tk{animation:gPanHalf 2.6s ease-in-out infinite alternate}
.gl-ca-truckl .s1,.gl-ca-truckr .s1{animation:gPan 2.6s ease-in-out infinite alternate}
.gl-ca-tiltu,.gl-ca-pedu,.gl-ca-craneu{--v:1}
.gl-ca-tiltd,.gl-ca-pedd,.gl-ca-craned{--v:-1}
.gl-ca-tiltu .wo,.gl-ca-tiltd .wo{animation:gTilt 2.6s ease-in-out infinite alternate}
.gl-ca-pedu .s1,.gl-ca-pedd .s1{animation:gTilt 2.6s ease-in-out infinite alternate}
.gl-ca-pedu .tk,.gl-ca-pedd .tk{animation:gTiltHalf 2.6s ease-in-out infinite alternate}
.gl-ca-craneu .wo,.gl-ca-craned .wo{animation:gCrane 3s ease-in-out infinite alternate}
.gl-ca-orbit .tk,.gl-ca-arc .tk{animation:gPan 3s ease-in-out infinite alternate}
.gl-ca-orbit .s1,.gl-ca-arc .s1{animation:gSquash 3s ease-in-out infinite}
.gl-ca-arc .wo{animation:gPanHalf 3s ease-in-out infinite alternate}
.gl-ca-trackside .tk,.gl-ca-trackrear .tk{animation:gStream 1.7s linear infinite}
.gl-ca-trackrear .s1{animation:gBob 1.7s ease-in-out infinite}
.gl-ca-handheld .wo{animation:gJit .9s steps(1,end) infinite}
.gl-ca-whip .wo{animation:gWhip 2.4s ease-in-out infinite}
.gl-ca-rack .s2{display:block;transform:scale(2.3) translate(-6px,4px);
  animation:gFocusA 3s ease-in-out infinite alternate}
.gl-ca-rack .s1{animation:gFocusB 3s ease-in-out infinite alternate}
/* Locked off is the one tile that does nothing, and it is not an omission:
   beside twenty moving tiles, stillness is the clearest available statement of
   what "the camera does not move" means. */

/* Action. Two subjects, which is what separates this group from every other
   one on the palette — the whole reason it exists is what happens between
   them. `[class*=]` rather than a second class from the builder: the glyph
   name is the data, and a shadow flag alongside it is a thing to keep in step. */
.gl[class*="gl-ac-"] .s2{display:block;transform:translateX(6px)}
.gl[class*="gl-ac-"] .s1{transform:translateX(-6px)}
.gl-ac-fight .s1{animation:gJabA .62s ease-in-out infinite}
.gl-ac-fight .s2{animation:gJabB .62s ease-in-out infinite}
.gl-ac-kiss .s1{animation:gCloseA 2.6s ease-in-out infinite}
.gl-ac-kiss .s2{animation:gCloseB 2.6s ease-in-out infinite}
.gl-ac-embrace .s1{animation:gCloseA 3.2s ease-in-out infinite;opacity:.8}
.gl-ac-embrace .s2{animation:gCloseB 3.2s ease-in-out infinite;opacity:.8}
.gl-ac-talk .s1{animation:gSayA 2.2s ease-in-out infinite}
.gl-ac-talk .s2{animation:gSayB 2.2s ease-in-out infinite}
.gl-ac-laugh .s1{animation:gLaughA 1.1s ease-in-out infinite}
.gl-ac-laugh .s2{animation:gLaughB 1.1s ease-in-out infinite}
.gl-ac-walktalk .tk{animation:gStream 2.4s linear infinite}
.gl-ac-walktalk .s1{animation:gStepA 1s ease-in-out infinite}
.gl-ac-walktalk .s2{animation:gStepB 1s ease-in-out infinite}
.gl-ac-chase .tk{animation:gStream 1.1s linear infinite}
.gl-ac-chase .s1{animation:gStepA .7s ease-in-out infinite;opacity:.5}
.gl-ac-chase .s2{animation:gStepB .7s ease-in-out infinite}
.gl-ac-handoff .ob{display:block;animation:gPass 2.8s ease-in-out infinite}
/* `.gl.` on these two, and only these two. The group's base rule is
   `.gl[class*="gl-ac-"] .s2` — an attribute selector, so (0,3,0) — and a plain
   `.gl-ac-reveal .s2` is (0,2,0) and loses. Both of these are the actions with
   one performer rather than two, so losing meant the two tiles that are not
   about a pair were the only evidence that the base rule outranked them. */
.gl.gl-ac-reveal .s2,.gl.gl-ac-turn .s2{display:none}
.gl.gl-ac-reveal .s1{transform:none;animation:gReveal 3s ease-in-out infinite}
.gl-ac-reveal .lt{opacity:.22;animation:gWipe 3s ease-in-out infinite}
.gl.gl-ac-turn .s1{transform:none;animation:gTurn 2.6s ease-in-out infinite}

@keyframes gScale{to{transform:scale(1.5)}}
@keyframes gCreep{to{transform:scale(1.08)}}
@keyframes gPan{to{transform:translateX(calc(var(--d,1) * 11px))}}
@keyframes gPanHalf{to{transform:translateX(calc(var(--d,1) * 4.5px))}}
@keyframes gTilt{to{transform:translateY(calc(var(--v,1) * 7px))}}
@keyframes gTiltHalf{to{transform:translateY(calc(var(--v,1) * 2.5px))}}
@keyframes gCrane{to{transform:translateY(calc(var(--v,1) * 9px)) scale(.84)}}
@keyframes gSquash{0%,100%{transform:scaleX(1)}50%{transform:scaleX(.55)}}
/* Exactly one post spacing, so the loop has no seam to see. */
@keyframes gStream{from{transform:translateX(10px)}to{transform:translateX(-10px)}}
@keyframes gBob{0%,100%{transform:translateY(0)}50%{transform:translateY(-1.1px)}}
@keyframes gJit{0%{transform:translate(0,0)}20%{transform:translate(.9px,-.6px)}
  40%{transform:translate(-.8px,.5px)}60%{transform:translate(.5px,.9px)}
  80%{transform:translate(-.6px,-.5px)}}
@keyframes gWhip{0%,26%{transform:translateX(13px);filter:none}
  46%,54%{transform:translateX(0);filter:blur(1.7px)}
  74%,100%{transform:translateX(-13px);filter:none}}
@keyframes gFocusA{from{filter:blur(0);opacity:.85}to{filter:blur(1.4px);opacity:.4}}
@keyframes gFocusB{from{filter:blur(1.4px);opacity:.4}to{filter:blur(0);opacity:.95}}
@keyframes gJabA{0%,100%{transform:translateX(-6px)}50%{transform:translateX(-2.2px)}}
@keyframes gJabB{0%,100%{transform:translateX(6px)}50%{transform:translateX(2.2px)}}
@keyframes gCloseA{0%{transform:translateX(-8px)}45%,100%{transform:translateX(-2.4px)}}
@keyframes gCloseB{0%{transform:translateX(8px)}45%,100%{transform:translateX(2.4px)}}
@keyframes gSayA{0%,44%,100%{transform:translateX(-6px) scale(1)}
  20%{transform:translateX(-6px) scale(1.14)}}
@keyframes gSayB{0%,60%,100%{transform:translateX(6px) scale(1)}
  78%{transform:translateX(6px) scale(1.14)}}
@keyframes gLaughA{0%,100%{transform:translate(-6px,0)}50%{transform:translate(-6px,-1.6px)}}
@keyframes gLaughB{0%,100%{transform:translate(6px,0)}50%{transform:translate(6px,-1.4px)}}
@keyframes gStepA{0%,100%{transform:translate(-6px,0)}50%{transform:translate(-6px,-1.2px)}}
@keyframes gStepB{0%,100%{transform:translate(6px,-1.2px)}50%{transform:translate(6px,0)}}
@keyframes gPass{0%,10%{transform:translateX(-5px)}55%,100%{transform:translateX(5px)}}
@keyframes gReveal{0%,15%{opacity:0}55%,100%{opacity:.9}}
@keyframes gWipe{0%,10%{transform:scaleX(7) translateX(1px)}60%,100%{transform:scaleX(7) translateX(-9px)}}
@keyframes gTurn{0%,100%{transform:scaleX(1)}45%,55%{transform:scaleX(.28)}}
/* Frozen mid-move rather than switched off: a still diagram is still a
   diagram, and a tile that reverts to the neutral skeleton would make half the
   camera group identical to each other. */
@media (prefers-reduced-motion:reduce){
  .gl *{animation-play-state:paused !important;animation-delay:-1.2s !important}
}

/* The pill rail ----------------------------------------------------------
   One rail, shared by Image and Video, because the prompt is shared and a pill
   is part of the prompt. It costs nothing while empty — not a collapsed row, no
   element at all — which is the condition for putting anything above the
   options strip at all. */
#shot-rail{margin:8px 2px 0;gap:6px}
#shot-rail:empty{display:none;margin:0}
.spill{display:inline-flex;align-items:center;gap:7px;background:rgba(255,255,255,.07);
  border:1px solid rgba(255,255,255,.1);border-radius:999px;padding:3px 5px 3px 7px;
  color:#ddd;font-size:12.5px;max-width:100%}
.spill>.gl{width:26px;height:18px;flex:none;opacity:.85}
/* Frozen in the rail. The tile is the animated version because motion is what
   teaches a move you have not picked yet; a pill is the record of a decision
   already made, and twenty of them moving under the prompt is motion competing
   with the canvas for information you already have — the word is right there.
   Paused mid-move rather than switched off, and for the same reason
   reduced-motion pauses instead of cancelling: a camera glyph reverted to the
   neutral skeleton is the same drawing for all twenty-one of them. */
.spill .gl *{animation-play-state:paused;animation-delay:-1.2s}
.spill b{font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
/* An empty valued pill is a decision started and not finished, and it says so
   by staying visibly empty rather than by erroring. */
.spill.val b{color:var(--dim)}
.spill.val b.set{color:#ddd;font-style:italic}
.spill.val:not(.open){cursor:text}
.spill.open{background:rgba(255,255,255,.11);border-color:rgba(255,255,255,.2)}
/* width:auto and flex:none on both, because the base rule three hundred lines
   up is `input,textarea,select{width:100%}` — inside a flex pill that made the
   language select 245px of a 480px pill and squeezed the line of dialogue,
   which is the only thing in there anyone is reading, down to 156. */
.spill input{border:0;background:none;color:#fff;font-size:12.5px;padding:2px 0;
  flex:1 1 250px;width:auto;min-width:110px;max-width:min(38vw,340px)}
.spill select{border:0;background:rgba(255,255,255,.09);color:#ccc;font-size:11px;
  border-radius:6px;padding:0 3px;height:19px;flex:none;width:auto}
.spill .x{border:0;background:none;color:var(--mut);cursor:pointer;font-size:13px;
  line-height:1;width:19px;height:19px;border-radius:50%;flex:none}
.spill .x:hover{background:rgba(255,255,255,.15);color:#fff}
/* Dim, never gone. The rail is shared and the models are not: a pill this
   model does not read has to stay where you put it, because switching back is
   one click and a rail that forgets is worse than a rail that dims. */
.spill.off{opacity:.32}

/* What the model will actually read. Collapsed to one 11px line, which is the
   whole argument for it existing: the alternative to showing the compiled
   document is a three-minute render to find out. */
#shot-peek{margin:7px 2px 0}
#shot-peek>button{border:0;background:none;color:var(--dim);font-size:11.5px;
  cursor:pointer;padding:2px 0;display:inline-flex;align-items:center;gap:5px}
#shot-peek>button:hover{color:var(--fg)}
#shot-peek>button::before{content:"";width:0;height:0;border:4px solid transparent;
  border-left-color:currentColor;border-right:0;transition:transform .12s}
#shot-peek.open>button::before{transform:rotate(90deg) translateX(1px)}
#shot-peek pre{margin:7px 0 0;padding:10px 12px;border-radius:11px;background:rgba(255,255,255,.04);
  border:1px solid var(--line);font:11.5px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;
  color:#c8c8c8;white-space:pre-wrap;word-break:break-word;max-height:184px;overflow:auto}

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
/* The queue reports where you started it, under the head it belongs to, rather
   than in one shared strip at the top of Settings — with a button per family
   there is no longer a single place that "the download" means. */
.fam-prog{margin:0 2px 12px}

/* The size popover. Aspect is shown as a rectangle at its own proportions,
   because that is the one representation of a ratio nobody has to read — the
   same argument the shot tiles make for a camera move, and the reason this is
   a palette rather than a longer select. */

/* The sampling popover. A form, not a palette — a sampler name and a step count
   are not things a picture of them could teach, which is the line the shot
   tiles sit on the other side of. */
.menu.form{padding:9px;width:auto;min-width:250px}
.form .frow{display:grid;grid-template-columns:74px 1fr;align-items:center;gap:8px;
  margin:0 0 7px;font-size:11.5px;color:var(--dim)}
.form .frow>span{white-space:nowrap}
.form .frow select,.form .frow input{height:30px;padding:0 7px;font-size:12px}
/* The one control whose name explains nothing, so it gets the sentence the
   others do not need. Spanning both columns because a hint indented under a
   74px label is a hint shaped like a value. */
.form .frow>i{grid-column:1/-1;font-style:normal;font-size:10.5px;color:var(--mut);line-height:1.4}
.form .sz-reset{width:100%;margin-top:2px;padding:7px 0;border:1px solid var(--line);
  border-radius:9px;background:none;color:var(--dim);cursor:pointer;font:500 11px/1 inherit}
.form .sz-reset:hover{color:var(--fg);border-color:rgba(255,255,255,.3)}
/* A dot, not a colour change: the button's text is the resolved numbers, and
   recolouring those would say "warning" about a value you deliberately chose. */
#g-sampling,#v-sampling{width:auto;padding:0 11px;font-size:11.5px;color:var(--fg);
  font-variant-numeric:tabular-nums}
#g-sampling.edited::after,#v-sampling.edited::after{content:'';display:inline-block;width:4px;height:4px;margin-left:6px;
  border-radius:50%;background:var(--fg);opacity:.55;vertical-align:1.5px}
.menu.sizer{padding:10px;width:auto;min-width:236px}
.sizer .ars{display:grid;grid-template-columns:repeat(4,1fr);gap:4px}
.sizer .ar{display:flex;flex-direction:column;align-items:center;justify-content:flex-end;
  gap:5px;height:56px;padding:5px 2px;border:1px solid transparent;border-radius:9px;
  background:none;color:var(--dim);cursor:pointer;font:500 10px/1 inherit}
.sizer .ar i{display:block;border:1.5px solid currentColor;border-radius:2px;opacity:.75}
.sizer .ar:hover{background:rgba(255,255,255,.07);color:var(--fg)}
.sizer .ar.on{background:rgba(255,255,255,.13);color:var(--fg);border-color:rgba(255,255,255,.22)}
/* Scale is a separate row because it is a separate decision: the shape you want
   and how much of it you can afford are not the same question, and the old
   single select forced them to be answered together. */
.sizer .scales{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-top:9px}
.sizer .sc{padding:7px 0;border:1px solid var(--line);border-radius:9px;background:none;
  color:var(--dim);cursor:pointer;font:500 11.5px/1 inherit}
.sizer .sc:hover{color:var(--fg);border-color:rgba(255,255,255,.3)}
.sizer .sc.on{background:rgba(255,255,255,.13);color:var(--fg);border-color:rgba(255,255,255,.26)}
.sizer .sz-custom{display:flex;align-items:center;gap:6px;margin-top:9px}
.sizer .sz-custom label{display:flex;align-items:center;gap:5px;margin:0;flex:1;
  font-size:10.5px;color:var(--dim)}
.sizer .sz-custom input{width:100%;height:30px;padding:0 7px;font-size:12px;text-align:center}
.sizer .sz-swap{flex:none;width:26px;height:26px;display:grid;place-items:center;border:0;
  background:none;color:var(--dim);cursor:pointer;border-radius:7px;margin-top:12px}
.sizer .sz-swap:hover{background:rgba(255,255,255,.09);color:var(--fg)}
.sizer .sz-swap svg{width:14px;height:14px}
.sizer .sz-note{margin:8px 2px 1px;font-size:10.5px}
/* Wide enough to hold "16:9 · 2304×1296" without the strip reflowing when the
   scale changes under it. */
#g-size,#v-size{width:auto;padding:0 11px;font-size:11.5px;color:var(--fg);font-variant-numeric:tabular-nums}

/* A row per LoRA, not a card per LoRA. The catalogue below is cards because each
   entry there is a decision with a size, a repo and a licence attached; this is
   a list you scan for a name you recognise, and at a dozen LoRAs the card's
   16px of padding is the difference between reading it and scrolling it. */
.lora-row{display:flex;align-items:center;gap:12px;padding:9px 2px;border-top:1px solid var(--line)}
.lora-row:first-child{border-top:0;padding-top:2px}
.lora-row b{font-size:13px;font-weight:600}
/* Always visible, unlike the dataset card's ×, which only appears on hover
   because it sits on top of a picture it would otherwise cover. There is nothing
   under this one, and the whole reason to open this card is to find it. */
.lora-x{border:0;background:none;color:var(--dim);cursor:pointer;font:15px/1 inherit;
  padding:6px 8px;border-radius:9px;flex:none}
.lora-x:hover{background:rgba(248,113,113,.14);color:#f87171}
.kv{display:grid;grid-template-columns:132px 1fr;gap:5px 14px;font-size:13px}
.kv dt{color:var(--mut)}
.kv dd{color:#e8e8e8;word-break:break-word}
/* Flex, not grid. As a grid container with `place-items:center` the image was
   the only in-flow item, in an implicit auto-sized row, and `align-items:center`
   stops that row stretching — so the row was sized from the image and the
   image's `max-height:100%` was a percentage of a height derived from itself.
   Cyclic, so the browser drops the constraint. `max-width:100%` kept working
   because the inline axis is definite, which is exactly why this only showed up
   on images taller than the viewport: a 640x1536 rendered at 640x1536 and ran
   592px off the bottom of the screen, and the strip you could see read as a
   centre crop. A flex item's percentages resolve against the container's
   content box, which `inset:0` makes definite, so `100%` means the screen. */
.lb{position:fixed;inset:0;background:rgba(0,0,0,.93);z-index:70;
  display:flex;align-items:center;justify-content:center;padding:34px}
.lb-track{position:absolute;inset:0;display:flex;touch-action:pan-y;will-change:transform}
/* Only while settling. During the drag the transform is written every frame, and
   a transition on that lags the thumb — which is the whole thing this is for. */
.lb-track.snap{transition:transform .3s cubic-bezier(.22,.61,.36,1)}
.lb-slide{flex:0 0 100%;height:100%;display:grid;place-items:center;padding:30px}
/* Belt and braces with draggable="false": Safari honours -webkit-user-drag and
   ignores the attribute on some elements. */
.lb img,.lb video{-webkit-user-drag:none;user-select:none;-webkit-user-select:none}
.lb img,.lb video{max-width:100%;max-height:100%;width:auto;height:auto;
  min-width:0;min-height:0;object-fit:contain}
.lb .x{position:absolute;top:14px;right:16px;width:40px;height:40px;border:0;background:none;
  color:#fff;padding:8px;cursor:pointer;z-index:3;
  filter:drop-shadow(0 1px 3px rgba(0,0,0,.7))}
.lb .x:hover{opacity:.75}
/* Chrome off. Navigation lives in the top corners while you are moving between
   layers; once you have stopped on one picture there is nowhere left to go, so
   it gets out of the way until you ask for it. */
.lb.bare .x,.lb.bare .lb-all,.lb.bare .lb-nav,.lb.bare .lb-at{opacity:0;pointer-events:none}
.lb .x,.lb .lb-all,.lb .lb-nav,.lb .lb-at{transition:opacity .16s ease}
/* Paging, for the pointer that cannot swipe. Big, edge-anchored and mostly
   transparent — they sit over the picture, so they earn their place by being
   where a thumb already is rather than by being visible. */
.lb .lb-nav{position:absolute;top:50%;transform:translateY(-50%);width:52px;height:78px;
  border:0;background:rgba(0,0,0,.4);color:#e8e8e8;cursor:pointer;padding:22px;
  border-radius:12px;opacity:.5;transition:opacity .12s}
.lb .lb-nav:hover{opacity:1;background:rgba(0,0,0,.66)}
.lb .lb-nav[disabled]{opacity:.12;cursor:default}
.lb .lb-nav.prev{left:14px}
.lb .lb-nav.next{right:14px;transform:translateY(-50%) rotate(180deg)}
/* Where you are in the set. The one thing swiping cannot tell you, and the
   reason you can flick through twenty takes without losing your place. */
.lb .lb-at{position:absolute;top:20px;left:50%;transform:translateX(-50%);
  font:500 12px/1 inherit;color:#bbb;font-variant-numeric:tabular-nums}
.lb .lb-all{position:absolute;top:14px;right:60px;width:40px;height:40px;border:0;
  background:none;color:#fff;padding:9px;cursor:pointer;z-index:3;
  filter:drop-shadow(0 1px 3px rgba(0,0,0,.7))}
.lb .lb-all:hover{opacity:.75}
/* The last generation, beside Generate — and only where that is the shortest
   way back to it.
   
   Below 1024px the gallery has no other home: the drawer stacks, so there is no
   column beside the canvas and the header is the far corner of a tall screen.
   Above it the drawer is a column that is already next to the picture and the
   header button already opens it, so this is a second door onto a room with a
   door — and it lands in the composer, which is the row this whole redesign has
   been trying to empty. Desktop was not the problem the thumbnail solved, so it
   does not get the fix. */
.shot-back{display:none}
@media (max-width:1024px){
  .shot-back{width:36px;height:36px;flex:none;padding:0;overflow:hidden;border-radius:10px;
    border:1px solid rgba(255,255,255,.22);background:none;cursor:pointer;display:block}
  .shot-back:hover{border-color:rgba(255,255,255,.5)}
  .shot-back img,.shot-back video{width:100%;height:100%;object-fit:cover;display:block}
  .shot-back.hide{display:none}
}
@media (hover:none) and (max-width:1024px){ .shot-back{width:44px;height:44px} }

/* Datasets -------------------------------------------------------------- */
/* flex-column, not the default: a <button> centres its content vertically, so
   once one card in the row is taller than another the covers float in the
   middle of their cards with black above them. */
.ds-card{border:1px solid var(--line);background:rgba(255,255,255,.02);border-radius:14px;overflow:hidden;
  cursor:pointer;text-align:left;padding:0;color:inherit;font:inherit;
  display:flex;flex-direction:column;align-items:stretch}
.ds-card:hover{border-color:rgba(255,255,255,.24)}
/* The set you are looking at, and the one Start training will use. */
.ds-card.sel{border-color:rgba(255,255,255,.55)}
/* Not kept yet. The border is the whole statement — a dashed edge is what a
   provisional thing looks like, and it costs the card no words to say it. The
   same dash and the same alpha as the drop target, deliberately: a draft is
   what came off it, and at --line's 10% the gaps only made the card look
   fainter than its neighbours rather than different from them. */
.ds-card.draft{border-style:dashed;border-color:rgba(255,255,255,.2)}
/* The delete target has to sit beside the card rather than inside it: the card
   is a <button>, and a button cannot contain one. */
/* width:100% because a <button> shrinks to its content: without it the cards
   end at wherever each name happens to stop and the rail loses its edge. */
.ds-row{position:relative;min-width:0}
.ds-row .ds-card{width:100%}
.ds-row .ds-x{position:absolute;top:50%;right:9px;transform:translateY(-50%);
  width:24px;height:24px;border:0;border-radius:50%;background:rgba(0,0,0,.62);
  color:#eee;cursor:pointer;font-size:13px;line-height:1;padding:0;opacity:0;
  transition:opacity .12s}
.ds-row:hover .ds-x,.ds-row .ds-x:focus-visible{opacity:1}
/* Which group a card is in, said once above the group rather than on every
   card in it. */
.ds-group{grid-column:1/-1;margin:14px 2px 2px;font-size:11px;letter-spacing:.06em;
  text-transform:uppercase;color:var(--dim)}
.ds-group:first-child{margin-top:0}
.ds-group span{text-transform:none;letter-spacing:0}
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
/* The third line is cut through the glyphs rather than between them, and that
   is left alone deliberately. It is not a rendering fault dressed as a feature:
   a sliced line appears exactly when the caption runs past the box and never
   otherwise, so it is a perfectly correlated "there is more" that costs no
   pixels and no element. Snapping the height to whole lines was tried and
   reverted — it buys a tidier edge and pays for it by making a four-line
   caption end flush, indistinguishable from one that really stops there. */
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
/* The strip's version: a dashed outline when empty and the frame itself once
   filled. The thumbnail is the label — a filled first tile says
   "image-to-video" more directly than the words do.

   That was only ever true of the filled half. Empty, the row was four 36px
   dashed squares of near-identical weight — two fixed keyframe slots and two
   add-buttons for a tray — told apart by tooltip and a 1px rule, which is the
   same hover-to-find-out failure `.opt>.lead` was added to fix on the numeric
   fields. So an empty tile is named and a filled one collapses back to the
   square: the words are scaffolding for a picture that is not there yet, and
   the moment it arrives they are in its way. */
/* The way most sets begin, so it is the size of that fact: the whole canvas
   when nothing is chosen, and a target you can hit without aiming. */
.drop.hero{flex:1;min-height:0;display:flex;flex-direction:column;align-items:center;
  justify-content:center;border-radius:20px;padding:34px}
.drop.hero .drop-face{text-align:center;pointer-events:none}
.drop.hero .drop-face input{pointer-events:auto}
.drop.hero .glyph{width:40px;height:40px;margin:0 auto 16px;opacity:.4}
.drop.hero b{font-size:15px;font-weight:600}
.drop.mini{height:36px;flex:none;padding:0 11px 0 9px;border-radius:10px;overflow:hidden;
  background:rgba(255,255,255,.03);display:flex;align-items:center;justify-content:center;
  gap:7px;color:var(--dim)}
.drop.mini:hover{border-color:rgba(255,255,255,.4);color:var(--fg)}
/* :not(.lead), because label() inserts a <span> too and this rule pinned it to
   the icon's 16px box — every label was clipped to "Pictu" by the overflow the
   tile needs for its thumbnail. */
.drop.mini>span:not(.lead){padding:0;display:grid;place-items:center;width:16px;height:16px;flex:none}
/* After the icon, not before it: label() inserts afterbegin, and the glyph is
   what the eye lands on first. */
.drop.mini>.lead{order:2}
.drop.mini img{width:100%;height:100%;max-height:none;object-fit:cover;border-radius:0}
.drop.mini.set{width:36px;padding:0;border-style:solid;border-color:rgba(255,255,255,.28)}
.drop.mini.set>.lead{display:none}
/* Out of play this run because its opposite number is filled. Dimmed rather
   than removed: the row's whole job is to show that keyframes and references
   are alternatives, and a control that disappears when you use its neighbour
   teaches nothing except that the page lost it. */
.drop.mini.off{opacity:.3;pointer-events:none}
/* Dimmed like .off and, unlike it, still hoverable. .off means "the choice you
   already made put this out of play for this run", which the row's own shape
   explains; this means "the weight it needs is not on the volume", which
   nothing explains — and a tile with pointer-events:none cannot be hovered, so
   it could not deliver the one sentence that would. */
.drop.mini.locked{opacity:.3;cursor:default}
#canvas-acts{position:absolute;top:12px;right:12px;z-index:4;display:flex;gap:6px;
  opacity:.32;transition:opacity .12s}
#canvas:hover #canvas-acts,#canvas-acts:focus-within{opacity:1}
#canvas-acts .ico{width:30px;height:30px;background:rgba(0,0,0,.55);backdrop-filter:blur(8px);
  border-radius:9px}
#canvas-acts .ico:hover{background:rgba(0,0,0,.78);color:var(--fg)}
/* No pointer to hover with.
   
   Half a dozen controls on this page are revealed by :hover, which on a
   trackpad is a light touch and on an iPad is nothing at all — the gallery
   card's download and delete and the dataset tile's remove were at opacity:0
   with no way to reach them. They are not "hard to find" on touch, they are
   absent, and the feature behind them may as well not have shipped.
   
   The quiet-at-rest ones are a different case and stay quiet, just less so: the
   canvas actions and the shot actions are legible at .32 with a mouse because
   hover is one move away, and a tablet has no such move — so they sit up where
   they can be read and tapped without hunting.
   
   hover:none rather than pointer:coarse, because the question is whether hover
   exists, not how precise the pointer is: a trackpad-less touchscreen laptop
   fails this test too and has exactly the same problem. */
@media (hover:none){
  .gal .quick{opacity:1}
  .tile .rm{opacity:1}
  .shot .acts,#canvas-acts{opacity:.85}
  /* Handles ride on .sel already, which a tap sets — so a box you have selected
     is resizable and one you have not is not cluttered with eight dots. */
}
/* 44px is Apple's minimum and these are 36. Only the composer strip, because
   that is what you drive a render from; the settings sheet is a form you visit
   rather than a surface you work on. */
@media (hover:none) and (max-width:1024px){
  .opt,.opt.ib,.s,.drop.mini,.kinds button{min-height:44px}
  .opt select,.opt input{height:42px}
  .b{min-height:44px}
}
/* The region map. Same height as the 36px controls beside it so the bar stays
   one row, and the frame is drawn rather than bordered so the boxes sit inside
   the picture's proportions rather than inside a button's. */
.rmap{width:52px;height:36px;flex:none;padding:4px 6px;border:1px solid var(--line);
  border-radius:10px;background:rgba(255,255,255,.03);cursor:pointer}
.rmap:hover{border-color:rgba(255,255,255,.4)}
.rmap:focus-visible{outline:2px solid rgba(255,255,255,.5);outline-offset:1px}
.rmap svg{width:100%;height:100%;display:block;overflow:visible}
.rmap .fr{fill:none;stroke:rgba(255,255,255,.22);stroke-width:1}
/* Unselected boxes are outlines and the selected one is filled, which is the
   same distinction the canvas draws — a box you are editing against boxes that
   are merely there. */
.rmap .bx{fill:rgba(255,255,255,.10);stroke:rgba(255,255,255,.45);stroke-width:1;cursor:pointer}
.rmap .bx:hover{fill:rgba(255,255,255,.22)}
.rmap .bx.on{fill:rgba(255,255,255,.82);stroke:#fff}
.pad{padding:26px}

/* Reference chips. Numbered, because the number is the <Picture n> the prompt
   refers to — it is data, not decoration. */
.ref{position:relative;width:64px;height:64px;border-radius:11px;overflow:hidden;border:1px solid var(--line);
  background:rgba(255,255,255,.04)}
.ref img,.ref video{width:100%;height:100%;object-fit:cover;display:block}
.ref b{position:absolute;left:4px;top:3px;font-size:10px;font-weight:600;color:#fff;
  background:rgba(0,0,0,.7);padding:1px 5px;border-radius:4px}
.ref button.x{position:absolute;top:3px;right:3px;width:19px;height:19px;border:0;border-radius:50%;
  background:rgba(0,0,0,.66);color:#eee;font-size:11px;line-height:1;cursor:pointer}
/* What this picture is *for*, across the foot of the chip it belongs to.
   Always present, never empty, because this is the control that finally makes
   "do not describe your reference image" something other than advice — there
   is somewhere for that description to go now, and it is one click. A tile that
   only appeared once a role was set would be a feature you had to already know
   about to find, which is the state the keyframe pair was rescued from. */
.ref .role{position:absolute;left:0;right:0;bottom:0;border:0;padding:2px 3px;
  background:rgba(0,0,0,.72);color:#e8e8e8;font-size:9.5px;line-height:1.3;text-align:center;
  cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ref .role:hover{background:rgba(0,0,0,.88)}
.ref .role.none{color:var(--dim)}
/* Regions --------------------------------------------------------------- */
/* A region is a rectangle on the canvas, so it is drawn on the canvas. The
   four coordinates were the right parameter and the wrong primary: "0.5 0 0.5
   1" is a rectangle you rebuild in your head every time, which is why the row
   that carried those numbers also had to carry a 32px picture of them. That
   picture is this, at the size of the frame and grabbable — and the numbers,
   still here in the inspector, are now a readout that moves while you drag.
   Dragging is what teaches them; they never taught the dragging.

   It also fixes what the stack cost: a row was ~74px, so the eight regions the
   backend allows were ~592px of console against a 54dvh cap — the feature's
   fullest state broke the rule the console exists to hold. One inspector row
   is the same height whether there is one box or eight. */
/* `.frame`, not `.stage` — `.view > .stage` is already the canvas-plus-console
   column, and a second `.stage` carrying aspect-ratio and a max-height would
   have landed on it and collapsed the whole view. Third time this file has
   been bitten by a reused class name; see the notes on `.lb` and on
   `.blank`/`.empty`. */
/* Both dimensions in pixels, set by layoutFrame() off the canvas the way
   --shot-h is. `aspect-ratio` with an auto width is the obvious way to write
   this and does not work here: the frame is a flex item in a column, so its
   width is the cross size, and the ratio never transferred into it — the box
   came out 2px wide, which is exactly its two borders and nothing else. A dvh
   sum would be wrong for the height anyway the moment the console grew, so
   both numbers were going to be measured regardless. */
.frame{position:relative;align-self:center;flex:none;
  width:var(--frame-w,0);height:var(--frame-h,0);
  border-radius:16px;border:1px solid var(--line);background:rgba(255,255,255,.02);
  overflow:hidden;touch-action:none}
/* Absolute, not a flow child: a child sized height:100% would establish the
   container's height itself and aspect-ratio would never apply — the same trap
   documented on .tile .ph. */
.frame>.plate{position:absolute;inset:0;width:100%;height:100%;object-fit:cover}
/* Thirds. Faint enough to be a viewfinder rather than a grid you are working
   against, and gone the moment there is a picture underneath to judge. */
.frame>.thirds{position:absolute;inset:0;pointer-events:none;opacity:.5;
  background:
    linear-gradient(to right,transparent calc(33.333% - .5px),rgba(255,255,255,.07) calc(33.333% - .5px),
      rgba(255,255,255,.07) calc(33.333% + .5px),transparent calc(33.333% + .5px)),
    linear-gradient(to right,transparent calc(66.666% - .5px),rgba(255,255,255,.07) calc(66.666% - .5px),
      rgba(255,255,255,.07) calc(66.666% + .5px),transparent calc(66.666% + .5px)),
    linear-gradient(to bottom,transparent calc(33.333% - .5px),rgba(255,255,255,.07) calc(33.333% - .5px),
      rgba(255,255,255,.07) calc(33.333% + .5px),transparent calc(33.333% + .5px)),
    linear-gradient(to bottom,transparent calc(66.666% - .5px),rgba(255,255,255,.07) calc(66.666% - .5px),
      rgba(255,255,255,.07) calc(66.666% + .5px),transparent calc(66.666% + .5px))}
.frame.hot,.shot.hot{outline:1px dashed rgba(255,255,255,.45);outline-offset:-4px}

/* The layer is reparented between the frame and the first still, so every
   coordinate in it is a percentage and nothing measures its host. */
/* Shown by intent, not by mode.
   
   The boxes used to stand for as long as regions were armed, which meant a
   continuous white rectangle across every picture you rendered in the mode you
   render most in — chrome painted over the one thing the page exists to show.
   They are visible now while you are actually working on a region: the caret in
   the region bar, a box under the pointer mid-drag, or a file over the window,
   which is the moment "you can drop that here" needs saying.
   
   pointer-events goes with the paint. An invisible box that still swallows
   clicks would make inspecting your own render place regions by accident, and
   an invisible box that only *looks* gone is not the fix that was asked for.
   The way back in is the region bar, which is on screen the whole time regions
   are armed — click into its prompt and the boxes come back. */
#region-layer{position:absolute;inset:0;touch-action:none;cursor:crosshair;
  opacity:0;pointer-events:none;transition:opacity .16s ease}
#region-layer.show{opacity:1;pointer-events:auto}
#region-layer.off{display:none}
/* A region of the picture, not a line on top of it. The 1px white stroke read
   as UI at every size and fought whatever was underneath — bright plate or
   dark, it was the most contrasty thing in the frame. Corner brackets plus a
   barely-there wash say "this area" and leave the middle of the box clear,
   which is where the render you are judging actually is. */
.rbox{position:absolute;border-radius:5px;cursor:move;touch-action:none;overflow:hidden;
  background:rgba(255,255,255,.045);
  box-shadow:inset 0 0 0 1px rgba(255,255,255,.16),inset 0 0 0 2px rgba(0,0,0,.18)}
/* The brackets. Drawn with two gradients per corner so there is no extra
   element per box — eight boxes would otherwise be thirty-two more nodes on a
   layer that is redrawn on every pointermove. */
.rbox::before,.rbox::after{content:'';position:absolute;width:15px;height:15px;
  pointer-events:none;opacity:.9}
.rbox::before{left:-1px;top:-1px;
  border-left:2px solid rgba(255,255,255,.92);border-top:2px solid rgba(255,255,255,.92);
  border-radius:5px 0 0 0;filter:drop-shadow(0 0 1px rgba(0,0,0,.6))}
.rbox::after{right:-1px;bottom:-1px;
  border-right:2px solid rgba(255,255,255,.92);border-bottom:2px solid rgba(255,255,255,.92);
  border-radius:0 0 5px 0;filter:drop-shadow(0 0 1px rgba(0,0,0,.6))}
/* Solid when the box holds an identity — a resolvable LoRA or a photo — and
   faint when it holds neither, which is the same distinction the 32px plots
   drew and the one that decides what comes out: an empty rectangle is filled
   by the scene prompt, a box with an identity in it is a person. */
.rbox{opacity:.4}
.rbox.armed{opacity:1}
.rbox.sel{background:rgba(255,255,255,.08);
  box-shadow:inset 0 0 0 1px rgba(255,255,255,.42),inset 0 0 0 2px rgba(0,0,0,.22)}
.rbox>.face{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:.5;
  pointer-events:none}
.rbox>.tag{position:absolute;left:0;top:0;max-width:100%;padding:3px 7px;
  background:rgba(0,0,0,.6);backdrop-filter:blur(8px);color:#fff;
  border-radius:0 0 6px 0;font:500 11px/1.25 inherit;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis;pointer-events:none}
.rbox>.tag em{font-style:normal;color:var(--mut)}
/* Handles are hidden until the box is worth resizing — eight dots on every box
   at eight boxes is 64 dots on a picture you are trying to look at. */
.rbox>i{position:absolute;width:11px;height:11px;border-radius:3px;
  background:#fff;box-shadow:0 0 0 1px rgba(0,0,0,.5);opacity:0;transition:opacity .12s}
.rbox:hover>i,.rbox.sel>i{opacity:1}
.rbox>i[data-h=nw]{left:-5px;top:-5px;cursor:nwse-resize}
.rbox>i[data-h=ne]{right:-5px;top:-5px;cursor:nesw-resize}
.rbox>i[data-h=sw]{left:-5px;bottom:-5px;cursor:nesw-resize}
.rbox>i[data-h=se]{right:-5px;bottom:-5px;cursor:nwse-resize}
.rbox>i[data-h=n]{left:50%;top:-5px;margin-left:-5px;cursor:ns-resize}
.rbox>i[data-h=s]{left:50%;bottom:-5px;margin-left:-5px;cursor:ns-resize}
.rbox>i[data-h=w]{left:-5px;top:50%;margin-top:-5px;cursor:ew-resize}
.rbox>i[data-h=e]{right:-5px;top:50%;margin-top:-5px;cursor:ew-resize}
/* Snap guides: drawn only while a drag is landing on one, so the line is
   feedback rather than furniture. */
#region-layer>.guide{position:absolute;background:rgba(255,255,255,.55);pointer-events:none}
#region-layer>.guide.v{top:0;bottom:0;width:1px}
#region-layer>.guide.h{left:0;right:0;height:1px}

/* Drag-intent reveal -----------------------------------------------------
   Every one of this app's best gestures is a drop, and every one of them was
   invisible until you had already guessed it: a photo onto a region box is
   that character's likeness, a photo onto the bare canvas is the world the
   render happens inside, files onto an open contact sheet join the set. None
   of those can be advertised at rest without putting furniture on a canvas
   the whole layout exists to keep clear.

   So they are advertised at the only moment they are relevant. Dragging a file
   over the window *is* the question "where can this go", and this block is the
   answer to it. `dragging` is on the body only while a file is actually over
   the page, so at rest every selector here matches nothing and the page is
   exactly what it was.

   Two levels, because "you may drop here" and "here is what dropping does" are
   different questions. Everything eligible outlines itself; only the target
   under the cursor says what it is. Naming all of them at once would put eight
   captions on a picture at eight boxes, which is the wall of text the region
   rows were deleted for, redrawn on the canvas. */
body.dragging .can-drop:not(.hot):not(.locked){border-color:rgba(255,255,255,.34)}
body.dragging .frame.can-drop:not(.hot),body.dragging .shot.can-drop:not(.hot),
body.dragging #ds-sheet:not(.hot){outline:1px dashed rgba(255,255,255,.2);outline-offset:-4px}
#ds-sheet{position:relative}
#ds-sheet.hot{outline:1px dashed rgba(255,255,255,.45);outline-offset:-4px}
/* The caption. Generated content, so at rest it is not an element that exists
   and is hidden — it is an element that was never built. */
body.dragging .frame.hot::after,body.dragging .shot.hot::after,
body.dragging .rbox.drop-hit::after,body.dragging #ds-sheet.hot::after{
  content:attr(data-drop);position:absolute;left:50%;top:50%;
  transform:translate(-50%,-50%);padding:5px 11px;border-radius:999px;
  background:rgba(0,0,0,.74);backdrop-filter:blur(8px);color:#f5f5f5;
  font:500 12px/1 inherit;white-space:nowrap;max-width:calc(100% - 12px);
  overflow:hidden;text-overflow:ellipsis;pointer-events:none;z-index:6}
/* The contact sheet is the one target taller than the window — it is the thing
   that scrolls — so centring on the element puts the caption at the middle of
   forty images, which is off-screen for all but the shortest sets. Fixed to the
   viewport instead, low enough to clear the tile under the cursor and high
   enough to clear the training bar. */
body.dragging #ds-sheet.hot::after{position:fixed;top:auto;bottom:104px;
  transform:translateX(-50%);z-index:50}
/* A box under the cursor is lit the way a selected one is, and additionally
   pulled to full opacity — an unarmed box sits at .34, which is legible as
   "nothing in this one yet" and illegible as "this is the one you are about to
   drop on". */
body.dragging .rbox.drop-hit{opacity:1;border-color:#fff}

.wrap{display:flex;flex-wrap:wrap;gap:8px;align-items:center}

/* Train ----------------------------------------------------------------- */
.grid2{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}

@media(max-width:1180px){:root{--drawer:284px}}
@media(max-width:1024px){
  /* Scrollable, but still at least a windowful. `height:auto` alone dropped the
     fill-the-viewport behaviour along with the fixed height, so on anything
     taller than the content — a 900x1000 window, which is an ordinary half of a
     laptop screen — the page ended at 708px and the canvas collapsed to 268 with
     a third of the window left black underneath it. min-height puts the floor
     back: the column fills the viewport and the canvas, being the only flexible
     row, takes the slack. It can still grow past that and scroll, which is what
     `height:auto` was for on a genuinely short screen. */
  body{overflow:auto;height:auto;min-height:100dvh}
  .views{position:static;min-height:0}
  .view{position:static;flex-direction:column;min-height:calc(100dvh - var(--head))}
  .canvas{overflow:visible}
  .console{max-height:none;padding:13px 16px 15px}
  /* The gallery grid crops to squares here, which is the one place this app
     deliberately trades information for density — and the trade is right
     because the screen is the constraint rather than the design. Desktop keeps
     uncropped thumbnails because 320px of column can afford a 4:3 that letter-
     boxes; a phone cannot, and a ragged grid of mixed aspects on a 390px screen
     is a column of two-inch pictures with gaps between them. macOS Photos and
     iOS Photos make exactly this split, and for exactly this reason.
     
     Recognition survives the crop: you are scanning for "the one with the ship"
     and the centre of the frame carries that. The whole frame is one tap away
     in the viewer, which is where you look at a picture rather than find it. */
  #gal-grid{grid-template-columns:repeat(auto-fill,minmax(104px,1fr));gap:2px}
  #gal-grid .gal{border:0;border-radius:0;background:none}
  #gal-grid .media{aspect-ratio:1;width:100%;height:auto;object-fit:cover}
  /* The card's chrome goes with it. A timestamp and a menu under every square
     is the furniture this layout exists to remove, and both live in the viewer
     the square opens. */
  #gal-grid .foot{display:none}
  #gal-grid .quick{opacity:0}
  #gal-grid .gal:focus-within .quick{opacity:1}

  /* Stacked, the drawer stops being a column and becomes a strip. Left as a
     wrapping grid it grew downward without limit — at a dozen outputs it was
     several screens of gallery under a canvas you had to scroll back up to,
     which inverts what the drawer is for: the work you made is meant to sit
     beside the work you are making, not bury it.

     So it is capped and the row scrolls sideways instead. 192px is about a
     104px thumbnail once the head and the card foot are paid for, which is
     enough to recognise a shot by and not enough to compete with the canvas.
     Horizontal scrolling is the right axis here for the same reason vertical
     is the cheap one above: whichever direction the container is short in is
     the direction its contents must not grow. */
  /* A height, not a max-height. Everything below this distributes the track
     with flex, and flex has nothing to distribute unless something in the chain
     is definite — with `max-height` the drawer sized from content, the content
     sized from flex, and the whole strip resolved to zero. The closed state is
     spelled out beside it because `height` would otherwise keep the bar on
     screen when the gallery is shut. */
  .drawer{flex:none;border-left:0;border-top:1px solid rgba(255,255,255,.07);
    height:192px;overflow:hidden;display:flex;flex-direction:column}
  .studio.nodrawer .drawer{height:0;border-top:0}
  .drawer-in{width:auto;padding:10px 14px;flex:1;min-height:0;
    display:flex;flex-direction:column}
  /* grid-template-columns is reset rather than left to be overridden: .grid is
     still display:grid at this point in the cascade, and a stale template on a
     flex container is the kind of leftover that only shows up at one width. */
  .drawer .grid{display:flex;grid-template-columns:none;gap:10px;
    flex:1;min-height:0;overflow-x:auto;overflow-y:hidden}
  /* The card sizes from its height, not the container's width — the media
     keeps 4:3 and the width follows, so a strip of mixed aspects still reads
     as one row of even-height thumbnails. */
  /* A fixed card width, and the media contains inside it. Letting the width
     follow each image's own aspect was the first attempt and it does not work:
     an <img> flex item in a column takes its width from the intrinsic size, so
     the row came out at 1346px-wide cards instead of thumbnails. The whole
     frame still shows — object-fit stays `contain`, so a portrait letterboxes
     rather than being cropped to fit a tidy row, which is the same trade the
     gallery grid already refuses to make. */
  /* The media carries an explicit height rather than stretching to the row.
     height:100% on the card does not resolve here — the grid's own height comes
     from `flex:1`, which is definite to the layout engine and not to a
     percentage — so the cards sized from intrinsic media instead and a portrait
     came out 299px tall inside a 192px drawer. 100px is what the cap leaves
     once the head, the padding and the card foot are paid for, and stating it
     is what makes every thumbnail the same height. */
  /* The card stretches to the track and the media takes what the foot does not
     want. Hard-coding a media height was the previous attempt and it is the
     wrong shape of answer: the track's height comes from a chain of paddings
     and a head, so any number written here is right until one of them changes
     and silently slices the foot off afterwards. `flex:1 1 0` rather than the
     `1 1 auto` that failed before — an auto basis is the content size, which
     for an <img> is its intrinsic height, which is exactly what needed
     overriding. */
  .drawer .grid>.gal{flex:0 0 158px;min-height:0;display:flex;flex-direction:column}
  .drawer .grid>.gal .media{flex:1 1 0;min-height:0;height:auto;width:100%;
    aspect-ratio:auto;object-fit:contain}
  /* The foot pays for itself twice over at this size and has to get cheaper.
     At the desktop padding the card came to 143px inside a 134px content box,
     so every timestamp in the strip was sliced through the middle — the kind of
     overflow that looks like a font problem and is really an arithmetic one.
     The kind glyph and the menu are what the strip needs (a still and a clip are
     otherwise the same black rectangle); the timestamp is the part that can go,
     since the row is already newest-first. */
  .drawer .grid>.gal .foot{flex:none;padding:2px 4px 2px 6px;min-height:0;gap:4px}
  .drawer .grid>.gal .foot .kind{width:16px;height:16px;padding:0}
  .drawer .grid>.gal .foot .when{display:none}
  .drawer .grid>.gal .foot .more{width:20px;height:20px}
  .grid2{grid-template-columns:1fr}
}
</style></head><body>

<header class="top">
  <!-- The wordmark is the way home, which is the only reason Generate needs no
       nav item: you are already there, or you are one click from it. -->
  <button class="brand" id="go-home">Visionary</button>
  <!-- Train's two halves. In Generate this slot is empty — Generate's own
       switch lives down in the composer, beside the prompt it applies to. -->
  <span class="grow"></span>
  <button class="door" id="door"></button>
  <span class="sep"></span>
  <!-- Off at load, matching the `nodrawer` the studio opens with: the drawer
       is raw material for the thing you are making, and on a page you have
       just opened there is nothing being made yet — so the canvas gets the
       320px until you ask for the gallery back. -->
  <button class="ico" id="t-drawer" title="Recent work"></button>
  <button class="ico" id="t-settings" title="Settings"></button>
</header>

<div class="views">

<!-- ============================ GENERATE ============================ -->
<!-- Canvas first, console under it. Image and video are one place: which one
     you get is a property of what you are making, not an address you navigate
     to, so it sits inside the prompt field rather than in the chrome. -->
<div class="view studio nodrawer" id="v-generate">
 <div class="stage">
  <div class="canvas" id="canvas">
    <!-- No copy. An empty frame above a focused prompt field is already the
         whole instruction, and a sentence telling you to type is a sentence
         that will be read on every visit forever to be useful once. -->
    <div id="canvas-empty" class="blank"><div class="glyph" id="canvas-glyph"></div></div>
    <!-- The frame regional mode draws on: the render's own aspect, at the size
         the canvas can give it. It replaces the placeholder rather than sitting
         beside it, because an empty frame and an empty-state glyph are two
         answers to the same question. -->
    <div id="frame" class="frame hide"><div class="thirds"></div></div>
    <!-- Lives here so it exists before anything reparents it; drawRegions()
         moves it onto whichever host is showing. -->
    <div id="region-layer" class="off"></div>
    <!-- Clear, and full-screen. Both belong to the result rather than to the
         composer, so they live on the canvas and not in the strip — and both
         are quiet at rest for the reason `.shot .acts` is: a control on top of
         the picture must not compete with it. -->
    <div id="canvas-acts" class="hide">
      <button class="ico" id="canvas-full" title="Full screen — Space"></button>
      <button class="ico" id="canvas-clear" title="Clear the canvas"></button>
    </div>
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
    <div class="field">
      <textarea id="prompt" rows="1" placeholder="Describe an image…"></textarea>
      <!-- The same field in a different sign. One buffer for both kinds, like
           the prompt above it and for the same reason: what you are steering
           away from does not stop being true when the sentence becomes a clip.
           Hidden outright on a model that reads no negative — H3 is
           guidance-distilled, Krea 2 Turbo runs at CFG 1.0, and on both of
           those a negative prompt is a promise the sampler will not keep. -->
      <textarea id="neg" rows="1" class="hide"
                placeholder="Negative prompt — what to steer away from"></textarea>
      <button type="button" id="neg-toggle" class="neg-t hide"
              title="Which prompt you are writing. Click to switch; the negative is only read at CFG above 1."></button>
      <div class="bar2">
        <div class="kinds" id="kinds">
          <button data-kind="image" class="on">Image</button>
          <button data-kind="video">Video</button>
        </div>
      </div>
    </div>
    <!-- What you picked off the shot palette. One rail for both kinds, because
         the prompt is shared and a pill is part of the prompt; empty, it is not
         a collapsed row but no element at all, which is the condition for
         anything being allowed to sit above the options strip. -->
    <div class="wrap" id="shot-rail"></div>
    <!-- And what those pills turn into. H3 reads a six-field document, and
         until this line existed the only way to find out what it was going to
         be given was to spend three minutes rendering it. Collapsed it is
         eleven pixels of grey text; the compiled document comes from the same
         route that compiles the real run, so it cannot drift from it. -->
    <div id="shot-peek" class="hide"><button type="button">what the model reads</button></div>
    <!-- Only ever says what is wrong. A line confirming the LoRAs you can
         already read in the prompt above it would be the page telling you
         what you can see; a name that resolves to nothing is the one thing
         the prompt cannot show on its own. -->
    <p class="muted warn" id="lora-note"></p>
    <!-- One row for whichever box is selected, not one row per box. What you
         touch on the canvas decides what this is about, which is why it can
         stay a single 36px line at eight regions where the old stack was
         ~592px. The four coordinates are still here — they are the escape
         hatch, and while you drag they are the readout that teaches what they
         mean. -->
    <div id="region-bar" class="opts hide">
      <!-- The frame in miniature. The old per-region rows needed a 32px picture
           of the coordinates beside each one to be legible at all — that picture
           was the part that worked, and one row per box was the part that did
           not. So there is one map, the same size at eight boxes as at one, and
           it is how you reach a box when the boxes are off the render. -->
      <button id="r-map" class="rmap" title="Which box you are editing. Click one to select it; ← → step."></button>
      <button class="drop mini" id="r-ref" data-lb="Photo"
        title="A photo of this character. Pulls the box toward that likeness during sampling — stacks with the LoRA, and works without one.">
        <img id="r-ref-thumb" class="hide" alt=""><span id="r-ref-hint"></span>
      </button>
      <!-- Direction for one performer, not a second scene description. The
           split is the only rule this feature has and it is worth the two
           sentences: the prompt above is what every performer is standing in,
           this is what this one is doing, and the box already said where. -->
      <div class="opt wide"><input id="r-prompt"
        placeholder="a man in a denim jacket, laughing &lt;lora:name:1.3&gt;"
        title="This performer only — who they are and what they are doing. The scene, the light and the lens go in the prompt above; where they stand is the box. Do not write a position here, it is already said."></div>
      <div class="opt n" data-lb="Strength"><input id="r-strength" inputmode="decimal"
        data-step="0.05" data-bigstep="0.25"
        title="How hard this box's LoRA is applied. The node pack's guidance is 1.3–1.4 for a character. Writes the number in the token."></div>
      <span class="vr"></span>
      <div class="opt n" data-lb="X"><input data-r="x" inputmode="decimal"
        data-step="0.01" data-bigstep="0.1"
        title="Left edge, as a fraction of the width. 0 is the left of the canvas."></div>
      <div class="opt n" data-lb="Y"><input data-r="y" inputmode="decimal"
        data-step="0.01" data-bigstep="0.1"
        title="Top edge, as a fraction of the height. 0 is the top of the canvas."></div>
      <div class="opt n" data-lb="W"><input data-r="width" inputmode="decimal"
        data-step="0.01" data-bigstep="0.1"
        title="How much of the canvas width this box covers, 0 to 1."></div>
      <div class="opt n" data-lb="H"><input data-r="height" inputmode="decimal"
        data-step="0.01" data-bigstep="0.1"
        title="How much of the canvas height this box covers, 0 to 1."></div>
      <!-- Render-scoped, unlike everything to the left of the rule, which is
           about whichever box is selected. They lived in Advanced, which meant
           the two controls that only exist when regions are armed were behind a
           drawer that had nothing else to do with regions. -->
      <span class="vr" id="g-region-vr"></span>
      <button class="opt ib" id="g-arrange" data-ico="arrange"
        title="Distribute the boxes evenly"></button>
      <div class="opt n" id="g-region-base-wrap" data-lb="Global"><input id="g-region-base"
        value="1" inputmode="decimal" data-step="0.05" data-bigstep="0.25"
        title="Multiplies every region's LoRA strength at once. 1 uses the strengths as written."></div>
      <button class="drop mini hide" id="g-drop-scene" data-lb="Scene"
        title="Scene photo. The picture is generated inside it — lighting, perspective and shadows integrate.">
        <img id="g-thumb-scene" class="hide" alt=""><span id="g-hint-scene"></span>
      </button>
      <button class="drop mini hide" id="g-drop-outfit" data-lb="Outfit"
        title="Outfit or object photo. Transferred onto the subjects rather than pasted into the frame.">
        <img id="g-thumb-outfit" class="hide" alt=""><span id="g-hint-outfit"></span>
      </button>
      <span class="actions">
        <button class="opt ib" id="r-del" data-ico="trash"
          title="Remove this box — or select it and press ⌫"></button>
      </span>
    </div>
    <p class="muted hide" id="region-note" style="margin:7px 2px 0"></p>
    <div id="gen-prog" class="hide" style="margin-top:9px"><div class="bar"><i style="width:0%"></i></div><div class="row" style="gap:10px;margin-top:6px"><p class="muted grow" style="margin:0"></p><button class="s" data-cancel>Cancel</button></div></div>
    <div id="vid-prog" class="hide" style="margin-top:9px"><div class="bar"><i style="width:0%"></i></div><div class="row" style="gap:10px;margin-top:6px"><p class="muted grow" style="margin:0"></p><button class="s" data-cancel>Cancel</button></div></div>
    <p class="muted warn" id="gen-note" style="margin:8px 2px 0"></p>
    <p class="muted warn" id="vid-note" style="margin:8px 2px 0"></p>

    <!-- IMAGE -->
    <div id="c-image">
      <div class="opts">
        <div class="opt"><select id="g-model"></select></div>
        <!-- The ratio was never the parameter — the pixels are. So this picker
             and the Width/Height boxes under Advanced are one control: picking
             a ratio writes the boxes, typing in the boxes selects Custom, and
             Custom is the only option that has to spell out its own size. -->
        <!-- 3:4 is here because the swap button under Advanced put it here:
             every other landscape preset had a portrait counterpart to flip
             into and 4:3 did not, so the one ratio the page opens on was the
             one whose flip landed on Custom. -->
        <!-- Aspect and resolution were two controls pretending to be one, and
             the seam was that every preset was 1024-based: picking 16:9 chose a
             shape *and* silently chose ~1 MP, and the only route to the same
             shape at 2K was to work out 2304x1296 by hand in two boxes parked
             at the far end of Advanced. They are one control because there is
             only ever one width and one height — the ratio picks the shape, the
             scale picks how much of it, and the button shows the pair it
             resolved to. Same move as the shot palette: a closed vocabulary
             costs one button, and everything it can say lives behind it. -->
        <button class="opt ib" id="g-size" title="Shape and resolution"></button>
        <!-- The state the popover is a view over. Kept as real form elements
             rather than plain variables so readSize, swapSize, reuse() and the
             arrow-key nudges all keep addressing what they always did — the
             popover writes here and reads back, and is free to not exist. -->
        <div class="hide" id="g-size-state">
          <select id="g-aspect">
            <option value="1024x1024">1:1</option><option value="1152x896" selected>4:3</option>
            <option value="1216x832">3:2</option><option value="1344x768">16:9</option>
            <option value="896x1152">3:4</option>
            <option value="832x1216">2:3</option><option value="768x1344">9:16</option>
            <option value="custom">Custom</option>
          </select>
          <select id="g-scale">
            <option value="1" selected>1K</option>
            <option value="1.5">1.5K</option>
            <option value="2">2K</option>
          </select>
          <input id="g-w" inputmode="numeric"><input id="g-h" inputmode="numeric">
          <button id="g-swap"></button>
        </div>
        <!-- Sampling state. Same arrangement as #g-size-state: the popover is a
             view over these, so fillSelect, the submit body, reuse() and the
             arrow-key nudges are all unchanged by the drawer going away. -->
        <div class="hide" id="g-samp-state">
          <select id="g-sampler"></select>
          <select id="g-scheduler"></select>
          <input id="g-steps" placeholder="auto" inputmode="numeric">
          <input id="g-cfg" placeholder="auto" inputmode="decimal" data-step="0.1" data-bigstep="1">
          <input id="g-shift" placeholder="1.15" inputmode="decimal" data-step="0.05" data-bigstep="0.5">
          <!-- Demoted out of the strip. A seed is *reused off a result* — the
               gesture happens after a render, not before — and a batch count is
               a run parameter like steps, not a thing a take varies by. Both
               stay real elements here so reuse(), the submit body and the
               arrow-key nudges address exactly what they always did. -->
          <input id="g-seed" placeholder="random" inputmode="numeric">
          <select id="g-n"><option>1</option><option>2</option><option>3</option><option>4</option></select>
        </div>
        <span class="vr"></span>
        <!-- A picker, not a row: it writes a <lora:name:1> into the prompt at
             the caret. The button is here for discovery — you cannot type a
             syntax you have never seen — and after that the prompt is the
             stack, so a fifth LoRA costs the canvas nothing. -->
        <button class="s" id="add-lora" style="height:36px;padding:0 13px">+ LoRA</button>
        <!-- One icon for the whole vocabulary. It is next to + LoRA because
             both write into the prompt rather than beside it — the difference
             is only that this one writes words the model was trained on and
             you would otherwise have to guess. -->
        <button class="opt ib" id="g-shot" data-ico="shot" data-lb="Shot"
          title="Framing, angle, light and tone, as words this model reads."></button>
        <!-- Out of Advanced, which is collapsed by default and was therefore
             hiding the feature this backend was swapped for. It belongs beside
             + LoRA because both answer "who is in this picture"; the difference
             is only whether it matters where they stand. -->
        <!-- Named, not just iconed. Moving it out of Advanced fixed where it
             was; it did not fix that a glyph cannot announce a capability
             nobody knows the app has. `+ LoRA` sits immediately to the left
             with a word on it, which made the asymmetry the thing you noticed
             about this button rather than the feature behind it. -->
        <button class="opt ib" id="g-regional" data-ico="regions" data-lb="Regions"
          title="Place each character in their own box on the canvas — one LoRA, or one photo, per box."></button>
        <span class="actions">
          <span class="muted" id="gen-model-line"></span>
          <!-- "Advanced" was a drawer, which is a name for where something is
               rather than what it does — and behind it sat five controls that
               are not advanced, they are just rarely changed. A drawer also
               charges the console a whole row the moment you open it to read
               one number. This is the shot palette's shape again: one button
               showing the value it resolved to, everything behind it. -->
          <button class="opt ib" id="g-sampling" title="Sampler, steps and guidance"></button>
          <button class="ico shot-back hide" id="g-last" title="Last generation"></button>
          <button class="b" id="go-gen">Generate</button>
        </span>
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
        <!-- Shape and how much of it, as one control — the same pair `g-size`
             already collapsed on the image side, which the video side never
             received. The scale row here is the model's own tiers, which differ
             per checkpoint. -->
        <button class="opt ib" id="v-size" title="Shape and resolution"></button>
        <div class="hide" id="v-size-state">
          <select id="v-aspect">
            <option value="21:9">21:9</option><option value="16:9" selected>16:9</option>
            <option value="4:3">4:3</option><option value="1:1">1:1</option>
            <option value="3:4">3:4</option><option value="9:16">9:16</option>
          </select>
          <select id="v-tier"></select>
        </div>
        <div class="opt"><select id="v-seconds"></select></div>
        <!-- Wan only, and the same picker the image side uses. The one thing
             the A14B pair forces — which expert a LoRA patches — rides in the
             token as a third field, and is read off the filename when the
             matched `high`/`low` pair names it. -->
        <button class="s hide" id="v-add-lora" style="height:36px;padding:0 13px">+ LoRA</button>
        <!-- Sampling state, the same arrangement as #g-samp-state. The three
             `-wrap` divs stay because syncVideoModel toggles them per model —
             H3 reads no CFG and only the A14B pair has a handover to place —
             and the popover reads those same classes to decide what to draw. -->
        <div class="hide" id="v-samp-state">
          <select id="v-sampler"></select>
          <select id="v-scheduler"></select>
          <input id="v-steps" inputmode="numeric">
          <span id="v-cfg-wrap" class="hide"><input id="v-cfg" inputmode="decimal"
            data-step="0.1" data-bigstep="1"></span>
          <span id="v-shift-wrap" class="hide"><input id="v-shift" inputmode="decimal"
            data-step="0.1" data-bigstep="1"></span>
          <span id="v-switch-wrap" class="hide"><input id="v-switch" placeholder="auto"
            inputmode="numeric"></span>
          <input id="v-seed" placeholder="random" inputmode="numeric">
        </div>
        <button class="opt ib" id="v-shot" data-ico="shot" data-lb="Shot"
          title="Shot size, camera move, light, action and sound — the fields H3 reads."></button>
        <span class="actions">
          <span class="muted" id="v-model-line"></span>
          <button class="opt ib" id="v-sampling" title="Sampler, steps and guidance"></button>
          <button class="ico shot-back hide" id="v-last" title="Last generation"></button>
          <button class="b" id="go-vid">Generate</button>
        </span>
      </div>

      <!-- Every picture this model can be given, in one row, because keyframes
           and references are the same decision made two ways and choosing one
           excludes the other — they load different transformers.

           They used to be two rows: keyframes among the numeric controls at
           the far right of the strip above, references down here. Two pairs of
           unlabelled 36px dashed tiles, forty-five pixels and one row apart,
           telling each other apart by tooltip. What that cost was not
           aesthetic — the keyframe tiles were never found at all, and dropping
           photos into the reference tray looked like filling keyframe slots
           that kept growing, which is exactly what the reference tray does and
           exactly what a keyframe pair must never look like. Side by side with
           a rule between them, the tray that grows and the two fixed slots are
           told apart by shape, which is the thing a tooltip could not do.

           The chips carry their own <Picture n> labels, which is the part the
           prompt actually refers to and the only part worth spelling out. -->
      <div class="opts" id="v-src-sec">
        <button class="drop mini" id="v-drop-first" data-lb="First frame"
                title="The clip starts on this image. Drop or click; click again to clear.">
          <img id="v-thumb-first" class="hide" alt=""><span id="v-hint-first"></span>
        </button>
        <button class="drop mini hide" id="v-drop-last" data-lb="Last frame"
                title="The clip ends on this image. Drop or click; click again to clear.">
          <img id="v-thumb-last" class="hide" alt=""><span id="v-hint-last"></span>
        </button>
        <span class="vr" id="v-src-vr"></span>
        <span class="wrap" id="v-refs"></span>
        <!-- Named for the token they produce, not for what they take: what you
             attach here is what the prompt then calls <Picture 1> / <Video 1>,
             and the chips are already lettered P1/V1 to match. -->
        <button class="drop mini" id="v-add-ref" data-lb="Picture"
                title="Add an image reference — the subject, redrawn in a new shot. The prompt refers to it as &lt;Picture 1&gt;."></button>
        <button class="drop mini" id="v-add-vid" data-lb="Video"
                title="Add a video reference. The prompt refers to it as &lt;Video 1&gt;."></button>
        <!-- The one control in this row that changes what a take costs rather
             than what it contains, and the two options are not close: reference
             tokens ride through every sampling step, so this is a per-step
             price. Both are bounded — see H3_REF_MAX_SIDE — which is the only
             reason "max detail" is offered at all. -->
        <div class="opt" id="v-ref-size-wrap"><select id="v-ref-size"
          title="How much of each reference the model reads. &quot;match canvas&quot; scales every picture to the clip's own pixel area; &quot;max detail&quot; hands over as much of it as the run allows — 1536px on the long side — and buys likeness at several times the sampling time.">
          <option value="match">match canvas</option><option value="max">max detail</option>
        </select></div>
        <span class="muted" id="v-ref-max" hidden>9</span><span class="muted" id="v-vid-max" hidden>3</span>
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
      <!-- Words rather than the × the cards carry: a glyph that means "this one"
           cannot also mean "every one of these", and the difference is the whole
           point of the button. The label tracks the filter so it names what is
           actually about to go. -->
      <button class="pill danger" id="gal-purge">Delete all</button>
    </div>
    <div id="gal-grid" class="grid"></div>
    <p class="muted" id="gal-empty"></p>
  </div>
</div>

<!-- ============================== TRAIN ============================== -->
<!-- ============================== TRAIN ============================== -->
<!-- Same shape as Generate, for the same reason: subject in the middle, your
     library on the right, the console pinned along the bottom. The console is
     what makes this layout work here — a set of eighty images is a long scroll,
     and the controls that start the run must not be somewhere down inside it. -->
<div class="view studio hide" id="v-train">
 <div class="stage">
  <div class="canvas" id="t-canvas">
    <div id="train-err"></div>
    <div id="ds-edit-err"></div>

    <!-- There are two ways to get a set, and they are not equal: mostly you
         drag one in. So the drop target is the screen when nothing is chosen,
         rather than a bar you find after creating something to put it in. -->
    <div class="drop hero" id="drop">
      <input type="file" id="files" multiple accept="image/*,.zip,.txt" class="hide">
      <!-- No name field. Naming was the toll on the one action this screen
           exists for, and it is charged for a decision — is this set worth
           keeping — that you cannot make before seeing the images. It moved to
           Save, on the sheet, where the images are. -->
      <div class="drop-face">
        <div class="glyph" id="drop-glyph"></div>
        <b id="drop-title">Drop images or a .zip</b>
        <div class="muted" id="drop-sub" style="margin-top:7px"></div>
      </div>
      <div id="up-prog" class="hide" style="margin-top:18px;width:min(420px,70%)">
        <div class="bar"><i style="width:0%"></i></div>
      </div>
    </div>

    <!-- A set is chosen: its contact sheet, which is the thing that scrolls. -->
    <!-- data-drop, because the whole sheet has always accepted a file drop and
         nothing on it said so — the browser's own default is to drop the file
         into whichever caption box happens to be under the cursor, which is
         what this overrides. -->
    <div id="ds-sheet" class="hide" data-drop="Add to this set">
      <div class="opts sheet-bar">
        <!-- A saved set states its name; a draft is still asking for one, so
             for a draft the name field *is* the title rather than sitting
             next to a heading that repeats it. -->
        <b id="ds-title" style="font-size:14px"></b>
        <div class="opt mid hide" id="ds-name-wrap" data-ico="tag">
          <input id="ds-save-name" placeholder="name to save it" spellcheck="false">
        </div>
        <button class="s hide" id="ds-save" title="Keep this set in datasets/">Save</button>
        <span class="muted" id="ds-count"></span>
        <span class="vr"></span>
        <button class="s" id="f-all" title="Show every image">All</button>
        <button class="s" id="f-uncap" title="Only images with no caption">Uncaptioned</button>
        <button class="s" id="f-notrig" title="Only captions missing the trigger word">No trigger</button>
        <span class="vr"></span>
        <button class="s" id="dens-down" title="Smaller tiles">−</button>
        <button class="s" id="dens-up" title="Larger tiles">+</button>
        <span class="actions">
          <span class="muted" id="ins-summary"></span>
          <!-- The sliders glyph is already spoken for: #toggle-adv and
               #t-toggle-adv both wear it for "more settings". A rebus that
               stands for two unrelated things is not naming either of them,
               and what is behind this one — write every caption with a vision
               model, then read back what the set is actually teaching — is
               not a settings drawer. So it says the word. -->
          <button class="opt ib" id="ins-toggle" data-ico="sliders" data-lb="Captions"
                  title="Write captions with a vision model, and see what this set is actually teaching."></button>
          <button class="s" id="ds-add">+ Images</button>
        </span>
      </div>

      <!-- Captioning is an action on these images, so it lives with them
           rather than in the training console below. -->
      <div id="ins-panel" class="hide adv" style="margin:0 0 14px">
        <div class="opts" style="margin-top:0">
          <div class="opt mid" data-ico="tag"><input id="ds-trig" placeholder="trigger word" spellcheck="false"></div>
          <button class="s" id="do-prepend" title="Put the trigger word at the front of every caption that lacks it">Fix</button>
          <span class="vr"></span>
          <!-- Three menus, no labels: "Character", "Medium" and "Qwen3-VL 8B"
               each name themselves, and none of them could be mistaken for
               another. What a preset *does* is the one thing the word cannot
               carry, so it goes in the note below rather than into a tooltip
               nobody hovers. -->
          <div class="opt"><select id="cap-preset"></select></div>
          <div class="opt"><select id="cap-len"><option value="short">Short</option><option value="medium" selected>Medium</option><option value="long">Long</option></select></div>
          <span class="vr"></span>
          <div class="opt"><select id="cap-model"></select></div>
          <label class="row" style="gap:7px;margin:0;color:#ddd;font-size:13px"><input type="checkbox" id="cap-over" style="width:auto"> Replace existing</label>
          <span class="actions"><button class="s" id="do-caption">Caption</button></span>
        </div>
        <p class="muted" id="cap-note" style="margin:8px 2px 0"></p>
        <div id="cap-prog" class="hide" style="margin-top:9px"><div class="bar"><i style="width:0%"></i></div><p class="muted" style="margin-top:6px"></p></div>
        <div id="ins-body" style="margin-top:13px"></div>
      </div>

      <div class="tiles" id="tiles"></div>
    </div>
  </div>

  <!-- The training console. Pinned, so eighty images cannot push it off. -->
  <div class="console" id="t-console">
    <div class="opts" style="margin-top:0">
      <div class="opt wide" data-ico="tag"><input id="lname" placeholder="LoRA name" spellcheck="false"></div>
      <div class="opt mid" data-ico="trigger"><input id="ltrig" placeholder="trigger word" spellcheck="false"></div>
      <span class="vr"></span>
      <!-- Named, not iconed. These are the three dials that decide whether a run
           is worth its hours, and a glyph beside a bare "32" cannot say which
           of rank, alpha, epochs or batch size you are looking at. The tooltip
           is spent on what the number does instead of repeating the label. -->
      <div class="opt n" data-lb="Rank"><input id="a-dim" type="number" value="32"
        title="Network dimension — how much the LoRA can learn. Higher fits more and overfits sooner."></div>
      <div class="opt n" data-lb="Epochs"><input id="a-epochs" type="number" value="30"
        title="Passes over the set. A checkpoint is saved each one, so this is also how many you get to choose between."></div>
      <!-- Not `.n`: the narrow box fits three digits, and 0.0001 is six. A
           learning rate clipped to "0.000" is worse than no field at all. -->
      <div class="opt" data-lb="Learning rate"><input id="a-lr" type="number" step="0.00001" value="0.0001"
        title="Step size per update. 1e-4 is the usual starting point for a rank-32 LoRA."></div>
      <span class="actions">
        <span class="muted" id="train-hint"></span>
        <button class="opt ib" id="t-toggle-adv" data-ico="sliders" title="Advanced"></button>
        <button class="b" id="go-train" disabled>Start training</button>
      </span>
    </div>
    <div id="train-adv" class="hide adv">
      <div class="opts" style="margin-top:0">
        <div class="opt n" data-lb="Alpha"><input id="a-alpha" type="number" value="32"
          title="Scales the LoRA's contribution: the effective strength is alpha ÷ rank. Equal to rank means no scaling."></div>
        <div class="opt n" data-lb="Resolution"><input id="a-res" type="number" step="64" value="1024"
          title="Training pixels per side. Images are bucketed to it; higher costs VRAM quadratically."></div>
        <div class="opt n" data-lb="Repeats"><input id="a-rep" type="number" value="1"
          title="How many times each image is seen per epoch. Raise it for a set too small to fill an epoch."></div>
        <div class="opt n" data-lb="Batch size"><input id="a-bs" type="number" value="1"
          title="Images per update. Higher is steadier and needs more VRAM."></div>
        <div class="opt n" data-lb="Seed"><input id="a-seed" type="number" value="42"
          title="Fixes shuffling and noise so two runs differing in one dial are actually comparable."></div>
        <span class="actions"><span class="muted">Krea 2 RAW · bf16</span></span>
      </div>
    </div>
    <div id="step-run" class="hide" style="margin-top:11px">
      <div class="row"><b id="run-phase" class="grow">Starting…</b><span class="muted" id="run-pct"></span></div>
      <div class="bar"><i id="run-bar" style="width:0%"></i></div>
      <div class="row" style="margin-top:9px">
        <span class="muted grow" id="run-meta"></span>
        <button class="s" id="do-stop">Stop &amp; keep checkpoints</button>
      </div>
      <div id="run-done" class="hide" style="margin-top:11px"></div>
    </div>
  </div>
 </div>

 <!-- Every set you already have, always open. The second way in. -->
 <aside class="drawer" id="ds-drawer">
   <div class="drawer-in">
     <div class="drawer-head">
       <span class="grow"></span>
       <button class="s" id="ds-fresh">+ New set</button>
     </div>
     <div id="ds-err"></div>
     <div id="ds-list" class="grid"></div>
     <p class="muted" id="ds-empty" style="margin-top:6px"></p>
   </div>
 </aside>
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
    <!-- The token, and only the token. "Download missing" used to sit in this
         row, which put the one button that pulls the entire catalogue — every
         family, including the ones this install will never run — next to a
         password field it has nothing to do with. Each family downloads itself
         now, so the button that is almost always the wrong scope is gone and
         the card is about the thing it is labelled with. -->
    <!-- Where the cards you fill in once live, which is what a GPU choice is:
         it is set per session and confirms a cold start when it changes, so it
         was 71px of composer for a decision no take varies by. -->
    <div class="card">
      <label>GPU</label>
      <div class="row" style="gap:10px">
        <div class="opt" data-lb="Images"><select id="g-gpu"></select></div>
        <div class="opt" data-lb="Video"><select id="v-gpu"></select></div>
      </div>
      <p class="muted" style="margin:9px 2px 0">Changing a card costs one cold start while the model loads. Runs after it are warm.</p>
    </div>

    <div class="card">
      <label>HuggingFace token</label>
      <div class="row">
        <input id="tok" type="password" class="grow" placeholder="hf_…" autocomplete="off">
        <button class="s" id="tok-save">Save</button>
      </div>
      <p class="muted" style="margin-top:8px">
        Needed for Krea 2 RAW and Turbo, which are gated. Accept the licence at
        huggingface.co/krea/Krea-2-Raw with the same account. <span id="tok-state"></span>
      </p>
    </div>

    <!-- The other way weights arrive. Most LoRAs worth having were never
         published to HuggingFace — they are a link someone sent you — and
         without this the only route onto the volume was to have trained it
         here. Same card shape as the token above it, deliberately: this is
         another place weights come from, not another kind of thing. -->
    <div class="card">
      <label>Google Drive</label>
      <div class="row">
        <input id="gd-url" class="grow" placeholder="Drive link or file id" autocomplete="off" spellcheck="false">
        <input id="gd-folder" placeholder="folder (optional)" autocomplete="off" spellcheck="false"
          style="width:158px;flex:none" title="Group the files under loras/{name}/ — for a matched pair that belongs together. Leave blank to drop them in loose.">
        <button class="b" id="gd-go" style="padding:9px 16px;font-size:13px">Download</button>
      </div>
      <p class="muted" style="margin-top:8px">
        Lands in <code>loras/</code>, ready to name in a prompt. Only
        <code>.safetensors</code> is kept — a folder's preview images and readme
        are left behind. The link has to be shared with anyone who has it.
        <span id="gd-state"></span>
      </p>
      <!-- No progress bar, unlike the card above. Drive does not say how big a
           file is before it sends it, so a bar here could only sit at zero for
           the length of the transfer — which is what "stuck" looks like. The
           byte count and the rate move, and moving is the whole job of this
           element. -->
      <div id="gd-prog" class="hide"><div class="row" style="gap:10px"><p class="muted grow" style="margin:0"></p><button class="s" data-cancel>Cancel</button></div></div>
    </div>

    <!-- What the two cards above write into, and what the trainer writes into,
         listed once. Until now the only view of loras/ was the `+ LoRA` picker,
         which offers files to type and has nothing to say about what they cost
         or how to be rid of one — so a LoRA that turned out badly stayed on the
         volume forever, and the only way off it was the Modal CLI. Under the
         gear rather than beside the picker for the same reason the weights are:
         this is plumbing, decided once, and not a thing to trip over while
         writing a prompt. -->
    <div class="card">
      <div class="row" style="align-items:baseline;margin-bottom:4px">
        <label class="grow" style="margin:0">LoRAs</label>
        <span class="muted" id="lora-total"></span>
      </div>
      <div id="lora-err" style="margin-top:10px"></div>
      <div id="lora-list"></div>
    </div>

    <div id="models"></div>
  </div>
</div>

<script>
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
// A failed request comes back as {error}, never as a thrown parse error. A 500
// from Modal is a plain-text traceback, so r.json() rejected inside whichever
// caller happened to make the request and took the rest of that function with
// it — loadState() died on `s.models.filter` and left the composer looking like
// a deployment with no weights on it.
const api=async(p,o)=>{
  let r;
  try{ r=await fetch(p,o) }catch(e){ return {error:String(e.message||e)} }
  const body=await r.text();
  try{ return JSON.parse(body) }
  catch{ return {error:`${r.status} ${r.statusText||''} ${body.slice(0,400)}`.trim()} }
};
const post=(p,b)=>api(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});

// setInterval for a poll, with the one thing setInterval cannot do: skip a tick
// while the last one is still out. Drop-in — same argument order, same id, so
// `clearInterval(t)` inside the body still ends it.
//
// Every poll here awaits a request, and setInterval fires on a clock rather
// than on a reply. /api/status reads a *network* Dict, so at 400ms a slow reply
// does not delay the next tick, it overlaps it, and three things follow.
//
// Responses land out of order, so the bar is painted by whichever reply arrives
// last rather than whichever is newest: step 14 lands, then step 12 lands on
// top of it and the bar walks backwards. That is the client half of "the
// progress bar goes nuts"; `_publish` is the server half.
//
// And in-flight polls hold connections. A browser gives one origin about six,
// and everything on this page comes off that origin — the gallery's covers, the
// stills on the canvas, and a <video> that re-requests byte ranges for as long
// as it plays. A pile of polls starves them: the clip stutters mid-playback and
// the grid comes back half-painted, neither of which looks like a poll loop,
// which is why this went unfound for so long.
const everyMs=(fn,ms)=>{
  let busy=false;
  const id=setInterval(async()=>{
    if(busy) return;
    busy=true;
    try{ await fn() } finally{ busy=false }
  },ms);
  return id;
};
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
  // Strip icons. The numeric fields that used to be in here are labelled now —
  // see the .opt rules — so what is left is the controls that have no value to
  // show at all: a disclosure toggle and two text fields whose placeholder
  // disappears the moment you type into them.
  sliders:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M4 7h10M18 7h2M4 17h4M12 17h8"/><circle cx="16" cy="7" r="2"/><circle cx="10" cy="17" r="2"/></svg>',
  refresh:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M20 11a8 8 0 1 0-.7 4.3"/><path d="M20 4.5V11h-6.5"/></svg>',
  // Two arrows crossing, not one double-headed one: the width and the height
  // trade places, and a single arrow with two heads reads as a dimension being
  // measured rather than two values being exchanged.
  swap:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 8.5h13"/><path d="M13.5 5l3.5 3.5-3.5 3.5"/><path d="M20 15.5H7"/><path d="M10.5 12L7 15.5 10.5 19"/></svg>',
  expand:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 4.5H4.5v5"/><path d="M14.5 19.5h5v-5"/><path d="M4.5 4.5l6 6"/><path d="M19.5 19.5l-6-6"/></svg>',
  first:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><rect x="3.5" y="5.5" width="17" height="13" rx="2.5"/><rect x="3.5" y="5.5" width="5" height="13" rx="2.5" fill="currentColor" stroke="none" opacity=".85"/></svg>',
  last:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><rect x="3.5" y="5.5" width="17" height="13" rx="2.5"/><rect x="15.5" y="5.5" width="5" height="13" rx="2.5" fill="currentColor" stroke="none" opacity=".85"/></svg>',
  tag:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"><path d="M3.5 11V4.5H10L20.5 15 15 20.5 3.5 11z"/><circle cx="7.5" cy="8" r="1.3" fill="currentColor" stroke="none"/></svg>',
  trigger:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><path d="M4 12h6"/><path d="M14 12h6"/><circle cx="12" cy="12" r="2.2"/></svg>',
  upload:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16V4"/><path d="M7.5 8.5L12 4l4.5 4.5"/><path d="M4 15v3.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V15"/></svg>',
  film:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><rect x="3.5" y="5.5" width="17" height="13" rx="2.5"/><path d="M8 5.5v13M16 5.5v13"/></svg>',
  // A room with a horizon, and a hanging garment. The two reference plates do
  // different things to the same render — one becomes the world, the other
  // becomes what the subjects are wearing — so drawing them alike would make
  // the pair of tiles a coin toss.
  scene:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><rect x="3.5" y="5.5" width="17" height="13" rx="2.5"/><path d="M3.5 15.5l4.6-4a1.8 1.8 0 0 1 2.4 0l5.4 4.7"/><path d="M14.6 13.2l1.6-1.4a1.8 1.8 0 0 1 2.4 0l1.9 1.7"/><circle cx="8.4" cy="9.6" r="1.3" fill="currentColor" stroke="none"/></svg>',
  outfit:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 4.5L5 7.2l1.4 3.2 1.8-.8V19.5h7.6V9.6l1.8.8L19 7.2l-4.5-2.7"/><path d="M9.5 4.5a2.5 2.5 0 0 0 5 0"/></svg>',
  // The feature, drawn: a frame with two boxes standing in it. Nothing more
  // abstract survived the test of being recognisable at 16px.
  regions:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><rect x="3" y="5" width="18" height="14" rx="2.5" opacity=".45"/><rect x="6" y="8.5" width="5" height="8" rx="1.4" fill="currentColor" stroke="none"/><rect x="13" y="8.5" width="5" height="8" rx="1.4" fill="currentColor" stroke="none" opacity=".55"/></svg>',
  // A frame with a subject in it and a move around it, which is the whole of
  // what is behind this button. Drawn as a viewfinder rather than a
  // clapperboard: a clapperboard means "video", and this button is on the
  // image side too.
  shot:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="5" width="18" height="14" rx="2.5" opacity=".45"/><ellipse cx="10" cy="12.5" rx="2.1" ry="3" fill="currentColor" stroke="none"/><path d="M14.6 9.4a4.4 4.4 0 0 1 0 6.2"/><path d="M17.4 7.6a7.2 7.2 0 0 1 0 9.8"/></svg>',
  arrange:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><rect x="3.5" y="6" width="5" height="12" rx="1.4"/><rect x="9.5" y="6" width="5" height="12" rx="1.4"/><rect x="15.5" y="6" width="5" height="12" rx="1.4"/></svg>',
  trash:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4.5 7h15"/><path d="M9.5 7V5.2A1.2 1.2 0 0 1 10.7 4h2.6a1.2 1.2 0 0 1 1.2 1.2V7"/><path d="M6.5 7l.8 11.3A1.7 1.7 0 0 0 9 20h6a1.7 1.7 0 0 0 1.7-1.7L17.5 7"/></svg>',
};
// Every control that asked for an icon instead of a label gets it here, so the
// markup names the idea ("dice") and only this table knows the path data.
$$('[data-ico]').forEach(el=>el.insertAdjacentHTML('afterbegin',ICON[el.dataset.ico]));
// And every control that asked for the word. Same mechanism as data-ico, and
// *nearly* exclusive with it: a pill carries one or the other, because a
// labelled icon is usually two ways to say one thing.
//
// Two buttons carry both, deliberately. The rule holds when the glyph draws
// something you already have a word for — nobody needs "Delete" written under
// a bin. It fails when the glyph is the only announcement of a capability the
// user does not know exists, which is the same distinction that put names back
// on the hyperparameters: an icon is a rebus for a word you already know, so
// it cannot teach you the word. #g-regional draws two boxes in a frame very
// well and still cannot say "this app does regional multi-character LoRA",
// and it sits next to `+ LoRA`, which spells itself out. #ins-toggle is the
// clincher: it wears the same sliders glyph as #toggle-adv and #t-toggle-adv,
// so that one glyph is standing for three unrelated panels and is therefore
// naming none of them.
//
// Factored out of the one-shot pass rather than left inline, because region
// rows are built long after load: a label injected only at startup is a label
// every dynamically-added pill silently goes without.
function label(root){
  [...root.querySelectorAll('[data-lb]')].forEach(el=>{
    if(!el.querySelector(':scope>.lead'))
      el.insertAdjacentHTML('afterbegin','<span class="lead">'+esc(el.dataset.lb)+'</span>');
  });
}
// Before label(), not after: these two write innerHTML rather than appending,
// so running them second silently deleted the .lead label() had just inserted
// and left the two add-buttons the only unnamed tiles in the row.
$('#v-add-ref').innerHTML='<span>'+ICON.photo+'</span>';
$('#v-add-vid').innerHTML='<span>'+ICON.film+'</span>';
label(document);
$('#drop-glyph').innerHTML=ICON.upload;
$('#gal-back').innerHTML=ICON.back;
$('#gal-refresh').innerHTML=ICON.refresh;
$('#gal-expand').innerHTML=ICON.expand;
$('#canvas-full').innerHTML=ICON.expand;
$('#canvas-clear').innerHTML=ICON.close;
$('#t-settings').innerHTML=ICON.gear;
$('#t-drawer').innerHTML=ICON.panel;
// The Camera app's move: the way back to what you made is a picture of what you
// made. An icon says "gallery", which you have to already want; a thumbnail says
// "this is the last thing you rendered", which is what you are reaching for.
// Falls back to the glyph with nothing on the volume, because a hole where a
// picture goes is worse than a symbol.
function paintLastShot(items){
  const it=(items||[])[0];
  // Beside Generate, not in the header. The Camera app puts the last frame next
  // to the shutter because those two are one loop: you press one and then want
  // the other. Putting it in the top bar meant pressing Generate at the bottom
  // of a 1194px screen and then reaching to the opposite corner to look at the
  // result — the hand crossing the whole device and coming back, for two things
  // that belong within a thumb's travel of each other.
  ['#g-last','#v-last'].forEach(sel=>{
    const b=$(sel); if(!b) return;
    b.classList.toggle('hide',!it);
    if(!it) return;
    const src=`/api/file/${it.job_id}/${it.files[0]}`;
    b.innerHTML = it.kind==='video'
      ? `<video src="${src}#t=0.04" preload="metadata" muted playsinline></video>`
      : `<img src="${src}" alt="">`;
  });
}
$$('#g-last,#v-last').forEach(b=>b.onclick=()=>{
  if((galItems||[]).length) viewAt(galItems,0);
});
$('#settings-x').innerHTML=ICON.close;

// ==================== SHELL ====================
// Generate is the page, not a destination — so it has no nav item, and the
// wordmark is how you get back to it. Train is the one door, on the right.
let mode='generate', kind='image', trainPct=null;

function setMode(m){
  mode=m;
  $('#v-generate').classList.toggle('hide',m!=='generate');
  $('#v-train').classList.toggle('hide',m!=='train');
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
  syncPromptHint();
  // The rail survives the switch along with the prompt, so what changes is
  // which of its pills the thing on the other side of the switch can read.
  drawShotRail();
  syncNeg();
  autoGrow($('.field.on-neg')?$('#neg'):$('#prompt'));
  syncCanvasView();
}
$$('#kinds button').forEach(b=>{
  b.insertAdjacentHTML('afterbegin',b.dataset.kind==='image'?ICON.photo:ICON.play);
  b.onclick=()=>setKind(b.dataset.kind);
});

// Enter generates. Which button that is belongs to the kind you are in, not to
// the key, so this is looked up rather than bound twice — and a disabled button
// is a model that is not downloaded, which Enter must not paper over.
function fireGenerate(){
  const b=$(kind==='image'?'#go-gen':'#go-vid');
  if(!b.disabled) b.click();
  return !b.disabled;
}
// Shift+Enter keeps the newline, because prompts here are prose and paragraphs
// in them are real. ⌘/Ctrl+Enter works too: it is what the muscle expects from
// every other box that submits, and costs nothing to honour.
// isComposing, because an IME's Enter is committing a character, not submitting.
$('#prompt').addEventListener('keydown',e=>{
  if(e.key!=='Enter'||e.isComposing||e.shiftKey||e.altKey) return;
  e.preventDefault();
  fireGenerate();
});
// And from any single-line box in the composer — you have just typed a seed or
// a step count, and reaching for the mouse to commit it is the wrong ending.
// Not the textareas: a negative prompt is prose with the same claim on Enter
// the positive one has.
$$('#c-image,#c-video,#region-bar').forEach(sec=>sec.addEventListener('keydown',e=>{
  if(e.key!=='Enter'||e.isComposing||!e.target.matches('input:not([type=file])')) return;
  e.preventDefault();
  e.target.blur();   // so a width still on its way to being snapped is snapped first
  fireGenerate();
}));

// ---------- ⌥← / ⌥→ : move the clause under the caret ----------
// A prompt is written by reordering it. "in soft window light" belongs before
// the subject as often as after, and moving it there by hand is a select, a
// cut, a click and a paste — four gestures, each with its own way of eating a
// comma or leaving a double space behind.
//
// The separators are slots and they do not move: the commas and line breaks
// stay exactly where they are and the text between them changes places. That is
// what keeps a prompt written across two lines at two lines, and a prompt with
// one comma at one comma, however many times you press the chord. Each slot
// also keeps its own leading and trailing whitespace, so a clause moving into
// the first position does not drag the space that used to precede it along.
//
// Returns false rather than doing nothing quietly when there is nowhere to go —
// at the ends, or against an empty slot, which is a trailing comma and not a
// clause. The caller lets the key fall through to the OS's word-jump there,
// which is the honest answer to ⌥← on the first clause in the box.
function moveClause(el,dir){
  const v=el.value, at=el.selectionStart;
  const slots=[];
  let s=0;
  for(let i=0;i<=v.length;i++){
    if(i!==v.length&&v[i]!==','&&v[i]!=='\n') continue;
    const t=v.slice(s,i), core=t.trim();
    slots.push({
      s, e:i, sep:v[i]||'', core,
      // An all-whitespace slot has no core to sit between a lead and a tail,
      // and splitting it into both would write the run out twice.
      lead: core ? t.slice(0,t.length-t.trimStart().length) : t,
      tail: core ? t.slice(t.trimEnd().length) : '',
    });
    s=i+1;
  }
  const i=slots.findIndex(sl=>at>=sl.s&&at<=sl.e), j=i+dir;
  if(i<0||j<0||j>=slots.length||!slots[i].core||!slots[j].core) return false;
  // Where the caret sits inside the clause it is holding, so a run of presses
  // keeps hold of it rather than moving it once and losing its place.
  const within=Math.min(slots[i].core.length,
                        Math.max(0, at-slots[i].s-slots[i].lead.length));
  [slots[i].core,slots[j].core]=[slots[j].core,slots[i].core];
  let out='', pos=0;
  slots.forEach((sl,k)=>{
    if(k===j) pos=out.length+sl.lead.length;
    out+=sl.lead+sl.core+sl.tail+sl.sep;
  });
  el.value=out;
  el.setSelectionRange(pos+within,pos+within);
  // Bubbling, because the note under the field is driven by an input listener
  // and a `<lora:…>` token that has just changed clause has not changed what it
  // resolves to — but the caret the LoRA picker tracks reads this too.
  el.dispatchEvent(new Event('input',{bubbles:true}));
  return true;
}
// The negatives take it as well: they are the same kind of comma-separated
// prose, and a chord that works in one box and not the one under it is a chord
// nobody trusts.
['#prompt','#neg'].forEach(sel=>$(sel).addEventListener('keydown',e=>{
  if(!e.altKey||e.metaKey||e.ctrlKey||e.isComposing) return;
  if(e.key!=='ArrowLeft'&&e.key!=='ArrowRight') return;
  if(moveClause(e.target,e.key==='ArrowRight'?1:-1)) e.preventDefault();
}));

// ---------- the prompt field ----------
// Measured, not asked for. The box opens at one row and grows with the text,
// which is the whole reason the resize grip could go: the height was never a
// preference, it was an observation the box can make itself.
//
// Capped, because the console is what the canvas is sized against — layoutFrame
// and the --shot-h sum both measure #canvas, and a field free to grow without
// limit is a field that can push the picture off screen one keystroke at a
// time. Past the cap it scrolls, which is the one place a scrollbar is the
// right answer: the prompt has stopped being glanceable anyway.
// The console's whole budget, and the prompt is what yields to it.
//
// Everything else here is either fixed or already conditional — the strip is one
// row, the rail only exists once you pick a pill, the region bar only once you
// arm regions. The prompt is the one part that grows without asking, and
// measuring showed it is also the part that breaks the budget on its own: at a
// flat 168px cap the worst case was 39.8% of a 1440x900 window, of which the
// prompt was 136.
//
// So the cap is not a number, it is what is left. The console measures its own
// non-prompt height and hands the remainder to the field, down to a floor of
// two lines — below that the box stops being a place you can write, and a
// budget that wins by making the prompt unusable has optimised the wrong thing.
// Past the cap it scrolls, which is the one place a scrollbar is right: a prompt
// that long has stopped being glanceable anyway.
const CONSOLE_BUDGET = 0.30;   // of the viewport
const FIELD_FLOOR = 52, FIELD_CEIL = 168;
const liveField=()=>$('.field.on-neg')?$('#neg'):$('#prompt');
function fieldMax(){
  const con=$('.console'); if(!con) return FIELD_CEIL;
  const other=con.getBoundingClientRect().height - liveField().getBoundingClientRect().height;
  return Math.max(FIELD_FLOOR, Math.min(FIELD_CEIL, innerHeight*CONSOLE_BUDGET - other));
}
let growing=false;
function autoGrow(el){
  if(!el||growing) return;
  growing=true;
  el.style.height='auto';
  el.style.height=Math.min(el.scrollHeight, fieldMax())+'px';
  growing=false;
}
// The budget is a fraction of the window, so it moves when the window does.
addEventListener('resize',()=>autoGrow(liveField()));
// And the console has to watch itself, because the prompt is not the only thing
// that grows: arming Regions adds a bar and picking pills adds a rail, and both
// happen long after the last keystroke. Without this the field kept whatever
// height it had won when it was the only claimant — measuring showed a long
// prompt at 30.0% climbing to 38.1% the moment regions and a full rail arrived,
// which is the budget being right once and then not again.
//
// It converges in one pass rather than oscillating: fieldMax subtracts the
// field's own height, so `other` does not move when the field does. The flag is
// for the 'auto' write inside autoGrow, which changes layout mid-measurement.
new ResizeObserver(()=>autoGrow(liveField())).observe($('.console'));
['#prompt','#neg'].forEach(sel=>$(sel).addEventListener('input',e=>autoGrow(e.target)));

// Whether this model reads a negative prompt at all.
//
// On the video side the model says so. On the image side nothing did, and that
// is the gap this closes: Krea 2 Turbo is distilled to CFG 1.0, where a
// negative prompt is not weak but *unread*, and the box sat at the top of
// Advanced regardless. Read off the effective CFG rather than the checkpoint
// name, so a Turbo run with CFG typed up to 5 gets the control back — the rule
// is about the number the sampler uses, not about which file is loaded.
function negAllowed(){
  if(kind==='video'){
    const m=videoModel();
    return !!(m&&m.supports&&m.supports.negative);
  }
  const typed=parseFloat($('#g-cfg').value);
  const def=((window.KREA2_DEFAULTS||{})[$('#g-model').value]||{}).cfg;
  const cfg=Number.isFinite(typed)?typed:def;
  return Number.isFinite(cfg)&&cfg>1;
}

// Which of the two textareas is showing, plus the dot that says the other one
// is not empty.
function setNegMode(on){
  const f=$('.field');
  f.classList.toggle('on-neg',on);
  $('#prompt').classList.toggle('hide',on);
  $('#neg').classList.toggle('hide',!on);
  $('#neg-toggle').textContent = on ? 'negative' : 'positive';
  autoGrow($(on?'#neg':'#prompt'));
  syncNeg();
}
function syncNeg(){
  const ok=negAllowed(), f=$('.field'), t=$('#neg-toggle');
  f.classList.toggle('has-neg',ok);
  t.classList.toggle('hide',!ok);
  // Switched away from under you: a model that reads no negative must not
  // leave you typing into the field it will not read. The text is kept — the
  // next model may well read it, and silently emptying a box someone wrote in
  // is the one thing worse than ignoring it.
  if(!ok&&f.classList.contains('on-neg')) return setNegMode(false);
  t.classList.toggle('filled', !!$('#neg').value.trim());
}
$('#neg-toggle').onclick=()=>{
  const to=!$('.field').classList.contains('on-neg');
  setNegMode(to);
  $(to?'#neg':'#prompt').focus();
};
$('#neg').addEventListener('input',syncNeg);
// A typed CFG can turn the control on for a checkpoint whose default is 1.0,
// so the box that decides has to say when it changes.
$('#g-cfg').addEventListener('input',syncNeg);

// ---------- ↑ / ↓ on a number ----------
// Every numeric box in the composer takes the arrows, ⌘ (or Ctrl) for the
// coarse step. Delegated from the two sections rather than bound per field for
// the same reason label() is a function: region rows are built long after load,
// and a handler attached only at startup is one every row added later silently
// goes without.
//
// 1 and 8 are the defaults because Width and Height are the boxes this was
// asked for, and 8 is the VAE's grid — ⌘↑ there lands on the next size the
// model can actually render rather than one it will floor. A field whose useful
// range is 1.0 to 1.4 carries its own data-step: stepping a shift of 1.15 by 8
// leaves behind every value the model accepts, which is a shortcut that cannot
// be right once. The coarse step is 8× the fine one unless data-bigstep says
// otherwise, so the two stay in proportion wherever the fine one is overridden.
//
// Nothing is snapped or clamped to the grid here. The boxes already do that on
// the way out — typing 1153 shows 1153 until you leave the field — and an arrow
// is a faster way to type a number, not a second way to commit one.
const dec=v=>(String(v).split('.')[1]||'').length;
// What ↑ counts from when the box is empty. The placeholder when it is a
// number, which is what the video row prints there; otherwise the table the
// image row's "auto" stands for, so ↑ on an empty Steps means "one more than
// the model would have used" rather than "1". A seed reading "random" has no
// such number and is left alone — walking a seed you have not drawn yet is not
// a thing the box can do.
function nudgeBase(el){
  const ph=parseFloat(el.placeholder);
  if(Number.isFinite(ph)) return ph;
  const d=(window.KREA2_DEFAULTS||{})[$('#g-model').value]||{};
  if(el.id==='g-steps') return d.steps;
  if(el.id==='g-cfg') return d.cfg;
  return null;
}
function nudgeNumber(el,dir,coarse){
  const fine=parseFloat(el.dataset.step)||1;
  const step=coarse ? (parseFloat(el.dataset.bigstep)||fine*8) : fine;
  const base=el.value!=='' ? parseFloat(el.value) : nudgeBase(el);
  if(!Number.isFinite(base)) return false;
  // Rounded to the decimals the value and the step between them carry: 1.15 +
  // 0.1 is 1.2500000000000002 in binary floating point, and the box would
  // print every digit of it.
  const next=Math.max(0, Number((base+dir*step).toFixed(
    Math.max(dec(step),dec(base)))));
  el.value=String(next);
  el.dispatchEvent(new Event('input',{bubbles:true}));
  return true;
}
$$('#c-image,#c-video,#region-bar').forEach(sec=>sec.addEventListener('keydown',e=>{
  if(e.key!=='ArrowUp'&&e.key!=='ArrowDown') return;
  if(e.altKey||e.shiftKey) return;
  if(!e.target.matches('input[inputmode=numeric],input[inputmode=decimal]')) return;
  if(nudgeNumber(e.target, e.key==='ArrowUp'?1:-1, e.metaKey||e.ctrlKey)) e.preventDefault();
}));

// A file is over the window: light every place it could go. See the
// drag-intent block in the stylesheet for what that means and why it is the
// only moment this app is willing to spend pixels explaining itself.
//
// Driven off `dragover` and a timer rather than dragenter/dragleave counting.
// The counting version is the textbook one and it is wrong here for the reason
// already recorded on wireCanvasDrop's own dragleave: every child element the
// cursor crosses fires its own leave, and on a page whose targets contain
// images, boxes and eight resize handles the depth counter drifts and the
// reveal strobes. dragover repeats while the drag is alive, so "still dragging"
// is a fact the browser re-states every few hundred milliseconds, and the only
// thing that needs guessing is when it stopped.
let dragEnd=null;
addEventListener('dragover',e=>{
  // Files only. Dragging a text selection out of the prompt field must not
  // make the page look like it wants to eat it.
  if(![...(e.dataTransfer?.types||[])].includes('Files')) return;
  document.body.classList.add('dragging');
  // The one moment "you can drop a photo on a box" needs saying, which is also
  // the only way a new user finds it at all.
  if(typeof syncRegionVis==='function') syncRegionVis();
  clearTimeout(dragEnd);
  dragEnd=setTimeout(dropOff,300);
});
// dragend fires on a drag that started inside the page; drop fires on one that
// came from outside and landed. Neither fires when a drag leaves the window
// entirely, which is what the timer is for.
['drop','dragend'].forEach(k=>addEventListener(k,dropOff));
function dropOff(){
  clearTimeout(dragEnd);
  document.body.classList.remove('dragging');
  if(typeof syncRegionVis==='function') syncRegionVis();
  // Every target's own dragleave clears its own highlight, and that is enough
  // right up until a drag leaves the window faster than the events describing
  // it — then a tile stays lit with nothing over it, which is a control
  // claiming to be a live drop target on a page where no drag is happening.
  // This is the one handler that knows the drag is over no matter how it
  // ended, so it is the one that owns the sweep.
  $$('.hot').forEach(el=>el.classList.remove('hot'));
  if(typeof clearCanvasDrop==='function') clearCanvasDrop();
}

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
  // The boxes are an image-side thing and the video canvas is the same element,
  // so switching kinds has to take the layer with it — otherwise the rectangles
  // sit over a clip they mean nothing to. drawRegions also owns the
  // canvas-empty toggle when regions are on, so it runs after this one.
  // window flag, not typeof — see the note in syncSize.
  if(window.REGIONS_READY) drawRegions();
  syncDropTargets();
  syncCanvasActs();
}

// Which of the big surfaces would actually take a file right now. The video
// canvas always would; the still canvas only inside Regions, because that is
// the only mode whose drop has a meaning — outside it there is no scene plate
// to become and no box to land in.
function syncDropTargets(){
  const live = kind==='video' || (window.REGIONS_READY && regionOn());
  $$('#canvas .frame, #canvas .shot').forEach(el=>el.classList.toggle('can-drop',live));
}

$('#t-drawer').onclick=()=>{
  const off=$('#v-generate').classList.toggle('nodrawer');
  $('#t-drawer').classList.toggle('on',!off);
};
$('#t-settings').onclick=()=>{ $('#settings').classList.remove('hide'); loadState(); };
$('#settings-x').onclick=()=>$('#settings').classList.add('hide');
$('#settings').onclick=e=>{ if(e.target.id==='settings') $('#settings').classList.add('hide') };


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
// Capture, because a menu anchored to a button inside a scrolling pane has to
// close when that pane moves under it — but the menu is itself a scroller when
// the LoRA list is long, and its own scroll reaches this listener the same way.
// Without the guard the list closed on the first wheel notch, which reads as
// "the picker cannot scroll" rather than "the picker is closing".
document.addEventListener('scroll',e=>{ if(!(menuEl&&menuEl.contains(e.target))) closeMenu() },true);

// Anchoring and teardown, shared by the menu and the shot palette. Factored
// out rather than copied: the scroll-close guard above is a bug that was fixed
// once, and a second floating element with its own copy of this is a second
// place for it to come back.
function floatBy(btn,el){
  closeMenu();
  document.body.appendChild(el);
  const r=btn.getBoundingClientRect(), w=el.offsetWidth, h=el.offsetHeight;
  el.style.left=Math.max(8,Math.min(r.right-w,innerWidth-w-8))+'px';
  // Clamped, because a menu tall enough to need flipping above its button is
  // also tall enough for r.top-h to land off the top of the window, and the
  // rows that go past the edge are unreachable — the scrollbar is inside the
  // menu, so nothing scrolls them back.
  const top=(r.bottom+h+10>innerHeight ? r.top-h-6 : r.bottom+6);
  el.style.top=Math.max(8,Math.min(top,innerHeight-h-8))+'px';
  menuEl=el;
  return el;
}

function openMenu(btn,items){
  const m=document.createElement('div'); m.className='menu';
  m.innerHTML=items.map((it,i)=>it.sep?'<hr>':
    `<button data-i="${i}" class="${it.danger?'danger':''}${it.on?' on':''}">${esc(it.label)}</button>`).join('');
  if(items.some(it=>it.on)) m.classList.add('checks');
  floatBy(btn,m);
  m.querySelectorAll('button').forEach(b=>b.onclick=()=>{ const it=items[+b.dataset.i]; closeMenu(); it.run() });
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

// What is on the canvas right now, if anything — the thing both the full-screen
// control and the key press act on. Read off the DOM rather than kept in a
// variable, because the canvas is written from four places (a finished run,
// Reuse, the gallery hand-off, a kind switch) and a fifth copy of the answer is
// a fifth thing to forget to update.
function canvasShot(){
  if(kind==='video'){ const v=$('#vid-out video'); return v ? {src:v.src, video:true} : null }
  const i=$('#gen-out .shot img'); return i ? {src:i.src, video:false} : null;
}
function syncCanvasActs(){
  $('#canvas-acts').classList.toggle('hide', !canvasShot());
}
$('#canvas-full').onclick=()=>{ const s=canvasShot(); if(s) lightbox(s.src,s.video) };
$('#canvas-clear').onclick=()=>{
  // The canvas only. The prompt, the pills, the boxes and the settings are all
  // still what you were working on — this clears the result, which is the one
  // thing "clear" can mean when everything else is an input you are mid-edit of.
  (kind==='video' ? $('#vid-out') : $('#gen-out')).innerHTML='';
  $(kind==='video' ? '#vid-meta' : '#gen-meta').innerHTML='';
  syncCanvasView(); syncCanvasActs();
};
// Space, because ⌘Space is Spotlight on a stock Mac and never reaches the page.
// Both are bound: ⌘Space works for anyone who has remapped Spotlight, and the
// bare key is the one that works out of the box. Guarded on where the caret is
// rather than on a modifier — a space inside the prompt is a space.
addEventListener('keydown',e=>{
  if(e.key!==' '&&e.code!=='Space') return;
  if(e.ctrlKey||e.altKey||e.shiftKey) return;
  const t=e.target;
  if(t&&(t.matches('input,textarea,select')||t.isContentEditable)) return;
  if(document.querySelector('.lb')||menuEl) return;
  const s=canvasShot(); if(!s) return;
  e.preventDefault(); lightbox(s.src,s.video);
});

// The set the viewer is walking, and where in it. Opening one picture and
// having to close it to see the next is the tax this removes: on a phone,
// twenty takes is twenty taps out and twenty back in.
let viewSet=[], viewIdx=0;
function viewAt(rows,i){
  viewSet=rows; viewIdx=i;
  const it=rows[i];
  lightbox(`/api/file/${it.job_id}/${it.files[0]}`, it.kind==='video');
}
function viewStep(d){
  const i=viewIdx+d;
  // Stops at the ends rather than wrapping. A set that loops has no edges, and
  // "am I back where I started" is the question you cannot answer while flicking.
  if(!viewSet.length||i<0||i>=viewSet.length) return;
  const keep=viewSet;
  document.querySelector('.lb')?.remove();
  viewAt(keep,i);
}

function lightbox(src,video){
  const el=document.createElement('div'); el.className='lb';
  const many=viewSet.length>1;
  // Three slides, not one. The middle is what you are looking at and the
  // neighbours are already in the DOM, because the drag has to show them: a
  // page that only swaps on release is a cut, and a cut does not tell you which
  // way you went or that there is anything either side. This is the one place
  // in the app where an animation is carrying information rather than decorating.
  const at=i=>{
    const it=viewSet[i]; if(!it) return '<div class="lb-slide"></div>';
    const u=`/api/file/${it.job_id}/${it.files[0]}`;
    // draggable=false, because an <img> is natively draggable: mousedown and
    // move starts an HTML image drag, the browser fires pointercancel, and the
    // gesture dies one frame in. Touch never takes that path, so the same code
    // tracked a thumb and refused a trackpad — the exact asymmetry an interface
    // this shape must not have.
    return `<div class="lb-slide">`+(it.kind==='video'
      ? `<video src="${u}" controls loop playsinline draggable="false"></video>`
      : `<img src="${u}" alt="" draggable="false">`)+`</div>`;
  };
  el.innerHTML=`<button class="x" title="Close">`
    +`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1"`
    +` stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l6 6-6 6"/></svg></button>`
    +(many?`<button class="lb-nav prev" ${viewIdx<=0?'disabled':''}>${ICON.back}</button>`
         +`<button class="lb-nav next" ${viewIdx>=viewSet.length-1?'disabled':''}>${ICON.back}</button>`
         +`<span class="lb-at">${viewIdx+1} / ${viewSet.length}</span>`:'')
    +`<button class="lb-all" title="All generations">`
    +`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"`
    +` stroke-linejoin="round"><rect x="3" y="3" width="7.5" height="7.5" rx="1.4"/>`
    +`<rect x="13.5" y="3" width="7.5" height="7.5" rx="1.4"/>`
    +`<rect x="3" y="13.5" width="7.5" height="7.5" rx="1.4"/>`
    +`<rect x="13.5" y="13.5" width="7.5" height="7.5" rx="1.4"/></svg></button>`
    +`<div class="lb-track">`
    +(many ? at(viewIdx-1)+at(viewIdx)+at(viewIdx+1)
           : `<div class="lb-slide">`+(video
               ? `<video src="${src}" controls autoplay loop playsinline draggable="false"></video>`
               : `<img src="${src}" alt="" draggable="false">`)+`</div>`)
    +`</div>`;
  const track=el.querySelector('.lb-track');
  if(many) track.style.transform='translateX(-100%)';

  const close=()=>{ el.remove(); document.removeEventListener('keydown',onKey); viewSet=[] };
  const onKey=e=>{
    if(e.key==='Escape') return close();
    if(e.key==='ArrowLeft') return viewStep(-1);
    if(e.key==='ArrowRight') return viewStep(1);
  };
  el.querySelector('.lb-all').onclick=()=>{ close(); openGallery() };
  if(many){
    el.querySelector('.prev').onclick=e=>{ e.stopPropagation(); viewStep(-1) };
    el.querySelector('.next').onclick=e=>{ e.stopPropagation(); viewStep(1) };
  }

  // The drag itself. While the pointer is down the track follows it one-to-one,
  // so both takes are on screen and the gesture is reversible — let go halfway
  // and it falls back to where it was. It only ever comes to rest on a take.
  let sx=null, sy=null, st=0, dragging=false, w=1, justDragged=false;
  const settle=(dx,commit)=>{
    track.classList.add('snap');
    track.style.transform=`translateX(${commit? (dx<0?'-200%':'0%') : '-100%'})`;
  };
  el.addEventListener('pointerdown',e=>{
    if(!many||e.target.closest('button')) return;
    sx=e.clientX; sy=e.clientY; st=e.timeStamp; dragging=false;
    w=el.getBoundingClientRect().width||1;
    track.classList.remove('snap');
    // Captured, so a drag that wanders off the picture — or off the window —
    // keeps arriving here instead of stopping wherever the pointer went.
    try{ el.setPointerCapture(e.pointerId) }catch(_){}
  });
  el.addEventListener('pointermove',e=>{
    if(sx==null) return;
    const dx=e.clientX-sx, dy=e.clientY-sy;
    // Committed to horizontal only once it is clearly horizontal, so a vertical
    // flick on a tall picture is still a scroll and not a half-page.
    if(!dragging){ if(Math.abs(dx)<10||Math.abs(dx)<=Math.abs(dy)) return; dragging=true }
    e.preventDefault();
    // Resistance at the ends: it moves, so the gesture is alive, but a third as
    // far, which is how an edge is felt rather than announced.
    const edge=(dx>0&&viewIdx<=0)||(dx<0&&viewIdx>=viewSet.length-1);
    track.style.transform=`translateX(calc(-100% + ${dx*(edge?0.3:1)}px))`;
  });
  const release=e=>{
    if(sx==null) return;
    const dx=e.clientX-sx; const was=dragging;
    sx=null; dragging=false;
    if(!was) return;
    // A drag ends in a click. Without this the click lands on the backdrop and
    // closes the viewer, so every successful swipe shut the thing it was
    // paging — and on a trackpad that is the only outcome you ever see.
    justDragged=true; setTimeout(()=>{ justDragged=false },0);
    // Distance or speed, not distance alone. A quarter of the width is a long
    // way on a 1200px viewer, and requiring it made every quick flick fall back
    // — the gesture people actually use to go through twenty takes is a short
    // fast one, not a slow haul across the screen. 0.45px/ms is about the speed
    // of a flick that means it and comfortably above a drag that is still
    // deciding, so either a long pull or a brisk flick pages and a slow short
    // nudge does not.
    const v=Math.abs(dx)/Math.max(1,e.timeStamp-st);
    const meant=Math.abs(dx)>w*0.25 || (v>0.45 && Math.abs(dx)>36);
    const commit=meant
      && !((dx>0&&viewIdx<=0)||(dx<0&&viewIdx>=viewSet.length-1));
    settle(dx,commit);
    if(!commit) return;
    const keep=viewSet, next=viewIdx+(dx<0?1:-1);
    // Swapped after the slide lands, so the rebuild is invisible: the picture
    // already sits where the new middle slide will put it.
    track.addEventListener('transitionend',()=>{
      document.querySelector('.lb')?.remove(); viewAt(keep,next);
    },{once:true});
  };
  el.addEventListener('pointerup',release);
  // Cancel means something took the gesture away, which is not the same as you
  // finishing it — so it goes back rather than committing to a take you may
  // never have asked for.
  el.addEventListener('pointercancel',()=>{
    if(sx==null) return;
    sx=null; dragging=false; settle(0,false);
  });
  el.onclick=e=>{
    if(justDragged) return;
    if(e.target.closest('.x')) return close();
    if(e.target.closest('button,video')) return;   // controls keep their own jobs
    // On the picture, a click asks for more of it — chrome off. Off the picture
    // is where the two devices differ, and they should: clicking the dark
    // surround to dismiss is a desktop convention old enough that removing it
    // reads as a broken dialog, while on glass the picture fills the screen and
    // that surround is a few pixels nobody aims at. So the mouse keeps its
    // click-away and touch does not have to grow one.
    if(e.target.closest('.lb-slide img,.lb-slide video')) return el.classList.toggle('bare');
    if(matchMedia('(hover:hover)').matches) return close();
    el.classList.toggle('bare');
  };
  document.addEventListener('keydown',onKey);
  document.body.appendChild(el);
}

// ==================== THE SHOT PALETTE ====================
// One icon opens a vocabulary. H3 reads a document with named fields and the
// composer offered a textarea for it, so where camera direction goes, whether
// tone belongs in the sentence, and what a comma does were all things you found
// out by spending three minutes rendering. None of that is a grammar anyone
// could infer — it is a schema, published in the model repo — so the page emits
// it and the pills are how you say what goes in it.
//
// The vocabulary is served, never written into this file. A copy here would be
// a second source of truth, and the first pill added on one side and not the
// other would compile to "No such shot pill" against the page that offered it.
let shotVocab=[], shotLangs=['English'], shotRoleDefs=[];
// The captioner's menus, served for the same reason the palette is: the
// instruction behind a preset lives on the server, so a copy of the labels here
// would be a menu offering keys `/api/caption` rejects by name.
let capPresets=[], capModels=[];
let shot=[];        // [{key,value?,lang?}] — the rail, in the order you built it
let shotOpen=null;  // which valued pill is expanded, if any
let refRoles=[];    // one role per image reference, positional

const shotGroup=k=>shotVocab.find(g=>g.key===k.split('.')[0]);
const shotItem=k=>{ const g=shotGroup(k);
  return g&&g.items.find(it=>it.key===k.split('.').slice(1).join('.')) };

// Whether the thing in front of you reads this pill at all. Two different
// reasons it might not, and they are worth different words: the image side has
// no camera and no soundtrack, and Wan is silent.
//
// Per item as well as per group, because one item disagrees with its group and
// it is the one that matters: on-screen text is a picture of words and every
// model can draw it, while a spoken line is audio. Given a group and no item,
// the answer is whether *any* of its items is live — which is what decides
// whether the whole section is dim.
function shotLive(g,it){
  if(kind==='image') return !!g.image;
  if(!it) return g.items.some(x=>shotLive(g,x));
  const need=it.needs||g.needs;
  return !need || !!((videoModel()||{supports:{}}).supports||{})[need];
}
function shotWhy(g){
  if(kind==='image') return 'video only';
  const m=videoModel();
  return m?m.label+' is silent':'';
}

// The tile skeleton. Nine shapes, and the class on the <svg> is what turns them
// into a push-in or a candle — see the stylesheet for why it is one drawing
// rather than eighty. The posts are spaced exactly 20 so a 20px stream loops
// with no seam, and they are what makes lateral movement visible at all: a
// horizontal line translated horizontally is a correct animation of nothing.
const GL_TICKS=[-34,-14,6,26,46,66];
const GL_GRAIN=[[7,6],[16,11],[27,6],[35,14],[10,20],[21,25],[38,22],[30,17],[5,13]];
function glyph(cls){
  return `<svg class="gl gl-${cls}" viewBox="0 0 44 30" aria-hidden="true">`
    +'<path class="lt" d="M4 -2H20L11 32H-8Z"/>'
    +'<g class="wo"><line class="hz" x1="-40" y1="21" x2="84" y2="21"/>'
    +'<g class="tk">'+GL_TICKS.map(x=>`<path d="M${x} 21v-4.5"/>`).join('')+'</g>'
    +'<ellipse class="s2" cx="22" cy="16" rx="3.2" ry="4.8"/>'
    +'<ellipse class="s1" cx="22" cy="16" rx="3.2" ry="4.8"/>'
    +'<circle class="ob" cx="22" cy="17.5" r="1.7"/></g>'
    +'<path class="pv" d="M3 30 13 21M41 30 31 21"/>'
    +'<path class="bu" d="M26 5h13a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-7l-4 3v-3a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2z"/>'
    +'<path class="tx" d="M12 22h20M17 26h10"/>'
    +'<g class="gr">'+GL_GRAIN.map(p=>`<circle cx="${p[0]}" cy="${p[1]}" r=".75"/>`).join('')+'</g>'
    +'<g class="bars"><rect x="-8" y="-1" width="60" height="4.6"/>'
    +'<rect x="-8" y="26.4" width="60" height="4.6"/></g>'
    +'<rect class="fr" x=".5" y=".5" width="43" height="29" rx="2.5"/></svg>';
}

function openPalette(btn){
  const el=document.createElement('div'); el.className='pal';
  // Live groups first, dead ones after. Dimming rather than hiding is the
  // established call — a control that vanishes teaches only that the page lost
  // it, and "this app moves the camera on video" is worth learning while you
  // are on the image side. But on Krea 2 that is fifty-seven of eighty-seven
  // tiles, and leaving them in vocabulary order meant scrolling through two
  // dead groups to reach a live one. Display order only: the compiler still
  // reads SHOT_VOCAB in its own order, which is what decides clause position.
  el.innerHTML=shotVocab.slice()
    .sort((a,b)=>(shotLive(a)?0:1)-(shotLive(b)?0:1))
    .map(g=>{
    const off=!shotLive(g);
    return `<section class="${off?'off':''}"><h4>${esc(g.label)}`
      +(off?`<i>${esc(shotWhy(g))}</i>`:'')+'</h4><div class="tiles">'
      // The tooltip is the phrase the pill writes, so hovering a tile shows the
      // exact wording it puts in the document. That is the one thing a glyph
      // and a two-word label between them still cannot say, and it is what
      // makes the palette teach the phrasing rather than replace it.
      +g.items.map(it=>{
        const k=g.key+'.'+it.key, dead=!shotLive(g,it);
        return `<button class="tl${shot.some(p=>p.key===k)?' on':''}${dead?' off':''}"`
          +` data-k="${esc(k)}" title="${esc(it.phrase||it.hint||'')}"${dead?' disabled':''}>`
          +glyph(it.glyph)+`<span>${esc(it.label)}</span></button>`;
      }).join('')+'</div></section>';
  }).join('');
  floatBy(btn,el);
  el.querySelectorAll('.tl:not(.off)').forEach(b=>b.onclick=()=>{
    const it=shotItem(b.dataset.k), added=!shot.some(p=>p.key===b.dataset.k);
    toggleShot(b.dataset.k);
    // A valued pill arrives with somewhere to type, and that is on the rail —
    // so adding one closes the palette rather than leaving a caret behind a
    // popover. Everything else leaves it open, because you are picking several
    // and a palette that shuts on the first click is one you reopen five times.
    if(added&&it.valued) return closeMenu();
    el.querySelectorAll('.tl').forEach(t=>
      t.classList.toggle('on',shot.some(p=>p.key===t.dataset.k)));
  });
}

function toggleShot(key){
  const g=shotGroup(key), it=shotItem(key);
  if(!g||!it) return;
  const at=shot.findIndex(p=>p.key===key);
  if(at>=0){ shot.splice(at,1); if(shotOpen===key) shotOpen=null }
  else{
    // The same exclusions `_validate_shot` applies, for a different reason.
    // There they make the rule true; here they make it legible — the guide
    // allows one camera move per clip, and a palette that let you stack three
    // would be teaching the opposite of what it compiles.
    const same=p=>shotGroup(p.key)===g;
    if(g.pick==='one'||it.solo) shot=shot.filter(p=>!same(p));
    else shot=shot.filter(p=>!(same(p)&&(shotItem(p.key)||{}).solo));
    const p={key};
    if(it.valued){ p.value=''; if(it.valued==='dialogue') p.lang=shotLangs[0]||'English' }
    shot.push(p);
    if(it.valued) shotOpen=key;
  }
  drawShotRail();
}

function drawShotRail(){
  const rail=$('#shot-rail');
  rail.innerHTML=shot.map(p=>{
    const g=shotGroup(p.key), it=shotItem(p.key);
    if(!g||!it) return '';
    const off=shotLive(g,it)?'':' off';
    if(it.valued&&shotOpen===p.key){
      // The language is a select, not a field, because the guide names the
      // eleven and forbids inventing one — and because a language tag is a
      // thing you cannot know to write, which is the whole reason dialogue is
      // a pill instead of something you type into the prompt.
      const langs=it.valued==='dialogue'
        ? '<select class="lang">'+shotLangs.map(l=>
            `<option${l===p.lang?' selected':''}>${esc(l)}</option>`).join('')+'</select>'
        : '';
      return `<span class="spill val open${off}" data-k="${esc(p.key)}">`+glyph(it.glyph)+langs
        +`<input class="v" value="${esc(p.value||'')}" placeholder="${esc(it.hint||it.label)}">`
        +'<button class="x" title="Remove">×</button></span>';
    }
    // Collapsed, a valued pill reads as what you chose rather than as a form.
    const words=it.valued?(p.value||it.label):it.label;
    return `<span class="spill${it.valued?' val':''}${off}" data-k="${esc(p.key)}">`+glyph(it.glyph)
      +`<b class="${it.valued&&p.value?'set':''}">${esc(words)}</b>`
      +'<button class="x" title="Remove">×</button></span>';
  }).join('');

  // mousedown, not click. An expanded pill's input has focus and its focusout
  // redraws the rail, so by the time a click on ✕ would fire, the button it was
  // aimed at has been replaced and the click lands on nothing.
  rail.querySelectorAll('.x').forEach(b=>b.addEventListener('mousedown',e=>{
    e.preventDefault(); toggleShot(b.parentElement.dataset.k);
  }));
  rail.querySelectorAll('.spill.val:not(.open)').forEach(el=>el.onclick=e=>{
    if(e.target.closest('.x')) return;
    shotOpen=el.dataset.k; drawShotRail();
  });

  const box=rail.querySelector('.spill.open');
  if(box){
    const input=box.querySelector('input'), sel=box.querySelector('select');
    const p=shot.find(x=>x.key===box.dataset.k);
    input.oninput=()=>{ p.value=input.value; syncShotPeek() };
    if(sel) sel.onchange=()=>{ p.lang=sel.value; syncShotPeek() };
    // focusout on the pill rather than blur on the input: the language select
    // is inside the same pill, and a blur handler collapsed it the moment you
    // reached for the one control the expansion exists to offer. This is the
    // keyboard half — tabbing away — and it is not enough on its own; see the
    // mousedown listener below.
    box.addEventListener('focusout',e=>{
      if(box.contains(e.relatedTarget)) return;
      shotOpen=null; drawShotRail();
    });
    input.focus();
    input.setSelectionRange(input.value.length,input.value.length);
  }
  $$('#g-shot,#v-shot').forEach(b=>b.classList.toggle('on',shot.length>0));
  syncShotPeek();
}

// What the model will actually be given. This is the answer to the expensive
// half of the problem: a take costs two to three minutes, so every question
// about the format used to be paid for at that rate. Compiled by the same route
// that compiles the real run — never a second implementation in here, which
// would be a preview that can disagree with what runs.
let peekOpen=false, peekTimer=null;
function syncShotPeek(){
  const box=$('#shot-peek');
  // Offered only when there is something to compile. With no pills the compiled
  // prompt is the typed one, and a disclosure that opens to show you your own
  // sentence back is a control with nothing to say.
  const has=shot.length>0||refRoles.some(Boolean);
  box.classList.toggle('hide',!has);
  box.classList.toggle('open',peekOpen&&has);
  const pre=box.querySelector('pre');
  if(!has||!peekOpen){ if(pre) pre.remove(); return }
  clearTimeout(peekTimer);
  peekTimer=setTimeout(async()=>{
    const r=await post('/api/compile',{
      kind, model:$('#v-model').value, prompt:promptText(), shot:readShot(),
      seconds:$('#v-seconds').value,
      // Never the bytes. This is re-fetched on every pill and every keystroke,
      // and what the compiler needs from a reference is that there is one.
      first_frame:!!keyframe.first, last_frame:!!keyframe.last,
      references:refs.length, ref_videos:refVids.length,
      ref_roles:refRoles.slice(0,refs.length),
    });
    let el=box.querySelector('pre');
    if(!el){ el=document.createElement('pre'); box.appendChild(el) }
    el.textContent=r&&r.prompt!=null?r.prompt:((r&&r.error)||'—');
  },220);
}
// The pointer half of collapsing an expanded pill, and the load-bearing one.
// focusout is the obvious mechanism and it only covers the case where focus
// lands somewhere that takes it: click a canvas, a label, a dead area of the
// bar, and focus does not move at all, so nothing fires and the pill stays a
// form. This is the same capture-phase mousedown that closes the menu, which
// is the pattern in this file that is known to hold. One listener, not one per
// redraw — the rail is rebuilt on every keystroke's worth of state change.
document.addEventListener('mousedown',e=>{
  if(!shotOpen) return;
  const box=$('#shot-rail .spill.open');
  if(box&&!box.contains(e.target)){ shotOpen=null; drawShotRail() }
},true);

$('#shot-peek>button').onclick=()=>{ peekOpen=!peekOpen; syncShotPeek() };

function readShot(){
  return shot.map(p=>{
    const o={key:p.key};
    if(p.value!==undefined) o.value=p.value;
    if(p.lang) o.lang=p.lang;
    return o;
  });
}

$('#g-shot').onclick=e=>openPalette(e.currentTarget);
$('#v-shot').onclick=e=>openPalette(e.currentTarget);

// ==================== MODELS (settings) ====================
// The volume check is a page-load thing, not a Settings thing: what it decides
// is whether Generate works, and that has to be answered before the gear is
// ever opened. It ran at boot already — but it ran *concurrently* with the
// gallery and the dataset list, and the reload it does on the server could not
// survive that, so the one call that mattered was the one that 500'd. Opening
// Settings then fixed it by accident, being the only /api/state on its own.
// Hence the retry: this answer is worth waiting a few seconds for.
let stateTries=0;
async function loadState(){
  const s=await api('/api/state');
  if(!s||!s.models){
    stateTries++;
    // Named, not blank. "No DiT on the volume" for an answer that never
    // arrived sends you to download 17 GB you already have.
    $('#gen-note').textContent = stateTries<6
      ? 'Checking the volume… ('+(s&&s.error?s.error:'no answer')+')'
      : 'Could not read the volume: '+((s&&s.error)||'no answer');
    if(stateTries<6) setTimeout(loadState,2500);
    return;
  }
  stateTries=0;
  $('#tok-state').innerHTML = s.hf_token_set?'<span class="ok">Token saved.</span>':'<span class="warn">No token saved.</span>';
  // Name the cost up front: how many are missing and how many GB that is.
  const miss=s.models.filter(m=>!m.present);
  // Grouped by family, in catalogue order. Twenty-odd flat cards is a wall you
  // scroll rather than a list you read, and the groups are the unit you
  // actually decide in: you want the Wan stack or you do not.
  const fams=[];
  s.models.forEach(m=>{
    const g=fams.find(f=>f.name===m.family);
    (g||fams[fams.push({name:m.family,items:[]})-1]).items.push(m);
  });
  // The family is the unit you decide in, so it is the unit that downloads.
  // A stack is four or five files that are only useful together — the Wan pair
  // is literally two halves of one model — and clicking them one at a time
  // meant watching a 4 GB file finish to be allowed to start the next, which is
  // a queue kept in a person rather than in the program. The button is indexed
  // rather than carrying the family name in an attribute: `fams` is in scope
  // where the handlers are wired, so nothing has to be escaped into markup and
  // read back out.
  $('#models').innerHTML = fams.map((f,i)=>{
    const left=f.items.filter(m=>!m.present);
    const size=left.reduce((a,m)=>a+m.approx_gb,0);
    return `
    <div class="fam">
      <div class="fam-head">
        <b>${esc(f.name)}</b>
        <span class="muted">${left.length?`${left.length} missing · ${size.toFixed(1)} GB`:'complete'}</span>
        ${left.length>1?`<button class="s" data-dl-fam="${i}">Download all ${left.length}</button>`:''}
      </div>
      <div class="fam-prog hide"><div class="bar"><i style="width:0%"></i></div><div class="row" style="gap:10px;margin-top:7px"><p class="muted grow" style="margin:0"></p><button class="s" data-cancel>Cancel</button></div></div>
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
            : `<button class="s" data-dl="${m.key}">Download ${m.approx_gb} GB</button>
               <button class="s hide" data-dl-cancel="${m.key}">Cancel</button>`}
          <div class="muted dl-state" id="dl-${m.key}"></div>
        </div>
      </div>`).join('')}
    </div>`;
  }).join('');
  $$('[data-dl]').forEach(b=>b.onclick=()=>startDownload(b.dataset.dl,b));
  $$('[data-dl-fam]').forEach(b=>b.onclick=()=>startFamilyDownload(fams[+b.dataset.dlFam],b));

  // Everything a `<lora:…>` token can name. Rebuilt on each poll so a freshly
  // trained LoRA is typeable without a reload — nothing holds a selection any
  // more, so there is no pick to preserve across the rebuild.
  window.MAX_LORAS=s.max_loras||6;
  window.MAX_REGIONS=s.max_regions||8;
  window.WAN_EXPERTS=s.wan_experts||['both','high','low'];
  // The palette's whole vocabulary, served rather than written into this file.
  // Rebuilt on each poll like the LoRA index, and the rail is redrawn from it:
  // a pill whose group has just changed which models read it has to re-dim
  // without a reload.
  shotVocab=s.shot_vocab||[];
  shotLangs=s.shot_langs||['English'];
  shotRoleDefs=s.shot_roles||[];
  drawShotRail();
  capPresets=s.caption_presets||[]; capModels=s.caption_models||[];
  const capDef=s.caption_defaults||{};
  fillCapSel($('#cap-preset'),capPresets,capDef.preset);
  fillCapSel($('#cap-model'),capModels,capDef.model);
  capNote();
  // Per-checkpoint steps and CFG. The boxes show "auto" rather than these, so
  // this is what ↑ on an empty one counts from — the number the backend would
  // have used, which is the only base that makes the first press mean anything.
  window.KREA2_DEFAULTS=s.krea2_defaults||{};
  // Polled rather than read once, so downloading the edit LoRA under the gear
  // makes the two plate tiles appear without a reload — the same way a freshly
  // trained LoRA becomes typeable.
  const hadEdit=window.HAS_EDIT_LORA;
  window.HAS_EDIT_LORA=!!s.edit_lora;
  // Re-runs the regional mode's own setter rather than duplicating what it
  // does, so the two tiles have exactly one rule deciding whether they show.
  // Setter, not the click handler: that would flip the mode off.
  if(hadEdit!==window.HAS_EDIT_LORA&&typeof setRegional==='function')
    setRegional(regionOn());
  loraIndex=s.loras.flatMap(l=>l.files.map(f=>{
    // Relative to loras/ and derived from the path rather than assembled from
    // the two labels, because the layout allows any nesting under a folder and
    // `folder + "/" + name` would quietly lose a level of it.
    const rel=String(f.path).split('/loras/').pop().replace(/\.safetensors$/i,'');
    // No `trigger` here. `/api/state` still serves `trigger_word` and the
    // dataset side still records it; the picker just stopped writing it into
    // the prompt, so carrying it on this index would be state nothing reads.
    return {path:f.path, rel, stem:rel.split('/').pop(),
            file:String(f.path).split('/').pop()};
  }));
  // The shortest name that still points at one file. A volume with one
  // `k3nan.safetensors` gets `<lora:k3nan:1>`; the matched Wan speed pairs,
  // which are both called `high`, get the folder that tells them apart.
  loraIndex.forEach(l=>{
    l.token = loraIndex.filter(x=>x.stem===l.stem).length===1 ? l.stem : l.rel;
  });
  $('#add-lora').disabled=!loraIndex.length;
  $('#v-add-lora').disabled=!loraIndex.length;
  syncLoraNote();
  // Same payload, the other view of it: what you can type, and what you can
  // throw away. Redrawn on every state load rather than only when Settings
  // opens, so the count in the header is never one training run out of date.
  drawLoras(s.loras);

  // Built once, and only once: the guard is what keeps a poll landing between
  // two clicks from resetting a sampler you just picked. Which also means the
  // default is applied here or nowhere — the lists arrive empty in the markup,
  // so a `selected` attribute has nothing to attach to.
  if(s.samplers&&!$('#g-sampler').options.length){
    $('#g-sampler').innerHTML=s.samplers.map(x=>`<option>${x}</option>`).join('');
    $('#g-scheduler').innerHTML=s.schedulers.map(x=>`<option>${x}</option>`).join('');
    // Served rather than spelled here, so the menu opens on exactly what the
    // backend would have used had the request left them out.
    const d=s.image_defaults||{};
    if(s.samplers.includes(d.sampler)) $('#g-sampler').value=d.sampler;
    if(s.schedulers.includes(d.scheduler)) $('#g-scheduler').value=d.scheduler;
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
  // Turbo first, against the catalogue's order: the catalogue is ordered by
  // what trains, this picker is read by what generates, and those disagree.
  // Eight steps against twenty-eight is the whole difference here, so falling
  // through to RAW because it happens to be listed first is a picker that
  // charges you three and a half times the sampling to open the page.
  const pick=['turbo','raw'].map(k=>s.models.find(m=>m.key===k)).filter(Boolean);
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
  // Both, because both read the model this line just chose. Setting `.value`
  // fires no change event, so the pair bound to `#g-model`'s onchange has to be
  // called by hand here — and syncNeg is the half that would go missing without
  // a symptom on the install that opens on Turbo. On a volume holding only RAW
  // it is `avail[0]` that selects a CFG of 5.5, and the negative prompt is a
  // control the sampler reads with nothing on the page offering it until you
  // touch the model select it is already on.
  syncModelLine(); syncNeg();
}

// GB past a gigabyte and MB under it, because a 1.8 GB weight and a 144 MB LoRA
// are both normal here and "0.14 GB" reads as a rounding error rather than a
// file. Two decimals to match what the catalogue's own sizes are rounded to.
const fmtBytes=b=>(b=+b||0)>=1e9?(b/1e9).toFixed(2)+' GB':Math.max(1,Math.round(b/1e6))+' MB';

// Everything under loras/, drawn from the same /api/state the picker's index is
// built from — so a LoRA that has just finished training is deletable without a
// reload, for the same reason it is typeable without one.
//
// The trigger word is here and nowhere else on the page. The picker stopped
// writing it into prompts, but "which one is alxcn" is precisely the question
// standing between you and a delete, and the answer was already in the payload.
function drawLoras(list){
  const total=list.reduce((a,l)=>a+(+l.bytes||0),0);
  $('#lora-total').textContent=list.length
    ? `${list.length} · ${fmtBytes(total)}` : '';
  $('#lora-list').innerHTML = list.length ? list.map((l,i)=>`
    <div class="lora-row">
      <div class="grow" style="min-width:0">
        <b>${esc(l.name)}</b>${l.trigger_word?` <code>${esc(l.trigger_word)}</code>`:''}
        <div class="muted">${l.files.length} file${l.files.length===1?'':'s'} · ${fmtBytes(l.bytes)}${l.catalogue?' · '+esc(l.catalogue):''}</div>
      </div>
      <button class="lora-x" data-del-lora="${i}" title="Delete">✕</button>
    </div>`).join('')
    // Both ways in are directly above this card, so the empty state points at
    // them rather than saying "nothing here" and leaving you to find them.
    : '<p class="muted" style="margin:2px 0 0">Nothing in <code>loras/</code> yet — train one, or paste a Drive link above.</p>';
  $$('[data-del-lora]').forEach(b=>b.onclick=()=>deleteLora(list[+b.dataset.delLora]));
}

// The dialog is the entire safety net: the route unlinks, and there is nothing
// behind it. So it says how much is going, and it says whether it can come back
// — those are two different sentences, because a Wan speed pair is a download
// and a LoRA you trained is however many hours that run took.
async function deleteLora(l){
  const n=l.files.length;
  if(!confirm(`Permanently delete “${l.name}”?\n\n`
    + `${n} file${n===1?'':'s'} (${fmtBytes(l.bytes)}) unlinked from the volume.\n`
    + (l.catalogue
        ? `It is part of ${l.catalogue} in the catalogue below, so it can be downloaded again.`
        : `This cannot be undone.`))) return;
  const r=await post('/api/loras/delete',{path:l.root});
  if(r.error){ errInto('#lora-err',r.error); return }
  errInto('#lora-err','');
  // The whole sheet, not just this list. Deleting a catalogue LoRA moves it back
  // to "missing" in the cards below and, for the identity-edit weight, takes the
  // two plate tiles off the strip — all of which loadState() already knows how
  // to do. Redrawing only the row that went would leave the page claiming a
  // weight it no longer has.
  loadState();
}
$('#tok-save').onclick=async()=>{
  const r=await post('/api/token',{hf_token:$('#tok').value});
  $('#tok').value=''; $('#tok-state').innerHTML=r.hf_token_set?'<span class="ok">Token saved.</span>':'<span class="warn">Cleared.</span>';
};
// Downloads are one-at-a-time, and the page says so by taking the other buttons
// away rather than by answering a press with a red message. The backend is
// idempotent regardless — this is what stops the question being asked, not what
// answers it, and the two disagree only for the moment a click is in flight.
// Deliberately not a loadState() call to re-enable. Rebuilding the model list
// would wipe the row the outcome was just written into, so "Cancelled." and a
// failure reason would both survive exactly as long as it took to re-render.
// Only a completed download changes what the list says, so only a completed
// download reloads it.
function downloadsBusy(on){
  $$('[data-dl],[data-dl-fam]').forEach(b=>{ b.disabled=on });
}

// One follower for every queued download. "This family's missing weights" and
// any other list differ only in which keys the server put in the queue — same
// record, same phases, same Cancel — so they get one loop rather than one each.
function followQueue(jobId, box, doneText){
  const bar=box.querySelector('i'), msg=box.querySelector('p');
  wireCancel(box, jobId);
  const t=everyMs(async()=>{
    const s=await api('/api/status/'+jobId);
    bar.style.width=(s.percent||0)+'%';
    msg.textContent=[s.phase||'Downloading…',s.mb_s&&`${s.mb_s} MB/s`].filter(Boolean).join(' · ');
    if(s.status==='completed'){ clearInterval(t); bar.style.width='100%';
      msg.innerHTML='<span class="ok">'+doneText+'</span>'; downloadsBusy(false); loadState(); }
    else if(s.status==='stopped'){ clearInterval(t);
      // Names what did land. A queue stopped four files in has done real work,
      // and "Cancelled." alone throws away the only record of which four.
      msg.textContent='Cancelled'+((s.downloaded||[]).length?` — ${s.downloaded.length} downloaded, ${(s.remaining||[]).length} not`:'.');
      downloadsBusy(false); }
    else if(s.status==='failed'){ clearInterval(t);
      msg.innerHTML='<span class="err">'+esc(s.error||'Download failed')+'</span>'; downloadsBusy(false); }
  },3000);
}

async function startFamilyDownload(fam, btn){
  const box=btn.closest('.fam').querySelector('.fam-prog'), msg=box.querySelector('p');
  downloadsBusy(true);
  box.classList.remove('hide'); box.querySelector('i').style.width='0%';
  msg.textContent='Starting…';
  // Token rides along, same as the old catalogue-wide button: pasting a key and
  // pressing Download is one action, and the gated families are exactly the
  // ones you paste it for.
  const r=await post('/api/download-missing',{family:fam.name,hf_token:$('#tok').value});
  $('#tok').value='';
  if(r.error){ msg.innerHTML='<span class="err">'+esc(r.error)+'</span>'; downloadsBusy(false); return }
  if(!r.job_id){ msg.textContent=r.note||'Nothing missing.'; downloadsBusy(false); loadState(); return }
  if(!r.mine){ msg.textContent=(r.busy_with||'Another download')+' is running.'; downloadsBusy(false); return }
  followQueue(r.job_id, box, `${fam.name} downloaded.`);
}
// ---------- Google Drive ----------
// Same poll loop as the weights above, against the same job record. The one
// thing it says that they do not is which files landed: a Drive folder's
// contents are not knowable up front, so "3 files · 0.4 GB" is the only
// confirmation that what arrived is what you meant to send.
$('#gd-go').onclick=async()=>{
  const b=$('#gd-go'), url=$('#gd-url').value.trim();
  const box=$('#gd-prog'), msg=box.querySelector('p');
  if(!url){ $('#gd-url').focus(); return }
  b.disabled=true; box.classList.remove('hide');
  msg.textContent='Starting…';
  const r=await post('/api/gdrive',{url,folder:$('#gd-folder').value.trim()});
  if(r.error){ msg.innerHTML='<span class="err">'+esc(r.error)+'</span>'; b.disabled=false; return }
  wireCancel(box,'dl_gdrive');
  const t=everyMs(async()=>{
    const s=await api('/api/status/dl_gdrive');
    if(s.status==='completed'){
      clearInterval(t); b.disabled=false;
      $('#gd-url').value=''; $('#gd-folder').value='';
      const n=(s.files||[]).length;
      msg.innerHTML='<span class="ok">'+esc(`${n} file${n===1?'':'s'} · ${s.size_gb} GB — `
        +(s.files||[]).join(', '))+'</span>'
        +((s.skipped||[]).length?'<br><span class="muted">Skipped: '+esc(s.skipped.join(', '))+'</span>':'');
      // The picker reads loraIndex, which only a state refresh rebuilds — so
      // without this the LoRA you just pulled is on the volume and untypeable.
      loadState();
    } else if(s.status==='stopped'){
      clearInterval(t); b.disabled=false;
      msg.textContent='Cancelled.';
    } else if(s.status==='failed'){
      clearInterval(t); b.disabled=false;
      msg.innerHTML='<span class="err">'+esc(s.error||'Download failed')+'</span>';
    } else {
      msg.textContent=[s.phase||'Downloading…',s.mb_s&&`${s.mb_s} MB/s`].filter(Boolean).join(' · ');
    }
  },3000);
};
$('#gd-url').addEventListener('keydown',e=>{ if(e.key==='Enter') $('#gd-go').click() });
$('#gd-folder').addEventListener('keydown',e=>{ if(e.key==='Enter') $('#gd-go').click() });

async function startDownload(key,btn){
  const el=$('#dl-'+key); el.textContent='Starting…';
  const cancelBtn=document.querySelector('[data-dl-cancel="'+key+'"]');
  downloadsBusy(true);
  const r=await post('/api/download',{key});
  // Unknown key is still an error; being busy is not one any more. `mine` false
  // means something else holds the uplink, which the disabled buttons should
  // have prevented — so it is a plain sentence on this row, not a red box.
  if(r&&r.error){ el.innerHTML='<span class="err">'+esc(r.error)+'</span>'; downloadsBusy(false); return }
  if(r&&!r.mine){ el.textContent=(r.busy_with||'Another download')+' is running.'; downloadsBusy(false); return }
  if(cancelBtn){
    cancelBtn.classList.remove('hide'); cancelBtn.disabled=false; cancelBtn.textContent='Cancel';
    cancelBtn.onclick=async()=>{ cancelBtn.disabled=true; cancelBtn.textContent='Stopping…'; await post('/api/stop/dl_'+key); };
  }
  const done=()=>{ if(cancelBtn)cancelBtn.classList.add('hide'); downloadsBusy(false); };
  const t=everyMs(async()=>{
    const s=await api('/api/status/dl_'+key);
    if(s.status==='completed'){clearInterval(t);el.innerHTML='<span class="ok">Done</span>';done();loadState();}
    else if(s.status==='stopped'){clearInterval(t);el.textContent='Cancelled.';done();}
    else if(s.status==='failed'){clearInterval(t);el.innerHTML='<span class="err">'+esc(s.error||'Failed')+'</span>';done();}
    // The rate, not just the phase. "Downloading…" is true of a transfer moving
    // at 90 MB/s and of one that has not moved a byte in four minutes, and
    // telling those apart without opening the Modal logs is the whole reason
    // the job publishes a byte count at all.
    else el.textContent=[s.phase||'Downloading…',s.mb_s&&`${s.mb_s} MB/s`].filter(Boolean).join(' · ');
  },3000);
}

// ==================== DATASETS ====================
// A dataset is a named folder; the editor is a view onto it. Nothing here is
// tied to a training run, which is the whole point of the section.
//
// A set you drop is a draft until you save it. Drafts are tied to this window:
// sessionStorage, not localStorage, because it survives a reload and dies with
// the tab — which is exactly the lifetime an unsaved thing should have. The
// heartbeat is what the server reads as "still open"; without it a draft is
// deleted once the grace period passes.
const SESSION=(()=>{
  let s=null;
  try{ s=sessionStorage.getItem('vis-session') }catch{}
  if(!s){
    s='s'+Math.random().toString(36).slice(2,10)+Date.now().toString(36);
    try{ sessionStorage.setItem('vis-session',s) }catch{}
  }
  return s;
})();
const beat=()=>post('/api/session',{session:SESSION});
let beatTimer=null;
// Only while there is something to keep alive. A ping reloads and commits the
// volume, and a tab left open on Generate for a day would do that seven hundred
// times to protect nothing — the sweep it drives also runs whenever the list is
// read, which is the moment it matters.
function keepAlive(on){
  if(on && !beatTimer) beatTimer=setInterval(()=>{ if(!document.hidden) beat() },120000);
  if(!on && beatTimer){ clearInterval(beatTimer); beatTimer=null }
}
// One on load whatever the state: it registers this window and sweeps out the
// drafts of the one you closed.
beat();
// A tab that was hidden for longer than the grace period has to say it is back
// before anything reads the list, or its own drafts look abandoned.
document.addEventListener('visibilitychange',()=>{ if(!document.hidden&&beatTimer) beat() });

let dsName=null, dsSaved=false, dsImages=[], dsInsight=null, dsFilter='all',
    dsDensity=2, capPoll=null;

async function loadDatasets(){
  const r=await api('/api/datasets');
  if(r.error){ errInto('#ds-err',r.error); return }
  const list=r.datasets||[];
  const card=d=>`
    <div class="ds-row">
      <button class="ds-card${d.name===dsName?' sel':''}${d.saved?'':' draft'}" data-open="${esc(d.name)}">
        ${d.cover
          ? `<img class="ds-cover" loading="lazy" src="/api/thumb/${encodeURIComponent(d.name)}/${encodeURIComponent(d.cover)}" alt="">`
          : '<div class="ds-cover empty">▤</div>'}
        <div class="ds-meta">
          <b>${esc(d.name)}</b>
          <div class="muted" style="margin-top:3px;font-size:12px">
            ${d.count} image${d.count===1?'':'s'}${d.uncaptioned?` · ${d.uncaptioned} uncaptioned`:''}
          </div>
        </div>
      </button>
      <button class="ds-x" data-del="${esc(d.name)}" title="Delete">×</button>
    </div>`;
  // Drafts first: they are the set you are working on, and the reason they are
  // labelled at all is that the label is a promise about what happens to them.
  const drafts=list.filter(d=>!d.saved), saved=list.filter(d=>d.saved);
  $('#ds-list').innerHTML =
    (drafts.length ? `<p class="ds-group">Unsaved <span>· cleared when you close the app</span></p>`
      + drafts.map(card).join('') : '')
    + (saved.length ? (drafts.length ? `<p class="ds-group">Saved</p>` : '')
      + saved.map(card).join('') : '');
  $('#ds-empty').textContent = list.length ? '' : '';
  $$('#ds-list [data-open]').forEach(b=>b.onclick=()=>openDataset(b.dataset.open));
  $$('#ds-list [data-del]').forEach(b=>b.onclick=()=>deleteSet(b.dataset.del));
  keepAlive(drafts.length>0);
  trainDatasets=list;
  const open=list.find(d=>d.name===dsName);
  dsSaved=!!(open&&open.saved);
  renderSetBar();
  syncTrainDataset();
}

// Back to the drop target. Nothing is destroyed — the set stays in the rail.
$('#ds-fresh').onclick=()=>{ dsName=null; dsSaved=false; showSheet(false); loadDatasets() };
$('#ds-add').onclick=()=>$('#files').click();

function showSheet(on){
  $('#ds-sheet').classList.toggle('hide',!on);
  $('#drop').classList.toggle('hide',on);
  syncTrainDataset();
}

// A name that is free to use, so dropping never stops to ask for one. A zip
// carries a name worth keeping; loose images do not, so they get a numbered
// one. Either way it is provisional — this is the draft's handle, and the name
// that lasts is the one typed into Save.
function suggestName(files){
  const zip=[...(files||[])].find(f=>/\.zip$/i.test(f.name));
  const raw=zip ? zip.name.replace(/\.zip$/i,'') : '';
  const clean=raw.replace(/[^A-Za-z0-9_-]/g,'_').slice(0,64);
  if(clean && !trainDatasets.some(d=>d.name===clean)) return clean;
  let n=1; while(trainDatasets.some(d=>d.name==='set_'+n)) n++;
  return 'set_'+n;
}

// One control, two states: a saved set is titled, a draft is named.
function renderSetBar(){
  $('#ds-title').textContent = dsSaved ? (dsName||'') : '';
  $('#ds-title').classList.toggle('hide',!dsSaved);
  $('#ds-name-wrap').classList.toggle('hide',dsSaved);
  $('#ds-save').classList.toggle('hide',dsSaved);
  // A zip dropped as `portraits.zip` already suggested `portraits`; carrying it
  // in means Save is usually one click rather than one click and some typing.
  if(!dsSaved && !$('#ds-save-name').value) $('#ds-save-name').value=dsName||'';
}

async function openDataset(name){
  dsName=name; dsFilter='all';
  dsSaved=!!(trainDatasets.find(d=>d.name===name)||{}).saved;
  $('#ds-save-name').value='';
  showSheet(true);
  renderSetBar();
  errInto('#ds-edit-err','');
  $$('#ds-list [data-open]').forEach(b=>b.classList.toggle('sel',b.dataset.open===name));
  await loadTiles();
}

async function loadTiles(){
  if(!dsName) return;
  const d=await api('/api/datasets/'+encodeURIComponent(dsName));
  if(d.error){ errInto('#ds-edit-err',d.error); return }
  dsImages=d.images||[];
  dsSaved=!!d.saved;
  renderSetBar();
  if(!$('#ds-trig').value) $('#ds-trig').value=d.trigger_word||'';
  await loadInsight();
  renderTiles();
}

// ---------- keep, or cull ----------
$('#ds-save').onclick=async()=>{
  const b=$('#ds-save'), name=$('#ds-save-name').value.trim();
  if(!name){ errInto('#ds-edit-err','Give the set a name to save it.'); $('#ds-save-name').focus(); return }
  b.disabled=true;
  const r=await post('/api/datasets/'+encodeURIComponent(dsName)+'/save',{name});
  b.disabled=false;
  if(r.error){ errInto('#ds-edit-err',r.error); return }
  errInto('#ds-edit-err','');
  await loadDatasets();
  await openDataset(r.name);
};
$('#ds-save-name').onkeydown=e=>{ if(e.key==='Enter'){ e.preventDefault(); $('#ds-save').click() } };

// Confirm on anything with images in it, saved or not. A draft is disposable by
// design, but forty files you just waited on an upload for are not, and the
// dashed border does not make a stray click on a small round × any less easy.
// The dialog is now the whole safety net — there is no .trash behind it — so it
// says how much is going and that it is not coming back.
async function deleteSet(name){
  const d=trainDatasets.find(x=>x.name===name)||{};
  if(d.count && !confirm(`Permanently delete “${name}”?\n\n`
    + `${d.count} image${d.count===1?'':'s'} and their captions are unlinked `
    + `from the volume. This cannot be undone.`)) return;
  const r=await post('/api/datasets/'+encodeURIComponent(name)+'/delete');
  if(r.error){ errInto('#ds-err',r.error); return }
  if(name===dsName){ dsName=null; dsSaved=false; showSheet(false) }
  await loadDatasets();
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
        <button class="rm" data-rm="${esc(i.name)}" title="Delete">×</button>
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
  if(!d){ $('#ins-body').innerHTML=''; $('#ins-summary').textContent=''; return }
  $('#ins-summary').textContent =
    (d.trigger_word && d.captioned ? `${d.with_trigger}/${d.captioned} trigger · ` : '')
    + `${d.captioned}/${d.images} captioned`;
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

// Refilled only when the served table changes. Unlike the LoRA index these two
// hold a selection, and a rebuild on every poll would put the preset you picked
// back to General somewhere between choosing it and pressing Caption.
function fillCapSel(sel,rows,def){
  const sig=rows.map(r=>r.key).join('|');
  if(sel.dataset.sig===sig) return;
  sel.dataset.sig=sig;
  const keep=sel.value;
  sel.innerHTML=rows.map(r=>`<option value="${esc(r.key)}">${esc(r.label)}</option>`).join('');
  sel.value=rows.some(r=>r.key===keep)?keep:(def||(rows[0]||{}).key||'');
}
// What the two words cannot say: which half of the picture a preset throws away,
// and that the second captioner is a download. Both come from the server, so the
// note and the instruction behind it can never disagree.
function capNote(){
  const p=capPresets.find(r=>r.key===$('#cap-preset').value);
  const m=capModels.find(r=>r.key===$('#cap-model').value);
  $('#cap-note').textContent=[p&&p.note,m&&m.note].filter(Boolean).join(' · ');
}
$('#cap-preset').onchange=capNote;
$('#cap-model').onchange=capNote;

$('#do-caption').onclick=async()=>{
  const btn=$('#do-caption'); btn.disabled=true;
  const box=$('#cap-prog'); box.classList.remove('hide');
  errInto('#ds-edit-err','');
  const r=await post('/api/caption',{dataset:dsName,trigger_word:$('#ds-trig').value.trim(),
    preset:$('#cap-preset').value,model:$('#cap-model').value,
    length:$('#cap-len').value,overwrite:$('#cap-over').checked});
  if(r.error){ errInto('#ds-edit-err',r.error); btn.disabled=false; box.classList.add('hide'); return }
  clearInterval(capPoll);
  capPoll=everyMs(async()=>{
    const s=await api('/api/status/'+r.job_id);
    box.querySelector('i').style.width=(s.percent||0)+'%';
    // Named while it loads. The first minute of a run against a captioner that
    // is not in the cache yet is a 17 GB pull, and a bare "Loading captioner…"
    // for twenty minutes is indistinguishable from a hang.
    box.querySelector('p').textContent=s.step?`Captioning ${s.step}/${s.total_steps}`
      :`Loading ${s.model_label||'captioner'}…`;
    // Refresh mid-run so captions land visibly rather than all at the end.
    if(s.step&&s.step%5===0) loadTiles();
    if(s.status==='completed'){ clearInterval(capPoll); box.classList.add('hide');
      btn.disabled=false; loadTiles();
      // A refusal wrote no file, so those images are still Uncaptioned and the
      // run otherwise looks like it worked. Say how many, and say the one thing
      // that fixes it — the other captioner is a menu away.
      if(s.refused) errInto('#ds-edit-err',
        `${s.refused} image${s.refused===1?'' :'s'} the captioner declined to describe`
        +(s.model==='qwen3vl'?'. Try the uncensored captioner.':'.'));
    }
    else if(s.status==='failed'){ clearInterval(capPoll); box.classList.add('hide');
      btn.disabled=false; errInto('#ds-edit-err',s.error||'Captioning failed'); }
  },2500);
};

// ---------- upload (fires immediately on drop; no prerequisites) ----------
const drop=$('#drop'), fin=$('#files'), dsSheet=$('#ds-sheet');
let uploading=false;
drop.onclick=()=>{ if(!uploading) fin.click() };
drop.ondragover=e=>{e.preventDefault();drop.classList.add('hot')};
drop.ondragleave=()=>drop.classList.remove('hot');
drop.classList.add('can-drop');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('hot');upload(e.dataTransfer.files)};
// The open sheet takes a drop too, and `upload` appends whenever a set is open.
// Adding to a set you already have was otherwise reachable only through the
// + Images button, so the obvious gesture did nothing — or worse, dropped the
// file into a caption box, which is what the browser does by default.
dsSheet.ondragover=e=>{e.preventDefault();dsSheet.classList.add('hot')};
dsSheet.ondragleave=e=>{ if(!dsSheet.contains(e.relatedTarget)) dsSheet.classList.remove('hot') };
dsSheet.ondrop=e=>{e.preventDefault();dsSheet.classList.remove('hot');upload(e.dataTransfer.files)};
fin.onchange=()=>{ upload(fin.files); fin.value=''; };   // reset so the same file can be re-picked

async function upload(list){
  const keep=[...list].filter(f=>/\.(png|jpe?g|webp|bmp|avif|zip|txt)$/i.test(f.name));
  if(!keep.length||uploading) return;
  // Dropping onto the empty screen is how most sets begin, so the set is
  // created here rather than being a thing you had to make first. It starts as
  // a draft: nothing you drop is committed to the library until you say so.
  if(!dsName){
    const name=suggestName(keep);
    const r=await post('/api/datasets',{name,session:SESSION});
    if(r.error){ errInto('#ds-err',r.error); return }
    await loadDatasets();
    await openDataset(name);
  }
  uploading=true; errInto('#ds-edit-err','');
  const box=$('#up-prog'); box.classList.remove('hide'); const bar=box.querySelector('i');
  $('#drop-title').textContent='Uploading…';
  $('#drop-sub').textContent=`${keep.length} file${keep.length>1?'s':''}`;

  const fd=new FormData();
  keep.forEach(f=>fd.append('files',f,f.name));
  fd.append('dataset',dsName);
  fd.append('session',SESSION);

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
    // The rail as well as the sheet: the image count Start training reads
    // lives in the list, so refreshing only the tiles left the button disabled
    // on a set you were looking at the contents of.
    await loadTiles();
    await loadDatasets();
  };
  x.onerror=()=>{ uploading=false; box.classList.add('hide');
    $('#drop-sub').textContent='';
    errInto('#ds-edit-err','Network error during upload.'); };
  x.send(fd);
}

// ==================== TRAIN ====================
// Train no longer owns a dataset; it picks one.
// Running or composing: the console shows one or the other, and the sheet
// above it is untouched either way.
const show=s=>{
  $('#step-run').classList.toggle('hide',s!=='run');
  $('#t-console').querySelector('.opts').classList.toggle('hide',s==='run');
  $('#train-adv').classList.toggle('hide',s==='run'||!$('#t-toggle-adv').classList.contains('on'));
};
let trainDatasets=[];

// The set you are looking at is the set you train. There is no second place
// to choose one, so there is nothing that can disagree with the contact sheet.
function syncTrainDataset(){
  const d=trainDatasets.find(x=>x.name===dsName);
  if(d){
    if(d.trigger_word && !$('#ltrig').value) $('#ltrig').value=d.trigger_word;
    // Only a saved set lends its name. A draft's handle is `set_3`, which is a
    // placeholder standing in for a name you have not chosen yet — proposing it
    // as the LoRA's name would turn it into one by default.
    if(d.saved && !$('#lname').value) $('#lname').value=d.name;
  }
  checkTrainReady();
}

function checkTrainReady(){
  const d=trainDatasets.find(x=>x.name===dsName);
  const ok=!!d&&d.count>0&&$('#lname').value.trim()&&$('#ltrig').value.trim();
  $('#go-train').disabled=!ok;
  $('#train-hint').textContent =
    !dsName ? 'Drop images, or pick a set' :
    !d||!d.count ? 'This set is empty' :
    (!$('#lname').value.trim()||!$('#ltrig').value.trim()) ? 'Name it and set a trigger word' : '';
}
$('#t-toggle-adv').onclick=()=>{
  $('#t-toggle-adv').classList.toggle('on',!$('#train-adv').classList.toggle('hide'));
};
$('#ins-toggle').onclick=()=>{
  $('#ins-toggle').classList.toggle('on',!$('#ins-panel').classList.toggle('hide'));
};
document.addEventListener('input',e=>{ if(e.target.id==='lname'||e.target.id==='ltrig') checkTrainReady() });

let trainJob=null;
$('#go-train').onclick=async()=>{
  $('#train-err').innerHTML='';
  const r=await post('/api/train',{dataset:dsName,lora_name:$('#lname').value.trim(),
    trigger_word:$('#ltrig').value.trim(),network_dim:$('#a-dim').value,network_alpha:$('#a-alpha').value,
    max_train_epochs:$('#a-epochs').value,learning_rate:$('#a-lr').value,resolution:$('#a-res').value,
    num_repeats:$('#a-rep').value,batch_size:$('#a-bs').value,seed:$('#a-seed').value});
  if(r.error){$('#train-err').innerHTML='<div class="err-box">'+r.error+'</div>';return}
  trainJob=r.job_id;
  show('run'); $('#run-done').classList.add('hide');
  poll=everyMs(pollTrain,3000); pollTrain();
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
// Only the thing the Sampling button cannot say. It used to print "8 steps ·
// CFG 1.0" beside a drawer that held those two numbers; now the button itself
// resolves and shows them, and a second copy in the strip would be the same
// fact twice — which is what this line was originally added to avoid.
function syncModelLine(){
  $('#gen-model-line').textContent = $('#g-model').value ? '' : 'No model downloaded';
  paintSampling();
}
$('#g-model').onchange=()=>{ syncModelLine(); syncNeg() };

// ---------- LoRAs, written in the prompt ----------
// A stack of rows was the wrong shape for this. Each row cost 56px of vertical
// space plus a wrapped select, so four LoRAs took 380px off the canvas — the one
// dimension the whole layout exists to protect — to hold four filenames and
// eight digits. And the row could not say the thing that actually matters,
// which is where in the sentence the LoRA applies.
//
// So they live in the prompt, in Automatic1111's syntax, because that is the
// notation anyone who has trained these models already types:
//
//     a portrait <lora:k3nan:0.4> in soft window light <lora:alxcn:1>
//
// Strength is optional and defaults to 1. A second number is the text encoder
// weight, which the backend already defaults to the UNet weight when omitted —
// so `<lora:x:0.8>` sets both, which is what one number has always meant here.
// On the video side the third field is the expert instead,
// because that is what Wan's A14B pair needs and a text-encoder weight is not a
// thing its model-only loader has.
//
// Cost to the canvas: zero. Four LoRAs are four words in a box that was
// already there.
let loraIndex=[];
// One capture and a split, rather than three optional groups: the field count
// varies by family and a regex that encodes that is a regex that has to change
// every time a family does.
const LORA_RE=/<lora:([^<>]*)>/gi;

// Resolution is by path under loras/ first, then by bare filename when that is
// unambiguous. The picker writes whichever of the two is unique, so a typed
// `<lora:k3nan:1>` works while two files called `high.safetensors` in different
// folders still have to be told apart — which is the exact case the video side
// hits with the matched speed pairs.
//
// Case is part of a filename, not noise. This folded it away before comparing,
// so `K3nan.safetensors` and `k3nan.safetensors` — two real files, two real
// LoRAs, which is exactly what the volume holds after a Drive pull and a
// training run disagree about capitalisation — collided into one ambiguous
// name. The result was not "picked the wrong one", it was worse: neither
// resolved, so both went untypeable and the note blamed a missing file for a
// file that is sitting right there. Nothing in the backend folds case; it
// addresses LoRAs by exact path, and ComfyUI validates them against a directory
// listing, so this was the only place on the path where two distinct files
// became one name.
//
// Exact first, then case-insensitively and only while that still points at one
// file: typing lowercase keeps working on every volume that does not actually
// hold a collision, and on one that does, the exact spelling always wins.
function resolveLora(name){
  const raw=String(name||'').trim().replace(/\.safetensors$/i,'');
  if(!raw) return null;
  const n=raw.toLowerCase();
  const exactRel=loraIndex.filter(l=>l.rel===raw);
  if(exactRel.length) return exactRel[0];
  const exactStem=loraIndex.filter(l=>l.stem===raw);
  if(exactStem.length===1) return exactStem[0];
  const byRel=loraIndex.filter(l=>l.rel.toLowerCase()===n);
  if(byRel.length===1) return byRel[0];
  const byStem=loraIndex.filter(l=>l.stem.toLowerCase()===n);
  return byStem.length===1 ? byStem[0] : null;
}

// Why a name did not resolve, which is a different question from whether it
// did. `<lora:high:1>` against a volume holding both Wan speed pairs is not a
// missing file — it is two files and no way to tell which — and sending you to
// look for a LoRA that is sitting right there is the worse of the two wrong
// answers.
//
// This one stays case-insensitive, and that is not an oversight left behind by
// the fix above: it only ever runs on a name that already failed to resolve, so
// its job is near-misses. A mistyped `<lora:K3NAN:1>` on a volume holding both
// casings answers "use K3nan or k3nan", which is the message that makes the
// difference between the two files visible.
function loraAlternatives(name){
  const n=String(name||'').trim().replace(/\.safetensors$/i,'').toLowerCase();
  return loraIndex.filter(l=>l.stem.toLowerCase()===n).map(l=>l.token);
}

// Every token in the text, with the offsets the ⌘↑/⌘↓ handler needs to rewrite
// one in place. Unresolved names are kept rather than dropped: a LoRA that
// silently does nothing is indistinguishable from a LoRA with no effect, which
// is the failure this whole file is written to avoid.
function parseLoras(text){
  const out=[];
  LORA_RE.lastIndex=0;
  let m;
  while((m=LORA_RE.exec(text))){
    const parts=m[1].split(':').map(s=>s.trim());
    out.push({
      start:m.index, end:m.index+m[0].length,
      name:parts[0]||'', a:parts[1]??'', b:parts[2]??'',
      hit:resolveLora(parts[0]),
    });
  }
  return out;
}

// What the text encoders see. The tokens are markup for this page, not
// language — leaving them in would have the model rendering the word "lora".
// Takes its text as an argument because region rows use the same syntax in
// their own field, and a second copy of this regex dance is a second place for
// the punctuation cleanup to drift.
function stripLoras(text){
  return text.replace(LORA_RE,' ').replace(/\s+/g,' ')
    .replace(/\s+([,.;:!?])/g,'$1').trim();
}
function promptText(){ return stripLoras($('#prompt').value) }

const loraNum=(v,d)=>{ const n=parseFloat(v); return Number.isFinite(n) ? n : d };

function readLoras(){
  return parseLoras($('#prompt').value).filter(t=>t.hit)
    .slice(0,window.MAX_LORAS||6).map(t=>({
      path:t.hit.path,
      unet:loraNum(t.a,1),
      // Left null on purpose when unwritten: the backend defaults the text
      // encoder weight to the UNet weight, so omitting it is a decision the
      // client does not have to duplicate — and duplicating it would freeze
      // today's default into every prompt ever saved.
      text_encoder:t.b===''?null:loraNum(t.b,null),
    }));
}

function readVidLoras(){
  const sup=(videoModel()||{supports:{}}).supports;
  if(!sup.loras) return [];
  return parseLoras($('#prompt').value).filter(t=>t.hit)
    .slice(0,window.MAX_LORAS||6).map(t=>({
      path:t.hit.path, unet:loraNum(t.a,1), expert:vidExpert(t),
    }));
}

// The matched speed pairs are named `high` and `low` inside one folder, so the
// file already says which expert it belongs to. Reading it beats making you
// write the same fact twice, and beats the silent quality loss of crossing them.
function vidExpert(t){
  const valid=window.WAN_EXPERTS||['both','high','low'];
  const b=String(t.b||'').toLowerCase();
  if(valid.includes(b)) return b;
  const n=t.hit.rel.toLowerCase();
  if(/(^|\/)high|high_noise/.test(n)) return 'high';
  if(/(^|\/)low|low_noise/.test(n)) return 'low';
  return 'both';
}

// The picker. Discovery only — you cannot type a syntax you have never seen —
// after which the prompt is the stack and this button is a shortcut.
function loraMenu(btn){
  if(!loraIndex.length) return;
  // Ticked means "already in the prompt", and the tick is what makes the
  // second click legible as a removal rather than a click that did nothing.
  const on=new Set(parseLoras(caretEl().value).filter(t=>t.hit).map(t=>t.hit.path));
  // Labelled with the token, which is the string this item is about to write
  // into the prompt. It used to be `folder · filename`, which spelled the same
  // word twice for every LoRA training produced — "my_style · my_style
  // .safetensors" — and spelled an extension that is true of every row. The
  // token is already the shortest name that points at one file, so it drops
  // both without losing the one case that needs the folder: the matched Wan
  // speed pairs, whose files are both called `high`.
  openMenu(btn, loraIndex.map(l=>({label:l.token, on:on.has(l.path), run:()=>insertLora(l)})));
}
$('#add-lora').onclick=e=>loraMenu(e.currentTarget);
$('#v-add-lora').onclick=e=>loraMenu(e.currentTarget);

// Where the caret was when a prompt field last had it, and which field that
// was. Clicking + LoRA takes focus away, and the menu takes it again, so by the
// time an item is chosen selectionStart is meaningless — it would put every
// pick at the end of the sentence, which is the one place the syntax makes no
// difference.
//
// Two fields, not one, since a region's prompt takes the same syntax at a
// smaller scope. Tracking which was last touched is what makes "select a box,
// click + LoRA, pick a name" put that character in that box — the shortest path
// this feature has, and the only one that involves no typing at all.
let promptCaret=0, caretTarget='#prompt';
const caretEl=()=>$(caretTarget)||$('#prompt');
['#prompt','#r-prompt'].forEach(sel=>
  ['keyup','click','select','focus','blur'].forEach(ev=>
    $(sel).addEventListener(ev,e=>{
      caretTarget=sel; promptCaret=e.target.selectionStart??0;
    })));

// Picking a LoRA that is already in the prompt takes it out again, rather than
// writing a second copy. Two tokens for one file is not a stack: apply_stack()
// patches both onto the clone chain, so the strengths compound into a number
// nobody chose and the picker looks like it silently did nothing. There is no
// prompt that wants the duplicate, so the second pick is free to mean the
// other thing.
function insertLora(l){
  const el=caretEl();
  const present=parseLoras(el.value).filter(t=>t.hit&&t.hit.path===l.path);
  if(present.length) return removeLora(l,present);
  // No trigger word is written. The picker used to prepend one, and the reason
  // it no longer does is that the right destination stopped being answerable:
  // the token goes where the caret is, but on the regional path the node
  // encodes the main prompt and nothing else — V12's caption compiler, the
  // thing that reads a box's words, only runs inside `_apply_edit_mode`. So a
  // trigger auto-written into a box would silently never reach the encoder,
  // and one auto-written into the main prompt instead would be the picker
  // editing a sentence the caret was nowhere near. Triggers are typed.
  const v=el.value;
  let at=Math.min(promptCaret,v.length);
  // Never inside another token. A pick deliberately leaves the caret on its own
  // strength so ⌘↑ can walk it, and clicking + LoRA again does not move it — so
  // two picks in a row wrote `<lora:alxcn: <lora:my_style:1> 1>`. Nothing on the
  // page says that is malformed; it silently loads one LoRA where you asked for
  // two, which is the failure this whole file exists to avoid.
  const inside=parseLoras(v).find(t=>at>t.start&&at<t.end);
  if(inside) at=inside.end;
  // Never welded to the neighbouring word: `<lora:x:1>` glued to the end of a
  // sentence survives the strip but reads as a typo while you are writing.
  const before=v.slice(0,at), after=v.slice(at);
  // 1.3 in a region, 1 everywhere else. The node pack's guidance for a
  // character LoRA is 1.3–1.4, and a picker that writes the known-weak value
  // into the one field where it is known to be weak is doing the wrong thing
  // quietly. The main prompt is a style stack far more often than a character,
  // so it keeps 1.
  const dflt=el.id==='r-prompt'?'1.3':'1';
  const tok=(before&&!/\s$/.test(before)?' ':'')+`<lora:${l.token}:${dflt}>`
           +(after&&!/^\s/.test(after)?' ':'');
  el.value=before+tok+after;
  // Caret onto the strength, selected, so the next thing you can do is ⌘↑ it
  // or type over it.
  const cur=before.length+tok.indexOf(':'+dflt+'>')+1;
  promptCaret=cur;
  el.focus(); el.setSelectionRange(cur,cur+dflt.length);
  el.dispatchEvent(new Event('input',{bubbles:true}));
  syncLoraNote();
}

function removeLora(l,toks){
  const el=caretEl();
  let v=el.value;
  // Back to front, so each splice leaves the earlier offsets valid.
  [...toks].sort((a,b)=>b.start-a.start).forEach(t=>{
    let s=t.start, e=t.end;
    // Take the space the token was welded to as well, or removing it leaves a
    // double gap mid-sentence. `[^\S\n]` and not `\s`: a prompt written across
    // several lines should not have its line breaks eaten by a picker click.
    if(/[^\S\n]/.test(v[e]||'')) e++;
    else if(/[^\S\n]/.test(v[s-1]||'')) s--;
    v=v.slice(0,s)+v.slice(e);
  });
  // Only the token. The trigger used to be stripped here as well, on the
  // grounds that this function had put it there — nothing puts it there now,
  // so every trigger in the field is something you typed, and a picker click
  // that deletes a word out of your sentence is the worse surprise.
  el.value=v;
  promptCaret=Math.min(promptCaret,v.length);
  el.focus(); el.setSelectionRange(promptCaret,promptCaret);
  el.dispatchEvent(new Event('input',{bubbles:true}));
  syncLoraNote();
}

// ⌘↑ / ⌘↓ on a strength, the way the same chord nudges attention weights in
// Automatic. Put the caret anywhere inside the brackets — the token under it is
// the one that moves, and the selection is restored so a run of presses walks
// the value instead of moving once and losing its place.
function nudgeLora(el,delta){
  const at=el.selectionStart;
  const t=parseLoras(el.value).find(t=>at>=t.start&&at<=t.end);
  if(!t) return false;
  // Clamped, not unbounded: past ±2 a LoRA stops styling the image and starts
  // destroying it, and the clamp is the cheapest place to say so.
  const next=Math.max(-2,Math.min(2,Math.round((loraNum(t.a,1)+delta)*100)/100));
  const body=[t.name,String(next)].concat(t.b?[t.b]:[]).join(':');
  const tok=`<lora:${body}>`;
  el.value=el.value.slice(0,t.start)+tok+el.value.slice(t.end);
  const num=t.start+tok.indexOf(':'+next)+1;
  el.setSelectionRange(num,num+String(next).length);
  // So a region's box redraws and its Strength cell follows: both are driven by
  // the token, and this is the one place that rewrites it without an keystroke.
  el.dispatchEvent(new Event('input',{bubbles:true}));
  syncLoraNote();
  return true;
}
// Both fields, because both hold the same syntax at different scopes and a
// strength you can walk in one but not the other is the kind of asymmetry
// nothing on screen would explain.
['#prompt','#r-prompt'].forEach(sel=>$(sel).addEventListener('keydown',e=>{
  if(!(e.metaKey||e.ctrlKey)||(e.key!=='ArrowUp'&&e.key!=='ArrowDown')) return;
  // Only swallowed when the caret was actually in a token; outside one, ⌘↑ is
  // still the caret-to-top the rest of the OS says it is.
  if(nudgeLora(e.target, e.key==='ArrowUp'?0.05:-0.05)) e.preventDefault();
}));

// Only ever a complaint. What is loaded is legible in the prompt; what is not
// loaded, and why, is the part the prompt cannot show.
function syncLoraNote(){
  // Region fields carry the same syntax, so an unresolvable name typed into a
  // box has to be reported by the same note — it is the one place on the page
  // that says a LoRA names no file, and a region silently rendering without
  // its character is exactly the failure it exists to catch.
  const regionToks=(typeof regions!=='undefined'?regions:[])
    .flatMap(r=>parseLoras(r.prompt||''));
  const toks=parseLoras($('#prompt').value);
  const bad=[...new Set([...toks,...regionToks].filter(t=>!t.hit).map(t=>t.name))];
  const max=window.MAX_LORAS||6;
  const sup=(typeof videoModel==='function'&&videoModel()||{supports:{}}).supports;
  const bits=[];
  // Ambiguous first and named individually: the fix is to copy one of the
  // alternatives, so listing them is the whole message.
  const vague=bad.filter(b=>loraAlternatives(b).length>1);
  const gone=bad.filter(b=>!loraAlternatives(b).length);
  vague.forEach(b=>bits.push(
    `"${b}" names ${loraAlternatives(b).length} LoRAs — use ${loraAlternatives(b).join(' or ')}.`));
  if(gone.length){
    bits.push(gone.length===1
      ? `No LoRA named "${gone[0]}" on the volume.`
      : `No LoRAs named ${gone.map(b=>`"${b}"`).join(', ')} on the volume.`);
  }
  if(toks.filter(t=>t.hit).length>max)
    bits.push(`Only the first ${max} LoRAs are applied.`);
  // One per box is the node's shape, and the backend rejects the rest rather
  // than applying the first — so say it here, while the second token is still
  // under the caret, instead of after a round trip.
  const rgn=(typeof regions!=='undefined'?regions:[]);
  rgn.forEach((r,i)=>{
    if(parseLoras(r.prompt||'').filter(t=>t.hit).length>1)
      bits.push(`Region ${i+1} names more than one LoRA — a region takes one.`);
  });
  // The same LoRA in the prompt and in a box is the one combination that
  // quietly undoes the feature: the prompt copy goes onto the global
  // LoraLoader chain and patches the whole canvas, so the box's mask is still
  // there and no longer separating anything. It looks like regional bleeding
  // rather than like two copies of one LoRA, which is why it has to be named.
  if(typeof regionOn==='function'&&regionOn()){
    const boxed=new Set(rgn.flatMap(r=>parseLoras(r.prompt||'').filter(t=>t.hit).map(t=>t.hit.path)));
    const both=[...new Set(toks.filter(t=>t.hit&&boxed.has(t.hit.path)).map(t=>t.hit.token))];
    both.forEach(n=>bits.push(
      `"${n}" is in the prompt and in a box — the prompt copy applies to the whole canvas and cancels the masking.`));
    // Region weight multiplies every box's own strength, so a zero here is not
    // a weak render — it is every boxed LoRA switched off. The node answers
    // that by returning the model unpatched, and a picture still comes back,
    // placed by the caption alone. That is what earns the line: nothing else on
    // the page tells that render apart from one the LoRAs actually ran in.
    if(parseFloat($('#g-region-base').value)===0&&boxed.size)
      bits.push('Region weight is 0 — every box’s LoRA is switched off.');
  }
  // The prompt is shared by image and video, so a stack typed for one is still
  // sitting there when you switch to the other. Saying that beats letting a
  // guidance-distilled checkpoint quietly ignore four of them.
  if(kind==='video'&&!sup.loras&&toks.some(t=>t.hit))
    bits.push(`${($('#v-model').selectedOptions[0]||{}).text||'This model'} takes no LoRAs — the ones in the prompt are ignored.`);
  $('#lora-note').textContent=bits.join(' ');
}
$('#prompt').addEventListener('input',syncLoraNote);
// The preview is of the whole document, and the typed sentence is most of it —
// so it has to follow the typing. Debounced inside syncShotPeek, and it does
// nothing at all while the disclosure is shut, which is nearly always.
$('#prompt').addEventListener('input',syncShotPeek);
// The weight is the one input that can silence every box without touching a
// token, so the note has to follow it as well as the prompt.
$('#g-region-base').addEventListener('input',syncLoraNote);

// Written back into the prompt, which is now the only place a stack lives. The
// canonical form goes in — full path when the bare name is ambiguous — so a
// reused prompt resolves to the same file the original run used.
function loraTokens(list,video){
  return (list||[]).map(l=>{
    // Image records the stem, video the filename; both match the way the two
    // stacks were read back before this, and an entry whose file is gone is
    // simply not written rather than becoming an unresolvable token.
    const hit=loraIndex.find(x=>video ? x.file===l.name : x.stem===l.name)
           || resolveLora(l.name);
    if(!hit) return '';
    const tail=video
      ? (l.expert&&l.expert!=='both'&&l.expert!==vidExpert({b:'',hit})?':'+l.expert:'')
      : (l.text_encoder!=null&&l.text_encoder!==l.unet?':'+l.text_encoder:'');
    return `<lora:${hit.token}:${l.unet??1}${tail}>`;
  }).filter(Boolean).join(' ');
}

// ---------- size ----------
// The ratio picker and the two pixel boxes are one control, not two that have
// to be kept in agreement. There is only ever a width and a height; the ratios
// are shortcuts to a pair of them, so picking one writes the boxes and typing
// in the boxes selects Custom. Nothing here can hold a size the other half
// disagrees with, because there is only one size.
//
// Custom is not something you choose — it is what the picker says once you have
// typed. It still carries the numbers, so the strip tells the truth about the
// size even with Advanced collapsed, which is the state it will be in most of
// the time.
// 8, the VAE's downscale — the finest grid a pixel size can actually land on.
// It was 16 (VAE 8x, then patch 2 in the DiT), which is the grid the DiT wants
// but not the one it needs: `SingleStreamDiT.forward` pads the latent up to the
// patch size and crops the result back to the unpadded shape, so an odd latent
// dimension costs one row of patches and nothing else. 16 was throwing away
// half the sizes that work. What has to stay true is only that the box never
// claims a size the model will not render, which 8 still guarantees.
const SNAP=8;
const snap=(v,d)=>{
  const n=parseInt(v,10);
  return Number.isFinite(n) ? Math.max(64,Math.min(2048,Math.floor(n/SNAP)*SNAP)) : d;
};
// The seven buckets stay exactly what they were and get multiplied, rather
// than being recomputed from the ratio at a new edge length. Krea 2 inherits
// Qwen-Image's trained buckets, and 1152x896 is one of them where the honest
// arithmetic for 4:3 at a 1024 short edge is 1365x1024 — a size nothing was
// trained on. Scaling a bucket keeps the shape the model knows and changes only
// how much of it there is; deriving one from the ratio would quietly leave the
// distribution at every scale including 1x.
const SIZE_SCALES=[['1','1K'],['1.5','1.5K'],['2','2K']];
const snap8=v=>Math.max(8,Math.round(v/8)*8);
function sizeScale(){ return +($('#g-scale')||{}).value||1 }
function readSize(){
  const a=$('#g-aspect').value;
  // Custom is literal. A number you typed is not a bucket to be multiplied —
  // scaling it would mean the box said 1153 and the model rendered 2306.
  if(a==='custom') return [snap($('#g-w').value,1024), snap($('#g-h').value,1024)];
  const p=a.split('x'), k=sizeScale();
  return [snap8(+p[0]*k), snap8(+p[1]*k)];
}
// Reduced, so Custom can say whether 992×1488 is still the 2:3 you meant.
// Only when it reduces to something a person actually says, though: 992×1024
// is honestly 31:32, and printing that is noise dressed as precision.
const ratio=(w,h)=>{
  const g=(a,b)=>b?g(b,a%b):a, d=g(w,h)||1, a=w/d, b=h/d;
  return (a<=21&&b<=21) ? `${a}:${b}` : '';
};
function syncSize(fromBoxes){
  const sel=$('#g-aspect');
  if(fromBoxes) sel.value='custom';
  const [w,h]=readSize();
  $('#g-w').value=w; $('#g-h').value=h;
  // Only Custom carries its numbers, and only because it has no name that
  // implies them. A preset says "16:9" and the Width/Height boxes say what
  // that is in pixels, updating as you switch — so putting the pixels in the
  // option label too would be the same fact in two places, and the one place
  // it would cost is the strip's width on a laptop.
  const custom=[...sel.options].find(o=>o.value==='custom');
  custom.textContent = sel.value==='custom'
    ? ['Custom',ratio(w,h),`${w}×${h}`].filter(Boolean).join(' · ') : 'Custom';
  // The frame is drawn at the render aspect, so it is wrong the moment that
  // changes — the ratio picker and the frame are one control too. The boxes
  // hold their fractions and get reshaped with it, which is the whole reason
  // they are stored normalised.
  //
  // Guarded on a window flag rather than `typeof drawRegions`, which is the
  // guard used elsewhere in this file and is the wrong one here: drawRegions is
  // a hoisted declaration, so typeof passes during the bootstrap call below —
  // and then it reads `regionOn`, a const whose initializer has not run, and
  // the whole script dies on a temporal-dead-zone error. A property on window
  // is the one thing that is safe to read before anything is initialised.
  if(window.REGIONS_READY) drawRegions();
  paintSizeBtn();
}
// "16:9 · 2304×1296" — the value, so the control needs no name. Custom prints
// the ratio it reduces to when there is one worth saying, which is the one
// thing the pixels alone cannot tell you.
function paintSizeBtn(){
  const [w,h]=readSize(), a=$('#g-aspect').value;
  // The option's own label, not ratio(w,h). These are trained buckets, and a
  // bucket is not its name: 1152x896 reduces to 9:7 and 1344x768 to 7:4. Both
  // are called 4:3 and 16:9 everywhere the models are documented, and printing
  // the honest fraction would rename two presets nobody would recognise —
  // Custom is the only case where the arithmetic is the best name available.
  const name = a==='custom' ? (ratio(w,h)||'Custom')
                            : $('#g-aspect').selectedOptions[0].textContent;
  $('#g-size').textContent=`${name} · ${w}×${h}`;
  $('#g-size').classList.toggle('on', !!menuEl && menuEl.classList.contains('sizer'));
}
$('#g-aspect').onchange=()=>syncSize(false);
$('#g-scale').onchange=()=>syncSize(false);
// The picker and the boxes are one control, so the swap has to be one too:
// 1152×896 flips to a 3:4 that is on the menu and gets selected there, rather
// than to a Custom that spells out a ratio the page could have named. It falls
// through to Custom only when the transpose really has no preset — which,
// since 3:4 landed, means only a size you typed yourself.
function swapSize(){
  // The bucket transposes, not the pixels. Transposing what readSize returns
  // looks for `1152x2016` among options that are all 1x, finds nothing, and
  // drops a perfectly ordinary 9:16 at 1.5K into Custom — so the swap silently
  // cost you the scale and the preset's name. The bucket's own transpose is
  // always on the menu, because the seven exist as transposed pairs.
  const a=$('#g-aspect').value;
  if(a!=='custom'){
    const [bw,bh]=a.split('x');
    const flipped=`${bh}x${bw}`;
    if([...$('#g-aspect').options].some(o=>o.value===flipped)){
      $('#g-aspect').value=flipped; syncSize(false); return;
    }
  }
  const [w,h]=readSize();
  $('#g-w').value=h; $('#g-h').value=w;
  const preset=[...$('#g-aspect').options].some(o=>o.value===`${h}x${w}`);
  if(preset) $('#g-aspect').value=`${h}x${w}`;
  syncSize(!preset);
}
$('#g-swap').onclick=swapSize;
$('#g-size').onclick=e=>openSizer(e.currentTarget);
$('#v-size').onclick=e=>openVidSizer(e.currentTarget);

// The video side's size control. Deliberately its own function rather than a
// `pre` parameter on openSizer: the image side addresses trained pixel buckets
// and a scale multiplier, the video side addresses ratio strings and a tier key
// whose labels come from the chosen model. Same shape on screen and the same
// `.sizer` styles, different vocabulary underneath — folding them together
// would mean one function branching on which of two unrelated things it holds.
function paintVidSize(){
  const b=$('#v-size'); if(!b) return;
  const m=videoModel()||{}, tiers=m.tiers||{};
  const a=$('#v-aspect').value;
  const tier=tiers[$('#v-tier').value]||Object.values(tiers)[0]||'';
  // The tier's own label, which already reads "768p" or "544p draft" — the
  // second word is a fact about the run and belongs on the button.
  b.textContent=[a,tier].filter(Boolean).join(' · ')||'Size';
  b.classList.toggle('on', !!menuEl && menuEl.classList.contains('sizer'));
}
function openVidSizer(btn){
  const el=document.createElement('div'); el.className='menu sizer';
  const draw=()=>{
    const cur=$('#v-aspect').value, curT=$('#v-tier').value;
    const tiles=[...$('#v-aspect').options].map(o=>{
      const [aw,ah]=o.value.split(':').map(Number);
      const long=26, sw=aw>=ah?long:Math.round(long*aw/ah), sh=ah>aw?long:Math.round(long*ah/aw);
      return `<button class="ar${o.value===cur?' on':''}" data-ar="${esc(o.value)}">
        <i style="width:${sw}px;height:${sh}px"></i><b>${esc(o.textContent)}</b></button>`;
    }).join('');
    // Read off the select rather than the model, because syncVideoModel has
    // already rebuilt it for this checkpoint — asking the model again would be
    // a second place deciding which tiers exist.
    const tiers=[...$('#v-tier').options].map(o=>
      `<button class="sc${o.value===curT?' on':''}" data-t="${esc(o.value)}">${esc(o.textContent)}</button>`).join('');
    el.innerHTML=`<div class="ars">${tiles}</div><div class="scales">${tiers}</div>`;
    el.querySelectorAll('[data-ar]').forEach(b=>b.onclick=()=>{
      $('#v-aspect').value=b.dataset.ar;
      $('#v-aspect').dispatchEvent(new Event('change',{bubbles:true}));
      paintVidSize(); draw();
    });
    el.querySelectorAll('[data-t]').forEach(b=>b.onclick=()=>{
      $('#v-tier').value=b.dataset.t; $('#v-tier').dataset.touched='1';
      $('#v-tier').dispatchEvent(new Event('change',{bubbles:true}));
      paintVidSize(); draw();
    });
  };
  draw(); floatBy(btn,el); paintVidSize();
}
$('#g-sampling').onclick=e=>openSampling(e.currentTarget,'g');
$('#v-sampling').onclick=e=>openSampling(e.currentTarget,'v');

// "8 steps · CFG 1.0", resolved rather than defaulted: a typed override shows,
// and a blank box shows what the checkpoint will actually use. The drawer's
// label said "Advanced", which is where something is rather than what it does.
function paintSampling(){
  paintOne('g',(window.KREA2_DEFAULTS||{})[$('#g-model').value]||{});
  if($('#v-sampling')) paintOne('v',(videoModel()||{}).defaults||{});
}
function paintOne(pre,d){
  const btn=$(`#${pre}-sampling`); if(!btn) return;
  const steps=$(`#${pre}-steps`).value||d.steps, cfg=$(`#${pre}-cfg`).value||d.cfg;
  // toFixed(1) because the defaults are 1.0 and 5.5 and the strip has always
  // printed them that way — an unformatted 1 reads as "off" next to a 5.5.
  const bits=[steps&&`${steps} steps`,
              cfg!=null&&cfg!==''&&Number.isFinite(+cfg)&&`CFG ${(+cfg).toFixed(1)}`].filter(Boolean);
  btn.textContent = bits.join(' · ')||'Sampling';
  // Marked when anything is overridden, because the resolved numbers alone
  // cannot say whether you chose them or the checkpoint did — and "why is this
  // 12 steps" is a question you ask days later, off a gallery card.
  const touched=['steps','cfg','shift','switch','sampler','scheduler']
    .some(k=>{ const x=$(`#${pre}-${k}`); return x&&x.dataset.touched==='1' });
  btn.classList.toggle('edited',touched);
}

// The five rarely-changed controls, behind one button. A form rather than a
// palette: a sampler name and a step count are not things a picture of them
// could teach, which is the line the shot tiles are on the other side of.
function openSampling(btn,pre){
  const el=document.createElement('div'); el.className='menu form';
  const row=(lb,html,hint)=>`<label class="frow"><span>${lb}</span>${html}${
    hint?`<i>${hint}</i>`:''}</label>`;
  const opts=(sel,cur)=>[...$(sel).options]
    .map(o=>`<option value="${esc(o.value)}"${o.value===cur?' selected':''}>${esc(o.textContent)}</option>`).join('');
  const id=k=>`#${pre}-${k}`;
  const d = pre==='g' ? ((window.KREA2_DEFAULTS||{})[$('#g-model').value]||{})
                      : ((videoModel()||{}).defaults||{});
  // A row the model does not read is not drawn. The video side already carries
  // that answer in the -wrap classes syncVideoModel maintains, so this asks
  // them rather than re-deriving what a checkpoint supports — two places
  // deciding that is how a control ends up present and ignored.
  const on=k=>{ const w=$(`#${pre}-${k}-wrap`); return !w || !w.classList.contains('hide') };
  const num=(k,lb,ph,step,big,hint)=> on(k) ? row(lb,
    `<input data-s="${id(k)}" inputmode="decimal" data-step="${step}" data-bigstep="${big}"
       placeholder="${ph}" value="${esc($(id(k)).value)}">`, hint) : '';
  el.innerHTML=
      row('Sampler',`<select data-s="${id('sampler')}">${opts(id('sampler'),$(id('sampler')).value)}</select>`)
    + row('Scheduler',`<select data-s="${id('scheduler')}">${opts(id('scheduler'),$(id('scheduler')).value)}</select>`)
    + row('Steps',`<input data-s="${id('steps')}" inputmode="numeric" placeholder="${d.steps??'auto'}"
        value="${esc($(id('steps')).value)}">`)
    + num('cfg','CFG', d.cfg??'auto', '0.1','1')
    + num('shift','Shift', d.shift??(pre==='g'?'1.15':'auto'), pre==='g'?'0.05':'0.1', pre==='g'?'0.5':'1',
        'Bends the noise schedule — higher spends more steps on composition and motion.')
    + (pre==='v' ? num('switch','Expert switch','auto','1','4',
        'Step at which the high-noise expert hands the latent to the low-noise one.') : '')
    + row('Seed',`<input data-s="${id('seed')}" inputmode="numeric" placeholder="random"
        value="${esc($(id('seed')).value)}">`,
        'Blank draws a new one. A seed worth keeping is on the render that used it.')
    + (pre==='g' ? row('Images',
        `<select data-s="#g-n">${opts('#g-n',$('#g-n').value)}</select>`) : '')
    + `<button class="sz-reset" type="button">Reset to the model’s defaults</button>`;
  el.querySelectorAll('[data-s]').forEach(f=>{
    const real=$(f.dataset.s);
    const push=()=>{
      real.value=f.value;
      // `touched` is what stops syncVideoModel-style redraws handing a chosen
      // value back to a default, and the selects already rely on it.
      if(f.value!=='') real.dataset.touched='1'; else delete real.dataset.touched;
      // Both events. `change` is what the selects' own handlers listen for and
      // `input` is what syncNeg listens for — dispatching only change meant
      // typing CFG 5 here left the negative-prompt toggle asleep, which is the
      // one cross-control consequence any of these five numbers has.
      real.dispatchEvent(new Event('input',{bubbles:true}));
      real.dispatchEvent(new Event('change',{bubbles:true}));
      paintSampling();
    };
    f.onchange=push; f.oninput=push;
    f.onkeydown=ev=>{
      if(ev.key!=='ArrowUp'&&ev.key!=='ArrowDown') return;
      if(nudgeNumber(ev.target, ev.key==='ArrowUp'?1:-1, ev.metaKey||ev.ctrlKey)){ ev.preventDefault(); push() }
    };
  });
  el.querySelector('.sz-reset').onclick=()=>{
    ['steps','cfg','shift','switch','seed'].forEach(k=>{
      const x=$(id(k)); if(!x) return;
      x.value=''; delete x.dataset.touched;
    });
    ['sampler','scheduler'].forEach(k=>{ const x=$(id(k)); if(x) delete x.dataset.touched });
    closeMenu(); paintSampling(); syncNeg();
  };
  floatBy(btn,el);
}

// The size popover. Same lifecycle as the shot palette and the LoRA menu —
// floatBy owns the single floating element, the outside-mousedown close, the
// scroll-close and the viewport clamp — so there is one thing on this page that
// knows how a popover behaves.
//
// A view over #g-size-state, never a second copy of it: every click writes into
// the real select or the real inputs and calls syncSize, which is what reuse(),
// the frame and the arrow keys already read. Rebuilt on each change rather than
// patched, because at eight tiles and three scales the diff is more code than
// the redraw and one of them can be wrong.
function openSizer(btn){
  const el=document.createElement('div'); el.className='menu sizer';
  const draw=()=>{
    const a=$('#g-aspect').value, k=String(sizeScale()), [w,h]=readSize();
    const tiles=[...$('#g-aspect').options].filter(o=>o.value!=='custom').map(o=>{
      const [bw,bh]=o.value.split('x').map(Number);
      // Drawn at its own proportions inside a fixed box. A rectangle is the one
      // representation of an aspect ratio that needs no reading, which is the
      // same argument the shot tiles make for a dolly-out.
      const long=26, sw=bw>=bh?long:Math.round(long*bw/bh), sh=bh>bw?long:Math.round(long*bh/bw);
      return `<button class="ar${o.value===a?' on':''}" data-ar="${o.value}" title="${esc(o.textContent)}">
        <i style="width:${sw}px;height:${sh}px"></i><b>${esc(o.textContent)}</b></button>`;
    }).join('');
    el.innerHTML=`<div class="ars">${tiles}</div>
      <div class="scales">${SIZE_SCALES.map(([v,lb])=>
        `<button class="sc${v===k&&a!=='custom'?' on':''}" data-sc="${v}">${lb}</button>`).join('')}</div>
      <div class="sz-custom">
        <label>W<input id="sz-w" inputmode="numeric" value="${w}"></label>
        <button class="sz-swap" title="Swap width and height">${ICON.swap}</button>
        <label>H<input id="sz-h" inputmode="numeric" value="${h}"></label>
      </div>
      <p class="muted sz-note">${a==='custom'?'Custom — snapped to 8, the VAE grid.'
        :`${esc($('#g-aspect').selectedOptions[0].textContent)} at ${esc(SIZE_SCALES.find(x=>x[0]===k)[1])} · ${(w*h/1e6).toFixed(1)} MP`}</p>`;
    el.querySelectorAll('[data-ar]').forEach(b=>b.onclick=()=>{
      $('#g-aspect').value=b.dataset.ar; syncSize(false); draw();
    });
    el.querySelectorAll('[data-sc]').forEach(b=>b.onclick=()=>{
      // Choosing a scale on a custom size is how you get back to a bucket: it
      // has to pick one, or the button would light with nothing behind it.
      if($('#g-aspect').value==='custom') $('#g-aspect').value=nearestBucket(...readSize());
      $('#g-scale').value=b.dataset.sc; syncSize(false); draw();
    });
    const commit=()=>{
      $('#g-w').value=el.querySelector('#sz-w').value;
      $('#g-h').value=el.querySelector('#sz-h').value;
      syncSize(true); draw();
    };
    el.querySelectorAll('#sz-w,#sz-h').forEach(i=>{
      i.onchange=commit;
      // The same chord the strip's numbers take, so a size is nudged the way
      // every other number on this page is: ⌘ steps by 8, the VAE's grid.
      i.onkeydown=ev=>{
        if(ev.key!=='ArrowUp'&&ev.key!=='ArrowDown') return;
        if(nudgeNumber(ev.target, ev.key==='ArrowUp'?1:-1, ev.metaKey||ev.ctrlKey)){
          ev.preventDefault(); commit();
        }
      };
    });
    el.querySelector('.sz-swap').onclick=()=>{ swapSize(); draw() };
  };
  draw();
  floatBy(btn,el);
  paintSizeBtn();
}
// Which trained bucket a typed size is closest to in shape, so leaving Custom
// lands somewhere the model knows rather than on whichever option was selected
// before you started typing.
function nearestBucket(w,h){
  const r=w/h;
  return [...$('#g-aspect').options].filter(o=>o.value!=='custom')
    .map(o=>{ const [a,b]=o.value.split('x').map(Number); return {v:o.value,d:Math.abs(a/b-r)} })
    .sort((x,y)=>x.d-y.d)[0].v;
}
// On input, not change: switching the picker to Custom while you are still
// typing is what makes the two halves read as one control.
['#g-w','#g-h'].forEach(s=>{
  $(s).addEventListener('input',()=>{ $('#g-aspect').value='custom'; });
  // Snapped on the way out rather than per keystroke, which would fight you at
  // "10" on the way to "1000".
  $(s).addEventListener('change',()=>syncSize(true));
  $(s).addEventListener('blur',()=>syncSize(true));
});
syncSize(false);

// ---------- regions ----------
// A region is a rectangle on the canvas with an identity in it, so it is drawn
// on the canvas. The identity is a LoRA written into the box's own prompt in
// the same `<lora:name:1.3>` syntax the main prompt takes — a token in the main
// prompt is the whole canvas, a token in a box is that rectangle, and nothing
// has to explain which is which — or a reference photograph dropped onto the
// box, or both. One LoRA per box, because that is the node's shape; the backend
// rejects a second rather than applying the first and looking like it took both.
//
// State is an array with a render function, the same shape `refs`/`drawRefs`
// already uses, rather than the DOM-as-state the old rows were. The boxes are
// drawn rather than typed into, so reading values back off the DOM would buy
// nothing — and array order is load-bearing: V9's `_pair_boxes` matches box i
// to row i by original index, so the boxes and the rows have to come out of one
// list in one order or every face lands in the wrong rectangle with no error.
let regions=[], rsel=-1;

const MIN_SIDE=0.04;          // below this a box is not grabbable, and 0 is rejected
const SNAP_TO=[0,1/4,1/3,1/2,2/3,3/4,1];
const SNAP_EPS=0.015;         // fraction of the frame, so it feels the same at any size

const clamp01=v=>Math.max(0,Math.min(1,Number.isFinite(v)?v:0));
const regionOn=()=>$('#g-regional').classList.contains('on');

// A box is "armed" when it holds an identity: a LoRA name that resolves to a
// file, or a photograph. That is the distinction that decides what comes out —
// an empty rectangle is filled by the scene prompt, a box with an identity in
// it is a person — and it is the same one the old 32px plots drew.
const regionArmed=r=>!!(r.ref||parseLoras(r.prompt||'').some(t=>t.hit));

// What the box calls itself. The LoRA name is the identity, so it wins; the
// prompt is the fallback because a photo-only box still has words worth showing.
function regionTag(r){
  const hit=parseLoras(r.prompt||'').find(t=>t.hit);
  if(hit) return esc(hit.hit.token);
  const words=stripLoras(r.prompt||'').trim();
  if(words) return esc(words.length>28?words.slice(0,27)+'…':words);
  return r.ref?'<em>photo</em>':'';
}

// ---------- the frame ----------
// Sized off the canvas, never off the viewport — the console under it grows and
// shrinks with what is open, so a dvh sum is wrong the moment anyone opens
// Advanced. Same measurement `layoutShots` does, and for the same reason.
function layoutFrame(){
  const f=$('#frame');
  if(f.classList.contains('hide')) return;
  const [rw,rh]=readSize(), ar=(rw&&rh)?rw/rh:1;
  const box=$('#canvas'), cs=getComputedStyle(box);
  // Every subtrahend measured, none guessed — same reasoning as layoutShots,
  // and the caption is a real element with a real height.
  const availH=box.clientHeight-parseFloat(cs.paddingTop)-parseFloat(cs.paddingBottom)
               -($('#gen-meta').offsetHeight||0)-12;
  const availW=box.clientWidth-parseFloat(cs.paddingLeft)-parseFloat(cs.paddingRight);
  // A container with no width has not been laid out yet — a hidden view, a tab
  // restored in the background. Writing that measurement through would set the
  // frame to 0×0 and every drag on it would divide by zero; keeping the last
  // real numbers costs nothing, because the ResizeObserver fires again the
  // moment it does have a size.
  if(availW<=0) return;
  // Whichever axis runs out first. A floor, because a console with everything
  // open can leave less height than a frame you could actually drag inside —
  // at that point letting the canvas scroll beats offering a 40px target.
  let h=Math.max(160,Math.min(availH,availW/ar));
  let w=h*ar;
  if(w>availW){ w=availW; h=w/ar }
  f.style.setProperty('--frame-w',Math.round(w)+'px');
  f.style.setProperty('--frame-h',Math.round(h)+'px');
}

// Where the boxes are drawn right now. One layer, reparented — every coordinate
// in it is a percentage, so it does not care which element it is inside.
//
// A still wins over the frame because adjusting boxes against the picture you
// actually got is the whole point of them still being there after a render; a
// plate wins over the still because a plate is what you are composing *into*,
// so the last render is no longer the subject. With a batch it is the first
// still only: one set of boxes applies to the whole batch, and drawing them
// four times would say otherwise.
function regionHost(){
  if(kind!=='image') return null;
  const shot=$('#gen-out').children.length&&!plate.scene&&!plate.outfit
    ? $('#gen-out').querySelector('.shot') : null;
  return shot||$('#frame');
}

function drawRegions(){
  const layer=$('#region-layer'), host=regionHost(), on=regionOn();
  // Defensive, because losing this element once cost three bugs that looked
  // like three features breaking. Anything that replaces the innerHTML of a
  // container the layer has been re-homed into deletes it, and the symptom is
  // never "the layer is gone" — it is every caller of drawRegions dying at its
  // first statement. Fail quiet and visible rather than taking setRegional,
  // syncCanvasView and syncRegionNote down with it.
  if(!layer){ console.warn('[regions] layer element is gone'); return }
  // The frame only shows when it is the host: with a render on the canvas the
  // boxes live on that instead, and an empty frame beside it would be a second
  // place to look.
  $('#frame').classList.toggle('hide',!(on&&host===$('#frame')));
  $('#canvas-empty').classList.toggle('hide',
    !!$('#gen-out').children.length||!!$('#vid-out').children.length||(on&&kind==='image'));
  layer.classList.toggle('off',!on||!host);
  // Before the early return, not after it. The no-host case is exactly the one
  // that has to hide the inspector — switching to video leaves the toggle's
  // `on` class alone, so returning first left a region row floating over the
  // video composer with no boxes anywhere to explain it.
  if(!on||!host){ syncInspector(); syncRegionNote(); return }
  if(layer.parentElement!==host) host.appendChild(layer);
  layoutFrame();

  // Updated in place, not rebuilt. A drag calls this on every pointermove, and
  // replacing eight subtrees at that rate throws away focus, hover and the
  // keyboard's own target sixty times a second — one arrow key moved a box and
  // the next went nowhere, because the element it was aimed at no longer
  // existed.
  const els=[...layer.querySelectorAll('.rbox')];
  while(els.length>regions.length) els.pop().remove();
  while(els.length<regions.length){
    const el=document.createElement('div');
    el.className='rbox'; el.tabIndex=0;
    // What dropping a photo here does, read by the drag-reveal caption. Set on
    // the element rather than written into the CSS so the two gestures a box
    // answers to — a photo for the likeness, a token for the LoRA — stay
    // described in the same place the box is built.
    el.dataset.drop='This character';
    el.innerHTML='<img class="face hide" alt=""><span class="tag hide"></span>'
      +['nw','n','ne','e','se','s','sw','w'].map(h=>`<i data-h="${h}"></i>`).join('');
    layer.appendChild(el); els.push(el);
  }
  regions.forEach((r,i)=>{
    const el=els[i];
    el.dataset.i=i;
    el.classList.toggle('armed',regionArmed(r));
    el.classList.toggle('sel',i===rsel);
    el.style.left=(clamp01(r.x)*100)+'%'; el.style.top=(clamp01(r.y)*100)+'%';
    el.style.width=(Math.min(1-clamp01(r.x),clamp01(r.width))*100)+'%';
    el.style.height=(Math.min(1-clamp01(r.y),clamp01(r.height))*100)+'%';
    const face=el.querySelector('.face'), src=r.ref?'data:image/png;base64,'+r.ref:'';
    face.classList.toggle('hide',!r.ref);
    if(src&&face.getAttribute('src')!==src) face.setAttribute('src',src);
    const tag=el.querySelector('.tag'), t=regionTag(r);
    if(tag.innerHTML!==t) tag.innerHTML=t;
    tag.classList.toggle('hide',!t);
  });
  syncInspector(); syncRegionNote();
  syncRegionVis(); syncRegionBadge();
}

// ---------- selection and the inspector ----------
// The map. Redrawn from `regions`, which drawRegions already owns, so it cannot
// disagree with the canvas about where anything is.
function drawRegionMap(){
  const el=$('#r-map'); if(!el) return;
  const W=40,H=28;
  el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" aria-hidden="true">`
    +`<rect class="fr" x=".5" y=".5" width="${W-1}" height="${H-1}" rx="2"/>`
    +regions.map((r,i)=>{
        const x=Math.max(0,Math.min(1,+r.x||0)), y=Math.max(0,Math.min(1,+r.y||0));
        const w=Math.max(0,Math.min(1-x,r.width==null?1:+r.width));
        const h=Math.max(0,Math.min(1-y,r.height==null?1:+r.height));
        return `<rect class="bx${i===rsel?' on':''}" data-i="${i}" x="${(x*W).toFixed(2)}"`
          +` y="${(y*H).toFixed(2)}" width="${(w*W).toFixed(2)}"`
          +` height="${(h*H).toFixed(2)}" rx="1.2"/>`;
      }).join('')
    +`</svg>`;
  el.querySelectorAll('[data-i]').forEach(r=>r.onclick=e=>{
    e.stopPropagation();
    // Selecting a box means adjusting it, and adjusting means seeing it — so
    // this is also the way back onto the canvas after a render put them away.
    revealRegions();
    selectRegion(+r.dataset.i);
  });
}
// Arrows step between boxes while the map has focus. Bound here rather than on
// the bar, because ⌥←/⌥→ in a region's prompt already moves clauses and a chord
// that means two things in one row is a chord nobody trusts.
$('#r-map').addEventListener('keydown',e=>{
  if(e.key!=='ArrowLeft'&&e.key!=='ArrowRight') return;
  if(!regions.length) return;
  e.preventDefault(); revealRegions();
  const d=e.key==='ArrowRight'?1:-1;
  selectRegion((rsel+d+regions.length)%regions.length);
  $('#r-map').focus();
});

function selectRegion(i,focusPrompt){
  rsel=(i>=0&&i<regions.length)?i:-1;
  drawRegions();
  if(rsel<0) return;
  // A new box wants the caret in the field, because the next thing you do is
  // say who is in it. Clicking an existing one focuses the box itself: the
  // pointerdown handler calls preventDefault to own the drag, which also
  // suppresses the focus the click would normally have given it — so without
  // this the arrow keys and ⌫ only ever reached a box you had Tabbed to, and
  // clicking one then pressing delete did nothing at all.
  if(focusPrompt) $('#r-prompt').focus();
  else $('#region-layer').querySelector('.rbox[data-i="'+rsel+'"]')?.focus();
}

// The strength field is a *view* of the first number in the box's `<lora:…>`
// token, not a second place the number lives — the same relationship the ratio
// picker has to the width and height boxes. Blank when there is no token to
// read, because a strength with nothing to apply to is a number that lies.
function tokenStrength(text){
  const t=parseLoras(text||'').find(t=>t.hit);
  return t?loraNum(t.a,1):null;
}
function setTokenStrength(text,val){
  const t=parseLoras(text||'').find(t=>t.hit);
  if(!t) return text;
  const parts=t.name.split(':');
  parts[1]=String(val);
  if(parts[2]===undefined&&t.b==='') parts.length=2;
  return text.slice(0,t.start)+'<lora:'+parts.join(':')+'>'+text.slice(t.end);
}

// Which box the inspector is currently showing. Not the same as `rsel`: the
// fields are only rewritten when the selection actually moves, so that a
// redraw mid-keystroke does not yank the caret to the end of what you are
// typing.
let rshown=-1;
function syncInspector(){
  const bar=$('#region-bar'), r=regions[rsel];
  // Selection moved, so every field belongs to a different box now and the
  // focus guards below have to be overridden. Without this, clicking box 2
  // while the caret was still in the prompt left box 1's words on screen —
  // and the next keystroke wrote them into box 2, because the input handler
  // sends whatever is in the field to whatever `rsel` now points at. A stale
  // display would have been a nuisance; copying one performer's direction
  // onto another is the bug.
  const moved=rshown!==rsel;
  rshown=rsel;
  // `kind` as well as the toggle: the toggle lives inside the image composer,
  // which hides on a switch to video while keeping its `on` class — so without
  // this the inspector row outlives the feature it belongs to and floats above
  // the video controls.
  bar.classList.toggle('hide',!(regionOn()&&kind==='image'&&r));
  if(!r) return;
  // Never while it is being typed into: rewriting the field under the caret
  // moves it to the end of the line on every keystroke.
  if(moved||document.activeElement!==$('#r-prompt')) $('#r-prompt').value=r.prompt||'';
  const s=tokenStrength(r.prompt);
  const sf=$('#r-strength');
  if(moved||document.activeElement!==sf){ sf.value=s===null?'':String(s) }
  sf.disabled=s===null;
  sf.placeholder=s===null?'—':'';
  $$('#region-bar [data-r]').forEach(el=>{
    if(moved||document.activeElement!==el)
      el.value=(+r[el.dataset.r]).toFixed(3).replace(/0+$/,'').replace(/\.$/,'');
  });
  const img=$('#r-ref-thumb'), hint=$('#r-ref-hint');
  img.classList.toggle('hide',!r.ref);
  hint.classList.toggle('hide',!!r.ref);
  $('#r-ref').classList.toggle('set',!!r.ref);
  if(r.ref) img.src='data:image/png;base64,'+r.ref;
}

$('#r-prompt').addEventListener('input',()=>{
  if(!regions[rsel]) return;
  regions[rsel].prompt=$('#r-prompt').value;
  drawRegions(); syncLoraNote();
});
$('#r-strength').addEventListener('input',()=>{
  const r=regions[rsel]; if(!r) return;
  const v=parseFloat($('#r-strength').value);
  if(!Number.isFinite(v)) return;
  r.prompt=setTokenStrength(r.prompt,v);
  drawRegions(); syncLoraNote();
});
$$('#region-bar [data-r]').forEach(el=>el.addEventListener('input',()=>{
  const r=regions[rsel]; if(!r) return;
  const v=parseFloat(el.value);
  if(!Number.isFinite(v)) return;
  r[el.dataset.r]=clamp01(v);
  drawRegions();
}));
$('#r-del').onclick=()=>{
  if(rsel<0) return;
  regions.splice(rsel,1);
  selectRegion(Math.min(rsel,regions.length-1));
  syncLoraNote();
};

// ---------- drawing ----------
// The first pointer-drag code in this page. Pointer Events with capture, so a
// drag that leaves the frame still tracks and still ends — with mouse events a
// release outside the window leaves a box stuck to the cursor.
function frameXY(e,host){
  const b=host.getBoundingClientRect();
  return [clamp01((e.clientX-b.left)/b.width), clamp01((e.clientY-b.top)/b.height)];
}
// Snap to the frame's own landmarks and to the other boxes' edges. This is what
// makes clean columns a gesture instead of a menu; Alt turns it off for the
// times when the tidy answer is the wrong one.
// `snapEdge`, not `snap` — the size control already owns that name for rounding
// pixels to the VAE grid, and two of them in one script scope is a page that
// does not load at all.
function snapEdge(v,axis,skip,alt){
  if(alt) return v;
  let best=v, dist=SNAP_EPS;
  const cands=SNAP_TO.slice();
  regions.forEach((r,i)=>{
    if(i===skip) return;
    cands.push(axis==='x'?r.x:r.y, axis==='x'?r.x+r.width:r.y+r.height);
  });
  cands.forEach(c=>{ const d=Math.abs(v-c); if(d<dist){ dist=d; best=c } });
  return best;
}
function showGuides(r){
  const layer=$('#region-layer');
  layer.querySelectorAll('.guide').forEach(g=>g.remove());
  if(!r) return;
  const near=(v,axis)=>SNAP_TO.some(c=>Math.abs(v-c)<1e-6);
  [['x',r.x],['x',r.x+r.width]].forEach(([a,v])=>{
    if(!near(v,a)) return;
    const g=document.createElement('div'); g.className='guide v';
    g.style.left=(v*100)+'%'; layer.appendChild(g);
  });
  [['y',r.y],['y',r.y+r.height]].forEach(([a,v])=>{
    if(!near(v,a)) return;
    const g=document.createElement('div'); g.className='guide h';
    g.style.top=(v*100)+'%'; layer.appendChild(g);
  });
}

$('#region-layer').addEventListener('pointerdown',e=>{
  if(e.button!==0) return;
  const host=regionHost(); if(!host) return;
  // ⌘ means "a new one, here" and skips the hit test on purpose. Once a few
  // performers are placed there is often no bare canvas left to start a drag
  // on, and the alternative — move something out of the way, draw, move it
  // back — is three gestures to express one.
  const fresh=e.metaKey||e.ctrlKey;
  const boxEl=fresh?null:e.target.closest('.rbox');
  const handle=fresh?null:(e.target.dataset.h||null);
  const [px,py]=frameXY(e,host);
  let idx, mode, orig, grab;

  if(boxEl){
    idx=+boxEl.dataset.i;
    mode=handle||'move';
    orig={...regions[idx]};
    grab=[px-orig.x, py-orig.y];
    selectRegion(idx);
  }else{
    // A drag on bare canvas draws a new box. Capped, and silently — the cap is
    // the backend's and there is nothing useful to say about it mid-gesture.
    if(regions.length>=(window.MAX_REGIONS||8)) return;
    regions.push({prompt:'',ref:null,x:px,y:py,width:0,height:0});
    idx=regions.length-1; mode='se'; orig={...regions[idx]};
    grab=[0,0];
    rsel=idx;
    drawRegions();
  }

  e.preventDefault();
  // Capture so a drag that leaves the frame still tracks and still ends: with
  // plain listeners a release outside the window leaves the box stuck to the
  // cursor. Guarded because a pointer already gone by the time we ask throws
  // NotFoundError, and losing the capture is survivable where losing the rest
  // of this handler is not.
  try{ $('#region-layer').setPointerCapture(e.pointerId) }catch(_){}

  const move=ev=>{
    const [x,y]=frameXY(ev,host);
    const r=regions[idx]; if(!r) return;
    const alt=ev.altKey;
    if(mode==='move'){
      r.x=Math.min(Math.max(snapEdge(x-grab[0],'x',idx,alt),0),1-orig.width);
      r.y=Math.min(Math.max(snapEdge(y-grab[1],'y',idx,alt),0),1-orig.height);
      r.width=orig.width; r.height=orig.height;
    }else{
      let l=orig.x, t=orig.y, rt=orig.x+orig.width, bt=orig.y+orig.height;
      if(mode.includes('w')) l=snapEdge(x,'x',idx,alt);
      if(mode.includes('e')) rt=snapEdge(x,'x',idx,alt);
      if(mode.includes('n')) t=snapEdge(y,'y',idx,alt);
      if(mode.includes('s')) bt=snapEdge(y,'y',idx,alt);
      // Sorted rather than clamped, so dragging a handle past its opposite
      // flips the box the way every other editor does instead of jamming.
      r.x=Math.min(l,rt); r.width=Math.abs(rt-l);
      r.y=Math.min(t,bt); r.height=Math.abs(bt-t);
    }
    drawRegions(); showGuides(r);
  };
  const up=ev=>{
    $('#region-layer').removeEventListener('pointermove',move);
    $('#region-layer').removeEventListener('pointerup',up);
    $('#region-layer').removeEventListener('pointercancel',up);
    try{ $('#region-layer').releasePointerCapture(ev.pointerId) }catch(_){}
    showGuides(null);
    const r=regions[idx];
    if(r){
      // A click rather than a drag on bare canvas leaves a zero-area box, which
      // the backend rejects outright. Grow it to something usable instead of
      // erroring at Generate about a rectangle nobody meant to make.
      if(r.width<MIN_SIDE||r.height<MIN_SIDE){
        if(mode==='se'&&!orig.width){
          r.width=Math.max(r.width,0.28); r.height=Math.max(r.height,0.6);
          r.x=Math.min(r.x,1-r.width); r.y=Math.min(r.y,1-r.height);
        }else{
          r.width=Math.max(r.width,MIN_SIDE); r.height=Math.max(r.height,MIN_SIDE);
        }
      }
      selectRegion(idx, mode==='se'&&!orig.width);
    }
    syncLoraNote();
    // selectRegion focuses the bar, so the boxes stay up on their own from
    // here — the hold only had to survive the pointer being down.
    holdRegions(false);
  };
  holdRegions(true);
  $('#region-layer').addEventListener('pointermove',move);
  $('#region-layer').addEventListener('pointerup',up);
  $('#region-layer').addEventListener('pointercancel',up);
});

// Keyboard is the whole non-pointer path, so it has to move and delete, not
// just select. Steps match the inspector cells' own data-step/data-bigstep.
$('#region-layer').addEventListener('keydown',e=>{
  const el=e.target.closest('.rbox'); if(!el) return;
  const r=regions[+el.dataset.i]; if(!r) return;
  if(e.key==='Backspace'||e.key==='Delete'){
    e.preventDefault();
    regions.splice(+el.dataset.i,1);
    selectRegion(Math.min(+el.dataset.i,regions.length-1));
    syncLoraNote(); return;
  }
  const d={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]}[e.key];
  if(!d) return;
  e.preventDefault();
  const step=(e.metaKey||e.ctrlKey)?0.1:0.01;
  r.x=Math.min(Math.max(r.x+d[0]*step,0),1-r.width);
  r.y=Math.min(Math.max(r.y+d[1]*step,0),1-r.height);
  drawRegions();
  $('#region-layer').querySelector(`.rbox[data-i="${el.dataset.i}"]`)?.focus();
});
$('#region-layer').addEventListener('focusin',e=>{
  const el=e.target.closest('.rbox');
  if(el&&+el.dataset.i!==rsel) selectRegion(+el.dataset.i);
});

// ---------- reference photos ----------
// Downscaled before encoding, because eight photographs in one JSON body is the
// payload this feature invites. On the image side this is the only cap there is:
// the node's own `ref_max_side` is set to 0 in the graph so the resizing happens
// in exactly one place and the two cannot end up fighting over which one shrank
// the picture.
//
// The video references reuse it and are not that case — H3_REF_MAX_SIDE caps the
// staged file again, at the same number, because the gallery's "Use as reference"
// hand-off never passes through here and H3 has no `ref_max_side` to switch off.
// The two agreeing is not what makes that correct; the server binding is. If they
// ever drift, the picture is still capped on the side that decides.
const REF_MAX=1536;
function shrinkB64(file){
  return new Promise(res=>{
    const img=new Image(), url=URL.createObjectURL(file);
    img.onload=()=>{
      URL.revokeObjectURL(url);
      // Under the cap the bytes go verbatim, because the canvas only knows how
      // to hand back PNG and a 224 KB JPEG that needed no resizing came out of
      // it at 1.9 MB — 8.6x, for a picture nothing had asked to change. Nine of
      // those is 17 MB of base64 to save 2 MB, which is the fix costing more
      // than the thing it fixes. Same rule the volume uses for the orientation
      // tag and `_fit_reference` for the size: rewrite only what is out of
      // spec, pass the rest through untouched.
      if(img.width<=REF_MAX&&img.height<=REF_MAX) return res(toB64(file));
      const s=REF_MAX/Math.max(img.width,img.height);
      const c=document.createElement('canvas');
      c.width=Math.round(img.width*s); c.height=Math.round(img.height*s);
      c.getContext('2d').drawImage(img,0,0,c.width,c.height);
      res(c.toDataURL('image/png').split(',')[1]);
    };
    img.onerror=()=>{ URL.revokeObjectURL(url); res(null) };
    img.src=url;
  });
}

const refInput=document.createElement('input');
refInput.type='file'; refInput.accept='image/*'; refInput.className='hide';
$('#r-ref').appendChild(refInput);
$('#r-ref-hint').innerHTML=ICON.photo;
$('#r-ref').onclick=e=>{
  if(e.target===refInput) return;
  const r=regions[rsel]; if(!r) return;
  if(!r.ref) return refInput.click();
  r.ref=null; drawRegions();
};
refInput.onchange=async e=>{
  const f=e.target.files[0]; refInput.value='';
  if(!f||!f.type.startsWith('image/')||!regions[rsel]) return;
  regions[rsel].ref=await shrinkB64(f);
  drawRegions();
};

// Dropping onto a box is the gesture the box exists for; dropping onto bare
// canvas is the scene, which is a different thing entirely and gated on a
// weight that may not be downloaded. One listener, because the target decides.
function wireCanvasDrop(el){
  // The outline goes on the host, not on the layer. `hot` was being set here
  // for as long as this function has existed and the rule that draws it is
  // `.frame.hot,.shot.hot` — but #region-layer is a *child* of the frame or the
  // still, and carries no classes at all, so the selector never matched and the
  // canvas drop shipped with no feedback whatsoever. The one gesture that turns
  // a photograph into the world the render happens inside looked, to anyone who
  // had not read this file, like dragging a file onto a dead surface.
  const host=()=>el.parentElement;
  const paint=on=>{
    const h=host(); if(!h) return;
    h.classList.toggle('hot',on);
    // Named before the drop, not after. Without the weight this still lights
    // and still captions — the point of showing it at all is that the feature
    // exists and is one download away, which is a thing worth learning while
    // you are holding the photograph it would have used.
    h.dataset.drop = window.HAS_EDIT_LORA ? 'Scene' : 'Scene — needs a download';
  };
  el.addEventListener('dragover',e=>{
    if(!regionOn()) return;
    e.preventDefault();
    const hit=e.target.closest('.rbox');
    paint(!hit);
    // Only the box under the cursor names itself. Eight captions on eight boxes
    // is the same wall of text the per-region rows were removed for, drawn on
    // the picture this time.
    $$('#region-layer .rbox').forEach(b=>{
      b.classList.toggle('sel',b===hit||+b.dataset.i===rsel);
      b.classList.toggle('drop-hit',b===hit);
    });
  });
  // Guarded on relatedTarget: without it every child under the cursor fires a
  // dragleave and the highlight strobes across a surface this large.
  el.addEventListener('dragleave',e=>{
    if(!el.contains(e.relatedTarget)) clearCanvasDrop();
  });
  el.addEventListener('drop',async e=>{
    if(!regionOn()) return;
    e.preventDefault(); clearCanvasDrop();
    const f=e.dataTransfer.files[0];
    if(!f||!f.type.startsWith('image/')) return;
    const hit=e.target.closest('.rbox');
    // Said, not swallowed. Without the edit LoRA this branch used to return in
    // silence, which is indistinguishable from a drop the page never received —
    // and the drop target is now visibly lit, so refusing quietly would be a
    // promise the page made and then broke.
    if(!hit&&!window.HAS_EDIT_LORA){
      const note=$('#region-note');
      note.classList.remove('hide'); note.textContent=NEED_EDIT_LORA;
      return;
    }
    const b64=await shrinkB64(f);
    if(!b64) return;
    if(hit){
      regions[+hit.dataset.i].ref=b64;
      selectRegion(+hit.dataset.i);
    }else{
      setPlate('scene',b64);
    }
    drawRegions();
  });
}
// ---------- when the boxes are on screen ----------
// Four reasons, and they are all "you are working on a region right now":
// the caret is in the region bar, a box is mid-drag, a file is over the window,
// or regions were just armed and the two seeded rectangles are the instruction.
// Anything else — including looking at the render you just made — and they go.
let regionPeek = false;          // set while a drag or a fresh arm holds them open
// Armed means visible — with one exception, which is the whole complaint the
// first version overshot. Gating them on focus in the region bar was too
// strict: the boxes are the list, you place and size them by dragging, and a
// list you have to click a text field to see is a list you cannot use.
//
// The thing that was actually unbearable was narrower than that. It was white
// rectangles over a picture you just waited two minutes for. So a finished
// render is the only thing that puts them away, and the very next sign you are
// making the next one brings them back — a keystroke in the prompt, a touch on
// the canvas, a region control, arming the mode again. No gesture exists purely
// to recover them, because every route back is something you were going to do
// anyway.
let freshRender = false;
function regionsVisible(){
  if(!window.REGIONS_READY || !regionOn()) return false;
  if(regionPeek) return true;
  if(document.body.classList.contains('dragging')) return true;
  return !freshRender;
}
// A result landed: the canvas is for looking at until you touch something.
// A result landed: the boxes come off the picture. They are still armed and
// still masking their LoRAs — this is only about what is drawn.
function shotLanded(){ freshRender=true; syncRegionVis() }
// And you asked for them back. Deliberately narrow: the 50/50 split is right
// most of the time, so a set of boxes is placed once and rendered against dozens
// of times, changing only the prompt. Restoring them on a keystroke would put
// rectangles back over the picture on the single most common action in the app
// — the original complaint, arriving through the fix for it. Only an explicit
// ask counts: the pill on the result, the mode button, or the region bar, which
// you reach only when adjusting a box.
function revealRegions(){
  if(!freshRender) return;
  freshRender=false; syncRegionVis();
}
function syncRegionVis(){
  const l=$('#region-layer'); if(l) l.classList.toggle('show', regionsVisible());
  if(typeof drawRegionMap==='function') drawRegionMap();
}
// The count, so the mode stays legible with nothing on the picture. Without it
// a regional render and a plain one look identical right up until the result.
function syncRegionBadge(){
  const on=window.REGIONS_READY&&regionOn(), n=on?regions.length:0;
  $('#g-regional').classList.toggle('counted',n>0);
  $('#g-regional').dataset.count=n||'';
}
// focusin/focusout rather than focus/blur: those do not bubble, and the bar
// holds a dozen fields — binding each one is a dozen places to forget the next
// control added to the row.
// Not the prompt. Writing the next take is the common path and says nothing
// about whether you want to look at the boxes.
$('#region-bar').addEventListener('focusin',revealRegions);
['focusin','focusout'].forEach(ev=>document.addEventListener(ev,()=>setTimeout(syncRegionVis,0)));
// A drag that starts on the layer has to keep them up even though the pointer
// leaves the field that revealed them.
function holdRegions(on){ regionPeek=on; syncRegionVis() }

function clearCanvasDrop(){
  const el=$('#region-layer');
  if(el.parentElement) el.parentElement.classList.remove('hot');
  // `sel` too, and back to whichever box actually is selected. dragover paints
  // the box under the cursor as selected so you can see what you are aiming at;
  // a drag abandoned over that box used to leave the paint behind, so the
  // console was inspecting one box while two looked chosen.
  $$('#region-layer .rbox').forEach(b=>{
    b.classList.remove('drop-hit');
    b.classList.toggle('sel',+b.dataset.i===rsel);
  });
}
wireCanvasDrop($('#region-layer'));

// ---------- the two plates ----------
// Held as base64 the way the video keyframes are. A scene plate also becomes the
// frame's backdrop, because a frame you can see the background in is the
// difference between placing people in a room and placing rectangles in a void.
const plate={scene:null,outfit:null};
function setPlate(slot,b64){
  plate[slot]=b64;
  const img=$('#g-thumb-'+slot), hint=$('#g-hint-'+slot), box=$('#g-drop-'+slot);
  img.classList.toggle('hide',!b64);
  hint.classList.toggle('hide',!!b64);
  box.classList.toggle('set',!!b64);
  if(b64) img.src='data:image/png;base64,'+b64;
  syncFrameBackdrop();
  drawRegions();
}
function syncFrameBackdrop(){
  const f=$('#frame');
  let el=f.querySelector('.plate');
  if(plate.scene){
    if(!el){ el=document.createElement('img'); el.className='plate'; f.prepend(el) }
    el.src='data:image/png;base64,'+plate.scene;
  }else if(el) el.remove();
  // Thirds are a viewfinder for an empty frame; over a photograph they are
  // just lines on someone's picture.
  f.querySelector('.thirds').classList.toggle('hide',!!plate.scene);
}
function wirePlate(slot){
  const box=$('#g-drop-'+slot), hint=$('#g-hint-'+slot);
  hint.innerHTML=ICON[slot];
  const input=document.createElement('input');
  input.type='file'; input.accept='image/*'; input.className='hide';
  box.appendChild(input);
  const take=async f=>{
    if(!f||!f.type.startsWith('image/'))return;
    setPlate(slot,await shrinkB64(f));
  };
  // A second click on a filled tile clears it. There is no ✕ because the tile
  // is 36px and a hit target inside it would be smaller than a fingertip —
  // the same reason the keyframe tiles do not carry one either.
  // `locked` is visible and hoverable so it can say why, which means it is also
  // clickable and droppable unless every entry point checks. A locked tile that
  // opened a file picker and then swallowed the picture would be worse than the
  // hidden tile this replaced.
  box.classList.add('can-drop');
  const locked=()=>box.classList.contains('locked');
  box.onclick=e=>{
    if(e.target===input||locked()) return;
    if(!plate[slot]) return input.click();
    setPlate(slot,null);
  };
  input.onchange=e=>{ take(e.target.files[0]); input.value='' };
  box.ondragover=e=>{ if(locked())return; e.preventDefault();box.classList.add('hot') };
  box.ondragleave=()=>box.classList.remove('hot');
  box.ondrop=e=>{ if(locked())return;
    e.preventDefault();box.classList.remove('hot');take(e.dataTransfer.files[0]) };
}
wirePlate('scene'); wirePlate('outfit');

// ---------- arrange ----------
// Overwrites, unlike the Columns/Rows select it replaces: that filled only the
// coordinates nobody had typed into, and once boxes are drawn there is no such
// thing as an untouched coordinate.
function distribute(cols){
  const n=regions.length; if(!n) return;
  regions.forEach((r,i)=>{
    r.x=cols?i/n:0; r.y=cols?0:i/n;
    r.width=cols?1/n:1; r.height=cols?1:1/n;
  });
  drawRegions();
}
$('#g-arrange').onclick=e=>openMenu(e.currentTarget,[
  {label:'Distribute in columns',run:()=>distribute(true)},
  {label:'Distribute in rows',run:()=>distribute(false)},
]);

// What the boxes cannot show: which engine this run is about to take. They are
// genuinely different — one sampling pass against masked LoRA deltas, or a
// krea2edit compose that regenerates the whole frame around the plate — and
// the second is several times slower, which is worth knowing before you press
// Generate rather than after. A region's own photo is neither: it is a latent
// mold on the fast path, so it never moves the run onto the slow one.
function syncRegionNote(){
  const n=readRegions().length;
  const el=$('#region-note');
  el.classList.toggle('hide',!regionOn());
  if(!regionOn()||!n){ el.textContent=''; return }
  const molds=regions.filter(r=>r.ref).length;
  const tail=molds?` ${molds} with a reference photo.`:'';
  // A box with words but no identity is placed by the description alone —
  // there is no LoRA delta to mask, so it is a soft placement rather than a
  // guaranteed one. Worth saying, because the two kinds of box look identical
  // on the canvas and do not hold their ground equally.
  const soft=regions.filter(r=>stripLoras(r.prompt||'').trim()
    && !r.ref && !parseLoras(r.prompt||'').some(t=>t.hit)).length;
  const softNote=soft
    ? ` ${soft} described only — placed by the words, not held by a mask.` : '';
  el.textContent = (plate.scene||plate.outfit)
    ? `${n} region${n>1?'s':''} composed into the reference — slower, and it re-renders the whole frame.${tail}`
    : `${n} region${n>1?'s':''}, one pass. Each LoRA is masked to its box.${tail}${softNote}`;
}

// Captured from the markup before anything overwrites them, so the sentence
// describing what a scene plate does still lives beside the tile that takes one
// rather than in a second table here that has to be kept in step with it.
const PLATE_TITLE={'#g-drop-scene':$('#g-drop-scene').title,
                   '#g-drop-outfit':$('#g-drop-outfit').title};
const NEED_EDIT_LORA=
  'Scene and outfit transfer need the Krea 2 identity-edit LoRA — download it under Settings.';

// Set, not toggle: `reuse()` and the edit-LoRA refresh both need to put the
// mode into a known state, and a flip called from those would turn regional
// *off* on a card that has regions whenever it happened to already be on.
function setRegional(on){
  $('#g-regional').classList.toggle('on',on);
  ['#g-arrange','#g-region-base-wrap','#g-region-vr']
    .forEach(s=>$(s).classList.toggle('hide',!on));
  // The plates ride on two conditions, not one: regions on, and the weight
  // they need actually downloaded. Region photos ride on neither — a mold is
  // not an extra_ref plate, so it needs no edit LoRA and never switches paths.
  //
  // The two conditions get two different treatments, and the split is the
  // point. Regions off: the tiles are gone, because there is nothing to put a
  // scene behind. Weight missing: the tiles are *dimmed*, the same `.off` the
  // keyframe pair uses for "out of play this run", because an install without
  // the edit LoRA is one download away from scene and outfit transfer and had
  // no way of learning either existed — the controls were absent, so there was
  // nothing to be curious about. This is the line: a model-gated control the
  // model will never read stays hidden, since a control that is present and
  // ignored is worse than one that is absent. A weight-gated control is not
  // that; it is a purchase you have not made yet, and hiding it hides the
  // decision rather than the capability.
  ['#g-drop-scene','#g-drop-outfit'].forEach(s=>{
    $(s).classList.toggle('hide',!on);
    $(s).classList.toggle('locked',!window.HAS_EDIT_LORA);
    $(s).title = window.HAS_EDIT_LORA ? PLATE_TITLE[s] : NEED_EDIT_LORA;
  });
  syncDropTargets();
  // Two half-width columns, seeded. Two rectangles appearing on the canvas is
  // the whole instruction — a sentence telling you to drag would be read on
  // every visit forever to be useful once.
  if(on&&!regions.length){
    regions=[{prompt:'',ref:null,x:0,y:0,width:0.5,height:1},
             {prompt:'',ref:null,x:0.5,y:0,width:0.5,height:1}];
    rsel=0;
  }
  drawRegions(); syncCanvasView(); syncLoraNote();
  // Arming puts the caret in the region prompt, which is what makes the two
  // seeded rectangles visible — the instruction only works if you can see it.
  // It is also the honest answer to "how do I get the boxes back": the same
  // field, every time, rather than a second gesture to learn.
  if(on) $('#r-prompt').focus();
  syncRegionVis();
}
$('#g-regional').onclick=()=>{
  // Reveal before toggle. While regions are armed and a fresh render has put
  // them away, this button's job is to bring them back — disarming the mode
  // from a button lit with a count, without the boxes ever being on screen, is
  // the one destructive reading this control has never had.
  if(regionOn() && freshRender) return revealRegions();
  setRegional(!regionOn());
};

function readRegions(){
  if(!regionOn()) return [];
  return regions.map(r=>{
    const toks=parseLoras(r.prompt||'').filter(t=>t.hit);
    return {
      // Stripped, same as the main prompt: the token is markup for this page,
      // and a text encoder handed the word "lora" renders it.
      prompt:stripLoras(r.prompt||''),
      // Sent as a stack even though a region takes one, so the backend can say
      // "a region takes one" about the second rather than this quietly
      // dropping it.
      loras:toks.map(t=>({path:t.hit.path,unet:loraNum(t.a,1)})),
      ref:r.ref||null,
      x:r.x, y:r.y, width:r.width, height:r.height,
    };
  // A box with no words, no LoRA and no photograph is an empty rectangle, and
  // dropping it here is safe in a way filtering server-side would not be: the
  // boxes and the rows are built from this one list in this one order, so they
  // stay paired by index whatever comes out.
  }).filter(r=>r.prompt||r.loras.length||r.ref);
}

// Everything the region code needs now exists, so the callers that run before
// this point — syncSize's bootstrap, syncCanvasView — are free to redraw.
window.REGIONS_READY=true;
drawRegions();

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
  // 22px for the actions strip under each still — the same number as
  // `.shot{padding-bottom}`, and subtracted per row rather than once, because a
  // batch is two rows and each of them has a strip.
  const ACTS_H=30;
  g.style.setProperty('--shot-h', (n<=2 ? h-ACTS_H : h/2-16-ACTS_H)+'px');
}
// The canvas changes height whenever the console does, so the fit is recomputed
// rather than set once at generation time.
new ResizeObserver(()=>{
  const n=$('#gen-out').children.length;
  if(n) layoutShots(n);
  // The frame is measured off the canvas the same way the stills are, so it
  // has to be remeasured on the same signal.
  layoutFrame();
}).observe(document.querySelector('#canvas'));

$('#go-gen').onclick=async()=>{
  const p=promptText(), regions=readRegions();
  if(!p&&!regions.length)return;
  $('#gen-err').innerHTML=''; $('#gen-meta').textContent='';
  const btn=$('#go-gen'); btn.disabled=true;
  const box=$('#gen-prog'); box.classList.remove('hide'); box.querySelector('p').textContent='Queued…';
  const [w,h]=readSize();
  const r=await post('/api/generate',{
    prompt:p, negative_prompt:negAllowed()?$('#neg').value:'', model:$('#g-model').value,
    shot:readShot(),
    loras:readLoras(), regions, region_weight:$('#g-region-base').value,
    // Only when there are boxes to compose around — the backend rejects a
    // plate without regions, and sending one anyway would turn a hidden tile
    // that still holds an image into an error nobody could see the cause of.
    scene:regions.length?plate.scene:null, outfit:regions.length?plate.outfit:null,
    width:w, height:h, num_images:$('#g-n').value, seed:$('#g-seed').value,
    sampler:$('#g-sampler').value, scheduler:$('#g-scheduler').value,
    steps:$('#g-steps').value, cfg_scale:$('#g-cfg').value, shift:$('#g-shift').value,
    gpu:$('#g-gpu').value,
  });
  if(r.error){$('#gen-err').innerHTML='<div class="err-box">'+r.error+'</div>';btn.disabled=false;box.classList.add('hide');return}
  wireCancel(box, r.job_id);
  const t=everyMs(async()=>{
    const s=await api('/api/status/'+r.job_id);
    box.querySelector('i').style.width=(s.percent||0)+'%';
    box.querySelector('p').textContent=s.step?`Step ${s.step}/${s.total_steps}`:(s.phase||'Working…');
    if(s.status==='completed'){
      clearInterval(t); btn.disabled=false; box.classList.add('hide');
      // Streamed by filename, not fetched as base64 first. The status record
      // already carries `files`, so the extra /api/outputs round trip was
      // buying nothing except a JSON body with megabytes of base64 in it —
      // which has to arrive whole, and be decoded, before any of the four
      // stills can paint. The gallery was moved off that inlining for exactly
      // this reason; the canvas, the one surface where the wait is being
      // watched, was still doing it. Now each <img> paints as its own bytes
      // land, off the same route and the same cache the gallery uses.
      const files=s.files||[];
      layoutShots(files.length);
      // The region layer lives *inside* the first `.shot` while there are
      // results, so the boxes land on the picture with no measurement. That
      // makes the innerHTML below its executioner: replacing the contents of
      // #gen-out deletes the layer and every pointer and keyboard listener
      // bound to it, and `$('#region-layer')` is null from then on. What that
      // looked like was three unrelated bugs — the boxes vanished after a
      // render, the inspector row would not close, and the Regional toggle
      // appeared dead — because `drawRegions()` threw on its first line and
      // took the rest of `setRegional` with it, including the note that would
      // have said the boxes had no LoRA in them. Park it somewhere the wipe
      // cannot reach; `drawRegions` re-homes it a few lines down.
      $('#canvas').appendChild($('#region-layer'));
      // Each still carries its own way into video. Two, because they are
      // genuinely different jobs: a first frame is the shot the clip starts on,
      // a reference is a subject the clip is about.
      $('#gen-out').innerHTML=files.map((f,n)=>
        `<figure class="shot"><img src="/api/file/${r.job_id}/${encodeURIComponent(f)}" alt=""`+
        ` decoding="async" fetchpriority="high">`+
        `<span class="acts">`+
        `<button data-n="${n}" data-as="first" title="Animate — use as the first frame of a clip">${ICON.play}</button>`+
        `<button data-n="${n}" data-as="reference" title="Use as a reference image">${ICON.photo}</button>`+
        `</span></figure>`).join('')
        || '<p class="muted">Saved to '+(s.output_dir||'')+'</p>';
      $$('#gen-out .acts button').forEach(b=>b.onclick=()=>
        handoffFile(r.job_id, files[+b.dataset.n], b.dataset.as));
      // The same viewer the gallery card opens. A still on the canvas is fitted
      // to whatever the console left it — at four-up, half of that — so the one
      // thing you want next is to see it at size, and it should not cost a trip
      // through the gallery to a copy of the image already on screen.
      $$('#gen-out img').forEach(im=>im.onclick=()=>lightbox(im.src,false));
      // Surface which LoRAs actually matched — a stack that silently no-ops
      // looks identical to a stack that had no effect.
      //
      // `===false`, not `!l.applied`. The image report emits {name, unet,
      // text_encoder} and no `applied` key at all, so the falsy test called
      // every LoRA on every render unapplied — a warning that is always on is
      // worse than no warning, and this one cost real time during a debug by
      // pointing at a healthy LoRA while the actual fault was elsewhere. The
      // other two readers of this field already default the other way:
      // `_infotext` uses `l.get("applied", True)` and the gallery card uses
      // `l.applied===false`.
      const skipped=(s.loras||[]).filter(l=>l.applied===false);
      // The one clause here that is not a fact about the render but a report
      // that something you asked for did not happen, so it carries `.warn` —
      // the same amber #lora-note uses for a name that resolves to no file.
      // Set in the same grey as "6.2s" it was a caption the eye reads past,
      // which is the wrong place to hide "your LoRA did nothing".
      $('#gen-meta').innerHTML=[
        (s.seeds||[]).join(', ')&&esc('seed '+(s.seeds||[]).join(', ')),
        s.sampler&&esc(`${s.sampler} · ${s.steps} steps · CFG ${s.cfg_scale}`),
        s.duration_s&&esc(`${s.duration_s}s`),
        skipped.length&&('<span class="warn">'+esc('not applied: '+skipped
          .map(l=>l.name+(l.reason?` (${l.reason})`:'')).join(', '))+'</span>'),
      ].filter(Boolean).join(' · ');
      shotLanded();
      syncCanvasView(); loadGallery();
    } else if(s.status==='stopped'){
      clearInterval(t); btn.disabled=false; box.classList.add('hide');
      // Said out loud. A cancelled run used to just hide the bar, which is the
      // same thing the screen does when a run finishes with nothing to show.
      $('#gen-meta').textContent='Cancelled.';
    } else if(s.status==='failed'){
      clearInterval(t); btn.disabled=false; box.classList.add('hide');
      $('#gen-err').innerHTML='<div class="err-box">'+(s.error||'Generation failed')+'</div>';
    }
  },POLL_MS);
};

// A run is cancellable from the moment it has an id, which is the moment the
// spawn returns — before that there is nothing to name. The stop is cooperative:
// the job unwinds at the next step boundary and the container stays warm, so the
// button reports that it asked rather than claiming the run is already over.
// Disabled rather than removed, because a Cancel that vanishes on click reads as
// a click that missed.
// The gap between "the image exists" and "the image is on screen" is bounded by
// how often this asks. At 2s a still that finished the instant after a poll sat
// on the volume for the better part of two seconds with the bar showing the last
// step — which is most of the perceived wait on an 8-step Turbo run that samples
// in four. A poll is a Dict read in the web container, so this is one of the
// cheapest seconds in the application to buy back.
const POLL_MS=400;

function wireCancel(box, jobId){
  const b=box.querySelector('[data-cancel]');
  if(!b) return;
  b.disabled=false; b.textContent='Cancel';
  b.onclick=async()=>{
    b.disabled=true; b.textContent='Stopping…';
    const r=await post('/api/stop/'+jobId);
    if(r&&r.error){ b.disabled=false; b.textContent='Cancel' }
  };
}

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
  // The row itself always stands: every video model takes a first frame, so
  // there is no model whose sources row is empty. It is the reference half
  // that comes and goes, and the rule with it — a divider with nothing on one
  // side of it is a line that means nothing.
  ['#v-add-ref','#v-add-vid','#v-ref-size-wrap','#v-src-vr'].forEach(
    s=>$(s).classList.toggle('hide',!sup.references));
  syncNeg();
  $('#v-cfg-wrap').classList.toggle('hide',!sup.cfg);
  $('#v-shift-wrap').classList.toggle('hide',!sup.cfg);
  $('#v-switch-wrap').classList.toggle('hide',!sup.experts);
  $('#v-drop-last').classList.toggle('hide',!sup.last_frame);
  // The prompt survives a model change, and so do the LoRAs written in it — so
  // whether this model reads them is something only the note can say. The pills
  // survive it too, and there the answer is visible rather than written: the
  // ones this model does not read go dim where they are.
  syncLoraNote();
  drawShotRail();

  // A model that cannot take what is already attached would fail at submit.
  // Dropping it here, where the section it came from is visibly gone, is the
  // only version of this that does not look like the request lost it.
  if(!sup.references&&(refs.length||refVids.length)){ refs=[]; refVids=[]; refRoles=[]; }
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
  paintSampling(); paintVidSize();
  drawRefs();
}
$('#v-model').onchange=syncVideoModel;

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
  // Each image chip also carries what it is *for*. The role compiles into the
  // prompt's `subject_definitions` — "<Subject 1> is the person in <Picture 1>"
  // — which is the whole answer to "you should not describe the picture you
  // attached, but there is nowhere else to put it so everyone does". Now there
  // is somewhere, and it is a menu rather than a sentence.
  const img=refs.map((b,i)=>{
    const spec=shotRoleDefs.find(r=>r.key===refRoles[i]);
    return `<div class="ref"><img src="data:image/png;base64,${b}" alt="">`+
    `<b>P${i+1}</b><button class="x" data-k="img" data-i="${i}" title="Remove">×</button>`+
    `<button class="role${spec?'':' none'}" data-r="${i}" `+
    `title="What this picture defines. It goes into the prompt as a subject, `+
    `so you never have to describe the photograph itself.">`+
    `${esc(spec?spec.label:'role')}</button></div>`;
  }).join('');
  const vid=refVids.map((b,i)=>
    // Same media fragment as the gallery card, and for the same reason: a
    // reference tile with no frame painted is an unlabelled black square,
    // which defeats the point of showing the reference you attached.
    // No role bar: the compiler builds subjects out of <Picture n> only, and a
    // menu that set something nothing reads is worse than no menu.
    `<div class="ref"><video src="data:video/mp4;base64,${b}#t=0.04" muted></video>`+
    `<b>V${i+1}</b><button class="x" data-k="vid" data-i="${i}" title="Remove">×</button></div>`).join('');
  $('#v-refs').innerHTML=img+vid;
  $$('#v-refs button.x').forEach(b=>b.onclick=()=>{
    const i=+b.dataset.i;
    (b.dataset.k==='img'?refs:refVids).splice(i,1);
    // The roles are positional — index i is <Picture i+1> — so removing the
    // second chip has to remove the second role with it. Left alone, every role
    // after the gap would have silently moved onto the wrong picture.
    if(b.dataset.k==='img') refRoles.splice(i,1);
    drawRefs();
  });
  $$('#v-refs button.role').forEach(b=>b.onclick=e=>openMenu(e.currentTarget,
    [{label:'No role',on:!refRoles[+b.dataset.r],
      run:()=>{ refRoles[+b.dataset.r]=''; drawRefs() }}].concat(
      shotRoleDefs.map(r=>({
        label:r.label, on:refRoles[+b.dataset.r]===r.key,
        run:()=>{ refRoles[+b.dataset.r]=r.key; drawRefs() },
      })))));
  // Exclusivity, shown where the choice is made rather than described under
  // it: whichever half is out of play this run goes inert. It is the same fact
  // the note carries, but the note is read after the click and this is read
  // before it. Nothing is hidden — a control that vanishes when you fill its
  // neighbour reads as a bug, and both halves have to stay visible for the row
  // to keep teaching that they are alternatives.
  const n=refs.length+refVids.length;
  const sup=(videoModel()||{supports:{}}).supports;
  const framed=!!(keyframe.first||keyframe.last);
  // References win when both are somehow attached, because that is what the
  // run does — so they are the half that stays live. Dimming both, which the
  // symmetric version did, left the gallery's "As reference" hand-off in a
  // state where a keyframe could not be cleared and a second reference could
  // not be added: the two controls locked each other out.
  ['first','last'].forEach(s=>$('#v-drop-'+s).classList.toggle('off',!!n));
  ['#v-add-ref','#v-add-vid'].forEach(s=>$(s).classList.toggle('off',framed&&!n));
  // "Keyframes are ignored" only when there is a keyframe to ignore. Said
  // unconditionally, it was the page's one mention of a control this layout
  // had already made unfindable — a warning about something you do not have,
  // pointing at somewhere you cannot see.
  $('#vid-note').textContent = n
    ? (framed ? `${n} reference${n>1?'s':''} — keyframes are ignored for this run.` : '')
    : (keyframe.first
        ? (sup.references?'':'Image-to-video. ')+'Canvas follows the first frame’s aspect ratio.'
        : '');
  // A role and a keyframe both change which document gets built, so the
  // preview has to follow them as well as the pills.
  syncShotPeek();
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
  // Images go through the same shrink the region photos do, for the payload
  // half of the reason H3_REF_MAX_SIDE gives: nine photographs straight off a
  // phone is tens of megabytes of base64 in one JSON body. The server caps them
  // again on arrival and that is the copy that binds — this is the one that
  // keeps the request from being the slowest part of pressing Generate.
  // Videos are sent whole: there is nothing here that can re-encode one.
  input.onchange=async e=>{
    for(const f of [...e.target.files].slice(0,max-bucket.length)){
      const b=await(isImg?shrinkB64(f):toB64(f));
      // shrinkB64 answers null for a file the canvas could not decode. Pushed
      // anyway it would be a chip with no picture and a base64 of "null" in the
      // request, which fails on the GPU rather than on the file it came from.
      if(b) bucket.push(b); else alert('Could not read that image.');
    }
    drawRefs();
  };
  input.click();
}
$('#v-add-ref').onclick=()=>pickRefs('img');
$('#v-add-vid').onclick=()=>pickRefs('vid');

// Everything that takes a file, wired one way.
//
// Written because the reference tray never had a drop handler at all — it was
// a click-only picker sitting in a row of tiles that all take drops, and the
// drag-intent reveal then lit it like the rest. A control that lights up under
// a dragged file and refuses it is worse than one that stays dark: the dark one
// is undiscovered, the lit one is a promise the page breaks while you watch.
//
// `can-drop` is set here rather than written into the markup so the two cannot
// drift: the class that makes a target glow is applied by the same call that
// gives it a handler, and an element with no handler has no way to get it.
function wireDropTarget(el, {accept, take, label}){
  if(!el) return;
  el.classList.add('can-drop');
  if(label) el.dataset.drop=label;
  const off=()=>el.classList.remove('hot');
  el.addEventListener('dragover',e=>{
    if(el.classList.contains('locked')||el.classList.contains('off')) return;
    // Files only, and only the kind this target takes. Without the type check
    // a video dragged onto the picture tray lights it, and the drop then
    // silently does nothing — which is the same broken promise one level down.
    if(![...(e.dataTransfer?.types||[])].includes('Files')) return;
    e.preventDefault(); el.classList.add('hot');
  });
  el.addEventListener('dragleave',e=>{ if(!el.contains(e.relatedTarget)) off() });
  el.addEventListener('drop',async e=>{
    if(el.classList.contains('locked')||el.classList.contains('off')) return;
    e.preventDefault(); off();
    const files=[...(e.dataTransfer?.files||[])].filter(f=>f.type.startsWith(accept));
    // Said, not swallowed. A file of the wrong kind landing on a tile that
    // just lit up for it has to say why nothing happened.
    if(!files.length){
      const want=accept==='image/'?'an image':'a video';
      return alert(`That tile takes ${want}.`);
    }
    await take(files);
  });
}

// The two halves of the reference tray, which is where this started.
[['#v-add-ref','img','image/','#v-ref-max',refs,'Picture reference'],
 ['#v-add-vid','vid','video/','#v-vid-max',refVids,'Video reference']]
  .forEach(([sel,kindOf,accept,maxSel,,label])=>wireDropTarget($(sel),{
    accept, label,
    take:async files=>{
      // Re-read the bucket and the cap at drop time rather than closing over
      // them: `refs` and `refVids` are reassigned wholesale by reuse() and by
      // syncVideoModel dropping references a model cannot take, so a captured
      // array would be pushing into a detached one.
      const isImg=kindOf==='img';
      const bucket=isImg?refs:refVids;
      const max=+$(maxSel).textContent;
      if(bucket.length>=max)
        return alert(`${max} ${isImg?'image':'video'} references is the model's limit.`);
      for(const f of files.slice(0,max-bucket.length)){
        const b=await(isImg?shrinkB64(f):toB64(f));
        if(b) bucket.push(b); else alert('Could not read that file.');
      }
      drawRefs();
    },
  }));

// The canvas itself. On the video side a dropped picture is the frame the clip
// starts on, which is the one reading that needs no mode: it is what the tile
// two rows down would have done, done on the largest target on screen.
wireDropTarget($('#vid-out'),{
  accept:'image/', label:'First frame',
  take:async files=>{
    const b=await toB64(files[0]);
    if(!b) return alert('Could not read that image.');
    setFrame('first',b); syncFrameCanvas(); drawRefs();
  },
});

function wireDrop(slot){
  const box=$('#v-drop-'+slot), img=$('#v-thumb-'+slot), hint=$('#v-hint-'+slot);
  box.classList.add('can-drop');
  hint.innerHTML=ICON[slot];
  const input=document.createElement('input');
  input.type='file'; input.accept='image/*'; input.className='hide';
  box.appendChild(input);

  const take=async f=>{
    if(!f||!f.type.startsWith('image/'))return;
    setFrame(slot, await toB64(f));
  };
  // A second click on a filled tile clears it, exactly as the scene and outfit
  // plates do — whose comment has always claimed these tiles behave the same
  // way. They did not: a keyframe could be replaced but never removed, so the
  // only route back to text-to-video was switching model and back. Harmless
  // while the two halves were unrelated rows; now that a keyframe puts the
  // reference tray out of play, a keyframe you cannot clear is a corner you
  // cannot get out of.
  box.onclick=e=>{
    if(e.target===input) return;
    if(!keyframe[slot]) return input.click();
    clearFrame(slot);
    if(slot==='first') syncFrameCanvas(); else drawRefs();
  };
  input.onchange=e=>{ take(e.target.files[0]); input.value='' };
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
  // Either slot puts the reference tray out of play, and drawRefs() is what
  // paints that — reached through the full sync for a first frame, which also
  // re-picks the checkpoint and locks the aspect, and directly for a last one.
  if(slot==='first') syncFrameCanvas(); else drawRefs();
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


$('#go-vid').onclick=async()=>{
  const p=promptText();
  if(!p)return;
  $('#vid-err').innerHTML=''; $('#vid-meta').textContent='';
  const btn=$('#go-vid'); btn.disabled=true;
  const box=$('#vid-prog'); box.classList.remove('hide');
  box.querySelector('i').style.width='0%';
  box.querySelector('p').textContent='Queued…';

  const r=await post('/api/video',{
    model:$('#v-model').value,
    prompt:p, negative_prompt:negAllowed()?$('#neg').value:'',
    aspect:$('#v-aspect').value, tier:$('#v-tier').value,
    seconds:$('#v-seconds').value, steps:$('#v-steps').value, seed:$('#v-seed').value,
    cfg:$('#v-cfg').value, shift:$('#v-shift').value, switch_at:$('#v-switch').value,
    sampler:$('#v-sampler').value, scheduler:$('#v-scheduler').value,
    loras:readVidLoras(), shot:readShot(), ref_roles:refRoles.slice(0,refs.length),
    first_frame:keyframe.first, last_frame:keyframe.last,
    references:refs, ref_videos:refVids,
    ref_size:$('#v-ref-size').value, gpu:$('#v-gpu').value,
  });
  if(r.error){
    $('#vid-err').innerHTML='<div class="err-box">'+r.error+'</div>';
    btn.disabled=false; box.classList.add('hide'); return;
  }
  wireCancel(box, r.job_id);

  const t=everyMs(async()=>{
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
      const src=`/api/file/${r.job_id}/${f}`;
      $('#vid-out').innerHTML=f
        ? `<video controls autoplay loop playsinline src="${src}"></video>`
          +`<button class="zoom" title="Full screen">${ICON.expand}</button>`
        : '<p class="muted">Saved to '+(s.output_dir||'')+'</p>';
      const zoom=$('#vid-out .zoom');
      if(zoom) zoom.onclick=()=>lightbox(src,true);
      const stack=readVidLoras();
      $('#vid-meta').textContent=[
        s.width&&`${s.width}×${s.height}`,
        s.seconds&&`${s.seconds}s · ${s.frames} frames · ${s.fps} fps`,
        s.seed!=null&&`seed ${s.seed}`,
        s.steps&&`${s.steps} steps`,
        stack.length&&`${stack.length} LoRA${stack.length>1?'s':''}`,
        s.duration_s&&`${s.duration_s}s`,
      ].filter(Boolean).join(' · ');
      shotLanded();
      syncCanvasView(); loadGallery();
    } else if(s.status==='stopped'){
      clearInterval(t); syncVideoModel(); box.classList.add('hide');
      $('#vid-meta').textContent='Cancelled.';
    } else if(s.status==='failed'){
      clearInterval(t); syncVideoModel(); box.classList.add('hide');
      $('#vid-err').innerHTML='<div class="err-box">'+(s.error||'Generation failed')+'</div>';
    }
  },POLL_MS);
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
  paintLastShot(galItems);
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
      <button data-act="del" title="Delete">${ICON.close}</button>
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
    card.querySelector('[data-open]').onclick=()=>viewAt(rows,+card.dataset.i);
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
  const rows=galShown();
  $('#gal-empty').textContent=rows.length?'':
    (galItems.length?'Nothing of that kind yet.':'Nothing generated yet.');
  $('#gal-grid').innerHTML=rows.map(galCard).join('');
  syncPurge();
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

// What the purge would take, in the words the filter is already using.
// A declaration, not a const: drawGallery() above calls it, and a const here
// would sit in the temporal dead zone for any path that draws before this line
// runs. Hoisting costs nothing and removes the ordering constraint.
function galShown(){ return galItems.filter(i=>galFilter==='all'||i.kind===galFilter) }
function syncPurge(){
  const n=galShown().length, b=$('#gal-purge');
  b.disabled=!n;
  b.textContent = !n ? 'Delete all'
    : `Delete ${n} ${galFilter==='all'?'result':galFilter}${n===1?'':'s'}`;
}
$('#gal-purge').onclick=async()=>{
  const rows=galShown(), n=rows.length;
  if(!n) return;
  // Spells out the count and the scope, because a filtered gallery and an
  // unfiltered one put the same button in the same place over very different
  // amounts of work.
  const what=galFilter==='all' ? `all ${n} results` : `${n} ${galFilter}${n===1?'':'s'}`;
  if(!confirm(`Permanently delete ${what}?\n\n`
    + `The files are unlinked from the volume. This cannot be undone.`)) return;
  const b=$('#gal-purge'); b.disabled=true; b.textContent='Deleting…';
  // The ids, not the filter: what goes is exactly what was counted in the
  // dialog, even if a run lands while it is open.
  const r=await post('/api/outputs/purge',
    {confirm:'delete', job_ids:rows.map(i=>i.job_id)});
  if(r.error) alert(r.error);
  await loadGallery();
};

// ---------- card actions ----------
function download(it){
  it.files.forEach(f=>{
    const a=document.createElement('a');
    a.href=`/api/file/${it.job_id}/${f}`; a.download=f;
    document.body.appendChild(a); a.click(); a.remove();
  });
}

async function remove(it){
  if(!confirm('Permanently delete this result?\n\nThe files are unlinked from the volume. This cannot be undone.')) return;
  await post(`/api/outputs/${it.job_id}/delete`);
  loadGallery();
}

// Fetch the bytes rather than reusing a data URL: the card is a streamed
// <img src>, so the base64 the video side needs does not exist client-side.
async function handoffFile(jobId,file,as){
  const blob=await (await fetch(`/api/file/${jobId}/${encodeURIComponent(file)}`)).blob();
  toVideo(await toB64(blob), as);
}
// The canvas hands off by (job, file) because that is all it has now that the
// stills are streamed rather than inlined; a gallery card hands off the same
// bytes through the same route, so it is the same call with the fields dug out.
const handoff=(it,as)=>handoffFile(it.job_id, it.files[0], as);

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
  // The pills, not the sentence they compiled into. A run whose prompt was a
  // six-field document has to come back as the rail that built it — restoring
  // the document into the prompt box would put a schema in a textarea and then
  // compile *that*, which is the one way this feature could corrupt a reuse.
  // `shot` is only in the sidecar when the compiler did something, so a card
  // from before the palette existed clears the rail rather than keeping
  // whatever happened to be on it.
  shot=(it.shot||[]).filter(p=>p&&shotItem(p.key));
  refRoles=(it.ref_roles||[]).slice();
  shotOpen=null;
  drawShotRail();
  if(it.kind==='image'){
    setKind('image');
    // Prompt and stack are one field now, so they are restored as one string.
    // A LoRA deleted since the run simply does not come back — the same thing
    // the old row-matching did, minus a row left behind to explain it.
    set('#prompt',[it.prompt_typed||it.prompt,loraTokens(it.loras,false)].filter(Boolean).join(' '));
    set('#neg',it.negative_prompt);
    if(it.model) $('#g-model').value=it.model;
    const size=`${it.width}x${it.height}`;
    if([...$('#g-aspect').options].some(o=>o.value===size)){
      $('#g-aspect').value=size; syncSize(false);
    } else if(it.width&&it.height){
      // Not a preset, so it was a custom size — and it has to come back as one,
      // or "reuse" would silently render a different picture than the card.
      $('#g-w').value=it.width; $('#g-h').value=it.height; syncSize(true);
    }
    // `seed` is what this backend records; `seeds` is what the Forge one did,
    // and sidecars written by it are still on the volume. Reading both keeps
    // every card in the gallery reusable rather than only the ones made since.
    set('#g-seed',it.seed??(it.seeds||[])[0]);
    set('#g-sampler',it.sampler); set('#g-scheduler',it.scheduler);
    set('#g-steps',it.steps); set('#g-cfg',it.cfg_scale); set('#g-shift',it.shift);

    // Regions come back or the reuse is a lie: which LoRA sat in which box is
    // the whole result of a regional render, and restoring the prompt without
    // them would put a one-subject picture behind a card showing two people.
    // The plates deliberately do not come back — they were uploaded bytes, not
    // something the sidecar keeps, and the note under the stack will say the
    // run is the one-pass kind until one is dropped again.
    // Not `regions` — that is the live array this writes into, and a local of
    // the same name here would shadow it and restore nothing.
    const saved=it.regions||[];
    regions=saved.map(r=>{
      // `r.lora` is the path relative to loras/, which is what resolveLora
      // matches first — no need to reduce it to a stem here, and reducing it
      // would break exactly the ambiguous names the full path exists to
      // disambiguate.
      const tok=r.lora&&r.lora!=='None'
        ? loraTokens([{name:r.lora,unet:r.strength}],false) : '';
      const box=r.box||[];
      return {
        prompt:[r.prompt,tok].filter(Boolean).join(' '),
        // The sidecar records whether a box had a photo, never the photo — it
        // was uploaded bytes staged into a container that is long gone. Same
        // as the plates, which do not come back either.
        ref:null,
        x:+box[0]||0, y:+box[1]||0,
        width:box[2]!=null?+box[2]:1, height:box[3]!=null?+box[3]:1,
      };
    });
    rsel=regions.length?0:-1;
    set('#g-region-base',it.region_weight);
    setRegional(regions.length>0);

    syncModelLine(); syncLoraNote(); syncNeg();
    autoGrow($('#prompt')); autoGrow($('#neg'));
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
    // Matched on the full filename under loras/, not the stem: `high.safetensors`
    // is the filename of both speed pairs, so a stem match would restore the
    // t2v LoRA into an i2v run without a word about it. loraTokens() writes
    // whichever name is unambiguous, which is the folder-qualified one here.
    set('#prompt',[it.prompt_typed||it.prompt,loraTokens(it.loras,true)].filter(Boolean).join(' '));
    set('#neg',it.negative_prompt);
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
    syncLoraNote(); syncNeg();
    autoGrow($('#prompt')); autoGrow($('#neg'));
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
    // The pills by name, so a run is readable without decompiling its
    // document — and readable a year later, when the labels may have moved
    // but `camera.pushin` still says what was picked.
    ['Shot',(it.shot||[]).map(p=>{
      const i=shotItem(p.key);
      return (i?i.label:p.key)+(p.value?`: “${p.value}”`:'');
    }).join(', ')],
    ['Reference roles',(it.ref_roles||[]).map((r,n)=>{
      const spec=shotRoleDefs.find(x=>x.key===r);
      return spec?`P${n+1} ${spec.label}`:'';
    }).filter(Boolean).join(', ')],
    ['Files',it.files.join(', ')],
    ['Created',it.created?new Date(it.created*1000).toLocaleString():''],
  ].filter(r=>r[1]!==undefined&&r[1]!==null&&r[1]!=='');
  const el=sheet(`
    <div class="sheet-head">
      <h1 class="grow">Metadata</h1>
      <button class="ico" data-close>${ICON.close}</button>
    </div>
    <label>Prompt</label>
    <textarea id="m-prompt" rows="5" readonly>${esc(it.prompt_typed||it.prompt||'')}</textarea>
    ${it.prompt_typed&&it.prompt_typed!==it.prompt?`
      <!-- Both, and in this order. What you wrote is what you recognise the
           run by; what ran is the six-field document, and it is here because
           the only other way to find out what the encoder was given is to
           render again. -->
      <label style="margin-top:12px">What the model read</label>
      <textarea rows="7" readonly>${esc(it.prompt||'')}</textarea>`:''}
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
    // What the box above it shows. Copying the compiled document from under a
    // button sitting beside a textarea containing the typed sentence is the
    // button doing something other than what you can see.
    await navigator.clipboard.writeText(it.prompt_typed||it.prompt||'');
    e.target.textContent='Copied';
  };
  el.querySelector('#m-reuse').onclick=()=>{ el.remove(); reuse(it) };
}

setNegMode(false);
setKind('image');
setMode('generate');
// Sequenced, and in this order. Both of these reload the volume server-side,
// and firing them together is what put two reloads on the same container in
// the same instant. The one that decides whether Generate works goes first.
loadState().then(loadDatasets);
</script></body></html>
"""
