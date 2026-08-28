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
import hashlib
import io
import json
import math
import os
import re
import shlex
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import zipfile
from collections import deque
from pathlib import Path
from typing import Any, Callable

import modal

# The name is the deployment, and therefore the URL: `modal deploy` replaces an
# app of the same name rather than standing a second one beside it. So this is
# env-driven for the reason VOLUME_NAME below is, and they are meant to be set
# together — a branch tried against the live name takes the live app down while
# it is being tried, which is a thing you find out by looking for your URL.
#
#     VISIONARY_APP=visionary-test VISIONARY_VOLUME=visionary-test \
#     VISIONARY_MODELS_VOLUME=visionary-test-models modal deploy app.py
#
# is a whole second copy: its own URL, its own records, its own weights.
APP_NAME = os.environ.get("VISIONARY_APP", "visionary")

# Storage this app owns. create_if_missing because a fresh account or a new
# VISIONARY_VOLUME should just work — the cost is that a typo'd name silently
# yields an empty volume rather than an error, so _require_models() prints the
# resolved name and what it actually found instead of a bare "not downloaded".
VOLUME_NAME = os.environ.get("VISIONARY_VOLUME", "visionary")
# version=2 decides what a fresh account creates; an existing volume binds by
# name whatever its version. The v1 era ended 2026-08-25: tools/cold_start.py
# measured v1 at 0.08 GB/s against v2's 1.61, which taxed every read on the
# platform — galleries, clips, datasets, and 240 of a cold H3 start's 300
# seconds. The v1 volumes survive under *-v1-retired, unreferenced.
volume = modal.Volume.from_name(VOLUME_NAME, version=2, create_if_missing=True)

# Weights live on their own v2 volume, and the number that put them there is
# 0.08 GB/s: that is what tools/cold_start.py measured the v1 volume serving,
# single-stream or 8-way alike, which priced a cold H3 start at 240 seconds of
# its 300 reading the checkpoint. The same probe read this volume at 1.61.
#
# **1.61 is the steady-state number, not the first one.** A file freshly seeded
# onto this volume reads once at roughly a tenth of that, then is fast forever
# after. On 2026-08-25 the migration re-downloaded every weight, and Krea 2's
# one slow read landed four hours later on a real render: 8.5 GB of text
# encoder at 87 MB/s, then 24 GB of DiT at 81 MB/s — the second charged to step
# 0 of 8, where comfy-aimdo shows it only as "Model Initializing" and the tqdm
# bar sits at 0% for five minutes. Measured against the 1.61 above it read as a
# regression, and three theories died before the next cold start came back
# normal. H3 looked immune purely because tools/cold_start.py had already read
# its weights that morning. **Read every weight once after seeding a volume**,
# or the first person to render pays for it and gets no hint why.
#
# v2 is Beta, and Modal's own docs decline to promise no data loss — which is
# exactly why *only* weights are here. A weight is a receipt: every file on
# this volume re-downloads from the gear. Datasets, LoRAs and characters are
# records, and records do not move to storage with a disclaimer. If v2 ever
# eats this volume, the cost is re-downloading checkpoints, not somebody's
# library.
#
# A sibling mount, not /workspace/models: Modal refuses nested mount paths
# outright ("volume mount paths cannot be nested"), so the models/ folder
# became /models. The layout under each mount is still the contract.
MODELS_VOLUME_NAME = os.environ.get("VISIONARY_MODELS_VOLUME", "visionary-models")
models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME, version=2,
                                       create_if_missing=True)

# Downloaded model weights live on their own volume, never on the data volume.
# Qwen3-VL-8B is ~17 GB, and with the HuggingFace cache pointed at /workspace
# every commit after a caption run had to snapshot those 17 GB — the job wrote
# all 80 captions in eight minutes and then sat in commit() for another fifteen.
# Separating them keeps a caption commit proportional to the captions.
hf_cache = modal.Volume.from_name("visionary-hf-cache", version=2,
                                  create_if_missing=True)
HF_CACHE = Path("/hf")

# Live job state (progress, stop flags) and the saved HF token. Dicts rather
# than Secrets so there is no CLI setup step — paste the key into the UI.
jobs = modal.Dict.from_name("visionary-jobs", create_if_missing=True)
config = modal.Dict.from_name("visionary-config", create_if_missing=True)

# Training sessions — the cards. A session outlives the run it started and the
# window that started it: it is the *setup* (which set, which name, which dials)
# plus a pointer at the last job spawned from it, so a finished run can be
# re-run with one dial changed instead of retyped.
#
# The whole index lives under one key rather than one key per session, and that
# is the `DL_ACTIVE` lesson rather than tidiness: this is a *network* Dict, and
# `_active_download()` scanning twenty-odd keys across it made a route take
# seven seconds to answer. The board polls, so a listing is one round trip.
#
# What is deliberately **not** in a session record is its status. A stored
# "running" is a claim about a container that may not exist — the Dict outlives
# every container, app and deploy that writes to it — so status is derived from
# the job record and its `beat` on every read. See `_session_view`.
sessions = modal.Dict.from_name("visionary-sessions", create_if_missing=True)
SESSION_INDEX = "index"

# Mount point is an internal detail; the layout under it is the contract.
# Weights are addressed by exact path, never scanned, so models/ is flat with
# descriptive filenames rather than the per-architecture directories a webui
# needs in order to populate its dropdowns.
# Env-driven because the mount point is exactly what changes off Modal: a local
# run re-roots both at a directory it owns, and every constant below derives
# from these two, so the layout *under* the root is identical either way. That
# is what makes a local datasets/ and a deployed one the same folder, an rsync
# apart. Modal refusing nested mounts is the only reason MODELS is a sibling
# rather than WORKSPACE/models — locally it can go back under it, which is
# where the module docstring says it started.
WORKSPACE = Path(os.environ.get("VISIONARY_WORKSPACE", "/workspace"))
MODELS = Path(os.environ.get("VISIONARY_MODELS", "/models"))
LORAS = WORKSPACE / "loras"
# A dataset is a named folder of images with .txt captions beside them — the
# same thing musubi reads, so the sidecars stay the source of truth and nothing
# we build here is required to get a dataset back out. Datasets outlive the
# training runs that use them; WORK is the per-run scratch copy musubi resizes
# and caches into, and is disposable.
DATASETS = WORKSPACE / "datasets"
# A set you have not kept yet. Identical folder shape to a dataset — images
# with .txt sidecars — so the contact sheet and the caption box never learn
# which root a set came from, and saving one is a move rather than a
# conversion. The split is the whole meaning of "saved": what is under
# datasets/ is your library, and nothing else is promised to survive.
#
# **On the container's disk, not the volume**, since 2026-09-01: the volume
# holds weights and what you pressed Save on, and an unsaved set is neither.
# It lives as long as the web container does — twenty minutes past the last
# request — and Save is the one gesture that reaches the volume. The cost to
# design for: anything that rents another container (captioning, dedupe on a
# GPU, training) reads a *saved* set, because this folder is invisible to
# every other machine. The owner refused a named exception: *"exceptions have
# a way of setting precedence … unsaved datasets aren't holy."*
# Env-driven for a reason that only shows up off Modal: "the container's disk"
# is /tmp in a container and is a real filesystem on a workstation, where /tmp
# is tmpfs on most Linux distributions — so an unsaved 400-image set would be
# spooled into RAM. Same default, same meaning, somewhere the launcher can put
# it. The Modal path does not read the variable and is unchanged.
DRAFTS = Path(os.environ.get("VISIONARY_DRAFTS", "/tmp/visionary-drafts"))
# Per-run scratch for the trainer's resized copy and the Drive puller's
# staging — each on the disk of the container doing the work, disposable.
# Same, and more sharply: WORK holds the trainer's resized copy of a dataset,
# which is gigabytes, and putting gigabytes on a tmpfs is putting them in RAM.
WORK = Path(os.environ.get("VISIONARY_WORK", "/tmp/visionary-work"))
OUTPUTS = WORKSPACE / "outputs"
# The Arsenal's first shelf: characters, saved deliberately and recalled by
# typing their name. A character is a folder of the files that make them —
# their pictures, their voice, a note.txt you can read in a terminal — because
# storage layout is the contract and nothing here is required to get your
# people back out. `character.json` beside them is the receipt that keeps what
# filenames cannot: which picture is the sheet, what each file provides.
CHARACTERS = WORKSPACE / "characters"
# The Playground's shelf. A saved workflow is the pure API-format graph —
# byte-for-byte what ComfyUI's /prompt accepts, so a stock install could run
# the file and nothing here is required to get an experiment back out. Our
# fields (created, what seeded it, exposed inputs) live beside each graph in a
# `{name}.meta.json` sidecar — the datasets split, so a reader that ignores us
# still gets a workflow.
WORKFLOWS = WORKSPACE / "workflows"
# The storyboards: one folder per board, `board.json` beside the pictures that
# were uploaded into it. A panel's picture is a *pointer* — into outputs/ for a
# render (job and file, the gallery's own address), or a bare filename for a
# picture dropped onto the board, which lives in this folder. Never a copy of a
# render, so unpinning deletes nothing and deleting a render leaves the panel's
# words standing. Legible without the app: `cat storyboard/*/board.json` is the
# film, and the folder is the whole import format — copy one in and it lists.
STORYBOARD = WORKSPACE / "storyboard"
STORYBOARD_MAX_PANELS = 400
# Per panel. A subject arrow is two points; the cap is generous for a client
# that is not this page.
STORYBOARD_MAX_ARROWS = 16
STORYBOARD_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
STORYBOARD_FILE_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}\.(png|jpg|jpeg|webp)$")
# Uploads are capped at the reference cap on the long side, because a panel's
# picture becomes a keyframe at the hand-off and the keyframe path resizes to
# this anyway — capping once on arrival means the wall and the run read the
# same bytes.
STORYBOARD_MAX_SIDE = 1536
# Node packs the Playground installs from git, one clone per folder, each
# carrying a `.visionary-pin` (source URL + resolved SHA). On the volume rather
# than in the image because installing one is a gesture, not a deploy — the
# generators pick the folder up through extra_model_paths.yaml at ComfyUI
# start, and a fresh pack loads on the next engine restart.
PLAYGROUND_NODES = WORKSPACE / "playground_nodes"
# /object_info, harvested by a CPU container and served to the editor.
# Derived, so it lives on the web container's disk and is rebuilt when a
# cold start finds it missing: the harvest job hands the catalogue back as
# its *return value* (megabytes, which a Dict polled every 400 ms is not
# for) and the web container reads that result once and keeps it here.
NODE_CATALOGUE = Path("/tmp/visionary-spool/node_catalogue.json")
# On the models volume, beside what it stages for. The download path promises
# "same filesystem, so this is an instant rename rather than a 26 GB copy" —
# staging on the data volume would quietly turn that rename into a five-minute
# cross-device copy read at the v1 volume's 0.08 GB/s.
STAGING = MODELS / ".cache" / "hf-staging"

# Deletions used to move here instead of unlinking. The name survives only so
# `_drop_legacy_trash` can clear what earlier versions left on the volume; no
# code writes into it any more.
LEGACY_TRASH_DIR = ".trash"
THUMB_DIR = ".thumbs"


def _set_cache(d: Path) -> Path:
    """Where a set's derived files live — thumbnails and dedupe fingerprints —
    on this container's disk, under the spool's LRU. They were `.thumbs/`
    inside the set on the volume; a cache the writer rebuilds does not belong
    beside the record it is derived from."""
    return SPOOL / "datasets" / d.name


# Env-driven for the reason WORKSPACE is: baked into trainer_image at this
# path, but locally it is a clone inside the venv that owns torch 2.5.1.
MUSUBI = Path(os.environ.get("VISIONARY_MUSUBI", "/opt/musubi-tuner"))
# Prepended to PATH for the trainer's subprocesses, so `python` and
# `accelerate` resolve to the environment musubi was installed into. Empty on
# Modal, where trainer_image *is* that environment. `accelerate` is a console
# script rather than a module, which is why this is a bin directory on PATH
# rather than an interpreter path like COMFY_PYTHON.
TRAIN_BIN = os.environ.get("VISIONARY_TRAIN_BIN", "")

# The default card for each family. Both default to Hopper for one reason
# each, and the reason is memory, not kernels. Images: Krea 2 used to have an
# A100-40GB of its own, sized against a measured 29.26 GiB peak at 1024px —
# that headroom does not survive V12, whose own regression notes record a
# 30.27 GiB block-mask build and a 17.88 GiB dense score tensor at the sequence
# lengths reference frames produce. Video: the H3 stack is 42.5 GB of weights
# before any activations. SageAttention used to be the other reason — compiled
# for sm_90 alone — until the arch list below grew Ada; it was a build arg
# functioning as a law.
GPU = os.environ.get("VISIONARY_IMAGE_GPU", "H100")
VIDEO_GPU = os.environ.get("VISIONARY_VIDEO_GPU", "H100")

# The card for the one-container mode, where a single ComfyUI serves both
# families. H200 because it is the only card here where both checkpoints stay
# resident at once (35 GB of Krea 2 and 42.5 GB of H3 in 141 GB); on an 80 GB
# card they take turns, paged through host RAM by ComfyUI's dynamic VRAM
# loading — seconds, not the 400 s volume reload, but not nothing.
BOTH_GPU = os.environ.get("VISIONARY_BOTH_GPU", "H200")

# Cards the UI may ask for, per class.
#
# Modal's Cls.with_options() returns a variant that autoscales independently of
# the base configuration — which is exactly the point (one class, any card) and
# exactly the cost (a card you have not used recently has no warm container, so
# picking it pays a cold start). The UI confirms before switching for that
# reason, and requests for the default are sent to the base class rather than
# through with_options so the common path keeps its warm container.
#
# Every card here must be in TORCH_CUDA_ARCH_LIST in comfy_image, or it loads
# the model fine and then falls back off SageAttention's kernels with no error
# to say so — the failure these lists exist to prevent. Hopper is 9.0, Ada
# (L40S) is 8.9; B200 is sm_100 and still needs "10.0" added and a rebuild
# forced. The A100 entries that used to be here went with sm_80, and could
# come back the same way L40S did.
#
# L40S is on every list because it is the cheap 48 GB card, and 48 GB is the
# number to know: a plain Krea 2 render fits (it ran on 40 GB), a regional
# render with reference frames may not, and H3's 42.5 GB of weights run with
# the DiT paged and the denoise loop several times slower than Hopper. The OOM
# error names the card so the second attempt is on a bigger one.
IMAGE_GPUS = ("H100", "H200", "L40S")
VIDEO_GPUS = ("H100", "H200", "L40S")
BOTH_GPUS = ("H200", "H100", "L40S")

# How much memory the card actually has, in GB, or 0 for "not stated".
#
# Passed in rather than measured, because the container that answers /api/state
# has no torch and no GPU — it is `web_image`, on CPU, deliberately. On Modal
# nothing sets it and it stays 0, which is right: every card in the two lists
# above is an 80 GB Hopper and there is nothing for the page to warn about.
# `tools/run_local.py` fills it from the card it found, and that is where the
# question starts to matter.
#
# **Reported, not enforced.** What a given card can and cannot render is exactly
# the sort of threshold this codebase refuses to invent — the one refusal below
# rests on numbers V12's own regression notes measured, and everything else
# waits for a rented box. A gate guessed here would be the app declining work it
# has never tried.
GPU_VRAM_GB = float(os.environ.get("VISIONARY_GPU_VRAM_GB") or 0)

# What the regional path peaks at, and it is not the weights. V12's own
# regression notes record a 30.27 GiB block-mask build and a 17.88 GiB dense
# score tensor at reference-frame sequence lengths — *activations*, which no
# amount of offloading or quantisation moves off the card. Krea 2 itself will
# stream onto a 16 GB card at a GGUF tier; a regional render on one will not,
# and the failure without this is an out-of-memory several minutes into a warm
# GPU rather than a form error in milliseconds.
REGIONAL_MIN_VRAM_GB = 32.0

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
# Pinned — and not for reproducibility, which is the reason that usually gets
# given and the one that matters least here. GenAI work is iterative; nobody
# needs this month's render back pixel-for-pixel in three.
#
# It is pinned because **a pin is the only thing that pulls upstream in.**
# Unpinning sounds like tracking latest and is the opposite: Modal's layer cache
# key is the command string, not the world, so `git clone --depth 1 <branch>` is
# a constant and that layer caches until something *above* it changes. Unpinned,
# the build sits on whatever HEAD was current the last time torch moved, with
# nothing recording which. Bumping a SHA is deliberate, guaranteed, and says
# what you got.
#
# Neither arrangement rebuilds when upstream churns — the cache key is the same
# either way — so the pin costs nothing for the hundred commits a week that
# never reach a render. What it lacks is a reason to look, and that is
# `tools/upstream.py`: it buckets what changed by the paths this app's behaviour
# rides on, so 168 files touched reads as "17 of them matter" rather than as a
# number to feel behind about.
#
# What upstream guarantees on its own, and what it does not: node ids, input
# ids, output slot indices and new required inputs are all migrated server-side
# on `POST /prompt` by `NodeReplace` — they maintain one for a node id with a
# *trailing space*. What has no migration is the arithmetic. In the range this
# bump crossed, `comfy/ldm/minimax/model.py` moved +150/-64 with every name
# intact.
COMFY_SHA = "924743af083c151296cc16f925aeab113b6484e8"  # 2026-08-22, +111
# The +1 over the previous pin is the entire point of the bump: it is the
# "Minimax-H3: Add missing special tokens" fix (#15808), twelve lines in
# comfy/text_encoders/minimax.py that end the gibberish speech mis-
# tokenized audio conditioning produced. The eleven commits upstream of
# it at bump time were partner nodes and dependency churn touching
# nothing tools/upstream.py watches, so they stay unbought.
COMFY = Path(os.environ.get("VISIONARY_COMFY", "/opt/comfyui"))
COMFY_PORT = 8188
# Which python runs it. `_Comfy.start()` spawns ComfyUI as a subprocess, so the
# interpreter is already a process boundary — and a process boundary is exactly
# what keeps torch 2.9.1+cu130 from having to agree with musubi's 2.5.1+cu124.
# On Modal that boundary is the image and this stays "python"; locally it names
# a venv built from the same image definition. The conflict that forces four
# images is therefore already solved in the code, and costs one string here.
COMFY_PYTHON = os.environ.get("VISIONARY_COMFY_PYTHON", "python")
# Extra argv for ComfyUI, which is where the offload flags go on a card that
# needs them. Empty on Modal, deliberately: an H100 holding the whole 42.5 GB
# has nothing to offload, and ComfyUI's automatic mode is what has always run
# there. `tools/run_local.py` fills it from the card it found.
#
# A passthrough rather than a table here, because the thresholds are the part
# nobody has measured yet — which card wants `--lowvram`, how much
# `--reserve-vram` a desktop actually needs — and a threshold argued from first
# principles is a threshold nobody has looked at. Keeping them in the launcher
# means the rented box can move them without touching this file.
COMFY_EXTRA_ARGS = shlex.split(os.environ.get("VISIONARY_COMFY_ARGS", ""))

# Regional multi-character LoRA for Krea 2, by a commit rather than a branch —
# the pack was pushed to twice in the week this landed, and a floating ref means
# the graph builder below can stop matching the node it builds for.
#
# Its whole surface on ComfyUI is public: it wraps the diffusion model through
# comfy.patcher_extension, swaps attention through the transformer_options
# `optimized_attention_override` hook, and loads LoRAs with
# comfy.sd.load_lora_for_models. All of it exists at COMFY_SHA, which is what
# makes this an install rather than a vendor — nothing here is patched, so
# forge/VENDOR.md has no successor.
#
# Re-checked across the 2026-08-03 -> 2026-08-22 bump, because a pinned pack
# against a moved host is the one place this arrangement could rot quietly: all
# **12** symbols it names survive, and `patcher_extension.py` is byte-identical.
# The one thing that did move is instructive rather than alarming —
# `optimized_attention_override` grew an optional `container_function` branch
# and kept `override(func, *args, **kwargs)` as the fallback, so a pack passing
# a plain callable takes the same path it always did. That is the shape of
# nearly every ComfyUI change in the range: additive, because breaking a node
# id or a hook breaks thousands of installs at once.
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

# Motion continuation for H3 scene chaining. A scene is longer than one
# generation (H3_MAX_FRAMES caps a take at ~14.4s), and what turns several
# generations into one scene is that the next opens where the last stopped.
# Opening on a *still* of the last frame loses the one thing a still cannot
# carry — momentum — and restarts the audio dead. This pack slices the previous
# clip's sampler latent (picture *and* sound) in as pinned context, so motion
# and audio genuinely continue across the join.
#
# Pinned like CLIFF, installed not vendored, GPL-3.0 like ComfyUI itself (we
# only talk to it over HTTP). Its runtime patches install lazily on the first
# Motion Context run and self-test first, refusing to run when ComfyUI's
# internals have moved — a named refusal, not silent wrong coordinates. Their
# own README: quality compounds down a chain and audio dulls faster than
# picture, so a long scene wants re-anchoring from references every few takes
# rather than an unbounded chain.
H3MC_SHA = "f80e36bc1d7887a143b12e6645313fd6b9cd2aee"  # 0.3.1, 2026-08-15
H3MC_REPO = "https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context"

# The pinned run sits at the head of the clip and is trimmed after decode, so a
# continued take must ask for its seconds *plus* the context or the person gets
# a clip shorter than the duration they set. 22 is the pack's default and its
# own recommendation for near-seamless joins; the choices are 17n+5 aligned.
H3MC_CONTEXT_FRAMES = 22
H3MC_AUDIO_CONTEXT = 24  # one second on the model's 40 Hz audio grid

# What a take's saved latent is called beside its clip: `{job}.context.safetensors`
# in the flat outputs/. ComfyUI's output folder is container disk, so the
# latent is harvested beside the clip — a chain that only worked while the
# container stayed warm would be a chain that breaks precisely on the long
# think between takes. The one honest second file in outputs/: bytes rather
# than a record, which is why it is not inside the clip.
H3MC_SUFFIX = ".context.safetensors"

# Style-by-reference: nkxx188's training-free K/V injection. This REPLACED the
# ostris reference-LoRA path, and the decision was a measurement, not an
# argument — tools/ab_style.py, same seed, two prompts: the ostris path leaked
# reference content both times (a woman rendered male, monogram fabric
# repeated onto dinner plates) while this one carried the style and kept the
# prompt's own subject and scene. It also needs no weight at all and ran ~9s
# against ~40s. Pinned for the same reason CLIFF_SHA is; single-author pack,
# same arity-risk class — re-check on every COMFY_SHA bump.
K2ST_SHA = "b30d495ab7e5626a2effc72a071430297643b718"  # 2026-07-21

# **There is no step cache on the H3 path, and a third one does not go back in
# on a wall-clock number.** Two families were tried and both are gone. CacheDiT
# came out in a day on cost — its wrapper taxed every *computed* step ~50% at
# 768p, more than the skips returned. TeaCache (`VisionaryStepCache`, ours)
# survived that test — 220.0s to 97.4s at production shape, computed steps at
# stock price — and came out on **fidelity**: it distorts H3 LoRAs. A skip does
# not drop a step, it re-applies the previous prediction while the sampler
# takes its normal stride, so the update is misplaced rather than missing —
# which is why the failure looks like a wrong face and never like a weak LoRA.
# The gate reads rel-L1 on the input latent and infers the output has not moved
# either; that inference is a claim about output-per-input gain, and a LoRA is
# a low-rank gain increase in exactly the subspace carrying identity. The
# harness that timed both is tools/ab_cache.py; docs/decisions.md carries the
# account. A fourth attempt needs a fidelity arm with LoRAs loaded, not a
# stopwatch.
K2ST_REPO = "https://github.com/nkxx188/ComfyUI-Krea2-StyleTransfer"
K2ST_REF_NODE = "Krea2StyleReference"
K2ST_TRANSFER_NODE = "Krea2StyleTransfer"

# GGUF, which is how Krea 2 reaches a 16 GB card: Q4_K_M is 7.2 GB against
# 26.3 for bf16. H3 needed no such thing — its small tiers are convrot
# safetensors that the loader already in the graph reads — so this is here for
# the image side alone.
#
# **This is a fork, which is the thing this repo does not do, so here is the
# argument.** The rule forbids *owning a patch*: `forge/` died because we
# maintained a diff against `backend/nn/krea.py`, and what replaced it was a
# clone at a SHA with nothing patched. This is that same shape — a clone, at a
# SHA, unmodified — and the only new fact is that the upstream is itself a fork.
#
# It is a strict superset, not a divergence: `git compare city96...molbal` is
# ahead 58, behind 0. Upstream cannot read Krea 2 files at all, has not been
# pushed since 2026-01-12, and the request for it (city96/ComfyUI-GGUF#464) has
# been open since June. Among those 58 commits are Krea-2 tensor handling, the
# Q8_CR quant type that `molbal/krea2-gguf` ships, and dynamic-VRAM support in
# the loaders, which is the half that matters on a card that has to stream.
#
# **The exit condition is #464 landing upstream**, and it is watched rather than
# remembered — `tools/upstream.py` reads these pins by AST already. It is a
# small upstream (55 stars against city96's 3960), so the pin is doing more work
# here than it does on the other three packs.
#
# In `comfy_image` rather than a local-only path, deliberately. Local-only means
# `_krea2_graph` grows a branch that is dead code on Modal and
# `tools/smoke_graphs.py` fails against an image that has no such node — the
# "second backend that becomes dead code" failure docs/decisions.md already
# names, arrived at from the other side. It costs an H100 one small clone and
# one import; nobody there will ever download a .gguf.
GGUF_REPO = "https://github.com/molbal/ComfyUI-GGUF"
GGUF_SHA = "4fb1ec6209c8eff5b339ca98dff60265bea09758"  # 2026-08-27
GGUF_UNET_NODE = "UnetLoaderGGUF"

# Our own node next to the pack's — one shim, see comfy_nodes/visionary_boxes.
COMFY_NODES_DIR = str(Path(__file__).parent / "comfy_nodes")

app = modal.App(APP_NAME)


# --------------------------------------------------------------------------
# Local mode — one card, one process, the same contract
#
# `modal deploy app.py` is still the entire install and nothing here runs under
# it: `VISIONARY_LOCAL` is set by tools/run_local.py and by nothing else. Every
# decorator, image, Volume and Dict above and below is still constructed
# unconditionally, because all of them are lazy — `Volume.from_name` hands back
# an unhydrated handle and the RPC does not fire until something uses it, which
# is why `import app` costs 0.15s with no credentials and no network.
#
# **Rebinding names rather than routing call sites through accessors is the
# whole trick.** The surface this file asks of a Dict is three operations, and
# of a Volume five; a shim answering those makes every existing call site work
# off Modal *and* every future one, with no edit. The alternative was a
# `_commit()` helper renamed across 28 sites — 28 chances to miss one, and no
# way to notice the miss until a stranger's render came back wrong.
#
# Dispatch is the deliberate exception and stays visible at its eight call
# sites, because dispatch is the one thing that genuinely differs: Modal starts
# a container per call, and a local run has one card that holds one model.
#
# `tools/smoke_local.py` asserts both halves — that the Dict and Volume surfaces
# stay closed, and that no ninth `.spawn(` appears outside `_spawn` — on a
# laptop, in ten seconds, because the person maintaining this will not be the
# person running it.
# --------------------------------------------------------------------------

LOCAL = bool(os.environ.get("VISIONARY_LOCAL"))


class _LocalDict(dict):
    """
    A `modal.Dict` for one process, which is what a local run has.

    Backed by a file only where the contents are a *record*. `jobs` is not one:
    locally the web server and the job are the same process, so a job cannot
    outlive the process that published it, and a record that survived a restart
    would be exactly the stale claim `beat` and `_download_alive` exist to
    disbelieve. `config` and `sessions` are records, and land as plain JSON —
    readable with `cat`, which is the storage-layout rule reaching the one piece
    of state that was never on the volume.
    """

    def __init__(self, path: "Path | None" = None):
        super().__init__()
        self._path = path
        self._lock = threading.RLock()
        if path and path.is_file():
            try:
                self.update(json.loads(path.read_text()))
            except Exception as exc:  # noqa: BLE001 — a corrupt file is not fatal
                # Named rather than swallowed: starting empty is survivable, but
                # silently starting empty reads as "my sessions are gone".
                print(f"[local] {path.name} unreadable, starting empty "
                      f"({type(exc).__name__}: {exc})", flush=True)

    def _flush(self) -> None:
        if not self._path:
            return
        # Staged then renamed, the `_download_weight` rule: a reader arriving
        # mid-write must see the old file whole rather than half of the new one.
        tmp = self._path.with_suffix(f".{threading.get_ident()}.part")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(dict(self), indent=1, default=str))
            tmp.replace(self._path)
        except Exception as exc:  # noqa: BLE001
            print(f"[local] could not write {self._path}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            tmp.unlink(missing_ok=True)

    def __setitem__(self, key, value):
        with self._lock:
            super().__setitem__(key, value)
            self._flush()

    def pop(self, key, *default):
        with self._lock:
            out = super().pop(key, *default)
            self._flush()
            return out


class _LocalCommit:
    """
    `volume.commit`, which locally has nothing to do.

    An object rather than a method because one caller reaches for
    `volume.commit.aio()` — `/api/upload`, the one `async def` route, where a
    blocking commit would stall the event loop for the whole container. That
    call has to keep working, so `.aio` has to exist.
    """

    def __call__(self) -> None:
        return None

    async def aio(self) -> None:
        return None


class _LocalEntry:
    """One row of `volume.listdir`, carrying the real enum so `e.type != FILE`
    in `_entries_by_rpc` compares against what it was written to compare."""

    __slots__ = ("path", "type", "mtime", "size")

    def __init__(self, path: str, mtime: float, size: int):
        self.path = path
        self.type = modal.volume.FileEntryType.FILE
        self.mtime = mtime
        self.size = size


class _LocalVolume:
    """
    A `modal.Volume` for a directory that is simply there.

    The read path this stands in for exists because a Modal mount can serve
    stale bytes and a reload is refusable while safetensors holds the volume
    open — so the gallery asks the *committed* state by RPC instead. None of
    that is true of a local directory: the mount is the record, there is no
    second copy to be behind, and a commit is a no-op rather than a cost.

    Answering `read_file` and `listdir` off the disk — rather than routing the
    gallery onto its `_entries_by_walk` fallback — is deliberate. The fallback
    prints a line saying the listing may be stale, and it should keep meaning
    that. A local run is not a degraded RPC; it is a filesystem that cannot lag.
    """

    def __init__(self, root: Path):
        self.root = root
        self.commit = _LocalCommit()

    def reload(self) -> None:
        return None

    def _resolve(self, rel: str) -> Path:
        return self.root / str(rel).lstrip("/")

    def read_file(self, rel: str):
        """Chunks, as a generator — so a missing file raises FileNotFoundError
        on iteration rather than at the call, which is where `_read_committed`
        catches it in order to try the other path spelling."""
        with open(self._resolve(rel), "rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    return
                yield chunk

    def listdir(self, path: str, recursive: bool = False):
        """Volume-relative paths, which is the spelling `_entries_by_rpc`
        already tolerates ("returned volume-relative in testing, even though
        '/outputs' went in")."""
        base = self._resolve(path)
        if not base.is_dir():
            return []
        walk = base.rglob("*") if recursive else base.iterdir()
        out = []
        for f in walk:
            try:
                if not f.is_file():
                    continue
                st = f.stat()
            except OSError:
                continue
            out.append(_LocalEntry(str(f.relative_to(self.root)),
                                   st.st_mtime, st.st_size))
        return out


class _Lane:
    """
    One queue, and as many workers as the hardware can actually run at once.

    This is not a second job system. It writes into the same `jobs` record
    through the same `_publish`, reports through the same `/api/status`, and
    answers the same `_request_stop` — what it adds is an ordering that Modal
    supplied by starting another container, and a local machine cannot.

    A queued job says how many are ahead of it and re-says it when that number
    changes, because a wait costs what it shows rather than what it takes, and
    "queued" alone is the still button the antifragile rule is about.
    """

    def __init__(self, name: str, workers: int):
        self.name = name
        self._pending: deque = deque()
        # Counted separately from the queue, because a job that has been picked
        # up is no longer *in* the queue and is still the thing everything else
        # is waiting for. Without this the first job to queue behind a running
        # render computed nothing ahead of it and so published nothing — a card
        # that says "queued" and never says what for, which is the still button
        # this whole mechanism exists to avoid.
        self._running = 0
        self._cv = threading.Condition()
        for i in range(workers):
            threading.Thread(target=self._work, name=f"lane-{name}-{i}",
                             daemon=True).start()

    def submit(self, fn, args, kwargs, job_id: "str | None") -> None:
        with self._cv:
            self._pending.append((fn, args, kwargs, job_id))
            ahead = len(self._pending) - 1 + self._running
            self._cv.notify()
        if job_id and ahead:
            _publish(job_id, status="queued", phase=_ahead(ahead))

    def _work(self) -> None:
        while True:
            with self._cv:
                while not self._pending:
                    self._cv.wait()
                fn, args, kwargs, job_id = self._pending.popleft()
                self._running += 1
                waiting, running = list(self._pending), self._running
            for i, (_, _, _, other) in enumerate(waiting):
                if other:
                    _publish(other, status="queued", phase=_ahead(i + running))
            if job_id and _stop_requested(job_id):
                # Stopped while it waited. Publishing the terminal status the
                # bodies publish — rather than starting the body so its own
                # `_stop_gate` can — is worth the duplication here only because
                # locally the alternative is loading 16-40 GB of checkpoint in
                # order to discover that nobody wants it.
                _publish(job_id, status="stopped", phase="stopped in the queue")
                _clear_stop(job_id)
                continue
            try:
                # `.local()`, not `get_raw_f()`: it is the only one of the two
                # that works for a Cls method, where it binds to the Obj's own
                # instance and runs `@modal.enter()` first. That lifecycle is
                # what loads the checkpoint, and `_on_gpu` memoizing the Obj is
                # what stops it running per request.
                fn.local(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — a worker must outlive a job
                print(f"[lane:{self.name}] {type(exc).__name__}: {exc}",
                      flush=True)
                if job_id:
                    _publish(job_id, status="failed",
                             error=f"{type(exc).__name__}: {exc}")
            finally:
                with self._cv:
                    self._running -= 1


def _ahead(n: int) -> str:
    return "queued — next up" if n == 1 else f"queued — {n} ahead"


# Which lane a function runs in, by `info.function_name`. Unknown names take
# the GPU lane on purpose: it is the serialising one, so a dispatch site added
# later is slow-but-correct rather than an out-of-memory crash, and
# `tools/smoke_local.py` names the ones it expects so a new one is still noticed.
_LANE_OF = {
    "download_job": "cpu",
    "download_missing_job": "cpu",
    "gdrive_job": "cpu",
    "caption_job": "gpu",
    "train_job": "gpu",
    "export_scene_job": "cpu",
    "migrate_outputs_job": "cpu",
    # Three placements share two method bodies up the MRO, and Modal names a
    # method by the class it was reached through — so every combination is a
    # name here. The table is spelled out rather than derived because
    # `tools/smoke_local.py` checks these names against the classes, and a
    # derived table would agree with itself while both halves were wrong.
    "ImageGenerator.generate_image": "gpu",
    "VideoGenerator.generate_video": "gpu",
    "BothGenerator.generate_image": "gpu",
    "BothGenerator.generate_video": "gpu",
    # The Playground drives the same warm ComfyUI, so it queues behind renders
    # for the same reason they queue behind each other.
    "ImageGenerator.playground": "gpu",
    "VideoGenerator.playground": "gpu",
    "BothGenerator.playground": "gpu",
    "ImageGenerator.restart_engine": "gpu",
    "VideoGenerator.restart_engine": "gpu",
    "BothGenerator.restart_engine": "gpu",
    "playground_catalogue_job": "gpu",
    # A knock on a door that is already open — locally the process is up and the
    # checkpoint loads on the first real request, so warming is what `_on_gpu`'s
    # memo already does.
    "ImageGenerator.warm": "inline",
    "VideoGenerator.warm": "inline",
    "BothGenerator.warm": "inline",
}
_LANES: "dict[str, _Lane]" = {}
_LANES_LOCK = threading.Lock()


def _lane(name: str) -> _Lane:
    with _LANES_LOCK:
        if name not in _LANES:
            # One GPU worker because there is one card: a second concurrent
            # render is an out-of-memory failure, not throughput. Two on the CPU
            # lane because that is what the downloads already are — `web_image`
            # has no GPU and `_active_download` is the thing that serialises
            # them, for the uplink's sake rather than the card's.
            _LANES[name] = _Lane(name, 2 if name == "cpu" else 1)
        return _LANES[name]


def _spawn(fn, *args, **kwargs):
    """
    Start a job and stop caring about it — the contract this app already has.

    On Modal that is `.spawn()`: fire and forget, the handle discarded, the
    `job_id` the only correlator. Locally it is a lane (see `_Lane`), which is
    the same contract with the ordering made explicit, because one card cannot
    do what a container-per-call did.

    `job_id` is read out of the keyword arguments rather than passed separately:
    every call that can actually queue — both generators, captioning, training —
    already passes it by keyword, and the ones that do not are the CPU downloads,
    which `_active_download` was already serialising by itself.
    """
    if not LOCAL:
        return fn.spawn(*args, **kwargs)
    try:
        name = fn.info.function_name
    except Exception:  # noqa: BLE001 — an unnamed function still has to run
        name = ""
    lane = _LANE_OF.get(name, "gpu")
    if lane == "inline":
        return None
    _lane(lane).submit(fn, args, kwargs, kwargs.get("job_id"))
    return None


# What Modal *mounts*, as against what this file reads and writes. On Modal
# they are the same three objects; locally they are deliberately not, because a
# `@app.function` decorator validates its `volumes=` against real Volumes at
# import time and would reject a shim — while the code below wants something
# that answers commit, reload, read_file and listdir against a directory.
#
# Captured here, above the rebind, so the decorators keep the real handles no
# matter what the names below them come to mean. A local run never mounts
# anything, so what these hold then is unused rather than wrong.
MODAL_VOLUMES = {"/workspace": volume, "/models": models_volume}
MODAL_VOLUMES_HF = {**MODAL_VOLUMES, str(HF_CACHE): hf_cache}
# The data volume alone, for the jobs that have no business holding the
# weights open: exporting a scene and migrating the output tree.
MODAL_VOLUMES_WS = {"/workspace": volume}


if LOCAL:
    # The rebind. Everything above this line is a definition; this is the only
    # place the running program changes shape, and it is four names.
    LOCAL_ROOT = WORKSPACE
    jobs = _LocalDict()
    config = _LocalDict(LOCAL_ROOT / ".local" / "config.json")
    sessions = _LocalDict(LOCAL_ROOT / ".local" / "sessions.json")
    # One object for all three mounts, because locally they are one filesystem.
    # `hf_cache` is only ever mounted, never called, so it rides along.
    volume = models_volume = hf_cache = _LocalVolume(WORKSPACE)
    # One card, so the picker offers one card. `GPU` and `VIDEO_GPU` are already
    # env-driven and the launcher fills them from the detected device, so the
    # menu — which builds itself out of these, like every other menu on the page
    # — needs no change on the front end, and `_on_gpu` always returns the base
    # runner because the request can only ever equal the default.
    IMAGE_GPUS = (GPU,)
    VIDEO_GPUS = (VIDEO_GPU,)


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
        # 11.3, not 11.1: AVIF decode arrived in Pillow's wheels at 11.2, and
        # `.avif` is in IMAGE_EXTS. On 11.1 every AVIF upload was invisible to
        # PIL while the browser drew it fine — thumbnail 500 (an empty tile),
        # no dimensions in the listing, no fingerprint for the duplicate scan —
        # so the set looked broken everywhere except the full-screen viewer.
        "pillow==11.3.0",
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
        # The similar class's eye: CLIP embeddings on CPU. onnxruntime rather
        # than torch because this container's whole ML duty is one 85 MB int8
        # encoder — a torch install is ~2 GB of image for the same matmul.
        # numpy is named although onnxruntime would drag it in, because app.py
        # imports it directly for the cosine matrix.
        "onnxruntime==1.27.0",
        "numpy==2.2.6",
    )
    # The model itself, pinned by revision and verified by checksum, baked at
    # build time. Not a gear-menu download: the duplicate review must work on a
    # fresh deploy with nothing chosen yet, and 85 MB is a dependency like a
    # wheel, not a checkpoint anybody picks. Layered before the front end so
    # editing a component does not re-pull it.
    .run_commands(
        "python -c \"import urllib.request; urllib.request.urlretrieve("
        "'https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/"
        "d15189d7028b43f1d3e65039190477f6af591c2a/onnx/vision_model_quantized.onnx',"
        " '/opt/similar_clip_b32_q8.onnx')\"",
        "python -c \"import hashlib; h = hashlib.sha256(open("
        "'/opt/similar_clip_b32_q8.onnx', 'rb').read()).hexdigest(); "
        "assert h == '583fd1110a514667812fee7d684952aaf82a99b959760c8d7dca7e0"
        "ab9839299', h\"",
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
    # ---- the front end, built into the image ------------------------------
    #
    # `modal deploy app.py` is the entire install, and that is the whole reason
    # the build happens here rather than on your machine. Mounting a local
    # `web/dist` would be simpler and would quietly make the deploy command a
    # lie: a fresh clone has no dist, and a stale one deploys whatever you last
    # built, which is the worst of the three because it looks like it worked.
    # Node is a build-time dependency only — nothing at runtime needs it.
    #
    # Layered in this order on purpose. The lockfile lands before the sources,
    # so editing a component re-runs `npm run build` (about a second) and not
    # `npm ci` (about a minute) — Modal invalidates from the first changed layer
    # down, so the order of these four lines is the difference.
    # A pinned tarball, not `apt_install("nodejs", "npm")`.
    #
    # Debian bookworm ships Node 18.20.4, and Vite 7 wants 20.19+ — it built
    # anyway and printed "Please upgrade your Node.js version" every time,
    # which is a build that works by luck on a floor upstream has already
    # declared unsupported. The same shape as the cudnn pin NVIDIA pruned:
    # a version resolved by somebody else's release schedule, quietly, until
    # the day it stops.
    #
    # NODE_VERSION is here rather than in a variable at the top because it is
    # the only place it can be read from — it belongs with the layer it builds,
    # the way COMFY_SHA belongs with the clone. Bump it deliberately.
    .apt_install("curl", "xz-utils")
    .run_commands(
        "curl -fsSL https://nodejs.org/dist/v24.19.0/node-v24.19.0-linux-x64.tar.xz"
        " -o /tmp/node.tar.xz",
        "mkdir -p /opt/node && tar -xJf /tmp/node.tar.xz -C /opt/node --strip-components=1",
        "rm /tmp/node.tar.xz",
        # Symlinked rather than put on PATH with .env, which *replaces* the
        # variable: node would be reachable and whatever Modal had put there
        # would not be.
        "ln -sf /opt/node/bin/node /opt/node/bin/npm /opt/node/bin/npx /usr/local/bin/",
    )
    .add_local_file("web/package.json", "/build/web/package.json", copy=True)
    .add_local_file("web/package-lock.json", "/build/web/package-lock.json", copy=True)
    .run_commands("cd /build/web && npm ci")
    .add_local_dir("web", "/build/web", copy=True, ignore=["node_modules", "dist"])
    .run_commands("cd /build/web && npm run build")
)

# **web_image plus ffmpeg, and it is a derived image rather than a fatter one.**
# The export stitches finished takes into one file, which is CPU work — so it
# gets a container, not a share of a GPU. What it must not get is a place in the
# image the *UI* runs from: `web` is the container the page waits on before it
# can draw anything, and ffmpeg's apt tree is ~100 MB of cold start charged to
# every session for a button pressed at the end of a scene. Derived from
# `web_image`, so every layer under it is the one already built and cached and
# the only new layer is the one this needs.
#
# It is also why `web_image has no ffmpeg` stays true, and why the two notes
# that say so — the clip cover in the gallery, the character-file transcode —
# are still accurate rather than quietly stale.
export_image = web_image.apt_install("ffmpeg")

trainer_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.10"
    )
    # git for the build-time clone; libgl1/libglib2.0-0 are opencv-python's
    # runtime libs (a musubi dependency). No unzip — extraction is pure Python.
    .apt_install("git", "libgl1", "libglib2.0-0")
    # torch first from the cu124 index so musubi's install sees it satisfied and
    # does not pull a default-CUDA wheel over the top.
    #
    # `extra_index_url` is pypi.org, and it is not decoration — without it this
    # build stopped working one day having changed nothing.
    #
    # torch 2.5.1 pins `nvidia-cudnn-cu12==9.1.0.70` exactly, and the PyTorch
    # index is a *proxy*: its listing for that package is a page of hrefs
    # pointing at pypi.nvidia.com. NVIDIA pruned 9.1.0.70 from their CDN — their
    # 9.1 line now starts at 9.1.1.17 — so the link went, and PyTorch's
    # generated listing went with it. pip was reading only the index it was
    # told to and that index had stopped carrying a version the wheel beside it
    # still requires. The failure names a package nothing in this file mentions.
    #
    # Nothing injected a second index; the base image sets no pip config at all.
    # That was checked, because the first explanation written here was that the
    # CUDA image put NVIDIA's index in front of PyPI, and it is not true.
    #
    # The rule the shape teaches: an index that proxies someone else's files
    # inherits their retention policy, so a pin is only as reproducible as the
    # *least durable* host in the chain that serves it. pypi.org still has the
    # file. Give pip somewhere else to look whenever the primary index is a
    # vendor's.
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        index_url="https://download.pytorch.org/whl/cu124",
        extra_index_url="https://pypi.org/simple",
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
        "accelerate==1.14.0",
        "huggingface_hub==1.29.0",
        "pillow==11.3.0",
        # transformers 5, and the major is the point rather than the accident:
        # this image exists so the captioner can move without re-litigating
        # musubi's pins or ComfyUI's, and it is the only one of the three that
        # can be on 5.x today. Exact, not `>=`, for the reason every pin here is
        # exact — a build that changes under you is a build you cannot bisect.
        "transformers==5.16.1",
    )
    # No hf_transfer any more. huggingface_hub 1.x pulls `hf-xet` as a hard
    # dependency and Xet *is* the accelerated path now, so the package this
    # image used to install and the env var that switched it on are both dead
    # weight — and an env var that names a library nobody installs is the
    # failure the old comment here was warning about, one release later.
    .env({"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
)


# Inference: ComfyUI on a CUDA 13 torch wheel. Images and video both.
#
# One image, one class body, three placements. Image and video are separate
# classes by default because each holds a checkpoint resident and
# `max_containers=1` is per class — on one class an image request waits behind
# a ten-minute clip. The third placement, `BothGenerator`, is that trade made
# on purpose from the gear: one warm card instead of two, and the wait. They
# all share the *build* because there is nothing per-family in it: the same
# ComfyUI, the same torch, the same attention kernels, and one node pack that
# only the image side loads.
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
    # TORCH_CUDA_ARCH_LIST must cover every card in IMAGE_GPUS, VIDEO_GPUS and
    # BOTH_GPUS. 9.0 is Hopper (H100/H200), 8.9 is Ada (L40S); add "10.0" for
    # B200 and force a rebuild, or the kernels will not load. One image means
    # one list, and a card missing from it is a model that loads and then runs
    # on the slow path with nothing to say so — which is how the A100 Krea 2
    # used to run on left, and how L40S came in: a string here, and a rebuild.
    .env({"TORCH_CUDA_ARCH_LIST": "8.9;9.0", "MAX_JOBS": "8"})
    # --no-build-isolation is required (the build imports the torch installed
    # above rather than a fresh one), but it also means pip supplies nothing:
    # without `wheel` the build dies on `invalid command 'bdist_wheel'`, and
    # without `ninja` torch's cpp_extension silently falls back to distutils
    # and compiles the CUDA kernels single-threaded.
    .pip_install("wheel", "setuptools", "packaging", "ninja")
    .run_commands(
        "git clone --depth 1 https://github.com/thu-ml/SageAttention /tmp/sage",
        # Upstream's setup.py turns TORCH_CUDA_ARCH_LIST into one -gencode list
        # and hands it to every extension. With Ada on the list that assembles
        # the Hopper-only `_qattn_sm90` kernel for sm_89 as well, and ptxas
        # refuses it — "Feature 'cp.async.bulk.tensor' requires .target sm_90
        # or higher" — which is how adding L40S broke the image. The Ada and
        # Ampere kernels compile for sm_90a fine, so only the sm90 extension's
        # list is narrowed to its own card; runtime dispatch is already by
        # `get_device_capability`, so nothing else has to know. It asserts on
        # its anchors rather than no-op'ing, so if upstream reshapes the file
        # the build stops here naming this patch instead of three thousand
        # ptxas lines later with the error above.
        "cd /tmp/sage && python -c '"
        r's = open("setup.py").read(); '
        r'old = "\"nvcc\": NVCC_FLAGS}"; '
        r'i = s.find("name=\"sageattention._qattn_sm90\""); '
        r'k = s.find(old, i); '
        r'j = s.find("extra_link_args", i); '
        r'a = "    if HAS_SM90:\n"; '
        r'assert 0 <= i < k < j and s.count(a) == 1, '
        r'"SageAttention setup.py changed shape: the sm90 -gencode narrowing in '
        r'app.py no longer finds its anchors, so re-read setup.py and re-fit it"; '
        r's = s[:k] + "\"nvcc\": NVCC_FLAGS_SM90}" + s[k + len(old):]; '
        r'd = "    NVCC_FLAGS_SM90 = [f for f in NVCC_FLAGS if f != \"-gencode\" '
        r'and not f.startswith(\"arch=\")] + [\"-gencode\", '
        r'\"arch=compute_90a,code=sm_90a\"]\n"; '
        r's = s.replace(a, d + a); '
        r'open("setup.py", "w").write(s)'
        "'",
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
        f"git clone {H3MC_REPO} {COMFY}/custom_nodes/h3_motion_context",
        f"cd {COMFY}/custom_nodes/h3_motion_context && git checkout {H3MC_SHA}",
        f"git clone {CLIFF_REPO} {COMFY}/custom_nodes/krea2_regional",
        f"cd {COMFY}/custom_nodes/krea2_regional && git checkout {CLIFF_SHA}",
        f"git clone {K2ST_REPO} {COMFY}/custom_nodes/krea2_styletransfer",
        f"cd {COMFY}/custom_nodes/krea2_styletransfer && git checkout {K2ST_SHA}",
        f"git clone {GGUF_REPO} {COMFY}/custom_nodes/gguf",
        f"cd {COMFY}/custom_nodes/gguf && git checkout {GGUF_SHA}",
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
    .add_local_dir(
        f"{COMFY_NODES_DIR}/visionary_free_regional",
        remote_path=f"{COMFY}/custom_nodes/visionary_free_regional",
    )
    .add_local_dir(
        f"{COMFY_NODES_DIR}/visionary_edit_arity",
        remote_path=f"{COMFY}/custom_nodes/visionary_edit_arity",
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
        "fits_vram_gb": 40.0,
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
        "fits_vram_gb": 40.0,
    },
    # ── Krea 2 on a card that cannot hold it ───────────────────────────────
    #
    # Q4_K_M is 7.2 GB against bf16's 26.3, which is the difference between a
    # 12 GB card and no card at all. K-quants rather than the classic Q4_0/Q5_1
    # line the other published repo carries: they quantise the tensors that
    # matter less, harder, which is the whole reason the K series exists.
    #
    # These need `GGUF_UNET_NODE` and therefore the pinned fork — see GGUF_SHA
    # for why a fork was taken and what ends it. Both sizes are the files' own,
    # off the HuggingFace API.
    #
    # **Not gated, where the weights they quantise are.** That is a fact about
    # a third-party re-upload, not permission granted: Krea's licence is
    # accepted on their repo and these do not ask. Worth knowing before pulling
    # one instead of the file it is a copy of.
    #
    # The text encoder stays bf16 at 8.9 GB and is not quantised here, so 16 GB
    # holds a Q4 DiT and a bf16 encoder only if ComfyUI evicts the encoder after
    # conditioning. That is the one thing the arithmetic cannot settle, and it
    # is on the rented-box list rather than written into the README.
    "krea2_turbo_q4": {
        "label": "Krea 2 Turbo (Q4_K_M)",
        "note": "8-step images on a 16 GB card",
        "family": "Krea 2 — images",
        "repo_id": "realrebelai/KREA-2_GGUFs",
        "filename": "TURBO/Krea-2-Turbo-Q4_K_M.gguf",
        "dest": MODELS / "krea2-turbo-Q4_K_M.gguf",
        "gated": False,
        "approx_gb": 7.2,
        "tier_of": "turbo",
        "fits_vram_gb": 16.0,
    },
    "krea2_turbo_q5": {
        "label": "Krea 2 Turbo (Q5_K_S)",
        "note": "8-step images, closer to full precision",
        "family": "Krea 2 — images",
        "repo_id": "realrebelai/KREA-2_GGUFs",
        "filename": "TURBO/Krea-2-Turbo-Q5_K_S.gguf",
        "dest": MODELS / "krea2-turbo-Q5_K_S.gguf",
        "gated": False,
        "approx_gb": 8.8,
        "tier_of": "turbo",
        "fits_vram_gb": 24.0,
    },
    "krea2_raw_q4": {
        "label": "Krea 2 RAW (Q4_K_M)",
        "note": "Full sampler, on a 16 GB card",
        "family": "Krea 2 — images",
        "repo_id": "realrebelai/KREA-2_GGUFs",
        "filename": "BASE/Krea-2-Base-Q4_K_M.gguf",
        "dest": MODELS / "krea2-raw-Q4_K_M.gguf",
        "gated": False,
        "approx_gb": 7.3,
        "tier_of": "raw",
        "fits_vram_gb": 16.0,
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
    # node's combo about a second directory.
    #
    # `internal` keeps it out of the pickers. "It appears in the picker like
    # any other file, which is honest" was written when the picker was prompt
    # syntax and typing `<lora:krea2_identity_edit_v1_2:1>` was a deliberate
    # act; the picker is a chip menu now, where the same entry is one mis-click
    # from loading the compose path's own machinery into an ordinary render.
    # The scene/outfit path still loads it by name, and Settings still lists,
    # downloads and deletes it — hidden from the pickers is not hidden.
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
        "arch": "krea2",
        "internal": True,
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
        "fits_vram_gb": 24.0,
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
        "fits_vram_gb": 24.0,
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
        "fits_vram_gb": 24.0,
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
    # ── Smaller tiers of the same three weights ────────────────────────────
    #
    # `tier_of` names the slot a row is an alternative for, and `fits_vram_gb`
    # is the card it is meant for. Nothing here is a mode: a tier is chosen by
    # downloading it, the same way every other weight on this page is chosen,
    # and `_slot_name()` picks between whatever you actually have.
    #
    # **Verified against the HuggingFace API, and every size below is the file's
    # own.** These are the same pruned-convrot lineage as the int8 rows above —
    # not GGUF — so they load through the loader already in the graph, which is
    # why 16 GB video costs a table entry rather than a node. That much is
    # inference from format and provenance and has not been run: the rented box
    # confirms it or these rows come out.
    #
    # NVFP4 is deliberately absent as a *default* anywhere. It has no compute
    # path before Blackwell — on Ada it is a memory saving and nothing else —
    # so it belongs to sm_120 and is not a tier to pick by size.
    "h3_dit_int4": {
        "label": "MiniMax-H3 (mixed int4)",
        "note": "Video DiT for a 16 GB card — t2v and i2v",
        "family": "MiniMax-H3 — video with sound",
        "repo_id": "Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot",
        "filename": "MiniMax_H3_FL2VA_pruned_mixed_int4_int8_convrot.safetensors",
        "dest": MODELS / "minimax_h3_fl2va_pruned_mixed_int4_int8_convrot.safetensors",
        "gated": False,
        "approx_gb": 15.9,
        "tier_of": "h3_dit",
        "fits_vram_gb": 16.0,
    },
    "h3_ref_dit_int4": {
        "label": "MiniMax-H3 Reference (mixed int4)",
        "note": "Reference DiT for a 16 GB card",
        "family": "MiniMax-H3 — video with sound",
        "repo_id": "Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot",
        "filename": "MiniMax_H3_Ref2VA_pruned_mixed_int4_int8_convrot.safetensors",
        "dest": MODELS / "minimax_h3_ref2va_pruned_mixed_int4_int8_convrot.safetensors",
        "gated": False,
        "approx_gb": 15.1,
        "tier_of": "h3_ref_dit",
        "fits_vram_gb": 16.0,
    },
    # The text encoder is the stubborn half of the 16 GB question: nvfp4 above
    # is 15.7 GB and this is 14.95, so neither is small. What makes 16 GB work
    # is that the encoder runs first and is evicted before the DiT loads —
    # sequential residency, which is the specific thing the rented box has to
    # confirm rather than the size arithmetic.
    "h3_te_int4": {
        "label": "Qwen3-VL 32B (int4)",
        "note": "H3 conditioner for a 16 GB card",
        "family": "MiniMax-H3 — video with sound",
        "repo_id": "Abiray/MiniMax-H3-GGUF",
        "filename": "text_encoders/qwen3vl_32b_minimax_h3_int4_convrot.safetensors",
        "dest": MODELS / "qwen3vl_32b_minimax_h3_int4_convrot.safetensors",
        "gated": False,
        "approx_gb": 15.0,
        "tier_of": "h3_te",
        "fits_vram_gb": 16.0,
    },
    # ── MiniMax-H3 speed LoRAs ─────────────────────────────────────────────
    #
    # MiniMax's own Lightning distillations, in the repo the H3 weights already
    # come from — 20 steps down to 8 or 4. They are the answer to the line this
    # file used to carry, that H3 had "no ecosystem for the int8 repackage":
    # there is one, it is first-party, and it was two directories from the
    # transformer we were already downloading.
    #
    # They load through `LoraLoaderModelOnly` like anything else, and the
    # `turbo_mode` toggle ComfyUI's tutorial describes is that node with a
    # switch in front of it. Worth writing down, because grepping
    # `comfy_extras/nodes_minimax_h3.py` for "turbo" returns nothing at either
    # COMFY_SHA or master and reads as proof the feature does not exist. It
    # does: the t2v and i2v templates wrap it in a *subgraph* and promote
    # `turbo_mode`, `turbo_model_strength` and `turbo_steps` as widgets, and
    # `video_minimax_h3_r2v.json` — which is not a subgraph — shows the parts
    # unwrapped: a `PrimitiveBoolean`, two `ComfySwitchNode`s choosing between
    # the bare DiT and `LoraLoaderModelOnly(DiT)` and between two step counts.
    #
    # So there is nothing to add here. That switch exists because a template is
    # a fixed graph and needs a way to turn a node off; this console has a LoRA
    # picker and a steps control, so the toggle is two controls it already has.
    #
    # One provenance note the templates carry and the repo does not: the 8-step
    # file is `lightx2v/Minimax-h3-Turbo` upstream and is *mirrored* into
    # Comfy-Org/MiniMax-H3. Pulled from the mirror so the whole family is one
    # repo id and one token.
    #
    # Paired to a transformer rather than to an expert: fl2v is the t2v/i2v
    # DiT, ref2v is the reference one. Crossing them is not something this
    # picker can prevent — see the note on `vidExpert` — so the labels say
    # which is which.
    "h3_speed_fl2v_8": {
        "label": "H3 speed · 8-step",
        "note": "Lightning turbo LoRA — t2v and i2v",
        "family": "MiniMax-H3 speed LoRAs",
        "repo_id": "Comfy-Org/MiniMax-H3",
        "filename": "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        "dest": LORAS / "h3-speed" / "fl2v-8step.safetensors",
        "gated": False,
        "approx_gb": 2.0,
        "arch": "h3",
    },
    "h3_speed_fl2v_4": {
        "label": "H3 speed · 4-step 768p",
        "note": "Lightning turbo LoRA — t2v and i2v, 768p",
        "family": "MiniMax-H3 speed LoRAs",
        "repo_id": "Comfy-Org/MiniMax-H3",
        "filename": "loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
        "dest": LORAS / "h3-speed" / "fl2v-4step-768p.safetensors",
        "gated": False,
        "approx_gb": 2.0,
        "arch": "h3",
    },
    "h3_speed_ref2v_4": {
        "label": "H3 speed · 4-step reference",
        "note": "Lightning turbo LoRA — ref2va only",
        "family": "MiniMax-H3 speed LoRAs",
        "repo_id": "Comfy-Org/MiniMax-H3",
        "filename": "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        "dest": LORAS / "h3-speed" / "ref2v-4step.safetensors",
        "gated": False,
        "approx_gb": 2.0,
        "arch": "h3",
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
# What the picker writes for one of these when nothing else says otherwise.
# Measured on first-run renders rather than argued: retroanime at the generic
# 1.0 with no phrase in the prompt reads as a grade over the picture, not a
# style — the weight is live and looks like it did nothing, which is the
# failure the LoRA note exists to catch. At 1.3 with the phrase present the
# style is unmistakably on. Served per entry so the page never hardcodes it.
KREA_STYLE_STRENGTH = 1.3
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
        "arch": "krea2",
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


def _catalogue_lora_roots() -> dict[str, dict[str, Any]]:
    """
    {listing row -> its catalogue spec} for every weight that lands in loras/.

    Keyed on the row rather than on the dest, because those are not the same
    thing: the identity-edit weight is a loose file at the top of loras/ while a
    trained LoRA is a folder holding its epochs, and it is the folder that one
    line of the LoRA listing stands for.

    The whole spec rather than the family alone, because the listing now serves
    three of its facts: `family` (the delete confirmation says a catalogue LoRA
    can be re-downloaded, where a trained one cannot), `arch` (what keeps an H3
    speed LoRA out of the image picker and a Krea style out of the video one —
    a LoRA for the wrong architecture loads nothing and warns nowhere), and
    `internal` (the identity-edit weight, which the compose path loads by name
    and a picker should never offer).
    """
    out: dict[str, dict[str, Any]] = {}
    for spec in MODEL_CATALOGUE.values():
        dest: Path = spec["dest"]
        if LORAS not in dest.parents:
            continue
        root = dest
        while root.parent != LORAS:
            root = root.parent
        out[str(root)] = spec
    return out


CATALOGUE_LORA_ROOTS = _catalogue_lora_roots()

# The three weights musubi is handed on the command line. Turbo used to have an
# alias here too, for the Forge pipeline that took absolute paths; ComfyUI's
# loaders take a basename inside a search path, so the graph builders read
# `MODEL_CATALOGUE[...]["dest"].name` and nothing needs a fourth constant.
RAW_PATH = MODEL_CATALOGUE["raw"]["dest"]
VAE_PATH = MODEL_CATALOGUE["vae"]["dest"]
TE_PATH = MODEL_CATALOGUE["text_encoder"]["dest"]

# Alternatives per slot, best first. Built from the catalogue rather than
# written twice, so a tier added above appears here by existing.
SLOT_TIERS: dict[str, list[str]] = {}
for _k, _spec in MODEL_CATALOGUE.items():
    if _spec.get("tier_of"):
        SLOT_TIERS.setdefault(_spec["tier_of"], []).append(_k)
for _base, _alts in SLOT_TIERS.items():
    # Largest first, which for one architecture is also best first: these are
    # quantisations of one checkpoint, so size is precision.
    _alts.sort(key=lambda k: MODEL_CATALOGUE[k]["approx_gb"], reverse=True)


def _dit_node(name: str) -> dict[str, Any]:
    """The loader for one diffusion model, chosen by what the file is.

    `UNETLoader` cannot read a GGUF and `UnetLoaderGGUF` takes no
    `weight_dtype`, so this is one conditional on a suffix rather than a mode:
    the file the slot resolved to already says which it is, and nobody picks a
    loader. Both graph builders use it, so an H3 GGUF — none of which is
    currently a catalogue row, since its small tiers are convrot safetensors —
    would need nothing here.
    """
    if name.endswith(".gguf"):
        return {"class_type": GGUF_UNET_NODE, "inputs": {"unet_name": name}}
    return {"class_type": "UNETLoader",
            "inputs": {"unet_name": name, "weight_dtype": "default"}}


def _slot_name(key: str) -> str:
    """
    Which file the graph should ask for in one slot, given what is on disk.

    **A working default, chosen by what you downloaded, with an override that
    does not argue.** The catalogue's own row is the answer unless a smaller
    tier is present and the card cannot hold the full one — so a Modal
    deployment, where `GPU_VRAM_GB` is 0 because nobody said, always resolves to
    exactly what it resolved to before this existed. Nothing about the deployed
    path changes.

    The override is `config["weight_tiers"][slot]`, set under the gear, and it
    is honoured whenever the file it names is actually present. It is not
    checked against the card: somebody who wants Q8 on a 12 GB card is asking
    for a slow render, not making a mistake, and a gate here would be this file
    having an opinion about a trade only the person rendering can price. The
    same latitude the custom captioner row takes, for the same reason.

    Falls back to the base row whenever nothing else resolves, including when
    the override names a file that has since been deleted — a stale override
    should cost a render at the default quality, never a job that cannot start.
    """
    alts = SLOT_TIERS.get(key)
    base = MODEL_CATALOGUE[key]["dest"]
    if not alts:
        return base.name

    chosen = (config.get("weight_tiers") or {}).get(key)
    if chosen:
        spec = MODEL_CATALOGUE.get(chosen)
        if spec and spec["dest"].is_file():
            return spec["dest"].name

    # Candidates, best first, and "best" is size because these are
    # quantisations of one checkpoint. `fits_vram_gb` is a *declared* fact per
    # row rather than arithmetic over `approx_gb`: a 15.7 GB encoder fits 16 GB
    # on paper and has nothing left to run in, and the difference between those
    # two readings is the whole reason the field exists.
    here = [k for k in (key, *alts) if MODEL_CATALOGUE[k]["dest"].is_file()]
    if not here:
        return base.name
    if not GPU_VRAM_GB:
        # Nobody said what the card is, which is every Modal deployment. The
        # catalogue's own row is the answer, exactly as it was before tiers.
        return base.name if base.is_file() else MODEL_CATALOGUE[here[0]]["dest"].name
    fits = [k for k in here
            if MODEL_CATALOGUE[k].get("fits_vram_gb", 0) <= GPU_VRAM_GB]
    # Nothing claims to fit: take the smallest that is here and let ComfyUI
    # stream it. Refusing would be this table overruling a card it has never
    # seen, and streaming slowly is a real answer where no render is not.
    pick = fits[0] if fits else here[-1]
    return MODEL_CATALOGUE[pick]["dest"].name

# JoyCaption, not Qwen3-VL, and still not a booru tagger.
#
# The original argument here was symmetry: Krea 2 reads its prompt through
# Qwen3-VL, so captioning with the same family should close the gap between how
# the dataset describes an image and how you describe the one you want. It is a
# good argument and it lost to the captions, which were not merely weak — they
# were wrong about the picture. A medium close-up of two people posing for a
# graduation photo came back as one person with her arm on a red bench. There is
# no bench. That is confabulation, not style, and it is the failure mode that
# makes a captioner worthless here: a caption nobody re-reads becomes a training
# target, and the LoRA learns the bench.
#
# **It was seeing the image.** That is the explanation the bench invites and it
# is the wrong one, checked rather than assumed: the template emits
# `<|vision_start|><|image_pad|><|vision_end|>` exactly once for the parts list
# we send, and the model had plainly found the scene — one subject with an arm
# resting on something, outdoors. It got the *binding* wrong. The arm was on her
# friend's shoulder, and the friend became a red bench.
#
# So this is attribute bleed, one level up: not "which garment is the red one"
# but "what is the thing she is touching", and it is the same failure a
# caption's adjective-to-noun binding has to hold against. A model that loses
# a binding does not signal it — it writes a confident sentence about furniture
# that is not there, and that sentence becomes a training target. No rewording
# of the instruction moved it, which is the tell that the instruction was never
# the variable.
#
# JoyCaption is trained *for this job*, on this instruction surface, and its
# default prompt beats our best-tuned one.
#
# So the presets were JoyCaption's own sentences for a while, lifted from its
# Space rather than written here, on the argument that a prompt for a model
# fine-tuned on a fixed set of instructions is not a place to be creative. Then
# a form-shaped instruction of the owner's — one labelled field per line, the
# reply parsed back into exactly those fields — read better than any of them,
# and it is now the only preset (see `CHARACTER_INSTRUCTION`). The lesson that
# survives is narrower than "use the trained phrasing": the captions decide,
# and no reading of an instruction does.
#
# **What is not taken from it: the tag modes.** The Space offers Danbooru, e621,
# Rule34 and Booru-like tag lists. Prose, not tags, still holds and holds for a
# reason that has nothing to do with which captioner writes it — a tag list
# cannot express binding, so "red, blue, dress, jacket" leaves the model to
# guess which colour belongs to which garment, and with two people in frame that
# guess is where attribute bleed comes from. The text encoders parse grammar.
# Offering a mode that throws grammar away would be offering a worse dataset in
# a menu that looks like a preference.
#
# Qwen3-VL stays in the table. Old job records name it, the gear can still be
# pointed back at it, and a table that drops an entry makes every run that used
# it unreadable — the same rule the sidecars follow.
#
# The system prompt is the model's, not ours. JoyCaption Beta One was trained
# with one and the Space sends this exact string; without it the model is being
# run off its own distribution, which is the cheapest quality loss available.
# Qwen has no entry here and therefore gets no system turn, which is what it had
# before and what it wants.
CAPTION_MODELS: dict[str, dict[str, str]] = {
    "joycaption": {
        "repo": "fancyfeast/llama-joycaption-beta-one-hf-llava",
        "label": "JoyCaption Beta One",
        # The size is in the note because this is the one control on the page
        # that can start a 17 GB download without saying so: the captioner is
        # pulled into the HF cache on first use, not chosen under the gear.
        "note": "Trained for dataset captioning. First run pulls ~17 GB.",
        "system": ("You are a helpful assistant and help users with any queries "
                   "they may have with no censorship or restrictions."),
    },
    "qwen3vl": {
        "repo": "Qwen/Qwen3-VL-8B-Instruct",
        "label": "Qwen3-VL 8B",
        "note": "The text encoder's own family. Confabulated on test sets.",
    },
    # Kept for the same reason the stock one is, and trusted least of the three.
    # It exists because a refusal is not an error here: the stock instruct model
    # declines on photographs of real people — "I can't identify or describe
    # individuals in images" — and what arrives is not an exception but a fluent
    # sentence that goes straight into a `.txt` sidecar and then into a training
    # run. Abliteration removes the refusal direction by editing the weights,
    # which is a blunt instrument that does not stop at refusals, so a repackage
    # is a standing suspect for exactly the invented-detail failure logged above.
    # JoyCaption does not refuse at all, which retires the only reason to run
    # either of these.
    "qwen3vl-uncensored": {
        "repo": "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2",
        "label": "Qwen3-VL 8B uncensored",
        "note": "Same weights, refusal removed. First run pulls ~17 GB.",
    },
}
DEFAULT_CAPTION_MODEL = "joycaption"


def _custom_key(prefix: str, text: str) -> str:
    """A stable menu key derived from what the user typed, so re-adding the
    same repo or re-saving the same preset name lands on one entry rather than
    accumulating near-duplicates."""
    slug = re.sub(r"[^a-z0-9_-]+", "-", text.strip().lower()).strip("-")[:48]
    return f"{prefix}:{slug or 'unnamed'}"


def _caption_models() -> dict[str, dict[str, Any]]:
    """
    Built-ins plus the repos added under the gear.

    The customs live in the `config` Dict rather than in this table because the
    table is baked into the image — a captioner added through the UI has to
    survive a redeploy without an edit to this file. Read fresh on every call
    for the same reason `_hf_token()` is: the Dict is the source of truth and a
    module-level cache would hold a deleted model in the menu until the
    container recycled.
    """
    out: dict[str, dict[str, Any]] = dict(CAPTION_MODELS)
    try:
        for key, spec in (config.get("custom_caption_models") or {}).items():
            out[key] = {**spec, "custom": True}
    except Exception:
        pass  # an unreachable Dict costs the customs, never the built-ins
    return out


def _caption_presets() -> dict[str, dict[str, Any]]:
    """Built-ins plus presets saved from the captioner row. Same shape and the
    same reasoning as `_caption_models()`."""
    out: dict[str, dict[str, Any]] = dict(CAPTION_PRESETS)
    try:
        for key, spec in (config.get("custom_caption_presets") or {}).items():
            out[key] = {**spec, "custom": True}
    except Exception:
        pass
    return out


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif"}
# A dataset holds clips too. Nothing trains on them — there is no video trainer
# here and no checkpoint to be one — but a file the volume already accepts
# should not be invisible
# to the page that lists what is in the folder. Counting them and telling them
# apart is the whole of it: the sidecar layout is identical, `{clip}.txt` beside
# `{clip}.mp4`, so nothing about the storage contract changes when the trainer
# arrives.
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# 507 tokens is the real budget; this is that in characters, with margin.
#
# Krea 2's conditioner tokenizes with `truncation=True` at
# `TextEncoderConfig.max_length + 34 - 5` = 541, of which the fixed 34-token
# prefix template ("Describe the image by detailing the color, shape...") is not
# yours — so a caption gets 507. Nothing here passes a flag that moves it; see
# `krea2_encoder.py` in musubi if it ever does.
#
# The old value was 1024 and was a founding constant with no argument behind it:
# ~217 tokens, 43% of what the encoder reads, silently thrown away. It never
# showed because Qwen wrote short. It started costing whole sentences the day
# JoyCaption and a `long` default arrived, which is the way an arbitrary
# threshold usually surfaces — not when it is set, but when something else
# changes underneath it.
#
# 2048 rather than the ~2400 that 507 tokens buys at the 4.73 chars/token
# measured on real caption prose. Two reasons for the margin, both about the
# caption growing after this cap is applied: `append` mode concatenates onto an
# existing sidecar, and denser text (quoted sign text, numbers, heavy
# punctuation) runs nearer 4.2 chars/token — where 2048 is 488 tokens and still
# inside the budget.
MAX_CAPTION_CHARS = 2048
THUMB_PX = 320

# One preset, and it is the owner's own text, verbatim.
#
# It used to be five, each composed out of JoyCaption's trained sentences with a
# leave-out clause of ours appended in the Space's grammar. Retired 2026-09-02
# for this one form-shaped instruction (the others live in git, to come back
# one at a time as they are re-earned): a field per line, the subject named by
# `{NAME}`, and every immutable — face, skin, eyes, age, body, ethnicity, hair
# colour and length — kept out of every field so the trigger word is the only
# place identity can live. The fields are the model's scratchpad; the last
# paragraph has it write the caption *from* them, and `_caption_extract` keeps
# only what follows that final CAPTION:, so the scratchpad never reaches a
# sidecar.
#
# Not reworded here, and not to be. The text is the product: it was tuned by
# reading what came back, and "improving" a sentence of it without re-reading
# the captions is the same mistake as rewriting a function signature to sound
# better. What composes around it on the server is exactly `{NAME}` and the
# CAPTION: cut, and the line under the box says so (see `_caption_instruction`).
CHARACTER_INSTRUCTION = """\
Describe this image for a text-to-image training caption. The subject is {NAME}. Output only the fields below, one per line, no extra text.

SHOT: one of extreme close-up / close-up / medium close-up / medium shot / cowboy shot / medium wide shot / wide shot / extreme wide shot
POSE: body position and what {NAME} is doing, in one sentence
EXPRESSION: facial expression and gaze direction, in a few words
HAIR ARRANGEMENT: only how the hair is worn (loose, tied back, braided, under a hat, wet, windblown). Never state color, length, or texture.
CLOTHING: every visible garment and accessory, with colors and materials
SETTING: location and background objects, with spatial relationships
LIGHTING: source, direction, quality, color temperature
CAMERA: angle, lens feel, depth of field
FLAWS: none (unless the image clearly contains a watermark, text overlay, visible motion blur, on-camera flash, or obvious compression artifacts — then name only what is clearly present)

Rules: Do not describe {NAME}'s face, skin, eyes, age, body type, ethnicity, or hair color/length anywhere in any field. Do not invent details you cannot see. Attach every adjective to the noun it belongs to, so it is unambiguous which garment or object each color and material describes.

Then, using only what you wrote above, write CAPTION: one or two natural sentences describing the image. Start with "{NAME}" and the shot type. Do not add anything not in the fields."""

# They are here rather than in the page for the reason `SHOT_VOCAB` is: the page
# should send `character`, not four hundred words of instruction it could edit
# into something the run cannot reproduce. What the page shows is the label and
# the note; the text prefills the box and the first keystroke makes it the run's.
CAPTION_PRESETS: dict[str, dict[str, str]] = {
    "character": {
        "label": "Character",
        "note": "One field per line, nothing about the face. Identity is the trigger's job.",
        "instruction": CHARACTER_INSTRUCTION,
    },
}
DEFAULT_CAPTION_PRESET = "character"

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


def _strip_leading_trigger(text: str, trigger_word: str) -> str:
    """
    Drop the trigger word from the front of a caption, however many times it
    is there.

    The instruction tells the model the trigger is prepended automatically and
    it complies most of the time — but a caption that opens with the trigger
    anyway used to get the prefix stacked on top, and a recaption over an
    already-doubled sidecar tripled it: "chgl, chgl, chgl, …". The loop is what
    heals those, one layer per run touched.

    Bounded at a word edge for the reason `prepend_trigger` uses `startswith`:
    a "cat" trigger must not eat the front of "category". Case-insensitive
    because the model recapitalises the token when it opens a sentence — which
    is exactly the case the exact-match checks kept missing.
    """
    t = trigger_word.strip()
    while t:
        head = text.lstrip()
        if not head.lower().startswith(t.lower()):
            return head if head != text else text
        rest = head[len(t):]
        if rest and (rest[0].isalnum() or rest[0] == "_"):
            return head  # a longer word that merely begins with the trigger
        text = rest.lstrip(" ,.:;-").lstrip()
    return text


def _caption_instruction(
    preset: str, trigger_word: str, instruction: str = "",
) -> str:
    """
    The prompt, and nothing the page does not show.

    `instruction` overrides the preset's body when the page sends one — the
    preset is a starting point the textarea prefills, and the first keystroke
    forks it into the user's own. The one thing done to it is `{NAME}` → the
    trigger word ("the subject" without one).

    **Nothing is appended.** Two things used to be, and both were invisible
    from the page. A sixty-word rulebook rode behind every preset until
    2026-08-29 — no lists, no markdown, no editorialising — and then a trigger
    clause ("prefixed automatically, so never write it yourself") until
    2026-09-02. The owner learned of the rulebook after it was gone, and the
    rule that came out of that is the one in CLAUDE.md: what the model sees is
    shown. A preset that wants either sentence carries it in its own text,
    where it can be read. The house prompt's reply is cut to its final
    CAPTION: by `_caption_extract`, and the one line under the box says so.

    `replace` rather than `format`, because this string is editable in the page.
    `.format()` on user-typed text raises on a stray brace, and the cost of that
    would be a form that rejects a caption instruction for containing "{".
    """
    presets = _caption_presets()
    spec = presets.get(preset) or presets[DEFAULT_CAPTION_PRESET]
    out = instruction.strip() or spec["instruction"]
    return out.replace("{NAME}", trigger_word or "the subject")


# The last thing the house prompt asks for, and the only thing kept.
CAPTION_MARK_RE = re.compile(r"CAPTION\s*:", re.I)


def _caption_extract(reply: str) -> str:
    """
    What follows the final `CAPTION:` in a reply, or the whole reply.

    The house prompt makes the model fill nine labelled fields and then write
    the caption from them. The fields are the scratchpad — they are what make
    the caption good, and they are not the caption — so the sidecar is the
    text after the last CAPTION: mark. The *last*, because a model that drafts
    twice writes the mark twice, and the final one is the one it meant.

    Written here rather than asked for in the prompt: "output only the
    caption" would cost the scratchpad, which is the whole method, and a rule
    stacked on a probabilistic writer is a coin flip anyway. This was the
    owner's original one-liner, `reply.split("CAPTION:")[-1]`, made
    case-blind and bounded.

    A reply with no mark comes back whole and is logged rather than emptied:
    whatever it is, the raw text says more about it than a blank sidecar.
    """
    marks = list(CAPTION_MARK_RE.finditer(reply))
    if not marks:
        print(f"[caption] no CAPTION: in reply, kept whole: {reply[:120]!r}")
        return reply.strip()
    return reply[marks[-1].end():].strip().strip("*").strip()


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


# **The stop flag lives in its own key, and that is not tidiness.** It used to
# be a field on the job record, which every `_publish` rewrites — and `_publish`
# is get-update-put against a *network* Dict under a **process-local** lock. The
# Stop route runs in the web container, so it never took that lock: it read the
# record, set `stop`, and wrote it back, while the GPU container's next publish
# was already holding a copy read before the press and put it back with
# `stop: False`. On the generate path a publish lands on every tqdm line, so the
# window was open several times a second and Stop mostly did nothing — the
# symptom being a run somebody eventually killed from the Modal dashboard.
#
# A key nobody merges cannot be clobbered by a merge. The record's own `stop`
# field is still read as a fallback, because a job already in flight when this
# deploys has one and nothing else would answer it.
def _attachment_weight(params: dict[str, Any]) -> tuple[int, float]:
    """
    How many pictures came with this request, and how many megabytes of base64.

    Counted rather than inferred: the fields differ per family and per mode, and
    a number the log is going to be read against has to be the real one.
    """
    blobs: list[str] = []
    for key in ("scene", "outfit", "first_frame", "last_frame"):
        if isinstance(params.get(key), str) and params[key]:
            blobs.append(params[key])
    for key in ("references", "ref_videos", "ref_audios"):
        for b in params.get(key) or []:
            if isinstance(b, str) and b:
                blobs.append(b)
    for r in params.get("regions") or []:
        if isinstance(r, dict) and isinstance(r.get("ref_image"), str) and r["ref_image"]:
            blobs.append(r["ref_image"])
    return len(blobs), sum(len(b) for b in blobs) / 1_000_000


def _note_queue_wait(tag: str, job_id: str, params: dict[str, Any]) -> float:
    """
    How long this job waited between being spawned and being given to a
    container, and a line when that is not instant.

    **The gap nothing could see.** The GPU classes are `max_containers=1` with
    `@modal.concurrent(max_inputs=1)`, so anything already holding that slot
    delays everything behind it. `/api/motion` used to hold it, on the same
    container renders run on, and a ten-minute wait came out of that — the trade
    was defended as "a suggestion queuing behind a render is fine on a
    single-user platform", which is true and is the wrong way round. Nothing
    shares that slot now.

    It is kept because the window it measures did not go with the thing that
    filled it: a cold start, a blob transfer and Modal's own scheduling all land
    here, and the next occupant of this gap will be something nobody predicted.
    Measured
    rather than inferred, because this window also holds cold starts, blob
    transfer and scheduling, and a line that named one of them would be a guess
    wearing a number.
    """
    t = params.get("queued_at")
    if not isinstance(t, (int, float)) or not t:
        return 0.0
    waited = max(0.0, time.time() - float(t))
    if waited >= QUEUE_WAIT_SLOW_S:
        print(f"[{tag}] {job_id} waited {waited:.1f}s to be delivered — the "
              f"container was busy, cold, or taking a large body", flush=True)
    return waited


def _log_spawn(kind: str, job_id: str, payload: dict[str, Any],
               t0: float) -> None:
    """
    Close the last dark segment: browser -> web container -> Modal.

    **The logs were as silent as the page**, which is worse, because the logs
    are the escape hatch — somebody who cannot tell what the app is doing opens
    the dashboard, and what was there was ComfyUI's own output either side of an
    eight-minute hole with nothing of ours in it at all. The GPU container's
    lines cover it from `accepted` onward; this covers what happens *before*
    that, which is where a 48 MB body actually travels: parsed here, then handed
    to Modal, which puts anything past a couple of megabytes into blob storage
    before the GPU container can be given it.

    The gap between this line and the container's `accepted` is that hop, and it
    could not be seen at all.
    """
    n, mb = _attachment_weight(payload)
    print(f"[api] {job_id} spawned in {time.time() - t0:.1f}s"
          + (f", {n} attachment{'' if n == 1 else 's'} / {mb:.1f} MB" if n else ""),
          flush=True)


def _pretty_model(name: str) -> str:
    """ComfyUI's class name as something worth showing somebody."""
    for raw, shown in (("MiniMaxH3VideoVAE", "the video VAE"),
                       ("MiniMaxH3TEModel_", "the text encoder"),
                       ("MiniMaxH3", "MiniMax-H3"),
                       ("AutoencodingEngine", "the VAE"),
                       ("Krea2", "Krea 2")):
        if name.startswith(raw):
            return shown
    return name


def _stop_key(job_id: str) -> str:
    return f"stop:{job_id}"


def _request_stop(job_id: str) -> None:
    jobs[_stop_key(job_id)] = True


def _clear_stop(job_id: str) -> None:
    """At the top of a run, because the key outlives the job that set it."""
    try:
        jobs.pop(_stop_key(job_id))
    except Exception:
        pass


def _stop_requested(job_id: str) -> bool:
    try:
        if jobs.get(_stop_key(job_id)):
            return True
        return bool((jobs.get(job_id) or {}).get("stop"))
    except Exception:
        return False


def _stop_gate(job_id: str, where: str) -> None:
    """
    Raise if Stop was pressed, at a point where nothing is holding a GPU.

    **The generate path used to read the flag in exactly one place** — inside
    `_await`, which is reached only once the graph has been posted. Everything
    before it, which on a cold container is where the minutes are, ran to
    completion whatever the person did: the button set a bit nobody looked at,
    the page said "Stopping…", and the render carried on to a picture nobody
    was waiting for any more. Somebody watching that goes to Modal and kills
    the app, which is what happened and how this was found.

    Cheap enough to call between phases: one read of a Dict this path already
    writes to several times.
    """
    if _stop_requested(job_id):
        print(f"[{job_id}] stop requested during {where}", flush=True)
        raise StopRequested(where)


def _hf_token() -> str | None:
    """The token pasted into the UI, stored in a Modal Dict. No Secrets needed."""
    return _pasted_key("hf_token")


def _pasted_key(name: str) -> str | None:
    """
    A credential typed into the gear and kept in a Modal Dict.

    One of these now. There were two for a while — the parse took a hosted
    model and an API key to reach it — and choosing local weights is what paid
    for the semantic layer's "no new controls": a hosted interpreter needed a
    key field in Settings, and weights need nothing, because the gear renders a
    catalogue family for free. `modal deploy app.py` stays the entire install.
    """
    try:
        return (config.get(name) or "").strip() or None
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
    if TRAIN_BIN:
        # In front, not appended: an `accelerate` on the system PATH would
        # otherwise win and launch the wrong interpreter's torch, which fails
        # as a CUDA error about the wrong architecture rather than as a
        # wrong-environment error.
        env["PATH"] = f"{TRAIN_BIN}{os.pathsep}{env.get('PATH', '')}"
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
                # tqdm's left-hand clock, which was captured by the regex and
                # then thrown away. Elapsed beside ETA is the pair the terminal
                # shows and the pair that answers "is this worth waiting for" —
                # an ETA alone cannot say whether it has been three minutes or
                # three hours, and a card whose run started before you opened
                # the window has no other way to know.
                fields["elapsed"] = m.group("el")
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
    generation after the first.

    **What absorbing it does not buy is a job that works.** This used to say the
    web container had already validated everything the job was about to open, so
    the reload here was the second look and not the only one. That is a
    confusion of two different questions: the web container proves the file is
    *on the volume*, and the GPU container needs it in *its own mount* to open
    it. A LoRA trained while this container was warm passed the first and failed
    the second, and the page said "No such LoRA" about an epoch the user could
    see in the picker. Freshness a container cannot get by reloading, it gets by
    asking Modal — see `_lora_visible`.

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

# How long an insisting reload waits out an open-files refusal, in seconds
# between attempts. Three tries, half a second of waiting at the very worst.
# Sized against what actually holds the descriptor: a still off a warm volume is
# tens of milliseconds, so the first pause covers the ordinary case, and a clip
# streaming to a slow connection is seconds — far past anything worth sleeping
# for inside a request handler. That one is what the False is for.
RELOAD_INSIST_BACKOFF = (0.15, 0.35)

# Above this, a reload is worth a line of its own. Under it there would be one
# per gallery miss and the log would be reloads.
RELOAD_SLOW_S = 2.0

# Above this, a request is worth a line. Under it the log would be `/api/status`
# at 400ms, which is a log made of polls and no more use than no log at all.
REQUEST_SLOW_S = 2.0

# Above this, a job that waited to reach a container is worth a line. Anything
# under is scheduling noise.
QUEUE_WAIT_SLOW_S = 5.0


def _reload_one(vol: Any, label: str) -> bool:
    """
    Bring one volume forward. False if it was refused for open files.

    Split from its sibling because the two refusals are independent facts and
    one `try` made them a single one: the data volume was reloaded first, so
    the moment anything on it was mapped the models reload was never attempted
    at all. A container that had loaded one LoRA therefore stopped seeing newly
    downloaded *weights* too, for a reason that had nothing to do with weights.

    The label is in the log line because "reload skipped" said which failure
    mode without ever saying which volume, and the two send you to different
    places: /workspace is LoRAs and results, /models is the checkpoint.
    """
    try:
        # **Timed, because this is the one step in a request with no bound on
        # it.** Everything else between accepting a job and queueing its graph
        # is arithmetic or a stat; this is a network call whose cost scales with
        # what is on the volume, and a volume grows with every render. It was
        # the prime suspect for an eight-minute gap and could not be confirmed
        # or cleared, which is the whole argument for the line.
        t0 = time.time()
        vol.reload()
        took = time.time() - t0
        if took > RELOAD_SLOW_S:
            print(f"[volume] {label} reload took {took:.1f}s", flush=True)
        return True
    except RuntimeError as exc:
        if "open files" not in str(exc):
            raise
        # Not silent: a stale view is a real cause of "that LoRA is not there",
        # and this line is the only thing that distinguishes it from a typo.
        print(f"[volume] {label} reload skipped, weights still mapped ({exc})",
              flush=True)
        return False


def _reload_volume_locked() -> bool:
    """
    Both volumes, one freshness step — and the answer is about the data one.

    Every caller of this asks about /workspace: a LoRA listing, a result
    folder, a dataset. /models is reloaded on the same trip because the gear
    downloads into it and a warm container should see that, but a refusal there
    is not the answer to anybody's question, and returning it as one made
    `_reload_insist` sleep half a second waiting out a mapped checkpoint it can
    never unmap.

    Data first, because that is the one whose staleness costs a render: a stale
    weight listing costs a gear-menu refresh, a stale LoRA listing costs "that
    LoRA is not there" in the middle of one.
    """
    ok = _reload_one(volume, "data")
    _reload_one(models_volume, "models")
    return ok


def _reload_insist() -> bool:
    """
    Reload, and try again briefly if it was refused. Returns whether it landed.

    For the routes whose answer is wrong rather than merely old without one:
    the two that delete out of the gallery, and — for a narrower reason than
    it used to be — listing it. All are asked about a run that finished moments
    ago, which is the one case a stale view cannot describe.

    Deleting is unchanged: `_listed` and `rmtree` read and mutate the mount, so
    a view too old to hold the folder refuses to delete work that is sitting
    right there.

    Listing does not call this any more, and neither does serving. The whole
    gallery read path — entry set, sidecars, covers, files — reads committed
    state by RPC and serves off the container's spool (see the block above
    `_entries_by_rpc`), so there is no mount left in it to be behind, and no
    descriptor of ours left on /workspace to refuse anyone's reload. What
    remains here are the paths that mutate the mount and are asked about
    something written seconds ago — deleting a result that just landed.

    Everywhere else keeps the single attempt. A reload is a freshness step, and
    for a route that is not being asked about something written seconds ago,
    a stale view costs a listing that catches up on the next request; sleeping
    in the handler would spend one of twenty slots to buy that.
    """
    ok = _reload_volume()
    for pause in RELOAD_INSIST_BACKOFF:
        if ok:
            break
        time.sleep(pause)
        ok = _reload_volume()
    return ok


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
    negative was ever cached for them, while the video weights were downloaded
    into a deployment that had already been asked whether they were there. The symptom
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


def _listed(path: Path, root: Path) -> bool:
    """
    Is `path` visible, asking each directory below `root` what it holds?

    `_sizes_on_disk` above is the same lesson at one level; this is it at
    several, and the extra levels are not theoretical. `/api/file` looks up
    `outputs/{job}/{name}` for a run that finished a moment ago, so the name
    that misses is the *directory* — `{job}` did not exist when this container
    last synced. A stat caches that miss below us, and `volume.reload()` brings
    the volume forward without invalidating a name already asked about, so the
    retry after the reload can answer "not there" about a file that is there.
    That is the 404-on-a-fresh-render whose symptom is a broken picture on the
    canvas with the run sitting on the volume.

    Walking down by readdir never consults a cached name: each level reports
    what it holds now. It is also how `_gallery()` has always listed, which is
    why the gallery could show a card whose image would not load.

    `root` is the confinement, not a convenience: it is the last directory
    taken on trust, so nothing above it is walked and a caller cannot use this
    to probe outside the tree it names.
    """
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    # Keyed by root as well as by path, because two roots now share this memo —
    # outputs/ and loras/ — and a bare relative path is not unique across them.
    # Nothing collides today, but the cost of one that did is a delete route
    # taking `_listed` at its word about a file under the *other* root.
    key = f"{root}|{rel}"
    if key in _LISTED_OK:
        return True
    here = root
    for part in rel.parts:
        try:
            if not any(e.name == part for e in os.scandir(here)):
                return False
        except OSError:
            return False
        here = here / part
    _LISTED_OK.add(key)
    return True


# Paths this container has already proven present, so a readdir is paid once.
#
# The walk above is O(entries in the directory) per level, and `outputs/` holds
# one directory per result — so on a volume with hundreds of them, confirming a
# job folder is a scan of the whole listing. A batch of four stills asks about
# the same folder four times, and the canvas asks the moment a run lands, which
# is exactly when a person is waiting.
#
# Only *positive* answers are kept. A miss is the case this function exists to
# re-ask, since the file it is looking for is one the GPU container may be
# writing right now; caching that would reintroduce the negative-dentry fault
# one layer up. A path that exists stops existing only through the three delete
# routes — two for results, one for a LoRA — and all three discard from here.
_LISTED_OK: set[str] = set()


def _forget_listed(rel: str, root: Path) -> None:
    """Drop a deleted path, and anything under it, from the positive cache.

    `root` is named rather than defaulted because the memo is keyed by it — see
    `_listed` — and a default would be a second place that decides which tree a
    caller meant. It would also be a module constant evaluated at def time,
    which `tools/_from_app.py` executes a *subset* of this file without."""
    pre = f"{root}|{rel}"
    for key in [k for k in _LISTED_OK if k == pre or k.startswith(f"{pre}/")]:
        _LISTED_OK.discard(key)


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
                # A tier is an alternative for another row's slot, and says
                # which card it is meant for. Both are absent on the ordinary
                # rows, which is every row on a deployment with no tiers pulled.
                "tier_of": spec.get("tier_of"),
                "fits_vram_gb": spec.get("fits_vram_gb"),
                # On the row that owns a slot: the alternatives, and which file
                # the graph will actually ask for. Derived here rather than in
                # the page, for the reason every other menu is — a second
                # implementation of the choice is a second answer to it.
                "tiers": SLOT_TIERS.get(key) or None,
                "using": _slot_name(key) if SLOT_TIERS.get(key) else None,
                "pinned": (config.get("weight_tiers") or {}).get(key),
            }
        )
    return out


# --------------------------------------------------------------------------
# Downloads — CPU only.
#
# Deliberately not on the GPU function: pulling 26 GB while an A100 idles is
# money burned for nothing. This runs on plain CPU at a fraction of the cost.
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Exporting a scene
#
# **A scene is several takes, and until this existed it stayed several takes.**
# H3 renders about 14.4 seconds at a time, so anything with more than one beat
# in it is several runs — and the platform's own claim is that what makes them
# one scene is that the cast, the look and the last frame carry across. That is
# true of the *making* and was false of the result: the takes sat in the gallery
# as unrelated clips, and the owner's reading is the one that matters — without
# this "it's just a gimmick".
#
# On CPU, for the reason the downloads are: never rent a GPU to do CPU work.
# --------------------------------------------------------------------------

EXPORT_JOB = "export_scene"


def _probe(path: Path) -> dict[str, Any]:
    """
    One take's shape, as ffprobe reports it.

    Read rather than assumed, because the fast path below turns on the answer:
    takes from one scene *usually* agree on codec, size and rate, and "usually"
    is not something to concatenate on. A scene whose middle take was rendered
    after somebody changed the size is the case that produces a file where the
    picture goes wrong halfway through and nothing said so.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"{path.name}: {out.stderr.strip()[:200] or 'unreadable'}")
    meta = json.loads(out.stdout or "{}")
    streams = meta.get("streams") or []
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not v:
        raise RuntimeError(f"{path.name} has no video track")
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    # The container's duration, then the video stream's. Needed only to make
    # silence the right length for a take that has no sound — see the graph —
    # and it is deliberately **not** part of the equality test below: two takes
    # of different lengths concatenate perfectly well, and treating a duration
    # as a shape mismatch would re-encode every scene.
    try:
        dur = float(meta.get("format", {}).get("duration")
                    or v.get("duration") or 0)
    except (TypeError, ValueError):
        dur = 0.0
    return {
        "w": int(v.get("width") or 0), "h": int(v.get("height") or 0),
        "vcodec": str(v.get("codec_name") or ""), "rate": str(v.get("r_frame_rate") or ""),
        "pix": str(v.get("pix_fmt") or ""),
        "acodec": str(a.get("codec_name") or "") if a else "",
        "arate": str(a.get("sample_rate") or "") if a else "",
        "alayout": str(a.get("channel_layout") or "") if a else "",
        "audio": a is not None,
        "dur": dur,
    }


@app.function(image=export_image, cpu=2.0, timeout=60 * 60,
              volumes=MODAL_VOLUMES_WS)
def export_scene_job(job_id: str, takes: list[dict]) -> dict[str, Any]:
    """
    Stitch a scene's takes into one file, in the order they were made.

    **Stream copy when the takes agree, re-encode only when they do not.** A
    copy is seconds and lossless; a re-encode is minutes and generational loss,
    and it is the wrong default for the case that is overwhelmingly common —
    every take out of one scene comes from one pipeline at one size. So the
    shapes are probed and the answer decides, and the phase says which happened
    rather than leaving a person to wonder why one export took forty seconds and
    the next took four.

    Audio is normalised rather than dropped. H3 renders sound, and a take with a
    silent stretch is still a take with an audio track — but a scene where one
    clip has none would make the concat filter fail on a stream that is not
    there, so silence is generated for it and the sound of the others survives.
    """
    _publish(job_id, status="running", percent=0, phase="Reading the takes")
    _reload_volume()
    out = OUTPUTS / f"{job_id}.mp4"
    try:
        srcs: list[Path] = []
        for i, t in enumerate(takes):
            name = Path(str(t.get("file") or "")).name
            f = OUTPUTS / name
            if not f.is_file():
                # Named, with the take number, because the fix is on the page:
                # a take deleted from the gallery is still in the scene's chain
                # until somebody takes it out of the chain too. Raised rather
                # than returned so it leaves by the one path that writes the
                # record — an early `return` here published nothing and the page
                # polled a job that would never answer, which is the exact
                # failure `_publish`'s own docstring records.
                raise RuntimeError(
                    f"Take {i + 1} is not on the volume any more ({name}). "
                    "Remove it from the scene and export again.")
            srcs.append(f)
        if not srcs:
            raise RuntimeError(
                "Nothing to export — the scene has no finished takes.")

        shapes = []
        for i, f in enumerate(srcs):
            _stop_gate(job_id, "probing")
            _publish(job_id, percent=int(5 * (i + 1) / len(srcs)),
                     phase=f"Reading take {i + 1} of {len(srcs)}")
            shapes.append(_probe(f))
        # Every field but the duration: a codec or a frame rate that differs is
        # exactly as unconcatenatable as a width, and it is the one you do not
        # see until the file is open. Length is the exception and has to be —
        # takes are different lengths by construction, and comparing whole dicts
        # would send every scene down the re-encode.
        shape = lambda s: {k: v for k, v in s.items() if k != "dur"}  # noqa: E731
        same = all(shape(s) == shape(shapes[0]) for s in shapes)

        OUTPUTS.mkdir(parents=True, exist_ok=True)
        # The record rides the scene's own metadata like every other clip;
        # `use_metadata_tags` is what lets ffmpeg carry a key of ours.
        record = _output_record(
            kind="video", job_id=job_id, model="export", prompt="",
            created=time.time(), takes=[s.name for s in srcs])
        tagged = ["-movflags", "use_metadata_tags+faststart",
                  "-metadata", f"{RECORD_KEY}={_record_json(record)}"]
        work = Path(tempfile.mkdtemp(prefix="export-"))
        if same:
            _publish(job_id, percent=10,
                     phase=f"Joining {len(srcs)} takes — no re-encode")
            # Absolute paths with `-safe 0`, and each one single-quoted with
            # embedded quotes escaped the way the demuxer's own parser wants.
            listing = "\n".join(
                "file '%s'" % str(f).replace("'", "'\\''") for f in srcs)
            (work / "takes.txt").write_text(listing + "\n")
            cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                   "-i", str(work / "takes.txt"), "-c", "copy",
                   *tagged, str(out)]
        else:
            _publish(job_id, percent=10,
                     phase=f"Joining {len(srcs)} takes — re-encoding, they differ")
            w, h = shapes[0]["w"] or 1280, shapes[0]["h"] or 720
            cmd = ["ffmpeg", "-y"]
            for f in srcs:
                cmd += ["-i", str(f)]
            # Scaled and padded to the first take's frame rather than stretched:
            # a take at another aspect is a different shot, not a mistake to
            # squash. SAR is reset because a mismatched sample aspect is the one
            # thing `concat` refuses after everything else has been made equal.
            parts, legs = [], ""
            for i, sh in enumerate(shapes):
                parts.append(
                    f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
                    f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=24[v{i}]")
                if sh["audio"]:
                    # `aformat`, not `aresample`: `concat` refuses streams whose
                    # channel layout differs as readily as ones whose rate does,
                    # and a mono take beside a stereo one is the ordinary case
                    # once anything but H3 has produced one of them.
                    parts.append(
                        f"[{i}:a]aformat=sample_rates=48000:channel_layouts=stereo"
                        f"[a{i}]")
                else:
                    # Silence exactly as long as that take's picture, so the
                    # tracks stay in step instead of the sound running ahead of
                    # the frames for the rest of the scene. `anullsrc` runs
                    # forever without `d`, which is why the duration is probed.
                    parts.append(
                        "anullsrc=channel_layout=stereo:sample_rate=48000:"
                        f"d={sh['dur'] or 1:.3f}[a{i}]")
                legs += f"[v{i}][a{i}]"
            graph = ";".join(parts) + f";{legs}concat=n={len(srcs)}:v=1:a=1[v][a]"
            cmd += ["-filter_complex", graph, "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                    *tagged, str(out)]

        _stop_gate(job_id, "stitching")
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=55 * 60)
        if run.returncode != 0 or not out.is_file():
            # ffmpeg's last line is the one that says why; the rest is a banner
            # naming every codec it was built with.
            tail = (run.stderr or "").strip().splitlines()
            raise RuntimeError("ffmpeg could not join the takes: "
                               + (tail[-1] if tail else "no output"))
        _publish(job_id, percent=95, phase="Committing")
        volume.commit()
        # `completed`, which is the word every other job in this file finishes
        # on and the one `pollStatus` reads. The record is what the page sees —
        # the return value is only this function's own answer — so both are
        # written, the way `_download_weight` does it.
        res = {"status": "completed", "job_id": job_id, "files": [out.name],
               "kind": "video", "takes": len(srcs), "percent": 100,
               "bytes": out.stat().st_size, "reencoded": not same}
        _publish(job_id, **res)
        return res
    except StopRequested:
        out.unlink(missing_ok=True)
        res = {"status": "stopped", "job_id": job_id}
        _publish(job_id, **res)
        return res
    except Exception as exc:  # noqa: BLE001 - the record is the only report
        out.unlink(missing_ok=True)
        res = {"status": "failed", "job_id": job_id, "error": str(exc)[:400]}
        _publish(job_id, **res)
        return res



MIGRATE_JOB = "migrate_outputs"


@app.function(image=export_image, cpu=2.0, timeout=4 * 60 * 60,
              volumes=MODAL_VOLUMES_WS)
def migrate_outputs_job(job_id: str) -> dict[str, Any]:
    """
    Move every job folder under outputs/ into the flat layout, once.

    Each folder's `visionary.json` becomes the record inside each of its
    files — PNGs re-encoded with the chunk, clips remuxed with the key, which
    is why this runs on the container that has ffmpeg — and the files take
    the run's id as their name: `{job}_{NN}.png`, `{job}.mp4`,
    `{job}.context.safetensors`. The folder goes with its caches. Progress is
    published because a volume with a year of renders on it is minutes of
    work, and a gallery that says "moving older results into place" is a
    gallery that has not gone blank.

    Per folder, so a crash mid-way leaves whole folders and whole files and
    never half of either; re-running picks up where it stopped.
    """
    _publish(job_id, status="running", percent=0, phase="Finding older results")
    _reload_volume()
    dirs = sorted(p for p in OUTPUTS.iterdir()
                  if p.is_dir() and not p.name.startswith(".")) if OUTPUTS.is_dir() else []
    moved, failed = 0, []
    for i, d in enumerate(dirs):
        _publish(job_id, percent=int(100 * i / max(1, len(dirs))),
                 phase=f"Moving {i + 1} of {len(dirs)} into place")
        try:
            record: dict[str, Any] = {}
            meta = d / LEGACY_META
            if meta.is_file():
                try:
                    record = json.loads(meta.read_text())
                except (OSError, ValueError):
                    record = {}
            record.setdefault("job_id", d.name)
            media = sorted(p for p in d.iterdir()
                           if p.is_file() and _keep_entry(p.name))
            for n, p in enumerate(media):
                ext = p.suffix.lower()
                # A batch keeps its index; a lone clip or a scene export
                # takes the run's bare name, which is what the flat writers
                # produce today.
                m = re.search(r"_(\d{2})$", p.stem)
                idx = m.group(1) if m else (f"{n:02d}" if ext == ".png" else None)
                name = f"{d.name}_{idx}{ext}" if idx else f"{d.name}{ext}"
                target = OUTPUTS / name
                if ext == ".png":
                    _png_with_record(p, target, record)
                elif ext == ".mp4":
                    _mp4_with_record(p, target, record)
                else:
                    shutil.copyfile(p, target)
            ctx = d / "context.safetensors"
            if ctx.is_file():
                shutil.copyfile(ctx, OUTPUTS / f"{d.name}{H3MC_SUFFIX}")
            shutil.rmtree(d, ignore_errors=True)
            moved += 1
        except Exception as exc:  # noqa: BLE001 — one folder must not stop the rest
            failed.append(f"{d.name}: {type(exc).__name__}: {exc}")
            print(f"[migrate] {d.name}: {type(exc).__name__}: {exc}", flush=True)
        if (i + 1) % 10 == 0:
            volume.commit()
    volume.commit()
    res = {"status": "completed", "job_id": job_id, "percent": 100,
           "moved": moved, "files": [],
           **({"failed": failed[:20]} if failed else {})}
    _publish(job_id, **res)
    return res


@app.function(image=web_image, cpu=2.0, timeout=4 * 60 * 60, volumes=MODAL_VOLUMES)
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
    expect_gb: float | None = None, note: str | Callable[[], str] = "",
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
        # Resolved every poll, not once at the call. A queued weight download
        # knows its position before it starts and passes a string; a Drive
        # folder learns which file of how many it is on only while it is
        # running, and a note fixed at call time would report the first file's
        # position for the length of the transfer.
        where = note() if callable(note) else note
        fields: dict[str, Any] = {
            "phase": " · ".join(x for x in (label, where, size) if x),
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
    # The models volume, because staging and dest both live there now — the
    # data volume saw nothing from this download.
    models_volume.commit()

    res = {
        "status": "completed",
        "key": key,
        "size_gb": round(dest.stat().st_size / 1e9, 2),
        "duration_s": round(time.time() - started, 1),
    }
    _publish(job_id, **res)
    print(f"[download] {spec['label']}: {res['size_gb']} GB in {res['duration_s']}s")
    return res


@app.function(image=web_image, cpu=2.0, timeout=6 * 60 * 60, volumes=MODAL_VOLUMES)
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


def _land_weights(paths: list[Path], dest_dir: Path) -> list[str]:
    """Move whole files off the stage into loras/, flattened onto their names."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    landed = []
    for p in paths:
        target = dest_dir / p.name
        if target.exists():
            target.unlink()
        shutil.move(str(p), target)
        landed.append(target.name)
    return landed


def _drive_plan(url: str, stage: Path, present: set[str], refetch: bool) -> dict[str, Any]:
    """
    List a Drive folder, and decide what actually has to cross the wire.

    `skip_download` answers with the folder's contents — an id, a path, and the
    local path each file would land at — without moving a byte, which is the
    whole reason a diff is possible here. What it does not answer with is a size
    or a checksum, so the only thing a remote file and a local one can be
    compared on is their name. That is why re-fetching is a switch rather than a
    freshness test: a weight replaced in place on Drive under the name it
    already had is indistinguishable from the one already on the volume, and a
    test that cannot see the bytes claiming otherwise would be worse than saying
    so.

    The second saving is the preview grid and the readme — dropped from the plan
    rather than downloaded and then filtered off the stage, which is the order
    this ran in before and it paid for every byte of them first.
    """
    import gdown

    listed = gdown.download_folder(url, output=str(stage), quiet=True,
                                   use_cookies=False, skip_download=True)
    if listed is None:
        # A folder gdown cannot read is a None and a line on stderr nobody is
        # reading, so the two causes are named here rather than left to be
        # inferred from a stage that stayed empty.
        raise RuntimeError(
            "Drive would not list that folder — either it is not shared with "
            "'Anyone with the link', or it holds more than 50 files, which is "
            "as far as the folder API goes.")

    weights = [(f, Path(f.local_path).name) for f in listed
               if f.path.lower().endswith(".safetensors")]
    return {
        "todo": [f for f, n in weights if refetch or n not in present],
        "already": [] if refetch else [n for _, n in weights if n in present],
        "skipped": [Path(f.path).name for f in listed
                    if not f.path.lower().endswith(".safetensors")],
    }


@app.function(image=web_image, cpu=2.0, timeout=4 * 60 * 60, volumes=MODAL_VOLUMES)
def gdrive_job(url: str, folder: str, refetch: bool = False) -> dict[str, Any]:
    """
    Pull one file or one folder off Google Drive into loras/.

    Staged first, then moved. Downloading straight into loras/ would put a
    half-written .safetensors in front of the picker — and the picker globs the
    directory live, so it would be offered, chosen, and fail inside a warm GPU
    container with a torch deserialization error thirty seconds into a run.
    Staging keeps a partial download invisible until it is a whole file.

    A folder is listed before anything is fetched, and a name already in the
    destination is left alone unless `refetch` says otherwise: re-pasting the
    link after two more epochs were uploaded costs the two epochs, not the whole
    run again. The comparison is by name, which is all Drive will give —
    `_drive_plan` has why, and why `refetch` is a switch because of it.
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
    # Resolved up here, before a byte moves, because the pull is now a diff
    # against what is in it.
    #
    # No folder given means the top level of loras/, where the listing already
    # treats a bare file as its own entry named for itself. A folder given means
    # loras/{folder}/, where the listing treats the files as versions of one
    # LoRA — which is right for a matched pair and wrong for a bag of unrelated
    # ones, so it stays a choice rather than a default.
    dest_dir = (LORAS / folder) if folder else LORAS
    # Matched the way the landing does, `.suffix.lower()` rather than a
    # `*.safetensors` glob: a file that arrived spelled .SAFETENSORS lands under
    # that name, and a glob that cannot see it is a diff that fetches it again
    # every time.
    present = ({p.name for p in dest_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".safetensors"}
               if dest_dir.exists() else set())

    stage = WORK / f"gdrive-{int(time.time())}"
    if stage.exists():
        shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f"[gdrive] {url} -> loras/{folder or ''}"
          + (f" ({len(present)} already there)" if present else ""))

    result: dict[str, Any] = {}
    done = threading.Event()
    # Written by the pull thread; read by the watcher every five seconds and by
    # the stop path once. `got` is what lets a stop keep the files that are
    # whole — the stage cannot be trusted wholesale there, because one of the
    # files in it is the half of a transfer that was interrupted, which is the
    # exact thing staging exists to keep away from the picker.
    live: dict[str, Any] = {"note": "", "got": [], "plan": {}}

    def pull() -> None:
        try:
            if _is_drive_folder(url):
                plan = _drive_plan(url, stage, present, refetch)
                live["plan"] = plan
                todo = plan["todo"]
                for i, f in enumerate(todo, 1):
                    # Between files, not inside one: the same cooperative stop
                    # the weight queue takes, at the only point where dropping
                    # out costs nothing already paid for.
                    if _stop_requested(job_id):
                        break
                    out = Path(f.local_path)
                    live["note"] = f"{i} of {len(todo)} · {out.name}"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    if gdown.download(id=f.id, output=str(out), quiet=True,
                                      use_cookies=False) is not None:
                        live["got"].append(out)
            else:
                # fuzzy, so a pasted browser URL works as well as a bare id —
                # which is the form the link actually arrives in. No diff on this
                # path: a file link carries its name in the headers of the
                # transfer itself, so there is nothing to compare until the bytes
                # a diff would have saved are already moving.
                gdown.download(url, output=str(stage) + "/", quiet=True, fuzzy=True)
        except Exception as exc:
            result["error"] = exc
        finally:
            done.set()

    threading.Thread(target=pull, daemon=True).start()
    state = _watch_download(stage, "Google Drive", job_id, done,
                            note=lambda: str(live["note"]))
    plan: dict[str, Any] = live["plan"]

    # Asked again here, because a stop that lands while the thread is between
    # files ends the pull on its own and the watcher returns on `done` without
    # ever having seen it.
    if state.get("stopped") or _stop_requested(job_id):
        landed = _land_weights(live["got"], dest_dir)
        shutil.rmtree(stage, ignore_errors=True)
        if landed:
            volume.commit()
        # What is whole is kept, not thrown away: a stop four files into a
        # folder has done real work, and discarding it would mean the next run's
        # diff has nothing to skip and pays for those four again.
        rest = [Path(f.local_path).name for f in plan.get("todo") or []]
        res = {"status": "stopped", "downloaded": landed,
               "remaining": [n for n in rest if n not in set(landed)]}
        _publish(job_id, **res)
        return res

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
    # picker has to keep stepping over. A folder link now knows what it left
    # behind from the listing, since those files were never fetched to be seen
    # on the stage; a file link has no listing and still answers off what landed.
    found = sorted(p for p in stage.rglob("*") if p.is_file())
    weights = [p for p in found if p.suffix.lower() == ".safetensors"]
    skipped = plan.get("skipped") or [p.name for p in found if p not in weights]
    already: list[str] = plan.get("already") or []

    if not weights and not already:
        shutil.rmtree(stage, ignore_errors=True)
        err = ("No .safetensors in that download"
               + (f" — found {', '.join(skipped[:6])}." if skipped else "."))
        _publish(job_id, status="failed", error=err)
        return {"status": "failed", "error": err}

    landed = _land_weights(weights, dest_dir)
    shutil.rmtree(stage, ignore_errors=True)
    volume.commit()

    res: dict[str, Any] = {
        "status": "completed",
        "percent": 100,
        "files": landed,
        "already": already,
        "skipped": skipped,
        "folder": folder,
        "size_gb": round(sum((dest_dir / n).stat().st_size for n in landed) / 1e9, 2),
        "duration_s": round(time.time() - started, 1),
    }
    # A sentence only when there is one to write. "Downloaded." is the whole
    # truth about a run that fetched everything it listed; it is a lie about one
    # that fetched two of five, and the three that did not cross the wire are
    # the only evidence the diff did anything at all.
    if already:
        where = f"loras/{folder}/" if folder else "loras/"
        res["note"] = (f"{len(landed)} new · {len(already)} already in {where}"
                       if landed else
                       f"Nothing new — {len(already)} already in {where}")
    _publish(job_id, **res)
    print(f"[gdrive] {len(landed)} file(s), {res['size_gb']} GB in {res['duration_s']}s"
          + (f" · {len(already)} skipped as present" if already else ""))
    return res


# --------------------------------------------------------------------------
# Captioning
# --------------------------------------------------------------------------


def _trimmed(caption: str) -> str:
    """
    Cut an over-long caption back to its last full sentence.

    This is the boundary handler, not the budget — the budget is
    MAX_CAPTION_CHARS and the argument for its value is up there. What this
    exists for is the shape of the cut: a hard slice puts a half-word at the end
    of a file the trainer reads as a sentence. Backing up to the last terminator
    costs a few words and leaves prose.

    It fires rarely now that the cap is the encoder's actual limit rather than
    half of it, which is the right frequency for a fallback. It is kept because
    "rarely" is not "never", and the run where it does fire is a run nobody is
    watching.

    Falls back to the hard slice when there is no terminator in range, because a
    caption that is one 1100-character sentence is still better truncated than
    dropped, and the alternative is a rule that silently empties a sidecar.
    """
    if len(caption) <= MAX_CAPTION_CHARS:
        return caption
    cut = caption[:MAX_CAPTION_CHARS]
    stop = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[: stop + 1] if stop > MAX_CAPTION_CHARS // 2 else cut


def _caption_processor(repo: str, cache_dir: str) -> Any:
    """
    Load a captioner's processor, with the pixel budget if it takes one.

    `min_pixels`/`max_pixels` cap the vision tower's token budget: Qwen3-VL
    scales patches with input resolution, so a 4000px training image would
    otherwise spend thousands of tokens on detail that never reaches the
    caption — slow, and no better. They are Qwen vocabulary. JoyCaption is a
    LLaVA on siglip2, which resizes to a fixed grid and has no such knob.

    Passed and then retried without, rather than branched on the repo id,
    because the menu is open and there is no list of which families take them.
    Checked on transformers 5.16.1 against all three built-ins: Qwen applies
    them, LlavaProcessor swallows them, nobody raises — so the retry is
    insurance rather than a live path, and it costs one extra config read on
    the first captioner that ever does refuse. Left in because the alternative
    is finding out inside a paid GPU job.
    """
    from transformers import AutoProcessor

    try:
        return AutoProcessor.from_pretrained(
            repo, cache_dir=cache_dir,
            min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28,
        )
    except (TypeError, ValueError) as exc:
        print(f"[caption] {repo} takes no pixel budget ({type(exc).__name__})")
        return AutoProcessor.from_pretrained(repo, cache_dir=cache_dir)


# Two message shapes, because the captioner menu is open.
#
# Qwen3-VL takes the image as a content *part* and its chat template walks the
# list. JoyCaption — a LLaVA on Llama 3.1, and the first captioner anyone added
# under the gear — ships a processor-level chat_template.json (not the tokenizer
# one, which is clean) whose first user turn is
# `message['content'].replace('<|reserved_special_token_69|>', '')`: it splices
# its own image placeholder in and wants `content` to be a plain string with the
# image handed to the processor beside it. The parts list against that raised
# `'list object' has no attribute 'replace'` — Jinja's wording rather than
# Python's `'list' object`, which is the tell that the failure was inside the
# template and not in this file.
#
# `AutoModelForImageTextToText` was picked so the menu could be any vision LM
# transformers maps. This is the other half of that promise, which was missing:
# the loader was general and the message shape was still Qwen's.
#
# Nothing here invents a placeholder token. A template that splices its own is
# the entire reason the flat shape exists; one that does not gets the parts
# list, where the processor does the expansion. There is no third case to guess
# at, and guessing would put a literal `<image>` in somebody's caption.
def _vlm_inputs(processor: Any, image: Any, instruction: str, shape: str,
                system: str = "") -> Any:
    # String content for the system turn under both shapes. Every template
    # that takes a system message at all takes a string one — Qwen branches on
    # `content is string` and JoyCaption only ever concatenates it — so a parts
    # list here would buy nothing and break the half that does not branch.
    head = [{"role": "system", "content": system}] if system else []
    if shape == "parts":
        convo = head + [{
            "role": "user",
            "content": [{"type": "image", "image": image},
                        {"type": "text", "text": instruction}],
        }]
        return processor.apply_chat_template(
            convo, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
    text = processor.apply_chat_template(
        head + [{"role": "user", "content": instruction}],
        tokenize=False, add_generation_prompt=True,
    )
    return processor(text=[text], images=[image], return_tensors="pt")


def _caption_shape(processor: Any, instruction: str, repo: str,
                   system: str = "") -> str:
    """
    Which shape this captioner accepts, settled once and before the loop.

    The shape cannot vary between images, so discovering it per image was
    eighty identical unreadable log lines, zero captions written, and an A100
    rented for the length of the set to produce them. A whole-run fact belongs
    to the run — and the run should end on it rather than walk the set.

    A probe render rather than reading the template as text, because the
    template is not the only thing on this path that can refuse: the processor
    call after it is where a placeholder that fails to expand shows up, and the
    flat shape has both halves. 64px is arbitrary and large enough that no
    patch-size or min_pixels arithmetic divides by zero on the way through.
    """
    from PIL import Image

    # Parts first, and that order is load-bearing rather than alphabetical.
    # Qwen3-VL's template *renders* the flat shape too — it just drops the
    # image on the floor while doing it — so a probe that asked flat first
    # would caption blanks fluently and never say why. The shape that fails
    # loudly when it is wrong goes first.
    tried = []
    for shape in ("parts", "flat"):
        try:
            _vlm_inputs(processor, Image.new("RGB", (64, 64)), instruction,
                        shape, system)
            return shape
        except Exception as exc:
            tried.append(f"{shape} → {type(exc).__name__}: {exc}")
    # Named, both attempts, because the two failures separate the causes: a
    # template error is a captioner this app can be taught, and a processor
    # with no image input at all is a text-only repo that got past the add
    # route's config check.
    raise RuntimeError(
        f"{repo} accepted neither message shape, so no image would have been "
        f"captioned. Tried {' · '.join(tried)}"
    )


# How long a caption run goes between commits. Small enough that the page's
# mid-run tile refresh shows real progress, large enough that the commit's
# pause is noise against the per-image inference it interrupts.
CAPTION_COMMIT_S = 20


def _caption_images(
    image_dir: Path, trigger_word: str, job_id: str,
    preset: str, write_mode: str, model_key: str,
    instruction: str = "", max_tokens: int = 320,
    temperature: float = 0.6, top_p: float = 0.9,
) -> tuple[int, int]:
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText

    every = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    # "skip" only fills empties; every other mode has something to do with a
    # caption that already exists, so it visits the whole set.
    todo = [
        p for p in every
        if write_mode != "skip"
        or not p.with_suffix(".txt").exists()
        or not p.with_suffix(".txt").read_text().strip()
    ]
    if not todo:
        return 0, 0

    models = _caption_models()
    spec = models.get(model_key) or models[DEFAULT_CAPTION_MODEL]
    repo = spec["repo"]
    print(f"[caption] {len(todo)}/{len(every)} images · {preset} · {repo}")
    _publish(job_id, phase="caption", step=0, total_steps=len(todo), percent=0)

    cache_dir = str(HF_CACHE)
    processor = _caption_processor(repo, cache_dir)
    # Extraction is for the house prompt only. A user's own instruction — a
    # draft, or a saved preset, which the page sends as an override — is kept
    # as the model writes it, even if it happens to say CAPTION: somewhere:
    # the owner's rule is that editing is binary when code depends on the
    # text, and a prompt whose output is chopped by rules it cannot see is the
    # half that drives people nuts.
    house = not instruction.strip() and preset in CAPTION_PRESETS
    instruction = _caption_instruction(preset, trigger_word, instruction)
    if house:
        print("[caption] house prompt · keeping what follows the final CAPTION:")
    # Ahead of the weights rather than after them, which is the whole payoff of
    # settling this once: a captioner whose template refuses the image is not
    # going to caption anything, and finding that out here costs a processor
    # load instead of a 17 GB pull and a model load.
    system = str(spec.get("system") or "")
    shape = _caption_shape(processor, instruction, repo, system)
    if shape != "parts":
        print(f"[caption] {repo} takes flat message content")
    # The Auto class rather than Qwen3VLForConditionalGeneration, because the
    # menu is no longer two known repos: a captioner added under the gear can be
    # any vision LM transformers maps — Qwen-VL, LLaVA, InternVL, Idefics. The
    # add route already proved the repo's config resolves and looks multimodal;
    # this load is the final authority, and its failure names the repo.
    model = AutoModelForImageTextToText.from_pretrained(
        repo, dtype=torch.bfloat16, device_map="cuda:0", cache_dir=cache_dir,
    )
    model.eval()
    # Persist the downloaded weights now, on their own volume, so the next cold
    # start reuses them and the dataset commit below stays small.
    try:
        hf_cache.commit()
    except Exception as exc:
        print(f"[caption] hf cache commit skipped: {exc}")

    written = refused = 0
    last_commit = time.time()
    for i, img_path in enumerate(todo, 1):
        if _stop_requested(job_id):
            print("[caption] stop requested")
            break
        try:
            image = _upright(Image.open(img_path)).convert("RGB")
            inputs = _vlm_inputs(processor, image, instruction, shape,
                                 system).to("cuda:0")
            # The processor hands back float32 pixels while the model is bf16.
            # Qwen3-VL's vision tower casts its input on the way in and never
            # minded; LLaVA's does not, so JoyCaption died on the first image
            # with a dtype mismatch the moment the template stopped being the
            # thing in front of it.
            if "pixel_values" in inputs and inputs["pixel_values"].is_floating_point():
                inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

            with torch.no_grad():
                # temperature 0 means greedy, and transformers refuses
                # temperature=0.0 with do_sample=True rather than inferring it.
                out = model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    **({"temperature": temperature, "top_p": top_p}
                       if temperature > 0 else {}),
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
                # The model is told the trigger is prepended automatically and
                # usually complies; when it does not, prepending over its copy
                # is where "chgl, chgl, …" came from — and each recaption run
                # stacked one more. Stripped here, once, so every branch below
                # composes from a caption that is known not to carry it.
                if house:
                    caption = _caption_extract(caption)
                # After the extraction, not before: the caption is told to
                # open with the trigger, and this is what keeps that from
                # doubling when the prepend below puts it there as well.
                caption = _strip_leading_trigger(caption, trigger_word)
                txt = img_path.with_suffix(".txt")
                existing = txt.read_text().strip() if txt.exists() else ""
                if write_mode == "append" and existing:
                    # The trigger already leads the existing text (or the Fix
                    # button will put it there); prefixing it again here would
                    # double it mid-caption.
                    final = f"{existing} {caption}"
                elif write_mode == "prepend" and existing:
                    # The new caption takes the front, so the trigger moves
                    # with it — stripped off the old text rather than left to
                    # appear twice. The helper loops, so a sidecar already
                    # doubled by the old bug comes out healed rather than
                    # carrying its history.
                    existing = _strip_leading_trigger(existing, trigger_word)
                    head = f"{trigger_word}, {caption}" if trigger_word else caption
                    final = f"{head} {existing}"
                else:
                    final = f"{trigger_word}, {caption}" if trigger_word else caption
                txt.write_text(_trimmed(final))
                written += 1
        except Exception as exc:
            print(f"[caption] {img_path.name} failed: {exc}")

        _publish(job_id, phase="caption", step=i, total_steps=len(todo),
                 percent=round(i / len(todo) * 100))
        # Committed as it goes, not only at the end: the page refreshes tiles
        # mid-run, and against an end-only commit that refresh could never
        # show a thing — "captions land visibly" was a fiction for the whole
        # run. Time-based, so the cost tracks the clock (about a second of
        # idle every twenty), not the size of the set.
        if time.time() - last_commit >= CAPTION_COMMIT_S:
            volume.commit()
            last_commit = time.time()

    del model
    torch.cuda.empty_cache()
    volume.commit()
    return written, refused


@app.function(
    # A100 rather than the training GPU: Qwen3-VL-8B in bf16 is ~17 GB, so this
    # does not need the headroom a rank-32 Krea 2 run does.
    image=caption_image, gpu="A100-40GB", cpu=2.0, timeout=2 * 60 * 60,
    volumes=MODAL_VOLUMES_HF,
)
def caption_job(
    job_id: str, dataset: str, trigger_word: str = "",
    preset: str = DEFAULT_CAPTION_PRESET,
    write_mode: str = "skip", model: str = DEFAULT_CAPTION_MODEL,
    instruction: str = "", max_tokens: int = 320,
    temperature: float = 0.6, top_p: float = 0.9,
) -> dict[str, Any]:
    models = _caption_models()
    spec = models.get(model) or models[DEFAULT_CAPTION_MODEL]
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
        src, trigger_word.strip(), job_id, preset, write_mode, model,
        instruction=instruction, max_tokens=max_tokens,
        temperature=temperature, top_p=top_p)
    res = {
        "status": "completed", "job_id": job_id, "dataset": dataset,
        "captioned": written, "refused": refused, "preset": preset,
        "model": model, "model_label": spec["label"],
        # The instruction is editable now, so the key alone no longer says what
        # ran — the record carries the exact text. Bounded at ~2 kB, so it does
        # not violate "keep the polled thing small", which is about results
        # that grow with output.
        "instruction": _caption_instruction(
            preset, trigger_word.strip(), instruction),
        "write_mode": write_mode,
        "duration_s": round(time.time() - started, 1),
        # A stopped run completes — the captions it wrote are real and the
        # loop unwound cleanly — but the page should not call it finished. The
        # flag is what lets "Stopping…" resolve into a readout that says so.
        "stopped": _stop_requested(job_id),
    }
    _publish(job_id, **res)
    return res


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


# What the trainer will accept, as tables rather than free strings.
#
# Same argument as CAPTION_PRESETS: the page builds its menus out of what
# /api/state serves, so a value that is not in one of these is the two sides
# having drifted — and the cost of guessing is not a form error, it is a GPU
# container that cold-starts, caches a dataset and then dies on argparse.
#
# The membership is chosen by what is *in the image*, which is the line that
# keeps this honest. CAME and Prodigy are the two optimizers anyone asks for
# next and neither is installed: `pip install -e .` on musubi brings
# bitsandbytes for adamw8bit and nothing else, so offering them would be
# offering a run that dies at import forty minutes in. They are one pip line
# away when someone wants them, and the table is where that decision goes.
TRAIN_OPTIMIZERS = {
    "adamw8bit": {"label": "AdamW 8-bit", "note": "bitsandbytes — the default, and the cheapest in VRAM"},
    "adamw": {"label": "AdamW", "note": "torch's own, a little steadier and a little heavier"},
    "adafactor": {"label": "Adafactor", "note": "lowest VRAM of the three; wants a lower learning rate"},
}
# transformers' schedulers, which is what musubi resolves these through. A
# constant rate is what a LoRA run has always used here; cosine is the one
# worth reaching for on a long run, because the last epochs stop overshooting.
LR_SCHEDULERS = {
    "constant": {"label": "Constant", "note": "the same rate throughout"},
    "constant_with_warmup": {"label": "Constant + warmup", "note": "eases in, then holds"},
    "cosine": {"label": "Cosine", "note": "decays to zero — steadier last epochs"},
    "cosine_with_restarts": {"label": "Cosine restarts", "note": "decays and jumps back up"},
    "linear": {"label": "Linear", "note": "straight line down to zero"},
    "polynomial": {"label": "Polynomial", "note": "a slower curve down"},
}
# Where in the noise schedule the training steps are drawn from. Krea 2 is
# flow-matching, so this and `discrete_flow_shift` are one decision in two
# fields: `shift` is the only sampling that reads the shift value, which is why
# the field is disabled beside every other one rather than quietly ignored.
TIMESTEP_SAMPLINGS = {
    "shift": {"label": "Shift", "note": "flow-matching, weighted by the shift value"},
    "sigmoid": {"label": "Sigmoid", "note": "concentrates on the middle of the schedule"},
    "uniform": {"label": "Uniform", "note": "every timestep equally likely"},
    "sigma": {"label": "Sigma", "note": "sampled on the sigma curve"},
}

# One place the dials' defaults are written. The page opens on these and the
# job applies them, so the menu cannot open on a value the backend would not
# have picked — two places spelling the same default is how they drift.
TRAIN_DEFAULTS = {
    "resolution": 1024, "batch_size": 1, "num_repeats": 1,
    "network_dim": 32, "network_alpha": 32, "learning_rate": 1e-4,
    "max_train_epochs": 30, "save_every_n_epochs": 5, "seed": 42,
    "optimizer_type": "adamw8bit", "lr_scheduler": "constant",
    "timestep_sampling": "shift", "discrete_flow_shift": 2.5,
    "fp8": False, "blocks_to_swap": 0,
}


@app.function(
    image=trainer_image, gpu=GPU, cpu=4.0, timeout=6 * 60 * 60,
    volumes=MODAL_VOLUMES,
)
def train_job(
    job_id: str, dataset: str, lora_name: str, trigger_word: str,
    resolution: int = 1024, batch_size: int = 1, num_repeats: int = 1,
    network_dim: int = 32, network_alpha: int = 32, learning_rate: float = 1e-4,
    max_train_epochs: int = 30, save_every_n_epochs: int = 5,
    discrete_flow_shift: float = 2.5, seed: int = 42,
    optimizer_type: str = "adamw8bit", lr_scheduler: str = "constant",
    timestep_sampling: str = "shift",
    fp8: bool = False, blocks_to_swap: int = 0,
    session: str = "",
) -> dict[str, Any]:
    """
    One LoRA, one container.

    **Nothing here is shared between runs, which is what makes two of them at
    once free.** There is no `max_containers` on this function, unlike the GPU
    classes below: those pin a replica because a loaded checkpoint is the thing
    worth keeping warm, and a training run loads its own weights, writes its own
    scratch under `work/{job_id}` and its own output folder, then goes away.
    Spawning it twice is two cards on the board and two bills, and no state to
    coordinate.

    TODO: video, and the trainer is not this one. musubi trains Krea 2 and Wan;
    **H3 trains under AI Toolkit**, so a video branch here is a second trainer
    rather than a fourth script name in `train_job` — which is the opposite
    shape from the one the Wan plan had, where only the script names changed.
    That makes it a real piece of work rather than a rename, and it is why a
    dataset already counts its clips: the storage contract does not change for
    whichever trainer arrives.
    """
    if not NAME_RE.match(job_id) or not NAME_RE.match(lora_name):
        raise ValueError("job_id and lora_name must be 1-64 chars of [A-Za-z0-9_-].")

    started = time.time()
    log: deque[str] = deque(maxlen=400)
    # `started` on the record, not just in this frame: the card is polled by a
    # window that may have been opened hours after the run began, so elapsed has
    # to be answerable from the record alone. `session` rides along for the same
    # reason the captioner carries its model label — the board maps job records
    # back to cards, and a run whose card was deleted mid-flight should still be
    # able to say which one it belonged to.
    # Merged, never assigned. The route writes a `queued` record *before* the
    # spawn — a trainer cold start is minutes, and a card with no job record
    # cannot say anything at all — so the flag a Stop pressed during that wait
    # set is already sitting in this record, and `jobs[job_id] = {...}` would
    # overwrite it with `stop: False`. The run would then ignore a stop the page
    # had already confirmed, which is the worst shape a stop can take: the
    # button reports success and the GPU keeps billing.
    _publish(job_id, status="running", phase="starting",
             started=started, session=session)
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
                # Four dials that were hardcoded here and are now the card's,
                # because they are the four a second run changes. The shift
                # value is only read when the sampling is `shift` — the page
                # says so by disabling the field rather than by taking a number
                # it will not use.
                "--timestep_sampling", timestep_sampling,
                "--weighting_scheme", "none",
                "--discrete_flow_shift", str(discrete_flow_shift),
                "--optimizer_type", optimizer_type,
                "--lr_scheduler", lr_scheduler,
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
                 "batch_size": batch_size, "num_repeats": num_repeats,
                 "optimizer_type": optimizer_type, "lr_scheduler": lr_scheduler,
                 "timestep_sampling": timestep_sampling,
                 "discrete_flow_shift": discrete_flow_shift,
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
# Sessions — a training run, as a thing that outlives the run
#
# A session is the setup: which set, which name, which dials, and a pointer at
# the last job spawned from it. Runs are concurrent because `train_job` shares
# nothing between them, so what the page needs is not a "current run" but a
# board — and a board needs records that survive the window that made them.
#
# Everything here reads and writes exactly one Dict key. See `sessions` at the
# top of the file for why that is not a shortcut.
# --------------------------------------------------------------------------


# How long a record may go without a beat before its claim to be running is not
# believed. Deliberately generous: `_run` beats on every tqdm line, but the
# gaps between them are the phases that print nothing — a cold container pulling
# the trainer image, the latent cache committing a large set — and calling one
# of those dead would show a failed card for a run that is fine. Twenty minutes
# of silence is a container that is gone.
SESSION_STALE_S = 20 * 60

# How many cards one listing carries. See `list_sessions` for why there is a
# bound at all and why it is reported rather than silent.
SESSION_LIST_MAX = 100


def _sessions_all() -> list[dict[str, Any]]:
    """Every session, newest first. One round trip."""
    try:
        index = sessions.get(SESSION_INDEX) or {}
    except Exception as exc:
        print(f"[sessions] read failed: {exc}")
        return []
    rows = [r for r in index.values() if isinstance(r, dict)]
    rows.sort(key=lambda r: -float(r.get("created") or 0))
    return rows


def _session_put(rec: dict[str, Any]) -> dict[str, Any]:
    """
    Merge one session into the index.

    Get-update-put against a network Dict, so it takes the same lock `_publish`
    takes and for the same reason — except that here the second writer is
    another *window* rather than another thread, which the lock cannot reach.
    That is the accepted trade: two windows creating a session in the same
    round trip is a lost card, not a lost run, and the alternative is a key per
    session and the seven-second listing that came with it.
    """
    with _SESSION_LOCK:
        index = sessions.get(SESSION_INDEX) or {}
        cur = index.get(rec["id"]) or {}
        cur.update(rec)
        cur["updated"] = time.time()
        index[rec["id"]] = cur
        sessions[SESSION_INDEX] = index
        return cur


def _session_get(sid: str) -> dict[str, Any] | None:
    index = sessions.get(SESSION_INDEX) or {}
    rec = index.get(sid)
    return rec if isinstance(rec, dict) else None


def _session_drop(sid: str) -> bool:
    with _SESSION_LOCK:
        index = sessions.get(SESSION_INDEX) or {}
        gone = index.pop(sid, None) is not None
        if gone:
            sessions[SESSION_INDEX] = index
        return gone


_SESSION_LOCK = threading.Lock()

# What a card reads off the live job record. Named rather than merged wholesale
# because the job record also carries `stop`, `session` and the log tail, and a
# dict polled every few seconds must not grow with what the job happens to
# publish next.
_SESSION_LIVE = (
    "phase", "percent", "step", "total_steps", "epoch", "total_epochs",
    "rate", "eta", "elapsed", "loss", "note", "output_dir", "files",
    "duration_s", "error", "started",
)


def _session_view(rec: dict[str, Any]) -> dict[str, Any]:
    """
    One card: the setup, plus whatever the run it points at is doing.

    **Status is derived here and stored nowhere.** A status written into the
    session record would be a claim about a container that outlives it — the
    Dict survives the container, the app, the deploy and the image rebuild — and
    the first version of this trusted such a field, which turned three
    interrupted runs into three cards that said "training" for good. The job
    record's `beat` is the only thing that can answer whether anything is
    actually running, so a stale one is rewritten to failed on the way past:
    the card, the Stop button and the Start button clear together rather than
    the page having three opinions.
    """
    out = {k: rec.get(k) for k in
           ("id", "lora_name", "trigger_word", "dataset", "params", "job_id",
            "created", "updated", "runs")}
    job_id = rec.get("job_id")
    if not job_id:
        return {**out, "status": "draft"}

    job = jobs.get(job_id) or {}
    status = str(job.get("status") or "")
    if not status:
        # The record is gone — the Dict is never swept, so in practice this is a
        # job spawned against an older deployment. Inactive and honest about it
        # beats a card stuck on "starting".
        return {**out, "status": "unknown",
                "note": "The record for this run has expired. Start it again to re-run it."}

    if status in ("running", "queued"):
        beat = float(job.get("beat") or job.get("started") or 0)
        if beat and time.time() - beat > SESSION_STALE_S:
            status = "failed"
            job = {**job, "status": status,
                   "error": "The container running this stopped reporting. "
                            "Checkpoints written before it did are still in loras/."}
            _publish(job_id, status=status, error=job["error"])

    live = {k: job[k] for k in _SESSION_LIVE if k in job}
    # The stop flag is never cleared — the job checks it between steps and
    # unwinds, and nothing goes back to unset it — so a card that read it alone
    # said "Stopping — finishing the step it is on" over a run that finished
    # unwinding an hour ago. It is a fact about a *live* run, so it is only
    # reported while there is one.
    # Through `_stop_requested`, so the card reflects the same fact the job
    # reads. Asking the record alone would show "stopping" only for a press
    # that landed in the merged field — which is the half that could be lost.
    stopping = _stop_requested(job_id) and status in ("running", "queued")
    return {**out, **live, "status": status, "stopping": stopping}


def _num(payload: dict, key: str, cast, default):
    try:
        v = payload.get(key)
        return cast(v) if v not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _train_params(payload: dict[str, Any]) -> dict[str, Any]:
    """
    The dials, normalised against `TRAIN_DEFAULTS` and the three tables.

    Raises ValueError with the offending value named, because this is reached
    from the form: a menu that has drifted from the server's table should say
    which value it sent, not fall back to a default and train something else.
    """
    d = TRAIN_DEFAULTS
    out = {
        "resolution": max(256, min(2048, _num(payload, "resolution", int, d["resolution"]))),
        "batch_size": max(1, min(64, _num(payload, "batch_size", int, d["batch_size"]))),
        "num_repeats": max(1, min(100, _num(payload, "num_repeats", int, d["num_repeats"]))),
        "network_dim": max(1, min(512, _num(payload, "network_dim", int, d["network_dim"]))),
        "network_alpha": max(1, min(512, _num(payload, "network_alpha", int, d["network_alpha"]))),
        "learning_rate": _num(payload, "learning_rate", float, d["learning_rate"]),
        "max_train_epochs": max(1, min(1000, _num(payload, "max_train_epochs", int, d["max_train_epochs"]))),
        "save_every_n_epochs": max(1, _num(payload, "save_every_n_epochs", int, d["save_every_n_epochs"])),
        "seed": _num(payload, "seed", int, d["seed"]),
        "discrete_flow_shift": _num(payload, "discrete_flow_shift", float, d["discrete_flow_shift"]),
        "blocks_to_swap": max(0, min(60, _num(payload, "blocks_to_swap", int, d["blocks_to_swap"]))),
        "fp8": bool(payload.get("fp8")),
    }
    for key, table in (("optimizer_type", TRAIN_OPTIMIZERS),
                       ("lr_scheduler", LR_SCHEDULERS),
                       ("timestep_sampling", TIMESTEP_SAMPLINGS)):
        v = str(payload.get(key) or d[key])
        if v not in table:
            raise ValueError(f"No {key.replace('_', ' ')} {v!r}. One of: {', '.join(table)}")
        out[key] = v
    return out


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
# `sa_solver` is the same class — a stochastic Adams solver whose noise is
# scheduled by a tau interval the solver owns — and it is in core at COMFY_SHA,
# so listing it is the whole install. Unmeasured at Turbo's 8 steps: its
# predictor is third order, and with that few steps the history it builds on
# is thin, so treat it as a RAW sampler until a Turbo render says otherwise.
SAMPLERS = [
    "er_sde", "euler", "res_multistep", "dpmpp_2m", "heun", "ddpm", "lcm",
    "deis", "ipndm", "sa_solver",
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
# What a style-reference render falls back to when the sampler is untouched.
# Measured, not preferred: at seed 42 the K/V engine under er_sde dropped the
# prompt's dog entirely, and under the pack's own tested euler_ancestral/simple
# kept it — its README locks a "tested route" and the sampler is part of it. A
# sampler the user actually set still wins; this only replaces the *default*,
# because a default is the app's own choice and the app should choose the one
# that was measured working.
STYLE_DEFAULTS = {"sampler": "euler_ancestral", "scheduler": "simple"}
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

# What ComfyUI prints, once per key, when a LoRA carries weights the loaded
# model has no parameter for — `comfy/sd.py`'s `load_lora_for_models`. It is a
# warning rather than an error there, and rightly so: a partial match is a real
# thing. Ours is the count, not the verdict.
COMFY_LORA_MISS = "NOT LOADED"

# What ComfyUI prints when `--use-sage-attention` actually took hold. Its
# *absence* is the whole point: the flag is accepted whatever happens, and
# kernels compiled for one architecture simply do not load on another — so the
# weights load, the pictures come out, and the run takes roughly twice as long
# with nothing anywhere saying why. On Hopper this has never been seen missing,
# which is exactly why it needs saying out loud somewhere that is not Hopper.
COMFY_SAGE_MARK = "sage attention"

# ComfyUI narrates the model loading that happens between accepting a graph and
# taking its first sampling step — on H3 that is 47 seconds of VAE, text encoder
# and a 20 GB DiT — and every word of it went to a log nobody was reading while
# the page showed one unchanging line.
#
# **A wait is bearable in proportion to what it shows, not to how long it is.**
# The person who found this put it exactly: they will sit through forty minutes
# of an agent working because every step is visible, and a still button for one
# minute is an eternity. So the last static stretch on this path is spent, and
# what spends it is output we were already printing.
COMFY_LOADING_RE = re.compile(r"Requested to load (?P<name>[\w.]+)")
COMFY_STAGED_RE = re.compile(
    r"Model (?P<name>[\w.]+) prepared for dynamic VRAM loading\.\s*(?P<mb>\d+)MB")

# H3's flow shifts, and the reason `MiniMaxH3SigmaShift` is normally absent from
# our graph: these are the *model's own* defaults — `MiniMaxH3.forward` reads
# `transformer_options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video)`
# with `sigma_shift_video=12.0, sigma_shift_audio=3.0`, and
# `supported_models.MiniMaxH3.sampling_settings` says `shift: 12.0` besides. So
# the node at its defaults is exactly equivalent to no node, and adding one
# unconditionally would be chrome on the graph.
#
# It stops being equivalent the moment a distilled LoRA is loaded: 20 steps down
# to 4 wants a different curve, and this is the only lever for one. The shift is
# the last word on the sampling curve, so it goes *after* the LoRAs, never
# before.
H3_SHIFT_VIDEO = 12.0
H3_SHIFT_AUDIO = 3.0


def _install_pack_requirements(tag: str) -> None:
    """
    Pay for the Playground's packs before ComfyUI imports them.

    A pack's Python dependencies have to be in this container's environment by
    the time ComfyUI walks custom_nodes, or the import fails and the pack
    silently is not there — require_nodes only guards our own. Installed at
    process start rather than baked into the image because installing a pack
    is a gesture in the Playground, not a deploy; the bill is a pip run on
    every cold start once packs exist, so each one prints what is being paid
    for (a wait costs what it shows).

    The torch trio is filtered out of every requirements file before pip sees
    it. A pack that pins torch would replace the cu130 wheel with a
    default-CUDA one and take SageAttention's ABI with it — the exact failure
    the comfy_image build notes warn about — and a pack that lists it unpinned
    is satisfied already. Skips are printed, not silent: a pack that truly
    needs a different torch needs a conversation, not a broken container.
    """
    packs = sorted(d for d in PLAYGROUND_NODES.iterdir()
                   if d.is_dir() and not d.name.startswith(".")) \
        if PLAYGROUND_NODES.is_dir() else []
    for pack in packs:
        req = pack / "requirements.txt"
        if not req.is_file():
            continue
        wanted, skipped = [], []
        for line in req.read_text().splitlines():
            bare = line.split("#", 1)[0].strip()
            if re.match(r"^(torch|torchvision|torchaudio)\b", bare):
                skipped.append(bare)
            elif bare:
                wanted.append(bare)
        if skipped:
            print(f"[{tag}] {pack.name}: leaving {', '.join(skipped)} to the "
                  "image's own pins", flush=True)
        if not wanted:
            continue
        print(f"[{tag}] installing {pack.name}'s requirements "
              f"({len(wanted)})", flush=True)
        done = subprocess.run(
            ["pip", "install", "--no-input", *wanted],
            capture_output=True, text=True)
        if done.returncode != 0:
            # The pack stays on the shelf and ComfyUI starts without it; the
            # error names the pack and carries pip's last words, because "a
            # node is missing" minutes later would not.
            tail = (done.stderr or done.stdout or "").strip().splitlines()[-8:]
            print(f"[{tag}] {pack.name}: pip failed — the pack will not "
                  "load this start\n" + "\n".join(tail), flush=True)


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
        # The card under this process, read once at start. An out-of-memory
        # error that does not say which card it ran out of is an error you hit
        # twice: the same graph on a 48 GB L40S and an 80 GB H100 fails and
        # fits, and the fix is under the gear, not in the graph.
        self.card = ""
        self.card_gb = 0
        # Keys a LoRA carried that this DiT has no home for. Counted rather than
        # logged-and-forgotten because it is the one LoRA failure with no
        # symptom: the graph runs, the clip arrives, and nothing anywhere says
        # the weights did nothing. See `_drain`.
        self._unmatched = 0
        self._told = False
        self._log: deque[str] = deque(maxlen=200)
        self._proc: subprocess.Popen | None = None
        # The unpacked execution_error of the last failed graph, so a caller
        # that wants the failing *node* (the Playground editor lights it up)
        # does not have to parse it back out of the message string.
        self.last_error: dict[str, Any] | None = None

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
        # Weights are absolute because they live on their own mount now, and
        # ComfyUI's extra_config joins with os.path.join — an absolute value
        # simply wins over base_path, which is the documented posix behaviour
        # rather than a trick. loras/ stays relative: it is on the data volume
        # the base_path names.
        #
        # Two loras roots, not one, and the second is the spool. A container
        # that has loaded a LoRA has it mapped off /workspace, which refuses
        # every later `volume.reload()`, which freezes the mount — so a LoRA
        # trained *while this container was warm* is invisible here while the
        # page shows it. `_lora_visible` pulls that file's committed bytes onto
        # local disk by RPC, and this root is what makes the name it already
        # answers to resolve to them. Same relative path under either root, so
        # nothing downstream can tell which one served it.
        #
        # Created before ComfyUI starts, empty or not: `folder_paths` records
        # the directories it walked and invalidates its filename cache when one
        # of them changes, and a root that did not exist at startup is a root
        # whose first file arrives unnoticed.
        SPOOL_LORAS.mkdir(parents=True, exist_ok=True)
        # custom_nodes is the Playground's shelf riding the same mechanism as
        # the weights: the volume layout is the contract and ComfyUI adapts to
        # it. Created before ComfyUI starts for the same reason the loras
        # roots are — a directory that did not exist at startup is one whose
        # first pack arrives unnoticed until the next restart.
        PLAYGROUND_NODES.mkdir(parents=True, exist_ok=True)
        (COMFY / "extra_model_paths.yaml").write_text(
            "visionary:\n"
            f"  base_path: {WORKSPACE}/\n"
            f"  diffusion_models: {MODELS}\n"
            # `unet` is the same folder under an older key, and ComfyUI-GGUF
            # reads that one — it registers `unet_gguf` against it rather than
            # against `diffusion_models`. Naming both costs a line and is the
            # difference between a quantised tier being offered and the loader
            # reporting an empty list, which reads as the file not having
            # downloaded.
            f"  unet: {MODELS}\n"
            f"  text_encoders: {MODELS}\n"
            f"  clip: {MODELS}\n"
            f"  vae: {MODELS}\n"
            "  loras: loras/\n"
            f"  custom_nodes: {PLAYGROUND_NODES}\n"
            "visionary_spool:\n"
            f"  base_path: {SPOOL}/\n"
            "  loras: loras/\n"
        )
        _install_pack_requirements(self.tag)

        # A fresh clone ships input/ and output/, but a changed --base-directory
        # or a pruned image would not, and the failure would surface as a
        # LoadImage error about a file we thought we had just written.
        (COMFY / "input").mkdir(parents=True, exist_ok=True)
        (COMFY / "output").mkdir(parents=True, exist_ok=True)

        self._proc = subprocess.Popen(
            [COMFY_PYTHON, "main.py", "--listen", "127.0.0.1", "--port", str(COMFY_PORT),
             "--disable-auto-launch", "--disable-metadata",
             # For H3, and only H3 — this is argv on both containers because
             # one `_Comfy` starts both, but the image graph opts back out
             # with a `ModelAttentionBackend` node. Sage's win is a
             # long-sequence win, and H3's packed video/text/audio sequence is
             # the only place here that has one; what Krea 2 got out of the
             # same flag was intermittent black blotches. The full account is
             # at the node, in `_krea2_graph`.
             #
             # H3 never passes an attention mask, so nothing on this path
             # depends on sage's mask support — which is as well, because it
             # has none: `sageattn` takes no `attn_mask`, ComfyUI reads that
             # off the signature into SAGE_ATTENTION_SUPPORTS_MASK, and a
             # masked call falls back per-call to PyTorch. That is a fact
             # about an upstream we clone unpinned, so it is a fact with a
             # date on it: if `attn_mask` ever appears in that signature,
             # masked calls stop falling back and start being quantized, and
             # ComfyUI's own Lightricks model has `low_precision_attention=
             # False, # sageattn mask support is unreliable` sitting next
             # door as the review of that.
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
             "--bf16-text-enc",
             # Last, so a local override wins over what is fixed above it.
             *COMFY_EXTRA_ARGS],
            cwd=str(COMFY),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        threading.Thread(target=self._drain, daemon=True).start()
        self._wait_ready()
        self._read_card()
        self._confirm_sage()

    def _read_card(self) -> None:
        """
        Which card, and how much of it, from ComfyUI's own view of the device.

        Best effort and once: the first `/system_stats` after ready. It feeds
        the out-of-memory error in `run` and one log line here, which is the
        line that says whether the cheap card somebody picked under the gear
        is the card this process actually got.
        """
        try:
            dev = ((self.get("/system_stats") or {}).get("devices") or [{}])[0]
            name = str(dev.get("name") or "")
            total = int(dev.get("vram_total") or 0)
        except Exception:
            return
        # "cuda:0 NVIDIA L40S : cudaMallocAsync" — the middle is the card.
        parts = [p.strip() for p in name.split(":") if p.strip()]
        self.card = parts[1] if len(parts) > 1 else name
        self.card_gb = round(total / 10**9)
        if self.card:
            print(f"[{self.tag}] on {self.card}, {total / 2**30:.1f} GiB",
                  flush=True)
    def _confirm_sage(self) -> None:
        """Say so if the attention backend we asked for did not arrive.

        Checked at startup rather than per render because it is a property of
        the process, and reported with the three facts that separate the causes:
        which card, what the kernels were built for, and what it costs. Krea 2
        is unaffected either way — `_krea2_graph` opts back out with a
        `ModelAttentionBackend` node, measured at 3.23s against sage's 3.26s —
        so the cost lands on H3 alone, which is the one path with a long enough
        sequence to have wanted it.
        """
        if any(COMFY_SAGE_MARK in line.lower() for line in self._log):
            return
        arch = os.environ.get("VISIONARY_GPU_ARCH", "unknown")
        print(f"[{self.tag}] SageAttention was requested and ComfyUI did not "
              f"confirm it.\n"
              f"[{self.tag}]   card:  {self.card or '?'} "
              f"(sm_{arch})\n"
              f"[{self.tag}]   built: TORCH_CUDA_ARCH_LIST="
              f"{os.environ.get('TORCH_CUDA_ARCH_LIST', '?')} — rebuild "
              f"SageAttention for this card\n"
              f"[{self.tag}]   cost:  H3 video falls back to PyTorch attention, "
              f"roughly half speed.\n"
              f"[{self.tag}]          Krea 2 is unaffected; it opts out of sage "
              f"in the graph anyway.", flush=True)

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
            # ComfyUI prints one of these per key it could not place, which on a
            # LoRA for the wrong architecture is hundreds. Counted here and
            # published once below, because `_publish` is a round trip to a
            # network Dict and doing it per key would cost more than the render.
            if COMFY_LORA_MISS in line:
                self._unmatched += 1
                continue
            # Named, and with a size when ComfyUI has one. Published before the
            # tqdm check because these arrive first and stop the moment it does.
            if self.job_id:
                load = COMFY_STAGED_RE.search(line) or COMFY_LOADING_RE.search(line)
                if load:
                    name = _pretty_model(load.group("name"))
                    mb = load.groupdict().get("mb")
                    _publish(self.job_id, phase=(
                        f"loading {name}" + (f" · {int(mb) / 1000:.1f} GB" if mb else "")))
                    continue

            m = TQDM_RE.search(line)
            if m and self.job_id:
                fields: dict[str, Any] = {
                    "phase": "generate",
                    "step": int(m.group("step")),
                    "total_steps": int(m.group("total")),
                    "percent": int(m.group("pct")),
                }
                # The first progress line is the moment every LoRA has finished
                # loading, so it is the earliest the count is final and the
                # cheapest place to say so — it rides a publish already going.
                if self._unmatched and not self._told:
                    self._told = True
                    fields["lora_note"] = (
                        f"{self._unmatched} LoRA weights did not match this "
                        f"model and were skipped — check the LoRA was trained "
                        f"for this architecture.")
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

    def restart(self) -> None:
        """
        Kill the warm engine on purpose and start a fresh one.

        `_revive()` replaces a ComfyUI that died; this is the lever for one
        that is alive and wrong — a pack that wrapped the resident model in
        place (the CacheDiT removal needed a container kill for exactly this),
        or a pack installed since this process walked custom_nodes. The bill
        is honest and stated where it is paid: the resident checkpoint dies
        with the process and reloads when the next graph asks for it.

        Runs as a queued method on the generator, so it waits its turn behind
        a job in flight rather than shooting one.
        """
        if self._proc is not None and self._proc.poll() is None:
            print(f"[{self.tag}] restarting ComfyUI on request", flush=True)
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=10)
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

        **That accumulation is now prevented rather than only recovered from.**
        `VisionaryFreeRegional` sits between the sampler and the decode on every
        regional graph and drops the session's device copies as the render ends
        — 1026 tensors a run, measured, with headroom flat at 46.7 GiB across
        three consecutive renders where it used to step down each time. This
        stays because prevention is not proof: a leak from somewhere else, or a
        single graph genuinely too big for the card, still lands here, and the
        node cannot help a run that has already failed.

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
        self._unmatched, self._told = 0, False
        self.last_error = None
        try:
            self._note_headroom()
            prompt_id = self.post("/prompt", {"prompt": graph})["prompt_id"]
            return self._await(job_id, prompt_id, what)
        except RuntimeError as exc:
            # StopRequested is not a RuntimeError, so a cancelled take does not
            # come through here and does not pay for a reload it never earned.
            if COMFY_OOM_MARK in str(exc):
                self._reclaim()
                if self.card:
                    # The card, because the same graph fits on the next size
                    # up and the fix is under the gear — see `card` in init.
                    raise RuntimeError(
                        f"{exc} This container is on {self.card} "
                        f"({self.card_gb} GB); a larger card is under the "
                        f"gear, and costs one cold start.") from exc
            raise
        finally:
            self.job_id = None

    def run_text(self, graph: dict[str, Any], timeout: float = 180.0) -> str:
        """
        Queue a graph whose output is words, and return them.

        **A sibling of `run`/`_await` rather than a branch inside them.** That
        pair extracts *filenames* — it walks `images`/`videos`/`gifs` and raises
        "saved nothing" when it finds none, which is exactly what a text output
        looks like to it. Teaching it a second output shape would put a rewrite's
        failure modes inside the path every render takes, and the render path is
        the one thing here that must not get more ways to go wrong.

        What it deliberately does *not* inherit is the forty-minute clip's
        tolerances: no stop check, because nothing offers to cancel a rewrite,
        and a short poll ceiling, because this answers in seconds and a rewrite
        that has hung for three minutes is a rewrite worth failing rather than
        waiting on.
        """
        self._revive()
        prompt_id = self.post("/prompt", {"prompt": graph})["prompt_id"]
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                entry = (self.get(f"/history/{prompt_id}") or {}).get(prompt_id)
            except OSError:
                # `get` has already established the process is alive, so this is
                # one dropped socket on a server that is still serving.
                time.sleep(0.5)
                continue
            if not entry:
                time.sleep(0.4)
                continue
            status = entry.get("status", {})
            if status.get("status_str") == "error" or not status.get("completed", True):
                raise RuntimeError(self._why_failed(status))
            for out in entry.get("outputs", {}).values():
                said = out.get("text")
                if said:
                    # The node returns a one-item list; ComfyUI passes `ui`
                    # through untouched.
                    return said[0] if isinstance(said, list) else str(said)
            raise RuntimeError(
                "the rewrite graph completed and returned no text.\n"
                + "\n".join(list(self._log)[-20:]))
        raise RuntimeError(f"the rewrite did not answer within {timeout:.0f}s.")

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

    def _why_failed(self, status: dict[str, Any]) -> str:
        """
        Turn ComfyUI's execution_error message into one line worth reading.

        The raw record nests the useful part — node type and exception — under
        a list of (event, payload) pairs, and a bare "status_str: error" sends
        you to the container logs for something the record already knows.

        The payload is also kept whole on `last_error`, because the message
        collapses node id and node type into one name and the Playground needs
        the id back to light up the node that broke.
        """
        for event, payload in status.get("messages") or []:
            if event == "execution_error":
                self.last_error = {
                    "node_id": payload.get("node_id"),
                    "node_type": payload.get("node_type"),
                    "message": payload.get("exception_message") or "failed",
                }
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


def _window_key(sid: str) -> str:
    return f"win:{sid}"


def _touch_session(sid: str) -> None:
    """Record that a window is still open. A timestamp in the sessions Dict
    is the entire payload — it used to be an empty file's mtime under
    `drafts/.sessions`, a volume write and a commit per beat per tab, for
    a fact that is liveness and not a record."""
    if not NAME_RE.match(sid or ""):
        return
    try:
        sessions[_window_key(sid)] = time.time()
    except Exception as exc:  # noqa: BLE001 — a missed beat is a late sweep, not a fault
        print(f"[session] beat {sid}: {type(exc).__name__}: {exc}", flush=True)


def _window_seen(sid: str) -> float:
    try:
        return float(sessions.get(_window_key(sid)) or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


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
        seen = _window_seen(sid) if NAME_RE.match(sid or "") else 0.0
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
        # The one deletion nobody asks for by name, so it is the one that must
        # explain itself in the log: what went, how much of it, whose window it
        # belonged to and how long that window had been silent. 80 images
        # vanishing mid-curation with no line anywhere is how this was learned.
        held = sum(1 for p in d.iterdir()
                   if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS)
        print(f"[sweep] retired draft {d.name!r}: {held} files, "
              f"session {sid or 'none'} quiet {int((now - seen) / 60)}m",
              flush=True)
        shutil.rmtree(d, ignore_errors=True)
        swept += 1

    # This is the app's periodic housekeeping pass, which makes it the one place
    # a one-time migration can run without waiting for the user to delete
    # something in each root. Each is a no-op on a volume that never had the
    # thing, and on one that did it runs once and then costs a stat.
    _drop_legacy_trash(DATASETS)
    _adopt_legacy_volume_drafts()
    _drop_legacy_thumbs(DATASETS)

    # Heartbeats outlive the drafts they kept alive; a day is well past the
    # point where one can still be protecting anything.
    try:
        for key in list(sessions.keys()):
            if isinstance(key, str) and key.startswith("win:"):
                if now - float(sessions.get(key) or 0.0) > 86400:
                    sessions.pop(key, None)
    except Exception as exc:  # noqa: BLE001 — housekeeping never fails a beat
        print(f"[sweep] heartbeats: {type(exc).__name__}: {exc}", flush=True)
    return swept


_ADOPTED_LEGACY = False


def _adopt_legacy_volume_drafts() -> None:
    """
    Drafts left on the volume from before drafts moved off it, kept rather
    than lost.

    A set under the old `drafts/` was never saved, so under the rule it has
    no claim on the volume — but deleting somebody's eighty images because a
    rule changed is the wrong side of that rule. Each one is moved into
    `datasets/` under its own name, once, and the log says so; deleting it
    is one press in Sets. `.sessions` markers go with the folder. Once per
    container, and a no-op on a volume that has no `drafts/`.
    """
    global _ADOPTED_LEGACY
    if _ADOPTED_LEGACY:
        return
    _ADOPTED_LEGACY = True
    legacy = WORKSPACE / "drafts"
    if not legacy.is_dir():
        return
    moved = 0
    for d in sorted(legacy.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        target = DATASETS / d.name
        if target.exists():
            target = DATASETS / f"{d.name}-recovered"
        try:
            DATASETS.mkdir(parents=True, exist_ok=True)
            shutil.move(str(d), str(target))
            _write_dataset_meta(target, session="")
            moved += 1
            print(f"[sweep] kept the unsaved set {d.name!r} as {target.name!r}: "
                  f"drafts live on the container now, and a rule change is "
                  f"not a reason to lose it", flush=True)
        except OSError as exc:
            print(f"[sweep] could not keep {d.name!r}: {exc}", flush=True)
    shutil.rmtree(legacy, ignore_errors=True)
    if moved:
        volume.commit()


def _drop_legacy_thumbs(root: Path) -> None:
    """The `.thumbs/` caches that used to sit inside every set on the volume.
    Derived, and rebuilt on the container now, so the volume copy is only
    bytes; swept once per set found holding one."""
    if not root.is_dir():
        return
    for d in root.iterdir():
        cache = d / THUMB_DIR
        if d.is_dir() and cache.is_dir():
            shutil.rmtree(cache, ignore_errors=True)


def _dataset_images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS) if d.is_dir() else []


def _dataset_videos(d: Path) -> list[Path]:
    """
    The clips in a set. Separate from the images rather than a `kind` filter on
    one walk, because every caller so far wants one or the other: the trainer
    trains images, the contact sheet counts both, and a caller that meant images
    and silently got clips is the failure this split makes impossible.
    """
    return sorted(p for p in d.iterdir() if p.suffix.lower() in VIDEO_EXTS) if d.is_dir() else []


def _caption_of(img: Path, overlay: dict[str, str] | None = None) -> str:
    txt = img.with_suffix(".txt")
    if overlay is not None and txt.name in overlay:
        return overlay[txt.name].strip()
    try:
        return txt.read_text().strip() if txt.is_file() else ""
    except OSError:
        return ""


# ── caption sidecars are read through a committed-state overlay ────────────
#
# The `.txt` sidecars are the one thing under datasets/ with a second writer:
# `caption_job` runs in its own container and commits, and this container's
# mount only catches up when a reload succeeds. During use it rarely does — a
# reload is refused while any request holds a file open, and the burst that
# opens a set (thumbs, headers, sidecars) holds descriptors for most of the
# window — and every dataset route ignored `_reload_volume()`'s False. The
# stale view was then served with a straight face: find & replace substituted
# in the mount's old text and committed it back over a caption run's output,
# and a refetch repainted captions that had already been replaced. The live
# web container's log holds the receipt: "reload skipped … path datasets is
# open".
#
# So the sidecars are read the way the gallery is read: committed state, by
# RPC, which no descriptor can refuse. One listdir per set answers every
# sidecar's committed (mtime, size) in a single round trip, and bytes are
# fetched only for files whose committed copy is ahead of the mount's — this
# container's own writes are always at least that new, so a set nobody else
# has touched costs the listdir and nothing more. The mount stays what it is
# everywhere else here: the write target.
_OVERLAY_CACHE: dict[str, tuple[int, int, str]] = {}
_OVERLAY_CACHE_MAX = 4096


def _committed_sidecars(d: Path) -> dict[str, tuple[int, int]]:
    """{sidecar name: (mtime, size)} in committed state, one RPC. Empty on
    any failure — the overlay then falls back to the mount, which is the view
    we were serving anyway."""
    try:
        rel = d.relative_to(WORKSPACE)
    except ValueError:
        return {}
    out: dict[str, tuple[int, int]] = {}
    try:
        for e in volume.listdir(f"/{rel}", recursive=False):
            if e.type != modal.volume.FileEntryType.FILE:
                continue
            name = e.path.rsplit("/", 1)[-1]
            if name.endswith(".txt"):
                out[name] = (int(e.mtime), int(e.size))
    except modal.exception.NotFoundError:
        pass  # nothing committed under this set yet — a brand-new draft
    except Exception as exc:  # noqa: BLE001 — any RPC failure falls back
        print(f"[overlay] listdir {rel} failed ({type(exc).__name__}: {exc})"
              f" — captions read off the mount, which may be stale", flush=True)
    return out


def _committed_sidecar_tree(root: Path) -> dict[str, dict[str, tuple[int, int]]]:
    """{set name: {sidecar: (mtime, size)}} for every set under one root, in
    one recursive RPC. The listing used to ask per set, which put one round
    trip per row on the route the page waits on with a blank screen — 2.4s
    on a handful of sets, sequentially, for information one call answers."""
    try:
        rel = root.relative_to(WORKSPACE)
    except ValueError:
        return {}
    out: dict[str, dict[str, tuple[int, int]]] = {}
    try:
        for e in volume.listdir(f"/{rel}", recursive=True):
            if e.type != modal.volume.FileEntryType.FILE:
                continue
            path = e.path.lstrip("/")
            if path.startswith(f"{rel}/"):
                path = path[len(f"{rel}/"):]
            parts = path.split("/")
            # Exactly set/sidecar.txt — anything deeper is .thumbs and its kin.
            if len(parts) != 2 or not parts[1].endswith(".txt"):
                continue
            out.setdefault(parts[0], {})[parts[1]] = (int(e.mtime), int(e.size))
    except modal.exception.NotFoundError:
        pass  # nothing committed under this root yet
    except Exception as exc:  # noqa: BLE001 — any RPC failure falls back
        print(f"[overlay] listdir {rel} failed ({type(exc).__name__}: {exc})"
              f" — set stats read off the mount, which may be stale", flush=True)
    return out


def _caption_overlay(d: Path,
                     committed: dict[str, tuple[int, int]] | None = None,
                     ) -> dict[str, str]:
    """{sidecar name: committed text} for every sidecar whose committed copy
    is ahead of this mount's. Raw text, not stripped — replace needs the
    bytes as written."""
    if committed is None:
        committed = _committed_sidecars(d)
    out: dict[str, str] = {}
    # A draft on this container's disk has no committed copy to be behind:
    # what is on disk is the only copy, written by this container.
    if not committed:
        return out
    rel_dir = d.relative_to(WORKSPACE)
    for name, (mt, size) in committed.items():
        p = d / name
        try:
            # Strict: the RPC's mtime is integer seconds, so a write of our
            # own in the same second keeps the mount's copy — the newer of
            # the two by construction.
            if int(p.stat().st_mtime) >= mt:
                continue
        except OSError:
            pass  # not on the mount at all — committed is the only copy
        rel = f"{rel_dir}/{name}"
        hit = _OVERLAY_CACHE.get(rel)
        if hit and hit[0] == mt and hit[1] == size:
            out[name] = hit[2]
            continue
        raw = _volume_bytes(rel)
        if raw is None:
            continue
        text = raw.decode("utf-8", "replace")
        if len(_OVERLAY_CACHE) >= _OVERLAY_CACHE_MAX:
            _OVERLAY_CACHE.clear()  # blunt, and sized to never fire in practice
        _OVERLAY_CACHE[rel] = (mt, size, text)
        out[name] = text
    return out


def _dataset_stats(d: Path,
                   committed: dict[str, tuple[int, int]] | None = None,
                   ) -> dict[str, Any]:
    """
    One scandir pass, and no caption file is ever opened.

    This used to call `_dataset_images` and `_dataset_videos` (a walk each),
    stat every media file for `modified`, and *read every sidecar in the set*
    to count "captioned" — and the listing calls it for every set, so opening
    Sets cost roughly three FUSE round trips per file in the whole library.
    The gallery never paid that (one sidecar per job, paginated), which is why
    the two screens felt like different products. Now a sidecar that exists
    and is non-empty counts as captioned — the panel inside a set still reads
    the real text, so the only thing this can miscount is a sidecar holding
    pure whitespace — and everything comes off one directory listing plus one
    cached stat per entry.

    Both kinds still count toward "captioned": a clip with no caption is as
    uncaptioned as a photograph with none, and a set that reported 24 of 24
    while holding six uncaptioned clips would be answering a question nobody
    asked.
    """
    images: list[str] = []
    videos: list[str] = []
    sidecars: set[str] = set()
    newest = 0.0
    try:
        with os.scandir(d) as it:
            for entry in it:
                try:
                    if not entry.is_file():
                        continue
                    name = entry.name
                    dot = name.rfind(".")
                    ext = name[dot:].lower() if dot >= 0 else ""
                    if ext in IMAGE_EXTS:
                        images.append(name)
                        newest = max(newest, entry.stat().st_mtime)
                    elif ext in VIDEO_EXTS:
                        videos.append(name)
                        newest = max(newest, entry.stat().st_mtime)
                    elif ext == ".txt" and entry.stat().st_size:
                        sidecars.add(name[:dot])
                except OSError:
                    continue  # a file swept mid-listing costs the file, not the set
    except OSError:
        pass
    # A sidecar the caption container committed that this mount has not
    # caught up to still counts: the rail's count is the first place a
    # finished run shows, and it read 0-of-80 until a reload happened to
    # succeed.
    if committed:
        sidecars.update(n[:-4] for n, (_, size) in committed.items() if size)
    images.sort()
    # The sidecar replaces the media suffix (`photo.jpg` → `photo.txt`), so
    # membership is by the same stem `with_suffix` produces.
    captioned = sum(1 for n in images + videos if n[: n.rfind(".")] in sidecars)
    meta = d / "dataset.json"
    info: dict[str, Any] = {}
    if meta.is_file():
        try:
            info = json.loads(meta.read_text())
        except (OSError, json.JSONDecodeError):
            info = {}
    return {
        "name": d.name,
        # `count` stays the image count, because that is what every existing
        # reader means by it — the trainer's gate, the rail's line, the button
        # that refuses an empty set. The clips are a second number beside it
        # rather than folded into the first: a set of 24 images and 6 clips is
        # not a set of 30 of anything, and today only one of those two numbers
        # can be trained on.
        "count": len(images),
        "videos": len(videos),
        "captioned": captioned,
        "uncaptioned": len(images) + len(videos) - captioned,
        "trigger_word": str(info.get("trigger_word") or ""),
        "modified": newest,
        # A clip has no cover — web_image has no ffmpeg — so a video-only set
        # shows the empty glyph rather than a broken thumbnail, which is what
        # /api/thumb would answer for one.
        "cover": images[0] if images else None,
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
    overlay = _caption_overlay(d)
    captions = [(p.name, _caption_of(p, overlay)) for p in images]
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


# --------------------------------------------------------------------------
# Duplicates
# --------------------------------------------------------------------------
# A set arrives with duplicates in it far more often than not: the same shoot
# exported twice, a JPEG saved beside the PNG it came from, a crop kept next to
# its original, a phone album pulled in through two different apps. They are
# not neutral. The trainer repeats every image the same number of times, so a
# picture that is present three times is trained three times as hard as the
# rest of the set — and the symptom of that ("everything comes out in that
# room") never points back at the folder it came from.
#
# Byte equality is the cheap half of the problem and the useless one. A
# duplicate that survived a re-encode, a resize or a colour-profile conversion
# has different bytes and is the same picture, which is exactly the case a
# `sha256` map reports as clean. So the grouping is perceptual — dHash, below —
# and the sha rides along beside it anyway, because "these are the same file"
# and "these are the same picture" are different facts and only the first one
# makes the choice for you.
#
# Nothing here decides anything. It groups, it ranks, and every ranking says
# which number decided it, because the keep/cut call is the one thing on this
# page that cannot be derived: "which of these two nearly identical frames is
# the better photograph" is not a measurement.

# Two classes, and the second one is not a softer first.
#
# A **duplicate** is one picture stored more than once — a re-encode, a resize,
# a reformat, a re-grade. Deleting all but one loses nothing, so a duplicate
# group arrives with a keeper already chosen.
#
# A **similar** pair is two photographs that look alike. On a training set that
# is usually a burst: four consecutive frames of one subject, all of them
# legitimately useful, and there is no deterministic way to prove the second is
# a re-save rather than the next shutter release. So a similar group is shown
# and nothing in it is preselected. Collapsing these two into one scale of
# confidence is the mistake this replaces — five tiers between them meant the
# common case (an export at the same size and format as its original) landed in
# the middle, where nothing is preselected and the keeper flow never runs.
#
# **Both hashes must agree, and the two are not doing the same job.** dHash
# reads edge gradients, pHash reads low-frequency DCT energy, and they fail
# independently — so an AND is far tighter than either alone and much tighter
# than accepting on whichever happens to be closer.
#
# Measured on a 731-image editorial set, 266,815 pairs. dHash is the
# discriminator: `dhash <= 6` on its own isolates exactly the three real
# duplicates in that folder, and the next-closest pair anywhere is 7 — but that
# pair is `d7 p32`, two entirely unrelated photographs, which is precisely what
# the pHash half is there to refuse. So the AND stays and the pHash bound is
# set by the *other* thing it has to survive.
#
# That thing is a re-grade. dHash barely moves under one — a quarter-stop
# brighter measures 3 bits, 1.4x measures 4 — while pHash climbs to 16, because
# a global exposure change with any clipping in it moves the coarse structure
# the DCT reads. A bound of 10 filed those as merely similar, which is wrong:
# the same picture, exported brighter, is a copy. Swept from 10 to 24 against
# the real folder the duplicate count never moves off 3, so the loosening is
# free there and is what makes the re-grade land.
#
# The gap between the two classes is therefore carried by dHash — 6 against 12
# — and that is the split the real data draws. Every burst-shaped pair in that
# folder (two frames of one runway look) measures dHash 8 to 11: outside
# `duplicate`, inside `similar`, which is exactly where a photograph that is
# merely alike belongs.
DUPLICATE_MATCH = {"dhash": 6, "phash": 16}
SIMILAR_MATCH = {"dhash": 12, "phash": 18}

# Crop detection, and the leash that is the whole reason it is safe.
#
# A reframed copy moves every edge, so its direct distance sits outside both
# thresholds while a centre crop of one lines up exactly with the other. Two
# crops per image, compared every way but original-to-original — that comparison
# is the direct one, already made.
#
# **The variants are not the hazard; the threshold they were read at was.**
# Taking the best of nine variant pairs is nine chances to draw a low number
# against an unrelated image, so at a loose threshold it is ruinous: measured on
# a 731-image editorial set, accepting variant matches at dhash<=20/phash<=24
# flags 813 pairs against the 9 the direct comparison finds. Read at
# DUPLICATE_MATCH instead, the same nine pairs add **zero** pairs to that set,
# and land an 80% centre crop of a real photograph at distance 0 on both hashes.
# Strictly stronger evidence, and it may only ever claim **similar** — never a
# duplicate, so a crop match cannot preselect anything for deletion. That is
# also right on its own terms: a deliberate crop of a training image is a
# variation somebody made on purpose.
#
# What it does not reach is a crop it has no variant for. Measured on the same
# photograph: 90% and 80% land, 70% and 60% are missed. Catching those needs
# scale-invariant keypoints rather than a fixed grid of centre crops, which is a
# different technique and a much heavier one. The shares below are the cheap
# half of the problem and they are honest about being that.
CROP_SHARES = (0.9, 0.8)
# The crop gate, deliberately its own number even though it agrees with
# DUPLICATE_MATCH today. They were the same constant for an afternoon, and
# loosening the duplicate rule's pHash bound from 10 to 16 — a change argued
# entirely about *re-grades* — silently changed which crops the pass finds. That
# is a coupling with no reason behind it: a crop of a file has not been
# re-exposed, so the pHash headroom a re-grade needs is headroom a crop match
# gets for free, and it gets it nine times over because this is a minimum across
# nine variant pairs. Two names, so the next threshold argument moves only the
# thing it is about.
CROP_MATCH = {"dhash": 6, "phash": 16}

# The similar class is measured by a learned embedding, and the hashes above
# decide only the duplicate class. The two-class design survives; what moved is
# which instrument reads the second class, and the lesson is worth its length:
# **the editorial set the thresholds came from contained only exact copies**,
# so SIMILAR_MATCH was drawn from data with no true near-duplicate in it — a
# line calibrated on the wrong question. Measured against ground truth that
# actually holds near-duplicates (INRIA Holidays, 500 groups of one scene
# photographed from shifted viewpoints, 1,110,795 pairs), the hash band's best
# case is 5% recall at 17 false pairs, and no threshold rescues it: at
# d24/p26 recall is still 18% while false accepts pass 22,000. Gradient and
# DCT hashes measure *storage* similarity, so they find copies and cannot find
# the next shutter release — 199 of Holidays' same-scene pairs sit at cosine
# 0.95+ with hash distances up to d40/p36, invisible to any band.
#
# 0.94, and both folders drew the line. Below it the embedding stops
# separating "the same scene re-shot" from "the same shoot, different
# content": on the 731-image editorial folder the band from 0.90 to 0.925
# holds a pair of *different models* on one backdrop (0.923) and one model in
# two *different looks* on one stage (0.925) — pairs the page must never call
# alike, because a claim like that is how the classifier stops being believed
# — and then nothing at all until the exact copies at 1.0. On Holidays, 0.94
# still reads every burst-tier pair (the product's actual target) at a 7e-5
# false rate; what it gives up is the walk-around-the-corner viewpoint band at
# 0.86–0.93, where this instrument genuinely cannot tell a re-take from a
# different photograph of the same trip — both folders put true and false
# pairs at the same cosine there, so no line in that band is honest.
#
# Rerun the calibration before moving it: `tools/tune_dupes.py` for the hash
# side, and a labelled folder (Holidays has groundtruth.json) for this one.
SIMILAR_COSINE = 0.94

# CLIP ViT-B/32's image encoder, int8 ONNX, baked into web_image at build time
# from a pinned revision with its checksum asserted — see the layer in
# `web_image`. Baked rather than downloaded under the gear because the
# duplicate review must work on a fresh deploy with nothing chosen yet: 85 MB
# is a dependency, not a checkpoint anybody picks. On a dev machine running
# the tools the file is simply absent and `_embedder()` answers None — the
# similar class then falls back to the hash band rather than disappearing.
EMB_MODEL = "/opt/similar_clip_b32_q8.onnx"
_EMB_MEAN = (0.48145466, 0.4578275, 0.40821073)
_EMB_STD = (0.26862954, 0.26130258, 0.27577711)
# One-slot cache: "session" appears after the first attempt, holding the ONNX
# session or None. A dict rather than a global-with-lock so the tools can pull
# this without seeding `threading`; the benign race is two threads building
# one session each and the second assignment winning.
_EMB_STATE: dict[str, Any] = {}

FINGERPRINT_FILE = "fingerprints.json"
# How long one scan request will spend measuring before it answers with what it
# has and asks to be called again.
#
# There is deliberately no job record, no spawn and no second route behind this.
# The measurement is already cached per file, so the cache *is* the progress
# state: a request measures what it can, writes what it measured, and reports
# how many are left. The next request picks up exactly where it stopped, and a
# container that dies mid-scan costs the images it had in hand rather than the
# folder. A job id would be a parallel contract for something the existing
# route can already resume.
#
# Ten seconds because a scan measures ~31 images a second on this image, so the
# common case — a training set of twenty to eighty — finishes in the first
# request and never sees the polling path at all. A 731-image source folder
# takes three.
SCAN_BUDGET_S = 10.0
# The cache is rewritten whole when this moves, rather than being migrated: it
# is a decode cache, and a rescan is the only thing a wrong one costs.
# 5: the decoder itself changed — Pillow 11.3 reads the AVIFs 11.1 could not,
# so every entry (and every cached failure) measured under 11.1 is stale.
# 6: fingerprints grew the CLIP embedding the similar class now reads.
FINGERPRINT_VERSION = 6
# Which encoding to prefer when two files are the same picture at the same size
# and the same weight. Lossless first: a PNG re-saved as JPEG can be recovered
# from neither direction, and the tie only ever arises between an original and
# a re-export of it.
_FORMAT_RANK = {"PNG": 0, "WEBP": 1, "AVIF": 2, "BMP": 3, "JPEG": 4}


def _dhash(im) -> int:
    """
    64 bits of horizontal gradient: is this pixel brighter than the one to its
    right, over a 9x8 grayscale.

    A gradient rather than a mean (aHash) because a re-export that shifts
    exposure, gamma or a colour profile moves every pixel the same way and
    leaves every *comparison between neighbours* alone. That is the whole
    reason two exports of one frame at different JPEG qualities land on the
    same hash while two different photographs of the same room do not.

    Takes an already-upright image: orientation is resolved on arrival, and a
    hash computed off the stored pixels would file a phone photo and its
    rotated copy as two different pictures.
    """
    from PIL import Image

    small = im.convert("L").resize((9, 8), Image.LANCZOS)
    px = small.load()
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | int(px[x, y] > px[x + 1, y])
    return bits


def _phash(im) -> int:
    """
    64 bits of low-frequency DCT energy, thresholded at its own median.

    Here to disagree with dHash rather than to outvote it. The two read
    different things — edges against coarse structure — so a pair that satisfies
    both is a pair two independent measurements agree about, and that agreement
    is what lets the thresholds sit far enough out to catch a re-grade without
    reaching a different photograph.

    The DC term is dropped before the median, because it is the average
    brightness of the whole frame: leaving it in makes the hash of a picture
    depend on how the picture was exposed, which is exactly the transform this
    is supposed to survive.
    """
    from PIL import Image

    small = im.convert("L").resize((32, 32), Image.LANCZOS)
    px = list(small.getdata())
    # Separable DCT-II: the row transform is computed once per (v, y) rather
    # than once per (v, u, y), which is the difference between ~2k and ~67k
    # cosine calls per hash.
    cos = [[math.cos((2 * i + 1) * k * math.pi / 64) for i in range(32)] for k in range(8)]
    rows = [[sum(px[y * 32 + x] * cos[u][x] for x in range(32)) for u in range(8)]
            for y in range(32)]
    coeffs = [sum(rows[y][u] * cos[v][y] for y in range(32))
              for v in range(8) for u in range(8)]
    median = sorted(coeffs[1:])[31]
    bits = 0
    for value in coeffs[1:]:
        bits = (bits << 1) | int(value > median)
    return bits


def _sharpness(im) -> float:
    """
    Mean absolute neighbour difference at a fixed 64x64.

    A tie-breaker between two copies of one picture and nothing more. At a fixed
    analysis size it answers "which of these two survived less blur and less
    smoothing", which is the question a re-encode raises; it is not a judgement
    about the photograph, and it is never compared across different pictures.
    """
    from PIL import Image

    small = im.convert("L").resize((64, 64), Image.LANCZOS)
    px = small.load()
    total = 0
    for y in range(63):
        for x in range(63):
            total += abs(px[x, y] - px[x + 1, y]) + abs(px[x, y] - px[x, y + 1])
    return round(total / (63 * 63 * 2), 2)


def _crop_variants(im) -> list:
    """The full frame first, then one centre crop per CROP_SHARES entry."""
    out = [im]
    w, h = im.size
    for share in CROP_SHARES:
        cw, ch = int(w * share), int(h * share)
        if cw >= 16 and ch >= 16:
            left, top = (w - cw) // 2, (h - ch) // 2
            out.append(im.crop((left, top, left + cw, top + ch)))
    return out


def _embedder():
    """
    The ONNX session behind the similar class, or None where the model is not.

    None is a mode, not an error: the tools run on machines whose python has no
    onnxruntime and whose disk has no model, and they still have to classify —
    they just fall back to the hash band for the similar class. The deployed
    image always has both, so a None *there* is worth the printed line.
    """
    if "session" not in _EMB_STATE:
        sess = None
        try:
            import onnxruntime

            if Path(EMB_MODEL).is_file():
                sess = onnxruntime.InferenceSession(
                    EMB_MODEL, providers=["CPUExecutionProvider"])
            else:
                print(f"[dupes] no similarity model at {EMB_MODEL}; "
                      "the similar class falls back to the hash band", flush=True)
        except Exception as exc:
            print(f"[dupes] similarity model unavailable ({exc}); "
                  "the similar class falls back to the hash band", flush=True)
        _EMB_STATE["session"] = sess
    return _EMB_STATE["session"]


def _embed(up) -> str | None:
    """
    A unit-normalised CLIP image embedding, quantised to int8 and base64ed.

    Quantised because the fingerprint cache is JSON read and rewritten whole
    per scan request: 512 floats print at ~4 KB an image where the int8 bytes
    are 684 characters, and at 1/127 per dimension the quantisation moves a
    cosine by less than a thousandth — nothing against a threshold of 0.90.

    Takes the already-upright image `_fingerprint` is holding open, so the
    embedding can never disagree with the hashes about which way is up.
    """
    sess = _embedder()
    if sess is None:
        return None
    try:
        import numpy as np
        from PIL import Image

        im = up.convert("RGB")
        w, h = im.size
        s = 224 / min(w, h)
        im = im.resize((max(224, round(w * s)), max(224, round(h * s))),
                       Image.BICUBIC)
        w, h = im.size
        left, top = (w - 224) // 2, (h - 224) // 2
        x = np.asarray(im.crop((left, top, left + 224, top + 224)),
                       dtype=np.float32) / 255.0
        # float32 arrays, not the bare tuples: numpy promotes float32 minus a
        # python tuple to float64, and the session refuses a double tensor.
        x = ((x - np.asarray(_EMB_MEAN, np.float32))
             / np.asarray(_EMB_STD, np.float32))
        v = sess.run(None, {"pixel_values": x.transpose(2, 0, 1)[None]})[0][0]
        v = v / (float(np.linalg.norm(v)) or 1.0)
        q = np.clip(np.round(v * 127.0), -127, 127).astype(np.int8)
        return base64.b64encode(q.tobytes()).decode("ascii")
    except Exception as exc:
        # An image the hashes could measure still fingerprints without its
        # embedding; the grouping falls back for the whole set and says so.
        print(f"[dupes] embedding failed: {exc}", flush=True)
        return None


def _emb_vec(rec: dict[str, Any]):
    """The stored embedding back as a unit numpy vector, or None."""
    b = rec.get("emb")
    if not b:
        return None
    try:
        import numpy as np

        q = np.frombuffer(base64.b64decode(b), dtype=np.int8).astype(np.float32)
        n = float(np.linalg.norm(q))
        return q / n if n else None
    except Exception:
        return None


def _fingerprint(img: Path) -> dict[str, Any] | None:
    """One image measured: its hashes, its bytes, its pixels, its encoding."""
    from PIL import Image

    try:
        st = img.stat()
        digest = hashlib.sha256()
        with img.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        with Image.open(img) as raw:
            fmt = (raw.format or img.suffix.lstrip(".")).upper()
            up = _upright(raw)
            w, h = up.size
            # Index 0 is the full frame, which is what every direct comparison
            # and every distance the page shows is measured from.
            variants = [[_dhash(v), _phash(v)] for v in _crop_variants(up)]
            rec = {"dhash": variants[0][0], "phash": variants[0][1],
                   "variants": variants, "sharpness": _sharpness(up)}
            emb = _embed(up)
            if emb:
                rec["emb"] = emb
    except Exception:
        # A file PIL cannot open is not a reason to fail the scan — but it is
        # never a normal state either: every extension the upload accepts is an
        # extension this image's Pillow decodes, so a failure here means the
        # two have drifted apart (AVIF sat in that gap for months, invisible to
        # every group it belonged in). The caller caches the failure so it is
        # not re-decoded every scan, counts it, and the count is reported.
        return None
    return {**rec, "sha": digest.hexdigest(), "bytes": st.st_size,
            "width": w, "height": h, "format": fmt, "mtime": st.st_mtime,
            "v": FINGERPRINT_VERSION, "stamp": [st.st_mtime_ns, st.st_size]}


def _fingerprints(d: Path, budget_s: float | None = None) -> tuple[dict, bool, int, list[str]]:
    """
    Every image in the set, measured once and cached beside the thumbnails.

    Cached for the reason thumbnails are: a scan decodes every image in the
    folder, and a two-hundred-image set of 12 MP phone photos is minutes of CPU
    that must not be paid again for looking at the second group. Keyed on
    `(mtime_ns, size)` rather than mtime alone, because replacing an image with
    another of the same age is exactly what an overwriting re-upload does, and
    on a version, because an entry measured by an older classifier is not a
    cheaper answer to today's question — it is a wrong one.

    Returns the map, whether anything was written, and how many images it did
    not reach — so the caller commits the volume once for a scan rather than
    once per file, and can answer "still measuring" instead of holding a request
    open for a minute.

    `budget_s` bounds the measuring, not the reading: everything already cached
    is collected however long the folder is, because that costs a dict lookup.
    Only the decodes are on the clock.
    """
    images = _dataset_images(d)
    cache_path = _set_cache(d) / FINGERPRINT_FILE
    cached: dict[str, Any] = {}
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            cached = {}

    out: dict[str, dict[str, Any]] = {}
    changed = False
    pending = 0
    measured = 0
    unreadable: list[str] = []
    deadline = None if budget_s is None else time.monotonic() + budget_s
    for img in images:
        try:
            st = img.stat()
        except OSError:
            continue
        was = cached.get(img.name)
        if (was and was.get("stamp") == [st.st_mtime_ns, st.st_size]
                and was.get("v") == FINGERPRINT_VERSION):
            out[img.name] = was
            if was.get("failed"):
                unreadable.append(img.name)
            continue
        # Out of time: leave it out of `out` entirely rather than half-measured.
        # It is absent from the cache too, which is exactly what makes the next
        # request pick it up — there is no separate cursor to keep in step.
        #
        # `measured` is what makes the loop terminate rather than merely slow
        # down. A budget check on its own is a check that can be true before the
        # first decode — at a zero budget it is true immediately, and then every
        # request skips every image, writes nothing and asks to be called again
        # forever. `smoke_dupes.py` drives exactly that case. In production the
        # same stall arrives as one image slower than the whole budget, which is
        # a 200 MP scan or a volume having a bad minute, and it would have hung
        # the panel on one file with no way to tell which.
        if measured and deadline is not None and time.monotonic() > deadline:
            pending += 1
            continue
        fp = _fingerprint(img)
        measured += 1
        changed = True
        if fp:
            out[img.name] = fp
        else:
            # Cached too, or a file that cannot be decoded is re-decoded on
            # every scan forever. The marker carries the stamp so a replaced
            # file is re-tried, and the version so a decoder upgrade (the AVIF
            # case: Pillow 11.1 → 11.3) re-tries everything it used to fail.
            out[img.name] = {"failed": True, "v": FINGERPRINT_VERSION,
                             "stamp": [st.st_mtime_ns, st.st_size]}
            unreadable.append(img.name)
    # A deleted image leaves its entry behind, which is why the rewrite is of
    # `out` rather than of `cached | out`: the cache must not grow forever on a
    # folder that is edited all day.
    if changed or len(cached) != len(out):
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(out))
            changed = True
        except OSError:
            pass
    good = {k: v for k, v in out.items() if not v.get("failed")}
    return good, changed, pending, unreadable


def _keep_rank(fp: dict[str, Any], name: str, captioned: bool) -> tuple:
    """
    Which of two copies of one picture is the one to keep.

    Pixels first, because resolution is the only axis on this list the trainer
    reads directly — a 2048px original and its 512px re-post are one picture and
    only one of them is worth bucketing. Then the lossless encoding, then
    sharpness, then weight: three ways of asking the same question, which is how
    much of the original survived. A caption last, because it is the only entry
    here that is about your work rather than the file, and it is cheap to move.

    Never applied across different pictures — see `_sharpness`.
    """
    return (
        -(fp["width"] * fp["height"]),
        _FORMAT_RANK.get(fp["format"], 9),
        -fp.get("sharpness", 0),
        -fp["bytes"],
        0 if captioned else 1,
        name,
    )


def _keep_reason(best: dict[str, Any], runner: dict[str, Any]) -> str:
    """
    The axis that decided, named — and named against the runner-up rather than
    against the whole group, because "nothing separates the top two" is the one
    statement that tells you your choice does not matter.

    Derived, so it is visible: a suggestion you have to re-derive by hand before
    you can trust it costs more than making the choice yourself.
    """
    if best["width"] * best["height"] != runner["width"] * runner["height"]:
        return f"most pixels · {best['megapixels']} MP"
    if best["format"] != runner["format"]:
        return f"{best['format']} over {runner['format']}"
    if best.get("sharpness", 0) != runner.get("sharpness", 0):
        return "sharpest copy at this resolution"
    if best["bytes"] != runner["bytes"]:
        return "same size, least compressed"
    if bool(best["caption"]) != bool(runner["caption"]):
        return "the only one captioned"
    return "identical in every respect — first by name"


def _link(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """
    What relates two images, if anything.

    Returns `kind` of "duplicate", "similar" or "", plus the evidence the page
    shows: both distances, and the transforms that would explain them. Every
    accept is an AND over the two hashes — see DUPLICATE_MATCH.
    """
    if a["sha"] == b["sha"]:
        return {"kind": "duplicate", "dhash": 0, "phash": 0, "same_file": True,
                "transforms": ["byte-for-byte identical"]}

    dd = (a["dhash"] ^ b["dhash"]).bit_count()
    dp = (a["phash"] ^ b["phash"]).bit_count()
    transforms = []
    if (a["width"], a["height"]) != (b["width"], b["height"]):
        transforms.append("resized")
    if a["format"] != b["format"]:
        transforms.append("reformatted")
    elif a["bytes"] != b["bytes"]:
        transforms.append("recompressed")

    if dd <= DUPLICATE_MATCH["dhash"] and dp <= DUPLICATE_MATCH["phash"]:
        kind = "duplicate"
    elif dd <= SIMILAR_MATCH["dhash"] and dp <= SIMILAR_MATCH["phash"]:
        kind = "similar"
    else:
        # The crop pass, on its leash. Every variant against every variant
        # except full-frame-to-full-frame, which is the comparison just made.
        hit = min(((x[0] ^ y[0]).bit_count(), (x[1] ^ y[1]).bit_count())
                  for i, x in enumerate(a["variants"])
                  for j, y in enumerate(b["variants"]) if i or j)
        if hit[0] <= CROP_MATCH["dhash"] and hit[1] <= CROP_MATCH["phash"]:
            return {"kind": "similar", "dhash": dd, "phash": dp, "same_file": False,
                    "transforms": [*transforms, "cropped"],
                    "crop_dhash": hit[0], "crop_phash": hit[1]}
        return {"kind": "", "dhash": dd, "phash": dp, "same_file": False,
                "transforms": transforms}
    return {"kind": kind, "dhash": dd, "phash": dp, "same_file": False,
            "transforms": transforms}


def _components(names: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    """Connected components, so an image is decided about once rather than once
    per pair it happens to resemble."""
    parent = {n: n for n in names}

    def find(n: str) -> str:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    out: dict[str, list[str]] = {}
    for n in names:
        out.setdefault(find(n), []).append(n)
    return [sorted(m) for m in out.values() if len(m) > 1]


def _duplicate_groups(d: Path, budget_s: float | None = None) -> dict[str, Any]:
    """
    The set's duplicate groups, then its similar ones.

    **Pairwise, over distinct fingerprints.** A BK-tree was here and was
    measured: at the radius this classifier uses it visits 96% of the tree per
    lookup, so it is the same sweep with a tree walk's overhead on top. What
    inverts the cost is deduplicating the hashes first — the pathological input
    for a pairwise scan, one picture present four hundred times, collapses to a
    single fingerprint and one comparison — and what makes a *rescan* free is
    the fingerprint cache. Neither of those is an index.

    **An image is in at most one group, and duplicates win.** Similar links are
    computed only between images no duplicate group already holds. A similar
    relationship between a copy and an outsider is therefore not shown until the
    copies are dealt with — which is the order the work happens in anyway: clear
    the duplicates, rescan, review what is merely alike. The invariant it buys
    is worth more than the edge it drops, because a name in two groups is a name
    you are asked about twice and can mark for deletion twice.
    """
    prints, wrote, pending, unreadable = _fingerprints(d, budget_s)
    if pending:
        # Half a folder groups into half the truth, and half the truth here is a
        # keeper suggested against copies that have not been looked at yet. So
        # nothing is grouped until everything is measured; what comes back is
        # the count, which is the only honest thing to draw.
        return {"scanning": True, "measured": len(prints) + len(unreadable),
                "total": len(prints) + len(unreadable) + pending,
                "groups": [], "images": len(prints),
                "unreadable": unreadable,
                "thresholds": {"duplicate": DUPLICATE_MATCH, "similar": SIMILAR_MATCH,
                               "similar_cosine": SIMILAR_COSINE, "crop": CROP_MATCH},
                "summary": {"duplicate_groups": 0, "duplicate_images": 0,
                            "similar_groups": 0, "similar_images": 0},
                "reclaim": 0, "_wrote": wrote}

    # Distinct fingerprints, so identical files are compared once. The sha is in
    # the key because two different pictures cannot share both hashes but two
    # copies of one file must stay distinguishable as *the same file*.
    by_print: dict[tuple, list[str]] = {}
    for name, fp in prints.items():
        by_print.setdefault(
            (tuple(tuple(v) for v in fp["variants"]), fp["sha"]), []).append(name)
    keys = list(by_print)

    # The similar class reads the embeddings when every image has one — all or
    # nothing, because a set half-measured by each instrument would group into
    # two different definitions of "alike". A build without the model writes no
    # embeddings, and the class falls back to the hash band it used to be.
    reps = [by_print[k][0] for k in keys]
    vecs = [_emb_vec(prints[r]) for r in reps]
    cosmat = None
    if vecs and all(v is not None for v in vecs):
        import numpy as np

        cosmat = np.stack(vecs) @ np.stack(vecs).T
    elif any(v is not None for v in vecs):
        missing = sum(1 for v in vecs if v is None)
        print(f"[dupes] {missing} of {len(vecs)} fingerprints lack embeddings; "
              "the similar class falls back to the hash band", flush=True)
    emb_of = ({n: v for k, v in zip(keys, vecs) for n in by_print[k]}
              if cosmat is not None else {})

    dup_edges: list[tuple[str, str]] = []
    sim_edges: list[tuple[str, str]] = []
    evidence: dict[tuple[str, str], dict[str, Any]] = {}
    for i in range(len(keys)):
        # Files sharing a fingerprint are the same picture by definition.
        same = by_print[keys[i]]
        for other in same[1:]:
            dup_edges.append((same[0], other))
        for j in range(i + 1, len(keys)):
            a, b = by_print[keys[i]][0], by_print[keys[j]][0]
            link = _link(prints[a], prints[b])
            if cosmat is not None and link["kind"] != "duplicate":
                # The embedding decides similar; the crop pass keeps its say
                # because a hard crop moves the global embedding while the
                # centre-crop hashes still land it. The hash similar band is
                # deliberately *not* consulted here — it is the line calibrated
                # on a set with no true near-duplicates in it.
                cos = float(cosmat[i, j])
                link["cosine"] = round(cos, 3)
                link["kind"] = ("similar"
                                if cos >= SIMILAR_COSINE
                                or "cropped" in link["transforms"] else "")
            if not link["kind"]:
                continue
            evidence[(a, b)] = link
            (dup_edges if link["kind"] == "duplicate" else sim_edges).append((a, b))

    names = list(prints)
    dup_groups = _components(names, dup_edges)
    held = {n for g in dup_groups for n in g}
    # Similar is computed over what is left, which is what keeps every image in
    # at most one group.
    sim_groups = _components([n for n in names if n not in held],
                             [(a, b) for a, b in sim_edges
                              if a not in held and b not in held])

    def build(members: list[str], kind: str) -> dict[str, Any]:
        rows = []
        for name in members:
            fp = prints[name]
            rows.append({
                "name": name, "caption": _caption_of(d / name), "bytes": fp["bytes"],
                "width": fp["width"], "height": fp["height"],
                # Rounded here rather than on the page: it is the one figure in
                # this row that is computed rather than read, and two consumers
                # rounding it differently is two answers to "which is bigger".
                "megapixels": round(fp["width"] * fp["height"] / 1e6, 1),
                "format": fp["format"], "sharpness": fp.get("sharpness", 0),
                "mtime": fp["mtime"],
            })
        rows.sort(key=lambda r: _keep_rank(prints[r["name"]], r["name"], bool(r["caption"])))
        keeper = rows[0]
        for r in rows:
            link = (_link(prints[keeper["name"]], prints[r["name"]])
                    if r["name"] != keeper["name"] else None)
            # The number that actually accepted an embedding-linked pair, so
            # the Match row can quote it instead of quoting hash distances
            # that rejected the pair — the crop rule's own lesson.
            cos = None
            if link is not None and emb_of:
                va, vb = emb_of.get(keeper["name"]), emb_of.get(r["name"])
                if va is not None and vb is not None:
                    cos = round(float(va @ vb), 3)
            r.update(
                cosine=cos,
                dhash_distance=link["dhash"] if link else 0,
                phash_distance=link["phash"] if link else 0,
                same_file=bool(link and link["same_file"]),
                transforms=link["transforms"] if link else [],
                # Only a crop match has these, and it is the one case where the
                # direct distance shown beside it is *outside* the threshold
                # that accepted the pair — because what accepted it was the
                # crop comparison. Reporting the first number without the
                # second is reporting a contradiction.
                crop_dhash=link.get("crop_dhash") if link else None,
                crop_phash=link.get("crop_phash") if link else None,
            )
        return {
            # Stable across a rescan so the page's per-group state survives one:
            # the alphabetically first member, which only moves when that member
            # is deleted and the group is therefore a different group.
            "key": members[0],
            "kind": kind,
            # Only a duplicate group preselects. A similar group is evidence, so
            # it arrives with everything kept and nothing to undo.
            "suggest": keeper["name"] if kind == "duplicate" else "",
            "why": _keep_reason(keeper, rows[1]) if kind == "duplicate" else "",
            "images": rows,
        }

    groups = ([build(g, "duplicate") for g in dup_groups]
              + [build(g, "similar") for g in sim_groups])
    # Duplicates first and the biggest first inside each class: a group of six
    # copies is the one press worth the most, and a pair of burst frames is a
    # judgement you should reach after the decided work is done.
    groups.sort(key=lambda g: (g["kind"] != "duplicate", -len(g["images"]), g["key"]))
    dupes = [g for g in groups if g["kind"] == "duplicate"]
    return {
        "scanning": False,
        "images": len(prints),
        # Should always be empty: everything the upload accepts, the scan
        # decodes. Reported by name rather than swallowed, because a file the
        # scan cannot see is excluded from every group it belongs in — which
        # reads as "the scan missed obvious duplicates", not as a decode fault.
        "unreadable": unreadable,
        "groups": groups,
        "thresholds": {"duplicate": DUPLICATE_MATCH, "similar": SIMILAR_MATCH,
                       "similar_cosine": SIMILAR_COSINE, "crop": CROP_MATCH},
        "summary": {
            "duplicate_groups": len(dupes),
            "duplicate_images": sum(len(g["images"]) for g in dupes),
            "similar_groups": len(groups) - len(dupes),
            "similar_images": sum(len(g["images"]) for g in groups if g["kind"] != "duplicate"),
        },
        # What accepting every suggestion would reclaim. Duplicates only —
        # nothing in a similar group is marked, so nothing in one is counted.
        "reclaim": sum(r["bytes"] for g in dupes for r in g["images"]
                       if r["name"] != g["suggest"]),
        "_wrote": wrote,
    }


_RUNNERS: "dict[tuple[str, str], Any]" = {}
_RUNNERS_LOCK = threading.Lock()


def _on_gpu(cls: Any, requested: Any, allowed: tuple[str, ...], default: str) -> Any:
    """
    Resolve a UI GPU choice to a runner to spawn from.

    Uses the base class when the choice is the default, so the ordinary request
    keeps hitting the container that is already warm; only a genuine switch pays
    for a variant. An unknown card falls back to the default rather than raising
    — a stale tab asking for a card that has been removed from the list should
    still get its picture.

    **Returns an instance, memoized per (class, card), and that is load-bearing
    rather than tidy.** A Cls method's `.local()` binds to the Obj's own user
    instance and runs `@modal.enter()` first, once per Obj — guarded on the
    instance, not on the class. The call sites used to construct a fresh Obj
    inline for every request, which remotely is free and locally means a fresh
    `_Comfy`: starting ComfyUI and reloading 35-42.5 GB of checkpoint for every
    single render, with the warm process the whole design rests on never being
    reached. Remotely the memo saves a hydration per request and changes nothing
    else — which is still a change to the deployed path, so it is on the
    rented-box list as deploy, render, redeploy, render.
    """
    gpu = str(requested or default)
    if gpu not in allowed:
        gpu = default
    key = (getattr(cls, "__name__", None) or str(cls), gpu)
    with _RUNNERS_LOCK:
        if key not in _RUNNERS:
            base = cls if gpu == default else cls.with_options(gpu=gpu)
            _RUNNERS[key] = base()
        return _RUNNERS[key]


def _host(family: str, payload: dict[str, Any]) -> Any:
    """
    The class a request lands on: its family's own, or the one-container
    class when the gear says so.

    `container` rides every request rather than living in server state so a
    tab that switched the mode does not change what another tab's next press
    runs on — the same reason `gpu` travels with the body. An absent or
    unknown value is the two-container default, because a stale tab must still
    get its picture.
    """
    if str(payload.get("container") or "") == "one":
        return _on_gpu(BothGenerator, payload.get("gpu"), BOTH_GPUS, BOTH_GPU)
    if family == "video":
        return _on_gpu(VideoGenerator, payload.get("gpu"), VIDEO_GPUS, VIDEO_GPU)
    return _on_gpu(ImageGenerator, payload.get("gpu"), IMAGE_GPUS, GPU)


MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp",
               # The dataset clip route serves whatever a set holds, and a set
               # is whatever was dropped on it — so the containers a browser
               # will actually play are spelled out rather than left to fall
               # back on video/mp4, which is a `<video>` that silently paints
               # nothing for a .mov it could otherwise decode.
               ".mp4": "video/mp4", ".mov": "video/quicktime",
               ".webm": "video/webm", ".mkv": "video/x-matroska",
               ".m4v": "video/x-m4v"}
OUTPUT_FILE_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}\.(png|jpg|webp|mp4)$")
# The run record's key, inside the file: a PNG `tEXt` chunk and an MP4
# metadata key by the same name. See `_output_record`.
RECORD_KEY = "visionary"
# What the job folders used to carry beside their files. Read by the
# migration only; nothing writes it any more.
LEGACY_META = "visionary.json"


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
        # Which plates the compose ran around. The mode already says krea2edit;
        # without this line an outfit render and a scene render carried the
        # same sheet, and neither said what the whole frame was regenerated
        # against. Older records have no `plates` — the get() is the tolerance.
        plates = report.get("plates") or []
        if plates:
            add("Plates", ", ".join(plates))
            # The two numbers a compose ran at. Same tolerance as `plates`:
            # records from before them simply omit the fields.
            add("Edit strength", report.get("edit_strength"))
            add("Compose seed", report.get("compose_seed"))
    # Outside the regions block — style is the whole-frame engine and runs
    # with no boxes at all.
    if report.get("style_refs"):
        add("Style refs", report["style_refs"])
        add("Style strength", report.get("style_strength"))
        # The caption goes in as a field rather than as the infotext's first
        # line, because that line is the prompt and A1111's format says so —
        # a reader pasting this back expects to get the sentence they wrote,
        # not the machine's expansion of it.
        add("Caption", report.get("caption"))

    lines.append(", ".join(pairs))
    return "\n".join(lines)


def _output_record(**fields: Any) -> dict[str, Any]:
    """
    The run's record — what you would want a week later: the prompt, the
    seed, the model, the pills, the boxes. It goes *inside* the file it
    describes, never beside it.

    It was a `visionary.json` beside the files in a folder per job, and that
    folder existed only to keep the record beside the batch. A file dragged out
    of the browser then left its record behind, which is the sentence-is-the-
    record thesis backwards; and the folder-per-job was the shape every cache
    and address in the app had grown around. The job Dict is still live state,
    not a record: it is polled during a run and means nothing afterwards.

    Never bytes. Every writer already keeps photographs out of the record —
    counts and booleans stand in for them — because this dict is also what the
    status route polls every 400 ms.
    """
    return dict(fields)


def _record_json(fields: dict[str, Any]) -> str:
    """ASCII on purpose: a PNG `tEXt` chunk is Latin-1, and PIL silently
    switches to an `iTXt` chunk for anything else — a second shape for the
    reader to know. Escaped JSON is the same record in one shape."""
    return json.dumps(fields, separators=(",", ":"))


def _png_with_record(src: Path, dst: Path, fields: dict[str, Any],
                     extra: dict[str, str] | None = None) -> None:
    """Re-encode `src` to `dst` with the record in a `tEXt` chunk, plus any
    other text chunks a writer wants beside it (A1111's `parameters`, ComfyUI's
    `prompt`). Text chunks land before the image data, which is what lets a
    reader take the record off the file's head."""
    from PIL import Image, PngImagePlugin

    with Image.open(src) as im:
        info = PngImagePlugin.PngInfo()
        for k, v in (extra or {}).items():
            info.add_text(k, v)
        info.add_text(RECORD_KEY, _record_json(fields))
        im.save(dst, pnginfo=info)


def _mp4_with_record(src: Path, dst: Path, fields: dict[str, Any],
                     extra: dict[str, str] | None = None) -> bool:
    """
    Copy `src` to `dst` with the record as an MP4 metadata key, streams
    untouched.

    `use_metadata_tags` is load-bearing: without it ffmpeg writes only the
    handful of tags it knows (`comment`, `title`) and silently drops ours.
    `faststart` puts `moov` — and the record inside it — at the head of the
    file, so a reader never has to pull a clip to learn what it is. Needs
    ffmpeg, which every container that writes a clip has; the web container
    does not, and never writes one.
    """
    cmd = ["ffmpeg", "-y", "-i", str(src), "-c", "copy",
           "-movflags", "use_metadata_tags+faststart"]
    for k, v in (extra or {}).items():
        cmd += ["-metadata", f"{k}={v}"]
    cmd += ["-metadata", f"{RECORD_KEY}={_record_json(fields)}", str(dst)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        tail = (done.stderr or "").strip().splitlines()[-1:]
        print(f"[record] {dst.name}: ffmpeg could not embed the record "
              f"({' '.join(tail)}) — copied plain", flush=True)
        shutil.copyfile(src, dst)
        return False
    return True


def _record_from_png(head: bytes) -> dict[str, Any] | None:
    """The record chunk out of a PNG's head. None when the head is too short
    to say; `{}` when the file has no record (the image data began)."""
    if not head.startswith(b"\x89PNG\r\n\x1a\n"):
        return {} if len(head) >= 8 else None
    pos = 8
    while pos + 8 <= len(head):
        length, ctype = struct.unpack(">I4s", head[pos:pos + 8])
        if ctype in (b"IDAT", b"IEND"):
            return {}
        body = head[pos + 8:pos + 8 + length]
        if len(body) < length:
            return None
        if ctype in (b"tEXt", b"iTXt"):
            key, _, val = body.partition(b"\x00")
            if key == RECORD_KEY.encode():
                if ctype == b"iTXt":
                    # flag, method, language\0, translated keyword\0, text.
                    # Read for a file written by some other tool; ours is
                    # always ASCII and lands in tEXt.
                    flag = val[0:1]
                    rest = val[2:].split(b"\x00", 2)
                    val = rest[2] if len(rest) == 3 else b""
                    if flag == b"\x01":
                        import zlib
                        val = zlib.decompress(val)
                try:
                    return json.loads(val.decode("utf-8", "replace"))
                except ValueError:
                    return {}
        pos += 8 + length + 4
    return None


def _mp4_atoms(buf: bytes, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        size, atype = struct.unpack(">I4s", buf[pos:pos + 8])
        hdr = 8
        if size == 1:
            if pos + 16 > end:
                return
            size = struct.unpack(">Q", buf[pos + 8:pos + 16])[0]
            hdr = 16
        elif size == 0:
            size = end - pos
        if size < hdr or pos + size > end:
            return
        yield atype, pos + hdr, pos + size
        pos += size


def _record_from_mp4(head: bytes) -> dict[str, Any] | None:
    """
    The record key out of `moov/udta/meta`, in ffmpeg's `use_metadata_tags`
    shape — a `keys` atom naming the entries, an `ilst` indexed from one —
    with the plain four-character `ilst` form accepted too. None until a
    whole `moov` is in the head; `{}` for a clip that has none.
    """
    for atype, a, b in _mp4_atoms(head, 0, len(head)):
        if atype == b"mdat":
            return {}  # data before moov: no record on the head
        if atype != b"moov":
            continue
        keys: list[bytes] = []
        items: dict[bytes, bytes] = {}
        for t, c, d in _mp4_atoms(head, a, b):
            if t != b"udta":
                continue
            for u, e, f in _mp4_atoms(head, c, d):
                if u != b"meta":
                    continue
                for v, g, h in _mp4_atoms(head, e + 4, f):
                    if v == b"keys":
                        n = struct.unpack(">I", head[g + 4:g + 8])[0]
                        p = g + 8
                        for _ in range(n):
                            ksize = struct.unpack(">I", head[p:p + 4])[0]
                            keys.append(head[p + 8:p + ksize])
                            p += ksize
                    elif v == b"ilst":
                        for w, i, j in _mp4_atoms(head, g, h):
                            for x, k, m in _mp4_atoms(head, i, j):
                                if x == b"data":
                                    items[w] = head[k + 8:m]
        for w, val in items.items():
            if w[0] == 0 and keys:
                idx = struct.unpack(">I", w)[0] - 1
                name = keys[idx] if 0 <= idx < len(keys) else b""
            else:
                name = w
            if name.split(b".")[-1] == RECORD_KEY.encode():
                try:
                    return json.loads(val.decode("utf-8"))
                except ValueError:
                    return {}
        return {}
    return None


# Past this much of a file with no record found, the file has none. A PNG's
# text chunks and a faststart clip's `moov` both sit in the first few KB.
_RECORD_HEAD_MAX = 4 << 20


def _read_record(rel: str) -> dict[str, Any]:
    """The record out of one committed file's head, by RPC — chunks pulled
    only until the record can be read, never the whole render."""
    parse = _record_from_png if rel.lower().endswith(".png") else _record_from_mp4
    head = b""
    try:
        for chunk in _read_committed(rel):
            head += chunk
            got = parse(head)
            if got is not None:
                return got
            if len(head) > _RECORD_HEAD_MAX:
                break
    except Exception as exc:  # noqa: BLE001 — a record is best-effort
        if not isinstance(exc, FileNotFoundError):
            print(f"[record] {rel}: {type(exc).__name__}: {exc}", flush=True)
        return {}
    return parse(head) or {}


def _group_of(name: str) -> str:
    """Which run a file belongs to, read off its name: `{job}_{NN}.png` and
    `{job}.mp4` both group under `{job}`. The batch is a property of the name,
    which is what let the folder go."""
    stem = name.split(".")[0]
    return re.sub(r"_\d{2}$", "", stem)


def _keep_entry(name: str) -> bool:
    """
    Is `outputs/{name}` a result, rather than something beside one?

    Shared by both listing sources so they cannot disagree about what a
    gallery item is. A motion-context tensor sits beside its clip and is not
    a result; a dotfile never is.
    """
    return not name.startswith(".") and Path(name).suffix.lower() in MEDIA_TYPES


# ── results are served off the container, and the volume is the record ─────
#
# Every picture bug the gallery ever had traced back to one dependency:
# serving bytes off the mount makes the page's freshness hang on
# `volume.reload()`, and reload is refusable — by our own `FileResponse`
# descriptors most of all, so painting pictures froze the view that the next
# picture needed. The listing broke that dependency by asking Modal instead of
# the mount (`_entries_by_rpc`); these helpers do the same for the bytes.
# A file is pulled ONCE from the volume's committed state by RPC — which both
# job writers commit immediately, and which no open descriptor can refuse —
# onto the container's own disk, and served from there. Descriptors during a
# transfer are then held on local disk, never on /workspace, so serving
# pictures can no longer freeze anything. The mount stays for what it is:
# writes, deletes, and the durable record.
SPOOL = Path(os.environ.get("VISIONARY_SPOOL", "/tmp/visionary-spool"))
# Local disk, not local memory, and capped: a busy video session is gigabytes.
# Trimmed LRU so the spool follows the session around rather than growing with
# the volume — which is the entire difference between it and the mount.
SPOOL_MAX_BYTES = 4 << 30
_SPOOL_TRIM_LOCK = threading.Lock()


def _read_committed(rel: str):
    """Chunks of one committed file by RPC. `listdir` answered volume-relative
    paths for a rooted query in testing, so both spellings are tried here too —
    one line, and it survives either convention."""
    try:
        yield from volume.read_file(rel)
        return
    except FileNotFoundError:
        pass
    yield from volume.read_file("/" + rel)


def _volume_bytes(rel: str) -> bytes | None:
    """One small committed file in memory — sidecars, cached covers."""
    try:
        return b"".join(_read_committed(rel))
    except Exception:
        return None


def _spooled(rel: str) -> Path | None:
    """
    The container-local copy of one committed volume file, pulled once by RPC.

    Staged then renamed, the `_download_weight` rule: a reader that arrives
    mid-pull must see nothing rather than half a file. The temp name carries
    the thread id because two requests for the same fresh file race the pull,
    and two writers on one `.part` interleave; two whole files renaming over
    each other just means the second wins.
    """
    if LOCAL:
        # The spool is a container-local cache of a *network* volume. Here that
        # volume is a directory on the same disk, so spooling it copies every
        # byte you already own, to serve it from a second place.
        direct = WORKSPACE / rel
        return direct if direct.is_file() else None
    local = SPOOL / rel
    try:
        if local.is_file():
            # mtime is the spool's LRU clock; serving a file is what keeps it.
            os.utime(local)
            return local
    except OSError:
        pass
    tmp = local.with_name(f"{local.name}.{threading.get_ident()}.part")
    try:
        local.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as out:
            for chunk in _read_committed(rel):
                out.write(chunk)
        tmp.replace(local)
        _trim_spool()
        return local
    except Exception as exc:
        # A miss is a real answer (the file is not in committed state — never
        # written, or just deleted); anything else is worth its line.
        if not isinstance(exc, FileNotFoundError):
            print(f"[spool] {rel}: {type(exc).__name__}: {exc}", flush=True)
        tmp.unlink(missing_ok=True)
        return None


def _trim_spool() -> None:
    """Oldest-served first, down to 90% of the cap, so one trim buys headroom
    rather than running again on the next pull."""
    with _SPOOL_TRIM_LOCK:
        files = []
        for dirpath, _, names in os.walk(SPOOL):
            for n in names:
                p = Path(dirpath) / n
                try:
                    st = p.stat()
                except OSError:
                    continue
                files.append((st.st_mtime, st.st_size, p))
        total = sum(s for _, s, _ in files)
        if total <= SPOOL_MAX_BYTES:
            return
        files.sort()
        for _, size, p in files:
            p.unlink(missing_ok=True)
            total -= size
            if total <= SPOOL_MAX_BYTES * 0.9:
                break


def _entries_by_rpc() -> tuple[dict[str, list[tuple[str, float]]], list[str]]:
    """
    ({group: [(filename, mtime)]}, [legacy job folders]) asked of Modal, not
    of the mount.

    `volume.listdir` is a metadata RPC against the volume's committed state.
    It does not read `/workspace`, so it needs no `volume.reload()` — and
    therefore **cannot be refused for open files**, which is the whole reason
    it is here. One container serves everybody (`max_containers=1`), so a
    listing frozen by its own picture-loading was what everybody got.

    Non-recursive now: outputs/ is flat, and a file's run is read off its
    name. A directory under it is a job folder from before the record lived
    inside the file; those are reported separately so the migration can be
    started, and never listed as results while they wait — a legacy file has
    no address the page could ask for.
    """
    out: dict[str, list[tuple[str, float]]] = {}
    legacy: list[str] = []
    for e in volume.listdir("/outputs", recursive=False):
        name = e.path.rstrip("/").rsplit("/", 1)[-1]
        if e.type == modal.volume.FileEntryType.DIRECTORY:
            if not name.startswith("."):
                legacy.append(name)
            continue
        if e.type != modal.volume.FileEntryType.FILE or not _keep_entry(name):
            continue
        out.setdefault(_group_of(name), []).append((name, float(e.mtime)))
    return out, legacy


def _entries_by_walk() -> tuple[dict[str, list[tuple[str, float]]], list[str]]:
    """The same answer off the mount, for when the RPC cannot give one."""
    out: dict[str, list[tuple[str, float]]] = {}
    legacy: list[str] = []
    if not OUTPUTS.is_dir():
        return out, legacy
    for p in OUTPUTS.iterdir():
        if p.is_dir():
            if not p.name.startswith("."):
                legacy.append(p.name)
            continue
        if p.is_file() and _keep_entry(p.name):
            out.setdefault(_group_of(p.name), []).append((p.name, p.stat().st_mtime))
    return out, legacy


def _output_entries() -> tuple[dict[str, list[tuple[str, float]]], list[str]]:
    """
    ({group: [(filename, mtime)]}, legacy folders), by RPC, falling back to
    the mount.

    Not silent: a fallback that says nothing makes "the gallery is behind
    again" indistinguishable from "the RPC has been failing all week", which
    is the same reason `_reload_volume` prints when it skips. An empty or
    absent `outputs/` is not a failure — it is a volume nobody has generated
    on yet.
    """
    try:
        return _entries_by_rpc()
    except modal.exception.NotFoundError:
        return {}, []
    except Exception as exc:  # noqa: BLE001 — any RPC failure falls back
        print(f"[gallery] listdir failed ({type(exc).__name__}: {exc}) — "
              f"listing off the mount, which may be stale", flush=True)
        return _entries_by_walk()


def _group_files(group: str) -> list[str]:
    """Every file of one run in outputs/, results and the context tensor
    alike — what a delete takes. Off the mount, after the caller's reload."""
    if not NAME_RE.match(group):
        return []
    try:
        return sorted(p.name for p in OUTPUTS.iterdir()
                      if p.is_file() and _group_of(p.name) == group)
    except OSError:
        return []


def _gallery(limit: int = 200, before: float = 0.0) -> tuple[list[dict[str, Any]], int, list[str]]:
    """
    A page of runs, newest first, how many there are in total, and any job
    folders still waiting to be migrated.

    Keyed by what is on the volume, not by a job id the browser happened to
    keep: a reload, a redeploy, or a job whose record expired all leave the
    work reachable. A file with no record still lists — older results predate
    the record and are not less real for it.

    `before` is the previous page's last sort key, so paging is a window over
    a stable order rather than an offset into a list that grows under it.

    The record is read only for the page being returned, off the head of
    each run's first file by RPC — mtimes arrive without touching
    `/workspace`, so the sort and the window are both decided before a byte
    of picture is pulled, and a deep gallery costs the same per request as a
    shallow one.
    """
    entries, legacy = _output_entries()

    rows: list[tuple[float, str, list[str]]] = []
    for group, files in entries.items():
        files.sort(key=lambda f: f[0])
        rows.append((max(m for _, m in files), group, [n for n, _ in files]))

    # Descending, with the group breaking ties. `mtime` is integer seconds off
    # the RPC, so two runs in the same second tie, and nothing promises a stable
    # order underneath — an unstable sort reshuffles the grid between reloads,
    # which is indistinguishable from the staleness this listing exists to fix.
    rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
    total = len(rows)
    if before:
        rows = [r for r in rows if r[0] < before]

    # Records by RPC, like everything else this listing reads. Cached per run
    # because a finished file never changes — except the moments right after
    # it lands, so an *empty* answer is only cached once the run is old enough
    # that the record is either committed or never coming. Parallel, because
    # serial was the page's whole wait.
    page = rows[:limit]
    need = [(group, modified, names[0]) for modified, group, names in page
            if group not in _META_CACHE]
    if need:
        from concurrent.futures import ThreadPoolExecutor

        def fetch(row: tuple[str, float, str]) -> tuple[str, float, dict[str, Any]]:
            group, modified, first = row
            return group, modified, _read_record(f"outputs/{first}")

        with ThreadPoolExecutor(max_workers=min(16, len(need))) as ex:
            fetched = list(ex.map(fetch, need))
        now = time.time()
        for group, modified, meta in fetched:
            if meta or now - modified > 300:
                _META_CACHE[group] = meta
        fresh_meta = {group: meta for group, _, meta in fetched}
    else:
        fresh_meta = {}

    out: list[dict[str, Any]] = []
    for modified, group, names in page:
        meta = _META_CACHE.get(group, fresh_meta.get(group, {}))
        out.append({
            # The record first, so the derived fields win: the record
            # describes the run, the files are the result.
            **meta,
            "job_id": group,
            "kind": "video" if names[0].lower().endswith(".mp4") else "image",
            "files": names,
            "modified": modified,
        })
    return out, total, legacy


# Records already read, for the life of the container. Safe because a
# finished file is immutable; evicted on delete so a re-used id can never
# wear a dead run's metadata.
_META_CACHE: dict[str, dict[str, Any]] = {}


def _start_migration() -> str:
    """Spawn the outputs migration unless one is already running. The job
    record is the lock: a second listing while it runs reads `running` and
    does not spawn again."""
    try:
        cur = jobs.get(MIGRATE_JOB) or {}
    except Exception:  # noqa: BLE001 — a Dict hiccup must not stop a spawn
        cur = {}
    if cur.get("status") == "running":
        return MIGRATE_JOB
    _publish(MIGRATE_JOB, status="running", percent=0, phase="Queued", files=[])
    _spawn(migrate_outputs_job, MIGRATE_JOB)
    return MIGRATE_JOB


def _forget_output(group: str) -> None:
    """A deleted run takes its caches with it — the record entry, the spooled
    bytes and the covers — or a re-used id would wear a dead run's face."""
    _META_CACHE.pop(group, None)
    for d in (SPOOL / "outputs", SPOOL / "outputs" / ".covers"):
        try:
            for p in d.iterdir():
                if p.is_file() and _group_of(p.name) == group:
                    p.unlink(missing_ok=True)
        except OSError:
            pass


# Where a LoRA the mount cannot show us is pulled to, and the second directory
# ComfyUI searches for one. Under the spool because it is the same thing the
# gallery does for bytes — see the block above `_entries_by_rpc` — and for the
# same reason: an RPC read of committed state is the one read on this platform
# that no open file descriptor can refuse.
SPOOL_LORAS = SPOOL / "loras"


def _lora_visible(rel: str, path: Path) -> bool:
    """
    Can this container actually open `rel`, and make it so if it can only
    almost.

    Two looks, because a GPU container has two different ways to be wrong about
    a LoRA that exists.

    The mount is asked first, by walking down from loras/ rather than statting
    the file. `_sizes_on_disk` used to do this one level up, which covers the
    *file* appearing in a folder we have already listed and not the folder
    itself appearing — and a training run's first checkpoint creates the
    folder. A name this container has asked about and been told no is cached
    below us, and `volume.reload()` does not invalidate it, so the leaf scandir
    raised FileNotFoundError about a directory that is there. `_listed` is the
    function that already learned this for `outputs/{job}/{name}`; it is the
    same lesson and there is no reason for two of it.

    Then committed state, by RPC, which is the half that was missing entirely.
    A container holding a LoRA has that LoRA *mapped* off /workspace —
    safetensors maps the weights and ComfyUI keeps the patched state dict — so
    every `volume.reload()` for the rest of its life is refused for open files,
    and the mount freezes at whenever it last synced. The page kept catching up
    because the web container maps nothing; the GPU container did not, and told
    you the epoch you were watching get written did not exist. Ten minutes of
    scaledown was the only cure.

    `_spooled` pulls the committed bytes onto local disk and ComfyUI is pointed
    at that directory too (see `_Comfy.start`), so the name resolves to a real
    file under a searched root and loads exactly as it would have. It also
    means the copy this container maps is on /tmp, which is one fewer descriptor
    on /workspace — the same move, and the same payoff, as serving off the
    spool.

    The order matters and is not arbitrary: the mount is free and the RPC is a
    transfer, so a fresh container never pays for this and a frozen one pays
    once per checkpoint.
    """
    if _listed(path, LORAS):
        return True
    return _spooled(f"loras/{rel}") is not None


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

    The returned path is always the /workspace one even when `_lora_visible`
    had to fetch a copy to answer: that path is the LoRA's durable address, it
    is what goes in the record, and `name` below is derived from it. Which
    directory the bytes are read out of is ComfyUI's business and nobody
    else's.
    """
    path = Path(str(raw_path or "")).resolve()
    if LORAS.resolve() not in path.parents:
        raise ValueError(f"LoRA must be under loras/: {str(raw_path)!r}")
    rel = path.relative_to(LORAS.resolve()).as_posix()
    if not _lora_visible(rel, path):
        raise ValueError(
            f"No such LoRA: {rel}. Not on the volume and not in its committed "
            "state — it may have been deleted, or a training run renamed out "
            "from under this list. Reopen Settings to refresh it."
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
            # Krea 2's text encoder takes LoRA weights, unlike H3's — so this
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

# The long edge of a region's own photograph, and of the two plates, enforced
# where it binds. The page shrinks to this number before it uploads, and that
# copy is an optimisation: eight uncapped photographs base64'd into one JSON
# body is a request measured in tens of megabytes, sent by a browser to a route
# that had no opinion about it. The video side has had a server-side cap since
# `ref_image_size: "max"` was found by whoever had the biggest camera; this side
# never did, so the only thing standing between a phone's 4032x3024 and the VAE
# encoder was a `<canvas>` in the client.
#
# A separate constant from H3_REF_MAX_SIDE despite the equal value, because the
# reasons do not travel: H3's is the number below which the node's own scale is
# exactly 1.0, and this one is about the payload and the encoder. They are free
# to diverge, and a shared name would make that look like a mistake.
REGION_REF_MAX_SIDE = 1536


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
            # V9's per-row portrait toggle: render this region's LoRA into a
            # portrait and feed it back as a live reference frame during the
            # compose. Costs an extra render plus a model reload and forces
            # the sequential path — the node's own docstring prices it at
            # roughly 3x — which is why it is per-region and opt-in rather
            # than the global auto_portrait switch.
            "anchor": bool(entry.get("anchor")),
            # Marks a row the backend conjured rather than the user drew —
            # see the conjuring note in _ImageSide.generate_image. Carried
            # through validation because this function runs twice on the
            # same rows and dropping it on the second pass would un-mark it.
            "derived": bool(entry.get("derived")),
        })
    return out


def _validate_objects(raw: Any) -> list[dict[str, Any]]:
    """
    The free-role plates: V12's sockets 3 and 4, a photograph plus the user's
    own sentence about what it is.

    Two per render, because that is how many sockets the node has left after
    the scene and the outfit — a cap set by hardware, not taste. The note is
    required, and that is the outfit's own lesson generalised: krea2edit is
    instruction-driven, so a reference the prompt never mentions is encoded
    into the attention sequence and then does close to nothing. A no-op plate
    that looks attached is worse than a refusal, so the refusal happens here,
    in milliseconds, with the fix in the sentence.

    A plate is an object or a person — two of the node's five reference
    roles, and the two whose clauses have been read in its source rather than
    guessed. A person plate is a photograph standing in for somebody with no
    LoRA: the node writes "the person from the Nth reference, face unchanged"
    itself, which is why the note is optional there and required for objects —
    an unnamed object does nothing, an unnamed person already has a clause.

    Same stdlib-importable shape as `_validate_regions`, for the same reason:
    a malformed object should be a form error, not a dead job on a warm H100.
    """
    if not raw:
        return []
    if not isinstance(raw, list):
        raise ValueError("Objects must be a list.")
    if len(raw) > 2:
        raise ValueError("Two object references per render — the node has "
                         "two free sockets. Fold the rest into the prompt.")
    out = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict) or not isinstance(entry.get("image"), str) \
                or not entry.get("image"):
            raise ValueError(f"Object {i + 1} is missing its photograph.")
        role = str(entry.get("role") or "object").strip().lower()
        if role not in ("object", "person"):
            # "scene" and "style" are real roles upstream but arrive here by
            # other doors — the scene tile and the style engine — and letting
            # them in as objects would be a second way to do the first thing.
            raise ValueError(
                f"Object {i + 1} has role {role!r}; a free plate is "
                "\"object\" or \"person\"."
            )
        note = str(entry.get("note") or "").strip()
        if not note and role == "object":
            raise ValueError(
                f"Object {i + 1} needs a sentence saying what it is — "
                "\"a motorcycle she leans against\". A reference the prompt "
                "never mentions does nothing."
            )
        out.append({"image": entry["image"], "note": note, "role": role})
    return out


def _validate_style_refs(raw: Any) -> list[str]:
    """
    The style-by-reference photograph — one.

    One because the K/V engine's tested, no-leakage route is single-reference;
    the pack's own README calls its two-reference node an experiment. A cap
    set by what has been proven, not by the plumbing.
    """
    if not raw:
        return []
    if not isinstance(raw, list):
        raise ValueError("Style references must be a list.")
    if len(raw) > 1:
        raise ValueError("One style reference — the engine's no-leakage route "
                         "is single-reference; its two-ref mode is an "
                         "experiment nothing here has proven.")
    for i, r in enumerate(raw):
        if not isinstance(r, str) or not r:
            raise ValueError(f"Style reference {i + 1} is malformed.")
    return list(raw)


def _armed_regions(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    The rows the edit path will actually arm: a LoRA or a reference photo.

    Mirrors the node's own predicate — `has_lora(r) or has_ref(r)` on V9's
    edit path — because that filter decides whether V12's unified attention
    arms at all, and a plate composed around zero armed boxes is the
    "V12 unified attention was not armed" RuntimeError. Reads the validated
    rows, so `lora` is a name or "None" and `ref` is a string.
    """
    return [r for r in regions
            if r.get("lora") not in (None, "", "None") or r.get("ref")]


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
        # A conjured full-canvas row exists to arm the edit path, not to place
        # anybody — "here" is the whole frame, so a placement clause for it
        # says nothing and its "one single person only" would be wrong the
        # moment the picture being edited holds two people.
        if region.get("derived") and not described:
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
    objects: list[dict[str, Any]] | None = None,
    style_refs: list[str] | None = None,
    style_strength: float = 1.0,
    region_weight: float = 1.0,
    edit_strength: float = 0.7,
    compose_seed: int | None = None,
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
    dit = _slot_name("turbo" if model == "turbo" else "raw")
    te = _slot_name("text_encoder")
    vae = _slot_name("vae")

    graph: dict[str, Any] = {
        "dit": _dit_node(dit),
        # type="krea2" is what selects Krea2Tokenizer and the Qwen3-VL hidden
        # state Krea 2 reads. The file is the bf16 text encoder — the fp8_scaled
        # one carries ~504 extra weight_scale tensors and is rejected outright.
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": te, "type": "krea2", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        # Krea 2's latent format is `Wan21`: 16 channels, and
        # `latent_dimensions = 3`, so the sampler works on [B, 16, T, H/8, W/8].
        # `length: 1` gives ((1-1)//4)+1 = 1 frame — a video latent node asked
        # for a single frame.
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

    # Krea 2 renders on PyTorch attention, against the process-wide SageAttention.
    #
    # `--use-sage-attention` is argv on both containers and is argued for by H3
    # alone: a 33B model running full self-attention over a packed video/text/
    # audio sequence is where the 2x lives. Krea 2 collects none of it, and
    # that is measured rather than assumed — `tools/ab_sage.py`, six matched
    # renders at 1024px/8 steps, 3.23s median on PyTorch against 3.26s on
    # sage. A 1024px render is ~4k tokens; attention is not the bill. So the
    # flag was buying this path nothing, and charging it the following.
    #
    # On sm90 sage dispatches to
    # `sageattn_qk_int8_pv_fp8_cuda_sm90`, which quantizes V to FP8 e4m3 with
    # one scale per channel across the whole sequence, and it runs there with
    # both of its outlier mitigations off: ComfyUI hardcodes `smooth_k=False`
    # at the call, the kernel hardcodes `smooth_v=False` inside. One outlier
    # token then sets the scale for every other token in its channel, and the
    # tokens that lose their resolution are a contiguous block of the image —
    # which is the shape of the intermittent black blotches this was reported
    # for. Krea 2 hands that kernel the worst tensors in the building:
    # `txtfusion` runs attention over the twelve RAW Qwen3-VL layers, the same
    # very large activations that already cost us `--bf16-text-enc`. Different
    # prompt, different outliers, which is why it was intermittent rather than
    # obvious — and why the three prompts ab_sage.py runs did not reproduce
    # it. **That attribution is unproven here.** What is proven is the price
    # of ruling it out, and the price is nothing; upstream has this exact
    # kernel on report for the same class of artefact — thu-ml/
    # SageAttention#288, ComfyUI-WanVideoWrapper#1554 — which is enough to
    # spend nothing on.
    #
    # A node rather than a change to argv, because argv is shared with the
    # video container and H3 is the reason the flag is there. First on the
    # model chain so everything downstream inherits it: V12 chains to the
    # `previous` override rather than replacing it, and the style pack
    # forwards transformer_options, so both land here. The text encoder and
    # the VAE were never on this path — `optimized_attention_for_device(...,
    # small_input=True)` and `vae_attention()` do not offer sage at all.
    graph["attn"] = {"class_type": "ModelAttentionBackend",
                     "inputs": {"model": ["dit", 0],
                                "attention": "pytorch attention"}}

    # Prompt-level LoRAs are the whole canvas; a region's own LoRA is applied
    # by V12 inside its box and never appears in this chain. Both weights are
    # carried because Krea 2's text encoder takes LoRA patches — the video
    # side's model-only loader has no such second number.
    model_src, clip_src = ["attn", 0], ["clip", 0]
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
    # noise. Other video families *do* want ModelSamplingSD3 — their configs
    # take the 1000 default — which is exactly why this looked like a line worth
    # copying. Check the model config, not the neighbouring graph.
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
        # `portrait` is V9's per-row anchor: the region's LoRA rendered into a
        # portrait and fed back as a live reference frame during the compose.
        # It reads the row, not the global auto_portrait switch — which stays
        # False below precisely so that one anchored region does not anchor
        # them all.
        regions_json = json.dumps([
            {"lora": r["lora"], "strength": r["strength"], "enable": True,
             "ref_image": r.get("ref_image", ""), "prompt": r["prompt"],
             "portrait": bool(r.get("anchor"))}
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
            #
            # edit_lora_strength is the page's "Fidelity" number: the node's
            # 0.0–2.0 range, defaulting to its own 0.7. Low keeps the picture
            # being edited, high lets the instruction rewrite more of it.
            "edit_lora_strength": edit_strength,
            "compose_steps": 10,
            # The node's tooltip claims 0 reuses the incoming noise seed; its
            # code does no such thing — `_compose_once` gets seed + 10000 + i,
            # so 0 is a *fixed* seed and every multi-subject compose staged
            # identically however the main seed rolled. Following the main
            # seed makes staging reroll with the run and pin with the pin.
            "compose_seed": seed if compose_seed is None else compose_seed,
            "ref_max_side": 0,
            # Declared defaults, written out per the optional-inputs rule
            # above — these three were missed when that rule landed.
            # grounding_px is a real dial upstream (lower = stronger scene
            # edits, higher = stronger identity) and stays at the default
            # until somebody measures it.
            "grounding_px": 1024,
            "portrait_steps": 8,
            "portrait_seed": 0,
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
                # Sockets 3 and 4 are the free-role plates: the user's own
                # sentence about what each photograph is, under a role the
                # validator has pinned to "object" or "person". An object
                # note is validated non-empty before this builds, because an
                # unreferenced frame does close to nothing; a person plate
                # may arrive noteless because the node's own clause covers
                # it — "the person from the Nth reference, face unchanged".
                *({"role": o.get("role", "object"), "note": o["note"]}
                  for o in (objects or [])),
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
        # Sockets 3 and 4, in list order — refs_json above appended their notes
        # in the same order, and that pairing is positional, so these two loops
        # must walk the same list the same way or a note lands on the wrong
        # photograph.
        for i, obj in enumerate(objects or []):
            node = f"object{i + 1}"
            graph[node] = {"class_type": "LoadImage", "inputs": {"image": obj["image"]}}
            v12[f"extra_ref_{i + 3}"] = [node, 0]
            # Full canvas for the same reason the outfit's is: a sub-box would
            # hard-paste the photograph instead of composing the thing it shows.
            v12[f"extra_box_{i + 3}"] = "0,0,1,1"

        graph["v12"] = {"class_type": KREA2_REGIONAL_NODE, "inputs": v12}
        sampler_model, positive, negative = ["v12", 0], ["v12", 1], ["v12", 2]
        if scene or outfit or objects:
            # Only the plate path registers the fixed-arity edit wrappers this
            # adapts — see comfy_nodes/visionary_edit_arity for the host change
            # that made them un-callable. The plain path's session wrappers are
            # *args-safe and go through untouched, so the node would be a no-op
            # there; not wiring it keeps the graph a record of what mattered.
            graph["arity"] = {"class_type": "VisionaryEditArity",
                              "inputs": {"model": ["v12", 0]}}
            sampler_model = ["arity", 0]
    elif style_refs:
        # The K/V injection path — tools/ab_style.py is the measurement that
        # put it here over the ostris reference-LoRA. The reference is VAE-
        # encoded at the target size and its attention K/V statistics are
        # injected during sampling; no weight is loaded and the base model is
        # untouched. One reference, because the pack's own README calls the
        # two-ref node an experiment and its no-leakage claim is tested on one.
        graph["styleimg"] = {"class_type": "LoadImage",
                             "inputs": {"image": style_refs[0]}}
        graph["styleref"] = {
            "class_type": K2ST_REF_NODE,
            "inputs": {"vae": ["vae", 0], "target_latent": ["latent", 0],
                       "reference_image": ["styleimg", 0],
                       "fit": "crop", "upscale_method": "lanczos"},
        }
        graph["pos"] = {"class_type": "CLIPTextEncode",
                        "inputs": {"text": prompt, "clip": clip_src}}
        # The pack's recommended route zeroes the positive rather than
        # encoding an empty string — Turbo is CFG-free and this is the wiring
        # its shipped workflow carries; the A/B rendered clean with it.
        graph["neg"] = {"class_type": "ConditioningZeroOut",
                        "inputs": {"conditioning": ["pos", 0]}}
        graph["styletx"] = {
            "class_type": K2ST_TRANSFER_NODE,
            "inputs": {"model": model_src,
                       "reference_latent": ["styleref", 0],
                       "ref_conditioning": ["pos", 0],
                       "mode": "recommended",
                       # The page's one dial. The node's own tooltip gives it
                       # the semantics the ostris LoRA never had: an overall
                       # style mix where 0 genuinely disables the reference.
                       "style_strength": style_strength,
                       # The rest are the node's defaults, written out per the
                       # optional-inputs rule; `recommended` mode governs them.
                       "value_adain_strength": 0.65,
                       "ref_value_mix": 1.0,
                       "ref_k_strength": 1.06,
                       "rf_mode": "flowturbo_pc",
                       "gamma": 0.5,
                       "beta": 2.5,
                       "high_scale_start": 1.04,
                       "high_scale_end": 0.0,
                       "low_scale_start": 1.0,
                       "low_scale_end": 1.10,
                       "adain_strength": 0.85,
                       "blocks": "7-27"},
        }
        sampler_model, positive, negative = ["styletx", 0], ["pos", 0], ["neg", 0]
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
    # The regional session uploads every region's LoRA to the card and keeps
    # the copies on the patcher it returned, which ComfyUI's execution cache
    # holds — so without this each regional render leaves its LoRAs behind and
    # the next one starts with less room. `_reclaim()` is the recovery for that
    # and it costs a checkpoint reload; this is the prevention, and it costs a
    # re-upload of a few hundred megabytes on the next run.
    #
    # Wired through the latent rather than left dangling, because a node with no
    # edge into the result is one ComfyUI may schedule *before* the sampler —
    # which would free the tensors `_prepare` is about to build and turn a leak
    # into a rebuild every step.
    samples = ["sample", 0]
    if regions:
        graph["free_regional"] = {
            "class_type": "VisionaryFreeRegional",
            "inputs": {"latent": samples, "model": sampler_model},
        }
        samples = ["free_regional", 0]
    graph["decode"] = {"class_type": "VAEDecode",
                       "inputs": {"samples": samples, "vae": ["vae", 0]}}
    graph["save"] = {"class_type": "SaveImage",
                     "inputs": {"images": ["decode", 0],
                                "filename_prefix": "visionary"}}
    return graph


# The image family's half of the generator body. Not a Modal class: the three
# placements — image alone, video alone, both on one card — are declared
# together after the video half, and each inherits what it serves. Modal
# collects `@modal.method()` up the MRO, so the body lives once.
#
# The nodes this half needs at startup, checked once rather than per request:
# a custom node that fails to import leaves ComfyUI running happily without
# it, and the first symptom would otherwise be a queued graph rejected for an
# unknown class_type — which reads as our graph builder naming a node wrong,
# minutes into a warm GPU, when the traceback that explains it scrolled past
# during startup.
# GGUF_UNET_NODE is on this list for the reason the others are: a pack that
# fails to import leaves ComfyUI running happily without it, and the first
# symptom would otherwise be a graph rejected for an unknown class_type —
# minutes into a warm GPU, and only for whoever picked a quantised tier, which
# is the hardest kind of report to read.
IMAGE_NODES = (KREA2_REGIONAL_NODE, "VisionaryBoxes", "VisionaryFreeRegional",
               "VisionaryEditArity", K2ST_REF_NODE, K2ST_TRANSFER_NODE,
               GGUF_UNET_NODE)


class _ImageSide:
    """A Krea 2 render on this container's warm ComfyUI."""

    _comfy: _Comfy

    @modal.method()
    def generate_image(self, job_id: str,
                       params: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        # Same three lines as the video side, and for the same reason: the
        # window between accepting a job and queueing its graph was unlit, and
        # the phase claimed `loading` while the volume reload — the one
        # unbounded step in it — had not been started.
        print(f"[image] {job_id} accepted", flush=True)
        _note_queue_wait("image", job_id, params)
        _clear_stop(job_id)
        _publish(job_id, status="running", phase="reloading the volume")
        _reload_volume()

        model = "turbo" if str(params.get("model") or "turbo") != "raw" else "raw"

        try:
            # Inside the try, not above it: a missing weight raised out here
            # would leave the record saying "running" forever, and the UI
            # polling a job that is never going to answer.
            _require_models(model, "vae", "text_encoder")
            _stop_gate(job_id, "the volume reload")

            regions = _validate_regions(params.get("regions"))
            objects = _validate_objects(params.get("objects"))
            style_refs = _validate_style_refs(params.get("style_refs"))
            if style_refs and regions:
                # Two engines, one graph slot: the ostris patch rewires the
                # conditioning and V12 rewires the attention, and nothing has
                # proven them composed. Refused rather than silently picking
                # one — a render that quietly dropped your boxes or your style
                # would be the worse answer.
                raise ValueError(
                    "Style references and region boxes are two engines — "
                    "remove one. Style applies to the whole frame."
                )
            if style_refs:
                style_refs = [
                    self._comfy.stage(job_id, b64, f"style{i + 1}")
                    for i, b64 in enumerate(style_refs)
                ]
                for name in style_refs:
                    _fit_reference(COMFY / "input" / name,
                                   REGION_REF_MAX_SIDE, "image")
            plates = {}
            # Objects are plates too — the same sockets, the same engine, and
            # therefore the same gates. The loop reads slot names so its
            # errors keep naming the thing that was attached.
            plate_slots = [s for s in ("scene", "outfit") if params.get(s)]
            plate_slots += ["object"] * len(objects)
            # Validated here rather than at the bottom of this block because
            # the conjuring below moves one entry out of the chain; still
            # validated exactly once, which is what the note further down is
            # protecting. `loras_sent` keeps the user's own stack for the
            # record — same split as prompt/prompt_typed: the sidecar's
            # `loras` is what was chosen, the derived region row is where
            # the first one actually ran.
            loras = _validate_loras(params.get("loras"))
            loras_sent = [dict(l) for l in loras]
            if plate_slots and not regions:
                # A plate with no boxes used to be a form error ("draw a
                # box"), which taxed the commonest render — one character,
                # whole canvas, no boxes drawn — with a gesture that adds no
                # information: the box would be the full frame. The edit path
                # still needs an armed region (V12's own assertion), so the
                # run's first LoRA is moved into a conjured full-canvas row —
                # the same thing a hand-drawn full-frame box holding that
                # chip would be. Moved, not copied: the deltas add in one
                # forward, and a LoRA applied twice is the mottled-texture
                # failure the edit-strength default exists to avoid. What
                # the move costs is the chip's text-encoder strength — a
                # region applies unet-only — which matches what a drawn box
                # already does with the same LoRA.
                if not loras:
                    raise ValueError(
                        f"A {plate_slots[0]} reference needs an identity to "
                        "compose around — put a LoRA on the run, or draw a "
                        "region holding a LoRA or a photo."
                    )
                first, loras = loras[0], loras[1:]
                regions = [{
                    "x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0,
                    "prompt": "", "lora": first["name"],
                    "strength": first["unet"], "ref": "",
                    "anchor": False, "derived": True,
                }]
            for slot in plate_slots:
                if not _armed_regions(regions):
                    # The edit path only arms boxes that hold a LoRA or a
                    # photo (`has_lora(r) or has_ref(r)` in the node), so a
                    # plate over described-only boxes dies in V12's own
                    # assertion — "V12 unified attention was not armed" —
                    # nine seconds after a cold model load. Observed live
                    # on job gen202608241321556c9e. The structural zero
                    # belongs here, where it is a form error.
                    raise ValueError(
                        f"A {slot} reference needs a box holding an "
                        "identity — a LoRA or a photo. Boxes with only a "
                        "description cannot anchor the compose."
                    )
            if plate_slots:
                _require_models("krea2_edit")
            for slot in ("scene", "outfit"):
                if params.get(slot):
                    plates[slot] = self._comfy.stage(job_id, params[slot], slot)
                    _fit_reference(COMFY / "input" / plates[slot],
                                   REGION_REF_MAX_SIDE, "image")
            # The staged filename replaces the bytes in place, which is the
            # same shape `regions[i]["ref_image"]` takes below — the graph
            # reads what the stager wrote, and the base64 never travels past
            # this point.
            for i, obj in enumerate(objects):
                obj["image"] = self._comfy.stage(job_id, obj["image"],
                                                 f"object{i + 1}")
                _fit_reference(COMFY / "input" / obj["image"],
                               REGION_REF_MAX_SIDE, "image")

            # Each region's own photo, staged the same way the plates are and
            # deliberately without their two gates: a mold is not an
            # `extra_ref_*`, so it neither turns on krea2edit nor needs the
            # identity-edit weight. The staged name goes back onto the row,
            # which is what `_krea2_graph` reads into `regions_json`.
            # Capped on arrival, like the video side's references and for a
            # different reason — see REGION_REF_MAX_SIDE. Behind the route
            # rather than in the page, because the page is one of the ways in
            # and the reuse path is another.
            for i, region in enumerate(regions):
                if region["ref"]:
                    region["ref_image"] = self._comfy.stage(
                        job_id, region["ref"], f"region{i}")
                    _fit_reference(COMFY / "input" / region["ref_image"],
                                   REGION_REF_MAX_SIDE, "image")

            # `loras` was validated once, above the plate gates — again would
            # re-stat the volume, so a LoRA deleted during a run could fail
            # the job that already produced the picture.
            shift = float(params.get("shift") or 1.15)
            # A style run's default sampler is the engine's tested route — see
            # STYLE_DEFAULTS. The route resolves this too; here again because
            # a job is reachable without the route's baking.
            fallback = STYLE_DEFAULTS if style_refs else IMAGE_DEFAULTS
            sampler = str(params.get("sampler") or fallback["sampler"])
            scheduler = str(params.get("scheduler") or fallback["scheduler"])
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
            raw_style = params.get("style_strength")
            style_strength = 1.0 if raw_style in (None, "") else float(raw_style)
            # Clamped to the node's declared 0.0–2.0, not rejected — same
            # treatment as a box dragged off the canvas. 0 is a real state
            # (the instruction layer off, plates become near no-ops) and
            # reaches the node intact for the same reason region_weight's
            # zero does.
            raw_edit = params.get("edit_strength")
            edit_strength = (0.7 if raw_edit in (None, "")
                             else max(0.0, min(2.0, float(raw_edit))))
            raw_cseed = params.get("compose_seed")
            compose_seed = None if raw_cseed in (None, "") else int(raw_cseed)

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
                objects=objects, style_refs=style_refs,
                style_strength=style_strength,
                edit_strength=edit_strength, compose_seed=compose_seed,
                **plates,
            )

            if params.get("workflow_graph"):
                # The Playground toggle: the console's own graph feeds the
                # saved workflow by inherited node key — see _apply_workflow.
                # Applied here rather than in the route so the graph it feeds
                # is the one with staged filenames in it, not placeholders.
                graph = _apply_workflow(graph, params["workflow_graph"],
                                        params.get("workflow_extras"))

            info = {"width": width, "height": height, "seed": seed, "steps": steps}
            _stop_gate(job_id, "staging")
            # `loading`, not `generate` — ComfyUI has the graph and has not
            # sampled anything. `_drain` moves it on at the first real step.
            _publish(job_id, phase="loading", step=0, total_steps=steps,
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

        OUTPUTS.mkdir(parents=True, exist_ok=True)
        report = {
            "sampler": sampler, "scheduler": scheduler,
            "cfg_scale": cfg, "shift": shift,
            # `loras_sent`, not the chain that ran: when a row was conjured
            # the first chip moved into it, and the record keeps what was
            # *chosen* here — same split as prompt/prompt_typed. The derived
            # region row below is where it went; Reuse restores the chips.
            "loras": [{"name": l["name"], "unet": l["unet"],
                       "text_encoder": l["text_encoder"]} for l in loras_sent],
            # Boxes as a 4-list in the order the row's fields read, so `reuse`
            # can put them straight back into x/y/w/h without a schema.
            #
            # `ref` is a bool, never the photo. This dict is the job record and
            # the job record is polled every 400ms — eight base64 photographs in
            # something on that loop is the exact failure the "keep the polled
            # thing small" rule exists for. The staged file is gone with the
            # container anyway, so there is nothing here a caller could reuse.
            #
            # `anchor` and `derived` only when true — on every plain row they
            # would be two fields of noise, and older readers tolerate their
            # absence the same way they tolerate `ref`'s.
            "regions": [{"box": [r["x"], r["y"], r["width"], r["height"]],
                         "lora": r["lora"], "strength": r["strength"],
                         "prompt": r["prompt"],
                         "ref": bool(r.get("ref_image")),
                         **({"anchor": True} if r.get("anchor") else {}),
                         **({"derived": True} if r.get("derived") else {})}
                        for r in regions],
            "region_weight": region_weight,
            # Only when it differs from what was typed. A caption nobody wrote
            # is the one thing about a regional render that cannot be worked
            # out from the fields, and "why is he on the right" has no answer
            # without it — but repeating the prompt back on every plain render
            # would be the record describing itself.
            **({"caption": caption} if caption != typed else {}),
            # Which engine ran, because the three are not the same picture and
            # the numbers beside them do not say which one produced it.
            "mode": ("krea2edit" if plates or objects else "regional")
            if regions else ("style" if style_refs else "plain"),
            # A count, not the photos — same rule as the region `ref` bool.
            # Which pictures carried the style is not reconstructible from a
            # count, and that is the plate lesson accepted rather than fixed:
            # the bytes cannot live in a record polled every 400ms.
            **({"style_refs": len(style_refs),
                "style_strength": style_strength} if style_refs else {}),
            # And which plates it composed around — same rule as `ref` above:
            # the slot names, never the photos. Without this a krea2edit run's
            # record said the engine and not the reason, so an outfit render
            # was not distinguishable from a scene one; verified missing on job
            # gen2026082413552761e9. Older records lack the field and every
            # reader must keep tolerating that. A person plate may have no
            # note, so the role alone is the entry.
            **({"plates": sorted(plates)
                + [f"{o.get('role', 'object')}: {o['note']}" if o["note"]
                   else o.get("role", "object") for o in objects]}
               if plates or objects else {}),
            # The two numbers a compose ran at, recorded only when one ran —
            # compose_seed is the *effective* value, because "followed the
            # main seed" is not reconstructible after the fact.
            **({"edit_strength": edit_strength,
                "compose_seed": seed if compose_seed is None else compose_seed}
               if plates or objects else {}),
            **info,
        }
        record = _output_record(
            kind="image", job_id=job_id, model=model,
            prompt=str(params.get("prompt") or ""),
            negative_prompt=str(params.get("negative_prompt") or ""),
            created=time.time(), **_shot_meta(params), **report,
            # Which saved workflow the toggle ran, when one did — the record
            # is the only place that can still answer that a week later.
            **({"workflow_name": params["workflow"]}
               if params.get("workflow") else {}),
        )
        names = []
        for i, src in enumerate(out_names):
            name = f"{job_id}_{i:02d}.png"
            # Re-encoded rather than copied, because the record and the
            # parameters block have to go in and ComfyUI ran with
            # --disable-metadata. The alternative is a PNG whose settings live
            # only in a job record that expires, which is the thing every
            # existing tool — PNG Info tabs, ComfyUI, the gallery — reads a
            # file to avoid. A1111's `parameters` stays beside ours for those
            # tools; the per-image seed is theirs, because ComfyUI advances
            # the noise seed per latent in a batch and a block reporting the
            # same seed for all four cannot reproduce three of them.
            _png_with_record(COMFY / "output" / src, OUTPUTS / name, record, {
                "parameters": _infotext(
                    prompt=str(params.get("prompt") or ""),
                    negative_prompt=str(params.get("negative_prompt") or ""),
                    model=model, seed=seed + i, report=report),
            })
            names.append(name)
        volume.commit()

        # Only filenames go into the job record. The PNGs themselves are served
        # by /api/file/{name} off the volume — a 1024px base64 image is
        # megabytes, and this dict is polled several times a second. `files`
        # is also what the canvas builds its <img> tags from, so the completed
        # record is the last round trip a run needs.
        res = {
            "status": "completed", "job_id": job_id, "files": names,
            "model": model, "output_dir": str(OUTPUTS),
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

# **A still is a film still — a fragment of a take, decoded, one frame kept.**
# So it is a frame count and not a duration, and it sits below H3_MIN_FRAMES on
# purpose: that floor is the trained range for a clip somebody is going to
# *watch*, and none of what it protects applies to a run whose output is a
# picture. Duration and model stop being the same control, which is the whole
# point — the cast, the reference grammar and the audio labels are H3's, so a
# still made anywhere else was a second product with a second vocabulary.
#
# 22 is the second point on the 17n+5 grid the video VAE decodes on; 5 is the
# first and yields five candidates, 39 the third. Short lengths hold on ref2va
# with the cast's identity intact — measured, and the reason this is a length
# rather than a second image model. The number itself is a starting point from
# one set of runs, not a measurement of the optimum; the neighbours are one
# edit away.
#
# There is no image VAE here and there must not be one. The frames come out of
# the *video* VAE, which is what keeps hair, skin and fine contour — the T=1
# decoders soften exactly what a character keyframe is carrying.
H3_STILL_FRAMES = 22

# H3's canvas is a short edge, not a resolution: the aspect ratio picks the
# long edge. 768 is what it was trained at; 544 is the draft tier, which is
# roughly 2.3x faster per step and is the single biggest speed lever there is —
# far bigger than step count, and it costs detail rather than coherence.
H3_TIERS = {"full": 768, "draft": 544}

# ref2va's limits: 9 images, 3 videos of 2-15 s, 3 standalone audio clips, and
# 12 across all types. A video's own soundtrack rides along with it and does not
# count against the audio budget here, because it is packed as that video's
# rather than as its own <Audio n>.
#
# **12 is the binding one and it is the one nothing surfaces.** Nine photographs
# plus three voices is exactly the total, so a fully cast scene has no room left
# for a reference video at all. Three audio also caps how many *voices* one
# generation can clone — two people in a diner is fine, add the waitress and you
# are at the ceiling. A scene longer than one generation gets its own three per
# link, which is one of the things chaining buys.
MAX_H3_REFS = 9
MAX_H3_REF_VIDEOS = 3
# `<Audio N>`. Three, from the node's own `ref_audios` autogrow template, and it
# counts toward the same 12 as pictures and videos.
MAX_H3_REF_AUDIOS = 3
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


def _fit_reference(path: Path, cap: int = H3_REF_MAX_SIDE, tag: str = "video") -> None:
    """
    Shrink one staged reference in place, to whatever cap the caller is bound by.

    Bounding the file is what bounds the run: under 2048 on the short edge the
    node's "max" scale is exactly 1.0, so the pixels written here are the pixels
    the DiT sees, and only one place decided how big the picture is. That is the
    same reason `ref_max_side` is pinned to 0 on the image side — two things
    resizing the same photograph is two things to check when it comes out soft.

    Applied to both H3 modes. "match" is already bounded in tokens, but not in
    what PIL and the VAE encoder chew through on the way there, and one rule is
    easier to hold than one rule per mode.

    The image side passes its own cap, for its own reason — see
    REGION_REF_MAX_SIDE. Same mechanism, because two things resizing a
    photograph is the failure this function exists to prevent, and a second
    copy of it on the other route would be exactly that.

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
            if max(w, h) <= cap:
                return
            scale = cap / max(w, h)
            size = (max(1, round(w * scale)), max(1, round(h * scale)))
            src.resize(size, Image.LANCZOS).save(tmp, "PNG")
        tmp.replace(path)
        print(f"[{tag}] reference {path.name}: {w}x{h} -> {size[0]}x{size[1]}",
              flush=True)
    except Exception as exc:
        # A reference that cannot be read is not one that can be measured
        # either, and LoadImage is about to fail on it with a better message
        # than anything guessable from here.
        tmp.unlink(missing_ok=True)
        print(f"[{tag}] could not cap reference {path.name}: {exc}", flush=True)


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
    ref_audios: list[str] | None = None,
    ref_size: str = "match",
    loras: list[dict[str, Any]] | None = None,
    shift_video: float | None = None,
    shift_audio: float | None = None,
    save_context_as: str | None = None,
    load_context_from: str | None = None,
    still: bool = False,
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
    ref_mode = bool(references or ref_videos or ref_audios)
    dit = _slot_name("h3_ref_dit" if ref_mode else "h3_dit")
    te = _slot_name("h3_te")
    vae = _slot_name("h3_vae")
    avae = _slot_name("h3_audio_vae")

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
        "dit": _dit_node(dit),
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

    if still:
        # Refused by name rather than reached as a KeyError two lines down.
        # Both context inputs are about what a take opens from and what the
        # next one continues into, and a picture is neither end of a chain.
        if save_context_as or load_context_from:
            raise ValueError("A still has no motion context to save or "
                             "continue from.")
        # A picture has no soundtrack, so the audio decode, the muxer and the
        # video writer all come out. In a run with no references that orphans
        # the audio VAE — nothing reads it, so ComfyUI never loads it, and a
        # 605 MB decoder stops being paid for by every still.
        del graph["audio"], graph["video"]
        graph["save"] = {"class_type": "SaveImage",
                         "inputs": {"images": ["frames", 0],
                                    "filename_prefix": "visionary"}}
        # Falls through rather than returning: the LoRA stack and the sigma
        # shift are below, and a still is the run that needs them *most* —
        # `h3_speed_ref2v_4` is what makes one cheap enough to iterate on. An
        # early return here dropped it in a way nothing would have reported,
        # which is the failure the LoRA comment below already warns about.
        # Nothing between here and there touches the two nodes just deleted;
        # both motion-context blocks are unreachable, refused above.

    # ── motion continuation ─────────────────────────────────────────────────
    # Every H3 take saves its sampler latent beside the clip, because the *next*
    # take is the one that needs it and nothing knows at render time whether
    # there will be one. Picture and audio come out of the same latent, so this
    # single file is the whole context a continuation slices from.
    if save_context_as:
        graph["save_ctx"] = {
            "class_type": "MiniMaxH3MotionContextSaveLatent",
            "inputs": {"latent": ["sample", 0],
                       "filename_prefix": f"h3_context/{save_context_as}",
                       "clip_index": 0},
        }
    if load_context_from:
        # Between the stock conditioning node and the guider — the pack's own
        # stated wiring. The previous clip's latent supplies both picture and
        # sound, sliced straight out with no decode/re-encode round trip; the
        # pinned run lands at the head of this clip and Trim removes it after
        # decode, matching the audio cut to the frame cut so a chain does not
        # accumulate drift. Every optional input is spelled out — the house
        # rule, because INPUT_TYPES defaults and run() defaults disagree in
        # node packs more often than they should.
        graph["ctx_lat"] = {
            "class_type": "MiniMaxH3MotionContextLoadLatent",
            "inputs": {"latent_path": f"h3_context/{load_context_from}",
                       "clip_index": 0},
        }
        graph["mctx"] = {
            "class_type": "MiniMaxH3MotionContext",
            "inputs": {"conditioning": ["cond", 0], "vae": ["vae", 0],
                       "latent": ["cond", 1],
                       "context_length": str(H3MC_CONTEXT_FRAMES),
                       "audio_context_length": H3MC_AUDIO_CONTEXT,
                       "context_latent": ["ctx_lat", 0],
                       "audio_vae": ["avae", 0]},
        }
        graph["guider"]["inputs"]["conditioning"] = ["mctx", 0]
        graph["trim"] = {
            "class_type": "MiniMaxH3MotionContextTrim",
            "inputs": {"images": ["frames", 0], "trim_frames": ["mctx", 1],
                       "audio": ["audio", 0], "fps": H3_FPS,
                       "match_tail": True},
        }
        graph["video"]["inputs"]["images"] = ["trim", 0]
        graph["video"]["inputs"]["audio"] = ["trim", 1]

    # The LoRA stack, chained onto the DiT the way the video graph does it —
    # `LoraLoaderModelOnly` is architecture-agnostic weight patching, so what
    # decides whether a given file does anything is whether its keys map onto
    # this DiT, not whether ComfyUI knows the family. That is why an unmatched
    # LoRA is *reported* (see `_drain`) rather than assumed to have worked: a
    # file whose keys miss loads nothing, changes nothing, and looks exactly
    # like a LoRA that was simply subtle.
    #
    # H3 has one expert, so there is no high/low branch to target and the stack
    # is one chain. `/api/video` already refuses an expert-tagged LoRA for any
    # model whose `supports.experts` is false, which is this one.
    src = "dit"
    for i, lora in enumerate(loras or []):
        graph[f"lora{i}"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": [src, 0], "lora_name": lora["name"],
                       "strength_model": lora["unet"]},
        }
        src = f"lora{i}"
    # After the stack, because a distilled LoRA changes the schedule it wants and
    # the shift has to be the last word on the sampling curve. Absent unless
    # asked for: at the defaults above it is a no-op, and a node that does
    # nothing is a node somebody has to work out is doing nothing.
    if shift_video is not None or shift_audio is not None:
        graph["shift"] = {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {
                "model": [src, 0],
                "shift_video": H3_SHIFT_VIDEO if shift_video is None else float(shift_video),
                "shift_audio": H3_SHIFT_AUDIO if shift_audio is None else float(shift_audio),
            },
        }
        src = "shift"
    if src != "dit":
        # Both, not just the guider. A LoRA that patches model sampling would
        # otherwise have the sampler reading it and the schedule ignoring it,
        # which is a disagreement no error reports — and `MiniMaxH3SigmaShift`
        # is precisely a node that patches model sampling.
        graph["guider"]["inputs"]["model"] = [src, 0]
        graph["sigmas"]["inputs"]["model"] = [src, 0]

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

    # A standalone `<Audio N>` — a voice to clone the timbre of, a track to
    # reuse, a texture to reference. Its own autogrow group, and deliberately
    # *not* the one above: `ref_video_audios` means "this is that clip's
    # soundtrack", which is a claim about a video. An audio dropped on a
    # character is a claim about a person, and putting it in the video group
    # would be telling the model it belongs to a clip nobody attached.
    for i, name in enumerate(ref_audios or []):
        graph[f"refaud{i}"] = {"class_type": "LoadAudio", "inputs": {"audio": name}}
        cond[f"ref_audios.ref_audio_{i}"] = [f"refaud{i}", 0]
    return graph


# --------------------------------------------------------------------------
# What a video model is allowed to ask for
#
# **There is one video family, and there was briefly two.** Wan 2.2 lived beside
# H3 here — same container, same warm process, same job/status/stop contract,
# a graph builder and a table row apart — which was the proof of the claim this
# backend rests on: driving ComfyUI rather than porting model code means a
# second family costs a graph and a table, not an image.
#
# It is gone because it was not worth its room. Everything it was *for* had
# been overtaken: CFG and a negative prompt are levers H3 does not need, the
# LoRA argument that once justified it was wrong in both directions (MiniMax
# ship their own Lightning distillations), and its own training path was tabled
# on a 64 GB duplicate-weights bill nobody was going to pay. What it left
# behind is silent video, half the catalogue, and a console asking every
# question twice.
#
# The dispatch it proved is still here — `requires` is per task, `supports` is
# per model, and the run is family-agnostic. A second family is a row and a
# graph builder, and that is the sentence this section exists to keep true.
# --------------------------------------------------------------------------


# What the composer is allowed to ask for, per video model.
#
# Served to the page rather than written into it, for the reason the GPU list
# already is: which controls a model reads is a property of the model, and a
# copy in the HTML is a second source of truth that drifts the first time one
# of them changes. A control that is present but ignored is worse than one that
# is absent, so absent is what the model says it wants — which is why the keys
# here are the ones something actually reads. `cfg`, `negative` and `experts`
# were in this table and are not any more: with one guidance-distilled family
# they were three permanent `False`s, and a key whose only value is False is a
# branch the page keeps for a model that is not there.
VIDEO_MODELS: dict[str, dict[str, Any]] = {
    "h3": {
        "label": "MiniMax-H3",
        "note": "Sound and picture in one pass",
        "requires": {"fl2va": VIDEO_MODEL_KEYS, "ref2va": VIDEO_REF_MODEL_KEYS},
        "tiers": {"full": "768p", "draft": "544p draft"},
        # **Zero is a length, and that is the point.** A still used to mean
        # Krea 2, so asking for one frame threw away the cast, `<Subject N>`
        # and the audio labels — the duration control was a model switch
        # wearing a time label. Zero is now the short end of this model's own
        # range, so composing does not change surface when the answer is a
        # picture. A model that cannot make one simply has no 0 here, which is
        # why there is no `supports.still` beside it saying the same thing.
        "lengths": [0, 5, 6, 8, 10, 12, 14],
        "samplers": ["res_multistep", "euler", "dpmpp_2m", "sa_solver"],
        "schedulers": ["simple", "normal", "beta"],
        "defaults": {"steps": 20, "sampler": "res_multistep", "scheduler": "simple",
                     "tier": "full", "seconds": 5},
        # `loras: True` because `LoraLoaderModelOnly` patches any MODEL and this
        # platform's other half is a trainer — the ecosystem objection this
        # entry used to encode ("no ecosystem for the int8 repackage") is an
        # argument about what other people publish, not about what the graph can
        # load, and it is the wrong reason to refuse somebody their own weights.
        "supports": {"loras": True, "references": True, "last_frame": True,
                     "audio": True},
    },
}


def _video_model_status() -> list[dict[str, Any]]:
    """
    Every video model with what is actually on the volume for each of its tasks.

    Per task, not per model, because fl2va and ref2va load different
    transformers — reporting one "MiniMax-H3: missing" would send you to
    download 21 GB the run you are composing does not load.
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
#   needs  a `supports` key the video model must carry, so a pill the model
#          cannot read dims rather than disappearing — the established rule from
#          the keyframes/references row. Nothing dims today, because H3 reads
#          every group; it is the column a second family would arrive on. An
#          item may carry its own, which then stands in for the group's: see
#          `say.dialogue`.
#   glyph  the CSS class that animates the shared tile skeleton. Not a drawing:
#          forty bespoke SVGs would drift apart, one skeleton plus a class per
#          move does not.
SHOT_VOCAB: list[dict[str, Any]] = [
    {"key": "framing", "label": "Framing", "pick": "one", "join": "list",
     "slot": 40, "field": "visual", "image": True, "needs": None, "items": [
        {"key": "xwide", "label": "extreme wide", "glyph": "fr-xw",
         "phrase": "in an extreme wide shot"},
        {"key": "wide", "label": "wide", "glyph": "fr-w",
         "phrase": "in a wide shot"},
        {"key": "medium", "label": "medium", "glyph": "fr-m",
         "phrase": "in a medium shot"},
        {"key": "mcu", "label": "medium close-up", "glyph": "fr-mcu",
         "phrase": "in a medium close-up"},
        {"key": "cu", "label": "close-up", "glyph": "fr-cu",
         "phrase": "in a close-up"},
        {"key": "xcu", "label": "extreme close-up", "glyph": "fr-xcu",
         "phrase": "in an extreme close-up"},
        {"key": "ots", "label": "over-the-shoulder", "glyph": "fr-ots",
         "phrase": "in an over-the-shoulder shot"},
        {"key": "pov", "label": "POV", "glyph": "fr-pov",
         "phrase": "in a first-person point-of-view shot"},
    ]},
    {"key": "angle", "label": "Angle", "pick": "one", "join": "list",
     "slot": 40, "field": "visual", "image": True, "needs": None, "items": [
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
     "slot": 30, "field": "visual", "image": True, "needs": None, "items": [
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
     "slot": 30, "field": "visual", "image": True, "needs": None, "items": [
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
        {"key": "pushin", "verb": "pushes in", "label": "push in", "glyph": "ca-push",
         "phrase": "The camera pushes in slowly, a small and steady move."},
        {"key": "pullout", "verb": "pulls out", "label": "pull out", "glyph": "ca-pull",
         "phrase": "The camera pulls out slowly, a small and steady move."},
        {"key": "panl", "verb": "pans left", "label": "pan left", "glyph": "ca-panl",
         "phrase": "The camera pans left at a moderate speed, a medium-amplitude move."},
        {"key": "panr", "verb": "pans right", "label": "pan right", "glyph": "ca-panr",
         "phrase": "The camera pans right at a moderate speed, a medium-amplitude move."},
        {"key": "tiltu", "verb": "tilts up", "label": "tilt up", "glyph": "ca-tiltu",
         "phrase": "The camera tilts up slowly, a small move."},
        {"key": "tiltd", "verb": "tilts down", "label": "tilt down", "glyph": "ca-tiltd",
         "phrase": "The camera tilts down slowly, a small move."},
        {"key": "truckl", "verb": "trucks left", "label": "truck left", "glyph": "ca-truckl",
         "phrase": "The camera trucks left at a steady, moderate speed."},
        {"key": "truckr", "verb": "trucks right", "label": "truck right", "glyph": "ca-truckr",
         "phrase": "The camera trucks right at a steady, moderate speed."},
        {"key": "pedu", "verb": "pedestals up", "label": "pedestal up", "glyph": "ca-pedu",
         "phrase": "The camera pedestals up slowly, a small move."},
        {"key": "pedd", "verb": "pedestals down", "label": "pedestal down", "glyph": "ca-pedd",
         "phrase": "The camera pedestals down slowly, a small move."},
        {"key": "orbit", "verb": "orbits the subject", "label": "orbit", "glyph": "ca-orbit",
         "phrase": "The camera orbits the subject at a slow, steady speed."},
        {"key": "arc", "verb": "arcs around the subject", "label": "arc", "glyph": "ca-arc",
         "phrase": "The camera arcs around the subject in a slow, wide move."},
        {"key": "craneu", "verb": "cranes up", "label": "crane up", "glyph": "ca-craneu",
         "phrase": "The camera cranes up in a large, slow move."},
        {"key": "craned", "verb": "cranes down", "label": "crane down", "glyph": "ca-craned",
         "phrase": "The camera cranes down in a large, slow move."},
        {"key": "trackside", "verb": "tracks alongside the subject", "label": "track side", "glyph": "ca-trackside",
         "phrase": "A stable side-tracking shot keeps pace with the subject."},
        {"key": "trackrear", "verb": "tracks the subject from behind", "label": "track rear", "glyph": "ca-trackrear",
         "phrase": "A stable rear tracking shot follows the subject from behind."},
        {"key": "handheld", "label": "handheld", "glyph": "ca-handheld",
         "phrase": "The camera is handheld, with small, constant, organic movement."},
        {"key": "whip", "verb": "whip-pans", "label": "whip pan", "glyph": "ca-whip",
         "phrase": "The camera whip-pans, a large and fast move."},
        {"key": "rack", "label": "rack focus", "glyph": "ca-rack",
         "phrase": "The focus racks from the foreground to the subject."},
        {"key": "zoom", "verb": "zooms in", "label": "zoom in", "glyph": "ca-zoom",
         "phrase": "The lens zooms in slowly, a small and steady move."},
        # The three the storyboard's arrow language has that the palette did
        # not: every storyboard sheet draws zoom out and rotate, and the guide
        # lists both rolls as motion types. Added here rather than in the
        # page because the vocabulary is served, never written into the page.
        {"key": "zoomout", "verb": "zooms out", "label": "zoom out",
         "glyph": "ca-zoomout",
         "phrase": "The lens zooms out slowly, a small and steady move."},
        {"key": "rollcw", "verb": "rolls clockwise", "label": "roll clockwise",
         "glyph": "ca-rollcw",
         "phrase": "The camera rolls clockwise around the lens axis."},
        {"key": "rollccw", "verb": "rolls counterclockwise",
         "label": "roll counter", "glyph": "ca-rollccw",
         "phrase": "The camera rolls counterclockwise around the lens axis."},
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

# The guide's two other dimensions of a camera move — "motion type +
# amplitude + speed", base-en §4.3 — with its own rule that the middle of each
# is omitted. A camera pill carrying either compiles through `_camera_phrase`
# rather than its hand-written `phrase`; one carrying neither compiles exactly
# as it always did, which is what keeps every existing document unchanged.
# The storyboard is what sets them: an arrow's length is amplitude.
CAMERA_AMPS = ("small", "medium", "large")
CAMERA_SPEEDS = ("slow", "normal", "fast")


def _camera_phrase(item: dict[str, Any], amp: str | None,
                   speed: str | None) -> str:
    """`The camera pans right with large amplitude at fast speed.` — the
    guide's own construction, verbatim in shape, medium and normal omitted."""
    out = f"The camera {item['verb']}"
    if amp == "small":
        out += " with small amplitude"
    elif amp == "large":
        out += " with large amplitude"
    if speed == "slow":
        out += " at slow speed"
    elif speed == "fast":
        out += " at fast speed"
    return out + "."


# Sixteen runs is far more than any clause has needed. The bound exists so a
# payload cannot spell a sentence one character at a time, not because a
# seventeenth run would mean anything.
MAX_SPANS = 16

_ORIGIN_JOINS = frozenset(
    "the a an of in on at to and or with is are it its for from by as".split())


# A child that reads as a continuation of its anchor is the one a bare space
# joins correctly. Anything else is a clause in its own right and needs the
# comma its siblings already get.
_CONTINUES = re.compile(
    r"^(in|on|at|with|without|under|over|behind|beside|against|through|across|"
    r"from|to|by|into|onto|around|beneath|between|holding|wearing|carrying|"
    r"lit|facing|turned|set|seen|shot|leaning|standing|sitting|resting|"
    r"reaching|wrapped|covered|framed|and)\b", re.I)


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
        if item.get("verb"):
            # Only a move with a verb can take them — handheld and a locked-off
            # camera have no amplitude — and only when sent: absence keeps the
            # hand-written phrase, and with it every document written before
            # the storyboard could say "large".
            for field, allowed in (("amp", CAMERA_AMPS), ("speed", CAMERA_SPEEDS)):
                v = entry.get(field)
                if v in (None, ""):
                    continue
                if v not in allowed:
                    raise ValueError(f"No such camera {field}: {v!r}. "
                                     f"One of: {', '.join(allowed)}")
                pill[field] = str(v)
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
             references: Any, ref_videos: Any, ref_audios: Any = None) -> str:
    """
    Which of the guide's four tasks a request is.

    Deliberately a second, finer read than the one `/api/video` makes. That one
    collapses to `ref2va` or `fl2va`, which is exactly right for *which
    checkpoint loads* — first-only, last-only and both are the same weights —
    and too coarse for *which alignment instruction*, where they are three
    different sentences and getting it wrong tells the model a picture sits at a
    timestamp it does not.
    """
    if references or ref_videos or ref_audios:
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
        if item.get("verb") and (pill.get("amp") or pill.get("speed")):
            return _camera_phrase(item, pill.get("amp"), pill.get("speed"))
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


def _flat(text: str) -> str:
    """
    `_oneline` without the strip, because a span's edges are load-bearing.

    Spans tile a clause, so the space between "a ruined city" and "no colour left
    in it" belongs to one of them. `_oneline` strips, which welds the runs into
    "a ruined cityno colour left in it" — and the damage is silent, because the
    offsets it produces are still internally consistent.
    """
    return re.sub(r"\s+", " ", text)


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


def _shot_body(body: "str | list[str]",
               buckets: dict[tuple[int, str, str], list[str]],
               *, field: str = "visual") -> str:
    """
    The user's clauses with the pills folded in around them, in slot order.

    Each clause is closed with a full stop if it does not close itself, and
    otherwise left alone: it is the part of the document the user wrote, and
    rewriting someone's sentence is not something a compiler gets to do.

    **A string and a one-element list compile identically**, which is what makes
    the storyline additive rather than a migration. A plain typed prompt is one
    module; a storyline is several; the pills fold around either at the same
    slots, so nobody who ignores the new surface sees their output change.

    The clauses land together, at the first non-negative slot — they are one
    body with an internal order, not separate things to interleave pills
    between. That order is load-bearing: subjects come out of the model left to
    right in the order they are described, so a compiler that reordered them
    here would be moving people around the frame.
    """
    parts_in = [body] if isinstance(body, str) else list(body)
    pending = [t for t in (_close(_oneline(p)) for p in parts_in) if t]
    out: list[str] = []
    for slot, join, fld in sorted(k for k in buckets if k[2] == field):
        if slot >= 0 and pending:
            out.extend(pending)
            pending = []
        parts = buckets[(slot, join, fld)]
        out.append(_shot_sentence(parts) if join == "list" else " ".join(parts))
    out.extend(pending)
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


# ── the scene ───────────────────────────────────────────────────────────────
#
# What the composer sends instead of a sentence.
#
# The reason it exists is that MiniMax's own guides — `skills/h3-prompt-writing`
# in the model repo, vendored verbatim rather than paraphrased — document a grammar
# far larger than a text field can hold, and every part of it that is missing
# here is missing for the same reason: `_compile_h3_prompt` could not emit it
# because the composer never collected it.
#
#   [Shot N] markers, and cut times that strictly increase
#   speaker IDs assigned by order of vocal event and held across shots
#   <scenetrans> at *both* connecting points when a line crosses a cut
#   a bracketed task-type prefix on `summary`, from a closed table
#   `retention_analysis` as one line per label with a fixed relationship marker
#
# None of those is a sentence anybody could be expected to write. All of them
# are facts an interface knows: a cut time is a boundary between two rows, a
# speaker ID is a cast member with a line, a retention line is the set of shots
# a handle appears in.
#
# **The scene is the input; the document is a receipt.** Same relationship this
# file already draws between `prompt_typed` and the compiled `prompt`, and the
# reason the job record carries the scene as well as the string it produced.

MAX_CAST = 8
MAX_SCENE_SHOTS = 8
MAX_SCENE_LINE = 600

# H3's own limit on the prompt field: 7,000 characters including whitespace.
# It is in neither writing guide — it surfaced in the format scorer vendored
# under `tools/vendor/minimax_score/`, which is the argument for having taken a
# second reading of this format from somebody who trained on it.
#
# Checked after compiling rather than by bounding the inputs, because the
# document is several times what was typed and no bound on a shot line predicts
# it. A refusal naming the overflow beats a truncation nobody sees: the fields
# at the end are `overall_soundscape` and `non_diegetic_music`, so a document
# cut to length loses its audio and still looks well-formed.
MAX_H3_PROMPT = 7000

# The guide's relationship markers, which are *fixed English values in the
# output format* rather than prose we choose — ref-en §4.1 and §4.2 say so in
# those words. Two tables because visual content and audio do not share a
# vocabulary: `fully_copy` is meaningless about a photograph and
# `fully_preserved` is meaningless about a signal.
H3_RETENTION = ("fully_preserved", "partially_preserved",
                "attribute_transfer", "weak_reference")
H3_AUDIO_RETENTION = ("fully_copy", "partially_copy", "reference",
                      "weak_reference")

# `summary` opens with a bracketed task type and the set is closed (ref-en §3).
# Order is the guide's own table order, because several are joined with " + "
# and a stable order is what makes one scene compile to one document twice.
H3_TASK_TYPES = ("keyframe completion", "reference generation",
                 "video editing", "video continuation",
                 "audio reuse", "audio reference")

# What a reference *video* is doing, which is the only thing that can promote a
# task type past `reference generation`. The guide is explicit that the mere
# presence of a video does not create a type: "If a reference video provides
# only camera movement, cuts, or rhythm, it normally belongs to reference
# generation."
#
# **A member's video is always `reference`, on purpose.** Edit and continue are
# properties of the clip, not of anybody in it, so the page says them once — at
# clip level, through `scene.sources` — and offers no role control on a
# member's video row: a second place to say "this clip continues from that
# video" would be a second way to do the first thing. The two rows stay in this
# table because `_validate_scene` accepts them wherever a client puts them; the
# page simply never does.
H3_VIDEO_ROLES = {"reference": "reference generation",
                  "edit": "video editing",
                  "continue": "video continuation"}

# And what an `<Audio N>` is doing. The guide draws the same line here that it
# draws for video: copying the signal and referencing its character are two
# different task types, and it says which marker goes with which — `fully_copy`
# for a reuse, `reference` for a timbre. So the role decides both.
#
# Unlike the video roles this *is* a fact about one member's recording, so its
# control lives on that recording's row — the toggle in `Material.tsx`. It
# validated and compiled from the day the composer landed while nothing on the
# page could set it, which made every recording a timbre reference and the
# reuse half of this table dead; the same page-never-writes-it failure the
# `H3_SOURCES` block records, wearing a different constant.
H3_AUDIO_ROLES = {"reference": ("audio reference", "reference"),
                  "reuse": ("audio reuse", "fully_copy")}

# A slot is a role, and the role is decided by where the file was dropped.
# `noun` builds the guide's own construction — `<Subject N> is the {noun} in
# <Picture M>` — and `retain` builds the clause after the relationship marker.
# This is `SHOT_REF_ROLES` with a kind above it, because a place has no face
# and a character has no architecture, and a flat role list cannot say so.
# **There is one label, and the guide says so outright.** `<Subject N>` covers
# "People, animals, or objects / Scenes, backgrounds, or environments / Clothing,
# props, interfaces, or visual effects / Styles, actions, expressions, or poses"
# (ref-en §2.1) — four *examples* of what can be a subject, not four types of
# subject. The character/place/thing split that used to live here was ours, and
# it was the closed-vocabulary failure this codebase keeps finding: three kinds
# means three sayable things, and a style, a pose or an expression is a perfectly
# legal subject with no box to go in.
#
# What the kinds were buying was a noun in the definition line — `is the person
# in <Picture 1>` — standing exactly where the guide puts the *description*:
#
#     <Subject 1> is the young woman in <Picture 1>, with long dark hair, a blue
#     cardigan, and a thin silver necklace.
#
# So the description does that work now, and a subject with no description says
# only that it is shown, which is the honest amount to claim about a photograph
# nobody has said anything about.
H3_SUBJECT = "subject"

# A slot is the media and nothing else. The guide is equally explicit that the
# relationship is prose rather than a schema — "One subject may be defined by
# multiple reference assets, and one reference asset may provide multiple
# subjects" — with per-asset roles written as a clause:
#
#     <Subject 1> is the woman whose appearance comes from <Picture 1> and whose
#     walking motion comes from <Video 1>.
#
# which is why a reference carries a `note` and no longer a role name.
H3_SLOT_MEDIA = {"image": "image", "audio": "audio", "video": "video"}

# The named slots that used to exist, mapped to the channel they arrived on.
# Kept so a stale tab still validates rather than 400-ing on a vocabulary that
# was retired under it — the same courtesy `lora_path` gets one route up.
H3_LEGACY_SLOTS = {"face": "image", "wardrobe": "image", "body": "image",
                   "establishing": "image", "object": "image", "style": "image",
                   "voice": "audio", "motion": "video"}

# An audio reference is a *sibling* of the subject, not a property of it. The
# guide's own construction is `<Audio 1> is the voice-timbre reference for
# <Subject 1> (S1)` — its own line in `subject_definitions`, its own line in
# `retention_analysis` with an audio marker, and the speaker ID reused rather
# than assigned. So a voice file does not fold into the subject's definition
# the way a second photograph does; it gets a line of its own.
#
# Keyed by role, because a definition that contradicts its own retention line
# is worse than either alone: a `reuse` recording compiled as "the voice-timbre
# reference" while its retention said `fully_copy` — one document, two claims
# about whether the signal itself lands in the clip. Unfindable while nothing
# could set the role; the first compile after the control landed showed it.
H3_AUDIO_NOUN = {"reference": "voice-timbre reference",
                 "reuse": "source audio, reused directly,"}

# Clip-level sources — properties of the clip, not of anybody in it.
#
# They are **not cast references**: a video you are continuing from is not a
# subject's likeness. Each now has a writer on the page, and each writer is the
# gesture that already existed, given the meaning it looked like it had:
#
# - `keyframe`: the first-frame tile, when a scene is live. Sent as
#   `first_frame` beside references it was **silently ignored** — references
#   and keyframes load different transformers and references win — so the
#   storyboard loop (a still, opened as the first frame of a cast clip) died
#   with no error. `readScene` now rides the picture inside `references[]` and
#   names it here, which is the path a reference run actually reads.
# - `continue` / `edit`: the Video tile becomes the Source door under a
#   composer, and the chip's role button swaps between the two — one slot,
#   because this validator refuses the pair as two answers to what the clip is.
#
# `continue` is still not the `Continue` button: that copies an H3MC
# sampler-latent sidecar and passes `load_context_from` — motion carried as a
# *mechanism* — where this is a task type and a `retention_analysis` line,
# motion declared as *grammar*. They can coexist on one run.
#
# This block once read "were unreachable", past tense, directly above the
# constant that closed only the *compiler* half — and every reader after it
# took the whole gap for closed. That is how a validated, compiled, stored
# field went unwritten by any gesture for its whole life: the comment made the
# missing side unsearchable. If a claim in this block goes stale again, fix the
# claim the day it goes stale.
#
# `noun` is the guide's own definition wording, `retain` the parenthetical its
# `retention_analysis` entry carries, and `mark` the relationship marker — a
# structural reference is `weak_reference` because only the pacing survives,
# while a source being edited is `partially_preserved` because most of it does.
H3_SOURCES = {
    "keyframe": {"kind": "image", "task": "keyframe completion",
                 "noun": "keyframe anchor for the target video",
                 "retain": "keyframe anchor", "mark": "fully_preserved"},
    "continue": {"kind": "video", "task": "video continuation",
                 "noun": "source video the target video continues from",
                 "retain": "continuation source", "mark": "weak_reference"},
    "edit": {"kind": "video", "task": "video editing",
             "noun": "source video for the target video edit",
             "retain": "cut and pacing structure", "mark": "partially_preserved"},
}


def _shot_groups(pills: list[dict[str, Any]], *,
                 side: str) -> dict[str, list[str]]:
    """
    The chosen pills by vocabulary *group*, rather than by slot.

    `_shot_phrases` buckets by slot because prose has one axis and the slot is
    the position along it. A shot does not: framing opens it, the camera move is
    its own sentence after the action, and the sound folds into a different
    field entirely. Grouping by slot to get those apart means reading
    `(40, "list", "visual")` and hoping nobody renumbers the table — so this
    asks the question it means.
    """
    chosen = {p["key"]: p for p in pills}
    out: dict[str, list[str]] = {}
    for group in SHOT_VOCAB:
        if side == "image" and not group["image"]:
            continue
        for item in group["items"]:
            pill = chosen.get(f"{group['key']}.{item['key']}")
            if pill is None:
                continue
            text = _shot_text(group, item, pill)
            if text:
                out.setdefault(group["key"], []).append(text)
    return out


def _h3_clock(seconds: float) -> str:
    """`MM:SS.mmm`, which is the cut-time format and not negotiable."""
    minutes, rest = divmod(max(0.0, float(seconds)), 60.0)
    return f"{int(minutes):02d}:{rest:06.3f}"


def _validate_scene(raw: Any, *, n_refs: int, n_vids: int, n_auds: int = 0,
                    seconds: float) -> dict[str, Any] | None:
    """
    Normalise the composer's scene, or say exactly what is wrong with it.

    Importable from the CPU web container for the same reason `_validate_shot`
    is. What it adds over that one is the class of error a pill rail cannot
    have: a cast member pointing at a picture nobody uploaded, a handle used in
    a shot that names nobody, a voice file dropped on a face. Every one of those
    compiles to a *valid* document that quietly says the wrong thing, which is
    the failure this whole layer exists to remove.

    Returns None for "no scene", which is not an error — it is the degrade, and
    the flat typed+pills path stays exactly as it was.
    """
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"Not a scene: {raw!r}")

    shots_in = list(raw.get("shots") or [])
    cast_in = list(raw.get("cast") or [])
    if not shots_in:
        return None
    if len(shots_in) > MAX_SCENE_SHOTS:
        raise ValueError(f"{len(shots_in)} shots; at most {MAX_SCENE_SHOTS}.")
    if len(cast_in) > MAX_CAST:
        raise ValueError(f"{len(cast_in)} in the cast; at most {MAX_CAST}.")

    # ── the cast ────────────────────────────────────────────────────────────
    cast: list[dict[str, Any]] = []
    handles: dict[str, str] = {}
    for entry in cast_in:
        if not isinstance(entry, dict):
            raise ValueError(f"Not a cast member: {entry!r}")
        # There is one kind, because the guide has one label. Whatever a client
        # sends is normalised rather than refused: character/place/thing was our
        # vocabulary and a stale tab still speaks it.
        kind = H3_SUBJECT
        handle = _h3_handle(str(entry.get("name") or ""))
        if not handle:
            raise ValueError("A cast member needs a name — it is the @handle "
                             "the shots refer to it by.")
        if handle in handles:
            raise ValueError(f"Two in the cast are called @{handle}. A handle "
                             f"is how a shot names somebody, so it has to pick "
                             f"one of them.")
        cid = str(entry.get("id") or handle)
        handles[handle] = cid

        marker = str(entry.get("retention") or "fully_preserved")
        if marker not in H3_RETENTION:
            raise ValueError(f"No such retention marker: {marker!r}. "
                             f"One of: {', '.join(H3_RETENTION)}")

        refs: list[dict[str, Any]] = []
        for ref in list(entry.get("refs") or []):
            if not isinstance(ref, dict):
                raise ValueError(f"Not a reference: {ref!r}")
            media = str(ref.get("kind") or "image")
            slots = [str(s) for s in (ref.get("slots") or []) if s]
            if not slots:
                raise ValueError(f"@{handle} has a file with no slot. The slot "
                                 f"is the role — a file with none is a picture "
                                 f"the model is not told what to do with.")
            # A named slot is folded to the channel it arrived on. `face` and
            # `wardrobe` were two ways of saying "a picture of them", and what
            # each picture actually provides is a sentence now — see `note`.
            slots = [H3_LEGACY_SLOTS.get(sl, sl) for sl in slots]
            for slot in slots:
                if slot not in H3_SLOT_MEDIA:
                    raise ValueError(
                        f"@{handle} has a {slot!r} reference. "
                        f"One of: {', '.join(H3_SLOT_MEDIA)}")
                if slot != media:
                    raise ValueError(
                        f"@{handle}'s {slot} reference takes {slot}, not {media}.")
            index = int(ref.get("index") or 0)
            have = {"video": n_vids, "audio": n_auds}.get(media, n_refs)
            if not 0 <= index < have:
                raise ValueError(
                    f"@{handle} points at {media} {index + 1} and "
                    f"{have} {'was' if have == 1 else 'were'} uploaded.")
            role = str(ref.get("role") or "reference")
            if media == "video" and role not in H3_VIDEO_ROLES:
                raise ValueError(f"No such video role: {role!r}. "
                                 f"One of: {', '.join(H3_VIDEO_ROLES)}")
            if media == "audio" and role not in H3_AUDIO_ROLES:
                raise ValueError(f"No such audio role: {role!r}. "
                                 f"One of: {', '.join(H3_AUDIO_ROLES)}")
            # **What this asset provides, in the person's words.** The guide's
            # own construction, and the reason a role name is not needed:
            #   <Subject 1> is the woman whose appearance comes from <Picture 1>
            #   and whose walking motion comes from <Video 1>.
            note = _oneline(str(ref.get("note") or ""))
            if len(note) > SHOT_VALUE_MAX:
                raise ValueError(f"@{handle}'s note on {media} {index + 1} is "
                                 f"{len(note)} characters; at most {SHOT_VALUE_MAX}.")
            # A character sheet is a *typed* reference, not a note. Marking it
            # is what lets the compiler write the citation the format wants —
            # the sheet's views and on-image labels doing the defining — and
            # the division of labour is the platform's one rule restated:
            # templated instruction text is ours, intent is theirs. Only a
            # picture can be one; a "sheet" of audio is a category error worth
            # refusing while the request is still a form.
            sheet = bool(ref.get("sheet"))
            if sheet and media != "image":
                raise ValueError(f"@{handle} marks {media} {index + 1} as a "
                                 f"character sheet, and only an image can be "
                                 f"one.")
            refs.append({"kind": media, "index": index, "slots": slots,
                         "note": note, "sheet": sheet,
                         "role": role})
        cast.append({"id": cid, "kind": kind, "handle": handle,
                     "note": _oneline(str(entry.get("note") or "")),
                     "retention": marker, "refs": refs})

    # ── the shots ───────────────────────────────────────────────────────────
    shots: list[dict[str, Any]] = []
    for i, entry in enumerate(shots_in):
        if not isinstance(entry, dict):
            raise ValueError(f"Not a shot: {entry!r}")
        line = _oneline(str(entry.get("line") or ""))
        # An empty row used to compile to "the shot cuts to the scene" — a
        # sentence describing nothing, traceable to nothing anybody placed.
        # The row is the control; an empty one is a row you have not written
        # yet, and the honest answer is to say so rather than to invent a shot.
        if not line and not (entry.get("say") or {}).get("text"):
            raise ValueError(f"Shot {i + 1} is empty. Write what happens in it, "
                             f"or remove it.")
        if len(line) > MAX_SCENE_LINE:
            raise ValueError(f"Shot {i + 1} is {len(line)} characters; "
                             f"at most {MAX_SCENE_LINE}.")
        pills = _validate_shot(entry.get("pills"))
        for pill in pills:
            group, item = SHOT_ITEMS[pill["key"]]
            if item.get("valued") == "dialogue":
                raise ValueError(
                    "A scene's dialogue belongs to a speaker in the shot, not "
                    "to a pill — the pill has no way to say who is talking, "
                    "and the speaker ID is what H3 reads.")
        try:
            beats = float(entry.get("beats") or 1.0)
        except (TypeError, ValueError):
            raise ValueError(f"Shot {i + 1}'s length is not a number.")
        if beats <= 0:
            raise ValueError(f"Shot {i + 1} has no length.")

        say = entry.get("say") or {}
        if not isinstance(say, dict):
            raise ValueError(f"Not a line of dialogue: {say!r}")
        text = _oneline(str(say.get("text") or ""))
        who = say.get("who")
        who = [who] if isinstance(who, str) else [str(w) for w in (who or [])]
        if text:
            if len(text) > SHOT_VALUE_MAX:
                raise ValueError(f"Shot {i + 1}'s line is {len(text)} "
                                 f"characters; at most {SHOT_VALUE_MAX}.")
            known = {c["id"] for c in cast}
            for w in who:
                if w not in known:
                    raise ValueError(f"Shot {i + 1} is spoken by somebody who "
                                     f"is not in the cast: {w!r}")
            if not who and not say.get("voice"):
                raise ValueError(
                    f"Shot {i + 1} has a line and nobody to say it. Pick a "
                    f"speaker, or describe the voice for an unseen one.")
        lang = str(say.get("lang") or "English")
        if lang not in H3_LANGUAGES:
            raise ValueError(f"No such dialogue language: {lang!r}. "
                             f"One of: {', '.join(H3_LANGUAGES)}")
        stage = _validate_stage(entry.get("stage"),
                                cast_ids={c["id"] for c in cast},
                                kinds={c["id"]: c["kind"] for c in cast})
        shots.append({
            "line": line, "pills": pills, "beats": beats, "stage": stage,
            "say": {"who": who, "text": text, "lang": lang,
                    "voice": _oneline(str(say.get("voice") or "")),
                    "carry": bool(say.get("carry")),
                    "cutoff": bool(say.get("cutoff")),
                    "offscreen": bool(say.get("offscreen"))},
        })

    # ── clip-level sources ──────────────────────────────────────────────────
    raw_src = raw.get("sources") or {}
    if not isinstance(raw_src, dict):
        raise ValueError(f"Not a sources block: {raw_src!r}")
    sources: dict[str, list[int]] = {}
    for key, spec in H3_SOURCES.items():
        got = raw_src.get(key)
        if got in (None, "", []):
            continue
        idx = [got] if isinstance(got, int) else list(got)
        have = n_vids if spec["kind"] == "video" else n_refs
        for i in idx:
            if not isinstance(i, int) or not 0 <= i < have:
                raise ValueError(
                    f"The {key} source points at {spec['kind']} "
                    f"{(i + 1) if isinstance(i, int) else i!r} and "
                    f"{have} {'was' if have == 1 else 'were'} uploaded.")
        if key != "edit" and len(idx) > 1:
            raise ValueError(f"There is one {key} source, not {len(idx)}.")
        sources[key] = idx
    # The guide's own precedence, and the reason it is a refusal here rather
    # than a ranking: continuing a clip and editing it are different task types
    # and different documents, and picking one for somebody silently is the
    # class of guess this compiler does not make.
    if "continue" in sources and "edit" in sources:
        raise ValueError("A clip is either continued from or edited, not both — "
                         "they are different tasks and different documents.")

    # A handle nobody defined is the one failure that reads as the model
    # ignoring you: it compiles to the literal characters "@ava", which the
    # encoder renders as nothing at all.
    for i, s in enumerate(shots):
        for handle in _h3_handles(s["line"]):
            if handle not in handles:
                raise ValueError(
                    f"Shot {i + 1} mentions @{handle} and nobody in the cast "
                    f"is called that.")

    # The last shot's dialogue is the only one that can be cut off by the end
    # of the clip, and only the last one may carry — there is nothing after it.
    if shots[-1]["say"]["carry"]:
        raise ValueError("The last shot's line cannot carry across a cut; "
                         "there is no shot after it.")
    return {"cast": cast, "shots": shots, "sources": sources,
            "seconds": float(seconds),
            "style": _oneline(str(raw.get("style") or "")),
            "grade": _oneline(str(raw.get("grade") or "")),
            "handles": handles}


_H3_HANDLE = re.compile(r"@([a-z0-9_]+)", re.I)


def _h3_handle(name: str) -> str:
    """A name reduced to the characters a mention can be written in."""
    return re.sub(r"^_+|_+$", "", re.sub(r"[^a-z0-9_]+", "_", name.lower()))


def _h3_handles(line: str) -> list[str]:
    """The handles a shot mentions, in order, once each."""
    out: list[str] = []
    for m in _H3_HANDLE.finditer(line):
        h = m.group(1).lower()
        if h not in out:
            out.append(h)
    return out


# `overall_soundscape` has no safe empty value, which is the opposite of
# `non_diegetic_music` and the reason they are not one rule. The guide is
# explicit that `N/A` there means *complete silence, requested* — so emitting it
# because nobody picked a sound pill tells the model every ordinary clip is
# silent. It also warns that a blank field lets ambience creep in on its own.
# This is the one sentence that is true of every scene and specific about none.
H3_SOUNDSCAPE_DEFAULT = ("Ambient sound consistent with the scene continues "
                         "throughout the video.")


def _h3_subjects(cast: list[dict[str, Any]],
                 shots: "list[dict[str, Any]] | None" = None) -> dict[str, int]:
    """
    `<Subject N>` by cast id — and only for cast that carry a reference.

    A subject label is a claim that something in the target video comes from an
    uploaded asset. Somebody described in the prose and nowhere else is not one,
    and giving them a label points the model at a picture that does not exist.

    **Numbered by order of first mention, not by the order the cast was built
    in.** These are different films:

        two men fight inside a corridor that is constantly rotating
        a spinning corridor bounces two men around as they fight

    Same three subjects, same world, same photographs. What differs is which
    noun the sentence opens on — and in the second one the men are not even
    agents, they are things that get bounced. Nobody has to be asked which the
    scene is *about*, because the sentence already said, and `[Shot N]` is read
    in order top to bottom. Numbering off the cast array threw that away and
    handed `<Subject 1>` to whoever happened to be created first, which on the
    composer is whichever `@` you typed first *anywhere* — so writing shot two
    before shot one silently inverted it.

    Anyone visible but never mentioned still gets a number, appended in cast
    order: a location carrying an establishing photograph earns a subject label
    without any line naming it, and so does a character who only ever speaks.

    `shots` is optional so the ordering is a refinement rather than a
    precondition — a caller with no timeline still gets the old, stable answer
    instead of an exception.
    """
    visible = [c for c in cast
               if any(r["kind"] != "audio" for r in c["refs"])]
    if not shots:
        return {c["id"]: i + 1 for i, c in enumerate(visible)}
    by_handle = {c["handle"]: c["id"] for c in visible}
    order: list[str] = []
    for shot in shots:
        for handle in _h3_handles(shot["line"]):
            cid = by_handle.get(handle)
            if cid is not None and cid not in order:
                order.append(cid)
    order += [c["id"] for c in visible if c["id"] not in order]
    return {cid: i + 1 for i, cid in enumerate(order)}


def _h3_label(member: dict[str, Any], subjects: dict[str, int]) -> str:
    """
    How a cast member is written in the body.

    A subject gets its label. Anybody else gets what the person typed — their
    note if they wrote one, and otherwise their name — because inventing "the
    young woman" for a name we were given is the compiler writing prose, and
    upper-casing `ava` into `Ava` is the same edit `_shot_join` refuses to make.
    """
    n = subjects.get(member["id"])
    return f"<Subject {n}>" if n else (member["note"] or member["handle"])


def _h3_speakers(shots: list[dict[str, Any]]) -> dict[str, str]:
    """
    `(Sx)` by cast id, assigned once by order of actual vocal event.

    The guide's rule exactly: a speaker keeps the same ID across shots, and
    **characters who never vocalise receive no speaker ID at all**. That last
    clause is the one worth guarding — a scene bucket or a prop with an `(S)`
    on it is a speaker H3 will look for and never find.
    """
    out: dict[str, str] = {}
    for shot in shots:
        if not shot["say"]["text"]:
            continue
        for who in shot["say"]["who"]:
            if who not in out:
                out[who] = f"S{len(out) + 1}"
    return out


def _h3_resolve(line: str, cast: list[dict[str, Any]],
                subjects: dict[str, int]) -> str:
    """
    Mentions swapped for what the encoder should read, and nothing else touched.

    The article in front of a mention is absorbed when it resolves to a
    `<Subject N>`, which is already definite — otherwise "in the @diner"
    compiles to "in the <Subject 2>". No case is changed anywhere: the rest of
    the line is the person's sentence and rewriting it is not something a
    compiler gets to do.
    """
    by_handle = {c["handle"]: c for c in cast}

    def swap(m: "re.Match[str]") -> str:
        member = by_handle.get(m.group(2).lower())
        if not member:
            return m.group(0)
        label = _h3_label(member, subjects)
        article = m.group(1) or ""
        return label if (article and label.startswith("<Subject")) \
            else article + label

    return re.sub(r"(\b(?:the|a|an)\s+)?@([a-z0-9_]+)", swap, line, flags=re.I)


def _h3_task_types(scene: dict[str, Any], task: str) -> str:
    """
    The bracketed prefix `summary` opens with, from the guide's closed table.

    Derived rather than asked for, because every input to it is already a fact
    of the request: a keyframe is a keyframe, and what a reference video is
    *doing* is the one thing that can promote past `reference generation`. The
    guide's own warning is the rule encoded here — the mere presence of a video
    does not create a type.
    """
    kinds: set[str] = set()
    if task in ("i2va", "fl2va", "l2va"):
        kinds.add("keyframe completion")
    for key in scene.get("sources") or {}:
        kinds.add(H3_SOURCES[key]["task"])
    for member in scene["cast"]:
        for ref in member["refs"]:
            if ref["kind"] == "video":
                kinds.add(H3_VIDEO_ROLES[ref["role"]])
            elif ref["kind"] == "audio":
                kinds.add(H3_AUDIO_ROLES[ref["role"]][0])
            else:
                kinds.add("reference generation")
    ordered = [t for t in H3_TASK_TYPES if t in kinds]
    return f"[{' + '.join(ordered)}] " if ordered else ""


# ── blocking ────────────────────────────────────────────────────────────────
#
# Marks on a floor and a camera looking at them, and the sentences that fall
# out. This is the one thing in the composer that is *not* a vocabulary: no
# pill can say `she stands behind him, turned three-quarters away from the
# lens`, and across all 77 items in SHOT_VOCAB there is no way to say a subject
# is behind, beside, facing away, or screen left. The only place a subject is
# ever named in output is the hardcoded `<Subject 1>` in `_shot_text`.
#
# **Blocking does not add a vocabulary. It drives the one that exists.**
# Framing, angle and the camera move are *derived* here and fold through
# `_shot_phrases` exactly as a clicked pill would — an explicitly chosen pill
# still wins, because a person overriding a derivation is a person who has
# looked at it. What is genuinely new is only what geometry alone can say:
# where in the frame, how far back, which way they face, and how they stand to
# each other.
#
# The precedent is already shipping on the image side — `_compose_caption`
# turns rectangles into prose, and `_box_framing` already treats box height as
# a depth proxy. That is an approximated pinhole camera in production; this is
# the real relation.
#
# Coordinates: metres on a ground plane. `x` lateral (right positive), `z` away
# from the origin, `y` height. `yaw` in degrees, 0 looking along +z, clockwise
# seen from above. A 36x24mm sensor, so `lens` is the millimetre number
# everybody already thinks in.

STAGE_SENSOR_W = 36.0
STAGE_SENSOR_H = 24.0
STAGE_LENS = 35.0
# A standing adult and their eyeline. The whole size ladder is a ratio against
# frame height, so these two numbers set where every band falls — they are the
# closest thing this file has to a magic constant and they are worth naming
# rather than inlining.
STAGE_FIGURE_H = 1.7
STAGE_FIGURE_W = 0.5
STAGE_EYE_H = 1.6
STAGE_EYE_RATIO = STAGE_EYE_H / STAGE_FIGURE_H
STAGE_MAX_MARKS = 8

# Frame-height fractions. A person 2.4x the frame's height is a close-up
# because their head fills it; at 0.45 they are a figure in a landscape.
# Checked against a 35mm lens: 1m -> close-up, 2m -> medium, 3m -> wide,
# 6m -> extreme wide, which is what those distances look like through one.
STAGE_SIZE = ((4.0, "xcu"), (2.4, "cu"), (1.5, "mcu"),
              (1.0, "medium"), (0.45, "wide"), (0.0, "xwide"))

# Camera pitch onto the subject's eyeline, in degrees. The dead band is wide
# because a camera 10cm off eyeline at three metres is eye level to anybody
# watching, and calling it a low angle would be the arithmetic overruling the
# picture.
STAGE_ANGLE = ((45.0, "bird"), (7.0, "high"), (-7.0, "eye"),
               (-45.0, "low"), (-181.0, "worm"))

# How a subject stands to the lens, by the angle between where they face and
# where the camera is. This is the band that earns the whole feature: "I
# couldn't see his face at all" is the fact CLAUDE.md records the old parse
# dropping, and it is `back to the lens` here.
# Descending, like every other band table here, and it is worth saying why
# rather than leaving it as house style: `_stage_band` returns the first row
# whose edge the value clears, so an *ascending* table never matches and
# silently returns its last row. Written ascending, this one reported a subject
# looking straight down the barrel as having their back to the lens — the
# arithmetic was right and the lookup was inverted, which is the shape of bug
# that survives review and dies the moment somebody prints it.
STAGE_FACING = ((155.0, "with their back to the lens"),
                (115.0, "turned three-quarters away from the lens"),
                (65.0, "in profile to the lens"),
                (25.0, "turned three-quarters toward the lens"),
                (0.0, "facing the lens"))

# Tilt below which the camera sees the body it is riding. Anatomy rather than
# taste: eyes sit above and in front of the chest, so a level or raised gaze
# contains none of you and a lowered one picks up torso, then hands, then feet.
#
# It is here because a video model read this scene's POV act as a free camera
# move — correctly, on the pixels. **POV is legible only when your own limbs
# are in frame**, and Enter the Void's is legible to a person because they
# watched him get shot, which is context no encoder has. So a shot that rides a
# body and looks away from it renders as a camera that happens to be there, and
# the word "point-of-view" in the prompt is carrying weight it cannot hold.
STAGE_OWN_BODY = -25.0

# Metres between two people, as a body would read it rather than as a number.
STAGE_NEAR = ((3.0, "across the space from {other}"),
              (1.2, "a few steps from {other}"),
              (0.6, "within arm's reach of {other}"),
              (0.0, "close enough that their shoulders overlap"))


def _stage_norm(deg: float) -> float:
    """An angle folded into (-180, 180], where the bands are written."""
    d = (float(deg) + 180.0) % 360.0 - 180.0
    return d + 360.0 if d <= -180.0 else d


def _stage_band(value: float, table) -> str:
    for edge, name in table:
        if value >= edge:
            return name
    return table[-1][1]


def _stage_fov(lens: float, sensor: float) -> float:
    """Half-angle, in radians — every projection below wants the half."""
    return math.atan(sensor / (2.0 * max(1.0, float(lens))))


def _stage_dims(mark: dict[str, Any]) -> tuple[float, float, float, float]:
    """
    How big a mark is, how far off the floor it starts, and where it is aimed.

    A mark used to *be* a standing adult: `STAGE_FIGURE_H` and `STAGE_EYE_H`
    were read straight out of the projection, so every subject was 1.7m tall
    with its eyeline at 1.6 whatever it was. That is the one assumption this
    feature could not keep. A subject is whatever the shot is about — the
    reference that broke it has a body lying on tiles and a ceiling light
    fixture in the same frame, and the light is what the camera is looking at.
    So the figure constants are defaults now and the dimensions ride on the
    mark: a standing adult is 1.7 x 0.5 at base 0, a body on the floor is
    0.4 x 1.8 at base 0, a fixture is 0.4 x 0.4 at base 2.7.

    The aim point is the eyeline for a person; the same ratio on anything else
    lands just above centre, which is inside the dead band of every angle
    reading and does not need a rule of its own.
    """
    h = float(mark.get("h") or STAGE_FIGURE_H)
    w = float(mark.get("w") or STAGE_FIGURE_W)
    base = float(mark.get("base") or 0.0)
    return h, w, base, base + h * STAGE_EYE_RATIO


def _stage_see(cam: dict[str, Any], mark: dict[str, Any]) -> dict[str, Any]:
    """
    One mark through the camera: how far, how big, where in frame, facing what.

    Everything downstream reads this dict, so the trigonometry happens once and
    the sentence writers stay readable.
    """
    h, w, base, aim = _stage_dims(mark)
    dx = float(mark["x"]) - float(cam["x"])
    dz = float(mark["z"]) - float(cam["z"])
    # Two distances, and conflating them cost a whole shot. `flat` is the plan
    # view and is what bearing and pitch are measured against; `dist` is the
    # real one. Size used to read `flat`, so a camera craned three metres
    # straight up over somebody had not moved at all as far as the framing was
    # concerned — the derivation reported the same close-up from the floor and
    # from the ceiling.
    flat = math.hypot(dx, dz)
    dist = math.hypot(flat, aim - float(cam["y"]))
    # Bearing off the camera's own forward, not off world north.
    bearing = _stage_norm(math.degrees(math.atan2(dx, dz)) - float(cam["yaw"]))

    half_h = _stage_fov(cam["lens"], STAGE_SENSOR_W)
    half_v = _stage_fov(cam["lens"], STAGE_SENSOR_H)
    # -1 is the left edge of frame, +1 the right. Behind the camera reads as
    # off-frame rather than as a wrapped angle, which `tan` would otherwise do
    # silently and put somebody back in shot facing the wrong way.
    behind = abs(bearing) >= 90.0
    sx = 9.9 if behind else math.tan(math.radians(bearing)) / math.tan(half_h)

    # What fraction of the frame the thing spans, on whichever axis it spans
    # most. Height alone was right while every mark was a standing figure and
    # reads a body lying down as a wide shot from a metre away.
    d = max(0.05, dist)
    fw = w / (2.0 * d * math.tan(half_h))
    fh = h / (2.0 * d * math.tan(half_v))
    fill = max(fh, fw)
    # Vertical screen position, measured off the camera's own axis. -1 is the
    # top edge, +1 the bottom, exactly as `sx` runs left to right. There was no
    # `sy` while every lens pointed at the horizon and every mark was the same
    # height — and the two frame tests that grew up in its absence disagreed the
    # moment either changed: the projection culled a ceiling fixture above the
    # top edge while `in_frame`, horizontal alone, had the clause calling it
    # centre frame.
    rise = math.atan2(base + h / 2.0 - float(cam["y"]), max(0.05, flat))
    sy = -(math.tan(rise - math.radians(float(cam.get("tilt") or 0.0)))
           / math.tan(half_v))
    pitch = math.degrees(math.atan2(float(cam["y"]) - aim, max(0.05, flat)))
    # Where the mark faces, measured against where the camera is standing.
    to_cam = math.degrees(math.atan2(-dx, -dz))
    facing = abs(_stage_norm(float(mark["yaw"]) - to_cam))

    # In frame if any of it is, on both axes — the object's own extent counts.
    # A centre-based test is right for neither: a close-up puts a standing
    # figure's midpoint below the bottom edge while their head and shoulders
    # fill the shot.
    # **Behind is behind, whatever the arithmetic says.** `sx` carries 9.9 as a
    # sentinel rather than a coordinate, and allowing an object its own width
    # either side of the edge is what let that sentinel through: something close
    # enough subtends more than 8.9 frames, so a body the camera was standing
    # inside came back in frame and in extreme close-up.
    return {"dist": dist, "flat": flat, "sx": sx, "sy": sy, "fill": fill,
            "fw": fw, "fh": fh, "pitch": pitch, "behind": behind,
            "facing": facing,
            "in_frame": (not behind and abs(sx) <= 1.0 + fw
                         and abs(sy) <= 1.0 + fh),
            "h": h, "w": w, "base": base, "aim": aim,
            "size": _stage_band(fill, STAGE_SIZE),
            "angle": _stage_band(pitch, STAGE_ANGLE)}


def _stage_where(seen: dict[str, Any]) -> str:
    """Where in frame, and how far back — the two a pill cannot say."""
    sx = seen["sx"]
    if not seen["in_frame"]:
        # Which way out, now that there are two ways. Left and right were the
        # only answers while `in_frame` was a horizontal test, so a ceiling
        # fixture above the top edge was reported as off to one side.
        if abs(sx) > 1.0 + seen["fw"]:
            return "just off-frame left" if sx < 0 else "just off-frame right"
        return ("just above the frame" if seen["sy"] < 0
                else "just below the frame")
    side = ("screen left" if sx < -0.33
            else "screen right" if sx > 0.33 else "centre frame")
    depth = ("in the foreground" if seen["fill"] >= 1.5
             else "in the background" if seen["fill"] < 0.45 else "")
    return f"{side}, {depth}" if depth else side


def _stage_pills(stage: dict[str, Any], seconds: float,
                 chosen: set[str]) -> list[dict[str, Any]]:
    """
    Framing, angle and the camera move, read off the arrangement.

    Emitted as ordinary pills so they fold through `_shot_phrases` at their
    usual slots and nothing downstream learns a new shape. **A pill the person
    picked is never overwritten** — `chosen` is checked per group, because
    overriding a derivation is a decision and the arithmetic does not get to
    take it back.
    """
    marks = stage["marks"]
    out: list[dict[str, Any]] = []
    if not marks:
        return out
    a = _stage_read(stage["camera"], marks)
    b = _stage_read(_stage_end(stage["camera"], stage.get("path")), marks)
    # **Only when the two ends agree.** A pill is a steady state, so a shot
    # whose framing or angle changes has no pill that is true of it, and
    # emitting the opening one was the derivation describing the first frame
    # and calling it the shot. `_stage_arc` says the transition instead.
    if "framing" not in chosen and a[0] and a[0] == b[0]:
        out.append({"key": f"framing.{a[0]}"})
    if "angle" not in chosen and a[1] and a[1] == b[1]:
        out.append({"key": f"angle.{a[1]}"})
    return out


def _stage_others(cam: dict[str, Any],
                  marks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    The marks the camera can see, which is every mark except the one it *is*.

    `camera.on` binds the camera to a body, and the body then stops being a
    subject: you cannot see yourself, and asking `_stage_see` anyway returns
    `behind`, `sx = 9.9` and a distance of zero, which the lead picker reads as
    the nearest thing in the room and describes in extreme close-up.
    """
    rider = cam.get("on")
    return [m for m in marks if m.get("castId") != rider] if rider else marks


def _stage_lead(cam: dict[str, Any],
                marks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    The subject the shot is about: whatever fills most of the frame.

    Nearest was the rule while every mark was a person, and it inverts the
    moment they are not the same size. A bare bulb half a metre off the lens is
    nearer than the body on the floor three metres below it, so "nearest"
    called an overhead shot of a dying man an extreme wide shot of a light
    fixture. Prominence is the question a shot size is an answer to.

    **Nothing in frame means no lead, and no lead means no framing.** This used
    to fall back to the nearest mark overall, on the reasoning that a camera
    looking slightly away is still a shot of somebody — survivable while
    `in_frame` was a horizontal test and every mark stood on one floor. Once it
    was true on both axes the fallback started answering with whoever was
    *behind the lens*: stepping out of a body leaves the camera standing inside
    it, and the shot came back "nobody in frame. In an extreme close-up". A
    shot with no subject has no shot size, and stating one is worse than
    stating nothing.
    """
    framed = [s for s in (_stage_see(cam, m) for m in _stage_others(cam, marks))
              if s["in_frame"]]
    return max(framed, key=lambda s: s["fill"]) if framed else None


def _stage_read(cam: dict[str, Any],
                marks: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    """
    Framing and angle from one camera position.

    POV is a framing value like any other, which is what lets a shot open in
    one and end out of it. The angle still comes off the lead, and that is the
    point rather than an oversight: a camera lying on a bathroom floor looking
    up at a light fixture is a point-of-view shot *and* a worm's-eye view, and
    the second half is only knowable because the fixture is a mark.
    """
    lead = _stage_lead(cam, marks)
    angle = lead["angle"] if lead else None
    if cam.get("on"):
        return "pov", angle
    return (lead["size"] if lead else None), angle


def _stage_end(cam: dict[str, Any],
               path: dict[str, Any] | None) -> dict[str, Any]:
    """The camera where the path leaves it, in the same shape as the start."""
    if not path:
        return cam
    out = dict(cam)
    for k in ("x", "z", "y", "lens"):
        if path.get(k) is not None:
            out[k] = float(path[k])
    for k in ("yaw", "tilt"):
        if path.get(k) is not None:
            out[k] = _stage_norm(float(path[k]))
    # Present-and-null is the whole gesture. The reference this came from is
    # one continuous take that is a man's point of view until his soul leaves
    # the body — a camera detaching from a mark, not a cut, and not something
    # any pill can say.
    if "on" in path:
        out["on"] = path["on"] or None
    return out


def _stage_phrase(group: str, key: str | None) -> str:
    """A vocabulary phrase by key, so a derived sentence and a clicked pill
    say the same words."""
    if not key:
        return ""
    for g in SHOT_VOCAB:
        if g["key"] == group:
            for item in g["items"]:
                if item["key"] == key:
                    return item["phrase"]
    return ""


def _stage_arc(stage: dict[str, Any], chosen: set[str]) -> str:
    """
    What the shot becomes, when the move changes it.

    The guide asks for "continuous development -> result" and the derivation
    was reading `camera` and never `path`, so it stated the opening framing and
    stopped. Silent when nothing changes — a shot that holds its framing is
    described by its pills, and a sentence saying so twice is the enrichment
    this compiler exists not to do.
    """
    marks = stage.get("marks") or []
    if not marks or not stage.get("path"):
        return ""
    a = _stage_read(stage["camera"], marks)
    b = _stage_read(_stage_end(stage["camera"], stage["path"]), marks)
    moved = [i for i in (0, 1) if a[i] and b[i] and a[i] != b[i]]
    # A group the person set by hand is theirs, and a derived sentence that
    # contradicts their pill is worse than one that never ran.
    moved = [i for i in moved if ("framing", "angle")[i] not in chosen]
    if not moved:
        return ""

    def side(read, which):
        # Comma, not `_shot_join`: these are clauses inside one sentence, and
        # joining them as sentences ran "in a first-person point-of-view shot
        # shot from ground level" with nothing between them.
        return ", ".join(p for p in (
            _stage_phrase("framing", read[0]) if 0 in which else "",
            _stage_phrase("angle", read[1]) if 1 in which else "") if p)

    return _close(f"The shot opens {side(a, moved)} and ends "
                  f"{side(b, moved)}")


def _stage_move(cam: dict[str, Any], path: dict[str, Any],
                dist: float, seconds: float) -> str:
    """
    Which camera move a path *is*.

    Read as displacement in the camera's own frame rather than the world's: a
    metre along its forward axis is a push, a metre across it is a truck, and
    turning on the spot is a pan. That decomposition is why this can answer
    amplitude and speed at all — **8 of the 21 hand-written camera pills state
    neither**, and `trackside`/`trackrear` state nothing but a direction.
    """
    dx = float(path.get("x", cam["x"])) - float(cam["x"])
    dz = float(path.get("z", cam["z"])) - float(cam["z"])
    dy = float(path.get("y", cam["y"])) - float(cam["y"])
    dyaw = _stage_norm(float(path.get("yaw", cam["yaw"])) - float(cam["yaw"]))
    dtilt = _stage_norm(float(path.get("tilt", cam.get("tilt") or 0.0))
                        - float(cam.get("tilt") or 0.0))

    yaw = math.radians(float(cam["yaw"]))
    tilt = math.radians(float(cam.get("tilt") or 0.0))
    plan = dx * math.sin(yaw) + dz * math.cos(yaw)
    lateral = dx * math.cos(yaw) - dz * math.sin(yaw)
    # **In the camera's own frame, all three axes of it.** This function's own
    # docstring has always claimed that and did it in two dimensions: vertical
    # was compared against the *world's* up, which is the same thing only while
    # the lens points at the horizon. Once a camera can tilt, a camera aimed at
    # the floor and dropping toward a body on it is travelling along its own
    # forward axis — a push in — and the world-axis test called it a crane
    # down. Enter the Void's overhead is exactly that shot, and a video model
    # asked to describe it independently said "closer framing without a major
    # change in vertical viewpoint", which is the push and not the crane.
    forward = plan * math.cos(tilt) + dy * math.sin(tilt)
    rise = dy * math.cos(tilt) - plan * math.sin(tilt)
    travel = math.hypot(math.hypot(dx, dz), dy)

    # Nothing moved and nothing turned. `static` is a real answer, not a
    # fallthrough — a locked-off camera is a choice H3 reads.
    if travel < 0.05 and abs(dyaw) < 3.0 and abs(dtilt) < 3.0:
        return "static"
    # Turning on the spot, on whichever axis turned further. `tiltu`/`tiltd`
    # were in the vocabulary from the start and unreachable: this read yaw and
    # nothing else, so for as long as the camera could not tilt that was the
    # whole truth, and the moment it could, a tilt-only shot came back locked
    # off.
    if travel < 0.05:
        if abs(dtilt) > abs(dyaw):
            return "tiltu" if dtilt > 0 else "tiltd"
        return "panr" if dyaw > 0 else "panl"
    if abs(rise) > max(abs(forward), abs(lateral)):
        return "craneu" if rise > 0 else "craned"
    # An arc is a truck that keeps the subject centred, so the yaw has to have
    # turned *with* the move rather than against it.
    if abs(lateral) > abs(forward) and abs(dyaw) > 12.0:
        return "arc"
    if abs(lateral) > abs(forward):
        return "truckr" if lateral > 0 else "truckl"
    return "pushin" if forward > 0 else "pullout"


# The verb for each move. The pills' own phrases already carry an amplitude and
# a speed baked in ("pushes in slowly, a small and steady move"), so a blocked
# shot cannot reuse them — appending a measured amplitude produced "pushes in
# slowly, a small and steady move, a medium-amplitude move, quickly", which
# contradicts itself twice in one sentence. Blocking states all three
# dimensions itself, in the guide's own construction.
STAGE_VERB = {"pushin": "pushes in", "pullout": "pulls out",
              "panl": "pans left", "panr": "pans right",
              "tiltu": "tilts up", "tiltd": "tilts down",
              "truckl": "trucks left", "truckr": "trucks right",
              "craneu": "cranes up", "craned": "cranes down",
              "arc": "arcs around the subject", "static": "holds a static shot"}


def _stage_move_sentence(cam: dict[str, Any], path: dict[str, Any],
                         dist: "float | None", seconds: float) -> str:
    """The camera move with motion type, amplitude and speed — all three."""
    key = _stage_move(cam, path, dist, seconds)
    verb = STAGE_VERB.get(key, "moves")
    if key == "static":
        return "The camera holds a static shot."
    note = _stage_move_note(cam, path, dist, seconds)
    return f"The camera {verb}{(' ' + note) if note else ''}."


def _stage_move_note(cam: dict[str, Any], path: dict[str, Any],
                     dist: "float | None", seconds: float) -> str:
    """
    The amplitude and speed the pill's own phrase cannot carry.

    Amplitude is relative to how far away the subject is, because a metre is a
    large move at two metres and nothing at twenty. Speed is metres per second
    over the shot, which is the one place `seconds` is load-bearing.

    **With nobody in frame there is no amplitude**, only a speed. `dist` used to
    be a float that fell back to zero, which `max(0.5, dist)` then turned into
    "large amplitude" for any move over a quarter of a metre — an amplitude
    measured against a subject that is not there.
    """
    dx = float(path.get("x", cam["x"])) - float(cam["x"])
    dz = float(path.get("z", cam["z"])) - float(cam["z"])
    dy = float(path.get("y", cam["y"])) - float(cam["y"])
    travel = math.hypot(math.hypot(dx, dz), dy)
    if travel < 0.05:
        return ""
    rate = travel / max(0.5, float(seconds))
    speed = "fast" if rate > 0.8 else "slow" if rate < 0.25 else "moderate"
    if dist is None:
        return f"at {speed} speed"
    ratio = travel / max(0.5, float(dist))
    amp = ("large" if ratio > 0.5 else "small" if ratio < 0.18 else "medium")
    # The guide's own construction, verbatim: "with small amplitude at slow
    # speed". Not phrasing we chose.
    return f"with {amp} amplitude at {speed} speed"


def _stage_clauses(stage: dict[str, Any], label) -> list[str]:
    """
    One clause per body — where they are, which way they face, how they stand
    to whoever is nearest.

    **Ordered by screen position, left to right.** `_shot_body` already
    records that subjects come out of the model in the order they are
    described, and until now that order was whatever the person happened to
    type. An arrangement knows the real one.

    The relation **trails**, which is CLAUDE.md's rule: the subject opens the
    clause and how they stand to the others closes it, so the last thing read
    before the encoder moves on is what binds them.

    Naming the other subject is safe *here* specifically because it is a
    `<Subject N>` label rather than a fresh noun phrase — the guide's own
    examples do it (`<Subject 4> … enters holding the leash of <Subject 2>`),
    and a label is a reference, not a second attention site.
    """
    cam, marks = stage["camera"], _stage_others(stage["camera"], stage["marks"])
    seen = [(m, _stage_see(cam, m)) for m in marks]
    on = [(m, s) for m, s in seen if s["in_frame"]]
    on.sort(key=lambda ms: ms[1]["sx"])

    out: list[str] = []
    # The rider's own body, and it leads because it is the nearest thing in the
    # shot and the only thing that says whose eyes these are. Excluding the
    # rider outright was right about their face and wrong about the rest of
    # them — see `STAGE_OWN_BODY`.
    rider = cam.get("on")
    if rider and float(cam.get("tilt") or 0.0) < STAGE_OWN_BODY:
        out.append(f"{label(rider)}'s own arms and torso across the bottom of "
                   f"the frame, seen from their own eyes")
    if not on:
        return out
    for mark, s in on:
        who = label(mark["castId"])
        bits = [f"{who} {_stage_where(s)}"]
        if mark.get("faces", True):
            bits.append(_stage_band(s["facing"], STAGE_FACING))
        # Nearest *other* body, if there is one. One relation per subject —
        # a clause that relates everybody to everybody is a paragraph.
        others = [(o, os_) for o, os_ in seen if o is not mark]
        if others:
            # **In three dimensions.** Two people stand on the same floor, so
            # the plan distance was the whole answer for as long as every mark
            # was a person — and a bare bulb on the ceiling is 1.1m away across
            # the floor and 2.6m straight up, which came out as "within arm's
            # reach of the bulb".
            def gap_to(other: dict[str, Any]) -> float:
                return math.hypot(
                    math.hypot(float(other["x"]) - float(mark["x"]),
                               float(other["z"]) - float(mark["z"])),
                    _stage_dims(other)[3] - _stage_dims(mark)[3])
            near = min(others, key=lambda oo: gap_to(oo[0]))
            bits.append(_stage_band(gap_to(near[0]), STAGE_NEAR)
                        .format(other=label(near[0]["castId"])))
        out.append(", ".join(b for b in bits if b))
    return out


def _stage_boxes(stage: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Marks projected onto the image plane, as the rectangles regions already are.

    **This is the same arrangement the prose comes from, seen the other way.**
    A region is a normalised 0..1 frame-space box paired positionally to a LoRA
    — exactly what a mark becomes once you look through the camera at it — so
    blocking reaches Krea 2 as a projection rather than as a second feature.

    Clamped rather than rejected, the way `_validate_regions` clamps: a body
    half out of frame is a real shot, and the visible half is the right box for
    it. A mark entirely outside the frustum yields nothing at all, because a
    zero-area region is an error downstream and an off-screen body is not a
    request for one.
    """
    cam = stage["camera"]
    half_h = _stage_fov(cam["lens"], STAGE_SENSOR_W)
    half_v = _stage_fov(cam["lens"], STAGE_SENSOR_H)
    out: list[dict[str, Any]] = []
    for mark in _stage_others(cam, stage["marks"]):
        seen = _stage_see(cam, mark)
        # **The frame test lives in `_stage_see` and nowhere else.** This
        # function used to carry its own — horizontal in one place, horizontal
        # and vertical in the other — and two tests for one question are two
        # answers waiting to differ, which is what a camera that can tilt made
        # them do.
        if not seen["in_frame"]:
            continue
        w, h = seen["fw"], seen["fh"]
        cx = (seen["sx"] + 1.0) / 2.0
        cy = (seen["sy"] + 1.0) / 2.0
        x = min(max(cx - w / 2.0, 0.0), 1.0)
        y = min(max(cy - h / 2.0, 0.0), 1.0)
        # The mark's own fields ride along, because a projected box *is* the
        # region for that body and pairing them back up afterwards would be an
        # index dance over a list this function already filters.
        box = {"castId": mark["castId"], "x": x, "y": y,
               "width": min(w, 1.0 - x), "height": min(h, 1.0 - y),
               "prompt": mark.get("prompt") or "",
               "lora": mark.get("lora"), "strength": mark.get("strength")}
        if box["width"] > 0.0 and box["height"] > 0.0:
            out.append(box)
    return out


def _validate_stage(raw: Any, *, cast_ids: "set[str] | None",
                    kinds: "dict[str, str] | None" = None) -> dict[str, Any] | None:
    """
    A shot's blocking, or the reason it is not one.

    Returns None for "not blocked", which is not an error — it is the degrade,
    and a shot without a stage compiles exactly as it did before.
    """
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"Not a stage: {raw!r}")

    def num(d: Any, key: str, default: float) -> float:
        try:
            return float(d.get(key, default))
        except (TypeError, ValueError):
            raise ValueError(f"{key} is not a number: {d.get(key)!r}")

    raw_cam = raw.get("camera") or {}
    if not isinstance(raw_cam, dict):
        raise ValueError(f"Not a camera: {raw_cam!r}")
    lens = num(raw_cam, "lens", STAGE_LENS)
    if not 8.0 <= lens <= 300.0:
        raise ValueError(f"A {lens:g}mm lens is outside 8-300mm.")
    cam = {"x": num(raw_cam, "x", 0.0), "z": num(raw_cam, "z", 0.0),
           "y": num(raw_cam, "y", STAGE_EYE_H),
           "yaw": _stage_norm(num(raw_cam, "yaw", 0.0)), "lens": lens,
           "tilt": _stage_norm(num(raw_cam, "tilt", 0.0)),
           "on": str(raw_cam.get("on") or "") or None}

    marks: list[dict[str, Any]] = []
    for entry in list(raw.get("marks") or [])[:STAGE_MAX_MARKS + 1]:
        if not isinstance(entry, dict):
            raise ValueError(f"Not a mark: {entry!r}")
        who = str(entry.get("castId") or "")
        # A mark is a *body*, so it belongs to somebody — on the video side to
        # a cast member, because an unbound one would compile to a clause about
        # a person the document never defines. The image side has no cast and
        # passes `cast_ids=None`: there a mark carries its own sentence, the
        # way a region always has.
        if cast_ids is not None and who not in cast_ids:
            raise ValueError(f"A mark stands on the floor for somebody who is "
                             f"not in the cast: {who!r}")
        if who and any(m["castId"] == who for m in marks):
            raise ValueError(f"{who!r} has two marks in one shot. A body is in "
                             f"one place at a time.")
        # Size is the mark's, not the projection's. Bounded because these are
        # metres and a typo is a subject that fills every frame it is in or
        # none of them; 12m is a bus, 0.02m is a coin.
        dims = {}
        for key, default in (("h", STAGE_FIGURE_H), ("w", STAGE_FIGURE_W)):
            dims[key] = num(entry, key, default)
            if not 0.02 <= dims[key] <= 12.0:
                raise ValueError(f"{who or 'A mark'} is {dims[key]:g}m {key}, "
                                 f"which is outside 0.02-12m.")
        base = num(entry, "base", 0.0)
        if not -2.0 <= base <= 12.0:
            raise ValueError(f"{who or 'A mark'} stands {base:g}m off the "
                             f"floor, which is outside -2-12m.")
        marks.append({"castId": who, "x": num(entry, "x", 0.0),
                      "z": num(entry, "z", 3.0),
                      "yaw": _stage_norm(num(entry, "yaw", 180.0)),
                      "h": dims["h"], "w": dims["w"], "base": base,
                      # Only a person has a front. `STAGE_FACING` is the band
                      # that earns this whole feature and it is nonsense on a
                      # light fixture — "the bulb, facing the lens" is the
                      # arithmetic answering a question nobody asked of it.
                      # Every mark has a front. This used to read the cast
                      # kind — `character` faced the lens and a `thing` did
                      # not — and that table is gone, so a caller that knows
                      # better passes `faces` and nothing else guesses.
                      "faces": bool(entry.get("faces", True)),
                      "prompt": _oneline(str(entry.get("prompt") or "")),
                      "lora": entry.get("lora") or None,
                      "strength": entry.get("strength")})
    if len(marks) > STAGE_MAX_MARKS:
        raise ValueError(f"{len(marks)} marks; at most {STAGE_MAX_MARKS}.")
    if not marks:
        return None

    # A camera riding a body has to be riding one that is on the floor,
    # because the whole of what `on` does downstream is take that mark out of
    # the frame. Bound to nobody, it silently describes the shot as a point of
    # view belonging to no one and still puts the rider in his own close-up.
    if cam["on"] and not any(m["castId"] == cam["on"] for m in marks):
        raise ValueError(f"The camera is riding {cam['on']!r}, who has no mark "
                         f"in this shot.")

    path = raw.get("path") or None
    if path is not None and not isinstance(path, dict):
        raise ValueError(f"Not a camera path: {path!r}")
    if path is not None:
        path = dict(path)
        if path.get("lens") is not None:
            end_lens = num(path, "lens", lens)
            if not 8.0 <= end_lens <= 300.0:
                raise ValueError(f"The path ends on a {end_lens:g}mm lens, "
                                 f"outside 8-300mm.")
        if path.get("on"):
            rider = str(path["on"])
            if not any(m["castId"] == rider for m in marks):
                raise ValueError(f"The camera path ends riding {rider!r}, who "
                                 f"has no mark in this shot.")
            path["on"] = rider
    return {"camera": cam, "marks": marks, "path": path}


def _h3_shot_text(shot: dict[str, Any], i: int, *, at: float, secs: float,
                  cast: list[dict[str, Any]], subjects: dict[str, int],
                  speakers: dict[str, str], carried: str, lead: str) -> str:
    """
    One `[Shot N]` block, in the guide's own shape.

    `[Shot 1]` takes no timestamp and every later shot opens with a strictly
    increasing cut time — which is why the times come off the strip rather than
    being typed: two rows cannot be out of order, and a cut cannot land outside
    the clip.
    """
    # The pills fold exactly as they do everywhere else — `_shot_body` at the
    # vocabulary's own slots. The first version placed framing at the head of
    # the shot, because the guide opens Shot 1 with the composition, and it came
    # back as "[Shot 1] in a medium shot, shot at eye level, <Subject 1> sits
    # alone". The phrases are prepositional *by design*: they were written to
    # trail the sentence, and a compiler that repositions them is reinterpreting
    # a table it does not own. The composition still arrives inside Shot 1,
    # which is what the guide asks for; it arrives where the vocabulary puts it.
    # The camera comes out of the fold and is placed by hand, for one reason:
    # `<scenetrans>` has to sit at the *connecting point*, and with the camera
    # sentence folded in, the carry-in marker landed at the end of the shot —
    # after the move, several clauses downstream of the cut it is marking.
    # Everything else still folds at the vocabulary's own slots.
    # Blocking, folded in before anything else reads the pills. The derived
    # ones are *added* to what was clicked and never over it, so a person who
    # picked a close-up keeps it however the marks move.
    pills = list(shot["pills"])
    extra: list[str] = []
    if shot.get("stage"):
        chosen = {p["key"].split(".", 1)[0] for p in pills}
        pills += _stage_pills(shot["stage"], secs, chosen)
        extra = _stage_clauses(
            shot["stage"],
            lambda cid: _h3_label(next(c for c in cast if c["id"] == cid),
                                  subjects))

    groups = _shot_groups(pills, side="video")
    buckets = _shot_phrases([p for p in pills
                             if not p["key"].startswith("camera.")],
                            side="video")
    line = _h3_resolve(shot["line"], cast, subjects)
    # The clauses land with the prose rather than after the pills, because they
    # are about the subjects and the pills are about the lens.
    body_in = [line] + extra if line else extra
    visual = _shot_body(body_in, buckets) if (body_in or buckets) else ""

    parts: list[str] = []
    if i == 0:
        parts.append(_shot_join([p for p in (_close(lead), visual) if p]))
    else:
        # `the shot cuts to` is one of the guide's five listed cut verbs, and a
        # cut is required to introduce new information — which the body does,
        # rather than the clause announcing it.
        cut = f"At {_h3_clock(at)}, the shot cuts to"
        parts.append(f"{cut} {visual}" if visual else _close(f"{cut} the scene"))
    if carried:
        parts.append(f"<scenetrans> {carried}")
    st = shot.get("stage")
    if st and st.get("path") and st["marks"] and "camera" not in {
            p["key"].split(".", 1)[0] for p in shot["pills"]}:
        lead = _stage_lead(st["camera"], st["marks"])
        parts.append(_stage_move_sentence(
            st["camera"], st["path"],
            lead["dist"] if lead else None, secs))
        arc = _stage_arc(st, {p["key"].split(".", 1)[0]
                              for p in shot["pills"]})
        if arc:
            parts.append(arc)
    parts.extend(groups.get("camera", []))

    say = shot["say"]
    if say["text"]:
        ids = [speakers[w] for w in say["who"] if w in speakers]
        by_id = {c["id"]: c for c in cast}
        names = [_h3_label(by_id[w], subjects) for w in say["who"] if w in by_id]
        # A speaker who is not a defined subject is written as a stable voice
        # description followed by the ID, which is the guide's own provision for
        # a narrator or anyone off-screen.
        who = ", ".join(names) or say["voice"] or "An unseen voice"
        sid = f" ({','.join(ids)})" if ids else ""
        verb = ("says in an off-screen voiceover" if say["offscreen"]
                else "says")
        line = (f"{who}{sid} {verb}: "
                f"<d>[{say['lang']}] {say['text']}</d>")
        if say["offscreen"]:
            line += " while their lips remain completely closed."
        parts.append(line)
        if say["cutoff"]:
            parts.append("<cutoff> The line is truncated by the end of the "
                         "video.")
        elif say["carry"]:
            # Both connecting points, which is what the guide asks for and what
            # a single marker at the cut cannot express: the model has to know
            # the audio continues *into* the next shot as well as *out of* this
            # one.
            parts.append("<scenetrans> The line continues seamlessly across "
                         "the cut.")
    return f"[Shot {i + 1}] " + " ".join(p for p in parts if p)


def _compile_h3_scene(scene: dict[str, Any], *, task: str) -> str:
    """
    The document H3 reads, assembled from a cast and a timeline.

    Six fields in ref mode and three in base mode, each on its own line with its
    content following — the shape both of MiniMax's guides demonstrate. The
    order is theirs and is not ours to change: `subject_definitions` has to
    define a label before `summary` uses it, and `retention_analysis` has to
    come before the body that spends it.
    """
    cast, shots = scene["cast"], scene["shots"]
    # The timeline decides the numbering, not the order the cast was built in.
    # See `_h3_subjects`: the sentence already says what the scene is about.
    subjects = _h3_subjects(cast, shots)
    speakers = spk = _h3_speakers(shots)

    # Spans, not starts. A camera move needs the shot's *duration* to state a
    # speed, and the beat weight is not one — passing `beats` made a 0.9m push
    # over six seconds read as "fast".
    total = sum(s["beats"] for s in shots) or 1.0
    at, times = 0.0, []
    for shot in shots:
        span = shot["beats"] / total * scene["seconds"]
        times.append((at, at + span))
        at += span

    # ── the body ────────────────────────────────────────────────────────────
    sources = scene.get("sources") or {}
    ref_mode = (bool(subjects) or bool(sources)
                or any(r["kind"] == "audio" for c in cast for r in c["refs"]))
    style = scene["style"] or "Live-action, cinematic"
    grade = scene["grade"]
    # The one place the two guides genuinely disagree, and it is deliberate:
    # base mode states the style *after* `[Shot 1]`, full-reference mode
    # establishes it in a sentence *before* it (ref-en §5.2).
    lead = "" if ref_mode else ", ".join([p for p in (style, grade) if p])

    blocks, carried = [], ""
    for i, shot in enumerate(shots):
        blocks.append(_h3_shot_text(
            shot, i, at=times[i][0], secs=max(0.1, times[i][1] - times[i][0]),
            cast=cast, subjects=subjects,
            speakers=speakers, carried=carried, lead=lead))
        carried = ("The line from the previous shot carries over across the "
                   "transition." if shot["say"]["carry"] else "")
    body = "\n".join(blocks)
    if ref_mode:
        opener = f"The target video is in a {style.lower()} style"
        opener += f" with {grade}." if grade else "."
        body = f"{opener}\n{body}"

    lines: list[str] = []
    if ref_mode:
        # ── subject_definitions ─────────────────────────────────────────────
        defs: list[str] = []
        # **In subject order, not cast order.** Numbering moved to order of
        # first mention and this loop did not, so a cast built location-first
        # emitted `<Subject 2>` above `<Subject 1>` — a field whose whole job is
        # to define a label before anything spends it, listing them out of
        # sequence. The guide's examples run 1, 2, 3 and there is no reason to
        # differ. Audio siblings below still follow the cast, because each hangs
        # off a subject already defined rather than introducing a number.
        in_order = sorted((c for c in cast if subjects.get(c["id"])),
                          key=lambda c: subjects[c["id"]])
        for member in in_order:
            n = subjects[member["id"]]
            # One entry per subject listing every asset it is built from —
            # `<Subject 2> is the Samoyed in <Picture 2>, <Picture 3> and
            # <Picture 4>` — rather than one entry per asset. A subject is a
            # content unit; the files are where it came from.
            visual = [r for r in member["refs"] if r["kind"] != "audio"]
            # **The description carries what it is.** This used to open with a
            # noun off our own kind table — `is the person in <Picture 1>` —
            # standing exactly where the guide puts the description:
            #
            #   <Subject 1> is the young woman in <Picture 1>, with long dark
            #   hair, a blue cardigan, and a thin silver necklace.
            #
            # With nothing written about them there is nothing to claim, so the
            # line says only that they are shown.
            what = member["note"] or "shown"
            # Where a picture has its own note, the guide's per-asset form wins,
            # because it says which asset provides which half of the subject.
            # A character sheet outranks both: it is a *typed* reference, and
            # the citation is templated — the sheet's views and on-image labels
            # do the defining, which is MiniMax's own construction for a
            # labeled reference card. This sentence is the compiler's to write
            # and never the person's, by the division the platform runs on:
            # instruction text the format wants is ours; what the subject *is*
            # stays their words, riding in `what` untouched.
            sheets = [r for r in visual if r.get("sheet")]
            told = [r for r in visual if r.get("note") and not r.get("sheet")]
            rest = [r for r in visual
                    if not r.get("note") and not r.get("sheet")]
            labels = _h3_list([_h3_asset(r) for r in rest])
            line = f"<Subject {n}> is {what}" + (f" in {labels}" if labels else "") + "."
            for r in sheets:
                line += (f" {_h3_asset(r)} is their labeled character "
                         f"reference sheet; its views and written notes "
                         f"define their appearance.")
            # Each noted asset gets its own sentence rather than the guide's
            # `whose X comes from` clause. That form needs a bare noun and what
            # a person types is a noun *phrase* — "the olive field coat" — which
            # came out as "whose the olive field coat comes from <Picture 3>".
            # A sentence takes whatever they wrote and stays grammatical, and
            # the guide asks only that the definition explain what each asset
            # provides, not that it be one sentence.
            for r in told:
                line += f" {_h3_cap(r['note'])} comes from {_h3_asset(r)}."
            defs.append(line)
        for member in cast:
            n = subjects.get(member["id"])
            for ref in member["refs"]:
                if ref["kind"] != "audio":
                    continue
                who = f"<Subject {n}>" if n else (member["note"] or member["handle"])
                sid = spk.get(member["id"])
                # The note rides the definition, because base-en §4.4 asks for
                # exactly this — pitch, timbre, rate, accent — when a speaker
                # first appears. It validated and travelled and was then
                # dropped on the floor, which made the UI's "how they sound"
                # field a lie: worse than hiding a capability is offering one
                # that silently does nothing.
                defs.append(f"{_h3_asset(ref)} is the "
                            f"{H3_AUDIO_NOUN[ref['role']]} for "
                            f"{who}{f' ({sid})' if sid else ''}."
                            + (f" {_h3_cap(ref['note'])}."
                               if ref.get("note") else ""))
        # A source is its own label, never folded into a subject: the guide's
        # own line is `<Video 1> is the source video for the target video edit.`
        for key, idx in sources.items():
            spec = H3_SOURCES[key]
            for i in idx:
                defs.append(f"{_h3_asset({'kind': spec['kind'], 'index': i})} "
                            f"is the {spec['noun']}.")
        lines += ["subject_definitions:", "\n".join(defs)]

        # ── summary ─────────────────────────────────────────────────────────
        first = _first_sentence(_close(_h3_resolve(
            shots[0]["line"], cast, subjects))) or _first_sentence(body)
        # ref-en §3: "For video-editing tasks, begin the summary after the
        # task-type prefix with: The target video is an edited version of
        # <Video 1>." Not phrasing we chose — a required opening.
        lead_in = ""
        if "edit" in sources:
            lead_in = (f"The target video is an edited version of "
                       f"{_h3_asset({'kind': 'video', 'index': sources['edit'][0]})}. ")
        lines += ["summary:", _h3_task_types(scene, task) + lead_in + first]

        # ── retention_analysis ──────────────────────────────────────────────
        # One line per label, `(appears in …)`, then the fixed relationship
        # marker, then what has to survive. `(Sx)` never appears here — the
        # guide says so outright, and it is easy to leak in from the body.
        keep: list[str] = []
        for member in cast:
            n = subjects.get(member["id"])
            if not n:
                continue
            # Standing on the floor is appearing. Blocking a subject into a
            # shot without naming them in the prose is the ordinary case, and
            # reading only the line reported them as appearing nowhere.
            where = [f"[Shot {i + 1}]" for i, s in enumerate(shots)
                     if member["handle"] in _h3_handles(s["line"])
                     or any(m["castId"] == member["id"]
                            for m in ((s.get("stage") or {}).get("marks") or []))]
            # Visual references only. An audio reference's retain clause belongs
            # to the `<Audio N>` line, which says it with an audio marker;
            # claiming it here too makes the subject retain a timbre a picture
            # does not carry, under two different markers.
            #
            # What is retained is what the person said each picture provides,
            # and "appearance" where they said nothing — the same fallback the
            # slot phrases used to reach when a subject had no slot filled.
            visual_refs = [r for r in member["refs"] if r["kind"] != "audio"]
            what = [r["note"] for r in visual_refs
                    if r.get("note") and not r.get("sheet")]
            # The sheet's retention clause is templated with its citation: what
            # is preserved is what the sheet defines, said once in the sheet's
            # own terms rather than re-listed by hand.
            if any(r.get("sheet") for r in visual_refs):
                what.append("everything its reference sheet's views and "
                            "notes define")
            # A picture nobody annotated is still there for their appearance, so
            # noting *one* of several must not delete what the others are for.
            plain = [r for r in visual_refs
                     if not r.get("note") and not r.get("sheet")]
            if plain or not what:
                # Leading, not appended: "its appearance and the olive field
                # coat are retained" reads as a sentence, and the coat leading
                # promoted one picture's note to the subject's headline.
                what.insert(0, "appearance")
            what = list(dict.fromkeys(what))
            keep.append(
                f"<Subject {n}>"
                + (f" (appears in {', '.join(where)})" if where else "")
                # `its` only where the phrase has no determiner of its own.
                # The slot table's phrases were bare nouns — "facial structure,
                # hair and build" — and what a person writes is a noun phrase:
                # "its the olive field coat is retained".
                + f": {member['retention']} - "
                # `_h3_list`, not semicolons: "X; its Y are retained" is a
                # mangled sentence the moment a person's note joins the list,
                # and the guide's own retention lines read as prose.
                + _h3_list([w if re.match(r"^(the|a|an|his|her|their|everything)\b",
                                           w, re.I)
                            else f"its {w}" for w in what])
                # "its appearance are retained" — the plural was safe while
                # every phrase came from the slot table and read as a list.
                + (" are retained." if len(what) > 1 else " is retained."))
        # `(Sx)` is deliberately absent from every line here — the guide says so
        # outright, and the audio lines are where it would leak in, because the
        # definition above legitimately carries one.
        for member in cast:
            for ref in member["refs"]:
                if ref["kind"] != "audio":
                    continue
                task, marker = H3_AUDIO_ROLES[ref["role"]]
                keep.append(
                    f"{_h3_asset(ref)}: {marker} - "
                    + ("it is reused as the target video's audio."
                       if marker == "fully_copy" else
                       "the target speaker follows its voice timbre and "
                       "delivery without copying the original signal."))
        for key, idx in sources.items():
            spec = H3_SOURCES[key]
            for i in idx:
                keep.append(
                    f"{_h3_asset({'kind': spec['kind'], 'index': i})} "
                    f"({spec['retain']}): {spec['mark']} - it is used as the "
                    f"{spec['retain']} of the target video.")
        lines += ["retention_analysis:", "\n".join(keep)]
        lines += ["detailed_description:", body]
    else:
        align = H3_ALIGN[task].format(s=f"{scene['seconds']:.2f}")
        if align:
            lines += [align, ""]
        lines += ["integrated_multimodal_description:", body]

    sound = _h3_across(shots, "sound")
    lines += ["overall_soundscape:",
              _h3_cap(", ".join(sound))
              + (" continues" if len(sound) == 1 else " continue")
              + " throughout the video." if sound else H3_SOUNDSCAPE_DEFAULT]
    score = _h3_across(shots, "score")
    lines += ["non_diegetic_music:",
              f"{_h3_cap(', '.join(score))}." if score else "N/A"]
    return "\n".join(lines)


def _h3_across(shots: list[dict[str, Any]], group: str) -> list[str]:
    """One field's pills over every shot, in order, once each.

    Both audio fields are summaries of the *whole* clip — the guide keeps
    shot-synchronised sound in the body and puts only the continuous layer here
    — so a pill picked on three shots is one line in the soundscape.
    """
    return list(dict.fromkeys(
        p for s in shots
        for p in _shot_groups(s["pills"], side="video").get(group, [])))


def _h3_list(parts: list[str]) -> str:
    """A comma list with `and` before the last, which is how the guide writes."""
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return parts[0] if parts else ""
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _h3_asset(ref: dict[str, Any]) -> str:
    """`<Picture N>` / `<Video N>` — numbered per category, and 1-based.

    The index is the file's position in `references[]` or `ref_videos[]`, not
    its order of first appearance in the cast, because those lists are what the
    generator uploads and H3 numbers them as it receives them. A compiler that
    numbered by cast order would label the right picture with the wrong number,
    which is a valid document that points at somebody else's face.
    """
    kind = {"video": "Video", "audio": "Audio"}.get(ref["kind"], "Picture")
    return f"<{kind} {ref['index'] + 1}>"


def _h3_cap(text: str) -> str:
    """First letter up, for a field that is a sentence of our own making."""
    return text[0].upper() + text[1:] if text else text


def _compile_h3_prompt(*, typed: str, pills: list[dict[str, Any]],
                       task: str, seconds: float,
                       roles: list[str] | None = None,
                       scene: dict[str, Any] | None = None) -> str:
    """
    The document H3 actually reads, assembled from a sentence and some pills.

    Returns `typed` untouched when there is nothing to compile — no pills, no
    reference roles. That is not a shortcut, it is the contract: this feature
    must not change what a prompt written before it meant, and a bare sentence
    wrapped in field labels is a different input to the encoder than the bare
    sentence.
    """
    if scene:
        doc = _compile_h3_scene(scene, task=task)
        if len(doc) > MAX_H3_PROMPT:
            raise ValueError(
                f"The document is {len(doc)} characters and H3's prompt field "
                f"holds {MAX_H3_PROMPT}. Shorten a shot, or use fewer.")
        return doc
    roles = [r for r in (roles or [])]
    if not pills and not any(roles):
        return typed
    body_in = typed
    # The summary is the one line that says what happens, and with a storyline
    # that line already exists: the declaration is the first module by
    # construction. Falling back through the typed text to the description's
    # first sentence keeps the three input shapes answering the same field.
    lead = typed
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
        body = f"[Shot 1] {_shot_body(body_in, buckets)}"
        # The typed line is the summary — it is the one sentence in the document
        # that says what happens. Taking the description's first sentence
        # instead, which is what this did first, summarised a shot as "A
        # close-up, shot from a low angle": true, and about the lens rather than
        # the scene. The first sentence is the fallback for a prompt built
        # entirely out of pills.
        lines += [
            f"subject_definitions: {' '.join(subjects)}",
            f"summary: {_first_sentence(_close(lead)) or _first_sentence(body)}",
            f"retention_analysis: {' '.join(retain)}",
            f"detailed_description: {body}",
        ]
    else:
        align = H3_ALIGN[task].format(s=f"{float(seconds):.2f}")
        if align:
            lines.append(align)
        lines.append(f"integrated_multimodal_description: "
                     f"[Shot 1] {_shot_body(body_in, buckets)}")

    # `N/A` here used to be the default and it was a bug with no symptom: the
    # guide reserves it for *complete silence, requested*, so every clip nobody
    # picked a sound pill on was telling H3 it was silent. It also warns that a
    # blank field lets ambience creep in on its own, so there is no empty answer
    # — see `H3_SOUNDSCAPE_DEFAULT`. `non_diegetic_music` below keeps `N/A`,
    # because there the guide's rule is the opposite one.
    lines.append(f"overall_soundscape: "
                 f"{_shot_audio(buckets, 'sound') or H3_SOUNDSCAPE_DEFAULT}")
    # The default, and the line worth the whole feature: with no score pill the
    # document says there is no score, which is the one thing free prose could
    # never say and the reason every clip came back scored.
    lines.append(f"non_diegetic_music: "
                 f"{_shot_audio(buckets, 'score') or 'N/A'}")
    return "\n".join(lines)


def _compile_image_prompt(typed: str, pills: list[dict[str, Any]]) -> str:
    """
    The same pills as prose, because Krea 2 has no document to fill in.

    Action and the two audio groups never reach here at all — they are filtered
    by `image` in the vocabulary rather than dropped, so a pill the image side
    does not read is dim on the palette rather than silently ignored.

    **Nothing precedes the subject.** Whatever occupies the opening clause is
    what the picture is *about*: the model reads front to back, and a property
    promoted to that position stops being a property and becomes the character.
    One mechanic, three faces — light in the lead makes a picture of the light,
    which is a real thing to want in an abstract render; perspective in the lead
    makes a picture of perspective, which reads as wild overcorrection when
    someone wanted a slight angle; and a *compiler* that leads with either makes
    that choice on nobody's behalf.

    This used to do all three, and the accumulation was the worst of it. All
    four image groups shared slot -10, so every pill landed ahead of the subject
    and they **stacked**: light and tone are both `pick: many`, so a framing, an
    angle, three lights and two tones put seven clauses in front of the person,
    who arrived last. The first repair was a split performed here — one clause
    leads, the rest fall behind — which still promoted one thing and chose it by
    vocabulary order, so with framing and angle unset the winner was light.

    The fix belongs in the vocabulary, and both sides turned out to want the
    same thing: light and tone at 30, framing and angle at 40 beside the camera
    move. Video already had `camera` at 40 for the reason H3's guide gives —
    describe the move after the thing it moves around — and that principle
    covers the frame it moves *within* just as well, so there is one slot table
    rather than a per-side exception. Nothing is left here to arrange.

    Framing's phrases moved with its slot. They were noun phrases ("a medium
    close-up"), which read correctly only when *fused* into a subject noun —
    "An extreme close-up portrait" — and this machinery cannot fuse, it
    appositions. So the one arrangement where leading is right was the one it
    could never produce, while a demoted "A medium close-up." is a bare
    fragment. "in a medium close-up" reads alone and reads joined to an angle.
    """
    if not pills:
        return typed
    return _shot_body(typed, _shot_phrases(pills, side="image"))


def _shot_meta(params: dict[str, Any]) -> dict[str, Any]:
    """
    What to put in the sidecar beside `prompt`, which is only ever what ran.

    Only when the compiler did something. A sidecar that gains `prompt_typed`
    equal to `prompt` and `shot: []` on every plain run is two fields of noise
    on the file that has to still make sense in a year.

    **"Did something" was `typed != prompt` and that test inverted under
    replacement.** It was exact while a document was the person's sentence with
    clauses added: the compiler moved the text, so the two differed whenever
    there was anything to record. Once the replacement is written into the box,
    the box *is* the compiled text — they are equal on precisely the runs with
    the most to record, and the sidecar came back empty on every one of them,
    losing the document and the original sentence together. So the question is
    asked directly now: is there anything here that the prompt alone does not
    already say.
    """
    typed = str(params.get("prompt_typed") or "")
    if not typed:
        return {}
    original = str(params.get("prompt_original") or "")
    moved = typed != params.get("prompt")
    if not moved and not (params.get("shot") or original):
        return {}
    out: dict[str, Any] = {"prompt_typed": typed, "shot": params.get("shot") or []}
    # **Kept although nothing writes it any more.** A model replacing somebody's
    # sentence is what filled this, and there is no such model on the path now —
    # but a sidecar is read years after it is written, and dropping the field
    # would make every run that has one unreadable by the code that reads them.
    if original and original != typed:
        out["prompt_original"] = original
    roles = params.get("ref_roles") or []
    if any(roles):
        # The whole positional list, blanks included: a role's index is the
        # <Picture n> it belongs to, so compacting it would renumber the
        # subjects on the way back in.
        out["ref_roles"] = list(roles)
    return out


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
    crafted `../../` cannot name a checkpoint. One weight where the image
    stack carries two, because `LoraLoaderModelOnly` patches the DiT and H3's
    text encoder — Qwen3-VL 32B, through CLIPLoader with type="minimax" — is
    never in the chain. That is a missing *text-encoder* number, not a refusal
    of LoRAs: three are in the catalogue, and ComfyUI's own H3 templates wire
    this same model-only loader.
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

        out.append({
            "path": str(path),
            # Forward slashes, relative to loras/ — os.walk on Linux produces
            # exactly this, and it is what the combo is validated against.
            "name": path.relative_to(LORAS.resolve()).as_posix(),
            "unet": strength,
        })
    return out


# The video half of the generator body — see `_ImageSide` for why it is not a
# Modal class of its own. Named at startup for the reason the image side names
# its six: a node that failed to import leaves ComfyUI running happily without
# it, and the first symptom is otherwise a rewrite rejected for an unknown
# class_type — minutes into a session, with the traceback long scrolled. The
# call went missing once while the comment beside it survived — two sessions
# editing one tree — so the sentence and the tuple travel together now.
VIDEO_NODES = ("MiniMaxH3MotionContext", "MiniMaxH3MotionContextTrim",
               "MiniMaxH3MotionContextSaveLatent",
               "MiniMaxH3MotionContextLoadLatent")


class _VideoSide:
    """A MiniMax-H3 take on this container's warm ComfyUI."""

    _comfy: _Comfy

    @modal.method()
    def generate_video(self, job_id: str,
                       params: dict[str, Any]) -> dict[str, Any]:
        """
        Run one clip.

        Everything outside the graph — the record, the staging directory, the
        stop check, the file that lands on the volume and the sidecar beside it
        — is the same contract the image side uses, extended rather than
        duplicated. It was written to dispatch on family and carried two for a
        while; the dispatch is gone and the contract is not, which is the half
        worth keeping if a second family ever arrives.
        """
        started = time.time()
        # **The first line, and it is the one that was missing.** A ten-minute
        # gap between pressing Generate and ComfyUI receiving the graph could
        # not be attributed without it: Modal holding the input behind another
        # and this function being slow look identical from outside, and the
        # container prints nothing either way. Stamped here, the two are one
        # subtraction apart.
        print(f"[video] {job_id} accepted", flush=True)
        _note_queue_wait("video", job_id, params)
        _clear_stop(job_id)
        # **Everything from here to `run()` used to be silent**, and a ten-minute
        # gap between pressing Generate and ComfyUI receiving the graph was
        # therefore unattributable: the job record said `loading` from the first
        # line and `generate` from the last, with nothing in between and nothing
        # in the log either. A volume reload queues behind another, a reference
        # is base64 that has to reach disk and be resized, and a missing weight
        # walks the volume — each of those can be the minutes, and none of them
        # could be told apart afterwards. So each one names itself, to the log
        # for later and to the record for the person waiting.
        _publish(job_id, status="running", phase="reloading the volume")
        _reload_volume()

        model = str(params.get("model") or "h3")
        still = bool(params.get("still"))

        try:
            # Inside the try, not above it: a missing weight raised out here
            # would leave the record saying "running" forever, and the UI
            # polling a job that is never going to answer.
            def stage(blob: str, slot: str, ext: str = "png") -> str:
                return self._comfy.stage(job_id, blob, slot, ext)

            # Gated either side of the slow part, so Stop is answered while
            # this is still arithmetic and a file copy rather than after a
            # graph is on a GPU.
            _stop_gate(job_id, "the volume reload")
            # **Counted and weighed, because "staging the inputs" was still not
            # an answer.** Nine references is H3's maximum and each one arrives
            # as base64 in the request body — through the web container, through
            # Modal's blob store, back down here to be decoded and written. It
            # is the one part of this window whose cost is set by something the
            # person did, so it is the one that has to name itself.
            n_att, mb = _attachment_weight(params)
            _publish(job_id, phase=(f"staging {n_att} attachment"
                                    f"{'' if n_att == 1 else 's'}"
                                    f" · {mb:.0f} MB" if n_att else "staging"))
            t_plan = time.time()
            plan = self._plan_h3(params, stage, job_id=job_id)
            graph, info = plan["graph"], plan["info"]
            if params.get("workflow_graph"):
                # The Playground toggle — same fact as the image side: the
                # compiled graph feeds the saved workflow by inherited node
                # key, after staging so filenames are real. See
                # _apply_workflow.
                graph = _apply_workflow(graph, params["workflow_graph"],
                                        params.get("workflow_extras"))
            print(f"[video] ready to queue after {time.time() - started:.1f}s "
                  f"({time.time() - t_plan:.1f}s staging {n_att} attachments, "
                  f"{mb:.1f} MB)", flush=True)
            _stop_gate(job_id, "staging")

            # **`loading`, not `generate`.** ComfyUI has the graph and has not
            # sampled anything: on a warm H100 it is 82 seconds of VAE, text
            # encoder and DiT before the first step, and a bar sitting at 0%
            # under the word "generate" for that long is the page saying
            # something untrue. `_drain` moves it on when a step arrives.
            _publish(job_id, phase="loading", step=0,
                     total_steps=info["steps"], percent=0, **info)

            out_names = self._comfy.run(job_id, graph,
                                        what="image" if still else "video")
        except StopRequested:
            res = {"status": "stopped", "job_id": job_id, "files": [],
                   "duration_s": round(time.time() - started, 1)}
            _publish(job_id, **res)
            return res
        except Exception as exc:
            _publish(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
            raise

        OUTPUTS.mkdir(parents=True, exist_ok=True)
        record = _output_record(
            kind="image" if still else "video", job_id=job_id, model=model,
            prompt=params["prompt"], created=time.time(),
            **_shot_meta(params), **info, **plan["meta"],
            **({"workflow_name": params["workflow"]}
               if params.get("workflow") else {}),
        )
        if still:
            # **Every decoded frame is kept, and that is the feature rather
            # than the leftover.** Which frame is cleanest moves with the
            # prompt, the seed and the canvas, so choosing one here would be
            # this file guessing at the only part a person has to look at. One
            # run returns a strip, which is also the answer to H3 being too
            # expensive to think in: the cost amortises across the candidates.
            #
            # Indexed in the filename because that index is the record. A still
            # whose record says which take it came from but not which frame is
            # a still nobody can make again. `kind="image"` in the record
            # because what came out is a picture, and everything downstream
            # sorts on what a thing *is* rather than on which model made it.
            names = []
            for i, src in enumerate(out_names):
                names.append(f"{job_id}_{i:02d}.png")
                _png_with_record(COMFY / "output" / src, OUTPUTS / names[-1], record)
            volume.commit()
            res = {"status": "completed", "job_id": job_id, "files": names,
                   "output_dir": str(OUTPUTS), "model": model,
                   "duration_s": round(time.time() - started, 1), **info}
            _publish(job_id, **res)
            return res

        name = f"{job_id}.mp4"
        # One clip per graph, so the first is the only one. Taking [0] rather
        # than asserting length: a save node that ever emitted a poster frame
        # beside the mp4 should cost the poster, not the take. The record
        # rides the clip's own metadata; a clip ffmpeg cannot rewrite is
        # copied plain and the log says which.
        _mp4_with_record(COMFY / "output" / out_names[0], OUTPUTS / name, record)
        # The take's motion context, harvested beside its clip. ComfyUI's
        # output folder is container disk, and the take that needs this file is
        # the *next* one — which arrives after exactly the kind of gap that
        # scales a container to zero. Best-effort on purpose: a take whose
        # latent did not save is a take you can only continue from its last
        # frame, which is the degrade the page already handles, and failing a
        # finished three-minute render over its sidecar would be the wrong
        # trade twice.
        ctx = COMFY / "output" / "h3_context" / f"{job_id}_00000.safetensors"
        if ctx.exists():
            shutil.copyfile(ctx, OUTPUTS / f"{job_id}{H3MC_SUFFIX}")
            ctx.unlink()
        else:
            print(f"[video] {job_id} no motion context saved "
                  f"(wanted {ctx}) — Continue will fall back to the last "
                  f"frame", flush=True)
        volume.commit()

        res = {
            "status": "completed", "job_id": job_id, "files": [name],
            "output_dir": str(OUTPUTS), "model": model,
            "duration_s": round(time.time() - started, 1), **info,
        }
        _publish(job_id, **res)
        return res

    @staticmethod
    def _plan_h3(params: dict[str, Any], stage: Any,
                 job_id: str = "") -> dict[str, Any]:
        """Graph and shot description for a MiniMax-H3 take."""
        refs_b64 = list(params.get("references") or [])[:MAX_H3_REFS]
        vids_b64 = list(params.get("ref_videos") or [])[:MAX_H3_REF_VIDEOS]
        auds_b64 = list(params.get("ref_audios") or [])[:MAX_H3_REF_AUDIOS]
        _require_models(*(VIDEO_REF_MODEL_KEYS
                          if (refs_b64 or vids_b64 or auds_b64)
                          else VIDEO_MODEL_KEYS))

        still = bool(params.get("still"))
        width, height = _h3_canvas(params["aspect"], params["tier"])
        # `seconds` still reaches the *compiler* on a still, and is meant to:
        # the document describes a shot, and a still is a frame out of it. A
        # separate one-moment grammar would be a second way to say the thing
        # `[Shot 1]` already says.
        frames = H3_STILL_FRAMES if still else _h3_frames(params["seconds"])

        # A continued take asks for its seconds *plus* the pinned context,
        # because the pinned run sits at the head of the clip and is trimmed
        # after decode — without the bump the person sets 6s and gets ~5.1.
        # `_h3_frames` clamps at H3_MAX_FRAMES, so a take already at the
        # ceiling comes out shorter rather than refused, which is the right
        # trade for a cap the duration menu already respects.
        continue_from = str(params.get("continue_from") or "") or None
        if continue_from:
            src = OUTPUTS / f"{continue_from}{H3MC_SUFFIX}"
            # Diagnose here, on this container, with the three facts that
            # distinguish the causes: the job asked for, the path wanted, what
            # is actually there. The web route already checked the volume, so
            # reaching this means the file went away between the check and the
            # render — a deleted generation, most likely.
            if not src.exists():
                raise FileNotFoundError(
                    f"Motion context for {continue_from} is not on the volume "
                    f"(wanted {src}). The take it belonged to was probably "
                    f"deleted — clear the Motion tile to continue from its "
                    f"last frame instead.")
            dest = COMFY / "output" / "h3_context" / f"{continue_from}_00000.safetensors"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            frames = _h3_frames(params["seconds"]
                                + H3MC_CONTEXT_FRAMES / H3_FPS)
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
        # `LoadAudio` lists the input directory and filters by content type the
        # way `LoadVideo` does, so the extension is load-bearing here too.
        ref_audios = [stage(b, f"refaud{i}", "wav") for i, b in enumerate(auds_b64)]
        keyframes: dict[str, str] = {}
        if not (references or ref_videos or ref_audios):
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
        loras = _validate_video_loras(params.get("loras"))
        graph = _h3_graph(
            prompt=params["prompt"], width=width, height=height, frames=frames,
            seed=seed, steps=steps,
            sampler=params["sampler"], scheduler=params["scheduler"],
            references=references, ref_videos=ref_videos,
            ref_audios=ref_audios, ref_size=ref_size,
            loras=loras,
            shift_video=params.get("shift_video"),
            shift_audio=params.get("shift_audio"),
            still=still,
            save_context_as=None if still else (job_id or None),
            load_context_from=continue_from,
            **keyframes,
        )
        meta = {"mode": ("ref2va" if (references or ref_videos or ref_audios)
                         else "fl2va"),
                "sampler": params["sampler"], "scheduler": params["scheduler"],
                "references": len(references), "ref_videos": len(ref_videos),
                "ref_audios": len(ref_audios),
                # No `expert`: H3 has one, so recording a field whose only value
                # is "both" would be a sidecar implying a choice nobody had.
                "loras": [{"name": l["name"], "unet": l["unet"]}
                          for l in loras]}
        # Only where somebody moved it, for the reason `ref_size` is conditional:
        # a sidecar that records the model's own default on every take is a
        # sidecar you have to know the default to read.
        if params.get("shift_video") is not None:
            meta["shift_video"] = float(params["shift_video"])
        if params.get("shift_audio") is not None:
            meta["shift_audio"] = float(params["shift_audio"])
        # Only where it meant something. It is the one input on this path that
        # changes what a take costs without changing anything the sidecar
        # already records — two takes at the same canvas, frames and steps can
        # be minutes apart on this alone, and nothing said which was which.
        if references:
            meta["ref_size"] = ref_size
        # The sidecar records the join, because a chained clip read years from
        # now should say what it opens from — and the *delivered* length, not
        # the sampled one: the pinned context is trimmed before the file is
        # written, so `frames` here would overstate every continued take by
        # H3MC_CONTEXT_FRAMES.
        if continue_from:
            meta["continued_from"] = continue_from
        shown = frames - H3MC_CONTEXT_FRAMES if continue_from else frames
        info = {"width": width, "height": height, "frames": shown,
                "seed": seed, "steps": steps}
        if still:
            # **In `info` rather than `meta`, because the page reads this one.**
            # `info` is what reaches the completed job record, and the canvas
            # has to know it is holding a strip of frames rather than a clip
            # before it decides which element to mount — a `<video>` pointed at
            # a PNG shows nothing and reports nothing. `frames` beside it is
            # already the count, so there is no second number to record.
            info["still"] = True
        else:
            # No `seconds` or `fps` on a still. They are true of the fragment
            # that was sampled and false of the thing delivered, and a sidecar
            # saying a picture is 0.92 seconds long is worse than one that does
            # not say.
            info["seconds"] = round(shown / H3_FPS, 2)
            info["fps"] = H3_FPS
        return {"graph": graph, "info": info, "meta": meta}


class _Generator(_ImageSide, _VideoSide):
    """
    One warm ComfyUI, and whichever families the placement below serves.

    `families` is the only thing a placement sets. It decides which node
    packs are checked at startup and what the log lines are tagged with; the
    methods are the same on all three, so a request that lands on the wrong
    class still runs — it just runs on a card that may not have been chosen
    for it. The routes go through `_host`, which is where the choice is made.
    """

    families: tuple[str, ...] = ()

    @modal.enter()
    def setup(self):
        tag = "both" if len(self.families) > 1 else self.families[0]
        self._comfy = _Comfy(tag)
        self._comfy.start()
        if "image" in self.families:
            self._comfy.require_nodes(*IMAGE_NODES)
        if "video" in self.families:
            self._comfy.require_nodes(*VIDEO_NODES)
        # **Nothing but ComfyUI starts here.** Each container held a second
        # model for a while — Qwen3-VL beside the checkpoint, for the prompt
        # rewrite and then for the motion panel — and it cost more than it
        # returned. Loaded at `enter` as a *ComfyUI graph* it held the single
        # queue for the 132 seconds it takes, right at the moment a cold
        # container was about to be asked for a render; loaded lazily it still
        # took the slot on first press, because `@modal.concurrent(max_inputs=1)`
        # means a suggestion in flight is a render that cannot be delivered. A
        # ten-minute wait came out of that and was blamed on the wrong feature,
        # because nothing anywhere said so.

    @modal.method()
    def warm(self) -> dict[str, Any]:
        """A knock, so the page can start this container on load. `enter` is
        what does the work; arriving here at all means it has run."""
        return {"ok": True}

    @modal.method()
    def playground(self, job_id: str,
                   params: dict[str, Any]) -> dict[str, Any]:
        """A user-authored graph on this warm engine — see _playground_run."""
        return _playground_run(self._comfy, job_id, params)

    @modal.method()
    def restart_engine(self, job_id: str) -> dict[str, Any]:
        """Replace the warm ComfyUI on request — see _Comfy.restart."""
        return _restart_engine(self._comfy, job_id)


# Three placements of the one body. `max_containers=1` on each: a checkpoint
# is 35 or 42.5 GB, so a second replica costs a full cold load rather than
# sharing the warm one. `@modal.concurrent(max_inputs=1)`: one GPU, one
# sampling loop.

@app.cls(
    image=comfy_image, gpu=GPU, cpu=4.0, timeout=60 * 60,
    volumes=MODAL_VOLUMES,
    max_containers=1,
    scaledown_window=10 * 60,
)
@modal.concurrent(max_inputs=1)
class ImageGenerator(_Generator):
    """Holds a warm ComfyUI process, loaded with a Krea 2 checkpoint."""
    families = ("image",)


@app.cls(
    image=comfy_image, gpu=VIDEO_GPU, cpu=4.0, timeout=60 * 60,
    volumes=MODAL_VOLUMES,
    max_containers=1,
    # Longer than the image side's 10 min: 42.5 GB is a slow thing to reload,
    # and video is worked in takes — you watch one clip, then adjust and go again.
    scaledown_window=15 * 60,
)
@modal.concurrent(max_inputs=1)
class VideoGenerator(_Generator):
    """Holds a warm ComfyUI process, loaded with a video checkpoint."""
    families = ("video",)


@app.cls(
    image=comfy_image, gpu=BOTH_GPU, cpu=4.0, timeout=60 * 60,
    volumes=MODAL_VOLUMES,
    max_containers=1,
    # The video side's window, because the more expensive reload is the one
    # to keep warm.
    scaledown_window=15 * 60,
    # Host RAM for both checkpoints at once. On H200 they share the card; on an
    # 80 GB card ComfyUI pages the idle family out to RAM and back, and a
    # container without the room for that pays the 400 s volume reload on
    # every switch instead. 96 GiB is a starting figure, not a measurement —
    # read `vram before run` and the page-in time on the first H100 session
    # and move it.
    memory=96 * 1024,
)
@modal.concurrent(max_inputs=1)
class BothGenerator(_Generator):
    """
    One warm ComfyUI serving both families — the gear's one-container mode.

    The trade it makes on purpose: one warm card instead of two, and a
    picture that waits behind a clip in progress, because one class is one
    queue. The rule this bends — one container per loaded checkpoint — stays
    the default; this is the user choosing the other side of it, told what
    it costs.
    """
    families = ("image", "video")


# --------------------------------------------------------------------------
# Playground — user-authored graphs on the same engine
#
# The room where the graph our builders write is the thing on screen: seeded
# from the app's own live graph, rewired by hand, run on whichever warm
# generator was asked for. Not a second backend — the same _Comfy, the same
# containers, the same job/status/stop contract; the one difference is that
# the graph arrives built instead of being built here. Saved workflows are
# pure API-format JSON under workflows/ (the storage layout is the contract),
# node packs installed from git live under playground_nodes/ pinned by SHA,
# and every output embeds its own graph the way ComfyUI does, so the PNG *is*
# the workflow.
# --------------------------------------------------------------------------

# A workflow name becomes a filename on the volume, so the rule is the
# filename's — and the error says so, because "invalid name" explains nothing.
_WORKFLOW_NAME_RE = re.compile(r"[\w][\w .-]{0,63}")


def _storyboard_name(raw: Any) -> str:
    """A board's folder name. The same shape a workflow name has, lower-cased
    because it is a path on the volume and a URL segment at once."""
    name = str(raw or "").strip()
    if not STORYBOARD_NAME_RE.fullmatch(name):
        raise ValueError("A storyboard name is lowercase letters, digits, "
                         "dashes and underscores — up to 64 of them.")
    return name


def _validate_storyboard(raw: Any) -> dict:
    """
    Structural zeros only — the page writes the whole document back on every
    edit, so the only thing to refuse is a shape no reader could follow: a
    panel without an id, two panels sharing one, a picture pointer missing
    half its address, an arrow with a point outside the frame. Content is the
    person's and travels verbatim. A picture's `file` is path-stripped because
    it becomes half of a file address; a picture with no `job_id` lives in the
    board's own folder, and one with a job id is a render in outputs/.

    The pills are `_validate_shot`'s — a panel is a shot's intent, so it
    carries a shot's vocabulary, camera amplitude and speed included — and
    that is what makes the hand-off a copy rather than a translation.
    """
    if not isinstance(raw, dict):
        raise ValueError("The storyboard must be an object with `panels`.")
    panels_in = raw.get("panels")
    if not isinstance(panels_in, list):
        raise ValueError("`panels` must be a list.")
    if len(panels_in) > STORYBOARD_MAX_PANELS:
        raise ValueError(f"A storyboard holds at most {STORYBOARD_MAX_PANELS} "
                         f"panels; this one has {len(panels_in)}.")
    aspect = str(raw.get("aspect") or "16:9")
    if aspect not in VIDEO_ASPECTS:
        raise ValueError(f"Unknown aspect {aspect!r}. "
                         f"One of: {', '.join(VIDEO_ASPECTS)}")
    panels: list[dict] = []
    seen: set[str] = set()
    for i, p in enumerate(panels_in):
        if not isinstance(p, dict):
            raise ValueError(f"Panel {i + 1} is not an object.")
        pid = str(p.get("id") or "")
        if not pid or pid in seen:
            raise ValueError(f"Panel {i + 1} needs an id no other panel has.")
        seen.add(pid)
        pic = p.get("picture")
        if pic is not None:
            if not isinstance(pic, dict) or not pic.get("file"):
                raise ValueError(f"Panel {i + 1}'s picture needs a file.")
            # `gallery` says which folder: a render in outputs/, or a picture
            # dropped onto the board. Either way the file is the address.
            name = Path(str(pic["file"])).name
            if pic.get("gallery") and not OUTPUT_FILE_RE.match(name):
                raise ValueError(f"Panel {i + 1}'s picture names a render no "
                                 f"run could have written: {name!r}")
            pic = {"file": name, **({"gallery": True} if pic.get("gallery") else {})}
        fit = str(p.get("fit") or "crop")
        if fit not in ("crop", "whole"):
            raise ValueError(f"Panel {i + 1}'s fit is {fit!r}; crop or whole.")
        focus_in = p.get("focus") or [0.5, 0.5]
        try:
            focus = [min(1.0, max(0.0, float(focus_in[0]))),
                     min(1.0, max(0.0, float(focus_in[1])))]
        except (TypeError, ValueError, IndexError):
            raise ValueError(f"Panel {i + 1}'s focus is not a point.")
        arrows_in = p.get("motion") or []
        if not isinstance(arrows_in, list):
            raise ValueError(f"Panel {i + 1}'s motion is not a list.")
        if len(arrows_in) > STORYBOARD_MAX_ARROWS:
            raise ValueError(f"Panel {i + 1} has {len(arrows_in)} arrows; "
                             f"at most {STORYBOARD_MAX_ARROWS}.")
        motion: list[dict] = []
        for a in arrows_in:
            if not isinstance(a, dict):
                raise ValueError(f"Panel {i + 1} has an arrow that is not "
                                 f"an object.")
            try:
                pts = [[float(x), float(y)] for x, y in a.get("pts") or []]
            except (TypeError, ValueError):
                raise ValueError(f"Panel {i + 1} has an arrow with a point "
                                 f"that is not a pair of numbers.")
            if len(pts) < 2:
                raise ValueError(f"Panel {i + 1} has an arrow with one end.")
            if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in pts):
                raise ValueError(f"Panel {i + 1} has an arrow leaving the "
                                 f"frame — points are fractions of it.")
            motion.append({"id": str(a.get("id") or f"a{len(motion) + 1}"),
                           "pts": pts[:2],
                           "label": _oneline(str(a.get("label") or ""))[:80]})
        panels.append({"id": pid,
                       "prose": str(p.get("prose") or ""),
                       "note": str(p.get("note") or ""),
                       "picture": pic,
                       "fit": fit,
                       "focus": focus,
                       "pills": _validate_shot(p.get("pills")),
                       "motion": motion})
    return {"title": str(raw.get("title") or "")[:200],
            "aspect": aspect, "panels": panels}


def _workflow_name(raw: Any) -> str:
    name = str(raw or "").strip()
    if not _WORKFLOW_NAME_RE.fullmatch(name) or name.startswith("."):
        raise ValueError(
            f"Not a workflow name: {name!r}. Letters, digits, spaces, dots, "
            "dashes and underscores, up to 64 characters — it becomes "
            "workflows/{name}.json on the volume.")
    return name


def _load_workflow_for_run(payload: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """
    The toggle's read: resolve a render request's `workflow`, on CPU.

    None when nothing was toggled, which is every request until the day a
    workflow exists. Failures are ValueErrors because they are form errors —
    a workflow deleted since the menu loaded is the stale-tab case, and the
    answer belongs on the form, not in a dead job.
    """
    name = str(payload.get("workflow") or "").strip()
    if not name:
        return None
    name = _workflow_name(name)
    path = WORKFLOWS / f"{name}.json"
    if not path.is_file():
        _reload_volume()
    if not path.is_file():
        raise ValueError(f"No workflow named {name!r} — it may have been "
                         "deleted since the menu loaded.")
    try:
        graph = json.loads(path.read_text())
    except ValueError as exc:
        raise ValueError(f"{name}.json is not valid JSON: {exc}")
    return name, _validate_playground_graph(graph)


def _validate_playground_graph(graph: Any,
                               known: set[str] | None = None) -> dict[str, Any]:
    """
    Structural zeros only — the shapes /prompt would reject, caught on CPU.

    Judgement stays out of here: whether the wiring makes *sense* is ComfyUI's
    call on the GPU, where the error comes back naming the node. What belongs
    here is what would otherwise cost a cold start to discover: not a dict, a
    node without a class_type (the UI-format export, which is the mistake
    everyone makes once), a link pointing at a node that is not in the graph,
    a class_type this install has never heard of. The catalogue check only
    runs when a catalogue exists — a fresh install has none, and refusing
    every run until one is harvested would be the validator outranking the
    engine.
    """
    if not isinstance(graph, dict) or not graph:
        raise ValueError("A workflow is a non-empty object of nodes — the "
                         "API-format graph ComfyUI's /prompt takes.")
    unknown = []
    for key, node in graph.items():
        if (not isinstance(node, dict)
                or not isinstance(node.get("class_type"), str)):
            raise ValueError(
                f"Node {key!r} has no class_type — this looks like a "
                "UI-format export. The Playground reads API-format graphs "
                "(in ComfyUI: Export (API)).")
        inputs = node.get("inputs")
        if inputs is None:
            continue
        if not isinstance(inputs, dict):
            raise ValueError(f"Node {key!r}: inputs must be an object.")
        for iname, val in inputs.items():
            if isinstance(val, list):
                if (len(val) != 2 or not isinstance(val[0], (str, int))
                        or isinstance(val[0], bool)
                        or not isinstance(val[1], int)):
                    raise ValueError(
                        f"Node {key!r} input {iname!r}: a link is "
                        "[node_id, slot], and this is neither a link nor a "
                        "value.")
                if str(val[0]) not in graph:
                    raise ValueError(
                        f"Node {key!r} input {iname!r} links to {val[0]!r}, "
                        "which is not in the graph.")
        if known is not None and node["class_type"] not in known:
            unknown.append(f"{key}: {node['class_type']}")
    if unknown:
        raise ValueError(
            "Unknown node type" + ("s" if len(unknown) > 1 else "")
            + " — not in this install's catalogue: "
            + "; ".join(sorted(unknown)[:6])
            + ". Install the pack that provides "
            + ("them" if len(unknown) > 1 else "it")
            + ", or refresh the catalogue if it was just installed.")
    return graph


def _apply_workflow(default_graph: dict[str, Any],
                    workflow_graph: dict[str, Any],
                    extras: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Substitute the console's compiled values into a saved workflow.

    By inherited node key: a workflow begins life as this app's own graph, so
    a node that kept both its key and its class_type still belongs to the
    console — the compiled prompt reaches its encoder, the seed and steps
    reach its sampler, the canvas reaches its latent, exactly as if the
    builder had written it. A node the user replaced or renamed inherits
    nothing, which is the honest reading: rewiring is a statement that the
    console no longer owns that input. Wiring is never substituted — links
    are the workflow's own on both sides.

    `extras` are the adaptive inputs — {node_key: {input: value}} — checked
    to exist before they are written, so a typo is a form error rather than a
    control that silently controls nothing.
    """
    out = json.loads(json.dumps(workflow_graph))
    matched = 0
    for key, node in default_graph.items():
        target = out.get(key)
        if (not isinstance(target, dict)
                or target.get("class_type") != node.get("class_type")):
            continue
        for iname, val in (node.get("inputs") or {}).items():
            if isinstance(val, list):
                continue
            tin = target.setdefault("inputs", {})
            if iname in tin and not isinstance(tin[iname], list):
                tin[iname] = val
                matched += 1
    if not matched:
        raise ValueError(
            "This workflow inherited nothing the console can feed: no node "
            "kept both the key and the class_type of the app's own graph "
            f"(looked for {', '.join(sorted(default_graph))}; the workflow "
            f"has {', '.join(sorted(out))}). Re-seed it from the app's "
            "graph, or run it from the Playground, which sends it verbatim.")
    for nkey, vals in (extras or {}).items():
        node = out.get(str(nkey))
        if not isinstance(node, dict):
            raise ValueError(f"Exposed input names node {nkey!r}, which is "
                             "not in the workflow.")
        for iname, val in (vals or {}).items():
            inputs = node.setdefault("inputs", {})
            if iname not in inputs:
                raise ValueError(
                    f"Exposed input {nkey}.{iname} is not an input of "
                    f"{node.get('class_type')} in this workflow.")
            if isinstance(inputs[iname], list):
                raise ValueError(
                    f"Exposed input {nkey}.{iname} is wired, not a value — "
                    "unwire it in the Playground to control it here.")
            inputs[iname] = val
    return out


def _playground_run(comfy: _Comfy, job_id: str,
                    params: dict[str, Any]) -> dict[str, Any]:
    """
    Run one user-authored graph on whichever warm engine holds this call.

    A sibling of the two `generate`s rather than a branch inside them — same
    record, same staging directory, same stop gates, same volume layout and
    sidecar. Runs on the *shared* generators on purpose: a single user has no
    second workstream, so a quarantine container would cost a cold start to
    protect real work from the person doing the real work. What answers the
    risk instead is `restart()`, one gesture away in the room.
    """
    from PIL import Image, PngImagePlugin

    started = time.time()
    print(f"[playground] {job_id} accepted", flush=True)
    _note_queue_wait("playground", job_id, params)
    _clear_stop(job_id)
    _publish(job_id, status="running", phase="reloading the volume")
    _reload_volume()

    graph = params["graph"]
    try:
        _stop_gate(job_id, "the volume reload")
        atts = dict(params.get("attachments") or {})
        if atts:
            mb = sum(len(b) for b in atts.values()) * 3 / 4 / 2**20
            _publish(job_id, phase=f"staging {len(atts)} attachment"
                                   f"{'' if len(atts) == 1 else 's'}"
                                   f" · {mb:.0f} MB")
        for name, blob in atts.items():
            # Written under the name the graph already spells rather than a
            # minted one: the graph is the user's record, and rewriting its
            # LoadImage inputs would hand back a graph that differs from the
            # one that ran. Path-stripped so a name cannot climb out of
            # input/.
            safe = Path(name).name
            (COMFY / "input" / safe).write_bytes(base64.b64decode(blob))
            if safe.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                # Same range, same reason as the reference paths: every input
                # a run is priced by needs one, and a file does not come with
                # its own. See H3_REF_MAX_SIDE.
                _fit_reference(COMFY / "input" / safe, H3_REF_MAX_SIDE,
                               "playground")
        _stop_gate(job_id, "staging")

        # `loading`, not `generate` — ComfyUI has the graph and has sampled
        # nothing. `_drain` moves the phase on when a step arrives.
        _publish(job_id, phase="loading", step=0, percent=0)
        out_names = comfy.run(job_id, graph, what="output")

        OUTPUTS.mkdir(parents=True, exist_ok=True)
        graph_json = json.dumps(graph)
        clip_exts = (".mp4", ".mov", ".m4v", ".webm", ".mkv")
        # `kind` by what came out, because everything downstream sorts on
        # what a thing *is* rather than on which surface made it. The graph
        # goes in the record too: the PNG's `prompt` copy is for ComfyUI,
        # this one is for readers of the record.
        exts = [(COMFY / "output" / s).suffix.lower() for s in out_names]
        record = _output_record(
            kind="video" if any(e in clip_exts for e in exts) else "image",
            job_id=job_id, model="playground", prompt="", created=time.time(),
            workflow=graph,
            workflow_name=str(params.get("workflow_name") or "") or None,
        )
        names: list[str] = []
        for i, src in enumerate(out_names):
            srcp = COMFY / "output" / src
            ext = srcp.suffix.lower() or ".bin"
            name = f"{job_id}_{i:02d}{ext}"
            if ext == ".png":
                # Re-encoded rather than copied, same trade as the image
                # path: ComfyUI ran with --disable-metadata, and a PNG whose
                # graph lives only in a job record that expires is what every
                # existing tool reads a file to avoid. `prompt` is ComfyUI's
                # own key, so a stock install loads this straight back onto
                # its canvas — the PNG *is* the workflow.
                _png_with_record(srcp, OUTPUTS / name, record, {"prompt": graph_json})
            elif ext in clip_exts:
                # The same embed for a clip, via the container's ffmpeg: the
                # graph rides the comment tag beside the record, streams
                # copied untouched.
                _mp4_with_record(srcp, OUTPUTS / name, record, {"comment": graph_json})
            else:
                shutil.copyfile(srcp, OUTPUTS / name)
            names.append(name)
        volume.commit()

        res = {"status": "completed", "job_id": job_id, "files": names,
               "output_dir": str(OUTPUTS), "model": "playground",
               "duration_s": round(time.time() - started, 1)}
        _publish(job_id, **res)
        return res
    except StopRequested:
        res = {"status": "stopped", "job_id": job_id, "files": [],
               "duration_s": round(time.time() - started, 1)}
        _publish(job_id, **res)
        return res
    except Exception as exc:
        # The failing node travels beside the message, not inside it — the
        # editor lights the node up, and a record that only carried the
        # collapsed string would need parsing back apart.
        err = comfy.last_error or {}
        _publish(job_id, status="failed",
                 error=(str(exc) if isinstance(exc, RuntimeError)
                        else f"{type(exc).__name__}: {exc}"),
                 **({"error_node": str(err["node_id"]),
                     "error_node_type": err.get("node_type")}
                    if err.get("node_id") is not None else {}))
        raise


def _restart_engine(comfy: _Comfy, job_id: str) -> dict[str, Any]:
    """
    Replace the warm ComfyUI, as a job — queued, watched, and priced.

    A method call rather than a route side-effect so it rides
    `@modal.concurrent(max_inputs=1)`: a restart waits its turn behind a
    render instead of shooting one. A job record rather than a bare {ok}
    because the page has to be able to say what the button did — the next
    graph pays a checkpoint reload, and a lever whose cost is invisible is a
    lever nobody trusts.
    """
    started = time.time()
    print(f"[playground] {job_id} restart requested", flush=True)
    _clear_stop(job_id)
    _publish(job_id, status="running", phase="restarting the engine")
    try:
        comfy.restart()
    except Exception as exc:
        _publish(job_id, status="failed",
                 error=f"{type(exc).__name__}: {exc}")
        raise
    res = {"status": "completed", "job_id": job_id, "files": [],
           "duration_s": round(time.time() - started, 1),
           "note": "Engine restarted. The next run reloads its checkpoint."}
    _publish(job_id, **res)
    return res


@app.function(image=comfy_image, cpu=4.0, timeout=30 * 60,
              volumes=MODAL_VOLUMES)
def playground_catalogue_job(job_id: str, url: str | None = None,
                             ref: str | None = None) -> dict[str, Any]:
    """
    Install a pack from git (optionally) and re-harvest the node catalogue.

    On CPU, because reading a schema is exactly the work the
    never-rent-a-GPU rule is about: ComfyUI answers /object_info under
    --cpu, the same trick tools/smoke_graphs.py uses to validate graphs. On
    comfy_image, because describing the packs means importing them, and the
    import wants ComfyUI's own dependencies.

    The clone is staged and then renamed — the same shape the weights use —
    so a half-cloned pack is never on the shelf, and `.git` is dropped once
    the pin records the URL and the resolved SHA: the pin is the
    declaration, and a re-install years later is this exact tree.
    """
    import threading
    import urllib.request

    started = time.time()
    print(f"[playground] {job_id} accepted", flush=True)
    _clear_stop(job_id)
    _publish(job_id, status="running", phase="reloading the volume")
    _reload_volume()
    pack_name = sha = None
    try:
        _stop_gate(job_id, "the volume reload")
        if url:
            pack_name = re.sub(r"\.git$", "", url.rstrip("/").rsplit("/", 1)[-1])
            pack_name = re.sub(r"[^\w.-]", "-", pack_name).lstrip(".") or "pack"
            dest = PLAYGROUND_NODES / pack_name
            if dest.exists():
                raise ValueError(
                    f"{pack_name} is already installed ({dest}). Delete it "
                    "in the Playground first to reinstall.")
            _publish(job_id, phase=f"cloning {pack_name}")
            PLAYGROUND_NODES.mkdir(parents=True, exist_ok=True)
            tmp = PLAYGROUND_NODES / f".{pack_name}.cloning"
            shutil.rmtree(tmp, ignore_errors=True)
            done = subprocess.run(["git", "clone", url, str(tmp)],
                                  capture_output=True, text=True, timeout=600)
            if done.returncode != 0:
                raise RuntimeError(f"git clone failed for {url}:\n"
                                   + (done.stderr or "").strip()[-2000:])
            if ref:
                done = subprocess.run(
                    ["git", "-C", str(tmp), "checkout", ref],
                    capture_output=True, text=True)
                if done.returncode != 0:
                    raise RuntimeError(
                        f"git checkout {ref!r} failed:\n"
                        + (done.stderr or "").strip()[-2000:])
            sha = subprocess.run(["git", "-C", str(tmp), "rev-parse", "HEAD"],
                                 capture_output=True, text=True).stdout.strip()
            (tmp / ".visionary-pin").write_text(json.dumps(
                {"url": url, "ref": ref, "sha": sha,
                 "installed": time.time()}, indent=2))
            shutil.rmtree(tmp / ".git", ignore_errors=True)
            tmp.rename(dest)
            _stop_gate(job_id, "the clone")

        _publish(job_id, phase="installing pack requirements")
        _install_pack_requirements("playground")

        _publish(job_id, phase="starting ComfyUI on CPU")
        (COMFY / "extra_model_paths.yaml").write_text(
            "visionary:\n"
            f"  base_path: {WORKSPACE}/\n"
            f"  custom_nodes: {PLAYGROUND_NODES}\n")
        port = 8199
        proc = subprocess.Popen(
            ["python", "main.py", "--listen", "127.0.0.1",
             "--port", str(port), "--cpu",
             "--disable-auto-launch", "--disable-metadata"],
            cwd=str(COMFY), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        tail: deque[str] = deque(maxlen=60)

        def drain() -> None:
            for line in proc.stdout or []:
                tail.append(line.rstrip())

        threading.Thread(target=drain, daemon=True).start()
        try:
            deadline = time.time() + 300
            raw = None
            while time.time() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(
                        "ComfyUI exited during the harvest — a pack that "
                        "fails to import at startup is the usual cause, and "
                        "its traceback is below.\n" + "\n".join(tail))
                try:
                    raw = urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/object_info",
                        timeout=10).read()
                    break
                except OSError:
                    time.sleep(1.0)
            if raw is None:
                raise RuntimeError(
                    "ComfyUI did not answer /object_info within 300s.\n"
                    + "\n".join(tail))
        finally:
            proc.terminate()

        info = json.loads(raw)
        volume.commit()
        res = {"status": "completed", "job_id": job_id, "files": [],
               "nodes": len(info),
               "duration_s": round(time.time() - started, 1)}
        if url:
            res["pack"] = pack_name
            res["sha"] = (sha or "")[:12]
            # The warm generators walked custom_nodes before this pack
            # existed, so the note says what to press rather than leaving
            # "installed but not offered" to be discovered mid-graph.
            res["note"] = ("Installed. Restart the engine to load it into "
                           "a warm generator.")
        _publish(job_id, **res)
        # The catalogue rides the *return value*, never the record: the
        # record is polled every 400 ms and this is megabytes. The web
        # container reads the result once by call id and keeps it on its
        # own disk.
        return {**res, "catalogue": info}
    except StopRequested:
        res = {"status": "stopped", "job_id": job_id, "files": [],
               "duration_s": round(time.time() - started, 1)}
        _publish(job_id, **res)
        return res
    except Exception as exc:
        _publish(job_id, status="failed",
                 error=(str(exc) if isinstance(exc, (RuntimeError, ValueError))
                        else f"{type(exc).__name__}: {exc}"))
        raise


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
    image=web_image, cpu=1.0, timeout=900, volumes=MODAL_VOLUMES,
    max_containers=1,
    # Modal's maximum, and the one line that retires a family of bandages.
    # With no window set the container died about a minute after the last
    # request, so a cold start was the steady state and the mount was the
    # only place guaranteed to exist — which is why the read path, the
    # covers, the drafts and the session markers all ended up on the volume.
    # Twenty warm minutes cost cents; the spool stays warm across a coffee
    # and a draft on this disk outlives a glance away. See the Storage rule.
    scaledown_window=20 * 60,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    api = FastAPI()

    # **The first thing that happens to a request, and the last thing that was
    # visible.** `t_route` inside a handler starts after FastAPI has read the
    # body off the wire and parsed it — so a 48 MB upload spent its whole life
    # before any line this file prints. Somebody who gave up on the screen and
    # opened the dashboard found ComfyUI's own output either side of a gap with
    # nothing of ours in it, and this is the near end of that gap.
    #
    # Only the slow ones. `/api/status` is polled every 400ms and a line per
    # request would be a log made of polls — which is the same failure as no
    # log, arrived at from the other side.
    @api.middleware("http")
    async def _timed(request: Request, call_next):
        t0 = time.time()
        response = await call_next(request)
        took = time.time() - t0
        if took >= REQUEST_SLOW_S:
            n = request.headers.get("content-length")
            size = f", {int(n) / 1_000_000:.1f} MB in" if n and n.isdigit() else ""
            print(f"[api] {request.method} {request.url.path} took "
                  f"{took:.1f}s{size}", flush=True)
        return response

    # Where the build landed. A constant rather than a search, because a page
    # that cannot be found should say which path was empty — the same reason
    # _require_models() prints the path it wanted.
    # Built into web_image at this path, which is what keeps `modal deploy
    # app.py` the entire install. Locally the launcher runs the same
    # `npm ci && npm run build` and says where it put it; the 503 below lists
    # whatever it finds instead, which is the same answer either way.
    DIST = Path(os.environ.get("VISIONARY_DIST", "/build/web/dist"))

    # Hashed filenames, so the bytes at a given name never change and the
    # browser never needs to ask again. index.html is the opposite: it is the
    # one unhashed file and it is what points at the current hashes, so a cached
    # copy of it is a deploy that never arrives.
    #
    # check_dir=False because StaticFiles raises at construction on a missing
    # directory, and this line runs at import: a build that did not produce a
    # bundle would take the whole container down with a stack trace about a
    # path, rather than reaching the route below that explains what is missing.
    # A dead API is a worse answer than a page that says why it is empty.
    api.mount("/assets",
              StaticFiles(directory=str(DIST / "assets"), check_dir=False),
              name="assets")

    @api.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        page = DIST / "index.html"
        if not page.is_file():
            # Diagnosing itself rather than 500ing: this can only happen if the
            # image built without the front end, and the three facts that
            # separate "npm build failed" from "mounted at the wrong path" are
            # what it wants, not a traceback.
            listing = "\n".join(sorted(p.name for p in DIST.iterdir())) \
                if DIST.is_dir() else "(the directory does not exist)"
            return HTMLResponse(
                "<pre>No front end in this image.\n\n"
                f"wanted:  {page}\n"
                f"in {DIST}:\n{listing}\n</pre>",
                status_code=503,
            )
        return HTMLResponse(
            page.read_text(),
            headers={"cache-control": "no-store"},
        )

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
                    spec = CATALOGUE_LORA_ROOTS.get(str(d))
                    loras.append({
                        "name": d.name, "trigger_word": trigger,
                        "strength": None,
                        "path": str(files[0]),
                        # The sidecar is the tell: this platform's trainer wrote
                        # it, and the trainer trains Krea 2 RAW — so the folder's
                        # weights are Krea 2's. A folder with neither sidecar nor
                        # catalogue entry arrived by hand and claims nothing, so
                        # both pickers keep offering it.
                        "arch": (spec or {}).get(
                            "arch", "krea2" if meta.exists() else ""),
                        "internal": bool((spec or {}).get("internal")),
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
                        "catalogue": (spec or {}).get("family", ""),
                        "files": [{"name": f.name, "path": str(f)} for f in files],
                    })
                elif d.suffix == ".safetensors":
                    # No sidecar to read a trigger word out of, and no epochs to
                    # choose between — one file, one entry, named for itself. The
                    # catalogue may still know its phrase: the Krea style LoRAs
                    # land exactly here, and each is near-invisible until its
                    # trigger is in the prompt — so serving "" for them told the
                    # picker nothing about the one fact that decides whether the
                    # weight does anything on a first try.
                    spec = CATALOGUE_LORA_ROOTS.get(str(d))
                    loras.append({
                        "name": d.stem,
                        "trigger_word": KREA_STYLE_LORAS.get(d.stem, ""),
                        "strength": (KREA_STYLE_STRENGTH
                                     if d.stem in KREA_STYLE_LORAS else None),
                        "path": str(d),
                        "root": str(d),
                        "bytes": _tree_bytes(d),
                        "catalogue": (spec or {}).get("family", ""),
                        # A loose file claims no architecture unless the
                        # catalogue put it there — see the folder branch above.
                        "arch": (spec or {}).get("arch", ""),
                        "internal": bool((spec or {}).get("internal")),
                        "files": [{"name": d.name, "path": str(d)}],
                    })
        loras.sort(key=lambda l: l["name"].lower())
        # The menu is what the ✕ removes. Customs are deleted outright; a
        # built-in is baked into the image, so its ✕ hides it here instead —
        # while `_caption_models()` stays whole, because old job records and
        # the DEFAULT_CAPTION_MODEL fallback still resolve through it.
        try:
            hidden_caps = set(config.get("hidden_caption_models") or [])
        except Exception:
            hidden_caps = set()  # an unreachable Dict hides nothing, never empties the menu
        cap_models = {k: m for k, m in _caption_models().items()
                      if k not in hidden_caps}
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
                "both": {"options": list(BOTH_GPUS), "default": BOTH_GPU},
                # 0 where nobody said, which is every Modal deployment. A page
                # that wants to explain why a control is unavailable needs the
                # number, and deriving it from the card's *name* would be a
                # lookup table of every GPU there is.
                "vram_gb": GPU_VRAM_GB,
                "regional_min_vram_gb": REGIONAL_MIN_VRAM_GB,
            },
            "max_refs": MAX_H3_REFS,
            "max_ref_audios": MAX_H3_REF_AUDIOS,
            "max_ref_videos": MAX_H3_REF_VIDEOS,
            # Same reason as gpus: which controls each video model reads, and
            # what is on the volume for each of its tasks, are properties of
            # the deployment. The composer builds itself from this.
            "video_models": _video_model_status(),
            # The shot palette builds itself from these, for the same reason
            # the composer builds itself from `video_models`: a copy of the
            # vocabulary in the front end would be a second source of truth, and
            # the first pill added on one side and not the other would compile to
            # "No such shot pill" against the page that offered it.
            "shot_vocab": SHOT_VOCAB,
            "shot_langs": H3_LANGUAGES,
            "shot_roles": [dict(spec, key=k) for k, spec in SHOT_REF_ROLES.items()],
            # The instruction rides along now — the page shows it in a textarea
            # the preset prefills, so hiding it would be hiding the one thing
            # the control edits. Reproducibility moved with it: the job record
            # carries the exact text that ran, not just the key, so a run is
            # still replayable from its record after the preset changes.
            "caption_presets": [
                {"key": k, "label": p["label"], "note": p["note"],
                 "instruction": p["instruction"], "custom": bool(p.get("custom"))}
                for k, p in _caption_presets().items()
            ],
            # The repo is shown in the gear's Caption models section.
            "caption_models": [
                {"key": k, "label": m["label"], "note": m.get("note", ""),
                 "repo": m["repo"], "custom": bool(m.get("custom"))}
                for k, m in cap_models.items()
            ],
            # A hidden default would be a key the page's own menu does not
            # offer — the captioner select would sit blank on it — so the
            # default follows the menu.
            "caption_defaults": {
                "preset": DEFAULT_CAPTION_PRESET,
                "model": (DEFAULT_CAPTION_MODEL
                          if DEFAULT_CAPTION_MODEL in cap_models
                          else next(iter(cap_models), DEFAULT_CAPTION_MODEL)),
            },
            # The trainer's vocabulary, served for the reason every other table
            # here is: the form builds its menus out of this, so a value it can
            # send is a value the job will accept. A hardcoded list in the page
            # is a run that cold-starts a GPU to die on argparse.
            "train_optimizers": [dict(v, key=k) for k, v in TRAIN_OPTIMIZERS.items()],
            "lr_schedulers": [dict(v, key=k) for k, v in LR_SCHEDULERS.items()],
            "timestep_samplings": [dict(v, key=k) for k, v in TIMESTEP_SAMPLINGS.items()],
            "train_defaults": TRAIN_DEFAULTS,
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

        # The same discard the two output deletes do, for the same reason:
        # `_listed` keeps positives forever, so a LoRA proven present stays
        # proven present after it is gone unless the route that removed it says
        # so. Without this the next render validates a name ComfyUI will then
        # refuse as "Value not in list" — the failure a form error exists to
        # get in front of.
        _forget_listed(root.name, LORAS)
        _drop_legacy_trash(LORAS)
        volume.commit()
        return {"ok": True}

    @api.post("/api/token")
    def set_token(payload: dict) -> dict[str, Any]:
        if "hf_token" in payload:
            config["hf_token"] = str(payload.get("hf_token") or "").strip()
        return {"ok": True, "hf_token_set": bool(_hf_token())}

    @api.post("/api/warm")
    def warm_generator(payload: dict) -> dict[str, Any]:
        """
        Start the container this session will actually use, on page load.

        **It used to start the interpreter's L4, and that was worse than
        useless.** Nothing on any user-facing path answered there — yet every
        page load rented a card and paid a vLLM cold start for it. A warm-up
        that warms something no request will reach is not a no-op; it competes
        for the same quota as the container that *is* about to be asked for
        something.

        What it buys is exactly what `enter` does and nothing more: ComfyUI
        started and its custom nodes checked, about 25 seconds. **It does not
        leave the checkpoint resident** — this docstring claimed it did until
        2026-08-25, when a cold start spent 400 seconds loading weights that a
        page-load knock was supposed to have made resident. Weights load when
        the first graph asks for them, and that is deliberate: `setup` above
        carries the reason, which is that a model loaded at `enter` is loaded
        as a ComfyUI graph and holds the single queue for as long as it takes,
        at the exact moment a cold container is about to be asked for a render.

        `spawn`, so the page never waits, and errors are swallowed: a warm-up
        that fails is a slower first press, not something to put on screen.
        """
        kind = "video" if str(payload.get("kind") or "") == "video" else "image"
        try:
            # The base class, never a with_options variant: a knock on a card
            # nobody has picked would start a container nobody will use.
            _spawn(_host(kind, {"container": payload.get("container")}).warm)
        except Exception as exc:
            print(f"[warm] {kind} warm-up failed: {exc}", flush=True)
        return {"ok": True}

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
        _spawn(download_job, key)
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
        _spawn(gdrive_job, url, folder, bool(payload.get("refetch")))
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
        _spawn(download_missing_job, missing, job_id)
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
        sid = str(form.get("session") or "")
        if not appending:
            # Stamp the window before the first byte is written: an upload that
            # takes longer than the grace period would otherwise be writing into
            # a folder the sweep considers ownerless.
            _write_dataset_meta(raw, session=sid)
        elif sid and raw.parent == DRAFTS:
            # Re-stamp on append too. A draft created in a closed tab keeps
            # that tab's session id forever, so the window now uploading into
            # it was not the window keeping it alive — and fifteen quiet
            # minutes after the upload, the sweep took the whole set.
            _write_dataset_meta(raw, session=sid)

        count, zips = 0, []
        for up in form.getlist("files"):
            filename = getattr(up, "filename", None)
            if not filename:
                continue
            basename = Path(filename).name
            suffix = Path(basename).suffix.lower()
            if suffix not in IMAGE_EXTS and suffix not in VIDEO_EXTS \
                    and suffix not in {".zip", ".txt"}:
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
            elif suffix in VIDEO_EXTS:
                # No upright pass: rotation in a clip is a container-level
                # matrix that PIL cannot see and re-encoding to bake it in is a
                # transcode this container has no ffmpeg for. Noted rather than
                # half-done — see the TODO at `train_job`.
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
            return JSONResponse({"error": "No images or clips found in the upload."}, 400)

        # Thumbs are built on arrival, not on first view. The upload already
        # decoded every image once to bake the rotation in, and a set whose
        # first open fired eighty full-resolution decodes held the sheet
        # blank for thirty seconds — while holding the mount open, which is
        # what kept refusing the reloads (see the overlay note). Incremental:
        # an append stats the existing thumbs and builds only its own. Off
        # the event loop for the same reason the commit below is `.aio()` —
        # this is the one async handler, and a stretch of PIL decodes inline
        # would stall every poll the page has out.
        def _warm_thumbs() -> None:
            for img in _dataset_images(raw):
                try:
                    _ensure_thumb(raw, img)
                except Exception as exc:
                    print(f"[upload] thumb {img.name}: {exc}")

        import asyncio

        await asyncio.to_thread(_warm_thumbs)

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
        alive. This is also the *only* place drafts are swept now — the listing
        used to sweep too, and moving housekeeping off the route the page waits
        on is part of why Sets opens fast. A window left open on Generate still
        clears out the drafts of the one you closed, because the beat fires on
        load and then periodically wherever the app is open.
        """
        _reload_volume()
        DRAFTS.mkdir(parents=True, exist_ok=True)
        sid = str(payload.get("session") or "")
        _touch_session(sid)
        # Liveness follows attention, not authorship. A draft records the
        # session that *created* it, and that id dies with its tab — so a
        # draft reopened in a later window was being kept alive by a marker
        # nobody was touching, and the sweep took 80 images out from under the
        # person curating them. The beat therefore names the set on screen,
        # and a draft you are looking at becomes yours before anything sweeps.
        opened = str(payload.get("open") or "")
        if sid and opened and NAME_RE.match(sid) and NAME_RE.match(opened):
            od = DRAFTS / opened
            if od.is_dir():
                try:
                    cur = json.loads((od / "dataset.json").read_text()).get("session")
                except (OSError, json.JSONDecodeError):
                    cur = None
                if cur != sid:
                    _write_dataset_meta(od, session=sid)
        swept = _sweep_drafts()
        volume.commit()
        return {"ok": True, "swept": swept}

    @api.post("/api/export-scene")
    def export_scene(payload: dict) -> dict[str, Any]:
        """
        Queue a stitch of this scene's takes into one file.

        Validated here rather than in the job for the reason `/api/gdrive`
        gives: an empty chain is a form error, and finding it inside the job
        costs a container start before anything can say so. What this cannot
        check is whether the files are still on the volume — a reload settles
        that, and the job names the take by number when one is gone.

        One job id, not one per press: the export is a property of the scene on
        screen, so a second press replaces the first rather than racing it, the
        same way `GDRIVE_JOB` is one name.
        """
        takes = payload.get("takes")
        if not isinstance(takes, list) or not takes:
            return {"error": "Nothing to export — render a take first."}
        if len(takes) > 64:
            return {"error": f"{len(takes)} takes is past what one export "
                             "handles. Clear the scene and chain fewer."}
        rows = []
        for t in takes:
            if not isinstance(t, dict):
                continue
            name = Path(str(t.get("file") or "")).name
            if name and OUTPUT_FILE_RE.match(name):
                rows.append({"file": name})
        if not rows:
            return {"error": "The takes carried no filenames — reload and try "
                             "again."}
        _clear_stop(EXPORT_JOB)
        # Seeded before the spawn, so the first poll finds a record rather than
        # a 404 it has to read as "not started yet" — the contract every other
        # job here keeps.
        _publish(EXPORT_JOB, status="running", percent=0,
                 phase=f"Queued — {len(rows)} takes", files=[], error=None)
        _spawn(export_scene_job, EXPORT_JOB, rows)
        return {"ok": True, "job_id": EXPORT_JOB, "takes": len(rows)}

    # ── the Arsenal: characters ─────────────────────────────────────────────

    @api.get("/api/characters")
    def list_characters() -> dict[str, Any]:
        """
        The saved cast, for the mention picker. Names and notes only — the
        picker is typed into mid-sentence, so this must answer at popover
        speed, and the files come one at a time off their own route when
        somebody actually picks.
        """
        _reload_volume()
        CHARACTERS.mkdir(parents=True, exist_ok=True)
        out = []
        for d in sorted(CHARACTERS.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            meta = {}
            try:
                meta = json.loads((d / "character.json").read_text())
            except (OSError, json.JSONDecodeError):
                # A folder somebody made by hand is still a character — the
                # layout is the contract, and the receipt is optional the same
                # way a sidecar-less output still lists in the gallery.
                pass
            files = sorted(f.name for f in d.iterdir()
                           if f.is_file() and f.name != "character.json"
                           and not f.name.startswith(".")
                           and f.suffix.lower() != ".txt")
            row = {"handle": d.name,
                   "note": str(meta.get("note") or ""),
                   "retention": str(meta.get("retention") or ""),
                   "refs": meta.get("refs") or
                   [{"file": f} for f in files],
                   "files": files}
            # Only when there is one. The key is absent rather than null for a
            # character with no weight behind it, so the picker's rows do not
            # all carry a field three quarters of them cannot fill.
            if isinstance(meta.get("lora"), dict):
                row["lora"] = meta["lora"]
            out.append(row)
        return {"characters": out}

    @api.post("/api/characters/{handle}")
    def save_character(handle: str, payload: dict) -> dict[str, Any]:
        """
        Save one character, whole. Saving twice replaces — the folder is the
        record and a record does not accumulate versions of itself.

        Deliberate, never automatic: "it never remembers unless told" is the
        rule, and this route is the telling.
        """
        try:
            _check_name(handle)
        except ValueError as exc:
            return {"error": str(exc)}
        refs = payload.get("refs") or []
        weight = payload.get("lora")
        held = bool(isinstance(weight, dict) and str(weight.get("path") or ""))
        # **A LoRA counts as something to save, and this used to refuse it.**
        # "A character with no files is a name" was written when a character was
        # photographs, and on a platform whose other half trains weights it
        # refused the strongest case there is: somebody who exists as a LoRA and
        # no reference photo. Files *or* a weight — a name with neither is still
        # a name, and that is what this sentence was always about.
        if (not isinstance(refs, list) or not refs) and not held:
            return {"error": "A character with no photograph and no LoRA is a "
                             "name, and a name is already in your head. Attach "
                             "something first."}
        _reload_volume()
        d = CHARACTERS / handle
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        exts = {"image": "png", "audio": "wav", "video": "mp4"}
        recorded = []
        for i, r in enumerate(refs):
            kind = str(r.get("kind") or "image")
            blob = str(r.get("b64") or "")
            if not blob or kind not in exts:
                continue
            name = f"{i:02d}-{kind}.{exts[kind]}"
            try:
                (d / name).write_bytes(base64.b64decode(blob))
            except (ValueError, OSError) as exc:
                shutil.rmtree(d, ignore_errors=True)
                return {"error": f"Could not write {name}: {exc}"}
            entry: dict[str, Any] = {"file": name, "kind": kind}
            if r.get("note"):
                entry["note"] = _oneline(str(r["note"]))[:SHOT_VALUE_MAX]
            if r.get("sheet") and kind == "image":
                entry["sheet"] = True
            recorded.append(entry)
        note = _oneline(str(payload.get("note") or ""))[:SHOT_VALUE_MAX]
        # The note is its own .txt as well as a field in the receipt, because
        # the folder has to make sense in a terminal: cat note.txt is the
        # character, no app required.
        if note:
            (d / "note.txt").write_text(note + "\n")
        # **A pointer, never the weight.** The LoRA is already a file on this
        # volume under `loras/`, so what a character stores is the path to it —
        # copying a 300 MB weight into every character that uses it would make
        # saving a character cost what training one did, and two copies of one
        # file is two things to delete. The consequence is stated rather than
        # hidden: delete the LoRA and the character recalls without it, which
        # `hydrate` shows as a chip that is simply not there.
        #
        # It is here at all because this platform's other half is a trainer. A
        # character whose likeness *is* a trained weight is the case the whole
        # product is for, and a record that dropped it would save the least
        # valuable half of them.
        lora = payload.get("lora")
        record: dict[str, Any] = {
            "note": note,
            "retention": str(payload.get("retention") or ""),
            "refs": recorded,
            "saved": time.time(),
        }
        if isinstance(lora, dict) and str(lora.get("path") or ""):
            record["lora"] = {
                "path": str(lora["path"]),
                "rel": str(lora.get("rel") or ""),
                "strength": float(lora.get("strength") or 1),
            }
        (d / "character.json").write_text(json.dumps(record, indent=1))
        volume.commit()
        return {"ok": True, "handle": handle, "files": len(recorded)}

    @api.get("/api/character-file/{handle}/{name}")
    def character_file(handle: str, name: str) -> Any:
        """One saved file, streamed off the volume — bytes by their own route,
        never inlined into a listing the picker polls."""
        try:
            _check_name(handle)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        f = (CHARACTERS / handle / Path(name).name)
        if not f.is_file():
            _reload_volume()
        if not f.is_file():
            return JSONResponse({"error": f"No {name!r} for {handle!r}."},
                                status_code=404)
        return FileResponse(f)

    @api.get("/api/datasets")
    def list_datasets() -> dict[str, Any]:
        """
        Read-only, deliberately. The draft sweep used to run here too, which
        put a JSON read per draft — and a volume commit whenever anything
        swept — on the path the page waits on with a blank screen. The session
        heartbeat already sweeps, fires on page load and then periodically, and
        nobody is watching its latency; housekeeping lives there.
        """
        DATASETS.mkdir(parents=True, exist_ok=True)
        DRAFTS.mkdir(parents=True, exist_ok=True)
        # No reload, and stats carry the committed sidecar listing — one
        # recursive RPC for the whole library, never one per set: the per-set
        # version ran them sequentially and put 2.4s on the route the page
        # waits on with a blank screen. Drafts are this container's own disk
        # and have nothing committed to overlay.
        trees = {DATASETS: _committed_sidecar_tree(DATASETS), DRAFTS: {}}
        out = [
            _dataset_stats(d, trees[root].get(d.name))
            for root in (DATASETS, DRAFTS)
            for d in sorted(root.iterdir())
            if d.is_dir() and not d.name.startswith(".")
        ]
        out.sort(key=lambda r: -r["modified"])
        _warm_set_thumbs([DATASETS / r["name"] for r in out if r["saved"]])
        return {"datasets": out}

    _WARMED = {"done": False}

    def _warm_set_thumbs(sets: list[Path]) -> None:
        """
        Refill a cold container's thumbnails for the library, in the
        background, most recent set first.

        The upload builds a thumbnail on arrival, so a warm container never
        pays for one on first view. A container that has just started has
        none, and the grid would build them one tile at a time as it scrolled
        — which was the cost that used to justify keeping them on the volume.
        Once per container, bounded, and off the request thread: the listing
        returns at once and the grid fills as they land.
        """
        if _WARMED["done"]:
            return
        _WARMED["done"] = True

        def run() -> None:
            budget = 600
            for d in sets:
                for img in _dataset_images(d):
                    if budget <= 0:
                        return
                    try:
                        _ensure_thumb(d, img)
                    except Exception as exc:  # noqa: BLE001 — one bad file
                        print(f"[thumbs] {d.name}/{img.name}: {exc}", flush=True)
                    budget -= 1

        threading.Thread(target=run, name="warm-thumbs", daemon=True).start()

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
        # Across filesystems now — a draft is on this container's disk and
        # the library is the volume — so this is a copy and a delete, which
        # is what "saved" costs. The derived cache follows the name.
        shutil.move(str(d), str(target))
        if new != name and _set_cache(d).is_dir():
            shutil.rmtree(_set_cache(target), ignore_errors=True)
            shutil.move(str(_set_cache(d)), str(_set_cache(target)))
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
        d, err = _dataset_or_error(name)
        if err:
            return err
        # No reload: captions come through the committed-state overlay, and
        # everything else in this folder is this container's own writing.
        committed = _committed_sidecars(d)
        overlay = _caption_overlay(d, committed)

        from concurrent.futures import ThreadPoolExecutor

        from PIL import Image

        # The scan already paid for most dimensions: reuse its cache where the
        # stamp still matches, and read only the headers it has not covered.
        prints: dict[str, Any] = {}
        try:
            prints = json.loads((_set_cache(d) / FINGERPRINT_FILE).read_text())
        except (OSError, json.JSONDecodeError):
            prints = {}

        def measure(img: Path) -> dict[str, Any] | None:
            try:
                st = img.stat()
            except OSError:
                return None
            if img.suffix.lower() in VIDEO_EXTS:
                # No dimensions and no duration: both need a demuxer, and this
                # container has none. The tile paints its own first frame out of
                # bytes the browser fetches anyway — the same trade the gallery
                # card makes for a clip.
                return {"name": img.name, "kind": "video",
                        "caption": _caption_of(img, overlay),
                        "bytes": st.st_size, "mtime": st.st_mtime}
            # Pixel dimensions alongside filesize: together they are what
            # actually informs a keep/cut call. PIL parses the header only, so
            # this is a small read per file rather than a decode.
            w = h = None
            was = prints.get(img.name)
            if (was and was.get("stamp") == [st.st_mtime_ns, st.st_size]
                    and was.get("width")):
                w, h = was["width"], was["height"]
            else:
                try:
                    with Image.open(img) as im:
                        # The size after orientation, which is the size the
                        # browser draws and the size the bucketer will see.
                        # Reporting the stored one labelled a portrait photo
                        # "4032×3024".
                        w, h = _upright(im).size
                except Exception:
                    pass
            return {"name": img.name, "kind": "image",
                    "caption": _caption_of(img, overlay), "bytes": st.st_size,
                    "width": w, "height": h, "mtime": st.st_mtime}

        # Images first, then clips: a mixed set is browsed by kind far more
        # often than by name, and the filter above the grid is the same split.
        # Measured in parallel because each file is one FUSE round trip and the
        # sidecar beside it is a second: read one at a time on a cold volume,
        # an 80-image set held the sheet blank for ten-plus seconds.
        files = _dataset_images(d) + _dataset_videos(d)
        with ThreadPoolExecutor(max_workers=min(16, max(1, len(files)))) as ex:
            items = [r for r in ex.map(measure, files) if r]
        return {**_dataset_stats(d, committed), "images": items}

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
        shutil.rmtree(_set_cache(d), ignore_errors=True)
        _drop_legacy_trash(d.parent)
        volume.commit()
        return {"ok": True}

    def _ensure_thumb(d: Path, img: Path) -> Path:
        """The cached thumbnail for one image, built if the cache is stale.

        Cached by mtime, so re-editing a caption never re-encodes the image
        and replacing an image does invalidate it. On the container, never
        the volume: derived, and the writer that has the pixels rebuilds it —
        the upload warms it on arrival, and `_warm_set_thumbs` refills a cold
        container's for the library in the background.
        """
        from PIL import Image

        thumbs = _set_cache(d)
        thumbs.mkdir(parents=True, exist_ok=True)
        cached = thumbs / (img.stem + ".jpg")
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
        return cached

    @api.get("/api/thumb/{name}/{filename}")
    def thumb(name: str, filename: str):
        """One thumbnail, cached beside the dataset, served with a long
        max-age. The cache is usually warm — the upload builds it on arrival
        — so this is the fallback for sets that predate that."""
        from fastapi.responses import Response

        d, err = _dataset_or_error(name)
        if err:
            return JSONResponse(err, status_code=404)
        img = d / Path(filename).name  # basename only — no directory escape
        if img.suffix.lower() not in IMAGE_EXTS or not img.is_file():
            return JSONResponse({"error": "Image not found."}, status_code=404)

        try:
            data = _ensure_thumb(d, img).read_bytes()
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

    @api.get("/api/clip/{name}/{filename}")
    def dataset_clip(name: str, filename: str):
        """
        One clip off a dataset, streamed.

        `FileResponse` rather than the `Response(read_bytes())` its image
        sibling uses, and the difference is the whole reason this is a second
        route: a clip is tens of megabytes, so reading it into the container to
        avoid holding a descriptor would trade the thing that refuses
        `volume.reload()` for the thing that fills the container's memory. It
        also has to answer a Range request — a `<video>` seeking to `#t=` asks
        for the first few hundred kilobytes and nothing else, which is what
        makes a grid of clips affordable at all.

        The tiles gate this behind an IntersectionObserver for the same reason
        the gallery does: forty clips mounting at once is forty descriptors, and
        that is what froze the listing they were being shown in.
        """
        d, err = _dataset_or_error(name)
        if err:
            return JSONResponse(err, status_code=404)
        clip = d / Path(filename).name  # basename only — no directory escape
        if clip.suffix.lower() not in VIDEO_EXTS or not clip.is_file():
            return JSONResponse({"error": "Clip not found."}, status_code=404)
        return FileResponse(
            str(clip),
            media_type=MEDIA_TYPES.get(clip.suffix.lower(), "video/mp4"),
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @api.post("/api/datasets/{name}/caption")
    def save_caption(name: str, payload: dict) -> dict[str, Any]:
        """One caption, saved on blur. Bulk save was how edits went missing."""
        d, err = _dataset_or_error(name)
        if err:
            return err
        img = d / Path(str(payload.get("image") or "")).name
        # A clip's caption is a `.txt` beside it, same as an image's — the
        # sidecar layout is the contract, and a route that refused one would be
        # a caption box on the tile that silently saved nothing.
        if img.suffix.lower() not in IMAGE_EXTS | VIDEO_EXTS or not img.is_file():
            return {"error": "Image not found."}
        saved = str(payload.get("caption") or "").strip()[:MAX_CAPTION_CHARS]
        img.with_suffix(".txt").write_text(saved)
        volume.commit()
        # Echoed so the page patches its own state from the reply — the
        # response is the truth, and a refetch is a second chance to be wrong.
        return {"ok": True, "caption": saved}

    @api.post("/api/datasets/{name}/remove")
    def remove_image(name: str, payload: dict) -> dict[str, Any]:
        """
        Delete an image, its caption and its thumbnail. Not recoverable.

        Takes `image` or `images`, and the plural is the same route rather than
        a second one: a duplicate review resolves fourteen files in one press,
        and fourteen requests against a network volume is fourteen reloads,
        fourteen commits and a listing that is briefly right about a folder
        nobody is looking at any more. What the plural must not become is a
        looser guard — every name still goes through the same basename and
        extension check, and one bad name fails only itself.
        """
        _reload_volume()
        d, err = _dataset_or_error(name)
        if err:
            return err
        asked = payload.get("images")
        if not isinstance(asked, list):
            asked = [payload.get("image")]
        wanted = [str(x or "") for x in asked if str(x or "").strip()]
        if not wanted:
            return {"error": "No image named."}

        removed, missing = [], []
        for raw in wanted:
            img = d / Path(raw).name
            if img.suffix.lower() not in IMAGE_EXTS | VIDEO_EXTS or not img.is_file():
                missing.append(Path(raw).name)
                continue
            for part in (img, img.with_suffix(".txt")):
                part.unlink(missing_ok=True)
            (_set_cache(d) / (img.stem + ".jpg")).unlink(missing_ok=True)
            removed.append(img.name)
        _drop_legacy_trash(d)

        if not removed:
            # Singular and plural answer the same way they always did: one name
            # that resolves to nothing is an error, not a no-op reported as ok.
            return {"error": "Image not found." if len(wanted) == 1
                    else f"None of the {len(wanted)} images named are in this set."}
        # Nothing prunes the fingerprint cache here on purpose: the next scan
        # builds its map from the folder listing, so an entry for a file that is
        # gone is never read, and one that comes back under the same name is
        # caught by the (mtime, size) stamp.
        volume.commit()
        return {"ok": True, "removed": removed, "missing": missing,
                **_dataset_stats(d)}

    @api.get("/api/datasets/{name}/insight")
    def dataset_insight(name: str, trigger: str = "") -> dict[str, Any]:
        d, err = _dataset_or_error(name)
        if err:
            return err
        return _caption_insight(d, trigger)

    @api.get("/api/datasets/{name}/duplicates")
    def dataset_duplicates(name: str) -> dict[str, Any]:
        """
        The set's classified duplicate and review groups.

        Its own route rather than a field on `/insight`, because the two are
        priced differently: insight reads a few hundred `.txt` files and is
        refreshed on every caption edit, while this decodes every image in the
        folder the first time it runs. Folding it in would make saving one
        caption cost a full rescan.
        """
        _reload_volume()
        d, err = _dataset_or_error(name)
        if err:
            return err
        report = _duplicate_groups(d, SCAN_BUDGET_S)
        # Fingerprints are derived data, like thumbnails — but unlike a
        # thumbnail a rescan costs a decode of the whole folder, so the one
        # commit per scan is worth paying and the per-file one is not.
        if report.pop("_wrote", False):
            volume.commit()
        return report

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

        d, err = _dataset_or_error(name)
        if err:
            return err
        overlay = _caption_overlay(d)

        changed = 0
        for img in _dataset_images(d):
            txt = img.with_suffix(".txt")
            cur = _caption_of(img, overlay)
            if not cur:
                new = trigger
            else:
                # Composed from the caption with the trigger stripped, not
                # tested with a bare `startswith`: exact-case startswith let
                # "Chgl, …" collect a second "chgl, " on top, and a sidecar the
                # captioner had already doubled kept every copy. Building the
                # canonical form heals both, and skipping when it matches is
                # what keeps this idempotent.
                new = f"{trigger}, {_strip_leading_trigger(cur, trigger)}".rstrip(", ")
            if new == cur:
                continue
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
        if d.parent == DRAFTS:
            # The captioner is another machine, and a draft is on this one.
            return {"error": f"Save {name!r} first — captioning runs on a GPU "
                             f"container, and only a saved set is on the volume "
                             f"it reads."}
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
        presets, models = _caption_presets(), _caption_models()
        preset = str(payload.get("preset") or DEFAULT_CAPTION_PRESET)
        if preset not in presets:
            return {"error": f"No caption preset {preset!r}. "
                             f"One of: {', '.join(presets)}"}
        model = str(payload.get("model") or DEFAULT_CAPTION_MODEL)
        if model not in models:
            return {"error": f"No captioner {model!r}. "
                             f"One of: {', '.join(models)}"}
        write_mode = str(payload.get("write_mode") or "skip")
        if write_mode not in ("skip", "append", "prepend", "replace"):
            return {"error": f"No write mode {write_mode!r}. "
                             "One of: skip, append, prepend, replace"}

        # Clamped rather than refused: these arrive from spinner-less number
        # fields, and a typo of 3200 tokens should cost the typo, not the run.
        def _clamp(key: str, default: float, lo: float, hi: float) -> float:
            try:
                return min(hi, max(lo, float(payload.get(key, default))))
            except (TypeError, ValueError):
                return default

        job_id = f"cap{time.strftime('%Y%m%d%H%M%S')}{os.urandom(2).hex()}"
        _spawn(
            caption_job,
            job_id=job_id, dataset=name, trigger_word=trigger,
            preset=preset, model=model,
            write_mode=write_mode,
            instruction=str(payload.get("instruction") or "")[:4000],
            max_tokens=int(_clamp("max_tokens", 320, 16, 1024)),
            temperature=_clamp("temperature", 0.6, 0.0, 1.5),
            top_p=_clamp("top_p", 0.9, 0.05, 1.0),
        )
        return {"ok": True, "job_id": job_id}

    # ---- caption presets and models ---------------------------------------
    #
    # Both live in the `config` Dict beside the HF token, because they are the
    # same kind of thing: something typed into the UI once that every later
    # session should still have. The catalogue is not the model here — a
    # captioner is pulled into the HF cache on first use, not downloaded under
    # the gear — so these are rows in a menu, not entries with a `dest` path.

    @api.post("/api/caption/presets")
    def save_caption_preset(payload: dict) -> dict[str, Any]:
        label = str(payload.get("label") or "").strip()
        instruction = str(payload.get("instruction") or "").strip()
        if not label:
            return {"error": "A preset needs a name."}
        if not instruction:
            return {"error": "A preset needs an instruction."}
        key = _custom_key("preset", label)
        stored = dict(config.get("custom_caption_presets") or {})
        stored[key] = {
            "label": label, "instruction": instruction[:4000],
            # What the note line can say about a preset the server did not
            # write: whose it is.
            "note": "Your preset.",
        }
        config["custom_caption_presets"] = stored
        return {"ok": True, "key": key}

    @api.post("/api/caption/presets/delete")
    def delete_caption_preset(payload: dict) -> dict[str, Any]:
        key = str(payload.get("key") or "")
        stored = dict(config.get("custom_caption_presets") or {})
        if key not in stored:
            # Built-ins are not deletable — they are baked into the image, so a
            # delete could only hide one until the next deploy un-hid it.
            return {"error": f"No custom preset {key!r}."}
        del stored[key]
        config["custom_caption_presets"] = stored
        return {"ok": True}

    @api.post("/api/caption/models")
    def add_caption_model(payload: dict) -> dict[str, Any]:
        repo = str(payload.get("repo") or "").strip().strip("/")
        label = str(payload.get("label") or "").strip()
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
            return {"error": f"{repo or '(empty)'!s} is not a HuggingFace repo id. "
                             "The shape is owner/name, like Qwen/Qwen3-VL-8B-Instruct."}

        # Validated here, on the CPU container, because the alternative is a
        # cold GPU start and a 17 GB pull before a typo surfaces. config.json
        # answers both questions that matter in milliseconds: does the repo
        # resolve (typos, gated-without-token), and is it a vision LM at all.
        # The GPU load stays the final authority — transformers may still lack
        # a mapping for an exotic architecture — but that failure names the
        # repo and the architecture when it happens.
        from huggingface_hub import hf_hub_download
        try:
            cfg_path = hf_hub_download(
                repo, "config.json", token=_hf_token(),
                cache_dir=tempfile.mkdtemp(prefix="capcfg-"))
            cfg = json.loads(Path(cfg_path).read_text())
        except Exception as exc:
            hint = (" The repo is gated — paste an HF token above and accept "
                    "its licence." if "gated" in str(exc).lower() else "")
            return {"error": f"Could not read {repo}/config.json: {exc}.{hint}"}
        if not any("vision" in k for k in cfg):
            return {"error": f"{repo} does not look like a vision-language model "
                             f"(model_type {cfg.get('model_type')!r}, no vision "
                             "config). A captioner has to read images."}

        key = _custom_key("vlm", repo)
        stored = dict(config.get("custom_caption_models") or {})
        stored[key] = {
            "repo": repo, "label": label or repo.split("/")[-1],
            "note": "Added by you. First run pulls the weights.",
        }
        config["custom_caption_models"] = stored
        return {"ok": True, "key": key}

    @api.post("/api/caption/models/delete")
    def delete_caption_model(payload: dict) -> dict[str, Any]:
        key = str(payload.get("key") or "")
        stored = dict(config.get("custom_caption_models") or {})
        if key in stored:
            del stored[key]
            config["custom_caption_models"] = stored
            return {"ok": True}
        # A built-in cannot be deleted — it is baked into the image — so its ✕
        # hides it instead, and the hide lives in the config Dict so a redeploy
        # does not resurrect it. `state()` filters the menu; `_caption_models()`
        # stays whole so old job records and the default fallback still resolve.
        if key in CAPTION_MODELS:
            hidden = set(config.get("hidden_caption_models") or [])
            hidden.add(key)
            if not any(k not in hidden for k in _caption_models()):
                return {"error": "That is the last captioner on the menu, and "
                                 "every captioning run needs one. Add another "
                                 "model before removing it."}
            config["hidden_caption_models"] = sorted(hidden)
            return {"ok": True}
        return {"error": f"No captioner {key!r} — reopen Settings to refresh "
                         "the list."}

    @api.post("/api/models/tier")
    def set_weight_tier(payload: dict) -> dict[str, Any]:
        """
        Pin one slot to one file, or clear the pin and go back to automatic.

        The automatic answer is a working default — the largest tier on disk
        that claims to fit the card — so nobody has to make this choice to get
        started. This is for afterwards: somebody who wants to see what the
        bigger file looks like on a card that has to stream it, or who trusts
        their own measurement over the table's `fits_vram_gb`. It is not checked
        against the card, deliberately. That trade is priced by the person
        watching the render, and a refusal here would be this file having an
        opinion about their hardware. Same latitude the custom captioner row
        takes, and the pin lives in the same Dict so it survives a redeploy.
        """
        slot = str(payload.get("slot") or "")
        if slot not in SLOT_TIERS:
            return {"error": f"{slot!r} has no alternatives to choose between."}
        chosen = payload.get("key")
        pins = dict(config.get("weight_tiers") or {})
        if not chosen:
            pins.pop(slot, None)
            config["weight_tiers"] = pins
            return {"ok": True, "slot": slot, "using": _slot_name(slot)}
        if chosen not in (slot, *SLOT_TIERS[slot]):
            return {"error": f"{chosen!r} is not one of this slot's files."}
        spec = MODEL_CATALOGUE[chosen]
        if not spec["dest"].is_file():
            # The one refusal, and it is about a fact rather than a judgement:
            # a pin to a file nobody has downloaded would silently fall back,
            # which reads as the pin not working.
            return {"error": f"{spec['label']} is not downloaded yet — "
                             "get it under Models first."}
        pins[slot] = chosen
        config["weight_tiers"] = pins
        return {"ok": True, "slot": slot, "using": _slot_name(slot)}

    @api.post("/api/datasets/{name}/replace")
    def replace_in_captions(name: str, payload: dict) -> dict[str, Any]:
        """
        Find & replace across caption sidecars.

        The page sends the names in its current filtered view, so the filters
        are the targeting tool — "Uncaptioned" can never match anything, and a
        search narrows the blast radius to what is on screen. No names means
        the whole set, which is what an unfiltered view shows anyway.
        """
        find = str(payload.get("find") or "")
        if not find:
            return {"error": "Nothing to find."}
        replace = str(payload.get("replace") or "")
        match_case = bool(payload.get("match_case"))

        d, err = _dataset_or_error(name)
        if err:
            return err
        # Read through the overlay, never the bare mount. This is the route
        # that turned a stale view into committed data: it substituted in old
        # text and wrote the result back over a caption run's output, which
        # is why a replace could need three passes and still miss instances.
        overlay = _caption_overlay(d)

        wanted = payload.get("images")
        names = {str(n) for n in wanted} if isinstance(wanted, list) else None
        # A regex only for the case fold; the find string itself is literal.
        # The lambda replacement keeps backslashes in the replacement literal
        # too — re.sub would otherwise read "\1" as a group reference.
        pat = re.compile(re.escape(find), 0 if match_case else re.IGNORECASE)
        changed: dict[str, str] = {}
        # Clips too: their captions are the same sidecars, and a replace that
        # skipped them would be the caption box's silent no-op one route over.
        for img in _dataset_images(d) + _dataset_videos(d):
            if names is not None and img.name not in names:
                continue
            txt = img.with_suffix(".txt")
            if txt.name in overlay:
                cur = overlay[txt.name]
            elif txt.exists():
                cur = txt.read_text()
            else:
                continue
            new = pat.sub(lambda _m: replace, cur)
            if new != cur:
                final = new.strip()[:MAX_CAPTION_CHARS]
                txt.write_text(final)
                changed[img.name] = final
        volume.commit()
        # The new text rides the reply so the page patches its state directly
        # instead of refetching — see save_caption.
        return {"ok": True, "changed": len(changed), "captions": changed}

    # ---- training sessions ------------------------------------------------
    #
    # A card, not a page. The run used to be a console under the contact sheet,
    # which made "one at a time" a property of the UI rather than of the
    # backend — `train_job` shares nothing between runs and never did. What
    # these five routes add is the record that outlives the run: the setup you
    # can re-run, edit or delete, whether or not anything is training now.

    def _session_new() -> dict[str, Any]:
        return {
            "id": f"s{time.strftime('%Y%m%d%H%M%S')}{os.urandom(2).hex()}",
            "created": time.time(), "runs": 0, "job_id": "",
        }

    def _session_fields(payload: dict) -> dict[str, Any]:
        """
        The half of a session the form owns.

        Deliberately not validated the way starting one is: a session exists to
        be filled in over more than one sitting — picking "a new set" from the
        dataset menu saves the card and walks away to go make the set — so a
        half-written record is the normal state rather than an error. The
        refusals live on `start`, which is the moment the answers have to be
        real.
        """
        return {
            "lora_name": str(payload.get("lora_name") or "").strip()[:64],
            "trigger_word": str(payload.get("trigger_word") or "").strip()[:64],
            "dataset": str(payload.get("dataset") or "").strip()[:64],
            "params": _train_params(payload.get("params") or {}),
        }

    @api.get("/api/sessions")
    def list_sessions() -> dict[str, Any]:
        """
        Every card, with whatever its run is doing. One Dict read plus one per
        live job, and no volume touch at all — this is polled while a run is on,
        so the image and video counts a card shows are joined on the page out of
        the dataset listing it already holds rather than re-walked here.

        Bounded, and the bound is stated. Nothing sweeps a session — a card is
        the setup you re-run from, so it is deleted by hand or not at all — and
        this is polled every couple of seconds, which is exactly the shape the
        "keep the polled thing small" rule is about. The gallery's answer is the
        one taken here: serve the newest, say how many there are, and let the
        page say "showing 100 of 140" rather than quietly stopping at a hundred.
        """
        rows = _sessions_all()
        return {"sessions": [_session_view(r) for r in rows[:SESSION_LIST_MAX]],
                "total": len(rows)}

    @api.post("/api/sessions")
    def put_session(payload: dict) -> dict[str, Any]:
        """Create a card, or save an edit to one. `id` decides which."""
        sid = str(payload.get("id") or "").strip()
        rec = _session_get(sid) if sid else None
        if sid and not rec:
            return {"error": "That session is gone — it was deleted in another window."}
        if rec and _session_view(rec).get("status") in ("running", "queued"):
            # Editing the dials under a run would put the card and the process
            # out of step with no way to tell which one is the truth.
            return {"error": "That run is going. Stop it before editing the setup."}
        try:
            fields = _session_fields(payload)
        except ValueError as exc:
            return {"error": str(exc)}
        base = rec or _session_new()
        return {"ok": True, "session": _session_view(_session_put({**base, **fields}))}

    @api.post("/api/sessions/{sid}/start")
    def start_session(sid: str) -> dict[str, Any]:
        """
        Spawn the run this card describes.

        Idempotent against a double press the way `/api/download` is: a card
        already running answers with the job it is already running rather than
        starting a second one against the same output folder.
        """
        rec = _session_get(sid)
        if not rec:
            return {"error": "That session is gone — it was deleted in another window."}
        view = _session_view(rec)
        if view.get("status") in ("running", "queued"):
            return {"ok": True, "job_id": rec.get("job_id"), "already": True,
                    "session": view}

        _reload_volume()
        dataset = str(rec.get("dataset") or "")
        d, err = _dataset_or_error(dataset) if dataset else (None, {
            "error": "Pick a set to train on."})
        if err:
            return err
        if d.parent == DRAFTS:
            return {"error": f"Save {dataset!r} first — the trainer runs on a "
                             f"GPU container, and only a saved set is on the "
                             f"volume it reads."}
        if not _dataset_images(d):
            return {"error": f"{dataset!r} has no images."}
        lora_name = str(rec.get("lora_name") or "").strip()
        if not NAME_RE.match(lora_name):
            return {"error": "LoRA name: letters, digits, - and _ only."}
        trigger = str(rec.get("trigger_word") or "").strip()
        if not trigger:
            return {"error": "A trigger word is required."}
        try:
            params = _train_params(rec.get("params") or {})
        except ValueError as exc:
            return {"error": str(exc)}

        job_id = f"tr{time.strftime('%Y%m%d%H%M%S')}{os.urandom(2).hex()}"
        # Written before the spawn, not by the container. A trainer image cold
        # start is minutes, and a card whose job record does not exist yet is a
        # card that cannot say anything at all — which reads as a press that did
        # nothing, which is how you get two runs.
        jobs[job_id] = {"status": "queued", "phase": "queued", "stop": False,
                        "percent": 0, "session": sid,
                        "started": time.time(), "beat": time.time()}
        _spawn(
            train_job,
            job_id=job_id, dataset=dataset, lora_name=lora_name,
            trigger_word=trigger, session=sid, **params,
        )
        rec = _session_put({**rec, "job_id": job_id,
                            "runs": int(rec.get("runs") or 0) + 1})
        return {"ok": True, "job_id": job_id, "session": _session_view(rec)}

    @api.post("/api/sessions/{sid}/stop")
    def stop_session(sid: str) -> dict[str, Any]:
        """
        Cooperative, and the card stays. Checkpoints already written survive,
        which is what makes stopping a choice rather than a loss — and the run
        it stops is left on the card with the progress it reached, because that
        is what you re-run from.
        """
        rec = _session_get(sid)
        if not rec:
            return {"error": "That session is gone — it was deleted in another window."}
        job_id = str(rec.get("job_id") or "")
        if job_id:
            _request_stop(job_id)
        return {"ok": True, "session": _session_view(rec)}

    @api.post("/api/sessions/{sid}/delete")
    def delete_session(sid: str) -> dict[str, Any]:
        """
        The card goes. Anything it started is asked to stop on the way out —
        a deleted card that leaves a GPU container running is the one outcome
        nobody would predict from the word delete.

        Nothing on the volume is touched: the checkpoints under loras/ are the
        run's output, not the card's, and they are deleted where every other
        LoRA is deleted.
        """
        rec = _session_get(sid)
        if not rec:
            return {"ok": True, "gone": True}
        job_id = str(rec.get("job_id") or "")
        if job_id and _session_view(rec).get("status") in ("running", "queued"):
            _request_stop(job_id)
        _session_drop(sid)
        return {"ok": True}

    @api.post("/api/train")
    def train(payload: dict) -> dict[str, Any]:
        """
        Start a run in one call: make the card, then start it.

        Kept because a training run is startable without a page, and folded onto
        the session routes rather than kept beside them — the rule against a
        second way to do the first thing, applied to the one route that would
        otherwise have its own spawn, its own validation and its own idea of
        what a run is.
        """
        try:
            fields = _session_fields({**payload, "params": payload.get("params") or payload})
        except ValueError as exc:
            return {"error": str(exc)}
        rec = _session_put({**_session_new(), **fields})
        started = start_session(rec["id"])
        if started.get("error"):
            _session_drop(rec["id"])
        return started

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
            n_auds = max(0, min(MAX_H3_REF_AUDIOS, int(payload.get("ref_audios") or 0)))
            roles = _validate_ref_roles(payload.get("ref_roles"), n_refs)
        except (TypeError, ValueError) as exc:
            return {"error": str(exc)}

        if str(payload.get("kind") or "video") == "image":
            return {"prompt": _compile_image_prompt(typed, shot)}
        d = VIDEO_MODELS["h3"]["defaults"]
        # `or` rather than a `not in (None, "")` check, and deliberately: zero
        # seconds is a still, and `/api/video` compiles a still's document at
        # this same default because a still is a frame *out of* a shot. So zero
        # falling through to the default here is what keeps the preview and the
        # run saying the same thing — and a later edit "fixing" this to pass 0
        # through would give the composer a document the run does not use, which
        # is worse than no preview at all.
        try:
            seconds = float(payload.get("seconds") or d["seconds"])
        except (TypeError, ValueError):
            seconds = float(d["seconds"])
        # The scene is validated here rather than only on the way to the GPU,
        # because this route is what the composer polls on every keystroke: a
        # handle nobody defined should be a sentence under the timeline the
        # moment it is typed, not a refusal discovered at Generate.
        try:
            scene = _validate_scene(payload.get("scene"), n_refs=n_refs,
                                    n_vids=n_vids, n_auds=n_auds,
                                    seconds=seconds)
        except (TypeError, ValueError) as exc:
            return {"error": str(exc)}
        return {"prompt": _compile_h3_prompt(
            typed=typed, pills=shot, seconds=seconds, roles=roles, scene=scene,
            task=_h3_task(payload.get("first_frame"), payload.get("last_frame"),
                          n_refs, n_vids, n_auds),
        )}

    @api.post("/api/generate")
    def generate(payload: dict) -> dict[str, Any]:
        t_route = time.time()
        prompt = str(payload.get("prompt") or "").strip()
        regions = payload.get("regions") or []
        # A document is prose too, so it satisfies this the way a box does. The
        # guard is about having *something* to render, and "the prompt box is
        # empty but the model read a sentence out of it" is not a state this can
        # reach — but a client that sends only a document is not malformed, and
        # refusing it here would be the route disagreeing with the compiler
        # about what counts as a prompt.
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
            # Blocking, projected. **The same arrangement the video side turns
            # into prose, seen through the camera instead of described by it**
            # — a mark becomes the normalised 0..1 rectangle a region already
            # is, so this reaches Krea 2 as a projection rather than as a
            # second feature.
            #
            # Only when nothing was drawn. A hand-drawn box is somebody looking
            # at the frame and deciding, and an arrangement does not get to
            # overrule that any more than it overrules a chosen pill.
            stage = _validate_stage(payload.get("stage"), cast_ids=None)
            if stage and not regions:
                regions = [
                    {k: v for k, v in b.items()
                     if k != "castId" and v is not None}
                    for b in _stage_boxes(stage)
                ]
            regions = _validate_regions(regions)
            objects = _validate_objects(payload.get("objects"))
            style_refs = _validate_style_refs(payload.get("style_refs"))
            shot = _validate_shot(payload.get("shot"))
        except ValueError as exc:
            return {"error": str(exc)}

        # Only where the card has been named, so this is silent on Modal and on
        # any local run that did not say. The numbers are V12's own, not ours:
        # a block-mask build and a dense score tensor, both activations, both
        # larger than a consumer card whatever the checkpoint is quantised to.
        # In milliseconds and with the boxes still on screen, which is the
        # `_validate_loras()` rule — the alternative is an out-of-memory some
        # minutes into a warm GPU that reads as the app being broken.
        if regions and 0 < GPU_VRAM_GB < REGIONAL_MIN_VRAM_GB:
            return {"error": (
                f"Regional rendering needs about {REGIONAL_MIN_VRAM_GB:.0f} GB "
                f"and this card has {GPU_VRAM_GB:.0f} GB. The cost is a "
                f"30.3 GB block-mask and a 17.9 GB score tensor — working "
                f"memory, so a smaller checkpoint does not help. Remove the "
                f"boxes to render the frame whole, or run this one on a "
                f"rented card.")}

        # The job refuses this too; here it is a form error while both
        # attachments are on screen.
        if style_refs and regions:
            return {"error": "Style references and region boxes are two "
                             "engines — remove one. Style applies to the "
                             "whole frame."}

        # A plate still needs an identity to compose around, but a box is only
        # one way to hold it: with no boxes drawn, the job conjures a
        # full-canvas region out of the run's first LoRA chip — the commonest
        # render is one character with no boxes, and demanding a full-frame
        # box there was a gesture that added no information. Caught here when
        # neither exists, because the answer names two fixes and both are on
        # screen. Objects are plates too — same sockets, same engine.
        plated = [slot for slot in ("scene", "outfit") if payload.get(slot)]
        plated += ["object"] * len(objects)
        for slot in plated:
            if not regions:
                if not stack:
                    return {"error": f"A {slot} reference needs an identity "
                                     "to compose around — put a LoRA on the "
                                     "run, or draw a region holding a LoRA "
                                     "or a photo."}
                break
            # Same fact the job checks, caught while the plate is on screen:
            # the edit path only arms boxes holding a LoRA or a photo, and a
            # plate over described-only boxes is a dead job after a cold load
            # ("V12 unified attention was not armed").
            if not _armed_regions(regions):
                return {"error": f"A {slot} reference needs a box holding an "
                                 "identity — a LoRA or a photo. Boxes with "
                                 "only a description cannot anchor the "
                                 "compose."}

        # The Playground toggle, resolved here on CPU so a stale menu is a
        # form error rather than a cold H100.
        try:
            wf = _load_workflow_for_run(payload)
        except ValueError as exc:
            return {"error": str(exc)}

        job_id = f"gen{time.strftime('%Y%m%d%H%M%S')}{os.urandom(2).hex()}"
        _spawn(_host("image", payload).generate_image, job_id=job_id, params={
            **({"workflow": wf[0], "workflow_graph": wf[1],
                "workflow_extras": payload.get("workflow_extras") or {}}
               if wf else {}),
            # See the video side: the container subtracts this on arrival, and
            # the difference is a delivery that waited.
            "queued_at": time.time(),
            # Prose, not a document: Krea 2 has no fields to fill in, so the
            # same pills append their clauses to the sentence in the same order.
            # The rail is shared with the video side and the vocabulary decides
            # what crosses — camera, action, foley and score never reach here.
            "prompt": _compile_image_prompt(prompt, shot),
            "prompt_typed": prompt,
            "prompt_original": str(payload.get("prompt_original") or ""),
            "shot": shot,
            "negative_prompt": str(payload.get("negative_prompt") or ""),
            "model": str(payload.get("model") or "turbo"),
            "loras": stack,
            "regions": regions,
            "region_weight": num("region_weight", 1.0, float),
            # Base64, the same shape /api/video already takes its keyframes in.
            "scene": payload.get("scene"),
            "outfit": payload.get("outfit"),
            "objects": objects,
            "style_refs": style_refs,
            "style_strength": num("style_strength", 1.0, float),
            # None, not a default: the job derives compose_seed from the main
            # seed when unset, and edit_strength's default lives beside its
            # clamp so the two cannot drift.
            "edit_strength": num("edit_strength", None, float),
            "compose_seed": num("compose_seed", None, int),
            "width": num("width", 1024, int),
            "height": num("height", 1024, int),
            "num_images": max(1, min(4, num("num_images", 1, int))),
            "steps": num("steps", None, int),
            "cfg_scale": num("cfg_scale", None, float),
            "seed": num("seed", None, int),
            "sampler": str(payload.get("sampler")
                           or (STYLE_DEFAULTS if style_refs
                               else IMAGE_DEFAULTS)["sampler"]),
            "scheduler": str(payload.get("scheduler")
                             or (STYLE_DEFAULTS if style_refs
                                 else IMAGE_DEFAULTS)["scheduler"]),
            "shift": num("shift", 1.15, float),
        })
        _log_spawn("image", job_id, payload, t_route)
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
        t_route = time.time()
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

        # Motion continuation — the previous take's sampler latent, sliced in as
        # pinned context so motion and audio continue across the join instead of
        # restarting from a still. Checked here, on CPU, before a GPU is rented:
        # a missing latent is a form error, and the fix — clear the Motion tile,
        # fall back to the frame — is a thing to say while the person is still
        # at the controls. The job checks again at render time because the
        # volume can change between the two.
        continue_from = str(payload.get("continue_from") or "").strip() or None
        if still and continue_from:
            return {"error": "A still is one frame, so there is no motion to "
                             "continue — clear the Motion tile, or ask for a "
                             "take."}
        if continue_from:
            if not re.fullmatch(r"vid[0-9a-z]+", continue_from):
                return {"error": f"Not a video job id: {continue_from!r}"}
            if not (OUTPUTS / f"{continue_from}{H3MC_SUFFIX}").exists():
                _reload_volume()
            if not (OUTPUTS / f"{continue_from}{H3MC_SUFFIX}").exists():
                return {"error":
                        f"That take's motion context is no longer on the "
                        f"volume ({continue_from}) — it was rendered before "
                        f"chaining existed, or its generation was deleted. "
                        f"Clear the Motion tile to continue from its last "
                        f"frame instead."}
            # A pinned context anchors the opening the way a first frame would,
            # and the two are different transformers' jobs — sending both would
            # be two answers to "where does this take open". The page never
            # sends both; a stale tab might, and silence here would be the model
            # ignoring one of them with nothing saying which.
            if first:
                return {"error": "A motion continuation already opens where "
                                 "the last take ended — clear the first frame, "
                                 "or clear the Motion tile, but not both ways "
                                 "at once."}

        refs = [r for r in (payload.get("references") or []) if r][:MAX_H3_REFS]
        vids = [v for v in (payload.get("ref_videos") or []) if v][:MAX_H3_REF_VIDEOS]
        auds = [a for a in (payload.get("ref_audios") or []) if a][:MAX_H3_REF_AUDIOS]
        if (refs or vids or auds) and not supports["references"]:
            return {"error": f"{spec['label']} does not take references."}
        # Audio counts toward the same twelve. It was left out of this sum when
        # the audio channel landed, so 9 images + 3 videos + 3 audio passed a
        # check whose message says the limit is 12 — and the refusal would then
        # come from the node, mid-run, on a warm H100.
        if len(refs) + len(vids) + len(auds) > MAX_H3_REF_TOTAL:
            return {"error": f"{MAX_H3_REF_TOTAL} references in total is the "
                             f"model's limit ({len(refs)} images + "
                             f"{len(vids)} videos + {len(auds)} audio)."}

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
        aspect = str(payload.get("aspect") or "16:9")
        tier = str(payload.get("tier") or spec["defaults"]["tier"])
        try:
            task = "ref2va" if (refs or vids or auds) else "fl2va"
            _h3_canvas(aspect, tier)  # raises with the valid set named
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
        # **Read off the duration, not asked for**, the same way the task is
        # read off what was attached. Zero seconds is the one length whose
        # answer is a picture.
        still = seconds == 0
        if still and 0 not in (spec["lengths"] or []):
            return {"error": f"{spec['label']} does not make stills."}
        if still:
            # The compiler still needs a shot to describe, because a still is a
            # frame *out of* one — `[Shot 1]` and its timings are what the
            # document is made of, and a one-moment grammar beside it would be
            # a second way to say what that field already says. So the document
            # is compiled at the model's own default length and the sampler
            # renders the opening of it.
            seconds = float(d["seconds"])

        # The composer's timeline. Note that `scene` means something else one
        # route up: on `/api/generate` it is a base64 *plate* — a picture of a
        # place, frame-scope, beside `outfit`. Here it is the cast and the
        # shots. Two routes, two schemas, and the collision is written down
        # rather than renamed away because "scene" is the accurate word in both
        # and a reader who meets the second one cold should be told.
        try:
            scene = _validate_scene(payload.get("scene"), n_refs=len(refs),
                                    n_vids=len(vids), n_auds=len(auds),
                                    seconds=seconds)
        except (TypeError, ValueError) as exc:
            return {"error": str(exc)}

        # Compiled here rather than in the client, so there is one
        # implementation of the format, an unknown pill is a form error, and the
        # sidecar records exactly what the encoder was given.
        #
        # The task read here is finer than the one above. That one is right for
        # which checkpoint loads — first-only, last-only and both are the same
        # weights — and too coarse for the alignment instruction, where they are
        # three different sentences about where a picture sits in time.
        compiled = _compile_h3_prompt(
            typed=prompt, pills=shot, seconds=seconds, roles=roles,
            scene=scene, task=_h3_task(first, last, refs, vids, auds),
        )

        # **A document taken over by hand, and the one field that outranks the
        # compiler.** The page can open what would run as editable text — view
        # source rather than a disclosure — and an edit there has to be what
        # runs, or the surface is a lie about its own contents.
        #
        # It is a *separate* key from `prompt` on purpose, because the two halves
        # of a run mean different things and this is exactly the case that
        # separates them: `prompt_typed` stays the prose somebody wrote, and only
        # the receipt is overridden. Folding it into `prompt` would put a
        # six-field document into the sidecar's intent field, and Reuse would
        # then load a schema into the first shot's row and compile *that* on the
        # next run.
        # Stripped, never collapsed. `_oneline` exists because a newline inside a
        # *field* ends it early — and this is the document itself, where one field
        # per line is the format. Running it through that would fold six fields
        # into one.
        override = str(payload.get("prompt_compiled") or "").strip()
        if override:
            if len(override) > MAX_H3_PROMPT and model == "h3":
                return {"error": f"The document is {len(override)} characters and "
                                 f"H3's prompt field holds {MAX_H3_PROMPT}."}
            compiled = override

        # The Playground toggle — same CPU-side read as the image route.
        try:
            wf = _load_workflow_for_run(payload)
        except ValueError as exc:
            return {"error": str(exc)}

        job_id = f"vid{time.strftime('%Y%m%d%H%M%S')}{os.urandom(2).hex()}"
        _spawn(_host("video", payload).generate_video, job_id=job_id, params={
            **({"workflow": wf[0], "workflow_graph": wf[1],
                "workflow_extras": payload.get("workflow_extras") or {}}
               if wf else {}),
            # When it was handed to Modal. The container subtracts this on
            # arrival, which is the only way to see a delivery that waited —
            # see `_note_queue_wait`.
            "queued_at": time.time(),
            "model": model,
            # Both travel. `prompt` is what runs and is the only one the graph
            # sees; the other two are what you chose, and they exist so a
            # gallery card can show a sentence instead of a six-field document
            # and so Reuse puts the pills back rather than the output of them.
            "prompt": compiled,
            "prompt_typed": prompt,
            "prompt_original": str(payload.get("prompt_original") or ""),
            "shot": shot,
            "scene": scene,
            "continue_from": continue_from,
            "ref_roles": roles,
            # No negative prompt on this path at all. H3 is guidance-distilled,
            # so one would not be applied — and a sidecar that records an input
            # the model never read is a sidecar that lies about how the clip was
            # made. It was a per-model question while there were two families.
            "aspect": aspect,
            "tier": tier,
            # Both travel on a still. `seconds` is what the *document* was
            # compiled against — the shot the frame is taken out of — and
            # dropping it here would make the sidecar disagree with the prompt
            # it sits beside.
            "still": still,
            "seconds": seconds,
            "steps": max(1, min(60, num("steps", d["steps"], int))),
            "seed": num("seed", None, int),
            # H3's flow shifts. `None` means "the model's default", which is the
            # honest empty value — the node at rest is a no-op, and it stops
            # being one under a distilled LoRA, which is the only reason it is
            # in the graph. There was a shared `shift` key here for as long as
            # there were two families, and it defaulted to Wan's 8.0; reading it
            # on an H3 take put 8.0 against the model's own 12.0, which is the
            # trap that made these two keys separate in the first place.
            "shift_video": num("shift_video", None, float),
            "shift_audio": num("shift_audio", None, float),
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
            "ref_audios": auds,
            "ref_size": ref_size,
            "first_frame": first,
            "last_frame": last,
        })
        _log_spawn("video", job_id, payload, t_route)
        return {"ok": True, "job_id": job_id, "model": model, "mode": task}

    @api.get("/api/gallery")
    def gallery(before: float = 0.0, limit: int = 200) -> dict[str, Any]:
        """
        A page of everything on the volume, newest first — no job id required.

        `stale` used to mean "this listing may be missing the run you just
        made". It cannot mean that any more: `_output_entries` asks Modal
        rather than the mount, so the *set* is right whether or not a reload
        landed. What a refused reload still costs is the mount — the sidecars
        read below, and the covers this reply is about to send the page after.
        So `stale` now means: **the mount is behind this listing, so the newest
        items may be thin and their pictures may not resolve yet.**

        Narrower, and still worth reporting. It is the difference between
        "nothing new" and "not looked at", and the client's cue to come back
        once it has stopped loading pictures. Not an error, and not something
        to apologise for on screen.

        `limit` is clamped rather than trusted: the sidecar read is per item,
        so an unbounded page is an unbounded number of file reads on a route
        anyone can call.

        No reload, and no `_reload_insist` sleeping in the handler: every part
        of this listing — the entry set, the mtimes, the sidecars — now comes
        off committed state by RPC, and the covers the page asks for next are
        served off the spool the same way. `stale` is kept in the reply for
        the page that still reads it, and it is now always false, because
        there is no mount left in this path to be behind.
        """
        items, total, legacy = _gallery(limit=max(1, min(limit, 500)), before=before)
        out: dict[str, Any] = {"items": items, "total": total, "stale": False}
        if legacy:
            # Job folders from before the record lived inside the file. They
            # are moved into place by a one-time job, started here on first
            # sight; until it lands they are not results the page can address,
            # so the reply says how many are still on their way.
            out["migrating"] = {"job_id": _start_migration(), "pending": len(legacy)}
        return out

    @api.get("/api/file/{name}")
    def output_file(name: str):
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
        The separators have to be visible to the regex to be rejected. One
        segment: outputs/ is flat and the name carries its run.

        **Served off the spool, never the mount.** The mount needs a reload to
        see a fresh run, reload is refusable — by this route's own descriptors
        most of all — and every "broken picture on a run that is sitting on
        the volume" traced back to that loop. The spool pulls committed bytes
        by RPC, which no open file can refuse, so a render that has finished
        is a render this route can serve, first ask included. It also puts a
        clip's range requests on local disk: the browser re-asking for byte
        ranges used to be a descriptor on /workspace per ask.
        """
        if not OUTPUT_FILE_RE.match(name):
            return JSONResponse({"error": "Invalid name."}, status_code=400)

        path = _spooled(f"outputs/{name}")
        if path is None:
            # The mount, for the one thing the spool cannot answer: bytes
            # written on THIS container that were never committed, and any
            # RPC failure. `_sizes_on_disk` rather than `is_file()` because
            # a first miss cached a negative entry the reload does not clear.
            mounted = OUTPUTS / name
            if not mounted.is_file():
                _reload_volume()
                if not _sizes_on_disk([mounted])[mounted]:
                    return JSONResponse({"error": "Not found."}, status_code=404)
            path = mounted
        return FileResponse(
            str(path),
            media_type=MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"),
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @api.get("/api/cover/{name}")
    def output_cover(name: str):
        """
        One gallery cover: the same result at 320px, and never a descriptor.

        The grid had no thumbnail at all — a 232px cell was served the full
        1024px PNG, and the drawer, the mobile grid at 104px and the 36px
        last-generation button all did the same. That is roughly 37x the bytes
        it needs, but the bytes were the cheaper half of the cost.

        The expensive half is that `/api/file` answers with `FileResponse`,
        which holds a descriptor open on /workspace for the length of the
        transfer *to the client*. Reading the bytes into memory and answering
        with `Response` closes the descriptor before anything goes on the
        wire, which is the entire point of this route existing rather than a
        `?w=320` on the other one.

        **Derived, so it lives on this container and nowhere else.** A copy
        used to be written onto the volume "for the next container", which
        was reasoning correctly about a web container that died a minute
        after the last request. With the scaledown window at twenty minutes
        the spool is warm for a session, and a cover rebuilt after a genuine
        cold start is the honest price of not keeping caches on the volume.

        Named `/api/cover/...` rather than `/api/thumb/...` because the dataset
        thumbnail route is two segments: FastAPI resolves by registration
        order, and a gallery cover reaching that handler 404s as "Image not
        found" for a dataset that was never named.
        """
        from fastapi.responses import Response
        from PIL import Image

        if not OUTPUT_FILE_RE.match(name):
            return JSONResponse({"error": "Invalid name."}, status_code=400)
        if name.lower().endswith(".mp4"):
            # Not a fallthrough to the clip. A cover route that sometimes
            # answers with five megabytes of mp4 is the thing it exists to
            # prevent; the card falls back to a gated <video>, which paints a
            # frame from bytes it had to fetch anyway once it is on screen.
            return JSONResponse(
                {"error": "No cover for a clip: web_image has no ffmpeg."},
                status_code=404)

        # The size is in the cache name, so raising the constant invalidates
        # by construction. No mtime check beside it: a finished file never
        # changes.
        local = SPOOL / "outputs" / ".covers" / f"{name}@{THUMB_PX}.jpg"
        try:
            if local.is_file():
                os.utime(local)
                data = local.read_bytes()
            else:
                src = _spooled(f"outputs/{name}")
                if src is None:
                    # The mount as last resort, same two-step as /api/file.
                    src = OUTPUTS / name
                    if not src.is_file():
                        _reload_volume()
                        if not _sizes_on_disk([src])[src]:
                            return JSONResponse({"error": "Not found."},
                                                status_code=404)
                with Image.open(src) as im:
                    # Upright even for our own renders: the viewer shows this
                    # same file at full size and browsers rotate from EXIF
                    # while PIL does not, so without this the card and the
                    # full-screen view of one file disagree by 90°.
                    im = _upright(im).convert("RGB")
                    im.thumbnail((THUMB_PX, THUMB_PX), Image.LANCZOS)
                    buf = io.BytesIO()
                    im.save(buf, "JPEG", quality=78, optimize=True)
                data = buf.getvalue()
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_bytes(data)
        except Exception as exc:
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

        # A day rather than /api/file's hour: a finished file never changes,
        # so a cover derived from one is immutable in a way a dataset image —
        # which you can replace in place — is not.
        return Response(content=data, media_type="image/jpeg",
                        headers={"Cache-Control": "private, max-age=86400"})

    @api.post("/api/outputs/{job_id}/delete")
    def delete_output(job_id: str) -> dict[str, Any]:
        """Delete a run and every file of it — the results and the motion
        context beside them. Unlinked, not recoverable."""
        if not NAME_RE.match(job_id):
            return {"error": "Invalid job_id."}
        # Insisting, because the card you are most likely to delete on impulse
        # is the one that just appeared, and a refused reload turns that into
        # "Not found" about files that are sitting on the volume.
        _reload_insist()
        files = _group_files(job_id)
        if not files:
            return {"error": "Not found."}
        for name in files:
            (OUTPUTS / name).unlink(missing_ok=True)
            _forget_listed(name, OUTPUTS)
        _forget_output(job_id)
        volume.commit()
        return {"ok": True, "removed": len(files)}

    @api.post("/api/outputs/purge")
    def purge_outputs(payload: dict) -> dict[str, Any]:
        """
        Delete many runs in one request.

        Two guards, because deletion here does not go anywhere first.

        `confirm` has to be in the body, so a bare POST to a guessed URL cannot
        fire it. And the caller names the runs rather than describing them:
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

        _reload_insist()
        removed, missing = 0, []
        for job_id in dict.fromkeys(job_ids):
            files = _group_files(job_id)
            if files:
                for name in files:
                    (OUTPUTS / name).unlink(missing_ok=True)
                    _forget_listed(name, OUTPUTS)
                _forget_output(job_id)
                removed += 1
            else:
                missing.append(job_id)

        _drop_legacy_trash(OUTPUTS)
        volume.commit()
        # Named, not just subtracted from the count. The list is the agreement,
        # so a run in it that could not be found is the one thing this route
        # owes an answer about — and the cause is nearly always a view too old
        # to hold it, which is a different problem from a bad id.
        return {"ok": True, "removed": removed,
                **({"missing": missing} if missing else {})}

    @api.get("/api/outputs/zip")
    def zip_outputs(ids: str = ""):
        """
        One zip of the named results, built from committed state on local
        disk and streamed from there.

        A GET, because a download has to be a navigable URL — an <a> click is
        the one gesture every browser turns into a saved file without holding
        the bytes in page memory, which a fetch-into-blob would and a
        selection of clips would blow through. ZIP_STORED, because every
        member is already compressed media and deflating it again is CPU
        spent making the file marginally larger. Built
        from `_spooled`, so nothing here opens the mount and a slow selection
        cannot refuse anyone's reload.
        """
        from fastapi.responses import FileResponse
        from starlette.background import BackgroundTask

        job_ids = [j for j in ids.split(",") if j]
        if not job_ids or len(job_ids) > 500:
            return JSONResponse({"error": "ids is a comma-separated list of "
                                          "job ids (at most 500)."},
                                status_code=400)
        if any(not NAME_RE.match(j) for j in job_ids):
            return JSONResponse({"error": "Invalid job_id."}, status_code=400)

        entries, _legacy = _output_entries()
        picked = [(j, sorted(n for n, _ in entries.get(j, [])))
                  for j in dict.fromkeys(job_ids)]
        if not any(files for _, files in picked):
            return JSONResponse({"error": "None of those results were "
                                          "found."}, status_code=404)

        fd, tmp = tempfile.mkstemp(suffix=".zip", prefix="visionary-sel-")
        os.close(fd)
        out = Path(tmp)
        n = 0
        try:
            with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:
                for _job_id, files_ in picked:
                    for fname in files_:
                        src = _spooled(f"outputs/{fname}")
                        if src is not None:
                            # Its own name: the run is in it, so two runs'
                            # files can no longer both be called 00.png.
                            zf.write(src, fname)
                            n += 1
        except Exception:
            out.unlink(missing_ok=True)
            raise
        return FileResponse(
            str(out), media_type="application/zip",
            filename=f"visionary-{n}-files.zip",
            # Unlinked once the response has streamed. It lives on /tmp, so
            # the open descriptor refuses nothing while it does.
            background=BackgroundTask(out.unlink, missing_ok=True),
        )

    # There was a GET /api/outputs/{job_id} here that returned every PNG of a run
    # base64'd into one JSON body. It is gone rather than left unused: /api/file
    # already serves the same bytes, streamed and cacheable, and it is what the
    # gallery, the drawer and now the canvas all use. Two routes for one job,
    # where the second one is strictly slower, is the shape a future change
    # picks the wrong half of.

    # ── Playground ─────────────────────────────────────────────────────────

    @api.post("/api/playground/seed")
    def playground_seed(payload: dict) -> dict[str, Any]:
        """
        The app's own live graph, built from the console's state and returned
        without running — what the Playground opens on, so nothing is ever
        blank and every experiment starts from something that works.

        Attachment *bytes* do not cross into a seed: references and keyframes
        stage on the GPU container at run time, and a placeholder file here on
        CPU would fail `_fit_reference` on a picture that does not exist. What
        was attached is named back in `dropped` so the room can say so instead
        of the seed silently being smaller than the console.
        """
        kind = ("video" if str(payload.get("kind") or "image") == "video"
                else "image")
        prompt = str(payload.get("prompt") or "")

        def num(k, d, cast):
            try:
                v = payload.get(k)
                return cast(v) if v not in (None, "") else d
            except (TypeError, ValueError):
                return d

        _reload_volume()
        try:
            shot = _validate_shot(payload.get("shot"))
        except ValueError as exc:
            return {"error": str(exc)}
        seed_v = num("seed", None, int)
        if seed_v is None:
            seed_v = int.from_bytes(os.urandom(4), "big")
        dropped = [k for k in ("references", "ref_videos", "ref_audios",
                               "first_frame", "last_frame", "scene", "outfit",
                               "style_refs", "regions", "objects")
                   if payload.get(k)]

        try:
            if kind == "image":
                model = ("turbo" if str(payload.get("model") or "turbo")
                         != "raw" else "raw")
                graph = _krea2_graph(
                    model=model,
                    prompt=_compile_image_prompt(prompt, shot),
                    negative_prompt=str(payload.get("negative_prompt") or ""),
                    width=num("width", 1024, int),
                    height=num("height", 1024, int),
                    batch_size=max(1, min(4, num("num_images", 1, int))),
                    seed=seed_v,
                    steps=num("steps", KREA2_DEFAULTS[model]["steps"], int),
                    cfg=num("cfg_scale", KREA2_DEFAULTS[model]["cfg"], float),
                    shift=num("shift", 1.15, float),
                    sampler=str(payload.get("sampler")
                                or IMAGE_DEFAULTS["sampler"]),
                    scheduler=str(payload.get("scheduler")
                                  or IMAGE_DEFAULTS["scheduler"]),
                    loras=_validate_loras(payload.get("loras")),
                    regions=[],
                )
            else:
                d = VIDEO_MODELS["h3"]["defaults"]
                aspect = str(payload.get("aspect") or "16:9")
                tier = str(payload.get("tier") or d["tier"])
                seconds = num("seconds", float(d["seconds"]), float)
                still = seconds == 0
                if still:
                    # Same fact as /api/video: the document describes a shot
                    # and a still is a frame out of it, so it compiles at the
                    # model's own default length.
                    seconds = float(d["seconds"])
                width, height = _h3_canvas(aspect, tier)
                graph = _h3_graph(
                    prompt=_compile_h3_prompt(
                        typed=prompt, pills=shot, seconds=seconds, roles=[],
                        scene=None, task=_h3_task(None, None, [], [], [])),
                    width=width, height=height,
                    frames=(H3_STILL_FRAMES if still
                            else _h3_frames(seconds)),
                    seed=seed_v,
                    steps=max(1, min(60, num("steps", d["steps"], int))),
                    sampler=str(payload.get("sampler") or d["sampler"]),
                    scheduler=str(payload.get("scheduler") or d["scheduler"]),
                    loras=_validate_video_loras(payload.get("loras")),
                    shift_video=num("shift_video", None, float),
                    shift_audio=num("shift_audio", None, float),
                    still=still,
                )
        except ValueError as exc:
            return {"error": str(exc)}

        return {"ok": True, "kind": kind, "graph": graph,
                **({"dropped": dropped} if dropped else {})}

    @api.post("/api/playground/run")
    def playground_run(payload: dict) -> dict[str, Any]:
        """
        Queue one user-authored graph — the Playground's Generate.

        Validation here is structural zeros against the cached catalogue;
        whether the wiring makes sense is ComfyUI's call on the GPU, where a
        failure comes back naming the node. Status and Stop are the routes
        every other job already answers to.
        """
        t_route = time.time()
        known = None
        raw = _node_catalogue_bytes()
        if raw:
            try:
                known = set(json.loads(raw))
            except ValueError:
                known = None
        try:
            graph = _validate_playground_graph(payload.get("graph"), known)
        except ValueError as exc:
            return {"error": str(exc)}
        atts = payload.get("attachments") or {}
        if (not isinstance(atts, dict)
                or any(not isinstance(v, str) for v in atts.values())):
            return {"error": "attachments is {filename: base64}."}
        if len(atts) > 16:
            return {"error": f"{len(atts)} attachments — the cap is 16 a "
                             "run, because each one rides the request body."}
        host = ("video" if str(payload.get("host") or "image") == "video"
                else "image")
        job_id = f"pg{time.strftime('%Y%m%d%H%M%S')}{os.urandom(2).hex()}"
        _spawn(_host(host, payload).playground, job_id=job_id, params={
            "queued_at": time.time(),
            "graph": graph,
            "attachments": atts,
            "workflow_name": str(payload.get("workflow_name") or ""),
        })
        _log_spawn("playground", job_id, payload, t_route)
        return {"ok": True, "job_id": job_id}

    @api.post("/api/playground/restart")
    def playground_restart(payload: dict) -> dict[str, Any]:
        """The engine-restart lever — a job, so the page can watch it land."""
        host = ("video" if str(payload.get("host") or "image") == "video"
                else "image")
        job_id = f"pgrst{time.strftime('%Y%m%d%H%M%S')}{os.urandom(2).hex()}"
        _spawn(_host(host, payload).restart_engine, job_id=job_id)
        return {"ok": True, "job_id": job_id}

    @api.get("/api/playground/nodes")
    def playground_nodes():
        """
        The node catalogue — /object_info as one cached file.

        Committed state by RPC, never the mount: the writer is a CPU
        container. Raw bytes out, because the catalogue is megabytes and a
        decode-and-re-encode here would buy nothing.
        """
        from fastapi.responses import Response
        raw = _node_catalogue_bytes()
        if raw is None:
            # Nothing on this container and no result to read: harvest, as a
            # visible job, and say so — the room polls until it lands.
            return Response(json.dumps({"missing": True,
                                        "harvesting": _harvest_if_missing()}),
                            media_type="application/json")
        return Response(raw, media_type="application/json")

    def _spawn_catalogue(job_id: str, **kw: Any) -> None:
        """Spawn a harvest and remember the call, so its return value — the
        catalogue itself — can be read back by whichever web container is
        alive when it lands."""
        _publish(job_id, status="running", percent=0, phase="Queued", files=[])
        fc = _spawn(playground_catalogue_job, job_id=job_id, **kw)
        # None locally: there is one process and one disk, so the harvest writes
        # NODE_CATALOGUE where the reader already looks and the call id has
        # nothing to recover. Remotely it is the handle, and recording it is how
        # a *different* web container reads the result back.
        if fc is None:
            return
        try:
            config["node_catalogue_call"] = {"job_id": job_id, "call_id": fc.object_id}
        except Exception as exc:  # noqa: BLE001 — a lost id is a re-harvest later
            print(f"[catalogue] could not record call {fc.object_id}: {exc}", flush=True)

    def _node_catalogue_bytes() -> bytes | None:
        """
        The catalogue, off this container's disk — or off the last harvest's
        return value, read once and kept here. None when neither exists.

        Derived data, so it never touches the volume: a cold container reads
        the last harvest back by its call id, and only when Modal no longer
        holds that result does a harvest have to run again.
        """
        try:
            if NODE_CATALOGUE.is_file():
                return NODE_CATALOGUE.read_bytes()
        except OSError:
            pass
        try:
            call = config.get("node_catalogue_call") or {}
            cid = str(call.get("call_id") or "")
            if not cid:
                return None
            res = modal.FunctionCall.from_id(cid).get(timeout=0)
        except Exception:  # noqa: BLE001 — not done, expired, or gone
            return None
        info = (res or {}).get("catalogue") if isinstance(res, dict) else None
        if not info:
            return None
        raw = json.dumps(info).encode()
        try:
            NODE_CATALOGUE.parent.mkdir(parents=True, exist_ok=True)
            NODE_CATALOGUE.write_bytes(raw)
        except OSError:
            pass
        return raw

    def _harvest_if_missing() -> str:
        """One harvest at a time: the recorded call's job is the lock."""
        try:
            call = config.get("node_catalogue_call") or {}
            cur = jobs.get(str(call.get("job_id") or "")) or {}
            if cur.get("status") == "running":
                return str(call["job_id"])
        except Exception:  # noqa: BLE001
            pass
        job_id = f"pgcat{time.strftime('%Y%m%d%H%M%S')}{os.urandom(2).hex()}"
        _spawn_catalogue(job_id)
        return job_id

    @api.post("/api/playground/refresh")
    def playground_refresh() -> dict[str, Any]:
        """Re-harvest the catalogue — a visible job, since it boots ComfyUI."""
        job_id = f"pgcat{time.strftime('%Y%m%d%H%M%S')}{os.urandom(2).hex()}"
        _spawn_catalogue(job_id)
        return {"ok": True, "job_id": job_id}

    @api.get("/api/playground/packs")
    def playground_packs() -> dict[str, Any]:
        _reload_volume()
        rows = []
        if PLAYGROUND_NODES.is_dir():
            for d in sorted(PLAYGROUND_NODES.iterdir()):
                if not d.is_dir() or d.name.startswith("."):
                    continue
                pin = {}
                pf = d / ".visionary-pin"
                if pf.is_file():
                    try:
                        pin = json.loads(pf.read_text())
                    except ValueError:
                        pin = {}
                rows.append({"name": d.name, "url": pin.get("url"),
                             "sha": (pin.get("sha") or "")[:12],
                             "installed": pin.get("installed")})
        return {"packs": rows}

    @api.post("/api/playground/packs")
    def playground_pack_install(payload: dict) -> dict[str, Any]:
        url = str(payload.get("url") or "").strip()
        # An https git URL and nothing else — a pack install is a clone, and
        # the clone is the one thing on this surface that reaches out.
        if not re.fullmatch(r"https://[\w.-]+/[\w./~-]+", url):
            return {"error": "A pack installs from an https git URL — e.g. "
                             "https://github.com/owner/repo."}
        job_id = f"pack{time.strftime('%Y%m%d%H%M%S')}{os.urandom(2).hex()}"
        _spawn_catalogue(job_id, url=url,
                         ref=str(payload.get("ref") or "").strip() or None)
        return {"ok": True, "job_id": job_id}

    @api.post("/api/playground/packs/delete")
    def playground_pack_delete(payload: dict) -> dict[str, Any]:
        name = str(payload.get("name") or "")
        if not re.fullmatch(r"[\w.-]+", name) or name.startswith("."):
            return {"error": f"Not a pack name: {name!r}"}
        dest = PLAYGROUND_NODES / name
        if not dest.is_dir():
            _reload_volume()
        if not dest.is_dir():
            return {"error": f"No pack named {name!r} under "
                             f"{PLAYGROUND_NODES}."}
        shutil.rmtree(dest)
        volume.commit()
        # Same fact as install, stated on the delete: a warm engine keeps
        # what it already imported until its next restart.
        return {"ok": True,
                "note": "Removed from the shelf. A warm engine keeps what "
                        "it already imported until its next restart."}

    @api.get("/api/workflows")
    def list_workflows() -> dict[str, Any]:
        _reload_volume()
        rows = []
        if WORKFLOWS.is_dir():
            for f in sorted(WORKFLOWS.glob("*.json")):
                if f.name.endswith(".meta.json"):
                    continue
                meta = {}
                mp = f.with_name(f.stem + ".meta.json")
                if mp.is_file():
                    try:
                        meta = json.loads(mp.read_text())
                    except ValueError:
                        meta = {}
                rows.append({"name": f.stem,
                             "created": meta.get("created"),
                             "seed_of": meta.get("seed_of"),
                             "exposes": meta.get("exposes") or [],
                             "mtime": f.stat().st_mtime})
        return {"workflows": rows}

    @api.get("/api/workflows/{name}")
    def get_workflow(name: str) -> dict[str, Any]:
        try:
            name = _workflow_name(name)
        except ValueError as exc:
            return {"error": str(exc)}
        path = WORKFLOWS / f"{name}.json"
        if not path.is_file():
            _reload_volume()
        if not path.is_file():
            return {"error": f"No workflow named {name!r} under "
                             f"{WORKFLOWS}."}
        try:
            graph = json.loads(path.read_text())
        except ValueError as exc:
            return {"error": f"{name}.json is not valid JSON: {exc}"}
        meta = {}
        mp = WORKFLOWS / f"{name}.meta.json"
        if mp.is_file():
            try:
                meta = json.loads(mp.read_text())
            except ValueError:
                meta = {}
        return {"name": name, "graph": graph, "meta": meta}

    @api.post("/api/workflows/{name}")
    def save_workflow(name: str, payload: dict) -> dict[str, Any]:
        try:
            name = _workflow_name(name)
            graph = _validate_playground_graph(payload.get("graph"))
        except ValueError as exc:
            return {"error": str(exc)}
        WORKFLOWS.mkdir(parents=True, exist_ok=True)
        # The graph file stays pure API format — a stock ComfyUI could run
        # it — and our fields live beside it, the datasets split.
        (WORKFLOWS / f"{name}.json").write_text(json.dumps(graph, indent=2))
        meta = payload.get("meta") or {}
        (WORKFLOWS / f"{name}.meta.json").write_text(json.dumps(
            {"created": meta.get("created") or time.time(),
             "seed_of": meta.get("seed_of"),
             "exposes": meta.get("exposes") or []}, indent=2))
        volume.commit()
        return {"ok": True, "name": name}

    @api.post("/api/workflows/{name}/delete")
    def delete_workflow(name: str) -> dict[str, Any]:
        try:
            name = _workflow_name(name)
        except ValueError as exc:
            return {"error": str(exc)}
        path = WORKFLOWS / f"{name}.json"
        if not path.is_file():
            _reload_volume()
        if not path.is_file():
            return {"error": f"No workflow named {name!r}."}
        path.unlink()
        (WORKFLOWS / f"{name}.meta.json").unlink(missing_ok=True)
        volume.commit()
        return {"ok": True}

    # ── the storyboards ───────────────────────────────────────────────
    # A board travels whole, both ways. There is no per-panel route because
    # the page holds the sequence and order is the meaning — a panel saved on
    # its own would be a panel with no "and then". Several boards, one folder
    # each, because a scene you put down on Tuesday is picked up on Friday
    # beside the one you started on Thursday.
    def _storyboard_dir(name: str) -> Path:
        return STORYBOARD / _storyboard_name(name)

    def _read_board(path: Path) -> dict[str, Any] | None:
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        return raw if isinstance(raw, dict) else None

    @api.get("/api/storyboards")
    def list_storyboards() -> dict[str, Any]:
        if not STORYBOARD.is_dir():
            _reload_volume()
        rows: list[dict[str, Any]] = []
        for d in (sorted(STORYBOARD.iterdir()) if STORYBOARD.is_dir() else []):
            if not d.is_dir() or not STORYBOARD_NAME_RE.fullmatch(d.name):
                continue
            raw = _read_board(d / "board.json")
            if raw is None:
                continue
            panels = raw.get("panels") or []
            # The first picture is the board's face in the list — the same
            # pointer the panel holds, resolved by the page.
            cover = next((p.get("picture") for p in panels
                          if isinstance(p, dict) and p.get("picture")), None)
            rows.append({"name": d.name, "title": str(raw.get("title") or ""),
                         "panels": len(panels), "updated": raw.get("updated"),
                         "cover": cover})
        rows.sort(key=lambda r: -float(r["updated"] or 0))
        return {"boards": rows}

    @api.get("/api/storyboard/{name}")
    def get_storyboard(name: str) -> dict[str, Any]:
        try:
            path = _storyboard_dir(name) / "board.json"
        except ValueError as exc:
            return {"error": str(exc)}
        if not path.is_file():
            _reload_volume()
        if not path.is_file():
            return {"error": f"No storyboard called {name!r}."}
        raw = _read_board(path)
        if raw is None:
            return {"error": f"{path} is not valid JSON."}
        raw["name"] = name
        return {"board": raw}

    @api.post("/api/storyboard/{name}")
    def save_storyboard(name: str, payload: dict) -> dict[str, Any]:
        try:
            folder = _storyboard_dir(name)
            board = _validate_storyboard(payload.get("board"))
        except ValueError as exc:
            return {"error": str(exc)}
        board["updated"] = time.time()
        folder.mkdir(parents=True, exist_ok=True)
        # Staged then renamed, the weights rule: a reader arriving mid-write
        # sees the last whole document rather than half of this one.
        tmp = folder / "board.json.part"
        tmp.write_text(json.dumps(board, indent=2))
        tmp.replace(folder / "board.json")
        volume.commit()
        return {"ok": True, "updated": board["updated"]}

    @api.post("/api/storyboard/{name}/delete")
    def delete_storyboard(name: str) -> dict[str, Any]:
        """The folder, whole: the board and every picture uploaded into it.
        Renders it pointed at are in outputs/ and are not touched — the
        confirm on the page says exactly that."""
        try:
            folder = _storyboard_dir(name)
        except ValueError as exc:
            return {"error": str(exc)}
        if not folder.is_dir():
            _reload_volume()
        if not folder.is_dir():
            return {"error": f"No storyboard called {name!r}."}
        shutil.rmtree(folder)
        volume.commit()
        return {"ok": True}

    @api.post("/api/storyboard/{name}/upload")
    async def upload_storyboard(name: str, request: Request) -> JSONResponse:
        """
        Pictures dropped onto the board, into its own folder.

        Uprighted and capped on arrival, once, for the reason
        `STORYBOARD_MAX_SIDE` gives; re-encoded as PNG when the source is a
        format nothing downstream reads (avif, bmp). Names are minted here
        rather than kept, because two drops of `IMG_0001.jpg` from two
        cameras are two pictures.
        """
        from PIL import Image, ImageOps

        try:
            folder = _storyboard_dir(name)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, 400)
        folder.mkdir(parents=True, exist_ok=True)
        form = await request.form()
        out: list[dict[str, Any]] = []
        stamp = f"{int(time.time() * 1000):x}"
        for n, up in enumerate(form.getlist("files")):
            filename = getattr(up, "filename", None)
            if not filename:
                continue
            suffix = Path(filename).suffix.lower()
            if suffix not in IMAGE_EXTS:
                continue
            ext = suffix if suffix in (".png", ".jpg", ".jpeg", ".webp") else ".png"
            target = folder / f"{stamp}{n:02d}{ext}"
            tmp = target.with_name(target.name + ".part")
            with open(tmp, "wb") as fh:
                while chunk := await up.read(1024 * 1024):
                    fh.write(chunk)
            try:
                with Image.open(tmp) as im:
                    im = ImageOps.exif_transpose(im)
                    im.thumbnail((STORYBOARD_MAX_SIDE, STORYBOARD_MAX_SIDE))
                    if ext in (".jpg", ".jpeg") and im.mode not in ("RGB", "L"):
                        im = im.convert("RGB")
                    im.save(target)
                    w, h = im.size
            except Exception as exc:
                tmp.unlink(missing_ok=True)
                return JSONResponse(
                    {"error": f"{filename} is not a picture this can read: "
                              f"{type(exc).__name__}: {exc}"}, 400)
            tmp.unlink(missing_ok=True)
            out.append({"file": target.name, "width": w, "height": h})
        volume.commit()
        return JSONResponse({"files": out})

    @api.get("/api/storyboard/{name}/file/{file}")
    def storyboard_file(name: str, file: str):
        """One uploaded picture, off the spool like a render — see
        `output_file` for why the spool and not the mount."""
        try:
            folder = _storyboard_dir(name)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        if not STORYBOARD_FILE_RE.match(file):
            return JSONResponse({"error": "Invalid name."}, status_code=400)
        path = _spooled(f"storyboard/{folder.name}/{file}")
        if path is None:
            mounted = folder / file
            if not mounted.is_file():
                _reload_volume()
            if not mounted.is_file():
                return JSONResponse({"error": "Not found."}, status_code=404)
            path = mounted
        return FileResponse(
            str(path),
            media_type=MEDIA_TYPES.get(path.suffix.lower(), "image/png"),
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @api.get("/api/status/{job_id}")
    def status(job_id: str) -> dict[str, Any]:
        """
        There was a "warm reload" here — the poll that first saw "completed"
        pulled the volume forward so the canvas's first /api/file would not
        miss, insisting through refusals with up to half a second of sleep on
        the hot poll path. Retired with the mount, not just moved: the pixels
        are served off the spool from committed state now, so the first ask
        for a fresh run cannot miss, and nothing in the serving path needs a
        reload to happen at all.
        """
        try:
            return jobs.get(job_id) or {"status": "unknown"}
        except Exception as exc:
            return {"status": "unknown", "error": str(exc)}

    @api.post("/api/stop/{job_id}")
    def stop(job_id: str) -> dict[str, Any]:
        _request_stop(job_id)
        return {"ok": True}

    return api
