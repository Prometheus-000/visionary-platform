"""
Visionary — Krea 2 LoRA trainer on Modal.

Replaces the manual `modal shell` routine: no deadsnakes PPA, no `accelerate
config`, no hand-editing dataset.toml in nano, no huggingface-cli by hand.
Drop images in the browser, press train, get a LoRA in the folder Forge reads.

    Volume  forge-webui-repo  ->  /workspace         (sd-webui-forge-classic layout)
    DiT                           /workspace/models/Stable-diffusion/raw.safetensors
    VAE                           /workspace/models/VAE/qwen_image_vae.safetensors
    Text encoder                  /workspace/models/text_encoder/qwen3vl_4b_bf16.safetensors
    Output                        /workspace/models/Lora/{lora_name}/

Deploy:
    modal deploy backend/main.py
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import signal
import subprocess
import time
import zipfile
from collections import deque
from pathlib import Path
from typing import Any

import modal
from fastapi import Request
from fastapi.responses import JSONResponse

APP_NAME = "visionary-lora-trainer"

# Single module-level handle: volume.commit() must be called on the same object
# Modal mounted, or the flush is a no-op.
volume = modal.Volume.from_name("forge-webui-repo")

# One secret carries both keys, so setup is a single command:
#   modal secret create my-hf-secret HF_TOKEN=hf_xxx VISIONARY_API_KEY=pick-something
hf_secret = modal.Secret.from_name("my-hf-secret")

# Live job state, keyed by job_id. This is the side channel between the running
# GPU container and the status endpoint — FunctionCall.get() only resolves when
# the call *finishes*, so it can never report progress mid-run. The Dict is
# written by train_job and read by api_train_status; the stop flag travels the
# other way.
jobs = modal.Dict.from_name("visionary-jobs", create_if_missing=True)

WORKSPACE = Path("/workspace")
MODELS = WORKSPACE / "models"
DATASETS = WORKSPACE / "datasets"
UPLOADS = WORKSPACE / "uploads"
LORA_OUT = MODELS / "Lora"
MUSUBI = Path("/opt/musubi-tuner")

# Measured, not guessed: a real run peaked at 29.26 GiB of 40 GB at 1024px,
# batch 1, rank 32 — ~11 GB headroom despite raw.safetensors being 26.28 GB on
# disk. Gradient checkpointing + adamw8bit reclaim more than expected, so this
# fits comfortably and needs neither fp8 nor block swap.
#
# Alternatives: "L40S" is 48 GB at $1.95/hr (cheaper per hour than A100-40GB's
# $2.10, but ~864 GB/s vs ~1555 GB/s memory bandwidth, so likely slower per
# step — cheaper per epoch only below ~2.9 s/it). "A100-80GB" ($2.50/hr) buys
# capacity for 1280px or larger batches. "L4"/"A10" cannot hold this model.
GPU = "A100-40GB"

app = modal.App(APP_NAME)


# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------

# Featherweight: this one is on your critical path, it must cold-start fast.
web_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]==0.115.6",
    "python-multipart==0.0.20",
    # Thumbnails for the caption review grid are generated here rather than in
    # the GPU container — reviewing a dataset must not require booting an A100.
    "pillow==11.1.0",
)

# Heavy: runs detached, nobody watches it boot.
#
# torch goes in FIRST from the cu124 index so musubi's `pip install -e .` sees a
# satisfied requirement and does not drag in a default-CUDA wheel over it.
# Everything else (accelerate==1.6.0, transformers==4.57.6, bitsandbytes for
# adamw8bit, voluptuous, toml) comes from musubi's own pyproject — deliberately
# NOT re-pinned here, because pinning them separately fights those versions.
trainer_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install("git", "unzip", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/kohya-ss/musubi-tuner /opt/musubi-tuner",
        "cd /opt/musubi-tuner && pip install -e .",
    )
    # hf_transfer only (not huggingface_hub) so musubi's hub pin stays intact.
    # fastapi rides along because Modal imports this whole module in every
    # container, including this one, and the upload endpoint's signature
    # references fastapi.Request at module scope.
    .pip_install("hf_transfer==0.1.9", "fastapi==0.115.6")
    # Bake the Qwen3-VL tokenizer into the image.
    #
    # krea2_encoder.py loads the text encoder weights from the safetensors path
    # you pass, but still fetches the *tokenizer* by repo id at runtime
    # (`Qwen/Qwen3-VL-4B-Instruct`). That is a silent network dependency in the
    # middle of a paid GPU run. Baking it makes the caching step hermetic.
    #
    # Deliberately NOT under a volume-backed HF_HOME: /workspace does not exist
    # at build time, and anything written there would be shadowed by the volume
    # mount at runtime. This lands in the image's own cache instead.
    .run_commands(
        "python -c \"from transformers import AutoTokenizer, Qwen2TokenizerFast; "
        "r='Qwen/Qwen3-VL-4B-Instruct'; "
        "AutoTokenizer.from_pretrained(r); Qwen2TokenizerFast.from_pretrained(r)\""
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1"})
)


# --------------------------------------------------------------------------
# Weight manifest — verified against the HF API, not copied from the docs.
#
#  * docs/krea2.md puts the VAE in `Comfy-Org/Qwen-Image-Edit_ComfyUI`. That
#    path 404s; the file is in `Comfy-Org/Qwen-Image_ComfyUI` (no "-Edit").
#  * `krea/Krea-2-Raw` is gated (anonymous GET -> 401), hence the mandatory token.
# --------------------------------------------------------------------------

WEIGHTS: list[dict[str, Any]] = [
    {
        "label": "DiT (Krea 2 RAW)",
        "repo_id": "krea/Krea-2-Raw",
        "filename": "raw.safetensors",
        "dest": MODELS / "Stable-diffusion" / "raw.safetensors",
        "gated": True,
    },
    {
        "label": "VAE (Qwen-Image)",
        "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
        "filename": "split_files/vae/qwen_image_vae.safetensors",
        "dest": MODELS / "VAE" / "qwen_image_vae.safetensors",
        "gated": False,
    },
    {
        # bf16, NOT fp8_scaled — this is load-bearing.
        #
        # krea2_encoder._load_qwen3_vl_model builds a plain
        # Qwen3VLForConditionalGeneration, runs the ComfyUI state dict through a
        # pure key-rename (model. -> model.language_model.), then errors on any
        # unexpected key. ComfyUI's fp8_scaled checkpoint carries ~504 extra
        # `weight_scale` / `comfy_quant` tensors that the plain model has no
        # parameters for, so it dies with:
        #   "checkpoint did not match the model: missing=[], unexpected=[...]"
        # Note missing=[] — every needed weight was present; only the quant
        # metadata tripped the strict check. There is no fp8 path for the text
        # encoder (krea2_utils' fp8 machinery targets the DiT's "blocks." keys).
        #
        # Verified: bf16 = 713 tensors, zero weight_scale/comfy_quant keys.
        # Costs ~8.9 GB vs ~5.2 GB, but the TE is loaded only during caching and
        # freed before training, so it never competes for VRAM with the DiT.
        "label": "Text encoder (Qwen3-VL 4B bf16)",
        "repo_id": "Comfy-Org/Qwen3-VL",
        "filename": "text_encoders/qwen3vl_4b_bf16.safetensors",
        "dest": MODELS / "text_encoder" / "qwen3vl_4b_bf16.safetensors",
        "gated": False,
    },
]

DIT_PATH = WEIGHTS[0]["dest"]
VAE_PATH = WEIGHTS[1]["dest"]
TEXT_ENCODER_PATH = WEIGHTS[2]["dest"]

# Inference-only, so it is NOT in WEIGHTS (training must not pay 26 GB for a
# model it never loads). Fetched on demand the first time you generate.
# Turbo is the recommended sampler: 8 steps at guidance 1.0 with --mu 1.15,
# against RAW's 28 steps at 5.5 — same size on disk, ~3.5x fewer steps.
TURBO_WEIGHT: dict[str, Any] = {
    "label": "DiT (Krea 2 Turbo)",
    "repo_id": "krea/Krea-2-Turbo",
    "filename": "turbo.safetensors",
    "dest": MODELS / "Stable-diffusion" / "turbo.safetensors",
    "gated": True,  # verified: anonymous GET returns 401, same as RAW
}
TURBO_PATH = TURBO_WEIGHT["dest"]

# Generated images land here; Forge picks them up alongside its own outputs.
OUTPUTS = WORKSPACE / "outputs"

STAGING_DIR = WORKSPACE / ".cache" / "hf-staging"

CAPTION_MODEL = "fancyfeast/llama-joycaption-beta-one-hf-llava"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif"}
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_LOG_LINES = 400


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------


class AuthError(Exception):
    pass


def _check_key(supplied: str | None) -> None:
    """
    Shared-secret gate on every public endpoint.

    Fail-closed on purpose: these URLs are world-reachable and each call can
    start an A100. An open endpoint is somebody else's free GPU.
    """
    expected = (os.environ.get("VISIONARY_API_KEY") or "").strip()
    if not expected:
        raise AuthError(
            "VISIONARY_API_KEY is not set on the backend. Run:  modal secret create "
            "my-hf-secret HF_TOKEN=hf_xxx VISIONARY_API_KEY=pick-something  "
            "(use --force to overwrite an existing secret)"
        )
    if (supplied or "").strip() != expected:
        raise AuthError("Bad or missing access key.")


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _resolve_hf_token(request_token: str | None) -> str | None:
    """Modal Secret wins; the UI field is the fallback. Never logged, never stored."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token and token.strip():
        return token.strip()
    return (request_token or "").strip() or None


class StoppedByUser(Exception):
    """Raised when the user pressed Stop; not a failure."""


# musubi builds its bar as tqdm(range(max_train_steps), desc="steps") and adds
# loss via set_postfix, so lines look like:
#   steps:  12%|#1        | 29/240 [01:17<09:23,  2.68s/it, avg_loss=0.0731]
# tqdm redraws with \r, and text=True enables universal newlines (\r counts as a
# line terminator), so each redraw arrives here as its own line.
TQDM_RE = re.compile(
    r"(?P<pct>\d+)%\|.*?\|\s*(?P<cur>\d+)/(?P<total>\d+)\s*"
    # rate and eta are literal "?" on the very first redraw, before tqdm has an
    # estimate. Matching that case lets the UI show the total step count (and so
    # the true scale of the run) immediately instead of after the first step.
    r"\[(?P<elapsed>[\d:]+)<(?P<eta>[\d:?]+),\s*(?P<rate>[\d.]+|\?)(?P<unit>s/it|it/s)"
)
LOSS_RE = re.compile(r"avg_loss=(?P<loss>[\d.eE+-]+)")
EPOCH_RE = re.compile(r"epoch\s+(?P<cur>\d+)\s*/\s*(?P<total>\d+)", re.I)

# How often to check the stop flag. Each check is a network round-trip to the
# Dict, so this trades stop latency against overhead.
STOP_POLL_SECONDS = 2.0


def _publish(job_id: str, **fields: Any) -> None:
    """Merge fields into the job's live state. Never fatal — telemetry only."""
    try:
        state = jobs.get(job_id) or {}
        state.update(fields)
        state["updated_at"] = time.time()
        jobs[job_id] = state
    except Exception as exc:  # a telemetry blip must not kill a paid GPU run
        print(f"[progress] publish failed: {exc}")


def _stop_requested(job_id: str) -> bool:
    try:
        return bool((jobs.get(job_id) or {}).get("stop"))
    except Exception:
        return False


def _run(
    cmd: list[str],
    label: str,
    log: deque[str],
    job_id: str,
    cwd: Path = MUSUBI,
    parse_progress: bool = False,
) -> None:
    """
    Run a subprocess, streaming output to Modal logs and keeping a tail for the UI.

    When `parse_progress` is set, tqdm lines are decoded into step/percent/ETA
    and published for the frontend. The stop flag is polled between lines; on
    stop the child gets SIGTERM, then SIGKILL if it does not go quietly.

    Raises on non-zero exit with the last lines attached, so a failure surfaces
    the actual traceback instead of a bare exit code.
    """
    banner = f"$ {' '.join(cmd)}"
    print(f"\n===== {label} =====\n{banner}", flush=True)
    log.append(f"[{label}]")
    _publish(job_id, phase=label)

    env = {**os.environ, "PYTHONPATH": str(MUSUBI / "src")}
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        # Own process group, so a stop reaches accelerate's children too —
        # killing only the launcher would orphan the actual trainer.
        start_new_session=True,
    )
    assert proc.stdout is not None

    last_stop_check = 0.0
    last_publish = 0.0
    stopping = False

    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        print(line, flush=True)
        log.append(line)

        now = time.monotonic()

        if parse_progress:
            m = TQDM_RE.search(line)
            if m and now - last_publish > 1.0:
                last_publish = now
                fields: dict[str, Any] = {
                    "phase": label,
                    "percent": int(m.group("pct")),
                    "step": int(m.group("cur")),
                    "total_steps": int(m.group("total")),
                    "elapsed": m.group("elapsed"),
                    "eta": m.group("eta"),
                    "rate": f"{m.group('rate')}{m.group('unit')}",
                }
                if (lm := LOSS_RE.search(line)) :
                    fields["loss"] = lm.group("loss")
                if (em := EPOCH_RE.search(line)) :
                    fields["epoch"] = int(em.group("cur"))
                    fields["total_epochs"] = int(em.group("total"))
                _publish(job_id, **fields)

        if not stopping and now - last_stop_check > STOP_POLL_SECONDS:
            last_stop_check = now
            if _stop_requested(job_id):
                stopping = True
                print(f"[{job_id}] stop requested — terminating {label}", flush=True)
                _publish(job_id, phase="stopping")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass

    code = proc.wait()

    if stopping:
        try:  # make sure nothing survived the SIGTERM
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        raise StoppedByUser(label)

    if code != 0:
        tail = "\n".join(list(log)[-25:])
        raise RuntimeError(f"{label} failed (exit {code}).\n{tail}")


def _safe_extract_zip(zip_path: Path, dest: Path) -> int:
    """
    Extract image files from a zip, flattened.

    Guards against zip-slip (`../` members, absolute paths) by rebuilding each
    name from its basename rather than trusting the archive. Also skips macOS
    `__MACOSX/` resource forks and dotfiles, which otherwise land as junk
    "images" and poison the dataset.
    """
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            name = Path(member.filename).name  # drops any directory component
            if not name or name.startswith("."):
                continue
            if "__MACOSX" in member.filename:
                continue
            suffix = Path(name).suffix.lower()
            if suffix not in IMAGE_EXTS and suffix != ".txt":
                continue
            target = dest / name
            with zf.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            if suffix in IMAGE_EXTS:
                count += 1
    return count


def _ensure_weight(spec: dict[str, Any], token: str | None) -> dict[str, Any]:
    """Download one weight to its exact destination if not already present."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import (
        EntryNotFoundError,
        GatedRepoError,
        RepositoryNotFoundError,
    )

    dest: Path = spec["dest"]
    label = spec["label"]

    if dest.exists() and dest.stat().st_size > 0:
        print(f"[weights] {label}: present ({dest.stat().st_size / 1e9:.2f} GB)")
        return {"label": label, "action": "cached", "path": str(dest)}

    if spec["gated"] and not token:
        raise RuntimeError(
            f"{label} is in the gated repo '{spec['repo_id']}' and no HF token was "
            f"found. Accept the licence at https://huggingface.co/{spec['repo_id']} "
            "and set HF_TOKEN in the my-hf-secret Modal secret."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[weights] {label}: downloading {spec['repo_id']}/{spec['filename']}")
    started = time.monotonic()

    try:
        staged = hf_hub_download(
            repo_id=spec["repo_id"],
            filename=spec["filename"],
            local_dir=str(STAGING_DIR),
            token=token,
        )
    except GatedRepoError as exc:
        raise RuntimeError(
            f"{label}: '{spec['repo_id']}' is gated and this token was rejected. "
            "Accept the licence with the same account that issued the token."
        ) from exc
    except RepositoryNotFoundError as exc:
        raise RuntimeError(f"{label}: repo '{spec['repo_id']}' not found.") from exc
    except EntryNotFoundError as exc:
        raise RuntimeError(
            f"{label}: '{spec['filename']}' missing from '{spec['repo_id']}'."
        ) from exc

    shutil.move(staged, dest)
    print(
        f"[weights] {label}: {dest.stat().st_size / 1e9:.2f} GB "
        f"in {time.monotonic() - started:.0f}s"
    )
    # Commit per asset: a mid-run timeout then costs time, not the download.
    volume.commit()
    return {"label": label, "action": "downloaded", "path": str(dest)}


def _write_dataset_toml(
    toml_path: Path,
    image_dir: Path,
    cache_dir: Path,
    resolution: int,
    batch_size: int,
    num_repeats: int,
) -> str:
    """
    Generate dataset.toml.

    The key names here are `image_directory` and `cache_directory`. Musubi
    validates this file with a voluptuous Schema that does NOT set ALLOW_EXTRA,
    so the shorter `image_dir` / `cache_dir` spellings raise MultipleInvalid
    rather than being ignored. This is the single most common way this config
    fails silently-looking.
    """
    content = f"""[general]
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
    toml_path.write_text(content)
    print(f"[dataset] wrote {toml_path}\n{content}")
    return content


# JoyCaption responds to named styles rather than freeform instructions; these
# are the three that matter for LoRA datasets. "descriptive" is the default
# because natural-language captions are what Krea 2's Qwen3-VL encoder expects —
# booru tags train a different kind of conditioning.
CAPTION_STYLES: dict[str, str] = {
    "descriptive": (
        "Write a descriptive caption for this image in a formal tone. Describe "
        "the subject, its appearance, the composition, lighting, setting, and "
        "artistic style. Do not speculate about things you cannot see."
    ),
    "descriptive_casual": (
        "Write a descriptive caption for this image in a casual, natural tone. "
        "Describe what is happening, how it looks, and the overall mood."
    ),
    "tags": (
        "Write a comma-separated list of booru-style tags for this image, "
        "covering subject, clothing, pose, setting, lighting and style. "
        "Output only the tags."
    ),
}
CAPTION_LENGTHS: dict[str, str] = {
    "short": " Keep it to one concise sentence.",
    "medium": " Keep it to two or three sentences.",
    "long": " Be thorough and detailed, four or more sentences.",
}

# Never write a caption longer than the text encoder will read. Qwen3-VL's
# conditioner truncates, so anything past this is silently discarded.
MAX_CAPTION_CHARS = 1024


def _caption_images(
    image_dir: Path,
    trigger_word: str,
    log: deque[str],
    job_id: str,
    style: str = "descriptive",
    length: str = "medium",
    overwrite: bool = False,
) -> int:
    """
    Caption images with JoyCaption, writing a .txt sidecar next to each.

    JoyCaption is purpose-built for training captions — it is what most of the
    hosted trainers use — so its output drops straight into a LoRA dataset.

    `overwrite=False` skips images that already have a caption, which is what
    makes re-running safe after you have hand-edited some of them.
    """
    import torch
    from PIL import Image
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    every = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    todo = [
        p
        for p in every
        if overwrite or not p.with_suffix(".txt").exists()
        or not p.with_suffix(".txt").read_text().strip()
    ]
    if not todo:
        print("[caption] every image already has a caption — nothing to do")
        return 0

    print(f"[caption] loading {CAPTION_MODEL} for {len(todo)}/{len(every)} images")
    log.append(f"[caption] {len(todo)} images")
    _publish(job_id, phase="caption", step=0, total_steps=len(todo), percent=0)

    # Explicit cache_dir on the volume rather than a global HF_HOME: JoyCaption
    # is ~17 GB and should survive container restarts, while the small baked-in
    # Qwen tokenizer keeps resolving from the image.
    cache_dir = str(WORKSPACE / ".cache" / "huggingface")
    processor = AutoProcessor.from_pretrained(CAPTION_MODEL, cache_dir=cache_dir)
    model = LlavaForConditionalGeneration.from_pretrained(
        CAPTION_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        cache_dir=cache_dir,
    )
    model.eval()

    instruction = CAPTION_STYLES.get(style, CAPTION_STYLES["descriptive"])
    instruction += CAPTION_LENGTHS.get(length, CAPTION_LENGTHS["medium"])

    written = 0
    for i, img_path in enumerate(todo, 1):
        if _stop_requested(job_id):
            print("[caption] stop requested — leaving remaining images uncaptioned")
            break
        try:
            image = Image.open(img_path).convert("RGB")
            convo = [
                {"role": "system", "content": "You are a helpful image captioner."},
                {"role": "user", "content": f"<image>\n{instruction}"},
            ]
            convo_text = processor.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[convo_text], images=[image], return_tensors="pt"
            ).to("cuda:0")
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=320,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    suppress_tokens=None,
                )[0][inputs["input_ids"].shape[1]:]
            caption = processor.tokenizer.decode(out, skip_special_tokens=True).strip()

            # JoyCaption sometimes opens with a stock preamble; strip it so the
            # trigger word stays at the very front, where it does its work.
            for junk in ("This image shows ", "The image shows ", "This image depicts "):
                if caption.startswith(junk):
                    caption = caption[len(junk):]
                    caption = caption[0].upper() + caption[1:] if caption else caption
                    break

            final = f"{trigger_word}, {caption}" if trigger_word else caption
            img_path.with_suffix(".txt").write_text(final[:MAX_CAPTION_CHARS])
            written += 1
        except Exception as exc:  # one bad image must not kill the whole run
            print(f"[caption] {img_path.name} failed: {exc}")
            if trigger_word and not img_path.with_suffix(".txt").exists():
                img_path.with_suffix(".txt").write_text(trigger_word)

        pct = round(i / len(todo) * 100)
        _publish(job_id, phase="caption", step=i, total_steps=len(todo), percent=pct)
        if i % 5 == 0 or i == len(todo):
            print(f"[caption] {i}/{len(todo)}", flush=True)

    # Free the captioner before training claims the VRAM.
    del model
    torch.cuda.empty_cache()
    volume.commit()
    return written


# --------------------------------------------------------------------------
# Upload — browser posts here DIRECTLY.
#
# Not through the Next.js route: Vercel functions cap request bodies at 4.5 MB
# and return 413 FUNCTION_PAYLOAD_TOO_LARGE. Modal web endpoints accept 4 GiB,
# which is the whole reason the dataset goes straight here.
# --------------------------------------------------------------------------


@app.function(
    image=web_image,
    timeout=900,
    volumes={"/workspace": volume},
    secrets=[hf_secret],
)
@modal.fastapi_endpoint(method="POST", label="upload-dataset")
async def api_upload_dataset(request: Request) -> JSONResponse:
    """
    Accept a zip or loose images and stage them on the volume.

    Takes the raw Starlette Request so the multipart stream is written straight
    to disk instead of being buffered whole in memory — a 4 GiB dataset must not
    have to fit in RAM.

    POST multipart/form-data, header `X-Visionary-Key`
        files: one .zip, or many images (+ optional matching .txt captions)
      -> { "status": "uploaded", "job_id": ..., "image_count": N }
    """
    try:
        _check_key(request.headers.get("x-visionary-key"))
    except AuthError as exc:
        return JSONResponse({"status": "error", "error": str(exc)}, status_code=401)

    job_id = f"ds_{time.strftime('%Y%m%d_%H%M%S')}_{os.urandom(3).hex()}"
    raw_dir = UPLOADS / job_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    image_count = 0
    zips: list[Path] = []

    form = await request.form()
    uploads = form.getlist("files")
    if not uploads:
        return JSONResponse(
            {"status": "error", "error": "No files in the request."}, status_code=400
        )

    for upload in uploads:
        filename = getattr(upload, "filename", None)
        if not filename:
            continue
        name = Path(filename).name  # strip any path the browser sent
        suffix = Path(name).suffix.lower()
        if suffix not in IMAGE_EXTS and suffix not in {".zip", ".txt"}:
            continue

        target = raw_dir / name
        with open(target, "wb") as out:
            while chunk := await upload.read(1024 * 1024):
                out.write(chunk)

        if suffix == ".zip":
            zips.append(target)
        elif suffix in IMAGE_EXTS:
            image_count += 1

    for zip_path in zips:
        try:
            image_count += _safe_extract_zip(zip_path, raw_dir)
        except zipfile.BadZipFile:
            return JSONResponse(
                {"status": "error", "error": f"{zip_path.name} is not a valid zip."},
                status_code=400,
            )
        finally:
            zip_path.unlink(missing_ok=True)

    if image_count == 0:
        shutil.rmtree(raw_dir, ignore_errors=True)
        return JSONResponse(
            {"status": "error", "error": "No images found in the upload."},
            status_code=400,
        )

    volume.commit()
    return JSONResponse(
        {"status": "uploaded", "job_id": job_id, "image_count": image_count}
    )


# --------------------------------------------------------------------------
# Dataset review — thumbnails + captions
#
# Runs on the CPU web image, not the GPU one: opening the review grid must never
# boot an A100. Thumbnails are cached on the volume so re-opening is instant.
# --------------------------------------------------------------------------

THUMB_PX = 320
THUMB_DIR_NAME = ".thumbs"


def _thumb_data_uri(img_path: Path, thumb_dir: Path) -> str | None:
    """Return a small JPEG data URI, generating and caching it on first use."""
    from PIL import Image

    cached = thumb_dir / (img_path.stem + ".jpg")
    try:
        if not cached.exists() or cached.stat().st_mtime < img_path.stat().st_mtime:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                im.thumbnail((THUMB_PX, THUMB_PX), Image.LANCZOS)
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=78, optimize=True)
                cached.write_bytes(buf.getvalue())
        return "data:image/jpeg;base64," + base64.b64encode(cached.read_bytes()).decode()
    except Exception as exc:
        print(f"[thumb] {img_path.name}: {exc}")
        return None


@app.function(
    image=web_image,
    timeout=300,
    volumes={"/workspace": volume},
    secrets=[hf_secret],
)
@modal.fastapi_endpoint(method="POST", label="dataset")
def api_dataset(payload: dict) -> dict[str, Any]:
    """
    List a staged dataset: thumbnail + current caption for every image.

    Reads the upload directory rather than the training copy, because captions
    are reviewed *before* a run exists. train_job copies .txt sidecars along with
    the images, so edits made here are what training sees.

    POST { "job_id": str, "access_key": str }
    """
    try:
        _check_key(payload.get("access_key"))
    except AuthError as exc:
        return {"status": "error", "error": str(exc)}

    job_id = str(payload.get("job_id") or "").strip()
    if not NAME_RE.match(job_id):
        return {"status": "error", "error": "Invalid job_id."}

    volume.reload()
    src = UPLOADS / job_id
    if not src.is_dir():
        return {"status": "error", "error": f"No dataset staged for {job_id}."}

    thumb_dir = src / THUMB_DIR_NAME
    thumb_dir.mkdir(exist_ok=True)

    items = []
    for img in sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        txt = img.with_suffix(".txt")
        items.append(
            {
                "name": img.name,
                "caption": txt.read_text().strip() if txt.exists() else "",
                "thumb": _thumb_data_uri(img, thumb_dir),
            }
        )

    volume.commit()  # persist newly generated thumbnails
    return {
        "status": "ok",
        "job_id": job_id,
        "count": len(items),
        "captioned": sum(1 for i in items if i["caption"]),
        "images": items,
    }


@app.function(
    image=web_image,
    timeout=120,
    volumes={"/workspace": volume},
    secrets=[hf_secret],
)
@modal.fastapi_endpoint(method="POST", label="save-captions")
def api_save_captions(payload: dict) -> dict[str, Any]:
    """
    Persist hand-edited captions.

    POST { "job_id": str, "captions": {filename: caption}, "access_key": str }
    """
    try:
        _check_key(payload.get("access_key"))
    except AuthError as exc:
        return {"status": "error", "error": str(exc)}

    job_id = str(payload.get("job_id") or "").strip()
    captions = payload.get("captions")
    if not NAME_RE.match(job_id):
        return {"status": "error", "error": "Invalid job_id."}
    if not isinstance(captions, dict):
        return {"status": "error", "error": "captions must be an object."}

    volume.reload()
    src = UPLOADS / job_id
    if not src.is_dir():
        return {"status": "error", "error": f"No dataset staged for {job_id}."}

    written = 0
    for raw_name, caption in captions.items():
        # Basename only — a filename from the client must not escape the dir.
        name = Path(str(raw_name)).name
        img = src / name
        if not img.exists() or img.suffix.lower() not in IMAGE_EXTS:
            continue
        img.with_suffix(".txt").write_text(str(caption).strip()[:MAX_CAPTION_CHARS])
        written += 1

    volume.commit()
    return {"status": "ok", "saved": written}


@app.function(
    image=trainer_image,
    gpu=GPU,
    cpu=2.0,
    timeout=2 * 60 * 60,
    secrets=[hf_secret],
    volumes={"/workspace": volume},
)
def caption_job(
    job_id: str,
    trigger_word: str = "",
    style: str = "descriptive",
    length: str = "medium",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Caption a staged dataset as its own job, so it can be reviewed before training."""
    if not NAME_RE.match(job_id):
        raise ValueError("Invalid job_id.")

    started = time.time()
    log: deque[str] = deque(maxlen=MAX_LOG_LINES)
    jobs[job_id] = {
        "status": "running",
        "phase": "caption",
        "stop": False,
        "started_at": started,
    }

    volume.reload()
    src = UPLOADS / job_id
    if not src.is_dir():
        raise RuntimeError(f"No dataset staged for {job_id}.")

    written = _caption_images(
        src, trigger_word.strip(), log, job_id, style, length, overwrite
    )

    result = {
        "status": "completed",
        "job_id": job_id,
        "captioned": written,
        "duration_s": round(time.time() - started, 1),
    }
    _publish(job_id, **result)
    print(f"[{job_id}] captioned {written} images in {result['duration_s']}s")
    return result


@app.function(image=web_image, timeout=30, secrets=[hf_secret])
@modal.fastapi_endpoint(method="POST", label="caption")
def api_caption(payload: dict) -> dict[str, Any]:
    """
    Queue captioning for a staged dataset. Returns immediately, like training.

    POST { "job_id", "trigger_word", "style", "length", "overwrite", "access_key" }
    """
    try:
        _check_key(payload.get("access_key"))
    except AuthError as exc:
        return {"status": "error", "error": str(exc)}

    job_id = str(payload.get("job_id") or "").strip()
    if not NAME_RE.match(job_id):
        return {"status": "error", "error": "Invalid job_id."}

    style = str(payload.get("style") or "descriptive")
    if style not in CAPTION_STYLES:
        return {"status": "error", "error": f"Unknown style: {style}"}
    length = str(payload.get("length") or "medium")
    if length not in CAPTION_LENGTHS:
        return {"status": "error", "error": f"Unknown length: {length}"}

    call = caption_job.spawn(
        job_id=job_id,
        trigger_word=str(payload.get("trigger_word") or ""),
        style=style,
        length=length,
        overwrite=bool(payload.get("overwrite")),
    )
    try:
        jobs[job_id] = {"status": "queued", "phase": "caption", "stop": False}
    except Exception:
        pass

    return {"status": "queued", "job_id": job_id, "call_id": call.object_id}


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


@app.function(
    image=trainer_image,
    gpu=GPU,
    # Observed on the first real run: CPU sat at 1.04 cores against a 1.00-core
    # reservation while GPU utilization capped at ~82%. Two dataloader workers
    # plus a torch thread do not fit on one core, so the GPU was being starved
    # between steps. CPU is trivially cheap next to an A100 — give it room.
    cpu=4.0,
    timeout=6 * 60 * 60,  # 30 epochs on a real dataset is hours, not minutes
    secrets=[hf_secret],
    volumes={"/workspace": volume},
)
def train_job(
    job_id: str,
    lora_name: str,
    trigger_word: str,
    auto_caption: bool = False,
    hf_token: str | None = None,
    # --- advanced (all optional; defaults match the krea2 doc) ---
    resolution: int = 1024,
    batch_size: int = 1,
    num_repeats: int = 1,
    network_dim: int = 32,
    network_alpha: int = 32,
    learning_rate: float = 1e-4,
    max_train_epochs: int = 30,
    save_every_n_epochs: int = 1,
    # 0 = off. Set it to bound how much work a Stop discards.
    save_every_n_steps: int = 0,
    # Ring buffer for intermediate checkpoints. 30 epochs at save_every_n_epochs=1
    # otherwise leaves 30 files of ~0.5 GB each. Pruning only ever touches the
    # numbered epoch/step checkpoints — the final model is written separately as
    # plain "{lora_name}.safetensors", so it is never at risk. 0 = keep all.
    save_last_n_epochs: int = 3,
    save_last_n_steps: int = 0,
    discrete_flow_shift: float = 2.5,
    optimizer_type: str = "adamw8bit",
    seed: int = 42,
    # --- VRAM escape hatches, off by default ---
    # fp8 quantizes the 28 main blocks at load time, roughly halving DiT memory.
    # Off by default because it perturbs numerics; turn it on if training OOMs.
    # Musubi rejects --fp8_base without --fp8_scaled (plain fp8 would cast the
    # norms and break the model), so these are passed as a pair, never alone.
    fp8: bool = False,
    # Offload N transformer blocks to CPU. Slower per step, but it is the lever
    # that lets a too-small card finish at all. 0 disables.
    blocks_to_swap: int = 0,
) -> dict[str, Any]:
    """
    The whole pipeline, in one detached call:
    weights -> captions -> dataset.toml -> cache latents -> cache TE -> train.
    """
    if not NAME_RE.match(job_id) or not NAME_RE.match(lora_name):
        raise ValueError("job_id and lora_name must be 1-64 chars of [A-Za-z0-9_-].")

    started = time.time()
    log: deque[str] = deque(maxlen=MAX_LOG_LINES)

    # Fresh state, and critically a cleared stop flag — otherwise a stop left
    # over from a previous job with this id would kill the new one instantly.
    jobs[job_id] = {
        "status": "running",
        "phase": "starting",
        "lora_name": lora_name,
        "stop": False,
        "started_at": started,
    }

    volume.reload()

    src_dir = UPLOADS / job_id
    if not src_dir.is_dir():
        raise RuntimeError(f"Upload {job_id} not found — upload the dataset first.")

    # --- 1. weights ------------------------------------------------------
    token = _resolve_hf_token(hf_token)
    weights_report = [_ensure_weight(spec, token) for spec in WEIGHTS]

    # --- 2. dataset layout ----------------------------------------------
    work = DATASETS / job_id
    image_dir = work / "images"
    cache_dir = work / "cache"
    image_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for item in src_dir.iterdir():
        if item.suffix.lower() in IMAGE_EXTS or item.suffix.lower() == ".txt":
            shutil.copy2(item, image_dir / item.name)

    images = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        raise RuntimeError("No images to train on.")
    print(f"[dataset] {len(images)} images in {image_dir}")

    # --- 3. captions -----------------------------------------------------
    if auto_caption:
        _caption_images(image_dir, trigger_word, log, job_id)
    else:
        # Krea 2 trains on caption files. Any image without one gets the trigger
        # word alone — a bare trigger is a valid minimal caption, a missing .txt
        # is a hard error inside the cache step.
        for img in images:
            txt = img.with_suffix(".txt")
            if not txt.exists():
                txt.write_text(trigger_word)

    toml_path = work / "dataset.toml"
    _write_dataset_toml(
        toml_path, image_dir, cache_dir, resolution, batch_size, num_repeats
    )
    volume.commit()

    # --- 4. pre-caching --------------------------------------------------
    _run(
        [
            "python", "src/musubi_tuner/krea2_cache_latents.py",
            "--dataset_config", str(toml_path),
            "--vae", str(VAE_PATH),
        ],
        "cache latents",
        log,
        job_id,
    )
    _run(
        [
            "python", "src/musubi_tuner/krea2_cache_text_encoder_outputs.py",
            "--dataset_config", str(toml_path),
            "--text_encoder", str(TEXT_ENCODER_PATH),
            "--batch_size", "1",
        ],
        "cache text encoder outputs",
        log,
        job_id,
    )
    volume.commit()

    # --- 5. train --------------------------------------------------------
    # Grouped in its own subdirectory: with save_every_n_epochs=1 a 30-epoch run
    # emits 30 checkpoints, and dumping those loose into models/Lora would bury
    # your real LoRAs. Forge scans the Lora dir recursively, so it still finds them.
    out_dir = LORA_OUT / lora_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always a pair — musubi raises ValueError on --fp8_base without --fp8_scaled.
    mem_flags: list[str] = []
    if fp8:
        mem_flags += ["--fp8_base", "--fp8_scaled"]
    if blocks_to_swap > 0:
        mem_flags += ["--blocks_to_swap", str(blocks_to_swap)]
    if mem_flags:
        print(f"[train] memory options: {' '.join(mem_flags)}")

    # save_every_n_steps bounds how much a Stop costs you. musubi has no
    # interrupt-and-save hook (no SIGINT handler anywhere in trainer_base), so
    # stopping cannot flush a fresh checkpoint — it can only preserve the most
    # recent periodic one. Saving every N steps shrinks the window of work a
    # stop throws away; 0 leaves epoch-boundary saves as the only granularity.
    step_save_flags: list[str] = []
    if save_every_n_steps > 0:
        step_save_flags = ["--save_every_n_steps", str(save_every_n_steps)]
    if save_last_n_epochs > 0:
        step_save_flags += ["--save_last_n_epochs", str(save_last_n_epochs)]
    if save_every_n_steps > 0 and save_last_n_steps > 0:
        # Only meaningful alongside step saving; musubi computes the removal
        # point from both, so setting it without save_every_n_steps does nothing.
        step_save_flags += ["--save_last_n_steps", str(save_last_n_steps)]

    stopped = False
    try:
        _run(
            [
                # Explicit flags instead of `accelerate config` — that command is
                # interactive and cannot run in a detached container.
                "accelerate", "launch",
                "--num_processes", "1",
                "--num_machines", "1",
                "--mixed_precision", "bf16",
                "--dynamo_backend", "no",
                "--num_cpu_threads_per_process", "1",
                "src/musubi_tuner/krea2_train_network.py",
                # --dit is the RAW DiT and --vae is the Qwen VAE. Swapping these
                # two is the classic failure; they are different models.
                "--dit", str(DIT_PATH),
                "--vae", str(VAE_PATH),
                "--dataset_config", str(toml_path),
                "--sdpa",
                "--mixed_precision", "bf16",
                "--timestep_sampling", "shift",
                "--weighting_scheme", "none",
                "--discrete_flow_shift", str(discrete_flow_shift),
                "--optimizer_type", optimizer_type,
                "--learning_rate", str(learning_rate),
                "--gradient_checkpointing",
                "--max_data_loader_n_workers", "2",
                "--persistent_data_loader_workers",
                "--network_module", "networks.lora_krea2",
                "--network_dim", str(network_dim),
                "--network_alpha", str(network_alpha),
                "--max_train_epochs", str(max_train_epochs),
                "--save_every_n_epochs", str(save_every_n_epochs),
                "--seed", str(seed),
                "--output_dir", str(out_dir),
                # No .safetensors here — musubi appends the extension itself, and
                # passing it produces `name.safetensors.safetensors`.
                "--output_name", lora_name,
                *step_save_flags,
                *mem_flags,
            ],
            "train",
            log,
            job_id,
            parse_progress=True,
        )
    except StoppedByUser:
        # Not an error. Whatever periodic checkpoints already reached disk are
        # kept and reported; the run just ends early.
        stopped = True
        print(f"[{job_id}] stopped by user — keeping checkpoints written so far")
        log.append("[train] stopped by user")

    # --- 6. commit -------------------------------------------------------
    produced = sorted(p.name for p in out_dir.glob("*.safetensors"))
    (out_dir / "visionary.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "lora_name": lora_name,
                "trigger_word": trigger_word,
                "images": len(images),
                "auto_caption": auto_caption,
                "hyperparams": {
                    "resolution": resolution,
                    "batch_size": batch_size,
                    "num_repeats": num_repeats,
                    "network_dim": network_dim,
                    "network_alpha": network_alpha,
                    "learning_rate": learning_rate,
                    "max_train_epochs": max_train_epochs,
                    "discrete_flow_shift": discrete_flow_shift,
                    "optimizer_type": optimizer_type,
                    "seed": seed,
                },
                "files": produced,
                "stopped_early": stopped,
            },
            indent=2,
        )
    )
    volume.commit()

    result = {
        "status": "stopped" if stopped else "completed",
        "job_id": job_id,
        "lora_name": lora_name,
        "trigger_word": trigger_word,
        "images": len(images),
        "output_dir": str(out_dir),
        "files": produced,
        "weights": weights_report,
        "duration_s": round(time.time() - started, 1),
        "log_tail": list(log)[-40:],
    }
    if stopped and not produced:
        # Honest reporting: a stop before the first checkpoint interval leaves
        # nothing behind. Say so rather than presenting an empty success.
        result["note"] = (
            "Stopped before any checkpoint was written. Lower save_every_n_steps "
            "(or save_every_n_epochs) if you want to be able to stop early and "
            "keep partial progress."
        )

    _publish(job_id, **{k: v for k, v in result.items() if k != "log_tail"})
    print(f"[{job_id}] {result['status']} in {result['duration_s']}s -> {out_dir}")
    return result


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------


@app.function(image=web_image, timeout=60, volumes={"/workspace": volume}, secrets=[hf_secret])
@modal.fastapi_endpoint(method="POST", label="loras")
def api_list_loras(payload: dict) -> dict[str, Any]:
    """List trained LoRAs on the volume, newest first."""
    try:
        _check_key(payload.get("access_key"))
    except AuthError as exc:
        return {"status": "error", "error": str(exc)}

    volume.reload()
    if not LORA_OUT.is_dir():
        return {"status": "ok", "loras": []}

    found = []
    for d in LORA_OUT.iterdir():
        if not d.is_dir():
            continue
        # The final model is "{name}.safetensors"; numbered files are epoch
        # checkpoints. Offer the final one first, then checkpoints newest-first.
        final = d / f"{d.name}.safetensors"
        ckpts = sorted(
            (p for p in d.glob("*.safetensors") if p != final),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        meta = d / "visionary.json"
        trigger = ""
        if meta.exists():
            try:
                trigger = json.loads(meta.read_text()).get("trigger_word", "")
            except Exception:
                pass
        files = ([final] if final.exists() else []) + ckpts
        if not files:
            continue
        found.append(
            {
                "name": d.name,
                "trigger_word": trigger,
                "files": [f.name for f in files],
                "path": str(files[0]),
                "modified": files[0].stat().st_mtime,
            }
        )

    found.sort(key=lambda x: x["modified"], reverse=True)
    return {"status": "ok", "loras": found}


@app.function(
    image=trainer_image,
    gpu=GPU,
    cpu=2.0,
    timeout=60 * 60,
    secrets=[hf_secret],
    volumes={"/workspace": volume},
)
def generate_job(
    job_id: str,
    prompt: str,
    negative_prompt: str = "",
    model: str = "turbo",
    lora_path: str | None = None,
    lora_multiplier: float = 1.0,
    width: int = 1024,
    height: int = 1024,
    steps: int | None = None,
    guidance_scale: float | None = None,
    mu: float | None = None,
    seed: int | None = None,
    num_images: int = 1,
    hf_token: str | None = None,
) -> dict[str, Any]:
    """
    Generate images with krea2_generate_image.py, optionally with a trained LoRA.

    Turbo is the default: 8 steps at guidance 1.0 with mu 1.15, against RAW's 28
    steps at 5.5 — same weights on disk, roughly 3.5x fewer steps.
    """
    if not NAME_RE.match(job_id):
        raise ValueError("Invalid job_id.")
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("prompt is required.")

    started = time.time()
    log: deque[str] = deque(maxlen=MAX_LOG_LINES)
    jobs[job_id] = {"status": "running", "phase": "generate", "stop": False}
    volume.reload()

    # Turbo and RAW have genuinely different sampler settings; defaulting them
    # per-model avoids silently generating 8-step RAW images (which look broken).
    use_turbo = model != "raw"
    if steps is None:
        steps = 8 if use_turbo else 28
    if guidance_scale is None:
        guidance_scale = 1.0 if use_turbo else 5.5
    if mu is None and use_turbo:
        mu = 1.15  # RAW leaves mu unset so its resolution-aware default applies

    token = _resolve_hf_token(hf_token)
    if use_turbo:
        _ensure_weight(TURBO_WEIGHT, token)
    for spec in WEIGHTS if not use_turbo else WEIGHTS[1:]:
        _ensure_weight(spec, token)

    out_dir = OUTPUTS / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "src/musubi_tuner/krea2_generate_image.py",
        prompt,  # positional, not a flag
        "--dit", str(TURBO_PATH if use_turbo else DIT_PATH),
        "--vae", str(VAE_PATH),
        "--text_encoder", str(TEXT_ENCODER_PATH),
        "--width", str(width),
        "--height", str(height),
        "--steps", str(steps),
        "--guidance_scale", str(guidance_scale),
        # NOTE the hyphen: this is the only hyphenated option in the script
        # (dest="num_images"). "--num_images" is an unrecognised argument.
        "--num-images", str(num_images),
        "--save_path", str(out_dir),
        "--attn_mode", "torch",
    ]
    if mu is not None:
        cmd += ["--mu", str(mu)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if negative_prompt.strip() and guidance_scale > 1:
        # Only meaningful with CFG on; Turbo at guidance 1.0 ignores it.
        cmd += ["--negative_prompt", negative_prompt.strip()]
    if lora_path:
        p = Path(lora_path)
        if not p.is_file() or LORA_OUT not in p.parents:
            raise ValueError(f"LoRA not found under models/Lora: {lora_path}")
        cmd += ["--lora_weight", str(p), "--lora_multiplier", str(lora_multiplier)]

    before = {p.name for p in out_dir.glob("*.png")}
    _run(cmd, "generate", log, job_id)
    new_files = sorted(p for p in out_dir.glob("*.png") if p.name not in before)

    images = []
    for p in new_files:
        images.append(
            {
                "name": p.name,
                "path": str(p),
                "data": "data:image/png;base64,"
                + base64.b64encode(p.read_bytes()).decode(),
            }
        )

    volume.commit()
    result = {
        "status": "completed",
        "job_id": job_id,
        "prompt": prompt,
        "model": "turbo" if use_turbo else "raw",
        "steps": steps,
        "guidance_scale": guidance_scale,
        "output_dir": str(out_dir),
        "images": images,
        "duration_s": round(time.time() - started, 1),
    }
    _publish(job_id, **{k: v for k, v in result.items() if k != "images"})
    print(f"[{job_id}] generated {len(images)} image(s) in {result['duration_s']}s")
    return result


@app.function(image=web_image, timeout=30, secrets=[hf_secret])
@modal.fastapi_endpoint(method="POST", label="generate")
def api_generate(payload: dict) -> dict[str, Any]:
    """Queue a generation. Returns immediately with a call_id to poll."""
    try:
        _check_key(payload.get("access_key"))
    except AuthError as exc:
        return {"status": "error", "error": str(exc)}

    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        return {"status": "error", "error": "A prompt is required."}

    def num(key: str, default, cast):
        try:
            v = payload.get(key)
            return cast(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    job_id = f"gen_{time.strftime('%Y%m%d_%H%M%S')}_{os.urandom(3).hex()}"
    call = generate_job.spawn(
        job_id=job_id,
        prompt=prompt,
        negative_prompt=str(payload.get("negative_prompt") or ""),
        model=str(payload.get("model") or "turbo"),
        lora_path=payload.get("lora_path") or None,
        lora_multiplier=num("lora_multiplier", 1.0, float),
        width=num("width", 1024, int),
        height=num("height", 1024, int),
        steps=num("steps", None, int),
        guidance_scale=num("guidance_scale", None, float),
        mu=num("mu", None, float),
        seed=num("seed", None, int),
        num_images=max(1, min(4, num("num_images", 1, int))),
        hf_token=payload.get("hf_token") or None,
    )
    try:
        jobs[job_id] = {"status": "queued", "phase": "generate", "stop": False}
    except Exception:
        pass
    return {"status": "queued", "job_id": job_id, "call_id": call.object_id}


# --------------------------------------------------------------------------
# Trigger / status endpoints
# --------------------------------------------------------------------------


@app.function(image=web_image, timeout=30, secrets=[hf_secret])
@modal.fastapi_endpoint(method="POST", label="trigger-train")
def api_trigger_train(payload: dict) -> dict[str, Any]:
    """
    Queue a run and return immediately.

    `spawn()` is what keeps the client off a multi-hour request — the gateway
    gets a response in milliseconds and the GPU work continues detached.
    """
    try:
        _check_key(payload.get("access_key"))
    except AuthError as exc:
        return {"status": "error", "error": str(exc)}

    job_id = str(payload.get("job_id") or "").strip()
    lora_name = str(payload.get("lora_name") or "").strip()
    trigger_word = str(payload.get("trigger_word") or "").strip()

    if not NAME_RE.match(job_id):
        return {"status": "error", "error": "Invalid job_id."}
    if not NAME_RE.match(lora_name):
        return {
            "status": "error",
            "error": "LoRA name must be 1-64 chars of letters, digits, - or _.",
        }
    if not trigger_word:
        return {"status": "error", "error": "trigger_word is required."}

    def num(key: str, default, cast):
        try:
            return cast(payload[key]) if payload.get(key) is not None else default
        except (TypeError, ValueError):
            return default

    call = train_job.spawn(
        job_id=job_id,
        lora_name=lora_name,
        trigger_word=trigger_word,
        auto_caption=bool(payload.get("auto_caption")),
        hf_token=payload.get("hf_token") or None,
        resolution=num("resolution", 1024, int),
        batch_size=num("batch_size", 1, int),
        num_repeats=num("num_repeats", 1, int),
        network_dim=num("network_dim", 32, int),
        network_alpha=num("network_alpha", 32, int),
        learning_rate=num("learning_rate", 1e-4, float),
        max_train_epochs=num("max_train_epochs", 30, int),
        save_every_n_epochs=num("save_every_n_epochs", 1, int),
        save_every_n_steps=num("save_every_n_steps", 0, int),
        save_last_n_epochs=num("save_last_n_epochs", 3, int),
        save_last_n_steps=num("save_last_n_steps", 0, int),
        discrete_flow_shift=num("discrete_flow_shift", 2.5, float),
        optimizer_type=str(payload.get("optimizer_type") or "adamw8bit"),
        seed=num("seed", 42, int),
        fp8=bool(payload.get("fp8")),
        blocks_to_swap=num("blocks_to_swap", 0, int),
    )

    # Seed the state immediately so the first status poll — which can easily
    # beat the container's cold start — sees "queued" instead of nothing.
    try:
        jobs[job_id] = {
            "status": "queued",
            "phase": "queued",
            "lora_name": lora_name,
            "stop": False,
            "call_id": call.object_id,
        }
    except Exception as exc:
        print(f"[{job_id}] could not seed job state: {exc}")

    return {
        "status": "queued",
        "job_id": job_id,
        "lora_name": lora_name,
        "call_id": call.object_id,
        "output_dir": f"/workspace/models/Lora/{lora_name}",
    }


@app.function(image=web_image, timeout=30, secrets=[hf_secret])
@modal.fastapi_endpoint(method="POST", label="train-status")
def api_train_status(payload: dict) -> dict[str, Any]:
    """
    Poll a run: live phase/percent/ETA while it runs, full result when it ends.

    Two sources, deliberately. FunctionCall.get() is authoritative for the
    terminal outcome (including crashes), but only resolves once the call is
    over — it cannot report progress. The Dict carries the live telemetry the
    container publishes as it goes. Terminal state wins when both exist.

    POST { "call_id": str, "job_id": str, "access_key": str }
    """
    try:
        _check_key(payload.get("access_key"))
    except AuthError as exc:
        return {"status": "error", "error": str(exc)}

    call_id = str(payload.get("call_id") or "").strip()
    job_id = str(payload.get("job_id") or "").strip()
    if not call_id:
        return {"status": "error", "error": "call_id is required."}

    live: dict[str, Any] = {}
    if job_id:
        try:
            live = dict(jobs.get(job_id) or {})
        except Exception:
            live = {}
    live.pop("stop", None)  # internal control flag, not for the client

    try:
        call = modal.FunctionCall.from_id(call_id)
    except Exception as exc:
        return {"status": "error", "error": f"Unknown call_id: {exc}"}

    try:
        result = call.get(timeout=0)  # non-blocking poll
        if isinstance(result, dict):
            return {**live, **result}
        return {"status": "completed", "result": result}
    except TimeoutError:
        # Still running — hand back whatever the container has published.
        return {**live, "status": "running", "call_id": call_id, "job_id": job_id}
    except Exception as exc:
        return {
            **live,
            "status": "failed",
            "call_id": call_id,
            "job_id": job_id,
            "error": str(exc),
        }


@app.function(image=web_image, timeout=30, secrets=[hf_secret])
@modal.fastapi_endpoint(method="POST", label="stop-train")
def api_stop_train(payload: dict) -> dict[str, Any]:
    """
    Ask a running job to stop.

    Cooperative, not a kill: this sets a flag the training container polls every
    couple of seconds, which then SIGTERMs the whole accelerate process group and
    returns normally with whatever checkpoints reached disk.

    Honest limitation: musubi has no interrupt-and-save hook — there is no signal
    handler anywhere in its trainer — so stopping CANNOT flush a fresh checkpoint
    mid-epoch. It preserves the most recent periodic save. Set save_every_n_steps
    on the run if you want that window to be small.

    POST { "job_id": str, "access_key": str }
    """
    try:
        _check_key(payload.get("access_key"))
    except AuthError as exc:
        return {"status": "error", "error": str(exc)}

    job_id = str(payload.get("job_id") or "").strip()
    if not NAME_RE.match(job_id):
        return {"status": "error", "error": "Invalid job_id."}

    try:
        state = dict(jobs.get(job_id) or {})
    except Exception as exc:
        return {"status": "error", "error": f"Could not read job state: {exc}"}

    if not state:
        return {"status": "error", "error": f"No such job: {job_id}"}
    if state.get("status") in {"completed", "stopped", "failed"}:
        return {"status": state["status"], "note": "Job already finished."}

    state["stop"] = True
    jobs[job_id] = state
    return {
        "status": "stopping",
        "job_id": job_id,
        "note": "Keeps checkpoints already written; cannot flush a new one mid-epoch.",
    }


# --------------------------------------------------------------------------
# CLI:  modal run backend/main.py --images-dir ./my_dataset --lora-name my_style
# --------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    images_dir: str,
    lora_name: str,
    trigger_word: str = "",
    auto_caption: bool = False,
    max_train_epochs: int = 30,
    network_dim: int = 32,
) -> None:
    """Train straight from a local folder, no browser involved."""
    src = Path(images_dir).expanduser()
    if not src.is_dir():
        raise SystemExit(f"Not a directory: {src}")

    job_id = f"ds_{time.strftime('%Y%m%d_%H%M%S')}_cli"
    files = [p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise SystemExit(f"No images in {src}")

    print(f"Uploading {len(files)} images as {job_id}…")
    stage_upload.remote(
        job_id,
        [
            (p.name, p.read_bytes())
            for p in src.iterdir()
            if p.suffix.lower() in IMAGE_EXTS or p.suffix.lower() == ".txt"
        ],
    )

    print(json.dumps(
        train_job.remote(
            job_id=job_id,
            lora_name=lora_name,
            trigger_word=trigger_word or lora_name,
            auto_caption=auto_caption,
            max_train_epochs=max_train_epochs,
            network_dim=network_dim,
            network_alpha=network_dim,
        ),
        indent=2,
    ))


@app.function(image=web_image, timeout=1800, volumes={"/workspace": volume})
def stage_upload(job_id: str, files: list[tuple[str, bytes]]) -> int:
    """Upload helper for the CLI path (the browser uses api_upload_dataset)."""
    raw = UPLOADS / job_id
    raw.mkdir(parents=True, exist_ok=True)
    for name, blob in files:
        (raw / Path(name).name).write_bytes(blob)
    volume.commit()
    return len(files)
