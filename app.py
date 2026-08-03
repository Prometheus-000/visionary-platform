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
DATASETS = WORKSPACE / "datasets"
UPLOADS = WORKSPACE / "uploads"
OUTPUTS = WORKSPACE / "outputs"
STAGING = WORKSPACE / ".cache" / "hf-staging"
MUSUBI = Path("/opt/musubi-tuner")

# The vendored sd-webui-forge-classic backend that inference runs on. Lives next
# to this file so `modal deploy` from anywhere still finds it.
FORGE_DIR = str(Path(__file__).parent / "forge")
FORGE = Path("/opt/forge")

GPU = "A100-40GB"  # measured 29.26 GiB peak at 1024px, rank 32 — ~11 GiB spare

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
        "repo_id": "krea/Krea-2-Raw",
        "filename": "raw.safetensors",
        "dest": MODELS / "krea2-raw.safetensors",
        "gated": True,
        "approx_gb": 26.3,
    },
    "turbo": {
        "label": "Krea 2 Turbo",
        "note": "DiT for generating — 8 steps",
        "repo_id": "krea/Krea-2-Turbo",
        "filename": "turbo.safetensors",
        "dest": MODELS / "krea2-turbo.safetensors",
        "gated": True,
        "approx_gb": 26.3,
    },
    "vae": {
        "label": "Qwen Image VAE",
        "note": "Required for both",
        "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
        "filename": "split_files/vae/qwen_image_vae.safetensors",
        "dest": MODELS / "qwen-image-vae.safetensors",
        "gated": False,
        "approx_gb": 0.25,
    },
    "text_encoder": {
        "label": "Qwen3-VL 4B",
        "note": "Text encoder, bf16",
        "repo_id": "Comfy-Org/Qwen3-VL",
        "filename": "text_encoders/qwen3vl_4b_bf16.safetensors",
        "dest": MODELS / "qwen3vl-4b-bf16.safetensors",
        "gated": False,
        "approx_gb": 8.9,
    },
}

RAW_PATH = MODEL_CATALOGUE["raw"]["dest"]
TURBO_PATH = MODEL_CATALOGUE["turbo"]["dest"]
VAE_PATH = MODEL_CATALOGUE["vae"]["dest"]
TE_PATH = MODEL_CATALOGUE["text_encoder"]["dest"]

CAPTION_MODEL = "fancyfeast/llama-joycaption-beta-one-hf-llava"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif"}
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_CAPTION_CHARS = 1024
THUMB_PX = 320

CAPTION_STYLES = {
    "descriptive": (
        "Write a descriptive caption for this image in a formal tone. Describe the "
        "subject, its appearance, composition, lighting, setting and artistic style. "
        "Do not speculate about things you cannot see."
    ),
    "casual": (
        "Write a descriptive caption for this image in a casual, natural tone. "
        "Describe what is happening, how it looks, and the overall mood."
    ),
    "tags": (
        "Write a comma-separated list of booru-style tags for this image covering "
        "subject, clothing, pose, setting, lighting and style. Output only tags."
    ),
}
CAPTION_LENGTHS = {
    "short": " Keep it to one concise sentence.",
    "medium": " Keep it to two or three sentences.",
    "long": " Be thorough and detailed, four or more sentences.",
}


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _publish(job_id: str, **fields: Any) -> None:
    """Merge progress fields into the job record the UI polls."""
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
    missing = [MODEL_CATALOGUE[k] for k in keys if not MODEL_CATALOGUE[k]["dest"].exists()]
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


def _model_status() -> list[dict[str, Any]]:
    out = []
    for key, spec in MODEL_CATALOGUE.items():
        dest: Path = spec["dest"]
        present = dest.exists() and dest.stat().st_size > 0
        out.append(
            {
                "key": key,
                "label": spec["label"],
                "note": spec["note"],
                "repo_id": spec["repo_id"],
                "gated": spec["gated"],
                "approx_gb": spec["approx_gb"],
                "present": present,
                "size_gb": round(dest.stat().st_size / 1e9, 2) if present else 0,
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
    from transformers import AutoProcessor, LlavaForConditionalGeneration

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

    cache_dir = str(WORKSPACE / ".cache" / "huggingface")
    processor = AutoProcessor.from_pretrained(CAPTION_MODEL, cache_dir=cache_dir)
    model = LlavaForConditionalGeneration.from_pretrained(
        CAPTION_MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0", cache_dir=cache_dir,
    )
    model.eval()

    instruction = CAPTION_STYLES.get(style, CAPTION_STYLES["descriptive"])
    instruction += CAPTION_LENGTHS.get(length, CAPTION_LENGTHS["medium"])

    written = 0
    for i, img_path in enumerate(todo, 1):
        if _stop_requested(job_id):
            print("[caption] stop requested")
            break
        try:
            image = Image.open(img_path).convert("RGB")
            convo = [
                {"role": "system", "content": "You are a helpful image captioner."},
                {"role": "user", "content": f"<image>\n{instruction}"},
            ]
            text = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda:0")
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=320, do_sample=True,
                    temperature=0.6, top_p=0.9, suppress_tokens=None,
                )[0][inputs["input_ids"].shape[1]:]
            caption = processor.tokenizer.decode(out, skip_special_tokens=True).strip()

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
    image=trainer_image, gpu=GPU, cpu=2.0, timeout=2 * 60 * 60,
    volumes={"/workspace": volume},
)
def caption_job(
    job_id: str, trigger_word: str = "", style: str = "descriptive",
    length: str = "medium", overwrite: bool = False,
) -> dict[str, Any]:
    jobs[job_id] = {"status": "running", "phase": "caption", "stop": False}
    volume.reload()
    src = UPLOADS / job_id
    if not src.is_dir():
        raise RuntimeError(f"No dataset staged for {job_id}.")

    started = time.time()
    written = _caption_images(src, trigger_word.strip(), job_id, style, length, overwrite)
    res = {
        "status": "completed", "job_id": job_id, "captioned": written,
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
    job_id: str, lora_name: str, trigger_word: str,
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

    src = UPLOADS / job_id
    if not src.is_dir():
        raise RuntimeError(f"No dataset staged for {job_id}.")

    work = DATASETS / job_id
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
            {"job_id": job_id, "lora_name": lora_name, "trigger_word": trigger_word,
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
        "status": status, "job_id": job_id, "lora_name": lora_name,
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


def _validate_loras(raw: Any) -> list[dict[str, Any]]:
    """
    Validate the LoRA stack coming off the wire. Pure stdlib, no torch.

    Paths are confined to loras/ with `resolve()` before the check, so a
    crafted `../../` cannot read a checkpoint — or anything else — off the
    volume. Anything malformed raises rather than being silently dropped; a LoRA
    that quietly does not load looks exactly like a LoRA with no effect.

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
        raw_path = str(entry.get("path") or "")
        path = Path(raw_path).resolve()
        if not path.is_file() or LORAS.resolve() not in path.parents:
            raise ValueError(f"LoRA must be a file under loras/: {raw_path!r}")

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

        out_dir = OUTPUTS / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%H%M%S")
        names = []
        for i, image in enumerate(images):
            name = f"{stamp}_{i:02d}.png"
            image.save(out_dir / name)
            names.append(name)
        volume.commit()

        report = getattr(pipe, "last_report", {})

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
# Web app — UI + API on a single URL
# --------------------------------------------------------------------------


@app.function(
    image=web_image, cpu=1.0, timeout=900, volumes={"/workspace": volume},
    max_containers=1,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse

    api = FastAPI()

    @api.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return UI_HTML

    @api.get("/api/where")
    async def where() -> dict[str, Any]:
        """
        What this deployment can actually see on the volume.

        Cheap CPU check for when the Models tab and a GPU job disagree — the
        usual cause is the app resolving a different volume than the one
        holding the weights, so the resolved name is part of the answer.
        """
        volume.reload()
        tree: dict[str, Any] = {}
        for d in (MODELS, LORAS, UPLOADS, OUTPUTS):
            if d.is_dir():
                tree[str(d)] = sorted(
                    f"{p.name} ({p.stat().st_size / 1e9:.2f} GB)" if p.is_file() else f"{p.name}/"
                    for p in d.iterdir() if not p.name.startswith(".")
                )[:25]
            else:
                tree[str(d)] = "(directory does not exist)"
        return {"volume": VOLUME_NAME, "mounted_at": str(WORKSPACE), "contents": tree}

    @api.get("/api/state")
    async def state() -> dict[str, Any]:
        volume.reload()
        loras = []
        if LORAS.is_dir():
            for d in sorted(LORAS.iterdir()):
                if not d.is_dir():
                    continue
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
        return {
            "models": _model_status(),
            "loras": loras,
            "hf_token_set": bool(_hf_token()),
            "samplers": SAMPLERS,
            "schedulers": SCHEDULERS,
            "max_loras": MAX_LORAS,
        }

    @api.post("/api/token")
    async def set_token(payload: dict) -> dict[str, Any]:
        token = str(payload.get("hf_token") or "").strip()
        config["hf_token"] = token
        return {"ok": True, "hf_token_set": bool(token)}

    @api.post("/api/download")
    async def download(payload: dict) -> dict[str, Any]:
        key = str(payload.get("key") or "")
        if key not in MODEL_CATALOGUE:
            return {"error": f"Unknown model: {key}"}
        download_job.spawn(key)
        return {"ok": True, "job_id": f"dl_{key}"}

    @api.post("/api/download-missing")
    async def download_missing(payload: dict) -> dict[str, Any]:
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

        # Appending must never be able to delete a dataset that already exists,
        # so track whether this call created the directory.
        existing = str(form.get("job_id") or "").strip()
        appending = bool(existing) and bool(NAME_RE.match(existing)) and (UPLOADS / existing).is_dir()
        job_id = existing if appending else f"ds{time.strftime('%Y%m%d%H%M%S')}{os.urandom(2).hex()}"
        raw = UPLOADS / job_id
        raw.mkdir(parents=True, exist_ok=True)

        count, zips = 0, []
        for up in form.getlist("files"):
            filename = getattr(up, "filename", None)
            if not filename:
                continue
            name = Path(filename).name
            suffix = Path(name).suffix.lower()
            if suffix not in IMAGE_EXTS and suffix not in {".zip", ".txt"}:
                continue
            target = raw / name
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

        volume.commit()
        # Same-named files overwrite rather than duplicate, so re-dropping the
        # same folder is idempotent instead of doubling the dataset.
        total = sum(1 for p in raw.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        return JSONResponse({"job_id": job_id, "added": count, "count": total})

    @api.get("/api/dataset/{job_id}")
    async def dataset(job_id: str) -> dict[str, Any]:
        from PIL import Image

        if not NAME_RE.match(job_id):
            return {"error": "Invalid job_id."}
        volume.reload()
        src = UPLOADS / job_id
        if not src.is_dir():
            return {"error": "Dataset not found."}

        thumbs = src / ".thumbs"
        thumbs.mkdir(exist_ok=True)
        items = []
        for img in sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            cached = thumbs / (img.stem + ".jpg")
            data = None
            try:
                if not cached.exists() or cached.stat().st_mtime < img.stat().st_mtime:
                    with Image.open(img) as im:
                        im = im.convert("RGB")
                        im.thumbnail((THUMB_PX, THUMB_PX), Image.LANCZOS)
                        buf = io.BytesIO()
                        im.save(buf, "JPEG", quality=78, optimize=True)
                        cached.write_bytes(buf.getvalue())
                data = "data:image/jpeg;base64," + base64.b64encode(cached.read_bytes()).decode()
            except Exception as exc:
                print(f"[thumb] {img.name}: {exc}")
            txt = img.with_suffix(".txt")
            items.append({
                "name": img.name,
                "caption": txt.read_text().strip() if txt.exists() else "",
                "thumb": data,
            })
        volume.commit()
        return {"job_id": job_id, "images": items}

    @api.post("/api/captions")
    async def save_captions(payload: dict) -> dict[str, Any]:
        job_id = str(payload.get("job_id") or "")
        captions = payload.get("captions")
        if not NAME_RE.match(job_id) or not isinstance(captions, dict):
            return {"error": "Bad request."}
        volume.reload()
        src = UPLOADS / job_id
        if not src.is_dir():
            return {"error": "Dataset not found."}
        saved = 0
        for raw_name, caption in captions.items():
            # Basename only — a client filename must not escape the directory.
            img = src / Path(str(raw_name)).name
            if img.exists() and img.suffix.lower() in IMAGE_EXTS:
                img.with_suffix(".txt").write_text(str(caption).strip()[:MAX_CAPTION_CHARS])
                saved += 1
        volume.commit()
        return {"ok": True, "saved": saved}

    @api.post("/api/remove-image")
    async def remove_image(payload: dict) -> dict[str, Any]:
        """Drop one image from a staged dataset, with its caption and thumbnail."""
        job_id = str(payload.get("job_id") or "")
        if not NAME_RE.match(job_id):
            return {"error": "Invalid job_id."}
        volume.reload()
        src = UPLOADS / job_id
        if not src.is_dir():
            return {"error": "Dataset not found."}

        # Basename only — a client-supplied name must not escape the directory.
        name = Path(str(payload.get("name") or "")).name
        img = src / name
        if not name or img.suffix.lower() not in IMAGE_EXTS or not img.exists():
            return {"error": "Image not found."}

        img.unlink(missing_ok=True)
        img.with_suffix(".txt").unlink(missing_ok=True)
        (src / ".thumbs" / (img.stem + ".jpg")).unlink(missing_ok=True)

        volume.commit()
        remaining = sum(1 for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        return {"ok": True, "count": remaining}

    @api.post("/api/prepend-trigger")
    async def prepend_trigger(payload: dict) -> dict[str, Any]:
        """
        Put the trigger word at the front of every caption that lacks it.

        For imported datasets: your own .txt files are used verbatim, so a
        caption without the trigger word trains a LoRA the trigger cannot
        summon. This fixes that without discarding the text.

        Idempotent by design — the test is `startswith`, not `in`. A substring
        test would false-positive on short triggers (a "cat" LoRA would skip
        "a cat sitting"), and running this twice must never double the prefix.
        """
        job_id = str(payload.get("job_id") or "")
        trigger = str(payload.get("trigger_word") or "").strip()
        if not NAME_RE.match(job_id):
            return {"error": "Invalid job_id."}
        if not trigger:
            return {"error": "A trigger word is required."}

        volume.reload()
        src = UPLOADS / job_id
        if not src.is_dir():
            return {"error": "Dataset not found."}

        changed = 0
        for img in sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS):
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

        volume.commit()
        return {"ok": True, "changed": changed}

    @api.post("/api/caption")
    async def caption(payload: dict) -> dict[str, Any]:
        job_id = str(payload.get("job_id") or "")
        if not NAME_RE.match(job_id):
            return {"error": "Invalid job_id."}
        caption_job.spawn(
            job_id=job_id,
            trigger_word=str(payload.get("trigger_word") or ""),
            style=str(payload.get("style") or "descriptive"),
            length=str(payload.get("length") or "medium"),
            overwrite=bool(payload.get("overwrite")),
        )
        return {"ok": True, "job_id": job_id}

    @api.post("/api/train")
    async def train(payload: dict) -> dict[str, Any]:
        job_id = str(payload.get("job_id") or "")
        lora_name = str(payload.get("lora_name") or "").strip()
        trigger = str(payload.get("trigger_word") or "").strip()
        if not NAME_RE.match(job_id):
            return {"error": "Invalid job_id."}
        if not NAME_RE.match(lora_name):
            return {"error": "LoRA name: letters, digits, - and _ only."}
        if not trigger:
            return {"error": "A trigger word is required."}

        def num(k, d, cast):
            try:
                v = payload.get(k)
                return cast(v) if v not in (None, "") else d
            except (TypeError, ValueError):
                return d

        train_job.spawn(
            job_id=job_id, lora_name=lora_name, trigger_word=trigger,
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
    async def generate(payload: dict) -> dict[str, Any]:
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
        Generator().generate.spawn(job_id=job_id, params={
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

    @api.get("/api/outputs/{job_id}")
    async def outputs(job_id: str) -> dict[str, Any]:
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
    async def status(job_id: str) -> dict[str, Any]:
        try:
            return jobs.get(job_id) or {"status": "unknown"}
        except Exception as exc:
            return {"status": "unknown", "error": str(exc)}

    @api.post("/api/stop/{job_id}")
    async def stop(job_id: str) -> dict[str, Any]:
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
def stage_upload(job_id: str, files: list[tuple[str, bytes]]) -> int:
    """Land locally-read files on the volume. Browser uploads use /api/upload."""
    raw = UPLOADS / job_id
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
    # Underscores only — job_id must satisfy NAME_RE on the way back in.
    job_id = f"ds_{time.strftime('%Y%m%d_%H%M%S')}_cli"

    print(f"Uploading {n_images} images as {job_id}…")
    stage_upload.remote(job_id, payload)

    if caption:
        print("Captioning…")
        print(json.dumps(caption_job.remote(job_id=job_id, trigger_word=trigger), indent=2))

    print(json.dumps(
        train_job.remote(
            job_id=job_id,
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
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<title>Visionary</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#000;--panel:rgba(255,255,255,.04);--line:rgba(255,255,255,.10);--fg:#f5f5f5;--mut:#8a8a8a;--dim:#5a5a5a}
body{background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif;-webkit-font-smoothing:antialiased}
.app{display:flex;min-height:100dvh}
aside{width:232px;flex:0 0 232px;border-right:1px solid rgba(255,255,255,.07);padding:16px 12px;display:flex;flex-direction:column;gap:2px}
.brand{display:flex;align-items:center;gap:10px;padding:10px 12px;border-radius:12px;background:var(--panel);margin-bottom:18px}
.dot{width:22px;height:22px;border-radius:7px;background:linear-gradient(135deg,#8b5cf6,#3b82f6);display:grid;place-items:center;font-size:12px}
.seclabel{font-size:11px;color:var(--dim);padding:0 12px;margin-bottom:6px}
nav button{display:flex;align-items:center;gap:10px;width:100%;padding:8px 12px;border:0;border-radius:9px;background:none;color:var(--mut);font:inherit;text-align:left;cursor:pointer}
nav button:hover{background:rgba(255,255,255,.04);color:#ddd}
nav button.on{background:rgba(255,255,255,.09);color:var(--fg)}
nav .ic{width:20px;height:20px;border-radius:6px;display:grid;place-items:center;font-size:11px;background:linear-gradient(135deg,#8b5cf6,#3b82f6)}
nav .ic.g{background:linear-gradient(135deg,#38bdf8,#6366f1)}
nav .ic.m{background:linear-gradient(135deg,#f59e0b,#ef4444)}
main{flex:1;min-width:0;padding:28px 32px 80px;max-width:980px}
h1{font-size:19px;font-weight:600;margin-bottom:3px;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:13px;margin-bottom:22px}
.card{border:1px solid var(--line);background:rgba(255,255,255,.02);border-radius:16px;padding:16px;margin-bottom:12px}
.row{display:flex;align-items:center;gap:14px}
.grow{flex:1;min-width:0}
label{display:block;font-size:12px;color:var(--mut);margin-bottom:6px}
input,textarea,select{width:100%;background:rgba(255,255,255,.05);border:1px solid var(--line);border-radius:11px;padding:10px 12px;color:var(--fg);font:inherit;outline:none}
input:focus,textarea:focus,select:focus{border-color:rgba(255,255,255,.28)}
textarea{resize:vertical}
button.b{background:#fff;color:#000;border:0;border-radius:12px;padding:11px 18px;font:600 14px/1 inherit;cursor:pointer}
button.b:disabled{background:rgba(255,255,255,.2);color:rgba(0,0,0,.4);cursor:not-allowed}
button.s{background:rgba(255,255,255,.07);color:var(--fg);border:1px solid var(--line);border-radius:11px;padding:9px 15px;font:500 13px/1 inherit;cursor:pointer}
button.s:hover{background:rgba(255,255,255,.12)}
button.s:disabled{opacity:.4;cursor:not-allowed}
.pill{display:inline-flex;align-items:center;gap:6px;background:rgba(255,255,255,.06);border:0;border-radius:999px;padding:7px 13px;color:#ddd;font:13px inherit;cursor:pointer}
.pill.on{background:#fff;color:#000}
.grid2{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}
.grid4{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.bar{height:3px;background:rgba(255,255,255,.1);border-radius:99px;overflow:hidden;margin-top:10px}
.bar>i{display:block;height:100%;background:#fff;border-radius:99px;transition:width .4s}
.muted{color:var(--dim);font-size:12px}
.ok{color:#4ade80}.warn{color:#fbbf24}.err{color:#f87171}
.err-box{border:1px solid rgba(248,113,113,.25);background:rgba(248,113,113,.1);color:#fca5a5;border-radius:12px;padding:11px 14px;font-size:13px;margin-bottom:12px}
.tile{border:1px solid var(--line);border-radius:14px;padding:11px;display:flex;gap:11px}
.tile img{width:88px;height:88px;object-fit:cover;border-radius:10px;flex:0 0 88px}
.tile textarea{font-size:12px;min-height:76px}
.rm{background:none;border:0;color:var(--dim);font-size:17px;line-height:1;padding:2px 6px;border-radius:6px;cursor:pointer;flex:0 0 auto}
.rm:hover{background:rgba(248,113,113,.15);color:#f87171}
.rm:disabled{opacity:.3}
#tiles{display:grid;gap:10px;grid-template-columns:1fr}
@media(min-width:900px){#tiles{grid-template-columns:1fr 1fr}}
.drop{border:1px dashed rgba(255,255,255,.2);border-radius:16px;padding:34px;text-align:center;cursor:pointer}
.drop.hot{border-color:rgba(255,255,255,.45);background:rgba(255,255,255,.05)}
.hide{display:none}
.gen-out{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}
.gen-out img{width:100%;border-radius:14px;border:1px solid var(--line)}
.lora-row{display:flex;gap:8px;align-items:center;margin-bottom:8px}
.lora-row select{flex:1;min-width:0}
.lora-row input{width:70px;text-align:center}
.lora-row button{padding:8px 10px}
code{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:#bbb}
@media(max-width:760px){.app{flex-direction:column}aside{width:auto;flex:none;flex-direction:row;overflow-x:auto;border-right:0;border-bottom:1px solid rgba(255,255,255,.07)}.brand,.seclabel{display:none}nav{display:flex;gap:4px}nav button{white-space:nowrap}main{padding:20px 16px 60px}.grid2{grid-template-columns:1fr}}
</style></head><body>
<div class="app">
<aside>
  <div class="brand"><span class="dot">✦</span><b style="font-size:14px">Visionary</b></div>
  <div class="seclabel">Tools</div>
  <nav>
    <button data-v="models" class="on"><span class="ic m">↓</span>Models</button>
    <button data-v="train"><span class="ic">✦</span>Train LoRA</button>
    <button data-v="generate"><span class="ic g">▦</span>Image</button>
  </nav>
</aside>
<main>
  <!-- MODELS -->
  <section id="v-models">
    <h1>Models</h1>
    <p class="sub">Nothing downloads until you ask for it.</p>
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
  </section>

  <!-- TRAIN -->
  <section id="v-train" class="hide">
    <h1>Train LoRA</h1>
    <p class="sub">Krea 2 RAW · rank 32 · bf16</p>
    <div id="train-err"></div>

    <div id="step-build">
      <!-- Dropzone stays put: it is how you start AND how you add more. -->
      <div class="drop" id="drop">
        <div style="font-size:22px;opacity:.35">↑</div>
        <div style="margin-top:6px" id="drop-title">Drop images or a .zip</div>
        <div class="muted" style="margin-top:3px" id="drop-sub">or click to browse</div>
        <input type="file" id="files" multiple accept="image/*,.zip,.txt" class="hide">
        <div id="up-prog" class="hide"><div class="bar"><i style="width:0%"></i></div></div>
      </div>

      <!-- Everything below appears only once there is a dataset. -->
      <div id="dataset" class="hide">
        <div class="card" style="margin-top:12px">
          <div class="row" style="gap:8px;flex-wrap:wrap">
            <button class="s" id="do-caption">Auto-caption</button>
            <select id="cap-style" style="width:auto"><option value="descriptive">Descriptive</option><option value="casual">Casual</option><option value="tags">Tags</option></select>
            <select id="cap-len" style="width:auto"><option value="short">Short</option><option value="medium" selected>Medium</option><option value="long">Long</option></select>
            <label style="display:flex;align-items:center;gap:7px;margin:0;color:#ddd"><input type="checkbox" id="cap-over" style="width:auto"> Replace existing</label>
            <span class="grow"></span>
            <button class="s" id="do-prepend" title="Put the trigger word at the front of every caption that lacks it">Prepend trigger</button>
          </div>
          <div id="cap-prog" class="hide"><div class="bar"><i style="width:0%"></i></div><p class="muted" style="margin-top:7px"></p></div>
        </div>

        <div class="row" style="margin:14px 2px 8px"><span class="muted" id="cap-count"></span></div>
        <div id="tiles"></div>

        <!-- Name and trigger sit here, next to the action that uses them. -->
        <div class="card grid2" style="margin-top:14px">
          <div><label>LoRA name</label><input id="lname" placeholder="my_style" spellcheck="false"></div>
          <div><label>Trigger word</label><input id="ltrig" placeholder="ohwx_style" spellcheck="false"></div>
        </div>
        <details class="card"><summary class="muted" style="cursor:pointer">Advanced</summary>
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

    <div id="step-run" class="hide">
      <div class="card">
        <div class="row"><b id="run-phase" class="grow">Starting…</b><span class="muted" id="run-pct"></span></div>
        <div class="bar"><i id="run-bar" style="width:0%"></i></div>
        <p class="muted" id="run-meta" style="margin-top:9px"></p>
        <div class="row" style="margin-top:14px"><button class="s" id="do-stop">Stop &amp; keep checkpoints</button></div>
      </div>
      <div class="card hide" id="run-done"></div>
    </div>
  </section>

  <!-- GENERATE -->
  <section id="v-generate" class="hide">
    <h1>Image</h1>
    <p class="sub" id="gen-model-line">Krea 2 Turbo · 8 steps</p>
    <div id="gen-err"></div>
    <p class="muted warn" id="gen-note" style="margin:-10px 2px 12px"></p>
    <div class="card">
      <textarea id="prompt" rows="3" placeholder="Describe an image…"></textarea>
      <div class="row" style="gap:8px;flex-wrap:wrap;margin-top:10px">
        <!-- populated from /api/state; models not on the volume are disabled -->
        <select id="g-model" style="width:auto"></select>
        <select id="g-aspect" style="width:auto">
          <option value="1024x1024">1:1</option><option value="1152x896">4:3</option>
          <option value="1216x832">3:2</option><option value="1344x768">16:9</option>
          <option value="832x1216">2:3</option><option value="768x1344">9:16</option>
        </select>
        <select id="g-n" style="width:auto"><option>1</option><option>2</option><option>3</option><option>4</option></select>
        <input id="g-seed" placeholder="seed" style="width:92px" inputmode="numeric">
        <span class="grow"></span>
        <button class="b" id="go-gen">Generate</button>
      </div>

      <!-- LoRA stack. Rows are added by hand; order is the order they patch in. -->
      <div id="lora-stack" style="margin-top:12px"></div>
      <div class="row" style="gap:8px;margin-top:8px">
        <button class="s" id="add-lora">Add LoRA</button>
        <span class="grow"></span>
        <button class="s" id="toggle-adv">Advanced</button>
      </div>

      <div id="gen-adv" class="hide" style="margin-top:12px">
        <textarea id="g-neg" rows="2" placeholder="Negative prompt"></textarea>
        <div class="row" style="gap:8px;flex-wrap:wrap;margin-top:10px">
          <select id="g-sampler" style="width:auto"></select>
          <select id="g-scheduler" style="width:auto"></select>
          <input id="g-steps" placeholder="steps" style="width:78px" inputmode="numeric">
          <input id="g-cfg" placeholder="CFG" style="width:78px" inputmode="decimal">
          <input id="g-shift" placeholder="shift 1.15" style="width:96px" inputmode="decimal">
        </div>
        <div class="row" style="gap:8px;margin-top:12px">
          <label class="row" style="gap:6px"><input type="checkbox" id="g-regional"> Regional</label>
          <span class="grow"></span>
          <select id="g-region-dir" class="hide" style="width:auto"><option value="columns">Columns</option><option value="rows">Rows</option></select>
        </div>
        <div id="region-stack" class="hide" style="margin-top:8px"></div>
        <div class="row hide" id="region-add-row" style="gap:8px;margin-top:8px">
          <button class="s" id="add-region">Add region</button>
        </div>
      </div>

      <div id="gen-prog" class="hide"><div class="bar"><i style="width:0%"></i></div><p class="muted" style="margin-top:7px"></p></div>
    </div>
    <div id="gen-out" class="gen-out"></div>
    <p class="muted" id="gen-meta" style="margin:10px 2px"></p>
  </section>
</main>
</div>
<script>
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const api=async(p,o)=>{const r=await fetch(p,o);return r.json()};
const post=(p,b)=>api(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});
let jobId=null, files=[], captions={}, saveT=null, poll=null;

// nav
$$('nav button').forEach(b=>b.onclick=()=>{
  $$('nav button').forEach(x=>x.classList.remove('on')); b.classList.add('on');
  ['models','train','generate'].forEach(v=>$('#v-'+v).classList.toggle('hide',v!==b.dataset.v));
});

// ---------- models ----------
async function loadState(){
  const s=await api('/api/state');
  $('#tok-state').innerHTML = s.hf_token_set?'<span class="ok">Token saved.</span>':'<span class="warn">No token saved.</span>';
  // Name the cost up front: how many are missing and how many GB that is.
  const miss=s.models.filter(m=>!m.present);
  const gb=miss.reduce((a,m)=>a+m.approx_gb,0);
  const dlAll=$('#dl-all');
  dlAll.disabled=!miss.length;
  dlAll.textContent=miss.length?`Download ${miss.length} missing · ${gb.toFixed(1)} GB`:'All models present';
  $('#models').innerHTML = s.models.map(m=>`
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
    </div>`).join('');
  $$('[data-dl]').forEach(b=>b.onclick=()=>startDownload(b.dataset.dl,b));

  // Options for every LoRA row, existing and future. Rebuilt on each poll so a
  // freshly trained LoRA appears without a reload; current picks are kept.
  window.MAX_LORAS=s.max_loras||6;
  loraOpts=s.loras.map(l=>l.files.map(f=>
    `<option value="${f.path}" data-t="${l.trigger_word||''}">${l.name} · ${f.name}</option>`).join('')).join('');
  $$('#lora-stack [data-f=path]').forEach(el=>{const v=el.value; el.innerHTML=loraOpts; el.value=v;});
  $('#add-lora').disabled=!s.loras.length;

  if(s.samplers&&!$('#g-sampler').options.length){
    $('#g-sampler').innerHTML=s.samplers.map(x=>`<option>${x}</option>`).join('');
    $('#g-scheduler').innerHTML=s.schedulers.map(x=>`<option>${x}</option>`).join('');
  }

  // Model picker reflects the volume rather than being hardcoded — otherwise it
  // claims both models are available even when neither is downloaded.
  const ms=$('#g-model'), prev=ms.value;
  const pick=s.models.filter(m=>m.key==='turbo'||m.key==='raw');
  ms.innerHTML=pick.map(m=>
    `<option value="${m.key}" ${m.present?'':'disabled'}>${m.label}${m.present?'':' — not downloaded'}</option>`).join('');
  const avail=pick.filter(m=>m.present).map(m=>m.key);
  ms.value = avail.includes(prev) ? prev : (avail[0]||'');
  const missing=pick.filter(m=>!m.present).length===pick.length;
  $('#go-gen').disabled=missing;
  $('#gen-note').textContent = missing
    ? 'No DiT on the volume — download Krea 2 Turbo on the Models tab.'
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
  if(!keep.length||uploading) return;
  uploading=true; $('#train-err').innerHTML='';
  const box=$('#up-prog'); box.classList.remove('hide'); const bar=box.querySelector('i');
  $('#drop-title').textContent='Uploading…';
  $('#drop-sub').textContent=`${keep.length} file${keep.length>1?'s':''}`;

  const fd=new FormData();
  keep.forEach(f=>fd.append('files',f,f.name));
  if(jobId) fd.append('job_id',jobId);          // append to the existing dataset

  // XHR, not fetch — fetch reports no upload progress.
  const x=new XMLHttpRequest();
  x.open('POST','/api/upload');
  x.upload.onprogress=e=>{ if(e.lengthComputable) bar.style.width=Math.round(e.loaded/e.total*100)+'%' };
  x.onload=async()=>{
    uploading=false; box.classList.add('hide'); bar.style.width='0%';
    let r={}; try{ r=JSON.parse(x.responseText) }catch{}
    if(r.error||x.status>=400||!r.job_id){
      // Show the status and any raw body — an opaque "failed" is not debuggable.
      const detail = r.error || (x.responseText||'').slice(0,300) || 'no response body';
      $('#train-err').innerHTML='<div class="err-box">Upload failed ('+x.status+'): '+
        detail.replace(/</g,'&lt;')+'</div>';
      resetDropLabel(); return;
    }
    jobId=r.job_id;
    $('#dataset').classList.remove('hide');
    await loadTiles();
  };
  x.onerror=()=>{ uploading=false; box.classList.add('hide'); resetDropLabel();
    $('#train-err').innerHTML='<div class="err-box">Network error during upload.</div>'; };
  x.send(fd);
}
function resetDropLabel(n){
  $('#drop-title').textContent = jobId ? 'Drop more images' : 'Drop images or a .zip';
  $('#drop-sub').textContent = 'or click to browse';
}
const show=s=>['build','run'].forEach(x=>$('#step-'+x).classList.toggle('hide',x!==s));

// ---------- review ----------
async function loadTiles(){
  const d=await api('/api/dataset/'+jobId);
  if(d.error){$('#train-err').innerHTML='<div class="err-box">'+d.error+'</div>';return}
  captions={};
  const esc=s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;');
  $('#tiles').innerHTML=d.images.map(i=>`
    <div class="tile" data-tile="${esc(i.name)}">
      ${i.thumb?`<img src="${i.thumb}" alt="">`:'<div style="width:88px;height:88px;border-radius:10px;background:rgba(255,255,255,.05);flex:0 0 88px"></div>'}
      <div class="grow">
        <div class="row" style="gap:8px;margin-bottom:5px">
          <code class="grow" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(i.name)}</code>
          <button class="rm" data-rm="${esc(i.name)}" title="Remove from dataset">×</button>
        </div>
        <textarea data-n="${esc(i.name)}" placeholder="No caption">${esc(i.caption)}</textarea>
      </div>
    </div>`).join('');
  $$('#tiles textarea').forEach(t=>t.oninput=()=>{
    captions[t.dataset.n]=t.value;
    clearTimeout(saveT); saveT=setTimeout(flush,1200);
  });
  $$('#tiles [data-rm]').forEach(b=>b.onclick=async()=>{
    b.disabled=true;
    const r=await post('/api/remove-image',{job_id:jobId,name:b.dataset.rm});
    if(r.error){$('#train-err').innerHTML='<div class="err-box">'+r.error+'</div>';b.disabled=false;return}
    delete captions[b.dataset.rm];
    b.closest('[data-tile]').remove();
    countTiles(r.count);
    if(!r.count){ $('#dataset').classList.add('hide'); jobId=null; }
    resetDropLabel();
  });
  countTiles(d.images.length, d.images.filter(i=>i.caption.trim()).length);
  resetDropLabel();
  checkTrainReady();
}
function countTiles(total, done){
  if(done===undefined) done=$$('#tiles textarea').filter(t=>t.value.trim()).length;
  if(total===undefined) total=$$('#tiles [data-tile]').length;
  $('#cap-count').textContent=`${total} image${total===1?'':'s'} · ${done} captioned`;
  checkTrainReady();
}
function checkTrainReady(){
  const n=$$('#tiles [data-tile]').length;
  const ok=n>0&&$('#lname').value.trim()&&$('#ltrig').value.trim();
  $('#go-train').disabled=!ok;
  $('#train-hint').textContent = !n ? '' :
    (!$('#lname').value.trim()||!$('#ltrig').value.trim()) ? 'Name it and set a trigger word to train' : '';
}
document.addEventListener('input',e=>{ if(e.target.id==='lname'||e.target.id==='ltrig') checkTrainReady() });
async function flush(){
  if(!Object.keys(captions).length) return;
  const send={...captions}; captions={};
  await post('/api/captions',{job_id:jobId,captions:send});
}
$('#do-prepend').onclick=async()=>{
  const trig=$('#ltrig').value.trim();
  if(!trig){$('#train-err').innerHTML='<div class="err-box">Set a trigger word first.</div>';return}
  clearTimeout(saveT); await flush();          // never clobber unsaved edits
  const b=$('#do-prepend'); b.disabled=true; const was=b.textContent;
  const r=await post('/api/prepend-trigger',{job_id:jobId,trigger_word:trig});
  if(r.error){$('#train-err').innerHTML='<div class="err-box">'+r.error+'</div>'}
  else{ b.textContent=r.changed?`Updated ${r.changed}`:'Already set'; await loadTiles(); }
  setTimeout(()=>{b.textContent=was;b.disabled=false},1800);
};
$('#do-caption').onclick=async()=>{
  clearTimeout(saveT); await flush();
  const btn=$('#do-caption'); btn.disabled=true;
  const box=$('#cap-prog'); box.classList.remove('hide');
  await post('/api/caption',{job_id:jobId,trigger_word:$('#ltrig').value.trim(),
    style:$('#cap-style').value,length:$('#cap-len').value,overwrite:$('#cap-over').checked});
  const t=setInterval(async()=>{
    const s=await api('/api/status/'+jobId);
    box.querySelector('i').style.width=(s.percent||0)+'%';
    box.querySelector('p').textContent=s.step?`Captioning ${s.step}/${s.total_steps}`:'Loading captioner…';
    if(s.status==='completed'){clearInterval(t);box.classList.add('hide');btn.disabled=false;loadTiles();}
    else if(s.status==='failed'){clearInterval(t);btn.disabled=false;
      $('#train-err').innerHTML='<div class="err-box">'+(s.error||'Captioning failed')+'</div>';}
  },2500);
};

// ---------- train ----------
$('#go-train').onclick=async()=>{
  clearTimeout(saveT); await flush();
  $('#train-err').innerHTML='';
  const r=await post('/api/train',{job_id:jobId,lora_name:$('#lname').value.trim(),
    trigger_word:$('#ltrig').value.trim(),network_dim:$('#a-dim').value,network_alpha:$('#a-alpha').value,
    max_train_epochs:$('#a-epochs').value,learning_rate:$('#a-lr').value,resolution:$('#a-res').value,
    num_repeats:$('#a-rep').value,batch_size:$('#a-bs').value,seed:$('#a-seed').value});
  if(r.error){$('#train-err').innerHTML='<div class="err-box">'+r.error+'</div>';return}
  show('run'); $('#run-done').classList.add('hide');
  poll=setInterval(pollTrain,3000); pollTrain();
};
async function pollTrain(){
  const s=await api('/api/status/'+jobId);
  $('#run-phase').textContent=s.phase||'Working…';
  $('#run-pct').textContent=s.percent!=null?s.percent+'%':'';
  $('#run-bar').style.width=(s.percent||0)+'%';
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
    $('#run-done').classList.remove('hide');
    $('#run-done').innerHTML=`<b>${s.status==='stopped'?'Stopped':'Training complete'}</b>
      <p class="muted" style="margin-top:7px">${s.note||''}</p>
      <p class="muted" style="margin-top:7px"><code>${s.output_dir||''}</code></p>
      <p class="muted" style="margin-top:5px">${(s.files||[]).length} checkpoint(s) · ${Math.round((s.duration_s||0)/60)} min</p>`;
    loadState();
  } else if(s.status==='failed'){
    clearInterval(poll);
    $('#train-err').innerHTML='<div class="err-box">'+(s.error||'Training failed')+'</div>';
  }
}
$('#do-stop').onclick=async()=>{ $('#do-stop').disabled=true; await post('/api/stop/'+jobId); };

// ---------- generate ----------
function syncModelLine(){
  const v=$('#g-model').value;
  $('#gen-model-line').textContent = !v ? 'No model downloaded'
    : v==='turbo' ? 'Krea 2 Turbo · 8 steps · CFG 1.0'
    : 'Krea 2 RAW · 28 steps · CFG 5.5';
}
$('#g-model').onchange=syncModelLine;
$('#toggle-adv').onclick=()=>$('#gen-adv').classList.toggle('hide');

// ---------- LoRA stack ----------
// One row per LoRA. Two weights each, the way Forge splits them: the first
// patches the DiT, the second the text encoder. Order matters — LoRAs patch in
// the order shown, so the arrows are authority, not decoration.
let loraOpts='';
function loraRow(sel,unet,te){
  const row=document.createElement('div');
  row.className='lora-row'; row.dataset.lora='1';
  row.innerHTML=`
    <select data-f="path">${loraOpts}</select>
    <input data-f="unet" inputmode="decimal" value="${unet??1}" title="UNet weight">
    <input data-f="te" inputmode="decimal" value="${te??1}" title="Text encoder weight">
    <button class="s" data-f="up">↑</button>
    <button class="s" data-f="down">↓</button>
    <button class="s" data-f="rm">✕</button>`;
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
  row.className='lora-row'; row.dataset.region='1';
  row.innerHTML=`
    <input data-f="prompt" placeholder="Region prompt" value="${prompt||''}" style="flex:1">
    <input data-f="x" inputmode="decimal" placeholder="x" style="width:62px">
    <input data-f="y" inputmode="decimal" placeholder="y" style="width:62px">
    <input data-f="width" inputmode="decimal" placeholder="w" style="width:62px">
    <input data-f="height" inputmode="decimal" placeholder="h" style="width:62px">
    <input data-f="weight" inputmode="decimal" value="1" style="width:62px" title="Weight">
    <button class="s" data-f="rm">✕</button>`;
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
  ['#region-stack','#region-add-row','#g-region-dir'].forEach(s=>$(s).classList.toggle('hide',!on));
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
  });
  if(r.error){$('#gen-err').innerHTML='<div class="err-box">'+r.error+'</div>';btn.disabled=false;box.classList.add('hide');return}
  const t=setInterval(async()=>{
    const s=await api('/api/status/'+r.job_id);
    box.querySelector('i').style.width=(s.percent||0)+'%';
    box.querySelector('p').textContent=s.step?`Step ${s.step}/${s.total_steps}`:(s.phase||'Working…');
    if(s.status==='completed'){
      clearInterval(t); btn.disabled=false; box.classList.add('hide');
      const out=await api('/api/outputs/'+r.job_id);
      $('#gen-out').innerHTML=(out.images||[]).map(i=>`<img src="${i.data}" alt="">`).join('')
        || '<p class="muted">Saved to '+(s.output_dir||'')+'</p>';
      // Surface which LoRAs actually matched — a stack that silently no-ops
      // looks identical to a stack that had no effect.
      const skipped=(s.loras||[]).filter(l=>!l.applied);
      $('#gen-meta').textContent=[
        (s.seeds||[]).join(', ')&&('seed '+(s.seeds||[]).join(', ')),
        s.sampler&&`${s.sampler} · ${s.steps} steps · CFG ${s.cfg_scale}`,
        s.duration_s&&`${s.duration_s}s`,
        skipped.length&&('not applied: '+skipped.map(l=>l.name+(l.reason?` (${l.reason})`:'')).join(', ')),
      ].filter(Boolean).join(' · ');
    } else if(s.status==='stopped'){
      clearInterval(t); btn.disabled=false; box.classList.add('hide');
    } else if(s.status==='failed'){
      clearInterval(t); btn.disabled=false; box.classList.add('hide');
      $('#gen-err').innerHTML='<div class="err-box">'+(s.error||'Generation failed')+'</div>';
    }
  },2000);
};

loadState();
</script></body></html>
"""
