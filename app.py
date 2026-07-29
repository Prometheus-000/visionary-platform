"""
Visionary — standalone Krea 2 LoRA trainer on Modal.

One file. One command. No Vercel, no npm, no Modal Secrets, no access keys.
The UI is served by Modal itself, so `modal deploy app.py` gives you a single
URL that is the whole application.

    modal deploy app.py

Portable across Modal accounts: every path below is fixed, so switching profiles
lands on the identical sd-webui-forge-classic (neo) layout. The volume must
already exist — if it does not, you are on the wrong profile.

    forge-webui-repo  ->  /workspace
      models/Stable-diffusion/raw.safetensors      Krea 2 RAW DiT   (training)
      models/Stable-diffusion/turbo.safetensors    Krea 2 Turbo DiT (inference)
      models/VAE/qwen_image_vae.safetensors
      models/text_encoder/qwen3vl_4b_bf16.safetensors
      models/Lora/{name}/                          trained output
      outputs/{job}/                               generated images
      datasets/{job}/, uploads/{job}/              working dirs

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

# Deliberately NO create_if_missing: every profile already has this volume, so
# a missing one means the wrong Modal profile is active. Failing loudly on
# deploy beats silently creating an empty volume and re-downloading 60 GB.
volume = modal.Volume.from_name("forge-webui-repo")

# Live job state (progress, stop flags) and the saved HF token. Dicts rather
# than Secrets so there is no CLI setup step — paste the key into the UI.
jobs = modal.Dict.from_name("visionary-jobs", create_if_missing=True)
config = modal.Dict.from_name("visionary-config", create_if_missing=True)

WORKSPACE = Path("/workspace")
MODELS = WORKSPACE / "models"
LORA_OUT = MODELS / "Lora"
DATASETS = WORKSPACE / "datasets"
UPLOADS = WORKSPACE / "uploads"
OUTPUTS = WORKSPACE / "outputs"
STAGING = WORKSPACE / ".cache" / "hf-staging"
MUSUBI = Path("/opt/musubi-tuner")

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
        "dest": MODELS / "Stable-diffusion" / "raw.safetensors",
        "gated": True,
        "approx_gb": 26.3,
    },
    "turbo": {
        "label": "Krea 2 Turbo",
        "note": "DiT for generating — 8 steps",
        "repo_id": "krea/Krea-2-Turbo",
        "filename": "turbo.safetensors",
        "dest": MODELS / "Stable-diffusion" / "turbo.safetensors",
        "gated": True,
        "approx_gb": 26.3,
    },
    "vae": {
        "label": "Qwen Image VAE",
        "note": "Required for both",
        "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
        "filename": "split_files/vae/qwen_image_vae.safetensors",
        "dest": MODELS / "VAE" / "qwen_image_vae.safetensors",
        "gated": False,
        "approx_gb": 0.25,
    },
    "text_encoder": {
        "label": "Qwen3-VL 4B",
        "note": "Text encoder, bf16",
        "repo_id": "Comfy-Org/Qwen3-VL",
        "filename": "text_encoders/qwen3vl_4b_bf16.safetensors",
        "dest": MODELS / "text_encoder" / "qwen3vl_4b_bf16.safetensors",
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
        "If those directories are empty, this Modal profile's forge-webui-repo "
        "is not the one holding your weights — check `modal profile current`.",
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
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import (
        EntryNotFoundError, GatedRepoError, RepositoryNotFoundError,
    )

    spec = MODEL_CATALOGUE[key]
    dest: Path = spec["dest"]
    job_id = f"dl_{key}"
    jobs[job_id] = {"status": "running", "phase": f"Downloading {spec['label']}", "percent": 0}

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

    out_dir = LORA_OUT / lora_name
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
# --------------------------------------------------------------------------


@app.function(
    image=trainer_image, gpu=GPU, cpu=2.0, timeout=60 * 60,
    volumes={"/workspace": volume},
)
def generate_job(
    job_id: str, prompt: str, model: str = "turbo",
    lora_path: str | None = None, lora_multiplier: float = 1.0,
    width: int = 1024, height: int = 1024, num_images: int = 1,
    steps: int | None = None, guidance_scale: float | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    started = time.time()
    log: deque[str] = deque(maxlen=200)
    jobs[job_id] = {"status": "running", "phase": "generate", "stop": False}
    volume.reload()

    use_turbo = model != "raw"
    dit = TURBO_PATH if use_turbo else RAW_PATH
    _require_models("turbo" if use_turbo else "raw", "vae", "text_encoder")

    # Turbo and RAW need genuinely different sampler settings; defaulting per
    # model avoids silently rendering 8-step RAW output, which looks broken.
    if steps is None:
        steps = 8 if use_turbo else 28
    if guidance_scale is None:
        guidance_scale = 1.0 if use_turbo else 5.5

    out_dir = OUTPUTS / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "src/musubi_tuner/krea2_generate_image.py",
        prompt,  # positional
        "--dit", str(dit), "--vae", str(VAE_PATH), "--text_encoder", str(TE_PATH),
        "--width", str(width), "--height", str(height),
        "--steps", str(steps), "--guidance_scale", str(guidance_scale),
        # Note the hyphen — the only hyphenated option in that script
        # (dest="num_images"). "--num_images" is an unrecognised argument.
        "--num-images", str(num_images),
        "--save_path", str(out_dir), "--attn_mode", "torch",
    ]
    if use_turbo:
        cmd += ["--mu", "1.15"]  # RAW leaves mu unset for its resolution-aware default
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if lora_path:
        p = Path(lora_path)
        if not p.is_file() or LORA_OUT not in p.parents:
            raise ValueError("LoRA must be a file under models/Lora.")
        cmd += ["--lora_weight", str(p), "--lora_multiplier", str(lora_multiplier)]

    before = {p.name for p in out_dir.glob("*.png")}
    _run(cmd, "generate", job_id, log)
    new = sorted(p for p in out_dir.glob("*.png") if p.name not in before)
    volume.commit()

    # Only filenames go into the job record. The PNGs themselves are served by
    # /api/outputs/{job_id} straight off the volume — a 1024px base64 image is
    # megabytes, and this dict is polled every few seconds.
    res = {
        "status": "completed", "job_id": job_id,
        "files": [p.name for p in new],
        "model": "turbo" if use_turbo else "raw", "steps": steps,
        "output_dir": str(out_dir), "duration_s": round(time.time() - started, 1),
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
        usual cause is the app running against a different Modal profile than
        the one holding the weights.
        """
        volume.reload()
        tree: dict[str, Any] = {}
        for d in (MODELS / "Stable-diffusion", MODELS / "VAE",
                  MODELS / "text_encoder", LORA_OUT, UPLOADS):
            if d.is_dir():
                tree[str(d)] = sorted(
                    f"{p.name} ({p.stat().st_size / 1e9:.2f} GB)" if p.is_file() else f"{p.name}/"
                    for p in d.iterdir() if not p.name.startswith(".")
                )[:25]
            else:
                tree[str(d)] = "(directory does not exist)"
        return {"volume": "forge-webui-repo", "mounted_at": str(WORKSPACE), "contents": tree}

    @api.get("/api/state")
    async def state() -> dict[str, Any]:
        volume.reload()
        loras = []
        if LORA_OUT.is_dir():
            for d in sorted(LORA_OUT.iterdir()):
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
        if not prompt:
            return {"error": "A prompt is required."}

        def num(k, d, cast):
            try:
                v = payload.get(k)
                return cast(v) if v not in (None, "") else d
            except (TypeError, ValueError):
                return d

        job_id = f"gen{time.strftime('%Y%m%d%H%M%S')}{os.urandom(2).hex()}"
        generate_job.spawn(
            job_id=job_id, prompt=prompt,
            model=str(payload.get("model") or "turbo"),
            lora_path=payload.get("lora_path") or None,
            lora_multiplier=num("lora_multiplier", 1.0, float),
            width=num("width", 1024, int), height=num("height", 1024, int),
            num_images=max(1, min(4, num("num_images", 1, int))),
            seed=num("seed", None, int),
        )
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
      </div>
      <p class="muted" style="margin-top:8px">
        Needed for Krea 2 RAW and Turbo, which are gated. Accept the licence at
        huggingface.co/krea/Krea-2-Raw with the same account. <span id="tok-state"></span>
      </p>
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
    <div class="card">
      <textarea id="prompt" rows="3" placeholder="Describe an image…"></textarea>
      <div class="row" style="gap:8px;flex-wrap:wrap;margin-top:10px">
        <select id="g-model" style="width:auto"><option value="turbo">Krea 2 Turbo</option><option value="raw">Krea 2 RAW</option></select>
        <select id="g-lora" style="width:auto"><option value="">No LoRA</option></select>
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
      <div id="gen-prog" class="hide"><div class="bar"><i style="width:0%"></i></div><p class="muted" style="margin-top:7px"></p></div>
    </div>
    <div id="gen-out" class="gen-out"></div>
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
  const sel=$('#g-lora'); const cur=sel.value;
  sel.innerHTML='<option value="">No LoRA</option>'+s.loras.map(l=>
    l.files.map(f=>`<option value="${f.path}" data-t="${l.trigger_word||''}">${l.name} · ${f.name}</option>`).join('')).join('');
  sel.value=cur;
}
$('#tok-save').onclick=async()=>{
  const r=await post('/api/token',{hf_token:$('#tok').value});
  $('#tok').value=''; $('#tok-state').innerHTML=r.hf_token_set?'<span class="ok">Token saved.</span>':'<span class="warn">Cleared.</span>';
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
$('#g-model').onchange=()=>{
  const t=$('#g-model').value==='turbo';
  $('#gen-model-line').textContent=t?'Krea 2 Turbo · 8 steps':'Krea 2 RAW · 28 steps';
};
$('#g-lora').onchange=()=>{
  const o=$('#g-lora').selectedOptions[0], t=o&&o.dataset.t;
  if(t&&!$('#prompt').value.includes(t)) $('#prompt').value=(t+', '+$('#prompt').value).trim().replace(/,\s*$/,'');
};
$('#go-gen').onclick=async()=>{
  const p=$('#prompt').value.trim(); if(!p)return;
  $('#gen-err').innerHTML=''; const btn=$('#go-gen'); btn.disabled=true;
  const box=$('#gen-prog'); box.classList.remove('hide'); box.querySelector('p').textContent='Queued…';
  const [w,h]=$('#g-aspect').value.split('x');
  const r=await post('/api/generate',{prompt:p,model:$('#g-model').value,
    lora_path:$('#g-lora').value||null,width:w,height:h,num_images:$('#g-n').value,seed:$('#g-seed').value});
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
    } else if(s.status==='failed'){
      clearInterval(t); btn.disabled=false; box.classList.add('hide');
      $('#gen-err').innerHTML='<div class="err-box">'+(s.error||'Generation failed')+'</div>';
    }
  },3000);
};

loadState();
</script></body></html>
"""
