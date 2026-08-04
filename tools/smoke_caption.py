"""
Smoke test for the Qwen3-VL captioner.

    modal run tools/smoke_caption.py           # class + processor exist, no weights
    modal run tools/smoke_caption.py --gpu     # load the model and caption one image

The cheap check is the one that matters after a transformers bump:
Qwen3VLForConditionalGeneration only exists from 4.57, and the failure mode of
pinning too low is an ImportError deep inside a paid GPU job.
"""

import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import (  # noqa: E402
    CAPTION_LENGTHS,
    CAPTION_MODEL,
    CAPTION_STYLES,
    HF_CACHE,
    caption_image,
    hf_cache,
    volume,
)

smoke = modal.App("visionary-caption-smoke")
smoke_image = caption_image.add_local_python_source("app")


@smoke.function(image=smoke_image, cpu=2.0, timeout=900)
def check() -> dict:
    """No GPU, no weights — prove the pinned transformers actually has the class."""
    import transformers
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: F401

    # Config only: downloads a few KB, not 16 GB, but still proves the repo id
    # is right and that this transformers can parse its architecture.
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(CAPTION_MODEL)

    return {
        "transformers": transformers.__version__,
        "model": CAPTION_MODEL,
        "arch": cfg.architectures,
        "styles": sorted(CAPTION_STYLES),
        "lengths": sorted(CAPTION_LENGTHS),
    }


@smoke.function(
    image=smoke_image, gpu="A100-40GB", cpu=4.0, timeout=60 * 60,
    volumes={"/workspace": volume, str(HF_CACHE): hf_cache},
)
def caption_one(dataset: str = "") -> dict:
    """Caption a single real image end to end and hand back the prose."""
    import time

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    volume.reload()

    root = Path("/workspace/datasets")
    if dataset:
        candidates = sorted(p for p in (root / dataset).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    else:
        candidates = sorted(root.rglob("*.jpg")) + sorted(root.rglob("*.png"))
    if not candidates:
        return {"error": f"no images under {root}"}

    img_path = candidates[0]
    cache_dir = str(HF_CACHE)

    started = time.time()
    processor = AutoProcessor.from_pretrained(
        CAPTION_MODEL, cache_dir=cache_dir,
        min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        CAPTION_MODEL, dtype=torch.bfloat16, device_map="cuda:0", cache_dir=cache_dir,
    )
    model.eval()
    load_s = round(time.time() - started, 1)

    instruction = CAPTION_STYLES["descriptive"] + CAPTION_LENGTHS["medium"]
    image = Image.open(img_path).convert("RGB")
    convo = [{
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}],
    }]
    inputs = processor.apply_chat_template(
        convo, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to("cuda:0")

    started = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=320, do_sample=True,
                             temperature=0.6, top_p=0.9)[0][inputs["input_ids"].shape[1]:]
    caption = processor.decode(out, skip_special_tokens=True).strip()

    volume.commit()
    return {
        "image": img_path.name, "size": image.size,
        "load_s": load_s, "caption_s": round(time.time() - started, 1),
        "caption": caption,
    }


@smoke.local_entrypoint()
def main(gpu: bool = False, dataset: str = ""):
    print(check.remote())
    if gpu:
        print(caption_one.remote(dataset))
