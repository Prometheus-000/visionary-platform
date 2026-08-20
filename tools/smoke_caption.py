"""
Smoke test for the Qwen3-VL captioner.

    modal run tools/smoke_caption.py           # class + processor exist, no weights
    modal run tools/smoke_caption.py --gpu     # load the model and caption one image
    modal run tools/smoke_caption.py --gpu --model qwen3vl-uncensored --preset character

The cheap check is the one that matters after a transformers bump:
Qwen3VLForConditionalGeneration only exists from 4.57, and the failure mode of
pinning too low is an ImportError deep inside a paid GPU job. It reads every
entry in CAPTION_MODELS rather than the default one, because a second repo id
in a table is a second thing that can be misspelt, and the spelling is not
checked anywhere until a GPU container has already started.
"""

import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import (  # noqa: E402
    CAPTION_LENGTHS,
    DEFAULT_CAPTION_MODEL,
    DEFAULT_CAPTION_PRESET,
    HF_CACHE,
    _caption_instruction,
    _caption_models,
    _caption_presets,
    _looks_like_refusal,
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
    # Both classes: the live path loads through the Auto class now (so custom
    # repos work), and the Qwen class is the floor the pin was chosen for.
    from transformers import (  # noqa: F401
        AutoModelForImageTextToText,
        AutoProcessor,
        Qwen3VLForConditionalGeneration,
    )

    # Config only: downloads a few KB per repo, not 17 GB each, but still proves
    # every repo id is right and that this transformers can parse each
    # architecture. A captioner whose weights load through the same class is the
    # whole basis for the picker being a repo id rather than a second code path,
    # so `arch` is the assertion, not decoration.
    from transformers import AutoConfig

    # The merged table, not the literal: a captioner added under the gear is a
    # repo id typed once and never config-checked again, so this is the one
    # place a later transformers bump can prove it still parses.
    models = {}
    for key, spec in _caption_models().items():
        try:
            models[key] = {"repo": spec["repo"],
                           "arch": AutoConfig.from_pretrained(spec["repo"]).architectures}
        except Exception as exc:
            models[key] = {"repo": spec["repo"], "error": f"{type(exc).__name__}: {exc}"}

    return {
        "transformers": transformers.__version__,
        "models": models,
        "presets": list(_caption_presets()),
        "lengths": sorted(CAPTION_LENGTHS),
    }


@smoke.function(
    image=smoke_image, gpu="A100-40GB", cpu=4.0, timeout=60 * 60,
    volumes={"/workspace": volume, str(HF_CACHE): hf_cache},
)
def caption_one(dataset: str = "", model: str = DEFAULT_CAPTION_MODEL,
                preset: str = DEFAULT_CAPTION_PRESET, trigger: str = "") -> dict:
    """Caption a single real image end to end and hand back the prose."""
    import time

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    volume.reload()

    # Both roots: a set is a draft until it is saved, so on a volume that has
    # only ever been dropped into there is nothing under datasets/ at all and
    # the no-argument form would report an empty install as a broken one.
    roots = [Path("/workspace/datasets"), Path("/workspace/drafts")]
    if dataset:
        src = next((r / dataset for r in roots if (r / dataset).is_dir()), roots[0] / dataset)
        candidates = sorted(p for p in src.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    else:
        candidates = sorted(p for r in roots if r.is_dir()
                            for p in list(r.rglob("*.jpg")) + list(r.rglob("*.png")))
    if not candidates:
        return {"error": f"no images under {' or '.join(str(r) for r in roots)}"}

    img_path = candidates[0]
    cache_dir = str(HF_CACHE)
    repo = _caption_models()[model]["repo"]

    started = time.time()
    processor = AutoProcessor.from_pretrained(
        repo, cache_dir=cache_dir,
        min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28,
    )
    # The same class the live path loads through, so what this smokes is the
    # path a run takes rather than a sibling of it.
    vlm = AutoModelForImageTextToText.from_pretrained(
        repo, dtype=torch.bfloat16, device_map="cuda:0", cache_dir=cache_dir,
    )
    vlm.eval()
    load_s = round(time.time() - started, 1)

    # The real builder, not a reimplementation of it — the point of running this
    # is to see what the preset actually sends.
    instruction = _caption_instruction(preset, "medium", trigger)
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
        out = vlm.generate(**inputs, max_new_tokens=320, do_sample=True,
                           temperature=0.6, top_p=0.9)[0][inputs["input_ids"].shape[1]:]
    caption = processor.decode(out, skip_special_tokens=True).strip()

    volume.commit()
    return {
        "image": img_path.name, "size": image.size,
        "model": model, "repo": repo, "preset": preset,
        "load_s": load_s, "caption_s": round(time.time() - started, 1),
        # Reported rather than left to be read: a decline is fluent prose, so
        # eyeballing the caption below is exactly how it gets missed.
        "refused": _looks_like_refusal(caption),
        "caption": caption,
    }


@smoke.local_entrypoint()
def main(gpu: bool = False, dataset: str = "",
         model: str = DEFAULT_CAPTION_MODEL, preset: str = DEFAULT_CAPTION_PRESET,
         trigger: str = ""):
    if model not in _caption_models():
        raise SystemExit(f"--model must be one of: {', '.join(_caption_models())}")
    if preset not in _caption_presets():
        raise SystemExit(f"--preset must be one of: {', '.join(_caption_presets())}")
    print(check.remote())
    if gpu:
        print(caption_one.remote(dataset, model, preset, trigger))
