"""
Rewrite a prompt with the model that is already going to read it.

Krea 2's text encoder is **Qwen3-VL-4B in bf16** — a decoder language model, not
a T5 or a CLIP — and it is resident in this container for the whole session
because every render goes through it. So the model that turns "empty diner, 3am"
into a prompt is the same one that will parse the result. That is the argument
for this node existing rather than a second container: not only that the weights
are already paid for, but that a rewriter which *is* the encoder writes in the
dialect the encoder reads.

**Why a second copy rather than the resident instance.** ComfyUI holds the
encoder as a conditioning provider: `comfy/text_encoders/krea2.py` taps twelve
raw hidden layers and concatenates them, with no LM head wired and no KV cache,
because none of that is needed to produce conditioning. Driving generation on
that object means reaching past ComfyUI's public surface at a pinned SHA and
writing a decode loop by hand. The only thing it would buy is the ~8 GiB this
costs — and Krea 2 is ~24 GiB of an 80 GB card, so that is not scarce. A
pinned-fragile dependency and our own sampler, traded for memory nobody is short
of, is the wrong way round.

**The stripped head, and why the file can still generate.** The ComfyUI
repackage drops `lm_head` — it is dead weight for conditioning. It costs nothing
here because Qwen3-VL-4B sets `tie_word_embeddings: true`, so the output
projection *is* `embed_tokens.weight`, which the file does carry. Verified
against the safetensors index before this was written: 713 tensors, all 36
layers, `lm_head` absent, `model.language_model.embed_tokens.weight` present.

Loaded from the local file against a pinned config rather than by repo id, so a
run needs no network and a HuggingFace outage cannot take renders with it.
"""

import gc
import os
import threading

# Module-level and guarded, because ComfyUI may execute two graphs back to back
# and `@modal.enter` has no equivalent here — the node is constructed per
# execution, so anything cached on `self` is cached for one call.
_LOCK = threading.Lock()
_LOADED: dict = {}

MODEL_FILE = os.environ.get(
    "VISIONARY_TE_FILE", "/workspace/models/qwen3vl-4b-bf16.safetensors")
# The tokenizer and config come from the base repo, baked into the image at
# build time so this resolves from cache. The weights never do — those are the
# local file above.
BASE_REPO = os.environ.get("VISIONARY_TE_REPO", "Qwen/Qwen3-VL-4B-Instruct")


def _load():
    """The text half of Qwen3-VL, once per container, on first use."""
    if _LOADED:
        return _LOADED
    with _LOCK:
        if _LOADED:  # won by another thread while we waited
            return _LOADED

        import torch
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoTokenizer, Qwen3VLForConditionalGeneration

        cfg = AutoConfig.from_pretrained(BASE_REPO)
        tok = AutoTokenizer.from_pretrained(BASE_REPO)

        state = load_file(MODEL_FILE)
        # **The vision tower is loaded even though this path never sees a
        # picture**, and the reason is worth the comment because the obvious
        # optimisation is what broke this the first time it ran.
        #
        # Dropping its 315 tensors saves ~1-2 GiB. It also leaves every
        # `model.visual.*` parameter on the meta device, because `assign=True`
        # only materialises what the state dict actually carries — and the
        # `.to("cuda")` below then dies with "Cannot copy out of meta tensor; no
        # data", which names the symptom and not the cause. Krea 2 is ~24 GiB of
        # an 80 GB card, so the memory was never the scarce thing; correctness
        # was.
        #
        # Tied embeddings: the head is the embedding matrix, so supplying it is
        # a reference rather than a copy and costs no memory.
        if "lm_head.weight" not in state:
            emb = state.get("model.language_model.embed_tokens.weight")
            if emb is None:
                raise RuntimeError(
                    f"{MODEL_FILE} has neither lm_head.weight nor "
                    "model.language_model.embed_tokens.weight — it is not the "
                    "Krea 2 text encoder this node expects.")
            state["lm_head.weight"] = emb

        # **Built on CPU with real storage, not on `meta`.** Meta init is the
        # fast way to do this and it does not work here: `load_state_dict`
        # materialises only what the state dict carries, and a model's
        # *non-persistent buffers* — the rotary `inv_freq` among them — are
        # created at init and appear in no checkpoint. They stay meta, they are
        # not reported by `missing`, and the failure surfaces two lines later as
        # "Cannot copy out of meta tensor" naming a device rather than a buffer.
        # Twice, from two different angles, before it was worth the forty
        # seconds this costs once per container.
        #
        # The default dtype is set around construction so the random init that
        # is about to be overwritten allocates 8 GiB rather than 16.
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = Qwen3VLForConditionalGeneration(cfg)
        finally:
            torch.set_default_dtype(prev)

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(
                f"{len(missing)} weights missing from {MODEL_FILE}. First few: "
                f"{missing[:5]}")

        model = model.to("cuda", dtype=torch.bfloat16).eval()
        del state
        gc.collect()
        _LOADED.update(model=model, tok=tok, torch=torch)
        return _LOADED


class VisionaryRewrite:
    """prose + instruction -> prose, on the resident encoder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prose": ("STRING", {"multiline": True, "default": ""}),
                "instruction": ("STRING", {"multiline": True, "default": ""}),
                "max_tokens": ("INT", {"default": 420, "min": 16, "max": 2048}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "visionary"
    # The answer has to come back out of ComfyUI, and an output node's `ui` dict
    # is the one channel that reaches `/history/{prompt_id}` — which the caller
    # already polls for renders. No second transport.
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Never cached. Two identical requests are two deliberate presses, and
        # ComfyUI returning the first one's output for the second would look
        # exactly like a model that ignored the button.
        return float("nan")

    def run(self, prose, instruction, max_tokens):
        got = _load()
        torch, model, tok = got["torch"], got["model"], got["tok"]

        text = tok.apply_chat_template(
            [{"role": "system", "content": instruction},
             {"role": "user", "content": prose}],
            tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            out = model.generate(
                **ids, max_new_tokens=int(max_tokens),
                # Temperature 0, matching `_plain_call`: the same sentence
                # rewritten twice should not come back two different pictures.
                # Pressing the button again is not how you ask for a variation.
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id)
        said = tok.decode(out[0][ids["input_ids"].shape[1]:],
                          skip_special_tokens=True)
        # Cleaning stays in app.py — `_clean_rewrite` is the one implementation
        # and it is already the thing every other backend's output goes through.
        return {"ui": {"text": [said]}, "result": (said,)}


NODE_CLASS_MAPPINGS = {"VisionaryRewrite": VisionaryRewrite}
NODE_DISPLAY_NAME_MAPPINGS = {"VisionaryRewrite": "Visionary Rewrite"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
