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
#
# **Loaded at container warm-up, not on first use, and the memory objection
# does not survive its own arithmetic.** This was lazy, and the cost landed on
# whoever pressed the button first: ~8 GB read off a *network* volume plus the
# CPU construction priced at forty seconds below. Minutes, charged to the one
# person in the system who is sitting there watching — and charged for a
# feature whose whole job happens *before* a generation, on a container that
# only exists because of one.
#
# It was then staged on CPU and promoted on first use, to keep 9 GiB off a card
# holding 42.5 GB of H3. That caution had the numbers backwards: 42.5 GB is
# what the int8 repackage buys *so that the model is resident rather than
# offloading every request*, on an 80 GB card. 42.5 + 9 leaves 28.5 GB. The
# half-measure bought nothing real and cost a second code path.
#
# One thing the arithmetic does not cover, so it is written down: this copy is
# outside ComfyUI's model management. `unload_all_models()` cannot see it and
# `/free` will not drop it, so the 9 GiB is subtracted from what ComfyUI
# believes it has for the whole life of the container. That is the same class
# of fact as the regional node's stranded LoRAs — see `_reclaim()` — and the
# reason it is fine here rather than a leak is that it is bounded, constant,
# and known at startup instead of growing per run.
_LOCK = threading.Lock()
_READY: dict = {}

MODEL_FILE = os.environ.get(
    "VISIONARY_TE_FILE", "/workspace/models/qwen3vl-4b-bf16.safetensors")
# The tokenizer and config come from the base repo, baked into the image at
# build time so this resolves from cache. The weights never do — those are the
# local file above.
BASE_REPO = os.environ.get("VISIONARY_TE_REPO", "Qwen/Qwen3-VL-4B-Instruct")


def _load():
    """The text half of Qwen3-VL, once per container, resident on the card."""
    if _READY:
        return _READY
    with _LOCK:
        if _READY:  # won by another thread while we waited
            return _READY

        import torch
        from safetensors.torch import load_file
        from transformers import (AutoConfig, AutoProcessor, AutoTokenizer,
                                  Qwen3VLForConditionalGeneration)

        # **`local_files_only`, or the bake buys nothing.** These three are
        # baked into the image precisely so a warm H100 never waits on
        # HuggingFace and an outage there cannot take renders with it — but a
        # repo id without this flag still reaches the Hub to check the revision
        # even when every file is already in the cache. So the files were local
        # and the *lookup* was not: three API calls on every container start,
        # for nothing, on the one path that had been documented as offline.
        _local = {"local_files_only": True}
        cfg = AutoConfig.from_pretrained(BASE_REPO, **_local)
        tok = AutoTokenizer.from_pretrained(BASE_REPO, **_local)
        # The processor is the vision path's tokenizer — it writes the image
        # placeholder tokens into the template and turns pixels into
        # `pixel_values`/`image_grid_thw`. Loaded unconditionally because it is
        # a config read, not weights, and a lazy load would make the first
        # image request the one that discovers the cache is incomplete.
        proc = AutoProcessor.from_pretrained(BASE_REPO, **_local)

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
        _READY.update(model=model, tok=tok, proc=proc, torch=torch)
        return _READY


class VisionaryRewrite:
    """prose + instruction (+ an optional frame) -> prose, on the resident encoder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prose": ("STRING", {"multiline": True, "default": ""}),
                "instruction": ("STRING", {"multiline": True, "default": ""}),
                "max_tokens": ("INT", {"default": 420, "min": 16, "max": 2048}),
            },
            # Base64 as a STRING rather than an IMAGE socket, deliberately: the
            # caller holds base64 (it is what the routes carry), the graph is
            # already an HTTP POST so the bytes ride the same channel as
            # everything else, and `LoadImage` would hand back a ComfyUI float
            # tensor that the processor wants converted straight back to PIL.
            # One input, no staging, no cleanup.
            "optional": {
                "image_b64": ("STRING", {"multiline": True, "default": ""}),
                # The container's warm-up knock. `@modal.enter` runs in the
                # Modal process and the weights live in ComfyUI's, so a graph
                # is the only way to reach across — and this is the existing
                # one rather than a second transport, which is why it is an
                # input here instead of a node of its own.
                "warm_only": ("BOOLEAN", {"default": False}),
            },
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

    def run(self, prose, instruction, max_tokens, image_b64="", warm_only=False):
        # Loaded and not a token further. The caller is the container saying
        # "do the slow part now, while nothing is waiting on you" — sampling
        # here would put a decode loop in front of the first real render for
        # nobody's benefit.
        if warm_only:
            _load()
            return {"ui": {"text": [""]}, "result": ("",)}

        got = _load()
        torch, model, tok = got["torch"], got["model"], got["tok"]

        if image_b64:
            # The vision path — the tower was resident all along (see `_load`),
            # so this is the plumbing that finally uses it. The message content
            # becomes a parts list because that is what makes the chat template
            # emit the vision placeholder tokens; a bare string never does,
            # and the failure is a model that silently answers without looking.
            import base64
            import io

            from PIL import Image
            try:
                img = Image.open(
                    io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
            except Exception as exc:
                raise RuntimeError(
                    f"image_b64 did not decode to an image: {exc}") from exc
            proc = got["proc"]
            text = proc.apply_chat_template(
                [{"role": "system", "content": instruction},
                 {"role": "user", "content": [
                     {"type": "image"},
                     {"type": "text", "text": prose}]}],
                tokenize=False, add_generation_prompt=True)
            ids = proc(text=[text], images=[img],
                       return_tensors="pt").to(model.device)
        else:
            # The text path, byte-for-byte what it was before the image input
            # existed — Enhance's behaviour is not this feature's to change.
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
        # Returned verbatim. Cleaning stays with the caller, which is now
        # `/api/motion` alone — it parses this into labelled sections, so a
        # tidier here would be a second thing deciding where a clause ends.
        return {"ui": {"text": [said]}, "result": (said,)}


NODE_CLASS_MAPPINGS = {"VisionaryRewrite": VisionaryRewrite}
NODE_DISPLAY_NAME_MAPPINGS = {"VisionaryRewrite": "Visionary Rewrite"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
