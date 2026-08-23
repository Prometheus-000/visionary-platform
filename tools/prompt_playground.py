"""
A bench for MiniMax's prompt grammar, deployed on its own.

    modal deploy tools/prompt_playground.py

Rough prose in, an H3 document out, so a question about the format costs
seconds rather than a two-to-three minute take. It is `geocine`'s Space —
https://huggingface.co/spaces/geocine/MiniMax-H3-Prompt-Enhancer-2.6B — run on
our own GPU because theirs is a shared ZeroGPU slice that throttles under any
real comparison.

**Its own app, not a route on app.py**, and that is the point rather than a
shortcut: this is a bench. It exists to answer whether a trained enhancer beats
a compiler on the four criteria CLAUDE.md judges by, and the answer might be
no. A bench that is wired into the product is a bench somebody has to unwire.

## What it is

`LiquidAI/LFM2.5-2.6B` fine-tuned on the guides — `geocine/…-2.6b`. Not a
template expander: the grammar is in the weights, which is why its system
prompt is 2 KB where ours is a compiler. Six of them, one per task, because
"T2VA has no alignment line" and "L2VA's line is bracketed" are different
instructions and mixing them leaks across tasks.

## Why the Space is vendored rather than transcribed

The system prompts, the user envelope and the post-processors come out of the
Space at a pinned revision. Transcribing them would make this a bench for a
copy — the same reason `_from_app.py` pulls out of app.py by AST rather than
holding a second vocabulary.

## What reading it turned up, before a single generation

Two independent corroborations of what `smoke_prompt.py` asserts, which is
worth more than it sounds because both were derived from the guides by hand:

- **Its alignment lines are byte-identical to `H3_ALIGN`**, including the
  guide's own asymmetry — FL2VA writes bare `Picture 1 (from Shot 1)` and L2VA
  writes bracketed `<Picture 1> (from [Shot N])`. Its docstring flags the
  inconsistency in the same words ours does and preserves it for the same
  reason. Two implementations, one oddity, kept twice on purpose.
- **Its `REF_TASKS` is `H3_TASK_TYPES`** — `reference_generation`,
  `keyframe_completion`, `video_editing`, `video_continuation`,
  `audio_reuse`, `audio_reference`, joined with `+`.

And one thing that argues *for* the compiler rather than against it: the Space
carries `postprocess.py`, `timestamps.py`, `ref_repair.py` and `meta_leak.py`
— **a trained model still needs deterministic repair on the way out.** The
question was never model-or-compiler; it is where the split falls.
"""

from pathlib import Path

import modal

APP_NAME = "visionary-prompt-playground"

# The Space, pinned. A branch would let the prompts move under a comparison
# that is only worth anything if both sides hold still.
SPACE_ID = "geocine/MiniMax-H3-Prompt-Enhancer-2.6B"
SPACE_REV = "main"
# Unpacked to a plain directory rather than used out of the HF cache, and the
# reason is subtle enough to be worth the line: the cache stores a snapshot's
# files as **symlinks into `blobs/`**, and the Space's `paths.py` locates its
# own prompts with `Path(__file__).resolve().parent.parent`. `resolve()`
# follows the symlink to the blob, so ROOT lands at the repo cache root and the
# prompts are looked for one directory that does not exist —
# `FileNotFoundError: …/spaces--geocine--…/prompts/system_base.txt`, with the
# `snapshots/<sha>/` segment silently gone. Their arithmetic is correct; our
# storage layout broke it. `local_dir` writes real files and it resolves.
SPACE_DIR = "/opt/enhancer"
MODEL_ID = "geocine/minimax-video-prompt-enhancer-2.6b"

# The safetensors rather than the Q4_K_M GGUF the Space serves. It quantizes
# because ZeroGPU hands out a slice and llama.cpp is what fits in one; we have
# a whole L4, and a bench that measures a 4-bit quantization cannot tell a
# weakness of the model from a weakness of the quantization.
GPU = "L4"

# The other arm. Krea 2's encoder, already resident in app.py's image container
# and already shown a frame by the motion path — so this arm costs no new
# weights, and it can *see*, which the fine-tune cannot: LFM2.5 is text-only and
# takes references as sentences somebody wrote about them.
#
# The comparison is therefore not "which model is better". It is **where the
# grammar should live** — baked into 2.6B of weights by fine-tuning, or handed
# to a bigger general model as instructions. This codebase has thirty blind
# render comparisons against the second bet with different rules, and it lost
# all thirty. Same shape, guide-faithful rules this time.
VLM_ID = "Qwen/Qwen3-VL-4B-Instruct"

# `1038lab/ComfyUI-MiniMax-H3-Promptor`, pinned, for one file: `vision_prompts.json`.
# Not installed as a node pack — it reaches its vision through OpenAI, Claude,
# Gemini or a separate Ollama, and we already have a VLM resident. What is worth
# taking is what it *asks a picture*, which is nine prompts somebody arrived at
# by doing this for real.
#
# And they line up with the composer's slots almost one for one, which is the
# result worth recording: Face -> Subject / Identity, Wardrobe -> the clothing
# half of it, Establishing -> Cinematic Composition, Style -> Style & Aesthetics.
# The slot is not only a tag on a file. **The slot is the vision prompt.**
PROMPTOR_REPO = "https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor"
PROMPTOR_SHA = "3fcaf7ad7a46851a8f4baaf4197b0cd439af95c7"

# Which of the nine a slot asks for. The clause this produces is the one thing
# `_compile_h3_scene` cannot write: it knows a picture was dropped on Face, and
# not that the face is a young woman with long dark hair.
SLOT_VISION = {
    "face": "Subject / Identity",
    "wardrobe": "Subject / Identity",
    "body": "Action / Emotion",
    "establishing": "Cinematic Composition",
    "style": "Style & Aesthetics",
    "object": "Prop & Object Interaction",
}

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    # transformers 5.x, and the floor is load-bearing rather than tidy: LFM2.5's
    # tokenizer_config names `TokenizersBackend`, a class that does not exist
    # before v5, and 4.57 fails at load with "Tokenizer class TokenizersBackend
    # does not exist or is not currently imported" — which reads as a corrupt
    # download rather than a version floor. Same shape as app.py's unpinned
    # vLLM: a tokenizer attribute that moved across a major.
    .pip_install(
        "transformers==5.15.1",
        "accelerate==1.14.0",
        # 1.x, because transformers 5 requires `huggingface-hub>=1.5.0` and the
        # 0.35 that app.py uses resolves to ResolutionImpossible rather than to
        # anything that names the floor. Pinned separately from app.py's on
        # purpose: this is its own app precisely so a bench's dependency
        # cascade cannot reach the thing that renders.
        "huggingface_hub==1.28.0",
        "fastapi[standard]==0.115.12",
    )
    .apt_install("git")
    .run_commands(
        f"git clone {PROMPTOR_REPO} /opt/promptor",
        f"cd /opt/promptor && git checkout {PROMPTOR_SHA}",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0"})
    # The third arm. `_compile_h3_scene` is pure stdlib by design — it runs on
    # the CPU web container so a bad pill is a form error — so it costs this
    # image nothing but two files, and it is the *shipping* compiler rather
    # than a copy of it.
    # **The repo's own layout, mirrored.** `_from_app.py` finds app.py with
    # `Path(__file__).resolve().parent.parent / "app.py"` — it expects to be in
    # `tools/` beside it. Flattening the two into one directory makes it look
    # for `/opt/app.py`, which is the second time in this file a vendored
    # module has located a sibling by walking up and been wrong because our
    # layout differed (the Space's `paths.py` was the first). The lesson both
    # times: copy somebody's tree the shape they wrote it, or their path
    # arithmetic is a bug you introduced.
    .add_local_file("app.py", "/opt/vis/app.py", copy=True)
    .add_local_file("tools/_from_app.py", "/opt/vis/tools/_from_app.py", copy=True)
    .add_local_file("tools/prompt_playground.html",
                    "/opt/vis/prompt_playground.html", copy=True)
    # Weights and Space baked in, for the reason app.py bakes the tokenizer it
    # needs: a bench you have to wait on is a bench you stop using.
    .run_commands(
        f"python -c \"from huggingface_hub import snapshot_download as d; "
        f"d('{MODEL_ID}'); "
        f"d('{SPACE_ID}', repo_type='space', revision='{SPACE_REV}', "
        f"local_dir='{SPACE_DIR}'); "
        f"d('{VLM_ID}')\""
    )
)


@app.cls(image=image, gpu=GPU, scaledown_window=5 * 60, max_containers=1)
@modal.concurrent(max_inputs=1)
class Enhancer:
    """One model, loaded once. `max_inputs=1` because one GPU runs one decode."""

    @modal.enter()
    def load(self):
        import sys

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # The Space's own package, imported rather than reimplemented — it owns
        # the user envelope and the post-processors, and a bench that rebuilt
        # them would be benching the rebuild.
        sys.path.insert(0, SPACE_DIR)
        from minimax.formatting.envelope import build_user_message, load_system
        from minimax.formatting.postprocess import postprocess_generation

        self._envelope = build_user_message
        self._system = load_system
        self._post = postprocess_generation

        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
        )
        self.model.eval()
        print(f"[playground] {MODEL_ID} on {GPU}", flush=True)

    @modal.method()
    def run(self, task: str, seconds: float, prompt: str,
            assets: list[str] | None = None,
            temperature: float = 0.0, max_new_tokens: int = 1200) -> dict:
        import torch

        messages = [
            {"role": "system", "content": self._system(task)},
            {"role": "user",
             "content": self._envelope(task, seconds, prompt, assets or None)},
        ]
        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        # LFM2.5's template opens a `<think>` block the fine-tune's targets
        # never contain, so leaving it in samples off-distribution. Lifted from
        # the Space's `generate.py` rather than guessed.
        if text.rstrip().endswith("<think>"):
            text = text.rstrip()[: -len("<think>")]

        ins = self.tok(text, return_tensors="pt").to("cuda")
        kw = {"max_new_tokens": max_new_tokens,
              "pad_token_id": self.tok.pad_token_id or self.tok.eos_token_id}
        # Greedy by default, which is the Space's own default and the only
        # setting under which two runs of one prompt are the same measurement.
        if temperature > 0:
            kw |= {"do_sample": True, "temperature": temperature, "top_k": 40}
        with torch.inference_mode():
            out = self.model.generate(**ins, **kw)

        raw = self.tok.decode(out[0][ins["input_ids"].shape[-1]:],
                              skip_special_tokens=True).strip()
        return {"raw": raw, "text": self._post(raw, task, float(seconds)),
                "task": task, "seconds": seconds}


@app.cls(image=image, gpu=GPU, scaledown_window=5 * 60, max_containers=1)
@modal.concurrent(max_inputs=1)
class Vision:
    """The same task, the same system prompts, a general model that can look.

    Deliberately the Space's prompts rather than prompts of our own: the arms
    have to differ in *one* thing. Give this one better instructions and the
    result says nothing about where the grammar belongs.
    """

    @modal.enter()
    def load(self):
        import sys

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        sys.path.insert(0, SPACE_DIR)
        from minimax.formatting.envelope import build_user_message, load_system
        from minimax.formatting.postprocess import postprocess_generation

        self._envelope, self._system, self._post = (
            build_user_message, load_system, postprocess_generation)
        self.proc = AutoProcessor.from_pretrained(VLM_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(
            VLM_ID, dtype=torch.bfloat16, device_map="cuda")
        self.model.eval()
        print(f"[playground] {VLM_ID} on {GPU}", flush=True)

    @modal.method()
    def describe(self, image_b64: str, slot: str = "face",
                 max_new_tokens: int = 160) -> dict:
        """One picture, one slot, one clause — the arm our compiler cannot write.

        Bounded hard on purpose. `subject_definitions` wants a clause, not a
        paragraph: the guide's own is *"the young woman in <Picture 1>, with
        long dark hair, a blue cardigan, and a thin silver necklace"* — three
        features and a stop. A VLM told to describe a photograph will write two
        hundred words, and every one past the clause is filler by the rule the
        composer is built on.
        """
        import base64
        import io as _io
        import json as _json
        from pathlib import Path

        import torch
        from PIL import Image

        presets = _json.loads(
            (Path("/opt/promptor") / "vision_prompts.json").read_text())["image_prompts"]
        mode = SLOT_VISION.get(slot, "Subject / Identity")
        ask = presets.get(mode) or presets["Subject / Identity"]
        if not isinstance(ask, str):
            ask = _json.dumps(ask)

        img = Image.open(_io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text":
             f"{ask}\n\nAnswer as a single noun phrase of at most 25 words, in the "
             f"shape 'the young woman, with long dark hair and a blue cardigan'. "
             f"No sentence, no preamble, no full stop."}]}]
        ins = self.proc.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = self.model.generate(**ins, max_new_tokens=max_new_tokens)
        text = self.proc.decode(out[0][ins["input_ids"].shape[-1]:],
                                skip_special_tokens=True).strip()
        # A model asked not to write a sentence writes one anyway often enough
        # that this is a regex rather than a request — `_clean_rewrite`'s rule.
        text = text.strip().strip('"').rstrip(".").split("\n")[0].strip()
        return {"slot": slot, "mode": mode, "clause": text}

    @modal.method()
    def run(self, task: str, seconds: float, prompt: str,
            assets: list[str] | None = None, image_b64: str | None = None,
            temperature: float = 0.0, max_new_tokens: int = 1200) -> dict:
        import base64
        import io as _io

        import torch
        from PIL import Image

        user: list[dict] = []
        # The whole reason this arm exists. A picture goes in as a picture, not
        # as somebody's sentence about one.
        if image_b64:
            img = Image.open(_io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
            user.append({"type": "image", "image": img})
        user.append({"type": "text",
                     "text": self._envelope(task, seconds, prompt, assets or None)})
        messages = [{"role": "system",
                     "content": [{"type": "text", "text": self._system(task)}]},
                    {"role": "user", "content": user}]

        ins = self.proc.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to("cuda")
        kw = {"max_new_tokens": max_new_tokens}
        if temperature > 0:
            kw |= {"do_sample": True, "temperature": temperature, "top_k": 40}
        with torch.inference_mode():
            out = self.model.generate(**ins, **kw)
        raw = self.proc.decode(out[0][ins["input_ids"].shape[-1]:],
                               skip_special_tokens=True).strip()
        return {"raw": raw, "text": self._post(raw, task, float(seconds)),
                "task": task, "seconds": seconds, "saw_image": bool(image_b64)}


@app.function(image=image, timeout=900)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web():
    import secrets
    import sys

    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    api = FastAPI()

    # **A 500 reaches the page as the five words "Internal Server Error", and
    # the browser then fails to parse them as JSON** — so the symptom is a
    # SyntaxError about the letter I, and the actual exception is in a log
    # nobody is looking at. Every route answers with its own traceback tail
    # instead. Same rule as `_require_models()`: an error a person can hit
    # twice is an error that should have explained itself the first time.
    @api.exception_handler(Exception)
    async def anything(_request, exc: Exception):
        import traceback
        return JSONResponse(status_code=200, content={
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-1800:],
        })

    # Guarded, because this ran at module scope and a `FileNotFoundError` in it
    # did not fail *an arm* — it failed `web()` itself, so the container could
    # not construct and the bench was gone rather than degraded. One arm being
    # unavailable is a column that says why; it is not an outage.
    VIS = None
    VIS_ERR = ""
    try:
        sys.path.insert(0, "/opt/vis/tools")
        from _from_app import SHOT, pull

        VIS = pull(SHOT)
    except Exception as exc:  # noqa: BLE001 — any failure here is the same failure
        VIS_ERR = f"our compiler is unavailable: {type(exc).__name__}: {exc}"
        print(f"[playground] {VIS_ERR}", flush=True)

    def ours(scene: dict, task: str, seconds: float) -> dict:
        if VIS is None:
            return {"error": VIS_ERR}
        """What ships today, given the same intent as a scene.

        Its input is a cast and rows rather than prose, which is the asymmetry
        the whole comparison turns on: it cannot be handed a sentence, and the
        other two cannot be handed a slot. Each gets the shape it takes, and
        what is compared is the document at the end.
        """
        try:
            v = VIS["_validate_scene"](
                scene, n_refs=len(scene.get("_refs") or []) or 9,
                n_vids=0, n_auds=0, seconds=seconds)
        except (TypeError, ValueError) as exc:
            return {"error": str(exc)}
        if v is None:
            return {"text": "", "error": "not a live scene — nothing to compile"}
        return {"text": VIS["_compile_h3_prompt"](
            typed="", pills=[], task=task, seconds=seconds, scene=v)}

    @api.post("/api/run")
    async def run(payload: dict) -> JSONResponse:
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            return JSONResponse({"error": "A prompt is required."})
        task = str(payload.get("task") or "T2VA")
        try:
            seconds = float(payload.get("seconds") or 5)
        except (TypeError, ValueError):
            seconds = 5.0
        assets = [a for a in (payload.get("assets") or []) if str(a).strip()]
        try:
            temp = float(payload.get("temperature") or 0)
        except (TypeError, ValueError):
            temp = 0.0
        # `.aio` rather than `.remote`, because this handler is async and a
        # blocking Modal call inside one blocks the event loop for the whole
        # generation — which on a cold GPU container is half a minute of the
        # page being unable to answer anything else.
        return JSONResponse(await Enhancer().run.remote.aio(
            task=task, seconds=seconds, prompt=prompt,
            assets=assets, temperature=temp))

    @api.post("/api/describe")
    async def describe(payload: dict) -> JSONResponse:
        img = payload.get("image")
        if not img:
            return JSONResponse({"error": "An image is required."})
        return JSONResponse(await Vision().describe.remote.aio(
            image_b64=str(img), slot=str(payload.get("slot") or "face")))

    @api.post("/api/ab")
    async def ab(payload: dict) -> JSONResponse:
        """Every arm on one intent, **assigned to columns at random.**

        The randomisation is not decoration. `judge_renders.py` hides which
        prompt made which picture for the reason this needs it: knowing the
        left one is ours is enough to decide it reads better.
        """
        prompt = str(payload.get("prompt") or "").strip()
        scene = payload.get("scene") or None
        if not prompt and not scene:
            return JSONResponse({"error": "A prompt or a scene is required."})
        task = str(payload.get("task") or "T2VA")
        try:
            seconds = float(payload.get("seconds") or 5)
        except (TypeError, ValueError):
            seconds = 5.0
        temp = float(payload.get("temperature") or 0)
        want = payload.get("arms") or ["compiler", "finetune", "vision"]

        # Spawned together. Two of these are cold GPU containers and running
        # them in sequence doubles a wait somebody is going to sit through.
        #
        # `.spawn.aio` / `.remote.aio` / `FunctionCall.get.aio` — verified
        # against modal 1.5.3 rather than inferred. `.aio` hangs off `remote`
        # and `spawn`, **not off the method**: `run.aio(...)` is an
        # AttributeError, `run.remote.aio(...)` is the call. The warning says
        # `await ...remote.aio(...)` and the dots in the middle are load-bearing.
        pending, out = {}, {}
        if "finetune" in want and prompt:
            pending["fine-tune 2.6B"] = await Enhancer().run.spawn.aio(
                task=task, seconds=seconds, prompt=prompt, assets=None,
                temperature=temp)
        if "vision" in want and prompt:
            pending["Qwen3-VL-4B + rules"] = await Vision().run.spawn.aio(
                task=task, seconds=seconds, prompt=prompt, assets=None,
                image_b64=payload.get("image") or None, temperature=temp)
        if "compiler" in want and scene:
            out["our compiler"] = ours(scene, task, seconds)
        for name, call in pending.items():
            out[name] = await call.get.aio()

        names = list(out)
        # Fisher-Yates over the arm names, so no column is anybody's home.
        for i in range(len(names) - 1, 0, -1):
            j = secrets.randbelow(i + 1)
            names[i], names[j] = names[j], names[i]
        return JSONResponse({
            "cols": [{"text": out[n].get("text", ""),
                      "error": out[n].get("error")} for n in names],
            "which": names,
        })

    @api.get("/")
    async def page() -> HTMLResponse:
        # **Read from a file, not held in a Python string.** It lived in a
        # `PAGE = """…"""` literal, and a patch that wrote `\n` inside it
        # produced a *real* newline in the JS rather than an escape — one
        # unterminated string, the whole script fails to parse, and every
        # handler including Generate silently never binds. The symptom is a
        # button that does nothing, which points at the backend.
        #
        # As a file it is just HTML: no double-escaping layer, and
        # `node --check` can read it.
        return HTMLResponse(Path("/opt/vis/prompt_playground.html").read_text())

    return api
