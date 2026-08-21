# One encoder, not two

The rewrite moves off its own copy of Qwen3-VL-4B and onto the one ComfyUI is
already holding to condition renders. Design note and spec, 2026-08-21.

## The claim this replaces

`CLAUDE.md` and `comfy_nodes/visionary_rewrite/__init__.py` both say the same
thing, at length, and it was true when it was written:

> ComfyUI holds the encoder as a conditioning provider: `comfy/text_encoders/krea2.py`
> taps twelve raw hidden layers and concatenates them, with no LM head wired and
> no KV cache, because none of that is needed to produce conditioning. Driving
> generation on that object means reaching past ComfyUI's public surface at a
> pinned SHA and writing a decode loop by hand.

At `COMFY_SHA = 16e3f3034f2bba1fff6c70cbd759339778555cd6` the decode loop is
already there, it is first-party, and it is on the public surface. Read rather
than assumed — every line below is from that commit:

- `comfy/text_encoders/llama.py`, `class BaseGenerate`: a pre-allocated KV cache
  (`init_kv_cache`), a generation loop, and a sampler carrying temperature,
  top-k, top-p, min-p, repetition penalty, presence penalty, a seed and stop
  tokens.
- `class BaseQwen3.logits` in the same file: `if self.model.config.lm_head:
  return self.model.lm_head(input)` and otherwise a linear against
  `embed_tokens.weight`. The tied-embedding case is handled upstream, by name.
  `Qwen3VL_4BConfig` is annotated `lm_head: bool = False  # 4B ties word embeddings`
  — which is exactly the fact our node's `_load()` discovered independently and
  patches into the state dict by hand.
- `comfy/text_encoders/qwen3vl.py`: `class Qwen3VL(BaseLlama, BaseQwen3,
  BaseGenerate, torch.nn.Module)`, and `Qwen3VLClipModel.generate(tokens, ...)`
  which tokenises, builds MRoPE position ids and DeepStack visual features, and
  calls the loop.
- `comfy/text_encoders/krea2.py`: `Krea2Qwen3VLClipModel(Qwen3VLClipModel)` —
  so the Krea 2 encoder inherits `generate` unchanged.
- `comfy/sd.py`: `CLIP.generate(...)` and `CLIP.decode(...)`, both public, the
  first going through `load_models_gpu` like every other model in the process.
- `comfy_extras/nodes_textgen.py`: a `TextGenerate` node doing precisely this,
  plus `TextGenerateLTX2Prompt`, an LTX-2 prompt enhancer built on it. Upstream
  shipped the feature this note is about.

So the trade the old comment described — "a pinned-fragile dependency and our
own sampler, traded for memory nobody is short of" — is not the trade on offer
any more. There is no sampler to write. What is left is a dependency on three
upstream names, which is a real cost and is stated in **Risks** below rather
than waved past.

## What this buys, stated exactly

It does **not** make Enhance faster. Same weights, same bf16, same greedy
decode: expect the same 2-9s warm. Anyone reading this expecting a latency win
should read "Out of scope" instead, where the thing that would actually move
perceived latency is named.

What it buys:

- **~8.9 GiB on the image container.** The second copy goes; the shared one was
  already resident because every render goes through it.
- **An invisible allocation becomes a managed one, on both containers.** The
  node's own comment admits the cost today: 9 GiB subtracted from what ComfyUI
  believes it has for the life of the container, which `unload_all_models()`
  cannot see and `/free` will not drop. That is the same class of fact as the
  regional node's stranded LoRAs. After this, `_reclaim()` reaches it.
- **A warm-up that heats the copy the renders use.** Today `_warm_rewrite`
  spends ~8 GB of network-volume read plus a ~40s CPU construction warming a
  model no render will ever touch, while the render's own encoder is still cold.
  After this the same knock loads the shared `CLIP`, so the first render is
  warmer too.
- **~200 lines and a `transformers` dependency deleted** from the node, and with
  them the three failure modes its comments exist to describe: the meta-device
  trap, the non-persistent-buffer trap, and the hand-grafted `lm_head`.

## The node

`comfy_nodes/visionary_rewrite/__init__.py` keeps its name — so `require_nodes`,
`_rewrite_backend` and both `rewrite()` methods keep working — and keeps its
four load-bearing properties: `OUTPUT_NODE` with a `ui` text payload (the one
channel that reaches `/history/{prompt_id}`, which `run_text` already polls),
`IS_CHANGED` returning NaN, the `image_b64` string input, and `warm_only`.

It loses `_load()`, `_LOCK`, `_READY`, `MODEL_FILE`, `BASE_REPO`, the
`AutoConfig`/`AutoTokenizer`/`AutoProcessor` triple, the `lm_head` graft and the
CPU-with-real-storage construction. ~259 lines to ~70.

It gains one required input, `"clip": ("CLIP",)`, and `run()` becomes:

1. Compose the chat string (see below) and, for the vision path, decode the
   base64 into a ComfyUI `IMAGE` tensor.
2. `clip.tokenize(text, image=img, skip_template=True, min_length=1)` — the
   kwargs `comfy_extras/nodes_textgen.py:TextGenerate.execute` passes.
3. `clip.generate(tokens, do_sample=False, max_length=int(max_tokens))`.
4. `clip.decode(ids)`, returned as `{"ui": {"text": [said]}, "result": (said,)}`.

`warm_only` runs the whole path with `max_length=1`. There is no `_load()` left
to call on its own, and one token costs less than a second code path that exists
only to avoid it — the point of the knock is that ComfyUI's loader runs, not
that nothing is sampled.

Cleaning stays in `app.py`. `_clean_rewrite` is the one implementation and it is
already what every backend's output goes through.

### Why not use `TextGenerate` directly

It is the reference implementation and not the shipping one, for two reasons
that are both about this app rather than about that node:

- It is not an `OUTPUT_NODE`. It returns `io.String.Output`, so nothing lands in
  `entry["outputs"]` and `run_text` — which reads `out.get("text")` — would find
  the graph completed and empty. A terminal node publishing to `ui` is needed
  either way, and once one exists it may as well be the whole node.
- It takes an `IMAGE` socket. The motion path holds base64, because that is what
  the route carries; an `IMAGE` socket means staging a file and a `LoadImage`,
  which is a second transport for a picture that already rides the POST.

### The chat template

Composed by hand and passed with `skip_template=True`:

```
<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{prose}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
```

Composing it rather than asking the tokenizer for it is forced, not preferred:
**no template on the Krea 2 path has a system turn.**
`Qwen3VLTokenizer.llama_template` is `<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n`
and `KREA2_TEMPLATE` is the fixed descriptor prompt the DiT conditions on. The
instruction has nowhere to go through either of them.

The think block is the second reason and would be enough on its own.
`Krea2Tokenizer.tokenize_with_weights` defaults `thinking=True` where its parent
defaults to `False`, and for Qwen3 that means *not* appending the empty block —
right for conditioning, where Krea 2 wants the no-think template, and wrong
here, where the block is what suppresses reasoning.

`skip_template` is the documented way past both, and `Qwen3VLTokenizer` sets it
automatically for any text starting with `<|im_start|>` anyway.

For the vision path the user turn opens with
`<|vision_start|><|image_pad|><|vision_end|>` and the image rides as
`clip.tokenize(..., image=tensor)`; `Qwen3VLTokenizer.tokenize_with_weights`
replaces the pad token with the image payload and
`Qwen3VL.preprocess_embed` runs the tower. The tensor is ComfyUI's `IMAGE`
layout — `(1, H, W, 3)`, float, 0-1 — built base64 → PIL → numpy.

The node caps the long side. The page already shrinks a frame to 1536px and
`/api/motion` refuses a payload past 8 MB, so this is the third bound and the
one that binds, which is `_fit_reference`'s rule: the browser's cap is an
optimisation and the server's is the contract.

## The graph, and why the loader line is copied exactly

```python
{"clip": {"class_type": "CLIPLoader",
          "inputs": {"clip_name": te, "type": "krea2", "device": "default"}},
 "rw":   {"class_type": "VisionaryRewrite",
          "inputs": {"clip": ["clip", 0], "prose": prose,
                     "instruction": instruction, "max_tokens": max_tokens,
                     "image_b64": image_b64}}}
```

Those three loader inputs are byte-identical to the ones `_krea2_graph` emits.
That is the whole mechanism: ComfyUI keys its execution cache on the input
signature, so the rewrite graph and the render graph resolve to the same cached
`CLIP` object and the same `ModelPatcher`. Drift in any of the three — a
different `device`, a `clip_name` spelled from somewhere else — silently makes
it two copies again, with no error and no log line, which is the failure this
paragraph exists to make findable.

It is built by one `_rewrite_graph()` in `app.py` with two callers, rather than
written out twice in `ImageGenerator.rewrite` and `VideoGenerator.rewrite` as it
is today. One builder is what makes it reachable from `smoke_graphs.py`.

`extra_model_paths.yaml` already maps `text_encoders: models/` and both
containers share `_Comfy`, so the file resolves on both with no build change.

## The video container, and the encoder that is there

An earlier draft of this note said the video container "has no Qwen3-VL to
share." That is wrong and worth correcting in place, because the true version is
more interesting than the false one. H3's text encoder **is** Qwen3-VL — the 32B
(`h3_te`), against Krea 2's 4B. What it is not is a model that can generate, and
the reasons are worth writing down because they are the reasons the LTX
comparison does not carry.

LTX-2's enhancer is cheap because LTX-2 loads `Gemma3ForConditionalGeneration` —
the complete causal LM, `lm_head` included. That is why upstream's
`TextGenerateLTX2Prompt` is a system prompt and nothing else. H3's checkpoint is
the opposite end of the same axis. From ComfyUI's own `Qwen3VL_32BConfig`,
comment included:

```python
# MiniMax H3 conditioning checkpoint: truncated to the first 50 of 64 layers,
# consumed as the unnormalized hidden state after layer 50 (no final norm, no lm_head)
num_hidden_layers: int = 50
lm_head: bool = False
final_norm: bool = False
```

Fourteen layers are not in the file, the final norm is not in the file, and
neither is a head. And unlike the 4B there is no free head to reconstruct:
checked against the published configs, Qwen3-VL-4B is `tie_word_embeddings:
true` while the 8B and 32B are both `false`. So `BaseQwen3.logits` falling back
to `embed_tokens.weight` — correct for the 4B — would on the 32B be an input
embedding matrix used as an output projection, which yields garbage rather than
an error. On top of that the file is nvfp4-AWQ.

So for **Phase 1** the video container loads Krea 2's 4B purely as a rewriter,
which is what it does today and what `VideoGenerator.rewrite`'s docstring
already says. What changes is that the copy becomes ComfyUI-managed rather than
invisible, so `/free` reaches it and `_reclaim()` can drop it on an OOM.

Which means **one node, one graph shape, both containers**. The alternative —
keep the loading node for video and add a sharing node for images — is two code
paths for one feature, and the second one would only ever run on the container
where nothing is shared.

Arithmetic, so nobody has to re-derive it: H3 is 42.5 GiB and the encoder 8.9,
against 80 GiB. Krea 2 is ~24 and the encoder 8.9. Neither is near the line, so
`load_models_gpu` has nothing to evict to make room.

## Phase 2: H3's own encoder, and the generation tail

Not out of scope — sequenced after, and recorded here so it is a decision rather
than an oversight.

`ethanfel/ComfyUI-MiniMax-H3-Guide` solves exactly the problem above with a
**Generation Tail Loader**. Its `hybrid_tail.py` docstring: *"The connected
standard MiniMax H3 CLIP owns the embedding, vision tower, and language layers
0..49. This module loads only layers 50..63, the final norm, and the LM head
while text is generated."* The tail is a separate weight file and is released
after decoding.

**Which file, and why it is not the obvious one.** `generate_with_tail`'s
`logits()` falls through to `self.model.lm_head(...)` only when the head weight
carries no `_qdata`/`scale`; any quantized layout reaches `_validate_int8_head`,
which raises on anything but `TensorWiseINT8Layout`. So the 5.40 GB nvfp4 tail
in that repo is rejected by that repo's own loader, and the one to take is the
**7.61 GB int8 ConvRot** tail. Our nvfp4 body underneath it is fine: layers
0–49 and 50–63 are separate modules exchanging hidden states, so the two
quantizations do not have to agree.

**Why it is worth doing at all**, given that it costs a fourth pinned SHA, a
7.61 GB catalogue entry and a per-press load: a 32B writing H3's prompts is a
better model than a 4B, and it is the encoder that will read the result. That is
this node's own founding argument — *a rewriter which is the encoder writes in
the dialect the encoder reads* — applied where it has the most force. It is also
the only claim here that cannot be settled by reading, which is why Phase 2
opens as a spike rather than as an implementation.

**Why it is sequenced second rather than folded in.** Three unknowns, none
measured: whether the pack imports headless, whether an nvfp4 body plus an int8
tail actually decodes, and whether the result writes better prose than the 4B in
bf16. `tools/prompt_ab.py` answers the third and is the gate. Phase 1 does not
depend on any of them, and if Phase 2 wins it changes two things and no more:
the video graph names a different loader, and the node gains one optional
`clip_tail` input that routes generation through `generate_with_tail`. That is
why Phase 1 does not need to anticipate it — an optional input is additive, and
building a hook for a phase that may not land is the speculation this file
argues against everywhere else.

**Pinned, not vendored** — the `CLIFF_SHA` pattern, for the reason that entry
gives: nothing in it is patched, so there is no `VENDOR.md` to keep in sync.

**On the per-press load.** It is real — 7.61 GB off the volume every time
Enhance is pressed on an H3 session — and it was accepted deliberately rather
than overlooked. The workflow is the argument: image prompts are enhanced while
brainstorming, where a wait lands between keystrokes, but video is worked in
takes and a session may enhance once in five renders. A load amortised over
minutes of sampling is not the same cost as one felt mid-thought.

**Wan gets nothing, and that is accepted.** `UMT5EncoderModel` is an encoder
stack with no decoder; there is no tail that makes it generate. Wan sessions
keep Enhance in Phase 1 because the 4B is still the video rewriter there. If
Phase 2 lands before Wan is retired, Wan loses Enhance, and the honest form of
that is the `VIDEO_MODELS[...]["supports"]` gate the composer already uses for
audio and negative prompts — a control the model will not honour should be
absent, not present and ignored.

## Why the render/rewrite handoff does not stall

The constraint this is built against: flipping between generating a picture and
rewriting a prompt must not cost a reload. Two mechanisms could make it, and
both were checked at `COMFY_SHA` rather than assumed:

1. **Execution-cache eviction.** The old classic cache called `clean_unused()`
   after every prompt, dropping any node not in the current graph — so an
   intervening rewrite would evict the render's loaders and the next render
   would re-read 35 GB off the volume. At this SHA `comfy/cli_args.py` makes
   `--cache-ram` the default and `comfy_execution/caching.py` implements it as
   `RAMPressureCache(LRUCache)`, which keeps results across prompts until system
   RAM headroom forces an eviction. Alternating graphs is the case it exists for.
2. **VRAM eviction.** `CLIP.generate` calls `load_models_gpu([patcher],
   memory_required=...)`, which frees other models only when free VRAM is short.
   See the arithmetic above.

Both are readings of source, not measurements, and the plan treats them that
way — see the table below.

## What else changes

- `comfy_image` drops the `AutoConfig`/`AutoTokenizer`/`AutoProcessor` bake. The
  node no longer imports `transformers` at all. `trainer_image` keeps its own
  bake; `krea2_encoder` still wants a tokenizer by repo id.
- `smoke_graphs.py` gains the rewrite graph as a variant, which is what makes a
  `VisionaryRewrite` that fails to import, or a `CLIP` input renamed upstream, a
  CPU-container failure in a minute rather than a warm-H100 failure at the first
  press.
- `smoke_rewrite.py`'s docstring paragraph — "`visionary_rewrite` loads the
  encoder against a pinned config — this is the check that the two have not
  drifted apart" — becomes false and is rewritten. The instruction to run it
  after a `COMFY_SHA` bump survives with a *different* reason: the rewrite now
  depends on `BaseGenerate`, `BaseQwen3.logits` and `CLIP.generate` being where
  they are.
- `CLAUDE.md`'s "A second copy in the same container, not the resident instance"
  section is rewritten in this file's own shape — the claim, then what retired
  it. The paragraphs on the stripped head and the meta-device trap go with the
  code they describe; the paragraph on `PARSE_GPU`'s concurrency objection
  stands unchanged.

## Risks, and the check for each

Every row is a thing that could be wrong, not a thing believed to be fine.

| Risk | Why it could break | Check |
|---|---|---|
| Cache does not survive alternation | `RAMPressureCache` evicts under system-RAM pressure and the container is `cpu=4.0` | render → rewrite → render on a warm container, and grep ComfyUI's stdout for a second text-encoder load line |
| Decode hits SageAttention | `--use-sage-attention` is process-wide, and sage asserts `mask is None` | `Llama2_.forward` picks `optimized_attention_for_device(device, mask=mask is not None, small_input=True)`; at `seq_len == 1` there is no mask and `small_input` is set, so it should be SDPA — assert it rather than trust the read |
| `<think>` does not tokenise | ComfyUI bundles a `qwen25_tokenizer`; `<think>` is a Qwen3 addition | round-trip `clip.tokenize` → `clip.decode` on the composed template and compare |
| The model sees different tokens | ComfyUI bundles its own `qwen25_tokenizer`; HF's is a different file | tokenize the composed string both ways, compare id lists — exact |
| The model is structurally wrong | wrong head, wrong layer count, mis-assembled template | first generated token must agree across the two paths |
| Fluent garbage | a wrong head produces confident nonsense that passes every other check | growth ratio inside the measured band, non-empty after `_clean_rewrite`, not caught by `_looks_like_refusal` |
| `COMFY_SHA` bump breaks the rewrite | three upstream names are now load-bearing where the old node depended on none | `smoke_rewrite.py` and `smoke_graphs.py`, both named in the bump checklist |

**Token-id parity is the gate, and prose equality is deliberately not asserted.**
An earlier draft of this note required byte-identical output on the grounds that
same weights plus greedy decode is deterministic. It is — given identical
logits, which these two paths will not produce. A different attention kernel, a
preallocated KV cache written in place against HF's dynamic one, a different
order of operations: all mathematically equivalent, none bitwise equal in bf16.
Somewhere in a 400-token decode two candidates land within a few ULPs, argmax
flips, and everything after diverges. That is arithmetic, not a defect, and a
gate that fails on it would send whoever runs it chasing numerics.

It is also the trap this repo has already paid for once. A byte-diff over prose
is a lexical check standing in for a judgement — the same shape as the coverage
and invention thresholds, and as the anchoring score `smoke_parse.py` keeps as
`anchor_sweep` so the next person can watch it fail rather than reinvent it.

So exactness is required where exactness exists. The token ids are deterministic
and comparable, and if the model sees the same ids the interesting risk is gone.
The first generated token is a structural smoke signal, cheap and near-certain
under agreement. The growth band is arithmetic over a number this file already
measured. Whether the prose is *better* is read, and if it is in doubt it goes
to `tools/prompt_ab.py` — blind, both orders, old path against new. Judgement in
the harness, arithmetic in the gate.

## Out of scope

- **Token streaming into the prompt box.** `BaseGenerate` runs
  `tqdm(range(max_length), desc="Generating tokens")`, which `_Comfy._drain`
  already has the regex for, so a progress hairline under the button is cheap;
  a real stream needs a Modal generator and SSE. This is the change that would
  move *perceived* latency, since weight-sharing moves memory rather than
  seconds — and it is a separate one.
- The `"interpreter"` arm and the `Interpreter` class. The seam stays; the
  point of a one-line backend switch is that it survives changes to the arm it
  is not selecting.
- `REWRITE_OPS`, `KREA_EXPANSION`, `_clean_rewrite`, `_rewrite_tokens`, and
  every route. The prose in and the prose out are unchanged, which is what
  makes the drift check meaningful.
- Sampling parameters. `do_sample=False` stays, for the reason the current node
  gives: pressing the button again is not how you ask for a variation.
