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

## The video container

It has no Qwen3-VL to share — H3 reads its own encoder and Wan reads umT5 — so
it loads Krea 2's purely as a rewriter. That is not a regression; it is what it
does today, and the docstring at `VideoGenerator.rewrite` already says so. What
changes is that the copy becomes ComfyUI-managed rather than invisible, so
`/free` reaches it and `_reclaim()` can drop it on an OOM.

Which means **one node, one graph shape, both containers**. The alternative —
keep the loading node for video and add a sharing node for images — is two code
paths for one feature, and the second one would only ever run on the container
where nothing is shared.

Arithmetic, so nobody has to re-derive it: H3 is 42.5 GiB and the encoder 8.9,
against 80 GiB. Krea 2 is ~24 and the encoder 8.9. Neither is near the line, so
`load_models_gpu` has nothing to evict to make room.

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
| Output drifts | different tokenizer, different sampler code, same weights | re-run `tools/rewrite-dump.jsonl`'s inputs through both paths and diff |
| `COMFY_SHA` bump breaks the rewrite | three upstream names are now load-bearing where the old node depended on none | `smoke_rewrite.py` and `smoke_graphs.py`, both named in the bump checklist |

The output-drift row is the gate. The same weights, greedy, at bf16 should mean
byte-identical prose; anything else is a bug worth finding before this lands
rather than a tolerance to accept.

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
