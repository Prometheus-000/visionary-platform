# Reusing a DiT's LLM/VLM Text Encoder as a Prompt-Rewriter LLM: Feasibility, Mechanics, and VRAM Economics

## TL;DR
- **Yes, it is technically possible and — in the case where you already pay the memory tax for a bf16 encoder — usually worth it**, but viability hinges entirely on *what class the pipeline loads*: pipelines that load the full `...ForConditionalGeneration` (Qwen-Image → `Qwen2_5_VLForConditionalGeneration`; LTX-2 → `Gemma3ForConditionalGeneration`) already have the `lm_head` resident and can call `.generate()` for free, whereas pipelines that load only the backbone (Krea 2 → `Qwen3VLModel`; HunyuanVideo → `LlamaModel`) do not — and Wan (umT5-XXL `UMT5EncoderModel`) *cannot generate at all*.
- **For Krea 2 specifically: it uses Qwen3-VL-4B-Instruct (not 8B), the 4B checkpoint has `tie_word_embeddings: true`, so the "missing" lm_head is free** — it is just `model.embed_tokens.weight` reused as the output projection, at zero weight cost. The community has *already built this*: `BennyDaBall/Krea-2-Engineer-V1` is a DoRA fine-tune of the exact Qwen3-VL encoder shipped in `krea/Krea-2-Turbo`, explicitly documented as "a real Qwen3-VL model, [so] it also runs as a chat / prompt-writer LLM," and LTX-2's official ComfyUI pack ships a Gemma-3 prompt enhancer that reuses the LTX text encoder.
- **But for your Modal scale-to-zero deployment, weight-sharing is the wrong optimization.** The dominant lever is temporal separation + Modal GPU memory snapshots, not sharing one module across two jobs. Run the rewriter *before* the DiT is resident (or as a tiny separate quantized Qwen3-4B GGUF at ~2.5 GB), and let Modal's snapshot/restore skip the weight-load cold start entirely. Weight-sharing saves at most the size of one encoder and only helps if both models would otherwise be co-resident — which, with proper sequential offload, they never are.

## Key Findings

**1. What class each pipeline instantiates for `text_encoder` (this determines everything):**

| Pipeline | `text_encoder` class (diffusers) | Has `lm_head`? | Can `.generate()` as-loaded? |
|---|---|---|---|
| **Krea 2** (`Krea2Pipeline`) | `Qwen3VLModel` (Qwen3-VL-4B-Instruct backbone) | No separate tensor, but **tied** → free | No as-loaded; trivially yes if wrapped in `Qwen3VLForConditionalGeneration` |
| **Qwen-Image / Qwen-Image-Edit** | `Qwen2_5_VLForConditionalGeneration` (7B/8.3B) | **Yes** (full model) | **Yes, immediately** |
| **HunyuanVideo** | `LlamaModel` (Llava-Llama-3-8B backbone) + `CLIPTextModel` | No | No as-loaded; needs `LlamaForCausalLM` |
| **HunyuanVideo-1.5** | `Qwen2_5_VLTextModel` (+ ByT5 `T5EncoderModel`) | No (text-only submodel) | No as-loaded |
| **LTX-2 / LTX-2.5** | `Gemma3ForConditionalGeneration` | **Yes** (full model) | **Yes, immediately** |
| **Wan 2.1 / 2.2** | `UMT5EncoderModel` (umT5-XXL) | N/A (encoder-only T5) | **Never** — it is an encoder, not decoder-only |

**2. Krea 2 is Qwen3-VL-4B-Instruct, tied embeddings — the lm_head is free.** The 4B `text_config` has `hidden_size: 2560`, `vocab_size: 151936`, `num_hidden_layers: 36`, `num_key_value_heads: 8`, and `tie_word_embeddings: true` (confirmed against the sibling Qwen3-4B-Instruct config.json: `"hidden_size": 2560, "num_hidden_layers": 36, "vocab_size": 151936, "tie_word_embeddings": true, "rope_theta": 5000000, "max_position_embeddings": 262144`). Because embeddings are tied, `lm_head.weight` *is* `model.embed_tokens.weight` — reconstructable at zero extra VRAM. The diffusers `text_encoder/model.safetensors` ships as a single ~8.88 GB bf16 file (≈4.44B params, which includes the Qwen3-VL vision tower — this is why ComfyUI vision-reference nodes work). The pipeline calls the encoder with `output_hidden_states=True` and taps **12 decoder layers (every 3rd: indices [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35])**, stacking them into a per-token conditioning of shape `(batch, seq, 12, 2560)` = **12 × 2560 = 30720 features** consumed by a `txtfusion` adapter in the DiT. (Confirmed by the `BennyDaBall/Krea-2-Engineer` GGUF card: ComfyUI's krea2 CLIP type "needs the full Qwen3-VL-4B (12-layer hidden-state stack + vision tower)... Loading it as CLIP fails at the sampler with 'Krea2 expects conditioning with 12x2560=30720 features ... but got 2560.'") This multi-layer tapping does **not** change what must be loaded — the backbone already produces all hidden states in one forward pass.

**3. If you *do* need to add an untied lm_head (only relevant if you used Qwen3-VL-8B):** Qwen3-VL-8B-Instruct has `hidden_size: 4096`, `vocab_size: 151936`, `num_hidden_layers: 36`, `tie_word_embeddings: false` (byte-confirmed from the model's config.json; note the known quirk that the field sits at root but is missing from `text_config`, which throws a vLLM AttributeError). An untied head is 151936 × 4096 = **622.3M params ≈ 1.16 GiB in bf16**. For the tied 4B (Krea's actual choice) this cost is **zero**.

**4. This reuse pattern already exists in the wild.** Confirmed real implementations:
- **`BennyDaBall/Krea-2-Engineer-V1`** — a DoRA fine-tune of "the Qwen3-VL text encoder that ships inside `krea/Krea-2-Turbo`, trained on the Z-Image-Engineer V7 corpus... Because the encoder is a real Qwen3-VL model, it also runs as a chat / prompt-writer LLM." Its GGUF companion is "the fine-tuned text tower only, repacked as a plain Qwen3 causal LM (general.architecture: qwen3, no vision tensors)."
- **LTX-2 official ComfyUI enhancer (`LTXVGemmaEnhancePrompt`)** — "uses the same Gemma 3 model that already sits in your LTX-2 workflow as the text encoder"; community caveat: "Running Gemma to enhance on top of running it to encode means the 22GB model is doing double duty."
- **`FNGarvin/Qwen2.5-VL-Clip-Loader-Prompt-Helper`** — a ComfyUI node whose stated purpose is "prompt expansion using the already-loaded text-encoder" for Qwen-Image, with an explicit "Dual-Use Architecture: Uses the same VRAM-loaded model for both high-level reasoning (Chat/Description) and low-level embedding generation."
- Unified understanding+generation research (Janus-Pro, BLIP3-o, Show-o2, MetaQuery, OmniGen2, BAGEL, Emu) is the academic version of the same idea, but those are purpose-built single backbones, not retrofits.

**5. VRAM economics favor temporal separation over weight-sharing** for a serverless single-GPU deployment.

## Details

### What is actually loaded (Question 1)
The decisive fact is the class in `model_index.json`. diffusers deliberately loads the *smallest sufficient* module:
- **Encoder-only pipelines** (Wan's umT5-XXL via `UMT5EncoderModel`) load a T5 *encoder stack* — there is no decoder and no `.generate()` path at all. Wan's own `generate.py` therefore ships a **completely separate** prompt-extend model, confirmed verbatim in the official Wan 2.2 README: "you can use models like Qwen/Qwen2.5-14B-Instruct, Qwen/Qwen2.5-7B-Instruct and Qwen/Qwen2.5-3B-Instruct. For image-to-video tasks, you can use models like Qwen/Qwen2.5-VL-7B-Instruct and Qwen/Qwen2.5-VL-3B-Instruct... modify the model used for extension with the parameter `--prompt_extend_model`." Reuse is impossible here: umT5 cannot rewrite prompts.
- **Backbone-only decoder pipelines** (Krea 2 → `Qwen3VLModel`; HunyuanVideo → `LlamaModel`) load the transformer body that outputs hidden states but instantiate no LM-head object. HunyuanVideo's `_get_llama_prompt_embeds` calls the encoder with `output_hidden_states=True` and takes `hidden_states[-(num_hidden_layers_to_skip+1)]` (default skip 2 → 3rd-from-last layer), after formatting with a template whose `crop_start` tokens are dropped (`llama_vec = llama_outputs.hidden_states[-3][:, crop_start:llama_attention_length]`). The weights for generation are physically present (a `LlamaModel`/`Qwen3VLModel` contains the full decoder stack); only the head class is absent. HunyuanVideo separately fine-tuned Hunyuan-Large as a *distinct* prompt-rewrite model (`tencent/HunyuanVideo-PromptRewrite`), i.e. Tencent chose *not* to reuse the encoder.
- **Full-model pipelines** (Qwen-Image → `Qwen2_5_VLForConditionalGeneration`; LTX-2 → `Gemma3ForConditionalGeneration`) load the entire causal-LM including `lm_head`. These are `.generate()`-ready with no extra weights.

For **Krea 2**, tied embeddings make the distinction moot: whether you load `Qwen3VLModel` or `Qwen3VLForConditionalGeneration`, the output projection is the same `embed_tokens` matrix. You can load `Qwen3VLForConditionalGeneration.from_pretrained(...)`, pass its inner `.model` (the `Qwen3VLModel`) as the pipeline's `text_encoder`, and keep the parent object for `.generate()` — same weights, two views, no duplication.

### Cost to add the head (Question 2)
- **Krea 2 / Qwen3-VL-4B (tied):** 0 extra params, 0 VRAM. Reuse `model.embed_tokens.weight`.
- **Qwen3-VL-8B (untied):** +622.3M params, +1.16 GiB bf16.
- **HunyuanVideo / Llama-3-8B:** Llama-3 is untied (vocab 128256 × hidden 4096 = 525.3M params ≈ 0.98 GiB bf16) — but the community text-encoder-only repackage (`Kijai/llava-llama-3-8b-text-encoder-tokenizer`) strips both the vision tower and the head, so you would re-add ~1 GB to generate.

### Practical mechanics (Question 3)
- **(a) `output_hidden_states=True` vs generation:** These are orthogonal. The encoder forward is a single full-sequence pass returning all hidden states; `.generate()` does incremental decoding. The *same weights* serve both; you just call different methods. You cannot do both in one call, but they are non-simultaneous by assumption.
- **(b) KV cache:** Generation allocates a KV cache; the encoder forward does not. For a 4B model rewriting a ~50→300-token prompt, the cache is a few hundred MB at most and is freed the moment `.generate()` returns. Call `torch.cuda.empty_cache()` before the denoise loop; because the rewrite finishes before denoising begins, the cache never coexists with peak DiT memory. (Note: Krea Realtime's *training/serving* KV-cache concerns — up to 25 GB per GPU — are specific to autoregressive *video* generation, not to a one-shot text rewrite; ignore that scale here.)
- **(c) Attention implementation:** The encoder path typically uses a full (causal, no-cache) SDPA/flash pass; generation uses incremental cached decoding. Both are supported by the same `attn_implementation` (SDPA or flash_attention_2) — no reconfiguration needed. For a batch rewrite call, plain SDPA is fine.
- **(d) Quantization degrades generation more than embedding extraction.** This is the biggest silent-quality trap. fp8/NVFP4 encoders (e.g. `qwen3vl_4b_fp8_scaled.safetensors`) are validated for *embedding extraction* feeding the DiT, but autoregressive decoding accumulates quantization error token-over-token; an fp8 rewriter can produce noticeably worse prose than the same model in bf16. If you reuse a quantized resident encoder for generation, expect degraded rewrites. A separate small bf16/Q5+ rewriter avoids this.
- **(e) Chat-template mismatch is critical.** The text-encoder path wraps the prompt in Krea's descriptor template (`<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>...`) and *strips the template prefix* so only prompt tokens condition the DiT (the diffusers offset analogous to `prompt_template_encode_start_idx` / HunyuanVideo's `crop_start`). The generation path needs a proper chat template with `add_generation_prompt=True`. These are different call sites; do not reuse the encoder's cropping logic for `.generate()`.
- **(f) Fine-tuning/adaptation risk.** Krea has not published that Krea 2's encoder was distilled away from instruct competence, and `Krea-2-Engineer-V1` demonstrates the stock encoder still functions as a capable Qwen3-VL LLM (it is a light DoRA fine-tune of that exact tower). Qwen-Image's encoder is a full instruct `Qwen2_5_VLForConditionalGeneration`. HunyuanVideo uses a competent `xtuner/llava-llama-3-8b`, but the diffusers repackage strips the head. Wan's umT5 is not an instruct model at all. Verdict: for Krea 2 and Qwen-Image the resident encoder is still a competent generator; always sanity-check output quality if the resident copy is quantized.

### Alternative VRAM-saving approaches (Question 5)
Approximate resident footprints (bf16 unless noted), single 80 GB-class or 24 GB GPU:
- **Krea 2 Turbo DiT:** 12B → ~24 GB bf16, ~12 GB fp8 (the ComfyUI FP8 diffusion model is reported at 12.01 GiB).
- **Krea 2 encoder (Qwen3-VL-4B):** ~8.9 GB bf16, ~4.5 GB fp8.
- **Wan 2.2 I2V (14B MoE, high+low-noise experts):** each expert 14B (~28 GB bf16); ComfyUI runs them fp8 (~14 GB each) and swaps.
- **umT5-XXL encoder:** ~11 GB bf16, ~5.5 GB fp8 (`umt5_xxl_fp8_e4m3fn_scaled.safetensors`).
- **A separate quantized rewriter (Qwen3-4B GGUF):** from the `Krea-2-Engineer-V1-GGUF` listing, F16 8.05 GB, Q8_0 ~4 GB ("near-lossless"), **Q4_K_M the recommended "sweet spot" (~2.5 GB)**, Q2_K 1.67 GB; Qwen3-1.7B ~1–1.5 GB; Qwen3-0.6B <1 GB.

Ranked for a Modal scale-to-zero single-GPU box running both Wan 2.2 I2V and Krea 2:
1. **Temporal separation (best).** Run the rewriter first, free it, then load the DiT. ComfyUI already does this for Krea 2 on 16 GB: it "loads and runs the Qwen3-VL text encoder to encode your prompt, then frees it before the diffusion sampling stage – so the encoder and the 12.01 GiB diffusion model are not both resident in VRAM at peak." Extend the same pattern to generation: rewrite → encode → free encoder → denoise. No weight-sharing needed.
2. **Modal GPU memory snapshots.** Per Modal's "GPU Memory Snapshots" post (July 30, 2025), Functions start "up to 10x faster than baseline" — a Parakeet/NeMo function drops from ~20 s (P0) to ~2 s, and a fully-loaded ViT from 8.5 s to 2.25 s; Modal's Mistral post reports Ministral-3B cold start falling ~118 s → ~12 s. An independent user measured a Qwen3-27B-FP8 vLLM cold start of 460 s cut ~6.5× via Modal snapshots. This attacks your actual pain (cold start = weight loading), which weight-sharing does not.
3. **Sequential/model CPU offload** (`enable_model_cpu_offload` moves whole modules on/off GPU around their forward call; `enable_sequential_cpu_offload` goes finer/slower). This already guarantees the encoder and DiT are not co-resident, erasing most of the benefit of weight-sharing. diffusers now also supports `device_map="cpu"` at pipeline init followed by `enable_model_cpu_offload()` for low-VRAM starts.
4. **Precompute/cache text embeddings** for repeated prompts — avoids re-running the encoder at all.
5. **Reuse the resident encoder for generation (the asked-about trick):** saves loading a second model (~2.5 GB for a small rewriter, or the full ~9 GB encoder if you'd otherwise duplicate it), but only helps if the encoder is *already* resident at rewrite time and you accept quantized-generation quality risk. On a scale-to-zero box, it saves less than snapshots + offload.

## Recommendations

**Stage 1 — Default architecture (do this):** Keep the rewriter and the DiT *temporally separated*, not weight-shared. On Modal: (a) load a small **separate** bf16 rewriter (Qwen3-4B-Instruct, or your existing Qwen3-VL-8B-Instruct captioning model if already warm) → generate the rich prompt → move it to CPU/free it; (b) load the Krea 2 / Wan 2.2 pipeline with `enable_model_cpu_offload()`; (c) enable **Modal GPU memory snapshots** so the weight-load cold start is skipped on restore. Benchmark cold-start p50/p90 and peak VRAM. This is the lowest-risk, highest-leverage path.

**Stage 2 — If (and only if) you are VRAM-bound at rewrite time and the encoder is already resident in bf16:** reuse it. Minimal Krea 2 recipe:
```python
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer
from diffusers import Krea2Pipeline
import torch

full = Qwen3VLForConditionalGeneration.from_pretrained(
    "krea/Krea-2-Turbo", subfolder="text_encoder", torch_dtype=torch.bfloat16
)
tok = AutoTokenizer.from_pretrained("krea/Krea-2-Turbo", subfolder="tokenizer")

# 1) rewrite (proper chat template, generation prompt):
msgs = [{"role": "system", "content": REWRITER_SYS},
        {"role": "user", "content": user_intent}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
out = full.generate(ids, max_new_tokens=300, temperature=0.7)
rich = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
torch.cuda.empty_cache()

# 2) hand the SAME weights to the pipeline as the backbone view:
pipe = Krea2Pipeline.from_pretrained(
    "krea/Krea-2-Turbo", text_encoder=full.model, torch_dtype=torch.bfloat16
).to("cuda")
img = pipe(rich, num_inference_steps=8, guidance_scale=0.0).images[0]
```
Because 4B embeddings are tied, `full.model` and `full` share the `embed_tokens`/head weights — no duplication. Let the pipeline apply Krea's descriptor template + prefix-crop for conditioning; use the plain chat template only for the rewrite.

**Do NOT** reuse a fp8/NVFP4 resident encoder for `.generate()` in production without an A/B quality check — quantized autoregressive decoding degrades prose. Prefer a small separate bf16/Q5+ rewriter if the resident copy is quantized.

**For Wan 2.2:** do not attempt reuse — umT5-XXL cannot generate. Use Wan's built-in `--prompt_extend_model` (a separate Qwen2.5-7B/14B or Qwen2.5-VL for I2V) or a small local GGUF rewriter, run before the DiT loads.

**Thresholds that change the recommendation:** If your GPU is large enough to hold rewriter + DiT + encoder simultaneously (e.g., ≥48 GB with fp8 DiT), skip all sharing — keep a dedicated bf16 rewriter warm. If cold start (not steady-state VRAM) is your only pain, snapshots alone solve it and no sharing is needed. Only if you are pinned to a small card (≤24 GB) *and* snapshots+offload still OOM should you reach for encoder reuse.

## Caveats
- The exact `architectures` string in Krea 2's gated `text_encoder/config.json` and the literal absence of an `lm_head.weight` tensor were inferred from the tied-4B lineage, the `Qwen3VLModel` pipeline class, and the single-file (no shard index) 8.88 GB layout — not byte-confirmed, because `krea/Krea-2-Raw`/`Turbo` are gated (a public 1:1 mirror, `ethanfel/Krea-2-Base-Diffusers`, shows the same `config.json` + single 8.88 GB `model.safetensors`). The behavior (tied head, backbone loads all hidden states) is certain regardless.
- The 8.88 GB encoder size implies the diffusers single-file **includes the Qwen3-VL vision tower**, even though T2I uses only the language path; ComfyUI vision-reference nodes exploit this, but it means you are not saving vision-tower VRAM by using the text path. The `Krea-2-Engineer` GGUF, by contrast, is text-tower-only.
- Some third-party stacks differ: ModelScope DiffSynth's `Krea2TextEncoder` wraps `Qwen3VLForConditionalGeneration` (full model), while HF diffusers uses `Qwen3VLModel` (backbone). Check your stack.
- "Krea 2" (the 12B MMDiT with Qwen3-VL-4B encoder, released June 22–23, 2026), "FLUX.1 Krea [dev]" (BFL/Krea collaboration, CLIP+T5 encoders — *not* an LLM encoder, so reuse is inapplicable), and "Krea Realtime 14B" (Self-Forcing distill of Wan 2.1 14B, umT5 encoder — reuse inapplicable) are three distinct models; only Krea 2 fits the reuse pattern.
- Quantized-generation quality, KV-cache timing, and chat-template correctness are the three silent-failure modes that will degrade output without throwing an error — validate each. A recurring community caution about all such enhancers (LTX, QwenVL) is that "this is an LLM, so it will confidently invent" details you didn't ask for — lower temperature and read the output before trusting it.