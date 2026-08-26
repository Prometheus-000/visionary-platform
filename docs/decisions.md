# Decisions and their receipts

Things that were built here and are no longer here, and things that were
considered and refused. Nothing in this file is loaded into a session — it is
the account you come back to when someone proposes rebuilding one of them.

Each entry keeps the measurement that ended it, because the measurement is the
part that does not have to be taken on trust. Where an entry still teaches a
live rule, that rule is stated in `.claude/rules/backend.md` or `web/CLAUDE.md`
and points back here.

---

## The semantic layer and the rewrite

*Live rules extracted from this: the system-prompt size budget, "a length is a
token cap", "meta-commentary is cut by a regex", and the judge/validator split.
See `.claude/rules/backend.md`.*

Two features lived in this space and both are gone. They are recorded together
because they failed the same way and the numbers cost real GPU time.

**The semantic layer** read the person's prose into a document of tagged
elements, marked which words were the model's, and compiled that to the prompt.
`/api/parse`, `PARSE_RULES`, a 4B on its own L4, a validator, an undo-aware
mirror painting provenance inline in the box. **The rewrite** replaced it with
one button and one instruction — Krea's own `docs/expansion.txt` — returning
prose rather than a document.

**The measurement that ended both.** Rendered blind, both orders, a win counted
only when both orders agree:

|  | beats bare | loses to bare | tie |
|---|---|---|---|
| the document layer, 30 comparisons | **0** | 4-10 | rest |
| the rewrite, 10 fragments | **3** | 1 | 6 |

The document layer never won a single pair, across two model sizes and two rule
sets. The rewrite did win — its three were the fragments the validator had been
refusing — and it was still not worth a button on every prompt, which is the
call the person made: *"you lose too much control"*, and *"it adds unnecessary
overhead and causes bugs because it interferes with the model's text encoders."*

Six things they established, each of which cost a measurement and none of which
depends on either feature existing:

- **A text metric cannot measure this.** Preserved, covered, round-tripped and
  idempotent all score a rewrite against the sentence it came from, so returning
  the sentence unchanged scores perfectly — which is exactly what the incumbent
  did: zero invented words across 27 fragments, read as maximum restraint, from
  a feature reaching 0% of renders. Eleven scored rows, every one a string
  comparison, not one asking whether the picture got better.
- **A threshold standing in for a question is the error to not repeat.** Two
  bounds were swept and both failed for one reason: a share of characters is
  dominated by how much the person typed. `empty diner, 3am` is 16 characters,
  so its budget was 28, and the prompts that made better pictures needed 332.
  Reading `night. no, late afternoon` *correctly* means dropping characters, so
  a correct reading scored 59% and was refused. A content-word variant was built
  to replace both and does not separate either — worst real 31%, worst evasion
  33%, because an evasion coincidentally shares a common word.
- **A system prompt has a size budget: 500–2000 characters.** `PARSE_RULES` grew
  from 2.9k to 10.2k one well-reasoned rule at a time and the output got worse,
  in two ways no check caught. It went lossy on a well-formed prompt, dropping
  the most distinctive thing in it. And it began **parroting its own examples**
  back as the answer — given three friends on a fire escape it returned phrases
  lifted verbatim from the instruction. So concrete examples came out with the
  wordcount, and that was not a coincidence: they *were* the parroting. No
  instruction is on a user-facing path any more — `_motion_instruction` was the
  last and went with the motion panel — so the budget is a fact about a class of
  thing rather than a check on a live string.
- **A length is a token cap, never an instruction.** "Between 60 and 100 words"
  produced 95, 122 and **617**. A model does not count.
- **Meta-commentary is cut by a regex, not asked away.** The model returned the
  prompt, then an arrow, then bullets explaining itself — helpfulness in the one
  place it is indistinguishable from failure, because the answer went into the
  box. An instruction not to preamble is a request.
- **The rewriter can be the encoder — and the bill for that is a shared queue.**
  Krea 2 reads its prompt through Qwen3-VL-4B, a decoder model already resident,
  so the rewrite ran warm in 2.2-9.2s on weights already paid for. Nobody else
  does this because Flux's encoder is T5-XXL and SD's is CLIP and both are
  encoder-only.

  **It is out of the tree now, and this entry is the record of why the trick was
  not worth its bill.** `comfy_nodes/visionary_rewrite`, `_rewrite_generator`
  and `/api/motion` are deleted; the video container loads one model again.

  **What it costs is the thing the ten-minute render was really about.** Riding
  the generator means riding its `@modal.concurrent(max_inputs=1)`, so a rewrite
  in flight is a `generate` that cannot even be *delivered* to the container —
  and on a cold one, that rewrite is itself queued behind ComfyUI's serial
  queue. Pressing Enhance therefore delayed a clip by the whole of its own
  cold start, and nothing anywhere said so. The person's own reading was that
  Enhance was to blame, and it was; the mechanism was not the one anybody would
  have guessed, and the payload theory that looked obvious was wrong — the same
  references were attached to the runs that were fast.

  So the load is **lazy now**, which inverts what this file used to say. Warming
  at `enter` was right while the rewrite was on every prompt; it posts a *graph*
  to do it, so every cold container spent 132 seconds refusing to render for a
  button that might never be pressed. One video-only panel is not worth that.
  The first press pays the load, a render never does, and the node's
  module-level `_READY` makes it once per container either way.

**What is left, and it is the useful half.** `tools/judge_prompts.py` marks a
rewrite against what the person said on four criteria — subject, tone, space,
fidelity — plus `lost` and `contradicted`, every verdict carrying a quote.
`tools/judge_renders.py` marks the pictures instead and is the only measurement
here that is not a proxy. `tools/serve_judge.py` opens a Sandbox for either.
The four criteria outlived their subject because they are about the *picture*,
and criterion 3 is what the next section is.

---

## The ten-minute render

*Live rule extracted from this: prefer the explanation with an unbounded shape
over the one with a computable ceiling, and anything that can take minutes says
which minutes they are, on both surfaces. See `.claude/rules/backend.md`.*

- **A silent wait is diagnosed by guessing, and the guess lands on whatever was
  added most recently.** This is the error rule above applied to the state that
  is not an error, and it cost more than any error here has. A render took ten
  minutes to start; the page said "generate, 0%" throughout; Stop did nothing;
  and the only way its owner learned it had begun was opening the Modal
  dashboard to kill the app. Their conclusion — reasonable, and wrong — was that
  the prompt rewrite was to blame, because it was the newest thing on that path.
  It was 48 MB of PNG references crossing the wire.

  **The counterpart failure is the investigator's, and it happened here too.**
  Handed a ten-minute gap, the diagnosis landed on 48 MB of PNG references —
  a real waste, fixed on its own merits, and *never a candidate*: parsing that
  body is 0.05s, decoding and writing all nine is 0.10s, and uploading it is
  0.4-15s depending on the connection. Sixteen seconds against four hundred and
  eighty, and none of it touches the GPU. The number was reached for because it
  was large, and never divided by a rate.

  So: **prefer the explanation with an unbounded shape over the one with a
  computable ceiling.** A payload has a ceiling — bytes over a rate, and it can
  be worked out in a minute. A queue does not: `VideoGenerator` is
  `max_containers=1` and `@modal.concurrent(max_inputs=1)`, so anything holding
  that slot delays everything behind it by however long it holds it, and a
  rewrite on a cold container holds it for its whole cold start. That is the
  shape a ten-minute wait has, and the arithmetic said so before any log did.

  **It was the queue.** The volume reload was the other unbounded candidate and
  is ruled out by the person who runs this — reloads do not take that long —
  which leaves the held input slot, and matches the observation that settled it:
  the runs that skipped Enhance were fast *with the same references attached*.
  The warm-up is lazy now so no render pays it, but the slot itself is inherent
  to riding the resident encoder, so `_note_queue_wait` measures the delivery
  gap rather than trying to prevent it.

  A feature was nearly retired for another feature's cost. **And the logs were
  as silent as the page, which is worse** — the logs are the escape hatch, so
  what somebody sees after giving up on the screen is ComfyUI's own output
  either side of an eight-minute hole with nothing of ours in it.

  **So anything that can take minutes says which minutes they are, on both
  surfaces.** The phase names the step ("reloading the volume", "staging 9
  attachments · 48 MB"). The log stamps `[api] spawned in Ns` when the route
  hands the job to Modal, `accepted` when the container is given it — the gap
  between those two is the hop a large body actually travels, and it could not
  be seen at all — then the unbounded steps by name. And the parts whose cost
  the person controls report the number they control.

---

## Wan 2.2, and what a second video family proved

*Live rules extracted from this: `VIDEO_MODELS` is served to the page so the
composer shows only controls the chosen model reads; an unmatched LoRA is
reported rather than assumed to have worked; `MiniMaxH3SigmaShift` is opt-in and
goes after the stack; `<Audio N>` is a sibling of the subject. See
`.claude/rules/backend.md`.*

**Wan 2.2 was here and is gone.** It shared the container, the warm ComfyUI
process, the job/status/stop contract and the output layout; what was
per-family was a graph builder and a row in `VIDEO_MODELS`. That is worth
recording as a *result* rather than a regret — it is the payoff of driving
ComfyUI instead of porting its model code, it was demonstrated rather than
argued, and it is the shape a third family should take.

Removing it cost about 700 lines and no contract. `requires` is still per task,
`supports` is still per model, and `run()` still dispatches nothing it does not
have to. That is the half worth keeping.

**What it cost while it was here** is the honest other half, and it is the
reason it went. Every question the console asked, it asked twice: a CFG box and
a flow shift and an expert-switch step that H3 reads none of, a negative prompt
on a guidance-distilled path, two tier tables, two frame grids, two compilers,
two graph builders, half the model catalogue, and a `needs` column on the shot
vocabulary whose only job was dimming audio pills for a silent model. The scene
composer landed on top of all of that and could only compile for one of them —
so the timeline was collected on both sides and flattened into prose on one,
with nothing on screen saying so. A second family that reads none of the first
one's grammar is not a second option, it is a second product.

Four facts about it are kept below, because each one cost a measurement and none
of them depends on Wan existing.

**The LoRA row said "no ecosystem for the int8 repackage" and was wrong on both
halves.** `LoraLoaderModelOnly` is architecture-agnostic weight patching — what
decides whether a file does anything is whether its keys map onto the DiT, not
whether ComfyUI knows the family — and MiniMax ship three Lightning
distillations in the same repo the H3 weights already come from: 8-step and
4-step for the fl2va transformer, 4-step for ref2va. They are catalogue entries
now, so they get the download UI and the picker for free.

Two things learned settling that, both of the shape "the obvious check answers
a narrower question than the one asked":

- **`turbo_mode` is real and is not a node input.** Grepping
  `comfy_extras/nodes_minimax_h3.py` for it returns nothing at our pin *or* at
  master, which reads as proof it does not exist. The t2v and i2v templates
  wrap the whole graph in a **subgraph** and promote `turbo_mode`,
  `turbo_model_strength` and `turbo_steps` as widgets on it;
  `video_minimax_h3_r2v.json` is not a subgraph and shows the parts unwrapped —
  a `PrimitiveBoolean` driving two `ComfySwitchNode`s that choose between the
  bare DiT and `LoraLoaderModelOnly(DiT)`, and between two step counts. A
  template is a fixed graph and needs a switch to turn a node off. This console
  has a LoRA picker and a steps field, so it needs neither.
- **`MiniMaxH3SigmaShift` exists and we had been right to omit it.** Its
  defaults (12.0 video, 3.0 audio) are the model's own —
  `MiniMaxH3.forward` reads
  `transformer_options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video)`
  against `sigma_shift_video=12.0, sigma_shift_audio=3.0`, and
  `supported_models.MiniMaxH3.sampling_settings` says `shift: 12.0` besides. So
  the node at rest is a no-op. It stops being one under a distilled LoRA, which
  is the only reason it is now in the graph at all — **opt-in, and after the
  stack**, because the shift is the last word on the sampling curve. The trap
  next to it: `/api/video`'s shared `shift` key falls back to
  `WAN_DEFAULT_SHIFT`, so reading *that* would have put 8.0 on every H3 take
  against the model's 12.0. `shift_video` and `shift_audio` are their own keys
  and `None` means the model's default.

**`<Audio N>` is wired, and it is a sibling of the subject rather than one of
its sources.** `MiniMaxH3ReferenceToVideo` has a `ref_audios` autogrow group
alongside `ref_images` and `ref_videos` — three clips, counting toward the same
twelve — and it is not `ref_video_audios`, which means "this is *that clip's*
soundtrack" and would tell the model a voice belongs to a video nobody attached.
The guide's construction is `<Audio 1> is the voice-timbre reference for
<Subject 1> (S1)`: its own line in `subject_definitions`, its own line in
`retention_analysis` with a marker from the *audio* table (`fully_copy` for a
reuse, `reference` for a timbre), and the speaker ID **reused, never assigned**.
So a voice file does not fold into a subject's definition the way a second
photograph does. And somebody with only a voice attached is not a `<Subject N>`
at all — nothing visible was uploaded for them, and a label would point the
model at a picture that does not exist.

**An unmatched LoRA is reported rather than assumed to have worked.** Keys that
do not map load nothing, the clip arrives, and it looks like a LoRA that was
simply subtle — the same "a row that does nothing looks like a row that did"
failure the expert guard already names. `_drain` counts ComfyUI's `NOT LOADED`
lines and publishes once on the first progress line, which is the moment loading
is finished and a publish that was going anyway.

`VIDEO_MODELS` is served to the page, so the composer shows only the controls
the chosen model reads. A control that is present but ignored is worse than one
that is absent — it is the UI making a promise the model will not keep.

---

## 2K is absent on purpose

*Live rule extracted from this: when the local module lands it extends the
existing job/status/stop contract, and upscaling this app's own output beats
upscaling a dropped file. See `.claude/rules/backend.md`.*

H3 generates at 768p here and there is no upscale, because the thing that makes
2K is not downloadable. **H3-Regenerate-2K is not a super-resolution module** —
it feeds the 768p result *plus the original multimodal context* back into the
base model to regenerate, which is what lets it recover small text and fine
detail that conventional SR has to guess. MiniMax's README is explicit that it
is withheld: *"this module is not yet open-sourced. We will release it once it
is ready."* H3-Context-IR, their prompt expansion, is withheld for the same
reason. Every `scripts/readme/full-2k-*.sh` in the repo posts to their hosted
platform with a bearer token and the video as a base64 data URL.

Building that hosted path was considered and rejected: it sends renders to a
third party, bills outside Modal, and is a second backend that becomes dead code
the day the local module ships. When it does ship it should extend the existing
job/status/stop contract rather than inventing a parallel one — it is another H3
task taking a video and a prompt, which `_h3_graph` and `/api/video` are already
shaped for.

One thing to build first when it lands: **upscaling this app's own output beats
upscaling a dropped file**, and not by a little. The method's whole advantage is
the original context, and a sidecar still holds the prompt, the shot pills and
the references. An external video arrives with none of that.

A mixture-of-experts pair was the one thing here with no image-side analogue:
*two* checkpoints split by noise level, sampled in sequence by two
`KSamplerAdvanced` nodes handing an unfinished latent over. It is why a video
LoRA row used to carry an `expert`, and why `readVidChips` now deliberately
sends one number — a field whose only value is "both" is a sidecar implying a
choice nobody had. Clips rendered while it existed still record theirs, and the
metadata sheet still reads it: **a reader that drops a field makes every run
that has one unreadable.**

---

## `forge/` is gone

*Live rules extracted from this: the image side is Hopper-only, and the offered
sampler/scheduler lists are checked against the node by `tools/smoke_graphs.py`.
See `.claude/rules/backend.md`.*

It used to say here that the image path *could* move to ComfyUI — Krea 2 is
supported natively (`Krea2` in `comfy/supported_models.py`,
`comfy.text_encoders.krea2`, shift 1.15, the same value Forge defaulted to) —
and that one thing stopped it being a rename: regional prompting.
`forge/krea2/regional.py` masked attention inside Krea 2's single-stream DiT
through a vendor patch to `backend/nn/krea.py`, because Forge Couple's
cross-attention design cannot reach a single-stream model at all. Rebuilding
that on ComfyUI was the cost, and it was not worth paying for a rename.

Somebody else paid it, and paid it better. `CLIFF_SHA` pins a node pack that
does regional multi-character LoRA on Krea 2 through ComfyUI's own hooks —
`comfy.patcher_extension` to wrap the diffusion model, and the
`optimized_attention_override` key in `transformer_options` to swap attention —
so nothing is patched and there is nothing to vendor. It is also a stronger
version of the feature: it multiplies each LoRA's activation delta by zero
outside its box, so there is no pathway left for one character's identity to
reach another's, where masking attention only made it unlikely.

Two things that were true of the old arrangement and are not true now:

- **Attention builds no longer conflict.** `forge/` deliberately installed
  neither sageattention nor flash_attn, because both assert `mask is None` and
  would have silently disabled regional prompting. The node pack runs its own
  FlexAttention kernel for the masked case and delegates unmasked blocks to
  whatever backend is installed, so `--use-sage-attention` is on for both
  paths and the two families share one image.
- **The image side is Hopper-only.** That is the bill for sharing: SageAttention
  is compiled for sm_90, so the A100-40GB Krea 2 used to run on is gone from
  `IMAGE_GPUS`. Moving either list means changing `TORCH_CUDA_ARCH_LIST` and
  forcing a rebuild.

What did *not* survive the move is Forge's sampler and scheduler menus. Those
were labels — "Euler a", "Automatic" — and `KSampler` validates `sampler_name`
against `comfy.samplers.KSAMPLER_NAMES`, so they were not spellings of ComfyUI
names, they were values it rejects. `tools/smoke_graphs.py` checks the offered
lists against the node now, which is the check that would have caught it.

## Krea 2 left SageAttention, and the flag stayed for H3

The `forge/` entry above says `--use-sage-attention` is on for both families.
It still is — as argv — but since 2026-08-25 the Krea 2 graph opts back out
with a `ModelAttentionBackend` node, and the receipt is `tools/ab_sage.py`:
six matched renders at 1024px, 3.23s median on PyTorch attention against
3.26s on sage. A 1024px render is ~4k tokens; attention is not where a Krea 2
render spends its time, so the flag bought nothing there — and on sm90 sage
dispatches to an FP8-PV kernel that runs with both of its outlier mitigations
off, which is the mechanism behind the intermittent black blotches the switch
was made to rule out. H3 keeps sage: a packed video/text/audio sequence is
long enough to pay for it.

## CacheDiT lasted one day

DBCache step-skipping for H3 (Jasonzzt/ComfyUI-CacheDiT over the cache-dit
library) went in at a measured 1.40x — `tools/ab_cache.py`, 47.9s to 34.2s on
a 20-step take — and came out the next morning on production logs. Both
numbers were real; they were measured at different shapes.

The harness measured 544p and 124 frames. Production runs 768p and eight-to-
ten-second takes, and there the wrapper's per-step bookkeeping stopped being
noise: computed steps ran ~15s where stock runs the same take at 8-10, a ~50%
inflation on the 15 steps that still compute, against 5 steps skipped. The
wrapper's own dashboard told the story honestly — 25% hit rate, "estimated
speedup 1.33x" — while the take got slower, because its estimate prices the
skips and not the overhead. Part of that overhead is a TaylorSeer calibrator
the wrapper enables by default and exposes no input to turn off; fixing it
means forking the pack.

Two lessons worth the price of admission. **A cache that skips a quarter of
the steps still loses if it taxes the other three quarters** — overhead
scales with tensor size, skips do not, so a draft-shape measurement cannot
stand in for production shape. And **`cache_dit.enable_cache` wraps the
resident model in place**: removing the node from the graph does not unwrap a
model a warm container already holds, which is why the removal shipped with a
container kill. If this comes back, it comes back measured at 768p on
ten-second takes, with the calibrator off, and with an unwrap story.

It did not come back; the other family shipped instead. TeaCache's whole-
output reuse tests the *latent*, not the hidden states, so its bookkeeping
does not grow with resolution — the same harness at production shape read
220.0s stock against 97.4s cached, computed steps at stock price. It ships
as `comfy_nodes/visionary_step_cache`, first-party, because the pack it was
proven with keeps its state in a node execute that ComfyUI caches — spent
state silently disables it from the second take on. Ours keys on the
sampler's sigma and resets itself.

It did not come back; the other family shipped instead. TeaCache's whole-
output reuse tests the *latent*, not the hidden states, so its bookkeeping
does not grow with resolution — the same harness at production shape read
220.0s stock against 97.4s cached, computed steps at stock price. It ships
as `comfy_nodes/visionary_step_cache`, first-party, because the pack it was
proven with keeps its state in a node execute that ComfyUI caches — spent
state silently disables it from the second take on. Ours keys on the
sampler's sigma and resets itself.
