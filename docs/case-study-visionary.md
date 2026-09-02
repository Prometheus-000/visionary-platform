# Visionary — A Serverless Creative Engine for Direct Spatial Control

**A single-file, zero-idle generation platform where the canvas is the interface, intent is the source of truth, and the prompt is a receipt.**

*A product design + design-engineering case study.*

---

## 1 · Overview

Visionary is a single-user studio for training LoRAs and generating stills and video, deployed as **one file to one URL**. `modal deploy app.py` installs the entire application — UI, API, and GPU jobs — with nothing to keep alive between sessions. There is no dashboard, no queue console, no separate front-end host: the front end is built into the image, and the whole system scales to zero when you close the tab.

The work covered here is the recent zero-to-one arc that turned Visionary from a competent generator behind a prompt box into a direct-manipulation instrument: a **semantic layer** that replaces the prompt rather than decorating it, **spatial regions** drawn straight onto latent space, and a **unified image/video continuum** with no mode switch — all riding a serverless backend whose engineering decisions are themselves UX decisions.

> **Thesis — Built for Dreamers.** The person using this is a filmmaker, not a prompt engineer. Their job is to describe a picture they already see; the tool's job is to speak whatever dialect each text encoder happens to need. Every format the user is asked to hold in their head is the tool making its problem into theirs.

### The middle path

The generative-AI interface landscape has settled into two failure modes, and Visionary is a deliberate refusal of both.

| Model | What it optimizes | What it costs the user |
|---|---|---|
| **The prompt box** (Midjourney, most hosted tools) | Approachability | The prompt *is* the state. Nothing accumulates, nothing is addressable, and a minor edit means rewriting the sentence and rerolling the whole picture. Method is copy-pasteable, so everyone converges on the same house style nobody chose. |
| **The node graph** (ComfyUI, A1111 stacks) | Total control | Control at the price of an engineering skill wearing an artist's clothes. The graph is the medium; the picture is an afterthought two hundred nodes away. |
| **Visionary** | Direct manipulation, system-first | You argue with the *picture*. The system owns the prompt. Spatial intent is expressed on the canvas, not in a settings column. |

The controlling design principle behind every decision below:

> **If a gesture cannot communicate intent without a label, the answer is a better gesture or a smarter reading of it — never a control panel.** The month-four failure this guards against is always the same one: something doesn't fit cleanly, and the cheapest fix is a panel. That is the moment a canvas becomes a node graph with better typography.

---

## 2 · The Semantic Layer — the mutable, disposable scene document

### The problem: enhancement is distrust

"Prompt enhance" features take your sentence, run it through an LLM, and send the result to the model. Three things are wrong with that, and they compound:

- The LLM's interpretation **replaces** your intent, silently.
- A minor correction forces a **full reroll** — there is nothing addressable to edit.
- A bare fragment is read as *keywords*. Type `the kitchen after the party` and the encoder renders a party in progress, because the word "party" is in the string. Type `empty diner, 3am` and you get daylight, because nothing anchored the clock.

The last point is the one that turned this from an ergonomics project into a measured one. **A fragment is where interpretation earns its place** — a self-correction discarded (`night. no, late afternoon` → a night scene) or a hedge rendered literally is exactly the failure an interpreter fixes and restraint cannot reach.

### The architecture: interpretation as untrusted input

The layer inserts a step between the person and the generators:

```
prose → interpreter → candidate document → deterministic validation → compiler → Krea 2 / MiniMax-H3 / Wan
```

The interpreter is a **text-only, abliterated Qwen3-4B** served by a local **vLLM** process on its own `@app.cls` (an L4), warmed by a ping on page load so the first fragment lands warm while the user is still typing — and scaling to zero between sessions. It is text-only on purpose: the parse reads a sentence, never a picture, so the vision tower would be weight and latency for a capability this stage never uses. The abliteration is the one local change, documented as a *procedure* (orthogonalize the refusal direction out of the residual stream) rather than a model, so a stale fork is an afternoon's re-run instead of a rewrite. The reason is specific to a schema-bound parse: **a refusal cannot arrive as recognizable prose; it arrives as an evasive storyline that satisfies the schema**, which nothing downstream can distinguish from a real one.

> **The single hardest-won invariant, and it overrules the original brief:** the user's prose is the source of truth. The document is a *derived, disposable interpretation* of that prose. Dropping a document must cost nothing, and it only costs nothing if the record survives it.

Four rules enforce that, each named where it lives in the code so the invariant is a place, not a paragraph:

1. **A document is valid only for the exact prose it was derived from.** Made *unrepresentable* rather than remembered: the store holds `doc` and exports only `docFor(prose)`. No caller can obtain elements without naming the string it believes they describe. A stale document sends no modules — server-side, re-derived, because the client is one more untrusted caller.
2. **Interpretation is untrusted probabilistic input.** The pipeline is never `Qwen → truth`. A *malformed* document is refused by name; an *untrustworthy* one is dropped and the run proceeds plain — no error, no toast, no banner, byte-for-byte the app as it was before any of this existed. Shape and trust are two functions because they are two behaviors.
3. **The layer enriches without taking ownership of the words.** The model may *insert*, never *revise*. A write that would take words off the person simply does not happen.
4. **Provenance is a fact about facts, not characters.** `a red winter coat` → `an oxblood down jacket` is the same fact in better words and carries no mark; `…and she looks visibly cold` adds a state nobody mentioned and is underlined — one gesture from gone.

### The trust surface is the interface

The marks live *inline in the prompt box*, on a mirror layer: every **element** carries a dotted underline (a thing the document can act on); the model's own words are laid over the top in **grey**. The only thing a grey run does that plain text cannot is **reroll** — rooted at the run's own end, revealed while the caret is inside it, gone when the caret leaves. Editing one needs no control at all: the edit drops the mark, the words turn dark, and they become yours.

> **Derived or invented, always visible.** This is the entire trust surface, and it needs no dialogue. It kills the chat panel, the assistant sidebar, and the clarifying question — every question asked is a small failure. Pick something, mark it invented, move on.

### The turn: what measurement retired

This is the part a portfolio piece is obligated to tell honestly, because it is the strongest part. The first build of this layer added an **arithmetic apparatus** — a coverage floor, an invention ceiling, a `_document_trust` score — to decide when to trust the model. Measured against the deployed interpreter, that apparatus is what made the feature **inert: it reached 0% of renders on finished prose.**

The failure was structural, not a mis-set number:

- On a fragment, a *good* enrichment scored **94% invention** and an *evasive* document **93%** — the good one higher. A share-of-characters metric is dominated by how much the person typed, not by what the document did.
- The invention ceiling permitted adding at most 1.78× what was typed. `empty diner, 3am` is 16 characters — an enrichment budget of 28 — while the prompt that produced the *better* picture needed 332. **The budget scaled with what you'd already written, so it gave least help to the input that needed most.**
- Rendered blind, both orders, a win counted only when both agree: **the document layer won 0 of 30 comparisons against the bare fragment** across two model sizes and two rule sets. Every configuration *compressed* the input; the prompts that actually win *grow* it 11–21×.

The element schema was the reason, and this is the finding that reframes the whole feature: **the document apparatus crippled the model — we were punishing it for doing its job.** A grammar whose unit is a short tagged fragment makes a model **decompose** where it needed to **write**, and the two thresholds then *selected for inaction*: they trusted the documents that stapled fragments of the sentence back onto itself and **refused the model exactly when it did the right thing.** Reading `night. no, late afternoon` correctly means dropping "night" — a correct reading scored 59% against the coverage floor and was rejected. The good enrichment that authored a real prompt tripped the invention ceiling. The scaffolding built to make the LLM safe was the thing making it useless, and it did so by penalizing precisely the judgment the feature existed to get.

**The deeper error was in the measurement, not the model.** Every check in the harness was **text-to-text** — preserved, covered, round-tripped, idempotent — scoring the document against the *characters of the sentence it came from*, and none of them scored the result against the picture. So the model was being judged for **character matching when it should have been judged for intent.** A metric built on word overlap can only ever reward the model for keeping the words and punish it for interpreting them — which is why *zero invented words across 27 fragments* read as maximum restraint and shipped as a feature that never fired. The fix was to stop measuring the proxy: render the pair and have a judge read the output against the original description (`does_it_help.py`, `prompt_ab.py`), and read four criteria that are all about the render — core subject extraction, emotional tone transfer, spatial logic, literal fidelity — rather than totalling a lexical score. **You cannot measure intent with a diff.**

**So the thresholds were deleted, the document path was switched off, and what replaced it is the better experience — not a smaller one.** The active version is a single instruction (`REWRITE_OPS`, Krea's own `expansion.txt` vendored verbatim) that returns **prose, not a document** — because the people who trained the encoder wrote the prompt for talking to it, and prose is the shape that lets the model write instead of chop. Blind A/B, rendered, both orders, wins counted only when both agree: **3 wins, 1 loss, 6 ties over 10 fragments** — the first configuration in the entire project to beat doing nothing, and the three wins are exactly the fragments the old validator refused.

It is the better *user* experience for reasons that have nothing to do with the A/B margin, and that is the point:

- **It is trustful rather than distrustful.** A distrustful feature alters your prompt and sends it to the model. This one hands back a new prompt, written into the same box you type in, that you can read, edit as plain text, accept, or throw away with ⌘Z. Nothing reaches the encoder that wasn't on screen first.
- **There is one job, so the model never classifies.** No "is this a fragment or a finished prompt" branch to get wrong; the one instruction is already conditional (a 16-character fragment grows 40×, a finished prompt 2.6×), and that behavior is emergent, not instructed.
- **It is honest about what it can't catch.** A document that reads the sentence and then contradicts one fact — `empty diner, 3am` coming back with daylight — is caught by nothing arithmetic. The thing that stands in for the missing check is exactly the UX obligation above: **the replacement is visible, so an error is something you see and delete, not something that renders silently.**

> **The lesson, generalized:** a probabilistic gate stacked on a probabilistic interpreter is two coin flips where the second one is invisible — and the second flip was landing on *inaction*. Keep arithmetic in the validator (structural zeros: *is this document about this prose at all?*), keep judgment in the test harness (a rubric a model reads, offline, where latency is free), and make the trust surface the interface itself: a visible, editable replacement beats any score, because the person is a better judge of their own picture than a coverage floor ever was.

---

## 3 · Spatial Regions — bounding boxes as latent masks

### The interaction

A region is drawn on the canvas and *everything about it is drawn on the canvas*. Drag on the frame to place a box; drag it to move it; drag a handle to size it. Touch one and it **opens** — a card rooted in the box's own near edge, holding its sentence, its LoRA strength, its reference photograph, and its four coordinates. **Selection is the open state.** There is no toggle, no sidebar, nothing to dismiss.

Each box masks a LoRA to its own rectangle, so two characters can be generated in one frame without their identities bleeding — the node pack multiplies each LoRA's activation delta by zero outside its box, so there is no pathway left for one character to reach the other's region. A box also takes a photograph directly (`ref_image`, a latent mold that pulls that rectangle toward that face during sampling), which means *a character with no training run behind it* — worth having on a platform whose other half is a trainer.

### The system discipline

The interaction is governed by a three-state model (`store.edit`) that exists to keep the render clean:

- **`off`** — the moment any render lands. Nothing is drawn. The boxes are still there, still masking, still sent with the next run; they are **addressable rather than drawn**. Hover names what you'd touch; a click opens it.
- **`content`** — the frequent act. One box's card, its own hairline for scope, *no coordinates* (they're the escape hatch for the rectangle, not for the sentence).
- **`geometry`** — the rare act, behind ⌘-click. Every box, its handles, snapping.

> **Nothing sits on top of a render.** A finished picture puts the boxes away every time, including after a render you asked for while editing them. This is chrome discipline as a hard rule: the boxes are *reachable* at all times and *drawn* at none. The four coordinates — `0.5 0 0.5 1` — were the right parameter and the wrong primary; the rectangle is not something you rebuild in your head, so dragging teaches the numbers and the numbers never taught the dragging.

Shot, lighting, and camera primitives work the same way: a **palette** (`SHOT_VOCAB`) of small animated tiles behind a labeled door, where a tile that *shows* a dolly-out teaches what no word can. The pills compile into the prompt; the closed vocabulary stays on the server and the page sends a key, so a run is reproducible from its record rather than from whatever text was in a field at the time. No persistent inspector, no layers list, no "control" tab — the standing veto that keeps this from becoming a node graph.

---

## 4 · The Image ↔ Video Continuum

Image and video are **one workspace** — shared prompt, shared canvas, shared gallery. There is no Image/Video mode switch. **Duration is the control:** `Still` runs Krea 2, any length routes to MiniMax-H3 or Wan 2.2. Measured, the two strips differed by exactly one thing — a seconds picker — so half the application had been sitting behind a toggle that gated one parameter.

> **Duration starts at zero.** A still is the default and time is something you add. Someone who wants one image should finish and leave without ever learning that motion exists.

The bridge is real, not cosmetic. The run record carries **two ids** — the render *on screen* and the job being *polled* — so pressing Generate never blanks the picture you're judging; the old render holds until the new one lands, which matters most when a video take is two to three minutes and you wanted to compare. "Animate" and "As reference" on a finished still are the first pieces of a still flowing into a clip with no round trip through the filesystem.

The two video families are not averaged into a false uniformity. `VIDEO_MODELS` is served to the page so the console shows **only the controls the chosen model reads** — Wan takes LoRAs, CFG, and a negative prompt; H3 is guidance-distilled and carries its own soundtrack. A control that is present but ignored is worse than one that is absent: it is the UI making a promise the model won't keep.

---

## 5 · Serverless Architecture as a Design Choice

The engineering here is not backstage plumbing — it is directly why the interface can be what it is.

### Zero-idle is a UX property

Because the whole app scales to zero on Modal, there is no "keep alive," no idle bill, and no lifecycle the user has to manage. The cost of that trade — cold starts — is paid where the user isn't looking: the interpreter is warmed by a **page-load ping** during the ~15 seconds the user spends typing, and a family of weights downloads on **CPU containers**, never on a rented GPU idling next to a 26 GB pull.

### One file, four images, isolated CUDA

`app.py` is deliberately **one file (~10.6k lines)**. The alternative — a package whose modules are imported by Modal image builds — trades one long file for a build-order problem. What is *not* merged are the container images, and the boundary is **CUDA isolation**, because the pins genuinely conflict:

| Image | Base | Purpose | Why isolated |
|---|---|---|---|
| `comfy_image` | **CUDA 13.0.3** (Ubuntu 24.04, py3.12) | Images **and** video inference | ComfyUI's quant backend needs `torch.version.cuda ≥ 13`; SageAttention is compiled from source (`-devel` base for `nvcc`), pinned to **`sm_90`** via `TORCH_CUDA_ARCH_LIST=9.0` |
| `trainer_image` | **CUDA 12.4.1** (py3.10) | LoRA training (musubi-tuner) | A separate `transformers` era; merging its pins would make every captioner bump re-litigate training |
| `caption_image` / parse | **CUDA 12.8.1** | Captioning + the Qwen parse (vLLM) | vLLM's inductor shells out to `nvcc`; a shared pin would force a compromise paid forever |
| `hf_cache` volume | — | Model-weight cache | Separated by *write pattern*: committing 17 GB of weights after writing 80 captions turned an 8-minute job into a 23-minute one |

> **Separate images only when pins conflict — and only then.** Images and video *share* `comfy_image` because nothing in it is per-family, and a fourth image would have been a second CUDA-13 build, a second SageAttention compile, and a second place to get the arch list wrong. Boundaries follow build cost, not tidiness.

**On Hopper, Ada and Blackwell:** the inference image compiles SageAttention for every card on its lists (`TORCH_CUDA_ARCH_LIST="8.9;9.0"`: Hopper `H100`/`H200` and Ada `L40S`), and that string was a law for a year before it was read as a build arg — the same string is what moved Krea 2 off the A100 it used to run on. Blackwell (`B200`, `sm_100`) is a *deliberate one-line pathway*, not a silent capability: add `10.0` and force a rebuild, or the kernels load the model and then fall back to the slow path. `max_containers=1` per GPU class keeps exactly one warm checkpoint resident, so a second replica never pays a cold load to share a job it could have queued behind; the one-container mode under the gear bends that on purpose, with the wait stated before the switch.

The payoff of this whole posture: adding Wan did not add a backend. It reused the container, the warm ComfyUI process, the job/status/stop contract, and the output layout — a graph builder and a row in `VIDEO_MODELS`. That is the shape any third family should take.

### The visual system — restraint as a design position

The interface is a dark, technical surface with **the canvas as the largest thing on screen at every moment**, because the picture is the reason the page exists. Everything adjustable lives in a bar *under* it, never a rail beside it — a settings column costs the image 384px of the one dimension it can't get back, and the dead-space measurements proved the picture is height-bound at every aspect ratio, so the bar always comes out of slack the picture wasn't using.

The CSS carries its Rams/Braun lineage explicitly — *less, but better*, expressed as a **named vocabulary** rather than a design system:

- A surface ramp of **six steps**, replacing 34 distinct white alphas that were "transcription drift, invisible in review because each instance is locally reasonable."
- **Four corner radii** for four shapes, where a popover, a card, an input, and a tile had been told apart by a single pixel of curvature.
- **Copy is a last resort** — design first, then an icon, then words — but corrected by measurement: *a bare number is not a value an icon can show*, so every hyperparameter field carries its name and the tooltip is promoted to saying what the number does.
- A hard **console budget of 30% of the viewport**, enforced in JS (`fieldMax()`), where the prompt field is the only thing allowed to yield.

> The vocabulary "constrains nothing and is deletable." What it actually buys is the next phase: **the Dynamic Canvas is a subtraction project** — chrome disappears and elements become addressable rather than drawn — and you cannot subtract consistently from 34 alphas, because every deletion becomes an archaeology question. Named, it's one line.

---

## 6 · Where this is going

Every feature above is a move toward one end state: **the canvas is where the work is done, not where the result appears.** Region prompts, LoRAs, and every attachment become a single gesture in a single place — a photo on a box is that character, a photo on the frame is the scene. What you touch decides the mode; nothing is labelled; nothing asks for confirmation, because undo is total. The prompt field itself is the endgame: a compilation target the user never has to see, because *nobody should have to learn a text encoder.*

The deeper bet is about **authorship**. A prompt is copy-pasteable, so a method transfers whole and an entire platform converges on a house style nobody chose — the reason every image on the big model-sharing sites looks the same. A conversation with a picture doesn't transfer that way, and a LoRA trained on your own photographs transfers even less: the model itself learns what *you* meant. That is why training is Phase 1 here rather than a bolt-on, why the picker reads your volume instead of a marketplace, and why "discover models" is a standing veto. **The technical decision and the artistic one are the same decision.**

> **Built for Dreamers** is not a tagline. It is the claim that the tool should take fragments — out of order, self-correcting, arriving in pieces — the way intent actually arrives, and hand back a picture you can argue with. The prompt is the machine's business. The intent is yours, and the sidecar keeps it as the record.

---

**Source:** [github.com/Prometheus-000/visionary-platform](https://github.com/Prometheus-000/visionary-platform)
