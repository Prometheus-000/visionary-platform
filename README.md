# Visionary

A generative studio that runs on your own [Modal](https://modal.com) account.
Train a LoRA on your own photographs, generate stills with it, and animate any
of them into a clip — one interface, one URL, nothing to keep running between
sessions.

![A written prompt and the picture it produced, on a canvas that holds the screen](docs/generate.png)

```bash
pip install modal
modal setup
modal deploy app.py
```

That is the whole install. The last command prints a URL, and the URL is the
application — interface, API and GPU jobs. Nothing runs on your machine, nothing
runs while you are not using it.

---

## What you can do

**Generate stills and video in one workspace.** Image and video share a canvas
and a gallery, and each keeps its own composer — switch away and back and
everything is as you left it. Duration is the switch: `Still` runs Krea 2, any
length runs MiniMax-H3. The controls on screen follow the model, so you only
ever see the ones it actually reads.

**Write prompts as pills, not paragraphs.** Camera moves, framing, lighting,
tone and audio come from a palette of seventy-seven animated tiles — a tile
*shows* you a dolly-out rather than making you guess the wording. What you type
keeps only what nothing else can say: who is in the shot and what happens. A
live preview shows the exact string the model will be handed, compiled by the
same code the run uses.

**Cast a scene by typing `@`.** The video side has no prompt box, because H3
reads a document rather than a sentence — shots, cut times, and who is speaking
in each. Type `@` mid-sentence and a picker floats off the caret; picking makes
a cast member and drops their name into the prose. Hand them a photograph and it
is what they look like, a recording and it is what they sound like. A shot's
slice of the clip is the length of what you wrote about it. Write one shot, cast
nobody, and the run is your typed text byte for byte.

**Place multiple characters with regional LoRAs.** Draw a box on the frame and
pick a LoRA on the card that box opens: it applies only inside the box, so two
trained identities stay separate instead of blending. Each box can also take a
reference photo; drop a photo on the bare canvas instead and it becomes the
scene the picture is generated inside.

**Train your own LoRAs.** Point the trainer at a folder of images and get a
`.safetensors` back. Runs live on a board rather than taking over a screen, so
you can **train several at once** — each is its own card showing live epoch,
step, rate and loss, and a finished card hands its LoRA files straight to the
LoRA picker. Start a run, add another, and leave; status is read off each job's
heartbeat, so a card reports what actually happened even if a container was
killed mid-step.

**Prepare datasets in the same place.** Images get prose captions from
Qwen3-VL-8B, with presets tuned to what you're training (character, style,
concept). A panel reads the set back to you — trigger-word coverage, caption
length, repeated clauses — and duplicate detection flags copies before they
train unevenly.

**Open the hood in the Playground.** A node room over the same engine: it
opens on the exact graph the app itself would run, and you rewire it, add
nodes, install packs from git, and run the result on the same warm GPU.
Workflows save as plain API-format JSON on your volume, every render embeds
its own graph the way ComfyUI does — the PNG *is* the workflow, droppable back
onto the room — and a saved workflow can stand in for the built-in graph from
the model menu, with the console still compiling your prompt into it.

![Seventy-seven tiles, each animating the move it names](docs/shot-palette.png)
![A take, the pills that produced it, and the gallery beside it](docs/video.png)
![A box per character, each LoRA masked to its own rectangle](docs/regional.png)
![Everything you have made, in one grid](docs/gallery.png)
![Captioning is a workspace, not a batch job](docs/dataset.png)

---

## Requirements

- A Modal account. `modal setup` walks you through auth in a browser.
- Python 3.10+ locally, only to run the `modal` CLI.
- A [HuggingFace](https://huggingface.co) account **if** you want Krea 2 — its
  weights are gated. Everything else downloads without one.

You do not need a local GPU, Docker, a `.env` file, or any Modal Secret. The
HuggingFace token is pasted into the UI and stored in a Modal Dict.

### If you would rather run it on your own card

Everything above stays true; this is a second way in, not a replacement. What it
needs:

- **NVIDIA, and a CUDA 13-capable driver** (r580 or newer). The inference wheels
  are cu130; the trainer's cu124 and the captioner's cu128 run under the same
  driver, which is what lets three environments share one machine.
- **VRAM.** 16 GB runs images and video at their quantised tiers; 24 GB runs the
  tiers the deployment uses. Regional multi-character rendering wants ~32 GB and
  is refused below it — see the table under **First run**.
- **System RAM**, and more than you would guess: a card smaller than the
  checkpoint streams the rest from host memory, so **64 GB** is the figure to aim
  at on a 16 GB card. This has never been a number this project had to publish,
  because a Modal container's RAM came with the GPU.
- **~30 GB of disk for the environments**, before any weights.
- Node, to build the front end. Same build the image runs, same pinned version.
- No Apple Silicon and no AMD. Every pin in the inference image is CUDA-versioned
  for a measured reason, so this does not run on the machine it was written on.

```bash
python3 tools/local_install.py --dry-run   # what would be installed, and from where
python3 tools/local_install.py             # build the environments
python3 tools/run_local.py                 # http://127.0.0.1:8790
```

`--models-dir ~/ComfyUI/models` points it at weights you already have rather than
downloading a second copy. Weights are addressed by exact filename, so a file you
own under a different name is invisible — the gear says which directory it looked
in and what is actually there, which is how you tell that from an empty folder.

**This runs the same `app.py` the deploy ships.** It is not a port and there is
no second copy to fall behind: `tools/run_local.py` sets one environment
variable, imports that file, and serves the FastAPI object it already builds. A
new route or a new shot in the palette arrives here because it is there.

---

## First run: get the weights

A fresh deployment has an empty volume — nothing downloads on its own, because
the full catalogue is ~206 GB and almost nobody wants all of it. Open the
deployed URL, click the gear, and pick what you need.

| Family                       |   Size | Gated | What it buys                              |
| ---------------------------- | -----: | ----- | ----------------------------------------- |
| Krea 2 — images              |  62 GB | yes   | training + still generation               |
| MiniMax-H3 — video           |  64 GB | no    | video **with a soundtrack**, references   |
| MiniMax-H3 speed LoRAs       |   3 GB | no    | fewer steps per clip                      |
| Krea 2 style LoRAs           |   4 GB | no    | Krea's own nine styles, for the prompt    |

### Smaller weights, for a card that cannot hold those

Every row above is the size the deployment runs. There are smaller ones, and
picking one is a download rather than a setting: the app resolves each slot to
the largest file on disk that fits the card it found, so pulling a tier is the
whole of choosing it.

| Slot                | Deployment       | 24 GB card       | 16 GB card         |
| ------------------- | ---------------: | ---------------: | -----------------: |
| Krea 2 Turbo        | 26.3 GB bf16     | 8.8 GB Q5_K_S    | **7.2 GB Q4_K_M**  |
| Krea 2 RAW          | 26.3 GB bf16     | —                | **7.3 GB Q4_K_M**  |
| H3 transformer      | 21.0 GB int8     | 21.0 GB int8     | **15.9 GB int4**   |
| H3 text encoder     | 15.7 GB nvfp4    | 15.7 GB nvfp4    | 15.7 GB nvfp4      |

The gear can pin a slot to any file you have downloaded, and it will not argue:
asking for the 21 GB file on a 16 GB card is asking for a slow render, not making
a mistake. A pin survives a redeploy, and a pin to a file you later delete falls
back to the default rather than failing the job.

**NVFP4 is not on this ladder on purpose.** It has no compute path before
Blackwell — on a 4090 it is a memory saving and nothing else — so it belongs to
an RTX 50-series card rather than to a size.

**The H3 text encoder has no smaller tier, and that was checked rather than
assumed.** A community int4 file exists and is 0.74 GB smaller; `python3
tools/probe_tiers.py` reads both headers over HTTP range requests and they are
not the same model to a loader — nvfp4 carries 2054 tensors, the int4 file 1602,
and the 452 missing ones are the entire scale apparatus the quantisation is
described by. So 16 GB video is the transformer coming down (21.0 to 15.9) plus
streaming, not a smaller encoder.

That probe is worth running before trusting any row here — it costs nothing, no
GPU and no download, and it is what removed the encoder row. What it cannot tell
you is whether a file that loads also *runs*: the kernels, the card and a render
are the GPU half, and **16 GB for Krea 2 still assumes ComfyUI evicts the 8.9 GB
text encoder before the transformer loads.** That assumption is on the
rented-box list. If it is wrong, those rows come out too.

You do not need a whole family. Video without references is the base set — the
fl2va transformer, the text encoder and the two VAEs — and the ref2va
transformer is one extra file on top of it rather than a second stack.

The **style LoRAs** are the cheapest way to see regional prompting work: put two
styles in two boxes and the hard seam between them — ink wash on one side,
motion blur on the other — is the masking, made visible.

Downloads run on CPU containers, never on a GPU, and report the bytes and rate
as they go. A transfer that goes quiet for four minutes is abandoned and resumed
from where it stopped, up to five times.

### Gated weights

Krea 2 RAW and Krea 2 Turbo need a HuggingFace token, and you must accept the
licence with the same account that issued it:

- <https://huggingface.co/krea/Krea-2-Raw>
- <https://huggingface.co/krea/Krea-2-Turbo>

Paste the token under the gear. If the licence has not been accepted, the error
says so and links the page rather than failing as a generic 403.

### LoRAs from Google Drive

Most LoRAs worth having were never published to HuggingFace — they are a link
someone sent you. Paste one under the gear and it lands in `loras/`, ready to
name in a prompt.

- A file link, a bare id, or a folder link all work.
- Only `.safetensors` is kept; a folder's preview grid and readme are skipped.
- Leave **folder** blank and files drop in loose, each its own entry. Give one
  and they are grouped as versions of a single LoRA under `loras/{folder}/`.
- The link must be shared with anyone who has it. An unshared file is named as
  that case explicitly, rather than surfacing a parse error.

---

## What runs underneath

**Train.** LoRA training for Krea 2 on
[musubi-tuner](https://github.com/kohya-ss/musubi-tuner). Point it at a folder
of images, get a `.safetensors` back. Each run loads its own weights and writes
its own output folder, so several train concurrently — the trainer is the one
GPU class here with no single-container pin.

**Caption.** Datasets are named folders of images with `.txt` sidecars beside
them — exactly what the trainer reads.

**Generate stills.** Krea 2 inference through a driven ComfyUI, with LoRA
stacking and regional multi-character LoRA.

**Generate video.** MiniMax-H3 through a driven ComfyUI — sound and picture in
one pass, from text, from a first and/or last frame, or from up to twelve
references (pictures, video and audio) through the ref2va transformer. It is
guidance-distilled, so there is no CFG and no negative prompt; LoRAs load
through `LoraLoaderModelOnly` like anything else, including MiniMax's own
Lightning distillations.

A second family (Wan 2.2) shared this path for a while and was removed. What it
proved is worth keeping: the container, the warm ComfyUI process and the job
contract are family-agnostic, so what a family costs is a graph builder and a
row of capabilities — the same row the UI reads to decide which controls to
show.

---

## Storage

One Modal Volume, mounted at `/workspace`:

```
models/               weights, flat, addressed by exact filename
loras/                trained LoRAs, one folder each; loose files work too
datasets/{name}/      images + .txt caption sidecars
outputs/{job}/        generated media + a visionary.json sidecar
work/, .cache/        disposable
```

The layout is the contract: datasets are folders of images with text files
beside them, so nothing here is required to get your data back out.

Run a second, isolated copy — its own URL, its own records, its own weights —
by naming all three:

```bash
VISIONARY_APP=visionary-test VISIONARY_VOLUME=visionary-test VISIONARY_MODELS_VOLUME=visionary-test-models modal deploy app.py
```

`VISIONARY_APP` is the one to remember. `modal deploy` replaces an app of the
same name, so setting only the volumes gives the test copy separate storage at
the *live* URL — which reads as the deployment having broken rather than as
having been replaced on purpose.

---

## GPUs and cost

Each job type picks its own class, and most are switchable in the UI:

| Job              | Default   | Options    |
| ---------------- | --------- | ---------- |
| Training         | A100-40GB | —          |
| Captioning       | A100-40GB | —          |
| Image generation | H100      | H100, H200, L40S |
| Video generation | H100      | H100, H200, L40S |
| Both, one container | H200  | H200, H100, L40S |

Image and video generation share one image, and its SageAttention kernels are
compiled for every card on these lists. What separates the cards is memory:
Krea 2 wants the headroom a regional render with reference photos peaks at,
and H3 is 42.5 GB of weights before any activations. L40S is the cheap 48 GB
card — a plain picture fits, a regional render with references may not, and
video runs with its weights paged and several times slower than an H100. An
out-of-memory error names the card it happened on.

"One container for both", under the gear, puts both families on one warm
card instead of two. On H200 both checkpoints stay resident; on H100 or L40S
they take turns through host memory. A picture queues behind a clip in
progress — that is the trade, and the setting says so before it is made.

Containers stay warm between requests (10 minutes for images, 15 for video) so
consecutive takes skip the model load, then scale to zero. You are billed for
GPU time while a job runs and while a container is warm — not for an idle
deployment. Bad inputs (a wrong LoRA path, an unknown aspect ratio, a missing
weight) are rejected on CPU in milliseconds, before a GPU is rented.

---

## Verifying a deployment

Five smoke tests, all cheap, all runnable against your own account:

```bash
modal run tools/smoke_graphs.py     # every graph validates against ComfyUI's node schema (CPU, no weights)
modal run tools/smoke_caption.py    # every captioner repo id resolves and parses (CPU); --gpu captions a real image
python3 tools/smoke_prompt.py       # the shot compiler matches MiniMax's published format (stdlib, no network)
python3 tools/smoke_pins.py         # every pinned wheel still resolves, before a deploy spends 20 minutes finding out
python3 tools/smoke_local.py        # the local runtime constructs, dispatches, queues and stops (no GPU, no account)
```

`smoke_local.py` is the one that guards the local build, and it guards a
specific thing: the *seam*. Features cannot drift, because there is one `app.py`
and the launcher imports it. What can drift is a ninth `.spawn()`, or a
`jobs.keys()` — lines that are perfectly good against Modal and mean nothing off
it. So it asserts those surfaces closed, on a laptop, in ten seconds.

**When a rented box is owed.** The Mac checks prove the seam and nothing about
what a card does with weights. Rent one whenever a pin, an image build step or
ComfyUI's argv changes, and run: a cold start; a Krea 2 render at bf16 and at
GGUF; an H3 take at 544p draft; a rank-32 train with `--fp8_base
--blocks_to_swap`; `Comfy-Kitchen … {'cuda': True}` in the log; and the four
`Visionary*` nodes still binding on a GGUF-loaded model — `VisionaryStepCache`
above all, because a silent miss there costs half the render speed with no error.

### What has been run end to end

Being honest about coverage, since "it deploys" is not "it works":

- **MiniMax-H3** — text-to-video run end to end on an H100. The compiler output
  is checked against the published format by `smoke_prompt.py` and scored 1.00
  against MiniMax's own format grader. Still unverified by ear: whether
  `non_diegetic_music: N/A` actually silences the soundtrack.
- **The scene composer** — driven and read, never measured against a render.
  `tools/prompt_ab.py` is the measurement that is not a proxy and has not run.
- **The local runtime** — the API and the page have been served off it, on a Mac
  with no CUDA: `/api/state` answers with the real catalogue, the real captioner
  list and all 77 shot tiles, and the job contract queues and stops under
  `smoke_local.py`. **No render has ever come out of it**, because the machine it
  was written on cannot run one. Everything downstream of "the graph is posted"
  is unverified there, and the quantised tiers are unverified entirely — see the
  rented-box list under **Verifying a deployment**. The Modal path is unchanged
  and is what the render evidence in this repo comes from.

---

## Working on the UI locally

The front end is React and TypeScript under `web/`, built by Vite into the image
at deploy time — so `modal deploy app.py` stays the whole install and no local
Node is needed to ship. For development, `npm run dev` in `web/` proxies to
`tools/preview_ui.py`, which serves the real prompt compilers and shot vocabulary
against stubbed jobs, so the entire UI is workable with no Modal account, no GPU
and nothing billed:

```bash
python3 tools/preview_ui.py
```

---

## Layout

```
app.py              the whole application — images, jobs, API, and the UI
comfy_nodes/        our own ComfyUI nodes — one shim, see visionary_boxes
tools/              smoke tests and the local UI preview server
tools/_from_app.py  pulls plain-Python pieces out of app.py by AST
CLAUDE.md           the design rationale — why the code is shaped the way it is
```

`app.py` is deliberately one file — long, but navigable by its banner comments,
and it keeps `modal deploy app.py` the whole install. If you are going to change
anything, read `CLAUDE.md` first: it explains the tradeoffs the code is holding,
several of which look like mistakes until you know what they avoid.

---

## Licensing

**[AGPL-3.0](LICENSE).** Strong copyleft with a network-use clause: section 13
means that if you modify this and let other people use it over a network, you owe
those users the corresponding source — deploying, not just distributing, counts.
Since this deploys as a web application by design, that is the normal case here.
Running your own private instance triggers nothing — including a local one on
your own card, which is a private instance like any other.

Handing somebody a modified copy to run is *distribution* rather than network
use, so sections 5 and 6 apply instead of 13: the same obligation, owed at a
different moment. Nothing about the local path changes what you may do; it
changes which paragraph says so.

The images install, rather than vendor, three upstreams:

- **[ComfyUI](https://github.com/Comfy-Org/ComfyUI)** — GPL-3.0. Cloned at
  `COMFY_SHA`, run as its own process, driven over its HTTP API. Not linked
  against or patched.
- **[Krea2 Regional Multi-LoRA](https://github.com/CliffNodes/Krea2-Multi-Character-Lora-Node-with-bounding-box-Scene-and-Outfit-Edit)**
  — MIT. Cloned at `CLIFF_SHA` into ComfyUI's `custom_nodes/`, unmodified.
- **[ComfyUI-GGUF](https://github.com/molbal/ComfyUI-GGUF)** — Apache-2.0, a
  fork of [city96's](https://github.com/city96/ComfyUI-GGUF). Cloned at
  `GGUF_SHA`, unmodified. It is the only pack that reads Krea 2 GGUF files;
  `tools/upstream.py` watches city96#464 and says when the fork can be dropped.

None of this is legal advice. Model weights carry their own separate licences —
Krea 2's in particular is gated and has terms you accept on HuggingFace. Nothing
here grants you rights to them.
