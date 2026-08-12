# Visionary

A generative studio that runs on your own [Modal](https://modal.com) account.
Train a LoRA on your photographs, generate stills with it, and animate any of
them into a clip — in one interface, on one URL, with nothing to keep alive
between sessions.

![The canvas holds the screen; the console is a bar beneath it](docs/generate.png)

It is a real interface, not a form in front of a script. The canvas is the
largest thing on screen at every moment, because the picture is the reason the
page exists. Everything you can change lives in a bar under it — never a rail
beside it, which would cost the image 384px of the one dimension it cannot get
back.

---

## The interface

**Image and video are one workspace.** They share the prompt, the canvas and the
gallery. The switch is a chip inside the prompt field, and the sentence survives
it — because a shot you described as a still is the same sentence you would
describe as a clip. There is no mode to navigate to and nothing to retype.

**The controls follow the model.** Wan 2.2 takes LoRAs, a negative prompt and
CFG; MiniMax-H3 is guidance-distilled and carries its own soundtrack, so it
offers none of those and offers references instead. Only the controls the chosen
model actually reads are on screen — a control that is present but ignored is
worse than one that is absent.

![Eighty-seven tiles, each animating the move it names](docs/shot-palette.png)

**The empty prompt box is the worst control on the page, so it is not the only
one.** MiniMax-H3 does not read a paragraph. It reads a document with named
fields, published in the model repo — and a textarea in front of that is why
nobody knows where camera direction goes, whether tone and genre matter, or
what to do with a reference image you were told not to describe. A documented
grammar presented as free prose reads as superstition, and a take is two to
three minutes, so every guess is paid for at that rate.

So the closed vocabulary is a palette: one icon in the strip, a popover of
small animated tiles, and a rail of pills under the prompt. The prompt field
keeps only what nothing else can say — who is in the shot and what happens.
This is the "a control that shows its own value gets no label" rule applied to
words instead of numbers, and it is the one place on the page where an icon can
teach: a tile *shows* a dolly-out, which is the thing neither the word nor a
static picture does. A dolly changes the relationship between subject and
background and a zoom does not, so push-in scales the subject faster than the
horizon and zoom scales both — a distinction no dropdown makes.

![What you picked, and the document it compiles to](docs/video.png)

Pick nothing and the compiler returns your typed text byte for byte, so every
prompt written before this still means what it meant. Pick something and the
document appears — and `what the model reads` shows the exact string the
encoder will be handed, compiled by the same route that compiles the real run,
so a preview cannot disagree with what happens. `non_diegetic_music: N/A` is
the default, and is worth the feature on its own: H3 invented a soundtrack for
every clip because nothing had ever told it not to.

One vocabulary, three destinations. Wan 2.2 gets prose with the audio pills
dropped, because it is silent and a sidecar recording an input the model never
read is a sidecar that lies. Krea 2 gets prose with camera, action and sound
filtered out — dimmed in the palette rather than hidden, with the group heading
saying why.

![A box per character, each LoRA masked to its own rectangle](docs/regional.png)

**Regional multi-character LoRA.** Draw a rectangle on the frame and write a
`<lora:name:1.3>` into it, and that LoRA's activation delta is multiplied by
zero everywhere outside the box — so there is no pathway left for one
character's identity to reach another's. The boxes *are* the list: drag to
place one, drag to move it, drag a handle to size it, and they snap to halves,
thirds and quarters and to each other. The console keeps one inspector row for
whichever box is selected, which is the same height at eight boxes as at one.

A box takes a photograph as well as a LoRA — a latent mold that pulls that
rectangle toward that face during sampling, which is worth having on a platform
whose other half is a trainer. Drop a photo on the bare canvas instead and it
becomes the **scene**: the picture is generated inside it, with lighting,
perspective and shadows integrated rather than the subjects pasted in. A second
tile takes an outfit. Both need the Krea 2 identity-edit weight, so without it
they are dimmed rather than hidden — a weight-gated control is a purchase you
have not made yet, and hiding it hides the decision rather than the capability.

**LoRAs are written in the prompt.** `<lora:my_style:0.8>`, the syntax anyone who
has trained these models already types. Strength defaults to 1 and the token
sits where the LoRA applies, so a fifth LoRA costs the canvas nothing — the rows
this replaced cost 380px of it for four filenames. `+ LoRA` still opens a
picker, because you cannot type a syntax you have never seen.

**Shape and resolution are one control, not two.** Every aspect preset used to be
1024-based, so picking 16:9 chose a shape *and* silently chose ~1 MP — and the
only route to the same shape at 2K was arithmetic in two boxes at the far end of
Advanced. One button now shows what it resolved to (`16:9 · 2016×1152`), with the
ratios as proportioned rectangles and the scale as a separate row. The buckets
are multiplied rather than recomputed, because Krea 2 inherits Qwen-Image's
trained sizes and the honest arithmetic for 4:3 at a 1024 short edge is a size
nothing was trained on.

**There is no Advanced drawer.** "Advanced" names where something is rather than
what it does, and behind it sat five controls that are not advanced — they are
rarely changed. Sampler, steps, guidance and shift live behind one button that
shows the values it resolved to, and it draws only the rows the chosen model
reads: MiniMax-H3 is guidance-distilled, so it gets no CFG row at all.

**The negative prompt is a mode on the prompt field.** A small marker in the
corner, and only on models that read one — Krea 2 Turbo is distilled to CFG 1.0,
where a negative prompt is not weak but unread. The gate is the effective CFG
rather than the checkpoint's name, so raising CFG brings the control back. A dot
appears when there is text on the other side, because otherwise the negative is
invisible from the positive.

**The console has a budget: 30% of the viewport.** Everything else in it is fixed
or conditional, so the prompt field is the only part that grows without asking —
and it is the part that yields. It takes whatever the budget has left, down to a
two-line floor, and re-measures when the region bar or the pill rail appears.

**It is designed for a tablet in portrait, and desktop inherits.** Below 1024px
the layout stacks, the gallery crops to a 1:1 grid, and the last generation
becomes a thumbnail beside Generate — the Camera app's arrangement, because you
press one and then want the other. Three faults found that way had been live on
desktop for months, including a drag that was broken on trackpads specifically.

**Nothing sits on top of a render.** Animate and As reference are icons under the
bottom-left corner that appear on hover. Regional boxes come off the picture the
moment a render lands and are reached again through a map of them in the console
— one control the same size at eight boxes as at one.

**Copy is a last resort — but a number is not a value it can show.** Design
first, then an icon, then words. A control that shows its own value gets no
label; the two keyframe tiles put the mark where the frame sits in the clip
rather than captioning themselves "first" and "last". Hyperparameters are the
exception, and they are the exception on purpose: "32" is a rank, an alpha, an
epoch count or a seed with equal plausibility, so every numeric field carries
its name and the tooltip says what the number *does*.

![Everything you have made, in one grid](docs/gallery.png)

**Your work stays beside your work.** The gallery is a drawer next to the canvas,
not a destination you leave the studio to visit, because the still you made an
hour ago is raw material for the clip you are making now. Any image can go
straight back to the prompt, or become the first frame of a video, without a
download and a re-upload. Open it full-width when you want the whole room.

![Captioning is a workspace, not a batch job](docs/dataset.png)

**Datasets are for reading, not just uploading.** Captions are written in prose
by Qwen3-VL-8B, because the text encoders these models use parse grammar — "red
dress, blue jacket" cannot say which garment is which, and a sentence can. The
panel beside the contact sheet reads the set back to you: trigger-word coverage,
caption length, duplicates, and the clauses your captions repeat, so you can see
what the LoRA is about to learn by accident.

**A preset is what the caption leaves out.** Whatever the captions name is what
the model learns to vary, and whatever they never name is what the trigger word
ends up owning — so **Character** describes pose, wardrobe, framing and light
and refuses to describe a face, **Style** describes the content and never the
look, and **Concept** describes everything around the thing you are training.
Each also names the flaws worth prompting away later: a watermark, a harsh
flash, a hand at the edge of frame. Pick the intent; the instruction behind it
lives on the server, so the run is reproducible from the job record.

Beside it is a captioner picker, because a refusal here is not an error. The
stock model declines on photographs of real people often enough to matter, and
what comes back is a fluent sentence that would land in a `.txt` sidecar and
train. Declines are detected and never written, and the second entry is the
same checkpoint with the refusal direction removed.

---

## Install

```bash
pip install modal
modal setup
modal deploy app.py
```

That is the whole install. The last command prints a URL, and the URL is the
application: interface, API and GPU jobs. Nothing runs on your machine, nothing
runs when you are not using it, and there is no config file to fill in first.

---

## What runs underneath

**Train.** LoRA training for Krea 2 on [musubi-tuner](https://github.com/kohya-ss/musubi-tuner).
Point it at a folder of images, get a `.safetensors` back.

**Caption.** Datasets are named folders of images with `.txt` sidecars beside
them, which is exactly what the trainer reads.

**Generate stills.** Krea 2 inference through the same driven ComfyUI the video
side uses, with LoRA stacking and regional multi-character LoRA — a box per
character, each LoRA masked to its own rectangle so two trained identities do
not blend into one another. Drop a photo in and the scene is regenerated around
the boxes instead of the subjects being pasted into it.

**Generate video.** Two families through a driven ComfyUI:

|                | MiniMax-H3              | Wan 2.2                    |
| -------------- | ----------------------- | -------------------------- |
| Audio          | yes, same latent        | silent                     |
| CFG / negative | no — guidance-distilled | yes                        |
| LoRAs          | no                      | yes                        |
| References     | ref2va checkpoint       | no                         |
| Experts        | one                     | two on A14B, one on the 5B |

Adding Wan did not add a backend. It reuses the container, the warm ComfyUI
process and the job contract; what is per-family is a graph builder and one row
of capabilities — which is also the row the composer reads to decide what to
show you.

---

## Requirements

- A Modal account. `modal setup` walks you through auth in a browser.
- Python 3.10+ locally, only to run the `modal` CLI.
- A [HuggingFace](https://huggingface.co) account **if** you want Krea 2 — its
  weights are gated. Everything else downloads without one.

You do not need a local GPU, Docker, a `.env` file, or any Modal Secret. The
HuggingFace token is pasted into the UI and stored in a Modal Dict.

---

## First run: get the weights

**Nothing downloads on its own.** A fresh deployment has an empty volume and
every model is opt-in, because the full catalogue is ~206 GB and almost nobody
wants all of it. Open the deployed URL, click the gear, and pick what you need.

| Family                       |   Size | Gated | What it buys                              |
| ---------------------------- | -----: | ----- | ----------------------------------------- |
| Krea 2 — images              |  62 GB | yes   | training + still generation               |
| MiniMax-H3 — video           |  64 GB | no    | video **with a soundtrack**, references   |
| Wan 2.2 — video              |  76 GB | no    | silent video, CFG, LoRA support           |
| Wan 2.2 speed LoRAs          |   5 GB | no    | fewer steps per clip                      |
| Krea 2 style LoRAs           |   4 GB | no    | Krea's own nine styles, for the prompt    |

The style LoRAs are the cheapest way to see regional prompting actually work.
Two character LoRAs in two boxes produce a picture of two people, and nothing
in that picture distinguishes "each LoRA was masked to its rectangle" from
"the model drew two people". Two *styles* do: ink wash on one side, motion blur
on the other and a hard seam between them is the masking, visible.

You do not need a whole family. The smallest useful video setup is Wan 2.2
TI2V 5B at **18 GB** — the 5B checkpoint, umT5-XXL and the 2.2 VAE — which does
both text-to-video and image-to-video on its own.

Downloads run on **CPU containers**, never on a GPU. Pulling 26 GB while an
A100 idles is money burned for nothing.

A transfer reports the bytes it has and the rate it is getting them at, and if
it goes quiet for four minutes it is abandoned and **resumed** from where it
stopped, up to five times. Both exist because of one failure: a 17 GB pull
stopping dead at 4 GB and the job staying "running" — no error, no log line, no
byte count — until the four-hour timeout collected it. A download that can hang
is survivable; one that can hang silently costs you the four hours before you
learn anything.

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

- A file link or a bare id both work; so does a folder link.
- Only `.safetensors` is kept. A folder's preview grid and readme are named as
  skipped rather than quietly copied onto the volume.
- Leave **folder** blank and the files drop in loose, each its own entry. Give
  one and they are grouped as versions of a single LoRA under
  `loras/{folder}/` — which is right for a matched pair and wrong for a bag of
  unrelated ones, so it stays your call.
- The link has to be shared with anyone who has it. Drive answers an unshared
  file with a sign-in page rather than an error, so the failure names that case
  explicitly instead of surfacing a parse error.

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

The layout is the contract, not the code. Datasets are folders of images with
text files beside them — the same thing the trainer reads — so nothing here is
required to get your data back out.

Run a second, isolated copy against its own storage by setting the volume name:

```bash
VISIONARY_VOLUME=visionary-test modal deploy app.py
```

---

## GPUs and what they cost you

Each job type picks its own class, and most are switchable in the UI:

| Job              | Default   | Options                        |
| ---------------- | --------- | ------------------------------ |
| Training         | A100-40GB | —                              |
| Captioning       | A100-40GB | —                              |
| Image generation | H100      | H100, H200                     |
| Video generation | H100      | H100, H200                     |

Image generation was an A100-40GB until it moved onto ComfyUI. Both inference
paths now share one image, and its SageAttention kernels are compiled for
Hopper — an A100 would load the weights, find no kernel, and quietly run slow.
The regional path wants the headroom regardless.

Containers stay warm between requests (10 minutes for images, 15 for video) so
consecutive takes skip the model load, then scale to zero. You are billed for
GPU time while a job runs and while a container is warm — not for the
deployment sitting idle.

Anything that can fail cheaply does. A bad LoRA path, an unknown aspect ratio
or a missing weight is rejected on CPU in milliseconds, before a GPU is rented.

---

## Verifying a deployment

Two smoke tests, both cheap, both runnable against your own account:

```bash
modal run tools/smoke_graphs.py
```

Checks every graph the app can build — the three Krea 2 shapes and all twelve
video variants across both families — against the real ComfyUI node schema on a
**CPU** container with no weights present. Catches a renamed node, a moved
input, a dangling link, a sampler the UI offers that ComfyUI does not have, and
a custom node that failed to import. It does not run a sampler, so it says
nothing about whether the picture looks right.

```bash
modal run tools/smoke_caption.py
```

Checks that the pinned transformers has the class and that **every** repo id in
the captioner picker resolves and parses, on a CPU container that downloads
config files rather than weights. `--gpu` loads one and captions a real image;
`--model` and `--preset` choose which captioner and which instruction, and the
result says whether the model refused.

```bash
python3 tools/smoke_prompt.py
```

Checks the shot compiler against the format MiniMax published: the alignment
sentences verbatim for each of the four tasks, the three field labels once each
in order, and a line of dialogue with commas, an ellipsis and a trailing
exclamation surviving byte for byte inside `<d>…</d>`. Pure stdlib and no
network — it reads the real compiler out of `app.py` by AST rather than
importing it, because importing `app.py` builds Modal image definitions at
module scope and wants credentials to answer a question about a string.

### What has actually been run end to end

Being honest about coverage, since "it deploys" is not "it works":

- **Wan 2.2 TI2V 5B** — text-to-video and image-to-video both verified on an
  H100, output inspected frame by frame.
- **MiniMax-H3** — text-to-video run end to end on an H100 and clips returned.
  The shot compiler's output has been checked against the published format by
  `smoke_prompt.py`, and `/api/compile` shows the same string the run is given.
  What is still unverified by ear is the audio: whether
  `non_diegetic_music: N/A` actually silences the invented soundtrack is an
  observation nobody has written down yet.
- **Wan 2.2 A14B** — graphs validate structurally. The two-expert handover
  cannot be checked structurally: wrong noise flags give a washed-out clip
  rather than an error, so only a real run will show it.

---

## Working on the UI locally

The front end is one self-contained string in `app.py` with no build step, so
it can be served locally against stubbed JSON instead of paying an image build
and a cold start per CSS change:

```bash
python3 tools/preview_ui.py 8777
```

The stubs are shaped to hold the awkward states — a missing model, an
uncaptioned dataset, a prompt too long to belong in a gallery card.

---

## Layout

```
app.py              the whole application — images, jobs, API, and the UI
comfy_nodes/        our own ComfyUI nodes — one shim, see visionary_boxes
tools/              smoke tests and the local UI preview server
tools/_from_app.py  pulls plain-Python pieces out of app.py by AST
CLAUDE.md           the design rationale — why the code is shaped the way it is
```

`_from_app.py` exists because two tools need the *real* thing rather than a
copy: a compiler checked against a reimplementation is checking the
reimplementation, and a palette previewed from a hand-written vocabulary is a
preview of a palette that does not exist.

`app.py` is deliberately one file. It is long, but the alternative — a package
whose modules are imported by Modal image builds — trades one long file for a
build-order problem, and the file is navigable by its banner comments.

If you are going to change anything, read `CLAUDE.md` first. It explains the
tradeoffs the code is holding, including several that look like mistakes until
you know what they are avoiding.

---

## Licensing

**[AGPL-3.0](LICENSE).** Worth understanding before you fork this or run it for
anyone but yourself.

This used to be an inheritance rather than a choice: `forge/` was a vendored
slice of [sd-webui-forge-classic](https://github.com/Haoming02/sd-webui-forge-classic),
which is AGPL-3.0, imported and executed on the image path. That tree is gone —
see CLAUDE.md for why — so the AGPL now comes from this repository's own
[LICENSE](LICENSE) and not from a dependency. AGPL-3.0 is strong copyleft with
a network-use clause: section 13 means that if you modify this and let other
people use it over a network, you owe those users the corresponding source —
deploying rather than distributing is not the loophole it is under the GPL.
Since this deploys as a web application by design, that clause is the normal
case here, not an edge one. Running your own private instance triggers nothing.

What the images now install, rather than vendor:

- **[ComfyUI](https://github.com/Comfy-Org/ComfyUI)** — GPL-3.0. Cloned into
  the container at the commit in `COMFY_SHA`, run as its own process, and
  driven over its HTTP API. Nothing here is linked against it or patched.
- **[Krea2 Regional Multi-LoRA](https://github.com/CliffNodes/Krea2-Multi-Character-Lora-Node-with-bounding-box-Scene-and-Outfit-Edit)**
  — MIT. Cloned at `CLIFF_SHA` into ComfyUI's `custom_nodes/`, unmodified.

None of that is legal advice, and the combination is worth a look of your own
if you plan to distribute this or run it for other people.

Model weights carry their own separate licences — Krea 2's in particular is
gated and has terms you accept on HuggingFace. Nothing here grants you rights
to them.
