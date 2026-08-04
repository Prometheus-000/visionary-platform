# Visionary

A single-user LoRA training and generation platform that runs entirely on
[Modal](https://modal.com). Train a LoRA on your own images, generate stills
with it, and turn those stills into video — one deployment, one URL, no
infrastructure to keep alive between sessions.

```bash
pip install modal
modal setup
modal deploy app.py
```

That is the whole install. The last command prints a URL, and the URL is the
application: UI, API and GPU jobs. Nothing runs on your machine, nothing runs
when you are not using it, and there is no config file to fill in first.

---

## What it does

**Train.** LoRA training for Krea 2 on [musubi-tuner](https://github.com/kohya-ss/musubi-tuner).
Point it at a folder of images, get a `.safetensors` back.

**Caption.** Datasets are named folders of images with `.txt` sidecars beside
them. Captioning uses Qwen3-VL-8B-Instruct and writes prose rather than tags,
because the text encoders these models use parse grammar.

**Generate stills.** Krea 2 inference on a vendored
[sd-webui-forge-classic](https://github.com/Haoming02/sd-webui-forge-classic)
backend, with LoRA stacking and regional prompting.

**Generate video.** Two families through a driven ComfyUI:

|                | MiniMax-H3              | Wan 2.2                    |
| -------------- | ----------------------- | -------------------------- |
| Audio          | yes, same latent        | silent                     |
| CFG / negative | no — guidance-distilled | yes                        |
| LoRAs          | no                      | yes                        |
| References     | ref2va checkpoint       | no                         |
| Experts        | one                     | two on A14B, one on the 5B |

The two are not interchangeable, and the composer only shows the controls the
chosen model actually reads — a control that is present but ignored is worse
than one that is absent.

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

You do not need a whole family. The smallest useful video setup is Wan 2.2
TI2V 5B at **18 GB** — the 5B checkpoint, umT5-XXL and the 2.2 VAE — which does
both text-to-video and image-to-video on its own.

Downloads run on **CPU containers**, never on a GPU. Pulling 26 GB while an
A100 idles is money burned for nothing.

### Gated weights

Krea 2 RAW and Krea 2 Turbo need a HuggingFace token, and you must accept the
licence with the same account that issued it:

- <https://huggingface.co/krea/Krea-2-Raw>
- <https://huggingface.co/krea/Krea-2-Turbo>

Paste the token on the Models panel. If the licence has not been accepted, the
error says so and links the page rather than failing as a generic 403.

---

## Storage

One Modal Volume, mounted at `/workspace`:

```
models/            weights, flat, addressed by exact filename
loras/{folder}/    one folder per trained LoRA
datasets/{name}/   images + .txt caption sidecars
outputs/{job}/     generated media + a visionary.json sidecar
work/, .cache/     disposable
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
| Image generation | A100-40GB | A100-40GB, A100-80GB, H100     |
| Video generation | H100      | H100, H200                     |

Containers stay warm between requests (10 minutes for images, 15 for video) so
consecutive takes skip the model load, then scale to zero. You are billed for
GPU time while a job runs and while a container is warm — not for the
deployment sitting idle.

Anything that can fail cheaply does. A bad LoRA path, an unknown aspect ratio
or a missing weight is rejected on CPU in milliseconds, before a GPU is rented.

---

## Training from the command line

No browser needed:

```bash
modal run app.py --images-dir ./photos --lora-name my_style
```

Add `--caption` to caption the folder first. The dataset is named after the
LoRA, so a run started here shows up on the Datasets tab and can be reused.

---

## Verifying a deployment

Two smoke tests, both cheap, both runnable against your own account:

```bash
modal run tools/smoke_video.py
```

Checks every video graph — all twelve variants across both families — against
the real ComfyUI node schema on a **CPU** container with no weights present.
Catches a renamed node, a moved input, a dangling link, and a sampler the UI
offers that ComfyUI does not have. It does not run a sampler, so it says
nothing about whether a clip looks right.

```bash
modal run tools/smoke_krea2.py --gpu --lora any
```

Exercises the Krea 2 loader, the LoRA stack and the regional prompting path.

### What has actually been run end to end

Being honest about coverage, since "it deploys" is not "it works":

- **Wan 2.2 TI2V 5B** — text-to-video and image-to-video both verified on an
  H100, output inspected frame by frame.
- **MiniMax-H3** — graphs validate structurally; no full run yet, so the audio
  path is unproven.
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
app.py     the whole application — images, jobs, API, and the UI
forge/     vendored sd-webui-forge-classic backend (see forge/VENDOR.md)
tools/     smoke tests and the local UI preview server
CLAUDE.md  the design rationale — why the code is shaped the way it is
```

`app.py` is deliberately one file. It is long, but the alternative — a package
whose modules are imported by Modal image builds — trades one long file for a
build-order problem, and the file is navigable by its banner comments.

If you are going to change anything, read `CLAUDE.md` first. It explains the
tradeoffs the code is holding, including several that look like mistakes until
you know what they are avoiding.

---

## Licensing

**[AGPL-3.0](LICENSE).** Not a preference — an inheritance, and worth
understanding before you fork this or run it for anyone but yourself.

`forge/` is a vendored slice of
[sd-webui-forge-classic](https://github.com/Haoming02/sd-webui-forge-classic),
which is AGPL-3.0, and it is imported and executed as part of the image
generation path rather than sitting there unused. AGPL-3.0 is strong copyleft
with a network-use clause: section 13 means that if you modify this and let
other people use it over a network, you owe those users the corresponding
source — deploying rather than distributing is not the loophole it is under
the GPL. `forge/modules_forge/packages/comfy/` carries GPL-3.0 on top of that.

Since this deploys as a web application by design, that clause is the normal
case here, not an edge one. Running your own private instance triggers nothing.

`forge/VENDOR.md` records the exact upstream commit and every local change, so
a sync is a diff rather than an archaeology exercise.

Model weights carry their own separate licences — Krea 2's in particular is
gated and has terms you accept on HuggingFace. Nothing here grants you rights
to them.
