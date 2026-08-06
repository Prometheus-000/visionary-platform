# Visionary

A single-user LoRA training and generation platform on Modal. `modal deploy app.py`
gives you one URL that is the whole application — UI, API and GPU jobs.

## Philosophy

Three words, in priority order when they conflict.

### Antifragile

A failure should teach you something and leave the system better able to survive
the next one. Concretely:

- **Errors diagnose themselves.** `_require_models()` does not say "not
  downloaded" — it prints the resolved volume name, the exact path it wanted,
  and what is actually on the volume, because those three facts distinguish a
  wrong Modal profile from a filename typo from a partial download. Any error a
  user can hit twice is an error that should have explained itself the first time.
- **Destructive is opt-in.** Culling an image moves it to `.trash/`. A mis-click
  costs a file move, not the file.
- **Stops are cooperative.** Jobs check a flag between steps and unwind cleanly,
  so the container survives and the next request is warm. Killing the process
  is what you do when there is no other lever, not the default.
- **Pin to what you can reproduce.** A commit SHA, not a branch or a floating
  ref. When upstream force-pushes, your build should not change under you.

### Scalable

Scale here means the axis that actually binds: not requests per second, but
**dataset size, model size, and cost per job**.

- **Never rent a GPU to do CPU work.** Downloads, uploads, thumbnails and
  validation run on CPU containers. Pulling 26 GB while an A100 idles is money
  burned. `_validate_loras()` is deliberately importable from the web container
  so a bad path is a form error in milliseconds, not a cold start and 35 GB of
  weight loading before it fails.
- **Keep the polled thing small.** Job records carry filenames; bytes are served
  off the volume by their own route. A dict polled every two seconds must never
  grow with the size of the result.
- **Separate storage by commit cost.** The HF cache lives on its own volume
  because committing 17 GB of model weights after writing 80 captions turned an
  eight-minute job into a twenty-three minute one. Storage boundaries follow
  write patterns, not tidiness.
- **One container per loaded checkpoint.** `max_containers=1` on GPU classes: a
  second replica pays a full cold load rather than sharing a warm one.

### Future-proof

Prefer the surface that will still be there, and that carries the *next* model
in for free.

- **Depend on maintained upstreams over owned forks.** Vendoring is a last
  resort, and when it happens it is recorded — see `forge/VENDOR.md`, which
  pins the source SHA and lists every local change, so a sync is a diff rather
  than an archaeology exercise.
- **Storage layout is the contract, not the code.** Datasets are folders of
  images with `.txt` sidecars beside them — the same thing the trainer reads.
  Nothing here is required to get your data back out.
- **Separate images when pins conflict.** Trainer, inference and captioning are
  three images because one shared `transformers` pin would mean every captioner
  bump re-litigates training. A dependency conflict resolved by compromise is a
  conflict you pay for forever.
- **Do not build a second way to do the first thing.** New capability extends
  the existing job/status/stop contract rather than inventing a parallel one.

## Layout

    app.py              the whole application — images, jobs, API, and UI_HTML
    forge/              vendored sd-webui-forge-classic backend (see VENDOR.md)
    ai-toolkit/         training reference
    tools/              smoke tests, the local UI preview

`app.py` is deliberately one file. It is long, but the alternative — a package
whose modules are imported by Modal image builds — trades one long file for a
build-order problem, and the file is navigable by its banner comments.

## Storage

    $VISIONARY_VOLUME (default "visionary")  ->  /workspace
      models/       weights, flat, descriptive filenames, addressed by exact path
      loras/{folder}/{name}.safetensors   trained output, any nesting
      loras/{name}.safetensors            loose files count too — see below
      datasets/{name}/  images + .txt caption sidecars — sets you saved
      drafts/{name}/    identical shape; sets you have not saved yet
      outputs/{job}/    generated media
      work/, .cache/    disposable

Set `VISIONARY_VOLUME` to run a second copy against its own storage.

### A folder is a LoRA; a loose file is also a LoRA

Training writes `loras/{name}/`, so a folder is one LoRA and the checkpoints in
it are that LoRA's epochs. Anything arriving another way — migrated off an older
volume, pulled off Google Drive, dropped in by hand — is a bare file at the top
level, and the folders-only walk skipped it silently. Four real LoRAs sat on the
volume invisible to the picker, which reads as "training never produced
anything" rather than "this listing has an opinion about directory layout." A
file a run can load is a file the picker has to offer.

This is also why a `<lora:…>` token resolves by the **shortest unambiguous
name**. One `k3nan.safetensors` on the volume is `<lora:k3nan:1>`; the matched
Wan speed pairs, whose files are both called `high` and `low`, are qualified by
their folder. When a name matches more than one file the note says which ones,
because "no LoRA named high" sends you looking for a file that is sitting right
there.

### Saving a set is a choice, and it is the only thing `drafts/` means

Dropping images makes a **draft**. It captions, filters and trains exactly like a
saved set — same folder, same sidecars, same code path — and the only difference
it has is which parent it sits under. Saving moves the folder into `datasets/`
under the name you type; the page never asks for one before the images are in
front of you, because "is this worth keeping" is not a question you can answer at
drop time. Most sets are dropped once to answer one question, and making each of
those a permanent named entry taxes the common case to serve the rare one.

A draft belongs to the window that made it. The page holds an id in
`sessionStorage` — surviving a reload, dying with the tab — and heartbeats it to
`/api/session`; a draft whose session has been quiet for fifteen minutes is
swept. There is no server-side "app closed" event to use instead: the web
container scales to zero on Modal's schedule, not yours, so a cold start would
be a lifecycle signal that means nothing about whether you are still working.

Sweeping and deleting both **move to `.trash/`**, per root. Deleting a set used
to `rmtree`, which made it the one unrecoverable action in an application where
culling a single image is not.

## Conventions

- **Comments explain why, not what.** Every non-obvious line in this codebase
  earns its comment by naming the failure that produced it. If a comment could
  be deleted without losing a fact, delete it.
- **No `from __future__ import annotations`.** It broke FastAPI's
  `get_type_hints()` against module globals and turned `/api/upload` into a 422.
  See the note at the top of `app.py`.
- **No Modal Secrets, no CLI setup.** The HF token is pasted into the UI and
  stored in a Modal Dict. `modal deploy app.py` is the entire install.
- **Nothing downloads on its own.** Weights are chosen explicitly, under the
  gear.
- **Prose, not tags.** Captions are sentences, because the text encoders these
  models use parse grammar. See the `CAPTION_MODEL` comment.

## The page

The UI is not organised the way this file is. There are three subsystems and
two domains, and the page follows the domains.

- **The canvas is the largest thing on screen, always.** Options live in a bar
  under it, never a rail beside it: a settings column costs the picture 384px
  of the one dimension it cannot get back, and vertical is the cheap axis. The
  bar is capped so its fullest state cannot push the canvas out of frame, and
  anything sized to fit the canvas measures the canvas — a `dvh` sum is wrong
  the moment the bar grows.
- **Generate is the page, not a destination.** It has no nav item. Train is one
  door, labelled with where it leads rather than where you are, so two things
  never look equally selected. It carries the training run's progress, because
  a run lasts hours and you are meant to leave and keep working.
- **Image and video are one workspace.** Shared prompt, canvas and gallery; the
  switch is a chip inside the prompt field and the prompt survives it. What
  differs is only the options, which rebuild from `VIDEO_MODELS` — see below.
- **Copy is a last resort — but a number is not a value it can show.** Design
  first, then an icon, then words. A control that shows its own value gets no
  label: "Krea 2 Turbo", "16:9", "720p", "5s" name themselves. Twice the icon
  was not enough and the design changed instead of a caption being added:
  keyframe tiles mark where the frame sits in the clip, and a tile that appears
  replaced the checkbox that used to reveal it.

  The hyperparameters were where this rule was pushed past what it can carry.
  "32" is a rank, an alpha, an epoch count or a seed with equal plausibility,
  and the icons standing in for those words failed the only test that matters:
  someone who has trained these models for five years had to hover every
  numeric field to find out what it was. An icon is a rebus for a word you
  already know — it cannot tell you *which* hyperparameter you are looking at.
  So every numeric field carries its name, and the tooltip is promoted from
  repeating that name to saying what the number does. The rule survives, with
  its scope corrected: a control that shows its own value gets no label, and a
  bare number is not a value.

- **LoRAs are written in the prompt, not stacked under it.**
  `<lora:name:0.8>`, Automatic1111's syntax, because it is the notation anyone
  who has trained these models already types. Strength defaults to 1; a second
  number is the text encoder weight on the image side and the Wan expert on the
  video side, and both are omitted far more often than not.

  A row per LoRA cost 56px plus a wrapped select — 380px of canvas for four
  filenames and eight digits — and it still could not say the thing that
  matters most, which is where in the sentence the LoRA applies. In the prompt
  a fifth LoRA costs the canvas nothing, and `+ LoRA` survives as a picker that
  writes a token at the caret, because you cannot type a syntax you have never
  seen. What the prompt cannot show — a name that resolves to no file, a stack
  past `MAX_LORAS`, a model that reads no LoRAs at all — is the only thing the
  note under the field ever says.

- **The ratio picker and the pixel boxes are one control.** There is only ever
  a width and a height; the ratios are shortcuts to a pair of them. Picking one
  writes the boxes, typing in the boxes selects Custom, and Custom is the only
  option that spells out its own size, because it is the only one with no name
  that implies it. Sizes snap to 16 on the way out — the pipeline floors to the
  patch grid regardless, and a box that keeps 1000 while the model renders 992
  is a box that lied to you.

`tools/preview_ui.py` serves `UI_HTML` against stubbed JSON, so the front end
is worked on locally instead of paying an image build and a cold start per CSS
change. Its stubs are shaped to hold the awkward states — a missing model, an
uncaptioned dataset, a prompt too long to belong in a gallery card.

## Phases

1. Krea 2 LoRA training (musubi-tuner) — done
2. Image inference (vendored Forge backend) + datasets and captioning — done
3. Video inference via ComfyUI — done
   - MiniMax-H3: t2v, i2v, first/last frame, ref2va, native soundtrack
   - Wan 2.2: A14B t2v/i2v/first-last and TI2V 5B, with LoRA stacking
4. Video LoRA training — Wan 2.2 is the target, which is why phase 3 loads LoRAs

The end state is one application where a generated still flows into a clip
without a round trip through the filesystem — the "Animate" and "As reference"
buttons on a finished image are the first piece of that.

### Two video families, one path

Adding Wan did not add a backend. It reuses the container, the warm ComfyUI
process, the job/status/stop contract and the output layout; what is per-family
is a graph builder and a row in `VIDEO_MODELS`. That is the payoff of driving
ComfyUI rather than porting its model code, and it is the shape any third
family should take.

They are not interchangeable, and the UI says so rather than averaging them:

|                | MiniMax-H3           | Wan 2.2                     |
| -------------- | -------------------- | --------------------------- |
| Audio          | yes, same latent     | silent                      |
| CFG / negative | no — guidance-distilled | yes                      |
| LoRAs          | no ecosystem for the int8 repackage | yes      |
| References     | ref2va checkpoint    | no                          |
| Experts        | one                  | two on A14B, one on the 5B  |

`VIDEO_MODELS` is served to the page, so the composer shows only the controls
the chosen model reads. A control that is present but ignored is worse than one
that is absent — it is the UI making a promise the model will not keep.

The A14B pair is the one thing with no image-side analogue: it is *two*
checkpoints split by noise level, sampled in sequence by two `KSamplerAdvanced`
nodes handing an unfinished latent over. So a video LoRA row carries an expert,
and the `wan22-speed-*` folders hold a matched `high`/`low` pair.

### The open question: is `forge/` redundant?

ComfyUI supports Krea 2 natively (`Krea2` in `comfy/supported_models.py`,
`comfy.text_encoders.krea2`, shift 1.15 — the same value `forge/` defaults to),
against the same weights already on the volume. So the image path *could* move
to the backend video already runs on, leaving one backend, one image and one
GPU class.

What stops it being a rename: regional prompting. `forge/krea2/regional.py`
masks attention inside Krea 2's single-stream DiT, which needed a vendor patch
to `backend/nn/krea.py`, and it exists because Forge Couple's cross-attention
design cannot reach a single-stream model at all. ComfyUI has the same
architectural problem, so that feature would have to be rebuilt, not ported.

That migration is worth doing and is *not* worth doing in the same change as
anything else: if images regress afterwards, the cause should be unambiguous.
Note also that `forge/` deliberately installs neither sageattention nor
flash_attn, because both assert `mask is None` and would silently disable
regional prompting — so the two backends want different attention builds and
cannot share one image without losing something.
