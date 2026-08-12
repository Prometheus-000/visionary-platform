# Visionary — Product Brief

**Launch release.** A single-user LoRA training and generation studio that runs
on your own Modal account. `modal deploy app.py` gives you one URL that is the
whole application — UI, API and GPU jobs.

---

## 1. The Core Problem & Value Proposition

### The problem, stated three ways

**Operationally: the pipeline is three tools that do not know about each other.**
Training a LoRA on your own photographs, generating stills with it, and animating
one of those stills into a clip are, today, three installs — a trainer with its
own dataset format, a node-graph UI with its own model directory, and a video
stack with a third set of weights. Each has its own conventions for where a
caption lives and what a checkpoint is called. The work of moving between them is
filesystem work, and it is paid on every iteration. A generated still that wants
to become a clip goes out to disk and comes back in by hand.

**Economically: the GPU is rented for the wrong work.** A persistent box sized
for a 26 GB DiT idles at that size while it pulls files, writes thumbnails and
validates a path. The alternative — hosted services — solves the idle problem by
taking the weights away from you: per-seat billing, your images on someone else's
storage, and no way to get a model out that you trained.

**Artistically, and this is the one that matters: the unit of authorship is a
string, and strings transfer.** Prompt engineering is a programming skill wearing
an artist's clothes. Because a prompt is copy-pasteable, somebody else's method
transfers whole — take their string, get their results, generate endless
variations in their voice without ever having their taste. The public proof is
Civitai, where every image looks the same. Not because the models cannot do
anything else, but because the prompt *is* the method, methods get shared, and a
shared method converges. That is what an aesthetic looks like when its unit of
authorship can be pasted.

There is a second, quieter version of the same failure. Every model arrives with
its own grammar — MiniMax-H3 wants a six-field document published in its repo,
Wan wants prose, Krea 2 wants prose with the camera clauses dropped. Asking a
person to hold three formats in their head so a checkpoint can be fed correctly
is the tool making its problem into theirs. A documented grammar presented as a
blank textarea reads as superstition: whether a comma or "the woman" versus "a
woman" changes the take is not something anyone can infer, and a video take is
two to three minutes, so every guess is paid for at that rate.

### Why now

Three things crossed a line at roughly the same time:

- **Open weights got good enough to own.** Krea 2 for stills, MiniMax-H3 for
  video that carries its own soundtrack in the same latent, Wan 2.2 for video
  that takes LoRAs. These are not demos — they are checkpoints a single person
  can train against and ship work from.
- **Serverless GPU made a single-user studio economically sane.** Per-second
  billing and scale-to-zero mean there is nothing to keep alive between sessions.
  The install is a deploy; the bill is the work.
- **The regional-LoRA problem got solved by somebody else, better.** Masking a
  single-stream DiT's attention used to require a vendored fork. A node pack now
  does it through public hooks and does it more strongly — multiplying each
  LoRA's activation delta by zero outside its box, so there is no pathway left
  for one character's identity to reach another's. The last reason to maintain a
  fork went away, and with it the last reason this had to be a research project
  rather than a product.

### The value proposition

**Own the weights, not the prompt.** Authorship at the prompt is thin — it is the
last few centimetres of a pipeline somebody else built. Authorship in a LoRA
trained on your own photographs is the model itself learning what you meant. That
is why training is Phase 1 here rather than an advanced feature bolted on later:
the trainer is in the application, the LoRA picker reads the volume rather than a
registry, and a dataset is a folder of your images with your captions beside
them. A conversation and a set of weights do not transfer the way a string does —
not because they cannot be copied, but because copying them gives you a record of
somebody else's session rather than an instrument you can point at your own.

**One deploy, one URL, no setup.** No Modal Secrets, no CLI configuration, no
database. The HuggingFace token is pasted into the UI and stored in a Modal Dict.
Nothing downloads on its own — weights are chosen explicitly.

**Your data stays a folder.** Datasets are images with `.txt` caption sidecars
beside them, which is exactly what the trainer reads. There is no export step and
no schema to fall out of sync with the files. Storage layout is the contract, not
the code — nothing here is required to get your data back out.

**The app knows what each model needs, so you do not.** One shot vocabulary
compiles to three different grammars. Only the controls a chosen model actually
reads appear on screen, because a control that is present but ignored is worse
than one that is absent — it is the interface making a promise the model will not
keep.

**Cost discipline is a design rule, not an optimisation pass.** Downloads,
uploads, thumbnails and validation run on CPU containers. Pulling 26 GB while an
A100 idles is money burned, and a bad LoRA path is a form error in milliseconds
rather than a cold start and 35 GB of weight loading before it fails.

---

## 2. Current Feature Architecture

Four phases are complete and in this release. The system is one Modal app defined
in one file; what follows is what that file contains.

### How it is built — three words, in priority order when they conflict

**Antifragile.** A failure should teach you something and leave the system better
able to survive the next one. Errors diagnose themselves: the missing-model error
prints the resolved volume name, the exact path it wanted, and what is actually
on the volume, because those three facts are what distinguish a wrong Modal
profile from a filename typo from a partial download. Any error a user can hit
twice is an error that should have explained itself the first time. Destructive
operations are asked for rather than softened — deletion unlinks, and the confirm
dialog that says what is going and how much of it is the safety net. A per-root
`.trash/` lived here for a while on the theory that a mis-click should cost a
file move rather than the file, but nothing ever surfaced it: it was not an undo
anyone could reach, it was a second copy of everything already thrown away, on a
volume whose only other way to reclaim space is the CLI. Stops are cooperative.
Pins are commit SHAs rather than branches, so an upstream force-push cannot
change your build under you.

**Scalable.** Scale here is not requests per second; it is dataset size, model
size and cost per job. Never rent a GPU to do CPU work. Keep the polled thing
small — a record polled every two seconds must never grow with the size of the
result. Separate storage by commit cost. One container per loaded checkpoint.

**Future-proof.** Prefer the surface that will still be there, and that carries
the *next* model in for free. Depend on maintained upstreams over owned forks.
Storage layout is the contract, not the code. Separate images when pins conflict,
and only then — a dependency conflict resolved by compromise is a conflict you
pay for forever, and the converse binds just as hard, which is why images and
video share one image rather than paying for a second CUDA build. Do not build a
second way to do the first thing.

### Runtime and deployment

| | |
|---|---|
| **Shape** | One Modal app, one web URL serving UI + ~38 API routes + GPU jobs |
| **Images** | Three — trainer, inference/ComfyUI, captioner. Separate only because their `transformers` pins genuinely conflict; images and video share one image because nothing in it is per-family |
| **GPU classes** | Two (image, video), H100/H200. Hopper-only is the bill for sharing one image: SageAttention is compiled for sm_90. `max_containers=1` each, so a warm checkpoint is shared rather than reloaded |
| **Volumes** | Two — the workspace, and the HuggingFace cache on its own volume because committing 17 GB of weights after writing 80 captions turned an 8-minute job into a 23-minute one |
| **Pins** | Two upstream commit SHAs (`COMFY_SHA`, `CLIFF_SHA`). Nothing vendored, nothing patched |
| **Multi-instance** | `VISIONARY_VOLUME` runs a second copy against its own storage |

### Storage — the volume is the contract

`models/` is flat and addressed by exact path; `loras/{folder}/` is what training
writes; `datasets/` and `drafts/` are the same shape as each other; `outputs/{job}/`
holds generated media; the rest is disposable scratch.

A folder is a LoRA and the checkpoints in it are that LoRA's epochs — but anything
arriving another way, migrated off an older volume or pulled off Drive, is a bare
file at the top level, and a folders-only walk skipped four real LoRAs silently.
That reads to the user as "training never produced anything" rather than "this
listing has an opinion about directory layout." A file a run can load is a file
the picker has to offer.

### Model catalogue and acquisition

Twenty-two entries across four families, every repo and filename verified against
the HuggingFace API rather than copied from docs.

- **Krea 2 — images.** RAW (26.3 GB, for training) and Turbo (26.3 GB, 8-step
  generation), Qwen Image VAE, Qwen3-VL 4B text encoder in bf16, and an optional
  identity-edit LoRA for scene and outfit transfer.
- **MiniMax-H3 — video with sound.** Two int8 DiTs (fl2va and ref2va, 21 GB
  each), a 32B nvfp4 text encoder, a video VAE and an audio VAE.
- **Wan 2.2 — video.** The A14B high/low-noise expert pair for t2v and i2v, the
  single 5B TI2V DiT, umT5-XXL, and both VAE generations.
- **Wan 2.2 speed LoRAs.** Matched LightX2V 4-step high/low pairs.

Acquisition is a first-class job, not a script:

- **A family downloads itself.** The group is the unit you decide in; the
  per-file queue used to be kept in a person rather than in the program.
- **One download at a time, and being busy is a state rather than an error.**
  Three concurrent pulls measured 4–12 MB/s each against ~31 MB/s for one, so the
  route is idempotent — a second press returns the job the first one started.
- **243.8 MB/s, measured.** `hf_transfer` on, against 30.6 MB/s on the plain
  backend for the same 21 GB file. The resume it costs stopped mattering when the
  largest weight in the catalogue started landing in under two minutes.
- **A long transfer publishes bytes.** A watcher polls bytes on disk and reports
  a count and a rate; a stall is abandoned and resumed, bounded by a retry count.
- **Staged, then moved.** A half-written `.safetensors` is never visible to the
  picker, so it can never be chosen and fail thirty seconds into a warm run.
- **Google Drive route**, on the same job/status/stop contract — most LoRAs worth
  having were never published to HuggingFace; they are a link someone sent you.

### Datasets and captioning

- **Saving is a choice.** Dropping images makes a draft that captions, filters and
  trains identically to a saved set; saving moves the folder under the name you
  type. The page never asks for a name before the images are in front of you.
  Drafts belong to the window that made them and are swept after fifteen minutes
  of silence, with the folder's own mtime counted as a heartbeat.
- **Orientation is resolved on arrival.** An uploaded image carrying an EXIF
  rotation tag is rewritten upright, once, so every downstream consumer — the
  page, the captioner, the trainer's bucketer — sees the same pixels. Fixed in
  each reader instead, this trains rotation into the LoRA.
- **Five caption presets, each one rule inverted.** Character describes pose,
  wardrobe, framing and light and refuses to describe a face; Style describes the
  content and never the look; Concept describes the context around the thing;
  General and Casual sit either side. What a caption *names* is what the model
  learns is free to vary — what it never names is what the trigger word ends up
  owning.
- **A refusal is caught, not written.** A stock instruct model declines to
  describe photographs of real people often enough to matter, and a decline is
  fluent prose that passes every downstream check. It is detected before it
  reaches a sidecar, counted by name, and the picker offers an abliterated
  repackage — same architecture, same loader, so the fix costs a repo id.
- Trigger-word prepend, per-image caption editing, an uncaptioned filter, and
  CPU-served thumbnails.

### Training

Krea 2 LoRA training on musubi-tuner: rank, alpha, learning rate, epochs,
save-every, resolution, discrete flow shift and seed, defaulting to a rank-32 /
alpha-32 / 1e-4 / 30-epoch run. Checkpoints land per epoch in `loras/{name}/`, so
a folder is a LoRA and the files in it are its epochs. Progress is carried on the
Train door itself, because a run lasts hours and you are meant to leave and keep
working.

### Image generation

- Krea 2 Turbo (8 steps, CFG 1.0 — the absence of a negative branch, not a low
  setting) and RAW (28 steps, CFG 5.5), on a resident checkpoint.
- **LoRAs are written in the prompt**, in Automatic1111's `<lora:name:0.8>`
  syntax, up to six. A name resolves by shortest unambiguous match, exact-case
  first — two files differing only in capitalisation are two LoRAs, and folding
  case made both untypeable. A `+ LoRA` picker writes a token at the caret,
  because you cannot type a syntax you have never seen.
- **Regions are drawn on the canvas**, up to eight. Each box takes its own prompt,
  its own LoRA at its own strength, and optionally its own reference photograph —
  a latent mold that pulls that rectangle toward that face during sampling, which
  gives you a character with no training run behind it. Boxes snap to halves,
  thirds, quarters and to each other; one inspector row holds the numbers for
  whichever box is selected, at the same height for eight boxes as for one.
- Scene and outfit transfer when the identity-edit LoRA is present.
- The ratio picker and the pixel boxes are one control, snapping to 8 on the way
  out because the pipeline floors to the VAE's grid regardless.

### Video — two families, one path

Adding Wan did not add a backend. It reuses the container, the warm ComfyUI
process, the job contract and the output layout; what is per-family is a graph
builder and a table row. They are not interchangeable, and the UI says so rather
than averaging them:

| | MiniMax-H3 | Wan 2.2 |
|---|---|---|
| Audio | Native stereo, same latent, one pass | Silent |
| CFG / negative | No — guidance-distilled | Yes |
| LoRAs | No ecosystem for the int8 repackage | Yes |
| References | Up to 9 images, 12 across all types | No |
| Experts | One | Two on A14B, one on the 5B |
| Tiers | 768p / 544p draft | 720p / 480p (14B), 704p / 480p (5B) |
| Length | 5–14 s at 24 fps | 2–5 s at 16 fps (14B) or 24 fps (5B) |
| Tasks | t2v, i2v, first/last frame, ref2va | t2v, i2v, first/last frame |

`VIDEO_MODELS` is served to the page, so the composer renders only what the
chosen model reads.

Two rules earned in this path and worth stating on their own:

- **Write out every optional input a node takes.** A ComfyUI node declares its
  defaults twice — once for the canvas widget, once in its `run()` signature —
  and for an input the graph omits, it is the signature that wins. In the pack we
  drive, two of them disagreed, so every plate render ran the identity-edit LoRA
  43% hot and downscaled every reference to 1024. Neither was visible in the
  graph, the sidecar or the page. Spelling every optional input out is a few
  lines, and it turns a disagreement upstream can introduce silently into one a
  diff shows.
- **Every input a run is priced by needs a range, and a file does not come with
  one.** Tier, seconds, aspect and steps are all bounded. "Max detail" on a
  reference was not: the node floors the short edge at 2048 and caps nothing, so
  a 4032×3024 straight off a phone arrived as 2720×2048 — 21,760 latent tokens
  against the 3,996 the same file would have cost at "match" — and reference
  tokens ride *every* sampling step, so the run died in step 0 of 8. References
  are bounded on arrival at 1536px now, and the resize bakes in the EXIF rotation
  rather than dropping the tag. The general rule: an option whose cost is set by
  something the page never measured is an option that will be found by whoever
  has the biggest camera.

### The prompt compiler

`SHOT_VOCAB` is one table with nine groups — Framing, Angle, Light, Tone, Action,
Speech & text, Camera, Sound, Score — and three compilers behind it. Each group
declares which side reads it and where its clause lands.

- **H3** gets the six-field document, with the alignment instruction held
  verbatim including the guide's own inconsistencies, because it is a contract
  with the checkpoint rather than phrasing we chose.
- **Wan** gets prose with the audio pills dropped by capability, not by field —
  dialogue is the case that breaks the simpler rule.
- **Krea 2** gets prose with camera, action and audio filtered out — and the
  palette *dims* what the model cannot read, with the group heading saying why.

Three rules hold it together: no pills means no document and the typed text
passes through byte-for-byte; the compiler closes your sentence and picks the
separator in front of it and does nothing else; and `non_diegetic_music: N/A` is
the default, which is worth the feature on its own because H3 invented a
soundtrack for every clip until something told it not to. `/api/compile` runs the
same compiler on the same container, so the disclosure under the pill rail shows
the exact string the encoder will be handed — a preview with its own
implementation is worse than no preview.

Reference chips carry a role — identity, wardrobe, location, style, prop or
action — which is what makes "do not describe the picture you attached"
enforceable rather than advice: there is now somewhere else for that description
to go, and it is one click. ⌥← / ⌥→ moves the clause under the caret one slot
along, because the separators are slots and the text between them changes places.

### The job contract, and what makes it survive

Every long operation — download, Drive pull, caption, train, generate, video —
publishes to the same record and answers the same status and stop routes. New
capability extends it rather than inventing a parallel one.

- **A job record has two writers, so publishing holds a lock.** The job thread
  writes phases; the log drain writes step counts. Interleaved without the lock,
  a stale read put `running` back over a `completed` in 15 modelled runs out of
  15, and the page polled a finished job forever.
- **A poll waits for its own reply.** `everyMs` replaces `setInterval`, which
  fires on a clock rather than on a reply — overlapping polls held connections
  against a six-per-origin budget and starved the very gallery covers and video
  byte-ranges they were painting.
- **A job record is a claim about a container, worth only what its age says.**
  Every publish stamps a beat, because a container killed mid-transfer never
  reaches its own terminal path and its record says "running" for good — and the
  Dict outlives the deploy, so rebuilding the image cannot clear it.
- **A dead ComfyUI is diagnosed and replaced, never re-raised as a socket
  error.** A GPU fault reaches us as a connection reset; the liveness check runs
  before anything blames the transport, and a fresh process starts at the top of
  the next run rather than every render being refused until the scaledown window
  expires. An OOM triggers a reclaim, charged to the job that already failed
  rather than to every job to prevent one, and free VRAM at the start of a run is
  recorded because it is the one number separating "this graph is too big" from
  "the last graph never gave the card back".
- **Stops are cooperative.** Jobs check a flag between steps and unwind cleanly,
  so the container survives and the next request is warm.

### The interface

One page, no build step, no dependencies. The canvas is the largest thing on
screen at every moment and the controls live in a bar under it, capped at 30% of
the viewport with the prompt field yielding to the budget. Image and video are
one workspace sharing prompt, canvas and gallery — the switch is a chip inside
the prompt field and the sentence survives it. The gallery has three layers, its
navigation lives in the top corners, and the last generation is a thumbnail
beside Generate where the hand that pressed it already is.

**The console pass has a measured result.** Controls are sorted by how often you
reach for them, not by whether they are per-take or per-session — scope sounds
right and fails on its own examples, because nobody picks a new aspect ratio per
render and CFG is not a thing you set once a session either, it is a thing you
almost never set. The seed is the case that proves it: different on every render,
so per-scope it belongs in the row, and yet nobody types a seed — you take it off
a result. Moving what a take does not change behind the model button took the
image strip from 1016px of controls to 732px, and the video strip from 979px to
652px. Controls also carry no chrome until the pointer is on them, because the
value is what makes a control a control and every one of these already shows its
own.

**Bar versus rail is settled, with evidence rather than taste.** Fitting each
render aspect into the canvas at 1512×982 leaves 0px of dead vertical space at
all five aspects and 152–1068px horizontal, so the picture is height-bound
everywhere: the bar always comes out of the picture, while four of five leave
room for a rail twice over. The bar stays anyway — a rail is a dated shape that
will read as of-its-decade long before this app stops being useful, and the
overlay that would have avoided both was built and rejected, because a backdrop
blur means judging a render through something whose legibility depends on what
you generated.

**Every picture the model can be given sits in one row, and the two halves dim
each other.** Keyframes and references are the same decision made two ways — they
load different transformers, so one excludes the other. As two separate rows of
unlabelled 36px tiles, the keyframe pair was never found at all, and dropping
photos into the reference tray looked like filling keyframe slots that kept
growing. Side by side with a rule between them, the tray that grows and the two
slots that do not are told apart by shape, which is the thing a tooltip could not
do. Whichever half is out of play goes dim rather than disappearing: a control
that vanishes when you fill its neighbour teaches nothing except that the page
lost it.

`tools/preview_ui.py` serves the page against stubbed JSON shaped to hold the
awkward states, so front-end work costs a reload rather than an image build and a
cold start. `tools/_from_app.py` pulls plain-Python pieces out by AST, so the
smoke tests check the real compiler rather than a reimplementation of it, and
`tools/smoke_graphs.py` validates every sampler and scheduler the UI offers
against what the node will actually accept.

---

## 3. The Long-Term Vision & Roadmap

### Where this is going, in one sentence

The console is scaffolding. The end state is a canvas where you argue with the
picture instead of authoring a prompt — and the prompt itself is an
implementation detail nobody should ever have to see.

### Horizon 1 — the next two quarters

**Phase 5: video LoRA training.** Wan 2.2 is the target, which is why the video
path already loads LoRAs and why a video LoRA row already carries an expert. This
extends the existing trainer rather than adding a second one, and it is the piece
that makes "own the weights" true on the video side as well as the image side.

**H3-Regenerate-2K, when and only when it ships locally.** 2K is absent on
purpose. The module is not super-resolution — it feeds the 768p result *plus the
original multimodal context* back into the base model, which is what recovers
small text and fine detail that conventional upscaling has to guess. MiniMax
withholds it. Building against their hosted endpoint was considered and rejected:
it sends renders to a third party, bills outside Modal, and becomes dead code the
day the local module lands. When it does land, it is another H3 task taking a
video and a prompt, which the existing graph builder and route are already shaped
for. One thing to build first: upscaling this app's own output beats upscaling a
dropped file, because the method's whole advantage is the original context and a
sidecar still holds the prompt, the pills and the references.

**Consolidation the release earns.** A generated still already flows into a clip
through "Animate" and "As reference"; that path widens as the two sides converge.

### Horizon 2 — Phase 6, the Dynamic Canvas

The canvas becomes where the work is done rather than where the result appears.
Every attachment a drop: a photo onto a box is that character, a photo onto the
frame is the scene, and keyframes and references arrive the same way instead of
through a tray of 32px tiles. What you touch answers with its own affordances, at
the place you touched it, and only the ones that object can respond to — not a
menu at a fixed corner, which is a panel in miniature parked at a coordinate.

**The capability is honestly tiered, and the tiers have very different costs:**

| Tier | What it is | Status |
|---|---|---|
| **Reachable now** | Touch-to-select and touch-to-edit. Segmentation on a finished render turns flat pixels into addressable regions; regional inpainting re-renders one mask. | An engineering step. The regional path is already element-as-component in primitive form — the gap is that boxes are author-declared rather than model-derived. |
| **Reachable but expensive** | Elements that persist across renders. | A LoRA already gives this for a character, which is why the platform trains them. Making it general means a LoRA per element and a training run per thing kept — which changes what a library *is*: weights, not images. |
| **Not there** | A frame computed from a spatial arrangement rather than authored, and a camera with real physical limits. | Causality between a 3D scene and a 2D image, which diffusion does not model. Needs a world model, or a hybrid where real 3D drives conditioning. Nothing off the shelf does it. |

The plan chases the first tier and leaves the third alone until the model exists.
Attempting it, failing, and concluding the whole direction is fantasy is the
specific mistake worth naming in advance.

### Horizon 3 — the end state, and what it vetoes

These are design constants, and each exists to kill a specific cheap fix. The
cheap fix is always a panel.

- **If a gesture cannot communicate intent without a label, the answer is a
  better gesture — never a control panel.** Every other rule is an instance of
  this one. It also kills the tutorial: coach marks and first-run overlays are
  labels in a costume.
- **The canvas never changes.** Character, wardrobe, action, location, camera —
  one surface. Kills tabs, rooms, modes, a back button. Today's Generate/Train
  split is on the wrong side of this.
- **What you touch decides the mode.** You do not enter character mode, you touch
  a face. Kills tool palettes and selected-tool state.
- **Everything is live, nothing is labelled.** Kills inspectors, sidebars,
  property panels — which is what the console is.
- **Nothing asks for confirmation.** Kills modals and the Generate button. The
  trade is that undo must be total, because reversibility is what replaces
  confirmation.
- **Duration starts at zero.** A still is the default and time is added. Someone
  who wants one image should finish and leave without learning motion exists.
- **The frame is computed, never authored.** You move the world, not the picture
  of it.
- **The camera is a character.** It has a want and real limits; it can be late,
  it can look away. Kills the camera as a settings group.
- **Derived or invented, always visible.** Kills the chat panel and clarifying
  questions — every question asked is a small failure, so pick something, mark it
  invented, move on.
- **The library is closed until you reach for it.** A drawer, not a workspace.
- **It never remembers unless told.** Kills personalisation and learned defaults.
  The first screen of the thousandth session is the first screen of the first.

### The strategic bet underneath all of it

**A prompt is a compilation target, not something the user writes.** Half of this
is already built and is the proof it works: one vocabulary, three compilers, and
nobody types `integrated_multimodal_description:`. Replacing the prompt field
entirely — a model reading intent instead of a table matching pills — is the same
idea with a better front half.

**Fragments are the expected input, not the degraded case.** People think in
bursts: a correction, a second thought that contradicts the first, the real point
arriving third. AI writing is caught precisely because it is too smooth, and a
tool that demands one clean flowing paragraph is demanding a human produce the
artefact of a machine before the machine will listen. Out of order, incomplete,
self-correcting, arriving in pieces over a minute — that is the normal shape of
intent, and the system's job is to take it that way.

**Intent is the durable record; the prompt is a receipt.** The sidecar already
keeps `prompt_typed` and `shot` alongside the compiled string, and the gallery and
Reuse prefer the typed one. This reads as a legibility choice and is really the
deeper one: it pays off the moment a model is replaced, because intent recompiles
for whatever comes next while a stored prompt is worth nothing to a checkpoint
that wants a different grammar. Reproducing a prompt was never worth protecting —
it is a guess at how to speak unnaturally to one text encoder on one particular
day.

**And a standing veto that will look unhelpful in a pull request.** Nothing in
this application browses, searches or recommends somebody else's weights. Pulling
a specific file you were sent is fine — that is what the Drive route is for. A
marketplace is not, because a "discover models" surface is exactly the feature
that would look obviously helpful and would undo the entire argument above.

### The risk worth naming

It is not the models, the latency or the scope. It is month four, when something
does not fit cleanly and the cheapest available fix is a panel. That is the
moment this becomes a node graph with better typography — and every rule in the
section above exists to be the thing that stops it.
