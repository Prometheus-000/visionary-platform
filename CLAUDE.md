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
- **Destructive is asked for, not softened.** Deletion unlinks. There was a
  `.trash/` per root for a while, on the theory that a mis-click should cost a
  file move rather than the file — but nothing ever surfaced it, so it was not
  an undo anyone could reach; it was a second copy of everything already thrown
  away, on a volume whose only other way to reclaim space is the Modal CLI. The
  confirm dialog is the safety net now, so it has to say what is going and how
  much of it: a dialog that undersells the blast radius is the failure mode this
  replaced. `_drop_legacy_trash()` clears what the old scheme left behind.
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
- **A poll waits for its own reply, and a burst of identical work is done
  once.** Small is not enough on its own: `setInterval` fires on a clock rather
  than on a reply, so at 400ms against a network Dict a slow answer does not
  delay the next tick, it overlaps it. Replies then land out of order and the
  bar is painted by whichever arrived last rather than whichever is newest —
  and the polls in flight hold connections, of which a browser gives one origin
  about six. The gallery's covers, the canvas stills and a `<video>`
  re-requesting byte ranges all come off that origin, so a pile of polls
  starves them: the clip stutters and the grid comes back half-painted. Neither
  symptom looks like a poll loop, which is why it went unfound. `everyMs` is
  the whole fix and is a drop-in — same arguments, same id, so
  `clearInterval(t)` inside the body still ends it.

  `_reload_volume()` is the same shape at the other end. Serialising it turned
  a race into a queue, and a queue of twelve identical reloads is the wrong
  half of that trade, because a gallery is a grid and a canvas is a batch — the
  misses arrive together and all want the same thing. The sequence number is
  read *before* queueing, so a caller overtaken by a reload that began after it
  asked rides on that one. One reload per in-flight window, not one overall:
  a reload that began before you asked is no evidence that a file written after
  it is visible, so a caller arriving mid-reload still runs its own. Twelve
  concurrent misses cost two rather than twelve — 3.00s of queue became 0.51s.
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
  resort, and the last resort ran out. `forge/` was here because Krea 2 needed
  a patch to `backend/nn/krea.py` to mask attention; it is gone because a node
  pack does the same job through ComfyUI's public hooks. What is left is two
  commit pins — `COMFY_SHA` and `CLIFF_SHA` — and nothing patched, so there is
  no `VENDOR.md` to keep and no sync to perform. If vendoring ever earns its
  place again, record it the way that file did: the source SHA and every local
  change, so a sync is a diff rather than an archaeology exercise.
- **Storage layout is the contract, not the code.** Datasets are folders of
  images with `.txt` sidecars beside them — the same thing the trainer reads.
  Nothing here is required to get your data back out.
- **Separate images when pins conflict — and only then.** Trainer, inference
  and captioning are three images because one shared `transformers` pin would
  mean every captioner bump re-litigates training. A dependency conflict
  resolved by compromise is a conflict you pay for forever. The converse is
  just as binding: images and video share `comfy_image` because nothing in it
  is per-family, and a fourth image would have been a second CUDA 13 build, a
  second SageAttention compile and a second place to get the arch list wrong.
- **Do not build a second way to do the first thing.** New capability extends
  the existing job/status/stop contract rather than inventing a parallel one.

## Layout

    app.py              the whole application — images, jobs, API, and UI_HTML
    comfy_nodes/        our own ComfyUI nodes — one shim, see visionary_boxes
    ai-toolkit/         training reference
    tools/              smoke tests, the local UI preview
    tools/_from_app.py  pulls plain-Python pieces out of app.py by AST

`_from_app.py` exists because two tools need the *real* thing rather than a
copy: `smoke_prompt.py` checking a compiler against a reimplementation would be
checking the reimplementation, and `preview_ui.py` drawing the shot palette from
a hand-written vocabulary would be a preview of a palette that does not exist.
Importing app.py is what it avoids — that pulls in modal and builds image
definitions at module scope, so it wants credentials and a network to answer a
question about a string.

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

### Orientation is resolved on arrival, not by each reader

An uploaded image is rewritten upright if it carries an EXIF orientation tag, so
the pixels on the volume are the pixels every consumer gets. This is a storage
decision rather than a display one because PIL does not rotate on open and
browsers do — the page showed a phone photo the right way up while the captioner
described a rotated scene and musubi, whose loader is a bare
`Image.open(...).convert("RGB")`, measured a 3024×4032 portrait as landscape,
put it in a landscape bucket, and trained the rotation into the LoRA. "Rotate" in
Finder or Photos usually writes the tag and leaves the pixels alone, so this
arrives far more often than it looks.

Fixing it in each reader would mean every future consumer has to know the tag
exists. Normalising once on arrival means none of them do — the same reason
captions are `.txt` sidecars. Files without the tag, which is nearly all of them,
are not touched at all. `_upright_copy()` repeats it into the training scratch
copy so sets uploaded before this still train upright, without rewriting bytes
already on the volume.

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

**Case is part of the name.** Resolution is exact first, and case-insensitive
only as a fallback that still has to land on one file. Folding case before
comparing made `K3nan.safetensors` and `k3nan.safetensors` — two files, two
LoRAs, which is what a Drive pull and a training run disagreeing about
capitalisation leaves behind — collide into one ambiguous name, and the failure
was not that it picked the wrong one. It was that *neither* resolved, so both
went untypeable and the note blamed a missing file for a file that is sitting
right there. Nothing in the backend folds case: it addresses LoRAs by exact path
and ComfyUI validates them against a directory listing, so the resolver was the
only place on the path where two distinct files became one name.

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

Sweeping and deleting both **unlink**. The sweep is the one deletion nobody asks
for by name, so the grace period is what protects it: fifteen minutes of silence
from the session, and the folder's own mtime counted as a heartbeat so an upload
still writing cannot be swept out from under itself.

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
  models use parse grammar. See the `CAPTION_MODELS` comment.
- **A caption preset is a decision about what to leave out, and a refusal is
  not an error.** Both halves of the captioner row exist for reasons the other
  cannot cover.

  What a caption *names* is what the model learns is free to vary; what it never
  names is what the trigger word ends up owning. So the presets in
  `CAPTION_PRESETS` are one rule inverted per intent — Character describes pose,
  wardrobe, framing and light and refuses to describe a face; Style describes
  the content and never the look; Concept describes the context around the
  thing. Each also names the flaws worth prompting away later, because a
  watermark nobody mentioned is a watermark the LoRA learned. The instruction
  stays on the server and the page sends a key, for the reason `SHOT_VOCAB`
  does: a run has to be reproducible from the job record rather than from
  whatever text was in a field at the time.

  The captioner picker is the other half. A stock instruct model declines to
  describe photographs of real people often enough to matter, and on a character
  set that is *every* image — but a decline is not an exception. It is fluent
  prose that passes every check downstream of it, lands in a `.txt` sidecar and
  trains, so the symptom is a LoRA that learned to say it cannot describe
  someone. `_looks_like_refusal()` therefore drops it before it is written,
  which leaves the image in the Uncaptioned filter where it can be found, and
  `CAPTION_MODELS` offers the abliterated repackage — same architecture, same
  loader, so the fix costs a repo id rather than a second code path. The count
  of refusals is reported by name and points at the other menu, because "run
  finished, nineteen of twenty-four captioned" is not a diagnosis.

  The regex is prefix-anchored for the same reason `prepend_trigger` uses
  `startswith`: "I cannot" inside a caption is a sentence about the picture, and
  a substring test would throw away real captions to catch a model talking about
  itself.
- **Reload through `_reload_volume()`, never `volume.reload()`.** Modal refuses
  a reload while anything on the volume is open, and a container holding a
  checkpoint always is — safetensors maps the weights straight off `/workspace`
  and the mapping outlives the descriptor. A bare reload therefore worked once
  per container and raised for the rest of its life. Freshness is not worth a
  dead container, so the open-files conflict is absorbed and logged; every
  other reload failure still raises.
- **A job record has two writers, so `_publish` holds a lock.** It does
  get-update-put against a *network* Dict, and the round trip is the window: a
  value is already stale when it arrives and the write lands after the caller
  decided. The job thread publishes phases and the terminal result; `_drain`
  publishes step counts as ComfyUI's tqdm line scrolls past. Interleaved, the
  drain reads `{step: 6}`, the job thread reads `{step: 6}` and writes
  `{step: 6, phase: …}` over the drain's `{step: 7}`, and the bar walks
  backwards — the server half of a symptom whose client half is the poll
  overlap above, which is why neither was reachable from where it was being
  looked for. On the last line it stops being cosmetic: a tqdm write that read
  the record before the job finished puts `status: "running"` back over a
  `completed` that was already there, and the page then polls a finished job
  until someone reloads it. Modelled with the latency where it actually sits,
  that lost the terminal status in 15 runs out of 15. The lock is process-local
  because `max_containers=1` means there is no second writer to coordinate
  with.
- **A long transfer publishes bytes, or it cannot be debugged.** `_download_weight`
  used to print one line at the start and one at the end. Between them, on a
  17 GB pull, it reported nothing — so "stalled at 4 GB" and "running fine at
  90 MB/s" were the same UI state and the same empty log, and the job sat there
  until the four-hour timeout. `_watch_download()` polls bytes on disk from a
  thread and publishes a count and a rate; if they stop moving for
  `DOWNLOAD_STALL_S` the attempt is abandoned and resumed, `DOWNLOAD_TRIES`
  times. It polls rather than hooking because `hf_hub_download` has no progress
  callback and the shape of what it writes changes across releases; summing the
  tree is indifferent to both.

  One thing to leave alone: **the worker takes its sink and its flag as
  arguments** — with a closure, an abandoned attempt finishing late writes its
  stale result into the *next* attempt's dict and sets the next attempt's event.

- **`hf_transfer` is on, and the resume it costs is not worth having.** This
  entry used to say the opposite — do not enable it on `web_image` without
  checking resume first — and the checking is what reversed it. Measured on
  this image against a 21 GB file: **30.6 MB/s on the plain requests backend,
  243.8 MB/s on hf_transfer.** That 8x was the whole distance between "over
  200 MB/s with the hf CLI" and a platform that needed a working day to fetch
  its own weights, and it had been left on the table by an unset env var while
  the package itself was installed.

  Both original objections were tested rather than argued. The missing progress
  hook was already answered by `_staged_bytes()`, which sums the tree and does
  not care which backend wrote it — those very numbers were taken that way. The
  resume objection is *true*: killed mid-file, the plain backend restarts at
  0.29 of 0.29 GB and hf_transfer at 0.00 of 5.09. It simply stopped mattering.
  At 244 MB/s the largest weight in the catalogue lands in under two minutes,
  which is shorter than `DOWNLOAD_STALL_S`, so a restart now costs less than
  the detector that triggers it. Resume was machinery for a fifteen-minute
  download and there is no longer one.

  Falling back to the plain backend on retries was measured too and does work —
  it picks up hf_transfer's bytes rather than starting over — and is
  deliberately not done. It buys a resume for a two-minute transfer at the
  price of an 8x slower one, and it puts a second backend on the failure path,
  where it would only ever run when things are already going wrong. Retries
  stay on hf_transfer and start the file over. That is also why
  `DOWNLOAD_STALL_S` stays conservative rather than being tuned down to match
  the new speed: an eager stall detector used to cost the bytes since the
  stall and now costs all of them.

- **A dead ComfyUI is diagnosed and replaced, never re-raised as a socket
  error.** A GPU that faults takes the process with it — Xid 31, an MMU fault,
  is the one that has happened — and the first thing that reaches us is a
  `ConnectionResetError` from the history poll, or a `ConnectionRefused` from
  the next `/prompt`. `_await` already had the right thing to say about a dead
  process, log tail and all; the check simply sat *downstream* of the call that
  could not survive to reach it, so the job record got urllib's exception
  instead: no CUDA error, no log, no mention of the GPU. `_check_alive()` runs
  before anything blames the transport, and it `wait()`s rather than `poll()`s
  because the socket resets a beat before the process is reaped and a bare poll
  in that gap reports the corpse alive.

  The second half is that the container outlives the process. `@modal.enter`
  runs once, so with `max_containers=1` a dead ComfyUI is not a degraded
  install — it is every render refused until the scaledown window expires,
  which is what three consecutive `ConnectionRefused` jobs on one container
  looked like. Xid 31 is an illegal address in a kernel, not a dead card, so
  the device runs the next graph perfectly well and `_revive()` starts a fresh
  process at the top of `run()`. A drop against a process that is demonstrably
  still alive is the third case and is a blip: retried, `COMFY_RESET_TRIES`
  times, because failing a forty-minute clip over one socket is the wrong
  trade.

  There is a fourth: alive, and with nothing left to allocate. ComfyUI answers
  its own OOM with `unload_all_models()`, which on this install does not reach
  the thing that filled the card — the regional node moves every region's LoRA
  onto the device in `_prepare()` and stores the copies on the patcher it
  returns, so they are held by the *execution cache*, which model management
  cannot see. Only `/free` drops that (`e.reset()`), and `free_memory` implies
  `unload_models` upstream, so there is no way to clear the node cache without
  also dropping the 24 GB checkpoint. Hence `_reclaim()` runs on an OOM and
  nowhere else: the reload is charged to a job that has already failed, rather
  than to every job to prevent one. This is why the symptom was "a few times,
  nothing reproducible" — a run that ran out of memory left behind the thing it
  ran out of memory on, and the next one started with less room than the last.
  `_note_headroom()` is the other half, because free VRAM at the *start* of a
  run is the one number that separates "this graph is too big" from "the last
  graph never gave the card back", and ComfyUI only prints its memory summary
  after it has already failed.

- **A family downloads itself; there is no button for the whole catalogue.**
  The group is the unit you decide in — you want the Wan stack or you do not —
  and clicking its files one at a time meant watching a 4 GB file finish to be
  allowed to start the next, which is a queue kept in a person rather than in
  the program. `download_missing_job` already walked a list sequentially, so
  this is a `family` parameter on the route it already had and a `job_id`
  argument on the job, not a second queue. The catalogue-wide "Download missing"
  was removed rather than kept beside it: it sat in the HuggingFace-token row,
  which put the one button that pulls every family — including the ones an
  install will never run — next to a password field it has nothing to do with.
  Two buttons doing overlapping things, one of them almost always the wrong
  scope, is worse than one. A family one file short shows no button at all,
  because that is what its own Download already is.

- **One download at a time, and being busy is a state rather than an error.**
  They share an uplink: three concurrent pulls measured 4-12 MB/s each against
  ~31 MB/s for one, so a second download is not a second download, it is the
  same bandwidth divided plus a container to pay for. `/api/download` is
  therefore idempotent — a second press returns the job the first one started,
  and the page removes the other buttons rather than answering a press with a
  red message. Pressing twice is not a mistake to correct; it is what anyone
  does when the first press appears to do nothing, which is how this arrived:
  `_active_download()` scanned `dl_{key}` across the whole catalogue, and on a
  network Dict those twenty-odd round trips made the route take seven seconds
  to answer. It is one pointer read now (`DL_ACTIVE`).

- **A job record is a claim about a container, worth only what its age says.**
  `jobs` is a *named* Modal Dict, so it outlives the container, the app, the
  deploy and the image rebuild. A container killed mid-transfer never reaches
  any of its own terminal paths, so its record says "running" for good. The
  first version of the concurrency guard above trusted that field and turned
  three corpses from one stopped app into a permanent refusal of every download
  that followed — and rebuilding the image could not clear it, because the Dict
  is not part of the image. Every `_publish` now stamps a `beat`, and
  `_download_alive()` believes a status only as far as it, rewriting a stale one
  to failed so the lock and the UI clear together. Anything that gates on a job
  record needs this; a status alone is not evidence that anything is running.
- **Weights are staged, then moved.** Both `_download_weight` and `gdrive_job`
  write to a staging directory and `shutil.move` into place. The picker globs
  `loras/` live, so a half-written `.safetensors` downloaded straight there is
  offered, chosen, and fails inside a warm GPU container thirty seconds into a
  run. Staging keeps a partial download invisible until it is a whole file, and
  the move is a rename because staging is on the same volume.

- **Write out every optional input a node takes.** A ComfyUI node declares its
  defaults twice — once in `INPUT_TYPES` for the canvas widget, once in its
  `run()` signature — and for an optional input the graph omits, it is the
  *signature* that wins. Those two disagree more often than they should. In the
  node pack we drive they disagreed on `edit_lora_strength` (0.7 vs 1.0) and
  `ref_max_side` (0 vs 1024), so every plate render ran the identity-edit LoRA
  43% hot, which its own tooltip warns gives "mottled, crumpled-looking
  texture", and downscaled every reference to 1024, which its tooltip says
  "costs likeness for speed". Neither was visible anywhere: not in the graph,
  not in the sidecar, not on the page. Spelling every optional input out is a
  few lines, and it turns a disagreement upstream can introduce silently into
  one a diff shows.

- **Every input a run is priced by needs a range, and a file does not come with
  one.** Tier, seconds, aspect, steps are all controls that cannot be set past
  what the card will do. `ref_image_size: "max"` looked like one more of those
  and was not: the node sizes a reference by `min(1.0, 2048 / min(w, h))`, which
  floors the short edge and caps nothing, so a 4032x3024 straight off a phone
  arrived as 2720x2048 — 21,760 latent tokens against "match"'s 3,996 — and a
  panorama arrived untouched at 31,000, because its short edge was already small
  enough to be left alone. Reference tokens ride every sampling step, so nine of
  them doubled the sequence and the run died in step 0 of 8 with ComfyUI's canned
  advice about a batch size the video path does not have.

  `H3_REF_MAX_SIDE` is the range, applied on arrival by `_fit_reference` rather
  than by asking the node for a smaller number: under 2048 on the short edge its
  scale is exactly 1.0, so bounding the staged file is what bounds the run and
  there is still only one thing deciding how big the picture is. It rewrites only
  when it resizes, and bakes the EXIF rotation in when it does — a resized copy
  saved without the tag would reach the DiT sideways, which is `_upright_inplace`
  from the other end: not a reader that forgets the tag, but a writer that drops
  it. The browser caps too, at the same number, for the payload; that copy is an
  optimisation and the server's is the one that binds, so drift costs nothing.

  The general rule: an option whose cost is set by something the page never
  measured is an option that will be found by whoever has the biggest camera.

## The page

The UI is not organised the way this file is. There are three subsystems and
two domains, and the page follows the domains.

- **The canvas is the largest thing on screen, always.** Options live in a bar
  under it, never a rail beside it: a settings column costs the picture 384px
  of the one dimension it cannot get back, and vertical is the cheap axis. The
  bar is capped so its fullest state cannot push the canvas out of frame, and
  anything sized to fit the canvas measures the canvas — a `dvh` sum is wrong
  the moment the bar grows.

  **"Vertical is the cheap axis" is false on a laptop, and the measurement is
  here so nobody re-derives it.** Fitting each render aspect into the canvas at
  1512x982 leaves this much unused:

  | render | dead ↔ | dead ↕ |
  |---|---|---|
  | 16:9 | 152px | **0px** |
  | 4:3 | 513px | **0px** |
  | 1:1 | 735px | **0px** |
  | 3:4 | 908px | **0px** |
  | 9:16 | 1068px | **0px** |

  The picture is height-bound at *every* aspect, so the bar always comes out of
  the picture, while four of five leave enough horizontal room for a rail twice
  over. On a tablet in portrait it inverts — 834x1194 leaves 297–469px vertical
  on landscape renders and nothing on portrait ones — so neither placement is
  right everywhere and the axis that is free depends on the render, not on
  taste. What is *not* an option is floating the console over the picture: it
  was built and rejected, because a backdrop blur means judging a render through
  something, and its legibility then depends on what you generated.

- **The console has a budget, and the prompt is what yields to it.** 30% of the
  viewport. Everything else in there is fixed or conditional — the strip is one
  row, the rail appears with the first pill, the region bar with the first box —
  so the prompt field is the only part that grows without asking, and measuring
  showed it was also the part that broke the budget alone: at a flat 168px cap
  the worst case was 39.8% of a 1440x900 window, 136 of which was the field.
  `fieldMax()` hands it whatever is left, down to a two-line floor, because a
  budget that wins by making the prompt unusable has optimised the wrong thing.
  A ResizeObserver on the console is the half that makes it true more than once:
  arming regions and picking pills both happen long after the last keystroke,
  and without it a long prompt sat at 30.0% and climbed to 38.1% when they
  arrived. It converges in one pass because `fieldMax` subtracts the field's own
  height, so `other` does not move when the field does.

- **A utility lives with the controls; navigation lives in the top corners.**
  Reaching the gallery is part of making something, like writing a prompt — so
  the last generation is a thumbnail beside Generate, 15px off the bottom, where
  the hand that just pressed Generate already is. The Camera app puts the last
  frame next to the shutter for that reason and it was read wrong once here:
  the pattern was copied into the header, which put the two halves of one loop
  at opposite corners of a 1194px screen.

  Moving *between* the gallery's layers is navigation, and it belongs in the top
  corners from the moment you are inside — not promoted there after a
  bottom-centre pill hands you over, which is what a "View all generations"
  button did and why it read as inconsistent. There are three layers, and the
  third is easy to miss: the paged view you swipe, the grid, and the picture
  with its chrome off, which is entered by *tapping* rather than swiping. A tap
  used to close the viewer, so leaving was doing double duty for looking
  closely and every attempt to see a render properly dismissed it.

- **The small screen is the design tool, not a port of the big one.** Three
  faults this session were live on desktop for months and only became visible
  under a phone or tablet: `appearance` had never been reset on any select, so
  desktop had been rendering macOS arrows as its chevron; the viewer's drag was
  broken on *trackpad* specifically, because `<img>` is natively draggable and
  the resulting HTML drag fires `pointercancel` — touch was the case that
  worked; and the gallery thumb was in the wrong corner at every size. Below
  1024px the layout stacks and the grid crops to squares, which is the one place
  this codebase trades information for density, and the trade is right because
  there the screen is the constraint rather than the design.
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

  A region's own field takes the same syntax, and that is the whole reason a
  region has no LoRA select. The two fields then mean one thing at two scopes —
  a token in the main prompt is the canvas, a token in a box is that box — so
  nothing has to explain which is which. One per box, because that is the
  node's shape; a second is rejected by name rather than quietly ignored. `+
  LoRA` writes into whichever of the two fields you last had the caret in, so
  "put this character here" is: draw a rectangle, click `+ LoRA`, pick a name.
  It writes `:1.3` into a box and `:1` into the main prompt, because the node
  pack's guidance for a character is 1.3–1.4 and the main prompt is a style
  stack far more often than a character.

- **A region is drawn on the canvas, not described under it.** The boxes are the
  list: drag on the frame to place one, drag it to move it, drag a handle to
  size it. The console keeps one inspector row for whichever box is selected.

  It used to be a row per region — a sentence and four coordinates — and the row
  needed a 32px picture of those coordinates beside it to be legible at all,
  which is the whole argument. "0.5 0 0.5 1" is a rectangle you rebuild in your
  head every time; the rectangle is not. The numbers survive in the inspector,
  where they are the escape hatch and, during a drag, a readout that moves —
  dragging teaches the numbers, and the numbers never taught the dragging.

  The cost was the other half. A row was ~74px, so the eight boxes the backend
  allows came to ~592px of console against a 54dvh cap: the feature's fullest
  state broke the rule the console exists to hold. One inspector row is the same
  height at eight boxes as at one.

  Boxes snap to halves, thirds and quarters and to each other, which is what
  makes an even split a gesture rather than a menu; Alt suppresses it. Arming
  the mode seeds two half-width columns rather than explaining anything, because
  two rectangles appearing on the canvas is the instruction. What the boxes are
  drawn *on* is the frame at the render's aspect, or the scene plate if one is
  dropped, or the last render — adjusting boxes against the picture you actually
  got is the reason they are still there afterwards.

- **Every picture the model can be given sits in one row, and the two halves
  dim each other.** Keyframes and references are the same decision made two
  ways — they load different transformers, so one excludes the other — and they
  used to be two rows: keyframes parked at the right of the strip among the
  numeric controls, references in their own row below. Two pairs of unlabelled
  36px dashed tiles, one row apart, telling each other apart by tooltip.

  What that cost was not tidiness. The keyframe tiles were never found at all,
  and dropping photos into the reference tray *looked* like filling keyframe
  slots that kept growing — which is exactly what the tray does and exactly what
  a fixed pair must never look like. Side by side with a rule between them, the
  tray that grows and the two slots that do not are told apart by shape, which
  is the thing a tooltip could not do. Whichever half is out of play goes dim
  rather than disappearing: the row's job is to show that these are
  alternatives, and a control that vanishes when you fill its neighbour teaches
  nothing except that the page lost it. References win when both are attached,
  because that is what the run does, so they are the half that stays live.

  The note under the field says "keyframes are ignored" only when there is a
  keyframe to ignore. It said it unconditionally, which made the page's one
  mention of keyframes a warning about something you did not have, pointing at
  a control this layout had already made unfindable.

- **A box takes a photograph as well as a LoRA.** `regions_json.ref_image` is a
  latent mold: it pulls that rectangle toward that face during sampling, stacks
  with the box's LoRA, and runs on the fast single-pass path. It is not an
  `extra_ref_*` plate, so — unlike the scene and outfit tiles — it needs no
  identity-edit weight and never moves the run onto the slow krea2edit compose.
  A box with a photo and no LoRA is a character with no training run behind it,
  which is worth having on a platform whose other half is a trainer.

  Photos are capped at 1536px on the long side before they are encoded, because
  eight of them in one JSON body is the payload this invites. That is
  deliberately the *only* cap: the node's own `ref_max_side` is set to 0 so the
  resizing happens in one place and the two cannot end up fighting over which
  one shrank the picture. The job record carries a bool, never the bytes — it is
  polled every 400ms.

- **A prompt is written by reordering it.** ⌥← / ⌥→ moves the clause under the
  caret one slot along, because "in soft window light" belongs before the
  subject as often as after and doing that by hand is a select, a cut, a click
  and a paste — four gestures, each with its own way of eating a comma. The
  separators are slots and they do not move: the commas and line breaks stay
  where they are and the text between them changes places, so a prompt written
  across two lines still has two lines however many times you press the chord.

- **The empty prompt box is the worst control on the page, so it is not the
  only one.** H3 does not read a paragraph; it reads a document with named
  fields, published in the model repo. The composer offered a textarea for it,
  and every symptom of that is the same symptom: there is no slot for camera
  direction, so every position is a guess; tone and genre belong to a clause
  with no name on screen; the place a reference image's description belongs is
  not on the page, so it goes in the only box there is. A documented grammar
  presented as free prose reads as superstition — whether a comma or "the woman"
  versus "a woman" changes the take is not something anyone can infer — and a
  take is two to three minutes, so every guess is paid for at that rate.

  So the closed vocabulary is a **palette**: one icon in the strip, a popover of
  small animated tiles, and a rail of pills under the prompt. The prompt field
  keeps only what nothing else can say — who is in the shot and what happens.
  This is the "a control that shows its own value gets no label" rule applied to
  words instead of numbers, and it is the one place on the page where an icon
  can teach: a tile *shows* a dolly-out, which is the thing neither the word nor
  a static picture does.

  Three rules hold the rest together:

  - **No pills, no document.** With nothing chosen the compiler returns the
    typed text byte-for-byte. Every prompt written before this still means what
    it meant, and the document only appears once you have said something that
    needs one.
  - **The compiler never rewrites your sentence.** It closes it if you did not,
    and it chooses the separator in front of it — a leading clause's full stop
    softens to a comma before a lowercase fragment, so "A medium close-up, a
    portrait of k3nan." rather than a capital that would silently turn a `k3nan`
    trigger word into `K3nan`. That is the whole extent of it. What is inside
    `<d>…</d>` is not touched at all, which is the guide's own rule and the one
    place ordinary tidying would corrupt the output invisibly.
  - **`non_diegetic_music: N/A` is the default, and is worth the feature on its
    own.** H3 invented a soundtrack for every clip because nothing had ever told
    it not to.

  `/api/compile` is the same compiler on the same CPU container, so the
  disclosure under the rail shows the exact document that would run. A preview
  with its own implementation is a preview that can disagree with the run, which
  is worse than none; and without it the only way to answer "where did my camera
  direction go" was to render again.

- **A reference chip carries what it is *for*.** Identity, wardrobe, location,
  style, prop or action, from the chip's own menu, compiling to the guide's
  `<Subject 1> is the person in <Picture 1>` and a matching retention line.
  This is what makes "do not describe the picture you attached" enforceable
  rather than advice: there is now somewhere for that description to go which is
  not the prompt field, and it is one click. Roleless chips run exactly as they
  did.

- **The ratio picker and the pixel boxes are one control.** There is only ever
  a width and a height; the ratios are shortcuts to a pair of them. Picking one
  writes the boxes, typing in the boxes selects Custom, and Custom is the only
  option that spells out its own size, because it is the only one with no name
  that implies it. Sizes snap to 8 on the way out — the pipeline floors to the
  VAE's grid regardless, and a box that keeps 1000 while the model renders 992
  is a box that lied to you. It was 16 for a while, on the theory that the DiT's
  patch of 2 needs an even latent. It does not: `SingleStreamDiT.forward` pads
  up to the patch size and crops the result back, so an odd latent costs one row
  of patches. Snapping to the coarser grid was not protecting anything — it was
  refusing every second size the model can render.

  The swap arrow between the two boxes belongs to the same control, which is
  what decides where it lands: transposing a preset re-selects the preset for
  the transposed pair rather than dropping to Custom, and 3:4 exists on the menu
  because 4:3 — the ratio the page opens on — was the one landscape entry with
  no portrait counterpart to flip into. Arrow keys are part of it too: ↑/↓ steps
  a numeric box by one and ⌘↑/⌘↓ by eight, which on Width and Height is the
  VAE's grid, so the coarse step always lands on a size the model can render.
  Nothing is snapped there, because typing 1153 already shows 1153 until you
  leave the field — an arrow is a faster way to type a number, not a second way
  to commit one. Fields whose useful range is 1.0 to 1.4 carry their own
  `data-step`; a shift of 1.15 stepped by 8 leaves behind every value the model
  accepts.

`tools/preview_ui.py` serves `UI_HTML` against stubbed JSON, so the front end
is worked on locally instead of paying an image build and a cold start per CSS
change. Its stubs are shaped to hold the awkward states — a missing model, an
uncaptioned dataset, a prompt too long to belong in a gallery card.

## Where the console redesign is up to

Two things are open, and both are further along than "not started" — the method
is settled and the measurements exist, so neither needs re-deriving.

**Promote and demote.** The rule is: what survives is what a render actually
varies by, and the phone is the forcing function rather than an opinion about
which controls feel advanced. Three are already demoted below 640px and the
argument generalises to every width — the GPU is set once and confirms a cold
start when it changes; a seed is reused off a result, so the gesture happens
*after* a render and not before; a batch count is a decision the Generate button
could carry. What is left is applying the same read to the rest, and moving the
three out of the composer entirely rather than only on a phone. Sampling, size
and the shot palette are already popovers, so the strip is a row of doors — the
open question is whether a row is still the right shape once everything in it
is one.

**Bar or rail.** See the dead-space table under "The page". Desktop has zero
vertical slack at every aspect and 513–1068px of horizontal at four of five, so
the bar is measurably the wrong default there; a tablet in portrait inverts it.
The overlay was built and rejected. What has not been tried is placing the
console in whichever margin the render leaves empty, which the app can already
compute because it fits the frame. `tools/preview_ui.py` holds the awkward
states for both.

## Phases

1. Krea 2 LoRA training (musubi-tuner) — done
2. Image inference + datasets and captioning — done
3. Video inference via ComfyUI — done
   - MiniMax-H3: t2v, i2v, first/last frame, ref2va, native soundtrack
   - Wan 2.2: A14B t2v/i2v/first-last and TI2V 5B, with LoRA stacking
4. Image inference onto the same ComfyUI, with regional multi-character
   LoRA — done. One backend, one image, two GPU classes.
   - A box per character, each LoRA masked to its own rectangle
   - Scene and outfit transfer, when the identity-edit LoRA is downloaded
5. Video LoRA training — Wan 2.2 is the target, which is why phase 3 loads LoRAs

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

### One vocabulary, three destinations

`SHOT_VOCAB` is a table, not three tables. Each group declares which side reads
it and where its clause lands, and the three compilers differ only in what they
do with the result:

- **H3** gets the document — an alignment instruction and three named fields, or
  the six-field reference form when pictures are attached. `H3_ALIGN` holds the
  four instruction sentences verbatim, including the guide's own inconsistency
  (i2va and l2va bracket their labels, fl2va does not), because they are a
  contract with the checkpoint rather than phrasing we chose. `_h3_task()` is a
  deliberately *finer* read than the one `/api/video` makes: that one collapses
  to `ref2va` or `fl2va`, which is right for which checkpoint loads and too
  coarse for the alignment instruction, where first-only, last-only and both are
  three different sentences about where a picture sits in time.
- **Wan** gets prose, and gets it with the audio pills dropped — the same way a
  negative prompt is dropped for H3, because a sidecar recording an input the
  model never read is a sidecar that lies. Dropped by `needs`, not by field:
  dialogue is the case that breaks the simpler rule, landing in the *visual*
  description and still being audio, and `<d>[English] …</d>` arriving at umT5
  is a pair of angle brackets in the prompt rather than a line anyone says.
  `needs` is therefore per item as well as per group.
- **Krea 2** gets prose with camera, action and both audio groups filtered out
  by `image` in the table. Filtered, not silently dropped — the palette dims
  what the thing in front of you cannot read, and the group heading says why.

The job carries `prompt` (what ran) and `prompt_typed` + `shot` (what you
chose), and the sidecar only gains the second pair when the compiler did
something. The gallery, Reuse and the metadata sheet all prefer the typed one: a
card showing a six-field document is a card you cannot read, and restoring a
document into the prompt box would compile *that* on the next run.

### The settled question: `forge/` is gone

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
