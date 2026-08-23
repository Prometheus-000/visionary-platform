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
- **A wait costs what it shows, not what it takes.** The clearest statement of
  it is the owner's own, and it is worth quoting rather than paraphrasing:

  > I can sit here with you for 30-40 minutes while you work on a feature
  > because I can see everything you do, your thought process, the subagent
  > tasks, everything, so the wait doesn't feel like I'm waiting. Looking at a
  > black screen do absolutely nothing after you hit the main button even for a
  > minute feels like an eternity.

  Forty minutes of visible work is comfortable; sixty seconds of a still button
  is not. So the lever on a slow path is almost never *make it faster* — it is
  **say what it is doing**, and say something different as often as the thing
  itself changes. A phase that holds one string for forty-seven seconds is a
  spinner with extra words.

  This is why the loading stretch is narrated per model rather than as "Loading
  the model…": ComfyUI already prints `Requested to load MiniMaxH3` and
  `Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged`, so the
  page can read "loading MiniMax-H3 · 20.0 GB" and change six times through a
  window that used to change none. Nothing was measured or optimised to get
  that; it was output already being printed to a log nobody was reading.

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

- **Stops are cooperative — and a cooperative stop has to be readable and
  read.** Jobs check a flag between steps and unwind cleanly, so the container
  survives and the next request is warm. Killing the process is what you do when
  there is no other lever, not the default.

  Both halves of that failed at once and neither was visible. The flag lived on
  the job record, which `_publish` rewrites with get-update-put against a
  *network* Dict under a **process-local** lock — and Stop is pressed in the web
  container, which never takes it, so a publish straddling the press put
  `stop: False` back. `smoke_stop.py` enumerates the six interleavings and two
  of them lose it. It has its own key now, because a merge cannot clobber what
  it does not touch. And the flag was read in exactly one place, inside
  `_await`, which is reached only after the graph is posted — so every minute
  before that ran to completion whatever anybody pressed. `_stop_gate` is called
  at each phase boundary, where nothing holds a GPU and unwinding is free.
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

### The user's prose is the record; everything derived from it is a receipt

This survived the deletion of everything that was built to serve it, which is
the strongest thing that can be said for it.

> **What a model produces from somebody's sentence is a derived, disposable
> interpretation. The sentence is the record.**

*Derived*: regenerable at any time. *Disposable*: refusable whole, at any point,
with no loss beyond an interpretation. It is the relationship this file draws
between `prompt_typed` and the compiled `prompt` — intent is what is kept,
everything downstream of it is a receipt.

**And the receipt outlives its writer.** `_shot_meta` still writes
`prompt_original`, and nothing on any path fills it any more. It stays because
a sidecar is read years after it is written, and a reader that drops the field
makes every run that has one unreadable.

### What was built here, what it measured, and why none of it is left

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


### Relations are the weakest link, and it is the reason blocking exists

The observation is theirs and it names the failure precisely: **when a prompt
falls apart it is almost always because nothing says how the subjects stand to
each other, and the render comes back with subjects as props** — three people
described one at a time arrive photographed one at a time, each squared to the
lens, none of them party to what is happening. That is a large part of what
reads as "the AI look".

**A relation does not have to name anybody**, which is the way out. Naming a
second subject inside this one's clause opens an attention site for somebody who
already has one, and they merge or one vanishes — a real encoder failure, the
same one regional prompting exists for. But contact, orientation, a shared
surface, a shared light, attention pointed out of frame: a hand on the back of
her chair, one boot up on the same railing, close enough that their shoulders
overlap. All of that renders and none of it opens a second attention site. So
the rule is **supply the relation, err toward supplying it, and write it as
geometry**.

**And the placement is theirs too: the link goes at the end of the clause.** The
subject opens it and how they stand to the others closes it, so the last thing
read before the encoder moves on is what binds them. Leading with the link makes
the *relation* the subject; burying it mid-clause loses it between two
descriptions.

**Every word of that is now `_stage_clauses` rather than an instruction**, which
is the whole point of blocking: a relation derived from where two people are
standing cannot be forgotten, parroted, or written about somebody who is not
there. `STAGE_NEAR` is the proximity band, `STAGE_FACING` is orientation, and
the clause is assembled subject-first with the relation trailing. See "Blocking"
under Phases.

A previous attempt at this shipped a `ties` field that was validated, stored and
**read by no compiler** — the one channel the rules pointed at for relating two
subjects was a field the encoder never saw. Worth remembering as the shape of
the failure rather than the instance: a relation that does not reach the prompt
is a relation that does not exist.

### Arithmetic in the validator, judgement in the harness

Thresholds were retired and then rebuilt one layer over, in the test suite, and
it took being asked *"isn't this the entire reason for using an LLM"* to see it.
A keyword check can prove a relation reached the prompt; it cannot tell a good
relation from a clumsy one, which is the only question worth asking about a
relation.

The distinction that was collapsed, and is now the rule:

- **The validator stays arithmetic.** It runs on every request, gates a render
  and degrades in silence. A probabilistic gate stacked on a probabilistic
  writer is two coin flips where the second one is invisible. What belongs there
  are structural zeros.
- **The harness gets a judge.** It runs when somebody is measuring, latency is
  free, and what it measures — did this understand the scene — is exactly what a
  keyword list cannot answer. `tools/judge_prompts.py` is that judge and its
  rubric is four criteria plus `lost` and `contradicted`.

Three things keep it honest and they are the parts to not skip. **The judge is
not the subject** — point it at different weights, because a model scoring its
own output agrees with itself, and the harness says so out loud when the two
match. **Every verdict carries a quote** from the text it is marking, so a score
with nothing behind it is visible; a judge that cannot quote the fault has
usually invented it. And **it is an instrument rather than an oracle** —
spot-check it by reading, which is the discipline the thresholds never got. What
earns it its place is not that it is right, it is that it is *repeatable*, which
reading by hand is not.

**What it still cannot do is look at the picture.** Every criterion is finally
about a render, and judging the text is one remove from that. `does_it_help.py`
renders the pair, `judge_renders.py` scores it blind in both orders, and
`prompt_ab.py` runs the three stages in order. That is the measurement that is
not a proxy, and until it runs, a text judge should be read as one.


## Layout

    app.py              the whole application — images, jobs, API
    web/                the front end: React + TypeScript, built by Vite
    web/src/scene/      the scene composer — what replaced the video prompt box
    comfy_nodes/        our own ComfyUI nodes — one shim, see visionary_boxes
    tools/              smoke tests, the local UI preview
    tools/tune_dupes.py where the duplicate thresholds come from — takes a folder
    tools/ui-checks/    parity checks — each takes a URL, runs on either page
    tools/_from_app.py  pulls plain-Python pieces out of app.py by AST
    tools/smoke_scene.py  the scene compiler and the blocking derivation
    tools/prompt_ab.py  render a prompt pair and have a vision model judge it —
                        the only measurement of this that is not a proxy, and
                        the one command that runs the three stages in order
    tools/does_it_help.py renders the same sentence two ways, one seed
    tools/judge_renders.py scores a rendered pair blind, both orders, or it is a tie
    tools/judge_prompts.py the same rubric on text, which is cheaper and a proxy
    tools/serve_judge.py  opens a Sandbox with either judge's model in it
    tools/upstream.py   what moved upstream that would reach a render —
                        the answer to "a pin means falling behind"

**The front end is built into the image, not mounted from your disk.** That is
what keeps `modal deploy app.py` the entire install: mounting a local
`web/dist` would be simpler and would quietly make the deploy command a lie —
a fresh clone has no dist, and a stale one deploys whatever you last built,
which is the worst of the three because it looks like it worked. Node is a
build-time dependency of the image; nothing at runtime needs it.

The lockfile is copied before the sources so that editing a component re-runs
`npm run build` and not `npm ci`. Modal invalidates from the first changed
layer down, so the order of those four lines is a minute per deploy.

`UI_HTML` is **gone**. It was the oracle the React port was checked against —
`preview_ui.py` served it with no flags and the shipped bundle with `--dist` —
and this paragraph said for a long time that it should go once a real deploy had
been exercised. It did. Nothing reads it now, and the arrow-key note under "The
page" is kept as the record of what it got wrong, because that fault is the kind
that survives a port.

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

This is also why a LoRA is **named by the shortest unambiguous name**. One
`k3nan.safetensors` on the volume is `k3nan`; the matched Wan speed pairs, whose
files are both called `high` and `low`, are qualified by their folder.

That rule used to be about *typing* — it decided what `<lora:…>` you could write
and the note said which files a name matched when it matched several. LoRAs are
chips now, so nothing is typed and nothing is ambiguous: the name is what the
chip reads, and it is picked from a list. What survives unchanged is why the
short form has to be *correct* rather than merely pretty — "high" would name two
real files, and a label that says `high` twice is a picker you cannot use.

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

**The resolver outlived the syntax that needed it.** Nobody types a name any
more, so nothing goes "untypeable" — but a *sidecar* records what ran as a name
rather than a path, so `reuse.ts` still starts from one and still has to land on
the right file. The failure would just arrive somewhere else now: a reused card
silently coming back with one fewer LoRA than the run it claims to reproduce.

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

### A duplicate is a copy; a similar image is a photograph

A set arrives with duplicates in it far more often than not — the same shoot
exported twice, a JPEG beside the PNG it came from, a phone album pulled in
through two apps. They are not neutral: the trainer repeats every image the same
number of times, so a picture present three times is trained three times as hard
as the rest of the set, and the symptom ("everything comes out in that room")
never points back at the folder it came from.

**Two classes, and the second is not a softer first.** A *duplicate* is one
picture stored more than once, so deleting all but one loses nothing and the
group arrives with a keeper chosen. A *similar* pair is two photographs that look
alike — on a training set that is usually a burst, all of it legitimately useful,
and there is no deterministic way to prove the second is a re-save rather than
the next shutter release. So a similar group is shown and **nothing in it is ever
preselected**.

A five-tier confidence scale between those two was built first and is the thing
to not rebuild. Measured on a 731-image editorial set, 266,815 pairs: the tiered
version flags **813 pairs**, the two classes flag **9**. Worse than the noise was
where the tiers put the common case — a rule demoting same-size, same-format
pairs to "possible" sent *most real duplicates* (exports at one size from one
tool) into a review list where nothing is preselected, so the keeper flow never
ran on the case it exists for.

**Both hashes must agree, and they are not doing the same job.** dHash reads edge
gradients, pHash reads low-frequency DCT energy, and they fail independently, so
an AND is far tighter than either alone and much tighter than accepting on
whichever is closer. On that same folder `dhash <= 6` *alone* isolates exactly
the three real duplicates with a 7-bit gap to anything else — but the pair at
that gap is `d7 p32`, two entirely unrelated photographs, which is precisely what
the pHash half refuses. The pHash bound is then set by the other thing it has to
survive: a re-grade barely moves dHash (a quarter-stop is 3 bits, 1.4x is 4)
while pHash climbs to 16. Swept from 10 to 24 the duplicate count never leaves 3,
so the loosening is free and is what makes "the same picture, exported brighter"
read as a copy. The gap between the classes is therefore carried by dHash, 6
against 12 — and every burst-shaped pair in that folder measures 8 to 11, which
is the split the data drew rather than one chosen for it.

**Crop matching is the spec's own scheme on a leash, and the leash is the whole
reason it is safe.** Two centre crops per image, compared every way but
full-frame-to-full-frame. Taking the best of nine variant pairs is nine chances
to draw a low number against an unrelated image, so read at a loose threshold it
is ruinous — that is where most of the 813 came from. Read at `CROP_MATCH` it
adds **zero** pairs across 266,815, and lands an exact 80% crop of a real
photograph at distance 0 on both hashes. It may only ever claim *similar*, never
a duplicate, so a crop match cannot preselect a deletion — which is also right on
its own terms, because a deliberate reframe of a training image is a variation
somebody made on purpose. `CROP_MATCH` is deliberately its own constant even
though it agrees with `DUPLICATE_MATCH` today: they were one constant for an
afternoon, and loosening the duplicate bound for re-grades silently changed which
crops were found.

**There is no index, and the measurement is why.** A BK-tree was built for this
and profiled: at the radius the classifier uses it visits **96% of the tree per
lookup**, so it is the same sweep with a tree walk's overhead on top. What
actually inverts the cost is deduplicating fingerprints before comparing — the
pathological input, one picture four hundred times, collapses to one comparison —
and 266,815 pairs then take **0.6s**. The binding cost was never the comparison;
it is the decode, at ~31ms an image.

**So the scan is resumable, and the cache is its only state.** A request measures
for `SCAN_BUDGET_S`, writes what it measured, and reports how many are left; the
page calls again. No job record, no spawn, no second route — the fingerprint
cache already holds the progress, so a container dying mid-scan costs the images
it had in hand rather than the folder. Nothing is grouped until everything is
measured, because half a folder groups into half the truth and half the truth
here is a keeper suggested against copies nobody has looked at.

One line in that loop is load-bearing: **at least one image is measured per
request whatever the budget.** A budget check alone can be true before the first
decode, and then every request skips every image, writes nothing, and asks to be
called again forever. `smoke_dupes.py` drives exactly that at a zero budget; in
production the same stall arrives as one image slower than the whole budget. The
page carries the other half of the same invariant, refusing to loop when a round
reports no progress.

**An image is in at most one group, and duplicates win.** Similar links are
computed only between images no duplicate group already holds. That drops a real
edge — a copy's relationship to an outsider is not shown until the copies are
dealt with — and it buys the invariant the whole review rests on: a name in two
groups is a name you are asked about twice and can mark for deletion twice. The
order it imposes is the order the work happens in anyway: clear the duplicates,
rescan, review what is merely alike.

**The selection is inverted, and that is the feature.** Every other delete
surface here asks you to name what goes, which is the wrong half of the question
for six near-identical frames of which you want one: naming the five is five
decisions to express one. So a duplicate group arrives with everything but the
keeper marked, and the gesture is *promotion* — touch a marked image and it
becomes a keeper too, touch it back to demote. **The last keeper cannot be
demoted**, and that single refusal is what makes the screen safe to move quickly
through, because no sequence of clicks deletes a group entirely.

The suggestion says which number decided it — "most pixels · 12.2 MP", "same
size, least compressed" — against the *runner-up* rather than the group, because
"nothing separates the top two" is the one statement that tells you your choice
does not matter. `Derived or invented, always visible`, on the one surface where
the derivation is a deletion.

**The facts are one grid with the pictures as its columns.** The question a group
asks is never "how big is this one", it is "which of these four", and that is
read across: resolution, megapixels, weight, encoding and detail are rows, the
label sits once in the gutter, the best cell in each row is marked and every
other one carries its distance from that best. A per-card stack of the same
numbers was the first version and it made you hold four values in your head to
compare them. Two things that grid taught, both found by driving it rather than
reading it: `min-width:max-content` sizes a column to its widest cell, which is
the nowrap caption, so one long sentence blew a 150px column to 430px and the
square thumbnail above it to 430px tall; and `.actions` only gets its flex rules
inside `.opts`, so Keep all and Reset wrapped and doubled the header's height.

The delete count is computed over **every** group, never the filtered view. A
number that changes when you touch a dropdown you did not think was a decision is
the specific way a confirm dialog stops being believed — and that dialog is the
whole safety net, so it states the count, the weight, and how many of the groups
involved are still carrying a suggestion nobody has opened.

`tune_dupes.py` is where every number above came from and is the tool to re-run
before moving one: point it at a real folder and it prints the distance
distributions, the margin from the nearest *rejected* pair, and the pairs closest
to each line by name. A threshold argued from first principles is a threshold
nobody has looked at.

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
  watermark nobody mentioned is a watermark the LoRA learned.

  The instruction used to stay on the server with the page sending a key, for
  reproducibility. That lock is gone: the row shows the instruction in a
  textarea the preset prefills, edits apply to that run, and Save keeps them as
  a preset of your own (in the `config` Dict, beside the custom captioner
  repos the gear's Caption models section takes). What the rule was protecting
  moved rather than died — the job record now carries the exact composed
  instruction that ran, so a run is still replayable after the preset changes.
  What is *not* editable is what composes around the body: the trigger clause,
  the length clause and `CAPTION_RULES`, because those are the sentences the
  refusal and preamble parsing depend on. Write modes (skip, append, prepend,
  replace) and the sampling numbers travel with the job the same way, and
  find & replace across sidecars is scoped by the page's current filter — the
  filters are the targeting tool, and the count is on the button before you
  press it.

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
- **The vLLM recipe is settled by running it, and it outlived the model it was
  written for.** The interpreter behind `/api/parse` was its own `@app.cls` on
  its own L4 — deleted with the feature — and every line of how it was served
  survives in `tools/serve_judge.py`, because a judge needs the same thing: a
  `devel` CUDA base, since vLLM's inductor shells out to `nvcc` and a slim image
  dies at engine init with "Could not find nvcc", several minutes after a
  successful model load, which is what made it look like a timeout; vLLM
  deliberately *unpinned* where everything else here is pinned, because 0.11.0
  fails on a tokenizer attribute a newer transformers dropped; `/health` probed
  with urllib and never curl, since the CUDA base has no curl and `command not
  found` every iteration is indistinguishable from a slow start.

  **What that arrangement got right and is worth keeping:** a second model does
  not go inside a generator process. Both generators carry
  `@modal.concurrent(max_inputs=1)` because one GPU runs one sampling loop, and
  `_publish`'s lock is process-local *because* `max_containers=1` means there is
  no second writer — the arrangement that lost the terminal status 15 runs out
  of 15 before the lock existed.

  **There is no longer an exception.** `/api/motion` was one — a button somebody
  pressed and waited for, running on the encoder the container already held —
  and the argument for it was that a suggestion queuing behind a render is
  correct on a single-user platform. True, and it is the wrong way round: what
  actually happened is a *render* queuing behind a *suggestion*, because the
  slot is one slot and it does not care which direction you were thinking in.
  Nothing shares a generator process now.


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

  **The leak itself is closed now, and the recovery above stays anyway.**
  `comfy_nodes/visionary_free_regional` sits between the sampler and the decode
  on every regional graph and drops the session's device copies as the render
  ends: **1026 tensors a run**, with headroom flat at 46.7 GiB across three
  consecutive regional renders where it used to step down each time. The node
  pack has no teardown of its own — its `run()` has a `finally` and it only
  unhooks the forward hooks, while `_prepare` is guarded by `if "down_d" in d`,
  so the device copies are a cache whose eviction was never written.

  Three things about it are deliberate. It is **a node rather than a patch**,
  because CLIFF_SHA's whole claim is that nothing in it is patched — an install,
  not a vendor, with no `VENDOR.md` to keep in sync. It finds the session **by
  shape rather than by name**, because the pack stores it four ways depending on
  which fallbacks fire in `comfy.patcher_extension` and `ModelPatcher`, and the
  first version matched the wrapper case and silently missed the `model_options`
  one — which is not a crash, it is the leak returning with a log line saying
  nothing is wrong. `smoke_free_regional.py` covers all four. And it **takes the
  latent and returns it**, so ComfyUI cannot schedule it before the sampler; a
  node with no edge into the result would be free to run first, and freeing
  before `_prepare` builds is a rebuild every step rather than a fix.

  What it does not do is make `_reclaim()` redundant. A graph genuinely too big
  for the card still lands there, and a node cannot help a run that has already
  failed.

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

  **A cap on the pixels is not a cap on the payload, and that took a ten-minute
  render to notice.** The browser's copy resizes to 1536 and then re-encoded as
  PNG, which is 8.6x a photograph's own encoding — a number `shrinkB64`'s own
  comment already had, and had applied only to the *pass-through* case. Every
  photo over the cap took the other branch. Nine references is H3's maximum, so
  the case to size for is nine: **48 MB of base64 in one body**, up from the
  browser, through the web container, into Modal's blob store and back down to
  the GPU, before a weight is read. At q0.92 the same nine are 3.8 MB.

  Lossless bought nothing: every reference is consumed at `LoadImage`'s index 0
  and no graph in this file takes the MASK output, so alpha is discarded
  downstream whatever is sent.

  The general rule: an option whose cost is set by something the page never
  measured is an option that will be found by whoever has the biggest camera —
  and *how many* is as much a cost as *how big*.

## The page

The UI is not organised the way this file is. There are three subsystems and
two domains, and the page follows the domains.

- **One canvas at a time, and a batch is frames of film rather than a contact
  sheet.** A batch of four used to be a two-up grid at half height each, and
  that arrangement is what broke the regions: one set of boxes applies to the
  whole batch, so there is no single surface to draw them on, and they landed on
  the first cell of the four you were trying to compare. Drawing them four times
  would have said the wrong thing; drawing them once said it about one arbitrary
  cell.

  So each result fills the canvas and `‹ 1 / 4 ›` steps between them, one
  sliding out as the next slides in. The result on screen *is* the canvas —
  which is the shape the rest of this file has been heading toward anyway, and
  the surface an inpaint acts on when there is one. Every frame stays mounted
  and the track is a transform, so stepping costs a compositor frame and nothing
  re-fetches.

  It also retires a rule rather than leaving it to rot: the still on the canvas
  used to open the lightbox on click, because *"a result you have to send to the
  gallery to look at properly is a result the canvas is only pretending to show
  you"* — true when a batch was four half-height thumbnails, and answered
  directly by showing every result at full canvas size. What replaced the click
  is one rule with no geography in it: **a click addresses whatever is under
  it** — over a region it opens that region, over bare picture it does nothing.
  Full screen is the expand button and Space. The alternative was a click that
  meant two different things depending on invisible geometry, which hover can
  disclose on a desktop and cannot on glass.

- **A render is replaced when the next one lands, not when it is asked for.**
  Pressing Generate used to blank the canvas immediately: `start()` reset the
  whole run record, so the thing you were judging disappeared for the length of
  the run and came back as something else. On the image side that is seconds; on
  the video side a take is two to three minutes, and the shot you were deciding
  about is gone for all of them — at exactly the moment you wanted to compare.

  So the run record carries two ids. `jobId`/`files` are the render *on screen*,
  moved only by `finish`, and `runId` is the job being polled. They are different
  renders for the length of a run, which is the whole point: `fileUrl(jobId, f)`
  keeps addressing the old job's bytes while the new one is sampled. A run that
  fails or is stopped leaves the picture up too — a request that never produced
  anything should not take away the last one that did.

  What reports the run instead is a hairline along the top edge of the canvas.
  The full centred bar is for a cold start, where there is nothing to keep.

  And when the render is a clip, it *plays* when it lands — muted, looping. The
  `muted` is load-bearing, not a taste: every browser refuses unmuted autoplay
  without a gesture, so removing it (say, to honour H3's soundtrack) does not
  produce autoplay with sound, it produces a 200s render arriving as a strip
  parked at 0:00 — the product's best moment spent asking for a click. The
  soundtrack is one tap away on the native controls.

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

- **The console is sorted by how often you reach for something, and by nothing
  else.** The row holds what you touch constantly — the dimensions, the LoRAs,
  the shot pills, the regions. One button holds what you touch rarely — the
  model, the sampler, steps, CFG, shift, the seed, the batch count.

  The axis is frequency, and the near miss worth recording is *scope* —
  per-generation against per-session — because it sounds right and fails on its
  own examples. Almost nothing in the row genuinely changes every take: you do
  not pick a new aspect ratio per render. And CFG is not a thing you set once a
  session either; it is a thing you almost never set. Scope does not predict
  where a hand goes. Frequency does, and it is the one you can answer by
  watching yourself work.

  It also dissolves the case that looked like an exception under the other two.
  The seed *is* different on every render, so by scope it belongs in the row —
  but nobody types a seed. You draw a random one, see something worth keeping,
  and take it off the result. Rarely reached for, so behind the button, and no
  special pleading required.

  The button is named for the model, because the model is the rarely-touched
  choice that decides what every frequently-touched one *means*: which sizes
  exist, whether there is a negative prompt to write, whether LoRAs load at all.
  Naming it `8 steps · CFG 1.0` named it after two values nobody chose — those
  are the checkpoint's defaults. The checkpoint is the choice; the numbers are
  its consequences.

- **The console has a budget, and the prompt is what yields to it.** 30% of the
  viewport. Everything else in there is fixed or conditional — the strip is one
  row, the rail appears with the first pill, and the boxes cost it nothing —
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

- **The note line under the strip is reserved, and Generate does not move under
  your finger.** The warnings (`#console-notes`) used to mount on demand, which
  read as the obvious economy — a line that says nothing should cost nothing —
  and missed where the cost went: the console is pinned to the bottom of the
  stage, so a note appearing shifted every control in it by one line, at
  exactly the moments a hand is over Generate (the LoRA notes arrive
  mid-typing, the keyframe note mid-attach). One 18px row held at rest is the
  price of a button that stays where you aimed; `fieldMax` absorbs it, so the
  30% budget still holds at every viewport. Two spans share the row, and only
  a second line — both notes at once, or one wrapping — still moves anything.

- **Typing with nothing focused lands in the prompt, not in the hotkeys.** A
  click that misses the field by a few pixels leaves focus on the body, and a
  sentence typed next used to arrive as a hotkey barrage — every space toggling
  the full-screen viewer, every backspace one keystroke from clearing the
  canvas. The first stray letter now focuses the visible prompt and lands in
  it; everything after types natively. Letters only, and the focus is
  synchronous: Space keeps its full-screen meaning when it is the *first* key
  pressed, because a sentence never starts with one — and a focus deferred a
  frame (the `applyWrite` pattern) let the whole sentence arrive as stray keys
  while it was still in flight.

- **An icon invites, a tooltip names, a click teaches — and for a control whose
  home you are already in, that is enough.** The rule survives; what changed is
  that neither of the two things it was being used to justify was a control.

  `#g-shot` and `#g-regional` were 34px glyphs in the settings strip, side by
  side, and together they flattened the two biggest features in the app into one
  confusing pair — with a count riding half-outside the regions button that read
  as an error pip rather than "2 regions". Both were doing the one job an icon
  cannot do: announcing a capability to someone who does not know it exists. The
  strip is a row you scan when you already know what you want, which is why an
  icon works there for a *control* and fails there for a *destination*.

  So each took a word, and one of them moved. Regions is a canvas verb — you place
  a character by drawing on the empty frame, and the empty canvas says so in words,
  once, where the attention already is, so it left the strip entirely. Shot stayed
  where it was and gained the word it was missing: it writes into the prompt, and
  `+ LoRA` beside it is the same kind of door. Giving it a row of its own was tried
  and cost 34px at rest for one button — see the palette note below. What is left
  in the strip are controls and two doors that say where they go, which is what the
  rule was always about.

  It does not generalise past the row. `#ins-toggle` keeps "Captions" because it
  wears the same sliders glyph as two unrelated panels, so its icon is not
  identifying anything, and the hyperparameters keep their names because a bare
  number is not a value. The rule is that an icon can carry a control whose glyph
  is unambiguous *and* whose home you are already in.

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

- **The cell is the picture's shape, not a box the picture is fitted into.**
  Thumbnails were always `contain` rather than `cover`, because cropping throws
  away the thing you opened the gallery to see — and `contain` inside a fixed
  4:3 cell pays for that in bars: a portrait render arrived smaller than the
  space spent on it, and a set of portraits was two thirds `--wash-2`. The box
  was the half of that decision that cost. It is gone on desktop, and the cards
  are packed into columns instead, so nothing is letterboxed and nothing is
  cropped.

  **It is twenty lines rather than a dependency, and the reason is reading
  order.** Every cheap way to do this reflows it. CSS `columns` fills a column
  top to bottom, so a listing that is newest first becomes newest *down the
  left edge* — and the top-left card being the last thing you made is the
  gallery's whole contract. The React packages that keep the order distribute
  by `i % cols` and give up the packing, so the columns end up as uneven as the
  pictures are. The ones that pack properly measure the DOM, which means
  reflowing as each image decodes. Native `grid-template-rows:masonry` is still
  being re-argued as `item-flow` and is not something to ship on yet.

  A greedy shortest-column pack has the property none of those has: with every
  column starting empty the first row *is* items 1..n left to right, and each
  item after that lands in the column that is currently shortest, so it reads
  approximately row-major. It is also stable under paging — greedy over a
  prefix is the prefix of greedy over the whole list — so appending a page
  never moves a card already on screen.

  **Nothing is measured, because the server already knows.** `/api/gallery` and
  `/api/dataset` carry the pixel dimensions of every item, so the layout is
  decided before a byte of picture is fetched and a card with a slow cover does
  not move the ones around it. The fallback for a result with no sidecar is the
  4:3 box it always had, and the fallback is written in both places on purpose:
  a packer and a stylesheet disagreeing about one card is a gap under it.

  **It stops at 1024px.** Below that the grid still crops to squares — the one
  place this app trades information for density — and the two layouts are
  different DOM rather than two stylesheets, because a masonry's column
  elements would hand that grid its cards in chronological order *down* each
  column. There is no CSS that unpicks that: `display:contents` on the columns
  flattens them into the grid in column order, which is the wrong order rather
  than no order.

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

  **And the other half, which this rule reads as licence for if you stop at the
  first sentence: the small screen is the right place to find a fault and the
  wrong place to decide every screen has it.** Two changes made for a phone
  reached desktop by accident and both were wrong there. The last-generation
  thumbnail was declared outside every media query, so it sat beside Generate at
  1512px — where the drawer is already a column next to the picture and the
  header button already opens it, making it a second door onto a room that has
  one. And the viewer lost click-away-to-close, because a tap meaning "show me
  more of this" is right on glass and overwrites a convention on a mouse old
  enough that its absence reads as a broken dialog.

  Both were found by grepping for rules declared outside a media query, which is
  worth doing after any pass that starts on a phone. Where the two genuinely
  differ, split on the *pointer* rather than the width — `(hover:none)` is
  asking the question the layout actually cares about, and a tablet with a
  keyboard is neither of the things a width test thinks it is.
- **The seed rolls until you type one, and nothing writes it back for you.** It
  used to: `finish()` put the run's seed into the field when a document existed
  and the field was blank, so that editing one assumption moved only what that
  edit implied. It was gated on the document precisely because the ungated
  version is a trap — *a seed that stopped rolling for a person who never
  engaged reads as "Generate is broken, it keeps making the same picture"* — and
  with the documents deleted the gate can never be true, so the behaviour went
  with them rather than being ungated by default.

  **What made that safe to lose is that the seed is on the render.** It is drawn
  once, sent to the sampler, written to the sidecar beside the file, shown on the
  metadata sheet and restored by Reuse. A generation knows its own seed forever;
  what is gone is only the input box filling itself in.

  **The condition for bringing it back is a surface where editing an image is
  the thing you are doing**, and their reasoning is what names it: the question
  a pin has to answer is *"is this person iterating, or starting something"*,
  and the document was a bad proxy for it — it meant "you used a feature", not
  "you meant to hold this frame". An edit surface answers it by construction.
  Changing one thing about a picture that already exists cannot be the start of
  something new, so there is nothing left to infer and the seed simply holds.

  There is no such surface yet. Inpainting is sketched under Phase 6 and the
  krea2edit compose exists for scene and outfit plates, but neither is somebody
  sitting in front of a render changing one thing about it. **Until that exists,
  a pin has no honest trigger** — which is why this is deleted rather than left
  behind a flag.

  **And a pin is never cleared when the size or the model changes.** Considered
  and rejected, and recorded because it looks like a bug for as long as it is
  not. Same seed, wider frame is a comparison people actually run; clearing
  deletes it. It would also be the first control here that silently empties
  another field — `setImg`/`setVid` are shallow patches, and the only writes of
  `seed: ''` are the two Reset buttons, which is a person asking. A value that
  empties itself when you touch a control you did not think was a decision is
  the specific way a surface stops being believed. The number is visible in its
  own field and one gesture from gone, and that visibility is what makes
  clearing it for you unnecessary rather than merely risky.


- **Generate is the page, not a destination.** It has no nav item. Train is one
  door, labelled with where it leads rather than where you are, so two things
  never look equally selected. It carries the training run's progress, because
  a run lasts hours and you are meant to leave and keep working.
- **Image and video share the canvas and the gallery. The composer is
  per-kind.** This used to say the prompt survives the switch too, because a
  shot described as a still is the same sentence you would describe as a clip.
  True of the sentence and false of everything around it: a Krea 2 LoRA is not a
  Wan LoRA, and carrying one across loaded it into a run that could not use it —
  silently, because the note warns about names that resolve to nothing and about
  models that read no LoRAs at all, and a LoRA for the wrong architecture is
  neither. Prompt, negative, the pill rail and the chips now swap with the kind,
  and both buffers are kept: switch away and back and everything is as you left
  it, which is the promise `img`/`vid` already made about model, size and seed.
  One live set and one dormant, swapped in `setKind` — the invalid state is
  unreachable because there is only ever one to read. What still differs beyond
  that is the options, which rebuild from `VIDEO_MODELS` — see below.

  **The split has happened, and the stash is now half of what it was.** The
  video side lost its prompt box to the scene composer: its prose lives in
  `scene.shots`, which belongs to one kind and is therefore not stashed at all —
  the same arrangement `regions` has. What still swaps is the negative, the pills
  and the chips. Two buffers behind one live slot was the right *shape* for the
  transition and it is still the right shape for what is left of it.
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

- **A trigger phrase is text. A LoRA is a file, and it is a chip.** This section
  used to defend `<lora:name:0.8>` in the prompt — Automatic1111's syntax, the
  notation anyone who has trained these models already types — against a row per
  LoRA that cost 56px plus a wrapped select, 380px of canvas for four filenames
  and eight digits. The row objection was right and is not what changed.

  **What changed is that the two things in that field were never one thing.** A
  trigger phrase reaches the encoder, so it belongs in the prompt, positioned
  where the sentence wants it. A LoRA never reaches the encoder at all —
  `stripLoras` deleted it from the string before `/api/generate` was called and
  the stack travelled in its own field. It was inline for exactly one reason: so
  a number could be typed beside it.

  **And the argument that kept it there does not hold for the main prompt.** The
  claim was that a row *could not say where in the sentence the LoRA applies* —
  true in a region, false on the canvas, and the since-deleted `useDocument` said
  so itself: *a token's position in the main prompt means nothing to the backend,
  which reads them into a stack.* So the canvas paid for a parser, a
  caret-targeting scheme and a drag subsystem to buy a property only a box has,
  and what the position means in a box is *which box*, which a control living on
  that box says without any syntax.

  So a chip: the name, then a circle carrying the strength, with the second value
  — text-encoder weight on image, Wan expert on video — disclosed on a click,
  because both are omitted far more often than not. They fold behind
  `▸ 4 LoRAs`, a disclosure built as one more `#shot-peek`: zero pixels at rest,
  in flow so nothing sits on top of a render, and it does not close on scroll,
  which `Popover` does and which disqualifies it for a box you type numbers into.
  The count is a word rather than a pip, for the reason the regions button
  learned. `+ LoRA` stays exactly what it was — a picker in the strip — and the
  division is the shot rail's: the door adds, the disclosure shows.

  **A region has a dropdown on its own card**, one per box because that is the
  node's shape, and picking a second replaces rather than being refused. That is
  what retired caret-targeting: `+ LoRA` writing into "whichever of the two
  fields you last had the caret in" was answering *where* with a guess about
  where you were looking.

  **The picker writes nothing.** No token, no strength, and no trigger phrase —
  it shows the known phrase as text to read and place yourself. Three of the
  note's lines went with the syntax and two of them went because they became
  *impossible*: a chip is picked from a list, so no name resolves to nothing and
  none resolves to two. The third, a LoRA whose phrase is missing from the prose,
  can still happen and is deliberately unsaid. The failure it named is real —
  the weight loads, the render changes a little, and it reads as a LoRA that did
  nothing — so this is a decision to manage it by hand, not a discovery that the
  warning was wrong. `/api/state` still serves `trigger_word` per entry, which is
  where a check or an agent reads the fact.

  **`<lora:…>` left in a prompt is now text.** Nothing parses it, converts it or
  migrates it. `stripLoras` survives on the send path alone so a reused prompt
  carrying one does not render the literal word "lora" — it removes text on the
  way out rather than reinterpreting it in the box.

  **Dragging a LoRA is gone; dragging a file is not.** `lora/drag.ts` and its
  private MIME type were there to answer *where* — a click inherited the caret, a
  drag named its own target. Once every target owns a control there is no *where*
  left, and the gesture was long: open a menu at the prompt bar, at the bottom of
  the screen, then haul up to a box on the canvas. A reference image comes from
  the Finder, where a drag is the only gesture there is, so every file drop is
  untouched. See `docs/design-notes/loras-are-not-text.md`.

  **Everything above about *placement* is the Krea 2 side, and only that.** The
  `▸ 4 LoRAs` disclosure, `+ LoRA` in the strip, the region dropdown, `LoraBox`,
  `LoraButton` and every id in `check_loras.py` describe a console with a prompt
  box in it. **The video side does not have one**, so none of them are there: a
  LoRA is a chip on the one rail, beside the cast and the shot pills, and the
  picker is a mark on the field's trailing edge. Do not port these controls
  across, and do not read a rule about `#lora-box` as a rule about video.

  What *is* shared is the idea and nothing else: **a LoRA is a file plugged into a
  module, a trigger phrase is text, and the two do not live in the same place.**
  That is the whole transferable claim. Where the chip sits, what opens it and
  what the count says are per-side, because the two sides no longer share a
  console at all.

- **A region is drawn on the canvas, and so is everything about it.** The boxes
  are the list: drag on the frame to place one, drag it to move it, drag a
  handle to size it. Touch one and it *opens* — a card rooted in the box's own
  near edge, holding its sentence, its LoRA strength, its photograph and its
  four coordinates. Selection is the open state; there is no toggle and nothing
  to dismiss.

  **Editing what is inside a box and redrawing the box are two different acts,
  and one control for both taxed the frequent one.** Changing a region's
  sentence or its photograph is what you do a dozen times an hour; moving the
  rectangle is rare, and mostly happens once before the first render. A sentence
  is not geometry, so the frequent act needs no rectangles on screen at all —
  which is what `store.edit` encodes, in three states:

  - `off`, the moment any render lands. Nothing is drawn. The boxes are still
    there, still masking their LoRAs, still sent with the next run; they are
    *addressable rather than drawn*. Hover names what you would be touching, a
    plain click opens it, and that is the whole entry — there is no chip and no
    glyph, because a control that reveals the boxes is a control competing with
    the picture it would reveal them on.
  - `content`, the frequent act: that one box's card, with only its own hairline
    for scope. No other rectangles, and **no four coordinates** — they are the
    escape hatch for the *rectangle*, so putting them over a render you are
    judging is the other scope leaking back in. It does carry that box's
    handles, and it is worth reading why that is not the same leak.
  - `geometry`, the rare one: every box, its handles, the snapping, the four
    coordinates and the frame's own card. Asked for with **⌘-click**, or a long
    press where there is no modifier to hold.

  **An open box is adjustable, and the gate was never about that box.** This
  read `no handles` for a version, on the argument above: a sentence is not
  geometry, so the frequent act needs no rectangles on screen. True of the
  rectangles you have *not* touched, and the whole force of it comes from there
  — what geometry cost was eight boxes and sixty-four handles arriving over a
  render you were judging. Nothing is drawn until you click, so the box whose
  card is open is not one of those; it is already the scope of a card, already
  carrying a hairline, and already the thing you are looking at. Being told to
  press again with a modifier held to move *that* rectangle is friction with
  nothing behind it — the same invented friction as gating geometry behind
  clearing the canvas, arrived at from the other direction. So the open box
  behaves exactly as it does in geometry: handles, drag to move, snapping. Every
  other box stays undrawn and stays a tap.

  The coordinates stay behind the gate anyway. Dragging is the gesture, they are
  the escape hatch from it, and a render you are judging is the wrong place for
  four numeric fields — which is the original sentence, unweakened, because it
  was about the numbers rather than about the rectangle.

  **⌘ means geometry, and it means it in both places.** ⌘-drag on the frame was
  already "a new box, here", so this is the existing meaning extended rather
  than a second rule. It gates the mode and nothing else: one press that both
  revealed the boxes and drew one would answer "show me the boxes" by adding a
  ninth to the eight it just showed you. Gating geometry behind *clearing the
  canvas* was the version before this, and it was invented friction — a step
  that exists to protect a render nobody was being asked to give up.

  The two-press rule holds *while there is something behind the gate*. With no
  boxes at all, "show me the boxes" shows nothing, so the reveal-only press was
  a dead press — and worse, the layer used to mount over a render only when a
  box already existed, so on a fresh session ⌘-drag over the picture hit bare
  pixels and did nothing at all: the only way to draw a first box was to clear
  the render you wanted to draw against. The layer now mounts over every
  render (it paints nothing in `off`, so an empty one costs no chrome), and a
  ⌘-drag with an empty set falls through the reveal and draws the first box in
  one gesture.

  The four coordinates were the right parameter and the wrong primary. "0.5 0
  0.5 1" is a rectangle you rebuild in your head every time; the rectangle is
  not. They survive in the card, where they are the escape hatch and, during a
  drag, a readout that moves — dragging teaches the numbers, and the numbers
  never taught the dragging.

  **The canvas has no layers, so the boxes do not have them either — and they
  had the worst kind, the kind nobody chose.** They are absolutely-positioned
  siblings in `regions` order, so the DOM supplied a z-order out of an array
  index and it decided every click, every hover and every drop: whichever box
  was drawn *last* won. A performer placed inside a wide background box could be
  reached and the background box could not, and the eight handles of anything
  underneath were simply unreachable. Nothing in this feature has ever had a
  front or a back — a box is an area of one picture — so the ordering was a
  storage detail the picture could see.

  The rule that replaces it is one this file already states, in the Phase 6
  list, for exactly this ambiguity: **resolve toward the smaller object.**
  Widening a selection is cheap and obvious; guessing large silently edits the
  wrong scope. `boxAt` takes the smallest box containing the point, and a handle
  — smaller than any box — comes first, but only where it is *drawn*. An
  invisible handle is not an object, and letting one win would give every box
  four edges of theft over its neighbours at a target nobody can see.

  **Hover had to move with it, and that is the part that looks optional.** CSS
  `:hover` follows paint order, which is the ordering just deleted, so the
  hairline saying "this is what you would be touching" would have named a
  different box than the click opened. It is `.rbox.under`, set by the same hit
  test — one answer, painted and acted on. Not `.hot`: that is already this
  page's word for a drag held over a drop target, and `check_drop.py` reads it
  as exactly that.

  A drop reads the same test, which is the case with no undo. It used to take
  the topmost box while the caption named the topmost box too — consistent, and
  both wrong the moment a small box sat inside a large one.

  Two arrangements preceded this and each fixed the last one's cost. A row per
  region was a sentence and four numbers that needed a 32px picture of those
  numbers beside it to be legible at all, and at ~74px each the eight boxes the
  backend allows came to ~592px of console against a 54dvh cap: the feature's
  fullest state broke the rule the console exists to hold. One shared inspector
  row fixed the height and left the other half — you dragged a rectangle at the
  top of the screen and said who was in it at the bottom, and the row could not
  say which of the two scopes in it you were looking at. The card costs the
  console nothing at all: `probe_console.py` measures arming at 0px, where the
  row cost 44.

  **A card is not a portal.** Every other floating thing here is `Popover`,
  positioned in viewport pixels, which is why it closes on scroll — right for a
  menu and absurd for the field you are typing into. The card is a child of the
  region layer, so it is a percentage of the same box the rectangles are a
  percentage of: it moves when they move and is clipped by the frame, which is
  the correct answer rather than a limitation. It also has three placements
  rather than two — under the box, over it, and *inside* it pinned to its own
  bottom edge, because the two rectangles arming seeds are full-height columns
  and the very first card anyone sees is the one with nowhere outside the box to
  go.

  It stays mounted through a drag and goes transparent instead. Unmounting was
  the first version and it re-ran the card's mount effects on release, which
  pulled focus into the prompt after *moving* a box — so ⌫ edited text instead
  of deleting the rectangle, which is exactly the keyboard fault this redesign
  set out to fix, reintroduced by the fix. `check_regions.py` found it on its
  first run, which is the argument for that file existing.

  Boxes snap to halves, thirds and quarters and to each other, which is what
  makes an even split a gesture rather than a menu; Alt suppresses it. Arming
  the mode seeds two half-width columns rather than explaining anything, because
  two rectangles appearing on the canvas is the instruction. What the boxes are
  drawn *on* is the frame at the render's aspect, or the scene plate if one is
  dropped, or the last render — adjusting boxes against the picture you actually
  got is the reason they are still there afterwards.

- **The frame is a place too, and it has the same card.** Scene, outfit and the
  region weight are about every box at once, so they cannot live on any one
  rectangle. They used to sit in the same row as the selected box's own fields,
  separated by a rule and by nothing else — two scopes in one strip, with the
  reader left to infer which was which. Now the scope *is* where the card is,
  and it is reached by one button in the corner of the layer, because a tap on
  bare canvas is already taken: it draws a rectangle. Escape is the keyboard
  half of the same thing. There is always exactly one card and it is about
  whatever is selected.

  The frame's card is also the only thing on screen that can say which engine
  the run takes — a plate moves the render onto a krea2edit compose that
  regenerates the whole frame and is several times slower, and no arrangement of
  rectangles shows that.

- **The map went with the row.** A 52px SVG of the boxes lived beside the
  inspector because the boxes come off the picture the moment a render lands,
  and something had to lead back to them. Two things did, and it was the lesser:
  the mode button reveals rather than disarms on the first press after a render,
  and a file dragged over the window brings them back on its own. What the map
  additionally did — reach a box that is overlapped or off-screen — is Tab,
  which cycles them and always could, once clicking one actually focused it.

  **`Nothing sits on top of a render` survives, and the three states above are
  what it costs to keep it.** It is written here because a canvas-native rework
  is exactly the moment someone assumes it was relaxed. A finished render puts
  the boxes away and the card with them, every time, including after the render
  you asked for while editing them — `off` is re-entered on every land, so the
  clean picture is never something you have to restore. A persistent hairline
  was considered and rejected for the same reason it always was: chrome on the
  one surface the layout exists to keep clear.

  What did change is the second half of the sentence. Nothing is *drawn* on a
  render; the regions are still *there*, and touching one opens it. That is not
  a relaxation, it is Phase 6's own rule arriving early — *every element
  addressable at all times, none of them drawn as a control* — and it is the
  first place in this app where the two halves of that rule are separable at
  all. The distinction to hold on to: this rule is about paint, and it never had
  anything to say about reach.

- **What a place carries is attachments, and an attachment has a role.** A
  photograph in a box is that character's likeness; the two frame-scope plates
  are the scene the picture is generated inside and an outfit transferred onto
  the subjects. They reach the backend under three different names — a region's
  `ref`, `scene`, `outfit` — and they are one thing: a picture, and what it is
  for. So the page holds one record for all three, and one function attaches it,
  taking the place as an argument. That argument is the entire difference
  between "this character" and "this scene".

  This is the axis the next capability arrives on, and it is worth writing down
  before there is anything at stake. **There is no ControlNet in this backend** —
  no node, no mask primitive, nothing in the catalogue — so this is a question
  about shape rather than a feature to build, which is the cheap time to answer
  it. The shape is already latent in three places and named in none: the region
  mold's V12 parameters are `ref_strength`, `ref_start_percent`,
  `ref_end_percent` and `ref_feather` scoped to a box, which is ControlNet's
  parameter set exactly and is hardcoded; `refs_json` is already a positional
  list of `{role, note}`, pinned to scene and object; and the video side's
  `SHOT_REF_ROLES` is the same vocabulary in the open.

  So a ControlNet is `depth`, `pose` or `edges` — a role on a picture, not a
  feature. V12 returns `(MODEL, +COND, −COND)`, so a `ControlNetApplyAdvanced`
  slots between it and the `KSampler` without touching anything above it, and
  adding one should be four things: a catalogue entry, which gets download UI
  and status for free; a preprocessor on a **CPU** container; a branch in
  `_krea2_graph`; and a row in the role table. **Zero new front-end surface** —
  dropped on the frame it is frame-wide, dropped on a box it is masked.

  **The standing veto: no conditioning panel, no layers list, no "control"
  tab.** When four roles exist, a list of them somewhere is the obvious cheap
  fix and it is precisely the month-four failure Phase 6 names. An attachment is
  drawn on the thing it is attached to, or it is not drawn. And `Derived or
  invented, always visible` lands here too: a depth map is derived, so it has to
  be visible on the attachment and cheap to reroll.

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

- **Grey is whose, the underline is reach — and the mirror had to change layers
  to say the first one.** The prompt box marks what the interpreter supplied,
  inline in the sentence, and that is two channels rather than one. Every
  *element* carries a dotted underline, because an element is a thing the
  document can act on: one is a single gesture from being something else.
  Whether the words in it are the person's or the model's is a separate claim
  laid over the top, and only the model's are grey.

  An earlier note here said derived text carries no mark at all, on the grounds
  that underlining what somebody wrote would mark nearly every word. That was
  right about a rule which underlined all derived text and wrong about this one:
  an element is an anchor and what hangs off it, so the marks land on the
  handful of things in the sentence there is anything to *do* to, and the
  connective tissue between them carries nothing.

  What it cost is worth recording, because it looks like a regression on the way
  past. The mirror used to paint decoration onto transparent glyphs while the
  textarea supplied the visible text, and a *colour* is impossible that way
  round — the half that can be styled is the half nobody can see. So the ink
  moved one layer down: the mirror's glyphs are what you read and the textarea
  is transparent with `caret-color` set. It keeps the caret, the selection, the
  undo stack and every chord in `keys()`, which is the whole reason this is
  still a textarea rather than a contenteditable.

  **The only thing a grey run does that plain text cannot is reroll**, and that
  is the whole of its affordance: rooted at the run's own end, revealed while
  the caret is inside it, gone when it leaves. Editing one needs no control at
  all — `remap` drops the mark the edit landed on, so the words turn dark and
  become yours with no gesture and nothing to commit. An inline editable rooted
  in the run was considered and rejected: a second text surface competing with
  the one underneath it, buying nothing typing does not already do.

  A reroll lands three ways — new text, the same text, a rejection — and **the
  in-flight state sits on the affordance and settles identically for all
  three.** Not a pulse under the words: that is motion on the render surface to
  announce a null result, and a flicker that fired for *identical* while a
  rejection stayed silent would build a channel telling you which way the
  validator went, which is the one thing the silent degrade exists not to say.

- **A prompt is written by reordering it.** ⌥← / ⌥→ moves the clause under the
  caret one slot along, because "in soft window light" belongs before the
  subject as often as after and doing that by hand is a select, a cut, a click
  and a paste — four gestures, each with its own way of eating a comma. The
  separators are slots and they do not move: the commas and line breaks stay
  where they are and the text between them changes places, so a prompt written
  across two lines still has two lines however many times you press the chord.

- **The empty prompt box is the worst control on the page, so on the video side
  there is not one.** H3 does not read a paragraph; it reads a document with
  named fields, published in the model repo. The composer offered a textarea for
  it, and every symptom of that is the same symptom: there is no slot for camera
  direction, so every position is a guess; tone and genre belong to a clause
  with no name on screen; the place a reference image's description belongs is
  not on the page, so it goes in the only box there is. A documented grammar
  presented as free prose reads as superstition — whether a comma or "the woman"
  versus "a woman" changes the take is not something anyone can infer — and a
  take is two to three minutes, so every guess is paid for at that rate.

  The palette below was the first half of the answer and the **scene composer**
  is the second — see "The scene composer" further down, and
  `docs/design-notes/console-ladder.html` for the design it was built to.

  So the closed vocabulary is a **palette**: a door in the strip beside `+ LoRA`,
  a popover of small animated tiles, and the pills themselves under the prompt.
  The prompt field keeps only what nothing else can say — who is in the shot and
  what happens. This is the "a control that shows its own value gets no label"
  rule applied to words instead of numbers, and it is the one place on the page
  where an icon can teach: a tile *shows* a dolly-out, which is the thing neither
  the word nor a static picture does.

  **The door was a wordless glyph, and that — not its room — was the fault.** It
  was a 34px mark beside `+ LoRA`, in a row you scan when you already know what
  you want, next to a second opaque mark doing the same disappearing act for
  regions. Regions left that row for good, because it is a canvas verb. Shot did
  not: it writes into the prompt, and `+ LoRA` is already precedent for the strip
  hosting a door that does that. What it needed was a word.

  It carries the word "Shot", and the icon rule is what says so rather than what
  it breaks: a glyph can carry a control whose home you are already in, and it
  cannot announce a *destination* — and eighty-seven tiles behind one press is a
  destination. The tile beside the word is the teaser, and it **animates on hover
  and is frozen otherwise**, which is the pills' own rule: the page at rest runs
  nothing, so a loop under the prompt would be motion competing with the canvas
  for attention you have not asked it for. Hover is the asking. The glyph follows
  the kind — a dolly-out on video, a framing on images — because the camera group
  is video-only and teasing a move the model cannot make is a promise the run
  will not keep.

  **It sat at the head of the pill rail for a version, and the measurement is what
  moved it back.** The rail is the right *room* — the door and the words that come
  out of it in one place, needing no caption — and it was the wrong price. A row
  that exists only to hold one button costs 34px of a console capped at 30% of the
  viewport, at rest, forever: resting went 120px to 154px, while the rail carrying
  sixteen pills only ever added 29px on top of that, because then the row is
  holding something. `#shot-rail:empty{display:none}` is back, and the rule it
  encodes — the rail costs nothing until there is something in it — turns out to be
  load-bearing rather than tidy. A line per button is not a price this console can
  pay, and the general form is worth keeping: **a row is affordable when it carries
  content, and never when it carries one control.**

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

  **A tier is a multiplier, except where it is a family.** 1K, 1.5K and 2K are
  one trained set — ~1.03 MP on a 64 grid — multiplied, because scaling a bucket
  keeps the shape the model knows while deriving one from the ratio at a new edge
  length lands on a size nothing was trained on. 1.3K is not that set made bigger:
  it is ~1.62 MP on a 32 grid, and 1.25x the bucket agrees with it on 1:1 and
  misses every other row by 32-48px — 1440x1120 where the family's 4:3 is
  1472x1120. So that tier pins its eight sizes and the multiplier is what a bucket
  falls back to when it has no pin, which is what keeps adding a ninth ratio one
  line rather than a column somebody has to remember to fill in.

  The ratio set is one vocabulary across every tier, not a set that changes under
  the scale buttons — so 21:9 and 4:5 exist at 1K too, and theirs are the only two
  rows in the table that are derived rather than inherited. Derived to the label
  *exactly*, 1568x672 being 7:3 and 896x1120 being 4:5, because where the model was
  never given a bucket there is nothing to be faithful to except the name.

  The swap arrow between the two boxes belongs to the same control, which is
  what decides where it lands: transposing a preset re-selects the preset for
  the transposed pair rather than dropping to Custom. Five of the eight transpose
  within the menu and three do not, so 4:3, 21:9 and 4:5 **deselect** — Custom, at
  the transposed pixels.

  That is the answer and not the cost of one. 3:4 was on the menu purely to give
  this button somewhere to land, and it went with the ratio set; the flip of a
  ratio the menu has no counterpart for *is* a custom ratio, so a tile left lit
  there would be the picker claiming a shape it does not offer. Re-adding a ratio
  to keep a tile lit is the tail wagging the dog, and special-casing it in
  `swapSize` is worse — it lights a tile for a bucket the picker cannot otherwise
  reach. Arrow keys are part of it too: ↑/↓ steps
  a numeric box by one and ⌘↑/⌘↓ by eight, which on Width and Height is the
  VAE's grid, so the coarse step always lands on a size the model can render.
  Nothing is snapped there, because typing 1153 already shows 1153 until you
  leave the field — an arrow is a faster way to type a number, not a second way
  to commit one. Fields whose useful range is 1.0 to 1.4 carry their own
  `data-step`; a shift of 1.15 stepped by 8 leaves behind every value the model
  accepts.

  **That paragraph described an intent the page did not deliver for most of its
  life, and the note is kept because the shape of the fault recurs.** In
  `UI_HTML` the handler was delegated from three sections and the sizer popover
  was appended to `<body>`, so with it open there were two pairs of boxes —
  `#g-w`/`#g-h`, inside a listening section and invisible, and `#sz-w`/`#sz-h`,
  visible and outside every one of them. The arrows stepped the inputs nobody
  could focus and did nothing to the boxes you actually type in, and nobody
  noticed for months, because a rule that reads as shipped is one nobody thinks
  to test.

  The React front end does not inherit it: the component owns its own keys, so
  where a popover renders stops being able to break them.
  `tools/ui-checks/probe_size.py` asserts it and now passes every row — it used
  to fail two by design, and that exemption went with `UI_HTML`.

`tools/preview_ui.py` serves the built bundle against stubbed JSON, so the front
end is worked on locally instead of paying an image build and a cold start per
CSS change. Its stubs are shaped to hold the awkward states — a missing model, an
uncaptioned dataset, a prompt too long to belong in a gallery card.

**A stub that omits a menu is a preview of a control that does not exist**, and
it fails silently — the case that taught this was `rewrite_ops` missing from
`/api/state`, which rendered no Enhance button at all, so the one surface that
feature had was invisible in the very file that exists to make the front end
developable without a GPU. The button is gone now; the rule is not. Every menu
the page builds itself out of is pulled from app.py rather than transcribed, and
pulling one means pulling **what it references** — a subset naming a constant
nobody pulled raises `NameError` from inside app.py, which reads as a broken
preview rather than an incomplete pull. That is `_from_app.py`'s one failure
mode and the reason its subset is named rather than pattern-matched. It fired
twice in one session over `_stage_*`, which is the rule working.

## Where the console redesign got to

**Promote and demote — done.** The rule that settled it is under "The page":
sort by how often you reach for a control, not by whether it is per-take or
per-session. What you touch constantly is the row; what you touch rarely is
behind the model button. Measured, the image strip was 1016px
of controls and the video 979px, most of it spent on things a take does not
change. Seed and the batch count went into the Sampling popover — a seed is
*reused off a result*, so the gesture happens after a render and not before, and
a batch is a run parameter like steps. The GPU went under the gear, because it
is chosen once a session and already confirms a cold start when it changes. The
video side finally got the size control the image side had, collapsing a
separate aspect select and tier select into one button reading `16:9 · 768p`.
Result: 732px and 652px.

Two more have left the strip since, and not to a popover: `#g-shot` and
`#g-regional` were not controls at all. See the icon rule under "The page" —
each went to the surface it acts on, which leaves this row holding the size, the
LoRA picker and the model button, and nothing that has to introduce itself.

**Unbounded buttons — done.** Ten controls each in their own box is ten boxes
competing with the picture above them, and the chrome is not what makes a
control a control: the value is, and every one of these already shows its own.
So the box is spent only when it is doing something — the pointer is over it, or
a popover is open from it. Scoped to the console: Train keeps its edges, because
it is a form you fill in rather than a row you scan, and a form without field
edges is a worse form. The line that decides is the same one that governs
labels — a control whose value *is* its label can lose its box; a free-text
field cannot, because an empty one has nothing to show and no edge to aim at.

**Bar or rail — settled: the bar stays.** The dead-space table under "The page"
says a rail would fit on desktop, and it is still not the answer. A rail is a
dated shape that will read as of-its-decade long before this app stops being
useful, and the overlay that would have avoided both was built and rejected for
putting chrome on the render. What the measurements really argued for was less
in the bar, not the bar somewhere else — which is what the popovers did.

The direction instead is **unbounded buttons**: controls that carry no chrome
until they need it, so the strip stops looking like a strip. It is a step toward
the console disappearing entirely, which is the actual goal the rail was a
detour around.

## The scene composer

**The video side has no prompt box.** H3 reads a document with named fields —
shots, cut times, speaker IDs, a retention line per subject — and a text field
has no shots in it, which is why `_compile_h3_prompt` could not emit half the
grammar it was written against: the composer never collected it. The design is
`docs/design-notes/the-scene-is-the-prompt.md` for the argument and
**`docs/design-notes/console-ladder.html` for the layout**, and the second
supersedes `scene-composer.html`, which hides the canvas behind a 132px stub. The
ladder draws all five states at true proportion against `CONSOLE_BUDGET`, and its
own footnote is why to trust the picture over the prose: twice while it was being
drawn the two disagreed and the prose was the optimistic one both times.

**The degrade is exact, and everything rests on it.** One shot, no cast, no
pills, and `readScene` returns null: no `scene` key is sent and the run is the
typed text byte-for-byte, in a box that is identical to the prompt box it
replaced, under the same `#prompt` id. Nothing about the video side changes until
somebody asks for a second shot. That is what let this land in three commits
rather than one, and it is the rule to check first if any of it is ever moved.

- **The gesture creates the cast.** Typing `@` floats a picker off the caret;
  picking makes the member and drops the handle. So the cast has no empty state,
  because it is never on screen before it exists — which is `+ LoRA` inverted:
  there you open a picker in order to insert a name, here the name you are
  already typing *is* the picker. The three doors on the field's trailing edge
  reach the same thing without knowing the gesture.

  **Where the caret is comes from the mirror**, and that is the third job it has
  done. It is a glyph-for-glyph copy of the textarea, so a zero-width span
  spliced into it at the caret's index sits exactly where the caret does — the
  only way anything on this page can answer that question. It was built for
  provenance marks, kept when they were deleted on the grounds that unwinding it
  bought nothing, and it is now load-bearing again.

- **The share is a readout, not an input.** A shot's slice of the clip is the
  length of what you wrote about it. That is not a shortcut — it is what removes
  a 9px drag target instead of enlarging it to a thumb-sized one, which is the
  question a phone frame asked and the console could not afford to answer. Extent
  drives prominence; `beats` survives as the precision escape hatch, the way a
  region's four coordinates do.

- **Rows divide the field's existing allowance rather than adding to it.**
  `growRows` is `autoGrow` over n boxes and reduces to it exactly at n=1, so a
  four-shot scene costs what a long prompt costs and the 30% budget needs no
  second arithmetic. Measured at 1512x982 with three shots and the source row
  showing: 287px, 29.2%.

- **The slot is the role**, and a file over a slot that cannot take it does not
  highlight. The rejection is the absence of an invitation, which arrives before
  the drop rather than after it; read off `dataTransfer.items` during the drag,
  the one moment the answer is available and still useful. `DropTile` alerts
  instead and is right to — its tiles all take images, so a wrong file there is
  worth naming, and here the row *is* the table of what takes what.

- **One rail, three kinds of chip** — cast, shot pills, LoRAs. They were three
  mechanisms in two places and they are one tool: each is something attached to
  the thing you are making, and each opens its own card when touched.

- **The pool is keyed by what travels**, not by the file on disk. A photograph
  and its own re-export shrink to the same bytes, and two entries in
  `references[]` pointing at one picture would put every `<Picture N>` after the
  first off by one with nothing on screen saying so. It also makes the numbering
  derivable rather than kept in step by hand, which is what `refs`/`refRoles`
  had to do — not fixed, *unrepresentable*.

- **Everything that floats is portalled and fixed.** `.console` is
  `overflow:auto` — the rule that stops a full console pushing the canvas out of
  frame — so a child of it floating above its own top edge is clipped by it. Not
  `Popover` either: that closes on scroll, which is right for a menu and wrong
  for a box you type a name into. Two faults found here and both are the same
  shape: a ref read during the parent's render is null and nothing re-renders to
  correct it, so the menu placed itself at 0,0; and picking left the menu open on
  the name just chosen, which is a picker that will not take yes for an answer.

### The compiled document is view source

**Reading the prompt is not a step in composing.** It was frame 6 of the ladder
and that was the error: the scene is the source and the document is what it
compiles to, so the precedent is not a disclosure, it is devtools — and four
things follow rather than being chosen.

- **No button says "inspect".** Devtools is a chord and a context menu and the
  page never advertises it, so the composer carries nothing for this. ⌘⌥U, which
  is the browser's own chord for exactly this, or right-click the console — one
  of the two has to be findable without being told. It stands aside for a text
  field's spelling menu or an image's Save Image rather than deciding it outranks
  the platform.
- **It is not in the console budget, because it is not in the console.** It takes
  the canvas the way devtools takes the window, and nobody is judging a render
  while reading a prompt. The surface that kept breaking the budget stops
  competing for it.
- **A textarea, not a `<pre>`,** and the edit is what runs. Where the precedent
  stops applying: a devtools edit evaporates on reload because the source file is
  the truth you port back to, and here that would be a render you cannot
  reproduce. So editing **detaches** — one bit for the whole document, visible in
  the header, one gesture back. Not per-field pinning: six independent states
  nobody asked for, and an attempt to make a derived surface partly
  authoritative. Either the scene is driving it or you have taken it over, and
  you can always see which.
- **It shows derivation.** Put the caret in a `[Shot N]` block and that row lights
  up below. Only while attached — a detached document is arbitrary text and its
  markers are claims about nothing.

`prompt_compiled` is its own key on `/api/video` rather than a rewrite of
`prompt`, and this is the case that separates the two halves of a run:
`prompt_typed` stays the prose somebody wrote and only the receipt is overridden.
Folded together, the sidecar's intent field would hold a six-field schema and
Reuse would load it into the first shot's row and compile *that* on the next run.
It is stripped and never `_oneline`d — that function exists because a newline
inside a *field* ends it early, and this is the document, where one field per
line is the format.

### What it does not do yet

- **Blocking is not reachable from it.** `_stage_*` and `_stage_boxes` are live
  in the compiler and nothing fills them in, which is exactly the state the scene
  itself was in before this. `web/src/blocking/` is still the throwaway probe.
- **Wan gets the timeline and cannot read it.** `/api/video` compiles a scene
  only for H3; on Wan the rows join into prose. Nothing on screen says so, and by
  the rule that a control present but ignored is worse than one absent, something
  should.
- **`retention` reaches the document at its default and no control sets it.**
  That is a default that reaches the output rather than a `ties` — but the moment
  it wants a control, it wants one on the cast card and not in a panel.
- **None of it has been measured against a render.** It has been driven and read.
  `tools/prompt_ab.py` is the measurement that is not a proxy and it has not run.

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
5. Video LoRA training — Wan 2.2 was the target, which is why phase 3 loads
   LoRAs. **Tabled, and the reason is a number** — see below.
6. **The Dynamic Canvas** — next, and sketched rather than specified below

The end state is one application where a generated still flows into a clip
without a round trip through the filesystem — the "Animate" and "As reference"
buttons on a finished image are the first piece of that.

### Phase 5 — Wan LoRA training, and why it is tabled

The pipeline is nearly free and the weights are not. That is the whole finding,
and it is recorded here because everything above it argues the opposite — phase
3 loads LoRAs *so that* this could happen, and the trainer already runs musubi.

`train_job` is already the exact three-step shape Wan wants. Only the names change:

| | Krea 2, today | Wan 2.2 |
|---|---|---|
| latents | `krea2_cache_latents.py --vae` | `wan_cache_latents.py --vae` |
| text | `krea2_cache_text_encoder_outputs.py --text_encoder` | `wan_cache_text_encoder_outputs.py --t5` |
| train | `krea2_train_network.py`, `networks.lora_krea2` | `wan_train_network.py`, `networks.lora_wan`, `--task t2v-A14B` |

The A14B pair trains in one run rather than two — `--dit <low>
--dit_high_noise <high> --timestep_boundary 0.875` — so it extends the existing
job/status/stop contract with a recipe parameter and invents nothing. By the
rule against building a second way to do the first thing, this is the shape it
should take whenever it happens.

**What stops it: musubi cannot train the weights this platform downloads.** Its
Wan doc is explicit that *"fp8_scaled models are not supported even with
`--fp8_scaled`"*, and every 14B DiT in `MODELS` is `fp8_scaled` because that is
the right choice for inference. The text encoder is the same story — we hold
`umt5_xxl_fp8_e4m3fn_scaled.safetensors`, musubi wants
`models_t5_umt5-xxl-enc-bf16.pth`. Only the VAE is shared, and it is shared
exactly: `wan_2.1_vae.safetensors` is the file musubi's own doc names.

So training needs a second copy of models already on the volume, at a precision
inference does not want:

- `wan2.2_t2v_{high,low}_noise_14B_fp16.safetensors` — 26.6 GB each, 53.2 GB the pair
- the bf16 T5 `.pth` from `Wan-AI/Wan2.1-I2V-14B-720P` — about 11 GB

Roughly 64 GB, of which 53 is a duplicate at a different dtype. At 244 MB/s
that is four minutes of transfer, so the cost is storage rather than time — but
it is storage on a volume whose only way to reclaim space is the Modal CLI.

Two things follow, and they outlive the decision to wait:

- **The catalogue's one-entry-per-model assumption does not survive training.**
  Whenever this is picked up, those entries have to say why two precisions of
  one weight exist, or the next person reading the list deletes the one that
  looks redundant.
- **"Downloaded" and "trainable" become different questions.** A Train surface
  offering Wan cannot check `present` — it has to check for the training-capable
  file specifically, or it offers a run that dies after the dataset is cached.

Tabled rather than abandoned: nothing above is a blocker, it is a bill, and it
should be paid deliberately rather than by a pull request that quietly adds
64 GB to a catalogue.

### Phase 6 — The Dynamic Canvas

The canvas becomes where the work is done rather than where the result appears.
Region prompts and LoRAs set on the canvas. Every attachment a drop — a photo
onto a box is that character, a photo onto the frame is the scene, and the video
side's keyframes and reference pictures arrive the same way instead of through a
tray of 32px tiles. One gesture, one place, for every picture a model can be
given.

What you touch answers with its own affordances, at the place you touched it,
and only the ones that object can respond to. Not a menu at a fixed corner: that
is a panel in miniature parked at a coordinate, and it puts the same affordance
count on every object regardless of what the object is. The numeric escape hatch
— X, Y, W, H — stays reachable behind one of those affordances rather than
beside them, because it is precision work you occasionally need and a drag
cannot give.

**The region bar and the map are gone as of the canvas-native rework above**,
and the question below is answered: the map went with the row rather than being
rehomed, because the boxes still come off a finished render and two cheaper
things already lead back to them. The rest of this section stands. It
eliminates the region bar, which raised the question worth settling first,
because the answer decides whether this is a move or a deletion: of the three
things in that bar only one is regional. The scene and outfit plates only exist
when regions are armed, so they follow. The region prompt becomes canvas-native,
which is the point. But the map exists *because* the boxes hide on a finished
render — if the dynamic canvas keeps them legible over a result, the map has no
job left and should go with the row rather than be rehomed out of habit.

**Why the app does not already work this way.** The platform was built in a
familiar idiom on purpose — a console, a Generate button, panels, modes —
because the guts had to exist and be understood before a new surface could be
imagined on top of them. Driving ComfyUI, the job/status/stop contract, the
volume layout, regional masking, the prompt compiler, LoRA training: none of
that is guessable from a mockup, and designing a novel interaction over a
backend you do not yet understand produces a surface that cannot be built. So
the conventional version came first, and it is scaffolding rather than debt.
Read the console that way — it is what made the rest legible, and it is expected
to go.

**Part of this waits on models, not on effort.** Three tiers, with very
different costs:

- *Reachable now.* Touch-to-select and touch-to-edit. Segmentation on a finished
  render turns flat pixels into addressable regions, and regional inpainting
  re-renders one mask. The regional path here is already element-as-component in
  primitive form; the gap is that boxes are author-declared rather than
  model-derived, which is a segmentation step and not a research problem.
- *Reachable but expensive.* Elements that persist across renders. A LoRA gives
  that for a character, which is why this platform trains them — but making it
  general means a LoRA per element and a training run per thing kept, which
  changes what a library is: weights, not images.
- *Not there.* A frame computed from a spatial arrangement rather than authored,
  and a camera with real physical limits. That is causality between a 3D scene
  and a 2D image, which diffusion does not model. It needs a world model, or a
  hybrid where real 3D drives conditioning, and nothing off the shelf does it.

The consequence: the interaction is mostly reachable and the physics is not, so
this phase should chase the first tier and leave the third alone until the model
exists. Attempting it, failing, and concluding the whole direction is fantasy is
the specific mistake this paragraph exists to prevent.

### Where this is going, and what that vetoes

Phase 6 is the first move toward an end state worth stating in full, because
every one of these exists to kill a specific cheap fix — and the cheap fix is
always a panel. If one of these has never vetoed anything, it is not earning its
place.

**If a gesture cannot communicate intent without a label, the answer is a better
gesture or a smarter reading of it — never a control panel.** Every other rule
here is an instance of this one. It also kills the tutorial: onboarding, coach
marks, first-run overlays, the little animated hand. Those are labels in a
costume. A gesture that needs teaching is the wrong gesture.

**The canvas never changes.** Character, wardrobe, action, location, camera —
one surface. What changes is what the surface is willing to hear. Kills tabs,
rooms, modes, a spine, a back button. Today's Generate/Train/Datasets split is
on the wrong side of this.

**What you touch decides the mode.** You do not enter character mode, you touch
a face. Mode is a consequence of attention, never a precondition for it. Kills
tool palettes, mode switchers, a selected-tool state, anything you must do
*before* the thing you meant to do. Ambiguity resolves toward the smaller
object — a hand over a counter is the hand; widening is cheap and obvious, while
guessing large silently edits the wrong scope.

**Everything is live, nothing is labelled.** Every element addressable at all
times, none of them drawn as a control. This is the hardest engineering in the
product and it exists so the screen can stay empty. Kills inspectors, sidebars,
property panels, toolbars, persistent chrome — which is what the console is.

**Nothing asks for confirmation.** Move something and the frame changes. Kills
modals, "are you sure", staged edits, and the Generate button. The trade is that
undo must be total, because reversibility is what replaces confirmation.

**Duration starts at zero.** A still is the default and time is added. Someone
who wants one image should finish and leave without learning that motion exists.
Kills a timeline on first run and any flow treating a photograph as the
degenerate case.

**The frame is computed, never authored.** Arrangement produces the frame; you
never compose by dragging contents around a viewport. This is what makes direct
manipulation honest — you move the world, not the picture of it.

**The camera is a character.** It has a position, a path, a want and real limits.
It can be late, it can look away, it can be wrong. Kills the camera as a settings
group and lens choice as taste.

**Derived or invented, always visible.** What was read from your words is marked
one way; what was filled in for you is marked another and is cheap to reroll.
This is the entire trust surface and it needs no dialogue. Kills the chat panel,
the assistant sidebar, and clarifying questions — every question asked is a small
failure, so pick something, mark it invented, move on.

**The library is closed until you reach for it.** A drawer, not a workspace.
Browsed, not searched. Things are applied, never imported: edits in a scene are
scene-local, edits in the library propagate, and those are two different acts in
two different places. Kills a persistent rail, a docked browser, an asset
manager.

**It never remembers unless told.** Same words, same result, a year later. Kills
personalisation, suggestion engines, learned preferences, recently-used
defaults, anything that pre-populates. The first screen of the thousandth
session is the first screen of the first.

**On touch, and the trap in it.** Designing for a tablet is a constraint that
does the work for you: no hover kills progressive disclosure, no right-click
kills the hidden second layer, imprecise input forces you to manipulate objects
instead of widgets. But that argument runs ahead of the tools — keyboard and
pointer are first-class here and will be for a long time. What it is really
guarding against is a feature that works *only* with a cursor, because that is a
fork, and the touch half of a fork becomes the degraded half within two
releases. Design to the coarser input; let precision be a bonus. Hover earns its
place for exactly one thing — showing what you would be touching before you
touch it — which is spatial feedback about scope rather than a control being
revealed. "No hover" is not "remove the shortcuts".

**A prompt is a compilation target, not something the user writes.** Nobody
should have to learn a text encoder. Every model arrives with its own grammar —
H3 wants a six-field document, Wan wants prose, Krea 2 wants prose with the
camera clauses dropped — and asking a person to hold three formats in their head
so a checkpoint can be fed correctly is the tool making its problem into theirs.
The user says what they want; the app knows what each model needs; the prompt
itself is an implementation detail they should never have to see.

Half of this is already built and is the proof it works: `SHOT_VOCAB` is one
vocabulary with three compilers behind it, and nobody types
`integrated_multimodal_description:`. Replacing the prompt field entirely — a
model reading intent instead of a table matching pills — is the same idea with
a better front half, and the direction this is going.

Reproducing a prompt is not worth protecting in the first place. A prompt is a
guess at how to speak unnaturally to one text encoder — an artefact of which
checkpoint you happened to be feeding on the day, with no value of its own. The
same intent producing a different prompt twice is not a defect to engineer
around; it is what a conversation is, and the second prompt is no less arbitrary
than the first.

**What is worth keeping is the intent, and that is what the sidecar should be
read as recording.** `prompt_typed` and `shot` are the durable half — what you
actually meant — and the compiled `prompt` is a receipt: this is what this
encoder was told, this once. Reuse, Copy and the metadata sheet prefer the typed
one, which reads as a legibility choice and is really the deeper one.

The gallery itself shows neither, and never has: a card carries the picture, its
kind and its age. Worth saying because "show the prompt on the card" is an
obvious-looking addition and it is wrong twice over — a six-field document is
not readable at thumbnail size, and a prompt is an implementation detail of
whichever encoder was being fed that day.

It also pays off the moment a model is replaced. Intent recompiles for whatever
comes next; a stored prompt is worth nothing to a checkpoint that wants a
different grammar, and every prompt in the gallery would otherwise be tied to
the encoder it was written against.

**And that is the actual shape of it: you argue with the picture, you do not
author a prompt.** "Make her older." "The light is too warm." "No, the other
one." Iteration is conversational and corrective, the way any real working
session is — including the ones that produced this file. What nobody should be
doing is nudging commas, swapping a period for a comma, or shuffling clauses
around a paragraph to see what the encoder does differently. That is a person
performing machine work because the machine will not do it for them.

The tell is in how AI writing is caught: it reads as machine-made because it is
*too smooth*. People think and write in bursts — fragments, a correction, a
second thought that contradicts the first, the real point arriving third. A tool
that demands one clean flowing paragraph is demanding that a human produce the
artefact of a machine before the machine will listen. Backwards.

So fragments are the expected input, not the degraded case. Out of order,
incomplete, self-correcting, arriving in pieces over a minute — that is the
normal shape of intent and the system's job is to take it that way. The prompt
that comes out the other side is the app's business, and stays in the sidecar so
a run can always be replayed.

**Why this matters beyond ergonomics: it puts authorship back in the work.**
The industry built extraordinary visual technology and then made it usable only
by engineers. Prompt engineering is a programming skill wearing an artist's
clothes, and it produced a strange artefact — a prompt is copy-pasteable, so
someone else's method transfers whole. Take their string, get their results,
generate endless variations in their voice without ever having their taste. That
is what hollows out AI-assisted work: the authored thing is a text file, and a
text file is a tool anybody can pick up.

The proof is public. Every image on Civitai looks the same, and it is not
because the models cannot do anything else — it is because the prompt is the
method, methods get shared, and a shared method converges. The same incantations
get pasted in front of every subject, the same negative boilerplate behind it,
the same dozen checkpoints and LoRAs underneath. The result is a house style that
nobody chose and nobody authored, arrived at by thousands of people
independently copying each other's strings. That is what an aesthetic looks like
when its unit of authorship is copy-pasteable.

A conversation does not transfer that way. Not because it cannot be copied —
anything can be copied — but because copying it gives you a record of somebody
else's session rather than an instrument you can point at your own. It is long,
situated, full of corrections that only make sense against what was on screen at
the time, and useless without the reactions that produced it. Nobody has the
same argument twice, including the person who had it.

**The strongest version of this is not conversation, it is that the instrument
is made of your own material.** Authorship at the prompt is thin — it is the
last few centimetres of a pipeline somebody else built. Authorship in something
trained or conditioned on your own photographs is the model itself learning what
you meant, and it is why training is Phase 1 here rather than an advanced
feature bolted on later. The trainer is in the application, the picker reads the
volume rather than a registry, and a dataset is a folder of your images with
your captions beside them. That shape is not an accident and it is not for
convenience: it is what a tool looks like when the person using it makes their
own instruments.

**This paragraph used to say "owning the weights", and that sentence was the
vehicle wearing the principle's clothes.** Every argument above it is about one
thing, and the direction matters: **the unit of authorship must not be
transferable.** A prompt fails that test precisely *because* it copies whole —
paste the string, get the look, bring no taste. A LoRA passes because it is not
a string. **And a set of your own photographs passes for exactly the same
reason** — copying it hands somebody your pictures, not your eye.

Stated the other way round this reads as a virtue, which is how the sentence
went wrong the first time: "owning the weights" sounds like a claim about
possession when it is a claim about *non-transferability*. The weights were one
way to hold your own material, not the reason it mattered, and promoting them to
the headline made a rule about a file format out of a rule about authorship.

That is not a pedantic distinction, because the format is already moving.
Personalisation is shifting from training toward reference conditioning —
IC-LoRA, ID-LoRA, multi-reference identity anchoring, a reusable character
object binding reference sets and wardrobe across models. `H3_CAST_KINDS` and
the cast buckets are already that object: slots holding *your* pictures, bound
to a person, read by whichever model is in front of them. Same authorship,
different vehicle. LoRA becomes the self-hosted tier rather than the definition.

**So the veto is about instruments, not about weights.** Nothing in this
application browses, searches or recommends somebody else's instruments —
weights, reference sets, character bibles, style packs, or whatever the next
format is called. Written the old way it had a hole exactly where the drift
happened: a "browse character packs" surface contains no weights at all, would
pass the letter of the rule, and would undo the paragraph above completely. It
is the same feature that looks obviously helpful in a pull request. Pulling a
specific file you were sent is fine — that is what the Drive route is for. A
marketplace is not, whatever it is a marketplace *of*.

Which is where this stops being a philosophy and becomes the file above: if
intent is the durable record and the prompt is a receipt, then what the sidecar
keeps *is* the authorship. The technical decision and the artistic one are the
same decision, and that is the strongest argument that the technical one is
right.

**The failure to guard against is not the model, the latency or the scope.** It
is month four, when something does not fit cleanly and the cheapest fix is a
panel. That is the moment this becomes a node graph with better typography.

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
| LoRAs          | yes                  | yes                         |
| References     | ref2va — pictures, videos and audio | no           |
| Experts        | one                  | two on A14B, one on the 5B  |

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

### 2K is absent on purpose

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
something. Reuse, Copy and the metadata sheet all prefer the typed one, because
restoring a document into the prompt box would compile *that* on the next run.
The gallery shows no prompt at all — see the note above.

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
