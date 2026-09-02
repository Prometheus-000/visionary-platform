---
paths:
  - "app.py"
  - "comfy_nodes/**"
  - "tools/**"
---

# The backend

Everything whose scope is `app.py`, the ComfyUI nodes and the tools. Loads when
Claude reads a file matching the paths above; the root `CLAUDE.md` holds the
rules that apply before any file is opened.

## Antifragile, in detail

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

- **Prefer the explanation with an unbounded shape over the one with a
  computable ceiling.** A payload has a ceiling — bytes over a rate, worked out
  in a minute. A queue does not. And anything that can take minutes says which
  minutes they are, on both the page and the log: the phase names the step, and
  `[api] spawned in Ns` / `accepted` stamp the hop a large body actually
  travels. The ten-minute render that established this, including the diagnosis
  that was wrong and why it was reached for, is in `docs/decisions.md`.
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

## Scalable

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

## Future-proof

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

## Relations, and the reason blocking exists

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

## Arithmetic in the validator, judgement in the harness

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

## Layout, and why it is shaped this way

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

Set `VISIONARY_VOLUME` to run a second copy against its own storage.

### The volume holds weights and what you saved, and nothing else

Stated by the owner on 2026-09-01, and the root cause was found the same
afternoon: the web function set no `scaledown_window`, so the container died
about a minute after the last request. A cold start was the steady state, and
under that fact the mount was the only place guaranteed to exist — so it
became the read path, and every cache, marker and draft followed it onto the
volume because nowhere else lasted two minutes. The reload freeze that every
gallery bug traced to was the *cost* of that dependence, not its cause. *"I
hate relying on modal volume for anything except model files or anything I
intentionally tell to save."* With the window at Modal's maximum the
container is what it was taken for: fast, free, roomy, and there when you come
back. Reading committed state by RPC stays, because a longer-lived container
sees a mount that lags longer — the two fixes are complementary.

The test before writing a file to the volume is which of two things it is:

- **Derived** — a thumbnail, an index, a heartbeat, a scratch copy, a node
  catalogue. It goes on the web container's own disk (`SPOOL`, fast, free,
  LRU-trimmed, and gone when the container is) or in a Modal Dict, and it must
  be rebuildable from what *is* on the volume. A cold start pays to rebuild
  it; that is the honest price of a container that scales to zero.
- **A record** — what you made, and what you meant by it. It goes *inside* the
  file it describes: a PNG text chunk, an MP4 comment atom. Never a JSON beside
  it, because a file dragged out of the browser then leaves its record behind,
  which is the sentence-is-the-record thesis backwards.

**There is no named exception, and that is deliberate.** `drafts/` looked like
one — an unsaved set kept on the volume so an upload survives the web
container dying — and the owner refused it: *"exceptions have a way of setting
precedence … unsaved datasets aren't holy. If I do a lot of work on a dataset,
I will simply press save."* So an unsaved set lives on the container's disk
and dies with it, which is what unsaved means, and Save is the one gesture
that reaches the volume. The consequence to design for: anything that rents a
container — captioning, dedupe, insight, training — reads a *saved* set,
because a draft on this container's disk is invisible to every other one.

**What the pass retired (2026-09-02), so nobody re-derives the list:** the
`visionary.json` sidecar (the record is a PNG `tEXt` chunk or an MP4
metadata key now, `_output_record` and `_read_record`); the job folders
(`outputs/` is flat, `{job}_{NN}.png`, `{job}.mp4`, and the run is read off
the name by `_group_of`); `.thumbs/` under runs and under sets (covers and
thumbnails are built by whoever has the pixels and kept in the spool);
`drafts/.sessions` (a timestamp in the sessions Dict); `drafts/` itself
(container disk); `work/` (each container's own scratch); and
`.node_catalogue.json` (the harvest's *return value*, read once by call id
and kept in the spool). A one-time job moves legacy job folders into place,
started on first sight from the listing; legacy volume drafts are adopted
into `datasets/` rather than lost, once, with a log line. The one honest
second file that survives is a clip's motion-context tensor, `{job}.context.
safetensors`, which is bytes rather than a record.

### A storyboard is a folder

`storyboard/<name>/board.json`, with the pictures dropped onto that board
beside it. A panel's picture is a pointer — `{file, gallery: true}` into the
flat `outputs/` for a render, `{file}` for an upload in this folder — never a copy, so
deleting a render leaves the panel's words standing and deleting a board
touches nothing in the gallery. The folder is the whole import format: copy
one in and `/api/storyboards` lists it. Uploads are uprighted and capped at
`STORYBOARD_MAX_SIDE` on arrival, once, because a panel's picture becomes a
keyframe at the hand-off and the keyframe path would resize to the same edge
anyway — one cap, in one place, so the wall and the run read the same bytes.
The board's pills are `_validate_shot`'s own, which is what makes a panel a
shot's intent rather than a translation of one.

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
`k3nan.safetensors` on the volume is `k3nan`; two folders whose files are both
called `high` and `low` are qualified by their folder.

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
more, so nothing goes "untypeable" — but the *record* keeps what ran as a name
rather than a path, so `reuse.ts` still starts from one and still has to land on
the right file. The failure would just arrive somewhere else now: a reused card
silently coming back with one fewer LoRA than the run it claims to reproduce.

### Saving a set is a choice, and it is the only thing `drafts/` means

Dropping images makes a **draft**. It filters, captions by hand and reviews
exactly like a saved set — same folder shape, same sidecars, same code path —
and the difference is where it sits: **on the web container's disk**, under
`DRAFTS`, not on the volume. Saving moves the folder into `datasets/` under
the name you type, which is a copy across filesystems and the one gesture
that reaches the volume; the page never asks for a name before the images
are in front of you, because "is this worth keeping" is not a question you
can answer at drop time.

What that costs, and is designed for: anything that rents another container
reads a *saved* set. The captioner and the trainer refuse a draft with a
sentence that names Save, because a folder on this container is invisible
to every other machine. The dedupe scan and the insight run in-process and
work on a draft as they are.

A draft lives as long as the container, twenty minutes past the last request,
and the drop surface says so. Within that life the window that made it still
heartbeats `/api/session` — a timestamp in the sessions Dict, never a file —
and a draft whose window has been quiet for fifteen minutes is swept, the
folder's own mtime counting as a heartbeat so an upload still writing cannot
be swept out from under itself. The overlay that reads committed captions
knows a draft has nothing committed and reads the disk.

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

**Both hashes must agree, and they decide only the duplicate class.** dHash
reads edge gradients, pHash reads low-frequency DCT energy, and they fail
independently, so an AND is far tighter than either alone and much tighter than
accepting on whichever is closer. On that same folder `dhash <= 6` *alone*
isolates exactly the three real duplicates with a 7-bit gap to anything else —
but the pair at that gap is `d7 p32`, two entirely unrelated photographs, which
is precisely what the pHash half refuses. The pHash bound is then set by the
other thing it has to survive: a re-grade barely moves dHash (a quarter-stop is
3 bits, 1.4x is 4) while pHash climbs to 16. Swept from 10 to 24 the duplicate
count never leaves 3, so the loosening is free and is what makes "the same
picture, exported brighter" read as a copy.

**The similar class is read by an embedding, because the hashes cannot read
it — and the calibration lesson is the part to keep.** The editorial folder
the thresholds came from contained only exact copies, so `SIMILAR_MATCH` was a
line drawn from data with no true near-duplicate in it. Measured against
ground truth that has them (INRIA Holidays, 500 same-scene groups, 1.1M
pairs), the hash band's best case is 5% recall, no threshold rescues it, and
199 burst-tier pairs sit at hash distances up to d40/p36 — a hash measures
*storage* similarity and cannot see the next shutter release. So similar is
now cosine over a CLIP ViT-B/32 embedding (int8 ONNX on CPU, baked into
`web_image` by pinned revision and checksum), at `SIMILAR_COSINE = 0.94`:
below that line both calibration folders put true re-takes and false pairs —
two different models on one backdrop at 0.923, one model in two looks at
0.925 — at the same cosine, so no lower line is honest. The hash band
survives only as the fallback where the model is absent (the tools on a dev
machine), and `SIMILAR_MATCH` is its constant.

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
  because that is a fact about the run rather than a preference — the token is
  prepended in Python once the caption is back, so the model has to be told both
  that the subject has a name and that writing it would double it. Nothing else
  composes around it any more. The length is substituted *into* the body at
  `{length}`, and the rulebook that used to ride behind every preset is gone:
  the bodies are JoyCaption's own trained instruction strings now, and stacking
  our wording for the same rules on top of a model already taught them in
  different words is one prompt arguing with itself. Write modes (skip, append, prepend,
  replace) and the sampling numbers travel with the job the same way, and
  find & replace across sidecars is scoped by the page's current filter — the
  filters are the targeting tool, and the count is on the button before you
  press it.

  The captioner picker is the other half, and the default moved. JoyCaption Beta
  One replaced Qwen3-VL because Qwen lost bindings on anything harder than a
  single subject — an arm resting on a friend's shoulder came back as an arm on
  a red bench, confidently, in prose nothing downstream can flag. The full
  record is above `CAPTION_MODELS`. Qwen stays in the table because old job
  records name it and a table that drops an entry makes those runs unreadable.

  A decline is the other thing that is not an exception. A stock instruct model
  declines to describe photographs of real people often enough to matter, and on
  a character set that is *every* image; what arrives is fluent prose that passes
  every check downstream of it, lands in a `.txt` sidecar and trains, so the
  symptom is a LoRA that learned to say it cannot describe someone.
  `_looks_like_refusal()` drops it before it is written, which leaves the image
  in the Uncaptioned filter where it can be found, and `CAPTION_MODELS` keeps the
  abliterated repackage for it — same architecture, same loader, so the fix costs
  a repo id rather than a second code path. JoyCaption does not refuse at all,
  which retires the reason that entry existed without retiring the entry. The
  count of refusals is reported by name and points at the other menu, because
  "run finished, nineteen of twenty-four captioned" is not a diagnosis.

  **The message shape is per-captioner and settled once per run.** The menu is
  any vision LM `AutoModelForImageTextToText` maps, and those do not agree on
  how an image reaches the template: Qwen takes it as a content part, JoyCaption
  wants `content` to be a plain string and splices its own placeholder in.
  `_caption_shape()` probe-renders both against a 64px image before the weights
  load and `_vlm_inputs()` builds whichever worked. Before the loop, because the
  shape cannot vary between images — discovering it per image was one unreadable
  line per file, zero captions, and an A100 rented for the length of the set.
  Parts is tried first and that order is load-bearing: Qwen's template renders
  the flat shape too, dropping the image while doing it.

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


- **Results are served off the container's spool; the volume is the record.**
  Every picture bug the gallery ever had traced back to one dependency:
  serving bytes off the mount makes the page's freshness hang on
  `volume.reload()`, and reload is refusable — by our own `FileResponse`
  descriptors most of all, so painting pictures froze the view the next
  picture needed, and a render that had just finished 404'd on the canvas
  while its files sat on the volume. The whole read path is off the mount
  now: the listing's entry set (`_entries_by_rpc`), the records
  (`_read_record`, off each file's head), the covers and the files themselves (`_spooled`) all
  read the volume's *committed state* by RPC, which both job writers commit
  immediately and which no open descriptor can refuse. A file is pulled once
  onto local disk and served from there — so a clip's range requests land on
  local disk too, and serving can no longer freeze anything. The spool is
  LRU-trimmed at `SPOOL_MAX_BYTES` so it follows the session around instead
  of growing with the volume, which is the entire difference between it and
  the mount. The mount keeps what it is for: writes, deletes, and datasets —
  which this container writes itself, so its own view of them is fresh by
  construction.

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
  The group is the unit you decide in — you want the video stack or you do not —
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

## The receipt outlives its writer

`_shot_meta` still writes `prompt_original`, and nothing on any path fills it
any more. It stays because a sidecar is read years after it is written, and a
reader that drops the field makes every run that has one unreadable. The same
reasoning is why `readVidChips` sends one number where an `expert` field used to
be: a field whose only value is "both" is a sidecar implying a choice nobody
had — but clips rendered while it existed still record theirs, and the metadata
sheet still reads it.

## One vocabulary, three destinations

`SHOT_VOCAB` is a table, not three tables. Each group declares which side reads
it and where its clause lands, and the compilers differ only in what they do
with the result. There are two now and there were three; the middle one is kept
in this list because it is the case that shaped `needs`:

- **H3** gets the document — an alignment instruction and three named fields, or
  the six-field reference form when pictures are attached. `H3_ALIGN` holds the
  four instruction sentences verbatim, including the guide's own inconsistency
  (i2va and l2va bracket their labels, fl2va does not), because they are a
  contract with the checkpoint rather than phrasing we chose. `_h3_task()` is a
  deliberately *finer* read than the one `/api/video` makes: that one collapses
  to `ref2va` or `fl2va`, which is right for which checkpoint loads and too
  coarse for the alignment instruction, where first-only, last-only and both are
  three different sentences about where a picture sits in time.
- **A silent family** got prose with the audio pills dropped — the same way a
  negative prompt is dropped for H3, because a sidecar recording an input the
  model never read is a sidecar that lies. Dropped by `needs`, not by field:
  dialogue is the case that breaks the simpler rule, landing in the *visual*
  description and still being audio, and `<d>[English] …</d>` arriving at a
  text-only encoder is a pair of angle brackets in the prompt rather than a line
  anyone says. `needs` is therefore per item as well as per group. Nothing dims
  today — H3 reads every group — and the column is kept as the one a second
  family arrives on, with `_shot_phrases(..., audio=False)` still the mechanism.
- **Krea 2** gets prose with camera, action and both audio groups filtered out
  by `image` in the table. Filtered, not silently dropped — the palette dims
  what the thing in front of you cannot read, and the group heading says why.

The job carries `prompt` (what ran) and `prompt_typed` + `shot` (what you
chose), and the sidecar only gains the second pair when the compiler did
something. Reuse, Copy and the metadata sheet all prefer the typed one, because
restoring a document into the prompt box would compile *that* on the next run.
The gallery shows no prompt at all — see the note above.

## Facts that outlived what taught them

The full accounts are in `docs/decisions.md`; these are the parts that still
bind.

- **`VIDEO_MODELS` is served to the page**, so the composer shows only the
  controls the chosen model reads. A control that is present but ignored is
  worse than one that is absent — it is the UI making a promise the model will
  not keep.
- **An unmatched LoRA is reported rather than assumed to have worked.** Keys
  that do not map load nothing, the clip arrives, and it looks like a LoRA that
  was simply subtle. `_drain` counts ComfyUI's `NOT LOADED` lines and publishes
  once on the first progress line.
- **`MiniMaxH3SigmaShift` is opt-in and goes after the stack.** Its defaults are
  the model's own, so at rest it is a no-op; it stops being one under a
  distilled LoRA. `shift_video` and `shift_audio` are their own keys, and
  reading `/api/video`'s shared `shift` would put 8.0 on every H3 take against
  the model's 12.0.
- **`<Subject N>` is numbered by order of first mention, `<Picture N>` by upload
  position, and the two are allowed to disagree.** These are different films:

      two men fight inside a corridor that is constantly rotating
      a spinning corridor bounces two men around as they fight

  Same three subjects, same photographs. What differs is which noun the sentence
  opens on — and in the second the men are not even agents, they are things that
  get bounced. Nobody has to be *asked* what a scene is about, because the
  sentence already said. Numbering off the cast array threw that away and handed
  `<Subject 1>` to whoever was created first, which on the composer is whichever
  `@` was typed first anywhere: writing shot two before shot one inverted it
  silently. Anyone visible but never mentioned still gets a number, appended in
  cast order — a location earns a label from its establishing photograph without
  a line naming it.

  `subject_definitions` is emitted in **subject** order for the same reason the
  field exists: it defines a label before anything spends it, and listing
  `<Subject 2>` above `<Subject 1>` is that job done backwards. That was a live
  defect for the length of one edit and the assertion caught it.

  A picture number is a position in `references[]` and nothing else, so a subject
  and its picture routinely carry different numbers. `smoke_scene.py` used to
  check this by reading the picture numbers down the defs and expecting 1, 2, 3 —
  which held only while both counters happened to walk the cast in the same
  order, so the test could not tell them apart. It asserts the claim directly now.

- **`<Audio N>` is a sibling of the subject, not one of its sources.** It gets
  its own line in `subject_definitions` and `retention_analysis`, and the
  speaker ID is reused, never assigned. Somebody with only a voice attached is
  not a `<Subject N>` at all.
- **The image side is Hopper-only.** SageAttention is compiled for sm_90.
  Moving `IMAGE_GPUS` or `VIDEO_GPUS` means changing `TORCH_CUDA_ARCH_LIST` and
  forcing a rebuild.
- **Offered sampler and scheduler lists are checked against the node.**
  `KSampler` validates `sampler_name` against `comfy.samplers.KSAMPLER_NAMES`;
  `tools/smoke_graphs.py` is the check.
- **When the 2K module ships it extends the job/status/stop contract**, and
  upscaling this app's own output beats upscaling a dropped file — the method's
  advantage is the original context, and a sidecar still holds it.
- **A system prompt has a size budget: 500–2000 characters.** Past it, output
  goes lossy and begins parroting its own examples back as the answer.
- **A length is a token cap, never an instruction.** "Between 60 and 100 words"
  produced 95, 122 and 617. A model does not count.
- **Meta-commentary is cut by a regex, not asked away.** An instruction not to
  preamble is a request.
