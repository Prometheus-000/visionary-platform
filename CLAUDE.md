# Visionary

A single-user LoRA training and generation platform. `modal deploy app.py` gives
you one URL that is the whole application — UI, API and GPU jobs. Modal is the
vehicle it runs on today, not part of what it is.

## Philosophy

Three words, in priority order when they conflict. Each expands in
`.claude/rules/backend.md`, with the failures that produced it.

**Antifragile.** A failure should teach you something and leave the system
better able to survive the next one. Errors carry the three facts that
distinguish the causes; any error a user can hit twice should have explained
itself the first time. Destructive is asked for, not softened — the confirm
dialog states the blast radius. A wait costs what it shows, not what it takes:
forty minutes of visible work is comfortable, sixty seconds of a still button is
not, so the lever on a slow path is almost never *make it faster*, it is **say
what it is doing**, and say something different as often as the thing itself
changes. Stops are cooperative, and a cooperative stop has to be readable and
read.

**Scalable.** Not requests per second — dataset size, model size, and cost per
job. Never rent a GPU to do CPU work. Keep the polled thing small, and let a
poll wait for its own reply rather than fire on a clock. Separate storage by
commit cost. One container per loaded checkpoint.

**Future-proof.** Prefer the surface that will still be there, and that carries
the *next* model in for free. Depend on maintained upstreams over owned forks.
Separate images when pins conflict — and only then.

### The product is the experience; everything else is a receipt

**The product is what the user experiences. It is not the stack, and the stack
does not get a share of the credit.** Modal, ComfyUI, MiniMax-H3, a GPU class,
one-file `app.py` — each is a derived, disposable interpretation of a workflow,
replaceable the day something serves the experience better.

That is what makes a teardown cheap rather than tragic. What accumulates is an
understanding of a workflow almost nobody has designed before, and it outlives
every implementation: the thesis below survived the deletion of everything built
to serve it, and `web/CLAUDE.md`'s veto list — the densest artefact in the
project — names no GPU, no model and no framework anywhere in it. Volatility in
this field is the only instrument that reveals what the product should be, so a
shock costs a receipt and pays the record. **Treat radical rebuilds as the
environment, not the emergency.**

**Ask what a thing makes possible before asking what it would cost us.** That
ordering is not politeness; it decides what you read. FastVideo was nearly
dismissed here because its numbers were measured on Blackwell and this is a
Hopper image — a string in a build arg, functioning as a law — and under that
frame `apps/dreamverse/` sat in its README and went unopened. Dreamverse
(hao-ai-lab/FastVideo, live at dreamverse.fastvideo.org) is a websocket session
where video generates continuously and you append prompts mid-rollout. It is the
first thing that could satisfy this project's own veto on the Generate button —
a veto written before anything could. Researching *harder* under the wrong frame
produced a more confident wrong answer, not a correction.

Three rules fall out of that:

- **Price the number before invoking a rule.** Conventions break ties; they do
  not outrank an order-of-magnitude claim. Split a headline into its parts and
  ask which part is actually new to us.
- **`docs/decisions.md` is a dated record, not a veto list.** Every entry is
  what the measurement said on one date, against the alternatives that existed
  then. A new candidate inherits the *test*, never the verdict.
- **The limitations worth hacking are the ones that do not look like
  decisions.** "You need 80 GB to train a 33B model" reads as physics; the real
  assumption is that all parameters are trainable at once, and rotating
  trainable windows hacks the simultaneity instead. An undocumented decision
  goes invisible, and an invisible decision hardens into a law — which is the
  reason this file exists at all.

### The user's prose is the record; everything derived from it is a receipt

This survived the deletion of everything that was built to serve it, which is
the strongest thing that can be said for it.

> **What a model produces from somebody's sentence is a derived, disposable
> interpretation. The sentence is the record.**

*Derived*: regenerable at any time. *Disposable*: refusable whole, at any point,
with no loss beyond an interpretation. That is the relationship between
`prompt_typed` and the compiled `prompt` — intent is what is kept, everything
downstream of it is a receipt. Reuse, Copy and the metadata sheet prefer the
typed one.

It pays off the moment a model is replaced: intent recompiles for whatever comes
next, while a stored prompt is worth nothing to a checkpoint that wants a
different grammar.

**A prompt is a compilation target, not something the user writes.** Nobody
should have to learn a text encoder. The user says what they want; the app knows
what each model needs; the prompt is an implementation detail they should never
have to see. Fragments are the expected input, not the degraded case — out of
order, incomplete, self-correcting. A tool that demands one clean flowing
paragraph is demanding that a human produce the artefact of a machine before the
machine will listen.

**The unit of authorship must not be transferable.** A prompt fails that test
precisely *because* it copies whole — paste the string, get the look, bring no
taste. A LoRA passes because it is not a string, and so does a set of your own
photographs. So nothing here browses, searches or recommends somebody else's
instruments — weights, reference sets, character bibles, style packs, or
whatever the next format is called. Pulling a specific file you were sent is
fine; that is what the Drive route is for. A marketplace is not, whatever it is
a marketplace *of*.

The full argument, and the veto list it produces, is in `docs/roadmap.md`.

## Hard rules

These apply before any file is opened. Everything else is scoped — see Pointers
below.

- **Nothing is built in a silo.** How a feature sits in the whole experience
  is front of mind *during* the build, not a pass after it. The definition of
  done includes the neighbors: every surface that reads or writes the changed
  concept, the other console, the checks and stubs that encode the old
  meaning. A control that renders but no longer travels is the failure this
  names — the composer shipped once with the keyframe tiles and both flat
  trays looking live and silently dead beside it. And when a rule in this file
  chafes, surface the judgment it proxies and assert that: the rules exist to
  keep an agent out of exactly this vacuum, and deleting one reopens it. The
  30% console budget is the worked example — retired for the judgment it stood
  in for, "the canvas is always dominant," not deleted.
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
- **Pin to what you can reproduce.** A commit SHA, not a branch or a floating
  ref. When upstream force-pushes, your build should not change under you. A pin
  is a declaration, not a binding — bumping one is editing a number, and what
  binds you is the cost of the bump. A pin that moves is a seed; a pin that has
  not moved in two years is a fossil.
- **Storage layout is the contract, not the code.** Datasets are folders of
  images with `.txt` sidecars beside them — the same thing the trainer reads.
  Nothing here is required to get your data back out.
- **Do not build a second way to do the first thing.** New capability extends
  the existing job/status/stop contract rather than inventing a parallel one.
- **A reader that drops a field makes every run that has one unreadable.** A
  sidecar is read years after it is written, so a field nothing fills any more
  still gets written and still gets read.
- **Prose, not tags.** Captions are sentences, because the text encoders these
  models use parse grammar. See the `CAPTION_MODELS` comment.
- **Arithmetic in the validator, judgement in the harness.** The validator runs
  on every request and gates a render, so what belongs there are structural
  zeros — a probabilistic gate stacked on a probabilistic writer is two coin
  flips where the second one is invisible. Judgement goes where latency is free.
- **`app.py` is deliberately one file**, navigable by its banner comments. The
  alternative trades one long file for a build-order problem in the Modal image
  builds.

## Pointers

| Where | What it holds | Loads |
| --- | --- | --- |
| `.claude/rules/backend.md` | `app.py`, the ComfyUI nodes, the tools: the philosophy in detail, storage behaviour, conventions, the shot vocabulary | on reading `app.py`, `comfy_nodes/**`, `tools/**` |
| `web/CLAUDE.md` | the page, the console budget, the canvas, region cards, LoRA chips, the shot palette, the scene composer, and the veto list | on reading anything under `web/` |
| `docs/decisions.md` | what was removed or refused, and the measurement that settled it — the semantic layer, the rewrite, `forge/`, Wan 2.2, 2K, the ten-minute render | never; read it for the measurement and the trap it names, not as a standing no |
| `docs/roadmap.md` | the phases, and the veto list in full | never; read it when deciding whether a new surface belongs |

Layout and the storage schema are not written down here because they are
readable: `ls`, and `app.py` around the `WORKSPACE` constants. What is written
down is why they are shaped that way, in the rules files above.

**The front end is built into the image, not mounted from your disk.** That is
what keeps `modal deploy app.py` the entire install. Node is a build-time
dependency of the image; nothing at runtime needs it.

## Documentation discipline

- Do not create .md files unless I explicitly ask for one.
- Never write session artifacts: no PLAN.md, SUMMARY.md, PHASE_*.md,
  *_COMPLETE.md, MIGRATION_NOTES.md, IMPLEMENTATION_*.md. Report that in chat.
- Docs live in exactly three places: README.md, CLAUDE.md, and docs/.
  Nothing else at repo root.
- If a change affects something already documented, edit the existing file.
  Do not add a new one alongside it.
- Before writing any .md, state which existing doc you considered editing instead.