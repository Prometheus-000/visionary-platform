# Visionary — Semantic Layer

> **Superseded in one respect, 2026-08-18 — read this before implementing any
> threshold in here.** This revision added an arithmetic apparatus the original
> plan (`The Semantic Layer - Plan-original.md`) does not contain: a coverage
> floor, an invention ceiling, `_document_trust`, `_document_matches`, and a
> "restraint" criterion for choosing the model. Measured against the deployed
> interpreter, that apparatus is what makes the feature inert — it reached 0% of
> renders on finished prose — and it cannot be repaired by moving the numbers:
> on a fragment a good enrichment scores 94% invention and an evasive document
> 93%, the good one higher.
>
> What survives is everything *structural* the original also had: `insertionOnly`
> on the write, `_merge_document` refusing to overwrite a derived element, and
> `doc.for === prompt` for staleness. What replaces the thresholds is not another
> number — it is that the compiled document is written into the prompt box in
> front of the person, in grey, so an evasion is visible rather than silent.
>
> The measurements are in CLAUDE.md under "Prompt replacement, and what it
> cost to get there". The criteria that replace the eleven-row matrix are read rather than
> totalled: core subject extraction, emotional tone transfer, spatial logic,
> literal feature fidelity.


An implementation brief for Claude Code. Read `CLAUDE.md` first; the tradeoffs it describes still hold and nothing here overrides them.

---

## What we are building and why

Today the prompt *is* the state. The user types prose, it goes to a generator, an image comes back. Nothing accumulates, nothing is addressable, and an edit means rewriting the sentence and rerolling the whole picture.

We are inserting a semantic layer between the user and the generators:

```
fragment → interpreter → scene document → renderer adapter → Krea 2 / MiniMax / Wan
```

The **scene document** becomes the source of truth. The LLM never writes the final prompt — it produces structure, and adapters compile structure into whatever each renderer wants. This is what makes model swaps survivable and edits local instead of global.

The first visible result is **underlined intent**: the user's own words are marked one way, the system's inferences another, and editing an inference changes only what it governs.

---

## Non-negotiables

These are the constraints that make the feature feel like an instrument instead of a slot machine. Violating any of them fails the task even if the code works.

1. **The UI never blocks on interpretation.** The sentence renders as typed. Underlines arrive when they arrive. There is no spinner on the text and no disabled state while parsing.

2. **Seed is pinned once a shot exists.** Editing a span must change what the span governs and as little else as possible. Seed only rerolls on explicit user action. Without this the whole feature demos as noise.

3. **User edits are immutable to the interpreter.** When a user touches a span, its `source` flips to `user` permanently. Re-interpretation may only rewrite spans still marked `model`. This is the entire contradiction-avoidance strategy — no merge algorithm, no conversation replay.

4. **Provenance is binary.** `user` or `model`. No confidence scores, no probabilities, no gradients. A number invites a UI that renders it, and a self-reported confidence from an uncalibrated model is a lie with decimals in it.

5. **No sampler settings in the scene document.** The scene says what is in it and what happens. It never says `cfg`, `steps`, `sampler`, or `shift`. The moment those appear, adapters stop being swappable and the abstraction is dead.

6. **Pick nothing and nothing changes.** A user who never engages with the semantic layer must get byte-for-byte the behaviour they get today. Every prompt written before this still means what it meant. This is already the rule for the shot compiler; it extends here.

---

## Phase 0 — vLLM interpreter service

**Goal:** Qwen3-VL answering structured-output requests on its own queue, never behind ComfyUI's.

- vLLM serving Qwen3-VL Instruct with an OpenAI-compatible endpoint, in the existing generation container as a separate process — the same shape as the driven ComfyUI process, not a replacement for it.
- **Use a small variant (2B/4B) for interpretation.** Span parsing does not need 8B, and a smaller model takes proportionally less of the GPU when it has to timeshare with a sampler. Keep 8B on the captioning container where quality matters and latency does not.
- Enable concurrent inputs on the Modal function so an interpret request is not blocked at the container boundary by an in-flight render.
- **Warm ping on page load.** The user opens the URL, then spends ~15 seconds typing. Spin the container in that window and the first fragment lands warm. This preserves scale-to-zero — do not add `keep_warm`.
- Constrained decoding against a JSON schema. Do not parse free prose into structure.
- Weights are opt-in under the gear like every other family, and download on CPU containers.

**Verify:** vision path works, not just text. Send an image, get a description back. The frame-reading loop in Phase 5 depends on it and it is the half nobody else can copy — find out now, not in six weeks.

**Smoke test:** `tools/smoke_interpret.py`, CPU-only, checks the endpoint resolves and the schema parses, in the spirit of `smoke_caption.py`.

---

## Phase 1 — Underlined intent

**Goal:** fragment in, marked sentence out.

Schema — the smallest thing that works:

```json
{
  "text": "Two steel battleships trade broadsides across a heavy grey sea...",
  "spans": [
    {"start": 0, "end": 21, "source": "user", "role": "subject"},
    {"start": 45, "end": 62, "source": "model", "role": "environment"}
  ],
  "entities": [
    {"id": "subject_01", "label": "battleship", "count": 2}
  ]
}
```

Character offsets, not token indices — the renderer needs to mark up text, and offsets survive re-tokenisation.

- Interpreter fills gaps in what the user wrote and marks every span with its source.
- Frontend renders `user` spans solid-underlined, `model` spans dotted. Nothing else. No badges, no sidebar, no chat.
- **Extend `what the model reads`** rather than replacing it. It already shows the exact compiled string via `/api/compile`. Now it shows the scene document and the string it compiled to — same route as the real run, so a preview still cannot disagree with what happens.
- A Krea 2 adapter and a MiniMax adapter compile the scene document to their respective formats. The existing shot compiler is the MiniMax adapter — wrap it, do not rewrite it.

**Eval, not vibes:** `tools/eval_interpret.py` — twenty fixed fragments with expected structure, runnable on CPU. Interpretation quality is the entire product and tuning it by feel is how it rots. This is `smoke_prompt.py` for the semantic layer.

---

## Phase 2 — Edit the assumption

**Goal:** change an inferred span, regenerate, and see only that change.

- Tap a dotted span → edit in place. On commit, `source` becomes `user` and it is now immutable to the interpreter.
- The scene document updates; the adapter recompiles; **the seed does not move.**
- A dotted span also offers reroll — re-infer that span alone, leaving every other span untouched.
- Interpreter calls receive the current document and rewrite only `model` spans. Never replay the conversation.

This phase is the demo. If editing one span visibly rerolls the whole image, the seed rule is broken — fix that before anything else.

---

## Phase 2.5 — Minimal arsenal

**Goal:** one named thing that persists, early.

Not the full arsenal. `loras/` already holds persistent entities and there is already a trainer — the gap is naming and promotion, and it is small.

- **Keep** on a take: give it a name, get an entity with an id and a stored reference.
- Entities are addressable in the scene document by id.
- Browsed and selected, never searched, never suggested. The drawer stays closed until reached for.

Placed here on purpose: every later phase should be tested against recurring content rather than one-off fragments, and identity continuity is the thing most in need of real usage data. Phase 6 is too late to start learning.

---

## Phase 3 — Touch to select

**Goal:** touching an object in the frame selects it semantically.

- **v0: the region box is already a grounded object.** It has a bbox and an identity. Touching an existing box selects its entity. Ship this first — it is nearly free and it proves the interaction.
- **v1: auto-detection** via the interpreter's grounding, producing bbox + label + persistent id per object.

v1 is the risky half. A detector that mislabels is worse than no detector, because the user reaches for the ship and gets the ocean. Do not ship v1 until v0 has been used enough to know what selection should feel like.

---

## Phase 4 — Contextual controls

**Goal:** the palette stops being globally available and starts being summoned.

The 87 animated tiles are the best asset in the codebase — a tile that *shows* a dolly-out teaches what no word does. Keep all of them. Change only when they appear.

- Selection determines the vocabulary. Touch a subject, get subject operations. Touch the frame, get camera and light.
- Never more than one group at a time, anchored to what was touched.
- Deselect and it disappears entirely.

---

## Phase 5 — Shot as one object

**Goal:** image and video stop being two things.

- `shot.duration = 0` is a still and routes to Krea 2. Above zero routes to a video family. **The model is derived, not chosen** — the capability row already knows which controls a model reads; it can pick the model too.
- One duration control replaces the Image/Video chip.
- Then: extend, continue, change first frame, change last frame become operations on one object rather than separate modes.
- **The frame-reading loop:** generation lands → interpreter reads it → visual state updates → the user can say "the ship should be farther away" and the system knows what ship, and what farther means in this frame. This is downstream of Phase 0's vision verification.

---

## Order of work

Phases 0 → 1 → 2 are one sprint and one feature. Do not start Phase 3 until a span edit regenerates cleanly with a pinned seed.

Do not build the full arsenal, auto-grounding, or video continuity in this sprint. They are all downstream of interpretation being good, and interpretation being good is not yet demonstrated.

---

## Definition of done for the sprint

A fragment is typed. It renders immediately. Marks appear shortly after. The user taps a dotted phrase, changes it, and hits generate — and the new image differs in the way the edit implied and nowhere else. One named entity from the arsenal is in the scene and looks the same as it did last session.

If that is true, the layer is real. If the image changes everywhere, it is not.

---

## Where the repo overruled this brief

Written after the sprint that built it, so the next reader is not left to
reconcile two documents on their own. Four places, and the first governs the
other three.

**1 · The user's prose is the source of truth, not the scene document.** This
brief says *"the scene document becomes the source of truth"*. It is the reverse:
the document is a derived, disposable interpretation of the prose, and every
degrade in the shipped design depends on that. Dropping a document has to cost
nothing, and it only costs nothing if the record survives it. The full argument
is in CLAUDE.md under "The user's prose is the record".

**2 · `origin` and text runs, not `source` and character offsets.** A model
cannot count characters and can copy its own words, so it emits *runs* of text
each marked with who wrote it, and the server converts those to offsets in the
one pass that has both halves in hand (`_spans_to_text`). Converting on either
side alone means one of them is guessing, and the guess is silent — a mark
landing three characters left still looks like a mark, over the wrong words.

**3 · A text-only abliterated Qwen3-4B, not Qwen3-VL.** The parse reads a
sentence; it does not look at a picture. The vision half is weight and latency
for a capability this stage never uses, and the refusal argument in
`docs/vendor-parse-model.md` is the one that decided the checkpoint: with a
schema bound, a refusal arrives as an *evasive storyline that satisfies the
schema*, which nothing downstream can tell from a real one.

**4 · No concurrency change on the generators.** The brief says to raise
concurrency on the generation container and run the parse inside it. Both
generators carry `@modal.concurrent(max_inputs=1)` because one GPU runs one
sampling loop, and `_publish`'s lock is process-local *because* `max_containers=1`
means there is no second writer — the arrangement that lost the terminal status
15 runs out of 15 before the lock existed. The interpreter is its own `@app.cls`
on an L4 instead, warmed by a ping on page load.

**And one thing the brief's definition of done does not cover.** It ends with
*"one named entity from the arsenal is in the scene and looks the same as it did
last session"*. The arsenal is not in this sprint — see the Decisions table in
`The Semantic Layer - Plan.md`, which scopes it to Phases 0 → 1 → 2. The rest of
that sentence is what shipped: type a fragment, see the assumptions marked
inline, reroll or edit one, and only what the edit implied moves.
