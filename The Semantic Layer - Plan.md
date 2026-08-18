# The semantic layer

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


## Context

Today the prompt is the state. Prose goes to a generator, an image comes back, and an edit
means rewriting the sentence and rerolling the whole picture. `semantic-layer-brief.md`
inserts a scene document between the two, so intent accumulates, edits are local, and model
swaps are survivable.

**Most of the backend is already built.** `PARSE_RULES`, `PARSE_SCHEMA`, `_spans_to_text`,
`_validate_modules`, `_prominence`, and all three renderer adapters
(`_compile_image_prompt` / `_compile_wan_prompt` / `_compile_h3_prompt`, each already taking
`modules`) exist in `app.py`. `/api/parse` and `/api/compile` are wired. The brief's
"wrap the existing shot compiler, do not rewrite it" is **done** — nobody should rewrite an
adapter in this sprint.

Three things are wrong, and one of them is the reason the feature has never been seen:

1. **The underline is dead code.** [marks.ts:44](web/src/console/marks.ts:44) places an
   element's marks only where `prompt.indexOf(e.text) >= 0`. An element that contains
   invented words is, by definition, *not* a substring of what was typed — so it is skipped
   by the "skipped rather than guessed at" rule and nothing is ever drawn. This holds in
   production *and* in `preview_ui.py`'s stub. `/api/parse`'s `text` key and the unused
   `remapCaret` are the half-finished other side of it: the page was always meant to put the
   document's prose in the box.
2. **The document never reaches a run.** `/api/generate` and `/api/video` have no `modules`
   parameter, `imageBody` sends none, and `_shot_meta` records none. It can be previewed and
   never run, so there is nothing to edit that a run would honour.
3. **The interpreter is the wrong one.** `PARSE_MODEL = "claude-sonnet-5"` and the whole
   `_anthropic_key` path are traces of an approach that was abandoned.
   [docs/vendor-parse-model.md](docs/vendor-parse-model.md) already pins the real decision —
   an abliterated **text-only** Qwen3-4B — with a refusal argument the brief does not engage:
   with a schema bound, a refusal arrives as an *evasive storyline that satisfies the schema*
   and nothing downstream can tell it from a real one.

Outcome: type a fragment, see the system's **candidate** assumptions marked inline, edit one
inline, press Generate, and only what the edit implied changes.

*Candidate* is load-bearing, which is why it is in the sentence rather than left to the word
already in the shape diagram. What comes back is a proposal about what was meant, and it is
one the validator is free to refuse whole. The model gets a turn at describing intent; it
never acquires authority over it.

## The shape

```
                         HUMAN
                           │
                           ▼
                  ┌─────────────────┐
                  │    Textarea     │   one surface, editable everywhere
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Qwen Interpreter│   untrusted, probabilistic,
                  │      (4B)       │   allowed to fail
                  └────────┬────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Semantic document    │   elements · provenance
              │      (candidate)       │   relationships · invented spans
              └────────────┬───────────┘
                           │
                   deterministic
                     validation        ← reject whole, never repair
                           │
                           ▼
              ┌────────────────────────┐
              │     Scene compiler     │   _compile_*_prompt, unchanged
              └────────────┬───────────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
          Krea 2          Wan             H3
```

A candidate that fails validation is dropped and the textarea's prose goes straight to the
compiler — the plain path, byte-for-byte today's app.

## Architecture invariants

The document architecture is deterministic. **The interpreter is not, and it is the largest
technical risk in this sprint — not the UI.** A 4B model asked to structure *"maybe a woman
in a red dress walking through some kind of futuristic airport, cinematic, probably evening"*
has no way to know which assumptions are legitimate, and the schema does not help: it
guarantees valid JSON, never a useful interpretation. The vendor doc's own finding is the
sharp edge — with a schema bound, **a refusal arrives as an evasive storyline that satisfies
the schema**, indistinguishable downstream from a real one.

The separation, in one line: **the user owns intent, the model proposes structure, the
validator enforces boundaries, the compiler executes.** Which makes the property all three
invariants serve, and the one to keep if the rest are ever rewritten:

> **The semantic document is not the source of truth for what the user intended. The user's
> prose is. The document is a derived, disposable interpretation of that prose.**

The brief says the opposite — *"the scene document becomes the source of truth"* — and is
overruled here, because every degrade in this plan depends on the reverse. Dropping a
document has to cost nothing, and it only costs nothing if the record survives it. Derived:
regenerable from the prose at any time. Disposable: refusable whole, at any point, with no
loss beyond an interpretation. It is the same relationship CLAUDE.md already draws between
`prompt_typed` and the compiled `prompt` — intent is what is kept, everything downstream of
it is a receipt — extended one layer up.

So these three are invariants, not goals. Each names where it is enforced, because an
invariant nobody can point at is a paragraph.

**1 · A semantic document is valid only for the exact prose state from which it was derived.**

Enforced by making the invalid state unrepresentable rather than by remembering a check at
each call site. `doc` is stored but **not readable**; the store exports only
`docFor(prose)`, which returns the elements or `null`. Nothing — `imageBody`, `videoBody`,
`Peek`, `reuse` — reads `s.doc.elements`. The prose is the key, always. Re-checked
server-side in `/api/generate` and `/api/video` by `_document_matches(typed, modules)`,
because the route receives the two independently and a client is one more untrusted caller.

**2 · Interpretation is untrusted probabilistic input.**

The pipeline is `Qwen → candidate document → deterministic validation → compiler`, never
`Qwen → semantic truth`. `_validate_modules` is the right instinct and does not go far
enough: it validates *shape* and cannot see the user's prose, so an evasive storyline passes
it intact. It gains the prose as an argument and four checks that are cheap, deterministic,
and each aimed at a named failure (see Stage 0).

**The model is allowed to fail, and failure degrades to plain prompt behaviour — silently
and totally.** No error, no toast, no banner, no partial document. A rejected interpretation
produces byte-for-byte today's app, which is the same guarantee non-negotiable #6 makes for
a user who never engages. A malformed semantic scene is never an acceptable output; the
absence of one always is.

Two failures, kept distinct: a **malformed** document (unknown role, bad depth) is refused by
name, the posture `_validate_shot` takes toward an unknown pill. An **untrustworthy or
stale** document is dropped and the run proceeds plain.

**3 · The semantic layer adds meaning without taking ownership of the user's words.**

The goal is **the minimum useful interpretation, not the best one.** Given *"a woman walks
into a room"*, the model must not decide young, elegant, modern office, afternoon, cinematic
— those make a richer prompt and every one of them is something the user now has to police.
The rule, in three lines, and it becomes the head of `PARSE_RULES`:

> **Explicit: preserve. Necessary structural inference: add. Optional creative detail: do not
> invent.**

"Necessary" is bounded by what the compiler structurally requires — an anchor for a dangling
clause, an ordering — and by nothing else. This has a mechanism too, not just an
instruction: an invention budget in the validator, and restraint as a first-class score in
`smoke_parse.py`.

**And a second sentence, which reads like the same rule and is not.** The one above bounds
how *much* is added. This one bounds what may be touched:

> **The document may enrich the user's words, but may not contradict, replace, or reinterpret
> an explicit fact.**

Enrichment is legal *because it is marked* — grey, underlined, one touch from a reroll, which
is the only terms invention was ever allowed on. Replacement is illegal however it is marked,
because a fact the user stated is not a slot the model gets to fill. The two look nothing
alike on screen and identical to any arithmetic over character ranges, which is the next
section's problem and the reason it is written down here first.

**The consequence for model choice.** The benchmark is not which model understands the
fragment best. It is **which model makes the fewest assumptions while respecting the boundary
between what the user said and what it inferred.** A 14B producing elaborate, beautiful
interpretations is *worse* here than a 4B producing sparse ones. Fidelity and restraint are
the axes; intelligence is not one.

## Decisions taken

| | Decision |
|---|---|
| **Invented text** | **Inline, in the sentence.** Per `visionary-reduced.png`: one flowing sentence, dark for the user's words, grey for the model's, underline marking what is addressable. Not a tail, not a separate surface. |
| **Interpreter** | **Local vLLM**, the pinned abliterated Qwen3-4B. The Anthropic path is removed, not deprecated. No tool-calling, no chatbot — `guided_json` constrained decoding. |
| **Storyline code** | **Kept, and none of its UI ships.** `web/src/storyline/` and `web/storyline.html` stay on disk for possible shape pre-visualisation. Vite builds `index.html` only (no `rollupOptions.input`), so they are already outside the bundle and cost the deploy nothing. |
| **Scope** | Brief Phases 0 → 1 → 2. Not 2.5, not touch-to-select, not auto-grounding. |

## The UI budget

Net **zero new controls**. Choosing local weights is what pays for it: a hosted interpreter
needed an API-key field in Settings, and weights need nothing — the gear renders a catalogue
family for free.

- **Added:** one treatment on text that already renders (`.mk-mirror` gains a grey class),
  and one reroll affordance that exists only while the caret is inside an invented run. The
  prompt stays a single `<textarea>` and stays editable everywhere — no second text surface.
- **Removed:** the Anthropic key plumbing (`_anthropic_key`, `anthropic==0.42.0`,
  `anthropic_key` on `/api/token` and `/api/state`, `PARSE_MODEL`), which was a Settings
  field about to be built and now never will be.
- **Not added:** no panel, no document view, no sidebar, no confirm. `Peek` keeps showing
  one `<pre>` of the compiled string — rendering the document *as a document* is the
  node-graph-with-better-typography failure CLAUDE.md names as the month-four risk.

## Stage 0 — the interpreter: harden, measure, then wire

Ordered that way deliberately. `stress_parse.py`'s own docstring already sets the standard —
*"the model choice should rest on a number rather than an argument"* — and invariant 2 means
the validator has to exist before there is anything worth measuring against.

### 0a · Push `_validate_modules` further

It gains the user's prose as an argument (it has never had it, which is why an evasive
storyline passes it intact) and four deterministic checks. Each maps to a real failure, and
**each rejects the whole document rather than repairing it** — a partially-trusted
interpretation is the one output invariant 2 forbids.

| Check | Catches |
|---|---|
| **Derived text is real.** Every `derived` run must appear in the user's prose, walked with `promptMarks`' forward-only cursor. | The evasive refusal, directly. A storyline the model wrote about something else has derived text the user never typed. This is the single highest-value check in the sprint. |
| **Invention budget.** Invented characters as a fraction of the whole, over a ceiling → reject. | *"Produce the minimum useful interpretation"* as a mechanism rather than a hope. An interpretation that is mostly invention is not an interpretation. |
| **Coverage.** The fraction of the typed prose covered by derived runs, under a floor → reject. | The model quietly dropping half of what the user said — invariant 3's other direction, and invisible without this. |
| **No invented subject.** An element with `role: "subject"` and `origin: "invented"` → reject. | `PARSE_RULES` already forbids it; this makes it checkable instead of hoped for. |

**Four checks, three questions — and a fourth question nothing here answers.** The rows above
read as one list and are not:

| Question | Asked by | What passing means |
|---|---|---|
| **Preservation** — did the user's prose survive? | derived text is real | every derived run is genuinely theirs |
| **Coverage** — was any of it silently dropped? | coverage floor | what they typed is accounted for |
| **Invention** — how much was added? | invention budget, no invented subject | a bounded amount, and never a subject |
| **Contradiction** — was an explicit fact changed? | *nothing, this sprint* | — |

**Coverage is not a proxy for the fourth, and reading it as one is the trap this table exists
to close.** Take 0c's own example. *"a woman in a red dress walks toward two guards"* comes
back with `two guards` marked derived and `heavily armed male bodyguards` beside it marked
invented. Preservation passes — those words are the user's. Coverage passes, and passes
*well*: the typed prose is covered end to end, better than a sparser interpretation would
manage. The budget likely passes too, since one added clause is not a breach. And the picture
now has armed male bodyguards in it, because the model decided the guards were armed and male.

That outcome is acceptable **only** because the addition is grey, addressable and cheap to
reroll — invariant 3's trade, working. It stops being acceptable the moment the same
substitution arrives as a *replacement* rather than an addition, and the distinction between
the two is semantic rather than textual: no arithmetic over character ranges can see it, and
a coverage number reported without this paragraph beside it says the document is faithful
when what it measured was that nothing went missing.

**So contradiction is measured rather than enforced, and named rather than left implicit.** No
detector ships in this sprint: it is a semantic judgement, and a bad one rejecting good
documents is worse than the failure it prevents. What ships instead invents nothing — the
sentence in `PARSE_RULES` (0b), a scored row in the 0c matrix, and corpus fragments in
`smoke_parse.py` that state a fact worth changing so a model that changes it is visible in a
number. If a deterministic version ever earns its place it attaches where the other four do,
as one more check inside `_validate_modules(…, prose=…)` — the signature this stage already
adds is the one it would need, so nothing here has to be reshaped to admit it later.

**Both thresholds come from the corpus, not from first principles.** This repo's own standard
— *"a threshold argued from first principles is a threshold nobody has looked at"*, the reason
`tune_dupes.py` exists. Sweep them over `smoke_parse.py`'s corpus and record the margin from
the nearest rejected document, the way the duplicate bounds were set. Do not ship a guessed
number.

### 0b · `PARSE_RULES` gains the restraint clause, at the head

It currently reads as *make a good structure*, and the invention-marking section
("invented — you supplied it... which is the only reason you are allowed to invent at all")
sits late enough to read as permission. **Both** of invariant 3's rules go **first**, where
they govern everything under them — the three-line restraint rule, and the sentence about
enrichment against replacement, which is the whole of what this sprint does about
contradiction and belongs where the model reads it rather than only where we argue it. The
marking section is then rewritten to point back at them rather than licensing invention.
`docs/krea2-prompt-template.md`'s findings stay — they are about *how to phrase what is
there*, not about adding things.

### 0c · Measure, then pick — and do not evaluate these like chatbots

`stress_parse.py` already spins a throwaway Sandbox per candidate. It becomes the comparison
harness, and the output is this matrix rather than a score:

```
                        Qwen3-4B   Qwen3-8B   Qwen3-14B
  ─────────────────────────────────────────────────────
  user text preserved      ?          ?           ?      ← highest weight
  semantic contradiction   ?          ?           ?      ← disqualifying
  invented spans           ?          ?           ?
  schema validity          ?          ?           ?
  relationship accuracy    ?          ?           ?
  entity accuracy          ?          ?           ?
  refusal behaviour        ?          ?           ?
  idempotency              ?          ?           ?
  reroll safety            ?          ?           ?
  latency                  ?          ?           ?
  VRAM                     ?          ?           ?
```

**User-text preservation carries the highest weight, and it is not the same axis as
invention.** The failure to design against:

> *"a woman in a red dress walks toward two guards"* → the model returns a beautiful
> structure in which **"two guards" has become "two heavily armed male bodyguards."**

That is worse for this product than a weaker interpretation, and a model that does it is
disqualified however well it scores elsewhere. Two validator checks catch it between them,
depending on how the model marks the substitution — worth knowing which does what:

- Marked **derived** → the **derived-text check** fires: those words are not in the prose.
- Marked **invented**, *in place of* the user's phrase → the derived-text check passes, and
  the **coverage check** fires: the user's own "two guards" is covered by nothing.
- Marked **invented**, *beside* the user's phrase, which survives and is covered →
  **nothing fires, and nothing should.** The addition is grey and one touch from a reroll,
  which is the only terms invention was ever allowed on.

The first two are why both checks exist; neither alone is sufficient. The third is why
neither of them is a contradiction detector, and it is not free merely because it is legal:
it is the attribute nobody asked for, arriving inside the rules. A model that does it
constantly loses on *invented spans*; a model that does it to a fact the fragment had already
settled loses on *semantic contradiction*. That is what the matrix scores and the validator
cannot reject.

The rest of the matrix, and what each row is measured by: *semantic contradiction* is scored
by hand against corpus fragments that state a fact worth changing — a count, a colour, a
relation — asking only whether the document still says that fact, which is a judgement a
person makes once per fragment and a validator cannot make at all; *invented spans* is the
invention rate; *relationship accuracy* scores `ties` against the corpus's expected relations;
*entity accuracy* scores whether subjects are the ones the fragment names, and no others;
*refusal behaviour* is the vendor doc's failure — a fragment with real material in it, scored
on whether the storyline is about that material or evasively about something else;
*idempotency* is `parse(parse(x).text)` returning the same document, which is what makes a
document survive a round trip through the box; *reroll safety* is the Stage 5 transaction,
scored on how often a reroll proposes touching derived content at all.

**Decision rule, from invariant 3:** at comparable preservation and fidelity, **take the most
restrained model, not the largest.** A 14B that interprets elaborately loses to a 4B that
interprets sparsely. If the pinned abliterated 4B wins, the vendor doc needs no change; if it
does not, that is worth knowing before a line of this is wired.

### 0d · Then serve it

Nothing above depends on where this runs, but it is what makes the feature visible at all.

**`tools/stress_parse.py` and `tools/smoke_parse.py::openai_compatible` are the blueprint,
already debugged.** Lift, do not re-derive:

- The image recipe and every comment justifying it — `nvidia/cuda:12.8.1-cudnn-devel` (vLLM's
  inductor shells out to `nvcc`; a slim base dies minutes after a successful model load),
  vLLM **unpinned** (0.11.0 fails on `Qwen2Tokenizer.all_special_tokens_extended`).
- `vllm serve --max-model-len 8192 --gpu-memory-utilization 0.90`, torch.compile left on
  (~3 min cold on an L4; `--enforce-eager` fails engine init).
- Health-probed with `urllib`, not `curl` — the CUDA base has no `curl` and the failure is
  indistinguishable from a slow start.
- The three structured-output dialects and their trap: vLLM 0.27 accepts `guided_json`
  without binding it and returns empty content, so a dialect must be recorded only once a
  response actually parses.

**Where it runs: its own `@app.cls`, L4, not inside a generator.** The brief says to raise
concurrency on the generation container. Do not.
[app.py:5335](app.py:5335) and [app.py:7461](app.py:7461) carry
`@modal.concurrent(max_inputs=1)  # one GPU, one sampling loop`, and `_publish`'s lock is
process-local *because* `max_containers=1` means there is no second writer — the finding
CLAUDE.md records as losing the terminal status 15 runs out of 15. A 4B at bf16 is ~8 GB;
an L4 is the measured target and is a fraction of an idle H100.

- Model class shaped like `_Comfy`: `@modal.enter()` starts the server,
  `_wait_ready()` polls `/health`, a drain thread mirrors stdout, and a `_revive()`
  equivalent restarts a dead process at the top of the call — same reasoning, `@modal.enter`
  runs once per container.
- Weights follow the **`CAPTION_MODELS` precedent**, not `MODEL_CATALOGUE`: a pinned repo id
  + revision from the vendor doc, served off the existing `hf_cache` volume.
  `MODEL_CATALOGUE` is built for single files with a `dest: Path`; vLLM wants a repo
  directory. Zero new gear UI. *Flag at review:* this trades against "nothing downloads on
  its own", exactly as the captioner already does; the alternative is a catalogue entry with
  friction.
- **Warm ping on page load**, per the brief — the user spends ~15s typing, which is the
  window. Preserves scale-to-zero; do not add `keep_warm`.

**Remove the Anthropic path in the same commit** so there is one interpreter, not two:
`PARSE_MODEL`, `_anthropic_key()` ([app.py:1229](app.py:1229)), the `anthropic_key` branches
in `/api/token` ([app.py:7930](app.py:7930)) and `/api/state` ([app.py:7825](app.py:7825)),
`anthropic==0.42.0` from `web_image` ([app.py:241](app.py:241)), and `smoke_parse.py`'s
`--backend hosted`. `_parse_storyline` ([app.py:6546](app.py:6546)) keeps its signature,
its docstring's "a model is one more untrusted caller" rule, and its `_validate_modules`
pass; only the call inside it changes.

## Stage 1 — the document reaches a run

Before the marks, so the marks have somewhere to live.

- **`store.ts`**: `doc: { for: string; elements: ParseElement[]; text: string } | null`,
  **not exported for direct reading**. The store exports `docFor(prose)` and nothing else —
  invariant 1, made unrepresentable rather than remembered. `imageBody`, `videoBody`, `Peek`
  and `reuse` all go through it; a grep for `.doc.elements` outside the store should return
  nothing, and is worth putting in `check_document.py`.
- A document that does not describe what is in the box is not sent, and the run degrades to
  today's byte-identical plain path — invariant 2's degrade, and what keeps non-negotiable #6
  true.
- **`useMarks` → `useDocument`** ([web/src/console/useMarks.ts](web/src/console/useMarks.ts)),
  writing `s.setDoc(...)` inside the existing `asked !== seen.current` staleness guard —
  that guard is already correct, it just gates one more write. `setDoc(null)` on failure or
  an empty prompt.
- **`imageBody`** ([useGenerate.ts:62](web/src/canvas/useGenerate.ts:62)) and `videoBody`
  ([useVideo.ts](web/src/video/useVideo.ts)) each gain one line.
- **`/api/generate`** ([app.py:8856](app.py:8856)) and **`/api/video`**
  ([app.py:8928](app.py:8928)): `_validate_modules(payload["modules"], prose=prompt)` inside
  the **existing** `try` beside `_validate_shot`, so a malformed document is a named form
  error on CPU before a cold H100 — and an untrusted or stale one is dropped by
  `_document_matches`, the run proceeding plain. Two failures, two behaviours, per invariant
  2. Pass `modules` to the compiler already being called; add `"modules": modules` to the
  spawn params. Widen `/api/generate`'s `if not prompt and not regions` guard to admit a
  document. Record on the job when a document was dropped — a degrade nobody can see in the
  logs is a degrade that gets diagnosed as a bad model.
- **`_shot_meta`** ([app.py:7184](app.py:7184)): emit `modules` when a document ran. The
  existing `typed == params["prompt"]` guard stays — a one-module document whose compile
  equals the typed text writes nothing, which is right: the run is reproducible from the
  typed prompt alone. Add a comment saying so. `modules` on the sidecar is a receipt beside
  the compiled string, never the record; `prompt_typed` stays the record, per the property at
  the head of the invariants.
- **`reuse.ts`** ([web/src/gallery/reuse.ts](web/src/gallery/reuse.ts)): restore `doc` with
  `for` set to the *same string* `setPrompt` writes. This is the plan's one real footgun, and
  it is why invariant 1 is a `docFor(prose)` accessor rather than a check to remember here.
  Cards and `MetaSheet` show nothing new.

**Checkpoint:** `probe_compile.py` diffs to zero.

## Stage 2 — marks in the sentence

The box holds the document's prose. Derived runs render normally, invented runs grey and
underlined, per the sketch.

- `useDocument` writes `r.text` to `s.prompt` when a parse lands, and `remapCaret`
  ([marks.ts:94](web/src/console/marks.ts:94)) — currently unused, written for exactly this —
  carries the caret across it.
- **`insertionOnly(typed, text, marks)`, new in `marks.ts`: the write is refused unless every
  one of the user's own runs survives, in order.** Compute the complement of the invented
  marks and walk it against `typed` with the same forward-only cursor `promptMarks` uses. If
  any derived run is missing, keep the typed text and drop the document. It is the client
  mirror of the validator's derived-text check and the last line of invariant 3: **the model
  may insert, never revise.** A write that would take ownership of the user's words does not
  happen, and the failure is silent — the box simply keeps what was typed.
- **No write while an IME is composing — and `PAUSE_MS` is why this is the common case
  rather than a rare one.** The textarea is controlled
  ([Field.tsx:119](web/src/console/Field.tsx:119)), so the write is a React value change, and
  a value change mid-composition destroys the composition buffer and the candidate window
  with it. The staleness guard does not cover this and cannot: `asked !== seen.current` asks
  whether the prose *moved*, and an open IME candidate window is precisely a prose state that
  has been **stable** for longer than the 500ms debounce while the user is mid-word. The
  parse timer is tuned to almost exactly the dwell time of the thing it must not interrupt.

  So the field tracks composition (`onCompositionStart` / `onCompositionEnd`) and **the write
  — only the write — is suppressed while it is true.** The parse still runs and the marks
  still land: the mirror is `aria-hidden` and painting it touches nothing the IME owns. A
  document arriving mid-composition is **dropped, not queued** — composition ends by changing
  the prose, so it is stale under invariant 1 at the moment it would have become usable, and
  the next pause re-parses. Nothing is invented here: `isComposing` is already this
  codebase's idiom in four places, one of them the Enter guard in this same handler
  ([Field.tsx:98](web/src/console/Field.tsx:98)). What is new is only that a *timer* needs the
  state where an event could carry the flag, which is what the two handlers are for.
- `.mk-i` splits into invented (grey + underline) and derived-run (underline only). Underline
  means *addressable*; colour means *whose*. Provenance stays binary — the underline is reach,
  not a third state.
- **The undo objection is already spent.** `moveClause`, `nudgeLora` and the `+ LoRA` caret
  sink ([Field.tsx:74,90,127](web/src/console/Field.tsx:74)) all write the textarea's value
  through React today, so native undo is already superseded on three paths. Add a one-slot
  `docUndo` in the store restoring the pre-parse `{prompt, doc}` pair, bound to ⌘Z in the
  field. One slot, because the gesture is one write at a time. **Captured at the write site,
  in the same `set()` that performs the write** — never at the moment the parse was asked
  for. The staleness guard makes those two strings equal today, which is exactly why the
  capture should not depend on it: undo staying correct is then a property of one statement
  rather than of a guard three lines away that a later change could loosen.
- `preview_ui.py`'s `/api/parse` stub ([tools/preview_ui.py:881](tools/preview_ui.py:881))
  gains a second shape: an element the prose does **not** contain. ~6 lines, and it is what
  makes the whole feature developable with no GPU and no Modal account — the file's stated
  reason to exist.

**Checkpoint:** an invention is visible inline, the caret never jumps, a parse landing during
an IME composition leaves the box alone, and `probe_console.py` shows the 30% console budget
holding.

## Stage 3 — the seed pin

Non-negotiable #2, with no new control.

- In `finish()` ([useGenerate.ts:117](web/src/canvas/useGenerate.ts:117),
  [useVideo.ts:79](web/src/video/useVideo.ts:79)) — both already read the seed off the record
  for `meta` — write it back to `s.img.seed` / `s.vid.seed` **only when a document exists and
  the field is blank**.
- **The `doc &&` condition resolves #2 against #6.** Read "pinned once a shot exists" as
  "once a *scene document* exists". A seed that stops rolling for someone who never engaged
  reads as *"Generate is broken — it keeps making the same picture."*
- **Switching image ↔ video carries no pin, and that is already true.** `s.img.seed` and
  `s.vid.seed` are separate fields ([store.ts:138](web/src/store.ts:138),
  [store.ts:153](web/src/store.ts:153)), each written by its own `finish()`. There is nothing
  to clear across the switch and nothing to build for it.
- **A pin is not cleared when the size, the model or any other parameter changes. Considered
  and rejected, and the reasons are worth keeping because this looks like a bug for as long
  as it is undocumented.** The case for clearing: a seed held across a resize buys no
  continuity. True — a different latent shape makes the same integer a different picture, so
  the pin stops *paying* rather than degrading anything, and nothing is broken by leaving it.
  The case against is three deep. Same seed, wider frame is a comparison people actually run,
  and clearing deletes it. It would be the first control here that silently empties another
  field: `setImg`/`setVid` are shallow patches ([store.ts:363](web/src/store.ts:363)), the one
  place that force-switches the video model on a reference drop leaves every dependent value
  alone ([App.tsx:302](web/src/App.tsx:302)), and the only writes of `seed: ''` in the
  codebase are the two Reset buttons — a person asking. And a value that empties itself when
  you touch a control you did not think was a decision is the specific way a surface stops
  being believed, which is CLAUDE.md's finding about the delete count moving under an
  unrelated dropdown, transferred whole. The number is visible in its own field and one
  gesture from gone; **that visibility is what makes clearing it for you unnecessary rather
  than merely risky**, and it is the same argument as the `SEED_HINT` sentence below.
  If it is ever felt as wrong in use the flip is one line in the size and model handlers —
  but it is then a stated behaviour and `SEED_HINT` has to say it too.
- **The reroll gesture already exists.** `SamplingButton.tsx:106`'s seed field shows
  `placeholder="random"` and now shows a number; `SEED_HINT` already reads *"Blank draws a
  new one"*; Reset already writes `seed: ''`. Add one sentence to `SEED_HINT` naming the new
  state — the carve-out CLAUDE.md already grants, since a field that silently stopped being
  random is a value that cannot show itself. **No lock glyph, no dice, no chip.**

**Checkpoint:** two Generates with an unchanged document give the same picture; with no
document, they do not; and changing the aspect with a document present leaves the seed
exactly where it was.

## Stage 4 — edit the assumption

**The whole prompt is editable, everywhere, always, and it stays one `<textarea>`.** Editing
a grey run is just typing into it: `remap` ([marks.ts:105](web/src/console/marks.ts:105))
drops any mark the edit landed on, so the words turn dark and become yours with no gesture,
no commit and no code. That is non-negotiable #3 already implemented, and it is the whole of
this stage's editing story.

**Explicitly rejected: an inline editable rooted in the run.** It is a second text surface
competing with the one underneath it, it drags in the focus-stealing trap `check_regions.py`
found on its first run (unmounting re-ran mount effects on release and stole focus into the
prompt, so ⌫ edited text instead of the thing selected), and it buys nothing typing does not
already do. This is the cheap-fix shape the vetoes exist to kill.

**A span is an object for exactly one thing: reroll.** That is the only response an invented
run has that plain text cannot give, and it is what makes invention acceptable at all —
ten-sentences #9: *marked another, and is cheap to reroll*.

- One affordance, rooted at the run's own end, revealed when the caret is inside a grey run
  and gone when it leaves. No card, no popover, no chip, nothing at rest.
- **Open sub-question, cheap to flip once felt:** caret-reveal versus an explicit tap.
  Caret-reveal is fewer gestures and appears while you type past a run; a tap is quieter and
  costs a gesture. Written as caret-reveal.
- On commit of an edit, update the document locally — that element's `origin` to `derived`,
  its `invented` runs cleared, `doc.for` updated in the same `set()`. **No re-parse.**
- Deleting a run's text deletes the element, with no confirmation. `docUndo` is the reversal.

## Stage 5 — reroll one span

Extend `/api/parse`; do not add a route.

- Two optional fields: `document` (a full element list) and `only` (an element id). When
  present, `_parse_storyline` appends a `PARSE_REROLL` paragraph to `PARSE_RULES` — beside it,
  on the server, for the reason `CAPTION_PRESETS` gives.

**`_merge_document(old, new, only)` is a transaction, and that is the whole design.**
User-authored content is immutable to the interpreter, enforced all-or-nothing:

```
old document → reroll(element X) → model proposes replacement
             → merge → validate → commit, or reject the entire replacement
```

- **Anything suspicious rejects the whole thing.** Any element other than `only` differing by
  so much as a character; any `derived` element touched; any `derived` run inside `only` that
  is not in the user's prose; the replacement's own invention over the budget; a failed
  re-validation of the merged result. One violation, and the *entire* replacement is discarded
  and the old document stands unchanged.
- **The budget is applied to the replacement itself, not only to the merged document, and
  that is a different rejection from a merged document that fails re-validation.** A
  whole-document check is a fraction, and a fraction dilutes: one lavishly invented element in
  a document of eight passes a ceiling that same element could never pass alone. So a reroll
  would become the way to buy invention the first parse would have refused — press the button
  until the answer is generous. Restraint that can be routed around by pressing a button twice
  is not a policy. Same ceiling, scoped to `only`: no second threshold to sweep, no second
  concept.
- **An identical replacement is a commit, not a rejection.** A reroll that comes back the
  same — same text, same marks — passes every check above, because all of them are about what
  *moved*: another element differing, a `derived` element touched, a `derived` run not in the
  prose, the replacement's budget, re-validation. Nothing moved, so nothing fires, and the
  transaction commits a document byte-identical to the one it replaced. Worth stating
  outright, because "anything suspicious rejects the whole thing" reads as an invitation to
  add an `unchanged?` guard, and a no-op reported as a failure is a bug that would look
  exactly like a broken interpreter.
- **What the press reports is the press, and it reports it on the affordance rather than on
  the run.** A reroll lands three ways — new text, the same text, a rejection — and two of
  them change nothing on screen, which really can read as frozen. The fix is not a pulse
  under the words. Three reasons, and the middle one is the disqualifying one: it is motion
  on the render surface to announce a null result; a flicker that fires for *identical* while
  a rejection stays silent builds a channel telling the user which way the validator went,
  which is the one thing invariant 2's silent degrade exists not to say; and a run that
  twitches is the grey underline making a claim about itself rather than about authorship.
  So the in-flight state sits on the affordance that was pressed — already mounted for as
  long as the caret is in the run — and settles identically for all three outcomes. That
  answers *did that register* without answering *what did it decide*, which is the right
  split. Cheap to flip if it is felt as insufficient in use; not cheap to un-leak.
- **No partial salvage, ever.** Keeping the good half of an ambiguous reroll produces a
  document nobody authored and nobody can reason about, and it is the state that makes a bug
  here undebuggable. A rejected reroll is one clean outcome: nothing moved.
- On screen a rejection is a no-op — the run stays grey and unchanged. Same degrade posture as
  invariant 2, and the same silence.
- The route's contract holds: failure returns `{ok: False, error}`, never a 500.

**Checkpoint — the sprint's definition of done:** reroll one span, seed held, and only what
the edit implied moves in the render.

## Must not move

- **`tools/ui-checks/probe_compile.py` — 402 pinned outputs.** Every existing case sends no
  `modules`; every one must come back byte-identical. Re-run and diff to zero after the route
  changes. This is "pick nothing and nothing changes", mechanised.
- **`tools/smoke_modules.py` — 11 golden prompts rebuilt byte-identical.** If one moves,
  `_shot_body` or `_module_clause` was changed and should not have been.
- **`tools/smoke_prompt.py`** — the pill compiler is not on this path.

## Tests

**Changed:** `probe_compile.py` gains an appended section (never a modified matrix) covering
multi-element and nested documents through all three compilers, including the one assertion
worth stating outright — *a one-element document compiles byte-identically to the same text
typed plain*, which turns `_shot_body`'s docstring claim into a check and is the single fact
non-negotiable #6 rests on. `smoke_modules.py` gains idempotency over a sidecar-restored
document, `_shot_meta`'s emit/omit pair, and **a case per validator check**, each built from
a hand-written adversarial document: an evasive storyline whose derived text is not in the
prose, one over the invention budget, one under the coverage floor, one with an invented
subject — every one rejected whole, and the run proceeding plain rather than partial.
`smoke_parse.py` drops `--backend hosted` and gains the matrix rows it can score on CPU against
a served endpoint — preservation, invention rate, relationship and entity accuracy, refusal
behaviour, idempotency, reroll safety, and semantic contradiction, which is the one row scored
against a hand-written expectation per fragment rather than computed — plus the
`_merge_document` transaction cases: a reroll touching a second element, a reroll touching
derived content, a reroll whose merged result fails re-validation, and — separately from that
last one — **a reroll whose replacement is valid on its own terms and over the invention budget
for the element it replaces**, which is the case that proves reroll is not an escape hatch
around restraint. Each asserts **the old document survives byte-identical**.
`stress_parse.py` becomes the 4B/8B/14B comparison harness printing the matrix with latency
and VRAM. `preview_ui.py` gains the second stub shape. `probe_console.py` gains a
document-present row.

**New:** `tools/smoke_interpret.py` (CPU-only, in the spirit of `smoke_caption.py`) — the
endpoint resolves, `PARSE_SCHEMA` is valid `$defs`/`$ref`, a dialect binds.
`tools/ui-checks/check_document.py` — typing never moves the caret; the box is only ever
written by an insertion; a parse landing mid-composition writes nothing; committing a run
stops it being grey; a stale `doc.for` sends no `modules`; the seed field shows a number after
a documented run, stays blank otherwise, and is untouched by a size change. The composition
row is driven over CDP (`Input.imeSetComposition`) rather than by dispatching a synthetic
`compositionstart`: a fabricated event tests the handler, and what is in doubt is the
browser — `check_regions.py`'s own rule that a driver poking past the interface can pass
while the interface is unreachable.

**Not built:** `tools/eval_interpret.py`. `smoke_parse.py` already *is* fidelity +
compliance, is backend-agnostic by design, and is what `stress_parse.py` drives. Grow its
corpus to the brief's twenty fragments instead — a second scoring tool is a second number
nobody can compare to the first.

## Verification

1. `python3.11 tools/stress_parse.py` across 4B / 8B / 14B — the full matrix per candidate,
   **before anything is wired**. The existing gate, at a few minutes of L4 each. User-text
   preservation is the highest-weight row; at comparable preservation, the most restrained
   model wins.
2. Sweep the invention budget and coverage floor over the corpus and record the margin from
   the nearest rejected document, the way `tune_dupes.py` sets a threshold. No guessed
   numbers ship.
3. `python3.11 tools/preview_ui.py` + `npm run dev` — the whole front end, no GPU, no Modal
   account. Both stub shapes exercise the marks and the inline edit.
4. `python3.11 tools/ui-checks/probe_compile.py` → **diff to zero**, then re-baseline only
   the appended section.
5. `python3.11 tools/smoke_modules.py && python3.11 tools/smoke_prompt.py`.
6. `python3.11 tools/ui-checks/check_document.py <url>` and `probe_console.py` against the
   dev server.
7. `modal deploy app.py`, then the end-to-end loop: type a fragment → marks appear → tap a
   grey run → edit → Generate → the picture changes where the edit implied and nowhere else.
   Reload, open the gallery, Reuse the take, confirm the document restores and Generate
   reproduces it.

## Docs to update in the same commit

- **CLAUDE.md** — the property the invariants serve and the three invariants themselves
  belong here, in "Philosophy", not only in this plan: they outlive the sprint and each one is
  the reason a future cheap fix does not ship. Record contradiction as named and deliberately
  unenforced, with the reason — otherwise the next person reads the coverage number as its
  proxy, which is the one misreading the four checks invite. Plus a
  "Conventions" entry for the interpreter tier (why its own `@app.cls` and not a process
  inside a generator; why the weights follow the captioner rather than the catalogue), and a
  "The page" entry for what the underline and the grey mean. Record the two swept thresholds
  and their margins the way the duplicate bounds are recorded.
- **`docs/vendor-parse-model.md`** — it is already correct and already names the model in use;
  add that it is now wired, and that `app.py` no longer carries a hosted fallback.
- **`semantic-layer-brief.md`** — record the four places the repo overruled it: the user's
  prose rather than the scene document as the source of truth, which governs the other three;
  `origin`/text runs over `source`/character offsets; the text-only abliterated 4B over
  Qwen3-VL; and no concurrency change on the generators.