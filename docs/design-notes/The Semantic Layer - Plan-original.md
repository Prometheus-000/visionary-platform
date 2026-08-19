What do you think of this plan?

# The semantic layer

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
   never run, so there is no source of truth to edit.
3. **The interpreter is the wrong one.** `PARSE_MODEL = "claude-sonnet-5"` and the whole
   `_anthropic_key` path are traces of an approach that was abandoned.
   [docs/vendor-parse-model.md](docs/vendor-parse-model.md) already pins the real decision —
   an abliterated **text-only** Qwen3-4B — with a refusal argument the brief does not engage:
   with a schema bound, a refusal arrives as an *evasive storyline that satisfies the schema*
   and nothing downstream can tell it from a real one.

Outcome: type a fragment, see the system's assumptions marked inline, edit one inline, press
Generate, and only what the edit implied changes.

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

## Stage 0 — the local interpreter

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

- **`store.ts`**: `doc: { for: string; elements: ParseElement[]; text: string } | null`.
  `for` is the prose it was parsed from and is the entire staleness mechanism.
- **`docLive(s)`** — one exported helper, used by `imageBody`, `videoBody` and `reuse`, so
  there is one comparison rather than three. **`modules` is sent only when
  `doc.for === stripLoras(s.prompt)`.** A document that does not describe what is in the box
  is not sent and the run degrades to today's byte-identical plain path. This is what keeps
  non-negotiable #6 true.
- **`useMarks` → `useDocument`** ([web/src/console/useMarks.ts](web/src/console/useMarks.ts)),
  writing `s.setDoc(...)` inside the existing `asked !== seen.current` staleness guard —
  that guard is already correct, it just gates one more write. `setDoc(null)` on failure or
  an empty prompt.
- **`imageBody`** ([useGenerate.ts:62](web/src/canvas/useGenerate.ts:62)) and `videoBody`
  ([useVideo.ts](web/src/video/useVideo.ts)) each gain one line.
- **`/api/generate`** ([app.py:8856](app.py:8856)) and **`/api/video`**
  ([app.py:8928](app.py:8928)): `_validate_modules` inside the **existing** `try` beside
  `_validate_shot`, so an unknown role is a named form error on CPU before a cold H100.
  Pass `modules` to the compiler already being called; add `"modules": modules` to the spawn
  params. Widen `/api/generate`'s `if not prompt and not regions` guard to admit a document.
- **`_shot_meta`** ([app.py:7184](app.py:7184)): emit `modules` when a document ran. The
  existing `typed == params["prompt"]` guard stays — a one-module document whose compile
  equals the typed text writes nothing, which is right: the run is reproducible from the
  typed prompt alone. Add a comment saying so.
- **`reuse.ts`** ([web/src/gallery/reuse.ts](web/src/gallery/reuse.ts)): restore `doc` with
  `for` set to the *same string* `setPrompt` writes. This is the plan's one real footgun —
  it is why `docLive` is a shared helper. Cards and `MetaSheet` show nothing new.

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
  any derived run is missing, keep the typed text and drop the document. This is
  `promptMarks`'s "skipped rather than guessed at" rule applied to the write, and it is what
  makes "the compiler never rewrites your sentence" survive one layer up: the model may
  *insert*, never revise.
- `.mk-i` splits into invented (grey + underline) and derived-run (underline only). Underline
  means *addressable*; colour means *whose*. Provenance stays binary — the underline is reach,
  not a third state.
- **The undo objection is already spent.** `moveClause`, `nudgeLora` and the `+ LoRA` caret
  sink ([Field.tsx:74,90,127](web/src/console/Field.tsx:74)) all write the textarea's value
  through React today, so native undo is already superseded on three paths. Add a one-slot
  `docUndo` in the store restoring the pre-parse `{prompt, doc}` pair, bound to ⌘Z in the
  field. One slot, because the gesture is one write at a time.
- `preview_ui.py`'s `/api/parse` stub ([tools/preview_ui.py:881](tools/preview_ui.py:881))
  gains a second shape: an element the prose does **not** contain. ~6 lines, and it is what
  makes the whole feature developable with no GPU and no Modal account — the file's stated
  reason to exist.

**Checkpoint:** an invention is visible inline, the caret never jumps, and
`probe_console.py` shows the 30% console budget holding.

## Stage 3 — the seed pin

Non-negotiable #2, with no new control.

- In `finish()` ([useGenerate.ts:117](web/src/canvas/useGenerate.ts:117),
  [useVideo.ts:79](web/src/video/useVideo.ts:79)) — both already read the seed off the record
  for `meta` — write it back to `s.img.seed` / `s.vid.seed` **only when a document exists and
  the field is blank**.
- **The `doc &&` condition resolves #2 against #6.** Read "pinned once a shot exists" as
  "once a *scene document* exists". A seed that stops rolling for someone who never engaged
  reads as *"Generate is broken — it keeps making the same picture."*
- **The reroll gesture already exists.** `SamplingButton.tsx:106`'s seed field shows
  `placeholder="random"` and now shows a number; `SEED_HINT` already reads *"Blank draws a
  new one"*; Reset already writes `seed: ''`. Add one sentence to `SEED_HINT` naming the new
  state — the carve-out CLAUDE.md already grants, since a field that silently stopped being
  random is a value that cannot show itself. **No lock glyph, no dice, no chip.**

**Checkpoint:** two Generates with an unchanged document give the same picture; with no
document, they do not.

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
- **`_merge_document(old, new, only)`, new: refuses to overwrite any element whose `origin`
  is `derived`.** Non-negotiable #3 enforced at the boundary, on CPU, by name — the posture
  `_validate_shot` already takes toward an unknown pill. A model that ignores the instruction
  cannot damage the user's words; it can only fail loudly.
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
document and `_shot_meta`'s emit/omit pair. `smoke_parse.py` drops `--backend hosted`, gains
the reroll shape and the `_merge_document` refusal. `preview_ui.py` gains the second stub
shape. `probe_console.py` gains a document-present row.

**New:** `tools/smoke_interpret.py` (CPU-only, in the spirit of `smoke_caption.py`) — the
endpoint resolves, `PARSE_SCHEMA` is valid `$defs`/`$ref`, a dialect binds.
`tools/ui-checks/check_document.py` — typing never moves the caret; the box is only ever
written by an insertion; committing a run stops it being grey; a stale `doc.for` sends no
`modules`; the seed field shows a number after a documented run and stays blank otherwise.

**Not built:** `tools/eval_interpret.py`. `smoke_parse.py` already *is* fidelity +
compliance, is backend-agnostic by design, and is what `stress_parse.py` drives. Grow its
corpus to the brief's twenty fragments instead — a second scoring tool is a second number
nobody can compare to the first.

## Verification

1. `python3.11 tools/stress_parse.py` — score the pinned model before wiring it in. This is
   the existing gate and it costs a few minutes of L4.
2. `python3.11 tools/preview_ui.py` + `npm run dev` — the whole front end, no GPU, no Modal
   account. Both stub shapes exercise the marks and the inline edit.
3. `python3.11 tools/ui-checks/probe_compile.py` → **diff to zero**, then re-baseline only
   the appended section.
4. `python3.11 tools/smoke_modules.py && python3.11 tools/smoke_prompt.py`.
5. `python3.11 tools/ui-checks/check_document.py <url>` and `probe_console.py` against the
   dev server.
6. `modal deploy app.py`, then the end-to-end loop: type a fragment → marks appear → tap a
   grey run → edit → Generate → the picture changes where the edit implied and nowhere else.
   Reload, open the gallery, Reuse the take, confirm the document restores and Generate
   reproduces it.

## Docs to update in the same commit

- **CLAUDE.md** — a "Conventions" entry for the interpreter tier (why its own `@app.cls` and
  not a process inside a generator; why the weights follow the captioner rather than the
  catalogue), and a "The page" entry for what the underline and the grey mean.
- **`docs/vendor-parse-model.md`** — it is already correct and already names the model in use;
  add that it is now wired, and that `app.py` no longer carries a hosted fallback.
- **`semantic-layer-brief.md`** — record the three places the repo overruled it: `origin`/text
  runs over `source`/character offsets, the text-only abliterated 4B over Qwen3-VL, and no
  concurrency change on the generators.