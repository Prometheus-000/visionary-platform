

# The shot palette — teaching the page to talk to H3

## Context

The composer offers **one textarea for a model that reads a multi-field
document.** That mismatch is the whole reported problem, and every symptom
falls out of it:

- *"No clue where to put camera directions"* — there is no slot for them, so
  every position is a guess.
- *"No idea if you need to describe medium or genre or tone"* — those belong to
  the scene-establishment clause, which has no name on screen.
- *"You shouldn't describe the image you're using, but everyone does"* — the
  place that description belongs (`subject_definitions`) is not on the page, so
  it goes in the only box there is.
- *"You have to guess whether a period or a comma or 'the woman' vs 'a woman'
  changes it"* — a documented grammar, presented as free prose, reads as
  superstition.
- *"Generations cost 2–3 minutes"* — so the guessing is paid for per guess.

MiniMax publishes the format in the model repo itself
([`VIDEO_PROMPT_WRITING_GUIDE_base_en.md`][base], [`_ref_en.md`][ref]). It is
not prose. It is a document, and the app can simply emit it.

Two facts make this cheap rather than a rewrite:

1. **`_h3_graph()` passes the prompt string straight into
   `MiniMaxH3ImageToVideo` / `MiniMaxH3ReferenceToVideo`** ([app.py:3979][g]).
   Whatever we assemble is exactly what the Qwen3-VL-32B conditioner sees.
   There is no template in the way.
2. **The reference chips already write H3's own label syntax.** The guide's
   labels are `<Subject N>` / `<Picture N>` / `<Video N>` / `<Audio N>`, and the
   composer has been lettering chips `<Picture 1>` / `<Video 1>` all along. The
   page is already half-speaking the language.

**No LLM.** The grammar is a table, the structure is a form, and the user's own
sentence is the one part that was never the problem. An LLM here would be a GPU
cold start, an external key, or both, to do work a `dict` does.

**No new text fields.** Anything with a closed vocabulary — camera, framing,
lens, lighting, tone, action, foley, score — becomes a **pill**, not a box to
type in. The prompt field keeps only what nothing else can say: who is in the
shot and what happens. This is the existing rule ("a control that shows its own
value gets no label") applied to words instead of numbers.

[base]: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md
[ref]: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md
[g]: app.py:3979

## The shape

One icon in the options strip, beside `+ LoRA`. It opens a **palette**: a
popover of small black-and-white animated tiles, grouped. Clicking a tile adds a
**pill** to a rail under the prompt field. The pills are the choice; the
compiler turns them into the right clause in the right field at submit.

    ┌─ canvas ──────────────────────────────────┐
    │                                           │
    └───────────────────────────────────────────┘
    ┌ prompt ───────────────────────────────────┐
    │ She reads a letter and stops walking.     │
    │                            [Image][Video] │
    └───────────────────────────────────────────┘
      ◐ close-up   ▷ push in   ☀ window light   ♪ solo piano     ← pill rail
      [model][16:9][720p][5s][seed][gpu] [◈ shot] [+ LoRA] [⚙][Generate]
                                          ↑ one icon, opens the palette

Nothing above costs console height until it is used: the rail is `display:none`
while empty, and the palette is a popover, not a panel.

## What the compiler emits

### H3 — the document

Per the guide, an optional alignment instruction followed by three named fields.
The alignment strings are **verbatim from the guide** and belong in a module
constant with the doc URL beside them, because they are a contract with the
checkpoint, not phrasing we chose:

| task | condition | first line |
|---|---|---|
| `t2va` | nothing attached | *(omitted — begins at the first field)* |
| `i2va` | first frame only | `For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.` |
| `fl2va` | first + last | `How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the S.SS-second mark of the target video.` |
| `l2va` | last frame only | `How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the target video.` |

`S.SS` is the clip length the composer already knows (`seconds`), so it is
filled, not left literal.

Then:

```
integrated_multimodal_description: <framing> <lighting/tone> <your sentence> <action> <camera>
overall_soundscape: <foley pills, one sentence>
non_diegetic_music: <score pills, one sentence>          ← "N/A" when no score pill
```

`non_diegetic_music: N/A` is the guide's own value for "no score" and is the
default. That single line is worth the whole feature on its own: H3 currently
invents a soundtrack for every clip because nothing ever told it not to.

Clause **order inside the description** is the compiler's, not the user's, which
is the point — it is the one thing you cannot get wrong by hand any more:
framing → light/tone → typed sentence → action → camera. Camera goes last
because the guide's rule is *type + amplitude + speed, written as natural
action*, after the thing it is moving around.

**Reference mode (`ref2va`)** uses the guide's six-field form
(`subject_definitions`, `summary`, `retention_analysis`, `detailed_description`,
then the two audio fields). `subject_definitions` is generated from the
reference chips' **roles** — see below — which is what finally makes "don't
describe your reference image" enforceable instead of advice.

### Krea 2 — prose, not a document

The image side has no document. The same pills append their clauses to the typed
sentence in the same order, and camera/action/foley/score pills never appear
there at all.

## Pills

Reuse the existing `.pill` class ([app.py:5949][p], already has an `.on` state)
inside a `.wrap` row ([app.py:6375][w]). A pill is a static mini-glyph plus its
word plus a `✕`. The palette tile is the animated version of the same glyph.

One rail, shared by Image and Video, because the prompt is shared and a pill is
part of the prompt. Pills a model does not read **go dim, not away** — the
established rule from the keyframes/references row: a control that vanishes when
you switch teaches only that the page lost it.

Groups, and which field each compiles into:

| group | pills | image | video | compiles into |
|---|---|---|---|---|
| Framing | extreme wide, wide, medium, medium close-up, close-up, extreme close-up, over-the-shoulder, POV | ● | ● | description, first |
| Angle | eye level, low, high, bird's eye, worm's eye, Dutch | ● | ● | description, first |
| Camera | push in, pull out, pan L/R, tilt up/down, truck L/R, pedestal up/down, orbit, arc, crane up/down, tracking (side/rear), handheld, whip pan, rack focus, zoom, static locked-off | — | ● | description, last |
| Light | window light, golden hour, overcast, hard sun, neon, candlelight, practical, silhouette, top light | ● | ● | description, second |
| Tone | documentary, noir, 16mm, anamorphic, high-key, desaturated, high contrast | ● | ● | description, second |
| Action | fight, kiss, conversation, walk and talk, embrace, chase, reveal, hand-off, turn to camera, laugh | — | ● | description, fourth |
| Sound | room tone, footsteps, wind, rain, traffic, crowd, breathing, cloth, water, fire | — | ● (H3 only) | `overall_soundscape` |
| Score | solo piano, strings, synth pad, percussion, guitar · slow/mid/driving · swelling/steady/fading | — | ● (H3 only) | `non_diegetic_music` |

Each entry is `{key, label, phrase, glyph, groups}` — `phrase` is the exact
wording the guide recommends ("slow push-in from a medium shot to a close-up",
"stable rear tracking shot"), so the pill teaches the phrasing it writes.

**One camera pill at a time.** The guide is explicit — one move per clip, add a
second only when the timing is spelled out — so picking a second camera pill
replaces the first rather than stacking. Same for framing and angle. Light, tone,
action, sound and score stack.

[p]: app.py:5949
[w]: app.py:6375

## Pills that carry a value

A closed vocabulary cannot hold a line of dialogue, and it cannot hold "ice
lightly tapping crystal" — the guide's own example of good foley. So a pill may
be **valued**: choosing it reveals a place to write, and nothing before that.

    ◐ close-up   ▷ push in   ☀ window light   ⊕ dialogue
                                              └─ picking it expands the pill ─┐
    ◐ close-up   ▷ push in   ☀ window light  [ EN ▾ | Take the morning…    ✕ ]

On blur the pill collapses to its own value — `❝ Take the morning with you.` —
so the rail reads as what you chose, not as a form. This is the same move the
page already made when a keyframe tile that *appears* replaced the checkbox that
used to reveal one: the control arrives with the decision, not before it.

An empty valued pill compiles to nothing and says so by staying visibly empty;
it is never a validation error.

| valued pill | field it holds | compiles to |
|---|---|---|
| Dialogue | a language (11, from the guide's list) + the line | `<Subject 1> (S1) says: <d>[English] …</d>` in the description |
| Sound · other | free foley | appended to `overall_soundscape` |
| Score · other | free instrumentation | appended to `non_diegetic_music` |
| On-screen text | the exact string | description, with the guide's do-not-translate framing |
| Subject | a name | seeds `<Subject N>` and lets other clauses refer to it |

**The language tag is not free text.** The guide names exactly eleven and
forbids inventing one, so it is a `select` inside the expanded pill, defaulting
to English — which is also why dialogue is a pill rather than something you type
into the prompt: typed in prose, the language tag is a thing you cannot know to
write.

The line itself is preserved **verbatim, punctuation included** — the compiler
must not strip, sentence-case or re-punctuate what is inside `<d>…</d>`, which
is an explicit rule in the guide and the one place the compiler's usual tidying
would silently corrupt the output.

## The palette

`openMenu()` ([app.py:7371][m]) is the wrong shape — it renders a list of text
buttons. Add a sibling `openPalette(btn, groups)` that shares its lifecycle
(the single floating element, the outside-mousedown close, the scroll-close with
the inner-scroll guard, the viewport clamp) and renders a grid of tiles instead.
Factor the positioning and teardown out of `openMenu` rather than copying it —
the scroll-close guard in particular is a bug that was already fixed once and
must not be fixed twice.

Glyphs are **one shared 40×28 inline SVG skeleton** — a frame rect, a subject
ellipse, a horizon line — animated by a CSS class per move (`.g-push`,
`.g-pan`, `.g-orbit`…) with a ~1.6s loop. Twelve keyframe rules cover the camera
group; bespoke SVG per tile would be forty drawings that drift apart. Black and
white, `prefers-reduced-motion` freezes them on the mid-frame.

Only the open palette animates, and it is removed from the DOM on close, so
there is no idle cost — the same reason `openMenu` is one element that moves.

[m]: app.py:7371

## Reference roles — the `subject_definitions` half

Each reference chip ([app.py:6233][c]) gains a role, set from the chip's own
menu: **identity · wardrobe · location · style · prop · action**. The role
compiles to `<Picture 2> defines the green jacket and cream shirt.` — the
guide's own construction.

This is what makes the "don't describe the picture" rule real: there is now
somewhere for that description to go that is not the prompt field, and it is one
click rather than a sentence.

Roleless chips keep working exactly as today. A chip with no role compiles to
nothing and the run is what it is now.

[c]: app.py:6233

## Where the code goes

**Backend** (`app.py`, all pure stdlib and importable from the web container —
same reason `_validate_loras()` is, so a bad selection is a form error in
milliseconds rather than a cold H100):

- `SHOT_VOCAB` — the table above, near `VIDEO_MODELS` (~app.py:4164). Served to
  the page via `/api/state` (~app.py:4817–4842) next to `video_models` and
  `wan_experts`, **not** written into `UI_HTML` — the existing rule that a copy
  in the HTML is a second source of truth.
- `H3_ALIGN` — the four verbatim strings, with the guide URL in the comment.
- `_h3_task(first, last, refs, vids)` — beside `_wan_task()` (app.py:4265).
  Today `/api/video` collapses to `"ref2va" if (refs or vids) else "fl2va"`
  (app.py:~5551), which is right for *which checkpoint loads* and too coarse for
  *which alignment instruction*: first-only, last-only and both are three
  different sentences. Keep the checkpoint choice as-is; this is a second, finer
  read used only by the compiler.
- `_compile_h3_prompt(...)` / `_compile_image_prompt(...)` — take the typed
  text, the pill keys, the ref roles and the task; return the string.

**Compiled in the route, not the client** — `/api/video` (app.py:5486) and
`/api/generate` — so there is one implementation, an unknown pill key is a form
error, and the sidecar records exactly what the encoder was given.

The job then carries both: `prompt` (the compiled document, what ran) and
`prompt_typed` + `shot` (what you chose). `_write_output_meta()` (app.py:2988)
takes `**fields`, so both fit with no signature change. **The gallery card and
`#vid-meta` must prefer `prompt_typed`** — a card showing a six-field document
is a card you cannot read.

**Frontend** (`UI_HTML`):

- Pill rail markup under `.field`, above `#lora-note`.
- `◈ shot` button in both option strips, next to `#add-lora` / `#v-add-lora`.
- `openPalette()` beside `openMenu()`; `SHOT_GLYPH` CSS near the `.menu` block
  (~app.py:6034).
- `readShot()` returning the pill keys + ref roles, wired into the two `post()`
  bodies (app.py:9426 and app.py:9821).
- Per-model filtering off the `supports` map that already exists — sound and
  score pills dim on Wan, which is silent; camera and action dim on Image.

`tools/preview_ui.py` stubs need a `shot_vocab` in the state fixture, and its
existing habit of holding the awkward states applies: a pill set long enough to
wrap the rail to two lines, and a model that reads none of them.

## Not in scope

Multi-shot timeline prompting — `[Shot 2] At 00:04.500, the camera cuts to…` —
is a real part of the format and deserves its own pass; it changes what a "take"
is, which the composer is not built around today. With it goes multi-speaker
dialogue: a second `(S2)` is only worth having once there are shots to cut
between, so the dialogue pill lands single-speaker and the compiler assigns
`(S1)` unconditionally.

Everything else in this plan is one shot, which is what the page already makes.

## Verification

1. `python@3.11 tools/preview_ui.py` — palette opens, tiles animate, pills add
   and remove, rail wraps, camera pills replace rather than stack, pills dim on
   Image and on Wan. A valued pill expands on add, collapses to its value on
   blur, and an empty one stays empty rather than erroring. No console errors.
2. A unit check of the compiler against the guide: for each of the four tasks,
   assert the first line matches `H3_ALIGN` verbatim and that the three field
   labels appear once each in order; assert a dialogue line containing commas,
   an ellipsis and a trailing exclamation survives inside `<d>…</d>`
   byte-for-byte. Belongs beside `tools/smoke_graphs.py`, which already exists
   to check offered values against what the backend accepts.
3. `modal deploy app.py`, then the run that motivated this: a real H3 clip with
   framing + camera + light pills and no score pill. Confirm the sidecar's
   `prompt` is the assembled document, `prompt_typed` is the typed sentence, the
   gallery card shows the typed one — and that the clip comes back **silent
   under the foley**, which is the observable proof `non_diegetic_music: N/A`
   reached the model.
4. Same prompt with and without the pills, at draft tier, as the honest
   before/after.