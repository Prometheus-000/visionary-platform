# The H3 scene composer — where it stands

Written at the end of the session that built the compiler, for whoever picks it
up next. Everything below was verified against the tree at `3f29405`, not
recalled — **and the first half of it has since been done**, at `6742ff8`,
`aba9892` and `cd09426`. The surface exists. What is left is blocking and the
measurement, which are steps 2 and 3 below.

## The one-sentence version

**The compiler was finished and had no surface. It has one now.**
`_validate_scene` and `_compile_h3_scene` turn a cast and a shot timeline into
MiniMax H3's published grammar, `_stage_*` derives camera and staging language
from an arrangement of marks, 145 assertions cover both — and the video console
is the thing that fills them in. `_stage_*` is still reachable from nothing but
the probe, which is the next piece of work.

## What exists

### The compiler, in `app.py`

| | |
|---|---|
| `_validate_scene` (7480) | normalises a cast, a shot timeline and its sources; refuses by name |
| `_compile_h3_scene` (8615) | the six-field reference form, or three fields when nothing is attached |
| `_h3_shot_text` (8514) | one `[Shot N]` block in the guide's shape |
| `H3_CAST_KINDS` (7391) | character / place / thing, with typed reference slots |
| `H3_RETENTION`, `H3_AUDIO_RETENTION` | the two marker vocabularies — they are not the same list |
| `_h3_subjects`, `_h3_label`, `_h3_speakers` | `<Subject N>` numbering, `(S1)` speaker IDs by order of vocal event |
| `MAX_H3_PROMPT` | 7000, taken from the vendored grader — stated in neither guide |

`tools/vendor/minimax_score/` is MiniMax's own format grader, pinned and
test-only. Our output scores **1.00** against it.

### Blocking, in `app.py` (`_stage_*`, 7841–8410)

A pinhole camera over a ground plane. Marks carry position, facing and their own
dimensions; the camera carries position, yaw, **tilt** and a lens. Out of it
comes shot size, angle, screen position, depth order, orientation, proximity, the
camera move with amplitude and speed — and the same marks projected to
normalised region rectangles for Krea 2.

It was measured against Enter the Void's death scene, and that reference broke
seven assumptions. Read the `_stage_*` comments before changing any of it; each
one names the failure it exists for.

### The probe, in `web/src/blocking/`

`web/blocking.html` → `blocking-main.tsx` → `Viewfinder.tsx`. A second page in
the existing Vite app, on the `storyline.html` precedent. **Throwaway** — nothing
in the product imports it.

```
drag a body        move a person on the floor
drag empty space   dolly and truck — you walk
two-finger ⇔ ⇕     turn your head, both axes
⌘ two-finger ⇕     crouch and rise
pinch              the lens
double-tap a body  see through their eyes
escape             step back out
⌥ tap a body       stand them up, lay them down
space              drop the camera's mark
```

`derive.ts` mirrors `_stage_*` in TypeScript. The duplication is deliberate and
is the thing to watch: `/api/compile` is the authority, and if the two ever
disagree, the TypeScript one is wrong.

Run it with `preview_start` on the `blocking` config in `.claude/launch.json`,
then open `/blocking.html`.

### `web/src/scene/model.ts`

260 lines mirroring `_validate_scene`. **Imported by nothing.** It is the shape
the composer will hold, written before the composer.

## The composer, built to `console-ladder.html`

That file is the design and it **supersedes `scene-composer.html`** on layout —
the earlier mockup is a scrolling page with a 132px canvas stub, and the ladder
redraws all five states at true 1512x982 proportion against `CONSOLE_BUDGET`.
Read the ladder. Its own footnote is the reason: twice while it was being drawn
the prose and the picture disagreed, and both times the prose was the optimistic
one.

    web/src/scene/model.ts     the shape, mirroring `_validate_scene`
    web/src/scene/Shots.tsx    the timeline — one shot is the prompt box
    web/src/scene/Mentions.tsx typing `@` is the picker
    web/src/scene/CastCard.tsx what a chip opens
    web/src/scene/Slot.tsx     a drop, and the role it lands on
    web/src/scene/Doors.tsx    cast / shot / LoRA on the field's trailing edge
    web/src/scene/SourcePane.tsx  the compiled document, as view source
    web/src/scene/pool.ts      one file, keyed by what travels

Five things it settled that are cheaper to read than to rediscover:

- **The degrade is exact and everything rests on it.** One shot, no cast, no
  pills: `readScene` returns null, no `scene` key is sent, and the run is the
  typed text byte-for-byte. The surface is identical to the prompt box it
  replaced, under the same `#prompt` id.
- **The share is a readout.** A shot's slice of the clip is the length of what
  you wrote about it, which removes a 9px drag target rather than growing it to
  a thumb. `beats` survives as the override.
- **Rows divide the field's allowance rather than adding to it.** `growRows` is
  `autoGrow` over n boxes and reduces to it exactly at n=1.
- **The mirror is how anything knows where the caret is.** It is a
  glyph-for-glyph copy of the textarea, so a zero-width span spliced in at the
  caret's index sits exactly where the caret does. Third job that mirror has
  done: marks, then mentions, now this.
- **`.console` is `overflow:auto`,** so every floating piece is portalled and
  fixed. Not `Popover`, which closes on scroll — right for a menu, wrong for a
  box you type a name into.

## What does not exist

- **Blocking is not reachable from the app.** Only the probe page. `_stage_*`
  and `_stage_boxes` are live in the compiler and nothing fills them in, which
  is exactly the state the *scene* was in before this session.
- **`retention` is not adjustable.** It reaches the document at its default and
  no control sets it. That is a default that reaches the output, not a `ties`.
- **Wan gets the timeline and cannot read it.** `/api/video` compiles a scene
  only for H3; on Wan the rows join into prose. Nothing says so on screen yet.
- Watch the word when you grep. `useGenerate.ts` sends `scene:` and
  `Inspector.tsx` describes one — those are the **image** side's scene *plate*,
  a photograph the frame is generated inside, routed to krea2edit. Unrelated to
  the H3 scene *document*, and the only collision of names in this area.

## What the session established that constrains the design

These cost real measurements. Do not re-derive them.

1. **A model writing prose at H3 is the wrong shape.** Two features failed this
   way — the semantic layer (0 wins in 30 blind render comparisons) and the
   motion panel. The owner's read of the second: *"it worked, but not as an h3
   prompt."* H3 reads a document with named fields. The compiler is the answer;
   a model writing the prompt is not.

2. **Nothing else goes in the generator process.** Both GPU classes are
   `max_containers=1` with `@modal.concurrent(max_inputs=1)`. Anything holding
   that slot delays every render behind it — which is what a ten-minute wait
   turned out to be, and it was blamed on the wrong feature for a day.

3. **A wait costs what it shows, not what it takes.** The owner's framing, and
   the most useful sentence of the session:

   > I can sit here with you for 30-40 minutes … because I can see everything
   > you do … Looking at a black screen do absolutely nothing after you hit the
   > main button even for a minute feels like an eternity.

   Any surface built here has to narrate itself. See the Antifragile section of
   `CLAUDE.md`.

4. **The validator stays arithmetic; the harness gets the judge.**
   `tools/judge_prompts.py` marks a compiled prompt against what the person said
   — four criteria plus `lost` and `contradicted`, every verdict carrying a
   quote. Criterion 3 (*space*) is precisely what blocking exists to answer.
   `tools/judge_renders.py` scores the pictures instead and is the only
   measurement here that is not a proxy. `tools/serve_judge.py` opens a Sandbox
   for either.

## Ranked next steps

1. ~~**The composer surface.**~~ Done — see above.
2. **Wire blocking to it.** `_compile_h3_scene` already takes a stage per shot,
   so this is a projection rather than a feature — and `_stage_boxes` gives
   Krea 2 regions from the same arrangement for free.
3. **Measure it.** `tools/prompt_ab.py` runs the three stages in order: render a
   pair, serve a vision model, judge blind in both orders. Blocked against
   unblocked from one scene, one seed. Until that runs, everything above is a
   proxy — **including the composer**, which has been driven and read and not
   once measured against a render.

## Open, and deliberately not decided

- **Pose beyond stand/lie.** `_stage_dims` takes `h`/`w`/`base` per mark, so a
  sitting figure is expressible today; nothing chooses those numbers for you.
- **Light as a transition.** Framing and angle state both ends of a move now;
  light and tone still describe one steady state. Enter the Void spends 42% of
  its runtime on a lighting event inside a single continuous shot, and there is
  nowhere to say that.
- **POV without limbs.** `STAGE_OWN_BODY` emits the rider's own arms and torso
  when the camera looks down far enough. Looking away from yourself, nothing in
  the frame says whose eyes these are — which is true of the film too, and is
  why a video model reading it saw no POV at all.
