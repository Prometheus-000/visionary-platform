# UX Audit — Visionary

*Audited 2026-08-20 · mode: **live + static** — live against `tools/preview_ui.py` (stub API,
real compilers/vocabulary, no Modal deploy) + vite dev at :5173; static over `web/src`
(~11.6k lines). GPU jobs were stubbed, so generation quality and real latency were out of
scope; everything about layout, flows, states, and copy was exercised for real.*

## Macro read

Visionary's interaction design is unusually deliberate — wait states on the expensive paths
are exemplary, destructive actions state their exact blast radius, and the disabled
Start-training button that explains itself step-by-step is a pattern most products never
reach. The gap is between the paths someone clearly polished (generate, download, caption)
and everything around them: a one-line CSS token bug is silently deleting the card
background across nine surfaces, eight smaller mutations give no feedback at all between
click and reply, dim text sits below accessibility contrast on every ground it's used on,
and the style system the code defines is bypassed more often than used. The product's own
philosophy — "errors diagnose themselves," "the dialog has to say what is going" — is the
right yardstick, and the P1s below are simply the places the implementation doesn't yet
meet it.

## Flow inventory

| Flow | Steps to goal | Notes |
|---|---|---|
| Generate a still | 3 (type → Generate → view) | No dead ends; strong wait state |
| Still → video | 1 (duration menu) | Controls swap correctly; canvas still vanishes (see finding) |
| Enhance a fragment | 1 | Silent when it fails (see finding) |
| Train a LoRA | ~5 (create → name → trigger → set → start) | Guided by self-explaining disabled button |
| Prepare a dataset | 3–4 (drop → caption → trigger → save) | Filters good; two disagreeing tallies (see finding) |
| Download weights / settings | 2–3 | Honest cost copy; several silent mutations |

No flow forces a detour into an adjacent workflow — cross-links (gallery → "Animate from
this frame", training card → LoRA picker) hand results forward instead. That's the right
shape.

---

## Findings

### Visual system (all screens)

**P1 — Card backgrounds are transparent everywhere: `--wash-1` is defined as itself**
- Evidence: `web/src/styles/ui.css:49` — `--wash-1:var(--wash-1)`. A self-referencing
  custom property is invalid at computed-value time, so `background:var(--wash-1)`
  resolves to nothing. Confirmed live: `.gal` computes `background-color: rgba(0,0,0,0)`,
  and `getComputedStyle(root).getPropertyValue('--wash-1')` returns empty. Nine consumers
  lose their ground: `.card` (521), `.shot` (648), `.gal` (711), `.sess` (1230),
  `.ds-card` (1341), `.tile` (1378), `.dupes-tableWrap` (1450), and ui.css:1652.
- Fix: `--wash-1: rgba(255,255,255,.03);` (the ramp's own comment implies a six-step
  scale under `--wash-2:.05` / `--wash-3:.07`). One line restores the intended card
  hierarchy across the app.

**P2 — `--dim` text fails contrast on every background it's used on**
- Evidence: `--dim:#5a5a5a` (ui.css:47). Computed ratios: **3.04:1** on `#000`,
  **2.82:1** on the `--wash-2` field ground, **2.74:1** on `#111` menus — all below the
  4.5:1 AA floor for normal text. It's used for real reading content at small sizes:
  `.muted` 12px (603), `.sub` 13px (520), `.foot .when` 11px (725), menu hints 11px on
  `#111` (758), control labels 14px (293), storyline labels 10.5–11.5px
  (`storyline/storyline.css:35,40,49,68`).
- Fix: raise `--dim` to `#767676` (4.54:1 on black), or split it: keep `#5a5a5a` strictly
  for disabled states and introduce a `--quiet` ≥ 4.5:1 for secondary text.

**P2 — Touch targets far below minimum, with only one touch-mode exception**
- Evidence: `.mk-reroll` **16×16px** (ui.css:441-443), `.ref button.x` 19×19 (1612),
  `.shot .acts button` 20×20 (689), drawer-strip `.foot .more` 20×20 (2038), seg buttons
  min-height 26px (586). Exactly one control is bumped to 44px under
  `@media (hover:none)`: `.shot-back` (1209).
- Fix: extend the existing `hover:none` bump into a general rule — minimum 44px hit area
  (padding or `::after` expansion, not necessarily visual size) for all interactive
  elements in touch mode.

**P3 — The token system exists and is routinely bypassed**
- Evidence (tallies over all CSS + inline styles): **border-radius: ~24 distinct values
  across 96 declarations, 21% tokenized** — and `--r-sheet:20px` (ui.css:71) is defined
  but never used; both sheets hardcode `20px` (1056, 1536). **Font sizes: 12 distinct px
  values, one single use of `--label-size`.** **Colors: 100 distinct literals** — 26
  white-alphas (.012–.92), three near-identical scrims (`rgba(0,0,0,.6/.62/.66)` ×5
  each), and 9 alphas of the semantic red; `#f87171`/`#4ade80`/`#fbbf24` appear as raw
  hex 6/5/3 times despite `--crit`/`--ok`/`--warn` existing (e.g. 496, 604, 143).
  **Spacing: 33 distinct values, zero tokens; the 5/6/7/8/9px band alone is 142 uses.**
- Fix: mechanical migration pass — radii to the four existing tokens, semantic hex to the
  three existing tokens, collapse the 5–9px band onto a 4/8 step. No new system needed;
  use the one already declared.

**P3 — 34 font-size declarations below 12px**
- Evidence: 9px (`.tile .clip .kind`, ui.css:1409), 10px ×10, 10.5px ×3, 11px ×15,
  11.5px ×5 (full list in ui.css/storyline.css/sandbox.css census).
- Fix: floor at 11px, and pair anything under 12px with a ≥4.5:1 color (compounds with
  the `--dim` finding — most sub-12px text is also dim).

### Generate flow

**P2 — Switching Still → video silently discards the canvas result**
- Evidence: observed live — generated a still, chose "8s" from the duration menu, canvas
  replaced by an empty video placeholder with no trace of the image or pointer to it (it
  survives only in the gallery panel, which was closed at the time).
- Fix: keep the last still visible as the default "first frame" candidate when switching
  to video (the First-frame slot already exists in the source row), or at minimum a
  transient "your still is in the gallery" affordance.

**P2 — A failed Enhance is indistinguishable from a no-op Enhance**
- Evidence: observed live — POST `/api/rewrite` returned 200 without a `text` field; the
  UI did nothing at all. Deliberate: `console/Rewrite.tsx:35-40` applies the result only
  `if (!failed(r) && r.ok && r.text)`, with the comment "the worst case here is the box
  unchanged." From the user's seat, press → flash → nothing could mean "my prompt was
  already good" or "the call died."
- Fix: on failure or empty result, show a transient note near the button ("Couldn't
  rewrite — try again"), keeping the box-unchanged behavior.

**P2 — Enhance's explanation names the wrong model in video mode**
- Evidence: observed live — in MiniMax-H3 video mode the button's tooltip/accessible name
  still reads "Restructures your prompt the way **Krea 2** reads best…". The note comes
  from server state and is rendered unconditionally (`console/Rewrite.tsx:52`,
  `title={o.note}`).
- Fix: either serve a mode-aware note, or make the copy model-neutral ("…the way the
  model reads best") — one word per concept includes model names.

**P3 — LoRA skip warning gives a diagnosis but no action**
- Evidence: canvas warning "not applied: gone (no matching keys)"
  (`canvas/Canvas.tsx:282`, reason attached at `canvas/useGenerate.ts:118`). "No matching
  keys" is accurate and unactionable; with a LoRA whose name reads as English ("gone")
  the whole line parses as gibberish.
- Fix: quote the name (`not applied: "gone" — its keys don't match this model; it was
  likely trained for a different base`), which names the cause *and* the likely fix.

### Errors & feedback (cross-cutting)

**P1 — A failed trigger-word save reports nothing at all**
- Evidence: `datasets/useDatasets.ts:184-187` — `commitTrigger` awaits `setDatasetMeta`
  and never checks `failed(r)`. The trigger word is what makes a LoRA invocable; a user
  who saw their trigger "stick" locally and then trains has spent real GPU money on a
  set without it. Misleads with cost attached.
- Fix: check the reply like every sibling call does and surface the existing err-box.

**P2 — Eight mutations have no pending state between click and reply**
- Evidence (handler locations): save HF token (`settings/Settings.tsx:52-57`, button
  :114), delete LoRA (:59-79, :189-190), remove caption model (:311-317 — while the *add*
  path right beside it does show `'Checking…'`, :294), delete set
  (`datasets/useDatasets.ts:166-181`), save set (`datasets/Editor.tsx:218-221` — Save
  stays enabled; double-submit possible), Fix/prepend trigger (`Editor.tsx:564-571`),
  delete session (`train/useSessions.ts:101-108`; ✕ stays enabled,
  `train/SessionCard.tsx:115-119`), gallery delete/purge (`gallery/Gallery.tsx:171-210`).
- Fix: one shared busy-button affordance (the codebase already has the pattern four ways:
  `dl.busy`, `s.rewriting`, `saving`, `deleting` — pick one and apply it to all eight).

**P2 — Fallback error strings are dead ends, and raw server bodies leak into the UI**
- Evidence: `'Generation failed'` (`canvas/useGenerate.ts:189`, `video/useVideo.ts:141`),
  `'Download failed'` (`settings/useDownload.ts:69`), `'Could not read that file.'`
  (`App.tsx:361`, `video/SourceRow.tsx:73`), `'Could not read that image.'`
  (`App.tsx:466`) — none offer a next step. Meanwhile `api/client.ts:37` builds
  `` `${status} ${statusText} ${body.slice(0,400)}` `` — up to 400 chars of what the
  comment itself calls "a plain-text traceback" — flowing verbatim into every err-box and
  `alert(r.error)` (`gallery/Gallery.tsx:179,207`); `datasets/Editor.tsx:108-109` leaks
  300 chars the same way.
- Fix: fallbacks name the action and a step ("Generation failed — the job log is in the
  Training panel"; "Could not read that file — is it an image or clip?"). Keep the
  traceback, but behind a "details" disclosure instead of the headline.

### Dataset editor

**P2 — Two disagreeing progress tallies sit side by side in the toolbar**
- Evidence: observed live — "12 images · 5 captioned" next to "14/19 trigger · 19/24
  captioned". They are different sources of truth rendered adjacently: a live client
  count (`datasets/Editor.tsx:178`, rendered :227) vs the server-computed `ds.insight`
  (:279-281), which lags until re-analysis.
- Fix: derive both from one source, or label the insight pair with its scope and
  staleness ("last analysis: 19/24") so the disagreement reads as freshness, not error.

**P3 — Unlabeled − / + control pair in the toolbar**
- Evidence: observed live — icon-only −/+ between the filters and the tallies with no
  visible label (thumbnail density, per `Editor.tsx:270-274`).
- Fix: a `title`/aria-label ("Smaller/larger tiles") — or fold into the existing seg
  control style with a tiny grid glyph.

### Training

**P2 — The session form mirrors the backend's TrainParams one-to-one**
- Evidence: 11 numeric dials from the `DIALS` map (`train/SessionForm.tsx:32-42`) + 3
  selects + fp8 checkbox + name/trigger/set ≈ **18 always-visible controls**, spread
  verbatim from `state.train_defaults` (`train/useSessions.ts:112-119`). A "More dials"
  expander exists, but Rank/Alpha/optimizer/scheduler/Flow shift sit outside it — the
  README's promise is "point the trainer at a folder," and the form's shape is the
  schema's, not that task's.
- Fix: front page = name, trigger, set, epochs; everything else behind More dials with
  the current defaults. The self-explaining footer already guides the four that matter.

### Responsiveness

**P2 — The mobile console wrap is unowned**
- Evidence: observed at 375px — the strip wraps into three ragged rows, "Still" orphaned
  on its own line, the last-shot thumbnail floating between the model name and Generate.
  The stylesheet knows: "Ten controls wrapping into four ragged rows" (ui.css:306-311);
  the only ≤640px accommodation is `gap:6px` (315-317). No horizontal overflow, so it
  functions — it just isn't designed.
- Fix: an explicit ≤640 order — row 1: prompt; row 2: duration · size · Shot · Enhance;
  row 3: model + Generate; thumbnail into the gallery door.
- Related hygiene (P3): Settings inputs `width:158, flex:'none'`
  (`settings/Settings.tsx:138, :340`) can't shrink and Settings has no media query;
  `storyline.css`/`sandbox.css` have zero media queries; `.shot img` has no reserved
  aspect box (ui.css:656) so a landing render reflows the canvas column.

### Gallery

**P3 — Download/Delete exist twice per card**
- Evidence: hover quick-actions (`gallery/Card.tsx:85`) and the same two entries again in
  each card's More menu (`gallery/Gallery.tsx:212-223`).
- Fix: quick actions on hover *or* in the menu, not both; the menu is the discoverable
  home, hover is the shortcut — if both stay, that's a defensible convention, but it's
  the only duplicated pathway in the app, so decide it deliberately.

### Terminology (one word per concept)

**P3 — "session" vs "run", and "LoRA" vs "checkpoint", for the same things**
- Evidence: the board says "N sessions" / "+ Create session" / "No sessions yet."
  (`train/Train.tsx:153, :169, :179`) while the header door for the same records says
  "Training · N **runs**" (`App.tsx:563`), and the board's own empty-state copy switches
  mid-paragraph (Train.tsx:181-183). One dialog uses both artifact nouns at once: "Any
  **checkpoints** it already wrote stay in **loras/**…" (`train/SessionCard.tsx:117-119`)
  while Settings manages the same files as "LoRAs" (`settings/Settings.tsx:172`).
- Fix: pick "session" for the record and "LoRA" for the artifact, everywhere; "run" and
  "checkpoint" survive only if given distinct meanings the UI actually teaches.

### Dev tooling (first-run experience of the repo itself)

**P2 — `npm run dev` gets a dead API out of the box: port defaults disagree**
- Evidence: hit live — vite proxies `/api` to `:8791` by default (`web/vite.config.ts:14`)
  but the stub binds `:8777` (`tools/preview_ui.py:56`). A fresh contributor following
  the documented loop gets proxy errors with no hint which side is wrong.
- Fix: make the two defaults equal (one-character change on either side).

---

## What's already right

- **Wait states on the paths that cost money are exemplary.** Generate flips to Stop with
  a canvas progress line and "H100 · Step 3/8"; downloads and captioning show live
  progress; stop-training disables itself (`canvas/useGenerate.ts:169`,
  `console/Console.tsx:244`, `settings/Settings.tsx:147`, `datasets/Editor.tsx:615`).
- **Progressive disclosure via the duration switch is the real thing.** Still ↔ video
  swaps the model, the placeholder copy, the source row, and drops +LoRA when the model
  can't take one — controls follow relevance, observed live.
- **Destructive dialogs state the blast radius**, including what's *excluded*: "The N
  older than these are not included." (`gallery/Gallery.tsx:187-210`). This is the
  standard the error copy should be held to.
- **The disabled Start button explains itself stepwise** — "Name the LoRA" → "Set a
  trigger word", updating live as the form fills (observed).
- **Task-verb menus**: "Reuse prompt & settings", "Animate from this frame", "Use as
  reference" — user tasks, not CRUD.
- **Honest state copy**: "UNSAVED · cleared when you close the app"; GPU picker: "Changing
  a card costs one cold start… Runs after it are warm."
- **Actionable errors exist as a house style to copy from**: "References need MiniMax-H3 —
  download it under Settings." (`App.tsx:371`), "Give the set a name to save it."
- **The IA is task-shaped, not route-shaped**: 33 API routes funnel into four surfaces;
  "Generate is the page, not a destination" (`App.tsx:33`).
- Media boxes mostly reserve aspect-ratio; `prefers-reduced-motion` is respected;
  `aria-busy` is used where busy states exist.

## Fix plan (by leverage)

1. **`--wash-1` one-liner** (S) — restores the card ground on nine surfaces. Resolves the
   P1 visual bug.
2. **`commitTrigger` failure check** (S) — the other P1; wire it to the existing err-box.
3. **Raise `--dim` to ≥ 4.5:1** (S) — clears the contrast failure across every screen at
   once; revisit the sub-12px sizes in the same pass.
4. **Shared busy-button treatment on the eight silent mutations** (M) — the pattern
   already exists four times in the codebase; unify and apply.
5. **Error-copy pass** (M) — fallback strings name action + next step; raw bodies behind a
   "details" disclosure; fix the stale Krea 2 note and the LoRA-skip wording.
6. **Mobile console layout at ≤640** (M) — own the wrap order; move the thumbnail.
7. **Vocabulary pass: session/LoRA everywhere** (S).
8. **SessionForm: four fields up front, dials behind More dials** (M).
9. **Port-default fix in dev tooling** (S).
10. **Token migration** (M/L) — radii/colors/spacing onto the declared tokens; delete or
    use `--r-sheet`.
