# UX Audit — Visionary

*2026-08-20 · live + static — stub API (`tools/preview_ui.py`) + vite dev at :5173, static
over `web/src` (~11.6k lines). GPU jobs stubbed, so output quality and real latency were
out of scope.*

> **Status: 20 of 21 findings fixed** (2026-08-20). F7 was declined and left in place;
> G7's spacing clause was deliberately left undone. Typecheck and production build both
> pass. See [What changed](#what-changed) at the foot.

## Macro read

Visionary's expensive paths are carefully designed — generate, download and caption all
show real progress, and destructive dialogs state their exact blast radius. The polish
stops at the edges: a one-line CSS bug is erasing card backgrounds app-wide, eight smaller
mutations give no feedback at all, and secondary text sits under the contrast floor on
every ground it uses. The product's own philosophy is the right yardstick; these are the
places the implementation hasn't reached it yet.

## Flows

| Flow | Steps | Notes |
|---|---|---|
| Generate a still | 3 | No dead ends; strong wait state |
| Still → video | 1 | Controls swap correctly; the still vanishes (F1) |
| Enhance a fragment | 1 | Silent when it fails (F2) |
| Train a LoRA | ~5 | Guided by a self-explaining disabled button |
| Prepare a dataset | 3–4 | Good filters; two disagreeing tallies (F8) |
| Download weights | 2–3 | Honest cost copy; silent mutations (G3) |

No flow detours into an adjacent workflow to finish — cross-links hand results forward.

## How to read this

Two tiers, **Global** then **By feature**, each sorted P1 → P2 → P3.

| Priority | Meaning | Count |
|---|---|---|
| **P1** | Blocks or misleads the user | 1 |
| **P2** | Visible friction | 11 |
| **P3** | Hygiene | 9 |

*Excluded: `web/storyline.html` and `src/storyline/` are a dev-only sandbox, absent from
the vite build input and from `dist/`, so no finding is filed against them.*

---

# Global (platform)

### G1 · P1 — Card backgrounds render transparent app-wide

- **Evidence:** `ui.css:49` sets `--wash-1:var(--wash-1)`, a self-reference that computes
  to nothing, stripping the ground from all nine consumers (`.card` 521, `.shot` 648,
  `.gal` 711, `.sess` 1230, `.ds-card` 1341, `.tile` 1378, `.dupes-tableWrap` 1450, 1652).
- **Fix:** set `--wash-1: rgba(255,255,255,.03)`, matching the ramp under `--wash-2:.05`.
- **Confirmed live:** `.gal` computes `background-color: rgba(0,0,0,0)`.

### G2 · P2 — Secondary text fails contrast on every background it uses

- **Evidence:** `--dim:#5a5a5a` (ui.css:47) computes to 3.04:1 on `#000`, 2.82:1 on the
  `--wash-2` field ground and 2.74:1 on `#111` menus, all under the 4.5:1 AA floor.
- **Scope:** it carries real reading text at 11–14px (`.muted` 603, `.sub` 520,
  `.foot .when` 725, menu hints 758, control labels 293).
- **Fix:** raise `--dim` to `#767676` (4.54:1) and reserve `#5a5a5a` for disabled states.

### G3 · P2 — Eight mutations show nothing between click and reply

- **Evidence:** save token (`Settings.tsx:52`), delete LoRA (:59), remove caption model
  (:311), delete set (`useDatasets.ts:166`), save set (`Editor.tsx:218`), prepend trigger
  (`Editor.tsx:564`), delete session (`useSessions.ts:101`), gallery delete/purge
  (`Gallery.tsx:171`).
- **Fix:** apply one shared busy-button treatment across all eight.
- **Note:** the codebase already has the pattern four ways (`dl.busy`, `s.rewriting`,
  `saving`, `deleting`).

### G4 · P2 — Error copy dead-ends, and raw tracebacks reach the screen

- **Evidence:** `'Generation failed'` (`useGenerate.ts:189`), `'Download failed'`
  (`useDownload.ts:69`) and `'Could not read that file.'` (`App.tsx:361`) offer no next
  step.
- **Evidence:** `api/client.ts:37` pipes 400 chars of raw response body into every
  err-box and `alert()`.
- **Fix:** give each fallback a next step and move the raw body behind a details toggle.

### G5 · P2 — Touch targets sit far below the 44px minimum

- **Evidence:** `.mk-reroll` is 16×16px (ui.css:441), `.ref button.x` 19×19 (1612),
  `.shot .acts button` 20×20 (689), `.foot .more` 20×20 (2038).
- **Fix:** extend the existing `@media (hover:none)` bump — today it lifts only
  `.shot-back` to 44px (1209) — into a blanket touch rule.

### G6 · P2 — `npm run dev` starts against a dead API

- **Evidence:** vite proxies `/api` to `:8791` (`vite.config.ts:14`) while the stub binds
  `:8777` (`tools/preview_ui.py:56`).
- **Fix:** align the two defaults.

### G7 · P3 — The declared token system is bypassed more than used

- **Evidence:** 96 border-radius declarations span ~24 values with 21% tokenized, and
  `--r-sheet` is defined but never used (hardcoded at 1056, 1536).
- **Evidence:** colors span 100 literals including 26 white-alphas; spacing spans 33
  values with no token at all.
- **Fix:** migrate radii and semantic colors onto the existing tokens, and collapse the
  5–9px band (142 uses) onto a 4/8 step.

### G8 · P3 — 26 font sizes fall below 12px

- **Evidence:** shipped CSS carries 9px ×1 (ui.css:1409), 10px ×10 and 11px ×15,
  including the `--label-size` token itself (ui.css:83).
- **Fix:** floor at 11px and pair anything under 12px with a ≥4.5:1 color.

### G9 · P3 — Two nouns each for "session" and "LoRA"

- **Evidence:** the board says "sessions" (`Train.tsx:153`) where the header says "runs"
  (`App.tsx:563`), and one dialog says "checkpoints" (`SessionCard.tsx:117`) where
  Settings says "LoRAs" (`Settings.tsx:172`).
- **Fix:** standardise on "session" for the record and "LoRA" for the artifact.

---

# By feature

## Generate & console

### F1 · P2 — Switching Still → video discards the canvas result

- **Evidence:** observed live — choosing "8s" after a generation replaced the image with
  an empty video placeholder, leaving no trace or pointer to it.
- **Fix:** carry the last still into the existing First-frame slot instead of clearing it.

### F2 · P2 — A failed Enhance looks identical to one that changed nothing

- **Evidence:** observed live — `/api/rewrite` returned 200 with no `text`, and
  `Rewrite.tsx:35-40` applies a result only when `r.ok && r.text`.
- **Fix:** show a transient "Couldn't rewrite — try again" on empty or failed replies.

### F3 · P2 — Enhance names the wrong model in video mode

- **Evidence:** observed live — the tooltip reads "the way Krea 2 reads best" while
  MiniMax-H3 is active, because the note renders unconditionally (`Rewrite.tsx:52`).
- **Fix:** serve a mode-aware note, or drop the model name from the copy.

### F4 · P2 — The mobile console wrap is undesigned

- **Evidence:** observed at 375px — ten controls wrap into three ragged rows with "Still"
  orphaned and the thumbnail stranded between the model name and Generate.
- **Fix:** define an explicit ≤640 row order and move the thumbnail into the gallery door.
- **Note:** the only current accommodation is `gap:6px` (ui.css:315).

### F5 · P3 — The LoRA skip warning diagnoses without advising

- **Evidence:** the canvas prints "not applied: gone (no matching keys)"
  (`Canvas.tsx:282`), which parses as gibberish when the LoRA name is an English word.
- **Fix:** quote the name and give the cause — `not applied: "gone" — trained for a
  different base model`.

### F6 · P3 — Canvas results reserve no space before they land

- **Evidence:** `.shot img` (ui.css:656) and `#vid-out video` (696) set no
  `aspect-ratio`, so a finished render reflows the canvas column.
- **Fix:** set `aspect-ratio` from the requested dimensions, as `.gal .media` (713) does.

## Sets (dataset editor)

### F7 · P3 — The trigger-word save ignores its reply

- **Evidence:** `useDatasets.ts:184-187` awaits `setDatasetMeta` without checking
  `failed(r)`, so a rejected save leaves the typed word on screen.
- **Fix:** check the reply and surface the existing err-box, as every sibling call does.
- **Status:** left as-is by request — training without a trigger word is a supported
  choice, so this is an unhandled reply rather than the P1 data-loss risk first filed.

### F8 · P2 — Two disagreeing tallies sit side by side

- **Evidence:** observed live — "12 images · 5 captioned" (client count,
  `Editor.tsx:178`) renders beside "19/24 captioned" (server insight, :279).
- **Fix:** derive both from one source, or timestamp the insight so the gap reads as
  staleness.

### F9 · P3 — The density control is unlabeled

- **Evidence:** observed live — an icon-only −/+ pair sits between the filters and
  tallies with no title or aria-label (`Editor.tsx:270-274`).
- **Fix:** give it the accessible name "Smaller/larger tiles".

## Training

### F10 · P2 — The training form mirrors the backend schema

- **Evidence:** 18 always-visible controls render from the `DIALS` map
  (`SessionForm.tsx:32-42`), spread verbatim from `train_defaults`
  (`useSessions.ts:112-119`).
- **Fix:** surface name, trigger, set and epochs, and move the rest behind the existing
  More dials.

## Gallery

### F11 · P3 — Download and Delete appear twice per card

- **Evidence:** both render as hover quick-actions (`Card.tsx:85`) and again in the same
  card's More menu (`Gallery.tsx:212-223`).
- **Fix:** keep them in the menu and drop the hover pair.

## Settings

### F12 · P3 — Settings inputs can't shrink

- **Evidence:** two inputs hardcode `width:158, flex:'none'` (`Settings.tsx:138`, `:340`)
  in a sheet with no media query.
- **Fix:** change to `flex: 1 1 158px` with a `min-width`.

---

## What's already right

- Wait states on the paths that cost money are exemplary — Generate flips to Stop with a
  live step counter (`useGenerate.ts:169`).
- The duration switch is real progressive disclosure, swapping model, copy and source row
  and dropping +LoRA when the model can't take one.
- Destructive dialogs state the blast radius including exclusions — "The N older than
  these are not included." (`Gallery.tsx:187`).
- The disabled Start button explains itself stepwise, from "Name the LoRA" to "Set a
  trigger word".
- Menus use task verbs — "Reuse prompt & settings", "Animate from this frame" — not CRUD.
- State copy is honest about cost and persistence — "UNSAVED · cleared when you close the
  app".
- Actionable errors already exist as a house style — "References need MiniMax-H3 —
  download it under Settings." (`App.tsx:371`).
- The IA is task-shaped, funnelling 33 API routes into four surfaces (`App.tsx:33`).

## What changed

| ID | What was done | Where |
|---|---|---|
| G1 | `--wash-1` given a real value, restoring the card ground on nine surfaces | `ui.css:49` |
| G2 | `--dim` raised to `#767676` — 4.54:1 on `--bg` | `ui.css:45` |
| G3 | New shared `useBusy` hook applied to all eight mutations | `ui/useBusy.ts` + 8 call sites |
| G4 | `ApiError` split into a sentence plus `detail`; new `ErrorNote` folds the traceback into a disclosure | `api/client.ts`, `ui/ErrorNote.tsx` |
| G5 | Touch block moved to the foot of the file so it wins, and extended to every glyph button | `ui.css` (end) |
| G6 | Stub default moved to 8791 to match vite and every check script | `preview_ui.py:56` |
| G7 | Radii 24 values → 6 tokens (93% tokenized); semantic hex → `--crit`/`--ok`/`--warn`; `--r-pill`, `--r-mark`, `--crit-line`, `--crit-fill` added | `ui.css` |
| G8 | Every sub-11px size raised; nothing below 11px remains | `ui.css` (11 sites) |
| G9 | "session" for the record, "LoRA" for the artifact, app-wide | `App.tsx`, `train/*`, `README.md` |
| F1 | The last still now lands in the video First-frame slot instead of vanishing | `App.tsx:402-464` |
| F2 | A failed or empty Enhance says so in the reserved note row | `Rewrite.tsx`, `Console.tsx` |
| F3 | Model name removed from the served copy at its source | `app.py:7293` |
| F4 | `≤640` flattens the strip into one wrap flow; run row on its own line; thumbnail dropped | `ui.css:319-360` |
| F5 | Skip warning quotes the LoRA and gives the likely cause | `useGenerate.ts:52-77` |
| F6 | `width`/`height` on the canvas `<img>` reserve the box before it lands | `useGenerate.ts`, `Canvas.tsx:215` |
| F7 | **Declined** — left exactly as it was | — |
| F8 | The contradicting server tally replaced by trigger coverage; one source per fact | `Editor.tsx:200-218` |
| F9 | Density control given a visible "Size" label and accessible names | `Editor.tsx:325-345` |
| F10 | Form opens on four fields; nine dials moved behind the existing More dials | `SessionForm.tsx` |
| F11 | Hover Download/Delete removed; menu keeps them and now survives `≤1024` | `Card.tsx`, `ui.css:2062` |
| F12 | Settings inputs given a flexible basis so the row reflows | `Settings.tsx:165, 400` |

Two things found while fixing, both handled: `#gal-grid .foot{display:none}` at `≤1024`
would have made Download and Delete unreachable on touch once F11 removed the hover pair,
and `tools/ui-checks/check_train.py` asserted the old form layout and was updated to
assert the new one instead.

### Deliberately not done

- **G7's spacing clause.** Collapsing the 5–9px band onto a 4/8 step means changing 71
  declarations (and 25 distinct values across 261 uses) that this stylesheet's own
  comments describe as pixel-tuned. That is a design decision with a visible result on
  every screen, not a mechanical migration, so it is left for a deliberate pass.
- **F7.** Training without a trigger word is a supported choice, so the unchecked reply
  is an unhandled error path rather than the data-loss risk first filed.
