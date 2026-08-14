# UI checks

Playwright and HTTP scripts that drive `tools/preview_ui.py` and assert what the
page actually does, rather than what a stylesheet says it should. They live here
because the alternative — a session scratchpad — is where the last several
rounds of them died, and each one encodes a fault that was expensive to find.

    /opt/homebrew/bin/python3.11 tools/preview_ui.py 8791 &
    python3.11 tools/ui-checks/baseline.py check

They assert against `preview_ui.py`'s stubs, so they need no Modal account, no
GPU and no deployment.

## The baseline

`baseline.py` is the one to run. It freezes what the page does into
`baseline/*.json`, committed, and `check` fails when any of it moves.

    baseline.py capture          # re-freeze (commit the diff, with a reason)
    baseline.py check            # exit 1 on drift
    baseline.py check compile    # one probe

This exists for the React port. 4,544 lines of the page are being rewritten, and
almost none of what makes them right is inferable from the code — the console
budget, the separator rule, which pills a silent model drops. Those survive a
rewrite only if something *fails* when they do not, and "someone looks at it" is
not that thing: every one of them looks fine when it is wrong.

A diff is a question, not automatically a bug. The answer belongs in the commit
that moves the baseline.

- `probe_compile.py` — 402 cases across all three compilers. HTTP only, no
  browser, so it runs anywhere. Pins the rules that have no other guard: no
  pills means the typed text byte-for-byte, the separator softening in front of
  a lowercase fragment, `needs` being per item so Wan drops dialogue as well as
  sound, and the four H3 alignment sentences.
- `probe_console.py` — the console against its 30% budget at three viewports.

## Parity checks

Every check here takes a URL now, so the same assertions run against either front
end and the outputs are compared directly:

    python3 tools/ui-checks/check_viewer.py                        # vanilla
    python3 tools/ui-checks/check_viewer.py http://localhost:5173  # React

`check_viewer.py`, `check_render.py` and `check_settings.py` were built this way.
`check_neg.py` and `check_drop.py` were not, and converting them is what found
three of the port's real gaps — so the conversion is worth recording as its own
lesson.

**What made them single-front-end was reaching into the page, not the URL.** They
switched sides with `setKind('video')` and reset state with `refs.length = 0;
drawRefs()`. None of that exists in a bundled front end, so both had to start
driving the way a person does: the kind chip inside the prompt field, the model
select inside the Sampling popover, a second click on a filled tile to clear it,
a chip's own ✕ to remove a reference. That is a better check on *either* front
end, for the reason the drag test already records — a driver poking at internals
is not a user, and the handler it pokes past may be the broken one.

Two smaller things fell out of it:

- **A report that leaves state behind makes the next report measure the wrong
  rule.** Keyframes and references put each other out of play, so dropping a file
  on the reference tray and then testing the keyframe tiles reports them DEAD for
  a feature working exactly as designed. `clear_refs()` and `clear_keyframes()`
  are between the rows for that reason.
- **Assert the meaning, not the mechanism.** The vanilla page always renders
  `#neg-toggle` and adds `.hide`; React does not render it at all. Both mean "this
  model reads no negative", so `toggleHidden` accepts absent as well as hidden —
  asserting on the mechanism would fail one implementation for a structural
  reason, which is the fault named below.

Two things `check_viewer.py` taught:

**Do not hold a reference across a rebuild.** The first version captured `.lb`
once and read the counter through it. The vanilla viewer removes and rebuilds
that element on `transitionend`, so the handle went stale and the check
reported "did not page" for a viewer that had paged — while React, which reuses
the element, passed. A parity check that passes one implementation for
structural reasons is worse than no parity check.

**A driver drag is not a drag.** `left_click_drag` produced no pointer events
on the viewer at all, so a real-input test asserted nothing and passed by doing
nothing. The drag is dispatched as synthetic `PointerEvent`s with real
timestamps instead, which exercises the same handlers. What that cannot cover
is the trackpad-versus-touch asymmetry that broke this before — `<img>` is
natively draggable, so a mouse drag started an HTML image drag and fired
`pointercancel` one frame in, while touch never took that path. The check
asserts `draggable="false"` because that is the fix; only real hardware proves
it still holds.

## The rest

- `check_neg.py` — the negative-prompt toggle and the auto-growing prompt field.
  Covers the models that read no negative, the CFG that wakes it, and the cap
  the field stops growing at. Both front ends answer 32px → 95px → 168px at one,
  four and forty lines.
- `check_drop.py` — every drop target, asserting each one *cancels* dragover.
  A target that does not cancel never receives the drop at all, which is how the
  reference tray shipped dead — and how the React port's video canvas shipped
  without its first-frame drop, which this is what caught.

- `check_regions.py` — the boxes and the card that opens out of them: a drag
  draws one, a release inside the threshold lands on the landmark and the same
  release with Alt does not, a box snaps to another box's edge, clicking one and
  pressing ⌫ deletes it, the card's numbers move with the drag, a render puts the
  boxes away and a file over the window brings them back, and a box drawn at the
  top of a still has its edge on the *picture* rather than 26px above it. The
  last two were broken when it was written; the ⌫ row broke again an hour later,
  when keeping the card mounted through a drag turned out to matter, and that is
  the row that caught it.
- `probe_size.py` — the ratio picker and the pixel boxes as one control: a
  preset writes the boxes, typing selects Custom, the swap transposes the
  *bucket* rather than the pixels, and nothing is snapped while you type. It
  used to fail two rows by design, because ⌘↑ on Width and Height never worked
  on the vanilla page — the handler was delegated from three sections and the
  sizer popover was appended to `<body>`, so the arrows stepped a pair of inputs
  nobody could focus. There is no vanilla page now and the component owns its
  own keys, so all rows pass and a failure here is a real one.
- `probe_lora.py` — what the note says about `<lora:…>` tokens. The stub volume
  holds `Portrait` and `portrait`, and `high` in both Wan speed folders, so the
  case rule and the shortest-unambiguous-name rule are checked against real
  collisions rather than a fixture invented to pass.
- `probe_clause.py` — ⌥← / ⌥→. The invariant is not that the text is unchanged
  — the point is that it changes — it is that the *separator sequence* is: same
  characters, same order, same count.
- `check_train.py` — Train and the dataset editor: every hyperparameter carries
  its name, drafts and saved sets are separated, the captioner offers two menus,
  and a run reports step, epoch, rate, ETA and a falling loss before it ends
  naming its checkpoints. Asserted by shape and terminal state, never by the
  value at an instant — the stub advances per poll rather than per second, so
  two front ends sampled at the same moment are legitimately on different steps.
- `check_meta.py` — the metadata sheet, Reuse and Copy: the only surface that
  shows a prompt at all. Leads with the typed sentence, shows the compiled
  document below it when they differ, and Reuse restores the typed one with its
  pills — restoring a document would compile *that* on the next run. The stub's
  video cards carry the pair, compiled through `_from_app` by the real
  compiler, so the branch runs against the document a run would actually
  produce rather than one transcribed into a fixture.

## Two things learned writing these

**A stale check is worse than no check.** `measure_console.py` drove `#kinds`,
`#toggle-adv` and `#v-toggle-adv`, none of which survived the console redesign.
It still ran, still printed a table, and the table described a page it had
failed to open. `probe_console.py` replaces it and asserts every selector
exists before measuring anything.

**Pin the invariant, not the symptom.** The obvious console assertion is
`height <= 30% of viewport`, and it fails at 1440x900 — the budget leaves the
field ~38px there, `FIELD_FLOOR` clamps it to 52, and the console lands at
31.6%. That is the documented trade, not a regression: below two lines the box
stops being somewhere you can write. Checking the symptom reports the design as
a bug on the shortest viewport anyone uses. What is pinned instead is
`field == max(FLOOR, min(CEIL, innerHeight*0.30 - other))`, which holds
everywhere and is what a React `fieldMax` has to reproduce.

## Against the shipped bundle

Every check takes a URL, and there are two targets left now that UI_HTML is
deleted:

    npm --prefix web run build && python3 tools/preview_ui.py 8791
    npm --prefix web run dev                    # :5173, proxying /api to 8791

The first is the one that matters before a deploy. `preview_ui.py` serves
`web/dist` as static files at absolute `/assets/…` paths, which is exactly how
the web container serves it and is not what the dev server does — a bundle that
works under `npm run dev` and 404s its own stylesheet when mounted is a failure
neither the dev server nor a unit test would show. Remember the build: the
static server reads `web/dist` off disk, so without it you are checking the
previous commit's front end and every row still passes.
