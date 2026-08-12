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

## The rest

- `check_neg.py` — the negative-prompt toggle and the auto-growing prompt field.
  Covers the models that read no negative, the CFG that wakes it, and the cap
  the field stops growing at.
- `check_drop.py` — every drop target, asserting each one *cancels* dragover.
  A target that does not cancel never receives the drop at all, which is how the
  reference tray shipped dead.

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
