"""
How tall the console gets, and whether that still fits the 30% budget.

Replaces the earlier `measure_console.py`, which had gone stale in a way worth
recording: it drove `#kinds`, `#toggle-adv` and `#v-toggle-adv`, none of which
survived the console redesign — Advanced became the model/sampling popovers and
the kind switch became a chip in the prompt field. It still ran, still printed a
table, and the numbers were of a page it was no longer opening. A harness that
reports on controls it failed to click is worse than no harness, so this one
asserts the selector exists before it measures anything.

The budget is 30% of the viewport. `fieldMax()` hands the prompt whatever is
left after everything else, so the worst case is not a long prompt on its own —
it is a long prompt with the pill rail wrapped *and* the region bar present,
because those arrive long after the last keystroke. That is the state the
ResizeObserver exists for, and the one that measured 38.1% without it.

**The assertion is the formula, not the 30%**, and the difference is the whole
reason this file is worth having. A bare `frac <= 0.30` fails at 1440x900 —
the budget leaves the field about 38px there, `FIELD_FLOOR` clamps it to 52,
and the console lands at 31.6%. That is not a regression, it is the documented
trade: below two lines the box stops being a place you can write, so the floor
is allowed to win. Checking the symptom would have reported the design as a bug
on the shortest viewport anyone uses. So what is pinned is

    field  == max(FLOOR, min(CEIL, innerHeight * 0.30 - other))

which is true at every viewport, says *why* the console is over when it is over,
and is exactly what a React `fieldMax` has to reproduce.
"""
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8791"

# Mirrored from the page, and deliberately not imported: if these three drift
# apart from `CONSOLE_BUDGET`/`FIELD_FLOOR`/`FIELD_CEIL` in the source, the
# check should fail rather than quietly follow the new numbers.
BUDGET, FIELD_FLOOR, FIELD_CEIL = 0.30, 52, 168

# 14" MBP, 13" MBP, 16" MBP. The 13" is the one that binds — it is the shortest
# viewport the console has to fit inside, so it is where the budget is decided.
VIEWPORTS = [(1512, 982), (1440, 900), (1728, 1117)]

LONG_PROMPT = "\n".join(
    "A dancer mid-turn in a white slip dress, backlit by a low sun" for _ in range(12))


def need(pg, sel):
    """A selector that has gone missing is a failed check, never a skipped one."""
    if pg.query_selector(sel) is None:
        raise AssertionError(f"selector vanished from the page: {sel}")
    return sel


def measure(pg, label, rows):
    m = pg.evaluate("""() => {
      const c = document.querySelector('.console');
      const v = document.querySelector('#canvas');
      // The live field is the negative box when that is the one on show, which
      // is what `liveField()` decides — measuring #prompt regardless would read
      // a hidden element's height on the models that swap it.
      const f = document.querySelector('.field.on-neg')
                  ? document.querySelector('#neg') : document.querySelector('#prompt');
      return {console: c ? c.getBoundingClientRect().height : null,
              canvas:  v ? v.getBoundingClientRect().height : null,
              field:   f ? f.getBoundingClientRect().height : null,
              // scrollHeight > clientHeight means the text wants more room than
              // it was given, which is the state where the cap has to bind.
              overflowing: f ? f.scrollHeight > f.clientHeight + 1 : false,
              vh: innerHeight};
    }""")
    if m["console"] is None or m["field"] is None:
        raise AssertionError(".console or the live field is not on the page")

    other = m["console"] - m["field"]
    want = max(FIELD_FLOOR, min(FIELD_CEIL, m["vh"] * BUDGET - other))
    rows.append({
        "state": label, "console": round(m["console"]), "canvas": round(m["canvas"] or 0),
        "field": round(m["field"]), "other": round(other),
        "want_field": round(want), "frac": round(m["console"] / m["vh"], 4),
        "overflowing": bool(m["overflowing"]),
        # Over budget is only legitimate when the floor is what forced it.
        "floored": round(want) <= FIELD_FLOOR,
    })


def pills(pg, side, n):
    """
    Fill the rail from the served vocabulary rather than a hand-written list.

    The keys are "{group}.{item}" and a wrong one is rejected by name, so a
    copy of the vocabulary here would be a rail of pills the compiler does not
    have — which is the same class of fault this file exists to catch.
    """
    # Through the palette, not by assigning a global. The old page exposed
    # `shot` and `drawShotRail()` on window and this reached straight for them,
    # which is why it kept passing after the port: `capture()` read a
    # module-level URL and ignored argv, so pointing it at the React build
    # measured the old page twice and reported it as agreement.
    pg.click(f"#{'v' if side == 'video' else 'g'}-shot")
    pg.wait_for_timeout(250)
    tiles = pg.locator(".tiles button.tl:not([disabled])")
    got = tiles.count()
    # Loud, because the failure this replaces was silent. A selector that
    # matches nothing measures a console with no rail on it and reports the
    # budget as comfortably met — which is exactly what "+ pill rail" showing
    # the same height as "resting" looks like.
    if got < n:
        raise AssertionError(
            f"wanted {n} shot tiles on the {side} side, found {got} — "
            "the palette markup moved and this probe is measuring nothing")
    for i in range(n):
        tiles.nth(i).click()
    pg.keyboard.press("Escape")


def run(pg, side):
    rows = []
    # The kind chip inside the prompt field, which is the only way a person
    # changes this too.
    if pg.evaluate("() => document.querySelector('#kind-toggle')?.title || ''"
                   ).lower().startswith("video") != (side == "video"):
        pg.click("#kind-toggle")
    pg.wait_for_timeout(400)
    measure(pg, f"{side} · resting", rows)

    if side == "image":
        pg.click("#g-regional")
        pg.wait_for_timeout(450)
        measure(pg, f"{side} · + regions", rows)

    pills(pg, side, 16)
    pg.wait_for_timeout(400)
    measure(pg, f"{side} · + pill rail", rows)

    pg.fill(need(pg, "#prompt"), LONG_PROMPT)
    pg.wait_for_timeout(500)
    measure(pg, f"{side} · + long prompt  << WORST", rows)

    # Back to rest by reloading, not by undoing sixteen clicks. Un-picking the
    # pills one at a time was the obvious version and it hangs: a `.spill` that
    # has scrolled behind the rail's overflow never becomes clickable, and
    # Playwright waits its full timeout on each one — sixteen pills times two
    # sides times three viewports. The page holds no state worth preserving
    # between sides, so a reload is both faster and exact.
    pg.reload(wait_until="networkidle")
    pg.wait_for_timeout(900)
    need(pg, "#prompt")
    return rows


def capture():
    out = {}
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="chrome")
        for vw, vh in VIEWPORTS:
            pg = b.new_page(viewport={"width": vw, "height": vh}, color_scheme="dark")
            errors = []
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.goto(URL, wait_until="networkidle", timeout=60_000)
            pg.wait_for_timeout(1200)
            need(pg, "#prompt")
            need(pg, "#canvas")
            rows = run(pg, "image") + run(pg, "video")
            if errors:
                raise AssertionError(f"page errors at {vw}x{vh}: {errors[:3]}")
            out[f"{vw}x{vh}"] = rows
            pg.close()
        b.close()
    return out


def report(data):
    """Returns the states where the field is not the height the formula wants."""
    bad = []
    for vp, rows in data.items():
        print(f"\n=== {vp} ===")
        print(f"  {'state':32} {'console':>8} {'field':>7} {'want':>6} {'% vp':>7}")
        for r in rows:
            # `fieldMax()` is a cap, not a target: `autoGrow` sets
            # min(scrollHeight, fieldMax()), so a one-line prompt sits at 32px
            # and is correct there. Two things to hold, then — the cap is never
            # exceeded, and it is actually *reached* once the text overflows,
            # which is the only state that proves the cap is wired up at all.
            # One pixel of slack: fractional padding at some zooms.
            ok = r["field"] <= r["want_field"] + 1
            if r["overflowing"]:
                ok = abs(r["field"] - r["want_field"]) <= 1
            note = ""
            if not ok:
                note = "   FIELD vs fieldMax()"
                bad.append((vp, r["state"], r["field"], r["want_field"]))
            elif r["frac"] > BUDGET:
                note = "   over 30% (floor, by design)" if r["floored"] else "   OVER 30%"
                if not r["floored"]:
                    bad.append((vp, r["state"], r["field"], r["want_field"]))
            print(f"  {r['state']:32} {r['console']:>6}px {r['field']:>5}px "
                  f"{r['want_field']:>4}px {r['frac']*100:>6.1f}%{note}")
    return bad


if __name__ == "__main__":
    data = capture()
    bad = report(data)
    if bad:
        print("\nfieldMax() contract broken:", file=sys.stderr)
        for vp, state, got, want in bad:
            print(f"  {vp}  {state}  field={got}px want={want}px", file=sys.stderr)
        sys.exit(1)
    print("\nfield height matches fieldMax() at every viewport and state")
