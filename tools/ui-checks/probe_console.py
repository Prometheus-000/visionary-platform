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
it is a long prompt with the pill rail wrapped and regions armed,
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
import re
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


def whole_lines(m, cap):
    """
    The cap, rounded down to a whole number of lines — `wholeLines` in
    `console/fieldMax.ts`, mirrored here for the reason the three constants
    above are: if the two drift apart, this should fail rather than follow.

    **The rule this replaces was `field == fieldMax()` exactly, and it was the
    right assertion for a cap that was applied raw.** It is not any more, and the
    fault it stopped catching is the reason: every constant in that file is a
    pixel count and a line is 21px on 8px of padding, so `FIELD_FLOOR` came out
    at 2.095 lines. The box permanently showed a 2px sliver of the line below,
    and once the text scrolled the browser moved it by whatever kept the caret
    in view — 14.5px, measured — leaving the caret inside a half-rendered line
    with the row above it sliced through the middle. The check passed
    throughout, because a box of 2.095 lines is exactly `fieldMax()`.

    So the contract is now *the largest whole number of lines that fits the cap*,
    which is strictly stronger: it still pins the cap, and it additionally pins
    that the height lands on a line.
    """
    line, chrome = m.get("line") or 0, m.get("chrome") or 0
    if line <= 0:
        return cap
    return chrome + max(1, int((cap - chrome) // line)) * line


def measure(pg, label, rows):
    m = pg.evaluate("""() => {
      const c = document.querySelector('.console');
      const v = document.querySelector('#canvas');
      // The live field is the negative box when that is the one on show, which
      // is what `liveField()` decides — measuring #prompt regardless would read
      // a hidden element's height on the models that swap it.
      const f = document.querySelector('.field.on-neg')
                  ? document.querySelector('#neg') : document.querySelector('#prompt');
      const cs = f ? getComputedStyle(f) : null;
      return {console: c ? c.getBoundingClientRect().height : null,
              canvas:  v ? v.getBoundingClientRect().height : null,
              field:   f ? f.getBoundingClientRect().height : null,
              // scrollHeight > clientHeight means the text wants more room than
              // it was given, which is the state where the cap has to bind.
              overflowing: f ? f.scrollHeight > f.clientHeight + 1 : false,
              // What a line costs, and what the box spends before the first one.
              // Read off the page rather than written here: the cap is quantised
              // to whole lines and the quantum is a property of the field's own
              // type, which differs between the prompt and a shot row.
              line: cs ? parseFloat(cs.lineHeight) : 0,
              chrome: cs ? parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom)
                           + parseFloat(cs.borderTopWidth)
                           + parseFloat(cs.borderBottomWidth) : 0,
              vh: innerHeight};
    }""")
    if m["console"] is None or m["field"] is None:
        raise AssertionError(".console or the live field is not on the page")

    other = m["console"] - m["field"]
    want = max(FIELD_FLOOR, min(FIELD_CEIL, m["vh"] * BUDGET - other))
    rows.append({
        "state": label, "console": round(m["console"]), "canvas": round(m["canvas"] or 0),
        "field": round(m["field"]), "other": round(other),
        "want_field": round(want), "want_lines": round(whole_lines(m, want)),
        "frac": round(m["console"] / m["vh"], 4),
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
    # The door is in the strip, one per side, sharing one palette. It carries a
    # word now rather than being a bare glyph — see `ShotDoor` — but the id is
    # unchanged, which is why this line did not have to move with it.
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


def to_side(pg, side):
    """
    Duration is the switch now — `Still` is a photograph and anything above it is
    a clip, so there is no image/video chip. Index rather than a label, because
    the seconds a model offers are per model: 0 is always Still and 1 its
    shortest clip.
    """
    want_video = side == "video"
    if pg.eval_on_selector(
        "#c-video", "e => e.classList.contains('hide')"
    ) == want_video:
        pg.click("#g-duration")
        pg.wait_for_selector(".menu button")
        pg.locator(".menu button").nth(1 if want_video else 0).click()
    pg.wait_for_timeout(400)


def run(pg, side):
    rows = []
    to_side(pg, side)
    measure(pg, f"{side} · resting", rows)

    if side == "image":
        # Arming lands boxes on the canvas, not a row in the console — which is the
        # whole point of this measurement: the region UI costs the console nothing,
        # so this row should read the same height as resting.
        #
        # Drawn rather than clicked. The empty canvas used to offer a "split into two
        # columns" button and no longer does, so the box arrives the way one always
        # actually arrives: a drag across the frame.
        pg.evaluate("""() => {
          const lay = document.querySelector('#region-layer');
          const b = lay.getBoundingClientRect();
          const at = (fx, fy) => ({ clientX: b.left + b.width * fx,
                                    clientY: b.top + b.height * fy,
                                    bubbles: true, cancelable: true, pointerId: 1,
                                    pointerType: 'mouse', button: 0, buttons: 1,
                                    isPrimary: true });
          lay.dispatchEvent(new PointerEvent('pointerdown', at(0.08, 0.10)));
          lay.dispatchEvent(new PointerEvent('pointermove', at(0.46, 0.90)));
          lay.dispatchEvent(new PointerEvent('pointerup', { ...at(0.46, 0.90), buttons: 0 }));
        }""")
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


def cuts(pg):
    """
    The gutter and the document have to agree about when a shot starts.

    `times()` normalises the shares against whatever number it is handed, so
    this is one expression in one file being right — and when it was wrong
    nothing failed, the timestamps beside every row were simply not the ones
    that would run. Three 4s shots read `05.33` under an 8s duration menu where
    the document said `At 00:08.000`, with the ruler two pixels away reading 12s
    and correct. A cut time is the only timing control H3 has — it timestamps a
    shot boundary and nothing else — so a readout that quietly lies about one is
    worth its own check rather than a heights row.

    Measured against `/api/compile`'s own answer, never against arithmetic
    repeated here: a check that recomputes the number it is checking agrees with
    itself no matter which of the two is wrong.
    """
    to_side(pg, "video")
    pg.fill(need(pg, "#prompt"), "A dancer turns under a streetlight.")
    for line in ("The camera pushes in on her hands.", "She steps out of the light."):
        pg.click(need(pg, ".tl-add"))
        pg.wait_for_timeout(250)
        pg.fill(need(pg, "#prompt"), line)
    # Pulled to unequal lengths on purpose: equal bars are the one arrangement
    # where a readout dividing by the wrong total can still land on the right
    # numbers, because the error is a ratio.
    grip = pg.locator(".tl-shot .tl-pull").first.bounding_box()
    y = grip["y"] + grip["height"] / 2
    pg.mouse.move(grip["x"] + grip["width"] / 2, y)
    pg.mouse.down()
    pg.mouse.move(grip["x"] + grip["width"] / 2 + 60, y, steps=8)
    pg.mouse.up()
    pg.wait_for_timeout(350)

    pg.click(need(pg, "#shot-peek button"))
    pg.wait_for_timeout(800)
    doc = pg.text_content(need(pg, "#shot-peek pre")) or ""
    stamps = {int(n): int(mm) * 60 + float(ss)
              for n, mm, ss in re.findall(r"\[Shot (\d+)\] At (\d\d):(\d\d\.\d\d\d)", doc)}
    # `[Shot 1]` carries no timestamp — that is the guide's rule and the
    # compiler's — so an empty map means the document never compiled, not that
    # the scene has one shot.
    if not stamps:
        raise AssertionError(f"no cut times in the compiled document: {doc[:300]!r}")

    bars = pg.locator(".tl-shot")
    for i in range(bars.count()):
        bars.nth(i).click()
        pg.wait_for_timeout(150)
        shown = pg.text_content(need(pg, ".tnum em")) or ""
        mins, _, rest = shown.rpartition(":")
        at = (int(mins) * 60 if mins else 0) + float(rest)
        want = stamps.get(i + 1, 0.0)
        if abs(at - want) > 0.01:
            raise AssertionError(
                f"shot {i + 1}: the gutter reads {shown} and the document says "
                f"{want:.3f} — the readout is on a different clock")
    print(f"  cut times agree with the document across {bars.count()} shots")
    pg.reload(wait_until="networkidle")
    pg.wait_for_timeout(900)
    need(pg, "#prompt")


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
            # Once, not per viewport: it is arithmetic over the bars and the
            # answer cannot depend on how wide the window is. Runs on the
            # binding viewport so a failure is reported where the console is
            # tightest anyway.
            if not out:
                cuts(pg)
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
                ok = abs(r["field"] - r["want_lines"]) <= 1
            note = ""
            if not ok:
                note = "   FIELD vs fieldMax()"
                bad.append((vp, r["state"], r["field"], r["want_lines"]))
            elif r["frac"] > BUDGET:
                note = "   over 30% (floor, by design)" if r["floored"] else "   OVER 30%"
                if not r["floored"]:
                    bad.append((vp, r["state"], r["field"], r["want_field"]))
            print(f"  {r['state']:32} {r['console']:>6}px {r['field']:>5}px "
                  f"{r['want_lines']:>4}px {r['frac']*100:>6.1f}%{note}")
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
