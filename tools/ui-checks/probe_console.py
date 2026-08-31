"""
The field fits its content up to three lines, and the canvas stays dominant.

Replaces the earlier `measure_console.py`, which had gone stale in a way worth
recording: it drove `#kinds`, `#toggle-adv` and `#v-toggle-adv`, none of which
survived the console redesign. A harness that reports on controls it failed to
click is worse than no harness, so this one asserts the selector exists before
it measures anything.

**The 30% budget this file used to defend is retired, on the owner's ruling:**
"That number has caused me more trouble than it's worth. It was never meant to
be exact. The point was that the canvas should always be dominant — the second
it's not, the platform becomes utilitarian. The prompt box should grow and
contract based on content." The budget arithmetic that derived the field's cap
from the viewport — with its floor, its quantiser and its ResizeObserver, each
defending the last — went with it; the history is in `fieldMax.ts`.

Two assertions now, and they are different kinds of claim:

    field == min(content, chrome + 3 * line)     # the formula, exact
    console / viewport < 0.50                    # dominance, a tripwire

The first is the page's contract and fails on any drift. The second is the
actual invariant the number was always standing in for: the canvas is the
largest thing on screen. A console at 50% is not something to squeeze a field
over — it is a layout that needs designing, and the ruling says so — so the
probe surfaces it as a failure rather than absorbing it, per the standing rule
that a broken limit is information: find which side is lying, never clamp the
one that is easier to silence.
"""
import re
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8791"

# Mirrored from the page, and deliberately not imported: if these drift apart
# from `FIELD_LINES` in `fieldMax.ts`, the check should fail rather than
# quietly follow the new numbers. DOMINANCE is the probe's own line in the
# sand — the ruling's "never come close to 50%", asserted at 50 so the failure
# names the moment the canvas stops being the largest thing on screen.
FIELD_LINES, DOMINANCE = 3, 0.50

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


def field_cap(m):
    """
    `cap()` in `console/fieldMax.ts`, mirrored rather than imported.

    Stated in lines, not pixels, which is what dissolved the half-rendered-row
    bug the old quantiser existed for: `chrome + 3 × line` is on a line
    boundary by construction, at every viewport, so there is no arbitrary
    pixel to round and no sliver for the caret to land in.
    """
    line, chrome = m.get("line") or 0, m.get("chrome") or 0
    if line <= 0:
        return 72  # FALLBACK_CAP, same mirror rule
    return chrome + FIELD_LINES * line


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
              // The content's own height — what the field would be with no cap.
              // A clamped textarea still reports it, which is what lets the
              // formula below be asserted from outside the page.
              scrollH: f ? f.scrollHeight : 0,
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

    # The page's whole contract: content, up to three lines of this field's
    # own type. `scrollHeight` of a clamped textarea is still the content's
    # height, which is what makes the formula assertable from outside.
    want = min(m["scrollH"], field_cap(m))
    rows.append({
        "state": label, "console": round(m["console"]), "canvas": round(m["canvas"] or 0),
        "field": round(m["field"]),
        "want_field": round(want), "want_lines": round(want),
        "frac": round(m["console"] / m["vh"], 4),
        "overflowing": bool(m["overflowing"]),
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
            # An equality now, not a cap-plus-proof: `autoGrow` sets exactly
            # min(content, three lines), and `want` is computed from the same
            # two measurements, so any daylight between them is drift. One
            # pixel of slack: fractional padding at some zooms.
            ok = abs(r["field"] - r["want_field"]) <= 1
            note = ""
            if not ok:
                note = "   FIELD vs three-line contract"
                bad.append((vp, r["state"], r["field"], r["want_lines"]))
            elif r["frac"] >= DOMINANCE:
                # Not absorbed and not softened: a console at half the window
                # is the canvas no longer dominant, which the ruling calls a
                # layout problem that needs designing — so it fails, loudly,
                # as one.
                note = "   CANVAS NOT DOMINANT"
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
    print("\nfield fits its content to three lines at every viewport, and the canvas stays dominant")
