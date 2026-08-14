"""
The boxes, and the card that opens out of them. Real pointer events, real keys.

    python3 tools/ui-checks/check_regions.py                        # web/dist
    python3 tools/ui-checks/check_regions.py http://localhost:5173  # dev server

There was no check on this feature at all beyond `check_drop.py`'s two rows,
which is how three of the things asserted below were live and broken: the layer
was 26px taller than the picture it masked over every render, clicking a box and
pressing ⌫ did nothing, and a file dragged over the window could not reveal the
boxes it was about to be dropped on.

Every row pins a *meaning* rather than a class name, because the whole surface
was rebuilt underneath these behaviours and a check written against the old
markup would have had to be rewritten with it — which is the same as not having
had one. Three of them pin a number instead: 0.5 for a snap landing, 0 for the
gap between a box's edge and the pixels it will mask, and `[]` for what an
unarmed page sends. Numbers survive a redesign; selectors are what it changes.

Driven by gestures throughout. There are no page globals to reach for in a
bundled front end, and there should not be: a driver poking past the interface
can pass while the interface is unreachable, which is exactly the state
`#g-drop-scene` was in for the twenty minutes before `check_drop.py` learned to
press the frame button.
"""
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8791"
fails = []

# A 2x2 PNG, so the page gets something it can really decode.
PNG = ("iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFElEQVR4nGP8"
       "z8Dwn4GBgYEJTAAAHAcCAKvHBh4AAAAASUVORK5CYII=")

# A drag on the layer, in the layer's own 0..1 — the same space the boxes and the
# backend use, so the assertions can be written in it and never touch a pixel.
DRAG = """
([x0, y0, x1, y1, alt]) => {
  const layer = document.querySelector('#region-layer');
  if (!layer) return 'MISSING';
  const b = layer.getBoundingClientRect();
  const pt = (fx, fy) => ({clientX: b.left + b.width * fx, clientY: b.top + b.height * fy});
  const opts = (p) => ({bubbles: true, cancelable: true, button: 0, pointerId: 7,
                        isPrimary: true, altKey: alt, ...p});
  // Down on whatever is actually under the point. Dispatching at the layer would
  // skip the hit test — `closest('.rbox')` reads the *target* — so every drag
  // would draw a new box and moving one could never be tested at all.
  const from = pt(x0, y0);
  const el = document.elementFromPoint(from.clientX, from.clientY) || layer;
  el.dispatchEvent(new PointerEvent('pointerdown', opts(from)));
  // Two moves: the handler coalesces to one store write per animation frame, and
  // a single move can land in the same frame as the down.
  layer.dispatchEvent(new PointerEvent('pointermove', opts(pt((x0 + x1) / 2, (y0 + y1) / 2))));
  layer.dispatchEvent(new PointerEvent('pointermove', opts(pt(x1, y1))));
  return new Promise((res) => requestAnimationFrame(() => requestAnimationFrame(() => {
    layer.dispatchEvent(new PointerEvent('pointerup', opts(pt(x1, y1))));
    res(true);
  })));
}
"""

# What the run would actually be sent, read off the boxes rather than off the DOM.
BOXES = """
() => [...document.querySelectorAll('#region-layer .rbox')].map((el) => {
  const b = el.getBoundingClientRect();
  const l = document.querySelector('#region-layer').getBoundingClientRect();
  return {x: (b.left - l.left) / l.width, y: (b.top - l.top) / l.height,
          w: b.width / l.width, h: b.height / l.height};
})
"""

# A real drop on the layer, aimed at a corner no rectangle covers — which is the
# scene, not a character. The target is what the handler branches on.
DROP_ON_CANVAS = """
([b64, fx, fy]) => {
  const layer = document.querySelector('#region-layer');
  const b = layer.getBoundingClientRect();
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  const dt = new DataTransfer();
  dt.items.add(new File([buf], 'x.png', {type: 'image/png'}));
  const x = b.left + b.width * fx, y = b.top + b.height * fy;
  const el = document.elementFromPoint(x, y) || layer;
  const ev = (t) => new DragEvent(t, {bubbles: true, cancelable: true, dataTransfer: dt,
                                      clientX: x, clientY: y});
  el.dispatchEvent(ev('dragover'));
  el.dispatchEvent(ev('drop'));
  return el.closest('.rbox') ? 'a box' : 'bare canvas';
}
"""

FILE_OVER = """
([b64, go]) => {
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  const dt = new DataTransfer();
  dt.items.add(new File([buf], 'x.png', {type: 'image/png'}));
  window.dispatchEvent(new DragEvent(go ? 'dragover' : 'drop', {bubbles: true, dataTransfer: dt}));
  return true;
}
"""


def near(a, b, eps=0.004):
    return abs(a - b) < eps


with sync_playwright() as pw:
    br = pw.chromium.launch(channel="chrome")
    pg = br.new_page(viewport={"width": 1400, "height": 950}, color_scheme="dark")
    pg.goto(URL, wait_until="networkidle", timeout=60_000)
    pg.wait_for_timeout(1200)
    print(f"\n=== {URL} ===")

    def check(label, ok, detail=""):
        if not ok:
            fails.append(label)
        print(f"  {'ok  ' if ok else 'FAIL'} {label:38} {detail}")

    def drag(x0, y0, x1, y1, alt=False):
        pg.evaluate(DRAG, [x0, y0, x1, y1, alt])
        pg.wait_for_timeout(140)

    def boxes():
        return pg.evaluate(BOXES)

    def tap(fx, fy):
        """Select whatever is at this fraction of the layer.

        Not `pg.click('.rbox')`. Playwright refuses to click an element whose
        centre another element covers, and regions are *allowed* to overlap —
        the two seeded columns already sit edge to edge, and the card the first
        one opens sits over the second. Refusing there is correct of Playwright
        and useless here, so the tap goes where a finger would and the page's own
        hit test decides what it landed on, which is the thing under test anyway.
        """
        pg.evaluate(DRAG, [fx, fy, fx, fy, False])
        pg.wait_for_timeout(140)

    def drop_last():
        """Delete the most recently drawn box, so the next drag starts from the
        same state the last one did."""
        b = boxes()
        if not b:
            return
        tap(b[-1]["x"] + b[-1]["w"] / 2, b[-1]["y"] + b[-1]["h"] / 2)
        pg.keyboard.press("Backspace")
        pg.wait_for_timeout(140)

    def clear_boxes():
        """Back to bare canvas, one selection and one ⌫ at a time. Deleting shifts
        every later index down, so this re-reads rather than counting down."""
        for _ in range(12):
            b = boxes()
            if not b:
                return
            tap(b[0]["x"] + b[0]["w"] / 2, b[0]["y"] + b[0]["h"] / 2)
            pg.keyboard.press("Backspace")
            pg.wait_for_timeout(140)

    def payload():
        """What `/api/generate` is actually handed. The only assertion here that
        cannot be made from the DOM, and the one that matters most: the boxes are
        a picture of the request, not the request.

        Intercepted at the network rather than by replacing `window.fetch` with a
        promise that never settles. That was the first version and it wedged the
        app in `running` — the canvas swaps the frame for a progress bar, so
        `#region-layer` stops existing and every row after it failed for a reason
        that had nothing to do with regions. A check that breaks the page it is
        measuring reports on a page nobody has.
        """
        seen = {}

        def grab(route, request):
            seen["body"] = request.post_data_json
            route.fulfill(status=200, content_type="application/json",
                          body='{"ok": true, "job_id": "gen000"}')

        pg.route("**/api/generate", grab)
        pg.click("#go-gen")
        pg.wait_for_timeout(400)
        pg.unroute("**/api/generate")
        # Back to a page with nothing running on it, because the fulfilled job id
        # is a real one in the stub and would land a render mid-check.
        pg.reload(wait_until="networkidle")
        pg.wait_for_timeout(900)
        return seen.get("body") or {}

    print("\nNOTHING ARMED")
    # The rule the whole feature rests on: with no boxes, a run is byte for byte
    # the run it would have been before this existed.
    pg.fill("#prompt", "a rooftop at golden hour")
    body = payload()
    check("no boxes sends regions: []", body.get("regions") == [], str(body.get("regions")))
    check("no boxes sends no scene plate", body.get("scene") is None, str(body.get("scene")))
    check("the prompt still goes as typed",
          body.get("prompt") == "a rooftop at golden hour", repr(body.get("prompt")))

    print("\nDRAWING, MOVING, SNAPPING")
    # Arming is a canvas gesture now, not a console glyph: the empty canvas is the
    # invitation, and its "split into two columns" button plants the same pair the
    # old `#g-regional` did.
    pg.click(".rinvite-b")
    pg.wait_for_timeout(500)
    seeded = boxes()
    check("the split seeds two columns", len(seeded) == 2, f"{len(seeded)} boxes")

    # Clearing them is also the ⌫ assertion. Before the focus fix this loop could
    # not have terminated: `pointerdown` calls `preventDefault` to own the drag,
    # which suppressed the focus the click would have given the box, so the layer's
    # own key handler — which finds its target with `closest('.rbox')` — never saw
    # a box at all and delete did nothing.
    clear_boxes()
    check("a box deletes on ⌫ after a click", boxes() == [], f"{len(boxes())} left")

    # 0.507 is inside SNAP_EPS (0.015) of a half and nowhere near it by eye. The
    # landing is the assertion: halves, thirds and quarters are what make an even
    # split a gesture rather than a menu.
    drag(0.02, 0.10, 0.507, 0.90)
    b = boxes()
    check("a drag draws one box", len(b) == 1, f"{len(b)} boxes")
    check("its edge snaps to the half", b and near(b[0]["x"] + b[0]["w"], 0.5),
          f"right edge {b[0]['x'] + b[0]['w']:.4f}" if b else "")

    # Alt, proved as a pair on one release point rather than as a single drag.
    # 0.6547 is 0.012 from two thirds — inside SNAP_EPS (0.015) and three times
    # the tolerance away from it, so the two outcomes are distinguishable. The
    # first version of this row released at 0.607, which is 0.06 from a third and
    # would not have snapped with or without the modifier: it asserted nothing and
    # passed, which is the failure mode a check is supposed to not have.
    THIRDS = 2 / 3
    drag(0.98, 0.10, 0.6547, 0.90)
    b = boxes()
    check("a release inside the threshold snaps", len(b) == 2 and near(b[1]["x"], THIRDS),
          f"left edge {b[1]['x']:.4f} vs {THIRDS:.4f}" if len(b) > 1 else f"{len(b)} boxes")
    drop_last()

    drag(0.98, 0.10, 0.6547, 0.90, alt=True)
    b = boxes()
    check("Alt suppresses it", len(b) == 2 and near(b[1]["x"], 0.6547),
          f"left edge {b[1]['x']:.4f} vs 0.6547" if len(b) > 1 else f"{len(b)} boxes")

    # And a third box dragged near the first box's edge lands on it. Frame
    # landmarks and box edges are one candidate pool; only the pool proves it.
    edge = b[0]["x"] + b[0]["w"]
    drag(0.02, 0.02, edge + 0.008, 0.06)
    b = boxes()
    check("a box snaps to another box's edge",
          len(b) == 3 and near(b[2]["x"] + b[2]["w"], edge, 0.006),
          f"{b[2]['x'] + b[2]['w']:.4f} vs {edge:.4f}" if len(b) > 2 else f"{len(b)} boxes")

    print("\nTHE CARD")
    b = boxes()
    tap(b[0]["x"] + b[0]["w"] / 2, b[0]["y"] + b[0]["h"] / 2)
    check("selecting a box opens its card", pg.locator("#region-inspector").count() == 1)
    check("the card is inside the layer",
          pg.evaluate("""() => {
            const c = document.querySelector('#region-inspector').getBoundingClientRect();
            const l = document.querySelector('#region-layer').getBoundingClientRect();
            return c.left >= l.left - 1 && c.right <= l.right + 1
                && c.top >= l.top - 1 && c.bottom <= l.bottom + 1;
          }"""))
    # The numbers are a readout that moves, which is the entire reason they
    # survived the row they used to live in.
    before = pg.input_value("#region-inspector .opt.n input >> nth=1")
    pg.evaluate(DRAG, [0.30, 0.50, 0.44, 0.62])
    pg.wait_for_timeout(200)
    after = pg.input_value("#region-inspector .opt.n input >> nth=1")
    check("the card's X follows the drag", before != after, f"{before} -> {after}")

    # Frame scope is a different card, reached by a gesture rather than by
    # knowing which of two things the row in front of you was about.
    pg.click(".rframe-btn")
    pg.wait_for_timeout(200)
    check("the frame button opens the frame's card",
          pg.locator("#frame-inspector").count() == 1
          and pg.locator("#region-inspector").count() == 0)
    check("the plates live there",
          pg.locator("#g-drop-scene").count() == 1 and pg.locator("#g-drop-outfit").count() == 1)
    check("the map went with the row", pg.locator(".rmap").count() == 0)
    check("the row went too", pg.locator("#region-bar").count() == 0)

    # A photo on bare canvas is the scene, and the scene is a different engine.
    # The drop has to say so: the plate becomes visible behind the boxes on its
    # own, but nothing about a rectangle shows that the run now recomposes the
    # whole frame and takes several times as long.
    # One small box in the top-left, so the rest of the frame is demonstrably bare
    # and the note has a live region to count. The first version dropped at a corner
    # the left-hand column happened to cover, and set that character's likeness
    # instead — which is the handler working and the check aiming badly.
    clear_boxes()
    drag(0.05, 0.05, 0.30, 0.30)
    pg.fill("#r-prompt", "a dancer mid-turn")
    pg.wait_for_timeout(200)
    landed = pg.evaluate(DROP_ON_CANVAS, [PNG, 0.75, 0.92])
    pg.wait_for_timeout(400)
    check("the drop reached bare canvas", landed == "bare canvas", landed)
    check("a scene dropped on bare canvas opens the frame's card",
          pg.locator("#frame-inspector").count() == 1)
    note = pg.locator("#region-note").inner_text() if pg.locator("#region-note").count() else ""
    check("and the card says the engine changed", "composed into the reference" in note,
          note[:64] or "(no note)")

    print("\nOVER A RENDER")
    # Clear the plate first — a second click on a filled tile is how DropTile
    # clears, and it has to go because a plate legitimately wins over the still:
    # what you are composing *into* is the plate, so the boxes stay on the frame
    # and there is no layer over a render to measure. Leaving it attached made the
    # geometry row below report `None`, which is the app being right and the check
    # being out of order.
    pg.click("#g-drop-scene")
    pg.wait_for_timeout(300)
    b = boxes()
    tap(b[0]["x"] + b[0]["w"] / 2, b[0]["y"] + b[0]["h"] / 2)
    pg.fill("#r-prompt", "a dancer mid-turn")
    # However many the rows above left behind. Hardcoding a count here couples this
    # section to the drawing section's arithmetic, so a box added there fails a row
    # about something else entirely.
    armed = len(boxes())
    pg.click("#go-gen")
    pg.wait_for_selector("#gen-out img", timeout=30_000)
    pg.wait_for_timeout(1200)

    # The layer's class *is* the mode now, and `.show` is gone with the reveal chip.
    # Asserted on `painted` rather than on the class alone, because the thing that
    # matters is whether anything is drawn on the render: a mode name is a promise and
    # a computed box-shadow is what the eye gets. Two of these rows used to test
    # `.show`, which stopped existing — and a selector that matches nothing reports
    # "away" for boxes that are sitting on the picture, so they passed by being wrong.
    mode = "() => document.querySelector('#region-layer')?.className ?? '(none)'"
    painted = """() => [...document.querySelectorAll('#region-layer .rbox')]
        .filter((b) => getComputedStyle(b).boxShadow !== 'none').length"""
    check("a render puts the boxes away", pg.evaluate(painted) == 0,
          f"{pg.evaluate(painted)} painted, mode {pg.evaluate(mode)!r}")
    check("and the regions are still armed under it", len(boxes()) == armed,
          f"{len(boxes())} of {armed} boxes")

    # The reveal the port lost. Without it the layer never enters geometry through the
    # drag, so the drop cannot be aimed at a box — which deletes the only moment anyone
    # discovers that a box takes a photograph.
    pg.evaluate(FILE_OVER, [PNG, True])
    pg.wait_for_timeout(200)
    check("a file over the window brings them back",
          pg.evaluate(mode) == "geometry", pg.evaluate(mode))
    check("and they can be dropped on", pg.evaluate(painted) == armed,
          f"{pg.evaluate(painted)} of {armed} painted")
    pg.evaluate(FILE_OVER, [PNG, False])
    pg.wait_for_timeout(400)
    check("and go away again after the drop", pg.evaluate(painted) == 0,
          f"{pg.evaluate(painted)} painted, mode {pg.evaluate(mode)!r}")

    # ⌘ is the gate, and it gates *only* the mode. Inside geometry the same modifier
    # means "a new box, here", so a press that did both would answer "show me the
    # boxes" by adding one to the eight it just showed you.
    pg.click("#region-layer", modifiers=["Meta"], position={"x": 12, "y": 12})
    pg.wait_for_timeout(250)
    check("⌘ over a render asks for geometry", pg.evaluate(mode) == "geometry",
          pg.evaluate(mode))
    check("and draws no extra box doing it", len(boxes()) == armed,
          f"{len(boxes())} of {armed} boxes")

    # A plain press is the frequent act: touch a performer, get their sentence — with
    # no rectangles and no coordinates over the render.
    pg.keyboard.press("Escape")
    pg.wait_for_timeout(200)
    b = boxes()
    tap(b[0]["x"] + b[0]["w"] / 2, b[0]["y"] + b[0]["h"] / 2)
    pg.wait_for_timeout(250)
    check("a plain touch opens that region's card", pg.evaluate(mode) == "content",
          pg.evaluate(mode))
    check("with its sentence ready to type",
          pg.evaluate("() => document.activeElement?.id") == "r-prompt",
          pg.evaluate("() => document.activeElement?.id"))
    check("and the coordinates stay out of it",
          pg.locator("#region-inspector .nums").count() == 0)

    # The geometry defect, pinned as a number. `.shot` reserves padding under the
    # picture for its two buttons, and an absolutely-positioned child resolves
    # `inset` against the padding box — so every box drawn over a render used to
    # be 26px taller than the pixels it was masking. Nothing on screen shows a box
    # and its mask disagreeing, which is why this needs a number rather than an eye.
    # Geometry over a render is asked for with ⌘ — there is no chip and no glyph, and
    # a plain click would open the region's card instead of drawing anything.
    pg.click("#region-layer", modifiers=["Meta"])
    pg.wait_for_timeout(300)
    gap = pg.evaluate("""() => {
      const img = document.querySelector('.shot img');
      const layer = document.querySelector('.shot > #region-layer');
      if (!img || !layer) return null;
      const i = img.getBoundingClientRect(), l = layer.getBoundingClientRect();
      return {top: Math.round(l.top - i.top), bottom: Math.round(l.bottom - i.bottom)};
    }""")
    check("the layer is the picture, not the padded box",
          gap == {"top": 0, "bottom": 0}, str(gap))

    br.close()

print("\n" + ("FAILED: " + ", ".join(fails) if fails
              else "the boxes are the list, and the card is the box"))
sys.exit(1 if fails else 0)
