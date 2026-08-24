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
([x0, y0, x1, y1, alt, meta]) => {
  const layer = document.querySelector('#region-layer');
  if (!layer) return 'MISSING';
  const b = layer.getBoundingClientRect();
  const pt = (fx, fy) => ({clientX: b.left + b.width * fx, clientY: b.top + b.height * fy});
  const opts = (p) => ({bubbles: true, cancelable: true, button: 0, pointerId: 7,
                        isPrimary: true, altKey: alt, metaKey: meta, ...p});
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

# The same gesture, stopped in the middle of itself.
#
# The card goes to opacity 0 for the length of a drag — a card over the rectangle
# you are dragging is a card in the way of the thing it describes — and for a
# while it took the four numbers with it, into the one moment they are worth
# reading. Nothing dispatched after `pointerup` can see that: the row below it
# passed throughout, because the card is back by the time it looks. So this one
# reads while the pointer is still down and lets go afterwards.
DRAG_MID = """
([x0, y0, x1, y1]) => {
  const layer = document.querySelector('#region-layer');
  if (!layer) return 'MISSING';
  const b = layer.getBoundingClientRect();
  const pt = (fx, fy) => ({clientX: b.left + b.width * fx, clientY: b.top + b.height * fy});
  const opts = (p) => ({bubbles: true, cancelable: true, button: 0, pointerId: 7,
                        isPrimary: true, ...p});
  const from = pt(x0, y0);
  const el = document.elementFromPoint(from.clientX, from.clientY) || layer;
  el.dispatchEvent(new PointerEvent('pointerdown', opts(from)));
  layer.dispatchEvent(new PointerEvent('pointermove', opts(pt((x0 + x1) / 2, (y0 + y1) / 2))));
  layer.dispatchEvent(new PointerEvent('pointermove', opts(pt(x1, y1))));
  return new Promise((res) => requestAnimationFrame(() => requestAnimationFrame(() => {
    const r = document.querySelector('#region-readout');
    const box = document.querySelector('#region-layer .rbox.sel');
    const out = {
      // Painted, not merely mounted: the fault being pinned here was a live
      // element at zero opacity, which every `count() === 1` in this file would
      // have called present.
      shown: !!r && getComputedStyle(r).opacity !== '0'
             && r.getBoundingClientRect().width > 0,
      // The children, not every span under it: each number carries its own label
      // in a nested one, and `querySelectorAll('span')` reads eight things where
      // there are four.
      nums: r ? [...r.children].map(
        (s) => parseFloat(s.textContent.replace(/[A-Z]/, ''))).filter((n) => !isNaN(n)) : [],
      x: box ? (box.getBoundingClientRect().left - b.left) / b.width : -1,
    };
    layer.dispatchEvent(new PointerEvent('pointerup', opts(pt(x1, y1))));
    res(out);
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


# Which box the layer thinks the pointer is over, without pressing anything. The
# hairline this lights is the only thing standing in for a box that is deliberately
# not drawn, so it has to name the box a click would open — and it cannot be read
# off `:hover`, which follows paint order and is the thing under test.
HOVER = """
([fx, fy]) => {
  const layer = document.querySelector('#region-layer');
  const b = layer.getBoundingClientRect();
  layer.dispatchEvent(new PointerEvent('pointermove', {
    bubbles: true, pointerId: 7, pointerType: 'mouse', isPrimary: true,
    clientX: b.left + b.width * fx, clientY: b.top + b.height * fy}));
  return true;
}
"""

# Read on its own, a paint later. The first version returned the class from inside
# the dispatch and failed every row while reporting the right answer in its own
# detail column — the detail called it a second time, by which point React had
# committed. A synchronous read of a state-driven class is a check that measures the
# previous gesture.
HOT = """
() => {
  const el = document.querySelector('#region-layer .rbox.under');
  return el ? Number(el.dataset.i) : -1;
}
"""

SELECTED = """
() => {
  const el = document.querySelector('#region-layer .rbox.sel');
  return el ? Number(el.dataset.i) : -1;
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

    def drag(x0, y0, x1, y1, alt=False, meta=False):
        pg.evaluate(DRAG, [x0, y0, x1, y1, alt, meta])
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

    def pick_lora(n):
        """Press row `n` of the open LoRA menu with a real pointer, over the canvas.

        Not `el.click()`, and not Playwright's own `.click()` on the locator either.
        A scripted click dispatches a bare `click` with no pointerdown in front of
        it, and the fault this exists for lives entirely in the pointerdown — the
        menu is a portal to <body> and a React child of the card, so React bubbled
        its press to `#region-layer`, which drew a rectangle, `preventDefault`ed
        the pointerdown and captured the pointer, leaving the row's own handler
        unreached. Every scripted press passed against that.

        Aimed at least 30px inside the layer for the same reason: the menu hangs
        off the left edge of the canvas, and a press on the part that overhangs
        misses the layer and works whatever is wrong.
        """
        row = pg.evaluate("(n) => document.querySelectorAll('.menu button')[n]"
                          ".getBoundingClientRect().toJSON()", n)
        lay = pg.eval_on_selector("#region-layer", "e=>e.getBoundingClientRect().toJSON()")
        x = max(row["x"] + 12, lay["x"] + 30)
        y = row["y"] + row["height"] / 2
        pg.mouse.move(x, y, steps=4)
        pg.mouse.down()
        pg.wait_for_timeout(40)
        pg.mouse.up()
        pg.wait_for_timeout(250)

    def dblclick(fx, fy):
        """The second gesture that makes a region, with a real mouse — the double
        click has to survive the two presses in front of it, and a dispatched
        `dblclick` would not prove that."""
        lay = pg.eval_on_selector("#region-layer", "e=>e.getBoundingClientRect().toJSON()")
        pg.mouse.dblclick(lay["x"] + lay["width"] * fx, lay["y"] + lay["height"] * fy)
        pg.wait_for_timeout(180)

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

    def selected():
        return pg.evaluate(SELECTED)

    def hover(fx, fy):
        pg.evaluate(HOVER, [fx, fy])
        pg.wait_for_timeout(120)
        return pg.evaluate(HOT)

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

    print("\nNO LAYERS")
    # The canvas has no z-order, so the boxes must not have one — and they had the
    # worst kind, the kind nobody chose: absolutely-positioned siblings in `regions`
    # order, so whichever was drawn *last* took every click, hover and drop. A
    # performer inside a wide background box could be reached and the background box
    # could not.
    #
    # Drawn in the order that fails. The small box first, then a big one over the top
    # of it, so the DOM's answer and the right answer are different boxes: paint order
    # says 1 and the rule says 0. A check that drew them the other way round would pass
    # against the bug.
    drag(0.40, 0.40, 0.60, 0.60, meta=True)
    drag(0.05, 0.05, 0.95, 0.95, meta=True)
    check("a box drawn over another still leaves two", len(boxes()) == 2,
          f"{len(boxes())} boxes")
    tap(0.50, 0.50)
    check("the smaller box takes the click", selected() == 0, f"box {selected()}")
    tap(0.10, 0.10)
    check("and the larger one takes it where it is alone", selected() == 1,
          f"box {selected()}")
    # Hover has to agree with the click or the hairline is a lie, and it is the only
    # thing naming a box that is not drawn.
    over = hover(0.50, 0.50)
    check("hover names the box the click would open", over == 0, f"box {over}")
    over = hover(0.10, 0.10)
    check("and follows the pointer back out", over == 1, f"box {over}")

    # Everything above is about *bodies*. A handle is the other half, and it is where
    # the layers survived longest. A handle was allowed to win only where it was
    # already lit; lighting is `sel` or `under`; `under` is decided by the bodies — so
    # the box lying underneath was never the one a hover named, its eight dots never
    # lit, and a press landing exactly on one of them fell through and *moved the box
    # on top*. A wide background box with a performer sitting on its right edge could
    # not be widened at all.
    #
    # ⌘ to draw both, because the rows above leave no bare canvas to start a drag on.
    drag(0.06, 0.28, 0.74, 0.70, meta=True)
    wide = len(boxes()) - 1
    drag(0.67, 0.40, 0.83, 0.60, meta=True)
    small = len(boxes()) - 1
    check("two more boxes to work with", len(boxes()) == small + 1 and wide != small,
          f"{len(boxes())} boxes")
    was = boxes()[wide]
    sat = boxes()[small]
    # Its east handle, at the middle of the very edge the small box is sitting on.
    hx, hy = was["x"] + was["w"], was["y"] + was["h"] / 2
    check("the small box really is covering that handle",
          sat["x"] < hx < sat["x"] + sat["w"] and sat["y"] < hy < sat["y"] + sat["h"],
          f"handle {hx:.3f},{hy:.3f} in {sat['x']:.3f},{sat['y']:.3f}"
          f"+{sat['w']:.3f}x{sat['h']:.3f}")
    drag(hx, hy, hx + 0.09, hy)
    now = boxes()[wide]
    check("a covered handle still takes the press", now["w"] > was["w"] + 0.04,
          f"width {was['w']:.3f} -> {now['w']:.3f}")
    check("and the box on top is not what moved", near(boxes()[small]["x"], sat["x"], 0.02),
          f"{boxes()[small]['x']:.3f} vs {sat['x']:.3f}")
    # A drop is not a gesture you can take back, so the caption and the landing have
    # to be the same box. Both come off the same hit test now.
    # A drop is not a gesture you can take back, so the caption and the landing have to
    # be the same box. The helper reports what `elementFromPoint` says, which is the
    # DOM's answer and the wrong one — the assertion is which box ends up holding a
    # likeness.
    pg.evaluate(DROP_ON_CANVAS, [PNG, 0.50, 0.50])
    pg.wait_for_timeout(400)
    faced = pg.evaluate("""() => [...document.querySelectorAll('#region-layer .rbox')]
        .filter((b) => b.querySelector('.face')).map((b) => Number(b.dataset.i))""")
    check("a photo lands on the box hover named", faced == [0], str(faced))
    clear_boxes()

    print("\nDRAWING, MOVING, SNAPPING")
    # Two boxes, drawn — which is the gesture that was always underneath the thing
    # this used to click. The empty canvas carried a "split into two columns" button
    # and no longer does: an empty state cannot demonstrate a spatial feature, so
    # what it advertised was regions at their least legible, permanently, on the
    # largest surface in the app. Drawing one is the arm now, and dropping a LoRA on
    # bare frame is the other.
    #
    # ⌘ throughout this section, because a plain drag on bare canvas no longer draws.
    # It could not stay: a card is dismissed by clicking outside it, most of the
    # canvas is outside every box, and a canvas where putting the card away leaves a
    # rectangle behind is a canvas you cannot put the card away on. The row below
    # pins the plain drag doing nothing, so the two halves cannot drift apart.
    drag(0.02, 0.05, 0.48, 0.95, meta=True)
    drag(0.52, 0.05, 0.98, 0.95, meta=True)
    seeded = boxes()
    check("two boxes drawn on the frame", len(seeded) == 2, f"{len(seeded)} boxes")

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
    # A plain one first, on the same bare canvas, so "⌘ draws" is measured against
    # "and nothing else does" rather than asserted alone.
    drag(0.02, 0.10, 0.507, 0.90)
    check("a plain drag on bare canvas draws nothing", boxes() == [],
          f"{len(boxes())} boxes")
    drag(0.02, 0.10, 0.507, 0.90, meta=True)
    b = boxes()
    check("⌘ draws one box", len(b) == 1, f"{len(b)} boxes")
    check("its edge snaps to the half", b and near(b[0]["x"] + b[0]["w"], 0.5),
          f"right edge {b[0]['x'] + b[0]['w']:.4f}" if b else "")

    # Alt, proved as a pair on one release point rather than as a single drag.
    # 0.6547 is 0.012 from two thirds — inside SNAP_EPS (0.015) and three times
    # the tolerance away from it, so the two outcomes are distinguishable. The
    # first version of this row released at 0.607, which is 0.06 from a third and
    # would not have snapped with or without the modifier: it asserted nothing and
    # passed, which is the failure mode a check is supposed to not have.
    THIRDS = 2 / 3
    drag(0.98, 0.10, 0.6547, 0.90, meta=True)
    b = boxes()
    check("a release inside the threshold snaps", len(b) == 2 and near(b[1]["x"], THIRDS),
          f"left edge {b[1]['x']:.4f} vs {THIRDS:.4f}" if len(b) > 1 else f"{len(b)} boxes")
    drop_last()

    drag(0.98, 0.10, 0.6547, 0.90, alt=True, meta=True)
    b = boxes()
    check("Alt suppresses it", len(b) == 2 and near(b[1]["x"], 0.6547),
          f"left edge {b[1]['x']:.4f} vs 0.6547" if len(b) > 1 else f"{len(b)} boxes")

    # And a third box dragged near the first box's edge lands on it. Frame
    # landmarks and box edges are one candidate pool; only the pool proves it.
    edge = b[0]["x"] + b[0]["w"]
    drag(0.02, 0.02, edge + 0.008, 0.06, meta=True)
    b = boxes()
    check("a box snaps to another box's edge",
          len(b) == 3 and near(b[2]["x"] + b[2]["w"], edge, 0.006),
          f"{b[2]['x'] + b[2]['w']:.4f} vs {edge:.4f}" if len(b) > 2 else f"{len(b)} boxes")

    print("\nTHE CARD")
    # **A card is a thing you open.** Selection used to be the open state, which put a
    # 296px panel over the picture on every gesture that touched a box — and the layer
    # refuses presses inside `.rins`, so a handle or a box lying under it stopped being
    # adjustable at all. Framing is a run of drags, so the drag row comes first: it is
    # the one that has to leave the picture alone.
    cards = lambda: pg.locator("#region-inspector").count()
    b = boxes()
    cx, cy = b[0]["x"] + b[0]["w"] / 2, b[0]["y"] + b[0]["h"] / 2
    drag(cx, cy, cx + 0.05, cy + 0.05)
    check("a drag on a box opens no card", cards() == 0, f"{cards()} cards")
    b = boxes()
    cx, cy = b[0]["x"] + b[0]["w"] / 2, b[0]["y"] + b[0]["h"] / 2
    tap(cx, cy)
    check("a click inside one does", cards() == 1, f"{cards()} cards")
    # The dismissal, and the thing it must not do. These two are one row in spirit:
    # the card is put away by clicking outside it, most of the canvas is outside every
    # box, and a canvas where that draws a rectangle is a canvas you cannot put the
    # card away on.
    was = len(boxes())
    tap(0.97, 0.03)
    check("a click outside it puts it away", cards() == 0, f"{cards()} cards")
    check("and leaves no new region behind", len(boxes()) == was,
          f"{was} -> {len(boxes())}")
    dblclick(0.90, 0.04)
    check("a double click on bare canvas makes one", len(boxes()) == was + 1,
          f"{was} -> {len(boxes())}")
    drop_last()
    tap(cx, cy)
    check("and the card comes back on a click", cards() == 1, f"{cards()} cards")
    check("the card is inside the layer",
          pg.evaluate("""() => {
            const c = document.querySelector('#region-inspector').getBoundingClientRect();
            const l = document.querySelector('#region-layer').getBoundingClientRect();
            return c.left >= l.left - 1 && c.right <= l.right + 1
                && c.top >= l.top - 1 && c.bottom <= l.bottom + 1;
          }"""))
    # The numbers are the box's, so they have to be the box's *now*. This used to
    # read them across a drag, back when a drag was something the card sat through;
    # the press that starts one is a press outside the card and puts it away, so what
    # is left to pin here is the reopen — the card that comes back has to describe the
    # rectangle that exists, not the one that was there when it last closed.
    before = pg.input_value("#region-inspector .opt.n input >> nth=1")
    pg.evaluate(DRAG, [0.30, 0.50, 0.44, 0.62])
    pg.wait_for_timeout(200)
    check("and the drag closes it on its way", cards() == 0, f"{cards()} cards")
    moved = boxes()[0]
    tap(moved["x"] + moved["w"] / 2, moved["y"] + moved["h"] / 2)
    after = pg.input_value("#region-inspector .opt.n input >> nth=1")
    check("the card reopens on the box as it is now", before != after,
          f"{before} -> {after}")

    # The numbers have to be readable *while* the box moves, which is the one moment
    # they teach anything and — now that a drag closes the card — the only surface
    # carrying them. `RegionLayer` writes the store once per animation frame expressly
    # to keep them moving; for a while that went to a card at opacity 0, and nothing
    # looking after the release could tell.
    mid = pg.evaluate(DRAG_MID, [0.44, 0.62, 0.36, 0.44])
    pg.wait_for_timeout(200)
    check("and are readable mid-drag",
          isinstance(mid, dict) and mid["shown"] and len(mid["nums"]) == 4,
          repr(mid if not isinstance(mid, dict) else mid["nums"]))
    check("saying where the rectangle is now",
          isinstance(mid, dict) and len(mid["nums"]) == 4 and near(mid["nums"][0], mid["x"]),
          f"readout {mid['nums'][0] if isinstance(mid, dict) and mid['nums'] else '?'} "
          f"vs box {mid['x']:.3f}" if isinstance(mid, dict) else repr(mid))

    # Open it again: the row above ended in a drag, and a drag is a press outside the
    # card. Everything below reads controls that only exist while one is up.
    b = boxes()
    tap(b[0]["x"] + b[0]["w"] / 2, b[0]["y"] + b[0]["h"] / 2)

    # Which character the box is, which is the one thing on the card that decides
    # what comes out of it. There was no row on this at all, and the control had
    # shipped carrying a class no stylesheet declares — so it drew as the UA's own
    # white button inside a black card, and with an empty index it was a dead white
    # button that looked exactly like a live one. Three rows: it is not wearing
    # browser chrome, the press opens the list, and picking arms the box.
    fill = pg.eval_on_selector("#r-lora", "e=>getComputedStyle(e).backgroundColor")
    check("the picker is not a UA button", fill in ("rgba(0, 0, 0, 0)", "transparent"), fill)
    label = pg.inner_text("#r-lora")
    check("empty, it names the act", "LoRA" in label and "No" not in label, label)
    pg.click("#r-lora")
    pg.wait_for_timeout(200)
    rows = pg.eval_on_selector_all(".menu button", "els=>els.length")
    check("the press opens the list", rows > 0, f"{rows} rows")
    was = boxes()
    pick_lora(0)
    check("picking one arms the box",
          pg.inner_text("#r-lora") != label
          and pg.locator("#region-layer .rbox.armed").count() > 0,
          pg.inner_text("#r-lora"))
    # The half that was broken, and the half a scripted `.click()` cannot see: the
    # menu is a portal to <body> and a React child of the card, so its press
    # bubbled to the layer, drew a rectangle and swallowed the click.
    check("and does not draw a rectangle doing it", len(boxes()) == len(was),
          f"{len(was)} -> {len(boxes())}")
    # And it comes back off, which is the row the menu only shows once there is
    # something to take off.
    pg.click("#r-lora")
    pg.wait_for_timeout(200)
    pick_lora(0)
    check("and comes back off", pg.inner_text("#r-lora") == label, pg.inner_text("#r-lora"))

    # Frame scope is a different card, reached by a gesture rather than by
    # knowing which of two things the row in front of you was about.
    pg.click(".rframe-btn")
    pg.wait_for_timeout(200)
    check("the frame button opens the frame's card",
          pg.locator("#frame-inspector").count() == 1
          and pg.locator("#region-inspector").count() == 0)
    # The tiles moved to the console's PlateRow — at rest, not behind the card
    # — so this row asserts the new home and that the card did NOT keep a copy:
    # two live homes for one attachment is the second-way failure.
    check("the plates live in the console, at rest",
          pg.locator("#g-plate-sec #g-drop-scene").count() == 1
          and pg.locator("#g-plate-sec #g-drop-outfit").count() == 1
          and pg.locator("#frame-inspector .drop").count() == 0)
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
    drag(0.05, 0.05, 0.30, 0.30, meta=True)
    tap(0.17, 0.17)
    pg.fill("#r-prompt", "a dancer mid-turn")
    pg.wait_for_timeout(200)
    # Described-only first, because that state changed: the edit path only
    # arms boxes holding a LoRA or a photo, so a plate over words alone is a
    # rejected run and the card has to say the requirement, not promise the
    # compose. Then the box is armed with a photo and the promise comes back.
    landed = pg.evaluate(DROP_ON_CANVAS, [PNG, 0.75, 0.92])
    pg.wait_for_timeout(400)
    check("the drop reached bare canvas", landed == "bare canvas", landed)
    check("a scene dropped on bare canvas opens the frame's card",
          pg.locator("#frame-inspector").count() == 1)
    note = pg.locator("#region-note").inner_text() if pg.locator("#region-note").count() else ""
    check("an unarmed box is told the compose cannot anchor",
          "needs a box holding an identity" in note, note[:64] or "(no note)")
    armed_hit = pg.evaluate(DROP_ON_CANVAS, [PNG, 0.17, 0.17])
    pg.wait_for_timeout(400)
    check("the arming drop reached the box", armed_hit != "bare canvas", armed_hit)
    pg.evaluate("() => document.querySelector('#g-frame')?.click()")
    pg.wait_for_timeout(300)
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

    # An open box is adjustable. Arming geometry was gated because it put *every*
    # rectangle over a render you were judging — which is an objection about the boxes
    # you have not touched, and nothing is drawn until you click one. Having clicked,
    # being asked to press again with a modifier held to move the rectangle in front of
    # you is friction with nothing behind it.
    was = boxes()[0]
    mid = (was["x"] + was["w"] / 2, was["y"] + was["h"] / 2)
    check("the open box shows its handles",
          pg.evaluate("""() => {
            const h = document.querySelector('#region-layer .rbox.sel > i');
            return !!h && getComputedStyle(h).display !== 'none';
          }"""))
    check("and no other box does",
          pg.evaluate("""() => [...document.querySelectorAll('#region-layer .rbox')]
            .filter((b) => !b.classList.contains('sel'))
            .every((b) => [...b.querySelectorAll('i')]
              .every((h) => getComputedStyle(h).display === 'none'))"""))
    drag(mid[0], mid[1], mid[0] + 0.125, mid[1] + 0.125)
    now = boxes()[0]
    check("dragging it moves it", not near(now["x"], was["x"], 0.02),
          f"{was['x']:.3f} -> {now['x']:.3f}")
    check("and the other boxes stay off the render", pg.evaluate(mode) == "content",
          pg.evaluate(mode))

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
