"""
Which drop targets actually accept a file? Real DataTransfer, real events.

    python3 tools/ui-checks/check_drop.py                        # vanilla
    python3 tools/ui-checks/check_drop.py http://localhost:5173  # React

Takes a URL, like check_viewer.py, because this is the shape every port check
should take — and because every target it names was re-implemented in the React
port, so a check that can only run against one of the two front ends is a check
that cannot say whether the port kept the behaviour.

Making it URL-parameterised meant giving up the vanilla page's globals. It used
to reset state with `refs.length = 0; drawRefs()` and switch sides with
`setKind('video')`, neither of which exists in a bundled front end — so the
driving is gestures now: click the kind chip, click a filled tile to clear it.
That is a better check on either front end for the reason the drag test already
records: a driver poking at internals is not a user, and the handler it is
poking past may be the broken one.
"""
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8791"
fails = []

# A 2x2 PNG, so the page gets something it can really decode.
PNG = ("iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFElEQVR4nGP8"
       "z8Dwn4GBgYEJTAAAHAcCAKvHBh4AAAAASUVORK5CYII=")

DROP = """
([sel, b64, mime, name]) => {
  const el = document.querySelector(sel);
  if (!el) return 'MISSING';
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  const file = new File([buf], name, {type: mime});
  const dt = new DataTransfer();
  dt.items.add(file);
  const fire = (type) => {
    const ev = new DragEvent(type, {bubbles: true, cancelable: true, dataTransfer: dt});
    el.dispatchEvent(ev);
    return ev;
  };
  fire('dragenter');
  const over = fire('dragover');
  const lit  = el.classList.contains('hot') || document.body.classList.contains('dragging');
  const drop = fire('drop');
  // A target that accepts a drop MUST cancel dragover — otherwise the browser
  // never delivers the drop at all. That is the whole test.
  return {acceptsDragover: over.defaultPrevented, litUp: lit,
          handledDrop: drop.defaultPrevented};
}
"""

with sync_playwright() as pw:
    b = pw.chromium.launch(channel="chrome")
    pg = b.new_page(viewport={"width": 1400, "height": 950}, color_scheme="dark")
    pg.goto(URL, wait_until="networkidle", timeout=60_000)
    pg.wait_for_timeout(1200)
    print(f"\n=== {URL} ===")

    def report(label, sel, mime="image/png", name="x.png"):
        r = pg.evaluate(DROP, [sel, PNG, mime, name])
        if r == "MISSING":
            print(f"  MISS {label:22} {sel:22} not on this page")
            fails.append(f"{label} (missing)")
            return
        ok = r["acceptsDragover"]
        if not ok:
            fails.append(label)
        print(f"  {'ok  ' if ok else 'DEAD'} {label:22} {sel:22} "
              f"dragover_cancelled={r['acceptsDragover']} lit={r['litUp']} "
              f"drop_handled={r['handledDrop']}")

    def to_video():
        """The chip inside the prompt field, which is the only way a user has."""
        if pg.eval_on_selector("#c-video", "e => e.classList.contains('hide')"):
            pg.click("#kind-toggle")
            pg.wait_for_timeout(500)

    def to_image():
        if pg.eval_on_selector("#c-image", "e => e.classList.contains('hide')"):
            pg.click("#kind-toggle")
            pg.wait_for_timeout(500)

    def clear_refs():
        """Every chip's own ✕. The tray and the keyframe pair put each other out of
        play, so a report that leaves a file attached makes the *next* report measure
        the exclusivity rule rather than the handler — which is what happened when this
        cleared state through the vanilla page's globals and the React port had none."""
        while pg.locator("#v-refs button.x").count():
            pg.locator("#v-refs button.x").first.click()
            pg.wait_for_timeout(120)

    def clear_keyframes():
        """A second click on a filled tile clears it — the same gesture a user has, and
        the one that was missing from the keyframe tiles for a while."""
        for sel in ("#v-drop-first", "#v-drop-last"):
            if pg.locator(f"{sel}.set").count():
                pg.click(sel)
                pg.wait_for_timeout(150)

    print("\nVIDEO side")
    to_video()
    # References first: a keyframe makes the tray legitimately inert, so testing after
    # one would measure the exclusivity rule, not the handler.
    report("add picture ref", "#v-add-ref")
    clear_refs()
    report("add video ref", "#v-add-vid", "video/mp4", "x.mp4")
    clear_refs()
    report("first keyframe", "#v-drop-first")
    report("last keyframe", "#v-drop-last")
    clear_keyframes()
    report("the video canvas", "#vid-out")
    clear_keyframes()

    print("\nIMAGE side")
    to_image()
    # Arming Regions is the whole reveal: the plates live in the region bar,
    # which only exists when regions do.
    pg.click("#g-regional")
    pg.wait_for_timeout(600)
    report("scene plate", "#g-drop-scene")
    report("outfit plate", "#g-drop-outfit")
    report("region layer", "#region-layer")
    report("a region box", "#region-layer .rbox")
    # `#canvas .frame` is deliberately not tested. The drop is listened for on
    # #region-layer, which covers the frame and paints `hot` onto the frame as
    # its *host* — so a real drag is cancelled on the layer and the frame never
    # needs its own listener. Dispatching synthetically at the frame cannot reach
    # the layer (the layer is its child, not its ancestor), so the row reported
    # DEAD for a target that is working, while `body.dragging` kept `lit` true
    # and hid the contradiction. A check that cries dead about a live control is
    # how a real dead one gets waved past.

    print("\nELSEWHERE (known good, as a control)")
    pg.click("#door")
    pg.wait_for_timeout(800)
    report("dataset hero drop", "#drop")
    b.close()

print("\n" + ("FAILED: " + ", ".join(fails) if fails else "every target accepts its drop"))
sys.exit(1 if fails else 0)
