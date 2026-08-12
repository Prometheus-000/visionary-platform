"""Which drop targets actually accept a file? Real DataTransfer, real events."""
from playwright.sync_api import sync_playwright

URL = "http://localhost:8791"

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

    def report(label, sel, mime="image/png", name="x.png"):
        r = pg.evaluate(DROP, [sel, PNG, mime, name])
        if r == "MISSING":
            print(f"  {label:22} {sel:18} MISSING")
            return
        ok = r["acceptsDragover"]
        print(f"  {'ok  ' if ok else 'DEAD'} {label:22} {sel:18} "
              f"dragover_cancelled={r['acceptsDragover']} lit={r['litUp']} "
              f"drop_handled={r['handledDrop']}")

    print("\nVIDEO side")
    pg.evaluate("()=>setKind('video')")
    pg.wait_for_timeout(600)
    # References first: a keyframe makes the tray legitimately inert, so
    # testing after one would measure the exclusivity rule, not the handler.
    report("add picture ref", "#v-add-ref")
    pg.evaluate("() => { refs.length = 0; refVids.length = 0; drawRefs(); }")
    report("add video ref", "#v-add-vid", "video/mp4", "x.mp4")
    pg.evaluate("() => { refs.length = 0; refVids.length = 0; drawRefs(); }")
    report("first keyframe", "#v-drop-first")
    report("last keyframe", "#v-drop-last")
    pg.evaluate("() => { clearFrame('first'); clearFrame('last'); drawRefs(); }")
    report("the video canvas", "#vid-out")

    print("\nIMAGE side")
    pg.evaluate("()=>setKind('image')")
    pg.wait_for_timeout(600)
    # Arming Regions is now the whole reveal: the plates moved out of the
    # drawer and into the region bar, which only exists when regions do.
    pg.evaluate("() => document.querySelector('#g-regional').click()")
    pg.wait_for_timeout(500)
    report("scene plate", "#g-drop-scene")
    report("outfit plate", "#g-drop-outfit")
    report("region layer", "#region-layer")
    report("a region box", "#region-layer .rbox")
    # `#canvas .frame` is deliberately not tested. `wireCanvasDrop` listens on
    # #region-layer, which covers the frame, and paints `hot` onto the frame as
    # its *host* — so a real drag is cancelled on the layer and the frame never
    # needs its own listener. Dispatching synthetically at the frame cannot
    # reach the layer (the layer is its child, not its ancestor), so the row
    # reported DEAD for a target that is working, while `body.dragging` kept
    # `lit` true and hid the contradiction. A check that cries dead about a live
    # control is how a real dead one gets waved past.

    print("\nELSEWHERE (known good, as a control)")
    pg.evaluate("() => { window.setMode && setMode('datasets') }")
    pg.wait_for_timeout(600)
    report("dataset hero drop", "#drop")
    b.close()
