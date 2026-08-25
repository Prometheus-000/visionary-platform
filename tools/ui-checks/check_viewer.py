"""
The gallery viewer's gesture, asserted against whichever front end you point at.

    python3 tools/ui-checks/check_viewer.py                  # the vanilla page
    python3 tools/ui-checks/check_viewer.py http://localhost:5173   # React

Both take the same selectors, which is the point: this is a parity check, not a
test of one implementation. Every assertion here is a fault that was expensive
to find or would be.

**The drag is dispatched synthetically, and that is not laziness.** A driver's
`left_click_drag` sends mouse events that never reached the element at all when
this was written, so a real-input test silently asserted nothing — it passed by
doing nothing, which is the worst outcome available. Synthetic `PointerEvent`s
with real timestamps exercise the same handlers the browser would, and the
thing being checked is the commit arithmetic rather than the browser's input
plumbing.

What is NOT covered, and needs a human on real hardware: that a trackpad and a
touchscreen both produce the pointer stream this simulates. That asymmetry is
exactly what broke here before — `<img>` is natively draggable, so a mouse drag
started an HTML image drag and the browser fired `pointercancel` one frame in,
while touch never took that path. `draggable="false"` is asserted below because
it is the fix, but only a real trackpad proves the fix still holds.
"""
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8791"

# A flick: six moves over ~120ms. Comfortably past both gates — a quarter of
# the width, and 0.45px/ms.
FLICK = """
async ([dx, steps, gap]) => {
  const lb = document.querySelector('.lb');
  if (!lb) return {error: 'no viewer open'};
  const track = lb.querySelector('.lb-track');
  const at = () => document.querySelector('.lb .lb-at')?.textContent;
  const fire = (type, x) => lb.dispatchEvent(new PointerEvent(type, {
    bubbles: true, cancelable: true, pointerId: 1, pointerType: 'mouse',
    clientX: x, clientY: 450, isPrimary: true,
  }));
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const x0 = Math.round(innerWidth / 2);

  const before = at();
  fire('pointerdown', x0);
  let mid = null;
  for (let i = 1; i <= steps; i++) {
    fire('pointermove', x0 + (dx / steps) * i);
    if (i === steps) mid = track.style.transform;
    await sleep(gap);
  }
  fire('pointerup', x0 + dx);
  await sleep(40);
  const settled = document.querySelector('.lb .lb-track').style.transform;
  await sleep(600);            // let the 0.3s snap land and the index swap
  return {before, mid, settled, after: at(),
          stillOpen: !!document.querySelector('.lb')};
}
"""

CANCEL = """
async () => {
  const lb = document.querySelector('.lb');
  const track = lb.querySelector('.lb-track');
  const fire = (type, x) => lb.dispatchEvent(new PointerEvent(type, {
    bubbles: true, cancelable: true, pointerId: 1, pointerType: 'mouse',
    clientX: x, clientY: 450, isPrimary: true,
  }));
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const x0 = Math.round(innerWidth / 2);
  const at = () => document.querySelector('.lb .lb-at')?.textContent;
  const before = at();
  fire('pointerdown', x0);
  for (let i = 1; i <= 6; i++) { fire('pointermove', x0 - i * 50); await sleep(20); }
  fire('pointercancel', x0 - 300);
  await sleep(500);
  return {before, after: at(),
          transform: document.querySelector('.lb .lb-track').style.transform};
}
"""

fails: list[str] = []


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(f"{label} {detail}".strip())


def open_viewer(pg):
    """
    Open the first card.

    A JS `.click()` rather than a driver click, because the drawer starts closed
    on the vanilla page: the card is in the DOM and off-screen, so a real click
    spends thirty seconds being told `#canvas-empty` intercepts pointer events.
    Opening the drawer first would work too and would be testing the drawer,
    which is not what this file is about — the handler is bound either way.
    """
    # The drawer must be open to hold cards at all now — a shut drawer renders
    # none, which is what stopped it fetching covers behind a closed panel.
    if not pg.evaluate(
        "() => document.querySelector('#t-drawer')?.classList.contains('on')"
    ):
        pg.click("#t-drawer")
    pg.wait_for_selector("#drawer-grid .gal .media", timeout=20_000)
    pg.evaluate("() => document.querySelector('#drawer-grid .gal .media').click()")
    pg.wait_for_selector(".lb", timeout=10_000)
    pg.wait_for_timeout(250)


with sync_playwright() as pw:
    b = pw.chromium.launch(channel="chrome")
    pg = b.new_page(viewport={"width": 1440, "height": 900}, color_scheme="dark")
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(URL, wait_until="networkidle", timeout=60_000)
    pg.wait_for_timeout(1200)

    print(f"\n=== {URL} ===")
    open_viewer(pg)

    shape = pg.evaluate("""() => {
      const lb = document.querySelector('.lb');
      const slides = [...lb.querySelectorAll('.lb-slide')];
      return {
        slides: slides.length,
        transform: lb.querySelector('.lb-track').style.transform,
        at: lb.querySelector('.lb-at')?.textContent,
        prevDisabled: lb.querySelector('.lb-nav.prev')?.disabled,
        // The fix that made a trackpad behave like a thumb.
        draggable: slides.map(s => s.querySelector('img,video')?.getAttribute('draggable') ?? null),
      };
    }""")
    # Three slides, because the drag has to show the neighbours: a page that
    # only swaps on release is a cut, and a cut does not say which way you went.
    check("three slides", shape["slides"] == 3, f"got {shape['slides']}")
    check("track centred", shape["transform"] == "translateX(-100%)", shape["transform"])
    check("prev disabled at the first", shape["prevDisabled"] is True)
    real = [d for d in shape["draggable"] if d is not None]
    check("media is draggable=false", real and all(d == "false" for d in real), str(shape["draggable"]))

    r = pg.evaluate(FLICK, [-300, 6, 20])
    check("a brisk flick pages forward", r.get("after") != r.get("before"),
          f"{r.get('before')} -> {r.get('after')}")
    check("mid-drag follows the pointer", "300px" in (r.get("mid") or ""), r.get("mid") or "")
    check("settles to the next slide", r.get("settled") == "translateX(-200%)", r.get("settled") or "")

    # A slow short nudge must not page. 24px over 300ms is under both gates.
    r2 = pg.evaluate(FLICK, [-24, 4, 75])
    check("a slow short nudge does not page", r2.get("after") == r2.get("before"),
          f"{r2.get('before')} -> {r2.get('after')}")
    check("and springs back", r2.get("settled") == "translateX(-100%)", r2.get("settled") or "")

    # Cancel means something took the gesture away, which is not you finishing
    # it — so it reverts rather than committing to a take you never asked for.
    r3 = pg.evaluate(CANCEL)
    check("pointercancel reverts", r3.get("after") == r3.get("before"),
          f"{r3.get('before')} -> {r3.get('after')}")
    check("cancel re-centres", r3.get("transform") == "translateX(-100%)", r3.get("transform") or "")

    # A tap on the picture asks for more of it. A tap used to close the viewer,
    # so every attempt to see a render properly dismissed it.
    pg.evaluate("() => document.querySelector('.lb-slide img,.lb-slide video')?.click()")
    pg.wait_for_timeout(150)
    bare = pg.evaluate("() => document.querySelector('.lb')?.classList.contains('bare')")
    check("tapping the picture turns chrome off", bare is True, f"bare={bare}")

    if errors:
        check("no page errors", False, str(errors[:2]))
    b.close()

print()
if fails:
    print(f"{len(fails)} failed:", file=sys.stderr)
    for f in fails:
        print(f"  {f}", file=sys.stderr)
    sys.exit(1)
print("viewer gesture intact")
