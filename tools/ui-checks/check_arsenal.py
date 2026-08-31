"""
That a character recalled from the Arsenal follows the Arsenal, asserted by colour.

    python3.11 tools/ui-checks/check_arsenal.py [http://localhost:8791]

The roadmap states the rule in one sentence and it has two halves — *things are
applied, never imported: edits in a scene are scene-local, edits in the library
propagate, and those are two different acts in two different places*. Only the
first half was built for a while: `hydrate` copied a saved character's bytes into
the scene and stopped, so re-shooting a reference left every scene that already
had that character holding the old picture. It was unreachable before the scene
survived a reload, which is why the second half landed with persistence.

**Everything here is asserted as a pixel value, not as a filename or a byte
count.** A reference that failed to update is a reference that still renders
perfectly — same shape, same size, same card, a different face. That is the
whole reason this file exists rather than a look: nothing about the failure is
visible unless you know what the picture was supposed to become. So each step
writes a flat swatch of a known colour into the library and reads the colour back
off the tile the page is actually showing.

Four colours, in order, because the interesting failures are off-by-one in time:
a check that goes red -> blue proves *a* change propagated, and a check that goes
red -> blue -> green -> yellow proves the page is following the library rather
than lagging it by one revision.

**The negative case is the last row and it is the one that erodes.**
`PoolFile.from` is set only by a recall and absent on a dropped file, so a
photograph dragged in from the Finder has to come through `refreshArsenal`
completely untouched. A mixed member — one recalled reference, one dropped —
pins both directions in one drive: widen the provenance test by accident and the
propagation rows all still pass while dropped files start chasing a library entry
they never came from. Raised by visionary-platform-b0.

**What this deliberately does not test: `scene.sources` surviving a repoint.**
`repointRefs` carries it, and it should — an id the pool has lost is filtered out
of the payload by `readScene`, so a missed rewrite would take a keyframe out of a
run with nothing on screen saying so. But no gesture on the page writes
`scene.sources` at all: `setScene` has no callers, the keyframe tiles write
`store.keyframe` instead, and `emptyScene` leaves it `{}`. The path is real in
the compiler and in `_validate_scene` and unreachable from the interface, so
there is nothing here to drive. Named rather than left out, the way `check_drop`
names `#canvas .frame`: a hole nobody wrote down becomes coverage everybody
assumes.
"""

import base64
import json
import struct
import sys
import urllib.request
import zlib

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8791"

fails: list[str] = []


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(label)


# Flat 2x2 swatches, written by hand rather than by Pillow — this directory runs
# with playwright and nothing else, and a dependency for four rectangles would be
# the tail wagging the dog. Distinct enough that a wrong one cannot be read as a
# right one under JPEG's chroma subsampling, which the page applies to anything
# it re-encodes.
RED, BLUE, GREEN, YELLOW = (220, 40, 40), (40, 60, 220), (40, 190, 90), (230, 200, 60)


def swatch(rgb: tuple[int, int, int]) -> str:
    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    row = b"\x00" + bytes(rgb) * 2
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(row * 2))
           + chunk(b"IEND", b""))
    return base64.b64encode(png).decode()


def shelve(handle: str, rgb, note="the woman in the olive coat") -> None:
    """Write a character to the Arsenal behind the page's back.

    Deliberately over HTTP rather than through the Save button: what is being
    tested is that the *page* follows the library, so the library has to be able
    to move without the page having touched it. Saving through the UI would leave
    the new bytes in the pool already and prove nothing.
    """
    body = json.dumps({"note": note, "retention": "",
                       "refs": [{"kind": "image", "b64": swatch(rgb)}]}).encode()
    req = urllib.request.Request(f"{URL}/api/characters/{handle}", data=body,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        if not json.loads(r.read()).get("ok"):
            raise AssertionError(f"could not shelve @{handle}")


# One pixel, decoded off whatever the element is actually showing. `naturalWidth`
# is waited on because an <img> whose src has just been set has no pixels yet and
# drawImage would silently produce transparent black — which reads as a colour
# mismatch and sends you looking for a propagation bug that is not there.
SAMPLE = """
async (sel) => {
  const el = document.querySelector(sel);
  if (!el) return null;
  const img = new Image();
  img.src = el.src;
  if (!img.complete) await new Promise((r) => { img.onload = r; img.onerror = r; });
  if (!img.naturalWidth) return null;
  const c = document.createElement('canvas');
  c.width = c.height = 1;
  c.getContext('2d').drawImage(img, 0, 0, 1, 1);
  const d = c.getContext('2d').getImageData(0, 0, 1, 1).data;
  return [d[0], d[1], d[2]];
}
"""


def near(got, want, tol=12) -> bool:
    """Within tolerance, because an image the page re-encodes to JPEG at 0.92 does
    not come back byte-identical — `shrinkB64` only re-encodes above 1536px, but
    asserting equality here would make this check depend on that threshold rather
    than on propagation."""
    return bool(got) and all(abs(a - b) <= tol for a, b in zip(got, want))


def sample(page, sel):
    return page.evaluate(SAMPLE, sel)


def main(pg):
    # ---- the image side --------------------------------------------------
    #
    # A box high on the frame, because the card is rooted on its own near edge
    # and one drawn low puts its name row above the canvas, where it is clipped
    # by the overflow that keeps the console from pushing the picture out of
    # frame. Nothing to do with the Arsenal; it is where the card fits.
    print("\nIMAGE SIDE")
    shelve("maya", RED)
    pg.goto(URL, wait_until="networkidle")
    pg.wait_for_selector("#canvas")
    layer = pg.locator("#region-layer")
    b = layer.bounding_box()
    pg.mouse.dblclick(b["x"] + b["width"] * 0.78, b["y"] + b["height"] * 0.12)
    pg.wait_for_timeout(500)
    pg.mouse.click(b["x"] + b["width"] * 0.86, b["y"] + b["height"] * 0.35)
    pg.wait_for_selector("#r-name", timeout=10000)

    # The name *is* the recall — see `CharacterRow`. mousedown rather than click
    # because the field has focus and a click blurs it first, taking the list
    # with it before the press lands.
    pg.fill("#r-name", "may")
    pg.wait_for_selector(".rcast button", timeout=10000)
    pg.locator(".rcast button").first.dispatch_event("mousedown")
    pg.wait_for_timeout(1500)
    check("a recall brings the character's photograph",
          near(sample(pg, "#r-ref img"), RED), str(sample(pg, "#r-ref img")))
    check("and their sentence, into an empty field",
          (pg.input_value("#r-prompt") or "").startswith("the woman"),
          pg.input_value("#r-prompt"))
    # The rectangle is the box's own. A saved character carries no geometry —
    # where somebody stands is a fact about this frame, not about them.
    check("and not the rectangle it was saved from",
          pg.locator("#region-layer .rbox").count() == 1)

    shelve("maya", BLUE)
    pg.goto(URL, wait_until="networkidle")
    pg.wait_for_selector("#canvas")
    pg.wait_for_timeout(2500)
    b = pg.locator("#region-layer").bounding_box()
    pg.mouse.click(b["x"] + b["width"] * 0.86, b["y"] + b["height"] * 0.35)
    pg.wait_for_selector("#r-ref img", timeout=10000)
    check("a restored scene re-reads the shelf",
          near(sample(pg, "#r-ref img"), BLUE), str(sample(pg, "#r-ref img")))

    # The half that needs no reload, and the one that decides what actually
    # renders: `start` awaits the refresh before the body is read.
    shelve("maya", GREEN)
    pg.keyboard.press("Escape")
    pg.fill("#prompt", "a rain-slick alley at dusk")
    pg.click("#go-gen")
    pg.wait_for_timeout(9000)
    pg.wait_for_selector("#canvas-clear", timeout=20000)
    pg.click("#canvas-clear")
    pg.wait_for_timeout(800)
    b = pg.locator("#region-layer").bounding_box()
    pg.mouse.click(b["x"] + b["width"] * 0.86, b["y"] + b["height"] * 0.35)
    pg.wait_for_selector("#r-ref img", timeout=10000)
    check("a run re-reads the shelf, with no reload",
          near(sample(pg, "#r-ref img"), GREEN), str(sample(pg, "#r-ref img")))
    pg.keyboard.press("Escape")

    # ---- the video side --------------------------------------------------
    #
    # The pool is keyed by content, so propagation here is a *re-point*: new
    # bytes are a new entry and every ref that named the old one has to be
    # rewritten onto it. The count is what tells a re-point from a second row.
    print("\nVIDEO SIDE")
    pg.click("#g-sampling")
    pg.wait_for_selector(".menu.form select")
    pg.select_option(".menu.form select", "video:h3")
    pg.keyboard.press("Escape")
    pg.wait_for_timeout(600)

    pg.click("#prompt")
    pg.type("#prompt", "@may")
    pg.wait_for_selector(".tment button", timeout=10000)
    pg.locator(".tment button").first.dispatch_event("mousedown")
    pg.wait_for_timeout(2000)
    check("the same character recalls into the cast",
          near(sample(pg, ".tref img"), GREEN), str(sample(pg, ".tref img")))

    # **The negative case.** A dropped file has no library behind it and must
    # come through the refresh untouched. Attached to the *same* member as the
    # recalled one, so a single run pins both directions at once.
    dropped = swatch((150, 90, 200))
    pg.set_input_files(".tmat input[type=file]", files=[{
        "name": "dropped.png", "mimeType": "image/png",
        "buffer": base64.b64decode(dropped)}])
    pg.wait_for_timeout(1200)
    check("a dropped photograph joins the same member",
          pg.locator(".tref").count() == 2, f"{pg.locator('.tref').count()} refs")

    shelve("maya", YELLOW)
    pg.fill("#prompt", "@maya steps off the train")
    pg.click("#go-vid")
    pg.wait_for_timeout(9000)
    pg.wait_for_selector(".chip.cast", timeout=25000)
    pg.locator(".chip.cast").first.click()
    pg.wait_for_selector(".tref img", timeout=10000)

    tiles = [pg.evaluate(SAMPLE, f".tref:nth-of-type({i + 1}) img")
             for i in range(pg.locator(".tref").count())]
    check("the recalled reference followed the shelf",
          any(near(t, YELLOW) for t in tiles), str(tiles))
    check("the dropped one did not",
          any(near(t, (150, 90, 200)) for t in tiles), str(tiles))
    # A re-point, not a second row: the old pool entry is gone and the ref names
    # the new id. Two rows here would mean `repointRefs` added rather than moved.
    check("and the member still holds exactly two references",
          pg.locator(".tref").count() == 2, f"{pg.locator('.tref').count()} refs")


with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    try:
        main(page)
    finally:
        browser.close()
    if errors:
        fails.append(f"page errors: {errors[:2]}")
        print(f"  FAIL page errors  {errors[:2]}")

print()
if fails:
    print("FAILED: " + "; ".join(fails))
    sys.exit(1)
print("a recalled character follows the Arsenal, and a dropped one does not")
