"""
The storyboard: a wall of boards, the two arrow kinds on a frame, a carried
panel, and the hand-off that lands on the canvas with both keyframes.

    /opt/homebrew/bin/python3.11 tools/preview_ui.py 8799 &
    python3.11 tools/ui-checks/check_storyboard.py http://localhost:8799

  * **The door opens on the most recent board**, with its stencils drawn:
    a camera pill on a panel is a stencil on its frame, not a word under it.
  * **A drag on bare picture is a subject's arrow**, hollow, with a name
    field at its tail — and the sentence it will write appears under the
    prose the moment it exists, marked as derived.
  * **The camera is picked from the palette's own tiles**, with amplitude and
    speed under them, and the tag under the frame reads the same words the
    document will.
  * **A panel is carried, not marked.** Press its number and travel, and it
    lands in the slot nearest the hand; the order on the wall is the order
    saved.
  * **Two selected panels are a first-and-last-frame take.** Animate lands on
    the canvas with both keyframe tiles filled, the first panel's prose in
    the row, and its camera on the rail.
  * **A dropped file is a panel** — uploaded into the board's folder and
    served back from it.
  * **Boards are a list**, and switching one is one press.
"""
import struct
import sys
import time
import zlib
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8791"
fails: list[str] = []


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(label)


def png(w: int, h: int) -> bytes:
    """A flat PNG, so the upload row needs no fixture on disk."""
    raw = b"".join(b"\x00" + b"\x80\x40\x20" * w for _ in range(h))
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def centre(loc):
    b = loc.bounding_box()
    return b["x"] + b["width"] / 2, b["y"] + b["height"] / 2


with sync_playwright() as pw:
    b = pw.chromium.launch()
    page = b.new_page(viewport={"width": 1512, "height": 982})
    page.goto(URL, wait_until="networkidle")

    # The door, and the wall behind it — on the most recent board.
    check("the header has a Storyboard door", page.locator("#door-storyboard").count() == 1)
    page.click("#door-storyboard")
    page.wait_for_selector("#v-storyboard", timeout=5000)
    page.wait_for_selector(".sbpanel", timeout=5000)
    panels = page.locator(".sbpanel")
    check("the wall opens on the most recent board", page.input_value("#sb-title") == "The Lookout",
          page.input_value("#sb-title"))
    check("every panel is on the wall", panels.count() == 8, str(panels.count()))
    check("a camera pill is a stencil on the frame", page.locator(".sbcam").count() == 7,
          str(page.locator(".sbcam").count()))
    check("a subject's arrow is hollow and on the picture", page.locator(".sbarrow").count() == 2,
          str(page.locator(".sbarrow").count()))
    check("the camera tag reads what the document will",
          page.locator(".sbpanel").nth(1).locator(".sbtag-camera").inner_text().strip()
          == "pan right · large · slow")

    # A drag on bare picture draws an arrow; a name goes on its tail; the
    # sentence appears under the prose.
    frame = panels.nth(3).locator(".sbframe")
    fb = frame.bounding_box()
    page.mouse.move(fb["x"] + fb["width"] * 0.2, fb["y"] + fb["height"] * 0.3)
    page.mouse.down()
    page.mouse.move(fb["x"] + fb["width"] * 0.8, fb["y"] + fb["height"] * 0.7, steps=6)
    page.mouse.up()
    check("a drag on the frame draws a subject's arrow", page.locator(".sbarrow").count() == 3,
          str(page.locator(".sbarrow").count()))
    label = panels.nth(3).locator(".sblabel")
    check("the arrow's name field is at its tail", label.count() == 1)
    label.fill("Maya")
    said = panels.nth(3).locator(".sbsaid").inner_text()
    check("the arrow's sentence is under the prose, with the name in it",
          said.startswith("Maya moves from the upper left"), said)
    page.keyboard.press("Escape")
    # Select it again — selecting an arrow is editing its name, so the field
    # takes the keys — and take it back with the mark at its head.
    panels.nth(3).locator("[data-arrow]").click()
    panels.nth(3).locator(".sbx").click()
    check("the mark at the head removes the arrow", page.locator(".sbarrow").count() == 2,
          str(page.locator(".sbarrow").count()))
    # A fresh arrow with no name: Backspace in the empty field takes it back.
    page.mouse.move(fb["x"] + fb["width"] * 0.3, fb["y"] + fb["height"] * 0.5)
    page.mouse.down()
    page.mouse.move(fb["x"] + fb["width"] * 0.7, fb["y"] + fb["height"] * 0.5, steps=4)
    page.mouse.up()
    page.keyboard.press("Backspace")
    check("Backspace in an empty name field removes the arrow", page.locator(".sbarrow").count() == 2,
          str(page.locator(".sbarrow").count()))

    # The camera, from the palette's own tiles, with its two dimensions.
    panels.nth(3).locator(".sbtag-camera").click()
    page.wait_for_selector(".sbpal", timeout=3000)
    page.locator(".sbpal .tl", has_text="pan right").click()
    page.wait_for_selector(".sbpal .sbdims", timeout=3000)
    page.locator(".sbpal .sbdims .seg button.s", has_text="large").click()
    page.keyboard.press("Escape")
    check("the camera picker writes the pill, and the tag says amplitude",
          panels.nth(3).locator(".sbtag-camera").inner_text().strip() == "pan right · large",
          panels.nth(3).locator(".sbtag-camera").inner_text())
    check("the stencil follows", page.locator(".sbcam").count() == 8)

    # Carry the first panel to the third slot.
    first_prose = panels.nth(0).locator(".sbprose").input_value()
    x0, y0 = centre(panels.nth(0).locator(".sbidx"))
    x1, y1 = centre(panels.nth(2).locator(".sbframe"))
    page.mouse.move(x0, y0)
    page.mouse.down()
    page.mouse.move(x0 + 8, y0 + 8)
    page.mouse.move(x1, y1, steps=10)
    check("the carried panel is on screen under the hand", page.locator(".sbghost").count() == 1)
    page.mouse.up()
    time.sleep(0.4)
    check("it lands in the slot nearest the hand",
          panels.nth(2).locator(".sbprose").input_value() == first_prose
          and panels.nth(0).locator(".sbprose").input_value().startswith("Maya enters"))
    check("the release put nothing in the selection", page.locator("#sb-animate").count() == 0)

    # Two numbers pressed without travelling: a pair, and the hand-off.
    panels.nth(0).locator(".sbidx").click()
    panels.nth(1).locator(".sbidx").click()
    page.wait_for_selector("#sb-animate", timeout=3000)
    check("two panels read as a first-and-last-frame take",
          page.locator("#sb-animate").inner_text().strip() == "Animate 1 → 2",
          page.locator("#sb-animate").inner_text())
    page.click("#sb-animate")
    page.wait_for_selector("#v-drop-first", timeout=5000)
    time.sleep(0.6)
    check("the first panel is the first frame", page.locator("#v-drop-first.set").count() == 1)
    check("the second panel is the last frame", page.locator("#v-drop-last.set").count() == 1)
    row = page.locator("#prompt").input_value()
    check("the panel's prose is the shot's line, and its arrow's sentence follows",
          row.startswith("Maya enters frame left") and "Maya enters from frame left" in row, row)
    check("the second panel's prose is where the shot ends", "The shot ends on the last frame" in row)
    rail = page.locator("#shot-rail .spill b").all_inner_texts()
    check("the camera rides the rail with its amplitude and speed",
          any("pan right · large · slow" in t for t in rail), repr(rail))
    check("the framing rides too", any(t.strip() == "wide" for t in rail), repr(rail))

    # Back to the wall: the pair is not remembered, the order is.
    page.click("#door-storyboard")
    page.wait_for_selector(".sbpanel", timeout=5000)
    check("the order saved is the order on the wall",
          page.locator(".sbpanel").nth(2).locator(".sbprose").input_value() == first_prose)

    # A picture chosen from the gallery: a trip through the Generate stage,
    # and the pin comes back to the panel that asked.
    page.locator(".sbpanel").nth(3).hover()
    page.locator(".sbpanel").nth(3).locator(".sbhead .ico").click()
    page.locator(".menu button", has_text="Choose a picture from the gallery").click()
    page.wait_for_selector("#gal-grid .gal", timeout=5000)
    page.locator('#gal-grid .gal[data-job="job000"]').hover()
    page.locator('#gal-grid .gal[data-job="job000"] .more').click()
    page.locator(".menu button", has_text="Add to storyboard").click()
    page.wait_for_selector("#v-storyboard .sbpanel", timeout=5000)
    page.wait_for_function(
        "!!document.querySelectorAll('.sbpanel')[3]?.querySelector('.sbframe img')", timeout=5000)
    src = page.locator(".sbpanel").nth(3).locator(".sbframe img").get_attribute("src") or ""
    check("a pin from the gallery lands in the panel that asked for it", "job000" in src, src)
    check("and the panel keeps its own words",
          page.locator(".sbpanel").nth(3).locator(".sbprose").input_value().startswith("She says the line"))

    # A file is a panel.
    tmp = Path(__file__).with_name("_storyboard_drop.png")
    tmp.write_bytes(png(64, 48))
    try:
        n = page.locator(".sbpanel").count()
        page.locator("#v-storyboard input[type=file]").set_input_files(str(tmp))
        page.wait_for_function(f"document.querySelectorAll('.sbpanel').length === {n + 1}", timeout=5000)
        src = page.locator(".sbpanel").last.locator(".sbframe img").get_attribute("src") or ""
        check("a dropped picture is a panel served from the board's own folder",
              "/api/storyboard/" in src and "/file/" in src, src)
    finally:
        tmp.unlink(missing_ok=True)

    # Boards are a list.
    page.click("#sb-boards")
    page.wait_for_selector(".menu button", timeout=3000)
    rows = page.locator(".menu button").all_inner_texts()
    check("the other board is one press away", any("Kitchen, morning" in r for r in rows), repr(rows))
    page.locator(".menu button", has_text="Kitchen, morning").click()
    page.wait_for_function("document.querySelector('#sb-title').value === 'Kitchen, morning'", timeout=5000)
    check("switching lands on its panels", page.locator(".sbpanel").count() == 1)

    # The document, read: the same compiler, the camera's own grammar.
    page.click("#sb-boards")
    page.locator(".menu button", has_text="Read the H3 document").click()
    page.wait_for_selector("#sb-doc textarea", timeout=5000)
    doc = page.locator("#sb-doc textarea").input_value()
    check("the document is H3's, with the camera in its grammar",
          "integrated_multimodal_description" in doc and "focus racks" in doc, doc[:200])

    b.close()

for f in fails:
    print(f"  FAIL  {f}")
print(f"\n{'PASS' if not fails else str(len(fails)) + ' FAILED'}")
raise SystemExit(1 if fails else 0)
