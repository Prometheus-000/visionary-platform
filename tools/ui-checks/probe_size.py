"""
The ratio picker and the pixel boxes, driven the way a person drives them.

    python3 tools/ui-checks/probe_size.py http://localhost:8791
    python3 tools/ui-checks/probe_size.py http://localhost:5173

They are one control, so every assertion here is about the two halves agreeing:
picking a ratio writes the boxes, typing in the boxes selects Custom, the swap
transposes the *bucket* rather than the pixels, and nothing is snapped while you
are still typing.

Driven through real clicks and real keys rather than by calling into the page,
for the reason the drag test already records: a driver poking at internals is
not a user, and the handler it pokes past may be the broken one.
"""
import re
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8791"
MAC = "Meta"

fails: list[str] = []


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(f"{label} {detail}".strip())


def boxes(pg):
    """The two pixel inputs, whatever they currently say."""
    return pg.evaluate("""() => {
      const i = [...document.querySelectorAll('.sizer .sz-custom input')];
      return i.map(x => x.value);
    }""")


def label(pg):
    return pg.eval_on_selector("#g-size", "b => b.textContent.trim()")


def selected(pg):
    return pg.evaluate("""() => {
      const on = document.querySelector('.sizer .ars .ar.on');
      return on ? on.querySelector('b')?.textContent?.trim() : null;
    }""")


def open_sizer(pg):
    if not pg.query_selector(".sizer"):
        pg.click("#g-size")
        pg.wait_for_selector(".sizer", timeout=5000)
    pg.wait_for_timeout(120)


with sync_playwright() as pw:
    b = pw.chromium.launch(channel="chrome")
    pg = b.new_page(viewport={"width": 1512, "height": 982}, color_scheme="dark")
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(URL, wait_until="networkidle", timeout=60_000)
    pg.wait_for_timeout(1200)

    print(f"\n=== {URL} ===")
    open_sizer(pg)

    # ---- picking a ratio writes the boxes ------------------------------------
    pg.click(".sizer .ars .ar:has(b:text-is('16:9'))")
    pg.wait_for_timeout(150)
    check("16:9 writes the boxes", boxes(pg) == ["1344", "768"], str(boxes(pg)))
    check("the button names the bucket, not the fraction",
          label(pg).startswith("16:9"), label(pg))

    # ---- the swap transposes the bucket, keeping the name and the scale -------
    pg.click(".sizer .sz-swap")
    pg.wait_for_timeout(150)
    check("swap lands on the transposed preset", selected(pg) == "9:16", str(selected(pg)))
    check("swap writes the transposed boxes", boxes(pg) == ["768", "1344"], str(boxes(pg)))

    # 4:3 is the ratio the page opens on and the one whose flip used to land on
    # Custom, which is why 3:4 is on the menu at all.
    pg.click(".sizer .ars .ar:has(b:text-is('4:3'))")
    pg.wait_for_timeout(120)
    pg.click(".sizer .sz-swap")
    pg.wait_for_timeout(150)
    check("4:3 flips into 3:4 rather than Custom", selected(pg) == "3:4", str(selected(pg)))

    # ---- the scale multiplies the bucket -------------------------------------
    pg.click(".sizer .ars .ar:has(b:text-is('16:9'))")
    pg.wait_for_timeout(120)
    pg.click(".sizer .scales .sc:text-is('1.5K')")
    pg.wait_for_timeout(150)
    check("1.5K multiplies the bucket", boxes(pg) == ["2016", "1152"], str(boxes(pg)))
    check("the name survives the scale", selected(pg) == "16:9", str(selected(pg)))

    # A scaled preset still transposes to its counterpart rather than to Custom:
    # transposing the *pixels* would look for 1152x2016 among 1x options, find
    # nothing, and cost you the scale and the name at once.
    pg.click(".sizer .sz-swap")
    pg.wait_for_timeout(150)
    check("a scaled preset keeps its scale through a swap",
          boxes(pg) == ["1152", "2016"], str(boxes(pg)))
    check("a scaled preset keeps its name through a swap",
          selected(pg) == "9:16", str(selected(pg)))

    # ---- typing selects Custom, and is not snapped mid-word -------------------
    w = pg.query_selector(".sizer .sz-custom input")
    w.click(click_count=3)
    pg.keyboard.type("1153")
    pg.wait_for_timeout(150)
    check("typing shows exactly what you typed", boxes(pg)[0] == "1153", str(boxes(pg)))

    pg.keyboard.press("Tab")
    pg.wait_for_timeout(200)
    # Floors to the VAE grid, because the pipeline floors regardless and a box
    # that keeps 1153 while the model renders 1152 is a box that lied to you.
    check("leaving the field snaps to 8", boxes(pg)[0] == "1152", str(boxes(pg)))
    check("a typed size is Custom", selected(pg) is None, str(selected(pg)))
    check("the button says Custom's arithmetic", "·" in label(pg), label(pg))

    # ---- arrows are a faster way to type a number ----------------------------
    w = pg.query_selector(".sizer .sz-custom input")
    w.click(click_count=3)
    pg.keyboard.type("1000")
    pg.keyboard.press("ArrowUp")
    pg.wait_for_timeout(120)
    check("↑ steps by one", boxes(pg)[0] == "1001", str(boxes(pg)))

    pg.keyboard.press(f"{MAC}+ArrowUp")
    pg.wait_for_timeout(120)
    # 8 on Width and Height is the VAE's grid, so the coarse step always lands
    # on a size the model can render.
    check("⌘↑ steps by eight", boxes(pg)[0] == "1009", str(boxes(pg)))

    pg.keyboard.press(f"{MAC}+ArrowDown")
    pg.keyboard.press("ArrowDown")
    pg.wait_for_timeout(120)
    check("the steps are symmetric", boxes(pg)[0] == "1000", str(boxes(pg)))

    if errors:
        check("no page errors", False, str(errors[:3]))
    b.close()

print()
if fails:
    print(f"{len(fails)} failed:", file=sys.stderr)
    for f in fails:
        print(f"  {f}", file=sys.stderr)
    sys.exit(1)
print("size control intact")
