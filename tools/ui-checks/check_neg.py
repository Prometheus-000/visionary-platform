"""
The negative-prompt toggle, and the field that grows and then stops.

    python3 tools/ui-checks/check_neg.py                        # vanilla
    python3 tools/ui-checks/check_neg.py http://localhost:5173  # React

Takes a URL for the reason check_viewer.py does. Getting there meant giving up
`window.kind` and `setKind('video')`: a bundled front end has no globals, so the
side is switched with the chip inside the prompt field and the model is picked in
the Sampling popover, which is where both front ends keep it. Reading the state
through the DOM the way a person reads it is also what makes the two answers
comparable at all.

**`toggleHidden` accepts absent as well as hidden**, and that is a real difference
rather than a fudge: the vanilla page renders `#neg-toggle` always and adds
`.hide`, React does not render it. Both mean "this model reads no negative", which
is the thing being checked — asserting on the mechanism would fail one
implementation for a structural reason, which the README already names as worse
than no check.
"""
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8791"
fails = []


def check(label, got, want):
    ok = got == want
    print(("  ok   " if ok else "  FAIL ") + f"{label}: {got!r}" + ("" if ok else f" != {want!r}"))
    if not ok:
        fails.append(label)


with sync_playwright() as pw:
    b = pw.chromium.launch(channel="chrome")
    pg = b.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=1,
                    color_scheme="dark")
    errs = []
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(URL, wait_until="networkidle", timeout=60_000)
    pg.wait_for_timeout(1200)
    print(f"\n=== {URL} ===")

    state = lambda: pg.evaluate("""() => {
        const t = document.querySelector('#neg-toggle');
        return {
          imageSide: !document.querySelector('#c-image').classList.contains('hide'),
          toggleHidden: !t || t.classList.contains('hide'),
          toggleText: t ? t.textContent.trim() : '',
          onNeg: document.querySelector('.field').classList.contains('on-neg'),
          filled: !!t && t.classList.contains('filled'),
          promptH: Math.round(document.querySelector('#prompt').getBoundingClientRect().height),
          resize: getComputedStyle(document.querySelector('#prompt')).resize,
          advHasNeg: !!document.querySelector('#gen-adv'),
        };
    }""")

    def side(want_image):
        """The chip inside the prompt field, which is the only route a person has."""
        if state()["imageSide"] != want_image:
            pg.click("#kind-toggle")
            pg.wait_for_timeout(500)

    def sampling(prefix):
        """Open the Sampling popover for this side, where the model select lives."""
        pg.click(f"#{prefix}-sampling")
        pg.wait_for_selector(".menu.form")
        pg.wait_for_timeout(200)

    def pick_model(prefix, label_substr):
        sampling(prefix)
        row = pg.locator(".menu.form .frow", has_text="Model").first
        sel = row.locator("select")
        value = sel.locator("option", has_text=label_substr).first.get_attribute("value")
        sel.select_option(value)
        pg.wait_for_timeout(350)
        pg.keyboard.press("Escape")
        pg.wait_for_timeout(250)

    print("\nimage / Krea 2 Turbo (CFG 1.0 — reads no negative)")
    side(True)
    pick_model("g", "Turbo")
    s = state()
    check("toggle hidden", s["toggleHidden"], True)
    check("the Advanced drawer is gone", s["advHasNeg"], False)
    check("resize grip gone", s["resize"], "none")
    print(f"  (one-row height {s['promptH']}px)")

    print("\nimage / Krea 2 RAW (CFG 5.5 — reads one)")
    pick_model("g", "RAW")
    check("toggle shown", state()["toggleHidden"], False)
    check("names the field you are in", state()["toggleText"], "positive")

    print("\nswitching into negative mode")
    pg.click("#neg-toggle")
    pg.wait_for_timeout(250)
    s = state()
    check("field marked", s["onNeg"], True)
    check("renames itself on switch", s["toggleText"], "negative")
    pg.fill("#neg", "blurry, watermark")
    pg.wait_for_timeout(200)
    pg.click("#neg-toggle")
    pg.wait_for_timeout(250)
    check("back on the prompt", state()["onNeg"], False)
    check("dot says the other side has text", state()["filled"], True)

    print("\ntyped CFG re-enables it on turbo")
    pick_model("g", "Turbo")
    check("hidden at CFG 1.0", state()["toggleHidden"], True)
    sampling("g")                      # CFG lives behind here now
    pg.locator(".menu.form .frow", has_text="CFG").first.locator("input").fill("5")
    pg.wait_for_timeout(350)
    check("back at CFG 5", state()["toggleHidden"], False)
    pg.click(".menu.form .sz-reset")
    pg.wait_for_timeout(350)
    check("hidden again at CFG blank", state()["toggleHidden"], True)

    print("\nvideo / MiniMax-H3 (guidance-distilled)")
    side(False)
    pick_model("v", "MiniMax-H3")
    check("toggle hidden on H3", state()["toggleHidden"], True)
    print("\nvideo / Wan 2.2 A14B (takes CFG)")
    pick_model("v", "A14B")
    check("toggle shown on Wan", state()["toggleHidden"], False)

    print("\nthe field grows and then stops")
    side(True)
    pg.fill("#prompt", "one line")
    pg.wait_for_timeout(250)
    h1 = state()["promptH"]
    pg.fill("#prompt", "\n".join(f"line {i}" for i in range(4)))
    pg.wait_for_timeout(300)
    h4 = state()["promptH"]
    pg.fill("#prompt", "\n".join(f"line {i}" for i in range(40)))
    pg.wait_for_timeout(350)
    h40 = state()["promptH"]
    print(f"  1 line {h1}px -> 4 lines {h4}px -> 40 lines {h40}px")
    check("grows with the text", h4 > h1, True)
    # 170, not 168: the cap is `min(FIELD_CEIL, innerHeight*0.30 - other)` and what is
    # asserted here is that it stops, not where — probe_console.py is what pins the
    # formula, because the formula is the invariant and this number is a symptom of it.
    check("capped at FIELD_CEIL", h40 <= 170, True)
    check("scrolls past the cap",
          pg.evaluate("() => { const e=document.querySelector('#prompt');"
                      " return e.scrollHeight > e.clientHeight }"),
          True)

    real = [e for e in errs if "favicon" not in e.lower() and "404" not in e]
    print(f"\nconsole errors: {real or 'none'}")
    if real:
        fails.append("console errors")
    b.close()

print("\n" + ("FAILED: " + ", ".join(fails) if fails else "all checks passed"))
sys.exit(1 if fails else 0)
