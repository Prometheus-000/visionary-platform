"""Exercise the negative-prompt toggle and the auto-growing field."""
from playwright.sync_api import sync_playwright

URL = "http://localhost:8791"
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

    def pick_model(sel, val):
        pg.evaluate("""([s, v]) => {
            const el = document.querySelector(s);
            el.value = v;
            el.dispatchEvent(new Event('change', {bubbles: true}));
        }""", [sel, val])

    state = lambda: pg.evaluate("""() => ({
        model: document.querySelector('#g-model').value,
        kind: window.kind,
        toggleHidden: document.querySelector('#neg-toggle').classList.contains('hide'),
        toggleText: document.querySelector('#neg-toggle').textContent,
        onNeg: document.querySelector('.field').classList.contains('on-neg'),
        filled: document.querySelector('#neg-toggle').classList.contains('filled'),
        promptH: Math.round(document.querySelector('#prompt').getBoundingClientRect().height),
        resize: getComputedStyle(document.querySelector('#prompt')).resize,
        advHasNeg: !!document.querySelector('#gen-adv'),
    })""")

    print("\nimage / Krea 2 Turbo (CFG 1.0 — reads no negative)")
    pick_model("#g-model", "turbo"); pg.wait_for_timeout(300)
    s = state()
    check("toggle hidden", s["toggleHidden"], True)
    check("the Advanced drawer is gone", s["advHasNeg"], False)
    check("resize grip gone", s["resize"], "none")
    one_row = s["promptH"]
    print(f"  (one-row height {one_row}px)")

    print("\nimage / Krea 2 RAW (CFG 5.5 — reads one)")
    pick_model("#g-model", "raw"); pg.wait_for_timeout(300)
    check("toggle shown", state()["toggleHidden"], False)
    check("names the field you are in", state()["toggleText"].strip(), "positive")

    print("\nswitching into negative mode")
    pg.click("#neg-toggle"); pg.wait_for_timeout(250)
    s = state()
    check("field marked", s["onNeg"], True)
    check("renames itself on switch", s["toggleText"].strip(), "negative")
    pg.fill("#neg", "blurry, watermark")
    pg.wait_for_timeout(200)
    pg.click("#neg-toggle"); pg.wait_for_timeout(250)
    check("back on the prompt", state()["onNeg"], False)
    check("dot says the other side has text", state()["filled"], True)

    print("\ntyped CFG re-enables it on turbo")
    pick_model("#g-model", "turbo"); pg.wait_for_timeout(250)
    check("hidden at CFG 1.0", state()["toggleHidden"], True)
    pg.click("#g-sampling"); pg.wait_for_timeout(250)   # CFG lives behind here now
    pg.fill('.menu.form [data-s="#g-cfg"]', "5"); pg.wait_for_timeout(300)
    check("back at CFG 5", state()["toggleHidden"], False)
    pg.click(".menu.form .sz-reset"); pg.wait_for_timeout(300)
    check("hidden again at CFG blank", state()["toggleHidden"], True)

    print("\nvideo / MiniMax-H3 (guidance-distilled)")
    pg.evaluate("()=>setKind('video')"); pg.wait_for_timeout(400)
    pick_model("#v-model", "h3"); pg.wait_for_timeout(400)
    check("toggle hidden on H3", state()["toggleHidden"], True)
    print("\nvideo / Wan 2.2 A14B (takes CFG)")
    pick_model("#v-model", "wan14b"); pg.wait_for_timeout(400)
    check("toggle shown on Wan", state()["toggleHidden"], False)

    print("\nthe field grows and then stops")
    pg.evaluate("()=>setKind('image')"); pg.wait_for_timeout(300)
    pg.fill("#prompt", "one line")
    pg.wait_for_timeout(200)
    h1 = state()["promptH"]
    pg.fill("#prompt", "\n".join(f"line {i}" for i in range(4)))
    pg.wait_for_timeout(250)
    h4 = state()["promptH"]
    pg.fill("#prompt", "\n".join(f"line {i}" for i in range(40)))
    pg.wait_for_timeout(300)
    h40 = state()["promptH"]
    print(f"  1 line {h1}px -> 4 lines {h4}px -> 40 lines {h40}px")
    check("grows with the text", h4 > h1, True)
    check("capped at FIELD_MAX", h40 <= 170, True)
    check("scrolls past the cap",
          pg.evaluate("() => { const e=document.querySelector('#prompt'); return e.scrollHeight > e.clientHeight }"),
          True)

    real = [e for e in errs if "favicon" not in e.lower()
            and "404" not in e]   # the stub server has no favicon route
    print(f"\nconsole errors: {real or 'none'}")
    if real:
        fails.append("console errors")
    b.close()

print("\n" + ("FAILED: " + ", ".join(fails) if fails else "all checks passed"))
