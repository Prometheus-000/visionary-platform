"""How tall does the console actually get, and how much of the viewport is that?"""
from playwright.sync_api import sync_playwright

URL = "http://localhost:8791"
VIEWPORTS = [(1512, 982), (1440, 900), (1728, 1117)]   # 14" MBP, 13" MBP, 16" MBP

LONG_PROMPT = "\n".join(
    "A dancer mid-turn in a white slip dress, backlit by a low sun" for _ in range(12))


def measure(pg, label, rows):
    h = pg.evaluate("() => document.querySelector('.console').getBoundingClientRect().height")
    cv = pg.evaluate("() => document.querySelector('#canvas').getBoundingClientRect().height")
    vh = pg.evaluate("() => innerHeight")
    rows.append((label, round(h), round(cv), h / vh))


with sync_playwright() as pw:
    b = pw.chromium.launch(channel="chrome")
    for vw, vhh in VIEWPORTS:
        pg = b.new_page(viewport={"width": vw, "height": vhh}, color_scheme="dark")
        pg.goto(URL, wait_until="networkidle", timeout=60_000)
        pg.wait_for_timeout(1200)
        rows = []

        pg.click('#kinds button[data-kind="image"]'); pg.wait_for_timeout(300)
        measure(pg, "image · resting", rows)

        pg.click("#toggle-adv"); pg.wait_for_timeout(300)
        measure(pg, "image · Advanced open", rows)

        pg.evaluate("() => document.querySelector('#g-regional').click()")
        pg.wait_for_timeout(400)
        measure(pg, "image · + Regions", rows)

        # Every shot pill the image side can read, to wrap the rail.
        pg.evaluate("""() => {
          shot = shotVocab.flatMap(g => g.items)
                   .slice(0, 14).map(i => ({key: i.key}));
          drawShotRail();
        }""")
        pg.wait_for_timeout(400)
        measure(pg, "image · + full pill rail", rows)

        pg.fill("#prompt", LONG_PROMPT); pg.wait_for_timeout(400)
        measure(pg, "image · + prompt at cap  << WORST", rows)

        pg.fill("#prompt", "a dancer"); pg.wait_for_timeout(200)
        pg.evaluate("() => { shot = []; drawShotRail(); }")
        pg.evaluate("() => document.querySelector('#g-regional').click()")
        pg.click("#toggle-adv"); pg.wait_for_timeout(300)

        pg.click('#kinds button[data-kind="video"]'); pg.wait_for_timeout(500)
        measure(pg, "video · resting", rows)
        pg.click("#v-toggle-adv"); pg.wait_for_timeout(300)
        pg.evaluate("""() => {
          shot = shotVocab.flatMap(g => g.items).slice(0, 16).map(i => ({key: i.key}));
          drawShotRail();
        }""")
        pg.fill("#prompt", LONG_PROMPT); pg.wait_for_timeout(500)
        measure(pg, "video · everything  << WORST", rows)

        print(f"\n=== {vw}x{vhh} ===")
        print(f"  {'state':34} {'console':>8} {'canvas':>8} {'% of viewport':>14}")
        for label, h, cv, frac in rows:
            flag = "  OVER 30%" if frac > 0.30 else ""
            print(f"  {label:34} {h:>6}px {cv:>7}px {frac*100:>12.1f}%{flag}")
        pg.close()
    b.close()
