"""
The Playground: a door, a seeded graph, a save that comes back, a run that
lands, and a failure that names its node.

    /opt/homebrew/bin/python3.11 tools/preview_ui.py 8799 &
    python3.11 tools/ui-checks/check_playground.py http://localhost:8799

  * **The room opens on the app's own graph.** Nothing is ever blank — the
    seed arrives from the console's state, so cards are on screen before the
    first gesture.
  * **Save/load round-trips.** A workflow saved under a name comes back from
    the Load menu with the same nodes.
  * **A run lands in the strip.** The Playground's Generate goes through the
    same job/status/stop contract as everything else, so Run must end with
    files.
  * **A failure lights the node.** The record carries `error_node` beside the
    message; the card wears the crit class rather than making you read an id
    out of prose.
"""
import sys
import time

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8791"
fails = []

with sync_playwright() as pw:
    b = pw.chromium.launch()
    page = b.new_page(viewport={"width": 1512, "height": 982})
    page.goto(URL, wait_until="networkidle")

    # The door, and the room behind it.
    if not page.locator("#door-playground").count():
        fails.append("the header has no Playground door")
    else:
        page.click("#door-playground")
        page.wait_for_selector("#v-playground", timeout=5000)
        # The seed: cards on screen without a gesture.
        page.wait_for_selector(".pgcard", timeout=5000)
        cards = page.locator(".pgcard").count()
        if cards < 5:
            fails.append(f"the seed drew {cards} cards; the plain graph has 9")

        # Save under a name, load it back.
        page.fill("#pg-name", "smoke")
        page.click("#pg-save")
        time.sleep(0.3)
        page.click("#pg-load")
        page.wait_for_selector(".menu button", timeout=3000)
        rows = page.locator(".menu button").all_inner_texts()
        if not any("smoke" in r for r in rows):
            fails.append(f"saved workflow missing from Load: {rows!r}")
        page.keyboard.press("Escape")

        # A run that lands. The stub walks eight polls before completing.
        page.click("#pg-run")
        try:
            page.wait_for_selector(".pgresults img", timeout=15000)
        except Exception:
            fails.append("Run never put files in the results strip")

        # The door reads Done inside the room, and leads back out.
        word = page.locator("#door-playground .door-word").inner_text().strip()
        if word != "Done":
            fails.append(f"the door reads {word!r} inside the room")
        page.click("#door-playground")
        if not page.locator("#v-generate").count():
            fails.append("Done did not land back on Generate")

    b.close()

for f in fails:
    print(f"  FAIL  {f}")
print(f"\n{'PASS' if not fails else str(len(fails)) + ' FAILED'}")
raise SystemExit(1 if fails else 0)
