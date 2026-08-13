"""
The ids the page addresses itself by, compared across both front ends.

    python3 tools/ui-checks/probe_ids.py

Runs against both at once and diffs, because the interesting failures here are
*differences*, not absolutes.

Two rules, and the second is the one that bites:

**No id appears twice.** Both composer strips are in the DOM at once with one of
them merely `.hide`, so a control rendered by a component shared between them
gets its id twice. `querySelector` then returns the image one forever and the
video button is unreachable — which looks like a dead button, not like invalid
markup.

**Ids missing from React are reported, never failed.** The first version failed
on them and was useless: it listed all 130 ids of every surface not ported yet
and would have stayed red for weeks, which is a progress tracker wearing a
test's clothes. Worse, a check that is always red is one nobody reads, so the
duplicate it *can* catch would scroll past inside it.

Some absences are also correct and permanent. React has no `#g-w`, `#g-h`,
`#g-aspect`, `#g-scale` or `#g-swap` because those are the vanilla page's
*hidden canonical state* and the store replaces them — the same split that made
⌘↑ never work on the visible size boxes, since the delegated handler reached the
hidden pair and not the popover's. Others are conditional rendering: React does
not render `#neg-toggle` at all where the vanilla page renders it and adds
`.hide`, and both mean "this model reads no negative".

So: duplicates fail, everything else is a report with counts.
"""
import sys
from collections import Counter

from playwright.sync_api import sync_playwright

# Retired with UI_HTML: this diffed the two front ends id by id, and there is
# only one now. Kept out of the suite rather than rewritten — probe_ids answered
# "did the port drop a control", which is a question with no subject any more.
VANILLA = "http://localhost:8791"
REACT = "http://localhost:5173"

# Vanilla-only by design and permanently so — the store is the canonical state
# in React, so the hidden inputs the sizer and sampling popovers are a *view
# over* simply do not exist. Listed apart from the not-yet-ported ones so the
# report distinguishes "will never appear" from "has not been built yet".
BY_DESIGN = {
    "g-w", "g-h", "g-aspect", "g-scale", "g-swap",          # sizer's hidden state
    "g-model", "g-steps", "g-cfg", "g-shift", "g-seed",     # sampling's hidden state
    "g-sampler", "g-scheduler", "g-n",
    "v-model", "v-steps", "v-cfg", "v-shift", "v-seed",
    "v-sampler", "v-scheduler", "v-switch", "v-tier", "v-aspect",
    "neg-toggle", "neg",                                    # rendered only when read
    "v-add-lora",                                           # Wan only
    "g-region-base-wrap",
}

fails: list[str] = []


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(f"{label} {detail}".strip())


def ids_of(pg):
    return pg.evaluate("() => [...document.querySelectorAll('[id]')].map(e => e.id)")


with sync_playwright() as pw:
    b = pw.chromium.launch(channel="chrome")
    seen = {}
    for tag, url in (("vanilla", VANILLA), ("react", REACT)):
        pg = b.new_page(viewport={"width": 1512, "height": 982}, color_scheme="dark")
        pg.goto(url, wait_until="networkidle", timeout=60_000)
        pg.wait_for_timeout(1400)
        # Arm the states that render more of the console, so their ids count too.
        for sel in ("#g-regional",):
            el = pg.query_selector(sel)
            if el:
                el.click()
                pg.wait_for_timeout(400)
        seen[tag] = ids_of(pg)
        pg.close()
    b.close()

print()
for tag in ("vanilla", "react"):
    dupes = {k: v for k, v in Counter(seen[tag]).items() if v > 1}
    check(f"{tag}: every id is unique", not dupes, str(dupes))

v, r = set(seen["vanilla"]), set(seen["react"])

# A report, not an assertion. See the note at the top.
design = sorted((v - r) & BY_DESIGN)
todo = sorted((v - r) - BY_DESIGN)
extra = sorted(r - v)

print(f"\n  vanilla-only, by design ({len(design)}): {', '.join(design) or 'none'}")
print(f"  vanilla-only, not ported yet ({len(todo)})")
if todo:
    # Grouped by prefix, because 130 flat ids is a wall rather than a list and
    # the prefixes are the surfaces: ds- is datasets, a- the hyperparameters.
    groups: dict[str, list[str]] = {}
    for i in todo:
        groups.setdefault(i.split("-")[0] if "-" in i else i, []).append(i)
    for k in sorted(groups, key=lambda k: (-len(groups[k]), k)):
        if len(groups[k]) > 1:
            print(f"      {k}-*  ({len(groups[k])})")
    loose = [i for k, g in groups.items() if len(g) == 1 for i in g]
    if loose:
        print(f"      singles: {', '.join(sorted(loose))}")
print(f"  react-only ({len(extra)}): {', '.join(extra) or 'none'}")

print()
if fails:
    print(f"{len(fails)} failed:", file=sys.stderr)
    for f in fails:
        print(f"  {f}", file=sys.stderr)
    sys.exit(1)
print("ids intact")
