"""
What Settings says about the catalogue, asserted against either front end.

    python3 tools/ui-checks/check_settings.py                        # vanilla
    python3 tools/ui-checks/check_settings.py http://localhost:5173  # React

Settings is where weights arrive and where they are deleted, so the things
worth pinning are the ones whose failure costs bandwidth or a file:

  * **A family one file short shows no button at all**, because that is what
    its own Download already is. Two buttons doing overlapping things, one of
    them almost always the wrong scope, is worse than one.
  * **There is no catalogue-wide Download.** It existed, in the token row,
    which put the one button that pulls every family — including the ones an
    install will never run — next to a password field it has nothing to do
    with. A check is the only thing that stops it being re-added as an
    obvious convenience.
  * **The GPU options are a property of what the images were compiled for**,
    not a preference. SageAttention is built for sm_90, so an A100 in this
    list is a run that loads the model and then cannot allocate a kernel.
  * The LoRA total, because it is the one number that says what the volume is
    actually holding.
"""
import re
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8791"

READ = """
() => {
  const fams = [...document.querySelectorAll('#models .fam')].map(f => {
    // Rows in one card per family now, not a card per model; a complete
    // family is folded and holds no rows at all, which the checks below are
    // fine with — its head still carries the note they read.
    const rows = [...f.querySelectorAll('.mrow')];
    // `.fam-dl`, not the head's first button — the fold toggle is a button
    // too, and it must never be mistaken for a 17 GB download.
    const famBtn = f.querySelector('.fam-head .fam-dl');
    return {
      name: f.querySelector('b')?.textContent?.trim(),
      note: f.querySelector('.fam-head .muted')?.textContent?.trim(),
      models: rows.length,
      // A row with a button is a weight that is not on the volume.
      missing: rows.filter(c => c.querySelector('button')).length,
      famButton: famBtn ? famBtn.textContent.trim() : null,
    };
  });
  return {
    families: fams,
    loraRows: document.querySelectorAll('#lora-list .lora-row').length,
    loraTotal: document.querySelector('#lora-total')?.textContent?.trim(),
    gpuImage: [...document.querySelectorAll('#g-gpu option')].map(o => o.textContent.trim()),
    gpuVideo: [...document.querySelectorAll('#v-gpu option')].map(o => o.textContent.trim()),
    // The button that must not come back.
    catalogueWide: [...document.querySelectorAll('button')]
      .map(b => (b.textContent || '').trim().toLowerCase())
      .filter(t => t === 'download missing' || t === 'download all missing'),
  };
}
"""

fails: list[str] = []


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(f"{label} {detail}".strip())


with sync_playwright() as pw:
    b = pw.chromium.launch(channel="chrome")
    pg = b.new_page(viewport={"width": 1440, "height": 900}, color_scheme="dark")
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(URL, wait_until="networkidle", timeout=60_000)
    pg.wait_for_timeout(1200)

    print(f"\n=== {URL} ===")
    pg.evaluate("() => document.querySelector('#t-settings').click()")
    pg.wait_for_selector("#models .fam", timeout=15_000)
    pg.wait_for_timeout(400)
    d = pg.evaluate(READ)

    check("families are grouped", len(d["families"]) > 0, f"{len(d['families'])} families")
    for f in d["families"]:
        n = f["missing"]
        if n > 1:
            check(f"{f['name']}: {n} missing offers one button",
                  f["famButton"] == f"Download all {n}", str(f["famButton"]))
        else:
            # The whole rule: one short is already served by its own Download.
            check(f"{f['name']}: {n} missing offers no family button",
                  f["famButton"] is None, str(f["famButton"]))
        check(f"{f['name']}: the note names the cost",
              bool(f["note"]) and ("missing" in f["note"] or f["note"] == "complete"),
              str(f["note"]))

    check("no catalogue-wide Download", not d["catalogueWide"], str(d["catalogueWide"]))
    check("image GPUs are Hopper only", d["gpuImage"] == ["H100", "H200"], str(d["gpuImage"]))
    check("video GPUs are Hopper only", d["gpuVideo"] == ["H100", "H200"], str(d["gpuVideo"]))

    # Not just "is there a total". `bool(total)` passed while the two front ends
    # printed 5.85 GB and 5.4 GB for the same eight files — a 1024-based
    # formatter with one decimal, which is defensible in isolation and wrong
    # beside a catalogue whose own sizes are decimal with two. The size unit is
    # a contract, so it is asserted as one.
    total = d["loraTotal"] or ""
    size = total.split("·")[-1].strip()
    check("the LoRA list totals itself", bool(total),
          f"{d['loraRows']} rows · {total}")
    check("sizes are decimal GB to two places (or whole MB)",
          bool(re.fullmatch(r"\d+\.\d{2} GB|\d+ MB", size)), repr(size))

    if errors:
        check("no page errors", False, str(errors[:2]))
    b.close()

print()
if fails:
    print(f"{len(fails)} failed:", file=sys.stderr)
    for f in fails:
        print(f"  {f}", file=sys.stderr)
    sys.exit(1)
print("settings intact")
