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
    not a preference. SageAttention is built for the arch list in comfy_image
    — Hopper and Ada now — so a card in this list that is not in that string
    is a run that loads the model and then cannot allocate a kernel. L40S is
    asserted present on every select because it is the one that was kept off
    by a build arg read as a law.
  * **"One container for both" collapses the two selects into one**, and the
    switch confirms before it is made — the trade (a picture behind a clip) is
    stated where it is chosen. A checkbox that silently flipped the class every
    request lands on would be the exact silent sharing the backend rule bans.
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
    gpuBoth: [...document.querySelectorAll('#b-gpu option')].map(o => o.textContent.trim()),
    oneContainer: document.querySelector('#one-container')?.checked ?? null,
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
    check("image GPUs are the compiled set", d["gpuImage"] == ["H100", "H200", "L40S"],
          str(d["gpuImage"]))
    check("video GPUs are the compiled set", d["gpuVideo"] == ["H100", "H200", "L40S"],
          str(d["gpuVideo"]))
    check("two containers at rest", d["oneContainer"] is False and not d["gpuBoth"],
          f"one={d['oneContainer']} both={d['gpuBoth']}")

    # Flip it. The confirm is the point — a declined one must leave the pair
    # standing, an accepted one must collapse them into the shared select.
    dialogs: list[str] = []

    def decline(dlg):
        dialogs.append(dlg.message)
        dlg.dismiss()

    pg.once("dialog", decline)
    pg.evaluate("() => document.querySelector('#one-container').click()")
    pg.wait_for_timeout(200)
    d2 = pg.evaluate(READ)
    check("the switch confirms", bool(dialogs) and "picture" in dialogs[0], str(dialogs[:1]))
    check("declined leaves two selects", d2["oneContainer"] is False and not d2["gpuBoth"],
          f"one={d2['oneContainer']} both={d2['gpuBoth']}")
    pg.once("dialog", lambda dlg: dlg.accept())
    pg.evaluate("() => document.querySelector('#one-container').click()")
    pg.wait_for_timeout(200)
    d3 = pg.evaluate(READ)
    check("accepted collapses to one select",
          d3["oneContainer"] is True and d3["gpuBoth"] == ["H200", "H100", "L40S"]
          and not d3["gpuImage"] and not d3["gpuVideo"],
          f"one={d3['oneContainer']} both={d3['gpuBoth']} image={d3['gpuImage']}")
    # The 48 GB card says what it costs before it is chosen, on every select.
    notes: list[str] = []

    def note(dlg):
        notes.append(dlg.message)
        dlg.dismiss()

    pg.once("dialog", note)
    pg.select_option("#b-gpu", "L40S")
    pg.wait_for_timeout(200)
    check("L40S names its memory", bool(notes) and "48 GB" in notes[0], str(notes[:1])[:120])
    check("declined card stays put", pg.evaluate("() => document.querySelector('#b-gpu').value") == "H200")

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
