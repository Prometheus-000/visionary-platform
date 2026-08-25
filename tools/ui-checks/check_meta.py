"""
The metadata sheet, Reuse and Copy — where the typed prompt earns its keep.

    python3 tools/ui-checks/check_meta.py                        # vanilla
    python3 tools/ui-checks/check_meta.py http://localhost:5173  # React

This is the only surface that shows a prompt at all. The gallery card shows
none — a six-field H3 document is not readable at thumbnail size, and a prompt
is an implementation detail of whichever encoder was being fed that day — so
everything about *which* prompt is shown is decided here.

The rule: `prompt_typed` and `shot` are the durable half, the compiled `prompt`
is a receipt. The sheet leads with the typed one because that is what you
recognise the run by, and Reuse restores the typed one because restoring a
document into the prompt box would compile *that* on the next run — a
double-compiled document, which is not what any prompt meant.

Both are shown when they differ, in that order, because the compiled one is
still the answer to "what did the encoder actually get".

The stub's video cards carry the pair, compiled by the real compiler through
`_from_app`, so this branch runs against the document a run would actually
produce rather than one transcribed into a fixture.
"""
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8791"

fails: list[str] = []


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(f"{label} {detail}".strip())


# A JS click rather than a driver click, for the reason check_viewer.py records:
# the card is off-screen until the drawer is open, and a driver click on an
# off-screen element is a scroll test.
#
# The drawer *is* opened first now, and that is not the same thing as testing it.
# It is closed on load — the canvas is the largest thing on screen — and the cards
# lazy-load their media by intersection, so with it shut no clip has a `<video>`
# for `wantVideo` to find and this file reports six failures for a page that is
# working. Reaching the sheet means reaching the drawer; the check has to do what
# a person does.
# By the sidecar, not by the mounted media. This used to find "a card that
# used the shot palette" by looking for a <video> — a proxy that depends on
# IntersectionObserver having fired, which this harness's browser sometimes
# never delivers (check_gallery already guards the same quirk). The drawer is
# items.slice(0, 24) in listing order, so the listing itself says which card
# is which, and the fact being tested — pills in the sidecar — is the fact
# used to pick the card.
OPEN_MENU = """
async (withPills) => {
  const g = await (await fetch('/api/gallery')).json();
  const idx = g.items.slice(0, 24).findIndex(i =>
    withPills ? (i.shot || []).length > 0
              : !(i.shot || []).length && i.kind === 'image');
  const card = [...document.querySelectorAll('#drawer-grid .gal')][idx];
  if (!card) return false;
  (card.querySelector('[data-act=menu]') || card.querySelector('.more')).click();
  return true;
}
"""
PICK = """
(label) => [...document.querySelectorAll('.menu button')]
  .find(b => (b.textContent || '').trim() === label)?.click()
"""
SHEET = """
() => {
  // The metadata sheet, not Settings — the vanilla page keeps Settings in the
  // DOM and hidden, so `.sheet` alone matches both.
  const box = document.querySelector('#meta-sheet') ||
    [...document.querySelectorAll('.sheet')].find(s => s.querySelector('textarea'));
  if (!box) return null;
  return {
    textareas: [...box.querySelectorAll('textarea')].map(t => (t.value || '').trim()),
    labels: [...box.querySelectorAll('label')].map(l => l.textContent.trim()),
  };
}
"""

with sync_playwright() as pw:
    b = pw.chromium.launch(channel="chrome")
    pg = b.new_page(viewport={"width": 1512, "height": 982}, color_scheme="dark")
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(URL, wait_until="networkidle", timeout=60_000)
    if not pg.evaluate(
        "() => document.querySelector('#t-drawer')?.classList.contains('on')"
    ):
        pg.click("#t-drawer")
    pg.wait_for_timeout(1500)
    pg.wait_for_selector("#drawer-grid .gal", timeout=25_000)

    print(f"\n=== {URL} ===")

    # ---- a card that used the shot palette ---------------------------------
    check("a card offers its menu", pg.evaluate(OPEN_MENU, True))
    pg.wait_for_timeout(500)
    pg.evaluate(PICK, "View metadata")
    pg.wait_for_timeout(800)

    d = pg.evaluate(SHEET)
    check("the sheet opens", d is not None)
    if d:
        tas = d["textareas"]
        check("it shows a prompt", len(tas) >= 1, f"{len(tas)} textareas")
        # Typed first. This is the whole rule: the document is what ran, the
        # sentence is what you meant, and you recognise the run by the sentence.
        check("the typed sentence leads",
              bool(tas) and "walks out of the shop" in tas[0],
              repr(tas[0][:70]) if tas else "")
        check("the typed one is not the document",
              bool(tas) and "integrated_multimodal_description" not in tas[0],
              repr(tas[0][:70]) if tas else "")
        # And the receipt is still reachable, because "what did the encoder get"
        # is a real question with a different answer.
        check("the compiled document is shown too",
              any("integrated_multimodal_description" in t for t in tas),
              str([t[:40] for t in tas]))
        # non_diegetic_music: N/A is the default and is worth the feature on its
        # own — H3 invented a soundtrack for every clip until something said not to.
        check("the document carries the soundtrack default",
              any("non_diegetic_music" in t for t in tas),
              str([t[:40] for t in tas]))

    check("Reuse is offered", pg.query_selector("#m-reuse") is not None)
    check("Copy is offered", pg.query_selector("#m-copy") is not None)

    # ---- Reuse puts the typed sentence back, never the document ------------
    pg.evaluate("() => document.querySelector('#m-reuse')?.click()")
    pg.wait_for_timeout(900)
    box = pg.evaluate("() => document.querySelector('#prompt')?.value || ''")
    check("Reuse restores the typed sentence", "walks out of the shop" in box, repr(box[:70]))
    check("Reuse does not restore the document",
          "integrated_multimodal_description" not in box, repr(box[:70]))
    # The pills come back with it — they are half of what "intent" means, and a
    # sentence restored without them recompiles to something else entirely.
    pills = pg.evaluate("""() => document.querySelectorAll('#shot-rail .spill').length""")
    check("and the shot pills come back with it", pills >= 3, f"{pills} pills")

    # ---- a card that never used it still works -----------------------------
    # No pills, no document: the compiler returns the typed text byte-for-byte,
    # so an older card has only `prompt` and the sheet must not go blank.
    pg.reload(wait_until="networkidle")
    # A reload shuts the drawer, and a shut drawer renders no cards now — so it
    # is reopened the way a person would reopen it.
    if not pg.evaluate(
        "() => document.querySelector('#t-drawer')?.classList.contains('on')"
    ):
        pg.click("#t-drawer")
    pg.wait_for_timeout(1500)
    pg.wait_for_selector("#drawer-grid .gal", timeout=25_000)
    pg.evaluate(OPEN_MENU, False)
    pg.wait_for_timeout(500)
    pg.evaluate(PICK, "View metadata")
    pg.wait_for_timeout(800)
    d2 = pg.evaluate(SHEET)
    check("a card with no typed prompt still shows one",
          bool(d2 and d2["textareas"] and d2["textareas"][0]),
          repr((d2 or {}).get("textareas", [""])[0][:60]))

    if errors:
        check("no page errors", False, str(errors[:2]))
    b.close()

print()
if fails:
    print(f"{len(fails)} failed:", file=sys.stderr)
    for f in fails:
        print(f"  {f}", file=sys.stderr)
    sys.exit(1)
print("metadata sheet intact")
