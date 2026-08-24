"""
What the canvas says after a render, asserted against either front end.

    python3 tools/ui-checks/check_render.py                        # vanilla
    python3 tools/ui-checks/check_render.py http://localhost:5173  # React

The caption under the stills is the whole surface for "what did this run
actually do", and every field in it comes off the job record by a name that is
easy to guess wrong. Writing the React version I reached for `seconds` and
`seed`; the record says `duration_s` and `seeds` — a list, because a batch has
one per image. Both wrong guesses render as an empty caption rather than an
error, which is exactly the failure this file exists to make loud.

The other assertion is the one that cost real debugging time upstream:
`applied === false`, never `!applied`. The image report emits no `applied` key
at all, so a falsy test marks every LoRA on every render as unapplied — a
warning that is always on, pointing at healthy LoRAs while the actual fault is
elsewhere. The stub carries one applied LoRA and one that is not, so a
regression to the falsy test shows up here as two names instead of one.
"""
import re
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8791"

fails: list[str] = []


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(f"{label} {detail}".strip())


# Typed through the native setter so React's onChange sees it; the vanilla page
# is happy either way.
TYPE = """
(text) => {
  const ta = document.querySelector('#prompt');
  const set = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set;
  set.call(ta, text);
  ta.dispatchEvent(new Event('input', {bubbles: true}));
}
"""

with sync_playwright() as pw:
    b = pw.chromium.launch(channel="chrome")
    pg = b.new_page(viewport={"width": 1440, "height": 900}, color_scheme="dark")
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(URL, wait_until="networkidle", timeout=60_000)
    pg.wait_for_timeout(1200)

    print(f"\n=== {URL} ===")
    pg.evaluate(TYPE, "a dancer in soft window light")
    pg.wait_for_timeout(200)
    pg.evaluate("() => document.querySelector('#go-gen').click()")

    # The stub takes a few seconds and steps its way there, same as a real run.
    pg.wait_for_selector("#gen-out .shot img", timeout=40_000)
    pg.wait_for_function(
        "() => { const i = document.querySelector('#gen-out .shot img');"
        " return i && i.naturalWidth > 0 }", timeout=40_000)
    pg.wait_for_timeout(600)

    d = pg.evaluate("""() => {
      const g = document.querySelector('#gen-out');
      const shots = [...document.querySelectorAll('#gen-out .shot')];
      const meta = document.querySelector('#gen-meta');
      return {
        shots: shots.length,
        painted: shots.map(s => s.querySelector('img')?.naturalWidth || 0),
        acts: shots[0]?.querySelectorAll('.acts button').length ?? 0,
        cols: g?.style.gridTemplateColumns || '',
        maxW: g?.style.maxWidth || '',
        shotH: g?.style.getPropertyValue('--shot-h') || '',
        meta: (meta?.textContent || '').trim(),
        warn: (meta?.querySelector('.warn')?.textContent || '').trim(),
      };
    }""")

    check("stills rendered", d["shots"] > 0, f"{d['shots']} shots")
    # Streamed by filename rather than inlined as base64: each <img> paints as
    # its own bytes land, off the same route and cache the gallery uses.
    check("every still actually painted", all(w > 0 for w in d["painted"]), str(d["painted"]))
    check("each still carries both handoffs", d["acts"] == 2, f"{d['acts']} buttons")

    # A batch is two columns; a single still gets the canvas. Both are set from
    # the measured fit rather than by CSS, so an empty --shot-h means the stills
    # are sized by nothing and will run under the console.
    two = d["shots"] > 1
    check("grid matches the batch size",
          ("repeat(2" in d["cols"]) == two, d["cols"])
    check("the fit was measured", bool(re.match(r"-?\d", d["shotH"].strip())), repr(d["shotH"]))

    # The caption. Field names live or die here.
    check("seeds are named", "seed" in d["meta"], d["meta"][:90])
    check("the sampler line is there", "steps" in d["meta"] and "CFG" in d["meta"], d["meta"][:90])
    check("the duration is there", bool(re.search(r"\d+(\.\d+)?s\b", d["meta"])), d["meta"][:90])

    # Exactly one unapplied LoRA in the stub. Two would mean the falsy test is
    # back and the healthy one is being reported too.
    check("the unapplied LoRA is named", "not applied" in d["warn"], repr(d["warn"]))
    check("only the unapplied one is named",
          "my_style" not in d["warn"] and "gone" in d["warn"], repr(d["warn"]))
    # Not "(" any more: skipNote deliberately retired the parenthesized
    # `gone (no matching keys)` form for a quoted name and a translated
    # cause — see its comment. Assert the cause, not the punctuation.
    check("the reason rides along", "keys don" in d["warn"], repr(d["warn"]))

    if errors:
        check("no page errors", False, str(errors[:2]))
    b.close()

print()
if fails:
    print(f"{len(fails)} failed:", file=sys.stderr)
    for f in fails:
        print(f"  {f}", file=sys.stderr)
    sys.exit(1)
print("render caption intact")
