"""
⌥← / ⌥→ — moving the clause under the caret one slot along.

    python3 tools/ui-checks/probe_clause.py http://localhost:8791
    python3 tools/ui-checks/probe_clause.py http://localhost:5173

The rule that makes this safe to press repeatedly: **the separators are slots
and they do not move.** The commas and line breaks stay exactly where they are
and the text between them changes places, so a prompt written across two lines
still has two lines however many times you press the chord, and a prompt with
one comma still has one comma.

At the ends, and against an empty slot — a trailing comma is not a clause — it
must decline rather than quietly doing nothing, so the key falls through to the
OS word-jump. That is asserted here as "the text is unchanged", which is the
observable half of the same thing.
"""
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8791"
ALT = "Alt"

fails: list[str] = []


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(f"{label} {detail}".strip())


SET = """
([text, caret]) => {
  const ta = document.querySelector('#prompt');
  const set = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set;
  set.call(ta, text);
  ta.dispatchEvent(new Event('input', {bubbles: true}));
  ta.focus();
  ta.setSelectionRange(caret, caret);
}
"""
GET = """() => document.querySelector('#prompt').value"""


def move(pg, text, caret, key):
    pg.evaluate(SET, [text, caret])
    pg.wait_for_timeout(120)
    pg.keyboard.press(f"{ALT}+{key}")
    pg.wait_for_timeout(160)
    return pg.evaluate(GET)


with sync_playwright() as pw:
    b = pw.chromium.launch(channel="chrome")
    pg = b.new_page(viewport={"width": 1512, "height": 982}, color_scheme="dark")
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(URL, wait_until="networkidle", timeout=60_000)
    pg.wait_for_timeout(1200)

    print(f"\n=== {URL} ===")

    base = "a portrait, in soft window light, 35mm"

    # Caret in the second clause, moved left: the two swap and the commas stay.
    got = move(pg, base, 16, "ArrowLeft")
    check("⌥← swaps the clause with the one before it",
          got == "in soft window light, a portrait, 35mm", repr(got))

    got = move(pg, base, 16, "ArrowRight")
    check("⌥→ swaps it with the one after",
          got == "a portrait, 35mm, in soft window light", repr(got))

    # The ends decline rather than wrapping.
    got = move(pg, base, 3, "ArrowLeft")
    check("⌥← on the first clause declines", got == base, repr(got))
    got = move(pg, base, 33, "ArrowRight")
    check("⌥→ on the last clause declines", got == base, repr(got))

    # Line breaks are slots too, and they stay put. The invariant is not "the
    # text is unchanged" — the whole point is that it changes — it is that the
    # *separator sequence* is unchanged: same characters, same order, same
    # count. A prompt written across two lines still has two lines, and a prompt
    # with one comma still has one comma, however many times you press.
    seps = "() => [...document.querySelector('#prompt').value].filter(c => c === ',' || c === '\\n').join('')"
    two = "a portrait, in soft light\n35mm, shallow focus"
    before = pg.evaluate(SET, [two, 30]) or pg.evaluate(seps)
    got = move(pg, two, 30, "ArrowLeft")
    after = pg.evaluate(seps)
    check("the separators do not move", after == before, f"{before!r} -> {after!r}")
    check("the clauses did swap", got != two, repr(got))

    # A trailing comma is a separator with nothing after it, not a clause.
    trail = "a portrait, in soft light,"
    got = move(pg, trail, 20, "ArrowRight")
    check("an empty slot is not a destination", got == trail, repr(got))

    # Leading and trailing whitespace inside a slot belongs to that slot, so a
    # clause moving into first position does not drag the old space with it.
    spaced = "a portrait,   in soft light, 35mm"
    got = move(pg, spaced, 18, "ArrowLeft")
    check("no double space is left behind", "  in soft light" not in got.strip(), repr(got))

    if errors:
        check("no page errors", False, str(errors[:3]))
    b.close()

print()
if fails:
    print(f"{len(fails)} failed:", file=sys.stderr)
    for f in fails:
        print(f"  {f}", file=sys.stderr)
    sys.exit(1)
print("clause moves intact")
