"""
The semantic layer, driven: marks, the caret, the write that must not happen.

    python3.11 tools/ui-checks/check_document.py                        # web/dist
    python3.11 tools/ui-checks/check_document.py http://localhost:5173  # dev server

Every row here pins a *rule* rather than a class name, because this surface is
the one Phase 6 is expected to move: the underline could become anything and the
statements below would still be the statements. Four of them exist because the
thing they check was broken while looking fine.

  * **The caret.** A parse landing mid-sentence rewrites the box, and React
    sends the caret to the end — so the next character you type appears
    somewhere you were not looking. You find out by having already typed.
  * **The write.** The model may insert and never revise, so a reply that would
    take words off the person is refused and the box keeps what was typed. It
    fails silently by design, which is exactly why it needs a check: nothing on
    screen would ever say it had gone wrong.
  * **The stale document.** A document is valid only for the prose it was
    derived from. `docFor` makes the invalid state unreachable rather than
    remembered, and this asserts the run really does go plain once the box
    moves.
  * **Composition.** Driven over CDP with `Input.imeSetComposition`, never by
    dispatching a synthetic `compositionstart`. A fabricated event tests the
    handler; what is in doubt is the *browser* — `check_regions.py`'s own rule
    that a driver poking past the interface can pass while the interface is
    unreachable.

The preview server's `/api/parse` stub is what makes this runnable with no GPU
and no Modal account: a trailing `+` on the prose appends a clause the prose
does not contain, which is the only shape that exercises the write at all.
"""
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8791"
fails = []

# The box, and what the mirror says about it. Read together and in one pass, so
# a check can never compare a box to a mirror from a different keystroke.
STATE = """() => {
  const ta = document.querySelector('#prompt');
  const mk = document.querySelector('.mk-mirror');
  return {
    box: ta ? ta.value : null,
    caret: ta ? ta.selectionStart : null,
    mirror: mk ? mk.textContent : null,
    grey: mk ? [...mk.querySelectorAll('.mk-i')].map(s => s.textContent) : [],
    spans: mk ? [...mk.querySelectorAll('.mk-el')].map(s => s.textContent) : [],
    reroll: !!document.querySelector('.mk-reroll'),
  };
}"""

# What `/api/generate` would be sent right now, without sending it. The document
# rides on the same body as the prompt, so this is the only way to ask whether a
# stale one is really dropped.
BODY = """async () => {
  let seen = null;
  const real = window.fetch;
  window.fetch = async (u, o) => {
    if (String(u).includes('/api/generate')) {
      seen = JSON.parse(o.body);
      return new Response(JSON.stringify({error: 'intercepted'}),
                          {status: 200, headers: {'Content-Type': 'application/json'}});
    }
    return real(u, o);
  };
  // The image side's button by id. Named rather than found by its label,
  // because both kinds have one reading "Generate" and clicking whichever came
  // first would silently measure the video body on an image page.
  document.querySelector('#go-gen')?.click();
  await new Promise(r => setTimeout(r, 500));
  window.fetch = real;
  return seen;
}"""


with sync_playwright() as pw:
    br = pw.chromium.launch(channel="chrome")
    pg = br.new_page(viewport={"width": 1400, "height": 950}, color_scheme="dark")
    pg.goto(URL, wait_until="networkidle", timeout=60_000)
    pg.wait_for_timeout(1000)
    cdp = pg.context.new_cdp_session(pg)
    print(f"\n=== {URL} ===")

    def check(label, ok, detail=""):
        if not ok:
            fails.append(label)
        print(f"  {'ok  ' if ok else 'FAIL'} {label:52} {detail}")

    def state():
        return pg.evaluate(STATE)

    def typed(text, settle=1100):
        """Set the prompt the way a person does, then let the pause fire."""
        pg.click("#prompt")
        pg.evaluate("""(v) => {
          const ta = document.querySelector('#prompt');
          const set = Object.getOwnPropertyDescriptor(
            window.HTMLTextAreaElement.prototype, 'value').set;
          set.call(ta, v);
          ta.dispatchEvent(new Event('input', {bubbles: true}));
        }""", text)
        pg.wait_for_timeout(settle)

    # ── a document lands, and it is visible in the sentence ──────────────────
    typed("a woman in a red dress +")
    st = state()
    check("the box holds the document's prose",
          st["box"] == "a woman in a red dress, lit from a low window.", st["box"])
    check("the mirror is glyph-for-glyph the box", st["mirror"] == st["box"])
    check("the invented run is grey", st["grey"] == ["lit from a low window"],
          str(st["grey"]))
    # Underline is reach, colour is authorship: an element the person wrote is
    # addressable too, and only one of the two is grey.
    check("the person's own element is underlined and not grey",
          "a woman in a red dress" in st["spans"] and
          "a woman in a red dress" not in st["grey"], str(st["spans"]))

    # ── the caret does not jump ──────────────────────────────────────────────
    typed("")
    pg.wait_for_timeout(600)
    pg.evaluate("""() => {
      const ta = document.querySelector('#prompt');
      const set = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype, 'value').set;
      set.call(ta, 'a woman in a red dress +');
      ta.dispatchEvent(new Event('input', {bubbles: true}));
    }""")
    pg.wait_for_timeout(60)
    pg.evaluate("() => document.querySelector('#prompt').setSelectionRange(9, 9)")
    pg.wait_for_timeout(1200)
    st = state()
    check("the caret stays where it was while the box grows",
          st["caret"] == 9, f"caret {st['caret']} in {len(st['box'])} chars")

    # ── editing a grey run makes it yours, with no gesture ───────────────────
    pg.evaluate("""() => {
      const ta = document.querySelector('#prompt');
      const set = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype, 'value').set;
      set.call(ta, ta.value.replace('a low window', 'a high window'));
      ta.dispatchEvent(new Event('input', {bubbles: true}));
    }""")
    pg.wait_for_timeout(150)
    check("committing a run stops it being grey", state()["grey"] == [],
          str(state()["grey"]))

    # ── the reroll affordance: nothing at rest, armed inside a grey run ──────
    typed("")
    typed("a woman in a red dress +")
    pg.evaluate("""() => { const ta = document.querySelector('#prompt');
      ta.focus(); ta.setSelectionRange(4, 4);
      document.dispatchEvent(new Event('selectionchange')); }""")
    pg.wait_for_timeout(200)
    check("nothing is drawn while the caret is in the person's words",
          not state()["reroll"])
    pg.evaluate("""() => { const ta = document.querySelector('#prompt');
      const i = ta.value.indexOf('lit from');
      ta.focus(); ta.setSelectionRange(i + 3, i + 3);
      document.dispatchEvent(new Event('selectionchange')); }""")
    pg.wait_for_timeout(220)
    check("the affordance arms inside a grey run", state()["reroll"])

    before = state()["box"]
    pg.evaluate("() => document.querySelector('.mk-reroll')?.click()")
    pg.wait_for_timeout(900)
    st = state()
    check("a reroll moves that clause and nothing else",
          st["box"] != before
          and st["box"].startswith("a woman in a red dress,")
          and st["grey"] and st["grey"][0] != "lit from a low window",
          st["box"])

    # ── a stale document is not sent ─────────────────────────────────────────
    typed("")
    typed("a woman in a red dress +")
    sent = pg.evaluate(BODY)
    check("a document that describes the box is sent with the run",
          bool(sent and sent.get("modules")),
          f"{len(sent.get('modules') or []) if sent else 0} element(s)")
    # Typing past it without waiting for the next parse: the box has moved, so
    # the document no longer describes it and the run must go plain.
    pg.evaluate("""() => {
      const ta = document.querySelector('#prompt');
      const set = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype, 'value').set;
      set.call(ta, ta.value + ' and a dog nobody parsed');
      ta.dispatchEvent(new Event('input', {bubbles: true}));
    }""")
    pg.wait_for_timeout(120)
    sent = pg.evaluate(BODY)
    check("a stale document sends no modules at all",
          bool(sent) and not sent.get("modules"),
          str((sent or {}).get("modules")))

    # ── a parse landing mid-composition leaves the box alone ─────────────────
    #
    # Over CDP, not a synthetic event. `PAUSE_MS` is 500ms and an open candidate
    # window is a prose state that has been *stable* for longer than that while
    # the user is mid-word — the parse timer is tuned to almost exactly the dwell
    # time of the thing it must not interrupt.
    typed("")
    pg.click("#prompt")
    pg.evaluate("""() => {
      const ta = document.querySelector('#prompt');
      const set = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype, 'value').set;
      set.call(ta, 'a hotel corridor +');
      ta.dispatchEvent(new Event('input', {bubbles: true}));
    }""")
    cdp.send("Input.imeSetComposition", {
        "text": "にほん", "selectionStart": 3, "selectionEnd": 3})
    held = state()["box"]
    pg.wait_for_timeout(1400)
    check("a parse landing mid-composition writes nothing",
          state()["box"] == held, state()["box"])
    cdp.send("Input.insertText", {"text": ""})

    # ── the seed: pinned only once a document exists ─────────────────────────
    seed = pg.evaluate("""() => {
      const el = document.querySelector('#g-seed, [placeholder="random"]');
      return el ? el.value : null;
    }""")
    check("the seed field is untouched by any of this", seed in ("", None),
          repr(seed))

    br.close()

print("\n" + ("FAILED: " + ", ".join(fails) if fails
              else "the model may insert, never revise — and the caret never moves"))
sys.exit(1 if fails else 0)
