"""
What the note under the prompt says about `<lora:…>` tokens.

    python3 tools/ui-checks/probe_lora.py http://localhost:8791
    python3 tools/ui-checks/probe_lora.py http://localhost:5173

The note is the only thing on the page that can say what the prompt cannot: a
name that resolves to no file, a name that resolves to two, a stack past
MAX_LORAS. The stub volume is built to hold every one of those — `Portrait` and
`portrait` are two real files differing only in case, and `high` exists in both
Wan speed folders — so the assertions here are about the rules, not the fixture.

The case rule is the one worth the most. Folding case before comparing makes two
distinct files one ambiguous name, and the failure is not "picked the wrong
one" — it is that *neither* resolves, so both go untypeable and the note blames
a missing file for a file sitting right there.
"""
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8791"

fails: list[str] = []


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(f"{label} {detail}".strip())


TYPE = """
(text) => {
  const ta = document.querySelector('#prompt');
  const set = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set;
  set.call(ta, text);
  ta.dispatchEvent(new Event('input', {bubbles: true}));
}
"""

NOTE = """() => (document.querySelector('#lora-note')?.textContent || '').trim()"""


def note_for(pg, text):
    pg.evaluate(TYPE, text)
    pg.wait_for_timeout(260)
    return pg.evaluate(NOTE)


with sync_playwright() as pw:
    b = pw.chromium.launch(channel="chrome")
    pg = b.new_page(viewport={"width": 1512, "height": 982}, color_scheme="dark")
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(URL, wait_until="networkidle", timeout=60_000)
    pg.wait_for_timeout(1400)

    print(f"\n=== {URL} ===")

    # A prompt with no tokens has nothing to report.
    n = note_for(pg, "a portrait in soft window light")
    check("silent with no tokens", n == "", repr(n))

    # Exact spelling wins outright, both ways round, on a volume that holds both.
    n = note_for(pg, "a portrait <lora:portrait:1>")
    check("exact lowercase resolves", n == "", repr(n))
    n = note_for(pg, "a portrait <lora:Portrait:1>")
    check("exact uppercase resolves", n == "", repr(n))

    # Neither spelling — case-insensitively this matches two files, so it must
    # say which rather than silently taking one.
    n = note_for(pg, "a portrait <lora:PORTRAIT:1>")
    check("an ambiguous case-fold names the candidates",
          "portrait" in n.lower() and ("Portrait" in n or "two" in n.lower() or "," in n),
          repr(n))

    # `high` is a real filename in both Wan speed folders. The note has to point
    # at the folder-qualified names, because "no LoRA named high" would send you
    # looking for a file that is sitting right there.
    n = note_for(pg, "a shot <lora:high:1>")
    check("an ambiguous stem names the folders",
          "wan22-speed" in n, repr(n))

    # Qualified by its folder, it resolves.
    n = note_for(pg, "a shot <lora:wan22-speed-t2v/high:1>")
    check("the folder-qualified name resolves", n == "", repr(n))

    # A name with no file behind it.
    n = note_for(pg, "a shot <lora:nosuchlora:1>")
    check("a missing name is named", "nosuchlora" in n, repr(n))

    # Past the cap. MAX_LORAS is 6 on the stub.
    many = " ".join(f"<lora:{x}:1>" for x in
                    ["my_style", "darkbrush", "sunsetblur", "portrait", "Portrait",
                     "krea2_identity_edit_v1_2", "wan22-speed-t2v/high"])
    n = note_for(pg, f"a shot {many}")
    check("a stack past the cap is reported",
          "6" in n or "most" in n.lower() or "max" in n.lower(), repr(n))

    if errors:
        check("no page errors", False, str(errors[:3]))
    b.close()

print()
if fails:
    print(f"{len(fails)} failed:", file=sys.stderr)
    for f in fails:
        print(f"  {f}", file=sys.stderr)
    sys.exit(1)
print("lora note intact")
