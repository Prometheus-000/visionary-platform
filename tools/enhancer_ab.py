"""
What the enhancer adds, and where it adds it.

    modal deploy tools/prompt_playground.py
    python3 tools/enhancer_ab.py --url https://…modal.run

Not "is it good". Two narrower questions that decide whether it belongs in the
product at all:

**Where does it add?** Which of the three or six fields each new fact lands in.
A model that only ever fills `overall_soundscape` and `non_diegetic_music` is
telling you those two fields are the gap, and the rest is a compiler's job.

**What does it add — and does the palette already have a word for it?** This is
the one that decides. `SHOT_VOCAB` is 75 phrases across camera, framing, angle,
light, tone, sound and score. If everything the model contributes lands inside
that vocabulary, it is an expensive pill-picker and the honest fix is a better
palette. **What it adds that the palette has no word for is the real finding**,
because that is a control the scene creator is missing — and per the rule the
composer is built on, anything in the document that traces to nothing a person
placed is filler until there is somewhere to place it.

The corpora are `smoke_parse.py`'s, pulled by AST rather than copied, for the
reason `_from_app.py` exists — and because the point is to run this thing on
the inputs *our own* layer failed:

- `ENRICH_CORPUS` — fragments, self-corrections, hedges. `empty diner, 3am` is
  the case CLAUDE.md records our document layer coming back *daylit*, and
  `night. no, late afternoon` is the correction it discarded.
- `RECALL_CORPUS` — recollections that end on a feeling word. The old rules
  kept that word verbatim 6 times out of 6, which renders as nothing.

Three checks beyond the counting, because they are the failures already on
record and a number will not show them:

- **contradiction** — does `3am` survive, or does the light argue with the clock
- **the feeling word** — echoed verbatim (bad) or staged (good)
- **narration** — does `I saw a guy…` reach the prompt as narration
"""

import argparse
import ast
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _from_app import SHOT, pull

G = pull(SHOT)
PARSE = Path(__file__).resolve().parent / "smoke_parse.py"

FIELDS = ("integrated_multimodal_description", "subject_definitions", "summary",
          "retention_analysis", "detailed_description",
          "overall_soundscape", "non_diegetic_music")

# Words that carry no content, so their arrival is not an addition.
STOP = set("""a an the and or but of in on at to for with from by as is are was were
be been being it its this that these those there here he she they them his her their
into over under across through while as if then than so very just also more most
some any all no not into onto off out up down back again about around near beside
between behind before after during until where when who whom which what how why""".split())


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP and len(w) > 2}


# Inputs chosen because there is **nothing in them to preserve**, so every word
# of the output is an addition and "what it adds" is isolated with no fidelity
# confound. The corpora above cannot do that: they all carry facts, so growth
# and preservation are tangled in one number.
#
# The first is the case to run before any other. A person walking down a street
# is the whole of it — no time, no weather, no camera, no sound, no reason. The
# model has to supply every one of those or produce nothing, and each is either
# a pill the palette already has or a control the composer is missing.
#
# It is also the right probe for **I2VA**, which is the opposite test: with a
# first frame supplied the guide says derive the style *from the picture* and
# develop forward. The same words should come back shorter and about motion. A
# model that writes the same paragraph either way is not reading the task.
CONTROLS = [
    ("street", "a person walking down the street"),
    ("door", "someone opens a door"),
    ("sitting", "a woman sitting down"),
    ("waiting", "two people waiting"),
]


def corpus(name: str) -> list[tuple[str, str]]:
    if name == "CONTROLS":
        return CONTROLS
    """A corpus out of smoke_parse.py without importing it."""
    for node in ast.parse(PARSE.read_text()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == name:
            val = ast.literal_eval(node.value)
            return [(f"{i}", v) if isinstance(v, str) else (v[0], v[1])
                    for i, v in enumerate(val)]
    raise SystemExit(f"{name} is not in smoke_parse.py any more")


def palette_words() -> set[str]:
    """Every content word the shot palette can already say."""
    out: set[str] = set()
    for g in G["SHOT_VOCAB"]:
        for it in g["items"]:
            out |= words(it.get("phrase") or "")
    return out


def by_field(doc: str) -> dict[str, str]:
    out, cur = {}, None
    for line in doc.split("\n"):
        head = line.split(":", 1)[0].strip()
        if head in FIELDS:
            cur = head
            out[cur] = line.split(":", 1)[1].strip() if ":" in line else ""
        elif cur:
            out[cur] += "\n" + line
    return out


CORPORA = {"controls": "CONTROLS", "fragments": "ENRICH_CORPUS",
           "recollections": "RECALL_CORPUS"}

FEELING = re.compile(r"\b(lonely|lonel(y|iness)|exhaust\w*|sad|angry|tired|lost|"
                     r"nostalgi\w*|melanchol\w*|eerie|peaceful|ominous|hopeful)\b", re.I)
NARRATION = re.compile(r"\b(I saw|I was|we were|I couldn't|I remember|there was this)\b", re.I)


def run(url: str, prompt: str, task: str, seconds: float) -> dict:
    body = json.dumps({"prompt": prompt, "task": task, "seconds": seconds}).encode()
    req = urllib.request.Request(f"{url.rstrip('/')}/api/run", data=body,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def report(url: str, which: str, task: str, seconds: float, limit: int) -> None:
    pal = palette_words()
    rows = corpus(CORPORA[which])[:limit]
    field_tally: dict[str, int] = {}
    covered = uncovered_all = 0
    novel: dict[str, int] = {}

    for name, prose in rows:
        try:
            got = run(url, prose, task, seconds)
        except Exception as exc:
            print(f"  {name:14} request failed: {exc}")
            continue
        if got.get("error"):
            print(f"  {name:14} {got['error']}")
            continue
        doc = got["text"]
        added = words(doc) - words(prose)
        inside = added & pal
        outside = added - pal
        covered += len(inside)
        uncovered_all += len(outside)
        for w in outside:
            novel[w] = novel.get(w, 0) + 1

        fields = by_field(doc)
        for f, body in fields.items():
            field_tally[f] = field_tally.get(f, 0) + len(words(body) - words(prose))

        grew = len(doc) / max(1, len(prose))
        flags = []
        if FEELING.search(prose) and FEELING.search(fields.get(
                "integrated_multimodal_description", "") + fields.get("detailed_description", "")):
            flags.append("FEELING ECHOED")
        if NARRATION.search(doc):
            flags.append("NARRATION")
        if re.search(r"\b3\s*a\.?m\.?\b|\bnight\b", prose, re.I) and re.search(
                r"\b(daylight|sunlit|morning sun|bright daylight|midday)\b", doc, re.I):
            flags.append("CONTRADICTS THE CLOCK")
        print(f"  {name:14} {grew:5.1f}x  +{len(added):3} words "
              f"({len(inside)} the palette has, {len(outside)} it does not)"
              + ("   " + " · ".join(flags) if flags else ""))

    total = covered + uncovered_all or 1
    print(f"\n  where it adds")
    for f, n in sorted(field_tally.items(), key=lambda x: -x[1]):
        print(f"    {n:5}  {f}")
    print(f"\n  what it adds")
    print(f"    {covered:5}  ({100*covered//total}%) the palette already has a pill for")
    print(f"    {uncovered_all:5}  ({100*uncovered_all//total}%) it does not")
    print(f"\n  the {min(25, len(novel))} commonest it has no word for — "
          f"each is a control the composer is missing, or filler")
    for w, n in sorted(novel.items(), key=lambda x: (-x[1], x[0]))[:25]:
        print(f"    {n:3}x  {w}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="the deployed playground")
    ap.add_argument("--corpus", choices=tuple(CORPORA), default="controls")
    ap.add_argument("--task", default="T2VA")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=99)
    ap.add_argument("--dry", action="store_true", help="print the corpus and stop")
    ap.add_argument("--across", metavar="PROMPT",
                    help="one prompt through T2VA and I2VA, side by side")
    a = ap.parse_args()

    rows = corpus(CORPORA[a.corpus])[:a.limit]
    if a.dry or not a.url:
        print(f"{len(rows)} inputs · {CORPORA[a.corpus]}\n")
        for name, prose in rows:
            print(f"  {name:14} {prose[:96]}{'…' if len(prose) > 96 else ''}")
        if not a.url:
            print("\npass --url to run them")
        return 0

    if a.across:
        pal = palette_words()
        for task in ("T2VA", "I2VA"):
            got = run(a.url, a.across, task, a.seconds)
            if got.get("error"):
                print(f"{task}: {got['error']}")
                continue
            doc, added = got["text"], words(got["text"]) - words(a.across)
            print(f"\n── {task} · {len(doc)} chars · +{len(added)} words "
                  f"({len(added & pal)} the palette has)\n")
            print(doc)
        return 0

    print(f"\n{a.corpus} · {a.task} · {a.seconds}s · {len(rows)} inputs\n")
    report(a.url, a.corpus, a.task, a.seconds, a.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
