"""
Is the rewrite path sound before a GPU is rented?

`/api/rewrite` is three things a model cannot be trusted to get right on its own:
an operation key it did not choose, a length it cannot count, and an answer it
likes to introduce, explain and wrap in quotes. Every one of those is handled
deterministically in `app.py`, which means every one of them is checkable here —
on a laptop, with no weights, no Modal account and no network.

    python3.11 tools/smoke_rewrite.py

In the spirit of `smoke_interpret.py`: this proves the plumbing, not the prose.
Whether the rewrite makes a *better picture* is `tools/prompt_ab.py`, which needs
a GPU and a judge and is the only measurement here that is not a proxy.

**Run this after any `COMFY_SHA` bump.** `comfy_image` takes its transformers
from ComfyUI's own `requirements.txt`, unpinned, and `visionary_rewrite` loads
the encoder against a pinned config — this is the check that the two have not
drifted apart in a way the cleaner would hide.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _from_app import pull  # noqa: E402

G = pull({
    "REWRITE_OPS", "KREA_EXPANSION", "_REWRITE_TAIL", "REWRITE_BACKEND",
    "_rewrite_tokens", "REWRITE_TOKEN_FLOOR", "REWRITE_TOKEN_CEILING",
    "_clean_rewrite", "_PREAMBLE", "_META", "_THINK", "_oneline",
    "_looks_like_refusal", "REFUSAL_RE", "MODULE_TEXT_MAX",
})


def ops() -> int:
    """
    One operation, and the facts it has to carry.

    It was three, and only one of them was ever measured — the blind A/B that
    beat the bare fragment 3-1 ran `expand` on every pair, which is
    `KREA_EXPANSION`. Balance and Enhance shipped unscored, and both turned out
    to be asking for behaviour that instruction already has: rule 7 makes
    expansion conditional, and grouping each subject with their own attributes
    under rule 2 balances a thin third subject without a word about prominence.
    """
    print("operations")
    bad = 0
    for key in ("enhance",):
        spec = G["REWRITE_OPS"].get(key)
        if not spec:
            print(f"  FAIL  {key} is missing"); bad += 1; continue
        why = []
        if not spec.get("label"):
            why.append("no label")
        if not spec.get("note"):
            why.append("no note")
        if not spec.get("instruction", "").strip():
            why.append("no instruction")
        bad += bool(why)
        print(f"  {'FAIL' if why else 'ok  '}  {key:8} {spec.get('label','?'):8} "
              f"{len(spec.get('instruction','')):5} chars"
              + (f"   [{', '.join(why)}]" if why else ""))
    if len(G["REWRITE_OPS"]) != 1:
        print(f"  FAIL  {len(G['REWRITE_OPS'])} operations — the row is one button")
        bad += 1

    # The cap is what holds the length — an instruction asking for one produced
    # 95, 122 and 617 words on three fragments. It scales now, because one job
    # serves both a fifteen-character fragment and a prompt rule 7 will only
    # polish, and a flat 200 truncates the second into a dropped sentence.
    tok = G["_rewrite_tokens"]
    floor, ceil = G["REWRITE_TOKEN_FLOOR"], G["REWRITE_TOKEN_CEILING"]
    for label, prose, want in (
            ("a fragment floors at the measured bound", "empty diner, 3am", floor),
            ("an empty prompt still floors", "", floor),
            ("a long prompt is not truncated", "x" * 1200, 600),
            ("and is bounded at the ceiling", "x" * 99999, ceil)):
        got = tok(prose)
        ok = got == want
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {label:44} {got:5} tokens")
    # Upstream's prompt is vendored verbatim; rule 7 is the line that makes it
    # cover Enhance as well as Expand, so its absence means the file drifted.
    krea = G["KREA_EXPANSION"]
    for probe, why in (("lightly polish", "rule 7, which is Enhance"),
                       ("Faithfulness First", "rule 1")):
        ok = probe in krea
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  KREA_EXPANSION carries {why}")
    return bad


CLEAN = [
    # (raw, expected, why)
    ("An empty diner at 3am, one tube flickering over the counter.",
     "An empty diner at 3am, one tube flickering over the counter.",
     "a clean answer is left alone"),
    ("Here is the expanded prompt: An empty diner at 3am.",
     "An empty diner at 3am.",
     "a preamble is stripped"),
    ('"An empty diner at 3am."',
     "An empty diner at 3am.",
     "wrapping quotes are not characters the model meant"),
    ("A lone fisherman hauling a net. → Revised version: A lone fisherman. "
     "**Change explained:** - 'feels' became 'seems'",
     "A lone fisherman hauling a net.",
     "meta-commentary is cut at the arrow"),
    ("What is the subject and mood? A diner, lonely. What lighting fits?\n"
     "Expanded prompt: An empty diner at 3am, rain on the glass.",
     "An empty diner at 3am, rain on the glass.",
     "a thinking block is dropped and the answer kept"),
    ("<think>The user wants a diner. Consider dawn or night.</think>\n"
     "An empty diner at 3am, rain on the glass.",
     "An empty diner at 3am, rain on the glass.",
     "a fenced reasoning block is dropped"),
    ("An empty diner at 3am. One tube flickers over the counter. Rain streaks the",
     "An empty diner at 3am. One tube flickers over the counter.",
     "the tail the token cap severed is dropped"),
    ("Notes on a winter morning, frost across the window.",
     "Notes on a winter morning, frost across the window.",
     "a prompt that opens with a meta word is not truncated"),
    ("A short one", "A short one",
     "too short to trim, kept rather than blanked"),
    ("What is the subject? A diner.", "What is the subject? A diner.",
     "thinking with no answer after it is left alone, not emptied"),
]


def clean() -> int:
    """`_clean_rewrite` — the one place an answer is made safe for the box."""
    print("\n_clean_rewrite — every case is a failure seen in a live run")
    bad = 0
    for raw, want, why in CLEAN:
        got = G["_clean_rewrite"](raw)
        ok = got == want
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {why}")
        if not ok:
            print(f"        want: {want!r}")
            print(f"        got : {got!r}")
    return bad


REFUSALS = [
    ("I cannot describe this scene.", True),
    ("I'm sorry, but I can't help with that request.", True),
    ("As an AI, I am unable to write that.", True),
    ("An empty diner at 3am, the counter running away to the right.", False),
    # The case anchoring exists for: a refusal *inside* a prompt is a sentence
    # about the picture, and a substring test would throw the prompt away.
    ("A prisoner says he cannot go back, lit by one bulb.", False),
]


def refusals() -> int:
    """A decline reaches the box as prose, so it is catchable — unlike a parse."""
    print("\nrefusal guard — prose declines are visible, schema-bound ones were not")
    bad = 0
    for text, want in REFUSALS:
        got = bool(G["_looks_like_refusal"](text))
        ok = got == want
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  "
              f"{'refusal' if got else 'prompt ':7}  {text[:52]}")
    return bad


def wiring() -> int:
    """The seam, and the tail every instruction is given."""
    print("\nwiring")
    bad = 0
    backend = G["REWRITE_BACKEND"]
    ok = backend in ("interpreter", "comfy")
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  REWRITE_BACKEND is {backend!r}")
    tail = G["_REWRITE_TAIL"]
    ok = "only the prompt" in tail.lower()
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  the shared tail asks for the prompt alone")
    # Every instruction is sent with the tail appended, so the pair has to fit
    # a system prompt without the operation's own text being crowded out.
    for key, spec in G["REWRITE_OPS"].items():
        total = len(spec["instruction"]) + len(tail)
        ok = total < 8000
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {key:8} instruction + tail = {total} chars")
    return bad


def main() -> int:
    bad = ops() + clean() + refusals() + wiring()
    print(f"\n{'PASS' if not bad else str(bad) + ' FAILED'}")
    print("\nThis proves the plumbing. Whether the rewrite makes a better picture\n"
          "is tools/prompt_ab.py, which renders the pair and judges it blind.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
