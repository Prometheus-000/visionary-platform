"""
Every prompt the three compilers can emit, captured so a port cannot change one.

`/api/compile` is the same compiler the run uses, so this needs no GPU and no
Modal — just `tools/preview_ui.py`. The matrix exists because the compilers are
the one part of the front end that is *not* moving: the page sends a pill list
and gets a document back, and a React rewrite that grows its own copy of that
logic would be the exact failure `/api/compile` was built to prevent. A stored
baseline turns "did the compiler drift" from a reading exercise into a diff.

What the cases are chosen to pin, beyond coverage:

  * **No pills, no document.** With nothing chosen the compiler returns the
    typed text byte-for-byte. This is the rule every prompt written before the
    palette depends on, so it is first and it is checked on all three sides.
  * **The separator, which is the only thing the compiler may touch.** A
    leading clause's full stop softens to a comma before a lowercase fragment,
    so a `k3nan` trigger word is never capitalised into a different token.
    `TYPED` therefore includes a lowercase fragment, an already-closed
    sentence, and one written across two lines.
  * **`needs`, which is per item and not only per group.** Wan is silent, so
    audio pills are dropped — but `dialogue` lands in the *visual* field while
    still being audio, which is the case a group-level rule gets wrong. Both
    are in the matrix on both sides.
  * **The four H3 alignment sentences.** first-only, last-only, both and refs
    are four different instructions, and `_h3_task()` reads them more finely
    than `/api/video` does. One case each.
"""
import json
import sys
import urllib.error
import urllib.request

URL = "http://localhost:8791"

# Typed halves worth pinning. The first is the byte-for-byte case; the rest are
# the three shapes the separator rule has to decide between.
TYPED = {
    "empty": "",
    "fragment": "a portrait of k3nan",           # lowercase: the comma case
    "closed": "A dancer turns under a streetlight.",
    "twoline": "A dancer turns.\nThe street is wet.",
}


def state():
    with urllib.request.urlopen(f"{URL}/api/state", timeout=30) as r:
        return json.load(r)


def compile_one(payload):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{URL}/api/compile", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}", "body": exc.read().decode()[:400]}


def cases(vocab):
    """
    The matrix. Deterministic and ordered, because the output is a stored
    artefact — a set iteration or a dict ordering that moves between runs would
    read as compiler drift on a diff.
    """
    # The wire key is "{group}.{item}" — `SHOT_ITEMS` is keyed that way, and a
    # bare item key is rejected by name rather than ignored.
    groups = {g["key"]: [f"{g['key']}.{i['key']}" for i in g["items"]] for g in vocab}
    valued = {f"{g['key']}.{i['key']}"
              for g in vocab for i in g["items"] if i.get("valued")}
    image_ok = {g["key"] for g in vocab if g.get("image")}
    audio_groups = {g["key"] for g in vocab if g.get("needs") == "audio"}

    def pill(key):
        """A valued pill carries its text; the compiler preserves it verbatim."""
        return {"key": key, "text": "Where were you?"} if key in valued else {"key": key}

    # One pill from each group, and every pill of each group. Sorted by group
    # key so a vocabulary gaining an item does not reshuffle everything above it.
    picks = []
    for gk in sorted(groups):
        if groups[gk]:
            picks.append((f"one:{gk}", [pill(groups[gk][0])]))
            # Every pill of one group, which on a `pick: one` group is also the
            # test that the compiler collapses rather than emitting all of them.
            picks.append((f"all:{gk}", [pill(k) for k in groups[gk]]))
    picks.append(("all:87", [pill(k) for gk in sorted(groups) for k in groups[gk]]))
    picks.append(("none", []))

    # The audio-on-a-silent-model case, both halves: a sound/score pill, which
    # a group-level `needs` catches, and dialogue, which it does not.
    audio_pills = [pill(groups[gk][0]) for gk in sorted(audio_groups) if groups[gk]]
    if groups.get("say"):
        picks.append(("audio:dialogue-only", [pill(groups["say"][0])]))
        picks.append(("audio:sound+dialogue", audio_pills + [pill(groups["say"][0])]))
    picks.append(("audio:groups-only", audio_pills))

    # An image-illegal pill on the image side: camera and action are filtered
    # by the table, and the page dims them rather than hiding them, so the
    # compiler is what has to actually drop them.
    picks.append(("image-illegal",
                  [pill(groups[gk][0]) for gk in sorted(set(groups) - image_ok) if groups[gk]]))

    for tname, typed in TYPED.items():
        for pname, pills in picks:
            yield f"image/{tname}/{pname}", {
                "kind": "image", "prompt": typed, "shot": pills}
            for model in ("wan14b", "wan5b"):
                yield f"{model}/{tname}/{pname}", {
                    "kind": "video", "model": model, "prompt": typed, "shot": pills}
            yield f"h3/{tname}/{pname}", {
                "kind": "video", "model": "h3", "prompt": typed, "shot": pills}

    # The four alignment instructions. One typed half is enough — what varies
    # here is the sentence H3 is handed about where a picture sits in time, and
    # that does not depend on what the user wrote.
    typed = TYPED["fragment"]
    base = {"kind": "video", "model": "h3", "prompt": typed, "shot": []}
    yield "h3/task/t2v", dict(base)
    yield "h3/task/i2v", dict(base, first_frame=True)
    yield "h3/task/l2v", dict(base, last_frame=True)
    yield "h3/task/fl2va", dict(base, first_frame=True, last_frame=True)
    for n in (1, 2, 9):
        yield f"h3/task/ref2va-{n}", dict(base, references=n)
    yield "h3/task/refvideo", dict(base, ref_videos=1)
    yield "h3/task/ref+frames", dict(base, references=2, first_frame=True)


def role_cases(roles):
    """
    A reference chip carries what it is *for*, and each role compiles to its own
    `<Subject n> is the …` line plus a matching retention line. Six roles, one
    case each, plus the mixed and roleless forms — roleless has to keep running
    exactly as it did before roles existed.
    """
    typed = TYPED["fragment"]
    base = {"kind": "video", "model": "h3", "prompt": typed, "shot": []}
    yield "h3/roles/none", dict(base, references=2)
    for r in roles:
        yield f"h3/roles/{r['key']}", dict(base, references=1, ref_roles=[r["key"]])
    yield "h3/roles/mixed", dict(
        base, references=3, ref_roles=[roles[0]["key"], roles[2]["key"], roles[3]["key"]])
    yield "h3/roles/partial", dict(base, references=2, ref_roles=[roles[0]["key"], ""])


# A document, one element per clause. Written out rather than generated so the
# nesting is visible: `_module_clause` folds a child straight onto its anchor,
# which is the difference between a light that is its own character and a light
# that falls on somebody.
DOC_FLAT = [
    {"id": "e1", "role": "text", "text": "a woman in a red dress"},
    {"id": "e2", "role": "light", "text": "lit from a low window",
     "origin": "invented"},
]
DOC_NESTED = [
    {"id": "e1", "role": "place", "text": "A hotel corridor", "children": [
        {"id": "e1a", "role": "text", "text": "with pale blue floral wallpaper"},
        {"id": "e1b", "role": "text", "text": "and honey-coloured door frames"}]},
    {"id": "e2", "role": "subject", "text": "Three figures stand at the far end"},
]
DOC_ONE = [{"id": "e1", "role": "text", "text": TYPED["fragment"]}]


def document_cases(vocab):
    """
    The document through all three compilers — **appended, never merged in.**

    Every case above sends no `modules` and must stay byte-identical forever;
    this is the new surface, kept in its own namespace so a diff says which of
    the two moved.

    The `one:plain` / `one:doc` pair is here because the obvious claim about it
    is **false, by exactly one full stop**, and pinning both halves is what keeps
    that from being rediscovered:

        typed plain   "a portrait of k3nan"
        one element   "a portrait of k3nan."

    `_shot_body`'s docstring says a string and a one-element list compile
    identically, and inside `_shot_body` they do. The difference is one level
    up: with no pills and no document the compiler returns the typed string
    *verbatim* — that is the "byte-for-byte today's app" guarantee — while
    anything chosen goes through `_close`, which closes a sentence the person
    left open. A one-element document is something chosen, so it is closed, the
    same way typed text is closed the moment a single pill is picked.

    So the invariant that non-negotiable #6 actually rests on is narrower and is
    already pinned by every case above: **with no document and no pills, the
    compile is the typed text and nothing else.** Someone who never engages with
    the semantic layer sees the app they had. Making the one-element case match
    plain typing would mean *not* closing the last clause of a document, which
    would then differ from every multi-element document and from the pills path
    — a smaller inconsistency traded for a larger one.
    """
    pills = [{"key": "framing.mcu"}, {"key": "light.window"}]
    for kind, model in (("image", None), ("video", "wan22"), ("video", "h3")):
        tag = kind if model is None else f"{kind}-{model}"
        base = {"kind": kind, "prompt": "", "shot": []}
        if model:
            base["model"] = model
        # Typed plain against the same text as a one-element document. Both
        # sides of the pair are captured so the baseline shows the equality
        # rather than asserting it somewhere the diff cannot see.
        yield f"doc/{tag}/one:plain", dict(base, prompt=TYPED["fragment"])
        yield f"doc/{tag}/one:doc", dict(base, modules=DOC_ONE)
        yield f"doc/{tag}/flat", dict(base, modules=DOC_FLAT)
        yield f"doc/{tag}/nested", dict(base, modules=DOC_NESTED)
        # With pills, which is where a document has to fold into the same slots
        # a typed string does.
        yield f"doc/{tag}/flat+pills", dict(base, modules=DOC_FLAT, shot=pills)
        yield f"doc/{tag}/nested+pills", dict(base, modules=DOC_NESTED, shot=pills)
        # A document *and* typed text: the compiler reads the document and
        # ignores the string, which is worth pinning because the opposite would
        # silently double the prompt.
        yield f"doc/{tag}/doc-wins", dict(base, prompt=TYPED["closed"],
                                          modules=DOC_FLAT)


def capture():
    st = state()
    out = {}
    for name, payload in cases(st["shot_vocab"]):
        out[name] = compile_one(payload)
    for name, payload in role_cases(st["shot_roles"]):
        out[name] = compile_one(payload)
    for name, payload in document_cases(st["shot_vocab"]):
        out[name] = compile_one(payload)
    return out


if __name__ == "__main__":
    result = capture()
    errs = {k: v for k, v in result.items() if "error" in v}
    json.dump(result, sys.stdout, indent=1, sort_keys=True, ensure_ascii=False)
    print(f"\n\n{len(result)} cases, {len(errs)} errors", file=sys.stderr)
    for k, v in list(errs.items())[:10]:
        print(f"  ERROR {k}: {v}", file=sys.stderr)
