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
  * **`needs`, which is per item and not only per group.** A silent family
    dropped the audio pills — but `dialogue` lands in the *visual* field while
    still being audio, which is the case a group-level rule gets wrong. Nothing
    is silent now, so the matrix pins that every group arrives; the per-item
    rule is what a second family would land on.
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


def capture():
    st = state()
    out = {}
    for name, payload in cases(st["shot_vocab"]):
        out[name] = compile_one(payload)
    for name, payload in role_cases(st["shot_roles"]):
        out[name] = compile_one(payload)
    return out


if __name__ == "__main__":
    result = capture()
    errs = {k: v for k, v in result.items() if "error" in v}
    json.dump(result, sys.stdout, indent=1, sort_keys=True, ensure_ascii=False)
    print(f"\n\n{len(result)} cases, {len(errs)} errors", file=sys.stderr)
    for k, v in list(errs.items())[:10]:
        print(f"  ERROR {k}: {v}", file=sys.stderr)
