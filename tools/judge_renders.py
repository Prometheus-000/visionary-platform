"""
Score the picture, not the text. The last proxy removed.

Everything else that measures this feature is one remove from the thing it is
about. `smoke_parse.py` scores a document against the sentence it came from;
`--judge` scores the rewritten prompt against the original description, which is
better and is still text. All four criteria are finally about a **render**, and
until something reads the render they are being answered by proxy.

`does_it_help.py` produces the pairs — the same fragment rendered bare and
rendered from what the model wrote, one seed, one size, the sentence as the only
variable. This reads those pairs and says which one better realises what the
person asked for.

    python3.11 tools/judge_renders.py --pairs out/            # local vLLM
    python3.11 tools/judge_renders.py --pairs out/ --backend http://…/v1

**The VLM variant of the same family**, which is the cheap part of this and the
reason it is worth doing at all: `docs/vendor-parse-model.md` already pins
`Qwen/Qwen3-VL-4B-Instruct` as the encoder the parse writes for. Serving it costs
the same L4 the parse costs, and it is not the model being judged — the parse
runs on the abliterated text-only fork, so this is a different checkpoint reading
a different modality, which is the separation `--judge` had to arrange
deliberately and this gets for free.

## What makes the verdict worth having

**Blind, by construction.** The two images go in as A and B in an order fixed by
a hash of the pair's name, and the prompts are never shown. The judge cannot know
which is the replacement, so it cannot prefer one for being longer, more
detailed, or more like a prompt — it can only prefer the picture. Every other
measurement in this repo could be gamed by writing more; this one cannot.

**Both directions of the same question.** A judge asked "which is better" will
pick one, always, and a coin flip reported as a preference is worse than no
number. So each pair is asked twice with the order swapped, and a pair only
counts as decided when both answers agree. Disagreement is reported as a tie
rather than resolved, because a judge that changes its mind when the images swap
places is telling you the two pictures are close.

**A quote, or it did not happen.** The verdict carries what in the image decided
it. The discipline is `--judge`'s and the reason is the same: a judge that cannot
point at the thing it marked has usually invented it.
"""

import argparse
import base64
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

RUBRIC = """\
You are shown two images, A and B, made from the same brief by different means.
Decide which one better realises the brief.

THE BRIEF: {brief}

Judge only these, in this order:
1. Does the image show what the brief asks for, including any state the brief
   implies? "After the party" means the party is over.
2. Is the feeling of the brief staged in the picture, rather than absent?
3. Do the things in the frame stand in a deliberate arrangement, or is the
   picture a list of objects?
4. Are the literal, specific facts of the brief present and correct?

Ignore which image looks more elaborate. A picture with more things in it is not
better; a picture that answers the brief is. If neither is clearly better, say
"tie" — that is a real answer and is preferred to a guess.

Return JSON: {{"winner": "A"|"B"|"tie", "because": "<one sentence naming what in
the winning image decided it>", "against": "<one sentence naming what the other
image got wrong or missed>"}}
"""


# Long edge in pixels before the image is sent. Qwen3-VL tiles an image into
# patches, so a 1152x864 render is thousands of vision tokens and *two* of them
# overrun a 16k window before a word of the rubric is read — which does not
# arrive as a context error, it arrives as a server that dies mid-decode and a
# Sandbox that terminates with no log. 768 is comfortably inside the window and
# is far more than the questions need: every one of the four criteria is about
# composition, staging and whether a named thing is present, and none of them is
# decided by detail this throws away.
JUDGE_LONG_EDGE = 768


def data_uri(path: Path) -> str:
    """The image, downscaled and re-encoded as JPEG, as a data URI."""
    try:
        from PIL import Image
    except ImportError:  # judged without Pillow: send it as it lies
        kind = "png" if path.suffix.lower() == ".png" else "jpeg"
        return f"data:image/{kind};base64," + base64.b64encode(path.read_bytes()).decode()

    import io
    with Image.open(path) as im:
        im = im.convert("RGB")
        if max(im.size) > JUDGE_LONG_EDGE:
            scale = JUDGE_LONG_EDGE / max(im.size)
            im = im.resize((round(im.width * scale), round(im.height * scale)),
                           Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def ask(base_url: str, model: str, brief: str, first: Path, second: Path,
        timeout: float = 300.0) -> dict:
    """One verdict. `first` is shown as A and `second` as B — the caller decides."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": RUBRIC.format(brief=brief)},
            {"type": "text", "text": "Image A:"},
            {"type": "image_url", "image_url": {"url": data_uri(first)}},
            {"type": "text", "text": "Image B:"},
            {"type": "image_url", "image_url": {"url": data_uri(second)}},
        ]}],
        "max_tokens": 400,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions", json.dumps(body).encode(),
        {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        said = json.load(r)["choices"][0]["message"]["content"]
    # The rubric asks for JSON and a 4B does not always oblige on the first
    # token. Pull the first object out rather than failing the pair — a judge
    # that answers well in a code fence is not a judge that failed.
    match = re.search(r"\{.*\}", said, re.S)
    if not match:
        return {"winner": "tie", "because": "", "against": "", "raw": said[:200]}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"winner": "tie", "because": "", "against": "", "raw": said[:200]}


def pairs_in(folder: Path) -> list[tuple[str, Path, Path]]:
    """`<name>_bare.png` beside `<name>_rich.png`, which is what `does_it_help` writes."""
    out = []
    for bare in sorted(folder.glob("*_bare.png")):
        rich = bare.with_name(bare.name.replace("_bare.png", "_rich.png"))
        if rich.exists():
            out.append((bare.name[: -len("_bare.png")], bare, rich))
    return out


def briefs_from(dump: Path | None) -> dict[str, str]:
    """The person's own fragment, keyed the way `does_it_help.py` names its files."""
    if not dump:
        return {}
    out = {}
    for line in dump.read_text().splitlines():
        if not line.startswith("ENRICH "):
            continue
        row = json.loads(line[7:])
        name = "".join(c if c.isalnum() else "_" for c in row["prose"])[:24]
        out[name] = row["prose"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, type=Path,
                    help="A folder of <name>_bare.png / <name>_rich.png")
    ap.add_argument("--briefs", type=Path, default=None,
                    help="The `--enrich` dump the pairs were rendered from, so "
                         "the judge is given the person's fragment rather than "
                         "a filename.")
    ap.add_argument("--backend", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    args = ap.parse_args()

    found = pairs_in(args.pairs)
    if not found:
        print(f"no *_bare.png / *_rich.png pairs in {args.pairs}", file=sys.stderr)
        return 1
    briefs = briefs_from(args.briefs)

    print(f"backend {args.backend}\nmodel   {args.model}\npairs   {len(found)}\n")
    print("  Each pair judged twice with the order swapped. A win counts only")
    print("  when both orders agree; disagreement is a tie, not a tiebreak.\n")

    wins = {"rich": 0, "bare": 0, "tie": 0}
    for name, bare, rich in found:
        brief = briefs.get(name) or name.replace("_", " ")
        # The order is fixed per pair rather than random, so a re-run of this
        # harness on the same folder gives the same answer — the property
        # `--judge` is kept for and the one reading by hand never had.
        rich_first = int(hashlib.sha1(name.encode()).hexdigest(), 16) % 2 == 0
        a, b = (rich, bare) if rich_first else (bare, rich)

        first = ask(args.backend.rstrip("/"), args.model, brief, a, b)
        second = ask(args.backend.rstrip("/"), args.model, brief, b, a)

        def side(v: dict, flipped: bool) -> str:
            w = str(v.get("winner", "tie")).strip().upper()
            if w not in ("A", "B"):
                return "tie"
            is_rich = (w == "A") == (rich_first != flipped)
            return "rich" if is_rich else "bare"

        one, two = side(first, False), side(second, True)
        verdict = one if one == two else "tie"
        wins[verdict] += 1
        mark = {"rich": "REPLACED", "bare": "bare    ", "tie": "tie     "}[verdict]
        print(f"  {mark}  {brief[:52]}")
        if verdict != "tie":
            because = (first if one == verdict else second).get("because", "")
            print(f"            {because[:96]}")
        elif one != two:
            print(f"            (order-dependent: {one} then {two} — the pair is close)")

    n = len(found)
    print(f"\n  replacement wins  {wins['rich']}/{n}")
    print(f"  bare wins         {wins['bare']}/{n}")
    print(f"  tie               {wins['tie']}/{n}")
    print("\n  Read this as the only measurement here that is not a proxy — and")
    print("  read a tie as a tie. A pair the judge flips on is two pictures that")
    print("  are genuinely close, which is a result about the feature rather")
    print("  than a failure of the instrument.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
