"""
Is the parse good enough to be trusted with somebody's sentence?

The parse is the only place in this platform where a model reads the *user's*
words, and it is about to run automatically on every pause in typing. It is also
the only component whose failure is invisible: a parse that quietly reformats
without applying any rule produces a storyline that looks right, compiles
cleanly, and reproduces every failure in `docs/krea2-prompt-template.md`.

So this scores two different things, and passing one of them proves nothing.

**Fidelity** — `compile(parse(x)) == x`. Automatic, over any corpus, no human.
This is the unusual property worth having: because compilation is near-lossless,
a parse can be checked against arbitrary input without anyone deciding what the
right answer was. It catches a parse that *dropped* something.

**Compliance** — targeted cases where the right answer is known. Each one is a
finding from the template document turned into an assertion. This catches a
parse that preserved every word and applied none of the rules, which fidelity
alone scores as perfect.

Backend-agnostic on purpose. It takes a callable, so the same instrument scores
a hosted model and a local one and the comparison means something. That is the
whole reason it exists before any weights are downloaded: the model choice
should be settled by a number, the way `hf_transfer` was, rather than argued.

    python3 tools/smoke_parse.py --backend hosted
    python3 tools/smoke_parse.py --backend http://localhost:8000/v1   # vLLM
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _from_app import pull  # noqa: E402

G = pull({
    "SHOT_VOCAB", "MODULE_ROLES", "MAX_MODULES", "MODULE_TEXT_MAX",
    "MAX_MODULE_DEPTH", "_module_clause", "_module_words", "_shot_phrases",
    "_shot_text", "_shot_sentence", "_shot_join", "_shot_body", "_close",
    "_oneline", "_flat", "_compile_image_prompt", "_validate_modules", "_module_texts",
    "_spans_to_text", "MAX_SPANS", "_SPAN",
    "_prominence", "PARSE_RULES", "_ELEMENT", "PARSE_SCHEMA", "PARSE_MODEL",
})

compile_prompt = lambda mods: G["_compile_image_prompt"]("", [], mods)
flat = lambda mods: [m for x in mods for m in [x, *flat(x.get("children") or [])]]


# ── the corpus ──────────────────────────────────────────────────────────────
#
# The three Gucci recreations. Fidelity is measured on these because they are
# the prompts every finding was derived from: a parse that cannot round-trip
# them has changed what the encoder is told, and the findings stop applying.

CORPUS = [
    "Three figures standing in a hotel corridor with pale blue floral wallpaper, "
    "honey-coloured wooden door frames and a deep blue carpet running away from "
    "the camera. On the left stands a tall woman in a purple beret, a sheer black "
    "lace long-sleeved top, a wide studded black belt and a teal suede skirt, with "
    "tall tan leather boots; one hand holds a patterned handbag at her hip. Beside "
    "her, to her right, stand two small girls of about eight, dressed identically "
    "in pale blue short-sleeved dresses with white collars and ribbon belts, white "
    "knee socks and black Mary Jane shoes, holding hands and standing shoulder to "
    "shoulder. All three face the camera directly, expressionless. Even shadowless "
    "light with no visible source. Shot straight on at chest height, rigidly "
    "symmetrical, sharp from front to back.",

    "Two young people sitting side by side on a slatted green wooden park bench in "
    "front of a weathered Roman brick ruin, with umbrella pines and dry grass "
    "behind them under a bright blue sky. Hard midday sun from the left throwing "
    "short crisp shadows. Shot on medium format at eye level.",

    "An extreme close-up portrait featuring pale, freckled skin and a single blue "
    "eye wrapped in reflective metallic gold ribbons. Strands of copper hair frame "
    "the top edge while the left ear softly blurs out of focus.",
]


# ── compliance: one case per finding ────────────────────────────────────────
#
# `want` is a predicate over the parsed storyline, not an expected string. The
# parse is allowed to phrase things its own way; what it is not allowed to do is
# skip a transformation. Each `why` is the failure the rule prevents.

def no_cross_perception(mods) -> bool:
    """A feeling *about* another subject must become this subject's own state."""
    joined = " ".join(m["text"].lower() for m in flat(mods))
    return not any(v in joined for v in
                   ("notices her", "notices him", "doesn't like that",
                    "dislikes that", "does not like that", "is jealous of"))


def has_a_tie(mods) -> bool:
    """`A is sitting with B` is a physical relation and must survive as a link."""
    return any(m.get("ties") for m in flat(mods))


def children_continue(mods) -> bool:
    """
    A child is joined straight onto its anchor, so it has to read as a
    continuation. A child that opens like a fresh sentence collides — "a hotel
    corridor pale blue floral wallpaper" — and the compiler may not repair it.
    """
    OPENERS = ("with", "in", "on", "under", "over", "against", "beside", "wearing",
               "holding", "lit", "and", "a", "an", "the", "its", "her", "his",
               "their", "carrying", "framed", "surrounded", "behind", "across")
    kids = [k for m in flat(mods) for k in (m.get("children") or [])]
    return all(k["text"].strip().lower().startswith(OPENERS) for k in kids) if kids else True


def no_invented_blankness(mods) -> bool:
    """A default that negates interiority is an instruction, not a neutral."""
    return not any("expressionless" in m["text"].lower()
                   for m in flat(mods) if m.get("origin") == "invented")


def geometry_not_adjective(mods) -> bool:
    """A quality has one setting and it is maximum; a fact carries an amount."""
    joined = " ".join(m["text"].lower() for m in flat(mods))
    bare = ("dramatic perspective", "extreme perspective", "strong perspective")
    return not any(b in joined for b in bare)


def subject_leads(mods) -> bool:
    """Whatever is first is what the picture is about. Never light, never camera."""
    if not mods:
        return True
    head = mods[0]["text"].lower()
    return not head.startswith(("lit ", "light", "shot ", "in a close", "in an ",
                                "hard ", "soft ", "backlit"))


CASES = [
    ("a mental relation becomes a visible state", no_cross_perception,
     "Three people in a bar. Person A is happy. Person C notices her and he "
     "doesn't like that she's talking to person B."),

    ("a physical relation becomes a tie", has_a_tie,
     "Two people on a sofa. Person A is sitting with person B."),

    ("a child is phrased as a continuation", children_continue,
     "A hotel corridor. Pale blue floral wallpaper and honey-coloured wooden "
     "door frames."),

    ("blankness is never invented", no_invented_blankness,
     "A woman at a kitchen table in the late afternoon."),

    ("perspective arrives as geometry", geometry_not_adjective,
     "Two people on a bench in front of a ruin. Give it some perspective."),

    ("nothing leads the subject", subject_leads,
     "Lit by a small window, k3nan sits reading."),

    ("a scene a stock model may balk at still parses", lambda m: len(m) > 0,
     "A recruit sits on a latrine floor at night in his underwear, head lowered, "
     "eyes up into the lens, a rifle across his knees and a wrong smile."),
]


# ── backends ────────────────────────────────────────────────────────────────

def hosted(model: str):
    """
    A baseline, so the local model has a number to beat rather than a feeling.

    Reads the key from the environment rather than the Modal Dict the app uses.
    That is not a departure from the no-Secrets rule — `tools/` runs on a laptop
    and never ships, and requiring a deploy to measure a model would mean the
    measurement happens after the decision instead of before it.
    """
    import os
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("Set ANTHROPIC_API_KEY to score the hosted baseline, "
                         "or pass --backend <vLLM url> for the local model.")
    import anthropic
    client = anthropic.Anthropic(api_key=key)

    def parse(prose: str):
        reply = client.messages.create(
            model=model, max_tokens=2048, system=G["PARSE_RULES"],
            tools=[G["PARSE_SCHEMA"]],
            tool_choice={"type": "tool", "name": "storyline"},
            messages=[{"role": "user", "content": prose}])
        for block in reply.content:
            if getattr(block, "type", None) == "tool_use":
                return G["_validate_modules"](block.input.get("elements") or [])
        raise ValueError("no storyline returned")

    return parse


def openai_compatible(base_url: str, model: str):
    """
    Any vLLM / SGLang server, which is how the local model is reached.

    `guided_json` rather than a polite request for JSON: with the schema bound,
    malformed output is not unlikely, it is unreachable. That is the difference
    between a 4B that works and one that works most of the time.
    """
    import urllib.request

    schema = G["PARSE_SCHEMA"]["input_schema"]
    # Spelled at the top level of the request, never under `extra_body` — that
    # is an OpenAI *client library* concept which flattens into the body, and
    # sent as a literal key vLLM ignores it silently. The schema would never
    # bind and the run would score an unconstrained model while reporting a
    # constrained one.
    DIALECTS = [
        ("guided_json", {"guided_json": schema,
                         "guided_decoding_backend": "xgrammar"}),
        ("structured_outputs", {"structured_outputs": {"json": schema}}),
        ("response_format", {"response_format": {
            "type": "json_schema",
            "json_schema": {"name": "storyline", "schema": schema,
                            "strict": True}}}),
    ]
    chosen: list[str] = []

    def call(extra: dict, prose: str) -> str:
        body = json.dumps({
            "model": model,
            "messages": [{"role": "system", "content": G["PARSE_RULES"]},
                         {"role": "user", "content": prose}],
            "max_tokens": 2048, "temperature": 0, **extra,
        }).encode()
        req = urllib.request.Request(f"{base_url}/chat/completions", body,
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)["choices"][0]["message"]["content"]

    def parse(prose: str):
        if not chosen:
            errs = []
            for name, extra in DIALECTS:
                try:
                    said = call(extra, prose)
                    # Recorded only once the response actually parses. Appending
                    # on a 200 was enough to lock every later call onto a dialect
                    # the server accepts and then ignores — vLLM 0.27 takes
                    # `guided_json` without binding it and returns empty content,
                    # so the run scored 0% on a model that works.
                    out = G["_validate_modules"](json.loads(said).get("elements") or [])
                    chosen.append(name)
                    print(f"  (schema bound via {name})")
                    return out
                except Exception as exc:
                    errs.append(f"{name}: {exc}")
                    if "said" in dir() and isinstance(said, str):
                        errs.append(f"    returned {len(said)} chars: {said[:200]!r}")
            raise SystemExit("No structured-output dialect accepted:\n  "
                             + "\n  ".join(errs))
        said = call(dict(DIALECTS[[d[0] for d in DIALECTS].index(chosen[0])][1]), prose)
        return G["_validate_modules"](json.loads(said).get("elements") or [])

    return parse


# ── the run ─────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="hosted",
                    help="'hosted', or an OpenAI-compatible base URL")
    ap.add_argument("--model", default=G["PARSE_MODEL"])
    args = ap.parse_args()
    parse = hosted(args.model) if args.backend == "hosted" else \
        openai_compatible(args.backend.rstrip("/"), args.model)

    print(f"backend: {args.backend}  model: {args.model}\n")

    print("fidelity — does the parse lose anything")
    kept, times = 0, []
    for prose in CORPUS:
        t0 = time.time()
        try:
            mods = parse(prose)
        except Exception as exc:
            print(f"  FAIL  {prose[:48]}…  ({exc})")
            continue
        times.append(time.time() - t0)
        out = compile_prompt(mods)
        ok = out == prose
        kept += ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {prose[:48]}…")
        if not ok:
            print(f"        got:  {out[:100]}…")

    print("\ncompliance — does it apply the rules")
    passed = 0
    for name, predicate, prose in CASES:
        try:
            mods = parse(prose)
            ok = bool(predicate(mods))
        except Exception as exc:
            ok = False
            print(f"  FAIL  {name}  ({exc})")
            continue
        passed += ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}")

    n = len(CORPUS) or 1
    print(f"\nfidelity   {kept}/{len(CORPUS)}  ({100 * kept // n}%)")
    print(f"compliance {passed}/{len(CASES)}  ({100 * passed // (len(CASES) or 1)}%)")
    if times:
        print(f"latency    {sum(times) / len(times):.1f}s mean, "
              f"{max(times):.1f}s worst")
    return 0 if kept == len(CORPUS) and passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
