"""
Mark a prompt rewrite against what the person actually said.

    python3 tools/judge_prompts.py DUMP --at http://127.0.0.1:8000/v1 --model M

**A judge, not an oracle.** It runs where latency is free and somebody is
measuring, never on a request path — the rule is that the validator stays
arithmetic and the harness gets the model, because a probabilistic gate stacked
on a probabilistic writer is two coin flips where the second one is invisible.

Three things keep it honest and they are the parts not to skip. **Point it at
different weights than the thing being judged** — a model scoring its own output
agrees with itself, and this says so out loud when the two match. **Every verdict
carries a quote**, so a score with nothing behind it is visible; a judge that
cannot quote the fault has usually invented it. And **spot-check it by reading**:
what earns it its place is not that it is right, it is that it is repeatable,
which reading by hand is not.

It outlived the parse subsystem it was written for because the four criteria are
about the *picture* rather than about a document — and criterion 3 is the one
blocking exists to answer: subjects described one at a time and never related
come out as cutouts, each squared to the lens.

The dump is one JSON object per line, each carrying `name`, `prose` and
`compiled`, optionally prefixed `ENRICH `.

**It scores text, which is one remove from the thing that matters.**
`judge_renders.py` marks the pictures themselves and is the measurement that is
not a proxy; this one is cheaper and reads the words.
"""

import argparse
import json
from pathlib import Path
from typing import Any


def _structured_call(base_url: str, model: str, system: str, user: str,
                     schema: dict[str, Any], chosen: list[str],
                     *, timeout: float = 300.0, max_tokens: int = 2048) -> str:
    """
    One chat completion with the schema **actually bound**, on any vLLM server.

    `guided_json` rather than a politely-worded request for JSON: with the
    grammar bound, malformed output is not unlikely, it is unreachable. That is
    the difference between a 4B that works and one that works most of the time.

    Three dialects because servers disagree about the spelling, and one trap
    that cost a whole run to find: **a dialect is recorded only once a response
    actually parses.** vLLM 0.27 accepts `guided_json` without binding it and
    returns empty content, so appending on a 200 was enough to lock every later
    call onto a dialect the server takes and then ignores — and the run scored
    0% on a model that works.

    Spelled at the top level of the request, never under `extra_body`: that is
    an OpenAI *client library* concept which flattens into the body, and sent as
    a literal key the server ignores it silently. The schema would never bind
    and a constrained run would be reported for an unconstrained one.

    `chosen` is the caller's one-slot memo of which dialect worked, so the
    negotiation happens once per process rather than per request.
    """
    import urllib.request

    dialects = [
        ("guided_json", {"guided_json": schema,
                         "guided_decoding_backend": "xgrammar"}),
        ("structured_outputs", {"structured_outputs": {"json": schema}}),
        ("response_format", {"response_format": {
            "type": "json_schema",
            "json_schema": {"name": "verdict", "schema": schema,
                            "strict": True}}}),
    ]

    def call(extra: dict[str, Any]) -> str:
        body = json.dumps({
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            # Temperature 0, so the same pair marked twice comes back the same
            # way. A judge that disagrees with itself measures nothing.
            "max_tokens": max_tokens, "temperature": 0, **extra,
        }).encode()
        req = urllib.request.Request(f"{base_url}/chat/completions", body,
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]

    if chosen:
        return call(dict(dict(dialects)[chosen[0]]))

    errs = []
    for name, extra in dialects:
        said = None
        try:
            said = call(extra)
            json.loads(said)          # parses, so the grammar really bound
            chosen.append(name)
            return said
        except Exception as exc:
            errs.append(f"{name}: {exc}"
                        + (f" (returned {len(said)} chars)" if isinstance(said, str) else ""))
    raise ValueError("No structured-output dialect bound the schema:\n  "
                     + "\n  ".join(errs))


JUDGE_RUBRIC = """\
You are scoring a prompt-replacement system for a text-to-image model.

Somebody wrote a description of a picture they want. A model rewrote it into a
prompt. You are given both, and you score the rewrite on four criteria. You are
not rewriting anything and you are not being helpful — you are marking.

1. subject — Is the thing the picture is *about* the subject of the prompt, or
   has it become a detail in a scene? "a colossal stone hand bursting from the
   dirt" is a prompt about a hand, not about a forest.
2. tone — Was an emotional statement translated into visual staging, or typed
   back out as a word? "It feels lonely" renders as nothing; isolation, empty
   frame, a stark light and a lost horizon render. If the original had no
   emotional statement, score this `n/a`.
3. space — Are the things in the picture placed in relation to each other —
   contact, orientation, a shared surface or light — or listed side by side?
   Subjects that are described one at a time and never related come out as
   cutouts, each squared to the lens. If the original had one subject and no
   relations to make, score `n/a`.
4. fidelity — Does every specific thing the person named survive exactly? A
   bright yellow sweater is bright yellow; a stopped clock is stopped. Rewording
   is fine and expected — "a red winter coat" as "an oxblood down jacket" is the
   same fact. Dropping it is not.

Also answer two things directly, because they are the failures that hide:

- lost: anything specific in the original that is not in the rewrite, quoted
  from the original. Empty if nothing was lost.
- contradicted: anything in the rewrite that disagrees with the original,
  quoted from the rewrite. "3am" answered with even daylight is the example.
  Empty if nothing contradicts.

Every verdict carries a quote from the text you are judging. If you cannot quote
it, you have not found it.
"""


_VERDICT = {
    "type": "object",
    "properties": {
        "score": {"type": "string", "enum": ["pass", "partial", "fail", "n/a"]},
        "quote": {"type": "string",
                  "description": "From the rewrite. Empty only when n/a."},
        "why": {"type": "string", "description": "One sentence."},
    },
    "required": ["score", "quote", "why"],
    "additionalProperties": False,
}


JUDGE_SCHEMA = {
    "input_schema": {
        "type": "object",
        "properties": {
            "subject": _VERDICT,
            "tone": _VERDICT,
            "space": _VERDICT,
            "fidelity": _VERDICT,
            "lost": {"type": "array", "items": {"type": "string"}, "maxItems": 8,
                     "description": "Quoted from the original."},
            "contradicted": {"type": "array", "items": {"type": "string"},
                             "maxItems": 8, "description": "Quoted from the rewrite."},
        },
        "required": ["subject", "tone", "space", "fidelity", "lost",
                     "contradicted"],
        "additionalProperties": False,
    }
}


def judge(path: str, base_url: str, model: str, subject_model: str = "") -> int:
    """Score a dump against the rubric, one call per row."""
    if subject_model and subject_model == model:
        print("  ! the judge is the model being judged — scores will agree with "
              "themselves. Point --judge at a different endpoint.\n")
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("ENRICH "):
            d = json.loads(line[7:])
            if d.get("compiled"):
                rows.append(d)
    chosen: list[str] = []
    totals: dict[str, int] = {}
    print(f"judging {len(rows)} scenes with {model}\n")
    for d in rows:
        user = (f"ORIGINAL\n{d['prose']}\n\nREWRITE\n{d['compiled']}")
        try:
            said = json.loads(_structured_call(
                base_url, model, JUDGE_RUBRIC, user,
                JUDGE_SCHEMA["input_schema"], chosen, max_tokens=1400))
        except Exception as exc:
            print(f"  {d.get('name', '?'):<12} JUDGE FAILED  {str(exc)[:60]}")
            continue
        marks = " ".join(f"{k[:4]}:{said[k]['score']:<7}"
                         for k in ("subject", "tone", "space", "fidelity"))
        for k in ("subject", "tone", "space", "fidelity"):
            if said[k]["score"] != "n/a":
                totals[k] = totals.get(k, 0) + (said[k]["score"] == "pass")
                totals[k + "_n"] = totals.get(k + "_n", 0) + 1
        print(f"  {d.get('name', '?'):<12} {marks}")
        for k in ("subject", "tone", "space", "fidelity"):
            if said[k]["score"] in ("fail", "partial"):
                print(f"      {k}: {said[k]['why'][:96]}")
        if said["lost"]:
            print(f"      lost: {'; '.join(said['lost'])[:110]}")
        if said["contradicted"]:
            print(f"      CONTRADICTED: {'; '.join(said['contradicted'])[:110]}")
    print()
    for k in ("subject", "tone", "space", "fidelity"):
        n = totals.get(k + "_n", 0)
        if n:
            print(f"  {k:<9} {totals.get(k, 0)}/{n} pass")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Mark a prompt rewrite against "
                                             "what the person actually said.")
    ap.add_argument("dump", help="one JSON object per line: name, prose, compiled")
    ap.add_argument("--at", required=True, metavar="BASE_URL",
                    help="an OpenAI-compatible /v1 — serve_judge.py opens one")
    ap.add_argument("--model", required=True)
    ap.add_argument("--subject-model", default="",
                    help="the weights that wrote the rewrites, so this can warn "
                         "when the judge is marking its own work")
    args = ap.parse_args()
    return judge(args.dump, args.at, args.model, args.subject_model)


if __name__ == "__main__":
    raise SystemExit(main())
