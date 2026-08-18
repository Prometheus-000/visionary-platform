"""
Is the interpreter reachable, and is the schema the thing that binds it?

CPU-only, in the spirit of `smoke_caption.py`: it answers the questions that do
not need weights, because those are the ones whose failure is hardest to read.
A schema that never binds does not produce an error — it produces an
unconstrained model returning prose, and the run scores zero on a model that
works. That is the failure this file exists to catch before a GPU is rented.

    python3.11 tools/smoke_interpret.py
    python3.11 tools/smoke_interpret.py --backend http://localhost:8000/v1

Without `--backend` the live half is skipped and said to be skipped, because a
harness that silently checks less than it claims is worse than one that checks
nothing.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _from_app import pull  # noqa: E402

G = pull({
    "PARSE_SCHEMA", "_ELEMENT", "_SPAN", "PARSE_RULES", "PARSE_REROLL",
    "PARSE_REPO", "PARSE_REVISION", "MAX_SPANS", "MAX_MODULES",
    "MODULE_ROLES", "MODULE_TEXT_MAX", "MAX_MODULE_DEPTH",
    "_structured_call", "_validate_modules", "_oneline", "_flat",
    "_spans_to_text",
})

fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


print("the schema — a grammar, not a suggestion")

schema = G["PARSE_SCHEMA"]["input_schema"]

# **The recursion is the whole reason this check exists.** `children` described
# as a bare object is what a schema looks like when nesting is written casually,
# and under constrained decoding it is a trap: an untyped object permits any
# JSON forever, so the grammar never pushes the model toward an end and it
# generates until the token cap. Measured on a 4B that was 40s per request and
# an empty completion — which reads as a broken model and is a broken schema.
check("`children` is a $ref, never a bare object",
      G["_ELEMENT"]["properties"]["children"]["items"] == {"$ref": "#/$defs/element"})
check("the $ref has a $defs to resolve against",
      "element" in schema.get("$defs", {}))
check("the element type is bounded on every array",
      all("maxItems" in G["_ELEMENT"]["properties"][k]
          for k in ("ties", "spans", "children")))
check("nothing extra may be emitted at either level",
      G["_ELEMENT"].get("additionalProperties") is False
      and schema.get("additionalProperties") is False)
check("the schema is serialisable, which is what a server receives",
      isinstance(json.dumps(schema), str))

# A span carries its own origin, because a flag on the pair says the wrong thing
# about whichever half it is wrong about.
check("a span is text plus an origin, both required",
      set(G["_SPAN"]["required"]) == {"text", "origin"})

print("\nthe rules — restraint first, where it governs what is under it")

rules = G["PARSE_RULES"]
head = rules[:rules.index("MARK EVERY ELEMENT")] if "MARK EVERY ELEMENT" in rules else ""
# Both of invariant 3's rules go *first*. The marking section sat early enough
# to read as permission to invent, which is the one thing it must not do.
check("the minimum-interpretation rule is in the head",
      "MINIMUM USEFUL INTERPRETATION" in head)
check("the enrichment-not-replacement rule is in the head",
      "MAY NOT CONTRADICT" in head)
check("marking comes after both, and points back at them",
      rules.index("MARK EVERY ELEMENT") > rules.index("MINIMUM USEFUL INTERPRETATION"))
check("the reroll paragraph is appended, never a second set of rules",
      G["PARSE_REROLL"] not in rules and "REROLL ONE ELEMENT" in G["PARSE_REROLL"])

print("\nthe weights — pinned to a revision, not a branch")
check("a repo id is pinned", bool(G["PARSE_REPO"]), G["PARSE_REPO"])
check("a full commit sha is pinned, not a branch name",
      len(G["PARSE_REVISION"]) == 40 and all(c in "0123456789abcdef"
                                             for c in G["PARSE_REVISION"]),
      G["PARSE_REVISION"])


def live(base_url: str) -> None:
    """The one question that needs a server: does a dialect actually bind."""
    print(f"\nthe endpoint — {base_url}")
    chosen: list[str] = []
    try:
        said = G["_structured_call"](
            base_url, G["PARSE_REPO"], G["PARSE_RULES"],
            "a woman in a red dress", schema, chosen, timeout=300)
    except Exception as exc:
        check("a structured-output dialect binds", False, f"({exc})")
        return
    check("a structured-output dialect binds", bool(chosen), f"via {chosen[0]}")
    try:
        mods = G["_validate_modules"](json.loads(said).get("elements") or [])
    except Exception as exc:
        check("what came back survives the validator", False, f"({exc})")
        return
    # Not "is it good" — that is `smoke_parse.py`'s question and it needs a
    # corpus. This only asks whether the pipe is connected end to end.
    check("what came back survives the validator", True, f"{len(mods)} element(s)")
    check("every element carries an id, which a reroll addresses it by",
          all(m.get("id") for m in mods))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="",
                    help="An OpenAI-compatible base URL. Omitted, the live half "
                         "is skipped and says so.")
    args = ap.parse_args()
    if args.backend:
        live(args.backend.rstrip("/"))
    else:
        print("\nthe endpoint — skipped, no --backend given")

    print()
    if fails:
        for f in fails:
            print(f"  {f}")
        print(f"{len(fails)} failure(s).")
        return 1
    print("The interpreter is wired the way the schema says it is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
