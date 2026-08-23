"""
Does this prompt beat that one? One command, judged by looking at the pictures.

**The only measurement of this feature that is not a proxy.** Everything else
scores text against text — preserved, covered, round-tripped, idempotent — and a
prompt can pass every one of those and make a worse picture, which is what
happened: the semantic layer scored as maximum restraint and lost 0-4 to doing
nothing. This renders both and has a vision model say which one answered the
brief.

    python3.11 tools/prompt_ab.py --url https://…modal.run --from enrich.jsonl

Three stages, and the two halves already existed — this is the one command that
runs them in order, which is the whole of what it adds:

  1. `does_it_help.py` renders each fragment twice against the deployed app,
     bare and rewritten, at one seed with the sentence as the only variable.
  2. `stress_parse.py --vision` serves a vision model in a throwaway Sandbox.
  3. `judge_renders.py` scores the pairs inside it.

`--from` takes a `smoke_parse.py --enrich` dump, so the
rewritten half is what a model actually wrote rather than what somebody hoped it
would write. That distinction is not academic: hand-written replacements won
their pairs and every model-written one lost.

## Reading the result

**A tie is a tie.** Each pair is judged twice with the images swapped and counts
only when both orders agree, so a judge with positional bias scores all ties and
no wins — verified against stub oracles rather than assumed. A pair it flips on
is two pictures that are genuinely close.

**The judge is not the subject.** It is a different checkpoint and a different
modality from the model being scored, which is the separation that keeps a model
from marking its own work.

**It is an instrument, not an oracle.** Spot-check by looking. What earns it its
place is not that it is right, it is that it is repeatable, which reading by
hand is not.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Render a prompt pair and judge which picture answered the brief.")
    ap.add_argument("--url", required=True, help="The deployed web URL")
    ap.add_argument("--from", dest="dump", default=None,
                    help="An --enrich / rewrite JSONL. Omit for the built-in pairs.")
    ap.add_argument("--out", default="tools/ab-pairs",
                    help="Where the renders land, and what the judge reads.")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--seed", type=int, default=774411)
    ap.add_argument("--judge-model", default="Qwen/Qwen3-VL-4B-Instruct",
                    help="A *vision* model, and deliberately not the one being judged.")
    ap.add_argument("--gpu", default="L4")
    ap.add_argument("--render-only", action="store_true",
                    help="Stop after the pictures — useful when you want to look "
                         "before spending a Sandbox on the judge.")
    args = ap.parse_args()

    out = Path(args.out)
    render = [sys.executable, str(ROOT / "tools" / "does_it_help.py"),
              "--url", args.url, "--out", str(out),
              "--limit", str(args.limit), "--seed", str(args.seed)]
    if args.dump:
        render += ["--from", args.dump]
    if (code := run(render)):
        print("render failed; not judging half a set", file=sys.stderr)
        return code

    pairs = sorted(out.glob("*_bare.png"))
    if not pairs:
        print(f"no pairs in {out} — nothing to judge", file=sys.stderr)
        return 1
    print(f"\n{len(pairs)} pairs rendered into {out}")
    if args.render_only:
        return 0

    judge = [sys.executable, str(ROOT / "tools" / "stress_parse.py"),
             "--vision", str(out), "--model", args.judge_model, "--gpu", args.gpu]
    if args.dump:
        judge += ["--vision-briefs", args.dump]
    return run(judge)


if __name__ == "__main__":
    raise SystemExit(main())
