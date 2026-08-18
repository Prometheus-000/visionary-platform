"""
Score a parse model in a throwaway Sandbox, before anything is deployed.

The model choice for `/api/parse` should rest on a measurement rather than an
argument — the same standard the `hf_transfer` entry in CLAUDE.md was settled
by, where measuring reversed a position that had been reasoned about carefully
and wrongly. **A number is not the same thing as a measurement, and reaching for
one is what went wrong here**: eleven scored rows, every one of them a string
comparison, and none of them asking whether the picture got better. This spins a GPU Sandbox, serves the candidate with vLLM, runs
`smoke_parse.py` inside it against localhost, prints the score and tears the
Sandbox down.

A Sandbox rather than a deployed function on purpose: nothing about the app
changes to run this, so a model that scores badly costs a few minutes of GPU
and leaves no trace. Only a model that earns it gets wired in.

    python3.11 tools/stress_parse.py                       # the pinned fork
    python3.11 tools/stress_parse.py --model A --model B   # side by side

**Pass more than one `--model` and this becomes the comparison harness**, which
is the gate the semantic layer is supposed to clear before anything is wired: a
Sandbox per candidate, the same corpus through each, and one table at the end.

**That decision rule was wrong, and the table it reads is measuring the wrong
thing.** It used to say the most restrained model wins and that a 14B
interpreting elaborately loses to a 4B interpreting sparsely. Every row in the
table is text-to-text — preserved, covered, round-tripped — so a model that
returns the sentence unchanged scores perfectly, and that is what the incumbent
did: zero invented words across 27 fragments, read as maximum restraint, with
the feature reaching 0% of renders. Restraint was also never a property of the
*model*: `PARSE_RULES` forbade invention, and swapping that one block
(`smoke_parse.py --enrich`) made the same weights author 9 of 18 fragments.

So the comparison to run is `--enrich --scenes`, and the criteria are read
rather than totalled:

    1 core subject      is what they were looking at the subject of the prompt
    2 emotional tone    "lonely" staged as isolation, not typed back out
    3 spatial logic     the man, the lamp and the wet ground in right relation
    4 literal fidelity  the blue tiles and the yellow sweater survive exactly

Semantic contradiction is still disqualifying and still enforced by nothing —
see CLAUDE.md, where the measurements behind all of this are recorded.
"""

import argparse
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
ABLITERATED = "huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated"

# vLLM brings its own CUDA wheels, so a slim base is enough and the build stays
# one layer. The three local files are the real compiler and the real harness —
# `_from_app.py` pulls from app.py rather than a copy, for the reason that file
# exists.
# vLLM unpinned rather than held at 0.11.0, which failed here on
# `Qwen2Tokenizer has no attribute all_special_tokens_extended` — its
# `get_cached_tokenizer` reaches for something a newer transformers dropped.
# Worth recording that this is *not* a defect in the fork: the base model
# declares the same `tokenizer_class`, so the fork was the first suspect and
# the wrong one.
# A `devel` CUDA base, not debian_slim: vLLM's inductor shells out to `nvcc`
# to build kernels, and on a slim image engine init dies with "Could not find
# nvcc and default cuda_home='/usr/local/cuda' doesn't exist" — several minutes
# after a successful model load, which is what made it look like a timeout.
# Same base family the repo already uses for `caption_image` and `comfy_image`.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install("vllm", "huggingface_hub")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": "/cache"})
    .add_local_file(ROOT / "app.py", "/work/app.py")
    .add_local_file(ROOT / "tools" / "_from_app.py", "/work/tools/_from_app.py")
    .add_local_file(ROOT / "tools" / "smoke_parse.py", "/work/tools/smoke_parse.py")
    # The 10.2k rules, carried so `--enrich long` can be run against `shipped`.
    # Shorter-is-better is a claim worth a measurement rather than a maxim.
    .add_local_file(ROOT / "tools" / "rules-long.txt", "/work/tools/rules-long.txt")
)

SCRIPT = r"""
set -o pipefail
echo "vllm $(python -c 'import vllm; print(vllm.__version__)') / transformers $(python -c 'import transformers; print(transformers.__version__)')"
echo "--- serving {model}"
# torch.compile is left on: it costs ~3 min on a cold L4 and is the
# configuration engine init is known to survive here. Caching it to the
# Volume was tried and is far worse — thousands of small artifacts over 9P
# never finished inside ten minutes. --enforce-eager was tried and engine
# init failed, so the compile is not what to economise on.
vllm serve {model} \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 {serve_extra} \
    > /tmp/vllm.log 2>&1 &

# Probed with python, not curl: the CUDA base has no curl, so `curl -sf` fails
# with "command not found" every iteration and the loop spins out its full
# window while the server sits there serving. Cost two runs to notice, because
# the symptom is identical to a slow start.
for i in $(seq 1 900); do
    if python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=3)" 2>/dev/null; then
        echo "--- up after ${{i}}s"; break
    fi
    if [ "$i" = "900" ]; then
        echo "--- never came up; root cause ---"
        grep -nE "Error|Exception|Traceback|CUDA|out of memory|Killed" /tmp/vllm.log \
            | grep -v "^.*ErrorHandler" | head -25
        echo "--- last 15 ---"; tail -15 /tmp/vllm.log
        exit 1
    fi
    sleep 1
done

# VRAM off vLLM's own load line, never nvidia-smi: --gpu-memory-utilization
# 0.90 makes the card read 90% full whatever the model weighs, so nvidia-smi
# measures the flag rather than the candidate. What is wanted is the weights,
# because that is the number that decides which card this can run on.
grep -oE "Model loading took [0-9.]+ (GiB|GB)" /tmp/vllm.log | head -1 \
    | sed -E "s/Model loading took ([0-9.]+).*/vram \1 GiB/" || true
grep -oE "GPU KV cache size: [0-9,]+ tokens" /tmp/vllm.log | head -1 || true

cd /work && {entry}
rc=$?
echo "--- vllm tail ---"; tail -5 /tmp/vllm.log
exit $rc
"""


def _serve_extra(args) -> str:
    """Extra `vllm serve` flags, which only the vision run needs.

    Two images is not a small ask: Qwen3-VL tiles an image into patches and a
    1152x864 render is thousands of tokens before a word of the rubric is read.
    Left at the text defaults the server accepts the request and then dies
    mid-decode, which arrives here as a terminated Sandbox and no log — the
    failure looks like infrastructure and is a context-length bug.

    `--limit-mm-per-prompt` is the declaration that two pictures per request is
    the contract; vLLM allocates multimodal cache from it, so leaving it at the
    default of one is the other half of the same crash.
    """
    if not args.vision:
        return ""
    return "--limit-mm-per-prompt '{\"image\":2}' --max-num-seqs 4"


def _entry(args, model: str) -> str:
    """The one command the Sandbox runs once the server is up.

    Built whole rather than through a `{extra}` placeholder, because
    `str.format` is single-pass: a nested field inside the substituted value
    comes out literal, and the failure is a Sandbox that starts, serves, and
    then runs a command with `{extra}` in it.
    """
    base = f"--backend http://localhost:8000/v1 --model {model}"
    if args.vision:
        return ("python tools/judge_renders.py --pairs /work/pairs " + base
                + (" --briefs /work/dump.log" if args.vision_briefs else ""))
    extra = " ".join(f for f in (
        f"--enrich {args.enrich}" if args.enrich else "",
        "--scenes" if args.scenes else "",
        "--judge /work/dump.log" if args.judge else "",
        "--dump" if args.dump else "",
        "--refusal" if args.refusal else "",
        "--patched" if args.patched else "") if f)
    return f"python tools/smoke_parse.py {base} {extra}".rstrip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=None,
                    help="Repeat to compare candidates. Defaults to the pinned "
                         "fork. Suffix `@GPU` to give one candidate its own card.")
    ap.add_argument("--gpu", default="L4",
                    help="4B at bf16 is ~8 GB; L4 is the realistic target.")
    ap.add_argument("--timeout", type=int, default=60 * 60)
    ap.add_argument("--patched", action="store_true",
                    help="Run with the proposed PARSE_RULES change appended.")
    ap.add_argument("--refusal", action="store_true",
                    help="Run the charged corpus only — the claim the vendored "
                         "fork rests on.")
    ap.add_argument("--enrich", nargs="?", const="block", default=None,
                    choices=["shipped", "long", "block", "full", "stage"],
                    help="Swap the restraint block for an authoring one. The "
                         "rules forbid invention, so no measurement of this "
                         "model has ever been a measurement of what it can do.")
    ap.add_argument("--judge", default=None, metavar="DUMP",
                    help="Serve the candidate as a *judge* and score this dump "
                         "against the four criteria. Use a different --model "
                         "from the one that wrote it: a model marking its own "
                         "work agrees with itself.")
    ap.add_argument("--scenes", action="store_true",
                    help="The recollection corpus rather than the fragments.")
    ap.add_argument("--vision", default=None, metavar="PAIRS_DIR",
                    help="Score rendered pairs instead of text — the only "
                         "measurement here that is not a proxy. Takes a folder "
                         "of <name>_bare.png / <name>_rich.png from "
                         "`does_it_help.py`, and wants a *vision* model: "
                         "--model Qwen/Qwen3-VL-4B-Instruct.")
    ap.add_argument("--vision-briefs", default=None, metavar="DUMP",
                    help="The --enrich dump the pairs came from, so the judge "
                         "reads the person's fragment rather than a filename.")
    ap.add_argument("--dump", action="store_true",
                    help="Ask the harness for raw documents instead of a score, "
                         "so a proposed trust change can be measured against real "
                         "output without renting the card once per idea.")
    args = ap.parse_args()

    # `repo@gpu`, because the candidates do not fit one card and a table split
    # across two invocations is a table nobody compares. A 14B at bf16 is ~28 GB
    # against an L4's 24, so it cannot be measured on the card the 4B is
    # measured on — which is itself a finding the VRAM row is there to carry,
    # not a detail to hide by running it somewhere else quietly.
    candidates = [(m.split("@")[0], (m.split("@") + [args.gpu])[1])
                  for m in (args.model or [ABLITERATED])]

    app = modal.App.lookup("visionary-stress-parse", create_if_missing=True)
    # The weights outlive the Sandbox even though nothing else does, so a second
    # candidate or a second attempt is a minute rather than eight.
    cache = modal.Volume.from_name("visionary-stress-hf", create_if_missing=True)
    print("candidates\n" + "\n".join(f"  {m}  on {g}" for m, g in candidates) + "\n")

    rows, worst = [], 0
    for model, gpu in candidates:
        print(f"\n{'=' * 72}\n=== {model}  on {gpu}\n{'=' * 72}")
        t0 = time.time()
        img = image
        if args.vision:
            # A directory rather than a file, because a pair is two images and
            # the judge wants them side by side. Added to the image for the same
            # reason the dump is: it is megabytes, and a command line is the
            # wrong place for it.
            img = image.add_local_dir(args.vision, "/work/pairs")
            img = img.add_local_file(ROOT / "tools" / "judge_renders.py",
                                     "/work/tools/judge_renders.py")
            if args.vision_briefs:
                img = img.add_local_file(args.vision_briefs, "/work/dump.log")
        if args.judge:
            # The dump rides in as a file rather than an argument: it is
            # thousands of characters of JSON per scene and a command line is
            # the wrong place for it.
            img = image.add_local_file(args.judge, "/work/dump.log")
        sb = modal.Sandbox.create(
            "bash", "-lc", SCRIPT.format(
                model=model, entry=_entry(args, model),
                serve_extra=_serve_extra(args)),
            app=app, image=img, gpu=gpu, timeout=args.timeout,
            volumes={"/cache": cache},
        )
        seen = []
        try:
            for line in sb.stdout:
                print(line, end="", flush=True)
                seen.append(line)
            err = sb.stderr.read()
            if err.strip():
                print("\n--- stderr ---\n" + err[-2000:], file=sys.stderr)
            sb.wait()
            code = sb.returncode
        finally:
            sb.terminate()
        worst = worst or code
        rows.append((model, gpu, _scores(seen), f"{time.time() - t0:.0f}s", code))

    if not (args.dump or args.refusal or args.enrich or args.judge or args.vision):
        _table(rows)
    return worst or 0


def _table(rows) -> None:
    """
    One table, printed last, because the per-candidate output is thousands of
    lines apart and nobody compares two models by scrolling.

    Rows are the criteria and columns are the candidates — the shape the plan
    draws it in, and the shape that matters: this table is read *across* a row
    to compare, never down a column to total. There is no score, deliberately.
    """
    print(f"\n\n{'=' * 72}\n=== the matrix\n{'=' * 72}\n")
    names = [m.split("/")[-1][:16] for m, *_ in rows]
    print(f"  {'':24}" + "".join(f"{n:>18}" for n in names))
    print(f"  {'':24}" + "".join(f"{g:>18}" for _, g, *_ in rows))
    print("  " + "─" * (24 + 18 * len(rows)))
    for key, label in ROWS:
        cells = "".join(f"{got.get(key, '—'):>18}" for _, _, got, _, _ in rows)
        mark = "  ←" if key in ("preserved", "contradiction") else ""
        print(f"  {label:24}{cells}{mark}")
    print(f"  {'wall clock':24}" + "".join(f"{w:>18}" for *_, w, _ in rows))
    # A nonzero exit is the *expected* case here and must not be read as a
    # broken run: `smoke_parse.py` returns 1 whenever a candidate scored less
    # than perfect, which is what a comparison is for. What actually invalidates
    # a column is a missing cell — the server never came up, or the harness died
    # partway — so that is what is flagged.
    for model, _, got, _, code in rows:
        absent = [label for key, label in ROWS if key not in got]
        if absent:
            print(f"  !! {model} (exit {code}) never reported: "
                  f"{', '.join(absent)} — read its column as incomplete")

    print("\n  ← the two rows that decide it. User-text preservation carries the"
          "\n    highest weight; semantic contradiction is disqualifying. At"
          "\n    comparable preservation and fidelity, take the most restrained"
          "\n    model rather than the largest — a 14B that interprets elaborately"
          "\n    loses to a 4B that interprets sparsely, because every assumption"
          "\n    it adds is one the person now has to notice and undo.")
    print("\n  Contradiction is facts replaced over facts stated, and it is the one"
          "\n    row nothing in the app can enforce: an in-place substitution marked"
          "\n    invented passes preservation and coverage both. That is why it is"
          "\n    measured here and why the column is worth reading before the rest.")


# The matrix, in the order the plan writes it: highest weight first, and the
# disqualifying row second so it cannot be skimmed past. `key` is what
# `smoke_parse.py` prints at the head of a line; `label` is what a reader needs.
ROWS = [
    ("preserved", "user text preserved"),
    ("empty", "degraded to nothing"),
    ("contradiction", "semantic contradiction"),
    ("inventionshare", "invented spans"),
    ("schema", "schema validity"),
    ("relations", "relationship accuracy"),
    ("entities", "entity accuracy"),
    ("refusal", "refusal behaviour"),
    ("idempotency", "idempotency"),
    ("reroll", "reroll safety"),
    ("fidelity", "round-trip fidelity"),
    ("compliance", "rule compliance"),
    ("matrixlatency", "latency"),
    ("vram", "VRAM (weights)"),
]


def _scores(lines: list[str]) -> dict[str, str]:
    """Every row `smoke_parse.py` prints, lifted back out of its output.

    Matched on a line *starting* with the key, which is why the harness prints
    the cells as bare `key value` lines at the end of its run: a table parsed
    out of prose is a table that breaks when somebody rewords a sentence.
    """
    out: dict[str, str] = {}
    for line in lines:
        head = line.split(maxsplit=1)
        if len(head) == 2 and head[0] in {k for k, _ in ROWS}:
            out[head[0]] = head[1].strip()
    return out


if __name__ == "__main__":
    raise SystemExit(main())
