"""
Serve a model in a throwaway Sandbox and mark something with it.

    python3.11 tools/serve_judge.py --dump enrich.jsonl        # score prompts
    python3.11 tools/serve_judge.py --pairs out/ --briefs d.log --model <vlm>

A Sandbox rather than a deployed function, on purpose: nothing about the app
changes to run this, so a judge that turns out to be useless costs a few minutes
of GPU and leaves no trace.

**It was `stress_parse.py`, which existed to choose a model for `/api/parse` by
measurement rather than by argument.** That route is gone and the reason it is
gone is the thing this file should still be read for: eleven scored rows, every
one of them a string comparison, and not one of them asking whether the picture
got better. A model that returned the sentence unchanged scored perfectly, which
is exactly what the incumbent did — zero invented words across 27 fragments,
read as maximum restraint, with the feature reaching 0% of renders.

So what survives is the serving, and the two things worth serving:

  --dump    text. `judge_prompts.py` marks a rewrite against what the person
            said, on four criteria that are read rather than totalled.
  --pairs   pictures. `judge_renders.py` scores rendered pairs blind, both
            orders, a win counted only when both agree. **The only measurement
            here that is not a proxy**, and what `prompt_ab.py` drives.

Every line of the vLLM recipe below was settled by running it, and each carries
the failure that put it there — the `devel` CUDA base, the unpinned vLLM, the
urllib health probe, the multimodal limits. None of it is taste.
"""

import argparse
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install("vllm", "huggingface_hub")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": "/cache"})
    .add_local_file(ROOT / "tools" / "judge_prompts.py", "/work/tools/judge_prompts.py")
    .add_local_file(ROOT / "tools" / "judge_renders.py", "/work/tools/judge_renders.py")
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


def _serve_extra(pairs: str) -> str:
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
    if not pairs:
        return ""
    return "--limit-mm-per-prompt '{\"image\":2}' --max-num-seqs 4"


def _entry(args, model: str) -> str:
    """The one command the Sandbox runs once the server is up.

    Built whole rather than through a `{extra}` placeholder, because
    `str.format` is single-pass: a nested field inside the substituted value
    comes out literal, and the failure is a Sandbox that starts, serves, and
    then runs a command with `{extra}` in it.
    """
    if args.pairs:
        return (f"python tools/judge_renders.py --pairs /work/pairs "
                f"--backend http://localhost:8000/v1 --model {model}"
                + (" --briefs /work/dump.log" if args.briefs else ""))
    return (f"python tools/judge_prompts.py /work/dump.log "
            f"--at http://localhost:8000/v1 --model {model}"
            + (f" --subject-model {args.subject_model}" if args.subject_model else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description="Serve a judge in a Sandbox and "
                                             "mark something with it.")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct",
                    help="The judge's weights. Point this at something other "
                         "than what wrote the thing being judged — a model "
                         "marking its own work agrees with itself.")
    ap.add_argument("--gpu", default="L4",
                    help="4B at bf16 is ~8 GB; L4 is the realistic target.")
    ap.add_argument("--timeout", type=int, default=60 * 60)
    ap.add_argument("--dump", default=None, metavar="JSONL",
                    help="Score prompts: one JSON object per line carrying "
                         "name, prose and compiled.")
    ap.add_argument("--pairs", default=None, metavar="DIR",
                    help="Score pictures: a folder of <name>_bare.png beside "
                         "<name>_rich.png, which is what does_it_help.py "
                         "writes. Wants a *vision* model.")
    ap.add_argument("--briefs", default=None, metavar="JSONL",
                    help="With --pairs, the dump the renders came from, so the "
                         "judge reads the person's sentence rather than a "
                         "filename.")
    ap.add_argument("--subject-model", default="",
                    help="With --dump, the weights that wrote the rewrites.")
    args = ap.parse_args()
    if bool(args.dump) == bool(args.pairs):
        ap.error("exactly one of --dump or --pairs")

    app = modal.App.lookup("visionary-serve-judge", create_if_missing=True)
    cache = modal.Volume.from_name("visionary-stress-hf", create_if_missing=True)

    img = image
    if args.pairs:
        # The pairs ride in on the image rather than a Volume: they are a few
        # megabytes and the judge wants them side by side.
        img = img.add_local_dir(args.pairs, "/work/pairs")
        if args.briefs:
            img = img.add_local_file(args.briefs, "/work/dump.log")
    else:
        # The dump rides in as a file rather than an argument: it is thousands
        # of characters of JSON per row and a command line is the wrong place
        # for it.
        img = img.add_local_file(args.dump, "/work/dump.log")

    t0 = time.time()
    sb = modal.Sandbox.create(
        "bash", "-lc", SCRIPT.format(
            model=args.model, entry=_entry(args, args.model),
            serve_extra=_serve_extra(args.pairs)),
        app=app, image=img, gpu=args.gpu, timeout=args.timeout,
        volumes={"/cache": cache},
    )
    try:
        for line in sb.stdout:
            print(line, end="", flush=True)
        err = sb.stderr.read()
        if err.strip():
            print("\n--- stderr ---\n" + err[-2000:], file=sys.stderr)
        sb.wait()
        code = sb.returncode
    finally:
        sb.terminate()
    print(f"\n--- {time.time() - t0:.0f}s")
    return code or 0


if __name__ == "__main__":
    raise SystemExit(main())
