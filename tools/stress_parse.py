"""
Score a parse model in a throwaway Sandbox, before anything is deployed.

The model choice for `/api/parse` should rest on a number rather than an
argument — the same standard the `hf_transfer` entry in CLAUDE.md was settled
by, where measuring reversed a position that had been reasoned about carefully
and wrongly. This spins a GPU Sandbox, serves the candidate with vLLM, runs
`smoke_parse.py` inside it against localhost, prints the score and tears the
Sandbox down.

A Sandbox rather than a deployed function on purpose: nothing about the app
changes to run this, so a model that scores badly costs a few minutes of GPU
and leaves no trace. Only a model that earns it gets wired in.

    python3.11 tools/stress_parse.py
    python3.11 tools/stress_parse.py --model Qwen/Qwen3-4B-Instruct-2507   # control
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
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
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

cd /work && python tools/smoke_parse.py \
    --backend http://localhost:8000/v1 --model {model}
rc=$?
echo "--- vllm tail ---"; tail -5 /tmp/vllm.log
exit $rc
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=ABLITERATED)
    ap.add_argument("--gpu", default="L4",
                    help="4B at bf16 is ~8 GB; L4 is the realistic target.")
    ap.add_argument("--timeout", type=int, default=45 * 60)
    args = ap.parse_args()

    app = modal.App.lookup("visionary-stress-parse", create_if_missing=True)
    # The weights outlive the Sandbox even though nothing else does, so a second
    # candidate or a second attempt is a minute rather than eight.
    cache = modal.Volume.from_name("visionary-stress-hf", create_if_missing=True)
    print(f"model   {args.model}\ngpu     {args.gpu}\n")

    t0 = time.time()
    sb = modal.Sandbox.create(
        "bash", "-lc", SCRIPT.format(model=args.model),
        app=app, image=image, gpu=args.gpu, timeout=args.timeout,
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

    print(f"\n{time.time() - t0:.0f}s wall, exit {code}")
    return code or 0


if __name__ == "__main__":
    raise SystemExit(main())
