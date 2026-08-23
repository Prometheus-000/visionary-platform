"""
What has changed upstream that this app would actually notice.

    python3 tools/upstream.py

`COMFY_SHA` and `CLIFF_SHA` are pins, and the objection to a pin is that you
fall behind without knowing. That objection is right and this is the answer to
it — not unpinning, which does not do what it sounds like.

**Unpinning does not track upstream.** Modal's layer cache key is the command
string, not the world: `git clone --depth 1 <branch>` is a constant, so the
layer caches until something *above* it changes. Unpinned, the build sits on
whatever HEAD was current the last time torch moved, and nothing records which.
A pin is therefore the only thing that pulls new code in deliberately — and the
only thing that says what you got. What it lacks is a reason to look, which is
what this prints.

The whole point is the **filter**. ComfyUI merges partner nodes, frontend
bumps, CI workflows and 3D mesh utilities all week; none of it reaches a render
here. So the commits are bucketed by the paths this app's behaviour actually
rides on, and a hundred commits that touch none of them is the answer "nothing
for you" rather than a number to feel behind about.

It reads the pins out of app.py by AST rather than being told them, for the
reason `_from_app.py` exists: a tool that has to be handed the version it is
checking is a tool that checks the wrong version the first time somebody
forgets.

Network, stdlib only, no Modal and no credentials.
"""

import ast
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app.py"

# What this app's behaviour rides on, and why each one is here. A path not in
# this table is churn as far as we are concerned — that judgement is the tool.
WATCHED = {
    "comfy/ldm/minimax/": "H3 transformer and VAE — the arithmetic behind every clip",
    "comfy_extras/nodes_minimax_h3.py": "the H3 nodes `_h3_graph` names",
    "comfy/text_encoders/minimax.py": "H3's Qwen3-VL conditioner",
    "comfy/ldm/krea2/": "Krea 2 transformer",
    "comfy_extras/nodes_krea2.py": "Krea 2 nodes",
    "comfy/text_encoders/krea2.py": "Krea 2's conditioner",
    "comfy/ldm/wan/": "Wan transformers",
    "comfy_extras/nodes_wan.py": "the Wan nodes `_wan_graph` names",
    "comfy/text_encoders/wan.py": "umT5",
    "comfy/supported_models.py": "sampling_settings — where H3's shift 12.0 lives",
    "comfy/model_sampling.py": "the shift maths ModelSamplingSD3 applies",
    "comfy/samplers.py": "KSAMPLER_NAMES and the scheduler list the page offers",
    "comfy/sd.py": "load_lora_for_models, and the 'NOT LOADED' line _drain counts",
    "comfy/lora.py": "whether a LoRA's keys map onto a DiT at all",
    "comfy/patcher_extension.py": "the hook the regional node pack wraps the model with",
    "comfy/ldm/modules/attention.py": "optimized_attention_override, and the sage call surface",
    "comfy/model_management.py": "unload_all_models, and what /free actually drops",
    "execution.py": "the OOM string `_reclaim` triggers on",
    "comfy/utils.py": "load_torch_file and common_upscale, which the pack calls",
    "requirements.txt": "what pip installs into the image beside our torch pin",
}

REPOS = {
    "COMFY_SHA": ("Comfy-Org/ComfyUI", WATCHED),
    # The node pack is small enough that every file in it is load-bearing, so
    # there is nothing to filter — any commit is worth reading.
    "CLIFF_SHA": (None, None),
}


def pins() -> dict[str, str]:
    """The SHAs as app.py currently holds them."""
    out = {}
    for node in ast.parse(APP.read_text()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in REPOS:
            out[node.targets[0].id] = ast.literal_eval(node.value)
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "CLIFF_REPO":
            out["_cliff_repo"] = ast.literal_eval(node.value)
    return out


def compare(repo: str, base: str, head: str = "HEAD") -> dict:
    url = f"https://api.github.com/repos/{repo}/compare/{base}...{head}"
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "visionary-upstream-check",
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def watched_for(path: str, table: dict) -> str | None:
    for prefix, why in table.items():
        if path == prefix or path.startswith(prefix):
            return why
    return None


def report(label: str, repo: str, sha: str, table: dict | None) -> bool:
    """Print one repo's state. True when something worth a bump moved."""
    print(f"\n{label}  {sha[:10]}  ({repo})")
    try:
        d = compare(repo, sha)
    except urllib.error.HTTPError as exc:
        print(f"  could not compare: HTTP {exc.code} — "
              f"{'rate limited, try again in a few minutes' if exc.code == 403 else exc.reason}")
        return False
    except urllib.error.URLError as exc:
        print(f"  could not compare: {exc.reason}")
        return False

    total = d.get("total_commits", 0)
    if not total:
        print("  up to date")
        return False

    files = d.get("files") or []
    # The compare endpoint caps at 300 files, and silently. A truncated list
    # that reports "nothing for you" is the one answer this tool must never
    # give wrongly, so it says when it cannot see the whole diff.
    capped = len(files) >= 300

    if table is None:
        print(f"  {total} commits behind — every file in this pack is load-bearing")
        for c in d.get("commits", [])[-10:]:
            print(f"    {c['sha'][:8]}  {c['commit']['author']['date'][:10]}  "
                  f"{c['commit']['message'].splitlines()[0][:66]}")
        return True

    hits = []
    for f in files:
        why = watched_for(f["filename"], table)
        if why:
            hits.append((f["filename"], f.get("additions", 0), f.get("deletions", 0), why))

    print(f"  {total} commits behind, {len(files)} files touched"
          + ("  (file list truncated at 300 — read this as a floor)" if capped else ""))
    if not hits:
        print("  nothing on a path this app rides on — no reason to bump")
        return False

    print(f"  {len(hits)} on paths that reach a render:\n")
    for name, add, rm, why in sorted(hits, key=lambda h: -(h[1] + h[2])):
        # A file the API could not diff reports +0/-0; say so rather than
        # printing a zero that reads as "nothing changed here".
        churn = f"+{add}/-{rm}" if (add or rm) else "size not reported"
        print(f"    {churn:>18}  {name}")
        print(f"    {'':>18}  {why}")
    return True


def main() -> int:
    p = pins()
    moved = False
    moved |= report("ComfyUI", "Comfy-Org/ComfyUI", p["COMFY_SHA"], WATCHED)

    repo = re.sub(r"^https://github\.com/", "", p.get("_cliff_repo", "")).rstrip("/")
    if repo:
        moved |= report("Regional node pack", repo, p["CLIFF_SHA"], None)

    print("\n" + ("Worth a bump. Then: modal run tools/smoke_graphs.py"
                  if moved else
                  "Nothing worth a rebuild."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
