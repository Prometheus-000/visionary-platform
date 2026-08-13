"""
Ask whether every pinned wheel in app.py still exists, before a deploy does.

    python3 tools/smoke_pins.py                 # every image
    python3 tools/smoke_pins.py trainer_image   # one of them

`modal deploy app.py` answers this question too, in about twenty minutes, by
downloading tens of gigabytes and compiling CUDA kernels — and then reporting
the answer as an image id and four interleaved logs. This asks it in about a
minute by resolving the pins and installing nothing.

**Why it runs remotely rather than on your laptop.** The obvious version is a
local `pip install --dry-run --platform manylinux2014_x86_64`, and it looks
like it works: it fails before a bad pin is fixed and passes after. It is
lying. With `--platform`, pip cannot evaluate `platform_system == "Linux"`
markers against a Darwin host, so it silently drops every `nvidia-*`
dependency — which is the exact class this file exists to catch. It failed for
a different reason than the deploy did. So the resolve happens in a Sandbox on
the image's own base layer: same OS, same interpreter, same indexes, and none
of the cost, because `--dry-run` downloads metadata rather than wheels.

**What it catches**, and it is one specific thing that has happened here: a
pinned version disappearing from the index that serves it. torch 2.5.1 requires
`nvidia-cudnn-cu12==9.1.0.70`; the PyTorch index is a proxy whose listing for
that package is a page of hrefs pointing at pypi.nvidia.com; NVIDIA pruned the
file; the listing lost the version; the build died naming a package app.py does
not mention. Nothing in the repo had changed.

**What it does not catch.** Each pip group is resolved against the bare base
rather than against the layers before it, so this answers "do these pins still
exist and agree with each other" and not "does the whole image converge". And
pins installed through `run_commands` — ComfyUI's requirements.txt, musubi's
`pip install -e .` — are invisible to it, because they are shell strings rather
than arguments. Both are deliberate: the failure being guarded against is
upstream deletion, which hits the declared pins first and hardest.

The pin lists are read out of app.py by AST rather than copied, for the reason
`_from_app.py` exists: a second list of versions is a list that drifts, and a
checker checking a copy is checking the copy.

**What it printed on the first run that is worth acting on.** `trainer_image`
now carries a pypi.org fallback because it was the one that broke;
`caption_image` and `comfy_image` name a PyTorch index with no fallback at all,
so they are exposed to exactly the same deletion. They resolve today. That is
the difference between a checker and a fix, and this file is only the checker.
"""
import ast
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"


def _const(node):
    """Literal args only. A pin built by an expression is one this cannot read,
    and reporting it as absent would be worse than skipping it."""
    return node.value if isinstance(node, ast.Constant) else None


def images():
    """Every `*_image = modal.Image...` in app.py, with its base and pin groups."""
    tree = ast.parse(APP.read_text())
    out = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.endswith("_image")):
            continue

        # The chain is built inside-out, so walk to the root and reverse.
        calls, cur = [], node.value
        while isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute):
            calls.append(cur)
            cur = cur.func.value
        calls.reverse()

        base, groups = None, []
        for call in calls:
            name = call.func.attr
            kw = {k.arg: _const(k.value) for k in call.keywords}
            if name == "from_registry":
                base = {"kind": "registry", "ref": _const(call.args[0]),
                        "python": kw.get("add_python")}
            elif name == "debian_slim":
                base = {"kind": "slim", "python": kw.get("python_version")}
            elif name == "pip_install":
                pkgs = [p for p in (_const(a) for a in call.args) if p]
                if pkgs:
                    groups.append({"pkgs": pkgs,
                                   "index_url": kw.get("index_url"),
                                   "extra_index_url": kw.get("extra_index_url")})
        if base and groups:
            out[node.targets[0].id] = {"base": base, "groups": groups}
    return out


def base_image(spec):
    if spec["kind"] == "registry":
        return modal.Image.from_registry(spec["ref"], add_python=spec["python"])
    return modal.Image.debian_slim(python_version=spec["python"])


def command(groups):
    """One shell script, so one sandbox answers for the whole image."""
    lines = ["set -u"]
    for i, g in enumerate(groups):
        flags = ""
        if g["index_url"]:
            flags += f" --index-url {g['index_url']}"
        if g["extra_index_url"]:
            flags += f" --extra-index-url {g['extra_index_url']}"
        pkgs = " ".join(f"'{p}'" for p in g["pkgs"])
        lines += [
            f"echo '@@@ group {i}'",
            # --dry-run resolves and reports; --ignore-installed makes the base
            # layer's own packages irrelevant to the answer.
            f"pip install --dry-run --ignore-installed --quiet{flags} {pkgs}"
            f" >/dev/null 2>/tmp/err{i} && echo '@@@ ok' ||"
            f" {{ echo '@@@ FAIL'; tail -4 /tmp/err{i}; }}",
        ]
    return "\n".join(lines)


def check(name, spec, app):
    print(f"\n=== {name} ===")
    for g in spec["groups"]:
        where = g["index_url"] or "pypi.org"
        extra = f" (+{g['extra_index_url']})" if g["extra_index_url"] else ""
        print(f"  {len(g['pkgs'])} pins from {where}{extra}")

    sb = modal.Sandbox.create(
        "sh", "-c", command(spec["groups"]),
        image=base_image(spec["base"]), app=app, timeout=900,
    )
    sb.wait()
    out = sb.stdout.read()

    bad = []
    idx = -1
    for line in out.splitlines():
        if line.startswith("@@@ group"):
            idx = int(line.split()[-1])
        elif line.startswith("@@@ ok"):
            print(f"  ok   group {idx}")
        elif line.startswith("@@@ FAIL"):
            bad.append(idx)
            print(f"  FAIL group {idx}: {', '.join(spec['groups'][idx]['pkgs'][:3])}…")
        elif bad and line.strip() and not line.startswith("@@@"):
            print(f"       {line.strip()[:150]}")
    return bad


def main():
    want = sys.argv[1:]
    found = images()
    if want:
        missing = [w for w in want if w not in found]
        if missing:
            sys.exit(f"no such image in app.py: {', '.join(missing)} "
                     f"(have: {', '.join(found)})")
        found = {k: v for k, v in found.items() if k in want}

    failed = {}
    app = modal.App("visionary-smoke-pins")
    with modal.enable_output(), app.run():
        for name, spec in found.items():
            bad = check(name, spec, app)
            if bad:
                failed[name] = bad

    print()
    if failed:
        for name, groups in failed.items():
            print(f"{name}: group(s) {groups} no longer resolve", file=sys.stderr)
        print("\nA pin that vanished is almost never yours to fix by changing the "
              "version — check whether the index still serves it, and give pip "
              "somewhere else to look before bumping anything.", file=sys.stderr)
        sys.exit(1)
    print(f"every declared pin still resolves ({len(found)} image(s))")


if __name__ == "__main__":
    main()
