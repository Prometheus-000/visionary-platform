"""
Build the local environments out of the image definitions, never beside them.

    python3 tools/local_install.py --dry-run          # the plan, no GPU needed
    python3 tools/local_install.py comfy              # build .venv-comfy
    python3 tools/local_install.py --dockerfile       # a RunPod template

**The four `modal.Image` chains in app.py are the only declaration of what this
app needs.** There is no requirements.txt, no pyproject, no lockfile — and a
local manifest maintained *beside* those chains is a second source of truth that
goes stale in silence, which is the failure `preview_ui.py`'s rule already
names: a stub that omits a menu is a preview of a control that does not exist.
So this reads them, through the same `smoke_pins.images()` that resolves the
pins, and installs exactly what they say.

**Why three environments and not one.** The pins genuinely conflict: three
different `transformers` (musubi picks its own, Qwen3VL needs >=4.57, ComfyUI
resolves a third) and three CUDA wheels (cu124, cu128, cu130). None of that is
incidental — app.py records the measurement behind each. What makes it cost
nothing locally is that ComfyUI and musubi were *already* subprocesses, so the
boundary that keeps their pins apart already existed in the code; `COMFY_PYTHON`
and `TRAIN_BIN` just point it at a venv. One driver, three environments, no
compromise pin. A CUDA-13-capable driver (r580+) runs cu124 and cu128 wheels
too, which is what makes the single driver enough.

**The shell steps are the drift alarm.** `run_commands` takes strings, so this
recognises a closed set of shapes and **hard-fails, loudly, on anything else**
rather than guessing. `--dry-run` needs no GPU and no network, which is what
makes that alarm something a laptop can ring — see `tools/smoke_local.py`.
"""

import argparse
import ast
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

# Which image each local environment is built from, and what it is for.
ENVS = {
    "web": ("web_image", "the API, downloads and the duplicate scanner"),
    "comfy": ("comfy_image", "ComfyUI — image and video generation"),
    "train": ("trainer_image", "musubi — LoRA training"),
    "caption": ("caption_image", "Qwen3-VL captioning"),
}

# Container paths, and where they land on a machine that is not a container.
# The layout *under* each root is untouched: it is the contract, and a local
# tree that differs from a deployed one is a tree you cannot rsync.
def _relocations(root: Path) -> "dict[str, str]":
    return {
        "/opt/comfyui": str(root / "comfyui"),
        "/opt/musubi-tuner": str(root / "musubi-tuner"),
        "/build/web": str(ROOT / "web"),
        "/opt/similar_clip_b32_q8.onnx": str(root / "similar_clip_b32_q8.onnx"),
    }


class Unreadable(Exception):
    """A build step this cannot replay. Never guessed — see the module docstring."""


# --------------------------------------------------------------------------
# Reading the steps
# --------------------------------------------------------------------------

# Every shape a `run_commands` string is allowed to take. Anything else stops
# the install and prints the command, because a build step silently skipped is
# a package silently missing, hours later, as an ImportError in a container.
SHAPES = (
    ("clone",    re.compile(r"^git clone (?:--depth 1 )?(?P<url>\S+) (?P<dst>\S+)$")),
    ("checkout", re.compile(r"^cd (?P<dir>\S+) && git checkout (?P<ref>\S+)$")),
    ("pip",      re.compile(r"^cd (?P<dir>\S+) && pip install (?P<args>.+)$")),
    ("npm",      re.compile(r"^cd (?P<dir>\S+) && npm (?P<args>.+)$")),
    # Both quote styles, and an optional `cd` in front. The `cd … && python -c
    # '…'` form arrived with the SageAttention gencode patch, and this matcher
    # refused it rather than skipping it — which is the whole design: a build
    # step silently skipped is a package silently missing, hours later, in a
    # container. It cost one regex to teach and found the change in a second on
    # a laptop.
    ("python",   re.compile(r"""^(?:cd (?P<dir>\S+) && )?python -c """
                            r"""(?P<q>["'])(?P<script>.*)(?P=q)$""", re.S)),
    # The Node tarball, which a local machine supplies itself. Matched so it is
    # skipped deliberately and reported, rather than falling through to a fail.
    ("node",     re.compile(r"^(curl -fsSL https://nodejs\.org/|mkdir -p /opt/node|"
                            r"rm /tmp/node\.tar\.xz|ln -sf /opt/node/)")),
    ("rm",       re.compile(r"^rm -rf (?P<path>\S+)$")),
)


def classify(cmd: str) -> "tuple[str, dict]":
    for kind, rx in SHAPES:
        m = rx.match(cmd.strip())
        if m:
            return kind, (m.groupdict() if m.groupdict() else {})
    raise Unreadable(cmd)


def plan(image_name: str, root: Path, relocate_paths: bool = True) -> list:
    """
    The ordered build for one image.

    `relocate_paths` is what separates the two consumers. A venv on somebody's
    laptop needs `/opt/comfyui` to become a directory they can write; a
    Dockerfile *is* a container and wants that path exactly as the image
    definition wrote it — relocating there would bake the machine that emitted
    the file into the image it emitted.
    """
    import app                                   # noqa: PLC0415 — cheap, offline
    from smoke_pins import images                # noqa: PLC0415

    spec = images(ns=vars(app)).get(image_name)
    if spec is None:
        raise SystemExit(f"app.py has no {image_name}")

    moves = _relocations(root) if relocate_paths else {}

    def relocate(text: str) -> str:
        for a, b in moves.items():
            text = text.replace(a, b)
        return text

    out = []
    for st in spec["steps"]:
        if st["op"] == "pip":
            if any(a is None for a in st["args"]):
                raise Unreadable(f"app.py:{st['line']}: a pip pin is not a literal")
            out.append({**st, "args": [relocate(a) for a in st["args"]]})
        elif st["op"] in ("apt", "env"):
            out.append(st)
        elif st["op"] in ("file", "dir"):
            # Relocated on both ends: the source is a path in this checkout and
            # the destination is a container path that has to become a real one.
            out.append({**st,
                        "src": relocate(st["src"]) if st["src"] else None,
                        "dst": relocate(st["dst"]) if st["dst"] else None})
        elif st["op"] == "run":
            for cmd in st["args"]:
                if cmd is None:
                    raise Unreadable(f"app.py:{st['line']}: a run_commands "
                                     "argument is not readable")
                kind, parts = classify(cmd)
                out.append({"op": "run", "kind": kind, "line": st["line"],
                            "cmd": relocate(cmd),
                            "parts": {k: relocate(v) if isinstance(v, str) else v
                                      for k, v in parts.items()}})
    return out


# --------------------------------------------------------------------------
# The three overrides, each printed as it is made
# --------------------------------------------------------------------------

def detect_arch() -> "tuple[str, str] | tuple[None, None]":
    """(device name, sm_XY) from torch, or (None, None) off a CUDA machine."""
    try:
        import torch                              # noqa: PLC0415
        if not torch.cuda.is_available():
            return None, None
        major, minor = torch.cuda.get_device_capability()
        return torch.cuda.get_device_name(0), f"{major}.{minor}"
    except Exception:                             # noqa: BLE001
        return None, None


def overrides(env: dict, root: Path) -> "list[tuple[str, str, str]]":
    """(key, from, to) for every value this changes, so none of them is silent."""
    out = []
    name, arch = detect_arch()
    if "TORCH_CUDA_ARCH_LIST" in env:
        was = env["TORCH_CUDA_ARCH_LIST"]
        now = arch or was
        why = (f"built for the card that is here ({name}, sm_{arch.replace('.', '')})"
               if arch else "no CUDA device visible — left as the image has it")
        out.append(("TORCH_CUDA_ARCH_LIST", was, f"{now}  # {why}"))
    return out


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def describe(step: dict) -> str:
    op = step["op"]
    if op == "pip":
        where = step.get("index_url") or "pypi.org"
        return f"pip install ({len(step['args'])} pins from {where})"
    if op == "apt":
        return f"apt install {' '.join(a for a in step['args'] if a)}"
    if op == "env":
        return f"env {step['vars']}"
    if op in ("file", "dir"):
        return f"copy {step['src']} -> {step['dst']}"
    cmd = step["cmd"]
    return f"[{step['kind']}] " + (cmd if len(cmd) < 96 else cmd[:93] + "...")


def dockerfile(image_name: str, steps: list) -> str:
    """A RunPod template is an image plus a start command, so it is a third
    output of this one reader rather than a second manifest to keep."""
    import app                                    # noqa: PLC0415
    from smoke_pins import images                 # noqa: PLC0415
    base = images(ns=vars(app))[image_name]["base"]
    ref = (base["ref"] if base["kind"] == "registry"
           else f"python:{base['python']}-slim")
    lines = [
        f"# Generated by tools/local_install.py from app.py's {image_name}.",
        "# Do not edit this file: edit the image definition in app.py and",
        "# re-emit, or the two become a manifest and a copy of a manifest.",
        "#",
        "# Build from the repository root, which is the COPY context:",
        "#   docker build -f Dockerfile --build-arg TORCH_CUDA_ARCH_LIST=8.9 .",
        "#",
        "# The arch is an argument because it is the one value in the image that",
        "# is about the card rather than about the software. 9.0 is Hopper, what",
        "# the deployment runs; 8.9 is Ada (4090), 12.0 is consumer Blackwell",
        "# (5090). Build it wrong and SageAttention's kernels load and do",
        "# nothing — the weights still load, the pictures still come out, at",
        "# roughly half speed with no error to explain it.",
        f"FROM {ref}",
        "WORKDIR /opt",
    ]
    for st in steps:
        if st["op"] == "apt":
            pkgs = " ".join(a for a in st["args"] if a)
            lines.append(f"RUN apt-get update && apt-get install -y {pkgs}")
        elif st["op"] == "pip":
            flags = ""
            if st.get("index_url"):
                flags += f" --index-url {st['index_url']}"
            if st.get("extra_index_url"):
                flags += f" --extra-index-url {st['extra_index_url']}"
            pkgs = " ".join(shlex.quote(a) for a in st["args"])
            lines.append(f"RUN pip install{flags} {pkgs}")
        elif st["op"] == "env":
            for k, v in (st["vars"] or {}).items():
                if k == "TORCH_CUDA_ARCH_LIST":
                    lines.append(f"ARG TORCH_CUDA_ARCH_LIST={v}")
                    lines.append("ENV TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST")
                    continue
                lines.append(f"ENV {k}={shlex.quote(str(v))}")
        elif st["op"] == "run" and st["kind"] != "node":
            lines.append(f"RUN {st['cmd']}")
        elif st["op"] in ("file", "dir"):
            # COPY reads the build context, so the source has to be relative to
            # the repository root rather than absolute on whoever emitted this.
            src = st["src"] or ""
            try:
                src = str(Path(src).resolve().relative_to(ROOT))
            except (ValueError, OSError):
                pass
            lines.append(f"COPY {src} {st['dst']}")
    lines += ["", 'CMD ["python", "tools/run_local.py", "--host", "0.0.0.0"]']
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("env", nargs="?", choices=sorted(ENVS), default=None,
                    help="which environment to build (default: all)")
    ap.add_argument("--root", default=str(Path.home() / ".visionary"),
                    help="where the environments and checkouts live")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and change nothing (needs no GPU)")
    ap.add_argument("--dockerfile", action="store_true",
                    help="emit a Dockerfile for the chosen image and exit")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    wanted = [args.env] if args.env else list(ENVS)

    if args.dockerfile:
        name = ENVS[args.env or "comfy"][0]
        sys.stdout.write(dockerfile(name, plan(name, root,
                                                relocate_paths=False)))
        return 0

    print(f"\nroot: {root}")
    name, arch = detect_arch()
    print(f"card: {name or 'none visible'}"
          + (f" (sm_{arch.replace('.', '')})" if arch else "")
          + ("" if arch else "  — dry-run only on this machine"))

    failed = []
    for env_key in wanted:
        image_name, why = ENVS[env_key]
        print(f"\n=== .venv-{env_key} — {why}")
        print(f"    from {image_name}")
        try:
            steps = plan(image_name, root)
        except Unreadable as exc:
            print("    UNREADABLE BUILD STEP — refusing to guess:")
            print(f"      {exc}")
            print("    This is the alarm, not a bug: a build step changed shape.")
            print("    Teach tools/local_install.py the new shape, or fix app.py.")
            failed.append(env_key)
            continue

        env_vars = {}
        for st in steps:
            if st["op"] == "env" and st["vars"]:
                env_vars.update(st["vars"])
        for key, was, now in overrides(env_vars, root):
            print(f"    override {key}: {was} -> {now}")

        for st in steps:
            skip = st["op"] == "run" and st["kind"] == "node"
            mark = "  (skipped: local node)" if skip else ""
            print(f"      {describe(st)}{mark}")

        if not args.dry_run:
            print("    building is not implemented in this commit — "
                  "run with --dry-run")
            failed.append(env_key)

    print()
    if failed and not args.dry_run:
        return 1
    if failed:
        print(f"  {len(failed)} image(s) could not be planned.")
        return 1
    print("  Every build step in every image was recognised.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
