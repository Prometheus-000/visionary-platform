"""
Run Visionary on the card in this machine. The local analogue of `modal deploy`.

    python3 tools/run_local.py                       # http://127.0.0.1:8790
    python3 tools/run_local.py --models-dir ~/ComfyUI/models
    python3 tools/run_local.py --api-only            # no front-end build

**This does not import a copy of anything.** It sets `VISIONARY_LOCAL`, imports
`app.py` — the same file `modal deploy` ships — and serves the FastAPI object
that file already builds. Every route, every field, every shot in the palette
and every node in a graph arrives here because it is there, with nobody having
done anything. That is the entire design: the local build cannot drift from the
deployed one, because there is only one of them.

What `VISIONARY_LOCAL` changes is the substrate — see the "Local mode" section
in app.py. Four names, one dispatch function.

**Weights are the expensive thing, so point at the ones you have.**
`--models-dir` aims the app at an existing ComfyUI models folder instead of
downloading its own; anything it does not find there is still a normal download
under the gear. Weights are addressed by exact filename, never scanned, so a
file you own under a different name is invisible — `_require_models()` prints
the directory it looked in and what is actually there, which is how you tell
that apart from an empty folder.

Requires: NVIDIA, a CUDA 13-capable driver (r580+), and the environments built
by `tools/local_install.py`.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HOME = Path(os.environ.get("VISIONARY_HOME", Path.home() / ".visionary"))


def detect_card() -> "tuple[str, str] | tuple[None, None]":
    """(name, sm_XY) off any torch that can see a device, or (None, None).

    Tried through each environment's own interpreter as well as this one,
    because the launcher's venv is the CPU one and need not have torch at all.
    """
    probe = ("import torch,json;"
             "d=torch.cuda.is_available();"
             "print(json.dumps([torch.cuda.get_device_name(0),"
             "'%d.%d'%torch.cuda.get_device_capability()] if d else [None,None]))")
    for exe in (sys.executable, DEFAULT_HOME / ".venv-comfy" / "bin" / "python"):
        try:
            out = subprocess.run([str(exe), "-c", probe], capture_output=True,
                                 text=True, timeout=60)
            if out.returncode == 0:
                import json
                name, cap = json.loads(out.stdout.strip().splitlines()[-1])
                if name:
                    return name, cap
        except Exception:                        # noqa: BLE001 — absence is an answer
            continue
    return None, None


def frontend_is_stale(dist: Path) -> "str | None":
    """Why the bundle needs rebuilding, or None if it does not.

    The image rebuilds on layer invalidation and never asks this question. Here
    it has to be asked, because the rule the image follows — never ship a stale
    dist — is the one thing a local build could quietly get wrong.
    """
    index = dist / "index.html"
    if not index.is_file():
        return "no build yet"
    built = index.stat().st_mtime
    src = ROOT / "web" / "src"
    newest = max((p.stat().st_mtime for p in src.rglob("*") if p.is_file()),
                 default=0)
    return "sources are newer than the build" if newest > built else None


def build_frontend(dist: Path) -> bool:
    why = frontend_is_stale(dist)
    if why is None:
        return True
    print(f"[web] rebuilding the front end — {why}")
    node = subprocess.run(["node", "--version"], capture_output=True, text=True)
    if node.returncode != 0:
        print("\n[web] Node is not on PATH, and the front end is built rather "
              "than shipped.\n"
              "      wanted:  the version web_image pins (see NODE_VERSION in "
              "app.py)\n"
              "      found:   nothing\n"
              "      fix:     install Node, or run with --api-only and use\n"
              "               `npm run dev` in web/ against this server.\n")
        return False
    web = ROOT / "web"
    if not (web / "node_modules").is_dir():
        print("[web] npm ci (first run only, about a minute)")
        if subprocess.run(["npm", "ci"], cwd=web).returncode != 0:
            return False
    return subprocess.run(["npm", "run", "build"], cwd=web).returncode == 0


def report_card(name, cap) -> None:
    """Say which card, which kernels, and what that costs — before anything
    slow starts. The failure this exists for is silent: SageAttention compiled
    for the wrong architecture loads the weights, produces the pictures, and
    runs at roughly half speed with nothing in any log to say so."""
    if not name:
        print("[gpu] no CUDA device visible. Generation and training will not "
              "run;\n      the page and its API will.")
        return
    sm = cap.replace(".", "")
    print(f"[gpu] {name} (sm_{sm})")
    if sm == "90":
        return
    note = {
        "89": "Ada: no FP4 datapath, so H3's NVFP4 text encoder is a memory "
              "saving rather than a speed-up. Prefer the GGUF or INT4 tiers.",
        "120": "consumer Blackwell: NVFP4 runs natively here, which is the "
               "tier to pick for H3.",
        "100": "Blackwell: NVFP4 runs natively here.",
        "80": "Ampere: below what the pinned kernels were built for.",
        "86": "Ampere: below what the pinned kernels were built for.",
    }.get(sm)
    print(f"      the deployment runs Hopper (sm_90); this is sm_{sm}."
          + (f"\n      {note}" if note else ""))
    print("      SageAttention must have been built for sm_" + sm +
          " — tools/local_install.py\n      derives that from this card. If it "
          "was not, ComfyUI runs the slow path\n      and says nothing.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--home", default=str(DEFAULT_HOME),
                    help="where environments, checkouts and the workspace live")
    ap.add_argument("--workspace", default=None,
                    help="records: datasets, LoRAs, outputs, characters")
    ap.add_argument("--models-dir", default=None,
                    help="an existing ComfyUI models folder to read weights from")
    ap.add_argument("--api-only", action="store_true",
                    help="skip the front-end build and serve the API alone")
    args = ap.parse_args()

    home = Path(args.home).expanduser()
    workspace = Path(args.workspace).expanduser() if args.workspace else home / "workspace"
    models = Path(args.models_dir).expanduser() if args.models_dir else workspace / "models"
    for d in (workspace, models):
        d.mkdir(parents=True, exist_ok=True)

    name, cap = detect_card()
    print()
    report_card(name, cap)

    dist = ROOT / "web" / "dist"
    if not args.api_only and not build_frontend(dist):
        print("[web] serving the API without a page. Use --api-only to make "
              "that the intent.")

    # Everything app.py reads, set before it is imported. `GPU`/`VIDEO_GPU`
    # carry the card's own name so the picker offers the card that is here —
    # the menu builds itself out of these, so no front-end change is needed.
    card = name or "local"
    os.environ.update({
        "VISIONARY_LOCAL": "1",
        "VISIONARY_WORKSPACE": str(workspace),
        "VISIONARY_MODELS": str(models),
        "VISIONARY_COMFY": str(home / "comfyui"),
        "VISIONARY_MUSUBI": str(home / "musubi-tuner"),
        "VISIONARY_COMFY_PYTHON": str(home / ".venv-comfy" / "bin" / "python"),
        "VISIONARY_TRAIN_BIN": str(home / ".venv-train" / "bin"),
        "VISIONARY_SPOOL": str(home / "spool"),
        "VISIONARY_DIST": str(dist),
        "VISIONARY_IMAGE_GPU": card,
        "VISIONARY_VIDEO_GPU": card,
    })

    sys.path.insert(0, str(ROOT))
    import app                                    # noqa: PLC0415

    print(f"\n[local] workspace {workspace}")
    print(f"[local] weights   {models}")
    print(f"[local] comfyui   {os.environ['VISIONARY_COMFY']}")

    try:
        import uvicorn                            # noqa: PLC0415
    except ModuleNotFoundError:
        # web_image installs plain `fastapi` rather than `fastapi[standard]`,
        # deliberately: Modal serves the ASGI app itself, so a bundled server is
        # dead weight there. It is not dead weight here — this is the one
        # package the local build needs that the deployment does not.
        print("\n[local] uvicorn is not installed, and locally there is no "
              "Modal to serve\n        the ASGI app. Install it into this "
              "environment:\n\n            pip install uvicorn\n")
        return 1

    api = app.web.get_raw_f()()
    print(f"[local] http://{args.host}:{args.port}\n")
    uvicorn.run(api, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
