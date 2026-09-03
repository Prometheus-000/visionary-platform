"""
Does the local runtime still fit the app it is a runtime for?

    python3 tools/smoke_local.py

Runs on a laptop: no GPU, no CUDA, no Modal account, no network. That is the
whole point of it. The local build is for people who are not the person
maintaining it, and the person maintaining it is on a Mac — so the only check
that will actually fire between rented boxes is one that costs ten seconds and
needs nothing.

**What can rot.** Not the features: there is one `app.py` and the local runtime
imports it, so a new route, a new field, a new shot in the vocabulary and a new
node in a graph all arrive locally with nobody doing anything. What can rot is
the *seam* — the four names `LOCAL` rebinds, and the one function dispatch goes
through. A ninth `.spawn()`, a `jobs.keys()`, a `volume.remove_file()`: each is
a perfectly good line to write against Modal, and each silently has no meaning
off it. So the surfaces are asserted closed rather than assumed closed, and the
failure lands here instead of on a stranger's 4090 an hour into a render.

Graph validity is deliberately not re-checked here — `tools/smoke_graphs.py`
does that against a real ComfyUI's `/object_info`, and a second opinion written
against a hand-made list would be a check on the list.
"""

import ast
import os
import pathlib
import sys
import tempfile
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP_PY = ROOT / "app.py"

# What a `modal.Dict` and a `modal.Volume` are allowed to be asked for. Every
# name here is answered by `_LocalDict` / `_LocalVolume` in app.py; anything
# else in the file is a call that works on Modal and does nothing off it.
# `keys` joined these when main's heartbeat sweep started iterating
# `sessions`. It belongs rather than being waved through: `modal.Dict`
# has it and `_LocalDict` inherits it from `dict`, so both sides really
# answer — which is what the second direction of the check below
# verifies, so widening this cannot quietly become a way to pass.
DICT_SURFACE = {"get", "pop", "keys"}
VOLUME_SURFACE = {"commit", "reload", "read_file", "listdir", "root"}

DICT_NAMES = {"jobs", "config", "sessions"}
VOLUME_NAMES = {"volume", "models_volume", "hf_cache"}


def _tree() -> ast.Module:
    return ast.parse(APP_PY.read_text())


def _load_local():
    """app.py in local mode, against a workspace that is a temp directory."""
    root = pathlib.Path(tempfile.mkdtemp(prefix="visionary-smoke-"))
    os.environ["VISIONARY_LOCAL"] = "1"
    os.environ["VISIONARY_WORKSPACE"] = str(root)
    os.environ["VISIONARY_MODELS"] = str(root / "models")
    # Scratch is deliberately not under the workspace, so it is pointed
    # somewhere the check can still verify it followed instructions.
    os.environ["VISIONARY_DRAFTS"] = str(root / "scratch" / "drafts")
    os.environ["VISIONARY_WORK"] = str(root / "scratch" / "work")
    os.environ["VISIONARY_SPOOL"] = str(root / "scratch" / "spool")
    sys.path.insert(0, str(ROOT))
    import app  # noqa: PLC0415
    return app, root


# --------------------------------------------------------------------------
# The checks. Each returns a list of complaints; empty means it passed.
# --------------------------------------------------------------------------

def check_imports_offline(app, root) -> list[str]:
    if not app.LOCAL:
        return ["VISIONARY_LOCAL was set and app.LOCAL is False"]
    if app.WORKSPACE != root:
        return [f"WORKSPACE is {app.WORKSPACE}, not the {root} it was given"]
    return []


def check_rebound(app, root) -> list[str]:
    bad = []
    for name in sorted(DICT_NAMES):
        got = type(getattr(app, name)).__name__
        if got != "_LocalDict":
            bad.append(f"{name} is {got}, not _LocalDict")
    for name in sorted(VOLUME_NAMES):
        got = type(getattr(app, name)).__name__
        if got != "_LocalVolume":
            bad.append(f"{name} is {got}, not _LocalVolume")
    # The decorators must still hold the real thing, or `modal deploy` breaks.
    for path, vol in app.MODAL_VOLUMES.items():
        if type(vol).__name__ != "Volume":
            bad.append(f"MODAL_VOLUMES[{path!r}] is {type(vol).__name__}, "
                       "not a real Volume — modal deploy would reject it")
    return bad


def check_surfaces_closed(app) -> list[str]:
    """Every attribute app.py asks of a Dict or a Volume, against what the
    shims answer. This is the assertion the whole seam rests on."""
    bad = []
    watched = {n: DICT_SURFACE for n in DICT_NAMES}
    watched.update({n: VOLUME_SURFACE for n in VOLUME_NAMES})
    for node in ast.walk(_tree()):
        if not isinstance(node, ast.Attribute):
            continue
        base = node.value
        if not isinstance(base, ast.Name) or base.id not in watched:
            continue
        if node.attr.startswith("__"):
            continue
        if node.attr not in watched[base.id]:
            bad.append(f"app.py:{node.lineno} asks for "
                       f"{base.id}.{node.attr}, which the shim does not answer")
    # And the other direction: the shims must really have what is claimed.
    for name, shim, surface in (("_LocalDict", app.jobs, DICT_SURFACE),
                                ("_LocalVolume", app.volume, VOLUME_SURFACE)):
        for attr in sorted(surface):
            if not hasattr(shim, attr):
                bad.append(f"{name} is missing {attr}")
    if not hasattr(app.volume.commit, "aio"):
        bad.append("volume.commit has no .aio — /api/upload awaits it")
    return bad


def check_dispatch_single_door(app) -> list[str]:
    """`.spawn()` may appear in exactly one place: inside `_spawn`."""
    bad = []
    tree = _tree()
    spawn_def = next((n for n in tree.body
                      if isinstance(n, ast.FunctionDef) and n.name == "_spawn"), None)
    if spawn_def is None:
        return ["app.py has no top-level _spawn"]
    inside = {ln for n in ast.walk(spawn_def)
              if (ln := getattr(n, "lineno", None)) is not None}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and node.attr == "spawn"
                and node.lineno not in inside):
            bad.append(f"app.py:{node.lineno} calls .spawn() outside _spawn — "
                       "it will do nothing locally")
    return bad


def check_lane_table(app) -> list[str]:
    """Every name in the lane table has to be a thing that exists, or the
    table is documentation of a dispatch target that has been renamed."""
    bad = []
    for name, lane in sorted(app._LANE_OF.items()):
        if lane not in ("gpu", "cpu", "inline"):
            bad.append(f"{name} routes to unknown lane {lane!r}")
        if "." in name:
            cls_name, meth = name.split(".", 1)
            cls = getattr(app, cls_name, None)
            if cls is None:
                bad.append(f"lane table names {cls_name}, which app.py has not")
                continue
            user = cls._get_user_cls()          # the one private API in the seam
            if not hasattr(user, meth):
                bad.append(f"lane table names {name}, which does not exist")
        else:
            fn = getattr(app, name, None)
            if fn is None:
                bad.append(f"lane table names {name}, which app.py has not")
            elif getattr(fn.info, "function_name", None) != name:
                bad.append(f"{name}.info.function_name is "
                           f"{getattr(fn.info, 'function_name', None)!r} — "
                           "lane routing keys off that string")
    return bad


def check_web_builds(app) -> list[str]:
    api = app.web.local()
    routes = [r for r in api.routes if hasattr(r, "path")]
    if len(routes) < 50:
        return [f"web() built only {len(routes)} routes"]
    check_web_builds.count = len(routes)
    return []


def check_job_contract(app) -> list[str]:
    """Seed, publish, poll, stop — the contract, driven through the local lane
    exactly as a route drives it."""
    bad = []
    api = app.web.local()
    status = next((r.endpoint for r in api.routes
                   if getattr(r, "path", "") == "/api/status/{job_id}"), None)
    if status is None:
        return ["no /api/status/{job_id} route to poll"]

    started, released, order = threading.Event(), threading.Event(), []

    class _Fake:
        """Stands in for a Modal Function: a name to route on and a body."""
        def __init__(self, name, body):
            self.info = type("I", (), {"function_name": name})()
            self._body = body

        def local(self, **kw):
            return self._body(**kw)

    def slow(job_id):
        order.append(f"start {job_id}")
        app._publish(job_id, status="running", phase="working")
        started.set()
        released.wait(5)
        app._publish(job_id, status="completed")
        order.append(f"end {job_id}")

    app.jobs["j1"] = {"status": "queued"}
    app.jobs["j2"] = {"status": "queued"}
    app._spawn(_Fake("train_job", slow), job_id="j1")
    if not started.wait(5):
        return ["the gpu lane never started the first job"]
    app._spawn(_Fake("caption_job", slow), job_id="j2")
    time.sleep(0.3)

    # j2 must be waiting behind j1, and must say so.
    rec2 = status("j2")
    if rec2.get("status") != "queued":
        bad.append(f"queued job reports {rec2.get('status')!r}, not 'queued'")
    if "ahead" not in str(rec2.get("phase", "")) and "next up" not in str(rec2.get("phase", "")):
        bad.append(f"a queued job's phase is {rec2.get('phase')!r} — it should "
                   "say what it is waiting behind")
    if "end j1" in order:
        bad.append("the gpu lane ran two jobs at once — one card, one render")

    # Stop, honoured while it waits.
    app._request_stop("j2")
    released.set()
    for _ in range(50):
        if status("j2").get("status") == "stopped":
            break
        time.sleep(0.1)
    else:
        bad.append(f"a queued job ignored Stop; it reports "
                   f"{status('j2').get('status')!r}")
    if app._stop_requested("j2"):
        bad.append("the stop flag outlived the job it stopped")
    if status("j1").get("status") != "completed":
        bad.append(f"the first job ended as {status('j1').get('status')!r}")
    return bad


# The volume's own split, as a test: a *record* is what you made and follows
# the workspace, and *derived* scratch deliberately does not — it lives on the
# container's disk and dies with it. This check demanded that everything follow
# the workspace, so main moving drafts and per-run scratch off the volume read
# as a regression when it was the rule being applied. What is worth asserting is
# not that scratch sits inside the workspace; it is that scratch is *relocatable
# at all*, because "the container's disk" is /tmp in a container and is tmpfs —
# memory — on most Linux workstations, where an unsaved set or a resized
# training copy would be gigabytes into RAM.
RECORD_PATHS = ("LORAS", "DATASETS", "OUTPUTS", "CHARACTERS", "WORKFLOWS")
SCRATCH_PATHS = (("DRAFTS", "VISIONARY_DRAFTS"),
                 ("WORK", "VISIONARY_WORK"),
                 ("SPOOL", "VISIONARY_SPOOL"))


def check_paths_follow_env(app, root) -> list[str]:
    bad = []
    for name in RECORD_PATHS:
        p = getattr(app, name, None)
        if p is None:
            bad.append(f"app.py has no {name}")
            continue
        if root not in p.parents and p != root:
            bad.append(f"{name} is {p}, outside the workspace it was given")
    if app.STAGING.parents and root not in app.STAGING.parents:
        bad.append(f"STAGING is {app.STAGING}, outside the workspace")
    for name, var in SCRATCH_PATHS:
        p = getattr(app, name, None)
        if p is None:
            bad.append(f"app.py has no {name}")
            continue
        if os.environ.get(var) and str(p) != os.environ[var]:
            bad.append(f"{name} is {p}, not the {os.environ[var]} it was given")
    return bad


def check_tier_resolution(app, root) -> list[str]:
    """
    `_slot_name` picks by card, with every candidate actually on disk.

    The branch this covers cannot run on Modal at all — `GPU_VRAM_GB` is 0
    there, so the early return hands back the catalogue's own row and the
    arithmetic below it is dead. That makes it exactly the kind of code that
    ships broken: the only machine that exercises it is a stranger's, and the
    first symptom is a 26 GB bf16 checkpoint chosen for a 12 GB card.

    The dests are pointed at empty temp files, because `here` filters on
    `is_file()` and a laptop has none of these weights. What is under test is
    the decision, not the download.
    """
    import copy                                    # noqa: PLC0415
    bad = []
    saved_vram = app.GPU_VRAM_GB
    saved_rows = copy.deepcopy({k: dict(v) for k, v in app.MODEL_CATALOGUE.items()})
    saved_cfg = app.config.get("weight_tiers")
    tmp = root / "tiers"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        for base, alts in app.SLOT_TIERS.items():
            for key in (base, *alts):
                f = tmp / app.MODEL_CATALOGUE[key]["dest"].name
                f.write_bytes(b"")
                app.MODEL_CATALOGUE[key]["dest"] = f

        for base in app.SLOT_TIERS:
            base_name = saved_rows[base]["dest"].name
            app.config.pop("weight_tiers", None)

            # Unstated is every Modal deployment, and must not move.
            app.GPU_VRAM_GB = 0
            if app._slot_name(base) != base_name:
                bad.append(f"{base}: an unstated card no longer resolves to "
                           f"{base_name} — the deployed path has changed")

            # A card that comfortably holds the full row keeps it.
            app.GPU_VRAM_GB = 80.0
            if app._slot_name(base) != base_name:
                bad.append(f"{base}: an 80 GB card resolved to "
                           f"{app._slot_name(base)}, not the full row")

            # A card below the full row's declared fit must come down a tier,
            # and must never be handed a file larger than the one it refused.
            app.GPU_VRAM_GB = 12.0
            got = app._slot_name(base)
            if got == base_name:
                bad.append(f"{base}: a 12 GB card was handed {base_name}, "
                           f"which declares {saved_rows[base].get('fits_vram_gb')} GB")
            elif got not in {saved_rows[k]["dest"].name for k in app.SLOT_TIERS[base]}:
                bad.append(f"{base}: a 12 GB card resolved to {got}, "
                           "which is not one of its tiers")

        # The override is honoured whatever the card, and a stale one falls back.
        slot = next(iter(app.SLOT_TIERS))
        alt = app.SLOT_TIERS[slot][-1]
        app.GPU_VRAM_GB = 80.0
        app.config["weight_tiers"] = {slot: alt}
        if app._slot_name(slot) != saved_rows[alt]["dest"].name:
            bad.append(f"{slot}: the gear override was not honoured on a big card")
        app.config["weight_tiers"] = {slot: "a_key_that_does_not_exist"}
        if app._slot_name(slot) != saved_rows[slot]["dest"].name:
            bad.append(f"{slot}: a stale override did not fall back to the base row")
    finally:
        app.GPU_VRAM_GB = saved_vram
        for k, v in saved_rows.items():
            app.MODEL_CATALOGUE[k]["dest"] = v["dest"]
        if saved_cfg is None:
            app.config.pop("weight_tiers", None)
        else:
            app.config["weight_tiers"] = saved_cfg
    return bad


CHECKS = (
    ("imports offline, in local mode", check_imports_offline, True),
    ("the four names are rebound, the mounts are not", check_rebound, True),
    ("the Dict and Volume surfaces are closed", check_surfaces_closed, False),
    ("dispatch has one door", check_dispatch_single_door, False),
    ("the lane table names things that exist", check_lane_table, False),
    ("web() builds every route", check_web_builds, False),
    ("dispatch, publish, queue, stop", check_job_contract, False),
    ("paths follow the workspace", check_paths_follow_env, True),
    ("tiers resolve by card", check_tier_resolution, True),
)


def main() -> int:
    app, root = _load_local()
    print(f"\napp.py in local mode, workspace {root}\n")
    failed = []
    for label, fn, wants_root in CHECKS:
        try:
            bad = fn(app, root) if wants_root else fn(app)
        except Exception as exc:  # noqa: BLE001 — a check that dies is a failure
            bad = [f"{type(exc).__name__}: {exc}"]
        note = ""
        if fn is check_web_builds and not bad:
            note = f" ({getattr(check_web_builds, 'count', '?')} routes)"
        print(f"  {'ok  ' if not bad else 'FAIL'}  {label}{note}")
        for line in bad:
            print(f"          {line}")
        if bad:
            failed.append(label)
    print()
    if failed:
        print(f"  {len(failed)} of {len(CHECKS)} checks failed.")
        return 1
    print("The seam holds: one app.py, four rebound names, one dispatch door.\n"
          "Graphs are checked against a real ComfyUI by tools/smoke_graphs.py;\n"
          "what a card actually does with the weights needs a rented box.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
