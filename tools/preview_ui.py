"""
The stubbed API the front end is developed against, and a static server for the
built bundle.

    npm --prefix web run build && python3 tools/preview_ui.py [port]

Two ways in, and they want different things:

- **`npm run dev` (:5173)** proxies /api here and serves the sources with HMR.
  That is the edit loop.
- **this server on its own** serves `web/dist` exactly as the web container
  does — absolute /assets paths, hashed filenames, nothing rewriting anything.
  That is the only way to exercise the artifact that actually ships, and it is
  what every check under tools/ui-checks/ is pointed at. A bundle that works
  under the dev server and 404s its own stylesheet when mounted is a failure
  neither the dev server nor a unit test would show.

What makes it worth having at all is the API half: the real prompt compilers
and the real shot vocabulary, pulled out of app.py by AST, answering against
stubbed jobs and files. So the whole front end is workable with no Modal
account, no GPU, no deployment and nothing billed.

Two deliberate choices:

- **app.py is re-read when it changes.** A vocabulary captured at import meant
  editing a pill and reloading showed the old palette, which reads as "my
  change did not work" rather than "the server is stale". Reload is the edit
  loop, so reload has to be honest.
- **Threaded.** A gallery is a grid, so opening it fires a dozen concurrent
  media requests. The single-threaded version dropped most of them and the
  covers came back blank — a layout bug that was not in the layout.

Stdlib only, and never imported by app.py: this must run on a laptop with no
torch, no modal, and no credentials.
"""

import base64
import json
import os
import re
import sys
import threading
import time
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from _from_app import (CAPTION, SHOT, TRAINER,
                       VIDEO, pull)

APP = Path(__file__).resolve().parent.parent / "app.py"
# Argument first, then $PORT, then a default. The env var is what lets a launcher
# hand out a free port instead of this file naming one: two of these cannot share
# 8791, so working on the page from two windows meant the second one refusing to
# start against a port the first had taken.
#
# 8791 and not 8777, because the default has to be the one everything else already
# assumes: web/vite.config.ts proxies /api there, and every check under
# tools/ui-checks/ points at it. It defaulted to 8777 while they did not, so the
# documented loop — this file with no argument, then `npm run dev` — served a page
# whose every request 502'd, and nothing on either side said which half was wrong.
PORT = int(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PORT") or 8791)
DIST = Path(__file__).resolve().parent.parent / "web" / "dist"

# The one thing here that is not a stub. The shot palette is eighty-odd tiles
# built from a table in app.py, and a hand-written copy of that table would be a
# preview of a palette that does not exist — the states worth looking at are
# exactly the ones a copy would get wrong. The compiler comes with it, so
# `/api/compile` shows the real document rather than a plausible one. The
# captioner's presets ride along for the same reason, in the same pull: two AST
# parses to answer one request would be paying twice for the same file.
#
# Re-pulled when app.py changes, for the reason at the top of this file: reload
# is the edit loop, so reload has to be honest, and a vocabulary captured at
# import is exactly the staleness this refuses to have. Keyed on mtime
# rather than re-parsed per request because every gallery thumbnail is a
# request and an AST parse of ten thousand lines is not free.
_APP_CACHE: dict = {}


def app_api() -> dict:
    stamp = APP.stat().st_mtime_ns
    if _APP_CACHE.get("stamp") != stamp:
        _APP_CACHE.update(stamp=stamp,
                          api=pull(SHOT | CAPTION | TRAINER))
    return _APP_CACHE["api"]


# --------------------------------------------------------------------------
# Stub payloads
#
# Shaped to exercise the states worth looking at, not the happy path only: a
# model that is missing, a dataset with uncaptioned images, a caption written
# as tags, a prompt long enough to prove the gallery is right not to show it.
# --------------------------------------------------------------------------

SIZES = [(1024, 1024), (1344, 768), (768, 1344), (1216, 832)]
LONG_PROMPT = (
    "ohwx_style a wide cinematic photograph of a lone figure walking a wet "
    "street at dusk, neon signage reflected in the puddles, shallow depth of "
    "field, 35mm, the light falling from a shopfront on the left, steam rising "
    "from a grate behind them, muted teal and amber grade, film grain"
) * 3

# What a clip that used the shot palette left behind.
#
# The sidecar gains `prompt_typed` and `shot` only when the compiler did
# something, so most cards carry neither and the ones that do are the case the
# metadata sheet's two-prompt branch exists for: H3 compiles to a six-field
# document, which is not a thing anyone recognises their own take by. Without a
# single item carrying these, that branch — and Reuse's whole reason for
# preferring the typed one — had never rendered here.
#
# Compiled by the real compiler rather than pasted, for the same reason
# /api/compile calls it: a document written by hand into this file is a document
# that drifts from the one a run would actually produce, and a preview that
# disagrees with the run is worse than no preview.
TYPED = "k3nan walks out of the shop and stops when he sees the car"
# `group.item`, which is what SHOT_ITEMS is keyed by — a bare "mcu" is rejected
# by name, and that rejection is the reason these are compiled here rather than
# transcribed: a pill key invented in this file would have shipped a document
# no run could reproduce.
SHOT_PILLS = [
    {"key": "framing.mcu"},
    {"key": "camera.pullout"},
    {"key": "light.overcast"},
]


def _h3_document() -> str:
    try:
        api = app_api()
        return api["_compile_h3_prompt"](
            typed=TYPED, pills=api["_validate_shot"](SHOT_PILLS),
            task="t2va", seconds=5, roles=[])
    except Exception:
        # The stub has to serve a gallery even when app.py cannot be parsed —
        # that is the state you are most likely to be in while editing it.
        return TYPED


# The storyboards, seeded with the states a panel is read in: a picture from
# the gallery rows below, a panel with prose and no picture, a camera move
# with its stencil, a subject's arrow with a name on it, a tall picture shown
# whole under a tilt. Pictures are pointers into GALLERY so a deleted row
# degrades the way the volume would — the words stay, the cover 404s. Uploads
# land in BOARD_FILES and are served back by name, which is the whole of what
# the route does.
STORYBOARDS = {
    "the-lookout-a1b2": {
        "title": "The Lookout",
        "aspect": "16:9",
        "panels": [
            {"id": "p1", "prose": "Empty coastal road at dusk, fog rolling over the guardrail.",
             "note": "hold long before she enters", "fit": "crop", "focus": [0.5, 0.5],
             "pills": [{"key": "framing.xwide"}, {"key": "camera.static"}], "motion": [],
             "picture": {"file": "job000_00.png", "gallery": True}},
            {"id": "p2", "prose": "Maya enters frame left, coat pulled tight, walking toward the lookout.",
             "note": "", "fit": "crop", "focus": [0.5, 0.5],
             "pills": [{"key": "framing.wide"},
                       {"key": "camera.panr", "amp": "large", "speed": "slow"}],
             "motion": [{"id": "a1", "pts": [[0.04, 0.62], [0.55, 0.5]], "label": "Maya"}],
             "picture": {"file": "job001_00.png", "gallery": True}},
            {"id": "p3", "prose": "Close on her hands gripping the rail, paint flaking under her fingers.",
             "note": "", "fit": "crop", "focus": [0.5, 0.5],
             "pills": [{"key": "framing.cu"}, {"key": "angle.high"},
                       {"key": "camera.pushin", "amp": "small", "speed": "slow"}],
             "motion": [],
             "picture": {"file": "job002_00.png", "gallery": True}},
            {"id": "p4", "prose": "She says the line — \u201cyou were supposed to wait.\u201d",
             "note": "frame undecided — over-shoulder vs profile", "fit": "crop",
             "focus": [0.5, 0.5], "pills": [], "motion": [], "picture": None},
            {"id": "p5", "prose": "Reverse: the sea, grey and enormous. No music.",
             "note": "", "fit": "whole", "focus": [0.5, 0.15],
             "pills": [{"key": "camera.tiltd", "amp": "large", "speed": "normal"}],
             "motion": [],
             "picture": {"file": "job010_00.png", "gallery": True}},
            {"id": "p6", "prose": "Elias appears beside her; neither looks at the other.",
             "note": "", "fit": "crop", "focus": [0.5, 0.5],
             "pills": [{"key": "framing.medium"}, {"key": "camera.zoom"}],
             "motion": [{"id": "a2", "pts": [[0.96, 0.55], [0.66, 0.52]], "label": "Elias"}],
             "picture": {"file": "job006_00.png", "gallery": True}},
            {"id": "p7", "prose": "The envelope changes hands, barely.",
             "note": "", "fit": "crop", "focus": [0.5, 0.5],
             "pills": [{"key": "camera.orbit", "amp": "medium", "speed": "slow"}],
             "motion": [],
             "picture": {"file": "job004_00.png", "gallery": True}},
            {"id": "p8", "prose": "Both figures tiny against the fog as the light dies.",
             "note": "mirror of panel 1 — same camera height", "fit": "crop",
             "focus": [0.5, 0.5],
             "pills": [{"key": "framing.xwide"},
                       {"key": "camera.craneu", "amp": "large", "speed": "slow"}],
             "motion": [],
             "picture": {"file": "job007_00.png", "gallery": True}},
        ],
        "updated": time.time(),
    },
    "kitchen-morning-c3d4": {
        "title": "Kitchen, morning",
        "aspect": "4:3",
        "panels": [
            {"id": "k1", "prose": "Steam off the kettle, the window fogged.",
             "note": "", "fit": "crop", "focus": [0.5, 0.5],
             "pills": [{"key": "camera.rack"}], "motion": [],
             "picture": {"file": "job009_00.png", "gallery": True}},
        ],
        "updated": time.time() - 86400 * 2,
    },
}
BOARD_FILES: dict[str, dict[str, bytes]] = {}


def _multipart(body: bytes, ctype: str) -> list[tuple[str, bytes]]:
    """The uploaded files out of a multipart body — name and bytes, nothing
    else, which is all the stub needs to hand them back by name."""
    m = re.search(r'boundary="?([^";]+)"?', ctype or "")
    if not m:
        return []
    out = []
    for part in body.split(b"--" + m.group(1).encode()):
        if b"\r\n\r\n" not in part:
            continue
        head, data = part.split(b"\r\n\r\n", 1)
        fn = re.search(rb'filename="([^"]*)"', head)
        if not fn or not fn.group(1):
            continue
        out.append((fn.group(1).decode(errors="replace"), data.rstrip(b"\r\n").rstrip(b"--")))
    return out


GALLERY = [
    {
        "job_id": f"job{i:03d}",
        "kind": "video" if i % 5 == 3 else "image",
        # Flat outputs/: the name carries the run, `{job}_{NN}.png` / `{job}.mp4`.
        "files": [f"job{i:03d}.mp4" if i % 5 == 3 else f"job{i:03d}_00.png"],
        "created": time.time() - i * 3600,
        "modified": time.time() - i * 3600,
        "prompt": _h3_document() if i % 5 == 3 else LONG_PROMPT,
        "negative_prompt": "blurry, low quality, watermark",
        "width": SIZES[i % 4][0],
        "height": SIZES[i % 4][1],
        **(
            {"seconds": 5, "frames": 120, "fps": 24, "seed": 4000 + i,
             "steps": 20, "sampler": "res_multistep", "scheduler": "simple",
             "references": 0, "ref_videos": 0,
             "prompt_typed": TYPED, "shot": SHOT_PILLS}
            if i % 5 == 3 else
            {"model": "turbo", "seeds": [4000 + i], "steps": 8, "cfg_scale": 1.0,
             "shift": 1.15, "sampler": "Euler", "scheduler": "Simple",
             "loras": [{"name": "my_style", "unet": 0.8, "text_encoder": 0.8,
                        "applied": True}],
             # Every fourth image card is a regional render, so `reuse()`'s
             # restore path is actually exercised — with an empty list here it
             # never ran, and the one thing reuse has to get right on a regional
             # card is which LoRA goes back into which rectangle. One box with a
             # LoRA and one with only a photo, because the photo-only box is the
             # one a filter is most likely to drop on the way back in.
             "regions": ([
                 {"box": [0.04, 0.08, 0.42, 0.86], "lora": "portrait.safetensors",
                  "strength": 1.35, "prompt": "a man in a long coat",
                  "ref": False},
                 {"box": [0.54, 0.08, 0.42, 0.86], "lora": "None",
                  "strength": 1.0, "prompt": "a woman", "ref": True},
             ] if i % 4 == 0 else []),
             "region_weight": 1.0}
        ),
    }
    # More than the drawer's 24-card cap, deliberately. The drawer renders a slice
    # and used to hand that same slice to the viewer, so a picture opened from it
    # dead-ended at 24 — a fault no stub shorter than the cap can express.
    for i in range(30)
]

# Two unsaved sets and three saved ones, because the rail has to be looked at
# with both groups in it: one group alone hides whether the headings, the dashed
# card and the spacing between the groups read as a difference or as damage.
DATASETS = [
    {"name": "set_2", "count": 31, "uncaptioned": 31, "cover": "0.png",
     "trigger_word": "", "saved": False},
    # A mixed set, because the two kind filters only appear when a set holds
    # both — so without one here the filter row's five-button state, the
    # two-number count line and the clip tile's own treatment are all
    # unreachable from this server.
    {"name": "roof_takes", "count": 9, "videos": 6, "uncaptioned": 4, "cover": "0.png",
     "trigger_word": "k3nan", "saved": True},
    {"name": "beach_walk", "count": 12, "uncaptioned": 3, "cover": "0.png",
     "trigger_word": "", "saved": False},
    # 24 = 18 numbered + the six files DUPE_FILES puts in the folder. The
    # duplicate review is about this set, so its count has to include them.
    {"name": "studio_portraits", "count": 24, "uncaptioned": 0, "cover": "0.png",
     "trigger_word": "ohwx_style", "saved": True},
    {"name": "street_night", "count": 41, "uncaptioned": 7, "cover": "0.png",
     "trigger_word": "ohwx_night", "saved": True},
    {"name": "product_flatlay", "count": 18, "uncaptioned": 18, "cover": "0.png",
     "trigger_word": "", "saved": True},
    {"name": "empty_set", "count": 0, "uncaptioned": 0, "cover": None,
     "trigger_word": "", "saved": True},
]

STATE = {
    "hf_token_set": True,
    "models": [
        {"key": "turbo", "label": "Krea 2 Turbo", "note": "8 steps, distilled",
         "family": "Krea 2 — images", "repo_id": "krea/Krea-2-Turbo", "present": True, "size_gb": 17.2,
         "approx_gb": 17.2, "gated": True},
        {"key": "raw", "label": "Krea 2 RAW", "note": "28 steps",
         "family": "Krea 2 — images", "repo_id": "krea/Krea-2-Raw", "present": False, "approx_gb": 17.2,
         "gated": True},
        {"key": "vae", "label": "VAE", "note": "", "family": "Krea 2 — images", "repo_id": "krea/Krea-2-VAE",
         "present": True, "size_gb": 0.3, "approx_gb": 0.3, "gated": False},
        {"key": "text_encoder", "label": "Text encoder", "note": "Qwen2.5-VL",
         "family": "Krea 2 — images", "repo_id": "krea/Krea-2-TE", "present": True, "size_gb": 9.1,
         "approx_gb": 9.1, "gated": False},
        # A second family, mostly missing. Krea 2 above is one file short, which
        # is the state where a family's Download-all correctly does not appear —
        # so on its own it left the button, its queue and its Cancel unreachable
        # here. This is the case the button exists for: a stack of files that are
        # only useful together, which is why the group is the unit you decide in.
        {"key": "h3_dit", "label": "MiniMax-H3 DiT", "note": "fl2va transformer",
         "family": "MiniMax-H3 — video", "repo_id": "MiniMaxAI/MiniMax-H3", "present": False,
         "approx_gb": 21.0, "gated": False},
        {"key": "h3_ref_dit", "label": "MiniMax-H3 ref DiT", "note": "ref2va transformer",
         "family": "MiniMax-H3 — video", "repo_id": "MiniMaxAI/MiniMax-H3", "present": False,
         "approx_gb": 21.0, "gated": False},
        {"key": "h3_vae", "label": "MiniMax-H3 VAE", "note": "",
         "family": "MiniMax-H3 — video", "repo_id": "MiniMaxAI/MiniMax-H3", "present": False,
         "approx_gb": 0.5, "gated": False},
        {"key": "h3_te", "label": "Qwen3-VL-4B", "note": "text encoder",
         "family": "MiniMax-H3 — video", "repo_id": "MiniMaxAI/MiniMax-H3", "present": True,
         "size_gb": 6.7, "approx_gb": 6.7, "gated": False},
    ],
    # Four shapes, because `<lora:…>` has to name each of them unambiguously:
    # a trained LoRA with epoch checkpoints, a bare file dropped at the top of
    # loras/ (which is what a Google Drive pull with no folder produces), two
    # matched pairs whose files are BOTH called high/low — the case that proves
    # the picker qualifies a colliding name with its folder instead of silently
    # offering the wrong one — and two files whose
    # names differ only in case, which is what a Drive pull and a training run
    # disagreeing about capitalisation leaves behind. Both of the last two have
    # to stay typeable: resolution is exact-first for exactly this reason.
    #
    # `root`, `bytes` and `catalogue` are what the Settings list reads: the row
    # to delete, what it costs, and whether the catalogue can put it back. The
    # speed pairs carry a `catalogue` and nothing else does, which is the pair of
    # confirm dialogs worth being able to read side by side — one says the delete
    # is a download and the other says it is permanent.
    "loras": [
        {"name": "my_style", "trigger_word": "ohwx_style", "arch": "krea2",
         "root": "/workspace/loras/my_style", "bytes": 613_400_000, "catalogue": "",
         "files": [
            {"path": "/workspace/loras/my_style/my_style.safetensors",
             "name": "my_style"},
            {"path": "/workspace/loras/my_style/my_style-000020.safetensors",
             "name": "my_style-000020"}]},
        # Two of the catalogue's Krea style LoRAs, named as they really land:
        # loose files at the top of loras/, because a folder would make the nine
        # of them one LoRA with nine epochs. They are here rather than invented
        # placeholders so a screenshot taken against this server names weights a
        # reader can actually download.
        {"name": "darkbrush", "trigger_word": "monochrome ink wash style",
         "strength": 1.3, "arch": "krea2",
         "root": "/workspace/loras/darkbrush.safetensors", "bytes": 469_291_992,
         "catalogue": "Krea 2 style LoRAs", "files": [
            {"path": "/workspace/loras/darkbrush.safetensors", "name": "darkbrush.safetensors"}]},
        {"name": "sunsetblur", "trigger_word": "ethereal motion blur style",
         "strength": 1.3, "arch": "krea2",
         "root": "/workspace/loras/sunsetblur.safetensors", "bytes": 469_291_992,
         "catalogue": "Krea 2 style LoRAs", "files": [
            {"path": "/workspace/loras/sunsetblur.safetensors", "name": "sunsetblur.safetensors"}]},
        # The case collision, which is the awkward state this pair exists to
        # hold: two real files whose names differ only in capitalisation, which
        # is what a Drive pull and a training run disagreeing leaves behind.
        # Folding case before comparing made both untypeable, so the resolver
        # must match exactly first — see the note in CLAUDE.md.
        {"name": "Portrait", "trigger_word": "",
         "root": "/workspace/loras/Portrait.safetensors", "bytes": 306_700_000,
         "catalogue": "", "files": [
            {"path": "/workspace/loras/Portrait.safetensors", "name": "Portrait.safetensors"}]},
        {"name": "portrait", "trigger_word": "",
         "root": "/workspace/loras/portrait.safetensors", "bytes": 76_400_000,
         "catalogue": "", "files": [
            {"path": "/workspace/loras/portrait.safetensors", "name": "portrait.safetensors"}]},
        # The catalogue's own loose file. It was missing from this list while
        # `edit_lora` below said it was on the volume, which the real /api/state
        # cannot do — it lands in loras/ and is listed like anything else there.
        # `internal` keeps it out of both pickers; Settings still lists it, and
        # this stub holding one internal row is what proves the pickers skip it.
        {"name": "krea2_identity_edit_v1_2", "trigger_word": "",
         "arch": "krea2", "internal": True,
         "root": "/workspace/loras/krea2_identity_edit_v1_2.safetensors",
         "bytes": 1_790_000_000, "catalogue": "Krea 2 — images", "files": [
            {"path": "/workspace/loras/krea2_identity_edit_v1_2.safetensors",
             "name": "krea2_identity_edit_v1_2.safetensors"}]},
        # Two folders whose files collide on the *same* two names, which is the
        # only arrangement that forces the picker to qualify. A matched pair of
        # speed LoRAs is where this arrives from in practice; the shape is what
        # matters, not the family it came from.
        {"name": "speed-a", "trigger_word": "",
         "root": "/workspace/loras/speed-a", "bytes": 1_060_000_000,
         "catalogue": "", "files": [
            {"path": "/workspace/loras/speed-a/high.safetensors", "name": "high"},
            {"path": "/workspace/loras/speed-a/low.safetensors", "name": "low"}]},
        {"name": "speed-b", "trigger_word": "",
         "root": "/workspace/loras/speed-b", "bytes": 1_060_000_000,
         "catalogue": "", "files": [
            {"path": "/workspace/loras/speed-b/high.safetensors", "name": "high"},
            {"path": "/workspace/loras/speed-b/low.safetensors", "name": "low"}]},
        # The catalogue's H3 speed folder, arch-tagged. The image picker must
        # not offer it and the video picker must — this row is the one that
        # proves the split, which no untagged row above can.
        {"name": "h3-speed", "trigger_word": "", "arch": "h3",
         "root": "/workspace/loras/h3-speed", "bytes": 2_000_000_000,
         "catalogue": "MiniMax-H3 speed LoRAs", "files": [
            {"path": "/workspace/loras/h3-speed/fl2v-8step.safetensors",
             "name": "fl2v-8step"}]},
    ],
    # One family, because there is one. This held three while there were two
    # more, on the rule that a stub with only one shape cannot catch a control
    # that fails to appear — still true, and the way to honour it now is a
    # *task* that is not downloaded rather than a family that does not exist.
    # `supports` is filled from app.py below rather than written here — see the
    # VIDEO set in `_from_app.py`. What stays hand-written is what the stub is
    # *for*: which tasks are ready, and which weights are pretend-downloaded.
    "video_models": [
        {"key": "h3", "label": "MiniMax-H3",
         "note": "Sound and picture in one pass",
         "tiers": {"full": "768p", "draft": "544p draft"},
         "samplers": ["res_multistep", "euler", "dpmpp_2m"],
         "schedulers": ["simple", "normal", "beta"],
         "defaults": {"steps": 20, "sampler": "res_multistep",
                      "scheduler": "simple", "tier": "full", "seconds": 5},
         # ref2va deliberately not downloaded: attaching a reference here should
         # disable Generate and name the missing transformer, which is the state
         # most worth being able to look at and is unreachable from a stub where
         # everything is present.
         "tasks": {"fl2va": {"ready": True, "missing": []},
                   "ref2va": {"ready": False,
                              "missing": ["MiniMax-H3 ref DiT"]}},
         "ready": True},
    ],
    # shot_vocab / shot_langs / shot_roles are added per request — see app_api().
    # `max_ref_audios` was missing here while app.py served it, which is this
    # file's one named failure mode — a stub that omits a field is a preview of a
    # control that does not exist, and it fails silently. The reference-budget
    # note reads all three.
    "max_loras": 6, "max_refs": 9, "max_ref_videos": 3, "max_ref_audios": 3,
    "max_regions": 8,
    # ComfyUI's spellings, which is what the image side sends into a graph now.
    # The old Forge labels ("Euler a", "Automatic") were not a different way of
    # writing these — they are values KSampler rejects.
    "samplers": ["euler", "res_multistep", "er_sde", "dpmpp_2m", "heun"],
    "schedulers": ["simple", "normal", "beta", "sgm_uniform", "karras"],
    # Deliberately not first in either list, which they are in app.py: the page
    # has to *select* these rather than fall through to whatever sits at the
    # top, and a stub that put them there could not tell the two apart.
    "image_defaults": {"sampler": "er_sde", "scheduler": "sgm_uniform"},
    "krea2_defaults": {"turbo": {"steps": 8, "cfg": 1.0},
                       "raw": {"steps": 28, "cfg": 5.5}},
    # True, because the awkward state here is the one with the controls
    # visible: two plate tiles next to a region stack is the fullest the bar
    # ever gets, and the cap that keeps it from pushing the canvas out of frame
    # is only worth checking against that. Flip it to see the tiles absent.
    "edit_lora": True,
    "gpus": {"image": {"options": ["H100", "H200"], "default": "H100"},
             "video": {"options": ["H100", "H200"], "default": "H100"}},
}

# The capabilities come from the deployment rather than being remembered. This
# ran as a hand-written table until H3 grew LoRAs and the preview kept rendering
# the composer without the button — which is the whole argument for pulling them:
# a stub that omits a menu is a preview of a control that does not exist.
_PULLED_VIDEO = pull(VIDEO)
_VIDEO = _PULLED_VIDEO["VIDEO_MODELS"]
# How many frames a still comes back as, so the stubbed strip is the arity that
# actually ships rather than a number chosen here.
H3_STILL_FRAMES = _PULLED_VIDEO["H3_STILL_FRAMES"]
for _m in STATE["video_models"]:
    _m["supports"] = dict(_VIDEO[_m["key"]]["supports"])
    # `lengths` for the same reason, found the same way. It was hand-written
    # here, and app.py grew a `0` — the length that means a still — which the
    # stub did not have, so the preview rendered a duration menu the deployment
    # does not serve and `Still` fell through to 5s. A vocabulary is the
    # deployment's, not this file's; what stays hand-written is which tasks are
    # ready and which weights are pretend-downloaded.
    _m["lengths"] = list(_VIDEO[_m["key"]]["lengths"])


# Which Drive outcome the next poll reports. Mutable module state, like
# DATASETS above and for the same reason: this flow is judged by what the card
# does over several polls, and a fixed reply cannot show a transfer moving.
GDRIVE = {"mode": "ok", "folder": "", "polls": 0}
_COLD = {"n": 0}
# What the config Dict holds in production: presets saved from the captioner
# row and captioners added under the gear. In memory so the preview can
# exercise add, appear-in-menu and delete without a server.
CUSTOM_PRESETS: dict = {}
CUSTOM_VLMS: dict = {}
# Built-ins hidden by their ✕ — the production route parks these in the config
# Dict; in memory here for the same reason the customs are.
HIDDEN_VLMS: set = set()

CAPTIONS = [
    "ohwx_style a photograph of a person seated by a window in soft daylight.",
    "ohwx_style a close portrait of a person against a plain grey backdrop.",
    "portrait, studio, grey backdrop, soft light, 85mm",          # tag-style
    "",                                                            # uncaptioned
    "a photograph of a person standing in a field.",               # no trigger
]


# A real training set is not square, and a stub that pretends otherwise cannot
# show an aspect-ratio bug — every dataset image here used to be 1024x1024, which
# is exactly why a full-screen viewer that squared its images went unnoticed. The
# extremes are in on purpose: a panorama and a tall crop are what break a viewer
# that constrains only one axis.
SHAPES = [(1024, 1024), (832, 1216), (1216, 832), (1536, 640), (640, 1536),
          (1200, 900), (900, 1200), (2048, 1024)]


def shape_of(name: str) -> tuple:
    """Dimensions for one dataset image, stable across the listing and the bytes.
    Keyed off the filename so /api/image and the JSON never disagree — a stub
    that reports 832x1216 and then serves a square is its own bug."""
    stem = "".join(c for c in name if c.isdigit()) or "0"
    return SHAPES[int(stem) % len(SHAPES)]


# The files the duplicate fixture is about, and they are *in* the listing.
#
# They were not, for a while: the report named `3 copy.png` and `8.jpg` while
# `images()` emitted `0.png … 23.png`, so deleting from the review removed four
# files the set had never contained. Everything still answered 200 and the image
# count never moved — which is precisely the shape of a broken batch delete, and
# the stub was generating it on its own. Same rule as `shape_of`: a stub that
# reports one thing and serves another is its own bug.
DUPE_FILES = [
    ("3 copy.png", 1536, 640, 1_820_000),
    ("8.jpg", 768, 320, 184_000),
    ("13.webp", 1536, 640, 1_210_000),
    ("17.jpg", 1216, 832, 612_000),
    ("20.jpg", 1216, 832, 902_000),
    ("21.jpg", 1638, 819, 1_020_000),
]


def images(n: int = 24, videos: int = 0) -> list:
    """`n` numbered images, then any clips, then the duplicate fixture's own
    files, so the two surfaces are talking about one folder."""
    out = []
    for i in range(n):
        w, h = shape_of(str(i))
        out.append({"name": f"{i}.png", "kind": "image",
                    "caption": CAPTIONS[i % len(CAPTIONS)],
                    "bytes": 780_000 + i * 4_100, "width": w, "height": h})
    # No width or height, exactly as the route answers for one: web_image has no
    # ffmpeg, so nothing on the server can measure a clip, and the tile that
    # printed "1024×1024" under one would be printing a number nobody produced.
    for i in range(videos):
        out.append({"name": f"clip_{i}.mp4", "kind": "video",
                    "caption": CAPTIONS[i % len(CAPTIONS)] if i % 2 else "",
                    "bytes": 18_400_000 + i * 900_000})
    for name, w, h, b in DUPE_FILES:
        out.append({"name": name, "caption": "", "bytes": b, "width": w, "height": h})
    return out


def duplicate_report(gone: set | None = None) -> dict:
    """
    The duplicate-review fixture, shaped like the real scan.

    The preview does not pull the scanner itself: that needs Pillow and a real
    folder of files, while this server deliberately remains stdlib-only. It
    does keep the *response* faithful, so the group-by-group surface is
    developed against pictures with genuinely different dimensions, weights and
    encodings rather than against a hand-drawn empty state.

    Both classes are here, because the whole safety model is the difference
    between them: a `duplicate` group carries a `suggest` and the page marks
    everything else, a `similar` group carries none and the page marks nothing.
    A fixture with only the first kind would let the second ship untested — and
    the second is the one that must never preselect a deletion.
    """
    def row(name: str, width: int, height: int, bytes_: int, fmt: str, *,
            dhash: int, phash: int, same_file: bool = False, caption: str = "",
            sharpness: float = 12.4, transforms: list | None = None,
            crop: tuple | None = None) -> dict:
        return {"name": name, "caption": caption, "width": width, "height": height,
                "bytes": bytes_, "format": fmt, "megapixels": round(width * height / 1e6, 1),
                "mtime": 0, "sharpness": sharpness,
                "dhash_distance": dhash, "phash_distance": phash,
                "same_file": same_file,
                "transforms": [] if transforms is None else transforms,
                "crop_dhash": crop[0] if crop else None,
                "crop_phash": crop[1] if crop else None}

    groups = [
        # One picture, three encodings, three sizes — and a byte-identical copy,
        # because "the same file" is the row where there is nothing to weigh.
        {"key": "3.png", "kind": "duplicate", "suggest": "3.png",
         "why": "most pixels · 1.0 MP", "images": [
             row("3.png", 1536, 640, 1_820_000, "PNG", dhash=0, phash=0, sharpness=14.1,
                 caption="ohwx_style a close portrait in soft daylight."),
             row("3 copy.png", 1536, 640, 1_820_000, "PNG", dhash=0, phash=0, same_file=True,
                 sharpness=14.1, transforms=["byte-for-byte identical"]),
             row("8.jpg", 768, 320, 184_000, "JPEG", dhash=1, phash=4, sharpness=9.8,
                 transforms=["resized", "reformatted"]),
             row("13.webp", 1536, 640, 1_210_000, "WEBP", dhash=0, phash=2, sharpness=13.6,
                 transforms=["reformatted"]),
         ]},
        # The tie the reason line exists for: same picture, same size, one of
        # them re-exported brighter.
        {"key": "2.png", "kind": "duplicate", "suggest": "2.png",
         "why": "PNG over JPEG", "images": [
             row("2.png", 1216, 832, 1_140_000, "PNG", dhash=0, phash=0, sharpness=11.2,
                 caption="ohwx_style standing in a field, soft daylight."),
             row("17.jpg", 1216, 832, 612_000, "JPEG", dhash=3, phash=12, sharpness=10.9,
                 transforms=["reformatted"]),
         ]},
        # Two frames off one burst. Nothing preselected, and the empty `suggest`
        # is what the page reads to know that.
        {"key": "4.png", "kind": "similar", "suggest": "", "why": "", "images": [
             row("4.png", 1216, 832, 940_000, "PNG", dhash=0, phash=0, sharpness=12.0),
             row("20.jpg", 1216, 832, 902_000, "JPEG", dhash=10, phash=14, sharpness=11.7,
                 transforms=["reformatted"]),
         ]},
        # A crop, which is always similar and never a duplicate — a deliberate
        # reframe of a training image is a variation somebody made on purpose.
        {"key": "6.png", "kind": "similar", "suggest": "", "why": "", "images": [
             row("6.png", 2048, 1024, 2_310_000, "PNG", dhash=0, phash=0, sharpness=15.3),
             row("21.jpg", 1638, 819, 1_020_000, "JPEG", dhash=18, phash=20, sharpness=15.1,
                 transforms=["resized", "reformatted", "cropped"], crop=(2, 4)),
         ]},
    ]
    # A deleted image leaves its group, and a group with one image left is not
    # a group any more. This is the half that makes the delete visible: without
    # it the panel deletes four files and redraws the same four cards.
    gone = gone or set()
    for g in groups:
        g["images"] = [i for i in g["images"] if i["name"] not in gone]
    groups = [g for g in groups if len(g["images"]) > 1]
    for g in groups:
        if g["kind"] == "duplicate" and g["suggest"] not in {i["name"] for i in g["images"]}:
            g["suggest"] = g["images"][0]["name"]

    dupes = [g for g in groups if g["kind"] == "duplicate"]
    return {
        "images": 24,
        "groups": groups,
        "thresholds": {"duplicate": {"dhash": 6, "phash": 16},
                       "similar": {"dhash": 12, "phash": 18},
                       "crop": {"dhash": 6, "phash": 16}},
        "summary": {
            "duplicate_groups": len(dupes),
            "duplicate_images": sum(len(g["images"]) for g in dupes),
            "similar_groups": len(groups) - len(dupes),
            "similar_images": sum(len(g["images"]) for g in groups
                                  if g["kind"] != "duplicate"),
        },
        "reclaim": sum(i["bytes"] for g in dupes for i in g["images"]
                       if i["name"] != g["suggest"]),
    }


INSIGHT = {
    "images": 24, "captioned": 19, "with_trigger": 14,
    "trigger_word": "ohwx_style", "median_words": 14, "thin": ["7.png", "12.png"],
    "duplicates": [{"images": ["3.png", "8.png", "13.png"]}],
    "tag_style": ["2.png", "17.png"],
    "phrases": [
        {"phrase": "soft daylight", "count": 11, "share": 0.61},
        {"phrase": "grey backdrop", "count": 7, "share": 0.37},
        {"phrase": "seated by a window", "count": 4, "share": 0.21},
    ],
}


# How far the fake scan has got, so the progress readout is reachable at all.
SCAN: dict = {}

# What the duplicate review has deleted, per dataset. In memory and mutated,
# for the reason the save and cull routes are: the delete is judged by what
# happens *next* — cards leaving the group, the group leaving the rail when it
# drops to one image, the count and the reclaim figure going down with them.
# A flat {"ok": true} leaves the page redrawing the folder it just emptied,
# which is the one outcome that would hide a broken batch delete.
REMOVED: dict = {}


# In-flight fake runs, keyed by job id: {polls, stopped}. Module state rather
# than per-request, because a run only reads as a run if consecutive polls
# disagree with each other.
RUNS: dict = {}
# The stub's arsenal. In memory, because the preview's whole job is driving the
# save-then-recall loop without a volume — a page reload keeps the server alive,
# so a "fresh session" still finds what the last one saved.
CHARS: dict = {}
# The export's own counter, so a stitch is watchable rather than instant.
EXPORT = {"polls": 0, "takes": 0}

# The Playground's shelf, in memory like CHARS and for the same reason: the
# room's whole loop is seed → rewire → save → load → run, and it has to be
# drivable without a volume.
WORKFLOWS_STUB: dict = {}
PACKS_STUB: list = []

# The stub's sketch of the plain Krea 2 graph — the *shape* the room opens on,
# not the builder's output (that lives in app.py and answers on a deployment).
# Kept to the node types the stub catalogue below describes, so every card the
# seed draws has a spec behind it.
PLAYGROUND_SEED = {
    "dit": {"class_type": "UNETLoader",
            "inputs": {"unet_name": "krea2_turbo.safetensors",
                       "weight_dtype": "default"}},
    "clip": {"class_type": "CLIPLoader",
             "inputs": {"clip_name": "qwen3vl_4b.safetensors",
                        "type": "krea2", "device": "default"}},
    "vae": {"class_type": "VAELoader",
            "inputs": {"vae_name": "krea2_vae.safetensors"}},
    "pos": {"class_type": "CLIPTextEncode",
            "inputs": {"text": "", "clip": ["clip", 0]}},
    "neg": {"class_type": "CLIPTextEncode",
            "inputs": {"text": "", "clip": ["clip", 0]}},
    "latent": {"class_type": "EmptySD3LatentImage",
               "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    "sample": {"class_type": "KSampler",
               "inputs": {"model": ["dit", 0], "positive": ["pos", 0],
                          "negative": ["neg", 0], "latent_image": ["latent", 0],
                          "seed": 4242, "steps": 8, "cfg": 1.0,
                          "sampler_name": "er_sde",
                          "scheduler": "sgm_uniform", "denoise": 1.0}},
    "decode": {"class_type": "VAEDecode",
               "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
    "save": {"class_type": "SaveImage",
             "inputs": {"images": ["decode", 0],
                        "filename_prefix": "visionary"}},
}

# Enough of /object_info for every node the seed uses, plus LoadImage for the
# file-widget path. Shapes copied from ComfyUI's own answers — [type, config]
# pairs, enums as arrays, image_upload marking the picture widget.
NODE_CATALOGUE_STUB = {
    "UNETLoader": {
        "input": {"required": {
            "unet_name": [["krea2_turbo.safetensors", "krea2_raw.safetensors"], {}],
            "weight_dtype": [["default", "fp8_e4m3fn"], {}]}},
        "output": ["MODEL"], "output_name": ["MODEL"],
        "display_name": "Load Diffusion Model", "category": "advanced/loaders"},
    "CLIPLoader": {
        "input": {"required": {
            "clip_name": [["qwen3vl_4b.safetensors"], {}],
            "type": [["krea2", "stable_diffusion"], {}]},
            "optional": {"device": [["default", "cpu"], {}]}},
        "output": ["CLIP"], "output_name": ["CLIP"],
        "display_name": "Load CLIP", "category": "advanced/loaders"},
    "VAELoader": {
        "input": {"required": {"vae_name": [["krea2_vae.safetensors"], {}]}},
        "output": ["VAE"], "output_name": ["VAE"],
        "display_name": "Load VAE", "category": "loaders"},
    "CLIPTextEncode": {
        "input": {"required": {
            "text": ["STRING", {"multiline": True, "default": ""}],
            "clip": ["CLIP", {}]}},
        "output": ["CONDITIONING"], "output_name": ["CONDITIONING"],
        "display_name": "CLIP Text Encode (Prompt)", "category": "conditioning"},
    "EmptySD3LatentImage": {
        "input": {"required": {
            "width": ["INT", {"default": 1024, "min": 16, "max": 16384}],
            "height": ["INT", {"default": 1024, "min": 16, "max": 16384}],
            "batch_size": ["INT", {"default": 1, "min": 1, "max": 4096}]}},
        "output": ["LATENT"], "output_name": ["LATENT"],
        "display_name": "Empty Latent Image (SD3)", "category": "latent/sd3"},
    "KSampler": {
        "input": {"required": {
            "model": ["MODEL", {}],
            "seed": ["INT", {"default": 0, "min": 0}],
            "steps": ["INT", {"default": 20, "min": 1, "max": 10000}],
            "cfg": ["FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}],
            "sampler_name": [["er_sde", "euler", "euler_ancestral",
                              "res_multistep"], {}],
            "scheduler": [["sgm_uniform", "simple", "normal"], {}],
            "positive": ["CONDITIONING", {}],
            "negative": ["CONDITIONING", {}],
            "latent_image": ["LATENT", {}],
            "denoise": ["FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}]}},
        "output": ["LATENT"], "output_name": ["LATENT"],
        "display_name": "KSampler", "category": "sampling"},
    "VAEDecode": {
        "input": {"required": {"samples": ["LATENT", {}], "vae": ["VAE", {}]}},
        "output": ["IMAGE"], "output_name": ["IMAGE"],
        "display_name": "VAE Decode", "category": "latent"},
    "SaveImage": {
        "input": {"required": {
            "images": ["IMAGE", {}],
            "filename_prefix": ["STRING", {"default": "ComfyUI"}]}},
        "output": [], "output_name": [], "output_node": True,
        "display_name": "Save Image", "category": "image"},
    "LoadImage": {
        "input": {"required": {
            "image": [["example.png"], {"image_upload": True}]}},
        "output": ["IMAGE", "MASK"], "output_name": ["IMAGE", "MASK"],
        "display_name": "Load Image", "category": "image"},
}


def train_status(job_id: str, name: str = "probe_lora") -> dict:
    """
    One training run's live fields, advanced by a poll.

    Called from `/api/status/{job}` and from the sessions listing, and written
    once for both: the board polls the listing rather than a status per card,
    so a second copy of this here would be a board that disagrees with the
    status route about the same run.
    """
    job = RUNS.setdefault(job_id, {"polls": 0, "stopped": False})
    job["polls"] += 1
    epochs, per_epoch = 4, 3
    total = epochs * per_epoch
    n = job["polls"]
    out = "/workspace/loras/" + name
    if job["stopped"]:
        # A stop is cooperative: the trainer finishes the step it is on and
        # unwinds, so for a moment the run is still running *and* stopping. Held
        # for one poll here rather than flipping straight to stopped, because
        # that moment is a state the card has words for and this is the only
        # server it can be developed against.
        job.setdefault("stopped_at", n)
        if n - job["stopped_at"] < 1:
            return {"status": "running", "phase": "training", "step": n,
                    "total_steps": total, "percent": int(min(n, total) * 100 / total),
                    "epoch": min(epochs, n // per_epoch + 1), "total_epochs": epochs}
        # Checkpoints already written survive a stop, which is what the dialog
        # promises by name.
        done = min(epochs, n // per_epoch)
        return {
            "status": "stopped", "percent": int(min(n, total) * 100 / total),
            "note": "Stopped after %d epoch%s." % (done, "" if done == 1 else "s"),
            "output_dir": out,
            "files": ["%s-%06d.safetensors" % (name, i + 1) for i in range(done)],
            "duration_s": 60 * n,
        }
    if n < total:
        return {
            "status": "running", "phase": "training",
            "step": n, "total_steps": total,
            "epoch": min(epochs, n // per_epoch + 1), "total_epochs": epochs,
            "rate": "1.8it/s", "eta": "%d:%02d" % ((total - n) // 2, 0),
            "elapsed": "%d:%02d" % (n // 2, (n * 30) % 60),
            "loss": round(0.182 - 0.011 * n, 4),
            "percent": int(n * 100 / total),
        }
    return {
        "status": "completed", "percent": 100,
        "note": "4 epochs, 4 LoRA files. Pick one in the LoRA picker.",
        "output_dir": out,
        "files": ["%s-%06d.safetensors" % (name, i + 1) for i in range(epochs)],
        "duration_s": 60 * total, "loss": 0.0709,
    }


# The board, seeded with the three states a card is read in: one training, one
# finished, and one saved half-written on the way to making a set — which is the
# state the "+ New set" path leaves behind and the only one with an instruction
# on it rather than a status.
def _params(**over) -> dict:
    return {**app_api()["TRAIN_DEFAULTS"], **over}


SESSIONS: list = []
# The server is threaded and the seed is lazy, so the first two requests race:
# both find SESSIONS empty, both build the list — paying the cold app.py pull
# inside _params(), which is the window — and both extend, and every card sits
# on the board twice under a duplicate React key. The dev page makes the race
# routine rather than rare: StrictMode mounts effects twice, so the first two
# fetches arrive together.
_SEED_LOCK = threading.Lock()


def seed_sessions() -> None:
    with _SEED_LOCK:
        if SESSIONS:
            return
        SESSIONS.extend([
            {"id": "s1", "lora_name": "k3nan_v3", "trigger_word": "k3nan",
             "dataset": "studio_portraits", "params": _params(network_dim=64, network_alpha=32),
             "job_id": "train900", "created": time.time() - 400, "runs": 1},
            {"id": "s2", "lora_name": "street_look", "trigger_word": "ohwx_night",
             "dataset": "street_night", "params": _params(max_train_epochs=12),
             "job_id": "train901", "created": time.time() - 9000, "runs": 2},
            {"id": "s3", "lora_name": "", "trigger_word": "", "dataset": "",
             "params": _params(), "job_id": "", "created": time.time() - 60, "runs": 0},
        ])
        # The finished one is finished on its first poll rather than four minutes
        # in: a board with nothing terminal on it cannot show Run again, Edit or
        # Delete, which is half of what a card does.
        RUNS["train901"] = {"polls": 99, "stopped": False}


def session_view(rec: dict) -> dict:
    """`_session_view` in app.py, to the extent a stub can be: status is derived
    from the run rather than stored, because a card that remembers being
    `running` is exactly the bug the real one refuses to have."""
    if not rec.get("job_id"):
        return {**rec, "status": "draft"}
    live = train_status(rec["job_id"], rec.get("lora_name") or "probe_lora")
    # Only while it is still unwinding, the same as `_session_view`: the flag is
    # never cleared, so a card reading it alone says "Stopping…" over a run that
    # stopped an hour ago.
    stopping = (bool(RUNS.get(rec["job_id"], {}).get("stopped"))
                and live.get("status") in ("running", "queued"))
    return {**rec, **live, "stopping": stopping}


def swatch(w: int, h: int, label: str, seed: int) -> bytes:
    """An SVG placeholder, so the tool needs no Pillow. Colour varies by seed
    because a grid of identical grey rectangles hides exactly the alignment and
    aspect-ratio bugs this server exists to catch."""
    hue = (seed * 47) % 360
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">'
        f'<rect width="100%" height="100%" fill="hsl({hue} 26% 33%)"/>'
        f'<rect x="8" y="8" width="{w - 16}" height="{h - 16}" fill="none" '
        f'stroke="#ffffff33" stroke-width="3"/>'
        f'<text x="20" y="40" font-family="ui-monospace,monospace" '
        f'font-size="22" fill="#e9e9e9">{escape(label)}</text>'
        f'<text x="20" y="70" font-family="ui-monospace,monospace" '
        f'font-size="16" fill="#ffffff88">{w}×{h}</text>'
        f'</svg>'
    ).encode()


# **A real 1280x720 clip, 2.3 KB, because a swatch has no dimensions.**
#
# Every `.mp4` this server was asked for came back as `swatch()` — an SVG — so
# `<video>` reported `videoWidth: 0` and laid out at the height of its own
# controls. That made a whole class of fault structurally invisible here: a clip
# that does not fit the canvas cannot be seen to not fit when the clip has no
# height. It is `web/CLAUDE.md`'s own named failure mode, arrived at from the
# other direction — a stub that omits a *property* is a preview of a render that
# cannot happen, and it fails silently.
#
# One second of flat grey at CRF 51, inlined rather than generated, so this file
# keeps the property the swatch was protecting: no Pillow, no ffmpeg, no
# dependency at all.
CLIP_720P = base64.b64decode(
    "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAOlbW9vdgAAAGxtdmhkAAAAAAAA"
    "AAAAAAAAAAAD6AAAA+gAAQAAAQAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAA"
    "AAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAtB0cmFrAAAAXHRr"
    "aGQAAAADAAAAAAAAAAAAAAABAAAAAAAAA+gAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAA"
    "AAAAAAABAAAAAAAAAAAAAAAAAABAAAAABQAAAALQAAAAAAAkZWR0cwAAABxlbHN0AAAAAAAA"
    "AAEAAAPoAAAIAAABAAAAAAJIbWRpYQAAACBtZGhkAAAAAAAAAAAAAAAAAAAwAAAAMABVxAAA"
    "AAAALWhkbHIAAAAAAAAAAHZpZGUAAAAAAAAAAAAAAABWaWRlb0hhbmRsZXIAAAAB821pbmYA"
    "AAAUdm1oZAAAAAEAAAAAAAAAAAAAACRkaW5mAAAAHGRyZWYAAAAAAAAAAQAAAAx1cmwgAAAA"
    "AQAAAbNzdGJsAAAAw3N0c2QAAAAAAAAAAQAAALNhdmMxAAAAAAAAAAEAAAAAAAAAAAAAAAAA"
    "AAAABQAC0ABIAAAASAAAAAAAAAABFUxhdmM2Mi4xMS4xMDAgbGlieDI2NAAAAAAAAAAAAAAA"
    "GP//AAAAOWF2Y0MBZAAy/+EAG2dkADKscgRAUAW7ARAAAAMAEAAAAwGA8YMYRgEAB2joQ4GU"
    "siz9+PgAAAAAEHBhc3AAAAABAAAAAQAAABRidHJ0AAAAAAAAKmgAAAAAAAAAGHN0dHMAAAAA"
    "AAAAAQAAAAwAAAQAAAAAFHN0c3MAAAAAAAAAAQAAAAEAAABIY3R0cwAAAAAAAAAHAAAAAQAA"
    "CAAAAAABAAAoAAAAAAEAABAAAAAAAwAAAAAAAAAEAAAEAAAAAAEAAAwAAAAAAQAABAAAAAAc"
    "c3RzYwAAAAAAAAABAAAAAQAAAAwAAAABAAAARHN0c3oAAAAAAAAAAAAAAAwAAAO7AAAAJQAA"
    "ACMAAAAkAAAAJAAAACQAAAAkAAAAJAAAACQAAAAkAAAAKgAAACQAAAAUc3RjbwAAAAAAAAAB"
    "AAAD1QAAAGF1ZHRhAAAAWW1ldGEAAAAAAAAAIWhkbHIAAAAAAAAAAG1kaXJhcHBsAAAAAAAA"
    "AAAAAAAALGlsc3QAAAAkqXRvbwAAABxkYXRhAAAAAQAAAABMYXZmNjIuMy4xMDAAAAAIZnJl"
    "ZQAABVVtZGF0AAACsQYF//+t3EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE2NSByMzIy"
    "MiBiMzU2MDVhIC0gSC4yNjQvTVBFRy00IEFWQyBjb2RlYyAtIENvcHlsZWZ0IDIwMDMtMjAy"
    "NSAtIGh0dHA6Ly93d3cudmlkZW9sYW4ub3JnL3gyNjQuaHRtbCAtIG9wdGlvbnM6IGNhYmFj"
    "PTEgcmVmPTE2IGRlYmxvY2s9MTowOjAgYW5hbHlzZT0weDM6MHgxMzMgbWU9dW1oIHN1Ym1l"
    "PTEwIHBzeT0xIHBzeV9yZD0xLjAwOjAuMDAgbWl4ZWRfcmVmPTEgbWVfcmFuZ2U9MjQgY2hy"
    "b21hX21lPTEgdHJlbGxpcz0yIDh4OGRjdD0xIGNxbT0wIGRlYWR6b25lPTIxLDExIGZhc3Rf"
    "cHNraXA9MSBjaHJvbWFfcXBfb2Zmc2V0PS0yIHRocmVhZHM9MTggbG9va2FoZWFkX3RocmVh"
    "ZHM9MyBzbGljZWRfdGhyZWFkcz0wIG5yPTAgZGVjaW1hdGU9MSBpbnRlcmxhY2VkPTAgYmx1"
    "cmF5X2NvbXBhdD0wIGNvbnN0cmFpbmVkX2ludHJhPTAgYmZyYW1lcz04IGJfcHlyYW1pZD0y"
    "IGJfYWRhcHQ9MiBiX2JpYXM9MCBkaXJlY3Q9MyB3ZWlnaHRiPTEgb3Blbl9nb3A9MCB3ZWln"
    "aHRwPTIga2V5aW50PTI1MCBrZXlpbnRfbWluPTEyIHNjZW5lY3V0PTQwIGludHJhX3JlZnJl"
    "c2g9MCByY19sb29rYWhlYWQ9NjAgcmM9Y3JmIG1idHJlZT0xIGNyZj01MS4wIHFjb21wPTAu"
    "NjAgcXBtaW49MCBxcG1heD02OSBxcHN0ZXA9NCBpcF9yYXRpbz0xLjQwIGFxPTE6MS4wMACA"
    "AAABAmWIgQAHf/D5+BNUYwAYAAADAAADAAADAAAWQAAACKgAAAcgAAAMYAAAFtgAAAMAAAMA"
    "AAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMA"
    "AAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMA"
    "AAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMA"
    "AAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMC+wAAACFBmgkt"
    "iF//AAADAAADAAADAAADAAADAAADAAADAAADAWUAAAAfQZ4QhxDPAAADAAADAAADAAADAAAD"
    "AAADAAADAAAEzQAAACABnhgmiFf/AAADAAADAAADAAADAAADAAADAAADAAAFTAAAACABnhhG"
    "iFf/AAADAAADAAADAAADAAADAAADAAADAAAFTQAAACABnhhmiFf/AAADAAADAAADAAADAAAD"
    "AAADAAADAAAFTQAAACABnhitSFf/AAADAAADAAADAAADAAADAAADAAADAAAFTQAAACABnhjN"
    "SFf/AAADAAADAAADAAADAAADAAADAAADAAAFTQAAACABnhjtSFf/AAADAAADAAADAAADAAAD"
    "AAADAAADAAAFTAAAACABnhkNSFf/AAADAAADAAADAAADAAADAAADAAADAAAFTAAAACZBmhlp"
    "NQIC0TKYEK8AAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMBlQAAACABniFFyFf/AAADAAADAAAD"
    "AAADAAADAAADAAADAAAFTA=="
)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # one line per media file is pure noise
        pass

    def reply(self, body, ctype="application/json", code=200):
        if isinstance(body, (dict, list)):
            body = json.dumps(body)
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        path = self.path.split("?")[0]

        # The board mutates in memory for the same reason the rail does: every
        # one of these is judged by what the card does next — a run that starts,
        # a card that goes, a half-written one that comes back with what you
        # typed still in it. A flat {"ok": true} leaves the page redrawing the
        # state it started in, which is the one state that needed no server.
        seed_sessions()
        m = re.match(r"/api/sessions/([^/]+)/(start|stop|delete)$", path)
        if m:
            sid, verb = m.group(1), m.group(2)
            rec = next((r for r in SESSIONS if r["id"] == sid), None)
            if not rec:
                return self.reply({"error": "That session is gone."})
            if verb == "delete":
                if rec.get("job_id"):
                    RUNS.setdefault(rec["job_id"], {"polls": 0})["stopped"] = True
                SESSIONS.remove(rec)
                return self.reply({"ok": True})
            if verb == "stop":
                RUNS.setdefault(rec["job_id"], {"polls": 0})["stopped"] = True
                return self.reply({"ok": True, "session": session_view(rec)})
            # A fresh job id per start, so Run again on a finished card is a new
            # run rather than the old one's terminal record showing through.
            rec["job_id"] = "train%03d" % (len(RUNS) + 902)
            rec["runs"] = int(rec.get("runs") or 0) + 1
            RUNS[rec["job_id"]] = {"polls": 0, "stopped": False}
            return self.reply({"ok": True, "job_id": rec["job_id"],
                               "session": session_view(rec)})

        if path == "/api/sessions":
            try:
                p = json.loads(body or b"{}")
            except json.JSONDecodeError:
                p = {}
            rec = next((r for r in SESSIONS if r["id"] == p.get("id")), None)
            fields = {"lora_name": str(p.get("lora_name") or ""),
                      "trigger_word": str(p.get("trigger_word") or ""),
                      "dataset": str(p.get("dataset") or ""),
                      "params": _params(**{k: v for k, v in (p.get("params") or {}).items()
                                           if k in app_api()["TRAIN_DEFAULTS"]})}
            if rec:
                rec.update(fields)
            else:
                rec = {"id": "s%d" % (len(SESSIONS) + 4), "job_id": "", "runs": 0,
                       "created": time.time(), **fields}
                SESSIONS.insert(0, rec)
            return self.reply({"ok": True, "session": session_view(rec)})

        # Saving and culling mutate the list in memory, because both are
        # judged by what the rail does next — a set moving out of Unsaved, a
        # card leaving and the heading going with it when it was the last one.
        # A flat {"ok": true} leaves the page redrawing the state it started in.
        m = re.match(r"/api/datasets/([^/]+)/remove$", path)
        if m:
            name = m.group(1)
            try:
                payload = json.loads(body or b"{}")
            except json.JSONDecodeError:
                payload = {}
            # Singular or plural, matching the route: the duplicate review
            # resolves a whole group in one press, and fourteen requests against
            # a network volume is fourteen reloads and fourteen commits.
            asked = payload.get("images")
            if not isinstance(asked, list):
                asked = [payload.get("image")]
            asked = [str(x) for x in asked if x]
            if not asked:
                return self.reply({"error": "No image named."})
            gone = REMOVED.setdefault(name, set())
            fresh = [n for n in asked if n not in gone]
            gone.update(fresh)
            row = next((d for d in DATASETS if d["name"] == name), None)
            if row:
                row["count"] = max(0, row["count"] - len(fresh))
            return self.reply({"ok": True, "removed": fresh,
                               "missing": [n for n in asked if n not in fresh],
                               **(row or {})})

        m = re.match(r"/api/datasets/([^/]+)/(save|delete)$", path)
        if m:
            name, verb = m.group(1), m.group(2)
            row = next((d for d in DATASETS if d["name"] == name), None)
            if not row:
                return self.reply({"error": f"No dataset named {name!r}."})
            if verb == "delete":
                DATASETS.remove(row)
                return self.reply({"ok": True})
            try:
                new = json.loads(body or b"{}").get("name") or name
            except json.JSONDecodeError:
                new = name
            row["name"], row["saved"] = new, True
            return self.reply({"ok": True, **row})

        # Mutates the list in memory for the same reason the dataset routes do:
        # what this is judged by is what the card does next — the row leaving,
        # the header's count and total going down with it, and the empty state
        # appearing once the last one goes. A flat {"ok": true} leaves the page
        # redrawing exactly what it started with.
        if path == "/api/loras/delete":
            try:
                root = str(json.loads(body or b"{}").get("path") or "")
            except json.JSONDecodeError:
                root = ""
            row = next((l for l in STATE["loras"] if l["root"] == root), None)
            if not row:
                # The stale-tab answer, worded as app.py words it: a basename,
                # because a full volume path in an error box is the thing you
                # have to read twice to find the one word that identifies it.
                return self.reply({
                    "error": f"No LoRA named {root.rsplit('/', 1)[-1]!r} on the "
                             "volume — reopen Settings to refresh the list."})
            STATE["loras"].remove(row)
            return self.reply({"ok": True})

        if path == "/api/export-scene":
            try:
                data = json.loads(body or b"{}")
            except json.JSONDecodeError:
                data = {}
            takes = [x for x in (data.get("takes") or [])
                     if isinstance(x, dict) and x.get("file")]
            if not takes:
                return self.reply({"error": "Nothing to export — render a take "
                                            "first."})
            EXPORT["polls"] = 0
            EXPORT["takes"] = len(takes)
            return self.reply({"ok": True, "job_id": "export_scene",
                               "takes": len(takes)})

        m = re.match(r"/api/characters/([^/]+)$", path)
        if m:
            try:
                data = json.loads(body or b"{}")
            except json.JSONDecodeError:
                data = {}
            refs = data.get("refs") or []
            weight = data.get("lora")
            held = bool(isinstance(weight, dict) and weight.get("path"))
            if not refs and not held:
                return self.reply({"error": "A character with no photograph and "
                                            "no LoRA is a name. Attach "
                                            "something first."})
            exts = {"image": "png", "audio": "wav", "video": "mp4"}
            rec = []
            for i2, r in enumerate(refs):
                kind = r.get("kind", "image")
                entry = {"file": "%02d-%s.%s" % (i2, kind, exts.get(kind, "bin")),
                         "kind": kind, "b64": r.get("b64", "")}
                if r.get("note"): entry["note"] = r["note"]
                if r.get("sheet"): entry["sheet"] = True
                rec.append(entry)
            saved = {"note": data.get("note", ""),
                     "retention": data.get("retention", ""),
                     "refs": rec}
            lora = data.get("lora")
            if isinstance(lora, dict) and lora.get("path"):
                saved["lora"] = {"path": str(lora["path"]),
                                 "rel": str(lora.get("rel") or ""),
                                 "strength": float(lora.get("strength") or 1)}
            CHARS[m.group(1)] = saved
            return self.reply({"ok": True, "handle": m.group(1),
                               "files": len(rec)})

        if path == "/api/datasets":
            try:
                new = json.loads(body or b"{}").get("name") or "set_x"
            except json.JSONDecodeError:
                new = "set_x"
            row = {"name": new, "count": 0, "uncaptioned": 0, "cover": None,
                   "trigger_word": "", "saved": False}
            DATASETS.insert(0, row)
            return self.reply({"ok": True, **row})

        # Caption presets and models mutate in memory, like the sessions board:
        # the states worth looking at are a saved preset appearing in the menu
        # and a custom captioner gaining its ✕ in Settings. The add validates
        # the repo shape and treats "text-only" in the id as the config-json
        # refusal, because those are the two errors the real route answers with
        # and neither is reachable by pasting a plausible repo at a stub.
        if path == "/api/caption/presets":
            try:
                p = json.loads(body or b"{}")
            except json.JSONDecodeError:
                p = {}
            label = str(p.get("label") or "").strip()
            if not label:
                return self.reply({"error": "A preset needs a name."})
            key = "preset:" + re.sub(r"[^a-z0-9_-]+", "-", label.lower()).strip("-")
            CUSTOM_PRESETS[key] = {
                "label": label, "instruction": str(p.get("instruction") or ""),
                "note": "Your preset."}
            return self.reply({"ok": True, "key": key})
        if path == "/api/caption/presets/delete":
            try:
                key = str(json.loads(body or b"{}").get("key") or "")
            except json.JSONDecodeError:
                key = ""
            if key not in CUSTOM_PRESETS:
                return self.reply({"error": f"No custom preset {key!r}."})
            del CUSTOM_PRESETS[key]
            return self.reply({"ok": True})
        if path == "/api/caption/models":
            try:
                p = json.loads(body or b"{}")
            except json.JSONDecodeError:
                p = {}
            repo = str(p.get("repo") or "").strip().strip("/")
            if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
                return self.reply({
                    "error": f"{repo or '(empty)'} is not a HuggingFace repo id. "
                             "The shape is owner/name, like Qwen/Qwen3-VL-8B-Instruct."})
            if "text-only" in repo:
                return self.reply({
                    "error": f"{repo} does not look like a vision-language model "
                             "(model_type 'llama', no vision config). A captioner "
                             "has to read images."})
            key = "vlm:" + re.sub(r"[^a-z0-9_-]+", "-", repo.lower()).strip("-")
            CUSTOM_VLMS[key] = {
                "repo": repo, "label": str(p.get("label") or "").strip() or repo.split("/")[-1],
                "note": "Added by you. First run pulls the weights."}
            return self.reply({"ok": True, "key": key})
        if path == "/api/caption/models/delete":
            try:
                key = str(json.loads(body or b"{}").get("key") or "")
            except json.JSONDecodeError:
                key = ""
            if key in CUSTOM_VLMS:
                del CUSTOM_VLMS[key]
                return self.reply({"ok": True})
            # Built-ins hide rather than delete, and the last row is refused —
            # shaped as the route shapes it.
            if key in app_api()["CAPTION_MODELS"]:
                hidden = HIDDEN_VLMS | {key}
                left = [k for k in app_api()["CAPTION_MODELS"]
                        if k not in hidden] + list(CUSTOM_VLMS)
                if not left:
                    return self.reply({
                        "error": "That is the last captioner on the menu, and "
                                 "every captioning run needs one. Add another "
                                 "model before removing it."})
                HIDDEN_VLMS.add(key)
                return self.reply({"ok": True})
            return self.reply({"error": f"No captioner {key!r} — reopen "
                                        "Settings to refresh the list."})

        # Find & replace answers the way the real route does now: the new
        # text rides the reply and the page patches its tiles from it, no
        # refetch. A stub that returned only the count would preview a
        # replace that visibly does nothing — the exact fault the real route
        # just stopped having.
        m = re.match(r"/api/datasets/([^/]+)/replace$", path)
        if m:
            try:
                p = json.loads(body or b"{}")
            except json.JSONDecodeError:
                p = {}
            find = str(p.get("find") or "")
            if not find:
                return self.reply({"error": "Nothing to find."})
            repl = str(p.get("replace") or "")
            asked = p.get("images") if isinstance(p.get("images"), list) else []
            caps = {str(name): f"a caption where {find!r} became {repl!r}"
                    for name in asked}
            return self.reply({"ok": True, "changed": len(caps),
                               "captions": caps})

        # Google Drive. Driven by what is in the url field, because the states
        # worth looking at here are the failures: a link that is not shared and
        # a folder with no weights in it are the two things that actually happen,
        # and neither is reachable by pasting a working link at a stub.
        #   ...error   -> the job fails
        #   ...slow    -> stays running, so the progress line can be watched
        #   anything else -> completes with two files and one skipped
        if path == "/api/gdrive":
            try:
                body_json = json.loads(body or b"{}")
            except json.JSONDecodeError:
                body_json = {}
            url = str(body_json.get("url") or "")
            folder = str(body_json.get("folder") or "")
            if not url:
                return self.reply({"error": "Paste a Google Drive link or file id."})
            # Mirrors NAME_RE in app.py. Worth stubbing rather than skipping:
            # this is the one error the route answers with instead of the job,
            # so it is the only one that appears without a single poll.
            if folder and not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", folder):
                return self.reply({
                    "error": "Folder name must be 1-64 chars of [A-Za-z0-9_-]."})
            GDRIVE["mode"] = ("error" if "error" in url
                              else "slow" if "slow" in url else "ok")
            GDRIVE["folder"] = folder
            GDRIVE["polls"] = 0
            return self.reply({"ok": True, "job_id": "dl_gdrive"})

        # Answered, not stubbed away: the page pings this on load and a 404 in
        # the console during every preview session is noise that trains you to
        # ignore the console. There is no interpreter here and there is nothing
        # to warm, which is exactly what `{"ok": true}` means to the caller —
        # nothing on screen depends on it having happened.
        if path == "/api/warm":
            return self.reply({"ok": True})

        # Not stubbed: this route is pure and cheap, and what it answers is the
        # only question the preview server cannot fake usefully. A hand-written
        # reply here would let the rail look right while compiling to something
        # else, which is the one bug this whole feature exists to prevent.
        if path == "/api/compile":
            try:
                p = json.loads(body or b"{}")
            except json.JSONDecodeError:
                p = {}
            api = app_api()
            try:
                pills = api["_validate_shot"](p.get("shot"))
                n_refs = max(0, min(9, int(p.get("references") or 0)))
                n_vids = max(0, min(3, int(p.get("ref_videos") or 0)))
                n_auds = max(0, min(3, int(p.get("ref_audios") or 0)))
                roles = api["_validate_ref_roles"](p.get("ref_roles"), n_refs)
            except (TypeError, ValueError) as exc:
                return self.reply({"error": str(exc)})
            typed = str(p.get("prompt") or "").strip()
            if p.get("kind") == "image":
                return self.reply(
                    {"prompt": api["_compile_image_prompt"](typed, pills)})
            try:
                seconds = float(p.get("seconds") or 5)
            except (TypeError, ValueError):
                seconds = 5.0
            # The scene, validated here for the reason the real route does it
            # here: this is what the composer polls, so a handle nobody defined
            # is a sentence under the rows the moment it is typed rather than a
            # refusal discovered at Generate. A stub that skipped it would be a
            # preview of a composer with no error messages.
            try:
                scene = api["_validate_scene"](p.get("scene"), n_refs=n_refs,
                                               n_vids=n_vids, n_auds=n_auds,
                                               seconds=seconds)
            except (TypeError, ValueError) as exc:
                return self.reply({"error": str(exc)})
            try:
                return self.reply({"prompt": api["_compile_h3_prompt"](
                    typed=typed, pills=pills, seconds=seconds, roles=roles,
                    scene=scene,
                    task=api["_h3_task"](p.get("first_frame"),
                                         p.get("last_frame"),
                                         n_refs, n_vids, n_auds),
                )})
            except ValueError as exc:
                # `_compile_h3_prompt` raises on an over-length document, which
                # is the one refusal that arrives from the compiler rather than
                # the validator — and it is the one somebody hits by writing.
                return self.reply({"error": str(exc)})

        # `/api/train` is the one-call form: make the card, then start it. The
        # page does not use it — the board posts a session and starts that — but
        # it is the route a script has, and a stub that answered `{"ok": true}`
        # with no job id is what taught the last front end to poll
        # /api/status/undefined and paint Done for a run nobody queued.
        if path == "/api/train":
            seed_sessions()
            try:
                p = json.loads(body or b"{}")
            except json.JSONDecodeError:
                p = {}
            job = "train%03d" % (len(RUNS) + 902)
            RUNS[job] = {"polls": 0, "stopped": False}
            rec = {"id": "s%d" % (len(SESSIONS) + 4), "job_id": job, "runs": 1,
                   "created": time.time(),
                   "lora_name": str(p.get("lora_name") or "probe_lora"),
                   "trigger_word": str(p.get("trigger_word") or ""),
                   "dataset": str(p.get("dataset") or ""), "params": _params()}
            SESSIONS.insert(0, rec)
            return self.reply({"ok": True, "job_id": job, "session": session_view(rec)})

        # `num_images` is echoed rather than fixed at two, because the batch is the
        # case the canvas is hardest to get right: one canvas at a time with `‹ 1 / 4 ›`
        # is only exercisable if the stub can actually return four. A fixed pair made
        # the strip look correct in the one arity that needs no navigation.
        if path == "/api/generate":
            try:
                n = int(json.loads(body or b"{}").get("num_images") or 1)
            except (json.JSONDecodeError, TypeError, ValueError):
                n = 1
            RUNS["gen000"] = {"polls": 0, "stopped": False, "n": max(1, min(8, n))}
            return self.reply({"ok": True, "job_id": "gen000"})

        # A clip, with the same shape the image side has. Worth stubbing rather than
        # falling through to the no-job-id reply: without it `/api/video` answered
        # `{"ok": true}`, the page polled `/api/status/undefined`, and the catch-all
        # said "completed" — so a take appeared instantly and every state a video run
        # is actually in, including the minutes of weight loading it opens with, was
        # unreachable here. That is the same fault the `/api/train` note below records.
        if path == "/api/video":
            # A new id per take, because the thing worth watching here is the *swap*:
            # the previous clip stays up until its replacement lands, and with one
            # reused id "replaced" and "never changed" are the same two frames.
            job = "vid%03d" % sum(1 for k in RUNS if k.startswith("vid"))
            # **Zero seconds is the other shape a video run lands in**, and the
            # stub has to be able to return it for the same reason `/api/generate`
            # echoes `num_images`: a canvas that only ever gets a clip makes the
            # strip look correct in the one case that needs no navigation. Read
            # off `seconds` exactly the way `/api/video` derives it, so the two
            # cannot drift into disagreeing about what a still is.
            # `not in (None, "")` rather than `or 5`, which is app.py's `num()`
            # helper spelled out and exists for exactly this value: zero is the
            # one `seconds` that means something and the one that is falsy, so
            # `or 5` turns every still back into a five-second clip.
            try:
                _s = json.loads(body or b"{}").get("seconds")
                secs = float(_s) if _s not in (None, "") else 5.0
            except (json.JSONDecodeError, TypeError, ValueError):
                secs = 5.0
            RUNS[job] = {"polls": 0, "stopped": False, "still": secs == 0}
            return self.reply({"ok": True, "job_id": job})

        # A family's queue. Worth stubbing rather than falling through to the
        # no-job-id reply, because the state this button spends all its time in
        # is the one the reply cannot produce: several files deep, one of them
        # named, a rate moving and Cancel live. The job id is derived the way
        # the route derives it so the page is exercised against an id it did not
        # invent.
        if path == "/api/download-missing":
            try:
                fam = str(json.loads(body or b"{}").get("family") or "")
            except json.JSONDecodeError:
                fam = ""
            job = "dl_fam_" + (re.sub(r"[^a-z0-9]+", "-", fam.lower()).strip("-")[:48] or "x")
            RUNS[job] = {"polls": 0, "stopped": False}
            return self.reply({"ok": True, "started": True, "job_id": job,
                               "mine": True, "missing": ["a", "b", "c"]})

        # ── Playground ─────────────────────────────────────────────────────
        m = re.match(r"/api/storyboard/([a-z0-9][a-z0-9_-]*)(?:/(delete|upload))?$", path)
        if m:
            name, verb = m.group(1), m.group(2)
            if verb == "delete":
                STORYBOARDS.pop(name, None)
                BOARD_FILES.pop(name, None)
                return self.reply({"ok": True})
            if verb == "upload":
                files = []
                stamp = "%x" % int(time.time() * 1000)
                for n, (fn, data) in enumerate(_multipart(body, self.headers.get("Content-Type", ""))):
                    ext = Path(fn).suffix.lower() or ".png"
                    fname = f"{stamp}{n:02d}{ext}"
                    BOARD_FILES.setdefault(name, {})[fname] = data
                    files.append({"file": fname, "width": 1024, "height": 1024})
                return self.reply({"files": files})
            # Whole document in, whole document out, like the route: the page
            # owns the order, so the stub holds exactly what it was last given.
            board = (json.loads(body or b"{}") or {}).get("board") or {}
            updated = time.time()
            STORYBOARDS[name] = {"title": str(board.get("title") or ""),
                                 "aspect": str(board.get("aspect") or "16:9"),
                                 "panels": list(board.get("panels") or []),
                                 "updated": updated}
            return self.reply({"ok": True, "updated": updated})

        if path == "/api/playground/seed":
            try:
                req = json.loads(body or b"{}")
            except json.JSONDecodeError:
                req = {}
            graph = json.loads(json.dumps(PLAYGROUND_SEED))
            graph["pos"]["inputs"]["text"] = str(req.get("prompt") or "")
            dropped = [k for k in ("references", "first_frame", "last_frame",
                                   "regions", "style_refs")
                       if req.get(k)]
            return self.reply({"ok": True, "kind": "image", "graph": graph,
                               **({"dropped": dropped} if dropped else {})})

        if path == "/api/playground/run":
            try:
                graph = json.loads(body or b"{}").get("graph") or {}
            except json.JSONDecodeError:
                graph = {}
            job = "pg%03d" % sum(1 for k in RUNS if k.startswith("pg"))
            # A node keyed "boom" fails the run naming itself, so the room's
            # error path — the lit-up card, the prose — is drivable here.
            RUNS[job] = {"polls": 0, "stopped": False,
                         "boom": "boom" in graph}
            return self.reply({"ok": True, "job_id": job})

        if path == "/api/playground/restart":
            job = "pgrst%03d" % sum(1 for k in RUNS if k.startswith("pgrst"))
            RUNS[job] = {"polls": 0, "stopped": False}
            return self.reply({"ok": True, "job_id": job})

        if path == "/api/playground/refresh":
            job = "pgcat%03d" % sum(1 for k in RUNS if k.startswith("pgcat"))
            RUNS[job] = {"polls": 0, "stopped": False}
            return self.reply({"ok": True, "job_id": job})

        if path == "/api/playground/packs":
            try:
                url = str(json.loads(body or b"{}").get("url") or "")
            except json.JSONDecodeError:
                url = ""
            name = re.sub(r"\.git$", "", url.rstrip("/").rsplit("/", 1)[-1]) \
                or "pack"
            PACKS_STUB.append({"name": name, "url": url, "sha": "abcdef123456",
                               "installed": 1_756_000_000})
            job = "pack%03d" % sum(1 for k in RUNS if k.startswith("pack"))
            RUNS[job] = {"polls": 0, "stopped": False, "pack": name}
            return self.reply({"ok": True, "job_id": job})

        if path == "/api/playground/packs/delete":
            try:
                name = str(json.loads(body or b"{}").get("name") or "")
            except json.JSONDecodeError:
                name = ""
            PACKS_STUB[:] = [p for p in PACKS_STUB if p["name"] != name]
            return self.reply({"ok": True, "note":
                "Removed from the shelf. A warm engine keeps what it already "
                "imported until its next restart."})

        m = re.match(r"/api/workflows/([^/]+)/delete$", path)
        if m:
            WORKFLOWS_STUB.pop(m.group(1), None)
            return self.reply({"ok": True})

        m = re.match(r"/api/workflows/([^/]+)$", path)
        if m:
            try:
                req = json.loads(body or b"{}")
            except json.JSONDecodeError:
                req = {}
            WORKFLOWS_STUB[m.group(1)] = {
                "graph": req.get("graph") or {},
                "meta": req.get("meta") or {},
            }
            return self.reply({"ok": True, "name": m.group(1)})

        m = re.match(r"/api/stop/(.+)$", path)
        if m:
            job = RUNS.get(m.group(1))
            if job:
                job["stopped"] = True
            return self.reply({"ok": True})

        # No job ids: a stub that starts jobs it cannot finish leaves the UI
        # polling a status that never lands, which looks like a hung backend.
        self.reply({"ok": True, "note": "preview server — no backend attached."})

    def do_GET(self):
        path = self.path.split("?")[0]

        # The built bundle, served the way the web container serves it. See the
        # note at the top of this file for why that is not the same thing as
        # what `npm run dev` serves.
        if path == "/":
            page = DIST / "index.html"
            if not page.is_file():
                # The one error this server can hit that is not a stub gap, and
                # it has exactly one cause. Saying so beats a traceback about a
                # missing file, which reads like the tool is broken.
                return self.reply(
                    "<pre>No build in web/dist.\n\n"
                    "  npm --prefix web run build\n</pre>",
                    "text/html; charset=utf-8", code=503)
            return self.reply(page.read_text(), "text/html; charset=utf-8")
        if path.startswith("/assets/"):
            f = DIST / path.lstrip("/")
            if not f.is_file():
                self.send_error(404)
                return
            kind = ("text/css" if f.suffix == ".css"
                    else "text/javascript" if f.suffix == ".js"
                    else "application/octet-stream")
            return self.reply(f.read_bytes(), kind)
        if path == "/api/state":
            api = app_api()
            # COLDSTART=n makes the first n answers claim the volume is empty, which is
            # the fault the page's re-check exists for and the only way to exercise it.
            _COLD["n"] += 1
            if _COLD["n"] <= int(os.environ.get("COLDSTART") or 0):
                return self.reply({**STATE, "models": [dict(m, present=False, size_gb=0)
                                                        for m in STATE["models"]],
                                   "loras": [], "shot_vocab": api["SHOT_VOCAB"],
                                   "shot_langs": api["H3_LANGUAGES"], "shot_roles": [],
                                   "caption_presets": [], "caption_models": [],
                                   "caption_defaults": {"preset": "", "model": ""}})
            return self.reply({
                **STATE,
                "shot_vocab": api["SHOT_VOCAB"],
                "shot_langs": api["H3_LANGUAGES"],
                "shot_roles": [dict(spec, key=k)
                               for k, spec in api["SHOT_REF_ROLES"].items()],
                # Shaped exactly as `state()` serves them — instruction and
                # custom flag included, because the textarea the preset prefills
                # is now the control this preview exists to exercise. The
                # in-memory customs ride behind the built-ins the way the
                # config Dict's do.
                "caption_presets": (
                    [{"key": k, "label": p["label"], "note": p["note"],
                      "instruction": p["instruction"], "custom": False}
                     for k, p in api["CAPTION_PRESETS"].items()]
                    + [{"key": k, **p, "custom": True}
                       for k, p in CUSTOM_PRESETS.items()]),
                "caption_models": (
                    [{"key": k, "label": m["label"], "note": m["note"],
                      "repo": m["repo"], "custom": False}
                     for k, m in api["CAPTION_MODELS"].items()
                     if k not in HIDDEN_VLMS]
                    + [{"key": k, **m, "custom": True}
                       for k, m in CUSTOM_VLMS.items()]),
                # The default follows the menu, as `state()` serves it — a
                # hidden default would sit blank in the captioner select.
                "caption_defaults": {
                    "preset": api["DEFAULT_CAPTION_PRESET"],
                    "model": (api["DEFAULT_CAPTION_MODEL"]
                              if api["DEFAULT_CAPTION_MODEL"] not in HIDDEN_VLMS
                              else next(
                                  (k for k in api["CAPTION_MODELS"]
                                   if k not in HIDDEN_VLMS),
                                  next(iter(CUSTOM_VLMS),
                                       api["DEFAULT_CAPTION_MODEL"])))},
                # The session form's three menus, shaped exactly as `state()`
                # serves them. Pulled rather than transcribed for the reason the
                # shot vocabulary is: a menu offering a value the route rejects
                # by name is a preview of a form that does not exist.
                "train_optimizers": [dict(v, key=k)
                                     for k, v in api["TRAIN_OPTIMIZERS"].items()],
                "lr_schedulers": [dict(v, key=k)
                                  for k, v in api["LR_SCHEDULERS"].items()],
                "timestep_samplings": [dict(v, key=k)
                                       for k, v in api["TIMESTEP_SAMPLINGS"].items()],
                "train_defaults": api["TRAIN_DEFAULTS"],
            })
        if path == "/api/gallery":
            # `total` and `stale` are half the gallery's contract now: the cap is
            # stated rather than silent, and a listing that knows it is behind says
            # so. Without them here the retry path and the "showing N of M" line are
            # unexercised in the one place they would be developed.
            #
            # `stale` is driven by a query flag rather than hard-coded false, because
            # the check that matters is "exactly one bounded retry chain, not a loop",
            # and that needs a server that can actually answer stale.
            q = parse_qs(urlparse(self.path).query)
            before = float((q.get("before") or ["0"])[0] or 0)
            limit = int((q.get("limit") or ["200"])[0] or 200)
            rows = [g for g in GALLERY
                    if not before or (g.get("modified") or 0) < before]
            return self.reply({"items": rows[:limit], "total": len(GALLERY),
                               "stale": (q.get("stale") or ["0"])[0] == "1"})
        if path == "/api/sessions":
            seed_sessions()
            # `total` alongside the rows, as the route serves it: the listing is
            # bounded because it is polled, and the board's "showing 3 of 140"
            # line has nothing to read without it.
            return self.reply({"sessions": [session_view(r) for r in SESSIONS],
                               "total": len(SESSIONS)})
        if path == "/api/characters":
            return self.reply({"characters": [
                {"handle": h, "note": c.get("note", ""),
                 "retention": c.get("retention", ""),
                 "refs": [{k: v for k, v in r.items() if k != "b64"}
                          for r in c.get("refs", [])],
                 "files": [r["file"] for r in c.get("refs", [])],
                 # Absent rather than null when there is none — the stub has to
                 # hold the shape app.py serves or the front end develops
                 # against a field that does not exist. This file's one named
                 # failure mode; see web/CLAUDE.md.
                 **({"lora": c["lora"]} if c.get("lora") else {})}
                for h, c in sorted(CHARS.items())]})
        m = re.match(r"/api/character-file/([^/]+)/(.+)$", path)
        if m:
            c = CHARS.get(m.group(1))
            r = next((x for x in (c or {}).get("refs", [])
                      if x["file"] == m.group(2)), None)
            if not r:
                return self.reply({"error": "no such file"}, code=404)
            body = base64.b64decode(r["b64"])
            ctype = {"image": "image/png", "audio": "audio/wav",
                     "video": "video/mp4"}[r.get("kind", "image")]
            self.send_response(200)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return None

        if path == "/api/datasets":
            return self.reply({"datasets": DATASETS})

        if path.endswith("/duplicates"):
            # The scan answers `scanning` for its first few calls before it
            # answers groups, because that path is otherwise unexercised
            # anywhere: a real scan only shows it on a folder big enough to
            # blow a ten-second budget, which is not a folder anyone keeps
            # beside a preview server. Module state rather than per-request,
            # for the reason RUNS is — progress only reads as progress if
            # consecutive polls disagree with each other.
            SCAN["calls"] = SCAN.get("calls", 0) + 1
            if SCAN["calls"] <= 4:
                measured = SCAN["calls"] * 5
                return self.reply({"scanning": True, "measured": measured, "total": 24,
                                   "images": measured, "groups": [], "reclaim": 0,
                                   "thresholds": {}, "summary": {
                                       "duplicate_groups": 0, "duplicate_images": 0,
                                       "similar_groups": 0, "similar_images": 0}})
            m = re.match(r"/api/datasets/([^/]+)/duplicates$", path)
            return self.reply(duplicate_report(REMOVED.get(m.group(1), set()) if m else set()))

        if path.endswith("/insight"):
            return self.reply(INSIGHT)
        m = re.match(r"/api/datasets/([^/]+)$", path)
        if m:
            name = m.group(1)
            row = next((d for d in DATASETS if d["name"] == name), None)
            gone = REMOVED.get(name, set())
            base = max(0, (row or {}).get("count", 0) - len(DUPE_FILES) + len(gone))
            return self.reply({
                "trigger_word": (row or {}).get("trigger_word", ""),
                "saved": (row or {}).get("saved", True),
                "images": [i for i in images(base, (row or {}).get("videos", 0))
                           if i["name"] not in gone],
            })

        # A clip off a set. The stub cannot mux an mp4, so what a tile gets is
        # the same swatch the gallery's clips get — the `<video>` paints nothing
        # from it, which leaves the frame black with its corner mark on it. That
        # is enough to develop the treatment against: what is being checked here
        # is the mark, the filter and the counts, and the only way to see a real
        # first frame is a real file on a real volume.
        m = re.match(r"/api/(?:thumb|image|clip)/[^/]+/(.+)$", path)
        if m:
            w, h = shape_of(m.group(1))
            return self.reply(swatch(w, h, m.group(1), hash(m.group(1))),
                              "image/svg+xml")
        # Any job id, not just the gallery's: the canvas now streams a finished
        # run's stills off this route by (job, file), so a pattern that only
        # matched `job\d+` served the gallery and left the canvas broken.
        m = re.match(r"/api/(?:file|cover)/([A-Za-z0-9_-]+\.[a-z0-9]+)$", path)
        if m:
            # One segment: the run is read off the name, as the route does.
            job = re.sub(r"_\d{2}$", "", m.group(1).split(".")[0])
            item = next((i for i in GALLERY if i["job_id"] == job), None)
            w, h = (item["width"], item["height"]) if item else (1024, 1024)
            # Labelled and coloured by *file*, not by job. Every frame of a batch
            # comes off one job id, so seeding on that alone drew four identical
            # rectangles — and the film strip's whole job is to move between them,
            # which is unfalsifiable when every frame looks the same.
            tag = m.group(1)
            # **A clip gets a clip.** This served the same SVG for `.mp4` as for
            # a still, on the reasoning that a card which renders beats a card
            # that is honest about being empty. True of a *card*, and it made
            # the canvas lie: `<video>` cannot decode an SVG, so it reported
            # `videoWidth: 0` and laid out at the height of its own controls —
            # roughly 150px, at any window size, for any clip. Every question
            # about how a clip sits on the canvas was therefore unanswerable
            # here, which is how a 720p clip came to overflow the canvas on the
            # deployed app with every check in this directory passing. See
            # `CLIP_720P`.
            if m.group(1).lower().endswith((".mp4", ".webm", ".mov")):
                return self.reply(CLIP_720P, "video/mp4")
            return self.reply(swatch(w, h, tag, hash(tag)), "image/svg+xml")

        # A finished job, with results. This used to land `{"images": []}`,
        # which meant the canvas — the largest thing on the page, and the one
        # the whole layout is built around — was the one region the preview
        # could never show in its filled state. Two images rather than one, so
        # the contact-sheet grid and the per-still hover actions are both
        # exercised; `files` alongside them so the video path lands too.
        # Its own status shape: no total to divide by, so no percent — the byte
        # count and the rate are the whole progress report, which is exactly the
        # state the real job is in against Drive.
        # The export, as a job that takes long enough to have a progress line.
        # Answering the first poll with "completed" would make the one state a
        # person actually watches — a scene being stitched — the one state this
        # server cannot show, which is the fault the generate stub records.
        if path == "/api/status/export_scene":
            EXPORT["polls"] += 1
            if EXPORT["polls"] < 4:
                return self.reply({
                    "status": "running", "percent": EXPORT["polls"] * 22,
                    "phase": f"Joining {EXPORT['takes']} takes — no re-encode"})
            return self.reply({
                "status": "completed", "percent": 100, "kind": "video",
                "files": ["export_scene.mp4"], "takes": EXPORT["takes"],
                "bytes": 18_400_000, "reencoded": False})

        if path == "/api/status/dl_gdrive":
            GDRIVE["polls"] += 1
            if GDRIVE["mode"] == "error":
                return self.reply({"status": "failed", "error":
                    "HTTPError: 403. If the file is not shared with 'Anyone with "
                    "the link', Drive serves a sign-in page instead of the file."})
            if GDRIVE["mode"] == "slow" or GDRIVE["polls"] < 2:
                gb = GDRIVE["polls"] * 0.4
                return self.reply({"status": "running", "mb_s": 88.4,
                                   "phase": f"Google Drive · {gb:.1f} GB",
                                   "downloaded_gb": gb})
            return self.reply({
                "status": "completed", "percent": 100, "size_gb": 0.43,
                "files": ["darkbrush.safetensors", "sunsetblur.safetensors"],
                "skipped": ["preview_grid.png", "README.md"],
                "folder": GDRIVE["folder"],
            })

        # A run that actually takes time, so the progress bar, the step line and
        # Cancel are all reachable without a deploy. The generate flow is the
        # most-used path on the page and the stub used to answer its very first
        # poll with "completed" — which meant the one state a user spends the
        # most time looking at was the one state this server could not show.
        m = re.match(r"/api/status/(gen\d+)$", path)
        if m:
            job = RUNS.setdefault(m.group(1), {"polls": 0, "stopped": False})
            job["polls"] += 1
            if job["stopped"]:
                return self.reply({"status": "stopped"})
            total = 8
            if job["polls"] < total:
                return self.reply({
                    "status": "running", "phase": "generate", "step": job["polls"],
                    "total_steps": total,
                    "percent": int(job["polls"] * 100 / total),
                })
            n = int(job.get("n") or 2)
            return self.reply({
                "status": "completed", "percent": 100,
                # Filenames, not base64 — the canvas streams these off /api/file
                # exactly as the gallery does, and stubbing the old inlined shape
                # would be testing a path the page no longer takes. As many as were
                # asked for, so the film strip has something to page through.
                "files": ["%s_%02d.png" % (m.group(1), i) for i in range(n)],
                "job_id": m.group(1), "output_dir": "/workspace/outputs/" + m.group(1),
                "seeds": [4242 + i for i in range(n)], "sampler": "Euler", "scheduler": "Simple",
                "steps": 8, "cfg_scale": 1.0, "shift": 1.15, "duration_s": 6.2,
                "width": 1024, "height": 1024,
                "loras": [{"name": "my_style", "unet": 0.8, "applied": True},
                          {"name": "gone", "unet": 1.0, "applied": False,
                           "reason": "no matching keys"}],
            })

        # The clip. Longer than the image run and opening on the phase that has no
        # step count, because the first minutes of a real take are 42.5 GB landing on
        # the card — which is the state the page names rather than showing a bar
        # sitting at zero looking stuck.
        m = re.match(r"/api/status/(vid\d+)$", path)
        if m:
            job = RUNS.setdefault(m.group(1), {"polls": 0, "stopped": False})
            job["polls"] += 1
            total = 10
            if job["stopped"]:
                return self.reply({"status": "stopped"})
            if job["polls"] <= 2:
                return self.reply({"status": "running", "phase": "loading", "percent": 0})
            if job["polls"] < total:
                step = job["polls"] - 2
                return self.reply({
                    "status": "running", "phase": "sampling", "step": step,
                    "total_steps": total - 2, "eta": "%ds" % (2 * (total - job["polls"])),
                    "percent": int(step * 100 / (total - 2)),
                })
            if job.get("still"):
                # **The other shape a video run lands in.** Stubbed for the same
                # reason `/api/generate` echoes `num_images`: a canvas that only
                # ever gets a clip makes the strip look right in the one case
                # that needs no navigation, and a still is 22 frames deep.
                #
                # No `seconds` and no `fps`, matching `_plan_h3` — they are true
                # of the fragment sampled and false of the picture delivered, so
                # a stub carrying them would hide that the meta line has to read
                # differently for a still.
                return self.reply({
                    "status": "completed", "percent": 100, "still": True,
                    "files": ["%s_%02d.png" % (m.group(1), i) for i in range(H3_STILL_FRAMES)],
                    "job_id": m.group(1), "width": 1344, "height": 768,
                    "frames": H3_STILL_FRAMES, "seed": 4242, "steps": 4,
                    "duration_s": 11.4,
                })
            return self.reply({
                "status": "completed", "percent": 100, "files": ["%s.mp4" % m.group(1)],
                "job_id": m.group(1), "width": 1280, "height": 720,
                "seconds": 5, "frames": 120, "fps": 24, "seed": 4242,
                "steps": 20, "duration_s": 214.0,
            })

        # Hours, not seconds — so the fields a long run is read by are the ones
        # stubbed: epoch over total, step over total, rate, elapsed against ETA
        # and a loss that actually falls. A bar alone cannot tell "training"
        # from "stuck", which is the whole reason the card's meta line exists.
        m = re.match(r"/api/status/(train\d+)$", path)
        if m:
            return self.reply(train_status(m.group(1)))

        # Three files, sequentially, with the queue position in the phase — the
        # thing a per-family button exists to show. Stopping mid-queue reports
        # what landed and what did not, because that is the state a cancelled
        # queue is actually in and "Cancelled." on its own loses it.
        m = re.match(r"/api/status/(dl_fam_[a-z0-9-]+)$", path)
        if m:
            job = RUNS.setdefault(m.group(1), {"polls": 0, "stopped": False})
            job["polls"] += 1
            names = ["Krea 2 Turbo", "Qwen3-VL 4B", "Krea 2 VAE"]
            per, total = 3, 9
            if job["stopped"]:
                got = min(len(names), job["polls"] // per)
                return self.reply({"status": "stopped", "downloaded": names[:got],
                                   "remaining": names[got:]})
            if job["polls"] < total:
                i = min(len(names) - 1, job["polls"] // per)
                gb = (job["polls"] % per + 1) * 5.4
                return self.reply({
                    "status": "running", "mb_s": 213.7, "downloaded_gb": gb,
                    "phase": f"{names[i]} · {i + 1} of {len(names)} · {gb:.1f} of 16.2 GB",
                    "percent": int(job["polls"] * 100 / total),
                })
            return self.reply({"status": "completed", "percent": 100,
                               "downloaded": names, "failed": []})

        # ── Playground ─────────────────────────────────────────────────────
        if path == "/api/playground/nodes":
            return self.reply(NODE_CATALOGUE_STUB)

        if path == "/api/playground/packs":
            return self.reply({"packs": PACKS_STUB})

        if path == "/api/storyboards":
            rows = []
            for name, b in STORYBOARDS.items():
                cover = next((p.get("picture") for p in b["panels"] if p.get("picture")), None)
                rows.append({"name": name, "title": b["title"], "panels": len(b["panels"]),
                             "updated": b["updated"], "cover": cover})
            rows.sort(key=lambda r: -(r["updated"] or 0))
            return self.reply({"boards": rows})
        m = re.match(r"/api/storyboard/([a-z0-9][a-z0-9_-]*)/file/([^/]+)$", path)
        if m:
            data = BOARD_FILES.get(m.group(1), {}).get(m.group(2))
            if data is None:
                return self.reply({"error": "Not found."}, code=404)
            ext = Path(m.group(2)).suffix.lower()
            return self.reply(data, {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                                     ".webp": "image/webp"}.get(ext, "image/png"))
        m = re.match(r"/api/storyboard/([a-z0-9][a-z0-9_-]*)$", path)
        if m:
            b = STORYBOARDS.get(m.group(1))
            if b is None:
                return self.reply({"error": f"No storyboard called {m.group(1)!r}."})
            return self.reply({"board": {**b, "name": m.group(1)}})

        if path == "/api/workflows":
            return self.reply({"workflows": [
                {"name": name,
                 "created": rec["meta"].get("created"),
                 "seed_of": rec["meta"].get("seed_of"),
                 "exposes": rec["meta"].get("exposes") or [],
                 "mtime": 1_756_000_000}
                for name, rec in sorted(WORKFLOWS_STUB.items())
            ]})

        m = re.match(r"/api/workflows/([^/]+)$", path)
        if m:
            rec = WORKFLOWS_STUB.get(m.group(1))
            if not rec:
                return self.reply({"error":
                    f"No workflow named {m.group(1)!r} under "
                    "/workspace/workflows."})
            return self.reply({"name": m.group(1), "graph": rec["graph"],
                               "meta": rec["meta"]})

        # A Playground run, with a failure mode: see /api/playground/run.
        m = re.match(r"/api/status/(pg\d+)$", path)
        if m:
            job = RUNS.setdefault(m.group(1), {"polls": 0, "stopped": False})
            job["polls"] += 1
            if job["stopped"]:
                return self.reply({"status": "stopped"})
            total = 8
            if job["polls"] < total:
                return self.reply({
                    "status": "running", "phase": "generate",
                    "step": job["polls"], "total_steps": total,
                    "percent": int(job["polls"] * 100 / total)})
            if job.get("boom"):
                return self.reply({
                    "status": "failed",
                    "error": "KSampler: mat1 and mat2 shapes cannot be "
                             "multiplied (77x768 and 2560x3072)",
                    "error_node": "boom", "error_node_type": "KSampler"})
            return self.reply({
                "status": "completed", "percent": 100,
                "files": [m.group(1) + "_00.png", m.group(1) + "_01.png"],
                "job_id": m.group(1), "model": "playground",
                "output_dir": "/workspace/outputs/" + m.group(1),
                "duration_s": 6.2})

        m = re.match(r"/api/status/(pgrst\d+)$", path)
        if m:
            job = RUNS.setdefault(m.group(1), {"polls": 0, "stopped": False})
            job["polls"] += 1
            if job["polls"] < 3:
                return self.reply({"status": "running",
                                   "phase": "restarting the engine"})
            return self.reply({"status": "completed", "files": [],
                               "note": "Engine restarted. The next run "
                                       "reloads its checkpoint."})

        m = re.match(r"/api/status/(pgcat\d+|pack\d+)$", path)
        if m:
            job = RUNS.setdefault(m.group(1), {"polls": 0, "stopped": False})
            job["polls"] += 1
            if job["polls"] < 2:
                return self.reply({"status": "running",
                                   "phase": "starting ComfyUI on CPU"})
            done = {"status": "completed", "files": [],
                    "nodes": len(NODE_CATALOGUE_STUB)}
            if job.get("pack"):
                done.update(pack=job["pack"], sha="abcdef123456",
                            note="Installed. Restart the engine to load it "
                                 "into a warm generator.")
            return self.reply(done)

        if path.startswith("/api/status/"):
            return self.reply({
                "status": "completed", "percent": 100, "files": ["%s.mp4" % m.group(1)],
                "seeds": [4242, 4243], "sampler": "Euler", "scheduler": "Simple",
                "steps": 8, "cfg_scale": 1.0, "shift": 1.15, "duration_s": 6.2,
                "width": 1024, "height": 1024,
                "seconds": 5, "frames": 120, "fps": 24, "seed": 4242,
                # One applied and one not: a stack that silently no-ops looks
                # identical to a stack that had no effect, and the line that
                # says which is which has to be visible somewhere.
                "loras": [{"name": "my_style", "unet": 0.8, "applied": True},
                          {"name": "gone", "unet": 1.0, "applied": False,
                           "reason": "no matching keys"}],
            })
        if path.startswith("/api/outputs/"):
            return self.reply({"images": [
                {"data": "data:image/svg+xml;base64," + base64.b64encode(
                    swatch(w, h, name, seed)).decode()}
                for name, (w, h, seed) in
                (("shot 1", (1024, 1024, 11)), ("shot 2", (1024, 1024, 29)))
            ]})

        self.reply({"error": f"No stub for {path}"}, code=404)


if __name__ == "__main__":
    print(f"Visionary UI preview  ->  http://127.0.0.1:{PORT}")
    print(f"Serving {DIST} with a stubbed API; compilers pulled from {APP}.")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
