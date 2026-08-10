"""
Serve UI_HTML on localhost with a stubbed API, so the frontend can be looked at
without a deploy.

The real UI is a @modal.asgi_app(): reaching it means building a remote image,
mounting the volume, and paying a cold start — minutes, for a CSS change. But
UI_HTML is one self-contained string with no build step, so the whole frontend
can be exercised locally against fake JSON. This is the dev server the project
does not otherwise have.

    python3 tools/preview_ui.py [port]

Two deliberate choices:

- **app.py is re-read on every request.** Caching it at import meant editing
  UI_HTML and reloading showed the old page, which reads as "my change did not
  work" rather than "the server is stale". Reload is the edit loop, so reload
  has to be honest.
- **Threaded.** A gallery is a grid, so opening it fires a dozen concurrent
  media requests. The single-threaded version dropped most of them and the
  covers came back blank — a layout bug that was not in the layout.

Stdlib only, and never imported by app.py: this must run on a laptop with no
torch, no modal, and no credentials.
"""

import base64
import json
import re
import sys
import time
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app.py"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8777


def ui_html() -> str:
    """Pull UI_HTML back out of app.py. Split, not import — importing app.py
    pulls in modal and builds image definitions at module scope."""
    src = APP.read_text()
    return src.split('UI_HTML = r"""', 1)[1].rsplit('"""', 1)[0]


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

GALLERY = [
    {
        "job_id": f"job{i:03d}",
        "kind": "video" if i % 5 == 3 else "image",
        "files": [f"{i:02d}.mp4" if i % 5 == 3 else f"{i:02d}.png"],
        "created": time.time() - i * 3600,
        "modified": time.time() - i * 3600,
        "prompt": LONG_PROMPT,
        "negative_prompt": "blurry, low quality, watermark",
        "width": SIZES[i % 4][0],
        "height": SIZES[i % 4][1],
        **(
            {"seconds": 5, "frames": 120, "fps": 24, "seed": 4000 + i,
             "steps": 20, "sampler": "res_multistep", "scheduler": "simple",
             "references": 0, "ref_videos": 0}
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
                 {"box": [0.04, 0.08, 0.42, 0.86], "lora": "k3nan.safetensors",
                  "strength": 1.35, "prompt": "a man in a long coat",
                  "ref": False},
                 {"box": [0.54, 0.08, 0.42, 0.86], "lora": "None",
                  "strength": 1.0, "prompt": "a woman", "ref": True},
             ] if i % 4 == 0 else []),
             "region_weight": 1.0}
        ),
    }
    for i in range(14)
]

# Two unsaved sets and three saved ones, because the rail has to be looked at
# with both groups in it: one group alone hides whether the headings, the dashed
# card and the spacing between the groups read as a difference or as damage.
DATASETS = [
    {"name": "set_2", "count": 31, "uncaptioned": 31, "cover": "0.png",
     "trigger_word": "", "saved": False},
    {"name": "beach_walk", "count": 12, "uncaptioned": 3, "cover": "0.png",
     "trigger_word": "", "saved": False},
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
        {"key": "wan_high", "label": "Wan 2.2 A14B high", "note": "high-noise expert",
         "family": "Wan 2.2 — video", "repo_id": "Comfy-Org/Wan_2.2", "present": False,
         "approx_gb": 14.3, "gated": False},
        {"key": "wan_low", "label": "Wan 2.2 A14B low", "note": "low-noise expert",
         "family": "Wan 2.2 — video", "repo_id": "Comfy-Org/Wan_2.2", "present": False,
         "approx_gb": 14.3, "gated": False},
        {"key": "wan_vae", "label": "Wan VAE", "note": "",
         "family": "Wan 2.2 — video", "repo_id": "Comfy-Org/Wan_2.2", "present": False,
         "approx_gb": 0.5, "gated": False},
        {"key": "umt5", "label": "UMT5 XXL", "note": "text encoder",
         "family": "Wan 2.2 — video", "repo_id": "Comfy-Org/Wan_2.2", "present": True,
         "size_gb": 6.7, "approx_gb": 6.7, "gated": False},
    ],
    # Four shapes, because `<lora:…>` has to name each of them unambiguously:
    # a trained LoRA with epoch checkpoints, a bare file dropped at the top of
    # loras/ (which is what a Google Drive pull with no folder produces), the
    # matched Wan speed pairs — whose files are BOTH called high/low, so they
    # are the case that proves the picker qualifies a colliding name with its
    # folder instead of silently offering the wrong expert — and two files whose
    # names differ only in case, which is what a Drive pull and a training run
    # disagreeing about capitalisation leaves behind. Both of the last two have
    # to stay typeable: resolution is exact-first for exactly this reason.
    "loras": [
        {"name": "my_style", "trigger_word": "ohwx_style", "files": [
            {"path": "/workspace/loras/my_style/my_style.safetensors",
             "name": "my_style"},
            {"path": "/workspace/loras/my_style/my_style-000020.safetensors",
             "name": "my_style-000020"}]},
        {"name": "alxcn", "trigger_word": "", "files": [
            {"path": "/workspace/loras/alxcn.safetensors", "name": "alxcn.safetensors"}]},
        {"name": "K3nan", "trigger_word": "", "files": [
            {"path": "/workspace/loras/K3nan.safetensors", "name": "K3nan.safetensors"}]},
        {"name": "k3nan", "trigger_word": "", "files": [
            {"path": "/workspace/loras/k3nan.safetensors", "name": "k3nan.safetensors"}]},
        {"name": "wan22-speed-t2v", "trigger_word": "", "files": [
            {"path": "/workspace/loras/wan22-speed-t2v/high.safetensors", "name": "high"},
            {"path": "/workspace/loras/wan22-speed-t2v/low.safetensors", "name": "low"}]},
        {"name": "wan22-speed-i2v", "trigger_word": "", "files": [
            {"path": "/workspace/loras/wan22-speed-i2v/high.safetensors", "name": "high"},
            {"path": "/workspace/loras/wan22-speed-i2v/low.safetensors", "name": "low"}]},
    ],
    # One of each shape the composer has to redraw for: audio + references and
    # no CFG (H3), the two-expert pair (A14B), and the single-expert 5B. A stub
    # with only one of them cannot catch a control that fails to appear.
    "video_models": [
        {"key": "h3", "label": "MiniMax-H3",
         "note": "Sound and picture in one pass",
         "tiers": {"full": "768p", "draft": "544p draft"},
         "lengths": [5, 6, 8, 10, 12, 14],
         "samplers": ["res_multistep", "euler", "dpmpp_2m"],
         "schedulers": ["simple", "normal", "beta"],
         "defaults": {"steps": 20, "sampler": "res_multistep",
                      "scheduler": "simple", "tier": "full", "seconds": 5},
         "supports": {"loras": False, "experts": False, "cfg": False,
                      "negative": False, "references": True,
                      "last_frame": True, "audio": True},
         "tasks": {"fl2va": {"ready": True, "missing": []},
                   "ref2va": {"ready": True, "missing": []}},
         "ready": True},
        {"key": "wan14b", "label": "Wan 2.2 A14B",
         "note": "Two experts · silent",
         "tiers": {"full": "720p", "draft": "480p draft"},
         "lengths": [2, 3, 4, 5],
         "samplers": ["euler", "uni_pc", "dpmpp_2m", "res_multistep"],
         "schedulers": ["simple", "normal", "beta"],
         "defaults": {"steps": 20, "cfg": 3.5, "shift": 8.0, "sampler": "euler",
                      "scheduler": "simple", "tier": "full", "seconds": 5},
         "supports": {"loras": True, "experts": True, "cfg": True,
                      "negative": True, "references": False,
                      "last_frame": True, "audio": False},
         # i2v deliberately not downloaded: attaching a first frame here should
         # disable Generate and name the missing pair, which is the state most
         # worth being able to look at.
         "tasks": {"t2v": {"ready": True, "missing": []},
                   "i2v": {"ready": False,
                           "missing": ["Wan 2.2 I2V · high noise",
                                       "Wan 2.2 I2V · low noise"]}},
         "ready": True},
        {"key": "wan5b", "label": "Wan 2.2 TI2V 5B",
         "note": "24 fps · silent",
         "tiers": {"full": "704p", "draft": "480p draft"},
         "lengths": [2, 3, 4, 5],
         "samplers": ["euler", "uni_pc", "dpmpp_2m", "res_multistep"],
         "schedulers": ["simple", "normal", "beta"],
         "defaults": {"steps": 30, "cfg": 5.0, "shift": 8.0, "sampler": "euler",
                      "scheduler": "simple", "tier": "full", "seconds": 5},
         "supports": {"loras": True, "experts": False, "cfg": True,
                      "negative": True, "references": False,
                      "last_frame": False, "audio": False},
         "tasks": {"t2v": {"ready": True, "missing": []},
                   "i2v": {"ready": True, "missing": []}},
         "ready": True},
    ],
    "wan_experts": ["both", "high", "low"],
    "max_loras": 6, "max_refs": 9, "max_ref_videos": 3,
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

# Which Drive outcome the next poll reports. Mutable module state, like
# DATASETS above and for the same reason: this flow is judged by what the card
# does over several polls, and a fixed reply cannot show a transfer moving.
GDRIVE = {"mode": "ok", "folder": "", "polls": 0}

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


def images(n: int = 24) -> list:
    out = []
    for i in range(n):
        w, h = shape_of(str(i))
        out.append({"name": f"{i}.png", "caption": CAPTIONS[i % len(CAPTIONS)],
                    "bytes": 780_000 + i * 4_100, "width": w, "height": h})
    return out


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


# In-flight fake runs, keyed by job id: {polls, stopped}. Module state rather
# than per-request, because a run only reads as a run if consecutive polls
# disagree with each other.
RUNS: dict = {}


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

        # Saving and culling mutate the list in memory, because both are
        # judged by what the rail does next — a set moving out of Unsaved, a
        # card leaving and the heading going with it when it was the last one.
        # A flat {"ok": true} leaves the page redrawing the state it started in.
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

        if path == "/api/datasets":
            try:
                new = json.loads(body or b"{}").get("name") or "set_x"
            except json.JSONDecodeError:
                new = "set_x"
            row = {"name": new, "count": 0, "uncaptioned": 0, "cover": None,
                   "trigger_word": "", "saved": False}
            DATASETS.insert(0, row)
            return self.reply({"ok": True, **row})

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

        if path == "/api/generate":
            RUNS["gen000"] = {"polls": 0, "stopped": False}
            return self.reply({"ok": True, "job_id": "gen000"})

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

        if path == "/":
            return self.reply(ui_html(), "text/html; charset=utf-8")
        if path == "/api/state":
            return self.reply(STATE)
        if path == "/api/gallery":
            return self.reply({"items": GALLERY})
        if path == "/api/datasets":
            return self.reply({"datasets": DATASETS})

        if path.endswith("/insight"):
            return self.reply(INSIGHT)
        m = re.match(r"/api/datasets/([^/]+)$", path)
        if m:
            name = m.group(1)
            row = next((d for d in DATASETS if d["name"] == name), None)
            return self.reply({
                "trigger_word": (row or {}).get("trigger_word", ""),
                "saved": (row or {}).get("saved", True),
                "images": images((row or {}).get("count", 0)),
            })

        m = re.match(r"/api/(?:thumb|image)/[^/]+/(.+)$", path)
        if m:
            w, h = shape_of(m.group(1))
            return self.reply(swatch(w, h, m.group(1), hash(m.group(1))),
                              "image/svg+xml")
        # Any job id, not just the gallery's: the canvas now streams a finished
        # run's stills off this route by (job, file), so a pattern that only
        # matched `job\d+` served the gallery and left the canvas broken.
        m = re.match(r"/api/file/([A-Za-z0-9_-]+)/(.+)$", path)
        if m:
            item = next((i for i in GALLERY if i["job_id"] == m.group(1)), None)
            w, h = (item["width"], item["height"]) if item else (1024, 1024)
            # Videos get a still too. A stub cannot mux a real clip, and a card
            # that renders is worth more here than a card that is honest about
            # being empty — the video path itself is only reachable on Modal.
            return self.reply(swatch(w, h, m.group(1), hash(m.group(1))),
                              "image/svg+xml")

        # A finished job, with results. This used to land `{"images": []}`,
        # which meant the canvas — the largest thing on the page, and the one
        # the whole layout is built around — was the one region the preview
        # could never show in its filled state. Two images rather than one, so
        # the contact-sheet grid and the per-still hover actions are both
        # exercised; `files` alongside them so the video path lands too.
        # Its own status shape: no total to divide by, so no percent — the byte
        # count and the rate are the whole progress report, which is exactly the
        # state the real job is in against Drive.
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
                "files": ["alxcn.safetensors", "k3nan.safetensors"],
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
            return self.reply({
                "status": "completed", "percent": 100,
                # Filenames, not base64 — the canvas streams these off /api/file
                # exactly as the gallery does, and stubbing the old inlined shape
                # would be testing a path the page no longer takes.
                "files": ["120000_00.png", "120000_01.png"],
                "job_id": m.group(1), "output_dir": "/workspace/outputs/" + m.group(1),
                "seeds": [4242, 4243], "sampler": "Euler", "scheduler": "Simple",
                "steps": 8, "cfg_scale": 1.0, "shift": 1.15, "duration_s": 6.2,
                "width": 1024, "height": 1024,
                "loras": [{"name": "my_style", "unet": 0.8, "applied": True},
                          {"name": "gone", "unet": 1.0, "applied": False,
                           "reason": "no matching keys"}],
            })

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

        if path.startswith("/api/status/"):
            return self.reply({
                "status": "completed", "percent": 100, "files": ["clip.mp4"],
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
    print(f"Serving UI_HTML from {APP}, re-read on every reload.")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
