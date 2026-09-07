"""
Dump the shot compiler's table and outputs as golden fixtures for the client's port.

    python3.11 tools/shot_fixtures.py [path/to/Fixtures/shot.json]

`SHOT_VOCAB` and its compilers are the one piece of Python a client ports
rather than calls — a prompt is a compilation target and the compiler is where
this project's knowledge of each model's grammar lives, so a client has to hold
it to show the document as it is composed. A port is a second implementation,
and the rule about those is that a preview with its own implementation is a
preview that can disagree with the run. This file is what makes the two agree
by measurement rather than by reading: every case here is compiled by the real
Python, pulled out of app.py by AST the way every smoke test pulls it, and the
client's tests replay the same inputs and demand the same bytes.

Re-run it whenever `SHOT_VOCAB`, a compiler or a validator changes, then run
the client's tests. A fixture that fails there is the port lagging the Python;
a fixture that fails to regenerate is the Python having grown a case the matrix
does not cover, and the matrix should grow with it.

**The client is a separate checkout, so the destination is an argument.** It
used to be a path inside this repository and could be a constant; a constant
now would write a fixture nobody reads, which is the failure this whole file
exists to prevent, arrived at from the other side. `VISIONARY_FIXTURES` in the
environment sets it once per machine; the argument wins over both. Sorted keys
and stable order, so a regeneration that changes nothing is a diff of nothing.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _from_app import SHOT, pull  # noqa: E402

G = pull(SHOT)

import os  # noqa: E402

_DEST = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("VISIONARY_FIXTURES"))
if not _DEST:
    sys.exit("Where to? Pass the client's Fixtures/shot.json, or set "
             "VISIONARY_FIXTURES. It is a separate checkout now, so there is "
             "no default that is not a guess.")
OUT = Path(_DEST).expanduser().resolve()
if OUT.is_dir():
    OUT = OUT / "shot.json"

# The typed halves `probe_compile.py` pins, for the same three reasons: the
# byte-for-byte case, the lowercase fragment the separator rule softens a full
# stop for, a sentence already closed, and one written across two lines.
TYPED = {
    "empty": "",
    "fragment": "a portrait of k3nan",
    "closed": "A dancer turns under a streetlight.",
    "twoline": "A dancer turns.\nThe street is wet.",
    "spaced": "  two   men  fight\tin a corridor ",
    "question": "who is at the door?",
    "quoted": 'she says "no"',
}

# Pill rails worth pinning. Names are what the client's test reports on a miss.
RAILS = {
    "none": [],
    "framing": [{"key": "framing.cu"}],
    "framing+angle": [{"key": "framing.cu"}, {"key": "angle.low"}],
    "angle+framing": [{"key": "angle.low"}, {"key": "framing.cu"}],
    "stacked": [{"key": "light.window"}, {"key": "light.golden"}, {"key": "light.candle"},
                {"key": "tone.noir"}, {"key": "tone.s16"}, {"key": "framing.mcu"},
                {"key": "angle.dutch"}],
    "camera": [{"key": "camera.pushin"}],
    "camera-amp": [{"key": "camera.panr", "amp": "large", "speed": "fast"}],
    "camera-amp-only": [{"key": "camera.tiltu", "amp": "small"}],
    "camera-speed-only": [{"key": "camera.orbit", "speed": "slow"}],
    "camera-medium": [{"key": "camera.craneu", "amp": "medium", "speed": "normal"}],
    "camera-noverb-amp": [{"key": "camera.handheld", "amp": "large"}],
    "dialogue": [{"key": "say.dialogue", "value": "Take the morning with you.", "lang": "English"}],
    "dialogue-fr": [{"key": "say.dialogue", "value": "Prends le matin avec toi, d'accord ?", "lang": "French"}],
    "dialogue-empty": [{"key": "say.dialogue", "value": ""}],
    "screen": [{"key": "say.screen", "value": "OPEN 24 HOURS"}],
    "both-say": [{"key": "say.screen", "value": "EXIT"},
                 {"key": "say.dialogue", "value": "This way.", "lang": "English"}],
    "sound": [{"key": "sound.rain"}, {"key": "sound.traffic"}],
    "sound-other": [{"key": "sound.other", "value": "ice tapping the side of a crystal glass"},
                    {"key": "sound.roomtone"}],
    "score": [{"key": "score.piano"}, {"key": "score.slow"}, {"key": "score.fading"}],
    "score-silent": [{"key": "score.silent"}],
    "score-then-silent": [{"key": "score.piano"}, {"key": "score.silent"}],
    "silent-then-score": [{"key": "score.silent"}, {"key": "score.strings"}],
    "score-other": [{"key": "score.other", "value": "a lone cello, barely there"}],
    "everything": [{"key": "framing.wide"}, {"key": "angle.high"}, {"key": "light.neon"},
                   {"key": "tone.anamorphic"}, {"key": "say.screen", "value": "BAR"},
                   {"key": "camera.trackside"}, {"key": "sound.crowd"}, {"key": "sound.cloth"},
                   {"key": "score.synth"}, {"key": "score.driving"}],
    "whitespace-value": [{"key": "say.screen", "value": "  two\n lines   here "}],
    "long-value": [{"key": "sound.other", "value": "x" * 450}],
    "string-entries": ["framing.xcu", "tone.desat"],
}

ROLES = {
    "none": [],
    "identity": ["identity"],
    "gap": ["", "wardrobe"],
    "all": ["identity", "wardrobe", "location", "style", "prop", "action"],
}

TASKS = ("t2va", "i2va", "l2va", "fl2va", "ref2va")


def compile_image(typed, rail):
    return G["_compile_image_prompt"](typed, G["_validate_shot"](rail))


def compile_video(typed, rail, *, task, seconds, roles, n_refs):
    return G["_compile_h3_prompt"](
        typed=typed, pills=G["_validate_shot"](rail), task=task, seconds=seconds,
        roles=G["_validate_ref_roles"](roles, n_refs))


def attempt(fn):
    try:
        return {"prompt": fn()}
    except ValueError as exc:
        return {"error": str(exc)}


cases = []
for tname, typed in TYPED.items():
    for rname, rail in RAILS.items():
        cases.append({"name": f"image/{tname}/{rname}", "kind": "image", "typed": typed,
                      "shot": rail, **attempt(lambda: compile_image(typed, rail))})

# The video matrix is the product of everything that changes the document:
# the typed shape, the rail, the task, and the roles. Kept to the combinations
# that exercise a different branch rather than the full product, which would
# be thousands of lines saying the same thing.
for tname in ("fragment", "closed", "twoline", "empty"):
    typed = TYPED[tname]
    for rname, rail in RAILS.items():
        for task in TASKS:
            roles = ROLES["identity"] if task == "ref2va" else []
            n_refs = 1 if task == "ref2va" else 0
            cases.append({"name": f"video/{tname}/{rname}/{task}", "kind": "video",
                          "typed": typed, "shot": rail, "task": task, "seconds": 5,
                          "roles": roles, "n_refs": n_refs,
                          **attempt(lambda: compile_video(typed, rail, task=task, seconds=5,
                                                          roles=roles, n_refs=n_refs))})
for rname, roles in ROLES.items():
    n_refs = max(2, len(roles))
    for rail in ("none", "framing", "dialogue"):
        cases.append({"name": f"video/roles/{rname}/{rail}", "kind": "video",
                      "typed": TYPED["closed"], "shot": RAILS[rail], "task": "ref2va",
                      "seconds": 8.5, "roles": roles, "n_refs": n_refs,
                      **attempt(lambda: compile_video(TYPED["closed"], RAILS[rail], task="ref2va",
                                                      seconds=8.5, roles=roles, n_refs=n_refs))})
# The clip length reaches the alignment sentence formatted to two places.
for secs in (5, 6, 8.5, 12.25, 14):
    cases.append({"name": f"video/seconds/{secs}", "kind": "video", "typed": TYPED["closed"],
                  "shot": RAILS["framing"], "task": "fl2va", "seconds": secs, "roles": [],
                  "n_refs": 0,
                  **attempt(lambda: compile_video(TYPED["closed"], RAILS["framing"], task="fl2va",
                                                  seconds=secs, roles=[], n_refs=0))})

# ── the scene ────────────────────────────────────────────────────────────────
# The composer's input: a cast and a timeline. One corpus shaped like the
# guide's own complete example — a location and two people who both speak,
# over three shots, with a line crossing a cut — then one variant per branch
# of the compiler, and one refusal per sentence the validator can say.


def three_shots():
    return {
        "style": "Live-action, cinematic",
        "cast": [
            {"id": "loc", "kind": "place", "name": "cafe", "refs": [
                {"kind": "image", "index": 0, "slots": ["establishing"]}]},
            {"id": "w", "kind": "character", "name": "Ava", "refs": [
                {"kind": "image", "index": 1, "slots": ["face", "wardrobe"]}]},
            {"id": "m", "kind": "character", "name": "Sam", "refs": [
                {"kind": "image", "index": 2, "slots": ["face"]}]},
        ],
        "shots": [
            {"line": "@ava sits in the @cafe holding a cookie", "beats": 1,
             "pills": ["framing.medium", "sound.roomtone"],
             "say": {"who": "w", "text": "Hey! Watch your dog!"}},
            {"line": "@sam beside her on the sofa", "beats": 1,
             "pills": ["framing.cu", "camera.pushin"],
             "say": {"who": "m", "text": "He just likes cookies more than me.",
                     "carry": True}},
            {"line": "@ava again, her annoyance softening", "beats": 1,
             "pills": ["framing.cu"],
             "say": {"who": "w", "text": "Well, he has good taste at least.",
                     "cutoff": True}},
        ],
    }


def with_shots(**patch):
    sc = three_shots()
    sc.update(patch)
    return sc


def scene_cases():
    silent = three_shots()
    for sh in silent["shots"]:
        sh["say"], sh["pills"] = {}, []
    pair = three_shots()
    pair["shots"][0]["say"] = {"who": ["w", "m"], "text": "Wait for us!"}
    pair["shots"][1]["say"] = {}
    pair["shots"][2]["say"] = {}
    vo = three_shots()
    vo["shots"][1]["say"]["offscreen"] = True
    vo["shots"][1]["say"]["carry"] = False
    shared = three_shots()
    shared["cast"][2]["refs"] = [{"kind": "image", "index": 1, "slots": ["face"]}]
    noted = three_shots()
    noted["cast"][1]["note"] = "the young woman with the cropped hair"
    noted["cast"][1]["refs"] = [
        {"kind": "image", "index": 1, "slots": ["image"]},
        {"kind": "image", "index": 2, "slots": ["image"], "note": "the olive field coat"},
    ]
    noted["cast"][2]["refs"] = [{"kind": "image", "index": 0, "slots": ["image"], "sheet": True}]
    noted["cast"][2]["retention"] = "attribute_transfer"
    noted["grade"] = "a warm, low-contrast grade"
    voice = three_shots()
    voice["cast"].append({"id": "n", "kind": "character", "name": "narrator", "refs": [
        {"kind": "audio", "index": 0, "slots": ["audio"], "note": "a low, unhurried voice"}]})
    voice["shots"][0]["say"] = {"who": "n", "text": "It began in a cafe.", "offscreen": True}
    reuse = three_shots()
    reuse["cast"][1]["refs"].append({"kind": "audio", "index": 0, "slots": ["voice"], "role": "reuse"})
    prose_only = {"cast": [{"id": "a", "kind": "character", "name": "ava", "note": "a woman in a red coat", "refs": []}],
                  "shots": [{"line": "@ava at the window", "beats": 1}]}
    unmentioned = with_shots(shots=[{"line": "rain on the glass", "beats": 2, "pills": ["sound.rain"]},
                                    {"line": "@sam looks up", "beats": 1}])
    motion = with_shots(cast=three_shots()["cast"] + [{"id": "d", "name": "dancer", "refs": [
        {"kind": "video", "index": 0, "slots": ["motion"], "note": "her walking motion"}]}])
    plain = {"shots": [{"line": "an empty diner at 3am", "beats": 1}]}
    plain_pills = {"shots": [{"line": "an empty diner at 3am", "beats": 1, "pills": ["framing.wide", "score.piano", "score.slow"]},
                             {"line": "the neon flickers", "beats": 3, "pills": ["camera.panl", "sound.rain"]}],
                   "style": "16mm documentary", "grade": "desaturated"}
    unseen = {"shots": [{"line": "an empty road at dawn", "beats": 1,
                         "say": {"text": "Nobody came back.", "voice": "a low, tired man's voice", "lang": "Spanish"}}]}
    articles = with_shots(shots=[{"line": "Ava walks into the @cafe and greets a @sam by the door; an @Ava smiles", "beats": 1}])

    ok = [
        ("corpus", three_shots(), dict(n_refs=3, task="ref2va", seconds=8.0)),
        ("corpus-i2va", three_shots(), dict(n_refs=3, task="i2va", seconds=8.0)),
        ("corpus-12s", three_shots(), dict(n_refs=3, task="ref2va", seconds=12.0)),
        ("silent", silent, dict(n_refs=3)),
        ("pair", pair, dict(n_refs=3)),
        ("voiceover", vo, dict(n_refs=3)),
        ("shared-file", shared, dict(n_refs=3)),
        ("notes-sheet-grade", noted, dict(n_refs=3)),
        ("narrator-audio", voice, dict(n_refs=3, n_auds=1)),
        ("audio-reuse", reuse, dict(n_refs=3, n_auds=1)),
        ("prose-only-cast", prose_only, dict(n_refs=0, task="t2va", seconds=5.0)),
        ("unmentioned-visible", unmentioned, dict(n_refs=3)),
        ("motion-video", motion, dict(n_refs=3, n_vids=1)),
        ("plain-base", plain, dict(n_refs=0, task="t2va", seconds=5.0)),
        ("plain-i2va", plain, dict(n_refs=0, task="i2va", seconds=5.0)),
        ("plain-fl2va", plain, dict(n_refs=0, task="fl2va", seconds=6.5)),
        ("plain-pills-style", plain_pills, dict(n_refs=0, task="t2va", seconds=10.0)),
        ("unseen-voice", unseen, dict(n_refs=0, task="t2va", seconds=5.0)),
        ("articles", articles, dict(n_refs=3)),
        ("keyframe-source", with_shots(sources={"keyframe": [0]}), dict(n_refs=3)),
        ("continue-source", with_shots(sources={"continue": 0}), dict(n_refs=3, n_vids=1)),
        ("edit-source", with_shots(sources={"edit": [0]}), dict(n_refs=3, n_vids=1)),
        ("no-scene-null", None, dict(n_refs=0, task="t2va")),
        ("no-scene-empty", {}, dict(n_refs=0, task="t2va")),
        ("no-scene-no-shots", {"cast": [], "shots": []}, dict(n_refs=0, task="t2va")),
    ]
    bad = [
        ("too-many-shots", {"shots": [{"line": "a", "beats": 1}] * 9}, {}),
        ("too-many-cast", with_shots(cast=[{"id": f"c{i}", "name": f"p{i}", "refs": []} for i in range(9)]), {}),
        ("no-name", with_shots(cast=[{"id": "x", "name": "!!!", "refs": []}]), {}),
        ("duplicate-handle", with_shots(cast=[{"id": "a", "name": "Ava", "refs": []}, {"id": "b", "name": "ava", "refs": []}]), {}),
        ("bad-retention", with_shots(cast=[{"id": "w", "name": "Ava", "retention": "mostly_ok", "refs": []}]), {}),
        ("ref-no-slot", with_shots(cast=[{"id": "w", "name": "Ava", "refs": [{"kind": "image", "index": 0, "slots": []}]}]), {}),
        ("bad-slot", with_shots(cast=[{"id": "w", "name": "Ava", "refs": [{"kind": "image", "index": 0, "slots": ["aura"]}]}]), {}),
        ("slot-media", with_shots(cast=[{"id": "w", "name": "Ava", "refs": [{"kind": "image", "index": 0, "slots": ["voice"]}]}]), {}),
        ("index-out", with_shots(cast=[{"id": "w", "name": "Ava", "refs": [{"kind": "image", "index": 7, "slots": ["face"]}]}]), {}),
        ("index-out-one", with_shots(cast=[{"id": "w", "name": "Ava", "refs": [{"kind": "video", "index": 1, "slots": ["video"]}]}]), dict(n_vids=1)),
        ("bad-video-role", with_shots(cast=[{"id": "w", "name": "Ava", "refs": [{"kind": "video", "index": 0, "slots": ["video"], "role": "star"}]}]), dict(n_vids=1)),
        ("bad-audio-role", with_shots(cast=[{"id": "w", "name": "Ava", "refs": [{"kind": "audio", "index": 0, "slots": ["audio"], "role": "sing"}]}]), dict(n_auds=1)),
        ("note-long", with_shots(cast=[{"id": "w", "name": "Ava", "refs": [{"kind": "image", "index": 0, "slots": ["image"], "note": "n" * 401}]}]), {}),
        ("sheet-audio", with_shots(cast=[{"id": "w", "name": "Ava", "refs": [{"kind": "audio", "index": 0, "slots": ["audio"], "sheet": True}]}]), dict(n_auds=1)),
        ("empty-shot", with_shots(shots=[{"line": "", "beats": 1}]), {}),
        ("long-line", with_shots(shots=[{"line": "y" * 601, "beats": 1}]), {}),
        ("dialogue-pill", with_shots(shots=[{"line": "@ava waits", "beats": 1, "pills": [{"key": "say.dialogue", "value": "hi"}]}]), {}),
        ("beats-text", with_shots(shots=[{"line": "@ava waits", "beats": "long"}]), {}),
        ("beats-zero", with_shots(shots=[{"line": "@ava waits", "beats": 0}]), {}),
        ("beats-negative", with_shots(shots=[{"line": "@ava waits", "beats": -2}]), {}),
        ("say-not-dict", with_shots(shots=[{"line": "@ava waits", "beats": 1, "say": "hello"}]), {}),
        ("say-long", with_shots(shots=[{"line": "@ava waits", "beats": 1, "say": {"who": "w", "text": "x" * 401}}]), {}),
        ("say-unknown", with_shots(shots=[{"line": "@ava waits", "beats": 1, "say": {"who": "zz", "text": "hi"}}]), {}),
        ("say-nobody", with_shots(shots=[{"line": "@ava waits", "beats": 1, "say": {"text": "hi"}}]), {}),
        ("say-lang", with_shots(shots=[{"line": "@ava waits", "beats": 1, "say": {"who": "w", "text": "hi", "lang": "Elvish"}}]), {}),
        ("sources-not-dict", with_shots(sources=[0]), {}),
        ("source-out", with_shots(sources={"keyframe": [5]}), {}),
        ("source-many", with_shots(sources={"continue": [0, 1]}), dict(n_vids=2)),
        ("source-both", with_shots(sources={"continue": [0], "edit": [0]}), dict(n_vids=1)),
        ("unknown-handle", with_shots(shots=[{"line": "@zed waits", "beats": 1}]), {}),
        ("last-carry", with_shots(shots=[{"line": "@ava waits", "beats": 1, "say": {"who": "w", "text": "hi", "carry": True}}]), {}),
        ("not-a-scene", [1, 2], {}),
        ("not-a-member", with_shots(cast=[7]), {}),
        ("not-a-ref", with_shots(cast=[{"id": "w", "name": "Ava", "refs": [7]}]), {}),
        ("not-a-shot", with_shots(shots=[7]), {}),
        ("doc-too-long", {"cast": [{"id": f"c{i}", "name": f"p{i}", "refs": [{"kind": "image", "index": i, "slots": ["image"]}]} for i in range(8)],
                          "shots": [{"line": f"@p{i} " + "y" * 580, "beats": 1} for i in range(8)]}, dict(n_refs=8)),
    ]
    out = []
    for name, sc, kw in ok + bad:
        n_refs = kw.get("n_refs", 3); n_vids = kw.get("n_vids", 0); n_auds = kw.get("n_auds", 0)
        task = kw.get("task", "ref2va"); seconds = kw.get("seconds", 8.0)
        case = {"name": f"scene/{name}", "kind": "video", "typed": "", "shot": [], "task": task,
                "seconds": seconds, "roles": [], "n_refs": n_refs, "n_vids": n_vids, "n_auds": n_auds,
                "scene": sc}
        try:
            v = G["_validate_scene"](sc, n_refs=n_refs, n_vids=n_vids, n_auds=n_auds, seconds=seconds)
            case["prompt"] = G["_compile_h3_prompt"](typed="", pills=[], task=task, seconds=seconds, scene=v)
        except ValueError as exc:
            case["error"] = str(exc)
        out.append(case)
    return out


cases += scene_cases()

# The validator on its own: what comes back, or the sentence that refuses it.
VALIDATE = {
    "unknown": [{"key": "framing.nope"}],
    "unknown-group": [{"key": "colour.red"}],
    "not-a-pill": [7],
    "duplicate": [{"key": "tone.noir"}, {"key": "tone.noir"}],
    "pick-one-replaces": [{"key": "framing.cu"}, {"key": "framing.wide"}],
    "pick-one-keeps-order": [{"key": "tone.noir"}, {"key": "framing.cu"}, {"key": "framing.wide"}],
    "solo-evicts": RAILS["score-then-silent"],
    "solo-evicted": RAILS["silent-then-score"],
    "bad-lang": [{"key": "say.dialogue", "value": "hi", "lang": "Klingon"}],
    "default-lang": [{"key": "say.dialogue", "value": "hi"}],
    "bad-amp": [{"key": "camera.panl", "amp": "huge"}],
    "bad-speed": [{"key": "camera.panl", "speed": "warp"}],
    "empty-amp": [{"key": "camera.panl", "amp": "", "speed": None}],
    "noverb-ignores-amp": RAILS["camera-noverb-amp"],
    "wrapped": {"pills": [{"key": "angle.eye"}]},
    "strings": RAILS["string-entries"],
    "value-collapsed": RAILS["whitespace-value"],
    "value-cut": RAILS["long-value"],
    "value-on-plain": [{"key": "framing.cu", "value": "ignored"}],
    "empty": [],
    "null": None,
}
validate = []
for name, raw in VALIDATE.items():
    try:
        validate.append({"name": name, "shot": raw, "pills": G["_validate_shot"](raw)})
    except ValueError as exc:
        validate.append({"name": name, "shot": raw, "error": str(exc)})

roles = []
for name, (raw, count) in {
    "none": ([], 2), "one": (["identity"], 2), "gap": (["", "wardrobe"], 3),
    "trimmed": (["identity", "wardrobe", "prop"], 2), "bad": (["face"], 1),
}.items():
    try:
        roles.append({"name": name, "roles": raw, "count": count,
                      "out": G["_validate_ref_roles"](raw, count)})
    except ValueError as exc:
        roles.append({"name": name, "roles": raw, "count": count, "error": str(exc)})

tasks = []
for first, last, refs, vids, auds in [
    (0, 0, 0, 0, 0), (1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (1, 1, 0, 0, 0),
    (0, 0, 1, 0, 0), (1, 1, 2, 0, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 1),
]:
    tasks.append({"first": first, "last": last, "refs": refs, "videos": vids, "audios": auds,
                  "task": G["_h3_task"](first, last, refs, vids, auds)})

fixture = {
    "vocab": G["SHOT_VOCAB"],
    "align": G["H3_ALIGN"],
    "languages": G["H3_LANGUAGES"],
    "roles": G["SHOT_REF_ROLES"],
    "camera_amps": list(G["CAMERA_AMPS"]),
    "camera_speeds": list(G["CAMERA_SPEEDS"]),
    "value_max": G["SHOT_VALUE_MAX"],
    "soundscape_default": G["H3_SOUNDSCAPE_DEFAULT"],
    "cases": cases,
    "validate": validate,
    "ref_roles": roles,
    "tasks": tasks,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(fixture, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
print(f"{len(cases)} compile cases, {len(validate)} validator cases → {OUT}")
