"""
Check the shot compiler against MiniMax's published prompt format.

    python3 tools/smoke_prompt.py

`smoke_graphs.py` covers the other half of the same question — is what we send
ComfyUI addressed to things that exist — and it needs a Modal container and a
ComfyUI process to answer it. This half needs neither: the compiler is pure
stdlib by design, because `/api/video` calls it on the CPU web container so a
bad pill is a form error rather than a cold H100.

What it proves, and why each one is worth an assertion:

- **The alignment lines are verbatim.** They are a contract with the checkpoint,
  not phrasing we chose, and the failure mode of getting one subtly wrong is a
  clip that silently anchors a picture to the wrong timestamp. Including the
  guide's own inconsistency — i2va and l2va bracket their labels, fl2va does
  not — which is the exact thing a well-meaning tidy-up would "fix".
- **`S.SS` is filled.** A placeholder that reaches the encoder is a placeholder
  the encoder conditions on.
- **The field labels appear once each, in the guide's order.** Both forms: the
  three-field base document and the six-field reference one.
- **Dialogue survives byte-for-byte inside `<d>…</d>`.** The guide is explicit
  that the line is preserved verbatim, punctuation included, and the compiler's
  ordinary tidying — closing an unclosed sentence, softening a full stop to a
  comma — is exactly what would corrupt it without anyone noticing.
- **No pills means no document.** The feature has to be additive: a prompt
  written before the palette existed must still reach the encoder unchanged.
- **`non_diegetic_music` is never blank.** `N/A` is the guide's own value for
  "no score", and it is the line this whole feature is worth having for.

Loaded out of app.py by AST rather than by import — see `_from_app.py`, which
both this and the UI preview server share so there is one way of doing it.
"""

from _from_app import MOTION, SHOT, pull

G = pull(SHOT | MOTION)
FAIL: list[str] = []


def check(name: str, got, want) -> None:
    if got != want:
        FAIL.append(f"{name}\n      got:  {got!r}\n      want: {want!r}")
    print(f"  {'FAIL' if got != want else ' ok '}  {name}", flush=True)


def refused(fn) -> bool:
    """Whether a bad input is named rather than quietly dropped."""
    try:
        fn()
    except ValueError:
        return True
    return False


def labels(doc: str) -> list[str]:
    """The field labels, in the order they appear, once each."""
    return [line.split(":", 1)[0] for line in doc.splitlines() if ": " in line
            and line.split(":", 1)[0].replace("_", "").isalpha()]


# --- the alignment instructions ------------------------------------------
# Quoted here independently of app.py rather than imported from it: a test that
# reads its expected value out of the thing it is testing proves the string is
# self-consistent and nothing else.
ALIGN = {
    "i2va": ("For the target video, at 0.00 seconds into the target video, "
             "<Picture 1> (from [Shot 1]) is fully referenced."),
    "fl2va": ("How the reference pictures align with the target video — "
              "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the "
              "target video; Picture 2 (from Shot N) aligns with the "
              "5.00-second mark of the target video."),
    "l2va": ("How the reference pictures align with the target video — "
             "<Picture 1> (from [Shot N]) aligns with the 5.00-second mark of "
             "the target video."),
}

print("\nalignment instructions", flush=True)
pills = G["_validate_shot"]([{"key": "framing.cu"}])
for task, want in ALIGN.items():
    doc = G["_compile_h3_prompt"](typed="She reads a letter.", pills=pills,
                                  task=task, seconds=5)
    check(f"{task} first line is verbatim", doc.splitlines()[0], want)

doc = G["_compile_h3_prompt"](typed="She reads a letter.", pills=pills,
                              task="t2va", seconds=5)
check("t2va begins at the first field", doc.splitlines()[0].split(":")[0],
      "integrated_multimodal_description")
check("S.SS is filled, not literal", "S.SS" in
      G["_compile_h3_prompt"](typed="x", pills=pills, task="l2va", seconds=12), False)
check("seconds reaches the sentence", "12.00-second" in
      G["_compile_h3_prompt"](typed="x", pills=pills, task="l2va", seconds=12), True)

# --- the two document shapes ---------------------------------------------
print("\ndocument shape", flush=True)
check("base form is the guide's three fields, in order", labels(doc),
      ["integrated_multimodal_description", "overall_soundscape",
       "non_diegetic_music"])

ref = G["_compile_h3_prompt"](typed="He walks in.", pills=pills, task="ref2va",
                              seconds=5, roles=["identity", "wardrobe"])
check("reference form is the guide's six fields, in order", labels(ref),
      ["subject_definitions", "summary", "retention_analysis",
       "detailed_description", "overall_soundscape", "non_diegetic_music"])
check("a role becomes a subject definition",
      "<Subject 1> is the person in <Picture 1>." in ref, True)
check("roles number subjects by their picture",
      "<Subject 2> is the wardrobe in <Picture 2>." in ref, True)
check("summary is what happens, not what lens",
      ref.split("summary: ")[1].splitlines()[0], "He walks in.")

# --- dialogue, byte for byte ---------------------------------------------
print("\ndialogue", flush=True)
LINE = "Take the morning with you, wherever you're going… please!"
said = G["_compile_h3_prompt"](
    typed="Two friends sit at a table.",
    pills=G["_validate_shot"]([
        {"key": "say.dialogue", "value": LINE, "lang": "English"}]),
    task="t2va", seconds=5)
check("the line survives inside <d>…</d> untouched",
      f"<d>[English] {LINE}</d>" in said, True)
check("the speaker is tagged (S1)", "<Subject 1> (S1) says:" in said, True)
check("a language outside the eleven is refused",
      refused(lambda: G["_validate_shot"](
          [{"key": "say.dialogue", "value": "hi", "lang": "Klingon"}])), True)
check("every one of the eleven is accepted",
      [l for l in G["H3_LANGUAGES"]
       if not G["_validate_shot"]([{"key": "say.dialogue", "value": "hi",
                                    "lang": l}])], [])

# --- additive by construction --------------------------------------------
print("\nadditive", flush=True)
TYPED = "She reads a letter and stops walking"
check("no pills, no roles: the typed text is what runs",
      G["_compile_h3_prompt"](typed=TYPED, pills=[], task="fl2va", seconds=5), TYPED)
check("no pills on the image side either",
      G["_compile_image_prompt"](TYPED, []), TYPED)
check("an empty valued pill compiles to nothing, not an error",
      "<d>" in G["_compile_h3_prompt"](
          typed="A room.",
          pills=G["_validate_shot"]([{"key": "say.dialogue", "value": ""}]),
          task="t2va", seconds=5), False)

# --- the silent default ---------------------------------------------------
print("\nsoundtrack", flush=True)
check("no score pill means the document says so",
      doc.splitlines()[-1], "non_diegetic_music: N/A")
scored = G["_compile_h3_prompt"](
    typed="A room.", task="t2va", seconds=5,
    pills=G["_validate_shot"]([{"key": "score.piano"}, {"key": "score.slow"}]))
check("score pills fold into one sentence", scored.splitlines()[-1],
      "non_diegetic_music: The soundtrack is solo piano, at a slow tempo.")
check("'no score' beats a score pill chosen before it",
      [p["key"] for p in G["_validate_shot"](
          [{"key": "score.piano"}, {"key": "score.silent"}])], ["score.silent"])
check("and loses to one chosen after it",
      [p["key"] for p in G["_validate_shot"](
          [{"key": "score.silent"}, {"key": "score.piano"}])], ["score.piano"])

# --- the silent families --------------------------------------------------
# The rail is shared across models, so what a silent one is handed is decided
# here and nowhere else. Dialogue is the case that breaks the obvious rule: it
# lives in the *visual* description and is still audio, so a filter by field
# would have posted `<d>[English] …</d>` to umT5, which reads angle brackets as
# angle brackets.
print("\nsilent families", flush=True)
silent = G["_validate_shot"]([
    {"key": "say.dialogue", "value": "Hello there", "lang": "English"},
    {"key": "say.screen", "value": "EXIT"},
    {"key": "framing.cu"}, {"key": "camera.pushin"},
    {"key": "sound.rain"}, {"key": "score.piano"}])
wan = G["_compile_wan_prompt"]("He walks in", silent)
check("a spoken line never reaches a silent model", "<d>" in wan, False)
check("on-screen text does — it is a picture of words", "EXIT" in wan, True)
check("neither foley nor score does",
      ("rain" in wan) or ("piano" in wan), False)
check("the camera move still does", "pushes in" in wan, True)
check("and H3 gets all of it",
      all(s in G["_compile_h3_prompt"](typed="He walks in", pills=silent,
                                       task="t2va", seconds=5)
          for s in ("<d>[English] Hello there</d>", "EXIT", "rain", "piano")), True)

# --- one move per clip ----------------------------------------------------
print("\nsingle-valued groups", flush=True)
check("a second camera pill replaces the first",
      [p["key"] for p in G["_validate_shot"](
          [{"key": "camera.pushin"}, {"key": "camera.orbit"}])], ["camera.orbit"])
check("light and tone stack",
      len(G["_validate_shot"]([{"key": "light.window"}, {"key": "light.neon"},
                               {"key": "tone.noir"}])), 3)

# --- the four tasks are told apart ---------------------------------------
print("\ntask detection", flush=True)
for args, want in [((None, None, [], []), "t2va"), (("a", None, [], []), "i2va"),
                   ((None, "b", [], []), "l2va"), (("a", "b", [], []), "fl2va"),
                   ((None, None, ["r"], []), "ref2va"),
                   (("a", "b", [], ["v"]), "ref2va")]:
    check(f"{args} -> {want}", G["_h3_task"](*args), want)

# --- the vocabulary itself ------------------------------------------------
print("\nvocabulary", flush=True)
check("every pill key is unique",
      len(G["SHOT_ITEMS"]),
      sum(len(g["items"]) for g in G["SHOT_VOCAB"]))
check("every non-valued pill has a phrase to write",
      [k for k, (_, it) in G["SHOT_ITEMS"].items()
       if not it.get("valued") and not it.get("solo") and not it["phrase"]], [])
check("every pill has a glyph class",
      [k for k, (_, it) in G["SHOT_ITEMS"].items() if not it.get("glyph")], [])
check("the image side reads no camera or audio pill",
      sorted(g["key"] for g in G["SHOT_VOCAB"] if not g["image"]),
      ["camera", "say", "score", "sound"])
check("an unknown pill key is refused, not dropped",
      refused(lambda: G["_validate_shot"]([{"key": "camera.dolly"}])), True)
check("an unknown reference role is refused",
      refused(lambda: G["_validate_ref_roles"](["stunt"], 1)), True)
check("roles are positional and padded to the reference count",
      G["_validate_ref_roles"](["identity"], 3), ["identity", "", ""])

# The one place the compiler's ordinary tidying would corrupt the user's text.
print("\npunctuation", flush=True)
check("a lowercase fragment continues the clause rather than following a stop",
      G["_compile_image_prompt"](
          "A hotel corridor. a tall woman on the left", []),
      "A hotel corridor. a tall woman on the left")
# The rule fires where clauses meet, which after the slot fix is between the
# subject and what follows it rather than ahead of the subject.
check("and it fires between a storyline's own clauses",
      G["_shot_join"](["A hotel corridor.", "a tall woman on the left."]),
      "A hotel corridor, a tall woman on the left.")
check("a trigger word's case is never changed",
      "ohwx_subject" in G["_compile_image_prompt"](
          "ohwx_subject sits by a window",
          G["_validate_shot"]([{"key": "framing.cu"}])), True)
check("a sentence that closes itself is not closed twice",
      G["_compile_image_prompt"]("She reads a letter.",
                                 G["_validate_shot"]([{"key": "framing.cu"}])),
      "She reads a letter. In a close-up.")
# Krea reads front-to-back, and the leading clause is what the subject sits in.
# Framing and angle are the frame, so they lead; light and tone modify the
# subject and cannot precede it, or the light becomes the thing described and
# the subject an apposition to it. This used to be "the first clause leads and
# the rest follow", which read correctly only because framing is first in
# vocabulary order — with framing and angle unset, light led.
check("nothing precedes the subject; light then frame follow it",
      G["_compile_image_prompt"](
          "a portrait of ohwx_subject",
          G["_validate_shot"]([{"key": "framing.mcu"}, {"key": "angle.low"},
                               {"key": "light.window"}])),
      "a portrait of ohwx_subject. Lit by soft daylight from a window. "
      "In a medium close-up, shot from a low angle.")
# The case the old split got wrong, and the reason this is structural now: no
# framing, no angle, and light must still not lead.
check("light alone still follows the subject",
      G["_compile_image_prompt"](
          "a portrait of ohwx_subject",
          G["_validate_shot"]([{"key": "light.window"}])),
      "a portrait of ohwx_subject. Lit by soft daylight from a window.")
check("tone alone still follows the subject",
      G["_compile_image_prompt"](
          "a portrait of ohwx_subject",
          G["_validate_shot"]([{"key": "tone.noir"}])),
      "a portrait of ohwx_subject. In high-contrast film noir.")

# The document is one field per line, so a newline anywhere inside it ends a
# field early and leaves the rest of the sentence looking like the start of
# another. A two-line prompt is a thing this page deliberately supports, so this
# is the ordinary case rather than a hostile one.
print("\nline structure", flush=True)
multi = G["_compile_h3_prompt"](
    typed="She reads a letter.\nThen she stops walking.",
    pills=G["_validate_shot"]([
        {"key": "framing.cu"},
        {"key": "sound.other", "value": "ice tapping\ncrystal"},
        {"key": "say.dialogue", "value": "Wait.\nPlease."}]),
    task="t2va", seconds=5)
check("a document has exactly one line per field",
      len(multi.splitlines()), 3)
check("a two-line prompt survives as one sentence",
      "She reads a letter. Then she stops walking." in multi, True)
check("and so does a pasted line break inside a valued pill",
      "<d>[English] Wait. Please.</d>" in multi, True)
check("every word and mark is still there",
      "ice tapping crystal" in multi, True)

# Two things that were shipped wrong for as long as this file has existed, and
# went unnoticed for one reason: nothing here pinned them. Both are one line in
# the guide and neither has a symptom you could see without reading the output.
print("\nwhat the guide requires and nothing asserted", flush=True)
base = G["_compile_h3_prompt"](
    typed="a woman at a window", task="t2va", seconds=5,
    pills=G["_validate_shot"](["framing.cu"]))
ref = G["_compile_h3_prompt"](typed="a woman at a window", pills=[],
                              task="ref2va", seconds=5, roles=["identity"])
check("the body opens with [Shot 1] — base-en §2.2 requires the marker",
      base.split("integrated_multimodal_description: ")[1].startswith("[Shot 1] "),
      True)
check("and so does the reference form's",
      "detailed_description: [Shot 1] " in ref, True)
# `N/A` in overall_soundscape means *complete silence, requested*. Emitting it
# because nobody picked a sound pill told every ordinary clip it was silent —
# and the guide separately warns that leaving the field blank lets ambience
# creep in, so there is no empty answer to fall back to either.
check("overall_soundscape is not N/A just because no sound pill was picked",
      "overall_soundscape: N/A" in base, False)
check("non_diegetic_music still is, which is the opposite rule and the point",
      "non_diegetic_music: N/A" in base, True)


# The one system prompt still on a user-facing path, and the budget is asserted
# rather than left to judgement for the reason PARSE_RULES records: it grew from
# 2.9k to 10.2k characters one well-reasoned rule at a time and the output got
# worse, because concrete examples come back parroted as the answer. This check
# lived in the enhance harness and moved here when that was deleted with the
# feature — a budget nobody asserts is a paragraph.
for _img in (True, False):
    for _aud in (True, False):
        _n = len(G["_motion_instruction"](image=_img, audio=_aud))
        check(f"the motion instruction fits its budget "
              f"(image={_img}, audio={_aud}): {_n} chars",
              500 <= _n <= 2000, True)

if FAIL:
    print("\n" + "\n".join("  " + f for f in FAIL), flush=True)
    raise SystemExit(f"\n{len(FAIL)} disagreement(s) with the published format.")
print("\nThe compiler writes the format MiniMax published.", flush=True)
