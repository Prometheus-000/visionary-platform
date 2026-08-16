"""
Do the known-good prompts come back byte-identical out of a storyline?

This is the only check that says whether the module compiler is a compiler or a
reimplementation. Eleven prompts are known to adhere — three Gucci recreations
the model produced without ever seeing the originals, and eight published with
the checkpoint — and every design decision in `docs/krea2-prompt-template.md`
was derived from reading them. If a storyline assembled out of their own clauses
does not reproduce them exactly, the module layer has changed what the encoder
is told, and every finding derived from those eleven stops applying to anything
this app now emits.

Byte-identical, not "close enough". The separator in front of a clause is the
one thing the compiler is allowed to choose, and it is exactly the thing that
would drift silently: a full stop where a comma belongs reads fine and is a
different input to the encoder.

`_from_app.py` pulls the real functions rather than a copy, for the reason that
file exists — a compiler checked against a reimplementation checks the
reimplementation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _from_app import pull  # noqa: E402

G = pull({
    "SHOT_VOCAB", "MODULE_ROLES", "MAX_MODULES", "MODULE_TEXT_MAX",
    "MAX_MODULE_DEPTH", "_module_clause", "_module_words",
    "_shot_phrases", "_shot_text", "_shot_sentence", "_shot_join", "_shot_body",
    "_close", "_oneline", "_flat", "_compile_image_prompt", "_validate_modules",
    "_spans_to_text", "MAX_SPANS",
    "_module_texts", "_prominence",
})

fails: list[tuple[str, str, str]] = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}", flush=True)
    if not ok:
        fails.append((name, str(got), str(want)))


def rebuild(name: str, modules: list[str], want: str) -> None:
    """A storyline of plain clauses, compiled with no pills, must equal `want`."""
    mods = G["_validate_modules"]([{"role": "text", "text": m} for m in modules])
    check(name, G["_compile_image_prompt"]("", [], mods), want)


# ── the three Gucci recreations ─────────────────────────────────────────────
print("\ngolden prompts, rebuilt from their own clauses", flush=True)

rebuild("corridor — three figures, one collapsed pair", [
    "Three figures standing in a hotel corridor with pale blue floral wallpaper, "
    "honey-coloured wooden door frames and a deep blue carpet running away from "
    "the camera.",
    "On the left stands a tall woman in a purple beret, a sheer black lace "
    "long-sleeved top, a wide studded black belt and a teal suede skirt, with "
    "tall tan leather boots; one hand holds a patterned handbag at her hip.",
    "Beside her, to her right, stand two small girls of about eight, dressed "
    "identically in pale blue short-sleeved dresses with white collars and "
    "ribbon belts, white knee socks and black Mary Jane shoes, holding hands "
    "and standing shoulder to shoulder.",
    "All three face the camera directly, expressionless.",
    "Even shadowless light with no visible source.",
    "Shot straight on at chest height, rigidly symmetrical, sharp from front to "
    "back.",
], "Three figures standing in a hotel corridor with pale blue floral wallpaper, "
   "honey-coloured wooden door frames and a deep blue carpet running away from "
   "the camera. On the left stands a tall woman in a purple beret, a sheer "
   "black lace long-sleeved top, a wide studded black belt and a teal suede "
   "skirt, with tall tan leather boots; one hand holds a patterned handbag at "
   "her hip. Beside her, to her right, stand two small girls of about eight, "
   "dressed identically in pale blue short-sleeved dresses with white collars "
   "and ribbon belts, white knee socks and black Mary Jane shoes, holding hands "
   "and standing shoulder to shoulder. All three face the camera directly, "
   "expressionless. Even shadowless light with no visible source. Shot straight "
   "on at chest height, rigidly symmetrical, sharp from front to back.")

rebuild("bathroom — three individuated, one overlapping", [
    "Three young people crowded into a small 1970s public bathroom with glossy "
    "pink square tiles on the walls and geometric amber and cream wallpaper "
    "above the tile line.",
    "On the left, a red-haired young woman leans her forearm against the wall "
    "beside a white ceramic hand dryer and rests her temple on her fingers with "
    "her eyes lowered; she wears an emerald green satin bomber jacket heavily "
    "embroidered with flowers over a pale pink pussy-bow blouse, and a long "
    "printed pleated skirt.",
    "In the centre a slim young man with dark curly hair stands facing the "
    "camera, his right arm raised against the wall above her head; he wears a "
    "sheer teal lace shirt embroidered with small birds, a brown leather belt, "
    "and grey flared jeans.",
    "On the right a blonde woman in oversized tortoiseshell glasses presses "
    "close behind him with both arms wrapped around his waist, wearing a dark "
    "brown fur coat appliqued with bright green leaves.",
    "Behind them a long red counter holds an orange basin beneath a wide mirror, "
    "lit by a fluorescent tube and two white globe pendant lamps.",
    "Shot on medium format, slightly wide, the whole room in focus.",
], "Three young people crowded into a small 1970s public bathroom with glossy "
   "pink square tiles on the walls and geometric amber and cream wallpaper "
   "above the tile line. On the left, a red-haired young woman leans her "
   "forearm against the wall beside a white ceramic hand dryer and rests her "
   "temple on her fingers with her eyes lowered; she wears an emerald green "
   "satin bomber jacket heavily embroidered with flowers over a pale pink "
   "pussy-bow blouse, and a long printed pleated skirt. In the centre a slim "
   "young man with dark curly hair stands facing the camera, his right arm "
   "raised against the wall above her head; he wears a sheer teal lace shirt "
   "embroidered with small birds, a brown leather belt, and grey flared jeans. "
   "On the right a blonde woman in oversized tortoiseshell glasses presses "
   "close behind him with both arms wrapped around his waist, wearing a dark "
   "brown fur coat appliqued with bright green leaves. Behind them a long red "
   "counter holds an orange basin beneath a wide mirror, lit by a fluorescent "
   "tube and two white globe pendant lamps. Shot on medium format, slightly "
   "wide, the whole room in focus.")

rebuild("bench — two copular subjects and a secondary", [
    "Two young people sitting side by side on a slatted green wooden park bench "
    "in front of a weathered Roman brick ruin, with umbrella pines and dry grass "
    "behind them under a bright blue sky.",
    "The person on the left is a young man in a black tailored blazer heavily "
    "embroidered with pink and blue flowers, a dusty red silk shirt and salmon "
    "pink wide-leg trousers with pink embroidered loafers.",
    "The person on the right is a young woman in a cream cable-knit cardigan "
    "with a red and blue vertical stripe down the front, red trousers and red "
    "patent boots, one leg drawn up onto the bench.",
    "Both look toward the camera, relaxed and unsmiling.",
    "A full-grown tiger walks slowly across the foreground from the left with "
    "its head lowered, passing in front of the bench and partly cropped by the "
    "frame edge.",
    "Hard midday sun from the left throwing short crisp shadows.",
    "Shot on medium format at eye level.",
], "Two young people sitting side by side on a slatted green wooden park bench "
   "in front of a weathered Roman brick ruin, with umbrella pines and dry grass "
   "behind them under a bright blue sky. The person on the left is a young man "
   "in a black tailored blazer heavily embroidered with pink and blue flowers, "
   "a dusty red silk shirt and salmon pink wide-leg trousers with pink "
   "embroidered loafers. The person on the right is a young woman in a cream "
   "cable-knit cardigan with a red and blue vertical stripe down the front, red "
   "trousers and red patent boots, one leg drawn up onto the bench. Both look "
   "toward the camera, relaxed and unsmiling. A full-grown tiger walks slowly "
   "across the foreground from the left with its head lowered, passing in front "
   "of the bench and partly cropped by the frame edge. Hard midday sun from the "
   "left throwing short crisp shadows. Shot on medium format at eye level.")

# ── published with the checkpoint ───────────────────────────────────────────
rebuild("macro eye — one clause, subject is the frame", [
    "An extreme close-up portrait featuring pale, freckled skin and a single "
    "blue eye wrapped in reflective metallic gold ribbons.",
    "Thin gold strips crisscross diagonally over the cheek and forehead, casting "
    "sharp, hard shadows onto the face.",
    "Strands of copper hair frame the top edge while the left ear softly blurs "
    "out of focus.",
    "Harsh, direct lighting highlights intricate skin pores and bright golden "
    "reflections, isolating the brightly lit features against a pitch-black "
    "background in a bold, high-contrast macro editorial style.",
], "An extreme close-up portrait featuring pale, freckled skin and a single "
   "blue eye wrapped in reflective metallic gold ribbons. Thin gold strips "
   "crisscross diagonally over the cheek and forehead, casting sharp, hard "
   "shadows onto the face. Strands of copper hair frame the top edge while the "
   "left ear softly blurs out of focus. Harsh, direct lighting highlights "
   "intricate skin pores and bright golden reflections, isolating the brightly "
   "lit features against a pitch-black background in a bold, high-contrast "
   "macro editorial style.")

rebuild("portrait — subject, secondary, background, light", [
    "A close-up portrait of a young East Asian woman with straight black hair, "
    "loose strands sweeping across her fair skin, and an intense gaze.",
    "She wears a light grey collared shirt with a black tie.",
    "A vibrant bouquet of pink and orange lilies with lush green leaves sits in "
    "the blurred right foreground.",
    "The background is a solid, striking crimson red.",
    "Soft, directional studio lighting highlights her facial features, creating "
    "a high-contrast composition with a shallow depth of field.",
], "A close-up portrait of a young East Asian woman with straight black hair, "
   "loose strands sweeping across her fair skin, and an intense gaze. She wears "
   "a light grey collared shirt with a black tie. A vibrant bouquet of pink and "
   "orange lilies with lush green leaves sits in the blurred right foreground. "
   "The background is a solid, striking crimson red. Soft, directional studio "
   "lighting highlights her facial features, creating a high-contrast "
   "composition with a shallow depth of field.")

rebuild("forest — a coordinate pair with sub-subjects", [
    "An anime illustration depicts a young boy and girl walking through a lush "
    "forest.",
    "The boy, on the left, wears a white short-sleeved shirt, a dark tie, and a "
    "blue cap.",
    "He has short brown hair and looks to his right with a curious expression.",
    "The girl, on the right, wears a white dress with a blue collar and cuffs, "
    "and her brown hair is tied back.",
    "She carries a woven basket over her right shoulder and also looks to her "
    "right with an inquisitive gaze.",
    "The forest background is filled with green foliage and trees, with sunlight "
    "filtering through the leaves.",
    "Large rocks are scattered in the foreground, with a small brown bird "
    "perched on a rock to the left.",
    "To the right, a small brown monkey is visible climbing a tree.",
    "Red and yellow flowers add pops of color to the scene.",
    "The overall style is characteristic of traditional hand-drawn animation, "
    "with soft lighting and a natural color palette.",
], "An anime illustration depicts a young boy and girl walking through a lush "
   "forest. The boy, on the left, wears a white short-sleeved shirt, a dark "
   "tie, and a blue cap. He has short brown hair and looks to his right with a "
   "curious expression. The girl, on the right, wears a white dress with a blue "
   "collar and cuffs, and her brown hair is tied back. She carries a woven "
   "basket over her right shoulder and also looks to her right with an "
   "inquisitive gaze. The forest background is filled with green foliage and "
   "trees, with sunlight filtering through the leaves. Large rocks are "
   "scattered in the foreground, with a small brown bird perched on a rock to "
   "the left. To the right, a small brown monkey is visible climbing a tree. "
   "Red and yellow flowers add pops of color to the scene. The overall style is "
   "characteristic of traditional hand-drawn animation, with soft lighting and "
   "a natural color palette.")

rebuild("sailor girl — one subject, painted register", [
    "A dynamic digital painting of a joyful girl in a sailor uniform stretching "
    "her arms high against a solid vibrant blue background.",
    "She has short dark windblown hair, amber eyes, and a bright smile.",
    "She wears a white shirt, striped blue collar, flowing red neckerchief, and "
    "a billowing blue pleated skirt.",
    "Expressive thick brushstrokes and bold shading emphasize energetic motion.",
], "A dynamic digital painting of a joyful girl in a sailor uniform stretching "
   "her arms high against a solid vibrant blue background. She has short dark "
   "windblown hair, amber eyes, and a bright smile. She wears a white shirt, "
   "striped blue collar, flowing red neckerchief, and a billowing blue pleated "
   "skirt. Expressive thick brushstrokes and bold shading emphasize energetic "
   "motion.")

rebuild("jester — zero-clause, object as subject", [
    "Stylized digital painting of a menacing jester figure rendered with bold, "
    "expressive brushstrokes and a vibrant, almost psychedelic color palette "
    "against a pitch-black background.",
    "Dynamic low-angle perspective forces a dramatic, imposing composition as "
    "the character leans forward, one leg raised high.",
    "The jester wears a classic multi-pointed hat with bells, a ruffled collar, "
    "puffed sleeves, harlequin-patterned shorts in muted gold and dark brown, "
    "and striped tights in alternating shades of purple, blue, and chartreuse.",
    "A heavily textured, flowing cape billows outward to the left, decorated "
    "with abstract, fluid patterns of saturated purples, greens, and iridescent "
    "hues resembling oil slicks or marbled paper.",
    "The figure's face is completely obscured, appearing as a smooth, faceless, "
    "pale mauve mask with a single, glowing bright white point of light in the "
    "center.",
    "In its right hand, clad in a grey-blue gauntlet, the jester grips a "
    "massive, ornate sword with a wide, glowing, ethereal white blade, its "
    "crossguard intricately sculpted.",
    "Lighting is dramatic and theatrical, casting strong shadows and "
    "highlighting the painterly texture, giving the artwork a dark fantasy, "
    "surreal aesthetic reminiscent of concept art.",
], "Stylized digital painting of a menacing jester figure rendered with bold, "
   "expressive brushstrokes and a vibrant, almost psychedelic color palette "
   "against a pitch-black background. Dynamic low-angle perspective forces a "
   "dramatic, imposing composition as the character leans forward, one leg "
   "raised high. The jester wears a classic multi-pointed hat with bells, a "
   "ruffled collar, puffed sleeves, harlequin-patterned shorts in muted gold "
   "and dark brown, and striped tights in alternating shades of purple, blue, "
   "and chartreuse. A heavily textured, flowing cape billows outward to the "
   "left, decorated with abstract, fluid patterns of saturated purples, greens, "
   "and iridescent hues resembling oil slicks or marbled paper. The figure's "
   "face is completely obscured, appearing as a smooth, faceless, pale mauve "
   "mask with a single, glowing bright white point of light in the center. In "
   "its right hand, clad in a grey-blue gauntlet, the jester grips a massive, "
   "ornate sword with a wide, glowing, ethereal white blade, its crossguard "
   "intricately sculpted. Lighting is dramatic and theatrical, casting strong "
   "shadows and highlighting the painterly texture, giving the artwork a dark "
   "fantasy, surreal aesthetic reminiscent of concept art.")

# ── the properties the storyline depends on ─────────────────────────────────
print("\nwhat the storyline needs to be true", flush=True)

# Order is placement. A compiler that sorted, deduplicated or normalised the
# body would move people around the frame, and nothing on screen would say why.
mods = G["_validate_modules"]([
    {"role": "subject", "text": "On the left, a red-haired woman."},
    {"role": "subject", "text": "In the centre, a dark-haired man."},
    {"role": "subject", "text": "On the right, a blonde woman."},
])
check("subject order survives compilation exactly",
      G["_compile_image_prompt"]("", [], mods),
      "On the left, a red-haired woman. In the centre, a dark-haired man. "
      "On the right, a blonde woman.")

check("reordering swaps who is where, and changes nothing else",
      G["_compile_image_prompt"]("", [], list(reversed(mods))),
      "On the right, a blonde woman. In the centre, a dark-haired man. "
      "On the left, a red-haired woman.")

# The whole no-regression claim: one module is a typed prompt.
check("a one-module storyline equals the typed prompt it came from",
      G["_compile_image_prompt"]("", [], G["_validate_modules"](
          [{"role": "text", "text": "a portrait of k3nan"}])),
      G["_compile_image_prompt"]("a portrait of k3nan", []) + ".")

# Pills fold around the storyline at the slots they already occupied, and light
# still cannot lead — the defect this session fixed must not return by a new door.
# The separator softens to a comma only before a *lowercase* continuation, which
# is the whole point of that rule — it exists so "A medium close-up. a portrait
# of k3nan." cannot happen without upper-casing somebody's trigger word. A module
# is a written sentence and keeps its capital, so the stop stays a stop.
check("nothing precedes a storyline; pills all follow it",
      G["_compile_image_prompt"]("", [{"key": "framing.mcu"},
                                      {"key": "light.window"}], mods),
      "On the left, a red-haired woman. In the centre, a dark-haired man. "
      "On the right, a blonde woman. Lit by soft daylight from a window. "
      "In a medium close-up.")

check("a lowercase module keeps its case and leads regardless",
      G["_compile_image_prompt"]("", [{"key": "framing.mcu"}], G["_validate_modules"](
          [{"role": "text", "text": "a portrait of k3nan"}])),
      "a portrait of k3nan. In a medium close-up.")

check("light alone still cannot lead a storyline",
      G["_compile_image_prompt"]("", [{"key": "light.window"}], mods[:1]),
      "On the left, a red-haired woman. Lit by soft daylight from a window.")

# Prominence is extent, and it belongs to the module rather than the slot.
share = G["_prominence"](mods)
check("shares sum to one at display precision",
      round(sum(s["share"] for s in share), 2), 1.0)
check("equal extent gives equal share", share[0]["share"], share[2]["share"])

longer = G["_validate_modules"]([
    {"role": "subject", "text": "On the left, a red-haired woman in an emerald "
                                "satin bomber embroidered with flowers over a "
                                "pale pink pussy-bow blouse."},
    {"role": "subject", "text": "In the centre, a dark-haired man."},
    {"role": "subject", "text": "On the right, a blonde woman."},
])
check("saying more raises that module's share",
      G["_prominence"](longer)[0]["share"] > share[0]["share"], True)

# Heat travels with the element: rearranging conserves every share.
check("rearranging conserves each module's share",
      sorted(s["share"] for s in G["_prominence"](list(reversed(longer)))),
      sorted(s["share"] for s in G["_prominence"](longer)))

# ── malformed input is refused by name, never dropped ───────────────────────
print("\nrefusals", flush=True)
for bad, why in [([{"role": "nonsense", "text": "x"}], "an unknown role"),
                 ([{"role": "text", "text": "x", "origin": "guessed"}],
                  "an unknown origin")]:
    try:
        G["_validate_modules"](bad)
        check(f"{why} is refused", "accepted", "refused")
    except ValueError:
        check(f"{why} is refused, not dropped", "refused", "refused")

check("an empty module compiles to nothing rather than an error",
      G["_validate_modules"]([{"role": "subject", "text": "  "},
                              {"role": "text", "text": "a portrait"}]),
      [{"role": "text", "text": "a portrait", "origin": "derived"}])

# ── spans: which words in a clause are the person's ─────────────────────────
#
# **The failure this is here to catch does not look like a failure.** A mark
# three characters left of where it belongs is still a mark — dotted underline,
# right shape, wrong words — and it asserts that somebody wrote something they
# did not. Nothing on screen suggests it should be doubted, and the only way to
# notice is to read the sentence and already know the answer. So every case
# below checks the *text under the offsets*, never the offsets alone.
print("\nspans", flush=True)

MIXED = [{"text": "Everything is covered in ", "origin": "derived"},
         {"text": "white dust", "origin": "invented"},
         {"text": ". A ruined city.", "origin": "derived"}]


def marked(mods: list[dict]) -> list[str]:
    """The words each mark actually covers, which is the only thing that matters."""
    return [mods[0]["text"][a:b] for a, b in mods[0].get("invented") or []]


text, marks = G["_spans_to_text"](MIXED)
check("runs join back to the clause with nothing added or dropped",
      text, "Everything is covered in white dust. A ruined city.")
check("the mark lands on the invented words",
      [text[a:b] for a, b in marks], ["white dust"])

# A run owns the space in front of it or the one behind it, never both.
check("a doubled space at a boundary does not shift the offsets",
      (lambda t, m: (t, [t[a:b] for a, b in m]))(
          *G["_spans_to_text"]([{"text": "a ", "origin": "derived"},
                                {"text": " b", "origin": "invented"}])),
      ("a b", ["b"]))
check("the clause owns its own edges, not the first and last run",
      (lambda t, m: (t, [t[a:b] for a, b in m]))(
          *G["_spans_to_text"]([{"text": "  lead", "origin": "invented"},
                                {"text": " tail  ", "origin": "derived"}])),
      ("lead tail", ["lead"]))
check("a newline inside a run collapses without welding two words together",
      G["_spans_to_text"]([{"text": "one\n\n", "origin": "derived"},
                           {"text": "two", "origin": "derived"}])[0],
      "one two")
check("two invented runs that touch are one underline, not two",
      (lambda t, m: (len(m), [t[a:b] for a, b in m]))(
          *G["_spans_to_text"]([{"text": "a ", "origin": "derived"},
                                {"text": "ruined", "origin": "invented"},
                                {"text": " city", "origin": "invented"}])),
      (1, ["ruined city"]))

check("through the validator, the marks still cover the invented words",
      marked(G["_validate_modules"]([{
          "role": "text", "text": "Everything is covered in white dust. A ruined city.",
          "origin": "derived", "spans": MIXED}])),
      ["white dust"])

# Dropped whole rather than repaired: no marks reads as "all theirs", which is
# what the clause meant before spans existed. A guessed repair reads as a fact.
check("runs that do not agree with the clause are dropped, not repaired",
      "invented" in G["_validate_modules"]([{
          "role": "text", "text": "a ruined city", "origin": "derived",
          "spans": [{"text": "a ruined", "origin": "derived"},
                    {"text": "metropolis", "origin": "invented"}]}])[0],
      False)
check("...and the clause itself survives that",
      G["_validate_modules"]([{
          "role": "text", "text": "a ruined city", "origin": "derived",
          "spans": [{"text": "a ruined", "origin": "derived"},
                    {"text": "metropolis", "origin": "invented"}]}])[0]["text"],
      "a ruined city")

# One field answers "which of these words are mine" at both granularities, so
# the page never has to consult `origin` as well.
check("an element invented whole needs no runs to be marked whole",
      marked(G["_validate_modules"]([{"role": "light", "text": "hard side light",
                                      "origin": "invented"}])),
      ["hard side light"])
check("a derived element with no runs carries no marks at all",
      "invented" in G["_validate_modules"]([{"role": "text", "text": "a woman",
                                             "origin": "derived"}])[0],
      False)
check("a child can be marked too",
      G["_validate_modules"]([{"role": "subject", "text": "a woman", "origin": "derived",
                               "children": [{"role": "text", "text": "in a purple beret",
                                             "origin": "invented"}]}])[0]["children"][0]
      .get("invented"),
      [[0, 17]])

# ── report ──────────────────────────────────────────────────────────────────
print()
if fails:
    for name, got, want in fails:
        print(f"  {name}\n      got:  {got!r}\n      want: {want!r}\n")
    print(f"{len(fails)} disagreement(s) with the known-good prompts.")
    raise SystemExit(1)
print("Every known-good prompt survives the storyline.")
