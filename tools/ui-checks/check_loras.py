"""
LoRAs are chips, and the composer is per-kind.

    /opt/homebrew/bin/python3.11 tools/preview_ui.py 8799 &
    python3.11 tools/ui-checks/check_loras.py http://localhost:8799

Every row here is a rule that used to be carried by `<lora:…>` in the prompt and
is now carried by structure instead — see `docs/design-notes/loras-are-not-text.md`.

  * **The box costs nothing at rest.** No LoRAs, no element. `#shot-rail:empty`'s
    rule: a row is affordable when it carries content, never when it carries one
    control.
  * **The count is a word.** The regions button failed here once — a count riding
    half-outside it read as an error pip rather than as "2 regions".
  * **Nothing writes into the prompt.** Picking used to insert a token, a
    strength and the trigger phrase — three edits to your sentence from one
    press. The box must be untouched.
  * **The composer is per-kind.** A Krea 2 LoRA is not an H3 LoRA. Switching used
    to carry one across and load it into a run that could not use it, silently.
"""
import sys
import time

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8791"
fails = []


def box(page):
    return page.eval_on_selector("#prompt", "el => el.value")


def pick(page, n=0):
    page.click("#add-lora")
    page.wait_for_selector(".menu button")
    page.locator(".menu button").nth(n).click()
    time.sleep(0.15)


with sync_playwright() as pw:
    b = pw.chromium.launch()
    page = b.new_page(viewport={"width": 1512, "height": 982})
    page.goto(URL, wait_until="networkidle")
    page.fill("#prompt", "a portrait in soft window light")

    if page.locator("#lora-box").count():
        fails.append("the box is drawn with no LoRAs on the canvas")

    pick(page, 0)
    if box(page) != "a portrait in soft window light":
        fails.append(f"picking wrote into the prompt: {box(page)!r}")
    if page.locator("#lora-box").count() != 1:
        fails.append("the box did not appear after a pick")
    summary = page.locator("#lora-box > button").inner_text().strip()
    if summary != "1 LoRA":
        fails.append(f"summary read {summary!r}, wanted '1 LoRA'")

    pick(page, 1)
    summary = page.locator("#lora-box > button").inner_text().strip()
    if summary != "2 LoRAs":
        fails.append(f"summary read {summary!r} with two picked")

    # Collapsed by default; the chips are what the disclosure holds.
    if page.locator("#lora-box .chip").count():
        fails.append("the chips are drawn while the box is shut")
    page.click("#lora-box > button")
    time.sleep(0.15)
    if page.locator("#lora-box .chip").count() != 2:
        fails.append("opening the box did not show two chips")

    # Picking one already picked takes it out — the tick is a state.
    pick(page, 0)
    if page.locator("#lora-box .chip").count() != 1:
        fails.append("picking a ticked row did not remove it")

    # ---- the switch -------------------------------------------------------
    def cross(want_video):
        """
        **Crossing consoles is a choice of model, not of duration.**

        This drove the Duration menu — `duration("5s")` to reach video and
        `duration("Still")` to come back — which was right while a still meant
        Krea 2. It is not any more: H3 makes a still at zero seconds, so both
        consoles answer that length and `Still` deliberately changes no engine.
        Left as it was, the round trip below never left H3, `#lora-box` never
        mounted, and the check died waiting thirty seconds for a selector that
        cannot appear — the same staleness `check_drop.py` carried, and this is
        its `cross()` verbatim so the two cannot drift into two spellings of one
        gesture. See `Duration.tsx` and `EngineRow` in `SamplingButton.tsx`.

        The button clicked is the *current* console's, which is the opposite of
        where we are going. Values are `kind:key` because the picker spans both
        families and a bare key could name either one.
        """
        if page.eval_on_selector(
            "#c-video", "e => e.classList.contains('hide')"
        ) != want_video:
            return
        page.click("#g-sampling" if want_video else "#v-sampling")
        page.wait_for_selector(".menu.form select")
        page.select_option(".menu.form select",
                           "video:h3" if want_video else "image:turbo")
        time.sleep(0.5)

    # **This block dies with video's prompt box, and what it is testing does
    # not.** Three lines below depend on `#prompt` existing on the video side:
    # the empty-check, the `fill`, and the come-back check that needs the video
    # side dirtied to prove the two buffers are separate.
    #
    # The claim underneath is "one live set and one dormant, swapped by kind,
    # neither leaking" — which survives the scene composer whole; video's buffer
    # just stops being one box and becomes the first shot row. So this is a
    # **re-point, not a delete**: aim the three at whatever video's input is
    # then, or the LoRA side loses a real guard for a reason that has nothing to
    # do with LoRAs. Deleting it would read as tidying up after a UI change and
    # would quietly drop the only check that the per-kind buffers do not bleed.
    cross(True)
    if page.locator("#lora-box").count():
        fails.append("an image LoRA followed the switch to video")
    if box(page) != "":
        fails.append(f"the image prompt followed the switch: {box(page)!r}")
    page.fill("#prompt", "a slow push in")
    cross(False)
    if box(page) != "a portrait in soft window light":
        fails.append(f"the image prompt did not come back: {box(page)!r}")
    if page.locator("#lora-box > button").inner_text().strip() != "1 LoRA":
        fails.append("the image chips did not come back")

    # ---- what the note still says ----------------------------------------
    #
    # Three of its lines are gone because they became impossible: a chip is
    # picked from a list, so no name resolves to nothing and none resolves to
    # two. A fourth — a LoRA whose trigger phrase is missing from the prose — is
    # gone because it is managed by hand. `_retired_probe_lora.py` is the record.
    #
    # What survives is the two it cannot see: a stack past the cap, and the same
    # LoRA on the canvas *and* in a box, which puts the canvas copy on the global
    # chain and cancels the masking it looks like it is doing.
    # `#lora-note` and not `#console-notes`: the container holds two spans and is
    # laid out to reserve its row whether or not either of them says anything, so
    # reading the container back gives '' even while the note is on screen.
    def note():
        el = page.query_selector("#lora-note")
        return (el.inner_html() if el else "").strip()

    # Index 1 is already on from the toggle above, so it is skipped — picking a
    # *ticked* row takes it off, which would keep the total under the cap and
    # make this pass for the wrong reason. Seven more takes it to eight.
    for n in (0, 2, 3, 4, 5, 6, 7):
        pick(page, n)
    n = note()
    if "6" not in n:
        fails.append(f"a stack past the cap is not reported: {n!r}")

    b.close()

for f in fails:
    print(f"  FAIL  {f}")
print(f"\n{'PASS' if not fails else str(len(fails)) + ' FAILED'}")
raise SystemExit(1 if fails else 0)
