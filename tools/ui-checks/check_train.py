"""
Train and the dataset editor, asserted against either front end.

    python3 tools/ui-checks/check_train.py                        # vanilla
    python3 tools/ui-checks/check_train.py http://localhost:5173  # React

Two surfaces, one door. What is worth pinning here is not the layout — the
baseline screenshots cover that — but the handful of rules that decide whether a
training run is reproducible or wasted:

  * **Every numeric field carries its name.** This is the one place the "a
    control that shows its own value gets no label" rule was pushed past what it
    can carry: "32" is a rank, an alpha, an epoch count or a seed with equal
    plausibility, and someone who has trained these models for five years had to
    hover every field to find out which. A bare number is not a value.
  * **A draft and a saved set are the same thing in two parents.** Both caption,
    filter and train identically, so the editor must offer the same controls for
    either; the only difference is that a draft can be saved and a saved set
    cannot be saved again.
  * **The captioner is two menus, not one.** A preset decides what to leave
    out; the model decides whether a photograph of a real person gets described
    at all. Both come from the server so the labels and the instruction behind
    them cannot disagree.
"""
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8791"

# name -> the label it must show. Matching app.py's ids exactly, because these
# are what every other tool addresses these fields by.
RESTING = {"a-dim": "Rank", "a-epochs": "Epochs", "a-lr": "Learning rate"}
ADVANCED = {"a-alpha": "Alpha", "a-res": "Resolution", "a-rep": "Repeats",
            "a-bs": "Batch size", "a-seed": "Seed"}

fails: list[str] = []


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(f"{label} {detail}".strip())


LABEL_OF = """
(id) => {
  const el = document.getElementById(id);
  if (!el) return null;
  const box = el.closest('[data-lb]');
  // Either the wrapper's data-lb or a rendered .lead — both are the name, and
  // which one carries it is a styling decision rather than a contract.
  return box?.dataset.lb || box?.querySelector('.lead')?.textContent?.trim() || null;
}
"""

with sync_playwright() as pw:
    b = pw.chromium.launch(channel="chrome")
    pg = b.new_page(viewport={"width": 1512, "height": 982}, color_scheme="dark")
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(URL, wait_until="networkidle", timeout=60_000)
    pg.wait_for_timeout(1300)

    print(f"\n=== {URL} ===")

    # Train is one door, labelled with where it leads rather than where you are.
    pg.click("#door")
    pg.wait_for_selector("#v-train", timeout=10_000)
    pg.wait_for_timeout(900)
    check("the door reaches Train", pg.is_visible("#t-console"))

    for i, want in RESTING.items():
        check(f"{i} carries its name", pg.evaluate(LABEL_OF, i) == want,
              str(pg.evaluate(LABEL_OF, i)))

    # The hyperparameters worth changing rarely are behind the sliders button,
    # which keeps its word because the glyph is worn by two unrelated panels.
    pg.click("#t-toggle-adv")
    pg.wait_for_timeout(400)
    for i, want in ADVANCED.items():
        check(f"{i} carries its name", pg.evaluate(LABEL_OF, i) == want,
              str(pg.evaluate(LABEL_OF, i)))

    # A run needs a name and a set; the button says so by staying disabled.
    check("Start training waits for a name and a set",
          pg.evaluate("() => !!document.querySelector('#go-train')?.disabled"))

    # ---- the dataset editor ------------------------------------------------
    rows = pg.evaluate("() => document.querySelectorAll('#ds-list [data-ds], #ds-list .ds-row').length")
    check("the rail lists sets", rows > 0, f"{rows} rows")

    # Drafts sit above saved sets under their own heading, because "unsaved,
    # cleared when you close the app" is the whole difference between them.
    heads = pg.evaluate("""() => [...document.querySelectorAll('#ds-list *')]
      .map(e => (e.textContent || '').trim().toUpperCase())
      .filter(t => t === 'SAVED' || t.startsWith('UNSAVED'))""")
    check("drafts and saved sets are separated",
          any(h.startswith("UNSAVED") for h in heads) and any(h == "SAVED" for h in heads),
          str(sorted(set(heads))[:3]))

    pg.click("#ds-list >> text=studio_portraits")
    pg.wait_for_timeout(1300)
    check("opening a set names it",
          "studio_portraits" in (pg.text_content("#ds-title") or ""),
          repr(pg.text_content("#ds-title")))
    check("and counts it", "image" in (pg.text_content("#ds-count") or ""),
          repr(pg.text_content("#ds-count")))

    for i in ("f-all", "f-uncap", "f-notrig", "dens-down", "dens-up"):
        check(f"#{i} is addressable", pg.query_selector(f"#{i}") is not None)

    # The captioner. Behind the panel in both, rendered-and-hidden in one and
    # not rendered in the other — which is the same thing to a person.
    pg.click("#ins-toggle")
    pg.wait_for_timeout(700)
    presets = pg.evaluate("() => [...document.querySelectorAll('#cap-preset option')].map(o => o.value)")
    models = pg.evaluate("() => [...document.querySelectorAll('#cap-model option')].map(o => o.value)")
    check("the preset menu is served, not written into the page", len(presets) >= 3, str(presets))
    # A stock instruct model declines to describe photographs of real people
    # often enough to matter, and on a character set that is every image — so the
    # abliterated repackage is the second entry rather than a second code path.
    check("a second captioner is offered", len(models) >= 2, str(models))

    # ---- a run, start to finish --------------------------------------------
    # Asserted by shape and by terminal state, never by the value at an instant:
    # the stub advances per poll rather than per second, so two front ends
    # sampled at the same moment are legitimately on different steps.
    pg.fill("#lname", "probe_lora")
    pg.fill("#ltrig", "ohwx")
    pg.wait_for_timeout(300)
    check("naming it and picking a set is enough to start",
          not pg.evaluate("() => !!document.querySelector('#go-train')?.disabled"))

    pg.click("#go-train")
    pg.wait_for_selector("#step-run", timeout=15_000)
    pg.wait_for_function(
        "() => /step \\d+\\/\\d+/.test(document.querySelector('#run-meta')?.textContent || '')",
        timeout=30_000)

    meta = pg.text_content("#run-meta") or ""
    # A bar alone cannot tell "training" from "stuck". Every one of these is a
    # number that has to move for the run to be legible, and the loss is the one
    # that says whether the hours are buying anything.
    for want in ("step", "epoch", "it/s", "ETA", "loss"):
        check(f"the run reports {want}", want in meta, meta[:70])

    check("a run in progress can be stopped",
          pg.query_selector("#do-stop") is not None)
    # "Stop & keep checkpoints" — the epochs already written survive, which is
    # what makes stopping a real choice rather than a loss.
    check("and says the checkpoints survive",
          "keep" in (pg.text_content("#do-stop") or "").lower(),
          repr(pg.text_content("#do-stop")))

    pg.wait_for_function(
        "() => (document.querySelector('#run-phase')?.textContent || '').trim() === 'Done'",
        timeout=90_000)
    done = (pg.text_content("#run-done") or "").replace("\n", " ")
    check("a finished run names what it produced",
          "checkpoint" in done.lower(), done.strip()[:80])
    check("and where it went",
          "loras/" in (pg.text_content("#run-done") or ""), done.strip()[:80])

    if errors:
        check("no page errors", False, str(errors[:2]))
    b.close()

print()
if fails:
    print(f"{len(fails)} failed:", file=sys.stderr)
    for f in fails:
        print(f"  {f}", file=sys.stderr)
    sys.exit(1)
print("train and datasets intact")
