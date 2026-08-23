"""
The training board and the dataset editor, asserted against the React front end.

    python3 tools/ui-checks/check_train.py                        # built bundle
    python3 tools/ui-checks/check_train.py http://localhost:5173  # dev server

What is worth pinning here is not the layout — the baseline screenshots cover
that — but the handful of rules that decide whether a training run is
reproducible or wasted:

  * **A run is a card, and there is no page for one.** `train_job` never shared
    anything between runs, so "one at a time" was a property of a page holding a
    single job in a variable. What a card has to carry is what the terminal
    carries: step over total, epoch over total, the rate, elapsed against ETA
    and the loss — a bar alone cannot tell training from stuck.
  * **Every numeric field carries its name.** This is the one place the "a
    control that shows its own value gets no label" rule was pushed past what it
    can carry: "32" is a rank, an alpha, an epoch count or a seed with equal
    plausibility. Rank sits beside alpha, because the effective strength is
    alpha ÷ rank and the two only mean anything as a pair.
  * **Cancelling is two acts, and the dialog separates them.** Stop keeps the
    card and the checkpoints; delete takes the card too; and doing nothing has
    to be reachable, or a stop nobody can take back is a button people learn not
    to press.
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

# name -> the label it must show. The ids survived the move out of the console
# and into the form on purpose: an id that changes with a rehousing is a check
# that silently stops checking.
# Epochs is the only dial left at rest. The form used to open on eighteen controls
# spread verbatim from the server's `train_defaults`, so its subject was the request
# body rather than the job; the other nine dials moved behind `#t-toggle-adv` at the
# same defaults. Everything is still reachable — this split is the assertion.
RESTING = {"a-epochs": "Epochs"}
ADVANCED = {"a-dim": "Rank", "a-alpha": "Alpha", "a-lr": "Learning rate",
            "a-bs": "Batch size", "a-res": "Resolution", "a-shift": "Flow shift",
            "a-rep": "Repeats", "a-seed": "Seed", "a-save": "Save every",
            "a-swap": "Blocks to swap"}

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
    pg.wait_for_selector("#sess-board", timeout=10_000)
    pg.wait_for_timeout(900)
    check("the door reaches the board", pg.is_visible("#sess-list"))

    cards = pg.query_selector_all("[data-session]")
    check("the board lists sessions", len(cards) > 0, f"{len(cards)} cards")

    # The dials are on the card because two runs against the same set differ
    # only in them, and a board where you cannot see which is which is a board
    # of identical rectangles.
    dials = (pg.text_content("[data-session] .sess-dials") or "").lower()
    for want in ("rank", "alpha", "batch", "lr", "epochs", "res"):
        check(f"a card shows its {want}", want in dials, dials[:80])
    # And the four that are *not* on it. The optimizer, the scheduler, the
    # timestep sampling and the flow shift are set once and then read the same
    # on every card, so on a board they were four columns of one repeated word.
    # Asserted by absence because the failure mode is a well-meaning re-add.
    for gone in ("optimizer", "schedule", "timesteps", "flow shift"):
        check(f"a card does not repeat its {gone}", gone not in dials, dials[:80])

    # 3:4, and it is the shape rather than the size that is checked: a card is
    # read in one fixation, which a 1180px row cannot be at any height.
    box = pg.evaluate("""() => { const c = document.querySelector('[data-session]');
        const r = c.getBoundingClientRect(); return {w: r.width, h: r.height} }""")
    ratio = box["h"] / box["w"] if box["w"] else 0
    check("a card is portrait", 1.28 < ratio < 1.40,
          f'{box["w"]:.0f}x{box["h"]:.0f} = {ratio:.2f}')

    # ---- the form ----------------------------------------------------------
    pg.click("#new-session")
    pg.wait_for_selector("#sess-form", timeout=10_000)
    pg.wait_for_timeout(400)
    for i, want in RESTING.items():
        check(f"{i} carries its name", pg.evaluate(LABEL_OF, i) == want,
              str(pg.evaluate(LABEL_OF, i)))

    # The other half of the same rule, and the one that rots first: a dial creeping
    # back onto the resting form is exactly how the eighteen-control version grew,
    # and it would pass every check below without this.
    check("the dials rest behind the disclosure",
          pg.evaluate("() => document.querySelector('#a-dim') === null"))

    pg.click("#t-toggle-adv")
    pg.wait_for_timeout(400)
    for i, want in ADVANCED.items():
        check(f"{i} carries its name", pg.evaluate(LABEL_OF, i) == want,
              str(pg.evaluate(LABEL_OF, i)))

    # Rank beside alpha: the effective strength is alpha ÷ rank, so they are one
    # decision, and they used to sit four fields apart with the epochs between.
    check("rank is next to alpha", pg.evaluate(
        """() => document.querySelector('#a-dim').closest('.opt').nextElementSibling
                 ?.querySelector('#a-alpha') !== null"""))

    # Only the shift sampling reads the shift value, so beside any other one the
    # field is disabled rather than quietly ignored.
    pg.select_option("#a-ts", "sigmoid")
    pg.wait_for_timeout(200)
    check("the flow shift follows the timestep sampling",
          pg.evaluate("() => !!document.querySelector('#a-shift')?.disabled"))
    pg.select_option("#a-ts", "shift")

    for i in ("a-opt", "a-sched", "a-ts"):
        opts = pg.evaluate(f"() => [...document.querySelectorAll('#{i} option')].map(o => o.value)")
        # Served, not written into the page: a menu offering a value the route
        # rejects by name is a run that cold-starts a GPU to die on argparse.
        check(f"#{i} is served by the server", len(opts) >= 3, str(opts[:4]))

    # A session needs a name, a trigger and a set; the button says so by staying
    # disabled, and the hint says which of the three is missing.
    check("Start training waits for a name and a set",
          pg.evaluate("() => !!document.querySelector('#go-train')?.disabled"))

    pg.fill("#lname", "probe_lora")
    pg.fill("#ltrig", "ohwx")
    pg.select_option("#sess-dataset", "studio_portraits")
    pg.wait_for_timeout(300)
    check("naming it and picking a set is enough to start",
          not pg.evaluate("() => !!document.querySelector('#go-train')?.disabled"))

    # ---- a run, start to finish --------------------------------------------
    # Asserted by shape and by terminal state, never by the value at an instant:
    # the stub advances per poll, so two front ends sampled at the same moment
    # are legitimately on different steps.
    pg.click("#go-train")
    pg.wait_for_selector('[data-session] [data-act="stop"]', timeout=20_000)
    # `\s*` rather than a literal space: the label is an <i> and the value is the
    # text node after it, so what separates them on screen is a margin and there
    # is no whitespace in `innerText` at all. Asserting the space asserted the
    # markup rather than the number.
    pg.wait_for_function(
        """() => [...document.querySelectorAll('[data-session]')].some(c =>
             /step\\s*\\d+\\/\\d+/.test(c.innerText))""",
        timeout=30_000)

    card = pg.query_selector('[data-session]:has([data-act="stop"])')
    meta = (card.inner_text() if card else "").replace("\n", " ")
    # A bar alone cannot tell "training" from "stuck". Every one of these is a
    # number that has to move for the run to be legible, and the loss is the one
    # that says whether the hours are buying anything.
    for want in ("step", "epoch", "it/s", "loss"):
        check(f"the running card reports {want}", want in meta, meta[:90])

    # Cancelling is two acts and doing nothing is a third, which is the one this
    # dialog exists to make available.
    pg.click('[data-session] [data-act="stop"]')
    pg.wait_for_selector("#stop-ask", timeout=10_000)
    for i, word in (("#ask-cancel", "keep training"), ("#ask-delete", "delete"),
                    ("#ask-stop", "checkpoint")):
        check(f"{i} offers its own outcome",
              word in (pg.text_content(i) or "").lower(),
              repr(pg.text_content(i)))

    pg.click("#ask-stop")
    pg.wait_for_function(
        """() => [...document.querySelectorAll('[data-session] .chip')].some(c =>
             /stopped/i.test(c.textContent))""",
        timeout=60_000)
    stopped = pg.query_selector('[data-session]:has(.chip.stopped)')
    text = (stopped.inner_text() if stopped else "").replace("\n", " ")
    # A stopped run keeps the card, its progress and its checkpoints — that is
    # what makes stopping a choice rather than a loss, and what you re-run from.
    check("a stopped run keeps what it reached", "%" in text, text[:80])
    # A stopped run draws its bar in red, because the fraction is the whole
    # point there — it is where the run got to. Green is reserved for a card
    # that is spending a GPU right now, and a finished one draws no bar at all:
    # white at 100% across a card is a divider, and read as one.
    check("a stopped run says so in the bar",
          pg.evaluate("""() => { const i = document.querySelector('.sess.stopped .bar > i');
              if (!i) return false;
              const [r,g,b] = getComputedStyle(i).backgroundColor.match(/\\d+/g).map(Number);
              return r > 180 && g < 140 && b < 140 }"""))
    check("a finished run draws no bar",
          pg.evaluate("() => !document.querySelector('.sess.completed .bar')"))

    # Re-running is not a button. Nobody re-runs a finished LoRA unchanged —
    # you come back to move a dial — so the press that matters is the one that
    # opens the form, and that is the whole face of the card.
    check("no card carries its own start button",
          pg.query_selector("[data-session] [data-act=start]") is None)
    pg.click('[data-session]:has(.chip.stopped) .sess-hit')
    pg.wait_for_selector("#sess-form", timeout=10_000)
    check("the card opens its settings", pg.is_visible("#go-train"))
    check("and Start training is there", "start" in (pg.text_content("#go-train") or "").lower())
    pg.keyboard.press("Escape")
    pg.wait_for_timeout(400)

    # ---- the dataset editor ------------------------------------------------
    # Behind a door now, not in a rail beside the board. Nothing on the board
    # reads the set list — the form picks from a menu and a card names its own —
    # so a permanent 320px column of them was width spent on a list no decision
    # on the screen consults.
    pg.click("#ds-door")
    pg.wait_for_selector("#ds-list", timeout=10_000)
    pg.wait_for_timeout(700)
    rows = pg.evaluate("() => document.querySelectorAll('#ds-list [data-ds], #ds-list .ds-row').length")
    check("the sets screen lists sets", rows > 0, f"{rows} rows")
    check("and offers the drop target beside them",
          pg.query_selector("#sets-screen #drop") is not None)

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
    # The way back to the board rides inside the sticky bar, because a set is a
    # long scroll and a back button that scrolls away is one you scroll up for.
    check("the way back to the board is in the bar",
          pg.evaluate("() => !!document.querySelector('.sheet-bar #ds-back')"))

    for i in ("f-all", "f-uncap", "f-notrig", "dens-down", "dens-up"):
        check(f"#{i} is addressable", pg.query_selector(f"#{i}") is not None)
    # A set of photographs offers no clip filter: a control that can only ever
    # empty the grid is worse than one that is absent, and the count line above
    # already says there are none.
    check("a set with no clips offers no clip filter",
          pg.query_selector("#f-vid") is None)

    # A mixed set counts the two kinds apart and tells the tiles apart. Only one
    # of those numbers can be trained on today, so a single total would hide
    # which.
    # Back to the index first. With a set open the index is not on screen —
    # under an open set a list of the other sets is the rail this screen exists
    # to have got rid of — so moving between sets goes through it, which is what
    # `‹ Sets` in the sheet bar is for.
    pg.click(".sheet-bar #ds-back")
    pg.wait_for_selector("#ds-list", timeout=10_000)
    pg.wait_for_timeout(600)
    pg.click("#ds-list >> text=roof_takes")
    pg.wait_for_timeout(1300)
    count = pg.text_content("#ds-count") or ""
    check("a mixed set counts images and clips apart",
          "image" in count and "clip" in count, repr(count))
    check("and offers both kind filters",
          pg.query_selector("#f-img") is not None and pg.query_selector("#f-vid") is not None)
    check("a clip tile is marked as one",
          pg.evaluate("() => document.querySelectorAll('#tiles .tile.vid').length") > 0)
    pg.click("#f-vid")
    pg.wait_for_timeout(400)
    check("the clip filter shows only clips", pg.evaluate(
        """() => { const t = document.querySelectorAll('#tiles .tile');
                   return t.length > 0 && [...t].every(x => x.classList.contains('vid')); }"""))
    pg.click("#f-all")

    # The captioner. Behind the panel, and both menus come from the server.
    pg.click("#ins-toggle")
    pg.wait_for_timeout(700)
    presets = pg.evaluate("() => [...document.querySelectorAll('#cap-preset option')].map(o => o.value)")
    models = pg.evaluate("() => [...document.querySelectorAll('#cap-model option')].map(o => o.value)")
    check("the preset menu is served, not written into the page", len(presets) >= 3, str(presets))
    # A stock instruct model declines to describe photographs of real people
    # often enough to matter, and on a character set that is every image — so the
    # abliterated repackage is the second entry rather than a second code path.
    check("a second captioner is offered", len(models) >= 2, str(models))

    if errors:
        check("no page errors", False, str(errors[:2]))
    b.close()

print()
if fails:
    print(f"{len(fails)} failed:", file=sys.stderr)
    for f in fails:
        print(f"  {f}", file=sys.stderr)
    sys.exit(1)
print("the board and the datasets are intact")
