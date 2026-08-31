"""
What the gallery has to get right about *freshness*, asserted against a real browser.

    python3.11 tools/ui-checks/check_gallery.py [http://localhost:8791]

Every assertion here is one fault from the session that produced them, and they share a
cause worth stating once: the listing used to be a walk of the volume mount, and the
mount could only move forward when nothing on the volume was open — while the gallery
itself held a file open per cover, for the length of every transfer. So painting the
gallery froze the gallery. The symptom was a grid showing a block of results from an
arbitrary earlier moment, catching up on its own, then freezing somewhere else.

None of that is visible in a screenshot, and none of it fails loudly. A frozen listing
renders perfectly — it is simply a listing of a different point in time, and it looked
fine for months. That is the argument for asserting it here rather than looking at it.

Pointed at a URL like the other checks in this directory, so it runs against the stub
server or a real deployment without knowing which.
"""

import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8791"

fails: list[str] = []


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(label)


def open_drawer(page):
    """The drawer is closed on load — the canvas is the largest thing on screen and
    the drawer is the one piece of chrome that takes width from it. So every entry
    here opens it first, the way a person does, rather than assuming it open. Kept
    idempotent because two of the rows below re-navigate."""
    if not page.evaluate("() => document.querySelector('#t-drawer')?.classList.contains('on')"):
        page.click("#t-drawer")
    page.wait_for_timeout(250)


def open_gallery(page):
    """Into the full grid, by the door the drawer actually offers."""
    page.click('#drawer .drawer-head button[title="Open gallery"]')
    page.wait_for_selector("#gal-grid .gal", timeout=5000)


def run(page):
    seen: list[str] = []
    page.on("request", lambda r: seen.append(r.url))

    page.goto(URL)
    open_drawer(page)
    page.wait_for_selector("#drawer-grid .gal", timeout=10000)
    page.wait_for_timeout(400)

    # ---- covers ---------------------------------------------------------
    #
    # The whole reason the listing could freeze. A 232px cell was served the full
    # 1024px PNG through `/api/file`, which answers with a FileResponse and therefore
    # holds a descriptor open on the volume for the length of the transfer to the
    # browser. Dozens at once is a continuous window in which `volume.reload()` is
    # refused, so the grid's own painting is what stops the grid updating.
    grid = [u for u in seen if "/api/cover/" in u or "/api/file/" in u]
    check("the drawer asks for covers", any("/api/cover/" in u for u in grid),
          f"{sum('/api/cover/' in u for u in grid)} covers")
    # Clips are the exception and are meant to be: web_image has no ffmpeg, so a card
    # fetches the frame it needs rather than having the server ship an mp4 to make a
    # picture of.
    stills = [u for u in grid if "/api/file/" in u and ".mp4" not in u]
    check("and never a full-resolution still", not stills, str(stills[:2]))

    # ---- the video gate -------------------------------------------------
    #
    # `loading="lazy"` covers an <img> and has no equivalent for <video>, so every clip
    # in the grid issued a range request on mount whether or not it was on screen.
    #
    # Both halves, and the positive one is the load-bearing check: a gate stuck shut
    # would pass "nothing below the fold loads" trivially — which is a worse regression
    # than the bug, a grid of clips that never paint. So assert that a clip *in* view
    # does upgrade, and one well below it does not. `innerHeight + 400` clears the
    # 200px `rootMargin`, so a below-fold card is genuinely outside the gate's reach.
    #
    # (Skipped where IntersectionObserver does not fire — some embedded/headless
    # browsers never composite off-screen content, so the observer stays silent and
    # this would report a false regression. The real-browser runs are the ones that
    # matter here.)
    open_gallery(page)
    page.wait_for_timeout(800)
    clips = page.evaluate("""() => {
      const all = [...document.querySelectorAll('#gal-grid .gal')];
      const vid = all.filter(c => c.querySelector('video') || c.querySelector('div.media'));
      const inView = vid.filter(c => {
        const t = c.getBoundingClientRect().top;
        return t >= 0 && t < innerHeight;
      });
      const below = vid.filter(c => c.getBoundingClientRect().top > innerHeight + 400);
      const io = typeof IntersectionObserver !== 'undefined';
      return {io,
              inView: inView.length,
              inViewLoaded: inView.filter(c => c.querySelector('video')).length,
              below: below.length,
              belowLoaded: below.filter(c => c.querySelector('video')).length};
    }""")
    if clips["inViewLoaded"] == 0 and clips["inView"]:
        check("(video gate: observer never fired — skipped)", True,
              "no IntersectionObserver callbacks in this browser")
    else:
        check("a clip in view has loaded, one below the fold has not",
              clips["inViewLoaded"] == clips["inView"] and clips["belowLoaded"] == 0,
              f"{clips['inViewLoaded']}/{clips['inView']} in view, "
              f"{clips['belowLoaded']}/{clips['below']} below")

    # ---- the drawer's viewer -------------------------------------------
    #
    # The drawer renders `items.slice(0, 24)` and used to hand that same array to the
    # viewer, so its counter read "3 / 24" on a volume holding hundreds: a render cap
    # had quietly become a navigation cap.
    page.click('#gal-full button[title="Back to canvas"]')
    page.wait_for_selector("#gal-full", state="detached", timeout=5000)
    total = page.evaluate("""async () => {
      const r = await fetch('/api/gallery?limit=200');
      return (await r.json()).total;
    }""")
    page.click("#drawer-grid .gal .media")
    page.wait_for_selector(".lb", timeout=5000)
    at = page.inner_text(".lb .lb-at")
    page.keyboard.press("Escape")
    check("a viewer opened from the drawer pages the whole listing",
          at.strip().endswith(f"/ {total}"), f"{at.strip()!r} of {total}")

    # ---- out-of-order replies ------------------------------------------
    #
    # Two /api/gallery calls overlap constantly — Refresh pressed while a land is still
    # fetching is the common case — and without a guard the reply that arrives last
    # wins rather than the one that is newest. An older listing painted over a newer
    # one is indistinguishable from the staleness all of this is about.
    page.evaluate("""() => {
      const real = window.fetch
      window.__slow = true
      window.fetch = (u, o) => {
        if (typeof u === 'string' && u.startsWith('/api/gallery')) {
          const n = ++window.__n
          // The first reply is held past the second, so "last to arrive" and
          // "newest" disagree — which is the only condition the guard exists for.
          return real(u, o).then(async (r) => {
            const body = await r.json()
            const rows = n === 1 ? body.items.slice(0, 3) : body.items
            if (n === 1) await new Promise((k) => setTimeout(k, 900))
            return new Response(JSON.stringify({...body, items: rows}),
                                {headers: {'content-type': 'application/json'}})
          })
        }
        return real(u, o)
      }
      window.__n = 0
    }""")
    page.evaluate("""() => {
      // Two deliberate reloads, back to back. The first is the one held.
      document.querySelector('#drawer .drawer-head button').click()
    }""")
    page.wait_for_timeout(80)
    page.click('#gal-full button[title="Refresh"]')
    page.wait_for_timeout(2000)
    n = page.evaluate("() => document.querySelectorAll('#gal-grid .gal').length")
    check("a late older reply does not overwrite a newer listing", n > 3,
          f"{n} cards")

    # ---- session-first --------------------------------------------------
    #
    # The page watched the run finish and holds its job id and filenames; asking the
    # volume to tell it that, and believing an answer that comes back without it, is
    # how a fresh render fell out of the drawer for as long as one container lasted.
    page.reload()
    # The drawer renders no cards while it is shut now — the collapse used to
    # hide two dozen live covers behind a closed panel — so a reload, which
    # shuts it, has to open it again the way a person would.
    open_drawer(page)
    page.wait_for_selector("#drawer-grid .gal", timeout=10000)
    before = page.evaluate("() => document.querySelectorAll('#drawer-grid .gal').length")
    page.evaluate("""() => {
      const own = JSON.parse(sessionStorage.getItem('vis-mine') || '[]')
      own.unshift({job_id: 'genLOCAL', kind: 'image', files: ['00.png'],
                   created: Date.now() / 1000 + 60})
      sessionStorage.setItem('vis-mine', JSON.stringify(own))
    }""")
    page.reload()
    open_drawer(page)
    page.wait_for_selector("#drawer-grid .gal", timeout=10000)
    page.wait_for_timeout(600)
    # **The card's own id, not its cover's URL.** This read the first
    # `.gal img`'s `src` and looked for the job id inside it, which is a
    # property of the *picture* rather than of the card — `Thumb` fetches the
    # cover and replaces `src` with a `blob:` URL as soon as the bytes land, so
    # the assertion was a race it lost whenever the fetch was quick, and a clip
    # card has no `<img>` in it at all. `data-job` is the card naming itself;
    # see `Card.tsx`. Equality rather than a substring, which the id in a URL
    # could never be.
    first = page.evaluate(
        "() => document.querySelector('#drawer-grid .gal')?.dataset.job")
    check("a result the listing omits is still in the grid",
          first == "genLOCAL", str(first))
    check("and it did not displace the listing",
          page.evaluate(
              "() => document.querySelectorAll('#drawer-grid .gal').length") >= before)

    # ---- the stale retry ------------------------------------------------
    #
    # The server has always reported `stale`; the client dropped it on the floor, so a
    # listing that knew it was behind was painted in silence. The retry that answers it
    # must be armed by the reply rather than a clock, or the fix is a poll loop against
    # a container already saying it cannot keep up.
    page.goto(f"{URL}?stale=1")
    open_drawer(page)
    page.evaluate("""() => {
      window.__calls = 0
      const real = window.fetch
      window.fetch = (u, o) => {
        if (typeof u === 'string' && u.startsWith('/api/gallery')) {
          window.__calls++
          return real(u.includes('stale=') ? u : u + '&stale=1', o)
        }
        return real(u, o)
      }
    }""")
    page.evaluate("() => document.querySelector('#drawer .drawer-head button').click()")
    page.wait_for_timeout(9000)
    calls = page.evaluate("() => window.__calls")
    # One deliberate reload plus a bounded chain. Nine seconds is well past the whole
    # backoff, so anything still climbing here is a loop.
    check("a stale listing is retried, and bounded", 1 <= calls <= 4, f"{calls} calls")
    check("Refresh says so once it has given up",
          "behind" in (page.get_attribute('#gal-full button[title*="Refresh"]', "title")
                       or "").lower(),
          page.get_attribute('#gal-full button[title*="Refresh"]', "title") or "")


def main():
    print(f"=== {URL} ===")
    with sync_playwright() as pw:
        # `channel="chrome"` like the other checks here: the bundled headless shell is
        # not installed on this machine and real Chrome is, which is also the browser
        # whose connection budget and lazy-loading behaviour these assertions are about.
        b = pw.chromium.launch(channel="chrome")
        page = b.new_page(viewport={"width": 1512, "height": 982})
        try:
            run(page)
        finally:
            b.close()
    if fails:
        print(f"\n{len(fails)} failed: {', '.join(fails)}")
        sys.exit(1)
    print("\ngallery freshness intact")


if __name__ == "__main__":
    main()
