"""
Lift the stylesheet out of `UI_HTML` into web/src/styles/, byte for byte.

    python3 tools/extract_css.py

The React port keeps this CSS rather than rewriting it, and the reason is in
what the 1,413 lines encode: the 30% console budget, the dead-space table
behind bar-not-rail, `contain` on thumbnails, the `(hover:none)` splits. None
of that is visible in a declaration — a Tailwind pass would land a page that
looks the same and has quietly discarded every measurement, with nothing
failing to say so.

Extracted rather than copied because `app.py` stays the source of truth for the
whole transition: both front ends are live until the React one replaces the
route, and a hand-copied stylesheet would start drifting on the first fix. Run
this again after any change to the `<style>` block.

The one edit made on the way through is `:root` → `:root, :host`, so the
variables resolve the same whether the bundle is mounted at the document root
or inside a shadow tree; nothing else is touched, including declaration order,
which matters because several rules here win on source order alone.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"
OUT = ROOT / "web" / "src" / "styles" / "ui.css"

BANNER = """/*
 * Extracted from UI_HTML in app.py by tools/extract_css.py — do not edit here.
 *
 * Every measured decision in this file is documented in CLAUDE.md under
 * "The page". The ones most easily lost to a tidy-up:
 *
 *   --- the console budget is 30%% of the viewport, and `fieldMax()` in JS is
 *       what enforces it; the CSS only supplies FIELD_FLOOR/FIELD_CEIL's twins
 *   --- thumbnails are `contain`, not `cover`: a ragged grid is the right
 *       trade, because cropping throws away the information you opened the
 *       gallery to see
 *   --- splits are on `(hover:none)` rather than width wherever the question
 *       is really about the pointer; a tablet with a keyboard is neither of
 *       the things a width test thinks it is
 *
 * %d lines, extracted %s.
 */
"""


def main():
    src = APP.read_text()
    try:
        i = src.index('UI_HTML = r"""')
        j = src.index('\n"""', i)
    except ValueError:
        sys.exit("could not find UI_HTML in app.py")

    ui = src[i:j]
    m = re.search(r"<style>(.*?)</style>", ui, re.S)
    if not m:
        sys.exit("no <style> block inside UI_HTML")

    css = m.group(1).strip("\n")
    # Variables have to resolve wherever the bundle is mounted. This is the only
    # rewrite performed, and it is additive — `:root` keeps working.
    css = css.replace(":root{", ":root, :host{", 1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    stamp = f"from app.py ({len(css.splitlines())} lines of CSS)"
    OUT.write_text(BANNER % (len(css.splitlines()), stamp) + "\n" + css + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} — {len(css.splitlines())} lines, "
          f"{len(css)} bytes")

    # Worth knowing at extraction time rather than after a phone-first pass:
    # a rule declared outside a media query reaches every screen, which is how
    # two touch-motivated changes landed on desktop by accident.
    media = len(re.findall(r"@media", css))
    frames = len(re.findall(r"@keyframes", css))
    props = len(set(re.findall(r"(--[a-z0-9-]+)\s*:", css)))
    print(f"  {media} @media blocks, {frames} @keyframes, {props} custom properties")
    if "prefers-reduced-motion" not in css:
        print("  note: no prefers-reduced-motion block — the animation pass "
              "must add one")


if __name__ == "__main__":
    main()
