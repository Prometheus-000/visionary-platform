"""
Turn the `ICON` table in UI_HTML into React components.

    python3 tools/extract_icons.py

Twenty-five inline SVGs. They are inline in the first place because a sprite
sheet or an icon font would be a second asset for a single-file app to serve,
and that reason survives the port — a bundled React app can afford a file, but
these are still the cheapest form and they are already drawn.

Generated rather than hand-transcribed for the obvious reason: converting
`stroke-width` to `strokeWidth` twenty-five times by eye is twenty-five chances
to typo a path, and a wrong `d` attribute is a shape nobody notices is wrong
until they look for it. Re-run after any change to the table.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"
OUT = ROOT / "web" / "src" / "icons.tsx"

# SVG attributes React spells differently. Anything not here is passed through,
# which is right for viewBox (already camel), d, x, y, rx, cx, cy, r, width…
RENAME = {
    "stroke-width": "strokeWidth",
    "stroke-linecap": "strokeLinecap",
    "stroke-linejoin": "strokeLinejoin",
    "stroke-dasharray": "strokeDasharray",
    "stroke-dashoffset": "strokeDashoffset",
    "fill-rule": "fillRule",
    "clip-rule": "clipRule",
    "stop-color": "stopColor",
    "class": "className",
}

HEADER = '''/*
 * Generated from the ICON table in app.py by tools/extract_icons.py.
 * Do not edit here — re-run the script.
 *
 * These are inline because a sprite sheet or an icon font would be a second
 * asset to serve for twenty-five small shapes, and they are sized by CSS
 * (`.opt>svg`, `.ico svg`) rather than by attributes, so they inherit
 * `currentColor` and whatever box they are dropped into.
 */
'''


def to_jsx(svg: str) -> str:
    def fix(m):
        name, value = m.group(1), m.group(2)
        return f'{RENAME.get(name, name)}="{value}"'

    jsx = re.sub(r'([a-zA-Z-]+)="([^"]*)"', fix, svg)
    # Self-close the void SVG elements the table uses unclosed.
    for tag in ("path", "rect", "circle", "line", "polyline", "polygon", "ellipse"):
        jsx = re.sub(rf"<{tag}([^>/]*)>", rf"<{tag}\1 />", jsx)
    return jsx


def component(name: str) -> str:
    return "Icon" + name[:1].upper() + name[1:]


def main():
    src = APP.read_text()
    i = src.index("const ICON={")
    depth, j = 0, i
    for k in range(i + len("const ICON="), len(src)):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                j = k + 1
                break
    else:
        sys.exit("could not find the end of the ICON table")

    block = src[i:j]
    # key:'<svg …>' — single-quoted, and none of the SVGs contain one.
    entries = re.findall(r"(\w+)\s*:\s*'(<svg.*?</svg>)'", block, re.S)
    if not entries:
        sys.exit("found the ICON table but no entries in it")

    out = [HEADER]
    for name, svg in entries:
        out.append(f"export function {component(name)}() {{\n  return (\n    {to_jsx(svg)}\n  )\n}}\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out))
    print(f"wrote {OUT.relative_to(ROOT)} — {len(entries)} icons")
    print("  " + ", ".join(component(n) for n, _ in entries))


if __name__ == "__main__":
    main()
