"""
The two matrices, as vector assets for a case study.

They are a pair and only mean anything together: the first is eleven criteria
scored against four models, every one of them text-to-text, on a feature that
turned out to reach 0% of renders. The second is the same question asked by
looking at the pictures instead — where the approach the first one was ranking
loses 0 of 30, and the thing that replaced it wins.

SVG rather than PNG so a portfolio can restyle them: every colour is a named
token at the top, text is text, and the layout is computed rather than placed.

    python3.11 tools/case_study_assets.py
"""
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path("docs/case-study")

INK, MUTED, FAINT = "#14181E", "#5E6975", "#8A95A1"
RULE, FIRM, SUNK = "#D6DDE4", "#B9C4CE", "#EDF1F4"
GROUND, SURFACE = "#F2F5F7", "#FFFFFF"
ACCENT, ACC_SF = "#2C5D71", "#E2EDF2"
GOOD, WARN, BAD = "#3C7355", "#966A20", "#9C4033"
BAD_SF, GOOD_SF = "#F4E3E0", "#E4EFE8"

SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MONO = "ui-monospace, 'SF Mono', SFMono-Regular, Menlo, Consolas, monospace"
SERIF = "ui-serif, 'Iowan Old Style', Palatino, Georgia, serif"


def t(x, y, s, *, size=13, fill=INK, family=SANS, weight="400", anchor="start",
      spacing=None, upper=False):
    s = escape(str(s))
    extra = f' letter-spacing="{spacing}"' if spacing else ""
    if upper:
        extra += ' style="text-transform:uppercase"'
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'fill="{fill}" font-weight="{weight}" text-anchor="{anchor}"'
            f'{extra}>{s}</text>')


def rect(x, y, w, h, fill, **kw):
    more = "".join(f' {k.replace("_","-")}="{v}"' for k, v in kw.items())
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}"{more}/>'


def line(x1, y1, x2, y2, stroke=RULE, width=1):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{stroke}" stroke-width="{width}"/>')


def head(w, eyebrow, title, deck, y=0):
    o = [t(56, y + 52, eyebrow, size=11, fill=ACCENT, family=MONO, spacing="1.6", upper=True),
         t(56, y + 96, title, size=34, family=SERIF, weight="600", spacing="-0.5")]
    yy = y + 128
    for ln in deck:
        o.append(t(56, yy, ln, size=14.5, fill=MUTED))
        yy += 22
    o.append(line(56, yy + 6, w - 56, yy + 6, INK, 2))
    return "".join(o), yy + 6


# ── 1 · judged by arithmetic ────────────────────────────────────────────────
#
# Every figure here is real and every one of them is text against text. That is
# the point of the asset: the matrix is not wrong, it is answering a question
# nobody needed answered.

COLS = ["4B abliterated\nthe fork · L4", "4B base\nInstruct-2507", "Qwen3-8B\nL4",
        "Qwen3-14B\nA100-40GB", "Deployed\nfork + validator"]

ROWS = [
    ("User text preserved",        ["5/8", "7/8", "7/8", "7/8", "5/8"], None),
    ("Semantic contradiction",     ["0/16", "0/16", "0/16", "0/16", "0/16"], None),
    ("Degraded to nothing",        ["0/8", "0/8", "0/8", "0/8", "3/8"], None),
    ("Spans marked invented",      ["4.4%", "0.0%", "0.0%", "9.3%", "4.4%"], None),
    ("Genuinely invented",         ["0.0%", "0.0%", "0.0%", "2 runs", "0.0%"], "bad"),
    ("Schema validity",            ["8/8", "8/8", "7/8", "8/8", "8/8"], None),
    ("Relationship accuracy",      ["1/8", "3/8", "3/8", "3/8", "1/8"], None),
    ("Entity accuracy",            ["8/8", "7/8", "7/8", "8/8", "5/8"], None),
    ("Refusal behaviour",          ["10/10", "9/10", "—", "—", "—"], None),
    ("Idempotency",                ["4/8", "6/8", "7/8", "4/8", "6/8"], None),
    ("Round-trip fidelity",        ["0/3", "0/3", "0/3", "0/3", "0/3"], "bad"),
    ("Rule compliance",            ["6/7", "7/7", "3/7", "4/7", "3/7"], None),
    ("Latency mean / worst",       ["9.4 / 17.2s", "8.0 / 16.0s", "4.4 / 11.0s",
                                    "4.3 / 8.7s", "10.0 / 26.9s"], None),
    ("VRAM · weights",             ["7.64 GiB", "7.64 GiB", "15.27 GiB",
                                    "27.52 GiB", "7.64 GiB"], None),
]


def asset_arithmetic():
    W, LX, CW, RH = 1500, 56, 214, 34
    hdr, y = head(W, "The apparatus that was replaced", "Judged by arithmetic", [
        "Fourteen criteria, four candidates, measured on live weights. Every row scores the",
        "document against the sentence it came from. Not one of them looks at a picture."])
    o = [hdr]
    x0 = LX + 268
    y += 34
    for i, c in enumerate(COLS):
        for j, ln in enumerate(c.split("\n")):
            o.append(t(x0 + i * CW + CW - 12, y + j * 15, ln, size=11,
                       fill=INK if j == 0 else FAINT, family=MONO,
                       weight="600" if j == 0 else "400", anchor="end"))
    y += 26
    o.append(line(LX, y, W - 56, y, FIRM, 1))

    for label, vals, mark in ROWS:
        band = BAD_SF if mark == "bad" else None
        if band:
            o.append(rect(LX, y, W - 112, RH, band))
        o.append(t(LX + 10, y + 22, label, size=13, weight="500"))
        for i, v in enumerate(vals):
            fill = BAD if (mark == "bad" and v not in ("2 runs",)) else INK
            if v == "—":
                fill = FAINT
            o.append(t(x0 + i * CW + CW - 12, y + 22, v, size=12.5, family=MONO,
                       fill=fill, weight="600" if mark == "bad" else "400",
                       anchor="end"))
        y += RH
        o.append(line(LX, y, W - 56, y, RULE, 1))

    # The bottom line, which is the whole reason the asset exists.
    y += 30
    o.append(rect(LX, y, W - 112, 96, ACC_SF))
    o.append(rect(LX, y, 4, 96, ACCENT))
    o.append(t(LX + 26, y + 36, "Every row scored. The feature reached 0% of renders.",
               size=19, family=SERIF, weight="600"))
    o.append(t(LX + 26, y + 64,
               "Fourteen ways to be right about characters, and no way to be wrong about the picture.",
               size=13.5, fill=MUTED))
    y += 96 + 44

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{y}" '
           f'viewBox="0 0 {W} {y}">{rect(0,0,W,y,GROUND)}{"".join(o)}</svg>')
    (OUT / "01-judged-by-arithmetic.svg").write_text(svg)
    print(f"  01-judged-by-arithmetic.svg  {W}x{y}")


# ── 2 · judged by looking ───────────────────────────────────────────────────

JUDGED = [
    ("4B fork · shipped rules",   "0/10", "4/10", "6/10", "0.73×", False),
    ("Qwen3-14B · shipped rules", "0/10", "1/10", "9/10", "0.94×", False),
    ("4B fork · long rules",      "0/10", "4/10", "6/10", "0.91×", False),
    ("Rewrite on the encoder",    "3/10", "1/10", "6/10", "21×",   True),
]


def asset_looking():
    W, LX, RH = 1500, 56, 44
    hdr, y = head(W, "What replaced it", "Judged by looking", [
        "The same question asked of the pictures. Each pair rendered at one seed with the sentence as",
        "the only variable, then scored by a vision model with the prompts hidden — twice, with the",
        "images swapped. A win counts only when both orders agree, so a biased judge scores all ties."])
    o = [hdr]

    cols = [("Configuration", LX + 10, "start"), ("Beats bare", 760, "end"),
            ("Loses to bare", 980, "end"), ("Tie", 1150, "end"),
            ("Median growth", 1420, "end")]
    y += 34
    for label, x, anc in cols:
        o.append(t(x, y, label, size=11, fill=FAINT, family=MONO, spacing="1.2",
                   anchor=anc, upper=True))
    y += 16
    o.append(line(LX, y, W - 56, y, FIRM, 1))

    for name, win, loss, tie, grow, is_new in JUDGED:
        if is_new:
            o.append(rect(LX, y, W - 112, RH, GOOD_SF))
        o.append(t(LX + 10, y + 28, name, size=13.5,
                   weight="600" if is_new else "500"))
        for val, x, fill in ((win, 760, GOOD if is_new else BAD),
                             (loss, 980, INK), (tie, 1150, MUTED),
                             (grow, 1420, GOOD if is_new else BAD)):
            o.append(t(x, y + 28, val, size=13, family=MONO, fill=fill,
                       weight="600" if is_new else "400", anchor="end"))
        y += RH
        o.append(line(LX, y, W - 56, y, RULE, 1))

    y += 30
    o.append(rect(LX, y, W - 112, 120, ACC_SF))
    o.append(rect(LX, y, 4, 120, ACCENT))
    o.append(t(LX + 26, y + 36, "Thirty comparisons. The document never won one.",
               size=19, family=SERIF, weight="600"))
    o.append(t(LX + 26, y + 66,
               "Every configuration compressed the fragment — 0.73× median, shorter than its input 10 times of 10 —",
               size=13.5, fill=MUTED))
    o.append(t(LX + 26, y + 88,
               "while the prompts that win grow it twenty-fold. The matrix could not see that, and it is the only thing that mattered.",
               size=13.5, fill=MUTED))
    y += 120 + 44

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{y}" '
           f'viewBox="0 0 {W} {y}">{rect(0,0,W,y,GROUND)}{"".join(o)}</svg>')
    (OUT / "02-judged-by-looking.svg").write_text(svg)
    print(f"  02-judged-by-looking.svg     {W}x{y}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    asset_arithmetic()
    asset_looking()
