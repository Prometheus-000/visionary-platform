"""
Where the duplicate thresholds come from.

    python3 tools/tune_dupes.py "/path/to/a/real/folder"

`smoke_dupes.py` asserts that the classifier still does what it did. This is the
other half: it measures a real folder and prints the distributions the
thresholds are chosen from, because a threshold argued from first principles is
a threshold nobody has looked at.

The number that matters is not how many duplicates it finds. It is **how close
the nearest unrelated pair comes**, because that is the whole margin — a
threshold set inside the tail of the unrelated distribution puts two different
photographs in front of you with one of them marked for deletion, which is the
failure that makes the feature untrustworthy rather than merely incomplete.
Runway and editorial sets are the hard case and the right thing to point this
at: forty frames of one look on one stage are genuinely different photographs
that a perceptual hash has every reason to confuse.

It writes nothing and deletes nothing. The fingerprint cache it builds is the
same one the app uses, beside the folder under `.thumbs/`.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _from_app import DUPES, pull

APP = pull(DUPES)


def quantiles(xs: list[int]) -> str:
    if not xs:
        return "none"
    s = sorted(xs)
    q = lambda f: s[min(len(s) - 1, int(len(s) * f))]  # noqa: E731
    return (f"min {s[0]:3d}  p1 {q(.01):3d}  p5 {q(.05):3d}  "
            f"median {q(.5):3d}  max {s[-1]:3d}")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.strip())
        return 2
    folder = Path(argv[1]).expanduser()
    if not folder.is_dir():
        print(f"not a folder: {folder}")
        return 2

    t0 = time.time()
    prints, wrote, _ = APP["_fingerprints"](folder)
    scan_s = time.time() - t0
    n = len(prints)
    if n < 2:
        print(f"{n} images — nothing to compare")
        return 2
    print(f"{n} images fingerprinted in {scan_s:.1f}s "
          f"({'measured' if wrote else 'from cache'}, {scan_s / n * 1000:.0f}ms each)\n")

    names = list(prints)
    dup, sim, crop, none_d, none_p, rejects = [], [], [], [], [], []
    D, S = APP["DUPLICATE_MATCH"], APP["SIMILAR_MATCH"]
    pairs = 0
    t0 = time.time()
    for i in range(n):
        a = prints[names[i]]
        for j in range(i + 1, n):
            b = prints[names[j]]
            pairs += 1
            link = APP["_link"](a, b)
            row = (link["dhash"], link["phash"], names[i], names[j])
            if link["kind"] == "duplicate":
                dup.append(row)
            elif "cropped" in link["transforms"]:
                crop.append(row)
            elif link["kind"] == "similar":
                sim.append(row)
            else:
                none_d.append(link["dhash"])
                none_p.append(link["phash"])
                # The real margin for an AND rule. A pair rejected with dhash 7
                # and phash 40 is not one bit from being accepted — it is 30.
                # So the distance that matters is how far the *worse* of the two
                # is from its threshold, and the minimum of that over every
                # rejected pair is the whole safety margin.
                rejects.append((max(link["dhash"] - S["dhash"],
                                    link["phash"] - S["phash"]),
                                link["dhash"], link["phash"], names[i], names[j]))
    compare_s = time.time() - t0
    print(f"{pairs:,} pairs compared in {compare_s:.1f}s\n")

    print(f"thresholds — duplicate dhash<={D['dhash']} AND phash<={D['phash']}"
          f" · similar dhash<={S['dhash']} AND phash<={S['phash']}\n")
    print(f"  duplicate pairs {len(dup):6,}")
    print(f"  similar pairs   {len(sim):6,}")
    print(f"  crop-only pairs {len(crop):6,}   (found by the crop pass alone)")
    print(f"  unrelated       {len(none_d):6,}\n")

    # The margin. How close does the *rejected* population come to the accepting
    # thresholds — and how far is the accepted population from them.
    print("unrelated pairs, distance distribution:")
    print(f"  dhash  {quantiles(none_d)}")
    print(f"  phash  {quantiles(none_p)}\n")

    # The margin, stated as one number: the nearest miss.
    rejects.sort()
    print("margin — how many bits the nearest REJECTED pairs are from `similar`:")
    print(f"  {quantiles([r[0] for r in rejects])}")
    for gap, dd, dp, a, b in rejects[:8]:
        print(f"  +{gap:2d} bits (d{dd:2d} p{dp:2d})  {a[:40]:40s}  {b[:40]}")
    print()
    if sim:
        print("similar pairs:")
        print(f"  dhash  {quantiles([r[0] for r in sim])}")
        print(f"  phash  {quantiles([r[1] for r in sim])}\n")

    groups = APP["_duplicate_groups"](folder)
    s = groups["summary"]
    print(f"grouped: {s['duplicate_groups']} duplicate groups "
          f"({s['duplicate_images']} images) · "
          f"{s['similar_groups']} similar groups ({s['similar_images']} images)")
    print(f"reclaim if every suggestion is accepted: "
          f"{groups['reclaim'] / 1e6:.0f} MB\n")

    # Eyeball fodder. Every judgement about whether a threshold is right is made
    # by looking at the pairs nearest the line, so they are named rather than
    # counted.
    def show(label: str, rows: list, key) -> None:
        if not rows:
            return
        print(f"{label} (closest to the line first):")
        for dd, dp, a, b in sorted(rows, key=key, reverse=True)[:12]:
            print(f"  d{dd:2d} p{dp:2d}  {a[:44]:44s}  {b[:44]}")
        print()

    show("WIDEST duplicate pairs — a false one here is a preselected deletion",
         dup, lambda r: (r[0], r[1]))
    show("CROP-ONLY pairs — this is the crop pass earning or not earning its place",
         crop, lambda r: (-r[0], -r[1]))
    show("WIDEST similar pairs — a false one here only costs you a look",
         sim, lambda r: (r[0], r[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
