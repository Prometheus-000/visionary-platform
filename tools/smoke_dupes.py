"""
Duplicate grouping, against images that were actually re-encoded.

    python3 tools/smoke_dupes.py

The whole claim of a perceptual hash is about files that a `sha256` map reports
as unrelated — a JPEG saved from a PNG, a half-size copy, a re-grade — so a
fixture that asserts against hand-written hashes asserts nothing. Every case
below is built by round-tripping real pixels through Pillow at the quality,
scale or exposure the case is named for, then grouped by the real
`_duplicate_groups` pulled out of app.py.

The negative rows matter more than the positives. A hash that groups everything
also groups every duplicate, and the failure it produces — two different
photographs of the same room filed as one picture, one of them suggested for
deletion — is the expensive one. This file is where the *behaviour* is pinned;
`tune_dupes.py` is where the thresholds behind it are chosen, against a real
folder, because a threshold argued from first principles is a threshold nobody
has looked at.

Some rows pin limits rather than features — how far the crop pass reaches, and
the crop whose partner sits inside a duplicate group and therefore waits its
turn. A limit nobody wrote down is a limit somebody re-discovers as a bug.

Stdlib and Pillow only. No Modal, no volume, no network.
"""

import math
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image, ImageEnhance

from _from_app import DUPES, pull

APP = pull(DUPES)


def scene(seed: int, size=(1024, 768)):
    """
    A deterministic picture with *large* structure in it.

    The first version of this drew per-pixel banding, and every row below
    failed: a 9x8 downsample of high-frequency noise averages to nothing in
    particular, so JPEG flipped bits at random and one picture's five encodings
    landed in two groups. That is a property of the fixture rather than of the
    hash — a photograph is low-frequency at the scale a thumbnail sees — and it
    is worth the note, because a fixture that fails a working hash is the kind
    of failure that gets the hash rewritten.
    """
    w, h = size
    im = Image.new("RGB", (w, h))
    px = im.load()
    for y in range(h):
        v = y / h
        for x in range(w):
            u = x / w
            px[x, y] = (
                int(128 + 120 * math.sin(6.0 * u + seed) * math.cos(4.0 * v + seed * 0.7)),
                int(128 + 120 * math.sin(3.0 * (u + v) + seed * 1.3)),
                int(128 + 120 * math.cos(5.0 * v - seed * 0.5) * math.sin(2.0 * u + seed)),
            )
    return im


def centre(im, share: float):
    w, h = im.size
    cw, ch = int(w * share), int(h * share)
    return im.crop(((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch))


def write(d: Path, name: str, im: Image.Image, caption: str = "", **save):
    im.save(d / name, **save)
    if caption:
        (d / name).with_suffix(".txt").write_text(caption)


def build(d: Path):
    """
    One folder holding every case, because grouping is a property of the folder.
    A case checked in isolation cannot fail by pulling in a neighbour, which is
    the failure that matters.
    """
    base, other, third = scene(1), scene(2), scene(3)

    # One picture, three encodings, two sizes.
    write(d, "orig.png", base)
    write(d, "export.jpg", base, quality=72)
    write(d, "small.jpg", base.resize((512, 384), Image.LANCZOS), quality=90)
    # A byte-identical copy — the case a sha catches, and the one where there is
    # nothing left to weigh.
    shutil.copyfile(d / "orig.png", d / "orig_copy.png")
    # A re-grade: every pixel moved, every comparison between neighbours intact.
    write(d, "brighter.jpg", ImageEnhance.Brightness(base).enhance(1.25), quality=88)
    # An exact 80% crop of a picture that already has copies. Its only partner is
    # inside a duplicate group, and similar links are computed over what is left
    # — so this lands in no group at all until the copies are dealt with. That
    # is the cost of "an image is in at most one group", pinned rather than
    # discovered later.
    write(d, "base_crop.jpg", centre(base, 0.8), quality=92)

    # A second picture, twice: the group whose top two differ only in encoding.
    write(d, "b_orig.png", other, caption="a caption worth keeping")
    write(d, "b_export.jpg", other, quality=70)

    # A third, as an original and a half-size re-post: the group whose top two
    # differ in pixels, which is the axis the suggestion leads with.
    write(d, "c_orig.png", third)
    write(d, "c_small.jpg", third.resize((512, 384), Image.LANCZOS), quality=88)

    # The crop pass, with nothing else in the way. 80% is a share the variants
    # cover exactly; 60% is not covered by any single variant and is reached
    # only because a 0.8 crop of one lines up with a 0.9 crop of the other —
    # which is worth pinning, because it is the reach of the pass rather than
    # its design.
    write(d, "d_orig.png", scene(5))
    write(d, "d_crop80.jpg", centre(scene(5), 0.8), quality=92)
    write(d, "e_orig.png", scene(7))
    write(d, "e_crop60.jpg", centre(scene(7), 0.6), quality=92)

    # The negative rows: neither of these is related to anything.
    write(d, "lone.png", scene(9))
    write(d, "lone2.png", scene(17))


def check(label: str, got, want) -> bool:
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'}  {label}: {got!r}" + ("" if ok else f" != {want!r}"))
    return ok


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="dupes-"))
    try:
        build(tmp)
        rep = APP["_duplicate_groups"](tmp)
        groups = {g["key"]: g for g in rep["groups"]}
        where = {r["name"]: g for g in rep["groups"] for r in g["images"]}
        kind = {n: g["kind"] for n, g in where.items()}

        rows = []
        rows.append(check("images scanned", rep["images"], 16))

        # ---- duplicates ---------------------------------------------------
        rows.append(check(
            "one picture, five files, one group",
            len({id(where.get(n)) for n in ("orig.png", "orig_copy.png", "export.jpg",
                                            "small.jpg", "brighter.jpg")}), 1))
        rows.append(check("and it is a duplicate group", kind.get("orig.png"), "duplicate"))
        rows.append(check("second picture is its own group",
                          where["b_orig.png"] is where["b_export.jpg"]
                          and where["b_orig.png"] is not where["orig.png"], True))
        rows.append(check("duplicate groups found", rep["summary"]["duplicate_groups"], 3))

        big = where["orig.png"]
        rows.append(check("keeper is the largest original", big["suggest"], "orig.png"))
        # The reason names the axis against the *runner-up*, which here is a
        # byte-identical copy: "there is nothing to choose between these two" is
        # the true statement about the top of this ranking.
        rows.append(check("and says why", big["why"],
                          "identical in every respect — first by name"))
        rows.append(check("the byte-identical copy is marked as one",
                          [r["same_file"] for r in big["images"] if r["name"] == "orig_copy.png"],
                          [True]))
        rows.append(check("a re-encode is not marked as one",
                          [r["same_file"] for r in big["images"] if r["name"] == "export.jpg"],
                          [False]))
        rows.append(check("a resize is named as one",
                          [r["transforms"] for r in big["images"] if r["name"] == "small.jpg"],
                          [["resized", "reformatted"]]))
        rows.append(check("megapixels are reported",
                          [r["megapixels"] for r in big["images"] if r["name"] == "orig.png"],
                          [0.8]))
        rows.append(check("format is reported",
                          sorted({r["format"] for r in big["images"]}), ["JPEG", "PNG"]))
        # Format outranks weight in `_keep_rank`, so a PNG beside its JPEG
        # export is decided on the encoding rather than on the byte count.
        rows.append(check("decided on the encoding", where["b_orig.png"]["why"],
                          "PNG over JPEG"))
        rows.append(check("decided on pixels", where["c_orig.png"]["why"],
                          "most pixels · 0.8 MP"))

        # ---- crops --------------------------------------------------------
        rows.append(check("an 80% crop is grouped", "d_crop80.jpg" in where, True))
        rows.append(check("as similar, never as a duplicate",
                          kind.get("d_crop80.jpg"), "similar"))
        rows.append(check("and the crop is named",
                          "cropped" in (where["d_crop80.jpg"]["images"][0]["transforms"]
                                        + where["d_crop80.jpg"]["images"][1]["transforms"]), True))
        rows.append(check("a similar group preselects nothing",
                          where["d_crop80.jpg"]["suggest"], ""))
        rows.append(check("a 60% crop is still reached", "e_crop60.jpg" in where, True))
        rows.append(check("and with the right partner",
                          sorted(r["name"] for r in where["e_crop60.jpg"]["images"]),
                          ["e_crop60.jpg", "e_orig.png"]))
        # The documented cost of one-group-per-image.
        rows.append(check("a crop whose partner has copies waits its turn",
                          "base_crop.jpg" in where, False))

        # ---- negatives ----------------------------------------------------
        rows.append(check("a lone image is in no group", "lone.png" in where, False))
        rows.append(check("a second lone image is in no group", "lone2.png" in where, False))
        rows.append(check("nothing unrelated was grouped",
                          rep["summary"]["duplicate_images"]
                          + rep["summary"]["similar_images"], 13))

        # ---- resuming -------------------------------------------------------
        # A scan that runs out of time answers with a count and asks to be
        # called again, and the cache it wrote is the only thing carrying its
        # place. Driven at a zero budget, which is the pathological case: every
        # call may measure at most one image, so this also proves the loop
        # cannot stall — a request that measures nothing would never converge.
        fresh = Path(tempfile.mkdtemp(prefix="dupes-resume-"))
        try:
            build(fresh)
            calls, seen, totals = 0, [], set()
            while True:
                part = APP["_duplicate_groups"](fresh, 0.0)
                calls += 1
                if not part["scanning"]:
                    break
                seen.append(part["measured"])
                totals.add(part["total"])
                if calls > 100:
                    break
            rows.append(check("a bounded scan converges", calls <= 100, True))
            rows.append(check("and it never goes backwards",
                              seen == sorted(seen), True))
            rows.append(check("naming one total the whole way",
                              len(totals) == 1 and totals == {16}, True))
            rows.append(check("the finished scan groups the same as an unbounded one",
                              [g["key"] for g in part["groups"]],
                              [g["key"] for g in rep["groups"]]))
            rows.append(check("and reports itself finished", part["scanning"], False))
        finally:
            shutil.rmtree(fresh, ignore_errors=True)

        # ---- the cache ----------------------------------------------------
        # A second scan must not re-measure anything: the cache is what makes a
        # rescan free, and a stamp that never matches is a cache that is only
        # ever written.
        rescan = APP["_duplicate_groups"](tmp)
        rows.append(check("a rescan writes nothing", rescan["_wrote"], False))
        rows.append(check("and finds the same groups",
                          [g["key"] for g in rescan["groups"]],
                          [g["key"] for g in rep["groups"]]))

        print()
        bad = rows.count(False)
        print(f"{len(rows) - bad}/{len(rows)} ok")
        return 1 if bad else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
