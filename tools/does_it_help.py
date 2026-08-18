"""
Does the interpretation make a better picture? Nobody had ever asked.

Every other measurement of the semantic layer — in `smoke_parse.py`, in the
matrix, in the plan — is **text-to-text**: preserved, covered, round-tripped,
idempotent. All of them score the document against the sentence it came from,
and none of them scores the sentence against the thing the app exists to make.
A feature can pass every one of them and be worth nothing, which is what
happened: the incumbent produced zero genuinely invented words across 27
fragments and scored as maximum restraint.

So this renders the same fragment twice on the live deployment — once as the
person typed it, once as the document compiles it — at one seed, one size, one
sampler, with the sentence as the only variable. It answers the founding
premise, which had never been tested: that an interpreted fragment beats a bare
one.

    python3.11 tools/does_it_help.py --url https://…modal.run          # the pairs below
    python3.11 tools/does_it_help.py --url … --from enrich.jsonl       # model output

`--from` takes `smoke_parse.py --enrich` output, which is the honest version:
the enriched half is then what the model wrote rather than what somebody hoped
it would write.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

# Hand-written, and only used when no dump is given. These are what a working
# interpreter would have said — written by hand because on 2026-08-17 no
# candidate produced anything to A/B against.
PAIRS = [
    ("diner", "empty diner, 3am",
     "An empty diner at 3am. Fluorescent tubes overhead, one of them flickering, "
     "throwing hard green light down the counter. Rain on the window laying "
     "streaks across the formica. Stools tucked under, nobody behind the "
     "register, a half-cleared plate at the far end. Shot from just inside the "
     "door at standing height, the counter running away to the right."),
    ("kitchen", "the kitchen after the party",
     "A kitchen after a party. Bottles and paper cups crowding the counter, an "
     "ashtray balanced on the hob, a bin bag slumped half-full by the door. One "
     "warm bulb still on over the sink and grey dawn coming through the blind "
     "behind it. Nobody in the room. Shot from the doorway at eye level, sharp "
     "front to back."),
]


def render(url: str, prompt: str, seed: int, tag: str) -> tuple[str, str] | None:
    """One image, and the job it came from. Everything but the sentence is fixed."""
    body = {"prompt": prompt, "modules": None, "negative_prompt": "",
            "model": "turbo", "shot": [], "loras": [], "regions": [],
            "region_weight": 1.0, "scene": None, "outfit": None,
            "width": 1152, "height": 864, "num_images": 1, "seed": seed,
            "sampler": "", "scheduler": "", "steps": "", "cfg_scale": "",
            "shift": "", "gpu": "H100"}
    req = urllib.request.Request(url + "/api/generate", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    job = r.get("job_id") or r.get("job")
    if not job:
        print(f"  {tag}: no job — {str(r)[:160]}")
        return None
    t0 = time.time()
    while time.time() - t0 < 900:
        st = json.load(urllib.request.urlopen(f"{url}/api/status/{job}", timeout=60))
        if st.get("status") in ("completed", "failed", "stopped"):
            files = st.get("files") or []
            print(f"  {tag}: {st['status']} in {time.time() - t0:.0f}s  {files}")
            return (job, files[0]) if files else None
        time.sleep(3)
    print(f"  {tag}: timed out")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="The deployed web URL")
    ap.add_argument("--seed", type=int, default=774411)
    ap.add_argument("--out", default="tools/parse-eval-2026-08-17",
                    help="Where the PNGs land")
    ap.add_argument("--from", dest="dump", default=None,
                    help="A `smoke_parse.py --enrich` JSONL — renders what the "
                         "model actually wrote instead of the pairs above.")
    ap.add_argument("--limit", type=int, default=4)
    args = ap.parse_args()

    pairs = PAIRS
    if args.dump:
        pairs = []
        for line in Path(args.dump).read_text().splitlines():
            if not line.startswith("ENRICH "):
                continue
            row = json.loads(line[7:])
            if row.get("compiled") and row["compiled"].strip() != row["prose"].strip():
                name = "".join(c if c.isalnum() else "_" for c in row["prose"])[:24]
                pairs.append((name, row["prose"], row["compiled"]))
        pairs = pairs[:args.limit]

    url = args.url.rstrip("/")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    got = {}
    for name, bare, rich in pairs:
        print(f"\n=== {name}\n  bare: {bare}\n  rich: {rich[:110]}…")
        for tag, prompt in (("bare", bare), ("rich", rich)):
            r = render(url, prompt, args.seed, tag)
            if not r:
                continue
            job, fn = r
            dest = out / f"{name}_{tag}.png"
            with urllib.request.urlopen(f"{url}/api/file/{job}/{fn}", timeout=300) as s:
                dest.write_bytes(s.read())
            got[f"{name}_{tag}"] = str(dest)
    print("\n" + json.dumps(got, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
