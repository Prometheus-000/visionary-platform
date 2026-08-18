"""Does enrichment make the picture better? Nobody has ever asked.

Every measurement in this repo and in this session is text-to-text: preserved,
covered, round-tripped, idempotent. The product question is text-to-image, and
it has never been run — which means the feature's founding premise, that an
interpreted fragment beats a bare one, is untested.

Same seed, same size, same sampler. The only variable is the sentence.
"""
import json, sys, time, urllib.request

URL = "https://deeepux--visionary-web.modal.run"
SEED = 774411

# Half-baked in exactly the sense that matters: a person types this and hopes
# the machine has an idea. The enriched half is hand-written, because **no
# candidate produces one** — the incumbent supplied zero genuinely invented
# words across 27 fragments, so there is nothing to A/B except what a working
# interpreter would have said.
PAIRS = [
    ("diner",
     "empty diner, 3am",
     "An empty diner at 3am. Fluorescent tubes overhead, one of them flickering, "
     "throwing hard green light down the counter. Rain on the window laying "
     "streaks across the formica. Stools tucked under, nobody behind the "
     "register, a half-cleared plate at the far end. Shot from just inside the "
     "door at standing height, the counter running away to the right."),
    ("kitchen",
     "the kitchen after the party",
     "A kitchen after a party. Bottles and paper cups crowding the counter, an "
     "ashtray balanced on the hob, a bin bag slumped half-full by the door. One "
     "warm bulb still on over the sink and grey dawn coming through the blind "
     "behind it. Nobody in the room. Shot from the doorway at eye level, sharp "
     "front to back."),
]

def post(path, body, timeout=300):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def render(prompt, tag):
    body = {"prompt": prompt, "modules": None, "negative_prompt": "",
            "model": "turbo", "shot": [], "loras": [], "regions": [],
            "region_weight": 1.0, "scene": None, "outfit": None,
            "width": 1152, "height": 864, "num_images": 1, "seed": SEED,
            "sampler": "", "scheduler": "", "steps": "", "cfg_scale": "",
            "shift": "", "gpu": "H100"}
    r = post("/api/generate", body)
    job = r.get("job_id") or r.get("job")
    if not job:
        print(f"  {tag}: no job — {str(r)[:160]}"); return None
    t0 = time.time()
    while time.time() - t0 < 900:
        st = json.load(urllib.request.urlopen(f"{URL}/api/status/{job}", timeout=60))
        if st.get("status") in ("completed", "failed", "stopped"):
            files = st.get("files") or []
            print(f"  {tag}: {st['status']} in {time.time()-t0:.0f}s  {files}")
            return (job, files[0]) if files else None
        time.sleep(3)
    print(f"  {tag}: timed out"); return None

out = {}
for name, bare, rich in PAIRS:
    print(f"\n=== {name}")
    for tag, prompt in (("bare", bare), ("rich", rich)):
        got = render(prompt, tag)
        if got: out[f"{name}_{tag}"] = got
print("\n" + json.dumps(out))
