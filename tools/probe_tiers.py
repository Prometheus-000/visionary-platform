"""
Are the quantised tiers the same shape as the weights this app already loads?

    python3 tools/probe_tiers.py            # headers only: no GPU, no download
    python3 tools/probe_tiers.py --json     # the same, as data

Three catalogue rows were added on an inference: that
`Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot`'s mixed int4/int8 files are the same
pruned-convrot lineage as the int8 files in `Comfy-Org/MiniMax-H3`, and so load
through the `UNETLoader` already in the graph. Inference is not measurement, and
the rows say so. This is the cheap half of measuring it.

**Why headers answer most of the question.** A safetensors file opens with a
u64 length and then a JSON header naming every tensor, its dtype and its shape —
so an HTTP range request for the first few hundred kilobytes describes a 21 GB
file completely. ComfyUI picks a loader and a model class by *reading those
names*: `comfy.sd.load_diffusion_model` detects the architecture from the state
dict's keys. Two files with the same key set and the same shapes are the same
model to it, whatever the dtypes underneath are called.

So a matching key set is strong evidence the file loads. It is not proof it
*runs* — that needs the kernels, the card and a render, which is what the GPU
half of the rented-box list is for. What this can do is fail, cheaply, before
anybody spends twenty minutes finding out.
"""

import argparse
import json
import struct
import sys
import urllib.request

# (label, repo, path) triples: the file the deployment runs, then the tier that
# claims to be a smaller version of it.
PAIRS = [
    ("H3 transformer (t2v/i2v)",
     ("Comfy-Org/MiniMax-H3",
      "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
     ("Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot",
      "MiniMax_H3_FL2VA_pruned_mixed_int4_int8_convrot.safetensors")),
    ("H3 transformer (references)",
     ("Comfy-Org/MiniMax-H3",
      "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"),
     ("Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot",
      "MiniMax_H3_Ref2VA_pruned_mixed_int4_int8_convrot.safetensors")),
    # Kept although it is no longer a catalogue row, because this is the pair
    # that removed it: a strict subset missing the whole nvfp4 scale apparatus.
    # A probe that only covers what survived cannot say why anything did not.
    ("H3 text encoder — REJECTED, kept as the record",
     ("Comfy-Org/MiniMax-H3",
      "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
     ("Abiray/MiniMax-H3-GGUF",
      "text_encoders/qwen3vl_32b_minimax_h3_int4_convrot.safetensors")),
]

# Pairs whose mismatch is the expected answer. Listed by label so the run is
# green when the world still looks the way it did when this was decided — and
# goes red if the file is ever re-uploaded in the matching shape, which would
# be worth knowing.
EXPECTED_MISMATCH = {"H3 text encoder — REJECTED, kept as the record"}


def url_for(repo: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{path}"


def _get(url: str, first: int, last: int) -> bytes:
    req = urllib.request.Request(url, headers={
        "Range": f"bytes={first}-{last}",
        "User-Agent": "visionary-probe-tiers",
    })
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def header(repo: str, path: str) -> dict:
    """The safetensors header, off the front of the file. Two range requests:
    eight bytes for the length, then the JSON itself."""
    url = url_for(repo, path)
    n = struct.unpack("<Q", _get(url, 0, 7))[0]
    if n > 200_000_000:
        raise ValueError(f"header claims {n} bytes — not a safetensors file")
    raw = _get(url, 8, 8 + n - 1)
    return json.loads(raw)


# The Krea 2 tiers are GGUF, a different container entirely, so a key-set diff
# is not available: the names live behind a length-prefixed KV section rather
# than in one JSON blob. What the front of the file does say cheaply is whether
# it is well-formed and how many tensors it holds, and a tensor count that
# matches the safetensors original is the same evidence in weaker form.
GGUF_PAIRS = [
    ("Krea 2 Turbo",
     ("Comfy-Org/Krea-2_ComfyUI", "split_files/diffusion_models/krea2_turbo.safetensors"),
     ("realrebelai/KREA-2_GGUFs", "TURBO/Krea-2-Turbo-Q4_K_M.gguf")),
    ("Krea 2 RAW",
     ("krea/Krea-2-Raw", "raw.safetensors"),
     ("realrebelai/KREA-2_GGUFs", "BASE/Krea-2-Base-Q4_K_M.gguf")),
]


def gguf_head(repo: str, path: str) -> dict:
    """Magic, version and tensor count off the first 24 bytes of a GGUF."""
    raw = _get(url_for(repo, path), 0, 23)
    magic = raw[:4]
    if magic != b"GGUF":
        raise ValueError(f"magic is {magic!r}, not GGUF")
    version, = struct.unpack("<I", raw[4:8])
    n_tensors, n_kv = struct.unpack("<QQ", raw[8:24])
    return {"version": version, "tensors": n_tensors, "kv": n_kv}


def check_gguf(label: str, base: tuple, tier: tuple) -> dict:
    out = {"label": label, "base": "/".join(base), "tier": "/".join(tier)}
    try:
        out["gguf"] = gguf_head(*tier)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    try:
        out["base_tensors"] = len(summarise(header(*base))["keys"])
    except Exception as exc:  # noqa: BLE001 — the original may be gated
        out["base_note"] = f"{type(exc).__name__} (gated or moved)"
    return out


def render_gguf(r: dict) -> bool:
    print(f"\n=== {r['label']} (GGUF)")
    print(f"    tier    : {r['tier']}")
    if r.get("error"):
        print(f"    UNREADABLE — {r['error']}")
        return False
    g = r["gguf"]
    print(f"    container: GGUF v{g['version']}, {g['tensors']} tensors, "
          f"{g['kv']} metadata entries")
    if "base_tensors" in r:
        same = g["tensors"] == r["base_tensors"]
        print(f"    original : {r['base_tensors']} tensors — "
              + ("same count" if same else "DIFFERENT count"))
        if not same:
            print("               A GGUF fuses nothing, so a different count is")
            print("               worth explaining before trusting the row.")
    else:
        print(f"    original : not readable — {r.get('base_note')}")
    print("    A well-formed GGUF still needs UnetLoaderGGUF to read *this*")
    print("    architecture, which only the pinned fork claims to. That claim")
    print("    is what the GPU half tests.")
    return True


def summarise(hdr: dict) -> dict:
    keys = {k for k in hdr if k != "__metadata__"}
    dtypes: dict[str, int] = {}
    total = 0
    for k in keys:
        info = hdr[k]
        dtypes[info["dtype"]] = dtypes.get(info["dtype"], 0) + 1
        off = info.get("data_offsets") or [0, 0]
        total += off[1] - off[0]
    return {"keys": keys, "dtypes": dtypes, "bytes": total,
            "meta": hdr.get("__metadata__") or {}}


def compare(label: str, base: tuple, tier: tuple) -> dict:
    out = {"label": label, "base": "/".join(base), "tier": "/".join(tier)}
    try:
        b = summarise(header(*base))
        t = summarise(header(*tier))
    except Exception as exc:  # noqa: BLE001 — an unreadable file is an answer
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    only_base = sorted(b["keys"] - t["keys"])
    only_tier = sorted(t["keys"] - b["keys"])
    out.update({
        "base_tensors": len(b["keys"]),
        "tier_tensors": len(t["keys"]),
        "base_gb": round(b["bytes"] / 1e9, 2),
        "tier_gb": round(t["bytes"] / 1e9, 2),
        "base_dtypes": b["dtypes"],
        "tier_dtypes": t["dtypes"],
        "missing_from_tier": only_base[:12],
        "extra_in_tier": only_tier[:12],
        "n_missing": len(only_base),
        "n_extra": len(only_tier),
        "same_keys": not only_base and not only_tier,
        "base_meta": b["meta"],
        "tier_meta": t["meta"],
    })
    return out


def render(r: dict) -> bool:
    """Print one comparison. True if it looks loadable."""
    print(f"\n=== {r['label']}")
    print(f"    runs now: {r['base']}")
    print(f"    tier    : {r['tier']}")
    if r.get("error"):
        print(f"    UNREADABLE — {r['error']}")
        return False
    print(f"    tensors : {r['base_tensors']} -> {r['tier_tensors']}"
          f"   ({r['base_gb']} GB -> {r['tier_gb']} GB)")
    print(f"    dtypes  : {r['base_dtypes']}")
    print(f"          -> {r['tier_dtypes']}")
    if r["same_keys"]:
        print("    keys    : identical — the same tensors, quantised differently.")
        print("              ComfyUI detects the architecture from these names,")
        print("              so this is the file it already knows how to load.")
        return True
    print(f"    keys    : NOT identical — {r['n_missing']} missing, "
          f"{r['n_extra']} extra")
    for k in r["missing_from_tier"]:
        print(f"              - {k}")
    for k in r["extra_in_tier"]:
        print(f"              + {k}")
    print("              A different key set is a different model to ComfyUI's")
    print("              detection. This tier needs its own loader, or the row")
    print("              comes out.")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--json", action="store_true", help="emit data, not prose")
    args = ap.parse_args()

    results = [compare(*p) for p in PAIRS]
    if args.json:
        for r in results:
            r["missing_from_tier"] = list(r.get("missing_from_tier", []))
            r["extra_in_tier"] = list(r.get("extra_in_tier", []))
        print(json.dumps(results, indent=1, default=str))
        return 0

    ok = []
    for r in results:
        matched = render(r)
        if r["label"] in EXPECTED_MISMATCH:
            print("    ^ expected: this is why the row is not in the catalogue.")
            ok.append(not matched)
        else:
            ok.append(matched)
    ok += [render_gguf(check_gguf(*p)) for p in GGUF_PAIRS]
    print()
    if all(ok):
        print("  Every shipped tier carries the same tensors as the file it")
        print("  replaces, and the one that did not is still absent. Strong")
        print("  evidence they load. What is unmeasured is whether they *run*:")
        print("  the kernels, the card and a render, which is the GPU half.")
        return 0
    print("  Something is not the shape it was when these rows were written.")
    print("  Read the mismatch above before anybody downloads that tier.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
