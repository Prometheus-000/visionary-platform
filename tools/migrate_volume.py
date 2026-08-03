"""
One-shot copy from the old forge-webui-repo volume into ours.

    modal run tools/migrate_volume.py                 # show the plan, copy nothing
    modal run tools/migrate_volume.py --apply         # copy the four weight files
    modal run tools/migrate_volume.py --apply --loras # bring trained LoRAs too

Both volumes are mounted into a single container, so the bytes never leave
Modal — a ~35 GB weight set moves at network-filesystem speed instead of being
pulled down and pushed back up. That is the only reason this is worth writing
rather than just re-downloading.

Nothing is deleted. The old volume is read-only here and stays exactly as it
is; delete it yourself once you are satisfied. This file is the only place the
old volume name survives, and it can be deleted with it.
"""

import shutil
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import VOLUME_NAME, volume  # noqa: E402

OLD_NAME = "forge-webui-repo"
old_volume = modal.Volume.from_name(OLD_NAME)  # no create_if_missing: it exists or we stop

migrate = modal.App("visionary-migrate")
image = modal.Image.debian_slim(python_version="3.11").add_local_python_source("app")

OLD = Path("/old")
NEW = Path("/new")

# Forge's layout on the left, ours on the right.
WEIGHTS = {
    "models/Stable-diffusion/raw.safetensors": "models/krea2-raw.safetensors",
    "models/Stable-diffusion/turbo.safetensors": "models/krea2-turbo.safetensors",
    "models/VAE/qwen_image_vae.safetensors": "models/qwen-image-vae.safetensors",
    "models/text_encoder/qwen3vl_4b_bf16.safetensors": "models/qwen3vl-4b-bf16.safetensors",
}
OLD_LORAS = "models/Lora"
NEW_LORAS = "loras"


def _plan(with_loras: bool) -> list[tuple[Path, Path, float]]:
    """Source, destination and GB for everything that needs copying."""
    jobs: list[tuple[Path, Path, float]] = []

    for src_rel, dst_rel in WEIGHTS.items():
        src, dst = OLD / src_rel, NEW / dst_rel
        if not src.is_file():
            print(f"  missing on {OLD_NAME}: {src_rel}")
            continue
        if dst.is_file() and dst.stat().st_size == src.stat().st_size:
            print(f"  already there: {dst_rel}")
            continue
        jobs.append((src, dst, src.stat().st_size / 1e9))

    if with_loras and (OLD / OLD_LORAS).is_dir():
        for src in sorted((OLD / OLD_LORAS).rglob("*.safetensors")):
            dst = NEW / NEW_LORAS / src.relative_to(OLD / OLD_LORAS)
            if dst.is_file() and dst.stat().st_size == src.stat().st_size:
                continue
            jobs.append((src, dst, src.stat().st_size / 1e9))

    return jobs


@migrate.function(
    image=image, cpu=4.0, timeout=4 * 60 * 60,
    volumes={str(OLD): old_volume, str(NEW): volume},
)
def run(apply: bool, with_loras: bool) -> dict:
    old_volume.reload()
    volume.reload()

    lora_count = len(list((OLD / OLD_LORAS).rglob("*.safetensors"))) if (OLD / OLD_LORAS).is_dir() else 0
    print(f"{OLD_NAME} -> {VOLUME_NAME}")
    print(f"LoRAs on the old volume: {lora_count}" + ("" if with_loras else "  (use --loras to bring them)"))

    jobs = _plan(with_loras)
    if not jobs:
        print("\nNothing to copy.")
        return {"copied": 0, "gb": 0.0}

    total = sum(gb for _, _, gb in jobs)
    print(f"\n{len(jobs)} file(s), {total:.1f} GB:")
    for src, dst, gb in jobs:
        print(f"  {src.relative_to(OLD)}  ->  {dst.relative_to(NEW)}  ({gb:.2f} GB)")

    if not apply:
        print("\nDry run. Re-run with --apply to copy.")
        return {"copied": 0, "gb": total, "dry_run": True}

    copied = 0.0
    for src, dst, gb in jobs:
        dst.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        # copyfile, not copy2: volume metadata is not worth preserving and
        # chmod on a mounted volume is a needless failure mode.
        shutil.copyfile(src, dst)
        copied += gb
        print(f"  {dst.relative_to(NEW)}  {gb:.2f} GB in {time.time() - started:.0f}s "
              f"({copied:.1f}/{total:.1f} GB)")

    volume.commit()
    print(f"\nDone. {len(jobs)} file(s), {copied:.1f} GB. {OLD_NAME} untouched.")
    return {"copied": len(jobs), "gb": round(copied, 1)}


@migrate.local_entrypoint()
def main(apply: bool = False, loras: bool = False):
    print(run.remote(apply, loras))
