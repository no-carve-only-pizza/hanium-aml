"""Create a small LFW sample set for quick attack experiments.

This copies images from data/raw/lfw into data/samples/lfw_100.
We keep a tiny sample first so FGSM/debug runs finish quickly.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def collect_images(raw_dir: Path) -> list[Path]:
    images = sorted(
        path for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        raise FileNotFoundError(f"No images found under {raw_dir}")
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy a deterministic sample from LFW.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/lfw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/samples/lfw_100"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = args.raw_dir.resolve()
    out_dir = args.out_dir.resolve()
    images = collect_images(raw_dir)

    rng = random.Random(args.seed)
    sample = rng.sample(images, k=min(args.count, len(images)))

    out_dir.mkdir(parents=True, exist_ok=True)
    for old_file in out_dir.glob("*.jpg"):
        old_file.unlink()

    manifest_rows = ["sample_id,person,source_path,sample_path"]
    for idx, src in enumerate(sample, start=1):
        person = src.parent.name
        dst = out_dir / f"{idx:04d}_{person}_{src.name}"
        shutil.copy2(src, dst)
        manifest_rows.append(f"{idx:04d},{person},{src},{dst}")

    manifest_path = out_dir / "manifest.csv"
    manifest_path.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")

    print(f"Copied {len(sample)} images")
    print(f"Sample dir: {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
