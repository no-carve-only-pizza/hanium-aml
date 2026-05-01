"""Build a small LFW identity-classification dataset.

The attack project needs a face identity classifier before targeted face attacks
make sense. This script selects the identities with the most images and writes
an ImageFolder-compatible dataset:

    data/processed/lfw_identity_10/
      train/<person>/*.jpg
      val/<person>/*.jpg
      test/<person>/*.jpg
      labels.json

ImageFolder can read this directly for ResNet-50 fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def collect_identity_images(raw_dir: Path) -> dict[str, list[Path]]:
    identities: dict[str, list[Path]] = {}
    for person_dir in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
        images = sorted(
            path for path in person_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        )
        if images:
            identities[person_dir.name] = images
    if not identities:
        raise FileNotFoundError(f"No identity folders found in {raw_dir}")
    return identities


def split_images(images: list[Path], train_ratio: float, val_ratio: float) -> dict[str, list[Path]]:
    n = len(images)
    train_end = max(1, int(n * train_ratio))
    val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))
    val_end = min(val_end, n - 1) if n >= 3 else train_end
    return {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare top-N LFW identities for classification.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/lfw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/lfw_identity_10"))
    parser.add_argument("--num-identities", type=int, default=10)
    parser.add_argument("--min-images", type=int, default=20)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-class", type=int, default=0, help="0 means use all selected images.")
    args = parser.parse_args()

    raw_dir = args.raw_dir.resolve()
    out_dir = args.out_dir.resolve()
    identities = collect_identity_images(raw_dir)

    selected = [
        (person, images)
        for person, images in sorted(identities.items(), key=lambda item: len(item[1]), reverse=True)
        if len(images) >= args.min_images
    ][: args.num_identities]
    if len(selected) < args.num_identities:
        raise ValueError(
            f"Only found {len(selected)} identities with >= {args.min_images} images; "
            f"lower --min-images or --num-identities."
        )

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ["train", "val", "test"]:
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    # torchvision.datasets.ImageFolder assigns labels by alphabetically sorted folder names.
    # Keep labels.json in the same order to avoid confusion during targeted attacks.
    class_names = sorted(person for person, _ in selected)
    labels = {person: idx for idx, person in enumerate(class_names)}
    image_count_rank = {person: rank for rank, (person, _) in enumerate(selected, start=1)}
    summary: list[dict[str, object]] = []

    for person, images in selected:
        images = images[:]
        rng.shuffle(images)
        if args.limit_per_class > 0:
            images = images[: args.limit_per_class]
        splits = split_images(images, args.train_ratio, args.val_ratio)
        for split, split_images_list in splits.items():
            dst_dir = out_dir / split / person
            dst_dir.mkdir(parents=True, exist_ok=True)
            for src in split_images_list:
                shutil.copy2(src, dst_dir / src.name)
        summary.append({
            "label": labels[person],
            "image_count_rank": image_count_rank[person],
            "person": person,
            "total": len(images),
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        })

    labels_path = out_dir / "labels.json"
    labels_path.write_text(json.dumps({"class_to_idx": labels, "summary": summary}, indent=2), encoding="utf-8")

    print(f"Prepared dataset: {out_dir}")
    for row in summary:
        print(
            f"[{row['label']}] {row['person']}: "
            f"train={row['train']} val={row['val']} test={row['test']} total={row['total']}"
        )
    print(f"Labels: {labels_path}")


if __name__ == "__main__":
    main()
