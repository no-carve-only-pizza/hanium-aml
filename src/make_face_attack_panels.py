"""Create visual comparison panels from attack metadata.

Each panel shows original, adversarial, and perturbation for successful attacks.
Run this separately for FGSM and PGD metadata files.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def make_panel(row: dict[str, str], out_path: Path) -> None:
    size = (224, 224)
    original = Image.open(row["file"]).convert("RGB").resize(size)
    adv = Image.open(row["adv_file"]).convert("RGB").resize(size)
    perturb = Image.open(row["perturbation_file"]).convert("RGB").resize(size)

    pad = 18
    title_h = 42
    label_h = 32
    width = size[0] * 3 + pad * 4
    height = title_h + size[1] + label_h + pad * 2
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(16)
    font = load_font(14)

    attack_name = row.get("attack", "attack")
    if row.get("epsilon") not in {None, ""}:
        param_text = f"eps={row['epsilon']}"
    elif row.get("theta") not in {None, ""}:
        param_text = f"theta={row['theta']}"
    else:
        param_text = "params=n/a"
    title = (
        f"{attack_name} {param_text} "
        f"{row['true_name']} -> {row['target_name']} | pred: {row['pred_before_name']} -> {row['pred_after_name']}"
    )
    draw.text((pad, 10), title[:110], fill="black", font=title_font)
    for idx, (image, label) in enumerate([
        (original, "original"),
        (adv, "adversarial"),
        (perturb, "perturbation"),
    ]):
        x = pad + idx * (size[0] + pad)
        y = title_h + pad
        canvas.paste(image, (x, y))
        draw.text((x, y + size[1] + 8), label, fill="black", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create attack visual panels.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/attack_panels"))
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--only-clean", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    with args.metadata.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    made = 0
    for row in rows:
        success = str(row.get("success", "")).lower() in {"true", "1", "yes"}
        clean_correct = row.get("clean_correct")
        if clean_correct in {None, ""}:
            clean_correct = row.get("pred_before") == row.get("true_label")
        else:
            clean_correct = str(clean_correct).lower() in {"true", "1", "yes"}
        if not success or (args.only_clean and not clean_correct):
            continue
        make_panel(row, args.out_dir / f"panel_{made + 1:02d}.jpg")
        made += 1
        if made >= args.count:
            break

    print(f"Created {made} panels in {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
