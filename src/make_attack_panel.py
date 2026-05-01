"""Create original / adversarial / perturbation comparison panels."""

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


def make_panel(original_path: Path, adv_path: Path, perturb_path: Path, out_path: Path, title: str) -> None:
    size = (224, 224)
    images = [
        Image.open(original_path).convert("RGB").resize(size),
        Image.open(adv_path).convert("RGB").resize(size),
        Image.open(perturb_path).convert("RGB").resize(size),
    ]
    labels = ["original", "targeted FGSM", "perturbation"]
    pad = 18
    label_h = 30
    title_h = 34
    width = size[0] * 3 + pad * 4
    height = title_h + size[1] + label_h + pad * 2
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = load_font(16)
    title_font = load_font(18)
    draw.text((pad, 8), title, fill="black", font=title_font)

    for idx, image in enumerate(images):
        x = pad + idx * (size[0] + pad)
        y = title_h + pad
        canvas.paste(image, (x, y))
        draw.text((x, y + size[1] + 6), labels[idx], fill="black", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make comparison panels from attack metadata.")
    parser.add_argument("--metadata", type=Path, default=Path("outputs/attacks/fgsm/metadata.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/attacks/fgsm/panels"))
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()

    with args.metadata.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {args.metadata}")

    made = 0
    for row in rows:
        if row.get("success") not in {"True", "true", "1"}:
            continue
        original = Path(row["file"])
        adv = Path(row["adv_file"])
        perturb = Path(row["perturbation_file"])
        title = f"eps={row['epsilon']} target={row.get('target_class_name', '')}"
        out = args.out_dir / f"panel_{made + 1:02d}.jpg"
        make_panel(original, adv, perturb, out, title)
        made += 1
        if made >= args.count:
            break

    print(f"Created {made} panels in {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
