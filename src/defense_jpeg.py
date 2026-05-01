"""Apply JPEG recompression defense to adversarial face images.

This baseline consumes `outputs/attacks/attack_index.csv`, applies JPEG
recompression to `adv_file`, re-runs the trained face classifier, and writes a
result CSV that can be joined back to attack metadata by `sample_id`.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from io import BytesIO
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from tqdm import tqdm

from device import get_device


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def jpeg_recompress(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=False)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def main() -> None:
    parser = argparse.ArgumentParser(description="JPEG recompression defense baseline.")
    parser.add_argument("--attack-index", type=Path, default=Path("outputs/attacks/attack_index.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/face_resnet50_lfw10/best.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/defenses/jpeg"))
    parser.add_argument("--quality", type=int, default=75)
    parser.add_argument("--limit", type=int, default=0, help="0 means use all filtered rows.")
    parser.add_argument("--only-success-on-clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attack-family", type=str, default="", help="Optional filter: fgsm, pgd, square, jsma.")
    args = parser.parse_args()

    if not 1 <= args.quality <= 100:
        raise ValueError("--quality must be between 1 and 100")

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    classes = ckpt["classes"]
    model = build_model(len(classes)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    attack_index = pd.read_csv(args.attack_index)
    if args.only_success_on_clean:
        attack_index = attack_index[
            bool_series(attack_index["clean_correct"]) & bool_series(attack_index["success_on_clean"])
        ]
    if args.attack_family:
        attack_index = attack_index[attack_index["attack_family"] == args.attack_family]
    if args.limit > 0:
        attack_index = attack_index.head(args.limit)
    if attack_index.empty:
        raise ValueError("No attack rows selected for JPEG defense")

    image_out_dir = args.out_dir / f"q{args.quality}" / "images"
    image_out_dir.mkdir(parents=True, exist_ok=True)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    to_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    def model_input(pixel_tensor: torch.Tensor) -> torch.Tensor:
        return (pixel_tensor - mean) / std

    rows: list[dict[str, object]] = []

    for _, attack in tqdm(attack_index.iterrows(), total=len(attack_index), desc=f"jpeg q={args.quality}"):
        adv_path = Path(str(attack["adv_file"]))
        if not adv_path.exists():
            # Keep the missing row visible in output rather than silently dropping it.
            rows.append({
                "sample_id": attack["sample_id"],
                "attack_family": attack["attack_family"],
                "attack": attack["attack"],
                "defense": "jpeg",
                "defense_params": json.dumps({"quality": args.quality}),
                "input_adv_file": str(adv_path),
                "defended_file": "",
                "pred_before_defense": attack.get("pred_after", ""),
                "pred_after_defense": "",
                "pred_after_defense_name": "",
                "target_label": attack["target_label"],
                "true_label": attack["true_label"],
                "attack_success_before_defense": attack["success_on_clean"],
                "attack_success_after_defense": "",
                "recovered": "",
                "target_conf_before_defense": attack.get("target_conf_after", ""),
                "target_conf_after_defense": "",
                "true_conf_after_defense": "",
                "defense_time_sec": "",
                "status": "missing_adv_file",
            })
            continue

        start = time.perf_counter()
        image = Image.open(adv_path).convert("RGB")
        defended = jpeg_recompress(image, args.quality)
        defended_path = image_out_dir / f"{attack['sample_id']}_jpeg_q{args.quality}.jpg"
        defended.save(defended_path, format="JPEG", quality=95)

        tensor = to_tensor(defended).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(model(model_input(tensor)), dim=1)
            pred_after_defense = int(probs.argmax(dim=1).item())
            target_label = int(attack["target_label"])
            true_label = int(attack["true_label"])
            target_conf_after = float(probs[0, target_label].cpu())
            true_conf_after = float(probs[0, true_label].cpu())
        elapsed = time.perf_counter() - start

        attack_success_after = pred_after_defense == target_label
        recovered = pred_after_defense == true_label

        rows.append({
            "sample_id": attack["sample_id"],
            "attack_family": attack["attack_family"],
            "attack": attack["attack"],
            "defense": "jpeg",
            "defense_params": json.dumps({"quality": args.quality}),
            "input_adv_file": str(adv_path),
            "defended_file": str(defended_path),
            "pred_before_defense": attack.get("pred_after", ""),
            "pred_after_defense": pred_after_defense,
            "pred_after_defense_name": classes[pred_after_defense],
            "target_label": target_label,
            "true_label": true_label,
            "attack_success_before_defense": attack["success_on_clean"],
            "attack_success_after_defense": attack_success_after,
            "recovered": recovered,
            "target_conf_before_defense": attack.get("target_conf_after", ""),
            "target_conf_after_defense": target_conf_after,
            "true_conf_after_defense": true_conf_after,
            "defense_time_sec": elapsed,
            "status": "ok",
        })

    result_path = args.out_dir / f"jpeg_results_q{args.quality}.csv"
    with result_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [row for row in rows if row["status"] == "ok"]
    if ok_rows:
        before_success = [row for row in ok_rows if str(row["attack_success_before_defense"]).lower() in {"true", "1"}]
        after_success_count = sum(bool(row["attack_success_after_defense"]) for row in before_success)
        recovered_count = sum(bool(row["recovered"]) for row in before_success)
        defense_success_rate = 1 - (after_success_count / len(before_success)) if before_success else 0.0
        recovery_rate = recovered_count / len(before_success) if before_success else 0.0
        avg_target_drop = sum(
            float(row["target_conf_before_defense"]) - float(row["target_conf_after_defense"])
            for row in ok_rows
        ) / len(ok_rows)
        avg_time = sum(float(row["defense_time_sec"]) for row in ok_rows) / len(ok_rows)
        print(f"Rows: {len(rows)} ok={len(ok_rows)}")
        print(f"Defense success rate: {defense_success_rate:.2%}")
        print(f"Recovery rate: {recovery_rate:.2%}")
        print(f"Avg target confidence drop: {avg_target_drop:.4f}")
        print(f"Avg defense time: {avg_time:.4f}s")
    else:
        print(f"Rows: {len(rows)} ok=0")
    print(f"Saved: {result_path.resolve()}")


if __name__ == "__main__":
    main()
