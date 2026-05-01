"""Run a simple targeted Square Attack against the trained LFW identity classifier.

This is a black-box style attack: it only uses model output probabilities, not
gradients. The implementation optimizes square patch perturbations inside an
L-infinity epsilon bound around the original image.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights
from tqdm import tqdm

from device import get_device


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    image = tensor.detach().cpu().clamp(0, 1).squeeze(0)
    transforms.ToPILImage()(image).save(path)


def tensor_norms(delta: torch.Tensor) -> tuple[int, float, float]:
    detached = delta.detach()
    l0 = int((detached.abs() > 1e-6).sum().cpu())
    l2 = float(detached.flatten().norm(p=2).cpu())
    linf = float(detached.abs().max().cpu())
    return l0, l2, linf


def square_size(iteration: int, max_queries: int, image_size: int, p_init: float) -> int:
    # Gradually shrink the square area as queries progress.
    progress = iteration / max(max_queries, 1)
    p = p_init * (1.0 - progress) + 0.01 * progress
    area = max(1, int(p * image_size * image_size))
    return max(1, min(image_size, int(math.sqrt(area))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted Square Attack for trained face identity classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/lfw_identity_10"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/face_resnet50_lfw10/best.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/attacks/square_face"))
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--max-queries", type=int, default=300)
    parser.add_argument("--p-init", type=float, default=0.08)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-class", type=int, default=-1, help="-1 means cyclic target: (true_label + 1) % num_classes")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    classes = ckpt["classes"]
    model = build_model(len(classes)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    to_pixel_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    def model_input(pixel_tensor: torch.Tensor) -> torch.Tensor:
        return (pixel_tensor - mean) / std

    def predict_probs(pixel_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(model(model_input(pixel_tensor)), dim=1)

    ds = datasets.ImageFolder(args.data_dir / "test")
    image_dir = args.out_dir / "images"
    perturb_dir = args.out_dir / "perturbations"
    image_dir.mkdir(parents=True, exist_ok=True)
    perturb_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    samples = ds.samples[: args.limit]
    if not samples:
        raise FileNotFoundError(f"No test images found under {args.data_dir / 'test'}")

    for image, true_label in tqdm(samples, desc="targeted face Square"):
        path = Path(image)
        clean = to_pixel_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        _, _, h, w = clean.shape
        target_label = args.target_class if args.target_class >= 0 else (true_label + 1) % len(classes)

        start = time.perf_counter()
        clean_probs = predict_probs(clean)
        pred_before = int(clean_probs.argmax(dim=1).item())
        true_conf_before = float(clean_probs[0, true_label].cpu())
        target_conf_before = float(clean_probs[0, target_label].cpu())

        delta = torch.empty_like(clean).uniform_(-args.epsilon, args.epsilon)
        adv = (clean + delta).clamp(0, 1)
        adv_probs = predict_probs(adv)
        best_target_conf = float(adv_probs[0, target_label].cpu())
        pred_after = int(adv_probs.argmax(dim=1).item())
        queries = 1

        for query_idx in range(1, args.max_queries + 1):
            if pred_after == target_label:
                break
            size = square_size(query_idx, args.max_queries, h, args.p_init)
            top = random.randint(0, h - size)
            left = random.randint(0, w - size)

            candidate_delta = delta.clone()
            patch = torch.empty((1, 3, size, size), device=device).uniform_(-args.epsilon, args.epsilon)
            candidate_delta[:, :, top:top + size, left:left + size] = patch
            candidate_delta = candidate_delta.clamp(-args.epsilon, args.epsilon)
            candidate = (clean + candidate_delta).clamp(0, 1)
            candidate_probs = predict_probs(candidate)
            queries += 1
            candidate_target_conf = float(candidate_probs[0, target_label].cpu())
            if candidate_target_conf >= best_target_conf:
                delta = candidate_delta
                adv = candidate
                adv_probs = candidate_probs
                best_target_conf = candidate_target_conf
                pred_after = int(adv_probs.argmax(dim=1).item())

        true_conf_after = float(adv_probs[0, true_label].cpu())
        target_conf_after = float(adv_probs[0, target_label].cpu())
        elapsed = time.perf_counter() - start

        final_delta = adv - clean
        visible_delta = (final_delta / (2 * args.epsilon)) + 0.5
        l0, l2, linf = tensor_norms(final_delta)
        clean_correct = pred_before == true_label
        target_success = pred_after == target_label
        target_success_on_clean = clean_correct and target_success

        stem = path.stem
        target_name_for_file = classes[target_label].replace("/", "_")
        suffix = f"to_{target_name_for_file}_eps{args.epsilon:.3f}_q{args.max_queries}"
        adv_path = image_dir / f"{stem}_{suffix}.jpg"
        perturb_path = perturb_dir / f"{stem}_{suffix}_perturbation.jpg"
        save_tensor_image(adv, adv_path)
        save_tensor_image(visible_delta, perturb_path)

        rows.append({
            "file": str(path),
            "adv_file": str(adv_path),
            "perturbation_file": str(perturb_path),
            "attack": "targeted_square",
            "epsilon": args.epsilon,
            "max_queries": args.max_queries,
            "queries_used": queries,
            "p_init": args.p_init,
            "success": target_success,
            "clean_correct": clean_correct,
            "success_on_clean": target_success_on_clean,
            "true_label": true_label,
            "true_name": classes[true_label],
            "target_label": target_label,
            "target_name": classes[target_label],
            "pred_before": pred_before,
            "pred_before_name": classes[pred_before],
            "pred_after": pred_after,
            "pred_after_name": classes[pred_after],
            "true_conf_before": true_conf_before,
            "true_conf_after": true_conf_after,
            "target_conf_before": target_conf_before,
            "target_conf_after": target_conf_after,
            "target_conf_gain": target_conf_after - target_conf_before,
            "l0": l0,
            "l2": l2,
            "linf": linf,
            "time_sec": elapsed,
        })

    metadata_path = args.out_dir / f"metadata_targeted_eps{args.epsilon:.3f}_queries{args.max_queries}.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    success_rate = sum(row["success"] for row in rows) / len(rows)
    clean_count = sum(row["clean_correct"] for row in rows)
    success_on_clean = sum(row["success_on_clean"] for row in rows)
    avg_queries = sum(int(row["queries_used"]) for row in rows) / len(rows)
    print(f"Device: {device}")
    print(f"Images: {len(rows)}")
    print(f"Target success rate all: {success_rate:.2%}")
    print(f"Target success rate on clean: {success_on_clean / clean_count:.2%}" if clean_count else "No clean-correct samples")
    print(f"Avg queries: {avg_queries:.1f}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
