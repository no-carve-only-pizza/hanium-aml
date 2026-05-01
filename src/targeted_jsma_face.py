"""Run a targeted JSMA-style attack against the trained LFW identity classifier.

This is a practical multi-pixel JSMA variant for 224x224 RGB images. Instead of
classic pairwise JSMA, which is very expensive at this resolution, each step
selects the top saliency pixels from the target-loss gradient and updates them
toward the target class. This matches the project plan's modified JSMA direction:
changing multiple pixels per iteration to improve attack efficiency.
"""

from __future__ import annotations

import argparse
import csv
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted multi-pixel JSMA for trained face identity classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/lfw_identity_10"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/face_resnet50_lfw10/best.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/attacks/jsma_face"))
    parser.add_argument("--theta", type=float, default=0.05, help="Pixel update size per selected channel.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--pixels-per-step", type=int, default=200)
    parser.add_argument("--max-pixel-ratio", type=float, default=0.05, help="Stop after modifying this fraction of image channels.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--target-class", type=int, default=-1, help="-1 means cyclic target: (true_label + 1) % num_classes")
    args = parser.parse_args()

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

    ds = datasets.ImageFolder(args.data_dir / "test")
    image_dir = args.out_dir / "images"
    perturb_dir = args.out_dir / "perturbations"
    image_dir.mkdir(parents=True, exist_ok=True)
    perturb_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    samples = ds.samples[: args.limit]
    if not samples:
        raise FileNotFoundError(f"No test images found under {args.data_dir / 'test'}")

    for image, true_label in tqdm(samples, desc="targeted face JSMA"):
        path = Path(image)
        clean = to_pixel_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        target_label = args.target_class if args.target_class >= 0 else (true_label + 1) % len(classes)
        target = torch.tensor([target_label], device=device)
        max_changed = max(1, int(clean.numel() * args.max_pixel_ratio))

        start = time.perf_counter()
        with torch.no_grad():
            clean_probs = F.softmax(model(model_input(clean)), dim=1)
            pred_before = int(clean_probs.argmax(dim=1).item())
            true_conf_before = float(clean_probs[0, true_label].cpu())
            target_conf_before = float(clean_probs[0, target_label].cpu())

        adv = clean.clone().detach()
        changed = torch.zeros_like(clean, dtype=torch.bool)
        steps_used = 0

        for step in range(args.steps):
            with torch.no_grad():
                current_pred = int(F.softmax(model(model_input(adv)), dim=1).argmax(dim=1).item())
            if current_pred == target_label or int(changed.sum().cpu()) >= max_changed:
                break

            attack_image = adv.clone().detach().requires_grad_(True)
            logits = model(model_input(attack_image))
            loss = F.cross_entropy(logits, target)
            model.zero_grad(set_to_none=True)
            loss.backward()

            # Targeted attack: reduce target loss. Larger saliency means stronger
            # expected target-confidence gain for a small pixel move.
            grad = attack_image.grad.detach()
            direction = -grad.sign()
            can_change = ((direction > 0) & (adv < 1.0)) | ((direction < 0) & (adv > 0.0))
            can_change = can_change & (~changed)
            saliency = grad.abs().masked_fill(~can_change, -1.0)
            remaining_budget = max_changed - int(changed.sum().cpu())
            k = min(args.pixels_per_step, remaining_budget, int(can_change.sum().cpu()))
            if k <= 0:
                break

            flat_idx = torch.topk(saliency.flatten(), k=k).indices
            flat_adv = adv.flatten()
            flat_dir = direction.flatten()
            flat_changed = changed.flatten()
            flat_adv[flat_idx] = (flat_adv[flat_idx] + args.theta * flat_dir[flat_idx]).clamp(0, 1)
            flat_changed[flat_idx] = True
            adv = flat_adv.view_as(adv).detach()
            changed = flat_changed.view_as(changed)
            steps_used = step + 1

        with torch.no_grad():
            adv_probs = F.softmax(model(model_input(adv)), dim=1)
            pred_after = int(adv_probs.argmax(dim=1).item())
            true_conf_after = float(adv_probs[0, true_label].cpu())
            target_conf_after = float(adv_probs[0, target_label].cpu())
        elapsed = time.perf_counter() - start

        delta = adv - clean
        visible_delta = (delta / (2 * max(args.theta, 1e-8))) + 0.5
        l0, l2, linf = tensor_norms(delta)
        clean_correct = pred_before == true_label
        target_success = pred_after == target_label
        target_success_on_clean = clean_correct and target_success

        stem = path.stem
        target_name_for_file = classes[target_label].replace("/", "_")
        suffix = f"to_{target_name_for_file}_theta{args.theta:.3f}_s{args.steps}_k{args.pixels_per_step}"
        adv_path = image_dir / f"{stem}_{suffix}.jpg"
        perturb_path = perturb_dir / f"{stem}_{suffix}_perturbation.jpg"
        save_tensor_image(adv, adv_path)
        save_tensor_image(visible_delta, perturb_path)

        rows.append({
            "file": str(path),
            "adv_file": str(adv_path),
            "perturbation_file": str(perturb_path),
            "attack": "targeted_jsma_multi_pixel",
            "theta": args.theta,
            "steps": args.steps,
            "steps_used": steps_used,
            "pixels_per_step": args.pixels_per_step,
            "max_pixel_ratio": args.max_pixel_ratio,
            "changed_channels": int(changed.sum().cpu()),
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

    metadata_path = args.out_dir / f"metadata_targeted_theta{args.theta:.3f}_steps{args.steps}_k{args.pixels_per_step}.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    success_rate = sum(row["success"] for row in rows) / len(rows)
    clean_count = sum(row["clean_correct"] for row in rows)
    success_on_clean = sum(row["success_on_clean"] for row in rows)
    avg_l0 = sum(int(row["l0"]) for row in rows) / len(rows)
    print(f"Device: {device}")
    print(f"Images: {len(rows)}")
    print(f"Target success rate all: {success_rate:.2%}")
    print(f"Target success rate on clean: {success_on_clean / clean_count:.2%}" if clean_count else "No clean-correct samples")
    print(f"Avg L0 changed channels: {avg_l0:.1f}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
