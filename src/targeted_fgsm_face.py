"""Run targeted FGSM against the trained LFW identity classifier."""

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


def fgsm_targeted(model: nn.Module, image: torch.Tensor, target: torch.Tensor, epsilon: float) -> torch.Tensor:
    image = image.clone().detach().requires_grad_(True)
    logits = model(image)
    loss = F.cross_entropy(logits, target)
    model.zero_grad(set_to_none=True)
    loss.backward()
    adv = image - epsilon * image.grad.sign()
    return adv.detach().clamp(0, 1)


def tensor_norms(delta: torch.Tensor) -> tuple[int, float, float]:
    detached = delta.detach()
    l0 = int((detached.abs() > 1e-6).sum().cpu())
    l2 = float(detached.flatten().norm(p=2).cpu())
    linf = float(detached.abs().max().cpu())
    return l0, l2, linf


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted FGSM for trained face identity classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/lfw_identity_10"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/face_resnet50_lfw10/best.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/attacks/fgsm_face"))
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--target-class", type=int, default=-1, help="-1 means cyclic target: (true_label + 1) % num_classes")
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    classes = ckpt["classes"]
    model = build_model(len(classes)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Attack in normalized model space; save a denormalized copy for inspection.
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(args.data_dir / "test", transform=eval_tf)

    image_dir = args.out_dir / "images"
    perturb_dir = args.out_dir / "perturbations"
    image_dir.mkdir(parents=True, exist_ok=True)
    perturb_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, (image, true_label) in enumerate(tqdm(ds.samples[: args.limit], desc="targeted face FGSM")):
        path = Path(image)
        pil = Image.open(path).convert("RGB")
        clean = eval_tf(pil).unsqueeze(0).to(device)
        true = torch.tensor([true_label], device=device)
        target_label = args.target_class if args.target_class >= 0 else (true_label + 1) % len(classes)
        target = torch.tensor([target_label], device=device)

        start = time.perf_counter()
        with torch.no_grad():
            clean_probs = F.softmax(model(clean), dim=1)
            pred_before = int(clean_probs.argmax(dim=1).item())
            true_conf_before = float(clean_probs[0, true_label].cpu())
            target_conf_before = float(clean_probs[0, target_label].cpu())

        adv = fgsm_targeted(model, clean, target, args.epsilon)
        with torch.no_grad():
            adv_probs = F.softmax(model(adv), dim=1)
            pred_after = int(adv_probs.argmax(dim=1).item())
            true_conf_after = float(adv_probs[0, true_label].cpu())
            target_conf_after = float(adv_probs[0, target_label].cpu())
        elapsed = time.perf_counter() - start

        clean_img = (clean * std + mean).clamp(0, 1)
        adv_img = (adv * std + mean).clamp(0, 1)
        delta = adv_img - clean_img
        visible_delta = (delta / (2 * args.epsilon)) + 0.5
        l0, l2, linf = tensor_norms(delta)

        stem = path.stem
        adv_path = image_dir / f"{stem}_to_{classes[target_label]}_eps{args.epsilon:.3f}.jpg"
        perturb_path = perturb_dir / f"{stem}_to_{classes[target_label]}_eps{args.epsilon:.3f}_perturbation.jpg"
        save_tensor_image(adv_img, adv_path)
        save_tensor_image(visible_delta, perturb_path)

        rows.append({
            "file": str(path),
            "adv_file": str(adv_path),
            "perturbation_file": str(perturb_path),
            "epsilon": args.epsilon,
            "success": pred_after == target_label,
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

    metadata_path = args.out_dir / f"metadata_targeted_eps{args.epsilon:.3f}.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    success_rate = sum(row["success"] for row in rows) / len(rows)
    print(f"Device: {device}")
    print(f"Images: {len(rows)}")
    print(f"Target success rate: {success_rate:.2%}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
