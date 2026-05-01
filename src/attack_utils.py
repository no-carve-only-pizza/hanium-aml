"""Shared utilities for face adversarial attack scripts."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights

from device import get_device

IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_face_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_face_model(checkpoint_path: Path) -> tuple[nn.Module, list[str], torch.device]:
    device = get_device()
    ckpt = torch.load(checkpoint_path, map_location=device)
    classes = ckpt["classes"]
    model = build_face_model(len(classes)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, classes, device


def imagenet_normalizer(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return mean, std


def pixel_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])


def model_input(pixel_tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (pixel_tensor - mean) / std


def load_test_samples(data_dir: Path, limit: int) -> list[tuple[str, int]]:
    ds = datasets.ImageFolder(data_dir / "test")
    samples = ds.samples[:limit] if limit > 0 else ds.samples
    if not samples:
        raise FileNotFoundError(f"No test images found under {data_dir / 'test'}")
    return samples


def load_clean_image(path: Path, to_pixel_tensor, device: torch.device) -> torch.Tensor:
    return to_pixel_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def predict_probs(model: nn.Module, pixel_tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return F.softmax(model(model_input(pixel_tensor, mean, std)), dim=1)


def target_for_label(true_label: int, target_class: int, num_classes: int) -> int:
    return target_class if target_class >= 0 else (true_label + 1) % num_classes


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    image = tensor.detach().cpu().clamp(0, 1).squeeze(0)
    transforms.ToPILImage()(image).save(path)


def tensor_norms(delta: torch.Tensor) -> tuple[int, float, float]:
    detached = delta.detach()
    l0 = int((detached.abs() > 1e-6).sum().cpu())
    l2 = float(detached.flatten().norm(p=2).cpu())
    linf = float(detached.abs().max().cpu())
    return l0, l2, linf


def safe_class_name(name: str) -> str:
    return name.replace("/", "_")


def base_attack_row(
    *,
    path: Path,
    adv_path: Path,
    perturb_path: Path,
    attack: str,
    classes: list[str],
    true_label: int,
    target_label: int,
    pred_before: int,
    pred_after: int,
    true_conf_before: float,
    true_conf_after: float,
    target_conf_before: float,
    target_conf_after: float,
    l0: int,
    l2: float,
    linf: float,
    time_sec: float,
) -> dict[str, object]:
    clean_correct = pred_before == true_label
    target_success = pred_after == target_label
    return {
        "file": str(path),
        "adv_file": str(adv_path),
        "perturbation_file": str(perturb_path),
        "attack": attack,
        "success": target_success,
        "clean_correct": clean_correct,
        "success_on_clean": clean_correct and target_success,
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
        "time_sec": time_sec,
    }


def write_metadata(rows: list[dict[str, object]], metadata_path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_attack_summary(rows: list[dict[str, object]], metadata_path: Path) -> None:
    success_rate = sum(bool(row["success"]) for row in rows) / len(rows)
    avg_gain = sum(float(row["target_conf_gain"]) for row in rows) / len(rows)
    print(f"Images: {len(rows)}")
    print(f"Target success rate: {success_rate:.2%}")
    print(f"Avg target confidence gain: {avg_gain:.4f}")
    print(f"Metadata: {metadata_path}")
