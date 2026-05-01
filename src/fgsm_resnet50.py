"""Run FGSM adversarial attack experiments on LFW sample images.

Beginner notes:
- The project plan names ResNet-50 as the baseline classifier.
- This script uses ImageNet pretrained ResNet-50 because a face identity model is not ready yet.
- By default it runs targeted FGSM toward one ImageNet target class.
- Later, the same targeted structure should point at a face identity classifier target label.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def safe_name(path: Path) -> str:
    return path.stem.replace(" ", "_")


def linf_norm(delta: torch.Tensor) -> float:
    return float(delta.detach().abs().max().cpu())


def l2_norm(delta: torch.Tensor) -> float:
    return float(delta.detach().flatten().norm(p=2).cpu())


def l0_norm(delta: torch.Tensor, threshold: float = 1e-6) -> int:
    return int((delta.detach().abs() > threshold).sum().cpu())


def confidence_margin(probs: torch.Tensor) -> float:
    top2 = torch.topk(probs.detach(), k=2, dim=1).values[0]
    return float((top2[0] - top2[1]).cpu())


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    image = tensor.detach().cpu().clamp(0, 1).squeeze(0)
    transforms.ToPILImage()(image).save(path)


def fgsm_attack(
    model: torch.nn.Module,
    forward_fn,
    image: torch.Tensor,
    label: torch.Tensor,
    epsilon: float,
    targeted: bool,
) -> torch.Tensor:
    image = image.clone().detach().requires_grad_(True)
    logits = forward_fn(image)
    loss = F.cross_entropy(logits, label)
    model.zero_grad(set_to_none=True)
    loss.backward()

    # Untargeted FGSM maximizes the current-label loss.
    # Targeted FGSM minimizes the target-label loss, so the sign is reversed.
    direction = -1 if targeted else 1
    adv_image = image + direction * epsilon * image.grad.sign()
    return adv_image.detach().clamp(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="FGSM attack experiment with ImageNet ResNet-50.")
    parser.add_argument("--sample-dir", type=Path, default=Path("data/samples/lfw_100"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/attacks/fgsm"))
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--targeted", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-class-id", type=int, default=837)
    args = parser.parse_args()

    sample_dir = args.sample_dir.resolve()
    out_dir = args.out_dir.resolve()
    image_out_dir = out_dir / "images"
    perturbation_out_dir = out_dir / "perturbations"
    image_out_dir.mkdir(parents=True, exist_ok=True)
    perturbation_out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path for path in sample_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No sample images found in {sample_dir}. Run src/make_sample.py first.")

    device = choose_device()
    weights = ResNet50_Weights.DEFAULT
    categories = weights.meta["categories"]
    if not 0 <= args.target_class_id < len(categories):
        raise ValueError(f"--target-class-id must be between 0 and {len(categories) - 1}")
    target_name = categories[args.target_class_id]
    model = models.resnet50(weights=weights).to(device).eval()

    # Keep tensors in [0, 1] so epsilon is easy to interpret.
    # Normalize only at model input time.
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    def model_input(x: torch.Tensor) -> torch.Tensor:
        return normalize(x.squeeze(0)).unsqueeze(0)

    rows: list[dict[str, object]] = []

    for image_path in tqdm(image_paths, desc="FGSM"):
        pil_image = Image.open(image_path).convert("RGB")
        clean = preprocess(pil_image).unsqueeze(0).to(device)

        start = time.perf_counter()
        with torch.no_grad():
            clean_logits = model(model_input(clean))
            clean_probs = F.softmax(clean_logits, dim=1)
            clean_pred = int(clean_probs.argmax(dim=1).item())
            attack_label_id = args.target_class_id if args.targeted else clean_pred
            attack_label = torch.tensor([attack_label_id], device=device)
            clean_conf = float(clean_probs[0, clean_pred].detach().cpu())
            clean_target_conf = float(clean_probs[0, args.target_class_id].detach().cpu())
            clean_margin = confidence_margin(clean_probs)

        adv = fgsm_attack(
            model=model,
            forward_fn=lambda x: model(model_input(x)),
            image=clean,
            label=attack_label,
            epsilon=args.epsilon,
            targeted=args.targeted,
        )

        with torch.no_grad():
            adv_logits = model(model_input(adv))
            adv_probs = F.softmax(adv_logits, dim=1)
            adv_pred = int(adv_probs.argmax(dim=1).item())
            adv_conf = float(adv_probs[0, adv_pred].detach().cpu())
            adv_target_conf = float(adv_probs[0, args.target_class_id].detach().cpu())
            adv_margin = confidence_margin(adv_probs)
        elapsed = time.perf_counter() - start

        delta = adv - clean
        attack_success = adv_pred == args.target_class_id if args.targeted else adv_pred != clean_pred
        cmd = clean_margin - adv_margin

        stem = safe_name(image_path)
        mode = "targeted" if args.targeted else "untargeted"
        adv_path = image_out_dir / f"{stem}_fgsm_{mode}_eps{args.epsilon:.3f}.jpg"
        perturb_path = perturbation_out_dir / f"{stem}_fgsm_{mode}_eps{args.epsilon:.3f}_perturbation.jpg"
        save_tensor_image(adv, adv_path)

        # Make tiny perturbations visible by rescaling them for human inspection.
        visible_delta = (delta / (2 * args.epsilon)) + 0.5
        save_tensor_image(visible_delta, perturb_path)

        rows.append({
            "file": str(image_path),
            "adv_file": str(adv_path),
            "perturbation_file": str(perturb_path),
            "attack": "fgsm",
            "targeted": args.targeted,
            "epsilon": args.epsilon,
            "target_class_id": args.target_class_id if args.targeted else "",
            "target_class_name": target_name if args.targeted else "",
            "success": attack_success,
            "pred_before_id": clean_pred,
            "pred_before_name": categories[clean_pred],
            "pred_after_id": adv_pred,
            "pred_after_name": categories[adv_pred],
            "confidence_before": clean_conf,
            "confidence_after": adv_conf,
            "target_confidence_before": clean_target_conf if args.targeted else "",
            "target_confidence_after": adv_target_conf if args.targeted else "",
            "margin_before": clean_margin,
            "margin_after": adv_margin,
            "cmd": cmd,
            "l0": l0_norm(delta),
            "l2": l2_norm(delta),
            "linf": linf_norm(delta),
            "time_sec": elapsed,
        })

    mode = "targeted" if args.targeted else "untargeted"
    metadata_path = out_dir / f"metadata_{mode}_eps{args.epsilon:.3f}.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Keep this stable alias for quick inspection by notebooks or teammates.
    latest_path = out_dir / "metadata.csv"
    latest_path.write_text(metadata_path.read_text(encoding="utf-8"), encoding="utf-8")

    success_count = sum(1 for row in rows if row["success"])
    print("\n=== FGSM finished ===")
    print(f"Device: {device}")
    print(f"Images: {len(rows)}")
    print(f"Mode: {mode}")
    if args.targeted:
        print(f"Target: {args.target_class_id} ({target_name})")
    print(f"Attack success rate: {success_count / len(rows):.2%}")
    print(f"Metadata: {metadata_path}")
    print(f"Latest metadata alias: {latest_path}")
    print(f"Adversarial images: {image_out_dir}")


if __name__ == "__main__":
    main()
