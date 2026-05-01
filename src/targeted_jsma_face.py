"""Run a targeted JSMA-style attack against the trained LFW identity classifier.

This is a practical multi-pixel JSMA variant for 224x224 RGB images. Instead of
classic pairwise JSMA, which is very expensive at this resolution, each step
selects the top saliency pixels from the target-loss gradient and updates them
toward the target class. This matches the project plan's modified JSMA direction:
changing multiple pixels per iteration to improve attack efficiency.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from attack_utils import (
    base_attack_row,
    imagenet_normalizer,
    load_clean_image,
    load_face_model,
    load_test_samples,
    model_input,
    pixel_transform,
    print_attack_summary,
    safe_class_name,
    save_tensor_image,
    target_for_label,
    tensor_norms,
    write_metadata,
)


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

    model, classes, device = load_face_model(args.checkpoint)
    mean, std = imagenet_normalizer(device)
    to_pixel_tensor = pixel_transform()

    image_dir = args.out_dir / "images"
    perturb_dir = args.out_dir / "perturbations"
    image_dir.mkdir(parents=True, exist_ok=True)
    perturb_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    samples = load_test_samples(args.data_dir, args.limit)

    for image, true_label in tqdm(samples, desc="targeted face JSMA"):
        path = Path(image)
        clean = load_clean_image(path, to_pixel_tensor, device)
        target_label = target_for_label(true_label, args.target_class, len(classes))
        target = torch.tensor([target_label], device=device)
        max_changed = max(1, int(clean.numel() * args.max_pixel_ratio))

        start = time.perf_counter()
        with torch.no_grad():
            clean_probs = F.softmax(model(model_input(clean, mean, std)), dim=1)
            pred_before = int(clean_probs.argmax(dim=1).item())
            true_conf_before = float(clean_probs[0, true_label].cpu())
            target_conf_before = float(clean_probs[0, target_label].cpu())

        adv = clean.clone().detach()
        changed = torch.zeros_like(clean, dtype=torch.bool)
        steps_used = 0

        for step in range(args.steps):
            with torch.no_grad():
                current_pred = int(F.softmax(model(model_input(adv, mean, std)), dim=1).argmax(dim=1).item())
            if current_pred == target_label or int(changed.sum().cpu()) >= max_changed:
                break

            attack_image = adv.clone().detach().requires_grad_(True)
            logits = model(model_input(attack_image, mean, std))
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
            adv_probs = F.softmax(model(model_input(adv, mean, std)), dim=1)
            pred_after = int(adv_probs.argmax(dim=1).item())
            true_conf_after = float(adv_probs[0, true_label].cpu())
            target_conf_after = float(adv_probs[0, target_label].cpu())
        elapsed = time.perf_counter() - start

        delta = adv - clean
        visible_delta = (delta / (2 * max(args.theta, 1e-8))) + 0.5
        l0, l2, linf = tensor_norms(delta)
        stem = path.stem
        target_name_for_file = safe_class_name(classes[target_label])
        suffix = f"to_{target_name_for_file}_theta{args.theta:.3f}_s{args.steps}_k{args.pixels_per_step}"
        adv_path = image_dir / f"{stem}_{suffix}.jpg"
        perturb_path = perturb_dir / f"{stem}_{suffix}_perturbation.jpg"
        save_tensor_image(adv, adv_path)
        save_tensor_image(visible_delta, perturb_path)

        row = base_attack_row(
            path=path,
            adv_path=adv_path,
            perturb_path=perturb_path,
            attack="targeted_jsma_multi_pixel",
            classes=classes,
            true_label=true_label,
            target_label=target_label,
            pred_before=pred_before,
            pred_after=pred_after,
            true_conf_before=true_conf_before,
            true_conf_after=true_conf_after,
            target_conf_before=target_conf_before,
            target_conf_after=target_conf_after,
            l0=l0,
            l2=l2,
            linf=linf,
            time_sec=elapsed,
        )
        row.update({
            "theta": args.theta,
            "steps": args.steps,
            "steps_used": steps_used,
            "pixels_per_step": args.pixels_per_step,
            "max_pixel_ratio": args.max_pixel_ratio,
            "changed_channels": int(changed.sum().cpu()),
        })
        rows.append(row)

    metadata_path = args.out_dir / f"metadata_targeted_theta{args.theta:.3f}_steps{args.steps}_k{args.pixels_per_step}.csv"
    write_metadata(rows, metadata_path)
    clean_count = sum(bool(row["clean_correct"]) for row in rows)
    success_on_clean = sum(bool(row["success_on_clean"]) for row in rows)
    avg_l0 = sum(int(row["l0"]) for row in rows) / len(rows)
    print(f"Device: {device}")
    print_attack_summary(rows, metadata_path)
    print(f"Target success rate on clean: {success_on_clean / clean_count:.2%}" if clean_count else "No clean-correct samples")
    print(f"Avg L0 changed channels: {avg_l0:.1f}")


if __name__ == "__main__":
    main()
