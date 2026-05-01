"""Run a targeted ZOO-style black-box attack against the face classifier.

ZOO estimates gradients with finite differences instead of using backpropagation.
This implementation samples a subset of pixel channels each iteration and updates
only those coordinates toward the target class. It is intentionally simple and
practical for project-scale experiments.
"""

from __future__ import annotations

import argparse
import random
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
    predict_probs,
    print_attack_summary,
    safe_class_name,
    save_tensor_image,
    target_for_label,
    tensor_norms,
    write_metadata,
)


def target_loss(model, image: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, target_label: int) -> torch.Tensor:
    with torch.no_grad():
        probs = F.softmax(model(model_input(image, mean, std)), dim=1)
        return -torch.log(probs[:, target_label].clamp_min(1e-12))


def estimate_coordinate_gradient(
    model,
    adv: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    target_label: int,
    coord_indices: torch.Tensor,
    finite_diff_h: float,
    batch_size: int,
) -> tuple[torch.Tensor, int]:
    flat_size = adv.numel()
    grad_values = torch.zeros(coord_indices.numel(), device=adv.device)
    base_flat = adv.flatten()
    queries = 0

    for start in range(0, coord_indices.numel(), batch_size):
        coords = coord_indices[start:start + batch_size]
        plus = adv.repeat(coords.numel(), 1, 1, 1)
        minus = adv.repeat(coords.numel(), 1, 1, 1)
        plus_flat = plus.view(coords.numel(), flat_size)
        minus_flat = minus.view(coords.numel(), flat_size)
        row_idx = torch.arange(coords.numel(), device=adv.device)
        plus_flat[row_idx, coords] = (base_flat[coords] + finite_diff_h).clamp(0, 1)
        minus_flat[row_idx, coords] = (base_flat[coords] - finite_diff_h).clamp(0, 1)

        losses_plus = target_loss(model, plus, mean, std, target_label)
        losses_minus = target_loss(model, minus, mean, std, target_label)
        grad_values[start:start + coords.numel()] = (losses_plus - losses_minus) / (2 * finite_diff_h)
        queries += coords.numel() * 2

    return grad_values, queries


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted ZOO-style attack for trained face identity classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/lfw_identity_10"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/face_resnet50_lfw10/best.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/attacks/zoo_face"))
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument("--coords-per-iter", type=int, default=64)
    parser.add_argument("--fd-batch-size", type=int, default=64)
    parser.add_argument("--finite-diff-h", type=float, default=0.001)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-class", type=int, default=-1, help="-1 means cyclic target: (true_label + 1) % num_classes")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, classes, device = load_face_model(args.checkpoint)
    mean, std = imagenet_normalizer(device)
    to_pixel_tensor = pixel_transform()
    samples = load_test_samples(args.data_dir, args.limit)

    image_dir = args.out_dir / "images"
    perturb_dir = args.out_dir / "perturbations"
    image_dir.mkdir(parents=True, exist_ok=True)
    perturb_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for image, true_label in tqdm(samples, desc="targeted face ZOO"):
        path = Path(image)
        clean = load_clean_image(path, to_pixel_tensor, device)
        target_label = target_for_label(true_label, args.target_class, len(classes))

        start = time.perf_counter()
        clean_probs = predict_probs(model, clean, mean, std)
        pred_before = int(clean_probs.argmax(dim=1).item())
        true_conf_before = float(clean_probs[0, true_label].cpu())
        target_conf_before = float(clean_probs[0, target_label].cpu())

        adv = clean.clone().detach()
        flat_size = adv.numel()
        queries = 0
        iterations = 0
        while queries < args.max_queries:
            probs = predict_probs(model, adv, mean, std)
            queries += 1
            pred_current = int(probs.argmax(dim=1).item())
            if pred_current == target_label:
                break

            remaining_query_budget = args.max_queries - queries
            max_coords = max(1, remaining_query_budget // 2)
            coord_count = min(args.coords_per_iter, max_coords, flat_size)
            coord_indices = torch.randperm(flat_size, device=device)[:coord_count]
            grad_values, grad_queries = estimate_coordinate_gradient(
                model=model,
                adv=adv,
                mean=mean,
                std=std,
                target_label=target_label,
                coord_indices=coord_indices,
                finite_diff_h=args.finite_diff_h,
                batch_size=args.fd_batch_size,
            )
            queries += grad_queries

            flat_adv = adv.flatten()
            flat_clean = clean.flatten()
            flat_adv[coord_indices] = flat_adv[coord_indices] - args.learning_rate * grad_values.sign()
            delta = (flat_adv - flat_clean).clamp(-args.epsilon, args.epsilon)
            flat_adv[:] = (flat_clean + delta).clamp(0, 1)
            adv = flat_adv.view_as(adv).detach()
            iterations += 1

        adv_probs = predict_probs(model, adv, mean, std)
        queries += 1
        pred_after = int(adv_probs.argmax(dim=1).item())
        true_conf_after = float(adv_probs[0, true_label].cpu())
        target_conf_after = float(adv_probs[0, target_label].cpu())
        elapsed = time.perf_counter() - start

        delta = adv - clean
        visible_delta = (delta / (2 * args.epsilon)) + 0.5
        l0, l2, linf = tensor_norms(delta)

        stem = path.stem
        target_name_for_file = safe_class_name(classes[target_label])
        suffix = f"to_{target_name_for_file}_eps{args.epsilon:.3f}_q{args.max_queries}"
        adv_path = image_dir / f"{stem}_{suffix}.jpg"
        perturb_path = perturb_dir / f"{stem}_{suffix}_perturbation.jpg"
        save_tensor_image(adv, adv_path)
        save_tensor_image(visible_delta, perturb_path)

        row = base_attack_row(
            path=path,
            adv_path=adv_path,
            perturb_path=perturb_path,
            attack="targeted_zoo",
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
            "epsilon": args.epsilon,
            "max_queries": args.max_queries,
            "queries_used": queries,
            "coords_per_iter": args.coords_per_iter,
            "finite_diff_h": args.finite_diff_h,
            "learning_rate": args.learning_rate,
            "iterations": iterations,
        })
        rows.append(row)

    metadata_path = args.out_dir / f"metadata_targeted_eps{args.epsilon:.3f}_queries{args.max_queries}.csv"
    write_metadata(rows, metadata_path)
    clean_count = sum(bool(row["clean_correct"]) for row in rows)
    success_on_clean = sum(bool(row["success_on_clean"]) for row in rows)
    avg_queries = sum(int(row["queries_used"]) for row in rows) / len(rows)
    print(f"Device: {device}")
    print_attack_summary(rows, metadata_path)
    print(f"Target success rate on clean: {success_on_clean / clean_count:.2%}" if clean_count else "No clean-correct samples")
    print(f"Avg queries: {avg_queries:.1f}")


if __name__ == "__main__":
    main()
