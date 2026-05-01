"""Run a simple targeted Square Attack against the trained LFW identity classifier.

This is a black-box style attack: it only uses model output probabilities, not
gradients. The implementation optimizes square patch perturbations inside an
L-infinity epsilon bound around the original image.
"""

from __future__ import annotations

import argparse
import math
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
    print_attack_summary,
    safe_class_name,
    save_tensor_image,
    target_for_label,
    tensor_norms,
    write_metadata,
)


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

    model, classes, device = load_face_model(args.checkpoint)
    mean, std = imagenet_normalizer(device)
    to_pixel_tensor = pixel_transform()

    def predict_probs(pixel_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(model(model_input(pixel_tensor, mean, std)), dim=1)

    image_dir = args.out_dir / "images"
    perturb_dir = args.out_dir / "perturbations"
    image_dir.mkdir(parents=True, exist_ok=True)
    perturb_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    samples = load_test_samples(args.data_dir, args.limit)

    for image, true_label in tqdm(samples, desc="targeted face Square"):
        path = Path(image)
        clean = load_clean_image(path, to_pixel_tensor, device)
        _, _, h, w = clean.shape
        target_label = target_for_label(true_label, args.target_class, len(classes))

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
            attack="targeted_square",
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
            "p_init": args.p_init,
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
