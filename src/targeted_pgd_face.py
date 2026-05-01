"""Run targeted PGD against the trained LFW identity classifier."""

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
    predict_probs,
    print_attack_summary,
    safe_class_name,
    save_tensor_image,
    target_for_label,
    tensor_norms,
    write_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted PGD for trained face identity classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/lfw_identity_10"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/face_resnet50_lfw10/best.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/attacks/pgd_face"))
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=0.003)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--random-start", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target-class", type=int, default=-1, help="-1 means cyclic target: (true_label + 1) % num_classes")
    args = parser.parse_args()

    model, classes, device = load_face_model(args.checkpoint)
    mean, std = imagenet_normalizer(device)
    to_pixel_tensor = pixel_transform()
    samples = load_test_samples(args.data_dir, args.limit)

    image_dir = args.out_dir / "images"
    perturb_dir = args.out_dir / "perturbations"
    image_dir.mkdir(parents=True, exist_ok=True)
    perturb_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for image, true_label in tqdm(samples, desc="targeted face PGD"):
        path = Path(image)
        clean = load_clean_image(path, to_pixel_tensor, device)
        target_label = target_for_label(true_label, args.target_class, len(classes))
        target = torch.tensor([target_label], device=device)

        start = time.perf_counter()
        clean_probs = predict_probs(model, clean, mean, std)
        pred_before = int(clean_probs.argmax(dim=1).item())
        true_conf_before = float(clean_probs[0, true_label].cpu())
        target_conf_before = float(clean_probs[0, target_label].cpu())

        if args.random_start:
            adv = (clean + torch.empty_like(clean).uniform_(-args.epsilon, args.epsilon)).clamp(0, 1)
        else:
            adv = clean.clone().detach()

        for _ in range(args.steps):
            adv = adv.clone().detach().requires_grad_(True)
            logits = model(model_input(adv, mean, std))
            loss = F.cross_entropy(logits, target)
            model.zero_grad(set_to_none=True)
            loss.backward()
            adv = adv - args.alpha * adv.grad.sign()
            delta = torch.clamp(adv - clean, min=-args.epsilon, max=args.epsilon)
            adv = (clean + delta).detach().clamp(0, 1)

        adv_probs = predict_probs(model, adv, mean, std)
        pred_after = int(adv_probs.argmax(dim=1).item())
        true_conf_after = float(adv_probs[0, true_label].cpu())
        target_conf_after = float(adv_probs[0, target_label].cpu())
        elapsed = time.perf_counter() - start

        delta = adv - clean
        visible_delta = (delta / (2 * args.epsilon)) + 0.5
        l0, l2, linf = tensor_norms(delta)

        stem = path.stem
        target_name_for_file = safe_class_name(classes[target_label])
        suffix = f"to_{target_name_for_file}_eps{args.epsilon:.3f}_a{args.alpha:.3f}_s{args.steps}"
        adv_path = image_dir / f"{stem}_{suffix}.jpg"
        perturb_path = perturb_dir / f"{stem}_{suffix}_perturbation.jpg"
        save_tensor_image(adv, adv_path)
        save_tensor_image(visible_delta, perturb_path)

        row = base_attack_row(
            path=path,
            adv_path=adv_path,
            perturb_path=perturb_path,
            attack="targeted_pgd",
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
            "alpha": args.alpha,
            "steps": args.steps,
            "random_start": args.random_start,
        })
        rows.append(row)

    metadata_path = args.out_dir / f"metadata_targeted_eps{args.epsilon:.3f}_alpha{args.alpha:.3f}_steps{args.steps}.csv"
    write_metadata(rows, metadata_path)
    print(f"Device: {device}")
    print_attack_summary(rows, metadata_path)


if __name__ == "__main__":
    main()
