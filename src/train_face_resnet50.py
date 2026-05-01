"""Fine-tune ResNet-50 on a small LFW identity dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights
from tqdm import tqdm

from device import get_device


def build_loaders(data_dir: Path, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader, DataLoader, datasets.ImageFolder]:
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, train_ds


def build_model(num_classes: int, freeze_backbone: bool) -> nn.Module:
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, optimizer=None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach().cpu()) * labels.size(0)
            total_correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu())
            total += labels.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet-50 face identity classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/lfw_identity_10"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/face_resnet50_lfw10"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    device = get_device()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader, train_ds = build_loaders(args.data_dir, args.batch_size, args.num_workers)
    model = build_model(num_classes=len(train_ds.classes), freeze_backbone=args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4)

    history_path = args.out_dir / "history.csv"
    best_path = args.out_dir / "best.pt"
    labels_path = args.out_dir / "labels.json"
    labels_path.write_text(json.dumps({"classes": train_ds.classes, "class_to_idx": train_ds.class_to_idx}, indent=2), encoding="utf-8")

    best_val_acc = -1.0
    rows = []
    print(f"Device: {device}")
    print(f"Classes: {len(train_ds.classes)}")
    print(f"Train batches: {len(train_loader)} Val batches: {len(val_loader)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        rows.append(row)
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} train_acc={train_acc:.2%} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2%}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": train_ds.classes,
                "class_to_idx": train_ds.class_to_idx,
                "freeze_backbone": args.freeze_backbone,
            }, best_path)
            print(f"saved best: {best_path}")

    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device)
    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps({
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "classes": train_ds.classes,
    }, indent=2), encoding="utf-8")
    print(f"Test loss={test_loss:.4f} test_acc={test_acc:.2%}")
    print(f"Checkpoint: {best_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
