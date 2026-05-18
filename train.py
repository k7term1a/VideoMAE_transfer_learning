import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from dataset import build_loaders
from model import build_model
from utils import compute_metrics


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_run_name(head: str, shot: str, epochs: int, lora_rank: int) -> str:
    prefix = f"lora{lora_rank}" if head == 'lora' else head
    return f"{prefix}_{shot}_{epochs}e"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VideoMAEv2 transfer learning")
    p.add_argument('--head',       required=True, choices=['linear', 'mlp', 'lora'])
    p.add_argument('--shot',       required=True,
                   help="Samples per class for training: 20 | 40 | full")
    p.add_argument('--epochs',     type=int, default=30)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lora_rank',  type=int, default=16)
    p.add_argument('--run_name',   default=None,
                   help="Override auto-generated output directory name")
    return p.parse_args()


def train_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler:    GradScaler,
    device:    torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device)
        labels       = labels.to(device)

        optimizer.zero_grad()
        with autocast():
            logits = model(pixel_values)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        bs             = labels.size(0)
        total_loss    += loss.item() * bs
        total_correct += (logits.argmax(1) == labels).sum().item()
        total         += bs

    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    all_preds, all_labels            = [], []

    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device)
        labels       = labels.to(device)

        with autocast():
            logits = model(pixel_values)
            loss   = criterion(logits, labels)

        bs             = labels.size(0)
        total_loss    += loss.item() * bs
        preds          = logits.argmax(1)
        total_correct += (preds == labels).sum().item()
        total         += bs
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_preds, all_labels)
    return total_loss / total, metrics['top1_acc'], metrics['mca']


def main() -> None:
    args = parse_args()
    set_seed(42)

    shot       = None if args.shot == 'full' else int(args.shot)
    run_name   = args.run_name or make_run_name(args.head, args.shot, args.epochs, args.lora_rank)
    output_dir = Path('runs') / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Run: {run_name} | Device: {device}")

    train_loader, valid_loader, _, class_to_idx, split_indices = \
        build_loaders(shot=shot, batch_size=args.batch_size)

    num_classes = len(class_to_idx)
    print(f"Classes: {num_classes} | Train samples: {len(train_loader.dataset)}")

    if split_indices is not None:
        with open(output_dir / 'split_index.json', 'w') as f:
            json.dump({'split_indices': split_indices, 'class_to_idx': class_to_idx}, f, indent=2)

    model = build_model(args.head, num_classes, lora_rank=args.lora_rank).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total_p:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = GradScaler()

    log_path = output_dir / 'train_log.csv'
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'val_mca'])

    best_val_acc     = 0.0
    patience_counter = 0
    PATIENCE         = 10

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss, _         = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, val_acc, val_mca = evaluate(model, valid_loader, criterion, device)
        scheduler.step()
        elapsed = time.perf_counter() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs} "
            f"| train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} "
            f"| val_acc={val_acc:.4f} "
            f"| val_mca={val_mca:.4f} "
            f"| {elapsed:.0f}s"
        )

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_acc, val_mca])

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            patience_counter = 0
            torch.save(
                {
                    'model_state':  model.state_dict(),
                    'class_to_idx': class_to_idx,
                    'head':         args.head,
                    'lora_rank':    args.lora_rank,
                    'epoch':        epoch,
                },
                output_dir / 'best.pt',
            )
            print(f"  ↑ new best ({best_val_acc:.4f}), checkpoint saved")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    print(f"\nDone. Best val_acc: {best_val_acc:.4f} | Output: {output_dir}")


if __name__ == '__main__':
    main()
