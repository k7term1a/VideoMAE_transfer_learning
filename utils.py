import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_train_log(log_path: Path, save_path: Path | None = None) -> None:
    rows = list(csv.DictReader(open(log_path)))
    epochs     = [int(r['epoch'])      for r in rows]
    train_loss = [float(r['train_loss']) for r in rows]
    val_loss   = [float(r['val_loss'])   for r in rows]
    val_acc    = [float(r['val_acc'])    for r in rows]
    val_mca    = [float(r['val_mca'])    for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_loss, label='train_loss')
    ax1.plot(epochs, val_loss,   label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend()

    ax2.plot(epochs, val_acc, label='val_acc')
    ax2.plot(epochs, val_mca, label='val_mca')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()

    plt.tight_layout()
    out = save_path or Path(log_path).parent / 'train_curve.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_head_comparison(epochs: int = 30, runs_root: Path = Path('runs'),
                         save_path: Path | None = None,
                         backbone_tag: str = 'k710') -> None:
    heads = ['linear', 'mlp', 'lora16', 'lora32']
    shots = ['20', '40', 'full']
    labels = {'linear': 'Linear', 'mlp': 'MLP', 'lora16': 'LoRA-16', 'lora32': 'LoRA-32'}

    # acc[head][shot] = top1_acc or None
    acc: dict[str, dict[str, float | None]] = {h: {} for h in heads}
    for head in heads:
        for shot in shots:
            p = runs_root / f'{backbone_tag}_{head}_{shot}_{epochs}e' / 'eval_results.json'
            if p.exists():
                acc[head][shot] = json.loads(p.read_text())['top1_acc']
            else:
                acc[head][shot] = None

    x      = np.arange(len(shots))
    width  = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, head in enumerate(heads):
        values = [acc[head][s] if acc[head][s] is not None else 0 for s in shots]
        bars   = ax.bar(x + i * width, values, width, label=labels[head])
        for bar, shot in zip(bars, shots):
            if acc[head][shot] is not None:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f'{acc[head][shot]:.3f}',
                        ha='center', va='bottom', fontsize=8)

    ax.axhline(0.58, color='gray',   linestyle='--', linewidth=1.2, label='Gemma 4 (58%)')
    ax.axhline(0.65, color='tomato', linestyle='--', linewidth=1.2, label='Swin-LoRA (65%)')

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'{s}-shot' if s != 'full' else 'full' for s in shots])
    ax.set_ylabel('Top-1 Accuracy')
    ax.set_title(f'Head Comparison — Top-1 Acc ({epochs} epochs)')
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()

    out = save_path or runs_root / 'head_comparison.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('logs', nargs='*', help='train_log.csv paths')
    parser.add_argument('--compare',      action='store_true', help='plot head comparison')
    parser.add_argument('--epochs',       type=int, default=30)
    parser.add_argument('--backbone_tag', default='k710')
    args = parser.parse_args()

    if args.compare:
        plot_head_comparison(epochs=args.epochs, backbone_tag=args.backbone_tag)
    else:
        for path in args.logs:
            plot_train_log(Path(path))


def compute_metrics(preds: list[int], labels: list[int]) -> dict:
    n        = len(labels)
    top1_acc = sum(p == l for p, l in zip(preds, labels)) / n

    correct: dict[int, int] = defaultdict(int)
    total:   dict[int, int] = defaultdict(int)
    for p, l in zip(preds, labels):
        total[l] += 1
        if p == l:
            correct[l] += 1

    per_class_acc = {cls: correct[cls] / total[cls] for cls in total}
    mca           = float(np.mean(list(per_class_acc.values())))

    return {'top1_acc': top1_acc, 'mca': mca, 'per_class_acc': per_class_acc}


def plot_confusion_matrix(
    labels:      list[int],
    preds:       list[int],
    class_names: list[str],
    save_path:   Path,
) -> None:
    n      = len(class_names)
    matrix = np.zeros((n, n), dtype=int)
    for l, p in zip(labels, preds):
        matrix[l][p] += 1

    per_class_acc = matrix.diagonal() / matrix.sum(axis=1).clip(min=1)
    worst3        = set(np.argsort(per_class_acc)[:3].tolist())

    fig, ax = plt.subplots(figsize=(max(10, n * 0.65), max(8, n * 0.55)))
    im      = ax.imshow(matrix, cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.03)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(
        [f"{'★ ' if i in worst3 else ''}{c}" for i, c in enumerate(class_names)],
        fontsize=7,
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix  (★ = worst 3 classes)')

    thresh = matrix.max() / 2
    for i in range(n):
        for j in range(n):
            if matrix[i, j] > 0:
                ax.text(j, i, str(matrix[i, j]),
                        ha='center', va='center', fontsize=6,
                        color='white' if matrix[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
