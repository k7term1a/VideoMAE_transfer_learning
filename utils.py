from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
