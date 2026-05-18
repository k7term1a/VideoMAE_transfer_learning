import argparse
import json
import time
from pathlib import Path

import torch
from torch.cuda.amp import autocast

from dataset import VideoDataset
from model import build_model
from utils import compute_metrics, plot_confusion_matrix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a VideoMAEv2 checkpoint on the test set")
    p.add_argument('--checkpoint', required=True, help="Path to best.pt")
    p.add_argument('--head',       required=True, choices=['linear', 'mlp', 'lora'])
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    ckpt_path  = Path(args.checkpoint)
    output_dir = ckpt_path.parent

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Checkpoint : {ckpt_path}")
    print(f"Device     : {device}")

    ckpt          = torch.load(ckpt_path, map_location=device, weights_only=False)
    class_to_idx  = ckpt['class_to_idx']
    lora_rank     = ckpt.get('lora_rank', 16)
    num_classes   = len(class_to_idx)
    idx_to_class  = {v: k for k, v in class_to_idx.items()}

    model = build_model(args.head, num_classes, lora_rank=lora_rank)
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()

    test_ds = VideoDataset('test', class_to_idx)
    print(f"Test samples: {len(test_ds)}")

    all_preds:  list[int]   = []
    all_labels: list[int]   = []
    times:      list[float] = []

    torch.cuda.reset_peak_memory_stats(device)

    for i in range(len(test_ds)):
        pixel_values, label = test_ds[i]
        pixel_values = pixel_values.unsqueeze(0).to(device)  # (1, T, C, H, W)

        t0 = time.perf_counter()
        with torch.no_grad(), autocast():
            logits = model(pixel_values)
        times.append(time.perf_counter() - t0)

        all_preds.append(logits.argmax(1).item())
        all_labels.append(label)

    vram_mib    = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    sec_per_vid = sum(times) / len(times)

    metrics = compute_metrics(all_preds, all_labels)
    results = {
        'top1_acc':      metrics['top1_acc'],
        'mca':           metrics['mca'],
        'per_class_acc': {idx_to_class[k]: v for k, v in metrics['per_class_acc'].items()},
        'sec_per_vid':   sec_per_vid,
        'vram_mib':      vram_mib,
    }

    with open(output_dir / 'eval_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    plot_confusion_matrix(
        all_labels, all_preds,
        class_names=[idx_to_class[i] for i in range(num_classes)],
        save_path=output_dir / 'confusion_matrix.png',
    )

    print(f"\nTop-1 Acc  : {metrics['top1_acc']:.4f}")
    print(f"MCA        : {metrics['mca']:.4f}")
    print(f"sec / vid  : {sec_per_vid:.4f}")
    print(f"VRAM peak  : {vram_mib:.1f} MiB")
    print(f"Saved to   : {output_dir}")


if __name__ == '__main__':
    main()
