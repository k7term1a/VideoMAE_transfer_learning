import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch.amp import autocast

from dataset import VideoDataset, _read_frames, _read_multi_clips
from model import build_model
from utils import compute_metrics, plot_confusion_matrix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a VideoMAEv2 checkpoint on the test set")
    p.add_argument('--checkpoint',  required=True, help="Path to best.pt")
    p.add_argument('--head',        required=True, choices=['linear', 'mlp', 'lora'])
    p.add_argument('--backbone',    default='vmaev2', choices=['k710', 'vmaev2'])
    p.add_argument('--no_cache',    action='store_true',
                   help='Time full pipeline: video decode + preprocess + forward')
    p.add_argument('--multi_clip',  action='store_true',
                   help='Multi-clip inference: extract 3 evenly-spaced clips, average logits. '
                        'Implies reading from video (bypasses cache). '
                        'Can be combined with --no_cache to also measure decode time.')
    p.add_argument('--n_clips',     type=int, default=3,
                   help='Number of clips for --multi_clip (default: 3)')
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

    model = build_model(args.head, num_classes, lora_rank=lora_rank, backbone=args.backbone)
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()

    test_ds = VideoDataset('test', class_to_idx)
    print(f"Test samples: {len(test_ds)}")

    all_preds:  list[int]   = []
    all_labels: list[int]   = []
    times:      list[float] = []

    torch.cuda.reset_peak_memory_stats(device)

    for i in range(len(test_ds)):
        path, label = test_ds.samples[i]

        if args.multi_clip:
            # ── Multi-clip inference ──────────────────────────────────────────
            # Optimisations:
            #   1. _read_multi_clips reads only the unique frames needed.
            #   2. Preprocessing for each clip runs in parallel (threads).
            #   3. All clips are batched into a single forward pass.
            t0    = time.perf_counter()
            clips = _read_multi_clips(path, n_clips=args.n_clips)
            if clips is None:
                print(f"  [warn] cannot read {path.name}, skipping")
                continue

            def preprocess_clip(clip: "np.ndarray") -> torch.Tensor:
                frame_list = [clip[j] for j in range(clip.shape[0])]
                return test_ds.processor(frame_list, return_tensors='pt')['pixel_values'].squeeze(0)

            # Parallel preprocessing (I/O-free CPU work → threads are sufficient)
            with ThreadPoolExecutor(max_workers=len(clips)) as pool:
                pixel_values_list = list(pool.map(preprocess_clip, clips))

            # Single batched forward pass  (batch = n_clips)
            pixel_values_batch = torch.stack(pixel_values_list, dim=0).to(device)
            with torch.no_grad(), autocast('cuda'):
                logits_batch = model(pixel_values_batch)          # (n_clips, num_classes)

            logits = logits_batch.mean(dim=0, keepdim=True)       # (1, num_classes)
            times.append(time.perf_counter() - t0)

        elif args.no_cache:
            # ── Single-clip, full pipeline (decode → preprocess → forward) ───
            t0     = time.perf_counter()
            frames = _read_frames(path)
            if frames is None:
                print(f"  [warn] cannot read {path.name}, skipping")
                continue
            frame_list   = [frames[j] for j in range(frames.shape[0])]
            inputs       = test_ds.processor(frame_list, return_tensors='pt')
            pixel_values = inputs['pixel_values'].squeeze(0).unsqueeze(0).to(device)
            with torch.no_grad(), autocast('cuda'):
                logits = model(pixel_values)
            times.append(time.perf_counter() - t0)

        else:
            # ── Single-clip, cached (forward only) ───────────────────────────
            pixel_values, label = test_ds[i]
            pixel_values = pixel_values.unsqueeze(0).to(device)
            t0 = time.perf_counter()
            with torch.no_grad(), autocast('cuda'):
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

    if args.multi_clip:
        timing_scope = f"decode+preprocess+{args.n_clips}×forward (multi-clip)"
    elif args.no_cache:
        timing_scope = "decode+preprocess+forward"
    else:
        timing_scope = "forward only (cached)"
    total_sec    = sum(times)
    print(f"\nTop-1 Acc  : {metrics['top1_acc']:.4f}")
    print(f"MCA        : {metrics['mca']:.4f}")
    print(f"sec / vid  : {sec_per_vid:.4f}  ({timing_scope})")
    print(f"total sec  : {total_sec:.2f}s  ({len(times)} videos)")
    print(f"VRAM peak  : {vram_mib:.1f} MiB")
    print(f"Saved to   : {output_dir}")


if __name__ == '__main__':
    main()
