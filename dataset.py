import os
import re
import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEImageProcessor

DATASET_ROOT = "/home/kaichi/VideoMAE/video-dataset/dataset-video-split"
CACHE_DIR    = "/home/kaichi/VideoMAE/cache"
NUM_FRAMES   = 16
SEED         = 42


def _parse_class(filename: str) -> str | None:
    # Format 1: Abuse001_x264.mp4 / Normal_Videos_006_x264.mp4
    m = re.match(r'^([A-Za-z_]+?)\d+_x264\.mp4$', filename)
    if m:
        return m.group(1).rstrip('_')
    # Format 2: Sitting_(40).mp4
    m = re.match(r'^(.+)_\(\d+\)\.mp4$', filename)
    if m:
        return m.group(1)
    return None


def get_classes() -> list[str]:
    classes: set[str] = set()
    for split in ('train', 'valid', 'test'):
        for f in (Path(DATASET_ROOT) / split).iterdir():
            cls = _parse_class(f.name)
            if cls:
                classes.add(cls)
    return sorted(classes)


def _read_frames(path: Path) -> np.ndarray | None:
    """Return (T, H, W, C) uint8 numpy array, or None on failure."""
    try:
        import decord
        vr      = decord.VideoReader(str(path), ctx=decord.cpu(0))
        total   = len(vr)
        indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
        return vr.get_batch(indices).asnumpy()
    except Exception:
        pass

    try:
        import torchvision
        frames_t, _, _ = torchvision.io.read_video(
            str(path), pts_unit='sec', output_format='THWC'
        )
        total   = len(frames_t)
        indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
        return frames_t[indices].numpy()
    except Exception:
        return None


class VideoDataset(Dataset):
    def __init__(
        self,
        split: str,
        class_to_idx: dict[str, int],
        shot: int | None = None,
        split_indices: list[int] | None = None,
    ):
        self.class_to_idx = class_to_idx
        self.processor    = VideoMAEImageProcessor.from_pretrained(
            "OpenGVLab/VideoMAEv2-Base"
        )
        os.makedirs(CACHE_DIR, exist_ok=True)

        split_dir   = Path(DATASET_ROOT) / split
        all_samples: list[tuple[Path, int]] = []
        for f in sorted(split_dir.iterdir()):
            cls = _parse_class(f.name)
            if cls in class_to_idx:
                all_samples.append((f, class_to_idx[cls]))

        if split == 'train' and shot is not None:
            if split_indices is not None:
                self.samples       = [all_samples[i] for i in split_indices]
                self.split_indices = split_indices
            else:
                by_class: dict[int, list[int]] = {}
                for i, (_, label) in enumerate(all_samples):
                    by_class.setdefault(label, []).append(i)

                rng      = random.Random(SEED)
                selected: list[int] = []
                for idxs in by_class.values():
                    shuffled = idxs[:]
                    rng.shuffle(shuffled)
                    selected.extend(shuffled[:shot])
                selected.sort()

                self.samples       = [all_samples[i] for i in selected]
                self.split_indices = selected
        else:
            self.samples       = all_samples
            self.split_indices = None

    def _cache_path(self, video_path: Path) -> Path:
        return Path(CACHE_DIR) / f"{video_path.parent.name}_{video_path.stem}.pt"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        cache       = self._cache_path(path)

        if cache.exists():
            pixel_values = torch.load(cache, weights_only=True)
        else:
            frames = _read_frames(path)
            if frames is None:
                warnings.warn(f"Cannot read {path}, substituting zeros")
                pixel_values = torch.zeros(NUM_FRAMES, 3, 224, 224)
            else:
                frame_list   = [frames[i] for i in range(NUM_FRAMES)]
                inputs       = self.processor(frame_list, return_tensors="pt")
                # processor output: (1, T, C, H, W) → squeeze → (T, C, H, W)
                pixel_values = inputs["pixel_values"].squeeze(0)
            torch.save(pixel_values, cache)

        return pixel_values, label


def build_loaders(
    shot: int | None,
    batch_size: int,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int], list[int] | None]:
    classes      = get_classes()
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_ds = VideoDataset('train', class_to_idx, shot=shot)
    valid_ds = VideoDataset('valid', class_to_idx)
    test_ds  = VideoDataset('test',  class_to_idx)

    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=True, drop_last=False)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    valid_loader = DataLoader(valid_ds, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    return train_loader, valid_loader, test_loader, class_to_idx, train_ds.split_indices
