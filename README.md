# VideoMAEv2 Transfer Learning

Human activity and suspicious behavior classification using VideoMAEv2-Base as a frozen backbone.

## Setup

```bash
uv pip install -r requirements.txt \
  --index-url https://download.pytorch.org/whl/cu118
```

## Before running

Open `model.py` and implement the four stub classes:

1. `LinearHead` — `Linear(768 → num_classes)`
2. `MLPHead` — `Linear(768→512) → GELU → Dropout(0.3) → Linear(512→num_classes)`
3. `LoRALinear` — LoRA wrapper for `nn.Linear`
4. `LoRAHead` — `Linear(768 → num_classes)` used with LoRA backbone

Start with `LinearHead`, verify training runs, then proceed to `MLPHead` and `LoRALinear`.

## Output directory naming

| Command | Output dir |
|---|---|
| `--head linear --shot full  --epochs 30` | `runs/linear_full_30e/` |
| `--head mlp    --shot 40   --epochs 30` | `runs/mlp_40_30e/` |
| `--head lora   --shot 20   --epochs 20` | `runs/lora16_20_20e/` |

## 40-shot experiments (three heads)

```bash
# Linear head
python train.py --head linear --shot 40 --epochs 30 --batch_size 8
python eval.py  --checkpoint runs/linear_40_30e/best.pt --head linear

# MLP head
python train.py --head mlp --shot 40 --epochs 30 --batch_size 8
python eval.py  --checkpoint runs/mlp_40_30e/best.pt --head mlp

# LoRA (rank=16)
python train.py --head lora --shot 40 --epochs 30 --batch_size 8
python eval.py  --checkpoint runs/lora16_40_30e/best.pt --head lora
```

## Outputs per run

| File | Content |
|---|---|
| `best.pt` | Best checkpoint (by val accuracy) |
| `train_log.csv` | Per-epoch train/val metrics |
| `split_index.json` | Shot-mode train subset indices (for reproducibility) |
| `eval_results.json` | Top-1 Acc, MCA, sec/vid, VRAM MiB |
| `confusion_matrix.png` | Confusion matrix with ★ marking the 3 worst classes |

## Dataset

21 classes — 8 normal activities and 13 suspicious/criminal behaviors.  
Pre-split: train 940 / valid 194 / test 200.
