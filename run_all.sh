#!/bin/bash
set -e

python train.py --head linear --shot 40 --epochs 30 --batch_size 8
python eval.py  --checkpoint runs/linear_40_30e/best.pt --head linear

python train.py --head mlp --shot 40 --epochs 30 --batch_size 8
python eval.py  --checkpoint runs/mlp_40_30e/best.pt --head mlp

python train.py --head lora --shot 40 --epochs 30 --batch_size 8
python eval.py  --checkpoint runs/lora16_40_30e/best.pt --head lora
