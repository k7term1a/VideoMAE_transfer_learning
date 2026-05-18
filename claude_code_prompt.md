# Claude Code Prompt：VideoMAE V2 Transfer Learning 訓練腳本

## 背景說明

我已經可以在算力機器上載入 VideoMAEv2-Base backbone 並取得 output tensor。現在需要在此基礎上建構完整的訓練 pipeline。

---

## 資料集

- 來源：Kaggle — Human Activity & Suspicious Behavior Video Dataset（daudshah/video-dataset）
- 機器上的資料集路徑：`/home/kaichi/VideoMAE/video-dataset/dataset-video-split/`

### 目錄結構

```
dataset-video-split/
├── train/   # 940 支影片
├── valid/   # 194 支影片
└── test/    # 200 支影片
```

**資料集已提供固定的 train/valid/test 切分，直接使用，不需重新分割。**

### 類別推斷規則

類別由**檔名前綴**決定，共兩種命名格式：

| 格式 | 範例 | 類別 |
|---|---|---|
| `{Class}{N}_x264.mp4` | `Abuse001_x264.mp4` | `Abuse` |
| `{Class}_{(N)}.mp4` | `Sitting_(40).mp4` | `Sitting` |

共 **21 類**（見下方列表）。注意資料集中存在髒資料：`Normal_Videos_006_x264.mp4`（前綴多一底線），需在解析時正規化為 `Normal_Videos`。

**正常行為（8 類）：**
```
Clapping, Meet_and_Split, Normal_Videos, Sitting, Standing_Still,
Walking, Walking_While_Reading_Book, Walking_While_Using_Phone
```

**可疑 / 犯罪行為（13 類）：**
```
Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting,
RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism
```

---

## 模型架構

### Backbone

- 模型：`OpenGVLab/VideoMAEv2-Base`（ViT-B，約 86M 參數）
- **所有 backbone 參數必須 frozen**（`requires_grad = False`）
- 輸入格式：`(B, T, C, H, W)`，其中 T=16、H=W=224
- 輸出：取最後一層 hidden state 的 `[CLS]` token 作為影片表示，shape 為 `(B, 768)`

### 分類頭（三種，透過 `--head` 參數切換）

分類頭實作由使用者自行撰寫於 `model.py`，規格如下：

| 名稱 | 架構 |
|---|---|
| `linear` | `Linear(768, num_classes)` |
| `mlp` | `Linear(768, 512)` → GELU → `Dropout(0.3)` → `Linear(512, num_classes)` |
| `lora` | LoRA adapter（rank=16）注入 backbone attention 的 Q/V projection，接 `Linear(768, num_classes)`；**此模式下 LoRA 參數不 frozen** |

**`model.py` 的 backbone 載入與整體組裝邏輯由 Claude 提供；分類頭部分留為介面，由使用者實作。**

---

## 輸入前處理

- 每支影片均勻取 **16 幀**
- Resize 至 **224×224**
- 使用 HuggingFace 的 `VideoMAEImageProcessor` 做 normalization
- 影片讀取優先使用 `decord`（速度較快），備用 `torchvision`
- 前處理結果需快取為 `.pt` 檔案，避免每個 epoch 重複讀取影片

---

## 訓練設定

| 項目 | 設定 |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 1e-4，搭配 Cosine Annealing Scheduler |
| Epochs | 30 |
| Loss | CrossEntropyLoss，label smoothing = 0.1 |
| 混合精度 | fp16（`torch.cuda.amp`） |
| Gradient Clipping | max_norm = 1.0 |
| Early Stopping | patience = 10，監控 val accuracy |

---

## Shot 實驗設定

透過 `--shot` 參數支援以下三種模式：

- `--shot 20`：每類取 20 筆做訓練
- `--shot 40`：每類取 40 筆做訓練（**主實驗，對齊基線**）
- `--shot full`：使用全部訓練資料

---

## 資料切分規則

資料集已提供固定的 train/valid/test 切分，直接使用：

- Full 模式：直接使用 `train/`、`valid/`、`test/` 三個資料夾
- Shot 模式（20/40）：從 `train/` 中每類採樣 N 筆，`valid/` 與 `test/` 保持不變（random seed = 42）
- **測試集在所有實驗中完全相同**（即原始 `test/` 資料夾），切分結果（shot 模式的 train 子集索引）存為 `split_index.json`

---

## 需記錄的指標

### 訓練中（每 epoch 印出 + 存入 CSV log）

- train loss、val loss
- val Top-1 Accuracy
- val Mean Class Accuracy（macro）

### 測試集最終評估

- Top-1 Accuracy
- Mean Class Accuracy（每類 recall 的 macro 平均）
- Per-class accuracy（供 confusion matrix 使用）
- 推論時間（sec/vid，對測試集逐筆單張推論取平均，不做 batching）
- VRAM peak 用量（MiB，使用 `torch.cuda.max_memory_allocated()`）

---

## 專案檔案結構

```
project/
├── requirements.txt
├── README.md
├── dataset.py        # VideoDataset 類別、前處理、快取邏輯
├── model.py          # backbone 載入 + 三種分類頭定義
├── train.py          # 主訓練腳本
├── eval.py           # 載入 checkpoint，跑測試集評估
└── utils.py          # metrics、logging、confusion matrix 繪圖
```

---

## 資料集路徑

資料集路徑以常數定義於 `dataset.py`：

```python
DATASET_ROOT = "/home/kaichi/VideoMAE/video-dataset"
```

所有腳本直接引用此常數，**不透過 CLI 參數傳入**。

---

## train.py 指令介面

`--output_dir` 由 `--head`、`--shot`、`--epochs` 自動組合，命名規則如下：
- linear/mlp：`./runs/{head}_{shot}_{epochs}e`
- lora：`./runs/lora{rank}_{shot}_{epochs}e`

例如：
- `--head linear --shot full --epochs 30` → `./runs/linear_full_30e`
- `--head lora --shot 20 --epochs 20` → `./runs/lora16_20_20e`（rank=16）
- `--head mlp --shot 40 --epochs 30` → `./runs/mlp_40_30e`

可透過 `--run_name` 覆蓋自動命名（選填）。

```bash
python train.py \
  --head [linear|mlp|lora] \
  --shot [20|40|full] \
  --epochs 30 \
  --batch_size 8 \
  [--run_name custom_name]
```

## eval.py 指令介面

使用者直接指定 checkpoint 路徑與 head 類型，無需推算 run 目錄。

```bash
python eval.py \
  --checkpoint ./runs/linear_full_30e/best.pt \
  --head [linear|mlp|lora]
```

---

## 輸出檔案（存至 output_dir）

| 檔案 | 內容 |
|---|---|
| `best.pt` | val accuracy 最佳的 checkpoint |
| `train_log.csv` | 每個 epoch 的訓練指標 |
| `confusion_matrix.png` | 測試集 confusion matrix，標注表現最差的類別 |
| `eval_results.json` | 最終測試指標（Top-1 Acc、MCA、sec/vid、VRAM MiB） |

---

## 限制與注意事項

- Python 3.10+，PyTorch 2.x
- 單 GPU 執行即可，不需要 DDP
- 若影片檔案損毀或無法讀取，**跳過並印出 warning，不可 crash**
- 程式啟動時統一設定 random seed = 42（torch、numpy、random、CUDA）
- **類別數不可 hardcode**，需從資料集資料夾結構自動推斷
- Backbone HuggingFace model ID：`OpenGVLab/VideoMAEv2-Base`

---

## 交付要求

請產生上述所有檔案，每個檔案必須完整可執行，**不可留 TODO 或 placeholder**。README 中需包含可完整重現 40-shot 三種分類頭實驗的指令範例。
