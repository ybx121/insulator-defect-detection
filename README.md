# insulator-defect-detection

输电线路绝缘子缺陷检测与识别研究工程。当前复现主线使用本地新数据集
`Dataset/labels`，在同一原始 train/val/test 划分上训练并比较 6 个 YOLO 系模型和
2 个 D-FINE 模型。

远程仓库只保存代码、配置、文档、训练日志和结果图，不使用 Git LFS，也不保存数据集
或模型权重。数据集需要由使用者另行放到下文指定位置；YOLO 与 D-FINE 的通用预训练
权重可按复现步骤下载。

> 数据可信度说明：当前 `Dataset/labels` 原始划分存在跨 train/val/test 的近重复图组。
> 本文结果是该固定划分上的可复现实验结果，可用于模型横向比较，但不能直接解释为
> 严格无泄漏的泛化性能。详细审计见
> `docs/optimization_execution_plan_dataset_labels.md`。

## 克隆仓库

普通 Git 克隆即可，不需要安装或运行 Git LFS：

```powershell
git clone git@github.com:ybx121/insulator-defect-detection.git
Set-Location .\insulator-defect-detection
```

远程仓库包含：

- 项目代码、模型 YAML、训练矩阵和依赖文件；
- 八个实验的日志、参数、指标 CSV、曲线和验证可视化；
- 完整队列状态 `runs/dataset_labels_retrain/status.json`。

远程仓库不包含：

- `Dataset/` 下的训练、验证和测试数据；
- `datasets/` 下生成的 COCO 或其他派生数据；
- 任意 `.pt`、`.pth`、`.onnx` 或 `.engine` 权重；
- `runs/third_party/D-FINE` 第三方源码和预训练权重。

## 数据集文件夹位置

其他人拿到的数据集格式与本项目一致，只需把数据集文件夹命名为 `labels`，并放到
项目根目录的 `Dataset` 文件夹下：

```text
insulator-defect-detection/
└── Dataset/
    └── labels/
        ├── data.yaml
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── labels/
            ├── train/
            ├── val/
            └── test/
```

训练配置使用的固定相对路径是：

```text
Dataset/labels/data.yaml
```

不要放成 `Dataset/labels/labels/images/...`，也不要修改
`configs/dataset_labels_train_matrix.yaml` 中的数据集路径。

## 八模型验证结果

排名以最佳 `mAP50-95` 为主。所有指标均来自 `Dataset/labels` 的原始 `val` 划分。

| 排名 | 模型（最佳权重） | 最佳轮次 | Precision | Recall | mAP50 | 最佳 mAP50-95 | 末轮 mAP50-95 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | D-FINE L 960 | 15/15 | 90.50% | 89.44% | 88.31% | **61.83%** | 61.83% |
| 2 | D-FINE M 960 | 13/15 | 89.72% | **90.14%** | **88.88%** | **61.56%** | 59.03% |
| 3 | YOLO11s Context 960 | 72/100 | **94.68%** | 81.67% | 88.54% | **57.33%** | 55.65% |
| 4 | YOLO11m 960 | 81/100 | 87.41% | 82.23% | 87.06% | **56.56%** | 55.09% |
| 5 | YOLO11s 960 | 73/100 | 89.89% | 80.62% | 86.90% | **55.82%** | 53.88% |
| 6 | YOLO11s 1280 | 78/100 | 91.47% | 83.63% | 86.68% | **55.23%** | 54.25% |
| 7 | YOLO11m-P2 960 | 92/100 | 91.60% | 77.15% | 86.34% | **54.62%** | 53.91% |
| 8 | GF-InsuYOLO 960 | 94/100 | 82.44% | 72.53% | 76.60% | **46.77%** | 46.56% |

按主排名指标 `mAP50-95`，当前最佳单模型是 D-FINE L 960；如果只看 mAP50 或
Recall，则 D-FINE M 960 略高。当前没有新的八模型集成结果，因此 README 不再把旧
`credible_fine_v2` 集成写成当前最佳结果。

## 从零复现当前最佳模型

下面以 Windows PowerShell、Python 3.11 和 NVIDIA CUDA GPU 为例。参考实验环境为：

```text
Python 3.11.15
PyTorch 2.13.0+cu126
Ultralytics 8.4.90
CUDA 12.6
NVIDIA GeForce RTX 4080 Laptop GPU
seed 20260731
```

### 1. 准备环境

```powershell
conda create -n insulator-defect python=3.11 -y
conda activate insulator-defect

pip uninstall -y torch torchvision torchaudio
pip install -r .\requirements-win-cuda.txt
pip install -r .\requirements.txt
pip install ultralytics==8.4.90
```

确认 CUDA：

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

最后两项应显示 `True` 和实际 NVIDIA GPU 名称。

### 2. 检查数据集位置

确认数据集文件夹名为 `labels`，并位于：

```text
Dataset/labels
```

先确认配置和三个划分存在：

```powershell
Test-Path .\Dataset\labels\data.yaml
Test-Path .\Dataset\labels\images\train
Test-Path .\Dataset\labels\images\val
Test-Path .\Dataset\labels\images\test
Test-Path .\Dataset\labels\labels\train
Test-Path .\Dataset\labels\labels\val
Test-Path .\Dataset\labels\labels\test
```

七项都必须输出 `True`。

### 3. 下载 YOLO 通用预训练权重

```powershell
python -c "from ultralytics import YOLO; YOLO('yolo11s.pt'); YOLO('yolo11m.pt')"
```

这些是通用初始化权重，不是本文八个实验的最佳权重。

### 4. 准备固定版本 D-FINE

检出本次实验使用的 D-FINE 提交：

```powershell
git clone https://github.com/Peterande/D-FINE.git .\runs\third_party\D-FINE
git -C .\runs\third_party\D-FINE checkout 7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6

Push-Location .\runs\third_party\D-FINE
git apply ..\..\..\patches\dfine_map50_checkpoint.patch
pip install -r .\requirements.txt
Pop-Location
```

生成最佳模型 D-FINE L 960 的 COCO 数据、配置并下载 COCO 预训练权重：

```powershell
python .\scripts\prepare_dfine.py `
  --dataset .\Dataset\labels `
  --output .\datasets\dataset_labels_coco `
  --dfine-root .\runs\third_party\D-FINE `
  --project-root . `
  --imgsz 960 `
  --epochs 15 `
  --batch 4 `
  --model-size l `
  --pretrained-variant coco `
  --config-name dfine_hgnetv2_l_dataset_labels_coco.yml `
  --run-name dfine_l_img960 `
  --download-weights `
  --overwrite
```

### 5. 训练当前最佳 D-FINE L 960

```powershell
Push-Location .\runs\third_party\D-FINE
python .\train.py `
  -c .\configs\dfine\custom\dfine_hgnetv2_l_dataset_labels_coco.yml `
  -t .\weights\dfine_l_coco.pth `
  --use-amp `
  --seed 20260731
Pop-Location
```

训练共 15 轮，batch 为 4，固定输入为 960。参考 RTX 4080 Laptop GPU 耗时约
3 小时 14 分钟。最佳权重应生成在：

```text
runs/dfine/dfine_l_img960/best_stg2.pth
```

参考验证结果：

```text
Precision       90.50%
Recall          89.44%
mAP50           88.31%
mAP50-95        61.83%
best epoch      15/15
```

### 6. 用统一评估器复核 D-FINE L

先把 D-FINE 预测导出为项目统一 JSON：

```powershell
python .\scripts\predict_dfine.py `
  --dfine-root .\runs\third_party\D-FINE `
  --config .\runs\third_party\D-FINE\configs\dfine\custom\dfine_hgnetv2_l_dataset_labels_coco.yml `
  --weights .\runs\dfine\dfine_l_img960\best_stg2.pth `
  --data .\Dataset\labels\data.yaml `
  --split val `
  --imgsz 960 `
  --batch 4 `
  --conf 0.001 `
  --device 0 `
  --output .\runs\eval\dataset_labels_dfine_l_img960_val_predictions.json
```

再计算统一指标：

```powershell
python .\scripts\evaluate_detector.py `
  --data .\Dataset\labels\data.yaml `
  --split val `
  --mode external `
  --external-predictions .\runs\eval\dataset_labels_dfine_l_img960_val_predictions.json `
  --conf 0.001 `
  --iou 0.70 `
  --seed 20260731 `
  --output .\runs\eval\dataset_labels_dfine_l_img960_val.json
```

评估时不要改变数据划分、图像尺寸或根据验证结果调阈值后再与表中结果比较。

## 从零复现全部八个模型

完整训练矩阵位于：

```text
configs/dataset_labels_train_matrix.yaml
```

矩阵固定：

```text
YOLO epochs       100
D-FINE epochs      15
seed         20260731
YOLO patience       30
workers              4
augment       moderate
device                0
```

先查看将要执行的全部命令：

```powershell
python .\scripts\train_dataset_labels_all.py --dry-run
```

队列会读取仓库内已经发布的完成状态并跳过 `complete` 项。真正从零重跑前，先保留
参考状态副本，再删除活动状态文件：

```powershell
Copy-Item `
  .\runs\dataset_labels_retrain\status.json `
  .\runs\dataset_labels_retrain\status.reference.json `
  -Force
Remove-Item .\runs\dataset_labels_retrain\status.json
```

为两个 D-FINE 任务预先生成配置并下载通用预训练权重：

```powershell
python .\scripts\prepare_dfine.py `
  --dataset .\Dataset\labels `
  --output .\datasets\dataset_labels_coco `
  --dfine-root .\runs\third_party\D-FINE `
  --project-root . `
  --imgsz 960 --epochs 15 --batch 4 `
  --model-size m --pretrained-variant coco `
  --config-name dfine_hgnetv2_m_dataset_labels_coco.yml `
  --run-name dfine_m_img960 `
  --download-weights --overwrite

python .\scripts\prepare_dfine.py `
  --dataset .\Dataset\labels `
  --output .\datasets\dataset_labels_coco `
  --dfine-root .\runs\third_party\D-FINE `
  --project-root . `
  --imgsz 960 --epochs 15 --batch 4 `
  --model-size l --pretrained-variant coco `
  --config-name dfine_hgnetv2_l_dataset_labels_coco.yml `
  --run-name dfine_l_img960 `
  --download-weights --overwrite
```

然后启动可续跑队列：

```powershell
python .\scripts\train_dataset_labels_all.py
```

八个模型串行训练在参考机器上总计约 24 小时。运行状态和日志位于：

```text
runs/dataset_labels_retrain/status.json
runs/dataset_labels_retrain/logs/
```

如果只训练某几个任务：

```powershell
python .\scripts\train_dataset_labels_all.py `
  --only yolo11s_img960 yolo11s_context_img960 dfine_l_img960
```

任务名必须来自：

```text
yolo11s_img960
yolo11s_img1280
yolo11m_img960
yolo11m_p2_img960
gf_insuyolo_img960
yolo11s_context_img960
dfine_m_img960
dfine_l_img960
```

## 使用已有 YOLO 权重

以 YOLO11s Context 960 为例验证：

```powershell
yolo detect val `
  model=.\runs\detect\runs\detect\dataset_labels_retrain\yolo11s_context_img960\weights\best.pt `
  data=.\Dataset\labels\data.yaml `
  split=val `
  imgsz=960 `
  batch=12 `
  conf=0.001 `
  iou=0.70 `
  rect=False `
  device=0
```

预测自定义图片目录：

```powershell
yolo detect predict `
  model=.\runs\detect\runs\detect\dataset_labels_retrain\yolo11s_context_img960\weights\best.pt `
  source=D:\Images\insulator-test `
  imgsz=960 `
  conf=0.25 `
  device=0 `
  save=True
```

GF-InsuYOLO 和 Context YAML 使用了项目自定义模块，优先通过项目的 `train.py`、
`infer.py` 或已注册自定义模块的代码入口加载；普通 YOLO11s/YOLO11m 权重可直接使用
Ultralytics CLI。

## 复现检查清单

- 数据必须位于 `Dataset/labels`，目录中不能再多套一层 `labels`。
- 三个划分图像/标签数必须为 `1568/196/196` 且一一对应。
- 类别顺序必须与 `data.yaml` 完全一致。
- 使用种子 `20260731`、固定原始划分和训练矩阵中的 batch/尺寸。
- YOLO 比较使用各任务的 `weights/best.pt`。
- D-FINE M/L 比较使用各任务的 `best_stg2.pth`。
- 不使用 Git LFS，也不要期望普通 `git clone` 下载数据或权重。
- 当前表格只代表原始验证划分；无泄漏新划分建立后必须重新训练并单独排名。

## 项目入口

- `configs/dataset_labels_train_matrix.yaml`：八模型完整训练矩阵。
- `scripts/train_dataset_labels_all.py`：可续跑串行训练队列。
- `train.py`：YOLO、P2、GF-InsuYOLO 和 Context 训练入口。
- `scripts/prepare_dfine.py`：数据转 COCO、D-FINE 配置生成和预训练权重下载。
- `scripts/predict_dfine.py`：D-FINE 预测导出。
- `scripts/evaluate_detector.py`：项目统一评估器。
- `runs/dataset_labels_retrain/status.json`：已完成八任务状态和耗时。
- `docs/optimization_execution_plan_dataset_labels.md`：数据审计、泄漏说明和后续优化方案。
