# insulator-defect-detection

输电线路绝缘子缺陷检测与识别研究工程。仓库同时保留防泄漏的
`credible_fine_v2` 历史基准，以及在用户指定 `Dataset/labels` 原始划分上完成的
YOLO、GF-InsuYOLO 和 D-FINE 八模型重训练结果。`Dataset/labels` 的原始划分存在
跨划分近重复图组，因此这批结果用于历史复现和后续优化，不应直接解释为无泄漏泛化性能。

## 完整克隆与发布内容

本仓库使用 Git LFS 保存模型权重。要获得代码、文档、训练结果和真实权重文件，
请安装 Git LFS 后执行：

```powershell
git lfs install
git clone git@github.com:ybx121/insulator-defect-detection.git
Set-Location .\insulator-defect-detection
git lfs pull
git lfs ls-files
```

不要依赖 GitHub 的“Download ZIP”代替上述流程，因为源码归档是否包含 LFS 对象
取决于仓库设置。正常 `git clone` 配合 Git LFS 会下载本次发布的完整内容。

仓库发布范围包括：

- 全部版本化代码、配置和文档；
- 八个重训练任务的日志、参数、指标、曲线、可视化样例和最佳权重；
- 队列状态与实验清单，位于 `runs/dataset_labels_retrain`。

八个发布权重的固定路径为：

```text
runs/detect/runs/detect/dataset_labels_retrain/yolo11s_img960/weights/best.pt
runs/detect/runs/detect/dataset_labels_retrain/yolo11s_img1280/weights/best.pt
runs/detect/runs/detect/dataset_labels_retrain/yolo11m_img960/weights/best.pt
runs/detect/runs/detect/dataset_labels_retrain/yolo11m_p2_img960/weights/best.pt
runs/detect/runs/detect/dataset_labels_retrain/gf_insuyolo_img960/weights/best.pt
runs/detect/runs/detect/dataset_labels_retrain/yolo11s_context_img960/weights/best.pt
runs/dfine/dfine_m_img960/best_stg2.pth
runs/dfine/dfine_l_img960/best_stg2.pth
```

为避免把本地派生数据和重复 checkpoint 推入仓库，以下内容明确不发布：

- 整个 `Dataset/`，包括本地训练用的 `Dataset/labels`；
- `datasets/`、`merged_dataset/` 和本地 `Dataset/labels.zip`；
- 八个保留任务中除上述最佳权重外的 `.pt`/`.pth` checkpoint；
- 第三方仓库、下载缓存和其他历史训练目录。

## 当前最好结果

所有结果均来自固定的 `rect=False` 方形 letterbox 验证协议：

| 结果 | 模型 | 数据集 | 划分 | mAP50 |
| --- | --- | --- | --- | ---: |
| 最佳单模型 | D-FINE-M，COCO 预训练 | `credible_fine_v2` | `val` | 0.8257 |
| 最佳集成 | YOLO 全图/P2/缺陷专家/ROI + D-FINE-M/L | `credible_fine_v2` | `val` | **0.8375** |

最佳集成的 200 次 Bootstrap mAP50 95% 置信区间为
`[0.8058, 0.8735]`，分类别 AP50 为：

```text
insulator_string       0.9651
broken_shell           0.7726
flashover_pollution    0.6651
missing_disc_drop      0.9474
```

这里的“当前最好”是验证集最好结果，不是锁定测试集结果。由于验证集
距离项目目标 `mAP50 > 0.95` 仍较远，锁定的 `test` 尚未用于最终评估。

权威实验记录：

- `docs/optimization_protocol.md`
- `configs/experiments.yaml`
- `runs/eval/credible_v2_dfine_ml_yolo_full_ensemble_val_bootstrap.json`

## 复现合同

四分类顺序固定为：

```text
0 insulator_string
1 broken_shell
2 flashover_pollution
3 missing_disc_drop
```

必须核对以下数据指纹：

```text
credible_fine_v1
5b0cab9c44f6985659841ee5dc463284582c711037c987ac78c0a8a6581eb113

credible_fine_v2
e45e9272436fcaea448f4e4edc7c252989f5be3db8b1a8400a5ba97d034e97e1
```

不同指纹代表不同基准，结果不能放在同一个排行榜中比较。

当前 v2 指纹实现还会记录解析后的本地源路径。要得到上面的精确 v2 指纹，
项目根目录需保持为 `E:\School\insulator-defect-detection`，并使用下文约定的
审核文件路径；换目录构建时必须把新指纹如实记录为另一个复现环境。

本文所说的“从零复现”从以下内容开始：

- 仓库内的原始 `Dataset`、`InsulatorDataSet` 和人工 CPLID 类别映射。
- 脚本下载的 CC BY 4.0 Supervisely 公开数据。
- 两份已经完成人工审核的 CSV。
- 全新的 Python 环境和从官方提交检出的 D-FINE。

人工审核决定是数据合同的一部分，不能由模型自动重新生成。开始前必须把实验归档中的
两份文件放到以下位置：

```text
annotations/credible_fine_v1_review_reviewed.csv
annotations/credible_v1_missing_label_candidates_reviewed.csv
```

它们的 SHA-256 必须分别为：

```text
7bc82e0310a208ce7e9448c9edcdc18e19d69626247fcf1c805f5697d5cdcbf0
331e6f15a42dc575103357b0e8069e3b4deb0dfc72540e2dae9f7e35934a5219
```

如果缺少这两份人工审核文件，只能重新进行人工审核，不能声称精确复现
`credible_fine_v2` 或当前最好结果。

可以用下面的命令核对文件：

```powershell
Get-FileHash -Algorithm SHA256 `
  .\annotations\credible_fine_v1_review_reviewed.csv, `
  .\annotations\credible_v1_missing_label_candidates_reviewed.csv
```

## 从零开始训练与验证

下面以 Windows PowerShell、Python 3.11 和 NVIDIA CUDA GPU 为例。所有命令均在
项目根目录执行。

### 1. 准备 Python 与 CUDA 环境

```powershell
conda create -n insulator-defect python=3.11 -y
conda activate insulator-defect

pip uninstall -y torch torchvision torchaudio
pip install -r .\requirements-win-cuda.txt
pip install -r .\requirements.txt
pip install ultralytics==8.4.90
```

确认 CUDA 可用：

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`torch.cuda.is_available()` 应输出 `True`。当前参考实验使用：

```text
Python 3.11.15
Ultralytics 8.4.90
CUDA 12.6
NVIDIA GeForce RTX 4080 Laptop GPU
seed 20260708
```

下载 YOLO11s 和 YOLO11m 官方预训练权重：

```powershell
python -c "from ultralytics import YOLO; YOLO('yolo11s.pt'); YOLO('yolo11m.pt')"
```

### 2. 准备固定版本的 D-FINE

当前结果使用 D-FINE 提交：

```text
7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6
```

检出代码、应用按 AP50 保存权重的补丁并安装依赖：

```powershell
git clone https://github.com/Peterande/D-FINE.git .\runs\third_party\D-FINE
git -C .\runs\third_party\D-FINE checkout 7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6

Push-Location .\runs\third_party\D-FINE
git apply ..\..\..\patches\dfine_map50_checkpoint.patch
pip install -r .\requirements.txt
Pop-Location
```

如果该目录已经存在，先用下面的命令确认提交和工作区状态，不要重复克隆或重复应用补丁：

```powershell
git -C .\runs\third_party\D-FINE rev-parse HEAD
git -C .\runs\third_party\D-FINE status --short
```

### 3. 从原始数据构建 `credible_fine_v1`

先把仓库中的异构标签统一为四分类 YOLO 数据。这里的 `unified_fine` 只作为
可信数据构建器的中间输入，不训练旧的 GF-InsuYOLO：

```powershell
python .\scripts\build_unified_dataset.py `
  --mode fine `
  --cplid-fine-labels .\annotations\cplid_defective_fine_labels.csv `
  --seed 20260708 `
  --overwrite
```

下载公开数据：

```powershell
python .\scripts\download_public_dataset.py `
  --output .\datasets\raw\supervisely
```

确认人工审核文件已经放在约定位置，然后构建 v1：

```powershell
python .\scripts\build_credible_dataset.py `
  --existing .\datasets\unified_fine `
  --public-root .\datasets\raw\supervisely `
  --output .\datasets\credible_fine_v1 `
  --review-csv .\annotations\credible_fine_v1_review_reviewed.csv `
  --seed 20260708 `
  --overwrite

python .\scripts\make_balanced_view.py `
  .\datasets\credible_fine_v1 `
  --overwrite

python .\scripts\validate_credible_dataset.py `
  .\datasets\credible_fine_v1
```

检查指纹：

```powershell
Get-Content .\datasets\credible_fine_v1\metadata\dataset_fingerprint.json
```

输出必须包含：

```text
5b0cab9c44f6985659841ee5dc463284582c711037c987ac78c0a8a6581eb113
```

### 4. 构建 `credible_fine_v2`

v2 只向训练集加入 41 张图中的 44 个人工确认框，不修改 `val`、`test`
标签或划分：

```powershell
New-Item -ItemType Directory -Force `
  .\datasets\credible_fine_v1\audit, `
  .\runs\audit\credible_v1_missing_label_consensus | Out-Null

Copy-Item -Force `
  .\annotations\credible_fine_v1_review_reviewed.csv `
  .\datasets\credible_fine_v1\audit\review_reviewed.csv

Copy-Item -Force `
  .\annotations\credible_v1_missing_label_candidates_reviewed.csv `
  .\runs\audit\credible_v1_missing_label_consensus\candidates_reviewed.csv

python .\scripts\apply_pseudo_labels.py `
  --source .\datasets\credible_fine_v1 `
  --proposals .\runs\audit\credible_v1_missing_label_consensus\candidates_reviewed.csv `
  --audit-review .\datasets\credible_fine_v1\audit\review_reviewed.csv `
  --output .\datasets\credible_fine_v2 `
  --min-score 0.65 `
  --min-agreement-iou 0.70 `
  --max-gt-iou 0.10 `
  --repeat-changed 0 `
  --require-review `
  --overwrite

python .\scripts\validate_credible_dataset.py `
  .\datasets\credible_fine_v2
```

检查最终指纹：

```powershell
Get-Content .\datasets\credible_fine_v2\metadata\dataset_fingerprint.json
```

输出必须包含：

```text
e45e9272436fcaea448f4e4edc7c252989f5be3db8b1a8400a5ba97d034e97e1
```

指纹不一致时立即停止，不要继续训练或把结果写入当前排行榜。

### 5. 复现最佳单模型 D-FINE-M

把 v2 导出为 COCO，生成 D-FINE-M 配置并下载 COCO 预训练权重：

```powershell
python .\scripts\prepare_dfine.py `
  --dataset .\datasets\credible_fine_v2 `
  --output .\datasets\credible_fine_v2_coco `
  --dfine-root .\runs\third_party\D-FINE `
  --project-root . `
  --imgsz 960 `
  --epochs 15 `
  --batch 4 `
  --model-size m `
  --pretrained-variant coco `
  --multi-scale `
  --config-name dfine_hgnetv2_m_insulator.yml `
  --run-name credible_fine_v2_dfine_m_960 `
  --download-weights `
  --overwrite
```

训练：

```powershell
Push-Location .\runs\third_party\D-FINE
python .\train.py `
  -c .\configs\dfine\custom\dfine_hgnetv2_m_insulator.yml `
  --use-amp `
  --seed=20260708 `
  -t .\weights\dfine_m_coco.pth
Pop-Location
```

当前最好单模型使用：

```text
runs/dfine/credible_fine_v2_dfine_m_960/best_stg2.pth
```

导出验证集预测：

```powershell
python .\scripts\predict_dfine.py `
  --dfine-root .\runs\third_party\D-FINE `
  --config .\runs\third_party\D-FINE\configs\dfine\custom\dfine_hgnetv2_m_insulator.yml `
  --weights .\runs\dfine\credible_fine_v2_dfine_m_960\best_stg2.pth `
  --data .\datasets\credible_fine_v2\data.yaml `
  --split val `
  --imgsz 960 `
  --batch 4 `
  --conf 0.001 `
  --device 0 `
  --output .\runs\eval\credible_v2_dfine_m_960_val_predictions.json
```

用项目统一评估器计算指标：

```powershell
python .\scripts\evaluate_detector.py `
  --data .\datasets\credible_fine_v2\data.yaml `
  --split val `
  --mode external `
  --external-predictions .\runs\eval\credible_v2_dfine_m_960_val_predictions.json `
  --conf 0.001 `
  --iou 0.70 `
  --seed 20260708 `
  --output .\runs\eval\credible_v2_dfine_m_960_val.json
```

参考结果：

```text
mAP50     0.8257
mAP50-95  0.5669
mAP75     0.5846
```

### 6. 准备最佳集成的 YOLO 数据视图

缺陷专家只保留三个缺陷类：

```powershell
python .\scripts\make_class_projection_dataset.py `
  .\datasets\credible_fine_v1 `
  .\datasets\credible_fine_v1_defects `
  --classes 1 2 3 `
  --overwrite
```

ROI 分支使用 15% 上下文、两个训练扰动，并在高效训练视图中只保留一个正样本扰动：

```powershell
python .\scripts\make_crop_dataset.py `
  --input .\datasets\credible_fine_v1 `
  --output .\datasets\credible_fine_v1_crops `
  --margin 0.15 `
  --train-jitter-count 2 `
  --jitter-center 0.05 `
  --jitter-scale 0.10 `
  --efficient-positive-jitters 1 `
  --seed 20260708 `
  --overwrite
```

### 7. 训练最佳集成的四个 YOLO 分支

训练 YOLO11s 全图基线：

```powershell
python .\train.py `
  --model .\yolo11s.pt `
  --data .\datasets\credible_fine_v1\data_unbalanced.yaml `
  --imgsz 960 `
  --epochs 40 `
  --batch 16 `
  --device 0 `
  --seed 20260708 `
  --augment-preset moderate `
  --close-mosaic 15 `
  --patience 20 `
  --name credible_v1_yolo11s_960_aug_moderate_40
```

训练 YOLO11m-P2：

```powershell
python .\train.py `
  --model .\configs\yolo11m_p2.yaml `
  --weights .\yolo11m.pt `
  --data .\datasets\credible_fine_v1\data.yaml `
  --imgsz 960 `
  --epochs 40 `
  --batch 6 `
  --device 0 `
  --seed 20260708 `
  --augment-preset moderate `
  --close-mosaic 15 `
  --patience 20 `
  --name credible_v1_yolo11m_p2_960_b6
```

从全图基线初始化三个缺陷类专家：

```powershell
python .\train.py `
  --model .\yolo11s.pt `
  --weights .\runs\detect\runs\credible_v1_yolo11s_960_aug_moderate_40\weights\best_map50.pt `
  --data .\datasets\credible_fine_v1_defects\data.yaml `
  --imgsz 960 `
  --epochs 30 `
  --batch 8 `
  --device 0 `
  --seed 20260708 `
  --optimizer AdamW `
  --lr0 0.0005 `
  --augment-preset moderate `
  --name credible_v1_yolo11s_defect_expert_960
```

训练 ROI 局部模型：

```powershell
python .\train.py `
  --model .\yolo11s.pt `
  --data .\datasets\credible_fine_v1_crops\data_efficient.yaml `
  --imgsz 640 `
  --epochs 30 `
  --batch 16 `
  --workers 4 `
  --device 0 `
  --seed 20260708 `
  --optimizer AdamW `
  --lr0 0.001 `
  --lrf 0.01 `
  --weight-decay 0.0005 `
  --augment-preset moderate `
  --close-mosaic 8 `
  --patience 10 `
  --name credible_v1_local_roi_yolo11s_640_efficient_v2
```

后续集成统一使用各目录中的 `weights/best_map50.pt`。如果同名输出目录已经存在，
Ultralytics 可能追加数字后缀；此时必须把后续路径改成实际目录，不能误用旧权重。

### 8. 训练集成用 D-FINE-L

重新导出同一个 v2 COCO 数据并生成固定 960 输入的 D-FINE-L 配置：

```powershell
python .\scripts\prepare_dfine.py `
  --dataset .\datasets\credible_fine_v2 `
  --output .\datasets\credible_fine_v2_coco `
  --dfine-root .\runs\third_party\D-FINE `
  --project-root . `
  --imgsz 960 `
  --epochs 12 `
  --batch 4 `
  --model-size l `
  --pretrained-variant coco `
  --config-name dfine_hgnetv2_l_insulator_coco.yml `
  --run-name credible_fine_v2_dfine_l_coco_960_b4_fixed `
  --download-weights `
  --overwrite
```

训练：

```powershell
Push-Location .\runs\third_party\D-FINE
python .\train.py `
  -c .\configs\dfine\custom\dfine_hgnetv2_l_insulator_coco.yml `
  --use-amp `
  --seed=20260708 `
  -t .\weights\dfine_l_coco.pth
Pop-Location
```

集成使用补丁保存的 AP50 最佳权重：

```text
runs/dfine/credible_fine_v2_dfine_l_coco_960_b4_fixed/best_map50.pth
```

导出验证集预测：

```powershell
python .\scripts\predict_dfine.py `
  --dfine-root .\runs\third_party\D-FINE `
  --config .\runs\third_party\D-FINE\configs\dfine\custom\dfine_hgnetv2_l_insulator_coco.yml `
  --weights .\runs\dfine\credible_fine_v2_dfine_l_coco_960_b4_fixed\best_map50.pth `
  --data .\datasets\credible_fine_v2\data.yaml `
  --split val `
  --imgsz 960 `
  --batch 4 `
  --conf 0.001 `
  --device 0 `
  --output .\runs\eval\credible_v2_dfine_l_coco_960_val_predictions.json
```

### 9. 复现当前最佳集成

最终集成组成如下：

```text
全图主模型        YOLO11s，4 类
P2 补充分支       YOLO11m-P2，4 类，class offset 0
缺陷专家          YOLO11s，3 类，class offset 1
ROI 局部分支      YOLO11s，3 类
外部预测 1        D-FINE-M，4 类
外部预测 2        D-FINE-L，4 类
```

使用保存下来的真实融合参数运行验证：

```powershell
python .\scripts\evaluate_detector.py `
  --weights .\runs\detect\runs\credible_v1_yolo11s_960_aug_moderate_40\weights\best_map50.pt `
  --local-weights .\runs\detect\runs\credible_v1_local_roi_yolo11s_640_efficient_v2\weights\best_map50.pt `
  --data .\datasets\credible_fine_v2\data.yaml `
  --split val `
  --mode ensemble-two-stage `
  --ensemble-weights `
    .\runs\detect\runs\credible_v1_yolo11m_p2_960_b6\weights\best_map50.pt `
    .\runs\detect\runs\credible_v1_yolo11s_defect_expert_960\weights\best_map50.pt `
  --external-predictions `
    .\runs\eval\credible_v2_dfine_m_960_val_predictions.json `
    .\runs\eval\credible_v2_dfine_l_coco_960_val_predictions.json `
  --ensemble-class-offsets 0 1 `
  --two-stage-fusion union `
  --imgsz 960 `
  --batch 8 `
  --local-imgsz 640 `
  --conf 0.001 `
  --operating-conf 0.25 `
  --iou 0.70 `
  --fusion-iou 0.55 `
  --device 0 `
  --bootstrap 200 `
  --seed 20260708 `
  --output .\runs\eval\credible_v2_dfine_ml_yolo_full_ensemble_val_bootstrap.json `
  --leaderboard .\runs\eval\leaderboard_credible_v2.csv
```

参考结果：

```text
mAP50       0.8375
mAP50-95    0.5727
mAP75       0.6135
mAP50 CI95  [0.8058, 0.8735]
```

评估器还会生成：

```text
runs/eval/credible_v2_dfine_ml_yolo_full_ensemble_val_bootstrap_per_class.csv
runs/eval/credible_v2_dfine_ml_yolo_full_ensemble_val_bootstrap_errors.csv
runs/eval/leaderboard_credible_v2.csv
```

### 10. 复现检查清单

只有同时满足以下条件，才可以把实验标记为当前结果的复现：

- v1、v2 指纹与本文完全一致。
- D-FINE 提交为 `7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6`。
- 所有训练和评估均使用种子 `20260708`。
- YOLO 使用 `best_map50.pt`，D-FINE-M 使用 `best_stg2.pth`，
  D-FINE-L 使用 `best_map50.pth`。
- D-FINE 预测置信度保留到 `0.001`，不提前裁掉低分框。
- 最终评估使用 `rect=False` 方形 letterbox、`fusion_iou=0.55`
  和 200 次 Bootstrap。
- 只在 `val` 上对照当前结果；未经实验门控不要使用锁定的 `test`。
- 报告同时记录 overall、分类别 AP、正常图误报率、数据指纹和完整权重路径。

## 项目入口

- `train.py`：带实验清单和 AP50 最佳权重保存的 Ultralytics 训练入口。
- `scripts/prepare_dfine.py`：YOLO 到 COCO 导出及 D-FINE 配置生成。
- `scripts/predict_dfine.py`：导出模型无关的 D-FINE 预测 JSON。
- `scripts/evaluate_detector.py`：统一单模型、外部预测和融合评估。
- `docs/optimization_protocol.md`：数据合同、实验门控、完整正负实验结论。
- `configs/experiments.yaml`：当前模型、权重、指标和报告索引。
