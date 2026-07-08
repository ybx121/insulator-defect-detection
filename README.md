# insulator-defect-detection

输电线路绝缘子缺陷检测与识别研究工程。当前实现了 GF-InsuYOLO
（Global-Fine Insulator YOLO）的数据工程、训练入口和全图/局部两阶段推理入口。

## Dataset Variants

本项目把原始数据统一为两套 YOLO 数据集：

- `datasets/unified_coarse`：2 类，`insulator_string`、`defect`，用于粗标预训练。
- `datasets/unified_fine`：4 类，`insulator_string`、`broken_shell`、`flashover_pollution`、`missing_disc_drop`，用于主模型训练和评测。
- `datasets/unified_fine_crops`：3 类局部 crop 缺陷集，用于第二阶段局部缺陷检测器。

类别映射：

- `Dataset` 原始 `insulator string` -> `insulator_string`
- `Dataset` 原始 `broken shell` -> `broken_shell`
- `Dataset` 原始 `flavor` -> `flashover_pollution`
- `Dataset` 原始 `diaochuan` -> `missing_disc_drop`
- CPLID normal 的 `insulator` -> `insulator_string`
- CPLID defective 的泛化 `defect` 只进入 coarse；进入 fine 前需要人工复标。

构建命令：

```bash
python3 scripts/build_unified_dataset.py --mode all --overwrite
python3 scripts/make_crop_dataset.py --input datasets/unified_fine --output datasets/unified_fine_crops --overwrite
```

校验命令：

```bash
python3 scripts/validate_yolo_dataset.py datasets/unified_coarse --num-classes 2
python3 scripts/validate_yolo_dataset.py datasets/unified_fine --num-classes 4
python3 scripts/validate_yolo_dataset.py datasets/unified_fine_crops --num-classes 3
```

## Manual Relabeling

CPLID 缺陷图的复标模板会自动生成：

```text
datasets/reports/cplid_defective_relabel_template.csv
```

把 `fine_class` 填为以下任一值：

```text
broken_shell
flashover_pollution
missing_disc_drop
```

然后重新构建 fine 数据集：

```bash
python3 scripts/build_unified_dataset.py \
  --mode fine \
  --cplid-fine-labels datasets/reports/cplid_defective_relabel_template.csv \
  --overwrite
```

重复图片和冲突标签报告：

```text
datasets/reports/dataset_duplicate_report_fine.csv
datasets/reports/dataset_duplicate_report_coarse.csv
```

## Training

建议使用 Python 3.11、PyTorch 2.x、Ultralytics 和 CUDA GPU 环境：

```bash
pip install -r requirements.txt
```

Stage 1：coarse 预训练：

```bash
python train.py \
  --model configs/gf_insuyolo.yaml \
  --data datasets/unified_coarse/data.yaml \
  --imgsz 960 \
  --epochs 120 \
  --name gf_insuyolo_coarse_960
```

Stage 2：fine 全图检测：

```bash
python train.py \
  --model configs/gf_insuyolo.yaml \
  --weights runs/gf_insuyolo_coarse_960/weights/best.pt \
  --data datasets/unified_fine/data.yaml \
  --imgsz 960 \
  --epochs 150 \
  --name gf_insuyolo_fine_960
```

Stage 3：局部 crop 缺陷检测：

```bash
python train.py \
  --model yolo11s.pt \
  --data datasets/unified_fine_crops/data.yaml \
  --imgsz 640 \
  --epochs 120 \
  --name gf_insuyolo_local_crop
```

推荐实验矩阵见：

```text
configs/experiments.yaml
```

## Inference

单阶段推理：

```bash
python infer.py \
  --weights runs/gf_insuyolo_fine_960/weights/best.pt \
  --source path/to/images \
  --output runs/infer/predictions.json
```

全图 + 局部两阶段推理：

```bash
python infer.py \
  --weights runs/gf_insuyolo_fine_960/weights/best.pt \
  --local-weights runs/gf_insuyolo_local_crop/weights/best.pt \
  --source path/to/images \
  --two-stage \
  --output runs/infer/two_stage_predictions.json
```

输出 JSON 每张图包含：

- `image`
- `has_defect`
- `insulator_boxes`
- `defect_boxes`
- 每个框的 `class_name`、`confidence`、`source_stage`

## Implemented Research Ideas

- 多源异构标注统一：YOLO/VOC 转换、类别别名规范化、元数据追踪。
- 粗到细训练：coarse 泛化缺陷预训练，再 fine 细粒度缺陷识别。
- 全图/局部协同：全图检测绝缘子串，局部 crop 放大后检测小缺陷。
- 小目标增强：`configs/gf_insuyolo.yaml` 增加 P2 检测头。
- 频域增强：`gf_insuyolo.modules.FrequencyEnhance` 在浅层特征增强高频纹理。
