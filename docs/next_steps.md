# Next Steps for Insulator Defect Detection

当前项目已经完成：

- CPLID 缺陷图复标
- `unified_coarse` 数据集构建
- `unified_fine` 数据集构建
- `unified_fine_crops` 局部 crop 数据集构建
- 数据集校验
- 训练入口和推理入口实现

下一步重点是先跑通训练链路，再做 baseline 和改进模型对比。

复标文件已提交到：

```text
annotations/cplid_defective_fine_labels.csv
```

如果需要从原始数据重新生成与当前一致的 fine 数据集，使用：

```bash
python scripts/build_unified_dataset.py \
  --mode all \
  --cplid-fine-labels annotations/cplid_defective_fine_labels.csv \
  --overwrite
```

## 1. 激活项目 Conda 环境

```bash
conda activate insulator-defect
```

当前环境 Python 版本：

```text
Python 3.11.15
```

如果训练环境还没有安装 PyTorch 和 Ultralytics，先安装依赖：

```bash
pip install -r requirements.txt
```

## 2. 先在 Mac 上跑 Smoke Test

这一步只验证训练流程能跑通，不追求精度。

Apple Silicon Mac 可以尝试使用 `mps`：

```bash
python train.py \
  --model yolo11n.pt \
  --data datasets/unified_fine/data.yaml \
  --imgsz 640 \
  --epochs 3 \
  --batch 2 \
  --device mps \
  --name smoke_yolo11n_fine
```

如果 `mps` 报错，可以改成 CPU：

```bash
python train.py \
  --model yolo11n.pt \
  --data datasets/unified_fine/data.yaml \
  --imgsz 640 \
  --epochs 3 \
  --batch 2 \
  --device cpu \
  --name smoke_yolo11n_fine_cpu
```

## 3. 在 NVIDIA GPU 上训练 Baseline

正式实验建议使用 CUDA GPU，例如 Colab、AutoDL、Kaggle 或实验室服务器。

先训练一个成熟 YOLO baseline：

```bash
python train.py \
  --model yolo11s.pt \
  --data datasets/unified_fine/data.yaml \
  --imgsz 960 \
  --epochs 100 \
  --batch 8 \
  --device 0 \
  --name baseline_yolo11s_fine
```

训练完成后记录：

- mAP@0.5
- mAP@0.5:0.95
- 每类 Precision
- 每类 Recall
- 缺陷类 Recall

## 4. 训练 GF-InsuYOLO Coarse 预训练模型

使用 2 类 coarse 数据集学习通用绝缘子和缺陷检测能力：

```bash
python train.py \
  --model configs/gf_insuyolo.yaml \
  --data datasets/unified_coarse/data.yaml \
  --imgsz 960 \
  --epochs 120 \
  --batch 8 \
  --device 0 \
  --name gf_insuyolo_coarse
```

## 5. 用 Coarse 权重 Fine-tune 四分类模型

```bash
python train.py \
  --model configs/gf_insuyolo.yaml \
  --weights runs/gf_insuyolo_coarse/weights/best.pt \
  --data datasets/unified_fine/data.yaml \
  --imgsz 960 \
  --epochs 150 \
  --batch 8 \
  --device 0 \
  --name gf_insuyolo_fine
```

四分类类别：

```text
0 insulator_string
1 broken_shell
2 flashover_pollution
3 missing_disc_drop
```

## 6. 训练局部 Crop 缺陷检测器

局部模型只检测 3 个缺陷类，不检测绝缘子串：

```bash
python train.py \
  --model yolo11s.pt \
  --data datasets/unified_fine_crops/data.yaml \
  --imgsz 640 \
  --epochs 100 \
  --batch 16 \
  --device 0 \
  --name local_crop_detector
```

局部 crop 类别：

```text
0 broken_shell
1 flashover_pollution
2 missing_disc_drop
```

## 7. 两阶段推理测试

使用全图模型检测绝缘子串和明显缺陷，再使用局部 crop 模型增强小缺陷识别：

```bash
python infer.py \
  --weights runs/gf_insuyolo_fine/weights/best.pt \
  --local-weights runs/local_crop_detector/weights/best.pt \
  --source test_images \
  --two-stage \
  --output runs/infer/two_stage_predictions.json
```

输出 JSON 中每张图包含：

- `image`
- `has_defect`
- `insulator_boxes`
- `defect_boxes`
- `class_name`
- `confidence`
- `source_stage`

## 8. 对比和消融实验

至少完成以下对比：

```text
baseline_yolo11s_fine
gf_insuyolo_fine
gf_insuyolo_fine + local_crop_detector
```

建议消融：

- 不使用 coarse 预训练
- 不使用局部 crop 检测器
- 不使用 P2 小目标检测头
- 不使用 FrequencyEnhance 频域增强模块

最终论文重点比较：

- 检测精度是否提升
- 小目标缺陷 Recall 是否提升
- 缺陷类别识别是否更稳定
- 两阶段推理是否减少漏检
