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

## 从零开始训练与验证

下面流程以 Windows PowerShell + NVIDIA CUDA 环境为例。Linux/macOS 可以把反引号
换成 `\`，路径分隔符换成 `/`。

### 1. 准备环境

建议使用 Python 3.11、PyTorch 2.x、Ultralytics 和 CUDA GPU 环境。

```powershell
conda activate insulator-defect
python --version
```

Windows NVIDIA GPU 机器建议先安装 CUDA 版 PyTorch，再安装项目依赖：

```powershell
pip uninstall -y torch torchvision torchaudio
pip install -r .\requirements-win-cuda.txt
pip install -r .\requirements.txt
```

确认 PyTorch 能看到 CUDA：

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

最后一行应为：

```text
True
```

如果没有 CUDA，也可以把后续命令里的 `--device 0` 改成 `--device cpu`，但训练会慢很多。

### 2. 构建数据集

如果需要从原始数据重新生成当前项目使用的数据集，执行：

```powershell
python .\scripts\build_unified_dataset.py `
  --mode all `
  --cplid-fine-labels .\annotations\cplid_defective_fine_labels.csv `
  --overwrite

python .\scripts\make_crop_dataset.py `
  --input .\datasets\unified_fine `
  --output .\datasets\unified_fine_crops `
  --overwrite
```

构建完成后应得到三套数据：

- `datasets/unified_coarse`：2 类，粗粒度预训练。
- `datasets/unified_fine`：4 类，全图主模型训练和验证。
- `datasets/unified_fine_crops`：3 类，局部 crop 缺陷检测器训练。

### 3. 校验数据集

训练前先检查标签格式、类别编号和图片路径：

```powershell
python .\scripts\validate_yolo_dataset.py .\datasets\unified_coarse --num-classes 2
python .\scripts\validate_yolo_dataset.py .\datasets\unified_fine --num-classes 4
python .\scripts\validate_yolo_dataset.py .\datasets\unified_fine_crops --num-classes 3
```

三条命令都通过后再开始训练。

### 4. 跑通 Smoke Test

这一步只验证环境、CUDA、数据集和训练入口能正常工作，不用于论文对比。

```powershell
python .\train.py `
  --model yolo11n.pt `
  --data .\datasets\unified_fine\data.yaml `
  --imgsz 640 `
  --epochs 3 `
  --batch 2 `
  --device 0 `
  --name smoke_yolo11n_fine_win_cuda
```

训练日志中出现下面类似内容，说明训练流程已经跑通：

```text
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
3 epochs completed
Results saved to ...
```

PowerShell 中如果进度条变成一行一行输出，通常只是终端宽度或进度条刷新显示问题，不影响训练结果。

### 5. 训练 Baseline

Baseline 使用 Ultralytics 官方 `yolo11s.pt`，直接在 `unified_fine` 四分类数据集上训练。
它不使用 GF-InsuYOLO 的频域增强、P2 小目标检测头、coarse 预训练或两阶段局部检测，
用于后续改进模型对比。

```powershell
python .\train.py `
  --model yolo11s.pt `
  --data .\datasets\unified_fine\data.yaml `
  --imgsz 960 `
  --epochs 100 `
  --batch 8 `
  --device 0 `
  --name baseline_yolo11s_fine
```

输出目录通常为：

```text
runs/detect/runs/baseline_yolo11s_fine
```

如果同名目录已存在，Ultralytics 会自动追加后缀，例如：

```text
runs/detect/runs/baseline_yolo11s_fine-2
```

当前已完成的一次 baseline 结果为：

```text
baseline_yolo11s_fine-2
Precision: 0.825
Recall:    0.760
mAP50:     0.780
mAP50-95:  0.528
```

分类别结果：

```text
insulator_string       mAP50=0.951  Recall=0.963
broken_shell           mAP50=0.606  Recall=0.565
flashover_pollution    mAP50=0.601  Recall=0.562
missing_disc_drop      mAP50=0.962  Recall=0.948
```

这个 baseline 的主要问题是 `broken_shell` 和 `flashover_pollution` 漏检较多，
后续 GF-InsuYOLO 重点观察这两个缺陷类的 Recall 和 mAP 是否提升。

### 6. 训练 GF-InsuYOLO Coarse 预训练模型

Coarse 阶段使用 2 类数据集：

```text
0 insulator_string
1 defect
```

模型结构来自 `configs/gf_insuyolo.yaml`，包含：

- P2 小目标检测头，用于提升小缺陷检测能力。
- `FrequencyEnhance` 浅层频域增强模块，用于增强裂纹、破损边缘、污闪纹理等高频信息。

训练命令：

```powershell
python .\train.py `
  --model .\configs\gf_insuyolo.yaml `
  --data .\datasets\unified_coarse\data.yaml `
  --imgsz 960 `
  --epochs 120 `
  --batch 8 `
  --device 0 `
  --name gf_insuyolo_coarse
```

训练完成后记录：

```text
runs/detect/runs/gf_insuyolo_coarse/weights/best.pt
```

如果目录被自动加后缀，后续 `--weights` 路径要同步改成实际目录名。

### 7. Fine-tune 四分类 GF-InsuYOLO

使用 coarse 阶段的 `best.pt` 作为初始化权重，在 `unified_fine` 四分类数据集上继续训练：

```powershell
python .\train.py `
  --model .\configs\gf_insuyolo.yaml `
  --weights .\runs\detect\runs\gf_insuyolo_coarse\weights\best.pt `
  --data .\datasets\unified_fine\data.yaml `
  --imgsz 960 `
  --epochs 150 `
  --batch 8 `
  --device 0 `
  --name gf_insuyolo_fine
```

四分类类别为：

```text
0 insulator_string
1 broken_shell
2 flashover_pollution
3 missing_disc_drop
```

训练完成后重点对比 baseline：

- overall `mAP50`
- overall `mAP50-95`
- 每类 Precision
- 每类 Recall
- `broken_shell` 和 `flashover_pollution` 的 Recall

### 8. 训练局部 Crop 缺陷检测器

局部模型只检测缺陷，不检测绝缘子串。类别为：

```text
0 broken_shell
1 flashover_pollution
2 missing_disc_drop
```

训练命令：

```powershell
python .\train.py `
  --model yolo11s.pt `
  --data .\datasets\unified_fine_crops\data.yaml `
  --imgsz 640 `
  --epochs 100 `
  --batch 16 `
  --device 0 `
  --name local_crop_detector
```

局部 crop 模型用于两阶段推理：第一阶段在全图中找绝缘子和明显缺陷，
第二阶段对绝缘子区域 crop 后再检测小缺陷。

### 9. 验证训练好的权重

训练完成后，Ultralytics 会自动验证 `best.pt` 并打印最终指标。也可以手动重新验证：

```powershell
yolo detect val `
  model=.\runs\detect\runs\baseline_yolo11s_fine-2\weights\best.pt `
  data=.\datasets\unified_fine\data.yaml `
  imgsz=960 `
  batch=8 `
  device=0
```

验证 GF-InsuYOLO fine 时，把 `model=` 换成对应权重：

```powershell
yolo detect val `
  model=.\runs\detect\runs\gf_insuyolo_fine\weights\best.pt `
  data=.\datasets\unified_fine\data.yaml `
  imgsz=960 `
  batch=8 `
  device=0
```

### 10. 查看训练效果图

每次训练的输出目录中都会生成结果图。最常用的是：

```text
results.png
confusion_matrix.png
confusion_matrix_normalized.png
BoxPR_curve.png
val_batch0_labels.jpg
val_batch0_pred.jpg
val_batch1_pred.jpg
val_batch2_pred.jpg
weights/best.pt
weights/last.pt
```

PowerShell 中可以直接打开目录：

```powershell
ii .\runs\detect\runs\baseline_yolo11s_fine-2
```

也可以直接打开关键图片：

```powershell
ii .\runs\detect\runs\baseline_yolo11s_fine-2\results.png
ii .\runs\detect\runs\baseline_yolo11s_fine-2\confusion_matrix_normalized.png
ii .\runs\detect\runs\baseline_yolo11s_fine-2\val_batch0_pred.jpg
```

查看重点：

- `results.png`：loss 是否下降，mAP 是否稳定上升。
- `confusion_matrix_normalized.png`：看哪些类别容易漏检或混淆。
- `BoxPR_curve.png`：看每个类别的 Precision-Recall 曲线。
- `val_batch*_pred.jpg`：看预测框是否漏检、误检、框偏移或重复框。

### 11. 使用模型预测图片

用训练好的 baseline 权重预测测试集图片：

```powershell
yolo detect predict `
  model=.\runs\detect\runs\baseline_yolo11s_fine-2\weights\best.pt `
  source=.\datasets\unified_fine\images\test `
  imgsz=960 `
  conf=0.25 `
  save=True
```

预测结果通常保存到：

```text
runs/detect/predict
```

使用项目的推理脚本做单阶段 JSON 输出：

```powershell
python .\infer.py `
  --weights .\runs\detect\runs\gf_insuyolo_fine\weights\best.pt `
  --source .\datasets\unified_fine\images\test `
  --output .\runs\infer\predictions.json
```

全图 + 局部两阶段推理：

```powershell
python .\infer.py `
  --weights .\runs\detect\runs\gf_insuyolo_fine\weights\best.pt `
  --local-weights .\runs\detect\runs\local_crop_detector\weights\best.pt `
  --source .\datasets\unified_fine\images\test `
  --two-stage `
  --output .\runs\infer\two_stage_predictions.json
```

输出 JSON 每张图包含：

- `image`
- `has_defect`
- `insulator_boxes`
- `defect_boxes`
- 每个框的 `class_name`、`confidence`、`source_stage`

### 12. 实验对比顺序

建议按下面顺序记录结果：

```text
1. baseline_yolo11s_fine
2. gf_insuyolo_coarse
3. gf_insuyolo_fine
4. local_crop_detector
5. gf_insuyolo_fine + local_crop_detector two-stage
```

最终至少比较：

```text
baseline_yolo11s_fine
gf_insuyolo_fine
gf_insuyolo_fine + local_crop_detector
```

推荐实验矩阵见：

```text
configs/experiments.yaml
```

## Implemented Research Ideas

- 多源异构标注统一：YOLO/VOC 转换、类别别名规范化、元数据追踪。
- 粗到细训练：coarse 泛化缺陷预训练，再 fine 细粒度缺陷识别。
- 全图/局部协同：全图检测绝缘子串，局部 crop 放大后检测小缺陷。
- 小目标增强：`configs/gf_insuyolo.yaml` 增加 P2 检测头。
- 频域增强：`gf_insuyolo.modules.FrequencyEnhance` 在浅层特征增强高频纹理。
