# Reproduce Mac Datasets on Windows

目标：在 Windows 电脑上重新生成和 Mac 当前项目一致的生成数据集。

由于 `datasets/` 和 `merged_dataset/` 是生成产物，体积较大，项目不会把它们提交到 Git。Windows 端需要用 Git 中的原始数据和脚本重新生成。

## 1. 拉取项目

```powershell
git clone git@github.com:ybx121/insulator-defect-detection.git
cd insulator-defect-detection
```

确认以下目录存在：

```text
Dataset/
InsulatorDataSet/
annotations/cplid_defective_fine_labels.csv
scripts/
```

其中 `annotations/cplid_defective_fine_labels.csv` 是 CPLID 248 张缺陷图的细分类复标文件。Windows 必须使用这个文件，才能生成和 Mac 一致的 `unified_fine`。

## 2. 创建 Conda 环境

建议使用和 Mac 一致的 Python 版本：

```powershell
conda create -y -n insulator-defect python=3.11 pillow pyyaml
conda activate insulator-defect
python --version
```

期望主版本为：

```text
Python 3.11.x
```

Mac 当前环境是：

```text
Python 3.11.15
```

## 3. 生成 Coarse 和 Fine 数据集

在项目根目录执行：

```powershell
python scripts/build_unified_dataset.py `
  --mode all `
  --cplid-fine-labels annotations/cplid_defective_fine_labels.csv `
  --overwrite
```

这会生成：

```text
datasets/unified_coarse
datasets/unified_fine
datasets/reports
```

期望数量：

```text
datasets/unified_coarse images 1960 labels 1960
datasets/unified_fine   images 1960 labels 1960
```

`unified_fine` 的类别统计应为：

```text
0 insulator_string      2476
1 broken_shell           584
2 flashover_pollution    196
3 missing_disc_drop      730
```

## 4. 生成局部 Crop 数据集

```powershell
python scripts/make_crop_dataset.py `
  --input datasets/unified_fine `
  --output datasets/unified_fine_crops `
  --overwrite
```

期望数量：

```text
datasets/unified_fine_crops images 1261 labels 1261
```

类别统计应为：

```text
0 broken_shell           572
1 flashover_pollution    192
2 missing_disc_drop      694
```

## 5. 生成旧版 merged_dataset

如果你还需要和 Mac 上同名的旧版 `merged_dataset/`，执行：

```powershell
python scripts/merge_datasets.py --overwrite
```

期望数量：

```text
merged_dataset images 2092 labels 2092
```

注意：`merged_dataset` 是早期保守合并版本，训练研究方案主要使用 `datasets/unified_*`。

## 6. 自动校验

依次运行：

```powershell
python scripts/validate_yolo_dataset.py datasets/unified_coarse --num-classes 2
python scripts/validate_yolo_dataset.py datasets/unified_fine --num-classes 4
python scripts/validate_yolo_dataset.py datasets/unified_fine_crops --num-classes 3
python scripts/validate_yolo_dataset.py merged_dataset --num-classes 4
```

全部应输出：

```text
validation passed
```

## 7. 快速数量检查

PowerShell 中可以运行：

```powershell
@(Get-ChildItem datasets/unified_coarse/images -Recurse -File).Count
@(Get-ChildItem datasets/unified_coarse/labels -Recurse -File).Count
@(Get-ChildItem datasets/unified_fine/images -Recurse -File).Count
@(Get-ChildItem datasets/unified_fine/labels -Recurse -File).Count
@(Get-ChildItem datasets/unified_fine_crops/images -Recurse -File).Count
@(Get-ChildItem datasets/unified_fine_crops/labels -Recurse -File).Count
@(Get-ChildItem merged_dataset/images -Recurse -File).Count
@(Get-ChildItem merged_dataset/labels -Recurse -File).Count
```

期望输出顺序：

```text
1960
1960
1960
1960
1261
1261
2092
2092
```

## 8. 关于“完全一样”

Windows 生成的数据集应与 Mac 保持一致：

- 相同原始数据来源
- 相同复标 CSV
- 相同随机种子
- 相同 train/val/test 切分
- 相同图片数量
- 相同标签数量
- 相同类别统计

唯一正常差异是 `data.yaml` 里的 `path:` 字段。它会写入 Windows 本机绝对路径，因此不会和 Mac 的绝对路径字符串完全一样。
