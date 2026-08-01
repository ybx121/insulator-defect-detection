# `Dataset/labels` 单数据集完整优化执行方案

本文是基于 2026-07-31 本地实测结果重新制定的当前执行方案。方案范围严格限定为：

```text
Dataset/labels
```

不把 `Supervisely`、`primary_full_v1`、历史伪标签、裁剪集或其他派生数据并入训练和正式验证。

> 核心结论：`best.pt` 在当前 `Dataset/labels` 验证划分上确实可以复现 `mAP50 ≈ 0.87`；但当前划分存在跨 train/val/test 的近重复图组，尤其集中影响 `broken_shell`。因此，现有 0.87 可以作为“历史复现成绩”，不能直接作为无泄漏泛化能力的最终证据。

<a id="toc"></a>
## 目录

- [1. 执行摘要](#section-1)
- [2. 数据集与权重合同](#section-2)
- [3. 本地验证方法与结果](#section-3)
- [4. 当前瓶颈的重新排序](#section-4)
- [5. 优化目标与评价原则](#section-5)
- [6. 不可违反的实验规则](#section-6)
- [7. 阶段 0：冻结当前历史基准](#section-7)
- [8. 阶段 1：构建无泄漏分组数据版本](#section-8)
- [9. 阶段 2：统一标注规范并进行盲审](#section-9)
- [10. 阶段 3：建立可比较的干净基线](#section-10)
- [11. 阶段 4：针对性训练优化](#section-11)
- [12. 阶段 5：统一评估与误差诊断](#section-12)
- [13. 实验矩阵、晋级门槛与停止条件](#section-13)
- [14. 锁定测试集评估](#section-14)
- [15. 推荐的一步一步执行顺序](#section-15)
- [16. 产物目录与完成检查表](#section-16)
- [17. 参考资料](#section-17)

<a id="section-1"></a>
## 1. 执行摘要

### 1.1 已确认的事实

1. `best.pt` 是标准 Ultralytics `YOLO11s` 四分类目标检测模型。
2. 权重不是 P2 模型，也不是 GF-InsuYOLO 或 D-FINE。
3. 权重内保存的旧训练数据配置为 `datasets/unified_fine/data.yaml`，输入尺寸为 960。
4. 权重内保存的历史指标为：
   - `mAP50 = 0.87563`；
   - `mAP50-95 = 0.59168`。
5. 在当前 `Dataset/labels` 的 196 张验证图上，本地重新验证得到：
   - Ultralytics 原生评估：`mAP50 = 0.87177`，`mAP50-95 = 0.58876`；
   - 项目 COCO 评估器：`mAP50 = 0.87019`，`mAP50-95 = 0.59013`。
6. 两套评估实现结果高度一致，可以确认“mAP50 约为 0.87”的说法能够在当前划分上复现。

### 1.2 不能忽略的限制

当前数据划分没有完全做到相似图组隔离：

- 没有发现跨划分的逐字节完全重复图；
- 发现 52 个近重复图组跨越原 train/val/test；
- 其中涉及 26 张验证图，占当前验证集的 13.3%；
- 这 26 张图包含 35 个 `broken_shell` 实例，占验证集该类 50 个实例的 70%。

排除这 26 张图后的诊断性结果为：

```text
mAP50      0.77867
mAP50-95   0.55162
```

其中 `broken_shell` AP50 从约 0.789 降至约 0.349。不过，排除后只剩 15 个 `broken_shell` 实例，类别组成发生明显变化，因此这个 0.77867 只能证明当前成绩对近重复组敏感，不能替代正式无泄漏验证结果。

### 1.3 当前最高优先级

当前正确的优化顺序是：

```text
冻结 0.87 历史复现结果
  → 按近重复图组重新划分数据
  → 冻结新的数据指纹
  → 对新验证集做隐藏预测的标注盲审
  → 从官方 COCO 预训练权重重新训练 YOLO11s
  → 再开展输入分辨率、P2、小目标和模型容量实验
  → 最后只评估一次锁定测试集
```

在无泄漏基准建立以前，不应把继续堆叠模型、增加集成分支或调验证集阈值作为第一优先级。

<a id="section-2"></a>
## 2. 数据集与权重合同

### 2.1 数据集规模

| 划分 | 图像数 | 类别 0 | 类别 1 | 类别 2 | 类别 3 | 目标总数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1568 | 2031 | 492 | 202 | 583 | 3308 |
| val | 196 | 267 | 50 | 29 | 80 | 426 |
| test | 196 | 278 | 53 | 16 | 72 | 419 |
| 合计 | 1960 | 2576 | 595 | 247 | 735 | 4153 |

类别顺序固定为：

```text
0 insulator_string
1 broken_shell
2 flashover_pollution
3 missing_disc_drop
```

### 2.2 当前数据指纹

本次只读计算得到的内容指纹为：

```text
245f2efa823a34fb0d2f409fff3ff6edf566bafb64966784176ee8be3923a4fc
```

计算规则为：

```text
sha256(sorted split|image_name|image_sha256|label_sha256)
```

共覆盖 1960 个图像—标签样本，读取并哈希 1,606,786,565 字节。

当前原始目录没有内置正式版本元数据，因此在进入新实验前，必须把上述指纹、划分清单、构建参数和类别统计写入新数据版本的 `metadata` 目录。

### 2.3 权重架构

| 项目 | 值 |
| --- | --- |
| 框架 | Ultralytics 8.4.90 |
| 任务 | 目标检测 |
| 网络 | YOLO11s |
| 模型配置 | `yolo11s.yaml`，scale=`s` |
| 类别数 | 4 |
| 检测层步长 | 8、16、32 |
| 未融合参数量 | 9,429,340 |
| 融合后参数量 | 9,414,348 |
| 融合后计算量 | 21.3 GFLOPs |
| 原训练输入 | 960 |
| 原训练 batch | 4 |
| 原训练 seed | 0 |
| 原训练 rect | false |

### 2.4 权重哈希与归档位置

原始权重：

```text
E:\School\insulator-defect-detection\best.pt
```

SHA-256：

```text
8687c9b674f86f74a7369bf891a8869d3a3d99d238e17a06eac45044dda9eae8
```

规范化归档副本：

```text
runs/detect/imported/yolo11s_unified_fine_img960_seed0_20260725/weights/
yolo11s_unified_fine_img960_seed0_legacy-best_sha8687c9b6.pt
```

权重清单：

```text
runs/detect/imported/yolo11s_unified_fine_img960_seed0_20260725/
checkpoint_manifest.json
```

原始 `best.pt` 被保留，归档副本与原文件的 SHA-256 完全一致。

<a id="section-3"></a>
## 3. 本地验证方法与结果

### 3.1 验证环境

```text
Python       3.11.15
PyTorch      2.13.0+cu126
Ultralytics  8.4.90
CUDA         12.6
GPU          NVIDIA GeForce RTX 4080 Laptop GPU
```

### 3.2 固定验证参数

```text
split       val
images      196
imgsz       960
batch       4
rect        false
conf        0.001
iou         0.7
max_det     300
tta         false
device      0
```

评估配置：

```text
configs/dataset_labels_eval.yaml
```

这个配置只引用 `Dataset/labels`，没有加入任何外部或辅助数据。

### 3.3 总体结果

| 评估实现 | mAP50 | mAP50-95 | AP75 |
| --- | ---: | ---: | ---: |
| Ultralytics 原生 | 0.87177 | 0.58876 | 未单独导出 |
| 项目 COCO 评估器 | 0.87019 | 0.59013 | 0.66648 |

两套实现的差异小于 0.002，属于实现细节造成的正常小幅差异。

### 3.4 分类别结果

以下使用项目 COCO 评估器结果：

| 类别 | AP50 | AP50-95 | AP75 |
| --- | ---: | ---: | ---: |
| `insulator_string` | 0.95373 | 0.81157 | 0.91678 |
| `broken_shell` | 0.78915 | 0.44440 | 0.42570 |
| `flashover_pollution` | 0.81432 | 0.48301 | 0.60361 |
| `missing_disc_drop` | 0.92359 | 0.62157 | 0.71983 |

### 3.5 目标尺寸结果

```text
AP-small    0.26733
AP-medium   0.63231
AP-large    0.57374
```

AP-small 明显低于 AP-medium，说明小目标仍是主要技术瓶颈之一。但在解决数据划分问题前，不应把该差距全部归因于网络结构。

### 3.6 运行点误报

在置信度阈值 0.25 下：

```text
无缺陷图像数            66
出现缺陷误报的图像数     0
无缺陷图像误报率          0.0%
```

这个结果说明当前模型在该验证划分上的主要问题更偏向缺陷召回和定位，不是正常图大量误报。

### 3.7 结果文件

原生验证：

```text
runs/validate/dataset_labels__yolo11s__img960__val/
```

项目 COCO 评估：

```text
runs/eval/dataset_labels__yolo11s__img960__val_coco.json
runs/eval/dataset_labels__yolo11s__img960__val_coco_per_class.csv
runs/eval/leaderboard_dataset_labels.csv
```

<a id="section-4"></a>
## 4. 当前瓶颈的重新排序

### 4.1 P0：跨划分近重复图组泄漏

这是当前最先需要解决的问题。

证据：

- 52 个感知相似图组跨越原 train/val/test；
- 26 张验证图处于跨划分近重复组；
- 28 张测试图也处于跨划分近重复组；
- 这些验证图集中包含 70% 的 `broken_shell` 实例。

影响：

- 当前 0.87 可能高估独立场景泛化能力；
- `broken_shell` AP 对这批相关样本高度敏感；
- 当前 test 也不能直接作为严格锁定、无泄漏测试集使用。

### 4.2 P0：`broken_shell` 独立泛化证据不足

当前验证集有 50 个 `broken_shell` 实例，其中 35 个位于跨划分近重复组。排除相关图后只剩 15 个实例，样本量过小。

因此，当前问题不是简单地说“broken AP 为 0.789，需要再加 0.16”，而是：

1. 先建立不含近重复泄漏的新验证集；
2. 确保新验证集中有足够数量、足够多样的 broken 实例；
3. 再判断模型是否真的缺少分类、定位或召回能力。

### 4.3 P1：`flashover_pollution` 样本量偏少

全数据集该类只有 247 个实例：

```text
train 202
val    29
test   16
```

16 个测试实例不足以稳定估计最终 AP。重新分组时必须做类别分层，同时保留场景独立性；如果无法同时满足，应优先补充真实独立样本，而不是通过复制或增强验证样本扩大计数。

### 4.4 P1：定位精度仍有明显提升空间

总体指标：

```text
AP50       0.87019
AP75       0.66648
AP50-95    0.59013
```

`broken_shell` 的 AP50 为 0.78915，但 AP75 只有 0.42570；这说明除了召回外，框边界一致性和精确定位也是瓶颈。

### 4.5 P1：小目标检测能力不足

AP-small 只有 0.26733，远低于 AP-medium。可能原因包括：

- P3 最小步长为 8，细小缺陷特征分辨率有限；
- 960 输入下部分缺陷仍然只占少量像素；
- 标注边界对微小目标的相对误差更敏感；
- 少数类训练样本不足。

### 4.6 P2：模型容量不是第一优先级

当前 YOLO11s 已经取得较强的绝缘子串和掉片类别结果。直接换 YOLO11m、D-FINE 或多模型集成可能增加分数，但无法修复：

- 数据泄漏；
- 类别样本不足；
- 标注不一致；
- 测试集不独立。

因此更大模型应放在无泄漏基线建立之后。

<a id="section-5"></a>
## 5. 优化目标与评价原则

### 5.1 两类基准必须分开

历史复现基准：

```text
Dataset/labels 当前划分
mAP50 ≈ 0.87
```

它只用于证明旧权重和旧成绩可复现。

正式优化基准：

```text
Dataset/labels 图像内容
按近重复图组重新划分
从通用预训练权重重新训练
```

正式模型比较只在新基准内部进行，不能把新划分分数和旧 0.87 直接解释为模型升降。

### 5.2 优化目标的顺序

1. 建立无泄漏、可复现的数据版本；
2. 建立 YOLO11s 干净基线；
3. 提升最弱类别的独立场景 AP；
4. 提升 AP75 和 AP50-95；
5. 提升 AP-small；
6. 保持正常图误报率可控；
7. 最后再冲击严格四分类 `mAP50 > 0.95`。

### 5.3 不以单一 overall 指标决策

每个候选实验至少报告：

- overall AP50、AP75、AP50-95；
- 四个类别的 AP50 和 AP50-95；
- AP-small、AP-medium、AP-large；
- 每类 GT 数量；
- 正常图误报率；
- group bootstrap 95% 置信区间；
- 两个随机种子的均值和标准差。

<a id="section-6"></a>
## 6. 不可违反的实验规则

1. 正式训练和验证只使用 `Dataset/labels` 中已有图像，不引入 Supervisely。
2. 原始 `Dataset/labels` 永久只读，不在原地改划分或覆盖标签。
3. 相似图和派生变体必须属于同一个 split。
4. test 不用于选模型、调阈值、调增强或决定早停。
5. 任何标签修改都创建新的数据版本和新指纹。
6. 验证集人工复核时隐藏所有模型预测。
7. 重分组后的正式模型必须从官方通用预训练权重重新训练。
8. 旧 `best.pt` 不得作为重分组基准的 warm start，因为它可能已经见过新 val/test 中的旧 train 图像或相似变体。
9. 所有模型比较必须使用相同数据指纹、相同评估器和相同推理参数。
10. 当前历史排行榜与新无泄漏排行榜物理隔离。

<a id="section-7"></a>
## 7. 阶段 0：冻结当前历史基准

### 7.1 已完成产物

- 数据只读内容指纹；
- 权重 SHA-256；
- 权重架构和训练参数；
- Ultralytics 原生验证；
- 项目 COCO 独立验证；
- 权重规范化归档；
- 数据配对、类别和坐标合法性检查；
- 精确重复和近重复划分审计。

### 7.2 复现命令

```powershell
python -c "from ultralytics import YOLO; YOLO('best.pt').val(data='configs/dataset_labels_eval.yaml', split='val', imgsz=960, batch=4, device='0', workers=0, conf=0.001, iou=0.7, max_det=300, rect=False, plots=True, save_json=True)"
```

项目 COCO 复核：

```powershell
python .\scripts\evaluate_detector.py `
  --weights .\best.pt `
  --data .\configs\dataset_labels_eval.yaml `
  --split val `
  --mode standard `
  --imgsz 960 `
  --batch 4 `
  --conf 0.001 `
  --operating-conf 0.25 `
  --iou 0.7 `
  --device 0 `
  --bootstrap 200 `
  --seed 20260731 `
  --output .\runs\eval\dataset_labels_yolo11s_val_bootstrap.json `
  --leaderboard .\runs\eval\leaderboard_dataset_labels_history.csv
```

### 7.3 阶段验收条件

- 指纹、权重哈希和评估参数已记录；
- 原生与 COCO mAP 差异小于 0.005；
- 旧权重归档副本哈希与原文件一致；
- 历史结果明确标记为“当前原划分结果”。

<a id="section-8"></a>
## 8. 阶段 1：构建无泄漏分组数据版本

### 8.1 新版本命名

建议输出：

```text
datasets/primary_labels_grouped_v1
```

这个版本只允许读取：

```text
Dataset/labels/images/**
Dataset/labels/labels/**
```

### 8.2 构建步骤

1. 收集 1960 个图像—标签对；
2. 检查缺失标签、孤立标签、非法类别和越界坐标；
3. 按图像内容哈希做精确去重；
4. 对标签冲突的完全重复图进入人工裁决，不能自动随机选择；
5. 使用感知哈希建立近重复组；
6. 将同一组整体分配到 train、val 或 test；
7. 按四类目标数量和图像数量做 group-aware 分层；
8. 输出每张图的原路径、组 ID、新 split、图像哈希和标签哈希；
9. 验证任何 group 不跨 split；
10. 计算新的数据指纹。

### 8.3 推荐划分策略

默认比例：

```text
train 80%
val   10%
test  10%
```

约束条件：

- 先保证 group 不泄漏，再追求类别比例接近；
- val 和 test 都必须包含四类；
- 每个弱类尽量覆盖多个独立拍摄系列；
- 不用复制、旋转或翻转图补足 val/test 数量；
- 增强只发生在训练时。

### 8.4 样本量不足时的处理

如果 group-aware split 后 `flashover_pollution` 的 val 或 test 实例过少：

1. 不降低 group 隔离要求；
2. 使用 group 级五折交叉验证评估模型开发阶段稳定性；
3. test 仍保持一次性最终评估；
4. 后续只补充来自新场景的该类真实图像。

### 8.5 关于旧权重的限制

如果重新划分把旧 train 图像分到新 val/test，旧 `best.pt` 已经见过这些图，不能用于正式分数。

因此：

- 旧权重可以用于生成审核候选；
- 旧权重可以作为部署参考；
- 旧权重不能作为新无泄漏验证集上的公平基线；
- 新基线必须从 `yolo11s.pt` 等通用预训练权重开始。

### 8.6 阶段验收条件

- 数据来源只有 `Dataset/labels`；
- 1960 个原始样本全部有明确去向或排除原因；
- 精确重复和近重复 group 跨 split 数量为 0；
- val/test 四类均有覆盖；
- 新指纹和 manifest 已生成；
- 原始目录没有变化。

<a id="section-9"></a>
## 9. 阶段 2：统一标注规范并进行盲审

### 9.1 为什么仍然需要复核

数据结构合法不代表语义标注一致。当前结果显示：

- `broken_shell` AP50 与 AP75 差距很大；
- `flashover_pollution` 样本较少；
- 小框的边界误差会显著影响 AP50-95；
- 新分组后验证集组成会改变。

### 9.2 “独立专家复核”的含义

这不是让模型针对验证集样本进行优化，也不是把验证样本加入训练。

正确含义是：

1. 审核人员查看原图和现有人工标签；
2. 审核界面隐藏模型预测、置信度和错误类型；
3. 按预先写好的类别和边界规范判断标签；
4. 两名审核者独立给出决定；
5. 冲突由第三人或共同会议裁决；
6. 审核完成后冻结标签，训练人员不能根据模型结果继续修改验证集。

### 9.3 标注规范必须明确

对每类定义：

- 正例边界；
- 负例边界；
- 最小可见面积；
- 遮挡和模糊处理；
- 同一缺陷多个部位是一个框还是多个框；
- 框住缺陷本体还是上下文区域；
- `ambiguous` 和 `ignore` 的使用条件。

重点统一：

```text
broken_shell：框破损缺口、整片破损区域，还是整片绝缘子片
flashover_pollution：框污染痕迹、放电痕迹，还是包含上下文的区域
missing_disc_drop：框缺失位置、剩余间隙，还是整段串结构
```

### 9.4 验证集审核流程

1. 新 split 生成后立即冻结图像清单；
2. 第一位审核者检查全部 val 图像，重点查漏标；
3. 第二位审核者独立检查全部弱类图像；
4. 对其余图像随机抽取至少 20% 复核；
5. 记录 `keep`、`modify`、`add`、`remove`、`ambiguous`；
6. 冲突裁决；
7. 构建新的标注版本和指纹；
8. 冻结验证标签。

### 9.5 训练集审核流程

训练集可以使用旧权重和多个候选模型挖掘疑似漏标，但模型输出只能作为候选：

- 高置信预测且无匹配 GT；
- 两个模型共同预测；
- 单模型预测但局部特征明显；
- 密集小目标区域；
- 审核者最终确认后才写入标签。

验证集审核不得采用这种模型引导方式。

### 9.6 阶段验收条件

- 标注规范已冻结；
- 验证审核未显示模型预测；
- 每个修改有审核人与原因；
- ambiguous 没有被当作普通背景；
- 新标签版本有独立指纹；
- 训练和验证修改记录物理隔离。

<a id="section-10"></a>
## 10. 阶段 3：建立可比较的干净基线

### 10.1 基线模型顺序

第一批只训练以下模型：

| 编号 | 模型 | 目的 |
| --- | --- | --- |
| B0 | YOLO11s，960 | 建立与历史架构最接近的干净基线 |
| B1 | YOLO11s，1280 | 验证提高输入分辨率的收益 |
| B2 | YOLO11m，960 | 验证模型容量收益 |
| B3 | YOLO11s-P2，960 | 验证更高分辨率检测头对小目标的收益 |

D-FINE、切片、两阶段和集成放到第二轮，避免一开始同时改变过多变量。

### 10.2 初始化方式

正式基线：

```text
yolo11s.pt
yolo11m.pt
```

禁止用归档的旧 `best.pt` 初始化正式基线，因为新划分可能包含它训练时见过的图像。

### 10.3 YOLO11s 基线命令模板

```powershell
python -c @"
from ultralytics import YOLO

model = YOLO('yolo11s.pt')
model.train(
    data='datasets/primary_labels_grouped_v1/data.yaml',
    epochs=100,
    patience=30,
    imgsz=960,
    batch=4,
    device=0,
    workers=8,
    seed=20260731,
    deterministic=True,
    rect=False,
    cos_lr=True,
    close_mosaic=10,
    project='runs/detect/primary_labels_grouped_v1',
    name='yolo11s_img960_seed20260731',
)
"@
```

第二个随机种子至少使用：

```text
20260801
```

### 10.4 基线对照要求

所有模型保持：

- 同一数据指纹；
- 同一 train/val/test；
- 相同 epoch 上限和早停规则；
- 相同评估尺寸；
- 相同 NMS 参数；
- 相同两个随机种子；
- 相同评估器。

### 10.5 阶段验收条件

- YOLO11s 两个种子完成；
- 结果能从日志和权重哈希复现；
- 指标包含分类别和目标尺寸分解；
- 没有使用旧 `best.pt` warm start；
- 新基线成为后续所有实验的对照组。

<a id="section-11"></a>
## 11. 阶段 4：针对性训练优化

### 11.1 优先级 A：小目标特征

在标注和划分可信后，依次测试：

1. 输入从 960 提高到 1280；
2. 使用 P2 检测头；
3. 对大图进行训练裁剪，但必须按原图 group 分配 split；
4. 使用 SAHI 做独立推理对照；
5. 必要时使用全图模型与局部模型融合。

一次实验只改变一个主要因素。

### 11.2 优先级 B：弱类采样

不要简单复制弱类图像。推荐：

- 以原图或 group 为采样单位；
- 提高含 `broken_shell`、`flashover_pollution` 的 group 被抽中概率；
- 保留足够普通图像，避免缺陷误报上升；
- 记录每类和每个 group 的实际曝光次数；
- 不把增强变体当作独立真实样本计数。

### 11.3 优先级 C：定位质量

针对 AP50 与 AP75 差距：

- 复核弱类框边界一致性；
- 分析预测中心命中但 IoU 不足的样本；
- 按目标尺寸统计定位误差；
- 比较 960 与 1280；
- 比较 P3 与 P2；
- 对极小框适当降低几何增强强度；
- 检查 mosaic 是否造成缺陷边界过度缩小。

### 11.4 优先级 D：缺失标注鲁棒训练

只有当人工审核确认训练集存在漏标时，才启动：

- ignore regions；
- 正负未标注学习；
- 高可信候选人工确认；
- teacher-student 或稀疏标注训练。

不能仅因为模型产生高置信预测就自动把预测写成标签。

### 11.5 优先级 E：更大模型和 D-FINE

当 YOLO11s 数据与小目标优化趋于稳定后再测试：

- YOLO11m；
- D-FINE-M；
- 单模型 TTA；
- 两模型融合；
- 全图与局部两阶段模型。

如果更大模型提升小于 0.01，且弱类和 AP-small 没有实质改善，应停止扩容。

<a id="section-12"></a>
## 12. 阶段 5：统一评估与误差诊断

### 12.1 固定评估参数

```text
imgsz          与实验约定一致
rect           false
conf           0.001
iou            0.7
max_det        300
tta            false，除非实验明确为 TTA
```

### 12.2 必报指标

- overall AP50、AP75、AP50-95；
- 每类 AP50、AP75、AP50-95；
- AP-small、AP-medium、AP-large；
- precision、recall；
- 正常图误报率；
- 混淆矩阵；
- FP、FN、定位错误和类别错误数量；
- 推理速度；
- 权重哈希和数据指纹。

### 12.3 置信区间

由于相似图必须作为一个统计单位，最终使用 group bootstrap，而不是单纯图片 bootstrap。

快速实验门控：

```text
不做 bootstrap 或只做 20 次
```

晋级候选：

```text
至少 200 次 group bootstrap
```

最终候选：

```text
至少 1000 次 group bootstrap
```

### 12.4 错误分类

每个错误归入：

```text
漏检
类别错误
定位不足
重复框
正常图误报
标注疑问
极小目标
强遮挡
边界截断
```

验证集上的“标注疑问”只能进入独立盲审流程，不能由训练人员直接修改。

<a id="section-13"></a>
## 13. 实验矩阵、晋级门槛与停止条件

### 13.1 第一轮实验矩阵

| 实验 | 数据 | 初始化 | 模型 | 尺寸 | 目的 |
| --- | --- | --- | --- | ---: | --- |
| H0 | 当前旧 split | 旧 `best.pt` | YOLO11s | 960 | 历史复现，已完成 |
| E0 | grouped v1 | `yolo11s.pt` | YOLO11s | 960 | 正式干净基线 |
| E1 | grouped v1 | `yolo11s.pt` | YOLO11s | 1280 | 分辨率收益 |
| E2 | grouped v1 | `yolo11m.pt` | YOLO11m | 960 | 容量收益 |
| E3 | grouped v1 | 通用预训练 | YOLO11s-P2 | 960 | 小目标特征收益 |
| E4 | grouped v1 | E0 初始化 | YOLO11s | 960 | group-aware 弱类采样 |

H0 与 E0-E4 不在同一排行榜比较绝对分数。

### 13.2 单次实验晋级门槛

相对 E0 同种子对照，至少满足：

- overall AP50-95 提升不少于 0.01；或
- 最弱类 AP50 提升不少于 0.03；或
- AP-small 提升不少于 0.03；
- 任一其他类别 AP50 不下降超过 0.02；
- 正常图误报率不增加超过 2 个百分点。

### 13.3 两种子晋级门槛

- 两个种子方向一致；
- 平均 AP50-95 优于 E0；
- 最弱类平均 AP50 不下降；
- 提升不是由一个相似图组贡献；
- 置信区间没有显示明显退化风险。

### 13.4 阶段性目标

新基准建立后按以下顺序推进：

1. 所有类别 AP50 不低于 0.75；
2. 所有类别 AP50 不低于 0.85；
3. overall AP50 达到 0.90；
4. overall AP50 达到 0.95；
5. group bootstrap 下界也达到项目约定门槛。

不要把旧 split 的 0.87 当作新 split 必须立即超过的数字，因为两个基准不可直接比较。

### 13.5 停止条件

- 连续三组实验平均提升小于 0.005：停止当前方向；
- 更大模型只提升强类：停止扩容；
- 训练 AP 上升而独立验证下降：检查过拟合和泄漏；
- 弱类结果由少数 group 主导：优先补数据；
- 标注争议没有解决：暂停结构搜索；
- 单模型已经稳定且集成收益小于 0.01：不再增加分支。

<a id="section-14"></a>
## 14. 锁定测试集评估

### 14.1 重新建立测试集

当前旧 test 含有 28 张属于跨原划分近重复组的图，因此不能直接作为严格无泄漏测试集。

grouped v1 构建时应重新建立 test，并满足：

- group 与 train、val 完全隔离；
- 模型开发期间不运行推理；
- 不查看分类别 test 指标来决定实验；
- 不在 test 上调置信度或 NMS；
- 最终模型、阈值和代码提交先冻结。

### 14.2 测试前冻结清单

- 数据指纹；
- Git 提交；
- 模型权重及 SHA-256；
- 推理尺寸；
- conf、IoU、max_det；
- TTA 设置；
- 评估器版本；
- 类别顺序；
- 错误分析模板。

### 14.3 最终报告

最终测试报告至少包含：

- overall 和 per-class AP；
- group bootstrap 置信区间；
- 每类 GT 数量；
- 正常图误报率；
- 推理速度和硬件；
- 失败案例；
- 数据和权重指纹；
- 与新验证基准的差异解释。

<a id="section-15"></a>
## 15. 推荐的一步一步执行顺序

### 第 1 步：冻结历史结果

- 保留当前数据指纹；
- 保留 `best.pt` 和归档副本；
- 保留两套评估结果；
- 标记 0.87 为旧 split 历史基准。

状态：已完成。

### 第 2 步：实现 primary-only 分组构建器

扩展现有构建工具，使其支持：

```text
只读取 Dataset/labels
不读取 public-root
按精确哈希去重
按感知哈希分组
按 group 分层划分
输出 manifest、统计和指纹
```

建议不要复用 `primary_full_v1` 名称，避免误以为含辅助数据。

### 第 3 步：构建 `primary_labels_grouped_v1`

- seed 固定为 20260731；
- pHash 距离先用 4；
- 检查距离 2、4、6 的敏感性；
- 选择能覆盖明显派生变体、又不过度合并不同场景的阈值；
- 生成 group 无泄漏报告。

### 第 4 步：制定标注规范

先抽取：

- 30 个 broken 样本；
- 30 个 flashover 样本；
- 20 个 missing 样本；
- 20 个正常或易混淆样本。

完成试标、冲突讨论和规范冻结。

### 第 5 步：盲审新验证集

- 全部 val 图像做一轮漏标检查；
- 全部弱类图像双人独立复核；
- 隐藏所有模型预测；
- 裁决后构建新标签版本；
- 冻结验证标签和指纹。

### 第 6 步：审核训练集漏标

- 旧 `best.pt` 只用于提出候选；
- 候选必须人工确认；
- 不自动写伪标签；
- ambiguous 使用 ignore，不当作背景。

### 第 7 步：训练 YOLO11s 两种子基线

从 `yolo11s.pt` 开始：

```text
imgsz=960
seed=20260731
seed=20260801
```

完成统一评估和 group bootstrap。

### 第 8 步：做最小消融矩阵

顺序：

```text
1280 输入
YOLO11m
P2
group-aware 弱类采样
训练裁剪
SAHI
D-FINE-M
```

每次只改变一个主要变量。

### 第 9 步：选择最终候选

- 两种子一致；
- 弱类和 AP-small 有真实改善；
- 正常图误报率可接受；
- group bootstrap 支持提升；
- 归档权重、配置和哈希。

### 第 10 步：一次性评估锁定测试集

测试完成后才允许查看 test 分类别指标和失败案例。

<a id="section-16"></a>
## 16. 产物目录与完成检查表

### 16.1 推荐目录

```text
configs/
├── dataset_labels_eval.yaml
└── primary_labels_grouped_v1_eval.yaml

datasets/
└── primary_labels_grouped_v1/
    ├── images/
    ├── labels/
    ├── data.yaml
    └── metadata/
        ├── dataset_fingerprint.json
        ├── split_manifest.csv
        ├── label_stats.csv
        ├── duplicate_resolution.csv
        └── split_leakage_audit.json

runs/
├── detect/
│   ├── imported/
│   │   └── yolo11s_unified_fine_img960_seed0_20260725/
│   └── primary_labels_grouped_v1/
├── audit/
│   └── primary_labels_grouped_v1/
├── eval/
│   ├── leaderboard_dataset_labels_history.csv
│   └── leaderboard_primary_labels_grouped_v1.csv
└── validate/
    └── dataset_labels__yolo11s__img960__val/
```

### 16.2 每个实验必须保存

- 数据指纹；
- Git 提交；
- 完整训练参数；
- 初始权重来源；
- 最终权重和哈希；
- 两个种子；
- 训练日志；
- 验证预测；
- 指标 JSON 和 per-class CSV；
- 错误清单；
- 是否通过门控的结论。

### 16.3 完成检查表

- [x] 确认模型架构为 YOLO11s
- [x] 确认权重类别顺序
- [x] 复现当前 split 的 mAP50≈0.87
- [x] 复现 mAP50-95≈0.59
- [x] 用第二套评估器交叉检查
- [x] 归档权重并校验 SHA-256
- [x] 计算当前数据内容指纹
- [x] 检查精确重复泄漏
- [x] 发现并量化近重复组泄漏
- [ ] 构建 primary-only group-aware 数据版本
- [ ] 完成新验证集盲审
- [ ] 从通用预训练权重训练两种子 YOLO11s
- [ ] 完成输入尺寸和 P2 消融
- [ ] 完成 group bootstrap
- [ ] 冻结最终候选
- [ ] 一次性评估新锁定测试集

<a id="section-17"></a>
## 17. 参考资料

### 17.1 当前项目证据

- `configs/dataset_labels_eval.yaml`
- `runs/eval/dataset_labels__yolo11s__img960__val_coco.json`
- `runs/eval/dataset_labels__yolo11s__img960__val_coco_per_class.csv`
- `runs/validate/dataset_labels__yolo11s__img960__val/`
- `runs/detect/imported/yolo11s_unified_fine_img960_seed0_20260725/checkpoint_manifest.json`

### 17.2 官方验证与训练说明

- Ultralytics Validation Mode: <https://docs.ultralytics.com/modes/val/>
- Ultralytics Train Mode: <https://docs.ultralytics.com/modes/train/>

### 17.3 小目标检测

- Feature Pyramid Networks for Object Detection: <https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html>
- QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection: <https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.pdf>
- SAHI: Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection: <https://arxiv.org/abs/2202.06934>

### 17.4 稀疏和缺失标注

- Object Detection as a Positive-Unlabeled Problem: <https://arxiv.org/abs/2002.04672>
- SparseDet: Improving Sparsely Annotated Object Detection with Pseudo-positive Mining: <https://openaccess.thecvf.com/content/ICCV2023/papers/Suri_SparseDet_Improving_Sparsely_Annotated_Object_Detection_with_Pseudo-positive_Mining_ICCV_2023_paper.pdf>

---

执行本方案时，最重要的纪律是区分“旧划分上可复现的 0.87”和“新无泄漏基准上的真实泛化性能”。只有在数据分组、标签规范和测试锁定都完成后，模型结构优化带来的提升才具有可信解释。
