# 绝缘子四分类缺陷检测完整优化执行方案

> [!IMPORTANT]
> 本文保留为早期 `credible_fine_v2/v3` 路线的历史方案。2026-07-31 已根据用户指定的单一数据集 `Dataset/labels`、`best.pt` 本地复现结果和近重复划分审计重新制定当前方案。后续执行请以 [`optimization_execution_plan_dataset_labels.md`](optimization_execution_plan_dataset_labels.md) 为准。

本文给出当前 `insulator-defect-detection` 项目从数据审核、可信基准构建、训练监督修正、模型实验到锁定测试集评估的完整执行流程。

方案针对当前最佳验证集结果 `mAP50 = 0.8375`，目标是在不发生数据泄漏、不修改评价口径制造提升的前提下，逐步逼近并验证严格四分类 `mAP50 > 0.95`。

> 当前核心判断：主要瓶颈不是模型容量，而是密集小缺陷、不完整或不一致标注、相关验证样本重复计权，以及弱类训练中的前景—背景监督冲突。继续单纯增加模型容量或集成分支的优先级较低。

<a id="toc"></a>
## 目录

- [1. 当前状态与瓶颈结论](#section-1)
- [2. 总体执行路线与不可违反的规则](#section-2)
- [3. 阶段 0：冻结当前 v2 基准](#section-3)
- [4. 阶段 1：制定统一标注规范](#section-4)
- [5. 阶段 2：盲审验证集](#section-5)
- [6. 阶段 3：审核训练集漏标](#section-6)
- [7. 阶段 4：构建 credible_fine_v3](#section-7)
- [8. 阶段 5：改进评估器](#section-8)
- [9. 阶段 6：在 v3 上重新建立单模型基线](#section-9)
- [10. 阶段 7：缺失标注鲁棒训练](#section-10)
- [11. 阶段 8：补充真实弱类数据](#section-11)
- [12. 阶段 9：逐片检测与分类](#section-12)
- [13. 阶段 10：实验矩阵、晋级与停止条件](#section-13)
- [14. 阶段 11：锁定测试集评估](#section-14)
- [15. 推荐的近期执行顺序](#section-15)
- [16. 产物目录约定](#section-16)
- [17. 风险清单与规避措施](#section-17)
- [18. 参考资料](#section-18)

<a id="section-1"></a>
## 1. 当前状态与瓶颈结论

### 1.1 当前权威基准

数据集：`datasets/credible_fine_v2`

数据指纹：

```text
e45e9272436fcaea448f4e4edc7c252989f5be3db8b1a8400a5ba97d034e97e1
```

当前最佳结果来自 YOLO 全图、P2、缺陷专家、ROI、D-FINE-M 和 D-FINE-L 的融合：

| 指标 | 当前值 |
| --- | ---: |
| mAP50 | 0.8375 |
| mAP50-95 | 0.5727 |
| mAP75 | 0.6135 |
| AP-small | 0.4417 |
| 正常图误报率 | 3.77% |

分类别 AP50：

| 类别 | AP50 |
| --- | ---: |
| `insulator_string` | 0.9651 |
| `broken_shell` | 0.7726 |
| `flashover_pollution` | 0.6651 |
| `missing_disc_drop` | 0.9474 |

权威结果文件：

- `runs/eval/credible_v2_dfine_ml_yolo_full_ensemble_val_bootstrap.json`
- `runs/eval/credible_v2_dfine_ml_yolo_full_ensemble_val_bootstrap_per_class.csv`
- `runs/eval/credible_v2_dfine_ml_yolo_full_ensemble_val_bootstrap_errors.csv`
- `runs/eval/leaderboard_credible_v2.csv`

### 1.2 主要瓶颈

| 优先级 | 瓶颈 | 本项目证据 | 判断 |
| --- | --- | --- | --- |
| P0 | `flashover_pollution` 标注与小目标问题 | AP50 为 0.6651，AP75 为 0.2782；验证集中约 69.6% 属于小目标，约 30.4% 的短边不足 16 px | 最大瓶颈 |
| P0 | `broken_shell` 密集实例漏标和边界不一致 | AP50 为 0.7726；验证集该类约 13.7% 的短边不足 16 px | 第二瓶颈 |
| P0 | 前景被评价或训练为背景 | 当前错误中有 680 个误报、87 个漏检、93 个定位错误，仅 2 个跨类混淆 | 不是单纯分类混淆 |
| P1 | 验证样本不完全独立 | Supervisely 验证集 184 张图来自 48 个原始图像组，平均每组 3.83 个变体 | 图片级 Bootstrap 可能低估不确定性 |
| P1 | 数据元信息漂移 | `label_stats.csv` 没有反映 v2 新增的 44 个训练框 | 影响审计和复现可信度 |
| P2 | 模型收益递减 | D-FINE-M 为 0.8257，六分支集成为 0.8375；M+L 仅增加约 0.0027 | 模型容量不是当前首要限制 |

### 1.3 到达 0.95 所需的弱类水平

如果两个强类保持当前水平，要使四分类平均 AP50 达到 0.95，则 `broken_shell` 和 `flashover_pollution` 的 AP50 平均需要达到约 0.9438。

这意味着：

- `broken_shell` 需要大幅提高；
- `flashover_pollution` 需要更大幅度提高；
- 常规模型改造带来的小幅收益不足以填补当前差距；
- 必须优先修正数据完整性、监督方式和评价独立性。

<a id="section-2"></a>
## 2. 总体执行路线与不可违反的规则

### 2.1 总体路线

```text
冻结 v2 基准
  → 建立统一标注规范
  → 盲审验证集并创建 v3
  → 审核训练集漏标
  → 修正分组评估与 Bootstrap
  → 在 v3 上重训 D-FINE-M 基线
  → 缺失标注鲁棒训练
  → 补充真实弱类数据
  → 逐片检测与分类
  → 最多两个候选进入锁定测试集
```

### 2.2 不可违反的规则

1. 验证集图片、裁剪图、增强图和人工修正结果不得加入训练集。
2. 验证集审核人员不得查看模型预测、模型置信度或错误排名。
3. 训练集可以使用模型产生审核候选，但候选必须人工确认后才能成为标签。
4. `credible_fine_v2` 永久保留，任何标签修改均创建新的数据版本和指纹。
5. v2 与 v3 的结果不得放入同一个排行榜直接比较。
6. 锁定测试集不得用于阈值选择、融合参数搜索、模型选择或错误驱动开发。
7. 所有训练和评估记录完整权重路径、数据指纹、Git 提交、随机种子和运行参数。
8. 未人工确认的模型框不得直接修改验证集或测试集标签。

<a id="section-3"></a>
## 3. 阶段 0：冻结当前 v2 基准

### 3.1 验证数据集

```powershell
python .\scripts\validate_credible_dataset.py `
  .\datasets\credible_fine_v2

Get-Content `
  .\datasets\credible_fine_v2\metadata\dataset_fingerprint.json
```

必须确认输出指纹为：

```text
e45e9272436fcaea448f4e4edc7c252989f5be3db8b1a8400a5ba97d034e97e1
```

### 3.2 冻结权威结果

检查并保留：

```text
runs/eval/credible_v2_dfine_ml_yolo_full_ensemble_val_bootstrap.json
runs/eval/credible_v2_dfine_ml_yolo_full_ensemble_val_bootstrap_per_class.csv
runs/eval/credible_v2_dfine_ml_yolo_full_ensemble_val_bootstrap_errors.csv
runs/eval/leaderboard_credible_v2.csv
```

### 3.3 建立基准清单

建议新增：

```text
runs/baseline/credible_v2_freeze_manifest.json
```

至少记录：

- 数据指纹；
- 验证集图片数；
- 各类别对象数；
- 权重绝对路径；
- 权重 SHA-256；
- D-FINE 提交；
- 当前 Git 提交；
- 评估参数；
- 当前指标；
- 生成时间和时区。

### 3.4 阶段验收

- v2 验证脚本通过；
- 指纹正确；
- 权威结果文件存在；
- 权重文件可读取；
- 当前结果可以使用固定命令复现；
- 后续实验不再修改 v2。

<a id="section-4"></a>
## 4. 阶段 1：制定统一标注规范

在人工复核前，必须先冻结类别定义和边界框规则，否则不同审核人员会继续产生不一致标签。

建议新增：

```text
annotations/credible_fine_v3_annotation_guideline.md
```

### 4.1 `broken_shell` 定义

应标注：

- 绝缘子片主体发生真实缺口、崩边或破碎；
- 缺陷已经改变绝缘子片的物理外轮廓；
- 图像证据足以排除遮挡、阴影、金具和模糊。

不应标注：

- 单纯反光；
- 正常玻璃片边缘；
- 被金具遮挡但无法确认破损；
- 分辨率不足以判断的实例。

### 4.2 `flashover_pollution` 定义

应标注：

- 表面存在可辨认的闪络、烧蚀、连续污秽或异常痕迹；
- 痕迹能够与正常材质颜色、阴影和反光可靠区分。

不应标注：

- 茶色或褐色玻璃本身；
- 单纯阴影；
- 阳光高光；
- 轻微颜色变化但证据不足。

### 4.3 `missing_disc_drop` 定义

应标注：

- 绝缘子串中存在明确缺片、掉串或结构性间隙；
- 缺失位置和范围能够确定。

不应标注：

- 透视导致的视觉间距变化；
- 遮挡造成的疑似缺片；
- 仅出现局部破损但绝缘子片仍存在的情况。

### 4.4 `ambiguous/ignore` 定义

以下实例不得被强行标成正例或负例：

- 目标小到无法可靠判断；
- 严重运动模糊；
- 大面积遮挡；
- 反光与闪络无法区分；
- 两名审核人员无法达成一致。

### 4.5 审核字段

审核 CSV 建议字段：

```text
image
group_id
reviewer
existing_box_id
original_class
decision
final_class
final_x1
final_y1
final_x2
final_y2
missing_instance
ambiguity_reason
notes
reviewed_at
```

`decision` 仅允许：

```text
confirmed
corrected
missing
wrong_class
rejected
ambiguous
```

### 4.6 阶段验收

- 类别定义无歧义；
- 正例、负例和 ambiguous 均有示例；
- 审核字段和合法状态固定；
- 两名审核人员使用同一版本规范；
- 审核开始后不得根据模型结果临时修改规则。

<a id="section-5"></a>
## 5. 阶段 2：盲审验证集

验证集审核的目的不是让模型记住验证样本，而是确认评价答案是否正确、完整和一致。

### 5.1 审核范围

验证集当前包含：

- 362 张图片；
- 217 个独立图像组；
- 190 个 `broken_shell` 标签；
- 237 个 `flashover_pollution` 标签；
- 222 张当前包含弱类标签的图片；
- 80 个当前包含弱类标签的图像组。

为了发现漏标，审核范围必须覆盖全部 362 张验证图，而不能只审核现有的 427 个弱类框。

审核顺序：

1. 先审核当前含弱类标签的 80 个独立组；
2. 再审核其余 137 个组，寻找未标注弱类实例；
3. 将 Supervisely 的原图及 `d/h/v` 变体放入同一次审核；
4. 逐个绝缘子串、逐片检查；
5. 对每个现有框判断类别、边界和可见性；
6. 对整幅图检查是否存在未标注实例。

### 5.2 盲审界面限制

审核界面只能显示：

- 原始图像；
- 原始人工标签；
- 冻结版类别定义。

不得显示：

- 模型预测框；
- 模型置信度；
- 错误严重度；
- 文件是否属于高错误样本；
- 当前模型预测类别；
- 集成模型是否达成共识。

### 5.3 双人复核流程

```text
审核者 A 独立审核
  → 审核者 B 独立审核
  → 自动比较结果
  → 分歧项交由第三人裁决
  → 形成冻结版 adjudicated.csv
```

建议新增脚本：

```text
scripts/build_blind_audit.py
scripts/merge_annotation_reviews.py
```

拟定命令：

```powershell
python .\scripts\build_blind_audit.py `
  --dataset .\datasets\credible_fine_v2 `
  --split val `
  --classes 1 2 `
  --include-all-images `
  --group-by family `
  --output .\runs\audit\credible_v3_val_blind
```

建议产物：

```text
runs/audit/credible_v3_val_blind/
├── reviewer_a.csv
├── reviewer_b.csv
├── adjudicated.csv
├── disagreement.csv
├── images/
└── index.html
```

### 5.4 阶段报告

必须报告：

- 原有框确认率；
- 错类数量；
- 边界修正数量；
- 新增漏标数量；
- rejected 数量；
- ambiguous 数量；
- 审核者间一致率；
- 分来源修改数量；
- 分类别修改数量；
- 分独立图像组修改数量。

### 5.5 阶段验收

- 全部 362 张验证图均有审核状态；
- 两名审核人员独立完成；
- 分歧全部裁决；
- 审核期间未显示模型预测；
- `adjudicated.csv` 冻结并计算 SHA-256；
- 此阶段不训练模型。

<a id="section-6"></a>
## 6. 阶段 3：审核训练集漏标

训练集可以使用模型辅助寻找候选，因为训练集属于允许模型学习的范围，但模型建议不能自动成为标签。

### 6.1 当前审核规模

| 来源 | 弱类图片 | 弱类独立组 |
| --- | ---: | ---: |
| Dataset | 534 | 382 |
| Supervisely | 899 | 293 |
| 合计 | 1433 | 675 |

### 6.2 使用同类别映射模型挖掘候选

现有脚本可以直接使用：

```powershell
python .\scripts\mine_missing_labels.py `
  --data .\datasets\credible_fine_v2\data.yaml `
  --weights `
    .\runs\detect\runs\credible_v1_yolo11s_960_aug_moderate_40\weights\best_map50.pt `
    .\runs\detect\runs\credible_v1_yolo11m_p2_960_b6\weights\best_map50.pt `
  --imgsz 960 `
  --batch 4 `
  --device 0 `
  --conf 0.45 `
  --iou 0.70 `
  --agreement-iou 0.55 `
  --max-gt-iou 0.20 `
  --max-candidates 1500 `
  --output .\runs\audit\credible_v3_train_candidates
```

不要直接加入三类缺陷专家权重，因为其类别编号相对四分类模型存在偏移，除非脚本已显式支持 class offset。

### 6.3 人工审核优先级

1. 两个模型共同检出的未匹配框；
2. `flashover_pollution` 候选；
3. `broken_shell` 候选；
4. 每个弱类独立图像组至少抽查一个原始样本；
5. 高密度绝缘子串；
6. 小于 16 px 的目标；
7. 模型认为是缺陷、但审核确认正常的硬负样本。

### 6.4 审核状态

训练候选允许显示模型框，但最终状态必须人工填写：

```text
approved
rejected
ambiguous
wrong_class
```

### 6.5 不再采用普通硬伪标签微调

当前 44 个人工确认框的低学习率微调仅达到 0.7716，说明以下做法不足：

- 只增加少量框；
- 继续把其余疑似前景当成背景；
- 从旧偏置检查点短程微调。

新一轮要求：

- 人工确认后才能加入；
- ambiguous 区域进入 ignore；
- 从通用预训练权重重新训练可信基线；
- 不把模型共识等同于真值。

### 6.6 阶段验收

- 候选包完整生成；
- 每个候选有人工状态；
- approved 与 rejected 均保留审核痕迹；
- 审核 CSV 计算 SHA-256；
- 训练集审核结果未用于修改 v2；
- 所有变更准备进入新版本 v3。

<a id="section-7"></a>
## 7. 阶段 4：构建 credible_fine_v3

现有 `apply_pseudo_labels.py` 适合训练集新增框，但不足以同时处理验证集修正、ignore 区域、实际标签统计和完整指纹，因此建议新增独立构建脚本。

建议新增：

```text
scripts/apply_reviewed_annotations.py
```

### 7.1 构建规则

- 图像划分与 v2 完全一致；
- 只应用已经裁决或人工批准的标签；
- 图片使用硬链接，避免重复占用空间；
- 训练、验证、测试成员不交换；
- v2 不做任何修改；
- ambiguous 区域保存为独立 sidecar；
- 指纹基于实际图片、标签和审核文件哈希；
- 所有派生变体继续保持在同一 split；
- 不把验证集修正同步到训练集。

### 7.2 拟定构建命令

```powershell
python .\scripts\apply_reviewed_annotations.py `
  --source .\datasets\credible_fine_v2 `
  --train-review .\runs\audit\credible_v3_train_candidates\candidates_reviewed.csv `
  --val-review .\runs\audit\credible_v3_val_blind\adjudicated.csv `
  --output .\datasets\credible_fine_v3 `
  --preserve-splits `
  --require-adjudicated
```

### 7.3 必须更新的元数据

```text
metadata/dataset_fingerprint.json
metadata/label_stats.csv
metadata/split_manifest.csv
metadata/review_manifest.json
metadata/ignore_regions.csv
metadata/source_distribution.csv
metadata/change_log.csv
```

### 7.4 指纹要求

指纹输入至少包含：

- 每张图片的稳定相对路径和内容哈希；
- 每个标签文件的规范化内容；
- split 和 group_id；
- 训练、验证审核文件的 SHA-256；
- ignore 区域内容；
- 类别顺序；
- 构建脚本版本。

### 7.5 验证

```powershell
python .\scripts\validate_credible_dataset.py `
  .\datasets\credible_fine_v3
```

额外检查：

- 标签文件数与图片数一致；
- 无跨划分 group；
- 类别只包含 0 到 3；
- 边界框位于图像范围内；
- `label_stats.csv` 与实际标签逐行统计一致；
- v3 指纹与 v2 不同；
- v2 目录内容未变化。

### 7.6 阶段验收

- v3 构建成功；
- 所有校验通过；
- 新指纹已记录；
- 变更日志可追溯到具体审核决定；
- v2 与 v3 排行榜物理隔离。

<a id="section-8"></a>
## 8. 阶段 5：改进评估器

当前评估器按图片进行 Bootstrap，但同一原始场景存在多个派生变体，应该增加按图像组重采样的评价方式。

建议扩展：

```text
scripts/evaluate_detector.py
```

新增参数：

```text
--bootstrap-unit image|family
--report-by-source
--report-by-size
--save-predictions
--ignore-regions
```

### 8.1 Family Bootstrap 算法

1. 从 `split_manifest.csv` 读取每张图的 `group_id`；
2. 随机有放回抽取 group；
3. 一个 group 被抽中时，其全部变体一起进入样本；
4. 为重复抽中的 group 复制新的 COCO image id；
5. 重新计算 COCO AP；
6. 重复 1000 次；
7. 输出 percentile 95% CI；
8. 同时保留 image-level CI，便于与旧结果对照。

### 8.2 新报告要求

每个实验至少输出：

- image-level AP；
- family-level AP；
- image-bootstrap 95% CI；
- family-bootstrap 95% CI；
- 分类别 AP50 和 AP75；
- 按来源 AP；
- 按尺寸 AP；
- 正常图误报率；
- 每个独立图像组的 FP 和 FN；
- ambiguous/ignore 命中数；
- 数据指纹和完整权重路径。

### 8.3 保存预测

所有单模型和集成评估应支持保存统一格式预测：

```text
runs/eval/<run_name>_predictions.json
```

目的：

- 不重新推理即可切换 image/family Bootstrap；
- 便于做来源、尺寸和错误分析；
- 避免多次推理造成配置漂移；
- 便于配对比较两个模型。

### 8.4 单元测试

在 `tests/test_pipeline.py` 增加：

- 同组变体必须一起进入 Family Bootstrap；
- ignore 区域内预测不计 FP；
- v2 与 v3 指纹不能写入同一个排行榜；
- 保存预测后重复评估结果一致；
- 来源和尺寸分层统计总数与总体一致；
- 随机种子固定时 Bootstrap 结果可复现。

### 8.5 阶段验收

- 新旧 image-level 指标一致；
- family-level 指标可生成；
- 1000 次 Bootstrap 可复现；
- ignore 逻辑通过测试；
- 保存预测与在线推理评估一致。

<a id="section-9"></a>
## 9. 阶段 6：在 v3 上重新建立单模型基线

优先训练 D-FINE-M，因为它是当前最强单模型。此阶段不加入新结构，用于分离“数据修正收益”和“模型方法收益”。

### 9.1 生成种子 20260708 配置

```powershell
python .\scripts\prepare_dfine.py `
  --dataset .\datasets\credible_fine_v3 `
  --output .\datasets\credible_fine_v3_coco `
  --dfine-root .\runs\third_party\D-FINE `
  --project-root . `
  --imgsz 960 `
  --epochs 15 `
  --batch 4 `
  --model-size m `
  --pretrained-variant coco `
  --multi-scale `
  --config-name dfine_hgnetv2_m_insulator_v3_seed20260708.yml `
  --run-name credible_fine_v3_dfine_m_960_seed20260708 `
  --overwrite
```

### 9.2 训练种子 20260708

```powershell
Push-Location .\runs\third_party\D-FINE

python .\train.py `
  -c .\configs\dfine\custom\dfine_hgnetv2_m_insulator_v3_seed20260708.yml `
  --use-amp `
  --seed=20260708 `
  -t .\weights\dfine_m_coco.pth

Pop-Location
```

### 9.3 训练第二个随机种子

重新生成独立的 `config-name` 和 `run-name`，使用：

```text
seed=20260709
```

不得让两个种子写入同一输出目录。

### 9.4 导出预测

```powershell
python .\scripts\predict_dfine.py `
  --dfine-root .\runs\third_party\D-FINE `
  --config .\runs\third_party\D-FINE\configs\dfine\custom\dfine_hgnetv2_m_insulator_v3_seed20260708.yml `
  --weights .\runs\dfine\credible_fine_v3_dfine_m_960_seed20260708\best_map50.pth `
  --data .\datasets\credible_fine_v3\data.yaml `
  --split val `
  --imgsz 960 `
  --batch 4 `
  --conf 0.001 `
  --device 0 `
  --output .\runs\eval\credible_v3_dfine_m_seed20260708_val_predictions.json
```

实际权重名称以补丁保存结果为准。如果只存在 `best_stg2.pth`，后续命令必须使用真实文件名。

### 9.5 评估

```powershell
python .\scripts\evaluate_detector.py `
  --data .\datasets\credible_fine_v3\data.yaml `
  --split val `
  --mode external `
  --external-predictions .\runs\eval\credible_v3_dfine_m_seed20260708_val_predictions.json `
  --conf 0.001 `
  --operating-conf 0.25 `
  --iou 0.70 `
  --bootstrap 1000 `
  --bootstrap-unit family `
  --seed 20260708 `
  --output .\runs\eval\credible_v3_dfine_m_seed20260708_val.json `
  --leaderboard .\runs\eval\leaderboard_credible_v3.csv
```

### 9.6 比较原则

不能直接拿 v3 分数与 v2 排行榜比较模型收益。正确对照是：

1. 在 v3 上重新评估原有模型；
2. 在 v3 上训练新基线；
3. 两者使用完全相同的 v3、评估器和 Bootstrap；
4. 再计算差值。

### 9.7 晋级条件

相对 v3 上的可比基线，候选必须满足：

- overall AP50 提升至少 0.02；
- `broken_shell` AP50 提升至少 0.03；
- `flashover_pollution` AP50 提升至少 0.03；
- 强类下降不超过 0.01；
- 正常图误报率上升不超过 1 个百分点；
- family-level 指标同步提升；
- 两个种子的提升方向一致。

<a id="section-10"></a>
## 10. 阶段 7：缺失标注鲁棒训练

如果 v3 清洗后弱类仍明显偏低，应解决“未标注区域被训练为背景”的监督问题。

### 10.1 先做 Ignore 实验

新增：

```text
metadata/ignore_regions.csv
```

格式示例：

```text
image,class_id,x1,y1,x2,y2,reason
```

训练规则：

- ignore 区域不作为正例；
- 区域内 anchor、proposal 或 query 不参与背景分类损失；
- 不产生边界框回归损失；
- 验证时 ignore 区域内的预测不计 FP；
- ignore 区域不能被当作已确认真值。

### 10.2 Ignore 消融实验

| 实验 | 标签 | 负样本处理 |
| --- | --- | --- |
| E1 | v3 | 标准背景损失 |
| E2 | v3 | ambiguous 区域忽略 |
| E3 | v3 | ambiguous 加高可信未标注区域忽略 |

只有 E2 或 E3 通过统一晋级门槛，才保留 ignore 方案。

### 10.3 PU 学习分支

标准检测损失默认“没有标注框就是背景”，PU 学习将其改成“没有标注框的区域可能是正例，也可能是背景”。

建议优先采用两阶段结构：

```text
FPN 骨干
  → PU-RPN
  → ROI Head
  → Focal Loss
```

推荐对比：

- 标准 Faster R-CNN-FPN-P2；
- 仅增加 Focal Loss；
- PU-RPN；
- PU-RPN 加 Focal Loss；
- SparseDet 风格伪正例挖掘。

### 10.4 受控变量

所有 PU 消融必须保持一致：

- 数据版本；
- 预训练权重；
- 输入尺寸；
- batch；
- epoch；
- 数据增强；
- 优化器；
- 随机种子；
- 评估器；
- ignore 区域。

只改变缺失标注处理方式。

### 10.5 阶段验收

- 基线和 PU 分支公平对照；
- 两个随机种子完成；
- 弱类和 family-level AP 同步提升；
- 正常图误报率未失控；
- 未人工确认框没有被写入验证标签；
- 所有损失修改具有单元测试。

<a id="section-11"></a>
## 11. 阶段 8：补充真实弱类数据

如果完成 v3 和 PU 训练后 overall AP50 仍低于约 0.88，应暂停通用架构搜索，优先补充真实、完整、独立的弱类数据。

### 11.1 优先申请官方 IDID

官方入口：

```text
https://ieee-dataport.org/competitions/epri-insulator-defect-image-dataset
```

官方数据包含：

- broken shell；
- flashed shell；
- good shell；
- insulator string。

### 11.2 导入要求

1. 核验许可和下载来源；
2. 不使用未经授权镜像；
3. 记录压缩包 SHA-256；
4. 与现有数据做完全重复检测；
5. 做保守近重复检测；
6. 按原始场景建立 group；
7. 人工抽查漏标；
8. 只将新增独立组加入训练集；
9. 保持 v3 验证集不变；
10. 创建新的训练数据版本和指纹。

### 11.3 数据采集重点

优先覆盖：

- 茶色玻璃硬负样本；
- 强反光；
- 阴影；
- 金具遮挡；
- 低对比度污秽；
- 轻度和重度闪络；
- 小缺口和大面积破损；
- 不同角度、距离和材质；
- 密集绝缘子串；
- 每片明确标为 good、broken 或 flashover。

### 11.4 分批学习曲线

新增数据按独立图像组分批加入，而不是一次全部混入。每批完成：

1. 构建新数据版本；
2. 固定 D-FINE-M 配置重新训练；
3. 记录弱类 AP；
4. 记录 family-level AP；
5. 记录正常图误报率；
6. 绘制独立图像组数量与 AP 的学习曲线；
7. 判断收益是否饱和。

派生增强图不得计作新的独立图像组。

<a id="section-12"></a>
## 12. 阶段 9：逐片检测与分类

这一步必须排在数据清洗和 good shell 标注之后。当前四分类数据没有完整的 good shell 实例标签，不能直接训练可靠的逐片分类器。

### 12.1 新结构

```text
D-FINE-M 检测绝缘子串
  → 估计字符串主轴
  → 旋转校正和紧致裁剪
  → 检测或枚举每个绝缘子片
  → good / broken / flashover 分类
  → 序列一致性约束
  → 映射回原图
```

### 12.2 与现有 ROI 分支的区别

现有 ROI 模型：

- 在裁剪后的字符串中继续自由检测缺陷；
- 未显式枚举每个物理绝缘子片；
- 仍容易产生重复框和背景误报。

新方案：

- 先枚举物理绝缘子片；
- 每片最多输出一个状态；
- 显式使用 good shell 作为负类；
- 利用相邻片外观、方向和间距；
- 将自由目标检测转化为受约束的逐片状态识别。

### 12.3 方向校正流程

1. 使用绝缘子串框或分割掩码估计主轴；
2. 将字符串旋转到水平或垂直标准方向；
3. 生成紧致 ROI，并保留少量上下文；
4. 在校正后的 ROI 中检测单片；
5. 对单片进行状态分类；
6. 将结果反变换到原图；
7. 使用原图四分类 COCO AP 做最终验收。

### 12.4 高分辨率策略

不要优先再次进行全图 1280 或 1536 训练。建议：

- 全图 960 定位字符串；
- 字符串 ROI 使用 1280；
- 仅在候选 ROI 计算高分辨率特征；
- 单独统计短边小于 16 px 和小于 32 px 的实例；
- 比较旋转校正前后 AP；
- 比较自由检测与逐片分类。

### 12.5 阶段验收

- 单片检测召回率达到可用水平；
- good shell 硬负样本误报受控；
- 弱类 AP 达到晋级门槛；
- 原图端到端 AP 提升，而不只是裁剪集内部指标提升；
- 方向校正失败时有回退路径；
- 处理时间满足应用要求。

<a id="section-13"></a>
## 13. 阶段 10：实验矩阵、晋级与停止条件

### 13.1 实验矩阵

| 编号 | 实验 | 目的 |
| --- | --- | --- |
| E0 | 原模型在 v3 上重新评估 | 分离标签修正与训练收益 |
| E1 | v3 D-FINE-M，两种子 | 建立新基线 |
| E2 | v3 加 ignore regions | 验证背景监督冲突 |
| E3 | PU-RPN 或 SparseDet | 验证缺失标注鲁棒学习 |
| E4 | 新增真实数据加 D-FINE-M | 验证数据规模和多样性 |
| E5 | 逐片检测与分类 | 验证结构性改进 |
| E6 | 最佳两个模型融合 | 形成最终候选 |

### 13.2 统一晋级门槛

候选相对同数据、同评估器的基线必须满足：

- overall AP50 提升至少 0.02；
- `broken_shell` AP50 提升至少 0.03；
- `flashover_pollution` AP50 提升至少 0.03；
- `insulator_string` 和 `missing_disc_drop` 下降不超过 0.01；
- 正常图误报率上升不超过 1 个百分点；
- family-level AP 同步提升；
- family-bootstrap CI 下界提高；
- 两个随机种子的提升方向一致。

### 13.3 结果解释

- 如果 E0 相对 v2 明显提高：旧评价主要受标签问题限制。
- 如果 E2 或 E3 明显提高：训练集漏标和背景监督冲突是主要问题。
- 如果清洗后 AP-small 仍很低：小目标分辨率仍是主要问题。
- 如果逐片分类提高而自由检测不提高：问题主要在检测范式。
- 如果新增独立真实数据持续提高：继续数据驱动优化。
- 如果所有有效实验连续三次提升小于 0.01：停止架构搜索。

### 13.4 强制停止条件

出现任一情况即停止对应路线：

- 强类下降超过 0.03；
- 正常图误报率超过 5%；
- 只在一个随机种子上提高；
- image-level 提高但 family-level 不提高；
- 收益主要来自一个重复变体图像组；
- 通过修改固定阈值而非改善 PR 曲线获得表面提升；
- 数据指纹不一致；
- 评估结果无法从保存预测复现；
- 连续三个有效实验提升均小于 0.01。

### 13.5 暂不优先的方向

在完成数据与监督修正前，不优先：

- D-FINE-XL 或继续增加模型容量；
- 更多注意力模块；
- 更多通用 BiFPN 变体；
- 全图 SAHI；
- 全图高分辨率穷举；
- 继续增加集成分支；
- 未人工确认的硬伪标签；
- 在验证集上搜索特定文件规则；
- 通过工作点阈值制造达标结果。

<a id="section-14"></a>
## 14. 阶段 11：锁定测试集评估

只有在验证集上选出最多两个候选后，才能使用锁定测试集。

### 14.1 测试前冻结内容

- Git 提交；
- 数据指纹；
- 权重 SHA-256；
- 推理脚本版本；
- 输入尺寸；
- 置信度保留阈值；
- NMS 或 WBF 参数；
- 融合权重；
- 类别顺序；
- 随机种子；
- 评估器版本；
- ignore 区域版本；
- 候选模型数量。

### 14.2 测试规则

- 不查看中间测试结果后重新调参；
- 不重新选择工作点阈值；
- 不增加第三个候选；
- 不根据测试错误修改模型；
- 不根据测试图片补充训练数据；
- 同时报告 image-level 和 family-level 指标；
- 使用 1000 次 Family Bootstrap；
- 保留完整预测和错误清单。

### 14.3 最终报告

至少报告：

- 数据版本和指纹；
- 候选模型定义；
- 权重哈希；
- overall AP50、AP75、AP50-95；
- 各类别 AP50 和 AP75；
- AP-small、AP-medium、AP-large；
- 正常图误报率；
- image-bootstrap CI；
- family-bootstrap CI；
- 分来源结果；
- 推理时间和硬件；
- 失败图像组；
- 已知限制。

### 14.4 达标判定

最低目标：

```text
锁定测试集四分类 mAP50 > 0.95
```

更严格、也更可信的目标：

```text
锁定测试集 family-bootstrap 95% CI 下界 > 0.95
```

<a id="section-15"></a>
## 15. 推荐的近期执行顺序

### 15.1 第一批：不消耗大规模 GPU

1. 冻结 v2 基准和权重哈希；
2. 编写 v3 标注规范；
3. 实现盲审包生成脚本；
4. 实现双人审核合并脚本；
5. 对 362 张验证图进行双人盲审；
6. 扩展评估器支持 Family Bootstrap；
7. 修复元数据统计验证。

### 15.2 第二批：训练集数据治理

1. 使用基线和 P2 模型挖掘训练候选；
2. 人工审核候选；
3. 抽查 675 个弱类独立组；
4. 建立硬负样本集合；
5. 形成训练审核裁决文件；
6. 构建并验证 `credible_fine_v3`。

### 15.3 第三批：可信基线

1. 在 v3 上重新评估旧 D-FINE-M；
2. 从 COCO 预训练权重训练 v3 D-FINE-M；
3. 完成两个随机种子；
4. 保存统一预测；
5. 生成 image 和 family 两套指标；
6. 判断数据修正是否已带来有效提升。

### 15.4 第四批：方法优化

1. Ignore 区域消融；
2. PU-RPN 或 SparseDet 分支；
3. 申请并导入官方 IDID；
4. 构建 good shell 逐片数据；
5. 训练逐片检测与分类结构；
6. 最多保留两个最终候选。

### 15.5 在上述步骤完成前暂停

- D-FINE-L 或 XL 扩容；
- 新注意力模块；
- 全图 SAHI；
- 新增更多集成模型；
- 未审核伪标签；
- 测试集评估。

<a id="section-16"></a>
## 16. 产物目录约定

建议统一使用：

```text
annotations/
├── credible_fine_v3_annotation_guideline.md
├── credible_fine_v3_val_adjudicated.csv
└── credible_fine_v3_train_reviewed.csv

datasets/
├── credible_fine_v2/
├── credible_fine_v3/
└── credible_fine_v3_coco/

runs/
├── audit/
│   ├── credible_v3_val_blind/
│   └── credible_v3_train_candidates/
├── baseline/
│   └── credible_v2_freeze_manifest.json
├── dfine/
│   ├── credible_fine_v3_dfine_m_960_seed20260708/
│   └── credible_fine_v3_dfine_m_960_seed20260709/
└── eval/
    ├── leaderboard_credible_v2.csv
    ├── leaderboard_credible_v3.csv
    └── credible_v3_*.json
```

每个实验目录必须包含：

- 运行参数；
- 数据指纹；
- Git 提交；
- 权重路径和哈希；
- 训练日志；
- 验证预测；
- 指标报告；
- 错误清单；
- 是否通过门控的结论。

<a id="section-17"></a>
## 17. 风险清单与规避措施

| 风险 | 影响 | 规避措施 |
| --- | --- | --- |
| 验证审核人员看到模型预测 | 形成验证集定向修正 | 验证审核界面完全隐藏预测 |
| 把验证裁剪加入训练 | 直接数据泄漏 | 数据构建器校验文件哈希和 group |
| v2 与 v3 混入同一排行榜 | 产生不可比结论 | 按数据指纹物理隔离排行榜 |
| 派生变体被视为独立样本 | CI 偏窄、场景重复计权 | Family Bootstrap 和 unique-family 报告 |
| 模型候选直接成为标签 | 伪标签确认偏差 | 只接受人工 approved 状态 |
| ambiguous 被当作背景 | 错误负监督 | 使用 ignore 区域 |
| 元数据与实际标签不一致 | 复现失败 | 构建后重新逐行统计并校验 |
| 只运行一个随机种子 | 偶然提升 | 所有晋级实验至少两个种子 |
| 在测试集上调阈值 | 测试泄漏 | 测试前冻结全部参数 |
| 继续堆叠模型 | 计算成本高、收益递减 | 使用统一门控和停止条件 |
| 未授权数据来源 | 许可和复现风险 | 只使用官方或许可明确的数据 |
| 逐片分类缺少 good shell | 无法学习可靠负类 | 获取或人工建立完整单片标签后再启动 |

<a id="section-18"></a>
## 18. 参考资料

### 18.1 当前项目资料

- `README.md`
- `docs/optimization_protocol.md`
- `configs/experiments.yaml`
- `runs/eval/credible_v2_dfine_ml_yolo_full_ensemble_val_bootstrap.json`
- `datasets/credible_fine_v2/metadata/label_stats.csv`
- `datasets/credible_fine_v2/metadata/split_manifest.csv`

### 18.2 缺失标注与 PU 学习

- Electrical insulator defect detection with incomplete annotations and imbalanced samples: <https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/gtd2.13107>
- Object Detection as a Positive-Unlabeled Problem: <https://arxiv.org/abs/2002.04672>
- SparseDet: Improving Sparsely Annotated Object Detection with Pseudo-positive Mining: <https://openaccess.thecvf.com/content/ICCV2023/papers/Suri_SparseDet_Improving_Sparsely_Annotated_Object_Detection_with_Pseudo-positive_Mining_ICCV_2023_paper.pdf>
- SparseDet official repository: <https://github.com/saksham-s/SparseDet>

### 18.3 小目标和高分辨率检测

- QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection: <https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.pdf>
- Feature Pyramid Networks for Object Detection: <https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html>
- SAHI: Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection: <https://arxiv.org/abs/2202.06934>

### 18.4 绝缘子数据与两阶段方法

- IEEE DataPort IDID: <https://ieee-dataport.org/competitions/epri-insulator-defect-image-dataset>
- Supervisely Insulator-Defect Detection Dataset: <https://ecosystem.supervisely.com/projects/aerial-power-infrastructure-detection-train-dataset>
- XAI-guided Insulator Anomaly Detection for Imbalanced Datasets: <https://arxiv.org/abs/2409.16821>

### 18.5 统计评估

- Cluster Bootstrap for hierarchical or clustered data: <https://pmc.ncbi.nlm.nih.gov/articles/PMC7148287/>

---

执行本方案时，每完成一个阶段，都应先确认阶段验收条件全部满足，再进入下一阶段。任何无法复现、指纹不一致、验证集泄漏或只在单一重复图像组上出现的提升，都不得作为有效优化结果。
