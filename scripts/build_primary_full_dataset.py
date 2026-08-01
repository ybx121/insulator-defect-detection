#!/usr/bin/env python3
"""Build a complete YOLO dataset with the newly labelled export as authority.

The primary dataset is expected to use the standard ``images/{split}`` and
``labels/{split}`` YOLO layout.  Independently labelled Supervisely data is
converted into the same four-class schema.  Historical labels and generated
datasets are deliberately not read by this builder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from build_credible_dataset import (
    IMAGE_EXTS,
    Label,
    Sample,
    fingerprint_samples,
    group_samples,
    public_family,
    stratified_group_split,
)


SPLITS = ("train", "val", "test")
PRIMARY_NAMES = ["insulator string", "broken shell", "flavor", "diaochuan"]
CANONICAL_NAMES = [
    "insulator_string",
    "broken_shell",
    "flashover_pollution",
    "missing_disc_drop",
]
PUBLIC_CLASS_MAP = {"insulator": 0, "broken": 1, "pollution-flashover": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=Path("Dataset/labels"))
    parser.add_argument("--public-root", type=Path, default=Path("datasets/raw/supervisely"))
    parser.add_argument("--output", type=Path, default=Path("datasets/primary_full_v1"))
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--phash-distance", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_yolo(path: Path) -> tuple[Label, ...]:
    labels: list[Label] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 columns")
        try:
            class_id = int(parts[0])
            x, y, width, height = (float(value) for value in parts[1:])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: non-numeric label") from exc
        if class_id not in range(4):
            raise ValueError(f"{path}:{line_number}: class id {class_id} is outside 0..3")
        if any(not 0 <= value <= 1 for value in (x, y, width, height)):
            raise ValueError(f"{path}:{line_number}: coordinate is outside 0..1")
        if width <= 0 or height <= 0:
            raise ValueError(f"{path}:{line_number}: box size must be positive")
        labels.append(Label(class_id, x, y, width, height))
    return tuple(labels)


def collect_primary(root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for split in SPLITS:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing primary split directories for {split}: {root}")
        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in label_dir.glob("*.txt")}
        missing_labels = sorted(image_stems - label_stems)
        orphan_labels = sorted(label_stems - image_stems)
        if missing_labels or orphan_labels:
            raise ValueError(
                f"Primary {split} pairing error: missing_labels={missing_labels[:10]}, "
                f"orphan_labels={orphan_labels[:10]}"
            )
        for image in images:
            sample = Sample(
                source="Primary_New_Labels",
                source_image=image,
                stem=image.stem,
                family=f"primary:{image.stem}",
                labels=read_yolo(label_dir / f"{image.stem}.txt"),
            )
            sample.original_split = split
            sample.original_label = label_dir / f"{image.stem}.txt"
            samples.append(sample)
    return samples


def collect_public(root: Path) -> tuple[list[Sample], list[dict[str, object]]]:
    samples: list[Sample] = []
    exclusions: list[dict[str, object]] = []
    labelled_subsets = (("Train", "train"), ("Train", "val"))
    for archive_name, subset in labelled_subsets:
        base = root / archive_name / subset
        image_dir, annotation_dir = base / "img", base / "ann"
        if not image_dir.is_dir() or not annotation_dir.is_dir():
            raise FileNotFoundError(f"Missing Supervisely subset: {base}")
        for image in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS):
            annotation_path = annotation_dir / f"{image.name}.json"
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            width = int(annotation["size"]["width"])
            height = int(annotation["size"]["height"])
            labels: list[Label] = []
            for obj in annotation.get("objects", []):
                title = obj["classTitle"]
                if title not in PUBLIC_CLASS_MAP:
                    raise ValueError(f"Unknown Supervisely class {title!r} in {annotation_path}")
                (x1, y1), (x2, y2) = obj["points"]["exterior"]
                x1, x2 = sorted((max(0.0, float(x1)), min(float(width), float(x2))))
                y1, y2 = sorted((max(0.0, float(y1)), min(float(height), float(y2))))
                if x2 <= x1 or y2 <= y1:
                    continue
                labels.append(
                    Label(
                        PUBLIC_CLASS_MAP[title],
                        ((x1 + x2) / 2) / width,
                        ((y1 + y2) / 2) / height,
                        (x2 - x1) / width,
                        (y2 - y1) / height,
                    )
                )
            if not labels:
                exclusions.append(
                    {
                        "source": "Supervisely_Insulator_Defect",
                        "source_image": str(image.resolve()),
                        "reason": "no_usable_annotations",
                    }
                )
                continue
            sample = Sample(
                source="Aux_Supervisely",
                source_image=image,
                stem=f"aux_supervisely_{image.stem}",
                family=f"aux_supervisely:{public_family(image.stem)}",
                labels=tuple(labels),
            )
            sample.original_split = subset
            sample.original_label = annotation_path
            samples.append(sample)

    test_base = root / "Test" / "test"
    for image in sorted(path for path in (test_base / "img").iterdir() if path.suffix.lower() in IMAGE_EXTS):
        exclusions.append(
            {
                "source": "Supervisely_Insulator_Defect",
                "source_image": str(image.resolve()),
                "reason": "public_test_has_no_annotations",
            }
        )
    return samples, exclusions


def label_signature(labels: tuple[Label, ...]) -> str:
    rows = sorted(label.yolo() for label in labels)
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def deduplicate_exact(
    samples: list[Sample],
) -> tuple[list[Sample], list[dict[str, object]]]:
    by_sha1: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_sha1[sample.sha1].append(sample)

    kept: list[Sample] = []
    report: list[dict[str, object]] = []
    for sha1, group in sorted(by_sha1.items()):
        group.sort(
            key=lambda sample: (
                sample.source != "Primary_New_Labels",
                -len(sample.labels),
                str(sample.source_image).lower(),
            )
        )
        selected = group[0]
        kept.append(selected)
        signatures = {label_signature(sample.labels) for sample in group}
        for sample in group:
            report.append(
                {
                    "sha1": sha1,
                    "group_size": len(group),
                    "selected": sample is selected,
                    "label_conflict": len(signatures) > 1,
                    "source": sample.source,
                    "source_image": str(sample.source_image.resolve()),
                    "label_signature": label_signature(sample.labels),
                }
            )
    return kept, [row for row in report if int(row["group_size"]) > 1]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def prepare_output(output: Path, overwrite: bool, protected: tuple[Path, ...]) -> None:
    output = output.resolve()
    if output in {Path(output.anchor), *protected}:
        raise ValueError(f"Refusing unsafe output path: {output}")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
        shutil.rmtree(output)
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output / "metadata").mkdir(parents=True, exist_ok=True)


def write_dataset(
    output: Path,
    assignments: dict[str, list[Sample]],
    groups: dict[str, list[Sample]],
    exclusions: list[dict[str, object]],
    duplicate_report: list[dict[str, object]],
    seed: int,
    overwrite: bool,
    protected: tuple[Path, ...],
) -> None:
    prepare_output(output, overwrite, protected)
    manifest: list[dict[str, object]] = []
    class_stats: Counter[tuple[str, int]] = Counter()
    source_stats: Counter[tuple[str, str]] = Counter()
    link_stats: Counter[str] = Counter()
    used_stems: Counter[str] = Counter()

    for split in SPLITS:
        for sample in sorted(assignments[split], key=lambda item: (item.source, item.stem)):
            used_stems[sample.stem] += 1
            suffix = "" if used_stems[sample.stem] == 1 else f"_{used_stems[sample.stem]}"
            stem = f"{sample.stem}{suffix}"
            image_path = output / "images" / split / f"{stem}{sample.source_image.suffix.lower()}"
            label_path = output / "labels" / split / f"{stem}.txt"
            link_stats[link_or_copy(sample.source_image, image_path)] += 1
            label_text = "\n".join(label.yolo() for label in sample.labels) + "\n"
            label_path.write_text(label_text, encoding="utf-8")
            label_sha256 = hashlib.sha256(label_text.encode("utf-8")).hexdigest()
            for label in sample.labels:
                class_stats[(split, label.class_id)] += 1
            source_stats[(split, sample.source)] += 1
            manifest.append(
                {
                    "split": split,
                    "group_id": sample.group_id,
                    "image": str(image_path.relative_to(output)),
                    "label": str(label_path.relative_to(output)),
                    "source": sample.source,
                    "source_original_split": sample.original_split,
                    "source_image": str(sample.source_image.resolve()),
                    "source_label": str(sample.original_label.resolve()),
                    "sha1": sample.sha1,
                    "label_sha256": label_sha256,
                    "width": sample.width,
                    "height": sample.height,
                    "classes": "|".join(str(value) for value in sorted({label.class_id for label in sample.labels})),
                    "object_count": len(sample.labels),
                }
            )

    stats_rows = [
        {
            "split": split,
            "class_id": class_id,
            "class_name": PRIMARY_NAMES[class_id],
            "canonical_name": CANONICAL_NAMES[class_id],
            "objects": class_stats[(split, class_id)],
        }
        for split in SPLITS
        for class_id in range(4)
    ]
    write_csv(output / "metadata" / "split_manifest.csv", manifest)
    write_csv(output / "metadata" / "label_stats.csv", stats_rows)
    write_csv(output / "metadata" / "duplicate_resolution.csv", duplicate_report)
    write_csv(output / "metadata" / "exclusions.csv", exclusions)

    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        group_splits[str(row["group_id"])].add(str(row["split"]))
    leaking_groups = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    if leaking_groups:
        raise RuntimeError(f"Duplicate groups leak across splits: {leaking_groups}")

    fingerprint_rows = sorted(
        f"{row['split']}|{row['sha1']}|{row['label_sha256']}|{row['source']}" for row in manifest
    )
    fingerprint = hashlib.sha256("\n".join(fingerprint_rows).encode("utf-8")).hexdigest()
    all_sources = sorted({sample.source for values in assignments.values() for sample in values})
    metadata = {
        "version": "primary_full_v1",
        "fingerprint": fingerprint,
        "seed": seed,
        "images": len(manifest),
        "objects": sum(int(row["object_count"]) for row in manifest),
        "groups": len(groups),
        "class_names": PRIMARY_NAMES,
        "canonical_class_names": CANONICAL_NAMES,
        "split_images": {split: len(assignments[split]) for split in SPLITS},
        "source_images": {
            split: {source: source_stats[(split, source)] for source in all_sources}
            for split in SPLITS
        },
        "excluded_images": len(exclusions),
        "empty_label_images": sum(1 for row in manifest if int(row["object_count"]) == 0),
        "exact_duplicates_removed": sum(1 for row in duplicate_report if not row["selected"]),
        "exact_duplicate_conflict_groups": len(
            {str(row["sha1"]) for row in duplicate_report if row["label_conflict"]}
        ),
        "link_method": dict(link_stats),
        "split_leakage_groups": 0,
        "policy": {
            "authoritative_source": "Dataset/labels",
            "historical_labels_used": False,
            "derived_datasets_used": False,
            "auxiliary_source": "datasets/raw/supervisely labelled train and val subsets",
            "exact_duplicate_precedence": "primary first, then richest label set",
        },
    }
    (output / "metadata" / "dataset_fingerprint.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "metadata" / "class_map.json").write_text(
        json.dumps(
            {
                "names": {str(index): name for index, name in enumerate(PRIMARY_NAMES)},
                "canonical_names": {str(index): name for index, name in enumerate(CANONICAL_NAMES)},
                "auxiliary_mapping": {
                    "insulator": 0,
                    "broken": 1,
                    "pollution-flashover": 2,
                    "missing_disc_drop": 3,
                },
                "notes": "Auxiliary source has no class-3 objects; class 3 is supplied by the primary dataset.",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {output.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                "nc: 4",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(PRIMARY_NAMES)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "\n".join(
            [
                "# primary_full_v1 数据集",
                "",
                "该版本以 `Dataset/labels` 的新人工标注为唯一主体和最高优先级，",
                "并仅加入独立的 Supervisely 有标注原图作为辅助数据。历史旧标签、",
                "伪标签、裁剪集、切片集、COCO 转换集和其他派生训练产物均未使用。",
                "",
                "## 类别",
                "",
                "- `0`: insulator string",
                "- `1`: broken shell",
                "- `2`: flavor（辅助源的 pollution-flashover）",
                "- `3`: diaochuan",
                "",
                "## 数据质量策略",
                "",
                "- 按图像内容 SHA-1 精确去重，新人工标注始终优先。",
                "- 相似图和同系列图被分到同一个 split，避免训练/验证/测试泄漏。",
                "- Supervisely 无标注测试图不当作负样本，记录在 `metadata/exclusions.csv`。",
                "- 主体新标注中的空标签文件按已审核负样本保留。",
                "- 所有输出标签均统一为 YOLO 的 `class x_center y_center width height` 格式。",
                "",
                "## 追溯与复现",
                "",
                "构建参数、统计和指纹见 `metadata/dataset_fingerprint.json`；",
                "每张图的来源及输出位置见 `metadata/split_manifest.csv`。",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    primary = args.primary.resolve()
    public_root = args.public_root.resolve()
    output = args.output.resolve()
    ratios = (args.train, args.val, args.test)

    primary_samples = collect_primary(primary)
    public_samples, exclusions = collect_public(public_root)
    samples = primary_samples + public_samples
    print(
        f"collected primary={len(primary_samples)} auxiliary={len(public_samples)} "
        f"excluded_unlabelled={len(exclusions)}"
    )
    fingerprint_samples(samples)
    samples, duplicate_report = deduplicate_exact(samples)
    print(
        f"exact deduplication kept={len(samples)} "
        f"removed={sum(1 for row in duplicate_report if not row['selected'])}"
    )
    groups = group_samples(samples, args.phash_distance)
    assignments = stratified_group_split(groups, ratios, args.seed)
    write_dataset(
        output,
        assignments,
        groups,
        exclusions,
        duplicate_report,
        args.seed,
        args.overwrite,
        (primary, public_root),
    )
    print(f"wrote {len(samples)} images in {len(groups)} groups to {output}")
    for split in SPLITS:
        counts = Counter(label.class_id for sample in assignments[split] for label in sample.labels)
        sources = Counter(sample.source for sample in assignments[split])
        print(
            f"{split}: images={len(assignments[split])} "
            f"objects={dict(sorted(counts.items()))} sources={dict(sorted(sources.items()))}"
        )


if __name__ == "__main__":
    main()
