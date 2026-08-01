#!/usr/bin/env python3
"""Build unified coarse/fine YOLO datasets for insulator defect research."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

FINE_NAMES = [
    "insulator_string",
    "broken_shell",
    "flashover_pollution",
    "missing_disc_drop",
]
COARSE_NAMES = ["insulator_string", "defect"]

DATASET_FINE_MAP = {
    0: 0,  # insulator string -> insulator_string
    1: 1,  # broken shell -> broken_shell
    2: 2,  # flavor -> flashover_pollution
    3: 3,  # diaochuan -> missing_disc_drop
}
DATASET_COARSE_MAP = {0: 0, 1: 1, 2: 1, 3: 1}
DATASET_ORIGINAL_NAMES = {
    0: "insulator string",
    1: "broken shell",
    2: "flavor",
    3: "diaochuan",
}


@dataclass(frozen=True)
class ObjectLabel:
    class_id: int
    original_class: str
    mapped_class: str
    x: float
    y: float
    w: float
    h: float

    def to_yolo(self) -> str:
        return f"{self.class_id} {self.x:.6f} {self.y:.6f} {self.w:.6f} {self.h:.6f}"


@dataclass(frozen=True)
class Sample:
    source_dataset: str
    source_image: Path
    output_stem: str
    image_sha1: str
    labels: tuple[ObjectLabel, ...]
    note: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root")
    parser.add_argument(
        "--mode",
        choices=["coarse", "fine", "all"],
        default="all",
        help="Dataset variant to build",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets"),
        help="Output root. Mode-specific folders are created inside this root.",
    )
    parser.add_argument(
        "--cplid-fine-labels",
        type=Path,
        help="CSV with columns image_stem,fine_class for CPLID defective relabeling",
    )
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument(
        "--dataset-conflict-policy",
        choices=["keep-richest", "skip"],
        default="keep-richest",
        help="How to handle identical Dataset images with different labels",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_images(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_yolo_rows(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label row in {path}:{line_number}: {raw!r}")
        class_id = int(float(parts[0]))
        coords = tuple(float(value) for value in parts[1:])
        if class_id not in DATASET_FINE_MAP:
            raise ValueError(f"Unknown class id {class_id} in {path}:{line_number}")
        if any(value < 0 or value > 1 for value in coords):
            raise ValueError(f"Coordinate out of range in {path}:{line_number}")
        rows.append((class_id, *coords))
    if not rows:
        raise ValueError(f"Empty label file: {path}")
    return rows


def map_dataset_labels(
    label_path: Path,
    class_map: dict[int, int],
    names: list[str],
) -> tuple[ObjectLabel, ...]:
    objects: list[ObjectLabel] = []
    for original_id, x, y, w, h in read_yolo_rows(label_path):
        mapped_id = class_map[original_id]
        objects.append(
            ObjectLabel(
                class_id=mapped_id,
                original_class=DATASET_ORIGINAL_NAMES[original_id],
                mapped_class=names[mapped_id],
                x=x,
                y=y,
                w=w,
                h=h,
            )
        )
    return tuple(objects)


def collect_dataset_samples(
    root: Path,
    mode: str,
    conflict_policy: str,
    report_dir: Path,
) -> list[Sample]:
    image_dir = root / "Dataset" / "datasets" / "mydata" / "images"
    label_dir = root / "Dataset" / "datasets" / "mydata" / "label"
    names = FINE_NAMES if mode == "fine" else COARSE_NAMES
    class_map = DATASET_FINE_MAP if mode == "fine" else DATASET_COARSE_MAP

    by_hash: dict[str, list[tuple[Path, tuple[ObjectLabel, ...]]]] = defaultdict(list)
    for image in iter_images(image_dir):
        label_path = label_dir / f"{image.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {image}")
        by_hash[sha1_file(image)].append((image, map_dataset_labels(label_path, class_map, names)))

    report_dir.mkdir(parents=True, exist_ok=True)
    conflict_path = report_dir / f"dataset_duplicate_report_{mode}.csv"
    samples: list[Sample] = []
    with conflict_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sha1",
                "kept",
                "status",
                "image",
                "object_count",
                "label_signature",
            ],
        )
        writer.writeheader()
        for digest, candidates in sorted(by_hash.items()):
            signatures = ["|".join(obj.to_yolo() for obj in labels) for _, labels in candidates]
            has_conflict = len(set(signatures)) > 1
            if len(candidates) > 1 and has_conflict and conflict_policy == "skip":
                kept_index = None
                status = "skipped_conflicting_duplicate"
            else:
                kept_index = max(range(len(candidates)), key=lambda idx: len(candidates[idx][1]))
                status = "kept_conflicting_richest" if has_conflict else "kept"
            for idx, (image, labels) in enumerate(candidates):
                writer.writerow(
                    {
                        "sha1": digest,
                        "kept": bool(kept_index == idx),
                        "status": status,
                        "image": str(image),
                        "object_count": len(labels),
                        "label_signature": signatures[idx],
                    }
                )
            if kept_index is None:
                continue
            image, labels = candidates[kept_index]
            samples.append(
                Sample(
                    source_dataset="Dataset",
                    source_image=image,
                    output_stem=f"dataset_{image.stem}",
                    image_sha1=digest,
                    labels=labels,
                    note=status,
                )
            )
    return samples


def parse_voc_objects(xml_path: Path) -> tuple[int, int, list[tuple[str, tuple[float, float, float, float]]]]:
    root = ET.parse(xml_path).getroot()
    width = int(root.findtext("size/width", "0"))
    height = int(root.findtext("size/height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}")
    objects: list[tuple[str, tuple[float, float, float, float]]] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = max(0.0, min(float(box.findtext("xmin", "0")), width))
        ymin = max(0.0, min(float(box.findtext("ymin", "0")), height))
        xmax = max(0.0, min(float(box.findtext("xmax", "0")), width))
        ymax = max(0.0, min(float(box.findtext("ymax", "0")), height))
        if xmax <= xmin or ymax <= ymin:
            continue
        x = ((xmin + xmax) / 2.0) / width
        y = ((ymin + ymax) / 2.0) / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        objects.append((name, (x, y, w, h)))
    return width, height, objects


def load_cplid_relabels(path: Path | None) -> dict[str, int]:
    if not path:
        return {}
    mapping: dict[str, int] = {}
    aliases = {
        "broken_shell": 1,
        "broken": 1,
        "flashover_pollution": 2,
        "flashover": 2,
        "pollution": 2,
        "missing_disc_drop": 3,
        "drop": 3,
        "missing": 3,
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"image_stem", "fine_class"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")
        for row in reader:
            stem = (row["image_stem"] or "").strip()
            label = (row["fine_class"] or "").strip()
            if not stem or not label:
                continue
            if label.isdigit():
                class_id = int(label)
            else:
                class_id = aliases.get(label)
                if class_id is None:
                    raise ValueError(f"Unknown fine class {label!r} in {path}")
            if class_id not in {1, 2, 3}:
                raise ValueError(f"CPLID defect fine_class must be 1, 2, or 3: {row}")
            mapping[stem] = class_id
    return mapping


def collect_cplid_normal(root: Path, mode: str) -> list[Sample]:
    base = root / "InsulatorDataSet" / "Normal_Insulators"
    samples: list[Sample] = []
    names = FINE_NAMES if mode == "fine" else COARSE_NAMES
    for image in iter_images(base / "images"):
        _, _, voc_objects = parse_voc_objects(base / "labels" / f"{image.stem}.xml")
        labels = tuple(
            ObjectLabel(0, original, names[0], x, y, w, h)
            for original, (x, y, w, h) in voc_objects
            if original == "insulator"
        )
        if not labels:
            raise ValueError(f"No insulator labels for {image}")
        samples.append(
            Sample(
                source_dataset="CPLID_Normal",
                source_image=image,
                output_stem=f"cplid_normal_{image.stem}",
                image_sha1=sha1_file(image),
                labels=labels,
            )
        )
    return samples


def collect_cplid_defective(
    root: Path,
    mode: str,
    relabels: dict[str, int],
    report_dir: Path,
) -> list[Sample]:
    base = root / "InsulatorDataSet" / "Defective_Insulators"
    samples: list[Sample] = []
    template_path = report_dir / "cplid_defective_relabel_template.csv"
    report_dir.mkdir(parents=True, exist_ok=True)
    with template_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_stem",
                "fine_class",
                "allowed_values",
                "source_image",
                "note",
            ],
        )
        writer.writeheader()
        for image in iter_images(base / "images"):
            note = "filled_from_cplid_fine_labels" if image.stem in relabels else "needs_manual_relabel"
            writer.writerow(
                {
                    "image_stem": image.stem,
                    "fine_class": FINE_NAMES[relabels[image.stem]] if image.stem in relabels else "",
                    "allowed_values": "broken_shell|flashover_pollution|missing_disc_drop",
                    "source_image": str(image),
                    "note": note,
                }
            )

    names = FINE_NAMES if mode == "fine" else COARSE_NAMES
    for image in iter_images(base / "images"):
        labels: list[ObjectLabel] = []
        _, _, insulators = parse_voc_objects(base / "labels" / "insulator" / f"{image.stem}.xml")
        for original, (x, y, w, h) in insulators:
            if original == "insulator":
                labels.append(ObjectLabel(0, original, names[0], x, y, w, h))
        _, _, defects = parse_voc_objects(base / "labels" / "defect" / f"{image.stem}.xml")
        if mode == "coarse":
            for original, (x, y, w, h) in defects:
                if original == "defect":
                    labels.append(ObjectLabel(1, original, names[1], x, y, w, h))
        else:
            if image.stem not in relabels:
                continue
            fine_id = relabels[image.stem]
            for original, (x, y, w, h) in defects:
                if original == "defect":
                    labels.append(ObjectLabel(fine_id, original, names[fine_id], x, y, w, h))
        if labels:
            samples.append(
                Sample(
                    source_dataset="CPLID_Defective",
                    source_image=image,
                    output_stem=f"cplid_defective_{image.stem}",
                    image_sha1=sha1_file(image),
                    labels=tuple(labels),
                    note="coarse_generic_defect" if mode == "coarse" else "manual_relabel_required",
                )
            )
    return samples


def split_samples(samples: list[Sample], seed: int, ratios: tuple[float, float, float]) -> dict[str, list[Sample]]:
    total_ratio = sum(ratios)
    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    shuffled = samples[:]
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * ratios[0])
    val_end = train_end + int(len(shuffled) * ratios[1])
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def write_dataset(output: Path, mode: str, splits: dict[str, list[Sample]], overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite")
        shutil.rmtree(output)
    names = FINE_NAMES if mode == "fine" else COARSE_NAMES
    for split in splits:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    metadata_dir = output / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    sample_rows: list[dict[str, object]] = []
    object_rows: list[dict[str, object]] = []
    class_counts: Counter[int] = Counter()
    source_counts: Counter[str] = Counter()
    used_stems: set[str] = set()

    for split, samples in splits.items():
        for sample in samples:
            stem = unique_stem(sample.output_stem, used_stems)
            image_name = f"{stem}{sample.source_image.suffix.lower()}"
            label_name = f"{stem}.txt"
            image_rel = Path("images") / split / image_name
            label_rel = Path("labels") / split / label_name
            shutil.copy2(sample.source_image, output / image_rel)
            (output / label_rel).write_text(
                "\n".join(label.to_yolo() for label in sample.labels) + "\n",
                encoding="utf-8",
            )
            source_counts[sample.source_dataset] += 1
            sample_rows.append(
                {
                    "split": split,
                    "image": str(image_rel),
                    "label": str(label_rel),
                    "source_dataset": sample.source_dataset,
                    "source_image": str(sample.source_image),
                    "image_sha1": sample.image_sha1,
                    "object_count": len(sample.labels),
                    "note": sample.note,
                }
            )
            for idx, label in enumerate(sample.labels):
                class_counts[label.class_id] += 1
                object_rows.append(
                    {
                        "split": split,
                        "image": str(image_rel),
                        "object_index": idx,
                        "source_dataset": sample.source_dataset,
                        "original_class": label.original_class,
                        "mapped_class": label.mapped_class,
                        "class_id": label.class_id,
                        "x_center": f"{label.x:.6f}",
                        "y_center": f"{label.y:.6f}",
                        "width": f"{label.w:.6f}",
                        "height": f"{label.h:.6f}",
                    }
                )

    write_csv(metadata_dir / "samples.csv", sample_rows)
    write_csv(metadata_dir / "objects.csv", object_rows)
    class_map = {
        "mode": mode,
        "names": {str(idx): name for idx, name in enumerate(names)},
        "dataset_original_names": DATASET_ORIGINAL_NAMES,
        "dataset_class_aliases": {
            "flavor": "flashover_pollution",
            "diaochuan": "missing_disc_drop",
        },
    }
    (metadata_dir / "class_map.json").write_text(json.dumps(class_map, indent=2), encoding="utf-8")
    (output / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {output.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                f"nc: {len(names)}",
                "names:",
                *[f"  {idx}: {name}" for idx, name in enumerate(names)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "\n".join(
            [
                f"# Unified {mode.title()} Insulator Dataset",
                "",
                f"Classes: {', '.join(names)}.",
                "",
                f"Samples by source: {dict(source_counts)}",
                f"Objects by class id: {dict(sorted(class_counts.items()))}",
                "",
                "Metadata files:",
                "",
                "- `metadata/samples.csv`",
                "- `metadata/objects.csv`",
                "- `metadata/class_map.json`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def unique_stem(stem: str, used: set[str]) -> str:
    candidate = stem
    counter = 1
    while candidate in used:
        counter += 1
        candidate = f"{stem}_{counter}"
    used.add(candidate)
    return candidate


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_mode(args: argparse.Namespace, mode: str) -> Path:
    project_root = args.root.resolve()
    output_root = args.output if args.output.is_absolute() else project_root / args.output
    output = output_root / f"unified_{mode}"
    report_dir = output_root / "reports"
    relabels = load_cplid_relabels(args.cplid_fine_labels)

    samples = []
    samples.extend(
        collect_dataset_samples(project_root, mode, args.dataset_conflict_policy, report_dir)
    )
    samples.extend(collect_cplid_normal(project_root, mode))
    samples.extend(collect_cplid_defective(project_root, mode, relabels, report_dir))
    splits = split_samples(samples, args.seed, (args.train, args.val, args.test))
    write_dataset(output, mode, splits, args.overwrite)
    print(f"{mode}: wrote {sum(len(items) for items in splits.values())} images to {output}")
    for split, split_samples_ in splits.items():
        by_source = Counter(sample.source_dataset for sample in split_samples_)
        print(f"  {split}: {len(split_samples_)} images {dict(by_source)}")
    return output


def main() -> None:
    args = parse_args()
    modes = ["coarse", "fine"] if args.mode == "all" else [args.mode]
    for mode in modes:
        build_mode(args, mode)


if __name__ == "__main__":
    main()
