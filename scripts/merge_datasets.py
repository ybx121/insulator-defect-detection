#!/usr/bin/env python3
"""Merge insulator datasets into a four-class YOLO dataset.

The merged dataset follows the conservative strategy:
- keep all samples from Dataset, preserving its four YOLO classes;
- add only normal CPLID samples from InsulatorDataSet/Normal_Insulators;
- map CPLID "insulator" boxes to class 0, "insulator string";
- skip CPLID defective samples because their defect labels are not fine-grained.
"""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


NAMES = ["insulator string", "broken shell", "flavor", "diaochuan"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class Sample:
    source: str
    stem: str
    image: Path
    label_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("merged_dataset"),
        help="Output dataset directory, relative to root unless absolute",
    )
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output directory before writing",
    )
    return parser.parse_args()


def image_files(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def normalize_yolo_label(label: Path) -> str:
    rows: list[str] = []
    for line_number, raw in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO row in {label}:{line_number}: {raw!r}")
        class_id = int(float(parts[0]))
        if class_id < 0 or class_id >= len(NAMES):
            raise ValueError(f"Invalid class id in {label}:{line_number}: {class_id}")
        coords = [float(value) for value in parts[1:]]
        if any(value < 0.0 or value > 1.0 for value in coords):
            raise ValueError(f"YOLO coordinate out of range in {label}:{line_number}")
        rows.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in coords))
    if not rows:
        raise ValueError(f"Empty label file: {label}")
    return "\n".join(rows) + "\n"


def collect_dataset_samples(root: Path) -> list[Sample]:
    base = root / "Dataset" / "datasets" / "mydata"
    image_dir = base / "images"
    label_dir = base / "label"
    samples: list[Sample] = []
    for image in image_files(image_dir):
        label = label_dir / f"{image.stem}.txt"
        if not label.exists():
            raise FileNotFoundError(f"Missing label for {image}")
        samples.append(
            Sample(
                source="dataset",
                stem=image.stem,
                image=image,
                label_text=normalize_yolo_label(label),
            )
        )
    return samples


def voc_box_to_yolo(box: ET.Element, width: int, height: int) -> tuple[float, float, float, float]:
    xmin = float(box.findtext("xmin", "0"))
    ymin = float(box.findtext("ymin", "0"))
    xmax = float(box.findtext("xmax", "0"))
    ymax = float(box.findtext("ymax", "0"))
    xmin = max(0.0, min(xmin, width))
    xmax = max(0.0, min(xmax, width))
    ymin = max(0.0, min(ymin, height))
    ymax = max(0.0, min(ymax, height))
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Invalid VOC box: {(xmin, ymin, xmax, ymax)}")
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return x_center, y_center, box_width, box_height


def convert_cplid_normal_xml(xml_path: Path) -> str:
    root = ET.parse(xml_path).getroot()
    width = int(root.findtext("size/width", "0"))
    height = int(root.findtext("size/height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}")
    rows: list[str] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name != "insulator":
            raise ValueError(f"Unexpected CPLID normal class {name!r} in {xml_path}")
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Missing bndbox in {xml_path}")
        coords = voc_box_to_yolo(box, width, height)
        rows.append("0 " + " ".join(f"{value:.6f}" for value in coords))
    if not rows:
        raise ValueError(f"No insulator objects in {xml_path}")
    return "\n".join(rows) + "\n"


def collect_cplid_normal_samples(root: Path) -> list[Sample]:
    base = root / "InsulatorDataSet" / "Normal_Insulators"
    image_dir = base / "images"
    label_dir = base / "labels"
    samples: list[Sample] = []
    for image in image_files(image_dir):
        xml_path = label_dir / f"{image.stem}.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"Missing CPLID normal XML for {image}")
        samples.append(
            Sample(
                source="cplid_normal",
                stem=image.stem,
                image=image,
                label_text=convert_cplid_normal_xml(xml_path),
            )
        )
    return samples


def split_samples(samples: list[Sample], seed: int, ratios: tuple[float, float, float]) -> dict[str, list[Sample]]:
    train_ratio, val_ratio, test_ratio = ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def write_dataset(output: Path, splits: dict[str, list[Sample]]) -> None:
    for split in splits:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    seen_names: set[str] = set()
    for split, samples in splits.items():
        for sample in samples:
            name = f"{sample.source}_{sample.stem}"
            if name in seen_names:
                raise ValueError(f"Duplicate output stem: {name}")
            seen_names.add(name)
            dst_image = output / "images" / split / f"{name}{sample.image.suffix.lower()}"
            dst_label = output / "labels" / split / f"{name}.txt"
            shutil.copy2(sample.image, dst_image)
            dst_label.write_text(sample.label_text, encoding="utf-8")

    data_yaml = "\n".join(
        [
            f"path: {output.resolve()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "",
            f"nc: {len(NAMES)}",
            "names:",
            *[f"  {idx}: {name}" for idx, name in enumerate(NAMES)],
            "",
        ]
    )
    (output / "data.yaml").write_text(data_yaml, encoding="utf-8")

    readme = "\n".join(
        [
            "# Merged Insulator Dataset",
            "",
            "Four-class YOLO dataset generated from:",
            "",
            "- `Dataset/datasets/mydata`: all images and YOLO labels.",
            "- `InsulatorDataSet/Normal_Insulators`: normal CPLID images, VOC XML converted to class 0 only.",
            "",
            "`InsulatorDataSet/Defective_Insulators` is intentionally excluded because its defect XML labels are generic `defect` annotations and cannot be mapped reliably to `broken shell`, `flavor`, or `diaochuan`.",
            "",
            "Classes:",
            "",
            *[f"- `{idx}`: `{name}`" for idx, name in enumerate(NAMES)],
            "",
        ]
    )
    (output / "README.md").write_text(readme, encoding="utf-8")


def summarize(splits: dict[str, list[Sample]]) -> None:
    print("Merged dataset summary")
    print("======================")
    for split, samples in splits.items():
        source_counts: dict[str, int] = {}
        object_counts = [0 for _ in NAMES]
        for sample in samples:
            source_counts[sample.source] = source_counts.get(sample.source, 0) + 1
            for row in sample.label_text.splitlines():
                object_counts[int(row.split()[0])] += 1
        print(f"{split}: {len(samples)} images, sources={source_counts}, objects={object_counts}")


def main() -> None:
    args = parse_args()
    project_root = args.root.resolve()
    output = args.output if args.output.is_absolute() else project_root / args.output

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")
        shutil.rmtree(output)

    dataset_samples = collect_dataset_samples(project_root)
    cplid_normal_samples = collect_cplid_normal_samples(project_root)
    samples = dataset_samples + cplid_normal_samples
    splits = split_samples(samples, args.seed, (args.train, args.val, args.test))
    write_dataset(output, splits)
    summarize(splits)
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
