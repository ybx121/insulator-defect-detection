#!/usr/bin/env python3
"""Validate a YOLO dataset directory."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--num-classes", type=int)
    return parser.parse_args()


def label_for_image(image: Path) -> Path:
    parts = list(image.parts)
    try:
        index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as exc:
        raise ValueError(f"Expected an images directory in {image}") from exc
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def resolve_images(root: Path, value: str | list[str]) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    images: list[Path] = []
    for raw in values:
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        if path.is_dir():
            images.extend(item for item in path.rglob("*") if item.suffix.lower() in IMAGE_EXTS)
        elif path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                raw_item = line.strip()
                if not raw_item:
                    continue
                item = Path(raw_item)
                if not item.is_absolute():
                    item = path.parent / item
                images.append(item)
        else:
            raise FileNotFoundError(path)
    return sorted(set(image.resolve() for image in images))


def validate_split(images: list[Path], split: str, num_classes: int | None) -> tuple[int, int, Counter[int], list[str]]:
    image_map = {str(path): path for path in images}
    labels = {str(path): label_for_image(path) for path in images}
    errors: list[str] = []
    for key, label in labels.items():
        if not label.exists():
            errors.append(f"{split}: missing label for {image_map[key]}")
    class_counts: Counter[int] = Counter()
    row_count = 0
    for label in sorted(set(labels.values())):
        if not label.exists():
            continue
        for line_number, raw in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            parts = raw.split()
            if len(parts) != 5:
                errors.append(f"{label}:{line_number}: expected 5 columns")
                continue
            try:
                class_id = int(float(parts[0]))
                coords = [float(value) for value in parts[1:]]
            except ValueError:
                errors.append(f"{label}:{line_number}: non-numeric row")
                continue
            if num_classes is not None and not 0 <= class_id < num_classes:
                errors.append(f"{label}:{line_number}: class id {class_id} outside 0..{num_classes - 1}")
            if any(value < 0 or value > 1 for value in coords):
                errors.append(f"{label}:{line_number}: coordinate outside 0..1")
            class_counts[class_id] += 1
            row_count += 1
    return len(images), sum(path.exists() for path in labels.values()), class_counts, errors


def main() -> None:
    args = parse_args()
    root = args.dataset.resolve()
    config = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    data_root = Path(config.get("path", root))
    if not data_root.is_absolute():
        data_root = root / data_root
    all_errors: list[str] = []
    total_images = 0
    total_labels = 0
    total_classes: Counter[int] = Counter()
    for split in ["train", "val", "test"]:
        image_paths = resolve_images(data_root.resolve(), config[split])
        images, labels, class_counts, errors = validate_split(image_paths, split, args.num_classes)
        total_images += images
        total_labels += labels
        total_classes.update(class_counts)
        all_errors.extend(errors)
        print(f"{split}: images={images} labels={labels} objects={dict(sorted(class_counts.items()))}")
    print(f"total: images={total_images} labels={total_labels} objects={dict(sorted(total_classes.items()))}")
    if all_errors:
        print("errors:")
        for error in all_errors[:200]:
            print(f"  - {error}")
        raise SystemExit(1)
    print("validation passed")


if __name__ == "__main__":
    main()
