#!/usr/bin/env python3
"""Validate a YOLO dataset directory."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--num-classes", type=int)
    return parser.parse_args()


def validate_split(root: Path, split: str, num_classes: int | None) -> tuple[int, int, Counter[int], list[str]]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    images = {p.stem: p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
    labels = {p.stem: p for p in label_dir.iterdir() if p.is_file() and p.suffix == ".txt"}
    errors: list[str] = []
    for stem in sorted(set(images) - set(labels)):
        errors.append(f"{split}: missing label for {images[stem]}")
    for stem in sorted(set(labels) - set(images)):
        errors.append(f"{split}: orphan label {labels[stem]}")
    class_counts: Counter[int] = Counter()
    row_count = 0
    for label in sorted(labels.values()):
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
    return len(images), len(labels), class_counts, errors


def main() -> None:
    args = parse_args()
    root = args.dataset.resolve()
    all_errors: list[str] = []
    total_images = 0
    total_labels = 0
    total_classes: Counter[int] = Counter()
    for split in ["train", "val", "test"]:
        images, labels, class_counts, errors = validate_split(root, split, args.num_classes)
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
