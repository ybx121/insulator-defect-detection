#!/usr/bin/env python3
"""Create a storage-efficient, rare-class-balanced YOLO training view."""

from __future__ import annotations

import argparse
import math
import os
import shutil
from collections import Counter
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--max-repeat", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def classes_in_label(path: Path) -> list[int]:
    return [int(float(raw.split()[0])) for raw in path.read_text(encoding="utf-8").splitlines() if raw.strip()]


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    if args.max_repeat < 1:
        raise ValueError("--max-repeat must be positive")
    root = args.dataset.resolve()
    image_source, label_source = root / "images" / "train", root / "labels" / "train"
    image_output, label_output = root / "images" / "train_balanced", root / "labels" / "train_balanced"
    if image_output.exists() or label_output.exists():
        if not args.overwrite:
            raise FileExistsError("Balanced view exists; pass --overwrite")
        shutil.rmtree(image_output, ignore_errors=True)
        shutil.rmtree(label_output, ignore_errors=True)
    image_output.mkdir(parents=True)
    label_output.mkdir(parents=True)

    images = sorted(path for path in image_source.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    image_classes = {
        image: classes_in_label(label_source / f"{image.stem}.txt") for image in images
    }
    counts = Counter(class_id for classes in image_classes.values() for class_id in classes if class_id > 0)
    largest = max(counts.values(), default=1)
    repeats = Counter()
    for image in images:
        defect_classes = {class_id for class_id in image_classes[image] if class_id > 0}
        repeat = max(
            [1, *[min(args.max_repeat, math.ceil(math.sqrt(largest / counts[class_id]))) for class_id in defect_classes]]
        )
        for copy_index in range(repeat):
            stem = image.stem if copy_index == 0 else f"{image.stem}_repeat{copy_index}"
            link_or_copy(image, image_output / f"{stem}{image.suffix.lower()}")
            link_or_copy(label_source / f"{image.stem}.txt", label_output / f"{stem}.txt")
        repeats[repeat] += 1
    print(f"balanced view: source_images={len(images)} output_images={sum(key * value for key, value in repeats.items())} repeat_histogram={dict(repeats)} class_objects={dict(counts)}")


if __name__ == "__main__":
    main()
