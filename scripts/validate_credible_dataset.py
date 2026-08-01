#!/usr/bin/env python3
"""Validate the versioned split and labels of a credible YOLO dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset.resolve()
    manifest_path = root / "metadata" / "split_manifest.csv"
    rows = list(csv.DictReader(manifest_path.open(newline="", encoding="utf-8")))
    if not rows:
        raise ValueError("Empty split manifest")
    group_splits: dict[str, set[str]] = defaultdict(set)
    split_classes: dict[str, set[int]] = defaultdict(set)
    split_images: dict[str, int] = defaultdict(int)
    errors: list[str] = []
    for row in rows:
        split = row["split"]
        group_splits[row["group_id"]].add(split)
        split_images[split] += 1
        image_path, label_path = root / row["image"], root / row["label"]
        if not image_path.exists() or not label_path.exists():
            errors.append(f"missing pair: {image_path} / {label_path}")
            continue
        for line_number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            parts = raw.split()
            if len(parts) != 5:
                errors.append(f"invalid row: {label_path}:{line_number}")
                continue
            class_id = int(float(parts[0]))
            coords = [float(value) for value in parts[1:]]
            if class_id not in range(4) or any(not 0 <= value <= 1 for value in coords) or coords[2] <= 0 or coords[3] <= 0:
                errors.append(f"invalid label: {label_path}:{line_number}")
            split_classes[split].add(class_id)
    leaking = {group: splits for group, splits in group_splits.items() if len(splits) != 1}
    if leaking:
        errors.append(f"groups cross splits: {list(leaking.items())[:5]}")
    for split in ("train", "val", "test"):
        if split_classes[split] != {0, 1, 2, 3}:
            errors.append(f"{split} classes: {split_classes[split]}")

    fingerprint_source = "\n".join(
        f"{row['split']}|{row['group_id']}|{row['sha1']}|{row['classes']}" for row in rows
    )
    split_fingerprint = hashlib.sha256(fingerprint_source.encode()).hexdigest()
    fingerprint_metadata = json.loads(
        (root / "metadata" / "dataset_fingerprint.json").read_text(encoding="utf-8")
    )
    expected = fingerprint_metadata["fingerprint"]
    source_fingerprint = fingerprint_metadata.get("source_fingerprint")
    if source_fingerprint:
        if split_fingerprint != source_fingerprint:
            errors.append(
                f"source split fingerprint mismatch: {split_fingerprint} != {source_fingerprint}"
            )
        settings = {
            key: value
            for key, value in fingerprint_metadata.items()
            if key not in {"dataset_version", "fingerprint"}
        }
        derived = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        if derived != expected:
            errors.append(f"derived fingerprint mismatch: {derived} != {expected}")
    elif split_fingerprint != expected:
        errors.append(f"fingerprint mismatch: {split_fingerprint} != {expected}")

    train_root = (root / "images" / "train").resolve()
    data_path = root / "data.yaml"
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    train_entry = Path(data["train"])
    if not train_entry.is_absolute():
        train_entry = root / train_entry
    if train_entry.is_file():
        for line in train_entry.read_text(encoding="utf-8").splitlines():
            path = Path(line).resolve()
            if train_root not in path.parents or not path.exists():
                errors.append(f"invalid training path: {path}")
    elif not train_entry.is_dir():
        errors.append(f"missing training source: {train_entry}")
    if errors:
        raise SystemExit("\n".join(errors[:30]))
    print(
        f"valid: images={len(rows)} groups={len(group_splits)} "
        f"splits={dict(split_images)} fingerprint={expected}"
    )


if __name__ == "__main__":
    main()
