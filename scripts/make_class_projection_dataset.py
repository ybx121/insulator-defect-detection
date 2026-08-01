#!/usr/bin/env python3
"""Create a hard-linked YOLO dataset with a projected class space."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        required=True,
        help="Source class IDs to retain, in the desired output-class order",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        help="Optionally retain only these source names from split_manifest.csv",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def project_label(source: Path, destination: Path, class_map: dict[int, int]) -> int:
    rows: list[str] = []
    if source.exists():
        for raw in source.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            fields = raw.split()
            source_class = int(float(fields[0]))
            if source_class in class_map:
                fields[0] = str(class_map[source_class])
                rows.append(" ".join(fields))
    destination.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if len(set(args.classes)) != len(args.classes):
        raise ValueError("--classes must not contain duplicates")
    source_yaml = source / "data_unbalanced.yaml"
    if not source_yaml.exists():
        raise FileNotFoundError(source_yaml)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    source_data = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    raw_names = source_data["names"]
    source_names = (
        [raw_names.get(index, raw_names.get(str(index))) for index in range(len(raw_names))]
        if isinstance(raw_names, dict)
        else list(raw_names)
    )
    invalid = [class_id for class_id in args.classes if class_id < 0 or class_id >= len(source_names)]
    if invalid:
        raise ValueError(f"Invalid source class IDs: {invalid}")
    class_map = {source_class: output_class for output_class, source_class in enumerate(args.classes)}
    allowed: dict[str, set[str]] | None = None
    if args.sources:
        requested = set(args.sources)
        allowed = {split: set() for split in ("train", "val", "test")}
        found_sources: set[str] = set()
        manifest_path = source / "metadata" / "split_manifest.csv"
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["source"] in requested:
                    allowed[row["split"]].add(Path(row["image"]).name)
                    found_sources.add(row["source"])
        missing = requested - found_sources
        if missing:
            raise ValueError(f"No samples found for sources: {sorted(missing)}")

    stats: dict[str, dict[str, int]] = {}
    digest = hashlib.sha256()
    for split in ("train", "val", "test"):
        source_images = source / "images" / split
        source_labels = source / "labels" / split
        output_images = output / "images" / split
        output_labels = output / "labels" / split
        output_images.mkdir(parents=True)
        output_labels.mkdir(parents=True)
        image_count = object_count = 0
        for image in sorted(path for path in source_images.iterdir() if path.suffix.lower() in IMAGE_EXTS):
            if allowed is not None and image.name not in allowed[split]:
                continue
            destination_image = output_images / image.name
            destination_label = output_labels / f"{image.stem}.txt"
            link_or_copy(image, destination_image)
            object_count += project_label(source_labels / f"{image.stem}.txt", destination_label, class_map)
            image_count += 1
            digest.update(f"{split}/{image.name}\n".encode())
            digest.update(destination_label.read_bytes())
        stats[split] = {"images": image_count, "objects": object_count}

    names = [source_names[class_id] for class_id in args.classes]
    data = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
        "nc": len(names),
    }
    (output / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    manifest = {
        "source": str(source),
        "source_fingerprint": json.loads(
            (source / "metadata" / "dataset_fingerprint.json").read_text(encoding="utf-8")
        ),
        "source_classes": args.classes,
        "sources": args.sources,
        "names": names,
        "class_map": class_map,
        "stats": stats,
        "projection_fingerprint": digest.hexdigest(),
    }
    (output / "projection_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
