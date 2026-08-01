#!/usr/bin/env python3
"""Build a versioned training dataset with high-confidence pseudo labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


NAMES = ["insulator_string", "broken_shell", "flashover_pollution", "missing_disc_drop"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument(
        "--audit-review",
        type=Path,
        help="Optional completed full-dataset audit CSV to preserve in metadata",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-score", type=float, default=0.65)
    parser.add_argument("--min-agreement-iou", type=float, default=0.70)
    parser.add_argument("--max-gt-iou", type=float, default=0.10)
    parser.add_argument("--repeat-changed", type=int, default=8)
    parser.add_argument(
        "--require-review",
        action="store_true",
        help="Only apply rows whose review_status is approved or accepted",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def accepted_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    with args.proposals.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if args.require_review:
        return [
            row
            for row in rows
            if row.get("review_status", "").strip().lower() in {"approved", "accepted"}
        ]
    return [
        row
        for row in rows
        if float(row["score"]) >= args.min_score
        and float(row["agreement_iou"]) >= args.min_agreement_iou
        and float(row["gt_any_iou"]) < args.max_gt_iou
    ]


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    proposals = args.proposals.resolve()
    audit_review = args.audit_review.resolve() if args.audit_review else None
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite")
        shutil.rmtree(output)
    if not (source / "images" / "train").is_dir():
        raise FileNotFoundError(f"invalid source dataset: {source}")
    if args.repeat_changed < 0:
        raise ValueError("--repeat-changed must be non-negative")

    for split in ["train", "val", "test"]:
        for image_path in (source / "images" / split).glob("*"):
            if image_path.is_file():
                link_or_copy(image_path, output / "images" / split / image_path.name)
        destination = output / "labels" / split
        destination.mkdir(parents=True, exist_ok=True)
        for label_path in (source / "labels" / split).glob("*.txt"):
            shutil.copy2(label_path, destination / label_path.name)

    rows = accepted_rows(args)
    added_by_class: Counter[str] = Counter()
    added_by_image: Counter[str] = Counter()
    for row in rows:
        image_path = output / "images" / "train" / row["image"]
        if not image_path.is_file():
            raise FileNotFoundError(f"proposal is not a train image: {row['image']}")
        with Image.open(image_path) as image:
            width, height = image.size
        x1, y1, x2, y2 = (float(row[key]) for key in ["x1", "y1", "x2", "y2"])
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(width), x2), min(float(height), y2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"invalid proposal box for {row['image']}")
        yolo = (
            int(row["class_id"]),
            ((x1 + x2) / 2) / width,
            ((y1 + y2) / 2) / height,
            (x2 - x1) / width,
            (y2 - y1) / height,
        )
        label_path = output / "labels" / "train" / f"{image_path.stem}.txt"
        existing = label_path.read_text(encoding="utf-8")
        if existing and not existing.endswith("\n"):
            existing += "\n"
        existing += f"{yolo[0]} {yolo[1]:.6f} {yolo[2]:.6f} {yolo[3]:.6f} {yolo[4]:.6f}\n"
        label_path.write_text(existing, encoding="utf-8")
        added_by_class[NAMES[yolo[0]]] += 1
        added_by_image[row["image"]] += 1

    output.mkdir(parents=True, exist_ok=True)
    train_images = sorted((output / "images" / "train").glob("*"))
    repeated = [output / "images" / "train" / name for name in sorted(added_by_image)]
    train_view = train_images + repeated * args.repeat_changed
    (output / "train_pseudo.txt").write_text(
        "\n".join(str(path.resolve()) for path in train_view) + "\n", encoding="utf-8"
    )
    (output / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {output}",
                "train: train_pseudo.txt",
                "val: images/val",
                "test: images/test",
                "",
                "nc: 4",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(NAMES)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    source_fingerprint_path = source / "metadata" / "dataset_fingerprint.json"
    source_fingerprint = json.loads(source_fingerprint_path.read_text(encoding="utf-8"))
    settings = {
        "source": str(source),
        "source_fingerprint": source_fingerprint.get("fingerprint"),
        "proposals": str(proposals),
        "proposals_sha256": sha256_file(proposals),
        "min_score": args.min_score,
        "min_agreement_iou": args.min_agreement_iou,
        "max_gt_iou": args.max_gt_iou,
        "accepted": len(rows),
        "images_changed": len(added_by_image),
        "repeat_changed": args.repeat_changed,
        "require_review": args.require_review,
        "train_view_images": len(train_view),
        "added_by_class": dict(added_by_class),
        "label_status": (
            "human-approved missing labels; training only"
            if args.require_review
            else "unreviewed high-confidence pseudo labels; training only"
        ),
    }
    if audit_review:
        if not audit_review.is_file():
            raise FileNotFoundError(audit_review)
        settings["audit_review"] = str(audit_review)
        settings["audit_review_sha256"] = sha256_file(audit_review)
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    metadata = output / "metadata"
    metadata.mkdir(exist_ok=True)
    source_metadata = source / "metadata"
    for name in ("split_manifest.csv", "duplicate_groups.csv", "label_stats.csv"):
        source_file = source_metadata / name
        if source_file.is_file():
            shutil.copy2(source_file, metadata / name)
    if audit_review:
        shutil.copy2(audit_review, metadata / "audit_review.csv")
    (metadata / "pseudo_label_manifest.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )
    (metadata / "dataset_fingerprint.json").write_text(
        json.dumps({"dataset_version": output.name, "fingerprint": fingerprint, **settings}, indent=2),
        encoding="utf-8",
    )
    print(f"built {output} with {len(rows)} pseudo labels in {len(added_by_image)} images")
    print(dict(added_by_class))


if __name__ == "__main__":
    main()
