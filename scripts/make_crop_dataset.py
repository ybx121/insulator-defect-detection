#!/usr/bin/env python3
"""Create a local crop dataset from insulator-string boxes."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
LOCAL_NAMES = ["broken_shell", "flashover_pollution", "missing_disc_drop"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("datasets/unified_fine"))
    parser.add_argument("--output", type=Path, default=Path("datasets/unified_fine_crops_v2"))
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument(
        "--train-jitter-count",
        type=int,
        default=1,
        help="Additional jittered crops per training insulator (default: 1)",
    )
    parser.add_argument(
        "--jitter-center",
        type=float,
        default=0.05,
        help="Maximum training crop center shift as a box-size fraction",
    )
    parser.add_argument(
        "--jitter-scale",
        type=float,
        default=0.10,
        help="Maximum training crop scale change as a fraction",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Recompute crop_manifest.json for an existing output without rewriting images",
    )
    parser.add_argument(
        "--efficient-positive-jitters",
        type=int,
        default=1,
        help="Positive jitter variants kept in the efficient training view",
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Reproduce the old behavior by discarding crops without defects",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        class_id, x, y, w, h = raw.split()
        rows.append((int(float(class_id)), float(x), float(y), float(w), float(h)))
    return rows


def yolo_to_xyxy(row: tuple[int, float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    _, x, y, w, h = row
    x1 = int(round((x - w / 2) * width))
    y1 = int(round((y - h / 2) * height))
    x2 = int(round((x + w / 2) * width))
    y2 = int(round((y + h / 2) * height))
    return x1, y1, x2, y2


def clip_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def expand_int_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    margin: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    dx = int(round((x2 - x1) * margin))
    dy = int(round((y2 - y1) * margin))
    return clip_box((x1 - dx, y1 - dy, x2 + dx, y2 + dy), width, height)


def jitter_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    rng: random.Random,
    center_jitter: float,
    scale_jitter: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2 + rng.uniform(-center_jitter, center_jitter) * box_w
    cy = (y1 + y2) / 2 + rng.uniform(-center_jitter, center_jitter) * box_h
    scale = 1.0 + rng.uniform(-scale_jitter, scale_jitter)
    jittered_w = max(2.0, box_w * scale)
    jittered_h = max(2.0, box_h * scale)
    jittered = (
        int(round(cx - jittered_w / 2)),
        int(round(cy - jittered_h / 2)),
        int(round(cx + jittered_w / 2)),
        int(round(cy + jittered_h / 2)),
    )
    clipped = clip_box(jittered, width, height)
    return clipped if clipped[2] > clipped[0] and clipped[3] > clipped[1] else box


def box_inside(
    defect: tuple[int, int, int, int],
    crop: tuple[int, int, int, int],
    min_center_inside: bool = True,
) -> bool:
    dx1, dy1, dx2, dy2 = defect
    cx, cy = (dx1 + dx2) / 2, (dy1 + dy2) / 2
    x1, y1, x2, y2 = crop
    if min_center_inside:
        return x1 <= cx <= x2 and y1 <= cy <= y2
    return not (dx2 <= x1 or dx1 >= x2 or dy2 <= y1 or dy1 >= y2)


def remap_to_crop(
    class_id: int,
    defect: tuple[int, int, int, int],
    crop: tuple[int, int, int, int],
) -> str | None:
    x1, y1, x2, y2 = crop
    dx1, dy1, dx2, dy2 = defect
    ix1, iy1 = max(dx1, x1), max(dy1, y1)
    ix2, iy2 = min(dx2, x2), min(dy2, y2)
    crop_w, crop_h = x2 - x1, y2 - y1
    if crop_w <= 0 or crop_h <= 0 or ix2 <= ix1 or iy2 <= iy1:
        return None
    local_id = class_id - 1
    cx = ((ix1 + ix2) / 2 - x1) / crop_w
    cy = ((iy1 + iy2) / 2 - y1) / crop_h
    bw = (ix2 - ix1) / crop_w
    bh = (iy2 - iy1) / crop_h
    return f"{local_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def summarize_existing(output: Path) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    for split in ["train", "val", "test"]:
        split_stats = {"crops": 0, "positive": 0, "negative": 0, "boxes": 0, "jittered": 0}
        label_root = output / "labels" / split
        for label_path in label_root.glob("*.txt"):
            rows = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            split_stats["crops"] += 1
            split_stats["boxes"] += len(rows)
            split_stats["jittered"] += int("_jitter" in label_path.stem)
            split_stats["positive" if rows else "negative"] += 1
        stats[split] = split_stats
    return stats


def write_manifest(
    output: Path,
    source: Path,
    args: argparse.Namespace,
    stats: dict[str, dict[str, int]],
) -> None:
    source_manifest = source / "metadata" / "split_manifest.csv"
    manifest = {
        "source": str(source),
        "source_split_manifest_sha256": (
            sha256_file(source_manifest) if source_manifest.is_file() else None
        ),
        "settings": {
            "margin": args.margin,
            "train_jitter_count": args.train_jitter_count,
            "jitter_center": args.jitter_center,
            "jitter_scale": args.jitter_scale,
            "positive_only": args.positive_only,
            "seed": args.seed,
            "efficient_positive_jitters": args.efficient_positive_jitters,
        },
        "classes": LOCAL_NAMES,
        "stats": stats,
    }
    (output / "crop_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def write_efficient_view(output: Path, positive_jitters: int) -> int:
    if positive_jitters < 0:
        raise ValueError("--efficient-positive-jitters must be non-negative")
    selected: list[Path] = []
    for image_path in sorted((output / "images" / "train").glob("*.jpg")):
        stem = image_path.stem
        if "_jitter" not in stem:
            selected.append(image_path.resolve())
            continue
        suffix = stem.rsplit("_jitter", 1)[-1]
        label_path = output / "labels" / "train" / f"{stem}.txt"
        if int(suffix) <= positive_jitters and label_path.read_text(encoding="utf-8").strip():
            selected.append(image_path.resolve())
    (output / "train_efficient.txt").write_text(
        "\n".join(str(path) for path in selected) + "\n", encoding="utf-8"
    )
    (output / "data_efficient.yaml").write_text(
        "\n".join(
            [
                f"path: {output}",
                "train: train_efficient.txt",
                "val: images/val",
                "test: images/test",
                "",
                "nc: 3",
                "names:",
                *[f"  {idx}: {name}" for idx, name in enumerate(LOCAL_NAMES)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return len(selected)


def main() -> None:
    args = parse_args()
    if args.margin < 0:
        raise ValueError("--margin must be non-negative")
    if args.train_jitter_count < 0:
        raise ValueError("--train-jitter-count must be non-negative")
    if not 0 <= args.jitter_center < 0.5:
        raise ValueError("--jitter-center must be in [0, 0.5)")
    if not 0 <= args.jitter_scale < 1.0:
        raise ValueError("--jitter-scale must be in [0, 1.0)")

    source = args.input.resolve()
    output = args.output.resolve()
    if args.manifest_only:
        if not output.is_dir():
            raise FileNotFoundError(f"crop output does not exist: {output}")
        stats = summarize_existing(output)
        stats["efficient_train"] = {"images": write_efficient_view(output, args.efficient_positive_jitters)}
        write_manifest(output, source, args, stats)
        print(f"updated {output / 'crop_manifest.json'}")
        return
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite")
        shutil.rmtree(output)
    for split in ["train", "val", "test"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    stats: dict[str, dict[str, int]] = {
        split: {"crops": 0, "positive": 0, "negative": 0, "boxes": 0, "jittered": 0}
        for split in ["train", "val", "test"]
    }
    for split in ["train", "val", "test"]:
        for image_path in sorted((source / "images" / split).iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            label_path = source / "labels" / split / f"{image_path.stem}.txt"
            rows = read_labels(label_path)
            insulators = [row for row in rows if row[0] == 0]
            defects = [row for row in rows if row[0] > 0]
            with Image.open(image_path) as image:
                width, height = image.size
                for idx, insulator in enumerate(insulators):
                    base_box = yolo_to_xyxy(insulator, width, height)
                    crop_boxes = [(expand_int_box(base_box, width, height, args.margin), False)]
                    if split == "train":
                        for _ in range(args.train_jitter_count):
                            shifted = jitter_box(
                                base_box,
                                width,
                                height,
                                rng,
                                args.jitter_center,
                                args.jitter_scale,
                            )
                            crop_boxes.append((expand_int_box(shifted, width, height, args.margin), True))

                    for variant, (crop, is_jittered) in enumerate(crop_boxes):
                        crop_rows: list[str] = []
                        for defect in defects:
                            defect_box = yolo_to_xyxy(defect, width, height)
                            if box_inside(defect_box, crop):
                                remapped = remap_to_crop(defect[0], defect_box, crop)
                                if remapped:
                                    crop_rows.append(remapped)
                        if args.positive_only and not crop_rows:
                            continue

                        suffix = "" if variant == 0 else f"_jitter{variant:02d}"
                        stem = f"{image_path.stem}_crop{idx:02d}{suffix}"
                        image.crop(crop).save(output / "images" / split / f"{stem}.jpg", quality=95)
                        label_text = "\n".join(crop_rows)
                        if label_text:
                            label_text += "\n"
                        (output / "labels" / split / f"{stem}.txt").write_text(
                            label_text, encoding="utf-8"
                        )

                        stats[split]["crops"] += 1
                        stats[split]["boxes"] += len(crop_rows)
                        stats[split]["jittered"] += int(is_jittered)
                        key = "positive" if crop_rows else "negative"
                        stats[split][key] += 1

    (output / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {output}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                "nc: 3",
                "names:",
                *[f"  {idx}: {name}" for idx, name in enumerate(LOCAL_NAMES)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    efficient_count = write_efficient_view(output, args.efficient_positive_jitters)
    stats["efficient_train"] = {"images": efficient_count}
    write_manifest(output, source, args, stats)
    total = sum(stats[split]["crops"] for split in ["train", "val", "test"])
    print(f"wrote {total} crop images to {output}")
    print(f"efficient train view: images={efficient_count}")
    for split in ["train", "val", "test"]:
        split_stats = stats[split]
        print(
            f"{split}: crops={split_stats['crops']} positive={split_stats['positive']} "
            f"negative={split_stats['negative']} boxes={split_stats['boxes']} "
            f"jittered={split_stats['jittered']}"
        )


if __name__ == "__main__":
    main()
