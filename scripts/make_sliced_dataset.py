#!/usr/bin/env python3
"""Create mixed full-frame and tiled training data for small defects."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


NAMES = ["insulator_string", "broken_shell", "flashover_pollution", "missing_disc_drop"]


@dataclass(frozen=True)
class Box:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("datasets/credible_fine_v1"))
    parser.add_argument("--output", type=Path, default=Path("datasets/credible_fine_v1_sliced"))
    parser.add_argument("--tile-size", type=int, default=768)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--min-visibility", type=float, default=0.6)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_boxes(path: Path, width: int, height: int) -> list[Box]:
    boxes: list[Box] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        class_id, x, y, box_w, box_h = raw.split()
        cx, cy = float(x) * width, float(y) * height
        pixel_w, pixel_h = float(box_w) * width, float(box_h) * height
        boxes.append(
            Box(
                int(float(class_id)),
                cx - pixel_w / 2,
                cy - pixel_h / 2,
                cx + pixel_w / 2,
                cy + pixel_h / 2,
            )
        )
    return boxes


def tile_origins(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, int(round(tile_size * (1 - overlap))))
    values = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if values[-1] != last:
        values.append(last)
    return values


def remap_box(
    box: Box,
    tile: tuple[int, int, int, int],
    min_visibility: float,
) -> tuple[int, float, float, float, float] | None:
    tx1, ty1, tx2, ty2 = tile
    center_x, center_y = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
    if not (tx1 <= center_x < tx2 and ty1 <= center_y < ty2):
        return None
    ix1, iy1 = max(box.x1, tx1), max(box.y1, ty1)
    ix2, iy2 = min(box.x2, tx2), min(box.y2, ty2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    original = max(1e-9, (box.x2 - box.x1) * (box.y2 - box.y1))
    if intersection / original < min_visibility:
        return None
    width, height = tx2 - tx1, ty2 - ty1
    return (
        box.class_id,
        ((ix1 + ix2) / 2 - tx1) / width,
        ((iy1 + iy2) / 2 - ty1) / height,
        (ix2 - ix1) / width,
        (iy2 - iy1) / height,
    )


def write_tile(
    image: Image.Image,
    rows: list[tuple[int, float, float, float, float]],
    tile: tuple[int, int, int, int],
    image_path: Path,
    label_path: Path,
) -> None:
    image.crop(tile).save(image_path, quality=95)
    text = "\n".join(
        f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        for class_id, x, y, w, h in rows
    )
    label_path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.tile_size <= 0 or not 0 <= args.overlap < 1:
        raise ValueError("Invalid tile geometry")
    if not 0 <= args.min_visibility <= 1 or args.negative_ratio < 0:
        raise ValueError("Invalid filtering settings")
    source, output = args.input.resolve(), args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        shutil.rmtree(output)
    image_output, label_output = output / "images" / "train", output / "labels" / "train"
    image_output.mkdir(parents=True)
    label_output.mkdir(parents=True)

    rng = random.Random(args.seed)
    positives: list[tuple[Path, Path, tuple[int, int, int, int], list[tuple[int, float, float, float, float]]]] = []
    negatives: list[tuple[Path, Path, tuple[int, int, int, int], list[tuple[int, float, float, float, float]]]] = []
    manifest_rows: list[dict[str, object]] = []
    for source_image in sorted((source / "images" / "train").iterdir()):
        if not source_image.is_file():
            continue
        label_path = source / "labels" / "train" / f"{source_image.stem}.txt"
        with Image.open(source_image) as image:
            width, height = image.size
        boxes = read_boxes(label_path, width, height)
        for y1 in tile_origins(height, args.tile_size, args.overlap):
            for x1 in tile_origins(width, args.tile_size, args.overlap):
                tile = (x1, y1, min(width, x1 + args.tile_size), min(height, y1 + args.tile_size))
                rows = [row for box in boxes if (row := remap_box(box, tile, args.min_visibility))]
                item = (source_image, label_path, tile, rows)
                if any(row[0] > 0 for row in rows):
                    positives.append(item)
                elif any(row[0] == 0 for row in rows):
                    negatives.append(item)

    rng.shuffle(negatives)
    selected = positives + negatives[: math.ceil(len(positives) * args.negative_ratio)]
    selected.sort(key=lambda item: (item[0].name, item[2]))
    for index, (source_image, _, tile, rows) in enumerate(selected):
        x1, y1, x2, y2 = tile
        stem = f"tile_{index:06d}_{source_image.stem}_{x1}_{y1}"
        with Image.open(source_image) as image:
            write_tile(
                image.convert("RGB"), rows, tile,
                image_output / f"{stem}.jpg", label_output / f"{stem}.txt"
            )
        manifest_rows.append(
            {
                "kind": "tile",
                "image": f"images/train/{stem}.jpg",
                "source_image": str(source_image),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "defect_objects": sum(row[0] > 0 for row in rows),
            }
        )

    # Full frames retain long-range context and the insulator-string class.
    for source_image in sorted((source / "images" / "train").iterdir()):
        if not source_image.is_file():
            continue
        stem = f"full_{source_image.stem}"
        shutil.copy2(source_image, image_output / f"{stem}{source_image.suffix.lower()}")
        shutil.copy2(source / "labels" / "train" / f"{source_image.stem}.txt", label_output / f"{stem}.txt")
        manifest_rows.append(
            {"kind": "full", "image": f"images/train/{stem}{source_image.suffix.lower()}", "source_image": str(source_image), "x1": 0, "y1": 0, "x2": "", "y2": "", "defect_objects": ""}
        )

    with (output / "slice_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    settings = vars(args).copy()
    settings.update({"positive_tiles": len(positives), "selected_negative_tiles": len(selected) - len(positives), "full_frames": len(list((source / 'images' / 'train').iterdir()))})
    (output / "slice_settings.json").write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")
    (output / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {output}",
                "train: images/train",
                f"val: {(source / 'images' / 'val').resolve()}",
                f"test: {(source / 'images' / 'test').resolve()}",
                "",
                "nc: 4",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(NAMES)],
                "",
            ]
        ), encoding="utf-8"
    )
    print(f"wrote {len(selected)} tiles and {settings['full_frames']} full frames to {output}")


if __name__ == "__main__":
    main()
