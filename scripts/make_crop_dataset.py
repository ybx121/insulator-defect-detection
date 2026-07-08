#!/usr/bin/env python3
"""Create a local crop dataset from insulator-string boxes."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
LOCAL_NAMES = ["broken_shell", "flashover_pollution", "missing_disc_drop"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("datasets/unified_fine"))
    parser.add_argument("--output", type=Path, default=Path("datasets/unified_fine_crops"))
    parser.add_argument("--margin", type=float, default=0.15)
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


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite")
        shutil.rmtree(output)
    for split in ["train", "val", "test"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    written = 0
    for split in ["train", "val", "test"]:
        for image_path in sorted((source / "images" / split).iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            label_path = source / "labels" / split / f"{image_path.stem}.txt"
            rows = read_labels(label_path)
            insulators = [row for row in rows if row[0] == 0]
            defects = [row for row in rows if row[0] > 0]
            if not defects:
                continue
            with Image.open(image_path) as image:
                width, height = image.size
                for idx, insulator in enumerate(insulators):
                    box = yolo_to_xyxy(insulator, width, height)
                    x1, y1, x2, y2 = box
                    dx = int(round((x2 - x1) * args.margin))
                    dy = int(round((y2 - y1) * args.margin))
                    crop = clip_box((x1 - dx, y1 - dy, x2 + dx, y2 + dy), width, height)
                    crop_rows: list[str] = []
                    for defect in defects:
                        defect_box = yolo_to_xyxy(defect, width, height)
                        if box_inside(defect_box, crop):
                            remapped = remap_to_crop(defect[0], defect_box, crop)
                            if remapped:
                                crop_rows.append(remapped)
                    if not crop_rows:
                        continue
                    crop_image = image.crop(crop)
                    stem = f"{image_path.stem}_crop{idx:02d}"
                    crop_image.save(output / "images" / split / f"{stem}.jpg", quality=95)
                    (output / "labels" / split / f"{stem}.txt").write_text(
                        "\n".join(crop_rows) + "\n", encoding="utf-8"
                    )
                    written += 1

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
    print(f"wrote {written} crop images to {output}")


if __name__ == "__main__":
    main()
