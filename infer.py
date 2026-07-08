#!/usr/bin/env python3
"""Run single-stage or global-local GF-InsuYOLO inference."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from PIL import Image

from gf_insuyolo.boxes import Detection, expand_box, nms, remap_crop_box


DEFAULT_NAMES = ["insulator_string", "broken_shell", "flashover_pollution", "missing_disc_drop"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Global detector weights")
    parser.add_argument("--local-weights", help="Optional crop detector weights")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/infer/predictions.json"))
    parser.add_argument("--two-stage", action="store_true")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--crop-margin", type=float, default=0.15)
    return parser.parse_args()


def iter_sources(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def result_to_detections(result, names: list[str], source_stage: str) -> list[Detection]:
    detections: list[Detection] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections
    for box in boxes:
        cls = int(box.cls.item())
        conf = float(box.conf.item())
        xyxy = tuple(float(value) for value in box.xyxy[0].tolist())
        detections.append(
            Detection(
                cls=cls,
                class_name=names[cls] if cls < len(names) else str(cls),
                confidence=conf,
                xyxy=xyxy,  # type: ignore[arg-type]
                source_stage=source_stage,
            )
        )
    return detections


def predict_one(model, image_path: Path, names: list[str], args: argparse.Namespace, stage: str) -> list[Detection]:
    results = model.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        verbose=False,
    )
    return result_to_detections(results[0], names, stage)


def run_two_stage(global_model, local_model, image_path: Path, args: argparse.Namespace) -> list[Detection]:
    global_dets = predict_one(global_model, image_path, DEFAULT_NAMES, args, "global")
    insulators = [det for det in global_dets if det.cls == 0]
    defects = [det for det in global_dets if det.cls > 0]

    with Image.open(image_path) as image:
        width, height = image.size
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for idx, insulator in enumerate(insulators):
                crop = expand_box(insulator.xyxy, width, height, args.crop_margin)
                if crop[2] <= crop[0] or crop[3] <= crop[1]:
                    continue
                crop_path = tmp_path / f"{image_path.stem}_crop{idx:02d}.jpg"
                image.crop(crop).save(crop_path, quality=95)
                local_results = local_model.predict(
                    source=str(crop_path),
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    verbose=False,
                )
                local_dets = result_to_detections(
                    local_results[0],
                    ["broken_shell", "flashover_pollution", "missing_disc_drop"],
                    "local_crop",
                )
                for local in local_dets:
                    remapped = remap_crop_box(local.xyxy, (crop[0], crop[1]))
                    defects.append(
                        Detection(
                            cls=local.cls + 1,
                            class_name=DEFAULT_NAMES[local.cls + 1],
                            confidence=local.confidence,
                            xyxy=remapped,
                            source_stage="local_crop",
                        )
                    )
    return nms(insulators + defects, args.iou)


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Install the inference environment with "
            "`pip install -r requirements.txt` in Python 3.11."
        ) from exc

    source_paths = iter_sources(Path(args.source))
    if not source_paths:
        raise SystemExit(f"No images found in {args.source}")

    global_model = YOLO(args.weights)
    local_model = YOLO(args.local_weights or args.weights)
    output_rows = []
    for image_path in source_paths:
        if args.two_stage:
            detections = run_two_stage(global_model, local_model, image_path, args)
        else:
            detections = nms(predict_one(global_model, image_path, DEFAULT_NAMES, args, "global"), args.iou)
        insulators = [det.as_json() for det in detections if det.cls == 0]
        defects = [det.as_json() for det in detections if det.cls > 0]
        output_rows.append(
            {
                "image": str(image_path),
                "has_defect": bool(defects),
                "insulator_boxes": insulators,
                "defect_boxes": defects,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(output_rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
