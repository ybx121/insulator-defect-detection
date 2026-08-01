#!/usr/bin/env python3
"""Evaluate the coarse-to-local detector on original full-resolution images."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gf_insuyolo.boxes import Detection, box_iou  # noqa: E402
from infer import DEFAULT_NAMES, iter_sources, register_custom_modules, run_two_stage  # noqa: E402


IOU_THRESHOLDS = np.arange(0.5, 0.96, 0.05)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-weights", required=True)
    parser.add_argument("--local-weights", required=True)
    parser.add_argument("--data", type=Path, default=Path("datasets/unified_fine/data.yaml"))
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--global-imgsz", type=int, default=960)
    parser.add_argument("--local-imgsz", type=int, default=640)
    parser.add_argument("--global-conf", type=float, default=0.15)
    parser.add_argument(
        "--local-conf",
        type=float,
        default=0.001,
        help="Low threshold used to build local precision-recall curves",
    )
    parser.add_argument(
        "--operating-local-conf",
        type=float,
        default=0.25,
        help="Local threshold used for reported operating-point P/R and false-positive rate",
    )
    parser.add_argument("--iou", type=float, default=0.55, help="Class-wise NMS IoU")
    parser.add_argument("--crop-margin", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--output", type=Path, default=Path("runs/eval/two_stage_test.json"))
    return parser.parse_args()


def resolve_dataset(data_path: Path, split: str) -> tuple[Path, Path, list[str]]:
    data_path = data_path.resolve()
    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root = Path(config.get("path", data_path.parent))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    images = Path(config[split])
    if not images.is_absolute():
        images = root / images
    images = images.resolve()
    parts = list(images.parts)
    try:
        image_part = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as exc:
        raise ValueError(f"Expected an 'images' directory in {images}") from exc
    parts[image_part] = "labels"
    labels = Path(*parts)
    raw_names = config["names"]
    if isinstance(raw_names, dict):
        names = [str(raw_names[index] if index in raw_names else raw_names[str(index)]) for index in range(len(raw_names))]
    else:
        names = [str(name) for name in raw_names]
    if names != DEFAULT_NAMES:
        raise ValueError(f"Expected fine classes {DEFAULT_NAMES}, got {names}")
    return images, labels, names


def read_ground_truth(label_path: Path, image_path: Path) -> list[Detection]:
    with Image.open(image_path) as image:
        width, height = image.size
    detections: list[Detection] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        class_id, x, y, box_w, box_h = raw.split()
        cls = int(float(class_id))
        cx, cy = float(x) * width, float(y) * height
        pixel_w, pixel_h = float(box_w) * width, float(box_h) * height
        detections.append(
            Detection(
                cls=cls,
                class_name=DEFAULT_NAMES[cls],
                confidence=1.0,
                xyxy=(
                    cx - pixel_w / 2,
                    cy - pixel_h / 2,
                    cx + pixel_w / 2,
                    cy + pixel_h / 2,
                ),
                source_stage="ground_truth",
            )
        )
    return detections


def interpolated_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    if not len(recall):
        return 0.0
    recall_points = np.linspace(0.0, 1.0, 101)
    values = [precision[recall >= point].max() if np.any(recall >= point) else 0.0 for point in recall_points]
    return float(np.mean(values))


def class_metrics(
    class_id: int,
    ground_truth: dict[str, list[Detection]],
    predictions: dict[str, list[Detection]],
) -> dict[str, float | int]:
    gt_by_image = {
        image_id: [det for det in detections if det.cls == class_id]
        for image_id, detections in ground_truth.items()
    }
    prediction_rows = sorted(
        (
            (image_id, det)
            for image_id, detections in predictions.items()
            for det in detections
            if det.cls == class_id
        ),
        key=lambda item: item[1].confidence,
        reverse=True,
    )
    target_count = sum(len(rows) for rows in gt_by_image.values())
    ap_values: list[float] = []
    operating_precision = 0.0
    operating_recall = 0.0
    for threshold_index, threshold in enumerate(IOU_THRESHOLDS):
        matched: dict[str, set[int]] = defaultdict(set)
        true_positive = np.zeros(len(prediction_rows), dtype=float)
        false_positive = np.zeros(len(prediction_rows), dtype=float)
        for prediction_index, (image_id, prediction) in enumerate(prediction_rows):
            candidates = gt_by_image.get(image_id, [])
            best_iou, best_index = 0.0, -1
            for candidate_index, candidate in enumerate(candidates):
                if candidate_index in matched[image_id]:
                    continue
                overlap = box_iou(prediction.xyxy, candidate.xyxy)
                if overlap > best_iou:
                    best_iou, best_index = overlap, candidate_index
            if best_index >= 0 and best_iou >= threshold:
                matched[image_id].add(best_index)
                true_positive[prediction_index] = 1.0
            else:
                false_positive[prediction_index] = 1.0

        cumulative_tp = np.cumsum(true_positive)
        cumulative_fp = np.cumsum(false_positive)
        recall = cumulative_tp / max(target_count, 1)
        precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
        ap_values.append(interpolated_ap(recall, precision))
        if threshold_index == 0 and len(prediction_rows):
            operating_precision = float(precision[-1])
            operating_recall = float(recall[-1])

    return {
        "targets": target_count,
        "predictions": len(prediction_rows),
        "precision_at_conf": operating_precision,
        "recall_at_conf": operating_recall,
        "ap50": ap_values[0],
        "ap50_95": float(np.mean(ap_values)),
    }


def main() -> None:
    args = parse_args()
    if args.local_conf > args.operating_local_conf:
        raise ValueError("--local-conf must not exceed --operating-local-conf")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is required for two-stage evaluation") from exc

    image_root, label_root, names = resolve_dataset(args.data, args.split)
    image_paths = iter_sources(image_root)
    if not image_paths:
        raise SystemExit(f"No images found in {image_root}")

    register_custom_modules()
    global_model = YOLO(args.global_weights)
    local_model = YOLO(args.local_weights)
    inference_args = argparse.Namespace(
        global_imgsz=args.global_imgsz,
        local_imgsz=args.local_imgsz,
        global_conf=args.global_conf,
        local_conf=args.local_conf,
        iou=args.iou,
        crop_margin=args.crop_margin,
        device=args.device,
    )

    ground_truth: dict[str, list[Detection]] = {}
    all_predictions: dict[str, list[Detection]] = {}
    operating_predictions: dict[str, list[Detection]] = {}
    proposal_count = 0
    normal_images = 0
    normal_images_with_false_positive = 0
    started = time.perf_counter()
    for index, image_path in enumerate(image_paths, start=1):
        relative = image_path.relative_to(image_root)
        image_id = relative.as_posix()
        label_path = (label_root / relative).with_suffix(".txt")
        targets = read_ground_truth(label_path, image_path)
        predictions = run_two_stage(global_model, local_model, image_path, inference_args)
        filtered = [
            det
            for det in predictions
            if det.source_stage != "local_crop" or det.confidence >= args.operating_local_conf
        ]
        ground_truth[image_id] = targets
        all_predictions[image_id] = predictions
        operating_predictions[image_id] = filtered
        proposal_count += sum(det.cls == 0 for det in predictions)
        if not any(det.cls > 0 for det in targets):
            normal_images += 1
            if any(det.cls > 0 for det in filtered):
                normal_images_with_false_positive += 1
        if index % 25 == 0 or index == len(image_paths):
            print(f"evaluated {index}/{len(image_paths)} images", flush=True)
    elapsed = time.perf_counter() - started

    per_class = {
        names[class_id]: class_metrics(class_id, ground_truth, operating_predictions)
        for class_id in range(len(names))
    }
    # AP requires the low-threshold local predictions, while P/R above is reported
    # at the deployment operating threshold.
    for class_id, name in enumerate(names):
        ap_metrics = class_metrics(class_id, ground_truth, all_predictions)
        per_class[name]["ap50"] = ap_metrics["ap50"]
        per_class[name]["ap50_95"] = ap_metrics["ap50_95"]

    defect_rows = [per_class[name] for name in names[1:]]
    report = {
        "dataset": str(args.data.resolve()),
        "split": args.split,
        "images": len(image_paths),
        "settings": {
            "global_weights": str(Path(args.global_weights).resolve()),
            "local_weights": str(Path(args.local_weights).resolve()),
            "global_imgsz": args.global_imgsz,
            "local_imgsz": args.local_imgsz,
            "global_conf": args.global_conf,
            "local_ap_conf": args.local_conf,
            "operating_local_conf": args.operating_local_conf,
            "nms_iou": args.iou,
            "crop_margin": args.crop_margin,
        },
        "defect_macro": {
            "mAP50": float(np.mean([row["ap50"] for row in defect_rows])),
            "mAP50_95": float(np.mean([row["ap50_95"] for row in defect_rows])),
            "precision_at_conf": float(np.mean([row["precision_at_conf"] for row in defect_rows])),
            "recall_at_conf": float(np.mean([row["recall_at_conf"] for row in defect_rows])),
        },
        "normal_false_positive": {
            "normal_images": normal_images,
            "images_with_false_positive": normal_images_with_false_positive,
            "rate": normal_images_with_false_positive / max(normal_images, 1),
        },
        "performance": {
            "seconds": elapsed,
            "milliseconds_per_image": elapsed * 1000 / len(image_paths),
            "average_crops_per_image": proposal_count / len(image_paths),
        },
        "per_class": per_class,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["defect_macro"], indent=2))
    print(f"wrote report to {args.output}")


if __name__ == "__main__":
    main()
