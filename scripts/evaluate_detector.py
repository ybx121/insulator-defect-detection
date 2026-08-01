#!/usr/bin/env python3
"""Evaluate standard, sliced, hybrid, or two-stage insulator detection."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NAMES = ["insulator_string", "broken_shell", "flashover_pollution", "missing_disc_drop"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class Box:
    class_id: int
    confidence: float
    xyxy: tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights")
    parser.add_argument("--local-weights")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--mode",
        choices=[
            "standard",
            "ensemble",
            "ensemble-two-stage",
            "sahi",
            "hybrid",
            "two-stage",
            "external",
        ],
        default="standard",
    )
    parser.add_argument("--ensemble-weights", nargs="*", default=[])
    parser.add_argument(
        "--external-predictions",
        type=Path,
        nargs="+",
        help="Model-independent prediction JSON to evaluate alone or fuse with the selected mode",
    )
    parser.add_argument(
        "--ensemble-class-offsets",
        type=int,
        nargs="*",
        default=[],
        help="Class-ID offset for each --ensemble-weights model (for projected experts)",
    )
    parser.add_argument(
        "--two-stage-fusion",
        choices=["union", "local-preferred", "local-only"],
        default="union",
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=1, help="Standard inference batch size")
    parser.add_argument("--local-imgsz", type=int, default=640)
    parser.add_argument("--slice-size", type=int, default=768)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--operating-conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--fusion-iou", type=float, default=0.55)
    parser.add_argument("--device", default="0")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--limit", type=int, help="Optional image limit for smoke tests")
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--verify-native", action="store_true")
    parser.add_argument("--tta", action="store_true", help="Use Ultralytics test-time augmentation")
    parser.add_argument("--output", type=Path, default=Path("runs/eval/evaluation.json"))
    parser.add_argument("--leaderboard", type=Path, default=Path("runs/eval/leaderboard.csv"))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    ensemble_modes = {"ensemble", "ensemble-two-stage"}
    if args.mode != "external" and not args.weights:
        raise ValueError(f"--weights is required for --mode {args.mode}")
    if args.mode == "external" and not args.external_predictions:
        raise ValueError("--external-predictions is required for --mode external")
    if args.ensemble_weights and args.mode not in ensemble_modes:
        raise ValueError(
            f"--ensemble-weights requires --mode ensemble or ensemble-two-stage, got {args.mode!r}"
        )
    if args.ensemble_class_offsets and args.mode not in ensemble_modes:
        raise ValueError(
            "--ensemble-class-offsets requires --mode ensemble or ensemble-two-stage"
        )


def read_external_predictions(
    path: Path,
    image_root: Path,
    expected_fingerprint: str,
    expected_split: str,
) -> dict[Path, list[Box]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset_fingerprint") != expected_fingerprint:
        raise ValueError(
            "External prediction fingerprint does not match the evaluation dataset: "
            f"{payload.get('dataset_fingerprint')} != {expected_fingerprint}"
        )
    if payload.get("split") != expected_split:
        raise ValueError(
            f"External prediction split {payload.get('split')!r} != {expected_split!r}"
        )
    output: dict[Path, list[Box]] = {}
    for relative, rows in payload["predictions"].items():
        output[(image_root / relative).resolve()] = [
            Box(
                class_id=int(row["class_id"]),
                confidence=float(row["confidence"]),
                xyxy=tuple(float(value) for value in row["xyxy"]),
            )
            for row in rows
        ]
    return output


def fuse_external_predictions(
    predictions: dict[Path, list[Box]],
    external: dict[Path, list[Box]],
    images: list[Path],
    iou_threshold: float,
) -> dict[Path, list[Box]]:
    fused: dict[Path, list[Box]] = {}
    for path in images:
        resolved = path.resolve()
        with Image.open(path) as image:
            width, height = image.size
        fused[resolved] = weighted_box_fusion(
            predictions.get(resolved, []) + external.get(resolved, []),
            iou_threshold,
            width,
            height,
        )
    return fused


def fuse_prediction_sets(
    prediction_sets: list[dict[Path, list[Box]]],
    images: list[Path],
    iou_threshold: float,
) -> dict[Path, list[Box]]:
    if not prediction_sets:
        return {path.resolve(): [] for path in images}
    if len(prediction_sets) == 1:
        return prediction_sets[0]
    fused: dict[Path, list[Box]] = {}
    for path in images:
        resolved = path.resolve()
        with Image.open(path) as image:
            width, height = image.size
        boxes = [
            box
            for predictions in prediction_sets
            for box in predictions.get(resolved, [])
        ]
        fused[resolved] = weighted_box_fusion(
            boxes, iou_threshold, width, height
        )
    return fused


def resolve_dataset(data_path: Path, split: str) -> tuple[Path, Path, dict[str, object]]:
    data_path = data_path.resolve()
    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root = Path(config.get("path", data_path.parent))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    image_root = Path(config[split])
    if not image_root.is_absolute():
        image_root = root / image_root
    image_root = image_root.resolve()
    parts = list(image_root.parts)
    try:
        image_index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as exc:
        raise ValueError(f"Expected an images directory in {image_root}") from exc
    parts[image_index] = "labels"
    label_root = Path(*parts)
    raw_names = config["names"]
    names = [raw_names.get(index, raw_names.get(str(index))) for index in range(len(raw_names))] if isinstance(raw_names, dict) else raw_names
    if list(names) != NAMES:
        raise ValueError(f"Expected {NAMES}, got {names}")
    fingerprint_path = root / "metadata" / "dataset_fingerprint.json"
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8")) if fingerprint_path.exists() else {"fingerprint": "unversioned"}
    return image_root, label_root, fingerprint


def collect_images(image_root: Path) -> list[Path]:
    return sorted(path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_EXTS)


def image_id(image: Path, image_root: Path) -> str:
    return image.relative_to(image_root).as_posix()


def read_ground_truth(images: list[Path], image_root: Path, label_root: Path) -> dict[str, list[Box]]:
    ground_truth: dict[str, list[Box]] = {}
    for image_path in images:
        relative = image_path.relative_to(image_root)
        label_path = (label_root / relative).with_suffix(".txt")
        with Image.open(image_path) as image:
            width, height = image.size
        boxes: list[Box] = []
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            class_id, x, y, box_w, box_h = raw.split()
            cx, cy = float(x) * width, float(y) * height
            pixel_w, pixel_h = float(box_w) * width, float(box_h) * height
            boxes.append(Box(int(float(class_id)), 1.0, (cx - pixel_w / 2, cy - pixel_h / 2, cx + pixel_w / 2, cy + pixel_h / 2)))
        ground_truth[relative.as_posix()] = boxes
    return ground_truth


def box_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-9)


def weighted_box_fusion(boxes: list[Box], iou_threshold: float, width: int, height: int) -> list[Box]:
    fused: list[Box] = []
    for class_id in range(len(NAMES)):
        candidates = sorted((box for box in boxes if box.class_id == class_id), key=lambda box: box.confidence, reverse=True)
        cluster_weights: list[float] = []
        cluster_weighted_coords: list[np.ndarray] = []
        cluster_confidences: list[float] = []
        for candidate in candidates:
            candidate_coords = np.asarray(candidate.xyxy, dtype=np.float64)
            if cluster_weights:
                cluster_coords = np.vstack(cluster_weighted_coords) / np.asarray(
                    cluster_weights, dtype=np.float64
                )[:, None]
                intersections_min = np.maximum(cluster_coords[:, :2], candidate_coords[:2])
                intersections_max = np.minimum(cluster_coords[:, 2:], candidate_coords[2:])
                intersection_sizes = np.maximum(0.0, intersections_max - intersections_min)
                intersections = intersection_sizes[:, 0] * intersection_sizes[:, 1]
                cluster_areas = np.maximum(
                    0.0, cluster_coords[:, 2] - cluster_coords[:, 0]
                ) * np.maximum(0.0, cluster_coords[:, 3] - cluster_coords[:, 1])
                candidate_area = max(
                    0.0, candidate_coords[2] - candidate_coords[0]
                ) * max(0.0, candidate_coords[3] - candidate_coords[1])
                ious = intersections / np.maximum(
                    cluster_areas + candidate_area - intersections, 1e-9
                )
                best = int(np.argmax(ious))
            else:
                ious = np.empty(0, dtype=np.float64)
                best = -1
            candidate_weight = max(candidate.confidence, 1e-6)
            if best >= 0 and ious[best] >= iou_threshold:
                cluster_weights[best] += candidate_weight
                cluster_weighted_coords[best] += candidate_coords * candidate_weight
                cluster_confidences[best] = max(
                    cluster_confidences[best], candidate.confidence
                )
            else:
                cluster_weights.append(candidate_weight)
                cluster_weighted_coords.append(candidate_coords * candidate_weight)
                cluster_confidences.append(candidate.confidence)
        for weight, weighted_coords, confidence in zip(
            cluster_weights, cluster_weighted_coords, cluster_confidences
        ):
            coords = weighted_coords / weight
            fused.append(
                Box(
                    class_id,
                    confidence,
                    (max(0.0, coords[0]), max(0.0, coords[1]), min(float(width), coords[2]), min(float(height), coords[3])),
                )
            )
    return fused


def _fused_xyxy(cluster: list[Box]) -> tuple[float, float, float, float]:
    weights = np.array([max(box.confidence, 1e-6) for box in cluster])
    coords = np.array([box.xyxy for box in cluster])
    values = np.average(coords, axis=0, weights=weights)
    return tuple(float(value) for value in values)


def predict_standard(
    model: object,
    images: list[Path],
    args: argparse.Namespace,
    class_offset: int = 0,
) -> dict[Path, list[Box]]:
    predictions: dict[Path, list[Box]] = {}
    for start in range(0, len(images), args.batch):
        chunk = images[start:start + args.batch]
        results = model.predict(
            source=[str(path) for path in chunk],
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
            augment=getattr(args, "tta", False),
            rect=False,
        )
        for source_path, result in zip(chunk, results):
            path = source_path.resolve()
            predictions[path] = [
                Box(int(class_id) + class_offset, float(confidence), tuple(float(value) for value in coords))
                for coords, confidence, class_id in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                    result.boxes.cls.cpu().tolist(),
                )
            ]
    return predictions


def predict_sahi(images: list[Path], args: argparse.Namespace) -> dict[Path, list[Box]]:
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
    except ImportError as exc:
        raise SystemExit("Install SAHI with `pip install -r requirements.txt`") from exc
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=args.weights,
        confidence_threshold=args.conf, device=args.device, image_size=args.imgsz,
    )
    predictions: dict[Path, list[Box]] = {}
    for index, path in enumerate(images, 1):
        result = get_sliced_prediction(
            str(path), detection_model,
            slice_height=args.slice_size, slice_width=args.slice_size,
            overlap_height_ratio=args.overlap, overlap_width_ratio=args.overlap,
            postprocess_type="GREEDYNMM", postprocess_match_threshold=args.iou,
            verbose=0,
        )
        predictions[path.resolve()] = [
            Box(int(obj.category.id), float(obj.score.value), tuple(float(v) for v in obj.bbox.to_xyxy()))
            for obj in result.object_prediction_list
        ]
        if index % 25 == 0 or index == len(images):
            print(f"SAHI evaluated {index}/{len(images)}", flush=True)
    return predictions


def predict_two_stage(images: list[Path], args: argparse.Namespace) -> dict[Path, list[Box]]:
    if not args.local_weights:
        raise ValueError("--local-weights is required for two-stage mode")
    from ultralytics import YOLO
    from infer import register_custom_modules, run_two_stage

    register_custom_modules()
    global_model, local_model = YOLO(args.weights), YOLO(args.local_weights)
    inference_args = argparse.Namespace(
        global_imgsz=args.imgsz, local_imgsz=args.local_imgsz,
        global_conf=args.conf, local_conf=args.conf, iou=args.iou,
        crop_margin=0.15, device=args.device, two_stage_fusion=args.two_stage_fusion,
    )
    predictions = {}
    for index, path in enumerate(images, 1):
        rows = run_two_stage(global_model, local_model, path, inference_args)
        predictions[path.resolve()] = [Box(row.cls, row.confidence, row.xyxy) for row in rows]
        if index % 25 == 0 or index == len(images):
            print(f"two-stage evaluated {index}/{len(images)}", flush=True)
    return predictions


def predict_ensemble(images: list[Path], args: argparse.Namespace) -> dict[Path, list[Box]]:
    if not args.ensemble_weights:
        raise ValueError("--ensemble-weights is required for ensemble mode")
    from ultralytics import YOLO

    offsets = getattr(args, "ensemble_class_offsets", [])
    if not offsets:
        offsets = [0] * len(args.ensemble_weights)
    if len(offsets) != len(args.ensemble_weights):
        raise ValueError(
            "--ensemble-class-offsets must provide one value per --ensemble-weights model"
        )
    all_predictions = [predict_standard(YOLO(args.weights), images, args)]
    all_predictions.extend(
        predict_standard(YOLO(weights), images, args, class_offset=offset)
        for weights, offset in zip(args.ensemble_weights, offsets)
    )
    predictions: dict[Path, list[Box]] = {}
    for path in images:
        with Image.open(path) as image:
            width, height = image.size
        combined = [box for rows in all_predictions for box in rows[path.resolve()]]
        predictions[path.resolve()] = weighted_box_fusion(
            combined, args.fusion_iou, width, height
        )
    return predictions


def extend_ensemble_with_local(
    images: list[Path],
    global_predictions: dict[Path, list[Box]],
    args: argparse.Namespace,
) -> dict[Path, list[Box]]:
    if not args.local_weights:
        raise ValueError("--local-weights is required for ensemble-two-stage mode")
    from ultralytics import YOLO
    from gf_insuyolo.boxes import expand_box, remap_crop_box
    from infer import model_names, predict_one

    local_model = YOLO(args.local_weights)
    local_names = model_names(local_model)
    predictions: dict[Path, list[Box]] = {}
    for index, path in enumerate(images, 1):
        global_rows = global_predictions[path.resolve()]
        local_rows: list[Box] = []
        with Image.open(path) as source:
            image = source.convert("RGB")
            width, height = image.size
            for insulator in (box for box in global_rows if box.class_id == 0):
                crop = expand_box(insulator.xyxy, width, height, 0.15)
                if crop[2] <= crop[0] or crop[3] <= crop[1]:
                    continue
                detections = predict_one(
                    local_model,
                    image.crop(crop),
                    local_names,
                    "local_crop",
                    args.local_imgsz,
                    args.conf,
                    args.iou,
                    args.device,
                )
                local_rows.extend(
                    Box(row.cls + 1, row.confidence, remap_crop_box(row.xyxy, crop[:2]))
                    for row in detections
                )
        predictions[path.resolve()] = weighted_box_fusion(
            global_rows + local_rows, args.fusion_iou, width, height
        )
        if index % 25 == 0 or index == len(images):
            print(f"ensemble-two-stage evaluated {index}/{len(images)}", flush=True)
    return predictions


def to_image_keyed(predictions: dict[Path, list[Box]], image_root: Path) -> dict[str, list[Box]]:
    return {path.relative_to(image_root).as_posix(): boxes for path, boxes in predictions.items()}


def coco_payload(ground_truth: dict[str, list[Box]], predictions: dict[str, list[Box]]) -> tuple[dict[str, object], list[dict[str, object]], dict[str, int]]:
    image_ids = {name: index + 1 for index, name in enumerate(sorted(ground_truth))}
    annotations, prediction_rows = [], []
    annotation_id = 1
    for name, boxes in ground_truth.items():
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy
            annotations.append({"id": annotation_id, "image_id": image_ids[name], "category_id": box.class_id + 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
            annotation_id += 1
    for name, boxes in predictions.items():
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy
            prediction_rows.append({"image_id": image_ids[name], "category_id": box.class_id + 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": box.confidence})
    dataset = {
        "info": {"description": "insulator credible evaluation"},
        "images": [{"id": value, "file_name": key} for key, value in image_ids.items()],
        "annotations": annotations,
        "categories": [{"id": index + 1, "name": name} for index, name in enumerate(NAMES)],
    }
    return dataset, prediction_rows, image_ids


def run_coco_eval(dataset: dict[str, object], predictions: list[dict[str, object]], category_id: int | None = None) -> list[float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = dataset
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(predictions) if predictions else coco_gt.loadRes([])
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        if category_id is not None:
            evaluator.params.catIds = [category_id]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return [float(value) for value in evaluator.stats]


def bootstrap_ci(
    dataset: dict[str, object],
    predictions: list[dict[str, object]],
    iterations: int,
    seed: int,
    metric_index: int = 0,
) -> dict[str, float] | None:
    if iterations <= 0:
        return None
    rng = random.Random(seed)
    source_ids = [row["id"] for row in dataset["images"]]
    gt_by_image: dict[int, list[dict[str, object]]] = {image_id: [] for image_id in source_ids}
    pred_by_image: dict[int, list[dict[str, object]]] = {image_id: [] for image_id in source_ids}
    for row in dataset["annotations"]:
        gt_by_image[row["image_id"]].append(row)
    for row in predictions:
        pred_by_image[row["image_id"]].append(row)
    values = []
    for _ in range(iterations):
        sampled = rng.choices(source_ids, k=len(source_ids))
        images, annotations, prediction_rows = [], [], []
        annotation_id = 1
        for new_id, old_id in enumerate(sampled, 1):
            images.append({"id": new_id, "file_name": f"bootstrap_{new_id}"})
            for original in gt_by_image[old_id]:
                row = dict(original, id=annotation_id, image_id=new_id)
                annotations.append(row)
                annotation_id += 1
            prediction_rows.extend(dict(row, image_id=new_id) for row in pred_by_image[old_id])
        cloned = dict(dataset, images=images, annotations=annotations)
        values.append(run_coco_eval(cloned, prediction_rows)[metric_index])
    return {"low": float(np.percentile(values, 2.5)), "high": float(np.percentile(values, 97.5)), "iterations": iterations}


def confusion_matrix(ground_truth: dict[str, list[Box]], predictions: dict[str, list[Box]], confidence: float, iou_threshold: float = 0.5) -> list[list[int]]:
    size = len(NAMES) + 1
    background = len(NAMES)
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for name, targets in ground_truth.items():
        candidates = sorted((box for box in predictions.get(name, []) if box.confidence >= confidence), key=lambda box: box.confidence, reverse=True)
        matched: set[int] = set()
        for candidate in candidates:
            best = max((index for index in range(len(targets)) if index not in matched), key=lambda index: box_iou(candidate.xyxy, targets[index].xyxy), default=-1)
            if best >= 0 and box_iou(candidate.xyxy, targets[best].xyxy) >= iou_threshold:
                matched.add(best)
                matrix[targets[best].class_id][candidate.class_id] += 1
            else:
                matrix[background][candidate.class_id] += 1
        for index, target in enumerate(targets):
            if index not in matched:
                matrix[target.class_id][background] += 1
    return matrix


def error_cases(
    ground_truth: dict[str, list[Box]],
    predictions: dict[str, list[Box]],
    confidence: float,
    iou_threshold: float = 0.5,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, targets in ground_truth.items():
        candidates = sorted(
            (box for box in predictions.get(name, []) if box.confidence >= confidence),
            key=lambda box: box.confidence,
            reverse=True,
        )
        matched: set[int] = set()
        mismatches = 0
        localized = 0
        false_positives = 0
        for candidate in candidates:
            best = max(
                (index for index in range(len(targets)) if index not in matched),
                key=lambda index: box_iou(candidate.xyxy, targets[index].xyxy),
                default=-1,
            )
            overlap = box_iou(candidate.xyxy, targets[best].xyxy) if best >= 0 else 0.0
            if best >= 0 and overlap >= iou_threshold:
                matched.add(best)
                mismatches += int(targets[best].class_id != candidate.class_id)
            elif best >= 0 and overlap >= 0.1 and targets[best].class_id == candidate.class_id:
                localized += 1
            else:
                false_positives += 1
        false_negatives = len(targets) - len(matched)
        severity = 3 * false_negatives + 2 * mismatches + localized + false_positives
        if severity:
            rows.append(
                {
                    "image": name,
                    "severity": severity,
                    "false_negatives": false_negatives,
                    "false_positives": false_positives,
                    "class_mismatches": mismatches,
                    "localization_errors_iou_0.1_to_0.5": localized,
                }
            )
    return sorted(rows, key=lambda row: (-int(row["severity"]), str(row["image"])))


def append_leaderboard(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        fingerprints = {item["dataset_fingerprint"] for item in existing}
        if fingerprints and fingerprints != {str(row["dataset_fingerprint"])}:
            raise ValueError(f"Refusing to mix dataset fingerprints in {path}: {fingerprints}")
    else:
        existing = []
    existing.append({key: str(value) for key, value in row.items()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(existing)


def main() -> None:
    args = parse_args()
    validate_args(args)
    from ultralytics import YOLO

    image_root, label_root, fingerprint = resolve_dataset(args.data, args.split)
    images = collect_images(image_root)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        images = images[: args.limit]
    ground_truth = read_ground_truth(images, image_root, label_root)
    started = time.perf_counter()
    model = None
    if args.mode == "external":
        predictions_by_path = {}
    elif args.mode == "standard":
        model = YOLO(args.weights)
        predictions_by_path = predict_standard(model, images, args)
    elif args.mode == "ensemble":
        predictions_by_path = predict_ensemble(images, args)
    elif args.mode == "ensemble-two-stage":
        predictions_by_path = extend_ensemble_with_local(
            images, predict_ensemble(images, args), args
        )
    elif args.mode == "sahi":
        predictions_by_path = predict_sahi(images, args)
    elif args.mode == "hybrid":
        model = YOLO(args.weights)
        standard = predict_standard(model, images, args)
        sliced = predict_sahi(images, args)
        predictions_by_path = {}
        for path in images:
            with Image.open(path) as image:
                width, height = image.size
            predictions_by_path[path.resolve()] = weighted_box_fusion(standard[path.resolve()] + sliced[path.resolve()], args.fusion_iou, width, height)
    else:
        predictions_by_path = predict_two_stage(images, args)
    if args.external_predictions:
        external_paths = (
            args.external_predictions
            if isinstance(args.external_predictions, list)
            else [args.external_predictions]
        )
        external_sets = [
            read_external_predictions(
                path.resolve(),
                image_root,
                fingerprint.get("fingerprint", "unversioned"),
                args.split,
            )
            for path in external_paths
        ]
        if args.mode == "external":
            predictions_by_path = fuse_prediction_sets(
                external_sets, images, args.fusion_iou
            )
        else:
            predictions_by_path = fuse_prediction_sets(
                [predictions_by_path, *external_sets], images, args.fusion_iou
            )
    elapsed = time.perf_counter() - started
    predictions = to_image_keyed(predictions_by_path, image_root)
    dataset, prediction_rows, _ = coco_payload(ground_truth, predictions)
    overall = run_coco_eval(dataset, prediction_rows)
    per_class = {}
    for index, name in enumerate(NAMES):
        class_stats = run_coco_eval(dataset, prediction_rows, index + 1)
        per_class[name] = {
            "map50_95": class_stats[0],
            "map50": class_stats[1],
            "map75": class_stats[2],
        }
    normal = [name for name, boxes in ground_truth.items() if not any(box.class_id > 0 for box in boxes)]
    normal_fp = sum(any(box.class_id > 0 and box.confidence >= args.operating_conf for box in predictions.get(name, [])) for name in normal)
    native_difference = None
    if args.verify_native and args.mode == "standard":
        metrics = model.val(
            data=str(args.data), split=args.split, imgsz=args.imgsz, batch=args.batch,
            conf=args.conf, iou=args.iou, device=args.device, plots=False, verbose=False,
            augment=args.tta,
        )
        native_difference = abs(float(metrics.box.map) - overall[0])
        if native_difference > 1e-4:
            raise RuntimeError(f"COCO/native mAP mismatch: {native_difference:.6f}")
    map50_ci95 = bootstrap_ci(dataset, prediction_rows, args.bootstrap, args.seed, metric_index=1)
    report = {
        "dataset": str(args.data.resolve()),
        "dataset_fingerprint": fingerprint.get("fingerprint", "unversioned"),
        "split": args.split,
        "mode": args.mode,
        "weights": str(Path(args.weights).resolve()) if args.weights else None,
        "images": len(images),
        "settings": {key: value for key, value in vars(args).items() if key not in {"output", "leaderboard"}},
        "metrics": {"map50_95": overall[0], "map50": overall[1], "map75": overall[2], "map_small": overall[3], "map_medium": overall[4], "map_large": overall[5]},
        "map50_ci95": map50_ci95,
        "map50_95_ci95": bootstrap_ci(dataset, prediction_rows, args.bootstrap, args.seed),
        "target": {
            "metric": "map50",
            "threshold": 0.95,
            "achieved": overall[1] > 0.95,
            "ci95_low_exceeds_threshold": bool(map50_ci95 and map50_ci95["low"] > 0.95),
        },
        "per_class": per_class,
        "normal_false_positive": {"images": len(normal), "images_with_fp": normal_fp, "rate": normal_fp / max(len(normal), 1)},
        "confusion_matrix_rows_gt_cols_prediction_last_is_background": confusion_matrix(ground_truth, predictions, args.operating_conf),
        "native_map_difference": native_difference,
        "performance": {"seconds": elapsed, "milliseconds_per_image": elapsed * 1000 / max(len(images), 1)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    per_class_path = args.output.with_name(f"{args.output.stem}_per_class.csv")
    with per_class_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_name", "map50_95", "map50", "map75"])
        writer.writeheader()
        writer.writerows({"class_name": name, **values} for name, values in per_class.items())
    error_path = args.output.with_name(f"{args.output.stem}_errors.csv")
    errors = error_cases(ground_truth, predictions, args.operating_conf)
    with error_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["image", "severity", "false_negatives", "false_positives", "class_mismatches", "localization_errors_iou_0.1_to_0.5"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors)
    append_leaderboard(
        args.leaderboard,
        {
            "dataset_fingerprint": report["dataset_fingerprint"], "split": args.split,
            "mode": args.mode, "weights": report["weights"], "imgsz": args.imgsz,
            "images": len(images),
            "map50_95": overall[0], "map50": overall[1], "map75": overall[2],
            "map_small": overall[3], "normal_fp_rate": report["normal_false_positive"]["rate"],
            "seconds": elapsed,
        },
    )
    print(json.dumps(report["metrics"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
