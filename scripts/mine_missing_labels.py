#!/usr/bin/env python3
"""Mine train-only missing-label candidates using cross-model consensus."""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_detector import (  # noqa: E402
    Box,
    box_iou,
    collect_images,
    predict_standard,
    read_ground_truth,
    resolve_dataset,
)


NAMES = ["insulator_string", "broken_shell", "flashover_pollution", "missing_disc_drop"]
COLORS = {1: "#e53935", 2: "#f9a825", 3: "#1565c0"}


@dataclass(frozen=True)
class Candidate:
    class_id: int
    score: float
    agreement_iou: float
    gt_same_iou: float
    gt_any_iou: float
    xyxy: tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/audit/missing_label_candidates"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--agreement-iou", type=float, default=0.55)
    parser.add_argument("--max-gt-iou", type=float, default=0.20)
    parser.add_argument("--max-candidates", type=int, default=500)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def weighted_pair(left: Box, right: Box) -> tuple[float, float, float, float]:
    total = max(left.confidence + right.confidence, 1e-9)
    return tuple(
        (left.confidence * a + right.confidence * b) / total
        for a, b in zip(left.xyxy, right.xyxy)
    )


def consensus_candidates(
    predictions: list[list[Box]],
    ground_truth: list[Box],
    agreement_iou: float,
    max_gt_iou: float,
) -> list[Candidate]:
    """Return defect predictions supported by every model but absent from labels."""
    if len(predictions) < 2:
        raise ValueError("At least two prediction sets are required")
    candidates: list[Candidate] = []
    used: list[set[int]] = [set() for _ in predictions]
    for first_index, first in enumerate(predictions[0]):
        if first.class_id == 0:
            continue
        matches = [(0, first_index, first)]
        for model_index, rows in enumerate(predictions[1:], 1):
            compatible = [
                (box_iou(first.xyxy, box.xyxy), index, box)
                for index, box in enumerate(rows)
                if index not in used[model_index] and box.class_id == first.class_id
            ]
            if not compatible:
                break
            overlap, index, box = max(compatible, key=lambda row: row[0])
            if overlap < agreement_iou:
                break
            matches.append((model_index, index, box))
        if len(matches) != len(predictions):
            continue

        fused = matches[0][2].xyxy
        confidence_weight = matches[0][2].confidence
        for _, _, box in matches[1:]:
            left = Box(first.class_id, confidence_weight, fused)
            fused = weighted_pair(left, box)
            confidence_weight += box.confidence
        same_iou = max(
            (box_iou(fused, gt.xyxy) for gt in ground_truth if gt.class_id == first.class_id),
            default=0.0,
        )
        any_iou = max(
            (box_iou(fused, gt.xyxy) for gt in ground_truth if gt.class_id > 0),
            default=0.0,
        )
        if any_iou >= max_gt_iou:
            continue
        for model_index, index, _ in matches:
            used[model_index].add(index)
        agreement = min(
            box_iou(matches[left][2].xyxy, matches[right][2].xyxy)
            for left in range(len(matches))
            for right in range(left + 1, len(matches))
        )
        candidates.append(
            Candidate(
                class_id=first.class_id,
                score=min(box.confidence for _, _, box in matches),
                agreement_iou=agreement,
                gt_same_iou=same_iou,
                gt_any_iou=any_iou,
                xyxy=fused,
            )
        )
    return sorted(candidates, key=lambda row: (row.score, row.agreement_iou), reverse=True)


def write_audit(
    output: Path,
    image_root: Path,
    rows: list[tuple[str, Candidate]],
    settings: dict[str, object],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    thumbnail_root = output / "thumbnails"
    thumbnail_root.mkdir(exist_ok=True)
    csv_rows: list[dict[str, object]] = []
    cards: list[str] = []
    for index, (relative, candidate) in enumerate(rows, 1):
        image_path = image_root / relative
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle(candidate.xyxy, outline=COLORS[candidate.class_id], width=max(2, image.width // 500))
        thumbnail = image.copy()
        thumbnail.thumbnail((720, 540))
        thumbnail_name = f"{index:04d}.jpg"
        thumbnail.save(thumbnail_root / thumbnail_name, quality=90)
        x1, y1, x2, y2 = candidate.xyxy
        csv_rows.append(
            {
                "image": relative,
                "class_id": candidate.class_id,
                "class_name": NAMES[candidate.class_id],
                "score": f"{candidate.score:.6f}",
                "agreement_iou": f"{candidate.agreement_iou:.6f}",
                "gt_same_iou": f"{candidate.gt_same_iou:.6f}",
                "gt_any_iou": f"{candidate.gt_any_iou:.6f}",
                "x1": f"{x1:.2f}", "y1": f"{y1:.2f}",
                "x2": f"{x2:.2f}", "y2": f"{y2:.2f}",
                "review_status": "pending",
            }
        )
        cards.append(
            "<article><img src='thumbnails/{thumb}'><p><b>{name}</b> "
            "score={score:.3f}, agreement={agreement:.3f}</p><code>{path}</code></article>".format(
                thumb=thumbnail_name,
                name=html.escape(NAMES[candidate.class_id]),
                score=candidate.score,
                agreement=candidate.agreement_iou,
                path=html.escape(relative),
            )
        )
    fields = list(csv_rows[0]) if csv_rows else [
        "image", "class_id", "class_name", "score", "agreement_iou", "gt_same_iou",
        "gt_any_iou", "x1", "y1", "x2", "y2", "review_status",
    ]
    with (output / "candidates.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    (output / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    page = """<!doctype html><meta charset='utf-8'><title>Missing label candidates</title>
<style>body{font:14px Arial;margin:24px}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}article{border:1px solid #ccc;padding:8px}img{width:100%;height:280px;object-fit:contain;background:#111}code{overflow-wrap:anywhere}</style>
<h1>Train-only missing label candidates</h1><p>Review candidates.csv; no labels are modified automatically.</p><main>""" + "".join(cards) + "</main>"
    (output / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if len(args.weights) < 2:
        raise SystemExit("--weights requires at least two checkpoints")
    from ultralytics import YOLO

    image_root, label_root, fingerprint = resolve_dataset(args.data, "train")
    images = collect_images(image_root)
    if args.limit:
        images = images[: args.limit]
    ground_truth = read_ground_truth(images, image_root, label_root)
    inference_args = argparse.Namespace(
        batch=args.batch, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device
    )
    model_predictions = []
    for weights in args.weights:
        print(f"predicting {weights}", flush=True)
        model_predictions.append(predict_standard(YOLO(weights), images, inference_args))

    rows: list[tuple[str, Candidate]] = []
    for image in images:
        relative = image.relative_to(image_root).as_posix()
        predictions = [model[image.resolve()] for model in model_predictions]
        for candidate in consensus_candidates(
            predictions, ground_truth[relative], args.agreement_iou, args.max_gt_iou
        ):
            rows.append((relative, candidate))
    rows.sort(key=lambda row: (row[1].score, row[1].agreement_iou), reverse=True)
    rows = rows[: args.max_candidates]
    settings = {
        "data": str(args.data.resolve()),
        "dataset_fingerprint": fingerprint.get("fingerprint"),
        "split": "train",
        "weights": [str(Path(value).resolve()) for value in args.weights],
        "imgsz": args.imgsz,
        "confidence": args.conf,
        "agreement_iou": args.agreement_iou,
        "max_gt_iou": args.max_gt_iou,
        "images": len(images),
        "candidates": len(rows),
    }
    write_audit(args.output.resolve(), image_root, rows, settings)
    print(f"wrote {len(rows)} candidates to {args.output.resolve()}")


if __name__ == "__main__":
    main()
