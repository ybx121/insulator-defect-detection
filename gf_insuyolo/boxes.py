"""Bounding-box utilities used by inference and validation scripts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    cls: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    source_stage: str

    def as_json(self) -> dict[str, object]:
        return {
            "class_id": self.cls,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 6),
            "bbox_xyxy": [round(float(v), 3) for v in self.xyxy],
            "source_stage": self.source_stage,
        }


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(detections: list[Detection], iou_threshold: float = 0.55) -> list[Detection]:
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if all(det.cls != prev.cls or box_iou(det.xyxy, prev.xyxy) < iou_threshold for prev in kept):
            kept.append(det)
    return kept


def expand_box(
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    margin: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    dx, dy = width * margin, height * margin
    return (
        max(0, int(round(x1 - dx))),
        max(0, int(round(y1 - dy))),
        min(image_width, int(round(x2 + dx))),
        min(image_height, int(round(y2 + dy))),
    )


def remap_crop_box(
    crop_box: tuple[float, float, float, float],
    crop_origin: tuple[int, int],
) -> tuple[float, float, float, float]:
    ox, oy = crop_origin
    x1, y1, x2, y2 = crop_box
    return x1 + ox, y1 + oy, x2 + ox, y2 + oy
