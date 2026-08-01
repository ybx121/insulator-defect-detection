#!/usr/bin/env python3
"""Run single-stage or global-local GF-InsuYOLO inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from gf_insuyolo.boxes import Detection, expand_box, nms, remap_crop_box


DEFAULT_NAMES = ["insulator_string", "broken_shell", "flashover_pollution", "missing_disc_drop"]
LOCAL_NAMES = DEFAULT_NAMES[1:]
LOCAL_TO_FINE_ID = {name: index + 1 for index, name in enumerate(LOCAL_NAMES)}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Global detector weights")
    parser.add_argument("--local-weights", help="Optional crop detector weights")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/infer/predictions.json"))
    parser.add_argument("--two-stage", action="store_true")
    parser.add_argument(
        "--two-stage-fusion",
        choices=["union", "local-preferred", "local-only"],
        default="union",
    )
    parser.add_argument("--global-imgsz", type=int, default=960)
    parser.add_argument("--local-imgsz", type=int, default=640)
    parser.add_argument("--global-conf", type=float, default=0.15)
    parser.add_argument("--local-conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, help="Legacy shortcut that sets both image sizes")
    parser.add_argument("--conf", type=float, help="Legacy shortcut that sets both confidence thresholds")
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--crop-margin", type=float, default=0.15)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()
    if args.imgsz is not None:
        args.global_imgsz = args.imgsz
        args.local_imgsz = args.imgsz
    if args.conf is not None:
        args.global_conf = args.conf
        args.local_conf = args.conf
    return args


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


def model_names(model) -> list[str]:
    names = model.names
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    return [str(name) for name in names]


def predict_one(
    model,
    source,
    names: list[str],
    stage: str,
    imgsz: int,
    conf: float,
    iou: float,
    device: str | None = None,
) -> list[Detection]:
    predict_kwargs = {
        "source": source,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "verbose": False,
        "rect": False,
    }
    if device is not None:
        predict_kwargs["device"] = device
    results = model.predict(
        **predict_kwargs,
    )
    return result_to_detections(results[0], names, stage)


def run_two_stage(global_model, local_model, image_path: Path, args: argparse.Namespace) -> list[Detection]:
    global_dets = predict_one(
        global_model,
        str(image_path),
        model_names(global_model),
        "global",
        args.global_imgsz,
        args.global_conf,
        args.iou,
        args.device,
    )
    insulators = [
        Detection(0, DEFAULT_NAMES[0], det.confidence, det.xyxy, det.source_stage)
        for det in global_dets
        if det.cls == 0
    ]
    # A coarse model's generic `defect` class has no valid fine-class mapping.
    # Preserve global defects only when their names already match the fine taxonomy.
    global_defects = [
        Detection(
            LOCAL_TO_FINE_ID[det.class_name],
            det.class_name,
            det.confidence,
            det.xyxy,
            det.source_stage,
        )
        for det in global_dets
        if det.class_name in LOCAL_TO_FINE_ID
    ]
    local_defects: list[Detection] = []
    crop_boxes: list[tuple[int, int, int, int]] = []
    local_names = model_names(local_model)

    with Image.open(image_path) as image:
        width, height = image.size
        for insulator in insulators:
            crop = expand_box(insulator.xyxy, width, height, args.crop_margin)
            if crop[2] <= crop[0] or crop[3] <= crop[1]:
                continue
            crop_boxes.append(crop)
            local_dets = predict_one(
                local_model,
                image.crop(crop),
                local_names,
                "local_crop",
                args.local_imgsz,
                args.local_conf,
                args.iou,
                args.device,
            )
            for local in local_dets:
                fine_id = LOCAL_TO_FINE_ID.get(local.class_name)
                if fine_id is None:
                    raise ValueError(
                        f"Unsupported local class {local.class_name!r}; expected one of {LOCAL_NAMES}"
                    )
                remapped = remap_crop_box(local.xyxy, (crop[0], crop[1]))
                local_defects.append(
                    Detection(
                        cls=fine_id,
                        class_name=local.class_name,
                        confidence=local.confidence,
                        xyxy=remapped,
                        source_stage="local_crop",
                    )
                )
    defects = combine_two_stage_defects(
        global_defects,
        local_defects,
        crop_boxes,
        getattr(args, "two_stage_fusion", "union"),
    )
    return nms(insulators + defects, args.iou)


def combine_two_stage_defects(
    global_defects: list[Detection],
    local_defects: list[Detection],
    crop_boxes: list[tuple[int, int, int, int]],
    mode: str,
) -> list[Detection]:
    if mode == "union":
        return global_defects + local_defects
    if mode == "local-only":
        return local_defects
    if mode != "local-preferred":
        raise ValueError(f"Unsupported two-stage fusion mode: {mode}")

    outside = []
    for defect in global_defects:
        x1, y1, x2, y2 = defect.xyxy
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if not any(rx1 <= cx <= rx2 and ry1 <= cy <= ry2 for rx1, ry1, rx2, ry2 in crop_boxes):
            outside.append(defect)
    return outside + local_defects


def register_custom_modules() -> None:
    from gf_insuyolo.modules import ContextGuidedEnhance, FrequencyEnhance
    import ultralytics.nn.tasks as tasks

    tasks.FrequencyEnhance = FrequencyEnhance
    tasks.ContextGuidedEnhance = ContextGuidedEnhance


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Install the inference environment with "
            "`pip install -r requirements.txt` in Python 3.11."
        ) from exc

    if args.two_stage and not args.local_weights:
        raise SystemExit("--two-stage requires --local-weights")

    source_paths = iter_sources(Path(args.source))
    if not source_paths:
        raise SystemExit(f"No images found in {args.source}")

    register_custom_modules()
    global_model = YOLO(args.weights)
    local_model = YOLO(args.local_weights) if args.local_weights else None
    output_rows = []
    for image_path in source_paths:
        if args.two_stage:
            assert local_model is not None
            detections = run_two_stage(global_model, local_model, image_path, args)
        else:
            detections = nms(
                predict_one(
                    global_model,
                    str(image_path),
                    model_names(global_model),
                    "global",
                    args.global_imgsz,
                    args.global_conf,
                    args.iou,
                    args.device,
                ),
                args.iou,
            )
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
