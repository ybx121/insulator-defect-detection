#!/usr/bin/env python3
"""Export D-FINE predictions in the project's model-independent JSON format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as vision_f
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_detector import collect_images, resolve_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfine-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_model(args: argparse.Namespace):
    dfine_root = args.dfine_root.resolve()
    if str(dfine_root) not in sys.path:
        sys.path.insert(0, str(dfine_root))
    from src.core import YAMLConfig

    config = YAMLConfig(str(args.config.resolve()), resume=str(args.weights.resolve()))
    config.yaml_cfg["HGNetv2"]["pretrained"] = False
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    config.model.load_state_dict(state)
    model = config.model.deploy().eval()
    postprocessor = config.postprocessor.deploy().eval()
    return model, postprocessor


def image_tensor(path: Path, size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    with Image.open(path) as source:
        image = source.convert("RGB")
        original_size = image.size
        image = vision_f.resize(image, [size, size])
        tensor = vision_f.pil_to_tensor(image).float().div_(255.0)
    return tensor, original_size


def main() -> None:
    args = parse_args()
    if args.batch <= 0:
        raise ValueError("--batch must be positive")
    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" else "cpu")
    image_root, _, fingerprint = resolve_dataset(args.data, args.split)
    images = collect_images(image_root)
    model, postprocessor = load_model(args)
    model.to(device)
    postprocessor.to(device)

    predictions: dict[str, list[dict[str, object]]] = {}
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, enabled=device.type == "cuda"
    ):
        for start in range(0, len(images), args.batch):
            paths = images[start : start + args.batch]
            loaded = [image_tensor(path, args.imgsz) for path in paths]
            batch = torch.stack([row[0] for row in loaded]).to(device)
            original_sizes = torch.tensor(
                [row[1] for row in loaded], dtype=torch.float32, device=device
            )
            labels, boxes, scores = postprocessor(model(batch), original_sizes)
            for path, image_labels, image_boxes, image_scores in zip(
                paths, labels, boxes, scores
            ):
                rows = []
                for class_id, box, score in zip(image_labels, image_boxes, image_scores):
                    confidence = float(score)
                    if confidence < args.conf:
                        continue
                    rows.append(
                        {
                            "class_id": int(class_id),
                            "confidence": confidence,
                            "xyxy": [float(value) for value in box],
                        }
                    )
                predictions[path.relative_to(image_root).as_posix()] = rows
            print(f"predicted {min(start + args.batch, len(images))}/{len(images)}", flush=True)

    payload = {
        "format_version": 1,
        "model_type": "dfine",
        "weights": str(args.weights.resolve()),
        "config": str(args.config.resolve()),
        "dataset": str(args.data.resolve()),
        "dataset_fingerprint": fingerprint.get("fingerprint", "unversioned"),
        "split": args.split,
        "imgsz": args.imgsz,
        "confidence": args.conf,
        "images": len(images),
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
