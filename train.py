#!/usr/bin/env python3
"""Train GF-InsuYOLO models with Ultralytics."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="configs/gf_insuyolo.yaml")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--weights", type=str, help="Optional checkpoint to finetune")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--project", type=str, default="runs")
    parser.add_argument("--name", type=str, default="gf_insuyolo")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def register_custom_modules() -> None:
    from gf_insuyolo.modules import FrequencyEnhance
    import ultralytics.nn.tasks as tasks

    tasks.FrequencyEnhance = FrequencyEnhance


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Install the training environment with "
            "`pip install -r requirements.txt` in Python 3.11."
        ) from exc

    register_custom_modules()
    model_source = args.weights or args.model
    model = YOLO(model_source)
    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "workers": args.workers,
        "resume": args.resume,
        "close_mosaic": 10,
        "cos_lr": True,
        "patience": 50,
        "plots": True,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    if args.weights and Path(args.model).suffix in {".yaml", ".yml"}:
        train_kwargs["cfg"] = args.model
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
