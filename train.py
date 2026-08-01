#!/usr/bin/env python3
"""Train insulator detectors with reproducible Ultralytics settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import time
from pathlib import Path


AUGMENT_PRESETS = {
    "default": {},
    "moderate": {
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.3,
        "degrees": 5.0,
        "translate": 0.1,
        "scale": 0.4,
        "shear": 2.0,
        "perspective": 0.0002,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
    },
    "defect_safe": {
        "hsv_h": 0.008,
        "hsv_s": 0.25,
        "hsv_v": 0.2,
        "degrees": 3.0,
        "translate": 0.05,
        "scale": 0.2,
        "shear": 1.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.5,
        "mixup": 0.0,
        "copy_paste": 0.0,
    },
    "none": {
        "augmentations": [],
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
    },
}


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
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--optimizer", type=str, default="auto")
    parser.add_argument("--lr0", type=float)
    parser.add_argument("--lrf", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--freeze", type=int)
    parser.add_argument("--warmup-epochs", type=float)
    parser.add_argument("--warmup-bias-lr", type=float)
    parser.add_argument("--warmup-momentum", type=float)
    parser.add_argument(
        "--context-only",
        action="store_true",
        help="Freeze YOLO11s layers 0-22 and Detect layer 26; train context layers 23-25",
    )
    parser.add_argument(
        "--detect-source-offset",
        type=int,
        default=0,
        help="Shift source Detect scale indices when transferring a head (P2 to P3 uses 1)",
    )
    parser.add_argument(
        "--detect-base-weights",
        help="Optional checkpoint used to initialize Detect before the primary shifted transfer",
    )
    parser.add_argument("--close-mosaic", type=int, default=15)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--augment-preset", choices=sorted(AUGMENT_PRESETS), default="moderate")
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def register_custom_modules() -> None:
    from gf_insuyolo.modules import ContextGuidedEnhance, FrequencyEnhance
    import ultralytics.nn.tasks as tasks

    tasks.FrequencyEnhance = FrequencyEnhance
    tasks.ContextGuidedEnhance = ContextGuidedEnhance


def transfer_detection_head(target: object, source: object, source_offset: int = 0) -> int:
    """Transfer shape-compatible final Detect parameters after YAML layer changes."""
    target_head = target.model.model[-1]
    source_head = source.model.model[-1]
    target_state = target_head.state_dict()
    source_state = source_head.state_dict()
    compatible = {}
    for target_key, target_value in target_state.items():
        source_key = re.sub(
            r"^((?:one2one_)?cv[23]\.)(\d+)(\.)",
            lambda match: f"{match.group(1)}{int(match.group(2)) + source_offset}{match.group(3)}",
            target_key,
        )
        value = source_state.get(source_key)
        if value is not None and target_value.shape == value.shape:
            compatible[target_key] = value
    target_head.load_state_dict(compatible, strict=False)
    return len(compatible)


def build_train_kwargs(args: argparse.Namespace) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "workers": args.workers,
        "resume": args.resume,
        "close_mosaic": args.close_mosaic,
        "cos_lr": args.cos_lr,
        "patience": args.patience,
        "plots": True,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "optimizer": args.optimizer,
        "cache": args.cache,
        "exist_ok": args.exist_ok,
    }
    optional = {
        "device": args.device,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "freeze": args.freeze,
        "warmup_epochs": args.warmup_epochs,
        "warmup_bias_lr": args.warmup_bias_lr,
        "warmup_momentum": args.warmup_momentum,
    }
    kwargs.update({key: value for key, value in optional.items() if value is not None})
    if args.context_only:
        if args.freeze is not None:
            raise ValueError("--context-only and --freeze cannot be used together")
        kwargs["freeze"] = [*range(23), 26]
    kwargs.update(AUGMENT_PRESETS[args.augment_preset])
    return kwargs


def sha256_if_file(value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_experiment_manifest(
    args: argparse.Namespace,
    train_kwargs: dict[str, object],
    model: object,
    elapsed_seconds: float,
) -> None:
    save_dir = Path(model.trainer.save_dir)
    manifest = {
        "command_args": vars(args),
        "resolved_train_args": train_kwargs,
        "elapsed_seconds": elapsed_seconds,
        "model_sha256": sha256_if_file(args.model),
        "weights_sha256": sha256_if_file(args.weights),
        "data_yaml_sha256": sha256_if_file(args.data),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch
        import ultralytics

        manifest.update(
            {
                "torch": torch.__version__,
                "ultralytics": ultralytics.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except ImportError:
        pass
    (save_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )


class BestMap50Checkpoint:
    """Keep the checkpoint that matches the project's primary validation metric."""

    def __init__(self) -> None:
        self.best = float("-inf")

    def __call__(self, trainer: object) -> None:
        score = float(trainer.metrics.get("metrics/mAP50(B)", float("-inf")))
        if score <= self.best:
            return
        self.best = score
        destination = Path(trainer.wdir) / "best_map50.pt"
        shutil.copy2(trainer.last, destination)
        destination.with_suffix(".json").write_text(
            json.dumps({"epoch": int(trainer.epoch), "map50": score}, indent=2),
            encoding="utf-8",
        )


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
    model_path = Path(args.model)
    if args.weights and model_path.suffix in {".yaml", ".yml"}:
        source_model = YOLO(args.weights)
        model = YOLO(args.model).load(args.weights)
        if args.detect_base_weights:
            detect_base = YOLO(args.detect_base_weights)
            base_transferred = transfer_detection_head(model, detect_base)
            print(f"Transferred {base_transferred} base Detect tensors")
            del detect_base
        transferred = transfer_detection_head(model, source_model, args.detect_source_offset)
        print(f"Transferred {transferred} shape-compatible Detect tensors")
        del source_model
    else:
        model_source = args.weights or args.model
        model = YOLO(model_source)
    model.add_callback("on_model_save", BestMap50Checkpoint())
    train_kwargs = build_train_kwargs(args)
    started = time.perf_counter()
    model.train(**train_kwargs)
    write_experiment_manifest(args, train_kwargs, model, time.perf_counter() - started)


if __name__ == "__main__":
    main()
