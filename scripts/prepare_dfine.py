#!/usr/bin/env python3
"""Export credible data to COCO and configure an optional D-FINE-M run."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

from PIL import Image


NAMES = ["insulator_string", "broken_shell", "flashover_pollution", "missing_disc_drop"]
PRETRAINED = {
    ("m", "coco"): (
        "dfine_m_coco.pth",
        "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_coco.pth",
    ),
    ("m", "obj2coco"): (
        "dfine_m_obj2coco.pth",
        "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_obj2coco.pth",
    ),
    ("l", "coco"): (
        "dfine_l_coco.pth",
        "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_coco.pth",
    ),
    ("l", "obj2coco"): (
        "dfine_l_obj2coco_e25.pth",
        "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_obj2coco_e25.pth",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/credible_fine_v1"))
    parser.add_argument("--output", type=Path, default=Path("datasets/credible_fine_v1_coco"))
    parser.add_argument("--dfine-root", type=Path, default=Path("runs/third_party/D-FINE"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--model-size", choices=["m", "l"], default="m")
    parser.add_argument("--pretrained-variant", choices=["coco", "obj2coco"], default="coco")
    parser.add_argument(
        "--multi-scale",
        action="store_true",
        help="Enable 0.75x-1.25x random batch resizing; fixed input is the memory-safe default",
    )
    parser.add_argument("--config-name", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--download-weights", action="store_true")
    return parser.parse_args()


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def yolo_boxes(path: Path, width: int, height: int) -> list[tuple[int, list[float]]]:
    boxes = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        class_id, x, y, box_w, box_h = raw.split()
        pixel_w, pixel_h = float(box_w) * width, float(box_h) * height
        boxes.append(
            (
                int(float(class_id)),
                [float(x) * width - pixel_w / 2, float(y) * height - pixel_h / 2, pixel_w, pixel_h],
            )
        )
    return boxes


def export_split(dataset: Path, output: Path, split: str) -> None:
    image_source, label_source = dataset / "images" / split, dataset / "labels" / split
    image_output = output / "images" / split
    image_output.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    annotation_id = 1
    for image_id, source in enumerate(sorted(image_source.iterdir()), 1):
        if not source.is_file():
            continue
        destination = image_output / source.name
        link_or_copy(source, destination)
        with Image.open(source) as image:
            width, height = image.size
        images.append({"id": image_id, "file_name": source.name, "width": width, "height": height})
        for class_id, box in yolo_boxes(label_source / f"{source.stem}.txt", width, height):
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": box,
                    "area": box[2] * box[3],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    payload = {
        "info": {"description": dataset.name},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": index, "name": name} for index, name in enumerate(NAMES)],
    }
    annotation_dir = output / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    (annotation_dir / f"instances_{split}.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"COCO {split}: images={len(images)} annotations={len(annotations)}")


def write_dfine_configs(args: argparse.Namespace, dataset: Path, dfine: Path) -> None:
    dataset_config = dfine / "configs" / "dataset" / "insulator_detection.yml"
    dataset_config.write_text(
        f"""task: detection

evaluator:
  type: CocoEvaluator
  iou_types: ['bbox']

num_classes: 4
remap_mscoco_category: False

train_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: {(dataset / 'images' / 'train').resolve().as_posix()}
    ann_file: {(dataset / 'annotations' / 'instances_train.json').resolve().as_posix()}
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: True
  num_workers: 4
  drop_last: True
  collate_fn:
    type: BatchImageCollateFunction

val_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: {(dataset / 'images' / 'val').resolve().as_posix()}
    ann_file: {(dataset / 'annotations' / 'instances_val.json').resolve().as_posix()}
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: False
  num_workers: 4
  drop_last: False
  collate_fn:
    type: BatchImageCollateFunction
""",
        encoding="utf-8",
    )
    stop_epoch = max(1, args.epochs - 5)
    base_size_repeat = 3 if args.multi_scale else "~"
    run_name = args.run_name or (
        f"{args.dataset.resolve().name}_dfine_{args.model_size}_{args.pretrained_variant}_{args.imgsz}"
    )
    config_name = args.config_name or (
        f"dfine_hgnetv2_{args.model_size}_insulator_{args.pretrained_variant}.yml"
    )
    if Path(config_name).name != config_name or not config_name.endswith(".yml"):
        raise ValueError("--config-name must be a .yml filename without directories")
    model_config = dfine / "configs" / "dfine" / "custom" / config_name
    if args.model_size == "l":
        architecture = """HGNetv2:
  name: 'B4'
  return_idx: [1, 2, 3]
  freeze_stem_only: True
  freeze_at: 0
  freeze_norm: True"""
        optimizer_params = """    - {params: '^(?=.*backbone)(?!.*norm|bn).*$', lr: 0.0000125}
    - {params: '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$', weight_decay: 0.}"""
    else:
        architecture = """DFINE:
  backbone: HGNetv2
HGNetv2:
  name: 'B2'
  return_idx: [1, 2, 3]
  freeze_at: -1
  freeze_norm: False
  use_lab: True
DFINETransformer:
  num_layers: 4
  eval_idx: -1
HybridEncoder:
  in_channels: [384, 768, 1536]
  hidden_dim: 256
  depth_mult: 0.67"""
        optimizer_params = """    - {params: '^(?=.*backbone)(?!.*norm|bn).*$', lr: 0.000025}
    - {params: '^(?=.*backbone)(?=.*norm|bn).*$', lr: 0.000025, weight_decay: 0.}
    - {params: '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$', weight_decay: 0.}"""
    model_config.write_text(
        f"""__include__: [
  '../../dataset/insulator_detection.yml',
  '../../runtime.yml',
  '../include/dataloader.yml',
  '../include/optimizer.yml',
  '../include/dfine_hgnetv2.yml',
]

output_dir: {(args.project_root.resolve() / 'runs' / 'dfine' / run_name).resolve().as_posix()}
eval_spatial_size: [{args.imgsz}, {args.imgsz}]

{architecture}
optimizer:
  type: AdamW
  params:
{optimizer_params}
  lr: 0.00025
  betas: [0.9, 0.999]
  weight_decay: 0.000125
epochs: {args.epochs}
train_dataloader:
  total_batch_size: {args.batch}
  dataset:
    transforms:
      ops:
        - {{type: RandomPhotometricDistort, p: 0.5}}
        - {{type: RandomZoomOut, fill: 0}}
        - {{type: RandomIoUCrop, p: 0.8}}
        - {{type: SanitizeBoundingBoxes, min_size: 1}}
        - {{type: RandomHorizontalFlip}}
        - {{type: Resize, size: [{args.imgsz}, {args.imgsz}]}}
        - {{type: SanitizeBoundingBoxes, min_size: 1}}
        - {{type: ConvertPILImage, dtype: 'float32', scale: True}}
        - {{type: ConvertBoxes, fmt: 'cxcywh', normalize: True}}
      policy:
        name: stop_epoch
        epoch: {stop_epoch}
        ops: ['RandomPhotometricDistort', 'RandomZoomOut', 'RandomIoUCrop']
  collate_fn:
    base_size: {args.imgsz}
    base_size_repeat: {base_size_repeat}
    stop_epoch: {stop_epoch}
    ema_restart_decay: 0.9999
val_dataloader:
  total_batch_size: {args.batch}
  dataset:
    transforms:
      ops:
        - {{type: Resize, size: [{args.imgsz}, {args.imgsz}]}}
        - {{type: ConvertPILImage, dtype: 'float32', scale: True}}
""",
        encoding="utf-8",
    )
    print(f"wrote {dataset_config} and {model_config}")


def main() -> None:
    args = parse_args()
    source, output, dfine = args.dataset.resolve(), args.output.resolve(), args.dfine_root.resolve()
    if not dfine.exists():
        raise FileNotFoundError(f"Clone D-FINE into {dfine}")
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite")
    for split in ("train", "val", "test"):
        export_split(source, output, split)
    write_dfine_configs(args, output, dfine)
    weight_name, pretrained_url = PRETRAINED[(args.model_size, args.pretrained_variant)]
    weights = dfine / "weights" / weight_name
    if args.download_weights and not weights.exists():
        weights.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(pretrained_url, weights)
        print(f"downloaded {weights}")
    config_name = args.config_name or (
        f"dfine_hgnetv2_{args.model_size}_insulator_{args.pretrained_variant}.yml"
    )
    print(
        "D-FINE command: "
        f"{sys.executable} train.py -c configs/dfine/custom/{config_name} "
        f"--use-amp --seed=20260708 -t weights/{weight_name}"
    )


if __name__ == "__main__":
    main()
