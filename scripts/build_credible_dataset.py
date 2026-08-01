#!/usr/bin/env python3
"""Build a leakage-resistant four-class insulator dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


NAMES = ["insulator_string", "broken_shell", "flashover_pollution", "missing_disc_drop"]
PUBLIC_CLASS_MAP = {"insulator": 0, "broken": 1, "pollution-flashover": 2}
SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class Label:
    class_id: int
    x: float
    y: float
    w: float
    h: float

    def yolo(self) -> str:
        return f"{self.class_id} {self.x:.6f} {self.y:.6f} {self.w:.6f} {self.h:.6f}"


@dataclass
class Sample:
    source: str
    source_image: Path
    stem: str
    family: str
    labels: tuple[Label, ...]
    sha1: str = ""
    dhash: int = 0
    width: int = 0
    height: int = 0
    group_id: str = ""
    split: str = ""


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", type=Path, default=Path("datasets/unified_fine"))
    parser.add_argument("--public-root", type=Path, default=Path("datasets/raw/supervisely"))
    parser.add_argument("--output", type=Path, default=Path("datasets/credible_fine_v1"))
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--phash-distance", type=int, default=4)
    parser.add_argument("--audit-size", type=int, default=400)
    parser.add_argument(
        "--review-csv",
        type=Path,
        help="Completed audit CSV; statuses are approved, corrected, rejected, or pending",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_yolo(path: Path) -> tuple[Label, ...]:
    labels: list[Label] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid row at {path}:{line_number}")
        class_id = int(float(parts[0]))
        values = [float(value) for value in parts[1:]]
        if class_id not in range(len(NAMES)) or any(not 0 <= value <= 1 for value in values):
            raise ValueError(f"Invalid label at {path}:{line_number}: {raw}")
        if values[2] <= 0 or values[3] <= 0:
            raise ValueError(f"Non-positive box at {path}:{line_number}: {raw}")
        labels.append(Label(class_id, *values))
    return tuple(labels)


def collect_existing(root: Path) -> list[Sample]:
    metadata_path = root / "metadata" / "samples.csv"
    rows = list(csv.DictReader(metadata_path.open(newline="", encoding="utf-8")))
    samples: list[Sample] = []
    for row in rows:
        image = root / Path(row["image"])
        label = root / Path(row["label"])
        source = row["source_dataset"]
        samples.append(
            Sample(
                source=source,
                source_image=image,
                stem=image.stem,
                family=f"{source}:{image.stem}",
                labels=read_yolo(label),
            )
        )
    return samples


def public_family(stem: str) -> str:
    return re.sub(r"[dhv]$", "", stem, flags=re.IGNORECASE)


def collect_public(root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for archive_name, subset in (("Train", "train"), ("Test", "test")):
        base = root / archive_name / subset
        image_dir, annotation_dir = base / "img", base / "ann"
        if not image_dir.exists():
            raise FileNotFoundError(
                f"Missing {image_dir}; run scripts/download_public_dataset.py first"
            )
        for image in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS):
            annotation = json.loads(
                (annotation_dir / f"{image.name}.json").read_text(encoding="utf-8")
            )
            width = int(annotation["size"]["width"])
            height = int(annotation["size"]["height"])
            labels: list[Label] = []
            for obj in annotation.get("objects", []):
                title = obj["classTitle"]
                if title not in PUBLIC_CLASS_MAP:
                    raise ValueError(f"Unknown public class {title!r}")
                (x1, y1), (x2, y2) = obj["points"]["exterior"]
                x1, x2 = sorted((max(0.0, float(x1)), min(float(width), float(x2))))
                y1, y2 = sorted((max(0.0, float(y1)), min(float(height), float(y2))))
                if x2 <= x1 or y2 <= y1:
                    continue
                labels.append(
                    Label(
                        PUBLIC_CLASS_MAP[title],
                        ((x1 + x2) / 2) / width,
                        ((y1 + y2) / 2) / height,
                        (x2 - x1) / width,
                        (y2 - y1) / height,
                    )
                )
            if labels:
                samples.append(
                    Sample(
                        source="Supervisely_Insulator_Defect",
                        source_image=image,
                        stem=f"public_{image.stem}",
                        family=f"public:{public_family(image.stem)}",
                        labels=tuple(labels),
                    )
                )
    return samples


def load_reviews(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def apply_reviews(samples: list[Sample], reviews: dict[str, dict[str, str]]) -> list[Sample]:
    aliases = {
        "broken_shell": 1,
        "flashover_pollution": 2,
        "missing_disc_drop": 3,
        "1": 1,
        "2": 2,
        "3": 3,
    }
    kept: list[Sample] = []
    for sample in samples:
        review = reviews.get(sample.stem)
        status = (review or {}).get("review_status", "pending").strip().lower()
        if status in {"", "pending", "approved"}:
            kept.append(sample)
            continue
        if status == "rejected":
            continue
        if status != "corrected":
            raise ValueError(f"Unknown review_status {status!r} for {sample.stem}")
        corrected_yolo = (review or {}).get("corrected_yolo", "").strip()
        corrected_class = (review or {}).get("corrected_defect_class", "").strip()
        if corrected_yolo:
            rows = corrected_yolo.replace("|", "\n")
            temporary = []
            for raw in rows.splitlines():
                parts = raw.split()
                if len(parts) != 5:
                    raise ValueError(f"Invalid corrected_yolo for {sample.stem}: {raw!r}")
                class_id = int(float(parts[0]))
                coords = [float(value) for value in parts[1:]]
                if class_id not in range(4) or any(not 0 <= value <= 1 for value in coords):
                    raise ValueError(f"Invalid corrected_yolo for {sample.stem}: {raw!r}")
                temporary.append(Label(class_id, *coords))
            sample.labels = tuple(temporary)
        elif corrected_class:
            if corrected_class not in aliases:
                raise ValueError(f"Invalid corrected_defect_class for {sample.stem}: {corrected_class}")
            class_id = aliases[corrected_class]
            sample.labels = tuple(
                Label(class_id if label.class_id > 0 else 0, label.x, label.y, label.w, label.h)
                for label in sample.labels
            )
        else:
            raise ValueError(f"Corrected review for {sample.stem} has no correction")
        kept.append(sample)
    return kept


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | int(
                pixels[row * 9 + column] > pixels[row * 9 + column + 1]
            )
    return value


def fingerprint_samples(samples: list[Sample]) -> None:
    for index, sample in enumerate(samples, 1):
        with Image.open(sample.source_image) as image:
            sample.width, sample.height = image.size
            sample.dhash = difference_hash(image)
        sample.sha1 = file_sha1(sample.source_image)
        if index % 250 == 0 or index == len(samples):
            print(f"fingerprinted {index}/{len(samples)} images", flush=True)


def group_samples(samples: list[Sample], phash_distance: int) -> dict[str, list[Sample]]:
    union = UnionFind(len(samples))
    by_sha: dict[str, int] = {}
    by_family: dict[str, int] = {}
    for index, sample in enumerate(samples):
        if sample.sha1 in by_sha:
            union.union(index, by_sha[sample.sha1])
        else:
            by_sha[sample.sha1] = index
        if sample.family in by_family:
            union.union(index, by_family[sample.family])
        else:
            by_family[sample.family] = index

    # Near-duplicate matching is deliberately conservative to avoid merging
    # visually simple but unrelated sky/background images.
    for left in range(len(samples)):
        first = samples[left]
        first_ratio = first.width / first.height
        for right in range(left + 1, len(samples)):
            second = samples[right]
            if abs(math.log(first_ratio / (second.width / second.height))) > 0.02:
                continue
            if (first.dhash ^ second.dhash).bit_count() <= phash_distance:
                union.union(left, right)

    roots = sorted({union.find(index) for index in range(len(samples))})
    root_ids = {root: f"group_{position:05d}" for position, root in enumerate(roots)}
    groups: dict[str, list[Sample]] = defaultdict(list)
    for index, sample in enumerate(samples):
        sample.group_id = root_ids[union.find(index)]
        groups[sample.group_id].append(sample)
    return dict(groups)


def group_vector(samples: list[Sample]) -> tuple[int, Counter[int], Counter[str]]:
    return (
        len(samples),
        Counter(label.class_id for sample in samples for label in sample.labels),
        Counter(sample.source for sample in samples),
    )


def stratified_group_split(
    groups: dict[str, list[Sample]], ratios: tuple[float, float, float], seed: int
) -> dict[str, list[Sample]]:
    if not math.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must sum to 1")
    total_images = sum(len(group) for group in groups.values())
    total_classes = Counter(label.class_id for group in groups.values() for s in group for label in s.labels)
    total_sources = Counter(s.source for group in groups.values() for s in group)
    targets = {
        split: {
            "images": total_images * ratio,
            "classes": {key: value * ratio for key, value in total_classes.items()},
            "sources": {key: value * ratio for key, value in total_sources.items()},
        }
        for split, ratio in zip(SPLITS, ratios)
    }
    current = {
        split: {"images": 0, "classes": Counter(), "sources": Counter()} for split in SPLITS
    }
    assignments: dict[str, list[Sample]] = {split: [] for split in SPLITS}
    rng = random.Random(seed)
    order = list(groups.items())
    rng.shuffle(order)
    order.sort(
        key=lambda item: (
            max((1 / max(total_classes[label.class_id], 1) for s in item[1] for label in s.labels), default=0),
            len(item[1]),
        ),
        reverse=True,
    )

    def delta(split: str, group: list[Sample]) -> float:
        image_count, class_counts, source_counts = group_vector(group)
        target = targets[split]
        state = current[split]
        score = 2.0 * _squared_delta(state["images"], image_count, target["images"])
        for key, value in class_counts.items():
            score += 4.0 * _squared_delta(
                state["classes"][key], value, target["classes"][key]
            )
        for key, value in source_counts.items():
            score += _squared_delta(state["sources"][key], value, target["sources"][key])
        return score

    for _, group in order:
        chosen = min(SPLITS, key=lambda split: (delta(split, group), SPLITS.index(split)))
        assignments[chosen].extend(group)
        image_count, class_counts, source_counts = group_vector(group)
        current[chosen]["images"] += image_count
        current[chosen]["classes"].update(class_counts)
        current[chosen]["sources"].update(source_counts)
        for sample in group:
            sample.split = chosen

    for split, split_samples in assignments.items():
        present = {label.class_id for sample in split_samples for label in sample.labels}
        if present != set(range(len(NAMES))):
            raise RuntimeError(f"{split} does not contain all classes: {present}")
    return assignments


def _squared_delta(current: float, addition: float, target: float) -> float:
    scale = max(target, 1.0)
    return ((current + addition - target) / scale) ** 2 - ((current - target) / scale) ** 2


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_audit(
    output: Path,
    samples: list[Sample],
    size: int,
    seed: int,
    reviews: dict[str, dict[str, str]],
) -> None:
    rng = random.Random(seed)
    cplid = [sample for sample in samples if sample.source == "CPLID_Defective"]
    remaining = [sample for sample in samples if sample not in cplid]
    remaining.sort(key=lambda sample: min(label.w * label.h for label in sample.labels))
    selected = list(cplid)
    selected.extend(remaining[: max(0, min(size, 350) - len(selected))])
    unselected = [sample for sample in remaining if sample not in selected]
    rng.shuffle(unselected)
    selected.extend(unselected[: max(0, size - len(selected))])
    selected = selected[:size]

    thumbnail_dir = output / "audit" / "thumbnails"
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    cards: list[str] = []
    colors = ["#2563eb", "#dc2626", "#ca8a04", "#16a34a"]
    for index, sample in enumerate(selected):
        with Image.open(sample.source_image) as raw:
            image = raw.convert("RGB")
        draw = ImageDraw.Draw(image)
        for label in sample.labels:
            x1 = (label.x - label.w / 2) * image.width
            y1 = (label.y - label.h / 2) * image.height
            x2 = (label.x + label.w / 2) * image.width
            y2 = (label.y + label.h / 2) * image.height
            draw.rectangle((x1, y1, x2, y2), outline=colors[label.class_id], width=max(2, image.width // 500))
            draw.text((x1, max(0, y1 - 14)), NAMES[label.class_id], fill=colors[label.class_id])
        image.thumbnail((520, 360), Image.Resampling.LANCZOS)
        thumb_name = f"{index:04d}_{sample.stem}.jpg"
        image.save(thumbnail_dir / thumb_name, quality=88)
        min_area = min(label.w * label.h for label in sample.labels)
        reasons = []
        if sample.source == "CPLID_Defective":
            reasons.append("CPLID synthetic class review")
        if min_area < 0.002:
            reasons.append("very small box")
        if not reasons:
            reasons.append("stratified random audit")
        existing = reviews.get(sample.stem, {})
        rows.append(
            {
                "sample_id": sample.stem,
                "split": sample.split,
                "source": sample.source,
                "source_image": str(sample.source_image.resolve()),
                "classes": "|".join(sorted({NAMES[label.class_id] for label in sample.labels})),
                "reason": "|".join(reasons),
                "review_status": existing.get("review_status", "pending"),
                "corrected_defect_class": existing.get("corrected_defect_class", ""),
                "corrected_yolo": existing.get("corrected_yolo", ""),
                "correction_notes": existing.get("correction_notes", ""),
            }
        )
        cards.append(
            "<article><img loading='lazy' src='thumbnails/{}'><strong>{}</strong>"
            "<span>{}</span><span>{}</span></article>".format(
                html.escape(thumb_name),
                html.escape(sample.stem),
                html.escape(sample.source),
                html.escape("; ".join(reasons)),
            )
        )
    write_csv(output / "audit" / "review.csv", rows)
    page = """<!doctype html><meta charset='utf-8'><title>Label audit</title>
<style>body{font:14px Arial;margin:20px;background:#f6f7f8;color:#171717}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}article{background:white;border:1px solid #ddd;padding:8px}img{width:100%;height:220px;object-fit:contain;background:#111}strong,span{display:block;margin-top:5px}</style>
<h1>Credible fine dataset label audit</h1><p>Record final decisions in review.csv.</p><main>""" + "".join(cards) + "</main>"
    (output / "audit" / "index.html").write_text(page, encoding="utf-8")


def write_dataset(
    output: Path,
    assignments: dict[str, list[Sample]],
    groups: dict[str, list[Sample]],
    overwrite: bool,
) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        shutil.rmtree(output)
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    class_stats: Counter[tuple[str, int]] = Counter()
    source_stats: Counter[tuple[str, str]] = Counter()
    output_paths: dict[int, Path] = {}
    used: Counter[str] = Counter()
    for split in SPLITS:
        for sample in sorted(assignments[split], key=lambda item: (item.source, item.stem)):
            used[sample.stem] += 1
            suffix = "" if used[sample.stem] == 1 else f"_{used[sample.stem]}"
            stem = f"{sample.stem}{suffix}"
            image_path = output / "images" / split / f"{stem}{sample.source_image.suffix.lower()}"
            label_path = output / "labels" / split / f"{stem}.txt"
            shutil.copy2(sample.source_image, image_path)
            label_path.write_text("\n".join(label.yolo() for label in sample.labels) + "\n", encoding="utf-8")
            output_paths[id(sample)] = image_path
            for label in sample.labels:
                class_stats[(split, label.class_id)] += 1
            source_stats[(split, sample.source)] += 1
            manifest.append(
                {
                    "dataset_version": "credible_fine_v1",
                    "split": split,
                    "group_id": sample.group_id,
                    "image": str(image_path.relative_to(output)),
                    "label": str(label_path.relative_to(output)),
                    "source": sample.source,
                    "source_image": str(sample.source_image.resolve()),
                    "family": sample.family,
                    "sha1": sample.sha1,
                    "dhash": f"{sample.dhash:016x}",
                    "width": sample.width,
                    "height": sample.height,
                    "classes": "|".join(str(i) for i in sorted({l.class_id for l in sample.labels})),
                    "object_count": len(sample.labels),
                }
            )

    train_counts = Counter(label.class_id for sample in assignments["train"] for label in sample.labels if label.class_id > 0)
    largest = max(train_counts.values(), default=1)
    balanced: list[str] = []
    balanced_image_dir = output / "images" / "train_balanced"
    balanced_label_dir = output / "labels" / "train_balanced"
    balanced_image_dir.mkdir(parents=True)
    balanced_label_dir.mkdir(parents=True)
    for sample in assignments["train"]:
        defect_classes = {label.class_id for label in sample.labels if label.class_id > 0}
        repeat = max(
            [1, *[min(3, max(1, math.ceil(math.sqrt(largest / train_counts[class_id])))) for class_id in defect_classes]]
        )
        source_image = output_paths[id(sample)]
        source_label = output / "labels" / "train" / f"{source_image.stem}.txt"
        for copy_index in range(repeat):
            stem = source_image.stem if copy_index == 0 else f"{source_image.stem}_repeat{copy_index}"
            image_link = balanced_image_dir / f"{stem}{source_image.suffix.lower()}"
            label_link = balanced_label_dir / f"{stem}.txt"
            link_or_copy(source_image, image_link)
            link_or_copy(source_label, label_link)
            balanced.append(str(image_link.resolve()))
    (output / "train_balanced.txt").write_text("\n".join(balanced) + "\n", encoding="utf-8")

    write_csv(output / "metadata" / "split_manifest.csv", manifest)
    duplicate_rows = []
    for group_id, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        for sample in group:
            duplicate_rows.append(
                {
                    "group_id": group_id,
                    "group_size": len(group),
                    "split": sample.split,
                    "source": sample.source,
                    "family": sample.family,
                    "sha1": sample.sha1,
                    "source_image": str(sample.source_image.resolve()),
                }
            )
    write_csv(output / "metadata" / "duplicate_groups.csv", duplicate_rows)
    stats_rows = [
        {"split": split, "class_id": class_id, "class_name": NAMES[class_id], "objects": class_stats[(split, class_id)]}
        for split in SPLITS for class_id in range(len(NAMES))
    ]
    write_csv(output / "metadata" / "label_stats.csv", stats_rows)

    fingerprint_source = "\n".join(
        f"{row['split']}|{row['group_id']}|{row['sha1']}|{row['classes']}" for row in manifest
    )
    fingerprint = hashlib.sha256(fingerprint_source.encode()).hexdigest()
    metadata = {
        "version": "credible_fine_v1",
        "fingerprint": fingerprint,
        "images": len(manifest),
        "groups": len(groups),
        "class_names": NAMES,
        "split_images": {split: len(assignments[split]) for split in SPLITS},
        "source_images": {
            split: {source: source_stats[(split, source)] for source in sorted({s.source for values in assignments.values() for s in values})}
            for split in SPLITS
        },
    }
    (output / "metadata" / "dataset_fingerprint.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (output / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {output.resolve()}",
                "train: images/train_balanced",
                "val: images/val",
                "test: images/test",
                "",
                "nc: 4",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(NAMES)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output / "data_unbalanced.yaml").write_text(
        "\n".join(
            [
                f"path: {output.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                "nc: 4",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(NAMES)],
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    reviews = load_reviews(args.review_csv.resolve() if args.review_csv else None)
    samples = collect_existing(args.existing.resolve()) + collect_public(args.public_root.resolve())
    samples = apply_reviews(samples, reviews)
    print(f"collected {len(samples)} images")
    fingerprint_samples(samples)
    groups = group_samples(samples, args.phash_distance)
    assignments = stratified_group_split(groups, (args.train, args.val, args.test), args.seed)
    write_dataset(args.output.resolve(), assignments, groups, args.overwrite)
    write_audit(args.output.resolve(), samples, args.audit_size, args.seed, reviews)
    print(f"wrote {len(samples)} images in {len(groups)} groups to {args.output.resolve()}")
    for split in SPLITS:
        counts = Counter(label.class_id for sample in assignments[split] for label in sample.labels)
        print(f"{split}: images={len(assignments[split])} objects={dict(counts)}")


if __name__ == "__main__":
    main()
