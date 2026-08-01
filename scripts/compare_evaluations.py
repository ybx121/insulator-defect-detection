#!/usr/bin/env python3
"""Apply the validation promotion gate to two evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFECT_CLASSES = ["broken_shell", "flashover_pollution", "missing_disc_drop"]
TARGET_METRIC = "map50"
TARGET_THRESHOLD = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--minimum-overall-gain", type=float, default=0.02)
    parser.add_argument("--maximum-class-drop", type=float, default=0.03)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def compare_reports(
    baseline: dict[str, object],
    candidate: dict[str, object],
    minimum_overall_gain: float = 0.02,
    maximum_class_drop: float = 0.03,
) -> dict[str, object]:
    identity = ("dataset_fingerprint", "split")
    for key in identity:
        if baseline[key] != candidate[key]:
            raise ValueError(f"Cannot compare different {key}: {baseline[key]} != {candidate[key]}")
    overall_gain = candidate["metrics"][TARGET_METRIC] - baseline["metrics"][TARGET_METRIC]
    class_deltas = {
        name: candidate["per_class"][name][TARGET_METRIC] - baseline["per_class"][name][TARGET_METRIC]
        for name in DEFECT_CLASSES
    }
    promoted = overall_gain >= minimum_overall_gain and min(class_deltas.values()) >= -maximum_class_drop
    ap_gap = candidate["metrics"]["map50"] - candidate["metrics"]["map75"]
    return {
        "promoted": promoted,
        "promotion_metric": TARGET_METRIC,
        "overall_gain": overall_gain,
        "per_defect_class_delta": class_deltas,
        "target": {
            "metric": TARGET_METRIC,
            "threshold": TARGET_THRESHOLD,
            "candidate_value": candidate["metrics"][TARGET_METRIC],
            "achieved": candidate["metrics"][TARGET_METRIC] > TARGET_THRESHOLD,
        },
        "localization_gap_map50_minus_map75": ap_gap,
        "recommendation": (
            "evaluate_localization_model" if ap_gap >= 0.15 else
            "prioritize_label_and_classification_review"
        ),
    }


def main() -> None:
    args = parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    result = compare_reports(
        baseline,
        candidate,
        minimum_overall_gain=args.minimum_overall_gain,
        maximum_class_drop=args.maximum_class_drop,
    )
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    raise SystemExit(0 if result["promoted"] else 2)


if __name__ == "__main__":
    main()
