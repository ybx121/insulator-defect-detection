#!/usr/bin/env python3
"""Run the reproducible YOLO11s augmentation screen and summarize AP50."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "detect" / "runs"
EVAL_ROOT = ROOT / "runs" / "eval"
LOG_ROOT = ROOT / "runs" / "logs"
DATA = ROOT / "datasets" / "credible_fine_v1" / "data_unbalanced.yaml"
EXPERIMENTS = {
    "moderate": ("credible_v1_yolo11s_960_aug_moderate_40", 40),
    "none": ("credible_v1_yolo11s_960_aug_none_40", 40),
    "defect_safe": ("credible_v1_yolo11s_960_aug_defect_safe_40", 40),
}


def completed_epochs(run_name: str) -> int:
    path = RUN_ROOT / run_name / "results.csv"
    if not path.exists():
        return 0
    return max(0, len(path.read_text(encoding="utf-8").splitlines()) - 1)


def run_logged(command: list[str], log_name: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / log_name
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def train(preset: str, run_name: str, epochs: int) -> None:
    if completed_epochs(run_name) >= epochs and (RUN_ROOT / run_name / "weights" / "best.pt").exists():
        print(f"skip completed training: {run_name}")
        return
    command = [
        sys.executable,
        str(ROOT / "train.py"),
        "--model", "yolo11s.pt",
        "--data", str(DATA),
        "--imgsz", "960",
        "--epochs", str(epochs),
        "--batch", "16",
        "--device", "0",
        "--workers", "8",
        "--seed", "20260708",
        "--augment-preset", preset,
        "--close-mosaic", "15",
        "--patience", "20",
        "--name", run_name,
    ]
    run_logged(command, f"{run_name}.log")


def evaluate(preset: str, run_name: str) -> Path:
    output = EVAL_ROOT / f"augmentation_{preset}_val.json"
    if output.exists():
        print(f"skip completed evaluation: {output.name}")
        return output
    weight_dir = RUN_ROOT / run_name / "weights"
    weights = weight_dir / "best_map50.pt"
    if not weights.exists():
        weights = weight_dir / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights: {weights}")
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_detector.py"),
        "--weights", str(weights),
        "--data", str(DATA),
        "--split", "val",
        "--mode", "standard",
        "--imgsz", "960",
        "--bootstrap", "0",
        "--device", "0",
        "--output", str(output),
        "--leaderboard", str(EVAL_ROOT / "credible_v1_leaderboard.csv"),
    ]
    run_logged(command, f"augmentation_{preset}_eval.log")
    return output


def summarize(reports: dict[str, Path]) -> None:
    rows = {}
    for preset, path in reports.items():
        report = json.loads(path.read_text(encoding="utf-8"))
        rows[preset] = {
            "map50": report["metrics"]["map50"],
            "map50_95_secondary": report["metrics"]["map50_95"],
            "per_class_map50": {
                name: values["map50"] for name, values in report["per_class"].items()
            },
        }
    winner = max(rows, key=lambda name: rows[name]["map50"])
    result = {
        "primary_metric": "map50",
        "target": 0.95,
        "winner": winner,
        "target_achieved_on_validation": rows[winner]["map50"] > 0.95,
        "results": rows,
    }
    output = EVAL_ROOT / "augmentation_ablation_summary.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main() -> None:
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    reports = {}
    for preset, (run_name, epochs) in EXPERIMENTS.items():
        train(preset, run_name, epochs)
        reports[preset] = evaluate(preset, run_name)
    summarize(reports)


if __name__ == "__main__":
    main()
