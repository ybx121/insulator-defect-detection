#!/usr/bin/env python3
"""Run every full-image detector on Dataset/labels as a resumable queue."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "configs" / "dataset_labels_train_matrix.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument(
        "--only",
        nargs="*",
        help="Optional job names. By default the complete matrix is run.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry jobs recorded as failed instead of leaving them failed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without starting training.",
    )
    return parser.parse_args()


def load_matrix(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Matrix must be a mapping: {resolved}")
    return data


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_logged(command: list[str], log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n[{utc_timestamp()}] COMMAND: {subprocess.list2cmdline(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return process.wait()


def selected(name: str, only: set[str] | None) -> bool:
    return only is None or name in only


def yolo_command(matrix: dict[str, Any], job: dict[str, Any]) -> list[str]:
    common = matrix["common"]
    command = [
        str(Path(matrix["python"])),
        str(ROOT / "train.py"),
        "--model",
        str(job["model"]),
        "--data",
        str(matrix["dataset"]),
        "--epochs",
        str(job.get("epochs", common["epochs"])),
        "--imgsz",
        str(job["imgsz"]),
        "--batch",
        str(job["batch"]),
        "--device",
        str(common["device"]),
        "--workers",
        str(common["workers"]),
        "--seed",
        str(matrix["seed"]),
        "--patience",
        str(common["patience"]),
        "--augment-preset",
        str(common["augment_preset"]),
        "--project",
        str(matrix["project"]),
        "--name",
        str(job["name"]),
        "--exist-ok",
    ]
    if job.get("weights"):
        command.extend(["--weights", str(job["weights"])])
    return command


def prepare_dfine(matrix: dict[str, Any], dry_run: bool) -> list[dict[str, Any]]:
    dfine = matrix["dfine"]
    python = str(Path(matrix["python"]))
    jobs = list(dfine.get("jobs", []))
    for index, job in enumerate(jobs):
        command = [
            python,
            str(ROOT / "scripts" / "prepare_dfine.py"),
            "--dataset",
            str(dfine["dataset"]),
            "--output",
            str(dfine["coco_output"]),
            "--dfine-root",
            str(dfine["root"]),
            "--project-root",
            str(ROOT),
            "--imgsz",
            str(dfine["imgsz"]),
            "--epochs",
            str(dfine["epochs"]),
            "--batch",
            str(job["batch"]),
            "--model-size",
            str(job["model_size"]),
            "--pretrained-variant",
            str(job["pretrained_variant"]),
            "--config-name",
            str(job["config_name"]),
            "--run-name",
            str(job["name"]),
        ]
        if index == 0:
            command.append("--overwrite")
        if dry_run:
            print(subprocess.list2cmdline(command))
            continue
        if index > 0:
            # The COCO export is shared. Temporarily move it out of the way so
            # prepare_dfine can write another model config without deleting it.
            output = (ROOT / dfine["coco_output"]).resolve()
            parked = output.with_name(output.name + ".prepared")
            if parked.exists():
                raise FileExistsError(parked)
            output.replace(parked)
            try:
                result = subprocess.run(command, cwd=ROOT, check=False)
            finally:
                if output.exists():
                    import shutil

                    shutil.rmtree(output)
                parked.replace(output)
        else:
            result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)
    return jobs


def dfine_command(matrix: dict[str, Any], job: dict[str, Any]) -> list[str]:
    dfine_root = (ROOT / matrix["dfine"]["root"]).resolve()
    config = dfine_root / "configs" / "dfine" / "custom" / job["config_name"]
    pretrained = dfine_root / "weights" / f"dfine_{job['model_size']}_{job['pretrained_variant']}.pth"
    return [
        str(Path(matrix["python"])),
        "train.py",
        "-c",
        str(config),
        "-t",
        str(pretrained),
        "--use-amp",
        "--seed",
        str(matrix["seed"]),
    ]


def main() -> None:
    args = parse_args()
    matrix = load_matrix(args.matrix)
    only = set(args.only) if args.only else None
    all_names = {
        *(str(job["name"]) for job in matrix.get("yolo_jobs", [])),
        *(str(job["name"]) for job in matrix.get("dfine", {}).get("jobs", [])),
    }
    unknown = set() if only is None else only - all_names
    if unknown:
        raise SystemExit(f"Unknown jobs: {', '.join(sorted(unknown))}")

    if args.dry_run:
        for job in matrix.get("yolo_jobs", []):
            if selected(str(job["name"]), only):
                print(subprocess.list2cmdline(yolo_command(matrix, job)))
        dfine_jobs = [
            job
            for job in matrix.get("dfine", {}).get("jobs", [])
            if selected(str(job["name"]), only)
        ]
        if dfine_jobs:
            prepare_dfine(
                {**matrix, "dfine": {**matrix["dfine"], "jobs": dfine_jobs}}, True
            )
        for job in dfine_jobs:
            print(subprocess.list2cmdline(dfine_command(matrix, job)))
        return

    queue_root = (ROOT / "runs" / "dataset_labels_retrain").resolve()
    status_path = queue_root / "status.json"
    logs = queue_root / "logs"
    status: dict[str, Any] = {
        "matrix": str(args.matrix.resolve()),
        "dataset": str((ROOT / matrix["dataset"]).resolve()),
        "pid": os.getpid(),
        "started_at": utc_timestamp(),
        "updated_at": utc_timestamp(),
        "state": "running",
        "jobs": {},
    }
    if status_path.exists():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        previous_jobs = previous.get("jobs", {})
        if previous_jobs:
            status["started_at"] = previous.get("started_at", status["started_at"])
            status["jobs"] = previous_jobs
    write_json(status_path, status)

    failures = 0
    for job in matrix.get("yolo_jobs", []):
        name = str(job["name"])
        if not selected(name, only):
            continue
        previous_state = status["jobs"].get(name, {}).get("state")
        if previous_state == "complete" or (previous_state == "failed" and not args.retry_failed):
            continue
        command = yolo_command(matrix, job)
        status["jobs"][name] = {"state": "running", "started_at": utc_timestamp()}
        status["updated_at"] = utc_timestamp()
        write_json(status_path, status)
        started = time.monotonic()
        code = run_logged(command, logs / f"{name}.log", ROOT)
        status["jobs"][name] = {
            "state": "complete" if code == 0 else "failed",
            "started_at": status["jobs"][name]["started_at"],
            "finished_at": utc_timestamp(),
            "elapsed_seconds": time.monotonic() - started,
            "returncode": code,
        }
        status["updated_at"] = utc_timestamp()
        write_json(status_path, status)
        failures += int(code != 0)

    dfine_jobs = [job for job in matrix.get("dfine", {}).get("jobs", []) if selected(str(job["name"]), only)]
    if dfine_jobs:
        prepare_dfine({**matrix, "dfine": {**matrix["dfine"], "jobs": dfine_jobs}}, False)

    for job in dfine_jobs:
        name = str(job["name"])
        previous_state = status["jobs"].get(name, {}).get("state")
        if previous_state == "complete" or (previous_state == "failed" and not args.retry_failed):
            continue
        command = dfine_command(matrix, job)
        status["jobs"][name] = {"state": "running", "started_at": utc_timestamp()}
        status["updated_at"] = utc_timestamp()
        write_json(status_path, status)
        started = time.monotonic()
        dfine_cwd = (ROOT / matrix["dfine"]["root"]).resolve()
        code = run_logged(command, logs / f"{name}.log", dfine_cwd)
        status["jobs"][name] = {
            "state": "complete" if code == 0 else "failed",
            "started_at": status["jobs"][name]["started_at"],
            "finished_at": utc_timestamp(),
            "elapsed_seconds": time.monotonic() - started,
            "returncode": code,
        }
        status["updated_at"] = utc_timestamp()
        write_json(status_path, status)
        failures += int(code != 0)

    status["state"] = "complete" if failures == 0 else "complete_with_failures"
    status["finished_at"] = utc_timestamp()
    status["updated_at"] = utc_timestamp()
    write_json(status_path, status)
    if failures:
        raise SystemExit(f"{failures} training job(s) failed; see {logs}")


if __name__ == "__main__":
    main()
