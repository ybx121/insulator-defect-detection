#!/usr/bin/env python3
"""Download and extract the CC BY 4.0 Supervisely insulator dataset."""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path


ARCHIVES = {
    "Train": "https://github.com/supervisely-ecosystem/aerial-power-infrastructure-detection-train-dataset/releases/download/v0.9.0/Train.tar",
    "Test": "https://github.com/supervisely-ecosystem/aerial-power-infrastructure-detection-test-dataset/releases/download/v0.9.0/Test.tar",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("datasets/raw/supervisely"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            target = (destination / member.name.lstrip("/\\")).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"Unsafe archive member: {member.name}")
        handle.extractall(destination, filter="data")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, url in ARCHIVES.items():
        archive = args.output / f"{name}.tar"
        destination = args.output / name
        if args.force or not archive.exists():
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, archive)
        marker = destination / "meta.json"
        if args.force or not marker.exists():
            destination.mkdir(parents=True, exist_ok=True)
            safe_extract(archive, destination)
        print(f"ready: {destination}")


if __name__ == "__main__":
    main()
