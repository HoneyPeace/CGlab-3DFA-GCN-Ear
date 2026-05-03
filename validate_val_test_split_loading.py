import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np


def shape_file_name(in_channels, partition):
    if in_channels == 7:
        return f"shape_{partition}.npy"
    if in_channels == 6:
        return f"shape_6ch_{partition}.npy"
    if in_channels == 3:
        return f"shape_3ch_{partition}.npy"
    return f"shape_{partition}.npy"


def split_paths(data_root, data_name, partition, in_channels):
    base_path = Path(data_root) / f"{data_name}-npy"
    return {
        "base": base_path,
        "shape": base_path / shape_file_name(in_channels, partition),
        "heatmap": base_path / f"Heat_data_{partition}.npy",
        "landmark": base_path / f"landmark_{partition}.npy",
    }


def resolve_validation_split(args):
    if args.val_dataset_name:
        return args.val_dataset_name, args.val_partition

    for candidate_name in [args.train_dataset_name, args.test_dataset_name]:
        paths = split_paths(args.data_root, candidate_name, args.val_partition, args.in_channels)
        if paths["shape"].exists() and paths["heatmap"].exists() and paths["landmark"].exists():
            return candidate_name, args.val_partition

    return args.test_dataset_name, "test"


def load_split(label, data_root, data_name, partition, in_channels):
    paths = split_paths(data_root, data_name, partition, in_channels)
    missing = [str(path) for key, path in paths.items() if key != "base" and not path.exists()]
    if missing:
        raise FileNotFoundError(f"{label} split missing files: {missing}")

    shape = np.load(paths["shape"], allow_pickle=True)
    heatmap = np.load(paths["heatmap"], allow_pickle=True)
    landmark = np.load(paths["landmark"], allow_pickle=True)

    summary = {
        "label": label,
        "data_name": data_name,
        "partition": partition,
        "shape_path": str(paths["shape"]),
        "heatmap_path": str(paths["heatmap"]),
        "landmark_path": str(paths["landmark"]),
        "num_samples": int(shape.shape[0]),
        "shape_shape": list(shape.shape),
        "heatmap_shape": list(heatmap.shape),
        "landmark_shape": list(landmark.shape),
        "shape_nan_count": int(np.isnan(shape).sum()),
        "heatmap_nan_count": int(np.isnan(heatmap).sum()),
        "landmark_nan_count": int(np.isnan(landmark).sum()),
        "shape_min": float(np.nanmin(shape)),
        "shape_max": float(np.nanmax(shape)),
        "heatmap_min": float(np.nanmin(heatmap)),
        "heatmap_max": float(np.nanmax(heatmap)),
        "landmark_min": float(np.nanmin(landmark)),
        "landmark_max": float(np.nanmax(landmark)),
    }
    return summary


def print_summary(summary):
    print(f"[CHECK] {summary['label']} data/partition: {summary['data_name']} / {summary['partition']}")
    print(f"[CHECK] {summary['label']} samples: {summary['num_samples']}")
    print(f"[CHECK] {summary['label']} shape: {summary['shape_shape']}")
    print(f"[CHECK] {summary['label']} heatmap: {summary['heatmap_shape']}")
    print(f"[CHECK] {summary['label']} landmark: {summary['landmark_shape']}")
    print(
        f"[CHECK] {summary['label']} NaN counts: "
        f"shape={summary['shape_nan_count']}, "
        f"heatmap={summary['heatmap_nan_count']}, "
        f"landmark={summary['landmark_nan_count']}"
    )
    print(
        f"[CHECK] {summary['label']} min/max: "
        f"shape={summary['shape_min']}/{summary['shape_max']}, "
        f"heatmap={summary['heatmap_min']}/{summary['heatmap_max']}, "
        f"landmark={summary['landmark_min']}/{summary['landmark_max']}"
    )


def main():
    parser = argparse.ArgumentParser(description="Validate train/val/test NPY split loading.")
    parser.add_argument("--data-root", type=str, default="../data")
    parser.add_argument("--train-dataset-name", type=str, default="train")
    parser.add_argument("--val-dataset-name", type=str, default="valiation")
    parser.add_argument("--val-partition", type=str, default="val")
    parser.add_argument("--test-dataset-name", type=str, default="test")
    parser.add_argument("--in-channels", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("validation_outputs"))
    args = parser.parse_args()

    val_dataset_name, val_partition = resolve_validation_split(args)

    summaries = [
        load_split("train", args.data_root, args.train_dataset_name, "train", args.in_channels),
        load_split("val", args.data_root, val_dataset_name, val_partition, args.in_channels),
        load_split("test", args.data_root, args.test_dataset_name, "test", args.in_channels),
    ]

    print("[CHECK] Train/Val/Test split loading validation")
    for summary in summaries:
        print_summary(summary)
        if summary["shape_nan_count"] != 0 or summary["heatmap_nan_count"] != 0 or summary["landmark_nan_count"] != 0:
            raise AssertionError(f"{summary['label']} split contains NaN values")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.output_dir / f"val_test_split_loading_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    with save_path.open("w", encoding="utf-8") as f:
        json.dump({"splits": summaries}, f, indent=2)

    print(f"[CHECK] Save path: {save_path}")
    print("[PASS] Train/Val/Test split loading validation complete")


if __name__ == "__main__":
    main()
