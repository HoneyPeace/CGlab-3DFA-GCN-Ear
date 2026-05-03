import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from loss import get_differentiable_coords


def tensor_stats(name, tensor):
    detached = tensor.detach()
    finite = torch.isfinite(detached)
    return {
        "name": name,
        "shape": list(detached.shape),
        "nan_count": int(torch.isnan(detached).sum().item()),
        "inf_count": int(torch.isinf(detached).sum().item()),
        "min": float(detached[finite].min().item()) if finite.any() else None,
        "max": float(detached[finite].max().item()) if finite.any() else None,
    }


def print_stats(stats):
    print(f"[CHECK] {stats['name']} shape: {stats['shape']}")
    print(f"[CHECK] {stats['name']} NaN count: {stats['nan_count']}")
    print(f"[CHECK] {stats['name']} Inf count: {stats['inf_count']}")
    print(f"[CHECK] {stats['name']} Min / Max: {stats['min']} / {stats['max']}")


def assert_clean(stats):
    if stats["nan_count"] != 0:
        raise AssertionError(f"{stats['name']} contains NaN values")
    if stats["inf_count"] != 0:
        raise AssertionError(f"{stats['name']} contains Inf values")


def main():
    parser = argparse.ArgumentParser(description="Validate final coordinate extraction from heatmap.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=32)
    parser.add_argument("--landmarks", type=int, default=4)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("validation_outputs"))
    args = parser.parse_args()

    torch.manual_seed(7)
    points = torch.randn(args.batch_size, args.num_points, 3)
    heatmap = torch.zeros(args.batch_size, args.landmarks, args.num_points)

    peak_idx = torch.zeros(args.batch_size, args.landmarks, dtype=torch.long)
    for b in range(args.batch_size):
        for l in range(args.landmarks):
            idx = (b * args.landmarks + l) % args.num_points
            peak_idx[b, l] = idx
            heatmap[b, l, idx] = 1.0

    pred_coords = get_differentiable_coords(points, heatmap, k=args.k)
    expected_coords = torch.gather(
        points.unsqueeze(1).expand(-1, args.landmarks, -1, -1),
        2,
        peak_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3),
    ).squeeze(2)

    max_abs_diff = torch.max(torch.abs(pred_coords - expected_coords)).item()

    stats_list = [
        tensor_stats("points", points),
        tensor_stats("heatmap", heatmap),
        tensor_stats("pred_coords", pred_coords),
        tensor_stats("expected_coords", expected_coords),
    ]

    print("[CHECK] Coordinate-from-heatmap validation")
    for stats in stats_list:
        print_stats(stats)
        assert_clean(stats)

    print(f"[CHECK] Expected pred_coords shape: {[args.batch_size, args.landmarks, 3]}")
    print(f"[CHECK] Actual pred_coords shape: {list(pred_coords.shape)}")
    print(f"[CHECK] Max |pred_coords - peak_point|: {max_abs_diff:.8f}")
    print(f"[CHECK] Save folder: {args.output_dir}")

    if list(pred_coords.shape) != [args.batch_size, args.landmarks, 3]:
        raise AssertionError("pred_coords shape mismatch")
    if max_abs_diff > 1e-6:
        raise AssertionError("coord extraction does not match heatmap peak when k=1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.output_dir / f"coord_from_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    summary = {
        "config": {
            "batch_size": args.batch_size,
            "num_points": args.num_points,
            "landmarks": args.landmarks,
            "k": args.k,
            "output_dir": str(args.output_dir),
        },
        "max_abs_diff": max_abs_diff,
        "tensor_stats": stats_list,
    }
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[CHECK] Save path: {save_path}")
    print("[PASS] Coordinate-from-heatmap validation complete")


if __name__ == "__main__":
    main()
