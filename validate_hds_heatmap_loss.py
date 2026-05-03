import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from loss import AdaptiveWingLoss, DeepPA_HierarchicalHeatmapLoss


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


def count_full_resolution_predictions(sem_list, target_hm):
    count = 0
    skipped = 0
    for sem in sem_list:
        pred = sem
        if pred.shape[1] != target_hm.shape[1]:
            pred = pred.permute(0, 2, 1).contiguous()
        if pred.shape[2] == target_hm.shape[2]:
            count += 1
        else:
            skipped += 1
    return count, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Validate main/aux HDS heatmap loss with synthetic tensors."
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=8192)
    parser.add_argument("--down-points", type=int, default=2048)
    parser.add_argument("--landmarks", type=int, default=36)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("validation_outputs"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    criterion = DeepPA_HierarchicalHeatmapLoss(AdaptiveWingLoss())

    target_hm = torch.rand(args.batch_size, args.landmarks, args.num_points)
    aux_full_bnl = torch.rand(args.batch_size, args.num_points, args.landmarks)
    aux_down_bnl = torch.rand(args.batch_size, args.down_points, args.landmarks)
    main_final_bln = torch.rand(args.batch_size, args.landmarks, args.num_points)

    old_like_sem_list = [aux_full_bnl, aux_down_bnl]
    patched_sem_list = [aux_full_bnl, aux_down_bnl, main_final_bln]

    old_main, old_aux = criterion(old_like_sem_list, target_hm)
    new_main, new_aux = criterion(patched_sem_list, target_hm)

    stats_list = [
        tensor_stats("target_hm", target_hm),
        tensor_stats("aux_full_bnl", aux_full_bnl),
        tensor_stats("aux_down_bnl", aux_down_bnl),
        tensor_stats("main_final_bln", main_final_bln),
        tensor_stats("old_like_L_main_hm", old_main),
        tensor_stats("old_like_L_aux_hm", old_aux),
        tensor_stats("patched_L_main_hm", new_main),
        tensor_stats("patched_L_aux_hm", new_aux),
    ]

    print("[CHECK] HDS heatmap loss validation")
    for stats in stats_list:
        print_stats(stats)
        assert_clean(stats)

    old_full_count, old_skipped = count_full_resolution_predictions(old_like_sem_list, target_hm)
    new_full_count, new_skipped = count_full_resolution_predictions(patched_sem_list, target_hm)

    print(f"[CHECK] Old-like full-resolution predictions: {old_full_count}")
    print(f"[CHECK] Old-like skipped downsampled predictions: {old_skipped}")
    print(f"[CHECK] Patched full-resolution predictions: {new_full_count}")
    print(f"[CHECK] Patched skipped downsampled predictions: {new_skipped}")
    print(f"[CHECK] Expected patched_L_main_hm > 0: {new_main.item():.8f}")
    print(f"[CHECK] Expected patched_L_aux_hm > 0: {new_aux.item():.8f}")

    if new_main.item() <= 0.0:
        raise AssertionError("patched_L_main_hm is not positive")
    if new_aux.item() <= 0.0:
        raise AssertionError("patched_L_aux_hm is not positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.output_dir / f"hds_heatmap_loss_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    summary = {
        "config": vars(args),
        "tensor_stats": stats_list,
        "old_full_resolution_predictions": old_full_count,
        "old_skipped_downsampled_predictions": old_skipped,
        "patched_full_resolution_predictions": new_full_count,
        "patched_skipped_downsampled_predictions": new_skipped,
    }
    summary["config"]["output_dir"] = str(summary["config"]["output_dir"])
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[CHECK] Save path: {save_path}")
    print("[PASS] HDS heatmap loss validation complete")


if __name__ == "__main__":
    main()
