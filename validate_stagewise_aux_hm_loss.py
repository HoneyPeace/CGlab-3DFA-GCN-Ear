import argparse
import json
import os
import time

import torch

from loss import AdaptiveWingLoss, DeepPA_HierarchicalHeatmapLoss


def tensor_stats(name, tensor):
    detached = tensor.detach()
    return {
        "name": name,
        "shape": list(detached.shape),
        "nan_count": int(torch.isnan(detached).sum().item()),
        "min": float(detached.min().item()),
        "max": float(detached.max().item()),
    }


def make_stage_indices(batch_size, full_points, stage_points):
    indices = []
    for n_points in stage_points:
        idx = torch.linspace(0, full_points - 1, steps=n_points).long()
        indices.append(idx.view(1, n_points).repeat(batch_size, 1))
    return indices


def main():
    parser = argparse.ArgumentParser(description="Validate stage-wise auxiliary heatmap loss shapes.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--landmarks", type=int, default=36)
    parser.add_argument("--num_points", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="validation_outputs")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    stage_points = [args.num_points, 2048, 512, 128]
    criterion = DeepPA_HierarchicalHeatmapLoss(AdaptiveWingLoss())

    target_hm = torch.rand(args.batch_size, args.landmarks, args.num_points)
    sem_list = [
        torch.rand(args.batch_size, args.num_points, args.landmarks),
        torch.rand(args.batch_size, 2048, args.landmarks),
        torch.rand(args.batch_size, 512, args.landmarks),
        torch.rand(args.batch_size, 128, args.landmarks),
        torch.rand(args.batch_size, args.landmarks, args.num_points),
    ]
    stage_indices = make_stage_indices(args.batch_size, args.num_points, stage_points)

    old_main, old_aux = criterion(sem_list, target_hm)
    new_main, new_aux = criterion(sem_list, target_hm, stage_indices)

    gathered_targets = [
        criterion._gather_target_hm(target_hm, idx)
        for idx in stage_indices
    ]

    stats = [
        tensor_stats("target_hm", target_hm),
        tensor_stats("old_main_loss", old_main),
        tensor_stats("old_aux_loss", old_aux),
        tensor_stats("stagewise_main_loss", new_main),
        tensor_stats("stagewise_aux_loss", new_aux),
    ]
    for stage_id, pred in enumerate(sem_list[:-1]):
        stats.append(tensor_stats(f"aux_pred_stage{stage_id}", pred))
        stats.append(tensor_stats(f"target_stage{stage_id}", gathered_targets[stage_id]))

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(
        args.save_dir,
        f"stagewise_aux_hm_loss_{time.strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("[CHECK] Stage point counts:", stage_points)
    print("[CHECK] Old aux loss uses full-resolution aux only:", f"{old_aux.item():.8f}")
    print("[CHECK] Stage-wise aux loss uses all aux stages:", f"{new_aux.item():.8f}")
    print("[CHECK] Saved stats:", save_path)

    if new_aux.item() <= 0.0:
        raise AssertionError("Stage-wise auxiliary loss should be positive for random tensors.")
    if any(item["nan_count"] > 0 for item in stats):
        raise AssertionError("NaN detected in validation tensors.")


if __name__ == "__main__":
    main()
