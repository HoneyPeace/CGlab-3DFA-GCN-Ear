import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn


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
    parser = argparse.ArgumentParser(
        description="Validate DeepPA/prior residual fusion identity path without datasets/checkpoints."
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=64)
    parser.add_argument("--stage-dim", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("validation_outputs"))
    args = parser.parse_args()

    torch.manual_seed(7)
    B, N, C = args.batch_size, args.num_points, args.stage_dim
    x = torch.randn(B, N, C)
    p_feat = torch.randn(B, N, C)

    fusion_mlp = nn.Sequential(
        nn.Linear(C * 2, C, bias=False),
        nn.BatchNorm1d(C),
        nn.GELU(),
    )
    fused = torch.cat([x, p_feat], dim=-1)
    mixed_residual = fusion_mlp(fused.view(-1, fused.shape[-1])).view(B, N, C)

    deeppa_base_out = x + mixed_residual
    prior_base_out = p_feat + mixed_residual
    identity_delta = prior_base_out - deeppa_base_out
    expected_delta = p_feat - x
    max_delta_error = torch.max(torch.abs(identity_delta - expected_delta)).item()

    stats_list = [
        tensor_stats("x_deeppa_feature", x),
        tensor_stats("p_feat_prior_feature", p_feat),
        tensor_stats("mixed_residual", mixed_residual),
        tensor_stats("deeppa_base_out", deeppa_base_out),
        tensor_stats("prior_base_out", prior_base_out),
    ]
    for stats in stats_list:
        print_stats(stats)
        assert_clean(stats)

    print(f"[CHECK] identity delta max abs error: {max_delta_error:.8f}")
    if max_delta_error > 1e-6:
        raise AssertionError("Prior-base fusion does not only swap the residual identity path")
    if list(prior_base_out.shape) != [B, N, C]:
        raise AssertionError("Prior-base output shape mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.output_dir / (
        f"prior_residual_fusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    report = {
        "batch_size": B,
        "num_points": N,
        "stage_dim": C,
        "max_delta_error": max_delta_error,
        "stats": stats_list,
        "save_path": str(save_path),
    }
    save_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[CHECK] Saved report: {save_path}")
    print("[PASS] Prior residual fusion harness passed.")


if __name__ == "__main__":
    main()
