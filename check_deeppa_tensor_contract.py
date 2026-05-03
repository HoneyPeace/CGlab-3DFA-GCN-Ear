import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn


def tensor_stats(name, tensor):
    detached = tensor.detach()
    finite = torch.isfinite(detached)
    stats = {
        "name": name,
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "nan_count": int(torch.isnan(detached).sum().item()),
        "inf_count": int(torch.isinf(detached).sum().item()),
        "min": float(detached[finite].min().item()) if finite.any() else None,
        "max": float(detached[finite].max().item()) if finite.any() else None,
    }
    return stats


def print_stats(stats):
    print(f"[CHECK] {stats['name']} shape: {stats['shape']}")
    print(f"[CHECK] {stats['name']} dtype/device: {stats['dtype']} / {stats['device']}")
    print(f"[CHECK] {stats['name']} NaN count: {stats['nan_count']}")
    print(f"[CHECK] {stats['name']} Inf count: {stats['inf_count']}")
    print(f"[CHECK] {stats['name']} Min / Max: {stats['min']} / {stats['max']}")


def assert_clean(stats):
    if stats["nan_count"] != 0:
        raise AssertionError(f"{stats['name']} contains NaN values")
    if stats["inf_count"] != 0:
        raise AssertionError(f"{stats['name']} contains Inf values")


def knn_indices(xyz, k):
    # xyz: [B, 3, N]
    points = xyz.transpose(1, 2).contiguous()
    dist = torch.cdist(points, points)
    return torch.topk(dist, k=k, dim=-1, largest=False)[1]


def gather_neighbors(features, idx):
    # features: [B, C, N], idx: [B, N, K] -> [B, C, N, K]
    b, c, n = features.shape
    k = idx.shape[-1]
    expanded = features.transpose(1, 2).contiguous()
    batch_idx = torch.arange(b, device=features.device).view(b, 1, 1).expand(b, n, k)
    gathered = expanded[batch_idx, idx, :]
    return gathered.permute(0, 3, 1, 2).contiguous()


def build_edge_features(features, k):
    # features: [B, C, N] -> edge: [B, 2C, N, K]
    idx = knn_indices(features[:, :3, :].contiguous(), k)
    neighbor = gather_neighbors(features, idx)
    center = features.unsqueeze(-1).expand_as(neighbor)
    return torch.cat([center, neighbor - center], dim=1)


class SimpleAdditionFusion(nn.Module):
    def __init__(self, dim, prior_dim):
        super().__init__()
        self.prior_proj = nn.Sequential(
            nn.Linear(prior_dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.GELU(),
        )

    def forward(self, x, prior):
        # x: [B, N, D], prior: [B, N, P]
        b, n, _ = x.shape
        p_feat = self.prior_proj(prior.reshape(b * n, -1)).reshape(b, n, -1)
        return x + p_feat


class InteractionResidualFusion(nn.Module):
    def __init__(self, dim, prior_dim):
        super().__init__()
        self.prior_proj = nn.Sequential(
            nn.Linear(prior_dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.GELU(),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.GELU(),
        )

    def forward(self, x, prior):
        # x: [B, N, D], prior: [B, N, P]
        b, n, _ = x.shape
        p_feat = self.prior_proj(prior.reshape(b * n, -1)).reshape(b, n, -1)
        fused = torch.cat([x, p_feat], dim=-1)
        mixed = self.fusion_mlp(fused.reshape(b * n, -1)).reshape(b, n, -1)
        return x + mixed


def maybe_write_summary(summary, output_dir):
    if output_dir is None:
        print("[CHECK] Save path: no file written; pass --summary-output-dir to write a summary.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = output_dir / f"deeppa_tensor_contract_{timestamp}.json"
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[CHECK] Save path: {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Validate DeepPA/PAConv tensor contracts with synthetic tensors.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=128)
    parser.add_argument("--in-channels", type=int, default=7)
    parser.add_argument("--landmarks", type=int, default=36)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=1344)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--summary-output-dir", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print("[CHECK] Starting DeepPA tensor contract validation")
    print(f"[CHECK] Synthetic input: B={args.batch_size}, C={args.in_channels}, N={args.num_points}")

    points = torch.randn(args.batch_size, args.in_channels, args.num_points)
    base_features = torch.randn(args.batch_size, args.num_points, args.feature_dim)
    prior_heatmap = torch.rand(args.batch_size, args.num_points, args.landmarks)
    prior_latent = torch.randn(args.batch_size, args.num_points, args.latent_dim)

    edge = build_edge_features(points, args.k)
    simple_heatmap = SimpleAdditionFusion(args.feature_dim, args.landmarks)
    legacy_heatmap = InteractionResidualFusion(args.feature_dim, args.landmarks)
    simple_latent = SimpleAdditionFusion(args.feature_dim, args.latent_dim)

    simple_heatmap_out = simple_heatmap(base_features, prior_heatmap)
    legacy_heatmap_out = legacy_heatmap(base_features, prior_heatmap)
    simple_latent_out = simple_latent(base_features, prior_latent)

    summaries = [
        tensor_stats("input_points", points),
        tensor_stats("edge_features", edge),
        tensor_stats("prior_heatmap", prior_heatmap),
        tensor_stats("prior_latent", prior_latent),
        tensor_stats("simple_heatmap_fusion_out", simple_heatmap_out),
        tensor_stats("legacy_interaction_fusion_out", legacy_heatmap_out),
        tensor_stats("simple_latent_fusion_out", simple_latent_out),
    ]

    for stats in summaries:
        print_stats(stats)
        assert_clean(stats)

    expected_edge_shape = [args.batch_size, args.in_channels * 2, args.num_points, args.k]
    print(f"[CHECK] Expected edge shape: {expected_edge_shape}")
    print(f"[CHECK] Actual edge shape: {list(edge.shape)}")
    if list(edge.shape) != expected_edge_shape:
        raise AssertionError("edge feature shape mismatch")

    expected_fusion_shape = [args.batch_size, args.num_points, args.feature_dim]
    for name, tensor in [
        ("simple_heatmap_fusion_out", simple_heatmap_out),
        ("legacy_interaction_fusion_out", legacy_heatmap_out),
        ("simple_latent_fusion_out", simple_latent_out),
    ]:
        print(f"[CHECK] Expected {name}: {expected_fusion_shape}")
        print(f"[CHECK] Actual {name}: {list(tensor.shape)}")
        if list(tensor.shape) != expected_fusion_shape:
            raise AssertionError(f"{name} shape mismatch")

    diff_simple_vs_legacy = torch.mean(torch.abs(simple_heatmap_out - legacy_heatmap_out)).item()
    diff_heatmap_vs_latent = torch.mean(torch.abs(simple_heatmap_out - simple_latent_out)).item()
    print(f"[CHECK] Mean |simple_heatmap - legacy_interaction|: {diff_simple_vs_legacy:.6f}")
    print(f"[CHECK] Mean |simple_heatmap - simple_latent|: {diff_heatmap_vs_latent:.6f}")

    summary = {
        "config": vars(args),
        "tensor_stats": summaries,
        "expected_edge_shape": expected_edge_shape,
        "expected_fusion_shape": expected_fusion_shape,
        "mean_abs_diff_simple_vs_legacy": diff_simple_vs_legacy,
        "mean_abs_diff_heatmap_vs_latent": diff_heatmap_vs_latent,
    }
    maybe_write_summary(summary, args.summary_output_dir)
    print("[PASS] DeepPA tensor contract validation complete")


if __name__ == "__main__":
    main()
