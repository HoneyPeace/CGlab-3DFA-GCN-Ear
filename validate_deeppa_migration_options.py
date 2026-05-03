import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from loss_controller import DeepPALossController
from PAConv.util.PAConv_util import get_graph_feature


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


def validate_center_geometry_edge(batch_size, num_points, k):
    points = torch.randn(batch_size, 7, num_points)
    points[:, 3:, :] = torch.arange(num_points).float().view(1, 1, num_points)

    edge = get_graph_feature(points, k=k)
    center_geom_from_edge = edge[:, 10:14, :, :]
    expected_center_geom = points[:, 3:, :].unsqueeze(-1).expand_as(center_geom_from_edge)
    max_abs_diff = torch.max(torch.abs(center_geom_from_edge - expected_center_geom)).item()

    print(f"[CHECK] PAConv 7ch edge shape: {list(edge.shape)}")
    print(f"[CHECK] Center eigen/curv max abs diff: {max_abs_diff:.8f}")
    if list(edge.shape) != [batch_size, 14, num_points, k]:
        raise AssertionError("PAConv 7ch edge shape mismatch")
    if max_abs_diff > 1e-6:
        raise AssertionError("PAConv 7ch edge does not preserve center eigen/curv channels")
    return edge, max_abs_diff


def validate_loss_controller(use_rlw):
    device = torch.device("cpu")
    L_main = torch.tensor(0.5, device=device)
    L_aux = torch.tensor(0.25, device=device)
    L_crd = torch.tensor(1.0, device=device)
    L_srf = torch.tensor(2.0, device=device)
    L_str = torch.tensor(3.0, device=device)

    controller = DeepPALossController(
        patience=5,
        base_hds=0.1,
        decay_step=0.05,
        min_heatmap_warmup=30,
        use_rlw_for_pred=use_rlw,
    )
    warmup_loss, warmup_weights = controller.compute_loss(
        "deeppa_frozen", 0, L_main, L_aux, L_crd, L_srf, L_str
    )

    controller.val_decay = 0.7
    transition_loss, transition_weights = controller.compute_loss(
        "deeppa_frozen", 30, L_main, L_aux, L_crd, L_srf, L_str
    )

    print(f"[CHECK] Warmup weights: {warmup_weights}")
    print(f"[CHECK] Transition weights: {transition_weights}")
    print(f"[CHECK] Warmup loss: {warmup_loss.item():.8f}")
    print(f"[CHECK] Transition loss: {transition_loss.item():.8f}")

    if warmup_weights["w_main"] != 1.0 or warmup_weights["w_geom"] != 0.0:
        raise AssertionError("Warmup must keep main heatmap weight at 1.0 and geometry at 0.0")
    if transition_weights["w_main"] != 0.7 or abs(transition_weights["w_geom"] - 0.3) > 1e-6:
        raise AssertionError("Transition weights do not follow val_decay")

    return warmup_loss, transition_loss, warmup_weights, transition_weights


def main():
    parser = argparse.ArgumentParser(description="Validate DeepPA migration options without datasets/checkpoints.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=64)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--use-rlw", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("validation_outputs"))
    args = parser.parse_args()

    torch.manual_seed(7)
    print("[CHECK] DeepPA migration option validation")

    edge, center_geom_diff = validate_center_geometry_edge(args.batch_size, args.num_points, args.k)
    warmup_loss, transition_loss, warmup_weights, transition_weights = validate_loss_controller(args.use_rlw)

    stats_list = [
        tensor_stats("edge_features", edge),
        tensor_stats("warmup_loss", warmup_loss),
        tensor_stats("transition_loss", transition_loss),
    ]
    for stats in stats_list:
        print_stats(stats)
        assert_clean(stats)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.output_dir / f"deeppa_migration_options_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    summary = {
        "config": {
            "batch_size": args.batch_size,
            "num_points": args.num_points,
            "k": args.k,
            "use_rlw": args.use_rlw,
            "output_dir": str(args.output_dir),
        },
        "center_geom_max_abs_diff": center_geom_diff,
        "warmup_weights": warmup_weights,
        "transition_weights": transition_weights,
        "tensor_stats": stats_list,
    }
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[CHECK] Save path: {save_path}")
    print("[PASS] DeepPA migration option validation complete")


if __name__ == "__main__":
    main()
