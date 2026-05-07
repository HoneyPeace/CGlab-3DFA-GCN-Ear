import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from My_args import parser
from augmentations import normalize_data, PointcloudScaleAndTranslate
from dataset import FaceLandmarkData
from loss import (
    AdaptiveWingLoss,
    CurvatureSurfaceLoss,
    DeepPA_HierarchicalHeatmapLoss,
    compute_structural_loss,
    focal_l1_loss,
)
from loss_controller import DeepPALossController
from optimizer_utils import build_training_optimizer
from train import UniversalPipeline, load_stage1_paconv_checkpoint, weight_init


def tensor_stats(name, tensor):
    detached = tensor.detach()
    return {
        "name": name,
        "shape": list(detached.shape),
        "min": float(detached.min().item()),
        "max": float(detached.max().item()),
        "mean": float(detached.mean().item()),
        "nan_count": int(torch.isnan(detached).sum().item()),
        "inf_count": int(torch.isinf(detached).sum().item()),
    }


def grad_summary(model, prefix, include_prefixes=None, exclude_prefixes=None):
    exclude_prefixes = exclude_prefixes or []
    total_sq = 0.0
    max_abs = 0.0
    param_tensors = 0
    tensors_with_grad = 0
    none_grad = 0
    zero_grad = 0
    top = []

    for name, param in model.named_parameters():
        if not name.startswith(prefix):
            continue
        if include_prefixes and not any(name.startswith(item) for item in include_prefixes):
            continue
        if any(name.startswith(excluded) for excluded in exclude_prefixes):
            continue

        param_tensors += 1
        if param.grad is None:
            none_grad += 1
            continue

        grad = param.grad.detach()
        grad_norm = float(grad.norm().item())
        grad_max = float(grad.abs().max().item()) if grad.numel() > 0 else 0.0
        tensors_with_grad += 1
        if grad_norm == 0.0:
            zero_grad += 1
        total_sq += grad_norm ** 2
        max_abs = max(max_abs, grad_max)
        top.append((grad_norm, grad_max, name))

    top.sort(reverse=True, key=lambda item: item[0])
    return {
        "prefix": prefix,
        "param_tensors": param_tensors,
        "tensors_with_grad": tensors_with_grad,
        "none_grad": none_grad,
        "zero_grad": zero_grad,
        "total_norm": total_sq ** 0.5,
        "max_abs": max_abs,
        "top": [
            {"name": name, "norm": norm, "max_abs": grad_max}
            for norm, grad_max, name in top[:8]
        ],
    }


def clone_params(model, prefix, include_prefixes=None):
    snapshots = {}
    for name, param in model.named_parameters():
        if not name.startswith(prefix):
            continue
        if include_prefixes and not any(name.startswith(item) for item in include_prefixes):
            continue
        snapshots[name] = param.detach().clone()
    return snapshots


def delta_summary(model, before):
    total_sq = 0.0
    max_abs = 0.0
    changed_tensors = 0
    top = []

    for name, before_tensor in before.items():
        after_tensor = dict(model.named_parameters())[name].detach()
        delta = after_tensor - before_tensor
        delta_norm = float(delta.norm().item())
        delta_max = float(delta.abs().max().item()) if delta.numel() > 0 else 0.0
        if delta_norm > 0.0:
            changed_tensors += 1
        total_sq += delta_norm ** 2
        max_abs = max(max_abs, delta_max)
        top.append((delta_norm, delta_max, name))

    top.sort(reverse=True, key=lambda item: item[0])
    return {
        "param_tensors": len(before),
        "changed_tensors": changed_tensors,
        "total_delta_norm": total_sq ** 0.5,
        "max_abs_delta": max_abs,
        "top": [
            {"name": name, "delta_norm": norm, "max_abs_delta": delta_max}
            for norm, delta_max, name in top[:8]
        ],
    }


def print_stats(stats):
    print(
        f"[CHECK] {stats['name']}: shape={stats['shape']}, "
        f"min={stats['min']:.8f}, max={stats['max']:.8f}, "
        f"mean={stats['mean']:.8f}, nan={stats['nan_count']}, inf={stats['inf_count']}"
    )


def print_grad_summary(title, summary):
    print(
        f"[CHECK] {title}: total_norm={summary['total_norm']:.12f}, "
        f"max_abs={summary['max_abs']:.12f}, "
        f"with_grad={summary['tensors_with_grad']}/{summary['param_tensors']}, "
        f"none_grad={summary['none_grad']}, zero_grad={summary['zero_grad']}"
    )
    for item in summary["top"][:3]:
        print(
            f"        top_grad {item['name']}: "
            f"norm={item['norm']:.12f}, max_abs={item['max_abs']:.12f}"
        )


def print_delta_summary(title, summary):
    print(
        f"[CHECK] {title}: total_delta_norm={summary['total_delta_norm']:.12f}, "
        f"max_abs_delta={summary['max_abs_delta']:.12f}, "
        f"changed={summary['changed_tensors']}/{summary['param_tensors']}"
    )
    for item in summary["top"][:3]:
        print(
            f"        top_delta {item['name']}: "
            f"norm={item['delta_norm']:.12f}, max_abs={item['max_abs_delta']:.12f}"
        )


def main():
    parser.add_argument(
        "--validation_output_dir",
        type=str,
        default="validation_outputs",
        help="Safe folder for validation summary JSON",
    )
    parser.add_argument(
        "--paconv_train_mode_for_check",
        action="store_true",
        help="Use train mode for PAConv during this validation only",
    )
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent
    os.chdir(repo_dir)

    if args.model.lower() not in ["frozen_aux_drop", "frozen_aux_fixed", "frozen_no_aux", "deeppa_frozen"]:
        raise ValueError("This harness is intended for frozen DeepPA/PAConv modes only.")
    if not args.unfreeze_paconv_in_frozen:
        print("[WARNING] --unfreeze_paconv_in_frozen is False. PAConv gradients are expected to be absent or zero.")
    if args.frozen_paconv_hm_weight <= 0.0:
        print("[WARNING] --frozen_paconv_hm_weight <= 0. PAConv heatmap head only receives indirect gradients, if any.")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    dataset = FaceLandmarkData(
        data_root=args.data_root,
        partition="train",
        data=args.train_dataset_name,
        in_channels=args.in_channels,
    )
    loader = DataLoader(dataset, num_workers=0, batch_size=args.batch_size, shuffle=False, drop_last=True)
    point, landmark, seg = next(iter(loader))
    point, landmark, seg = point.to(device), landmark.to(device), seg.to(device)

    model = UniversalPipeline(args, args.landmark_num, mode="frozen").to(device)
    model.apply(weight_init)

    paconv_path = os.path.join(args.output_root, "PAConv_Pretrained", "models", "Single_PAConv_last.t7")
    if not os.path.exists(paconv_path):
        raise FileNotFoundError(f"Missing PAConv pretrained checkpoint: {paconv_path}")
    load_stage1_paconv_checkpoint(model.stage1_paconv, paconv_path, device)

    model.train()
    if args.paconv_train_mode_for_check:
        model.stage1_paconv.train()
    else:
        model.stage1_paconv.eval()

    optimizer = build_training_optimizer(args, model, "frozen")
    optimizer.zero_grad()

    scale_and_translate = PointcloudScaleAndTranslate()
    point_normal, landmark_normal = normalize_data(point, landmark)
    point_normal, augmented_landmark = scale_and_translate(point_normal, landmark_normal)
    point_input = point_normal.permute(0, 2, 1).contiguous()
    points_for_coords = point_input[:, :3, :].permute(0, 2, 1).contiguous()
    target_hm = seg.permute(0, 2, 1).contiguous()

    hm_criterion = AdaptiveWingLoss().to(device)
    hierarchical_hm_loss = DeepPA_HierarchicalHeatmapLoss(hm_criterion).to(device)
    surface_criterion = CurvatureSurfaceLoss(
        k_p2p=args.plane_knn,
        k_curv=args.curv_knn,
        alpha=args.curv_alpha,
        beta=args.dir_beta,
    ).to(device)

    before_all = clone_params(model, "stage1_paconv.")
    heat_head_prefixes = [
        "stage1_paconv.conv6",
        "stage1_paconv.conv7",
        "stage1_paconv.conv8",
        "stage1_paconv.conv9",
    ]
    before_heat_head = clone_params(model, "stage1_paconv.", include_prefixes=heat_head_prefixes)

    pred_coords, sem_list, paconv_hm = model(point_input)
    paconv_hm_before = paconv_hm.detach().clone()

    L_pa = hm_criterion(paconv_hm, target_hm)
    L_main_hm, L_aux_hm = hierarchical_hm_loss(sem_list, target_hm)
    L_crd = focal_l1_loss(pred_coords, augmented_landmark, gamma=args.focal_gamma)
    L_srf, _, _, _ = surface_criterion(pred_coords, augmented_landmark, points_for_coords, disable_norm=False)
    L_str = compute_structural_loss(pred_coords, augmented_landmark)

    controller = DeepPALossController(
        patience=args.patience,
        base_hds=args.hds_buffer,
        decay_step=args.val_decay_step,
        min_heatmap_warmup=args.min_heatmap_warmup,
        use_rlw_for_pred=args.use_rlw_for_pred,
        aux_drop_epochs=args.aux_drop_epochs,
        frozen_paconv_hm_weight=args.frozen_paconv_hm_weight,
    )
    total_loss, weights = controller.compute_loss(
        args.model.lower(), 0, L_main_hm, L_aux_hm, L_crd, L_srf, L_str, L_pa
    )
    total_loss.backward()

    paconv_grad = grad_summary(model, "stage1_paconv.")
    heat_head_grad = grad_summary(model, "stage1_paconv.", include_prefixes=heat_head_prefixes)
    deeppa_grad = grad_summary(model, "stage2_deeppa.")

    optimizer.step()

    with torch.no_grad():
        model.train()
        if args.paconv_train_mode_for_check:
            model.stage1_paconv.train()
        else:
            model.stage1_paconv.eval()
        _, _, paconv_hm_after = model(point_input)
        L_pa_after = hm_criterion(paconv_hm_after, target_hm)
        heatmap_delta = (paconv_hm_after - paconv_hm_before).abs()

    delta_all = delta_summary(model, before_all)
    delta_heat_head = delta_summary(model, before_heat_head)

    stats = [
        tensor_stats("point_input", point_input),
        tensor_stats("target_hm", target_hm),
        tensor_stats("paconv_hm_before", paconv_hm_before),
        tensor_stats("paconv_hm_after", paconv_hm_after),
        tensor_stats("paconv_hm_abs_delta", heatmap_delta),
    ]

    print("[CHECK] Frozen PAConv gradient validation")
    print(f"[CHECK] Device: {device}")
    print(f"[CHECK] PAConv mode during check: {'train' if args.paconv_train_mode_for_check else 'eval'}")
    print(f"[CHECK] L_pa before: {L_pa.item():.12f}")
    print(f"[CHECK] L_pa after one optimizer step: {L_pa_after.item():.12f}")
    print(f"[CHECK] L_pa delta: {(L_pa_after.item() - L_pa.item()):.12f}")
    print(f"[CHECK] Total loss used for backward: {total_loss.item():.12f}")
    print(f"[CHECK] Weights: {weights}")
    for item in stats:
        print_stats(item)
    print_grad_summary("PAConv all grads", paconv_grad)
    print_grad_summary("PAConv heat-head grads", heat_head_grad)
    print_grad_summary("DeepPA grads", deeppa_grad)
    print_delta_summary("PAConv all param delta", delta_all)
    print_delta_summary("PAConv heat-head param delta", delta_heat_head)

    if paconv_grad["total_norm"] == 0.0:
        print("[WARNING] PAConv total gradient is zero.")
    if heat_head_grad["total_norm"] == 0.0:
        print("[WARNING] PAConv heatmap-head gradient is zero.")
    if delta_all["total_delta_norm"] == 0.0:
        print("[WARNING] PAConv parameters did not change after optimizer.step().")
    if heatmap_delta.max().item() == 0.0:
        print("[WARNING] PAConv heatmap output did not change after optimizer.step().")

    output_dir = Path(args.validation_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"frozen_paconv_grad_{time.strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        "config": {
            "model": args.model,
            "batch_size": args.batch_size,
            "num_points": args.num_points,
            "fusion_residual_base": args.fusion_residual_base,
            "unfreeze_paconv_in_frozen": args.unfreeze_paconv_in_frozen,
            "frozen_paconv_hm_weight": args.frozen_paconv_hm_weight,
            "frozen_paconv_lr_scale": args.frozen_paconv_lr_scale,
            "paconv_mode": "train" if args.paconv_train_mode_for_check else "eval",
        },
        "loss": {
            "L_pa_before": float(L_pa.item()),
            "L_pa_after": float(L_pa_after.item()),
            "L_pa_delta": float(L_pa_after.item() - L_pa.item()),
            "total_loss": float(total_loss.item()),
            "weights": weights,
        },
        "tensor_stats": stats,
        "grad": {
            "paconv_all": paconv_grad,
            "paconv_heat_head": heat_head_grad,
            "deeppa": deeppa_grad,
        },
        "param_delta": {
            "paconv_all": delta_all,
            "paconv_heat_head": delta_heat_head,
        },
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[CHECK] Save path: {save_path}")
    print("[PASS] Frozen PAConv gradient validation complete.")


if __name__ == "__main__":
    main()
