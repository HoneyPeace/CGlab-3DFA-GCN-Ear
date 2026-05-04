import argparse
from types import SimpleNamespace

import torch
import torch.nn as nn

from optimizer_utils import build_training_optimizer


class TinyFrozenPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1_paconv = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
        self.stage2_deeppa = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))


def count_group_params(param_group):
    return sum(p.numel() for p in param_group["params"])


def main():
    parser = argparse.ArgumentParser(
        description="Validate frozen PAConv optional unfreeze optimizer grouping."
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr-scale", type=float, default=0.1)
    args = parser.parse_args()

    model = TinyFrozenPipeline()
    for param in model.stage1_paconv.parameters():
        param.requires_grad = False

    frozen_args = SimpleNamespace(
        lr=args.lr,
        weight_decay=0.0,
        unfreeze_paconv_in_frozen=False,
        frozen_paconv_lr_scale=args.lr_scale,
    )
    frozen_opt = build_training_optimizer(frozen_args, model, "frozen")
    print(f"[CHECK] frozen param group count: {len(frozen_opt.param_groups)}")
    print(f"[CHECK] frozen group lr: {frozen_opt.param_groups[0]['lr']}")
    if len(frozen_opt.param_groups) != 1:
        raise AssertionError("Default frozen optimizer should keep one trainable group")

    for param in model.stage1_paconv.parameters():
        param.requires_grad = True

    unfreeze_args = SimpleNamespace(
        lr=args.lr,
        weight_decay=0.0,
        unfreeze_paconv_in_frozen=True,
        frozen_paconv_lr_scale=args.lr_scale,
    )
    unfreeze_opt = build_training_optimizer(unfreeze_args, model, "frozen")
    group_lrs = [group["lr"] for group in unfreeze_opt.param_groups]
    group_sizes = [count_group_params(group) for group in unfreeze_opt.param_groups]

    print(f"[CHECK] unfreeze param group count: {len(unfreeze_opt.param_groups)}")
    print(f"[CHECK] unfreeze group lrs: {group_lrs}")
    print(f"[CHECK] unfreeze group param counts: {group_sizes}")

    expected_paconv_lr = args.lr * args.lr_scale
    if len(unfreeze_opt.param_groups) != 2:
        raise AssertionError("Unfrozen PAConv optimizer should have DeepPA and PAConv groups")
    if abs(group_lrs[0] - args.lr) > 1e-12:
        raise AssertionError("DeepPA group learning rate mismatch")
    if abs(group_lrs[1] - expected_paconv_lr) > 1e-12:
        raise AssertionError("PAConv group learning-rate scale mismatch")

    print("[PASS] Frozen PAConv unfreeze validation passed.")


if __name__ == "__main__":
    main()
