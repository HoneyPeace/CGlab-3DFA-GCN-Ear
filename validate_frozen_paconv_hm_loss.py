import argparse

import torch

from loss_controller import DeepPALossController


def main():
    parser = argparse.ArgumentParser(description="Validate frozen PAConv heatmap loss contribution.")
    parser.add_argument("--hm-weight", type=float, default=0.1)
    parser.add_argument("--drop-epochs", type=int, default=15)
    args = parser.parse_args()

    L_main = torch.tensor(2.0)
    L_aux = torch.tensor(3.0)
    L_crd = torch.tensor(5.0)
    L_srf = torch.tensor(7.0)
    L_str = torch.tensor(11.0)
    L_pa = torch.tensor(13.0)

    no_pa_controller = DeepPALossController(
        min_heatmap_warmup=30,
        aux_drop_epochs=args.drop_epochs,
        frozen_paconv_hm_weight=0.0,
    )
    with_pa_controller = DeepPALossController(
        min_heatmap_warmup=30,
        aux_drop_epochs=args.drop_epochs,
        frozen_paconv_hm_weight=args.hm_weight,
    )

    no_pa_loss, no_pa_weights = no_pa_controller.compute_loss(
        "frozen_aux_drop", 0, L_main, L_aux, L_crd, L_srf, L_str, L_pa
    )
    with_pa_loss, with_pa_weights = with_pa_controller.compute_loss(
        "frozen_aux_drop", 0, L_main, L_aux, L_crd, L_srf, L_str, L_pa
    )

    expected_delta = args.hm_weight * L_pa.item()
    actual_delta = (with_pa_loss - no_pa_loss).item()

    print(f"[CHECK] no_pa_loss: {no_pa_loss.item():.8f}")
    print(f"[CHECK] with_pa_loss: {with_pa_loss.item():.8f}")
    print(f"[CHECK] expected PA loss delta: {expected_delta:.8f}")
    print(f"[CHECK] actual PA loss delta: {actual_delta:.8f}")
    print(f"[CHECK] W_Aux no_pa/with_pa: {no_pa_weights['w_hds']:.6f} / {with_pa_weights['w_hds']:.6f}")

    if abs(actual_delta - expected_delta) > 1e-6:
        raise AssertionError("Frozen PAConv heatmap loss weight is not applied correctly")
    print("[PASS] Frozen PAConv heatmap loss validation passed.")


if __name__ == "__main__":
    main()
