import torch

from loss_controller import DeepPALossController


def check_close(name, actual, expected, tol=1e-6):
    diff = abs(float(actual) - float(expected))
    print(f"[CHECK] {name}: actual={float(actual):.6f}, expected={float(expected):.6f}, diff={diff:.6f}")
    if diff > tol:
        raise AssertionError(f"{name} mismatch: {actual} != {expected}")


def main():
    device = torch.device("cpu")
    L_pa = torch.tensor(2.0, device=device)
    L_main = torch.tensor(0.0, device=device)
    L_aux = torch.tensor(0.0, device=device)
    L_crd = torch.tensor(3.0, device=device)
    L_srf = torch.tensor(5.0, device=device)
    L_str = torch.tensor(7.0, device=device)

    controller = DeepPALossController(patience=5, decay_step=0.05, min_heatmap_warmup=30)

    heat_only_loss, heat_only_weights = controller.compute_loss(
        "paconv_heat", 0, L_main, L_aux, L_crd, L_srf, L_str, L_pa
    )
    check_close("paconv_heat total loss", heat_only_loss.item(), 2.0)
    check_close("paconv_heat geom weight", heat_only_weights["w_geom"], 0.0)

    warmup_loss, warmup_weights = controller.compute_loss(
        "paconv_struct", 0, L_main, L_aux, L_crd, L_srf, L_str, L_pa
    )
    check_close("warmup total loss", warmup_loss.item(), 2.0)
    check_close("warmup PA weight", warmup_weights["w_main"], 1.0)
    check_close("warmup geom weight", warmup_weights["w_geom"], 0.0)

    for epoch in range(30):
        triggered = controller.update_patience(1.0, epoch)
        if triggered:
            raise AssertionError(f"patience triggered during warmup at epoch index {epoch}")
    check_close("val decay after warmup guard", controller.val_decay, 1.0)
    check_close("stagnation counter after warmup guard", controller.stagnation_counter, 0.0)

    for epoch in range(30, 34):
        controller.update_patience(1.1, epoch)
    check_close("val decay before patience count reaches 5", controller.val_decay, 1.0)

    triggered = controller.update_patience(1.1, 34)
    if not triggered:
        raise AssertionError("patience did not trigger after 5 stale epochs following warmup")
    check_close("val decay after first post-warmup trigger", controller.val_decay, 0.95)

    shifted_loss, shifted_weights = controller.compute_loss(
        "paconv_struct", 35, L_main, L_aux, L_crd, L_srf, L_str, L_pa
    )
    expected_shifted = (0.95 * 2.0) + (0.05 * (3.0 + 5.0 + 7.0))
    check_close("post-trigger total loss", shifted_loss.item(), expected_shifted)
    check_close("post-trigger PA weight", shifted_weights["w_main"], 0.95)
    check_close("post-trigger geom weight", shifted_weights["w_geom"], 0.05)

    print("[PASS] PAConv structural loss and warmup patience timing are valid")


if __name__ == "__main__":
    main()
