import json
import time
from pathlib import Path

import torch

from loss_controller import DeepPALossController


CHECK_EPOCHS = [1, 30, 31, 60, 61, 100, 250, 500]
WEIGHT_PAIRS = [
    (0.5, 0.5),
    (0.1, 0.9),
    (0.0, 1.0),
]
MODEL_NAMES = ["frozen_aux_drop", "deeppa_finetune", "deeppa_e2e"]
TOTAL_EPOCHS = 500
HEATMAP_EPOCHS = 60


def make_controller(main_weight, geom_weight):
    controller = DeepPALossController(aux_drop_epochs=30)
    controller.loss_schedule = "linear_three_phase"
    controller.fixed_heatmap_epochs = HEATMAP_EPOCHS
    controller.fixed_main_weight = main_weight
    controller.fixed_geom_weight = geom_weight
    controller.total_epochs = TOTAL_EPOCHS
    controller.e2e_aux_mode = "drop"
    return controller


def expected_weights(epoch, final_main, final_geom):
    if epoch <= HEATMAP_EPOCHS:
        return 1.0, 0.0
    transition_epochs = TOTAL_EPOCHS - HEATMAP_EPOCHS
    progress = (epoch - HEATMAP_EPOCHS) / float(transition_epochs)
    return 1.0 + (final_main - 1.0) * progress, final_geom * progress


def main():
    losses = {
        "L_main": torch.tensor(0.001),
        "L_aux": torch.tensor(0.01),
        "L_crd": torch.tensor(0.1),
        "L_srf": torch.tensor(0.02),
        "L_str": torch.tensor(0.03),
        "L_pa": torch.tensor(0.0),
    }
    all_records = []
    failures = []

    print("[CHECK] linear_three_phase schedule")
    print("[CHECK] heatmap-only epochs: 1-60")
    print("[CHECK] linear transition epochs: 61-500")

    for final_main, final_geom in WEIGHT_PAIRS:
        print(f"\n[CHECK] final Main_HM={final_main:.1f}, Geom={final_geom:.1f}")
        for model_name in MODEL_NAMES:
            print(f"[CHECK] Model={model_name}")
            controller = make_controller(final_main, final_geom)
            for epoch in CHECK_EPOCHS:
                total_loss, weights = controller.compute_loss(model_name, epoch - 1, **losses)
                expected_main, expected_geom = expected_weights(epoch, final_main, final_geom)
                patience_triggered = controller.update_patience(current_val_mm=1.0, epoch=epoch - 1)
                record = {
                    "model": model_name,
                    "epoch": epoch,
                    "target_final_main": final_main,
                    "target_final_geom": final_geom,
                    "w_main": float(weights["w_main"]),
                    "w_geom": float(weights["w_geom"]),
                    "w_aux": float(weights["w_hds"]),
                    "w_pa": float(weights["w_pa"]),
                    "w_dp": float(weights["w_dp"]),
                    "expected_main": expected_main,
                    "expected_geom": expected_geom,
                    "total_loss": float(total_loss),
                    "patience_triggered": bool(patience_triggered),
                }
                all_records.append(record)
                print(
                    "[CHECK] Ep{epoch:03d}: Main={w_main:.4f}, "
                    "Geom={w_geom:.4f}, PA={w_pa:.4f}, Aux={w_aux:.3f}".format(**record)
                )
                if abs(record["w_main"] - expected_main) > 1e-8:
                    failures.append(record)
                if abs(record["w_geom"] - expected_geom) > 1e-8:
                    failures.append(record)
                if model_name == "deeppa_e2e" and abs(record["w_pa"] - expected_main) > 1e-8:
                    failures.append(record)
                if model_name == "deeppa_finetune" and abs(record["w_pa"] - 0.1) > 1e-8:
                    failures.append(record)
                if patience_triggered:
                    failures.append(record)

    output_dir = Path("validation_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"linear_three_phase_schedule_{time.strftime('%Y%m%d_%H%M%S')}.json"
    save_path.write_text(json.dumps(all_records, indent=2), encoding="utf-8")
    print(f"\n[CHECK] Saved validation records: {save_path}")

    if failures:
        print("[FAIL] linear_three_phase schedule check failed")
        raise SystemExit(1)
    print("[PASS] linear_three_phase schedule weights are consistent")


if __name__ == "__main__":
    main()
