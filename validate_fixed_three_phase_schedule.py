import json
import time
from pathlib import Path

import torch

from loss_controller import DeepPALossController


CHECK_EPOCHS = [1, 30, 31, 60, 61, 100]
WEIGHT_PAIRS = [
    (1.0, 0.0),
    (0.8, 0.2),
    (0.6, 0.4),
    (0.4, 0.6),
    (0.2, 0.8),
]
MODEL_NAMES = ["frozen_aux_drop", "frozen_aux_fixed", "frozen_no_aux"]


def make_controller(main_weight, geom_weight):
    controller = DeepPALossController(aux_drop_epochs=30)
    controller.loss_schedule = "fixed_three_phase"
    controller.fixed_heatmap_epochs = 60
    controller.fixed_main_weight = main_weight
    controller.fixed_geom_weight = geom_weight
    return controller


def validate_pair(main_weight, geom_weight):
    records = []
    losses = {
        "L_main": torch.tensor(0.001),
        "L_aux": torch.tensor(0.01),
        "L_crd": torch.tensor(0.1),
        "L_srf": torch.tensor(0.02),
        "L_str": torch.tensor(0.03),
        "L_pa": torch.tensor(0.0),
    }

    for model_name in MODEL_NAMES:
        controller = make_controller(main_weight, geom_weight)
        for epoch in CHECK_EPOCHS:
            total_loss, weights = controller.compute_loss(model_name, epoch - 1, **losses)
            patience_triggered = controller.update_patience(current_val_mm=1.0, epoch=epoch - 1)
            record = {
                "model": model_name,
                "epoch": epoch,
                "w_main": float(weights["w_main"]),
                "w_geom": float(weights["w_geom"]),
                "w_aux": float(weights["w_hds"]),
                "total_loss": float(total_loss),
                "patience_triggered": bool(patience_triggered),
            }
            records.append(record)
    return records


def main():
    all_records = []
    print("[CHECK] fixed_three_phase schedule")
    print("[CHECK] heatmap-only epochs: 1-60")
    print("[CHECK] fixed-weight epochs: 61-end")

    for main_weight, geom_weight in WEIGHT_PAIRS:
        print(f"\n[CHECK] pair Main_HM={main_weight:.1f}, Geom={geom_weight:.1f}")
        records = validate_pair(main_weight, geom_weight)
        all_records.extend(records)
        for record in records:
            if record["model"] != "frozen_aux_drop":
                continue
            print(
                "[CHECK] {model} Ep{epoch:03d}: Main={w_main:.2f}, "
                "Geom={w_geom:.2f}, Aux={w_aux:.3f}, Patience={patience_triggered}".format(**record)
            )

    failures = []
    for record in all_records:
        epoch = record["epoch"]
        if epoch <= 60:
            if record["w_main"] != 1.0 or record["w_geom"] != 0.0:
                failures.append(record)
        else:
            pair_ok = any(
                abs(record["w_main"] - main) < 1e-8 and abs(record["w_geom"] - geom) < 1e-8
                for main, geom in WEIGHT_PAIRS
            )
            if not pair_ok:
                failures.append(record)
        if record["patience_triggered"]:
            failures.append(record)

    output_dir = Path("validation_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"fixed_three_phase_schedule_{time.strftime('%Y%m%d_%H%M%S')}.json"
    save_path.write_text(json.dumps(all_records, indent=2), encoding="utf-8")
    print(f"\n[CHECK] Saved validation records: {save_path}")

    if failures:
        print("[FAIL] fixed_three_phase schedule check failed")
        raise SystemExit(1)
    print("[PASS] fixed_three_phase schedule weights are consistent")


if __name__ == "__main__":
    main()
