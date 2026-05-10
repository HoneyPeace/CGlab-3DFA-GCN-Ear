import json
import time
from pathlib import Path

import torch

from loss_controller import DeepPALossController


CHECKS = [
    {"epoch": 1, "ramp_start": None, "pa": 1.0, "main": 0.0, "geom": 0.0, "aux": 0.0},
    {"epoch": 30, "ramp_start": None, "pa": 1.0, "main": 0.0, "geom": 0.0, "aux": 0.0},
    {"epoch": 31, "ramp_start": None, "pa": 0.1, "main": 1.0, "geom": 0.0, "aux": 0.0},
    {"epoch": 60, "ramp_start": None, "pa": 0.1, "main": 1.0, "geom": 0.0, "aux": 0.0},
    {"epoch": 81, "ramp_start": 80, "pa": 0.1, "main": 0.9966666667, "geom": 0.0033333333, "aux": 0.0},
    {"epoch": 110, "ramp_start": 80, "pa": 0.1, "main": 0.9, "geom": 0.1, "aux": 0.0},
    {"epoch": 500, "ramp_start": 80, "pa": 0.1, "main": 0.9, "geom": 0.1, "aux": 0.0},
]


def make_controller():
    controller = DeepPALossController(aux_drop_epochs=30)
    controller.loss_schedule = "train_hm_plateau"
    controller.e2e_aux_mode = "none"
    controller.e2e_staged_paconv = True
    controller.e2e_paconv_warmup_epochs = 30
    controller.e2e_paconv_final_weight = 0.1
    controller.fixed_main_weight = 0.9
    controller.fixed_geom_weight = 0.1
    controller.plateau_transition_epochs = 30
    return controller


def assert_close(label, actual, expected, failures, tol=1e-6):
    if abs(actual - expected) > tol:
        failures.append(f"{label}: expected {expected:.8f}, got {actual:.8f}")


def main():
    losses = {
        "L_main": torch.tensor(0.001),
        "L_aux": torch.tensor(0.01),
        "L_crd": torch.tensor(0.1),
        "L_srf": torch.tensor(0.02),
        "L_str": torch.tensor(0.03),
        "L_pa": torch.tensor(0.002),
    }
    records = []
    failures = []

    print("[CHECK] E2E staged PAConv schedule")
    print("[CHECK] Ep001-030: PAConv HM only")
    print("[CHECK] Ep031-plateau: PAConv HM 0.1 + DeepPA Main_HM 1.0")
    print("[CHECK] after train Main_HM plateau: 30-epoch ramp to Main_HM 0.9 / Geom 0.1")

    for check in CHECKS:
        controller = make_controller()
        controller.train_hm_plateau_ramp_start_epoch = check["ramp_start"]
        _, weights = controller.compute_loss("deeppa_e2e", check["epoch"] - 1, **losses)
        record = {
            "epoch": check["epoch"],
            "ramp_start": check["ramp_start"],
            "w_pa": float(weights["w_pa"]),
            "w_main": float(weights["w_main"]),
            "w_aux": float(weights["w_hds"]),
            "w_geom": float(weights["w_geom"]),
        }
        records.append(record)
        print(
            "[CHECK] Ep{epoch:03d}: PA={w_pa:.4f}, Main={w_main:.4f}, "
            "Aux={w_aux:.4f}, Geom={w_geom:.4f}".format(**record)
        )
        assert_close(f"Ep{check['epoch']} PA", record["w_pa"], check["pa"], failures)
        assert_close(f"Ep{check['epoch']} Main", record["w_main"], check["main"], failures)
        assert_close(f"Ep{check['epoch']} Aux", record["w_aux"], check["aux"], failures)
        assert_close(f"Ep{check['epoch']} Geom", record["w_geom"], check["geom"], failures)

    output_dir = Path("validation_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"e2e_staged_schedule_{time.strftime('%Y%m%d_%H%M%S')}.json"
    save_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"[CHECK] Saved validation records: {save_path}")

    if failures:
        print("[FAIL] E2E staged schedule check failed")
        for failure in failures:
            print(f"[FAIL] {failure}")
        raise SystemExit(1)
    print("[PASS] E2E staged schedule weights are consistent")


if __name__ == "__main__":
    main()
