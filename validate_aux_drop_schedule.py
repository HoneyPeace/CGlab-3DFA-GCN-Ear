import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from loss_controller import DeepPALossController


def check_schedule(drop_epochs):
    losses = [torch.tensor(1.0) for _ in range(5)]
    controller = DeepPALossController(
        min_heatmap_warmup=30,
        aux_drop_epochs=drop_epochs,
    )

    check_epochs = [0, 1, drop_epochs // 2, drop_epochs - 1, drop_epochs, drop_epochs + 1]
    rows = []
    for epoch in check_epochs:
        _, weights = controller.compute_loss(
            "frozen_aux_drop",
            epoch,
            losses[0],
            losses[1],
            losses[2],
            losses[3],
            losses[4],
        )
        rows.append({
            "epoch_arg": epoch,
            "printed_epoch": epoch + 1,
            "w_aux": float(weights["w_hds"]),
            "w_main": float(weights["w_main"]),
            "w_geom": float(weights["w_geom"]),
        })

    last_active = rows[-3]
    first_zero = rows[-2]
    if last_active["w_aux"] <= 0.0:
        raise AssertionError(f"Aux weight dropped too early for {drop_epochs} epochs")
    if first_zero["w_aux"] != 0.0:
        raise AssertionError(f"Aux weight did not reach zero at epoch_arg={drop_epochs}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Validate frozen_aux_drop 15/30 epoch schedules.")
    parser.add_argument("--drop-epochs", type=int, nargs="+", default=[15, 30])
    parser.add_argument("--output-dir", type=Path, default=Path("validation_outputs"))
    args = parser.parse_args()

    report = {}
    for drop_epochs in args.drop_epochs:
        if drop_epochs <= 0:
            raise ValueError("drop epochs must be positive")
        rows = check_schedule(drop_epochs)
        report[str(drop_epochs)] = rows
        print(f"[CHECK] aux_drop_epochs={drop_epochs}")
        for row in rows:
            print(
                "[CHECK] epoch_arg={epoch_arg:03d} printed_epoch={printed_epoch:03d} "
                "W_Aux={w_aux:.6f} W_Main={w_main:.6f} W_Geom={w_geom:.6f}".format(**row)
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.output_dir / f"aux_drop_schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[CHECK] Saved report: {save_path}")
    print("[PASS] Aux drop schedule validation passed.")


if __name__ == "__main__":
    main()
