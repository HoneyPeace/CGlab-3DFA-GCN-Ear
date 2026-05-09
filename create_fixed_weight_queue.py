import argparse
import json
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent


WEIGHTS = [
    (1.0, 0.0),
    (0.8, 0.2),
    (0.6, 0.4),
    (0.4, 0.6),
    (0.2, 0.8),
]


def weight_tag(main_weight, geom_weight):
    return "hm{:03d}_geom{:03d}".format(int(round(main_weight * 100)), int(round(geom_weight * 100)))


def main():
    parser = argparse.ArgumentParser(description="Create a fixed heatmap/geometry weight queue JSON.")
    parser.add_argument("--output", default="")
    parser.add_argument("--base_tag", default="prior_fixed_queue")
    parser.add_argument("--skip_hm100", action="store_true", help="Skip 1.0:0.0 if it is already running/done.")
    args = parser.parse_args()

    output = Path(args.output) if args.output else REPO_DIR / "debug_outputs" / "fixed_weight_queue.json"

    experiments = []
    for main_weight, geom_weight in WEIGHTS:
        if args.skip_hm100 and main_weight == 1.0 and geom_weight == 0.0:
            continue
        tag = "{}_{}".format(args.base_tag, weight_tag(main_weight, geom_weight))
        command = [
            r"C:\Users\CGlab\anaconda3\envs\EarLandMarking_DeepLA\python.exe",
            "run_frozen.py",
            "--need_resample", "False",
            "--exp_name", "S2G_Frozen_Refinement",
            "--stage1_exp_name", "S2G_PAconv_Refinement",
            "--stage1_user_tag", "finetune_paconv_heatmap",
            "--train_dataset_name", "train",
            "--val_dataset_name", "valiation",
            "--test_dataset_name", "test",
            "--val_partition", "val",
            "--Eval_DataType", "test",
            "--train_len", "200",
            "--batch_size", "4",
            "--num_points", "8192",
            "--geom_batch_size", "16",
            "--ablation_only", "frozen_aux_drop",
            "--fixed_heatmap_epochs", "60",
            "--user_tag", tag,
            "--fusion_residual_base", "prior",
            "--loss_schedule", "fixed_three_phase",
            "--fixed_main_weight", str(main_weight),
            "--fixed_geom_weight", str(geom_weight),
        ]
        experiments.append({
            "name": tag,
            "exp_name": "S2G_Frozen_Refinement",
            "user_tag": tag,
            "use_vs2019": True,
            "command": command,
        })

    queue = {
        "description": "Frozen DeepPA prior-base AuxDrop fixed heatmap/geometry weight sweep.",
        "use_vs2019": True,
        "exp_name": "S2G_Frozen_Refinement",
        "experiments": experiments,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[PASS] Queue written: {}".format(output))
    print("[INFO] Experiments: {}".format(len(experiments)))


if __name__ == "__main__":
    main()
