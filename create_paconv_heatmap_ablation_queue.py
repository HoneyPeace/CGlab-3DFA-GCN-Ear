import argparse
import json
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_PYTHON = r"C:\Users\CGlab\anaconda3\envs\EarLandMarking_DeepLA\python.exe"


BATCH_SWEEP = [
    (1, 1, "batch1"),
    (2, 1, "batch2"),
    (4, 1, "batch4"),
    (8, 1, "batch8"),
    (8, 2, "batch8x2"),
]

SIGMA_SWEEP = [
    (1.0, "sigma1"),
    (2.0, "sigma2"),
    (2.5, "sigma2p5"),
    (3.0, "sigma3"),
    (4.0, "sigma4"),
    (5.0, "sigma5"),
]

POINT_SWEEP = [
    (512, "pts512"),
    (1024, "pts1024"),
    (2048, "pts2048"),
    (4096, "pts4096"),
    (8192, "pts8192"),
    (16384, "pts16384"),
]


def str2bool_text(value):
    return "True" if value else "False"


def sigma_text(value):
    return "{:g}".format(value)


def make_command(args, tag, batch_size=4, accumulation_steps=1, sigma=2.5, num_points=8192):
    return [
        args.python,
        "run.py",
        "--model", "paconv_heat",
        "--need_resample", str2bool_text(args.need_resample),
        "--exp_name", args.exp_name,
        "--train_dataset_name", args.train_dataset_name,
        "--val_dataset_name", args.val_dataset_name,
        "--test_dataset_name", args.test_dataset_name,
        "--val_partition", args.val_partition,
        "--Eval_DataType", args.eval_datatype,
        "--train_len", str(args.train_len),
        "--epochs", str(args.epochs),
        "--batch_size", str(batch_size),
        "--accumulation_steps", str(accumulation_steps),
        "--num_points", str(num_points),
        "--sigma", sigma_text(sigma),
        "--geom_batch_size", str(args.geom_batch_size),
        "--user_tag", tag,
    ]


def add_experiment(experiments, args, group, label, command):
    tag = command[command.index("--user_tag") + 1]
    experiments.append({
        "name": "{}_{}".format(group, label),
        "group": group,
        "exp_name": args.exp_name,
        "user_tag": tag,
        "use_vs2019": True,
        "command": command,
    })


def build_experiments(args):
    experiments = []

    if args.include_batch:
        for batch_size, accumulation_steps, label in BATCH_SWEEP:
            tag = "{}_{}".format(args.base_tag, label)
            command = make_command(
                args,
                tag=tag,
                batch_size=batch_size,
                accumulation_steps=accumulation_steps,
                sigma=2.5,
                num_points=8192,
            )
            add_experiment(experiments, args, "batch", label, command)

    if args.include_sigma:
        for sigma, label in SIGMA_SWEEP:
            tag = "{}_{}".format(args.base_tag, label)
            command = make_command(
                args,
                tag=tag,
                batch_size=4,
                accumulation_steps=1,
                sigma=sigma,
                num_points=8192,
            )
            add_experiment(experiments, args, "sigma", label, command)

    if args.include_points:
        for num_points, label in POINT_SWEEP:
            tag = "{}_{}".format(args.base_tag, label)
            command = make_command(
                args,
                tag=tag,
                batch_size=4,
                accumulation_steps=1,
                sigma=2.5,
                num_points=num_points,
            )
            add_experiment(experiments, args, "points", label, command)

    return experiments


def main():
    parser = argparse.ArgumentParser(description="Create a PAConv heatmap-only one-variable ablation queue.")
    parser.add_argument("--output", default="")
    parser.add_argument("--base_tag", default="paconv_heat_base8192_sig2p5_b4")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--exp_name", default="S2G_PAconv_Refinement")
    parser.add_argument("--train_dataset_name", default="train")
    parser.add_argument("--val_dataset_name", default="valiation")
    parser.add_argument("--test_dataset_name", default="test")
    parser.add_argument("--val_partition", default="val")
    parser.add_argument("--eval_datatype", default="test")
    parser.add_argument("--train_len", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--geom_batch_size", type=int, default=16)
    parser.add_argument("--need_resample", action="store_true", default=True)
    parser.add_argument("--no_resample", dest="need_resample", action="store_false")
    parser.add_argument("--only", choices=["all", "batch", "sigma", "points"], default="all")
    args = parser.parse_args()
    args.include_batch = args.only in ["all", "batch"]
    args.include_sigma = args.only in ["all", "sigma"]
    args.include_points = args.only in ["all", "points"]

    output = Path(args.output) if args.output else REPO_DIR / "debug_outputs" / "paconv_heatmap_ablation_queue.json"
    experiments = build_experiments(args)

    queue = {
        "description": (
            "PAConv heatmap-only one-variable ablation from base setting "
            "num_points=8192, sigma=2.5, batch_size=4."
        ),
        "use_vs2019": True,
        "exp_name": args.exp_name,
        "base_setting": {
            "model": "paconv_heat",
            "num_points": 8192,
            "sigma": 2.5,
            "batch_size": 4,
            "accumulation_steps": 1,
            "train_len": args.train_len,
            "epochs": args.epochs,
            "need_resample": args.need_resample,
        },
        "experiments": experiments,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[PASS] Queue written: {}".format(output))
    print("[INFO] Experiments: {}".format(len(experiments)))
    print("[INFO] Run with:")
    print("       {} run_experiment_queue.py --queue {}".format(args.python, output))


if __name__ == "__main__":
    main()
