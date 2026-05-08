import os
import sys
import time
import subprocess


DEFAULT_WEIGHT_PAIRS = [
    (1.0, 0.0),
    (0.8, 0.2),
    (0.6, 0.4),
    (0.4, 0.6),
    (0.2, 0.8),
]


def pop_arg_value(args_list, flag, default=None):
    cleaned = []
    value = default
    skip_next = False
    for idx, arg in enumerate(args_list):
        if skip_next:
            skip_next = False
            continue
        if arg == flag:
            if idx + 1 < len(args_list):
                value = args_list[idx + 1]
                skip_next = True
            continue
        cleaned.append(arg)
    return cleaned, value


def pop_bool_flag(args_list, flag):
    cleaned = []
    found = False
    for arg in args_list:
        if arg == flag:
            found = True
            continue
        cleaned.append(arg)
    return cleaned, found


def set_arg_value(args_list, flag, value):
    updated_args = []
    skip_next = False
    for arg in args_list:
        if skip_next:
            skip_next = False
            continue
        if arg == flag:
            skip_next = True
            continue
        updated_args.append(arg)
    updated_args.extend([flag, str(value)])
    return updated_args


def parse_weight_pairs(text):
    if not text:
        return DEFAULT_WEIGHT_PAIRS
    pairs = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("weight_pairs must use 'main:geom' format, e.g. 0.8:0.2")
        main_text, geom_text = item.split(":", 1)
        pairs.append((float(main_text), float(geom_text)))
    if not pairs:
        raise ValueError("weight_pairs is empty")
    return pairs


def weight_tag(main_weight, geom_weight):
    main_id = int(round(main_weight * 100))
    geom_id = int(round(geom_weight * 100))
    return f"hm{main_id:03d}_geom{geom_id:03d}"


def save_sweep_command(commands):
    output_dir = os.path.join(os.getcwd(), "debug_outputs")
    os.makedirs(output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(output_dir, f"command_run_frozen_fixed_weight_sweep_{stamp}.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("[Working Directory]\n")
        f.write(os.getcwd() + "\n\n")
        f.write("[Sweep Command]\n")
        f.write(subprocess.list2cmdline([sys.executable] + sys.argv) + "\n\n")
        f.write("[Expanded Commands]\n")
        for command in commands:
            f.write(subprocess.list2cmdline(command) + "\n")
    print(f"[INFO] Sweep command saved to: {save_path}")


if __name__ == "__main__":
    user_args = sys.argv[1:]
    user_args, base_tag = pop_arg_value(user_args, "--user_tag", "fixed_weight_sweep")
    user_args, weight_pairs_text = pop_arg_value(user_args, "--weight_pairs", None)
    user_args, dry_run = pop_bool_flag(user_args, "--dry_run")
    weight_pairs = parse_weight_pairs(weight_pairs_text)

    for flag in [
        "--fusion_residual_base",
        "--loss_schedule",
        "--fixed_main_weight",
        "--fixed_geom_weight",
    ]:
        user_args, _ = pop_arg_value(user_args, flag, None)

    if "--fixed_heatmap_epochs" not in user_args:
        user_args = set_arg_value(user_args, "--fixed_heatmap_epochs", 60)

    commands = []
    for main_weight, geom_weight in weight_pairs:
        tag = f"{base_tag}_{weight_tag(main_weight, geom_weight)}"
        command_args = list(user_args)
        command_args = set_arg_value(command_args, "--user_tag", tag)
        command_args = set_arg_value(command_args, "--fusion_residual_base", "prior")
        command_args = set_arg_value(command_args, "--loss_schedule", "fixed_three_phase")
        command_args = set_arg_value(command_args, "--fixed_main_weight", main_weight)
        command_args = set_arg_value(command_args, "--fixed_geom_weight", geom_weight)
        commands.append([sys.executable, "run_frozen.py"] + command_args)

    save_sweep_command(commands)

    for idx, command in enumerate(commands, start=1):
        print("===============================================================")
        print(f"[SWEEP {idx}/{len(commands)}] {subprocess.list2cmdline(command)}")
        print("===============================================================")
        if dry_run:
            continue
        subprocess.run(command, check=True)
