# @Author: Researcher Park Pyeong-hwa & AI Assistant
# @File: run_finetune.py
# @Description:
# Lightweight finetune runner for the current Frozen DeepPA pipeline.
#
# This script keeps run_frozen.py as the canonical pipeline runner, but forces
# the PAConv-light-unfreeze setting into a separate entry point.

import os
import sys
import time
import subprocess


DEFAULT_ARGS = [
    ("--exp_name", "S2G_Finetune_Refinement"),
    ("--stage1_user_tag", "finetune_paconv_heatmap"),
]

FORCED_ARGS = [
    ("--unfreeze_paconv_in_frozen", "True"),
    ("--frozen_paconv_hm_weight", "0.1"),
    ("--frozen_paconv_lr_scale", "1.0"),
]

FINETUNE_ABLATIONS = [
    {
        "name": "AuxDrop30",
        "ablation_only": "frozen_aux_drop",
        "aux_drop_epochs": "30",
    },
    {
        "name": "NoAux",
        "ablation_only": "frozen_no_aux",
        "aux_drop_epochs": None,
    },
]


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
    updated_args.extend([flag, value])
    return updated_args


def remove_arg_value(args_list, flag):
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
    return updated_args


def ensure_arg_value(args_list, flag, value):
    if flag in args_list:
        return args_list
    return args_list + [flag, value]


def get_arg_value(args_list, flag, default=""):
    for idx, arg in enumerate(args_list):
        if arg == flag and idx + 1 < len(args_list):
            return args_list[idx + 1]
    return default


def save_pipeline_command_txt(command):
    output_dir = os.path.join(os.getcwd(), "debug_outputs")
    os.makedirs(output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(output_dir, f"command_run_finetune_{stamp}.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("[Working Directory]\n")
        f.write(os.getcwd() + "\n\n")
        f.write("[Command]\n")
        f.write(command + "\n")
    print(f"[INFO] Finetune command saved to: {save_path}")


def build_base_args(user_args):
    dry_run = "--dry_run" in user_args
    filtered_args = [arg for arg in user_args if arg != "--dry_run"]

    for flag, value in DEFAULT_ARGS:
        filtered_args = ensure_arg_value(filtered_args, flag, value)

    for flag, value in FORCED_ARGS:
        filtered_args = set_arg_value(filtered_args, flag, value)

    return filtered_args, dry_run


def build_finetune_commands(user_args):
    base_args, dry_run = build_base_args(user_args)
    commands = []

    for config in FINETUNE_ABLATIONS:
        ablation_args = set_arg_value(base_args, "--ablation_only", config["ablation_only"])
        if config["aux_drop_epochs"] is None:
            ablation_args = remove_arg_value(ablation_args, "--aux_drop_epochs")
        else:
            ablation_args = set_arg_value(ablation_args, "--aux_drop_epochs", config["aux_drop_epochs"])
        commands.append([sys.executable, "run_frozen.py"] + ablation_args)

    return commands, dry_run


def build_finetune_command(user_args):
    commands, dry_run = build_finetune_commands(user_args)
    return commands[0], dry_run


if __name__ == "__main__":
    commands, dry_run = build_finetune_commands(sys.argv[1:])
    delegated_args = commands[0][2:]
    command_texts = [subprocess.list2cmdline(command) for command in commands]
    command_text = "\n".join(command_texts)
    save_pipeline_command_txt(command_text)

    print("===============================================================")
    print(" [FINETUNE PIPELINE] PAConv light-unfreeze + AuxDrop30 / NoAux")
    print("    - Base runner       : run_frozen.py")
    print("    - PAConv unfreeze   : True")
    print("    - PAConv HM weight  : 0.1")
    print("    - PAConv LR scale   : 1.0")
    print("    - PAConv loss term  : + 0.1 * PA_HM")
    print("    - Ablation models   : frozen_aux_drop(30ep), frozen_no_aux")
    print(f"    - Residual base     : {get_arg_value(delegated_args, '--fusion_residual_base', 'prior')}")
    print(f"    - Loss schedule     : {get_arg_value(delegated_args, '--loss_schedule', 'val_adaptive')}")
    print(f"    - Result exp_name   : {get_arg_value(delegated_args, '--exp_name')}")
    print(f"    - Shared Stage1 tag : {get_arg_value(delegated_args, '--stage1_user_tag')}")
    print("===============================================================")
    print("[INFO] Delegated command:")
    for idx, text in enumerate(command_texts, start=1):
        print(f"[{idx}] {text}")

    if dry_run:
        print("[INFO] Dry run enabled. Training was not started.")
        sys.exit(0)

    for idx, command in enumerate(commands, start=1):
        print(f"[INFO] Running finetune ablation {idx}/{len(commands)}: {FINETUNE_ABLATIONS[idx - 1]['name']}")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            sys.exit(1)
