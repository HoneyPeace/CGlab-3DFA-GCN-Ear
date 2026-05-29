import os
import re
import subprocess
import sys
import time

from My_args import parser

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def get_latest_run_info(output_root, exp_name, tag=None):
    project_dir = os.path.join(output_root, exp_name)
    if not os.path.exists(project_dir):
        return None, None, None

    subdirs = [
        os.path.join(project_dir, d)
        for d in os.listdir(project_dir)
        if os.path.isdir(os.path.join(project_dir, d))
    ]
    if tag:
        subdirs = [d for d in subdirs if tag in os.path.basename(d)]
    if not subdirs:
        return None, None, None

    def get_actual_mtime(folder):
        latest_time = os.path.getmtime(folder)
        for root, _, files in os.walk(folder):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                try:
                    latest_time = max(latest_time, os.path.getmtime(file_path))
                except OSError:
                    pass
        return latest_time

    latest_subdir = max(subdirs, key=get_actual_mtime)
    folder_name = os.path.basename(latest_subdir)

    try:
        run_id = folder_name.split("_")[-1]
    except Exception:
        run_id = None

    match = re.search(r"_train(\d+)", folder_name)
    train_len = match.group(1) if match else None
    return run_id, train_len, latest_subdir


def find_eval_artifacts(run_dir):
    if not run_dir or not os.path.isdir(run_dir):
        return [], []
    files = os.listdir(run_dir)
    xlsx_files = [f for f in files if f.endswith(".xlsx") and "Results" in f]
    txt_files = [f for f in files if f.endswith(".txt") and "Results" in f]
    return xlsx_files, txt_files


def assert_eval_artifacts(run_dir):
    xlsx_files, txt_files = find_eval_artifacts(run_dir)
    if not xlsx_files or not txt_files:
        print("[ERROR] Evaluation artifacts missing after eval.")
        print(f"[ERROR] Run folder: {run_dir}")
        print(f"[ERROR] Results Excel files: {xlsx_files}")
        print(f"[ERROR] Results text files : {txt_files}")
        sys.exit(1)
    print(f"[CHECK] Evaluation artifacts found: {xlsx_files[-1]} / {txt_files[-1]}")


def has_arg(args_list, flag):
    return flag in args_list


def get_arg_value(args_list, flag, default=""):
    if flag in args_list:
        idx = args_list.index(flag)
        if idx + 1 < len(args_list):
            return args_list[idx + 1]
    return default


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


def get_eval_checkpoint_name(model_name):
    m_name = model_name.lower()
    if m_name in ["paconv", "paconv_heat", "paconv_struct"]:
        return "Single_PAConv_last.t7"
    if m_name == "frozen_aux_fixed":
        return "Frozen_Aux_Fixed_last.t7"
    if m_name == "frozen_aux_drop":
        return "Frozen_Aux_Drop_last.t7"
    if m_name == "frozen_no_aux":
        return "Frozen_No_Aux_last.t7"
    return f"{m_name}_last.t7"


def save_pipeline_command_txt(command_args):
    output_dir = os.path.join(os.getcwd(), "debug_outputs")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(
        output_dir,
        f"command_run_paconv_struct_{time.strftime('%Y%m%d%H%M%S')}.txt",
    )
    command = subprocess.list2cmdline([sys.executable] + command_args)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("[Working Directory]\n")
        f.write(os.getcwd() + "\n\n")
        f.write("[Command]\n")
        f.write(command + "\n")
    print(f"[INFO] Pipeline command saved to: {save_path}")


if __name__ == "__main__":
    args = parser.parse_args()
    user_args = sys.argv[1:]

    pipeline_model = args.model.lower() if has_arg(user_args, "--model") else "paconv_struct"
    pipeline_args = set_arg_value(user_args, "--model", pipeline_model)
    user_tag = get_arg_value(pipeline_args, "--user_tag", "")

    save_pipeline_command_txt(["run.py"] + pipeline_args)

    print("===============================================================")
    print(f" [PIPELINE START] Auto Training & Evaluation Pipeline ({pipeline_model})")
    print("===============================================================\n")

    print(">>> [PHASE 1] Executing train.py...")
    train_cmd = [sys.executable, "train.py"] + pipeline_args
    try:
        subprocess.run(train_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Training failed, stopping pipeline: {e}")
        sys.exit(1)

    run_id, train_len, run_dir = get_latest_run_info(args.output_root, args.exp_name, tag=user_tag)
    if not run_id:
        print("\n[ERROR] Could not find the newly created training run folder.")
        sys.exit(1)

    print(f"\n>>> [PHASE 2] Executing eval.py... (Run ID: {run_id} | Train Len: {train_len})")

    eval_args = set_arg_value(pipeline_args, "--run_id", run_id)
    if train_len and not has_arg(eval_args, "--train_len"):
        eval_args.extend(["--train_len", train_len])
    if not has_arg(eval_args, "--Eval_DataType"):
        eval_args.extend(["--Eval_DataType", "test"])
    if not has_arg(eval_args, "--model_epoch"):
        eval_args.extend(["--model_epoch", get_eval_checkpoint_name(pipeline_model)])

    eval_cmd = [sys.executable, "eval.py"] + eval_args
    try:
        subprocess.run(eval_cmd, check=True)
        assert_eval_artifacts(run_dir)
        print("\n===============================================================")
        print(" [PIPELINE SUCCESS] Training and evaluation completed.")
        print("===============================================================")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        sys.exit(1)
