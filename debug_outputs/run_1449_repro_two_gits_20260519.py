import datetime as dt
import json
import re
import subprocess
import sys
import time
from pathlib import Path


CURRENT_REPO = Path(r"C:\Users\CGlab\Desktop\Earlandmark\Ear 3DFA-GCN")
SEEDED_REPO = Path(r"C:\Users\CGlab\Desktop\Earlandmark\Ear 3DFA-GCN_seeded_pa_repro_360039a")
RESULTS_DIR = Path(r"C:\Users\CGlab\Desktop\Earlandmark\results")
RUN_ROOT = CURRENT_REPO / "debug_outputs" / "repro1449_two_gits_20260519"
VSDEVCMD = r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat"
PYTHON = r"C:\Users\CGlab\anaconda3\envs\EarLandMarking_DeepLA\python.exe"
CUTILS_PREBUILT = str(CURRENT_REPO / "debug_outputs" / "torch_ext_pa163_hmattnres3mm" / "cutils_")


BASE_ARGS = [
    "run_frozen.py",
    "--need_resample", "False",
    "--stage1_exp_name", "S2G_PAconv_Refinement",
    "--stage1_user_tag", "finetune_paconv_heatmap",
    "--train_dataset_name", "train",
    "--val_dataset_name", "valiation",
    "--test_dataset_name", "test",
    "--val_partition", "val",
    "--Eval_DataType", "test",
    "--epochs", "500",
    "--train_len", "200",
    "--batch_size", "4",
    "--num_points", "8192",
    "--geom_batch_size", "16",
    "--fusion_residual_base", "prior",
    "--ablation_only", "frozen_aux_drop",
    "--aux_drop_epochs", "30",
    "--use_stagewise_aux_hm", "True",
    "--decoder_fusion", "add",
    "--train_coord_readout", "sigmoid_xyz_pool",
    "--heatmap_activation_mode", "sigmoid",
    "--heatmap_loss_mode", "adaptive_wing",
    "--coord_loss_mode", "focal_l1",
    "--struct_loss_mode", "coord",
    "--surface_loss_mode", "topk",
    "--loss_schedule", "train_hm_plateau",
    "--plateau_start_epoch", "30",
    "--plateau_window", "20",
    "--plateau_patience", "10",
    "--plateau_threshold", "0.01",
    "--plateau_transition_epochs", "1",
    "--plateau_transition_mode", "linear",
    "--fixed_main_weight", "1.0",
    "--fixed_geom_weight", "1.0",
    "--model", "frozen_aux_drop",
    "--latent_injection_type", "raw",
    "--model_epoch", "Single_PAConv_last.t7",
    "--seed", "1",
    "--dataset_seed", "1",
]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_result_txt(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"Average\s+ME\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(?:±|짹|\+/-|\+-)\s*([0-9]+(?:\.[0-9]+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return float(match.group(1)), float(match.group(2))
    name_match = re.search(r"ME([0-9]+(?:\.[0-9]+)?)", path.name)
    return (float(name_match.group(1)), None) if name_match else (None, None)


def find_result(exp_name, user_tag, started):
    root = RESULTS_DIR / exp_name
    if not root.exists():
        return None
    candidates = []
    for result_dir in root.rglob("*"):
        if not result_dir.is_dir() or user_tag not in result_dir.name:
            continue
        try:
            if result_dir.stat().st_mtime < started - 5:
                continue
        except OSError:
            continue
        txt_files = sorted(result_dir.glob("*Results_ME*.txt"))
        if not txt_files:
            continue
        me, std = parse_result_txt(txt_files[-1])
        candidates.append({
            "result_dir": str(result_dir),
            "result_txt": str(txt_files[-1]),
            "me": me,
            "std": std,
        })
    if not candidates:
        return None
    candidates.sort(key=lambda item: Path(item["result_txt"]).stat().st_mtime)
    return candidates[-1]


def command_text(repo, args, torch_ext_dir, use_current_prebuilt):
    cmd = subprocess.list2cmdline([PYTHON] + args)
    env_parts = [
        f'call "{VSDEVCMD}" -arch=amd64 -host_arch=amd64',
        f'set "TORCH_EXTENSIONS_DIR={torch_ext_dir}"',
    ]
    if use_current_prebuilt:
        env_parts.append(f'set "DEEPLA_CUTILS_PREBUILT_DIR={CUTILS_PREBUILT}"')
    env_parts.append(cmd)
    return " && ".join(env_parts)


def patch_seeded_repo_noseed(enable):
    train_path = SEEDED_REPO / "train.py"
    backup_path = SEEDED_REPO / "train.py.repro1449_seeded_backup"
    if enable:
        if backup_path.exists():
            backup_path.unlink()
        original = train_path.read_text(encoding="utf-8")
        backup_path.write_text(original, encoding="utf-8")
        patched = original.replace(
            "    set_reproducible_seed(getattr(args, 'seed', 1))",
            "    print('[INFO] Local no-seed reproduction test: training seed is not fixed')",
        )
        train_path.write_text(patched, encoding="utf-8")
    else:
        if backup_path.exists():
            train_path.write_text(backup_path.read_text(encoding="utf-8"), encoding="utf-8")
            backup_path.unlink()


def run_one(index, total, item):
    run_dir = RUN_ROOT / f"{index:03d}_{item['name']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    status_path = run_dir / "status.json"
    command_path = run_dir / "command.txt"

    repo = item["repo"]
    args = BASE_ARGS + [
        "--exp_name", item["exp_name"],
        "--user_tag", item["user_tag"],
    ]
    if item.get("fix_train_seed"):
        args += ["--fix_train_seed", "True"]

    torch_ext_dir = str(run_dir / "torch_extensions")
    cmd = command_text(repo, args, torch_ext_dir, use_current_prebuilt=item.get("use_current_prebuilt", False))
    command_path.write_text(cmd + "\n", encoding="utf-8")

    started = time.time()
    status = {
        "index": index,
        "total": total,
        "name": item["name"],
        "repo": str(repo),
        "commit_label": item["commit_label"],
        "exp_name": item["exp_name"],
        "user_tag": item["user_tag"],
        "seed_mode": item["seed_mode"],
        "status": "running",
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
        "log_file": str(log_path),
        "command_file": str(command_path),
    }
    write_json(status_path, status)
    print(f"[RUN] Starting {index}/{total}: {item['name']}", flush=True)

    patched = bool(item.get("patch_seeded_noseed"))
    if patched:
        patch_seeded_repo_noseed(True)
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            process = subprocess.Popen(
                "cmd.exe /d /s /c " + cmd,
                cwd=str(repo),
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            while True:
                exit_code = process.poll()
                status.update({
                    "pid": process.pid,
                    "last_update": dt.datetime.now().isoformat(timespec="seconds"),
                    "elapsed_sec": int(time.time() - started),
                })
                write_json(status_path, status)
                if exit_code is not None:
                    break
                time.sleep(60)
    finally:
        if patched:
            patch_seeded_repo_noseed(False)

    result = find_result(item["exp_name"], item["user_tag"], started)
    status.update({
        "status": "success" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": int(time.time() - started),
        "result": result,
    })
    write_json(status_path, status)
    print(f"[RUN] Finished {index}/{total}: {item['name']} status={status['status']} result={result}", flush=True)
    return status


def main():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    runs = [
        {
            "name": "current_1449_sigmoidxyz_seedfree",
            "repo": CURRENT_REPO,
            "commit_label": "current 7a4fb9e",
            "exp_name": "S2G_Frozen_1449Repro_CurrentGit500",
            "user_tag": "origpaconv_1449_sigmoidxyz_current_seedfree_20260519",
            "seed_mode": "seed-free",
            "fix_train_seed": False,
            "use_current_prebuilt": True,
        },
        {
            "name": "current_1449_sigmoidxyz_seedfixed",
            "repo": CURRENT_REPO,
            "commit_label": "current 7a4fb9e",
            "exp_name": "S2G_Frozen_1449Repro_CurrentGit500",
            "user_tag": "origpaconv_1449_sigmoidxyz_current_seedfixed_20260519",
            "seed_mode": "seed-fixed",
            "fix_train_seed": True,
            "use_current_prebuilt": True,
        },
        {
            "name": "seeded360039a_1449_sigmoidxyz_seedfixed",
            "repo": SEEDED_REPO,
            "commit_label": "360039a origin/codex/seeded-pa-repro",
            "exp_name": "S2G_Frozen_1449Repro_Seeded360039a500",
            "user_tag": "origpaconv_1449_sigmoidxyz_360039a_seedfixed_20260519",
            "seed_mode": "seed-fixed by commit default",
            "fix_train_seed": False,
            "use_current_prebuilt": False,
        },
        {
            "name": "seeded360039a_1449_sigmoidxyz_local_noseed",
            "repo": SEEDED_REPO,
            "commit_label": "360039a + temporary local no-seed patch",
            "exp_name": "S2G_Frozen_1449Repro_Seeded360039a500",
            "user_tag": "origpaconv_1449_sigmoidxyz_360039a_localnoseed_20260519",
            "seed_mode": "local no-seed patch",
            "fix_train_seed": False,
            "use_current_prebuilt": False,
            "patch_seeded_noseed": True,
        },
    ]

    statuses = []
    for idx, item in enumerate(runs, start=1):
        status = run_one(idx, len(runs), item)
        statuses.append(status)
        if status["status"] != "success":
            break
    write_json(RUN_ROOT / "summary.json", {"runs": statuses})
    return 0 if statuses and all(item["status"] == "success" for item in statuses) else 1


if __name__ == "__main__":
    sys.exit(main())
