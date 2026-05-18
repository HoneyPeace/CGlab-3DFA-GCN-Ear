import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = (REPO_DIR / ".." / "results").resolve()
RUN_ROOT = REPO_DIR / "debug_outputs" / "hmr_seedcompare_then_clamps_20260519"
VSDEVCMD = r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat"
PYTHON = r"C:\Users\CGlab\anaconda3\envs\EarLandMarking_DeepLA\python.exe"
CUTILS_PREBUILT = r"C:\Users\CGlab\Desktop\Earlandmark\Ear 3DFA-GCN\debug_outputs\torch_ext_pa163_hmattnres3mm\cutils_"


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
    "--decoder_fusion", "prog_half_final320",
    "--train_coord_readout", "heatmap_attn_residual",
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
]


def parse_result_txt(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"Average\s+ME\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(?:짹|\+/-|\+-)\s*([0-9]+(?:\.[0-9]+)?)",
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
        txts = sorted(result_dir.glob("*Results_ME*.txt"))
        if txts:
            me, std = parse_result_txt(txts[-1])
            candidates.append({
                "result_dir": str(result_dir),
                "result_txt": str(txts[-1]),
                "me": me,
                "std": std,
            })
    if not candidates:
        return None
    candidates.sort(key=lambda item: Path(item["result_txt"]).stat().st_mtime)
    return candidates[-1]


def build_run(exp_name, user_tag, clamp_mm, seed_fixed):
    args = [PYTHON] + BASE_ARGS + [
        "--exp_name", exp_name,
        "--user_tag", user_tag,
        "--hm_attn_residual_max_mm", str(clamp_mm),
    ]
    if float(clamp_mm) <= 0.0:
        args += ["--hm_attn_residual_max_norm", "0.0"]
    if seed_fixed:
        args += ["--fix_train_seed", "True", "--seed", "1"]
    return args


def command_text(args, torch_ext_dir):
    command = subprocess.list2cmdline(args)
    return (
        f'call "{VSDEVCMD}" -arch=amd64 -host_arch=amd64 && '
        f'set "TORCH_EXTENSIONS_DIR={torch_ext_dir}" && '
        f'set "DEEPLA_CUTILS_PREBUILT_DIR={CUTILS_PREBUILT}" && '
        f"{command}"
    )


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_one(index, total, name, exp_name, user_tag, clamp_mm, seed_fixed):
    run_dir = RUN_ROOT / f"{index:03d}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch_ext_dir = str(run_dir / "torch_extensions")
    args = build_run(exp_name, user_tag, clamp_mm, seed_fixed)
    cmd = command_text(args, torch_ext_dir)

    (run_dir / "command.txt").write_text(cmd + "\n", encoding="utf-8")
    status_path = run_dir / "status.json"
    log_path = run_dir / "run.log"

    started = time.time()
    status = {
        "index": index,
        "total": total,
        "name": name,
        "exp_name": exp_name,
        "user_tag": user_tag,
        "clamp_mm": clamp_mm,
        "seed_fixed": seed_fixed,
        "status": "running",
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
        "log_file": str(log_path),
        "command_file": str(run_dir / "command.txt"),
    }
    write_json(status_path, status)
    print(f"[RUN] Starting {index}/{total}: {name}", flush=True)
    print(f"[RUN] Log: {log_path}", flush=True)

    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            "cmd.exe /d /s /c " + cmd,
            cwd=str(REPO_DIR),
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

    result = find_result(exp_name, user_tag, started)
    status.update({
        "status": "success" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": int(time.time() - started),
        "result": result,
    })
    write_json(status_path, status)
    print(f"[RUN] Finished {index}/{total}: {name} status={status['status']} result={result}", flush=True)
    return status


def main():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    exp_seed = "S2G_Frozen_HMR_SeedCompare500"
    initial = [
        {
            "name": "hmr3_seedfree_20260519",
            "exp_name": exp_seed,
            "user_tag": "pa163_hmr3_seedfree_retry_20260519",
            "clamp_mm": 3.0,
            "seed_fixed": False,
        },
        {
            "name": "hmr3_seedfixed_20260519",
            "exp_name": exp_seed,
            "user_tag": "pa163_hmr3_seedfixed_retry_20260519",
            "clamp_mm": 3.0,
            "seed_fixed": True,
        },
    ]

    results = []
    for idx, item in enumerate(initial, start=1):
        status = run_one(idx, 5, **item)
        results.append(status)
        if status.get("status") != "success":
            print("[STOP] Initial seed comparison failed; clamp follow-up skipped.", flush=True)
            write_json(RUN_ROOT / "summary.json", {"initial": results, "selected": None, "followup": []})
            return 1

    valid = [
        item for item in results
        if item.get("result") and item["result"].get("me") is not None
    ]
    if not valid:
        print("[STOP] No valid ME found from initial seed comparison.", flush=True)
        write_json(RUN_ROOT / "summary.json", {"initial": results, "selected": None, "followup": []})
        return 1

    selected = min(valid, key=lambda item: item["result"]["me"])
    selected_seed_fixed = bool(selected["seed_fixed"])
    selected_mode = "seedfixed" if selected_seed_fixed else "seedfree"
    print(f"[SELECT] {selected['name']} selected: ME={selected['result']['me']} mode={selected_mode}", flush=True)

    exp_clamp = "S2G_Frozen_HMR_ClampFollowup500"
    followup = [
        {
            "name": f"hmr2_from_{selected_mode}_20260519",
            "exp_name": exp_clamp,
            "user_tag": f"pa163_hmr2_from_{selected_mode}_20260519",
            "clamp_mm": 2.0,
            "seed_fixed": selected_seed_fixed,
        },
        {
            "name": f"hmr1_from_{selected_mode}_20260519",
            "exp_name": exp_clamp,
            "user_tag": f"pa163_hmr1_from_{selected_mode}_20260519",
            "clamp_mm": 1.0,
            "seed_fixed": selected_seed_fixed,
        },
        {
            "name": f"hmr0_from_{selected_mode}_20260519",
            "exp_name": exp_clamp,
            "user_tag": f"pa163_hmr0_from_{selected_mode}_20260519",
            "clamp_mm": 0.0,
            "seed_fixed": selected_seed_fixed,
        },
    ]

    followup_results = []
    for offset, item in enumerate(followup, start=3):
        status = run_one(offset, 5, **item)
        followup_results.append(status)
        if status.get("status") != "success":
            print("[STOP] Clamp follow-up failed.", flush=True)
            break

    write_json(
        RUN_ROOT / "summary.json",
        {"initial": results, "selected": selected, "followup": followup_results},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
