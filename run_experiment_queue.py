import argparse
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent
DEBUG_DIR = REPO_DIR / "debug_outputs"
DEFAULT_VSDEVCMD = r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat"
DEFAULT_PYTHON = r"C:\Users\CGlab\anaconda3\envs\EarLandMarking_DeepLA\python.exe"


def load_queue(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        data = {"experiments": data}
    if "experiments" not in data or not isinstance(data["experiments"], list):
        raise ValueError("Queue JSON must contain an 'experiments' list.")
    return data


def safe_name(text):
    text = str(text).strip() or "experiment"
    text = re.sub(r"[^0-9A-Za-z_.가-힣-]+", "_", text)
    return text[:120]


def split_command(command):
    if isinstance(command, list):
        return [str(item) for item in command]
    if isinstance(command, str):
        # Keep this conservative: queue files should preferably use list commands.
        return command
    raise ValueError("Experiment command must be a list or string.")


def command_to_text(command):
    if isinstance(command, list):
        return subprocess.list2cmdline(command)
    return command


def build_cmd_line(experiment, run_dir, queue):
    command = split_command(experiment["command"])
    command_text = command_to_text(command)

    use_vs2019 = bool(experiment.get("use_vs2019", queue.get("use_vs2019", True)))
    if not use_vs2019:
        return command_text

    vsdevcmd = experiment.get("vsdevcmd", queue.get("vsdevcmd", DEFAULT_VSDEVCMD))
    torch_ext_dir = experiment.get("torch_extensions_dir")
    if not torch_ext_dir:
        torch_ext_dir = str(run_dir / "torch_extensions")

    return (
        'call "{vsdevcmd}" -arch=amd64 -host_arch=amd64 && '
        'set "TORCH_EXTENSIONS_DIR={torch_ext_dir}" && '
        "{command}"
    ).format(vsdevcmd=vsdevcmd, torch_ext_dir=torch_ext_dir, command=command_text)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_result_txt(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    filename_match = re.search(r"ME([0-9]+(?:\.[0-9]+)?)", path.name)
    me = float(filename_match.group(1)) if filename_match else None
    std = None
    avg_match = re.search(
        r"Average\s+ME\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(?:±|\+/-|\+-)\s*([0-9]+(?:\.[0-9]+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if avg_match:
        me = float(avg_match.group(1))
        std = float(avg_match.group(2))
    else:
        plus_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:±|\+/-|\+-)\s*([0-9]+(?:\.[0-9]+)?)", text)
        if plus_match:
            if me is None:
                me = float(plus_match.group(1))
            std = float(plus_match.group(2))
    return {"result_txt": str(path), "me": me, "std": std}


def analyze_results(exp_name, user_tag, start_time):
    results_root = (REPO_DIR / ".." / "results" / exp_name).resolve()
    if not results_root.exists():
        return []

    matched_dirs = []
    for path in results_root.rglob("*"):
        if not path.is_dir():
            continue
        if user_tag and user_tag not in path.name:
            continue
        try:
            if path.stat().st_mtime < start_time - 5:
                continue
        except OSError:
            continue
        matched_dirs.append(path)

    summaries = []
    for result_dir in sorted(matched_dirs, key=lambda p: p.stat().st_mtime):
        txt_files = sorted(result_dir.glob("*Results_ME*.txt"))
        xlsx_files = sorted(result_dir.glob("Training_Log_*.xlsx"))
        command_files = sorted(result_dir.glob("command_*.txt"))
        summary = {
            "result_dir": str(result_dir),
            "result_txt": "",
            "me": None,
            "std": None,
            "training_log": str(xlsx_files[-1]) if xlsx_files else "",
            "command_file": str(command_files[-1]) if command_files else "",
        }
        if txt_files:
            parsed = parse_result_txt(txt_files[-1])
            summary.update(parsed)
        summaries.append(summary)
    return summaries


def append_csv(path, rows):
    if not rows:
        return
    fieldnames = ["queue_id", "index", "name", "status", "exit_code", "elapsed_sec", "me", "std", "result_dir", "result_txt"]
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def run_experiment(queue, experiment, index, total, queue_dir, args):
    name = experiment.get("name", "experiment_{}".format(index))
    run_dir = queue_dir / "{:03d}_{}".format(index, safe_name(name))
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "run.log"
    status_path = run_dir / "status.json"
    command_path = run_dir / "command.txt"

    command_text = build_cmd_line(experiment, run_dir, queue)
    command_path.write_text(command_text + "\n", encoding="utf-8")

    exp_name = experiment.get("exp_name", queue.get("exp_name", "S2G_Frozen_Refinement"))
    user_tag = experiment.get("user_tag", "")

    started = time.time()
    status = {
        "queue_id": queue_dir.name,
        "index": index,
        "total": total,
        "name": name,
        "status": "running",
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
        "command_file": str(command_path),
        "log_file": str(log_path),
    }
    write_json(status_path, status)
    print("[QUEUE] Starting {}/{}: {}".format(index, total, name))
    print("[QUEUE] Log: {}".format(log_path))

    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            ["cmd.exe", "/d", "/s", "/c", command_text],
            cwd=str(REPO_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        while True:
            exit_code = process.poll()
            elapsed = int(time.time() - started)
            status.update({
                "pid": process.pid,
                "elapsed_sec": elapsed,
                "last_update": dt.datetime.now().isoformat(timespec="seconds"),
            })
            write_json(status_path, status)
            if exit_code is not None:
                break
            time.sleep(args.poll_interval)

    elapsed = int(time.time() - started)
    status["exit_code"] = exit_code
    status["elapsed_sec"] = elapsed
    status["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
    status["status"] = "success" if exit_code == 0 else "failed"

    summaries = analyze_results(exp_name=exp_name, user_tag=user_tag, start_time=started)
    status["results"] = summaries
    write_json(status_path, status)

    flat_rows = []
    if summaries:
        for item in summaries:
            row = {
                "queue_id": queue_dir.name,
                "index": index,
                "name": name,
                "status": status["status"],
                "exit_code": exit_code,
                "elapsed_sec": elapsed,
                "me": item.get("me"),
                "std": item.get("std"),
                "result_dir": item.get("result_dir"),
                "result_txt": item.get("result_txt"),
            }
            flat_rows.append(row)
    else:
        flat_rows.append({
            "queue_id": queue_dir.name,
            "index": index,
            "name": name,
            "status": status["status"],
            "exit_code": exit_code,
            "elapsed_sec": elapsed,
        })
    append_csv(queue_dir / "summary.csv", flat_rows)

    print("[QUEUE] Finished {}/{}: {} ({})".format(index, total, name, status["status"]))
    return exit_code


def main():
    parser = argparse.ArgumentParser(description="Run experiments sequentially with logs and summaries.")
    parser.add_argument("--queue", required=True, help="Path to queue JSON.")
    parser.add_argument("--queue_id", default="", help="Optional queue run id.")
    parser.add_argument("--poll_interval", type=int, default=1800, help="Status update interval in seconds.")
    parser.add_argument("--continue_on_fail", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    queue_path = Path(args.queue).resolve()
    queue = load_queue(queue_path)
    queue_id = args.queue_id or dt.datetime.now().strftime("queue_%Y%m%d_%H%M%S")
    queue_dir = DEBUG_DIR / "experiment_queues" / safe_name(queue_id)
    queue_dir.mkdir(parents=True, exist_ok=True)
    (queue_dir / "queue.json").write_text(queue_path.read_text(encoding="utf-8"), encoding="utf-8")

    experiments = queue["experiments"]
    print("[QUEUE] Loaded {} experiments".format(len(experiments)))
    print("[QUEUE] Queue dir: {}".format(queue_dir))

    if args.dry_run:
        for idx, experiment in enumerate(experiments, start=1):
            name = experiment.get("name", "experiment_{}".format(idx))
            run_dir = queue_dir / "{:03d}_{}".format(idx, safe_name(name))
            command_text = build_cmd_line(experiment, run_dir, queue)
            print("===============================================================")
            print("[DRY RUN {}/{}] {}".format(idx, len(experiments), name))
            print(command_text)
        return 0

    for idx, experiment in enumerate(experiments, start=1):
        exit_code = run_experiment(queue, experiment, idx, len(experiments), queue_dir, args)
        if exit_code != 0 and not args.continue_on_fail:
            print("[QUEUE] Stopping because experiment failed. Use --continue_on_fail to continue.")
            return exit_code
    return 0


if __name__ == "__main__":
    sys.exit(main())
