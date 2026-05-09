import argparse
import datetime as _dt
import os
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

REPO_DIR = Path(__file__).resolve().parent
DEBUG_DIR = REPO_DIR / "debug_outputs"
RUN_BAT = REPO_DIR / "run_vs2019_frozen_fixed_weight_sweep_direct.bat"
PID_FILE = DEBUG_DIR / "active_vs2019_training.pid"
LOG_POINTER = DEBUG_DIR / "active_vs2019_training_log.txt"
COMMAND_FILE = DEBUG_DIR / "active_vs2019_training_command.txt"


def _read_text(path):
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text), encoding="utf-8")


def _is_pid_running(pid):
    if not pid:
        return False
    result = subprocess.run(
        ["tasklist", "/FI", "PID eq {}".format(pid), "/FO", "CSV", "/NH"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    return '"{}"'.format(pid) in result.stdout


def _tail_log(log_path, num_lines=40):
    if not log_path.exists():
        print("[WARNING] Log file does not exist yet.")
        return
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[-num_lines:]:
        print(line)


def start(args):
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    if args.extra and args.extra[0] == "--":
        args.extra = args.extra[1:]

    old_pid = _read_text(PID_FILE)
    if old_pid and _is_pid_running(old_pid):
        print("[ERROR] A tracked training process is already running. PID: {}".format(old_pid))
        print("[INFO] Use status_vs2019_training.bat or stop_vs2019_training.bat.")
        return 1

    if not RUN_BAT.exists():
        print("[ERROR] Runner not found: {}".format(RUN_BAT))
        return 1

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = DEBUG_DIR / "active_vs2019_training_{}.log".format(timestamp)
    command = ["cmd.exe", "/d", "/s", "/c", "call", str(RUN_BAT)] + args.extra

    _write_text(COMMAND_FILE, subprocess.list2cmdline(command))
    _write_text(LOG_POINTER, str(log_path))

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW

    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )

    _write_text(PID_FILE, str(process.pid))
    print("[PASS] Training process started. PID: {}".format(process.pid))
    print("[INFO] Log file: {}".format(log_path))
    print("[INFO] Status command: status_vs2019_training.bat")
    print("[INFO] Stop command: stop_vs2019_training.bat")
    return 0


def status(_args):
    pid = _read_text(PID_FILE)
    log_path_text = _read_text(LOG_POINTER)

    if pid:
        if _is_pid_running(pid):
            print("[INFO] Tracked training process is running. PID: {}".format(pid))
        else:
            print("[INFO] No running process found for tracked PID: {}".format(pid))
    else:
        print("[INFO] No tracked training PID file found.")

    if log_path_text:
        log_path = Path(log_path_text)
        print("[INFO] Log file: {}".format(log_path))
        print("[INFO] Last 40 log lines:")
        _tail_log(log_path)
    else:
        print("[WARNING] Log pointer not found: {}".format(LOG_POINTER))
    return 0


def stop(_args):
    pid = _read_text(PID_FILE)
    if not pid:
        print("[INFO] No tracked training PID file found.")
        return 0

    if not _is_pid_running(pid):
        print("[INFO] No running process found for tracked PID: {}".format(pid))
        return 0

    print("[INFO] Stopping tracked training process tree. PID: {}".format(pid))
    result = subprocess.run(
        ["taskkill", "/PID", pid, "/T", "/F"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        print("[ERROR] Failed to stop PID: {}".format(pid))
        return result.returncode

    print("[PASS] Stop requested for tracked training process tree.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Start, stop, or inspect tracked VS2019 training.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("extra", nargs=argparse.REMAINDER)

    subparsers.add_parser("status")
    subparsers.add_parser("stop")

    args = parser.parse_args()
    if args.command == "start":
        return start(args)
    if args.command == "status":
        return status(args)
    if args.command == "stop":
        return stop(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
