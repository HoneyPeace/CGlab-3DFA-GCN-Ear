import argparse
import datetime as dt
import subprocess
import time
from pathlib import Path


def read_tail(path, max_lines):
    if not path.exists():
        return ["[WARNING] Log file does not exist yet: {}".format(path)]
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if not lines:
        return ["[INFO] Log file exists but is empty: {}".format(path)]
    return lines[-max_lines:]


def process_status(pid):
    if not pid:
        return ["[INFO] No PID provided."]
    result = subprocess.run(
        ["tasklist", "/FI", "PID eq {}".format(pid), "/FO", "CSV", "/NH"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if any('"{}"'.format(pid) in line for line in lines):
        return ["[INFO] Tracked parent process is running. PID: {}".format(pid)] + lines
    return ["[INFO] Tracked parent process is not running. PID: {}".format(pid)] + lines


def python_process_status():
    command = (
        "Get-Process python -ErrorAction SilentlyContinue | "
        "Select-Object Id,CPU,StartTime,Path | Format-Table -AutoSize | Out-String"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    lines = [line.rstrip() for line in result.stdout.splitlines()]
    return lines or ["[INFO] No python process output."]


def write_snapshot(args):
    output_path = Path(args.output)
    history_path = Path(args.history) if args.history else None
    log_path = Path(args.log)

    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("[SNAPSHOT] {}".format(stamp))
    lines.append("[LOG] {}".format(log_path))
    lines.append("")
    lines.append("[PROCESS]")
    lines.extend(process_status(args.pid))
    lines.append("")
    lines.append("[PYTHON PROCESSES]")
    lines.extend(python_process_status())
    lines.append("")
    lines.append("[LAST {} LOG LINES]".format(args.tail_lines))
    lines.extend(read_tail(log_path, args.tail_lines))
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_text = "\n".join(lines)
    output_path.write_text(snapshot_text, encoding="utf-8")

    if history_path:
        with history_path.open("a", encoding="utf-8", errors="replace") as f:
            f.write(snapshot_text)
            f.write("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Write periodic snapshots of a training log.")
    parser.add_argument("--log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--history", default="")
    parser.add_argument("--pid", default="")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--tail_lines", type=int, default=80)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        write_snapshot(args)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
