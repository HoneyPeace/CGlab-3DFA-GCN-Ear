import os
import sys
import subprocess
import re
import time
from My_args import parser

def get_latest_run_info(output_root, exp_name):
    project_dir = os.path.join(output_root, exp_name)
    if not os.path.exists(project_dir): return None, None, None
    subdirs = [os.path.join(project_dir, d) for d in os.listdir(project_dir) if os.path.isdir(os.path.join(project_dir, d))]
    if not subdirs: return None, None, None
        
    def get_actual_mtime(folder):
        latest_time = os.path.getmtime(folder) 
        for root, dirs, files in os.walk(folder):
            for f in files:
                try: latest_time = max(latest_time, os.path.getmtime(os.path.join(root, f)))
                except OSError: pass
        return latest_time

    latest_subdir = max(subdirs, key=get_actual_mtime)
    folder_name = os.path.basename(latest_subdir)
    try: run_id = folder_name.split('_')[-1]
    except: run_id = None
    match = re.search(r'_train(\d+)', folder_name)
    train_len = match.group(1) if match else None
    
    return run_id, train_len, latest_subdir

def filter_args(args_list, keys_to_remove):
    filtered = []
    skip_next = False
    for arg in args_list:
        if skip_next:
            skip_next = False
            continue
        if arg in keys_to_remove:
            skip_next = True
            continue
        filtered.append(arg)
    return filtered

if __name__ == "__main__":
    args = parser.parse_args()
    raw_user_args = sys.argv[1:]
    # 강제 제어할 인자들 필터링
    keys_to_override = ['--use_loss_norm', '--use_warmup', '--user_tag', '--run_id']
    base_args = filter_args(raw_user_args, keys_to_override)
    
    print("===============================================================")
    print(" 🚀 [PC 1: E2E PIPELINE] 정규화(Norm) & 선학습(Warmup) 4-Way 비교")
    print("===============================================================\n")

    experiments = [
        {"tag": "NormOFF_WarmOFF", "norm": "False", "warm": "False", "desc": "1. [정규화 X | 선학습 X] 생 로스 전면 경쟁"},
        {"tag": "NormOFF_WarmON",  "norm": "False", "warm": "True",  "desc": "2. [정규화 X | 선학습 O] 생 로스 히트맵 웜업"},
        {"tag": "NormON_WarmOFF",  "norm": "True",  "warm": "False", "desc": "3. [정규화 O | 선학습 X] 1.0 체급 전면 경쟁"},
        {"tag": "NormON_WarmON",   "norm": "True",  "warm": "True",  "desc": "4. [정규화 O | 선학습 O] 1.0 체급 히트맵 웜업"}
    ]

    for exp in experiments:
        tag, norm, warm, desc = exp["tag"], exp["norm"], exp["warm"], exp["desc"]
        print(f"\n{'='*60}\n 🧪 [EXPERIMENT START] {desc}\n{'='*60}")

        print(f">>> [PHASE 1] Executing train.py for [{tag}]...")
        train_cmd = [sys.executable, "train.py"] + base_args + ["--use_loss_norm", norm, "--use_warmup", warm, "--user_tag", tag]
        subprocess.run(train_cmd, check=True)

        run_id, train_len, _ = get_latest_run_info(args.output_root, args.exp_name)
        print(f"\n>>> [PHASE 2] Executing eval_all.py for [{tag}]... (Run ID: {run_id})")

        eval_cmd = [sys.executable, "eval_all.py"] + base_args + ["--run_id", run_id, "--user_tag", tag]
        if train_len and "--train_len" not in base_args: eval_cmd.extend(["--train_len", train_len])
        if "--Eval_DataType" not in base_args: eval_cmd.extend(["--Eval_DataType", "test"])
            
        subprocess.run(eval_cmd, check=True)
        print(f">>> [SUCCESS] {tag} 완료! 10초 대기...\n")
        time.sleep(10)

    print("\n🎉 [PC 1] E2E 4가지 실험 세팅이 모두 종료되었습니다!")