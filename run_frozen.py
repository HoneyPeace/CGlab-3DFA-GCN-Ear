import os
import sys
import subprocess
import re
import time
from My_args import parser

def get_latest_run_info(output_root, exp_name):
    """지정된 실험 폴더에서 가장 최근에 생성된 폴더의 run_id와 경로를 추출합니다."""
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
    """자동화 스크립트가 제어할 인자들을 필터링합니다."""
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
    
    # 강제 제어할 인자 필터링
    keys_to_override = ['--model', '--use_loss_norm', '--user_tag', '--run_id', '--freeze_paconv', '--pretrained_paconv_path', '--use_warmup']
    base_args = filter_args(raw_user_args, keys_to_override)
    
    print("===============================================================")
    print(" 🧊 [FINAL ABLATION] 2-Teacher x 2-Normalization Comparison")
    print("===============================================================\n")

    # ---------------------------------------------------------
    # 단계 1: 2종류의 Teacher(PAConv) 모델 준비
    # ---------------------------------------------------------
    teacher_configs = [
        {"tag": "T1_NoWarmup", "warm": "False", "desc": "티처 1: 선학습 없이 학습된 PAConv"},
        {"tag": "T2_WithWarmup", "warm": "True",  "desc": "티처 2: 히트맵 선학습 기믹이 적용된 PAConv"}
    ]
    
    teacher_paths = {}

    for t_cfg in teacher_configs:
        tag, warm, desc = t_cfg["tag"], t_cfg["warm"], t_cfg["desc"]
        print(f"\n👨‍🏫 [PREPARE TEACHER] {desc}")
        
        train_cmd = [sys.executable, "train.py"] + base_args + ["--model", "PAConv", "--use_warmup", warm, "--user_tag", tag]
        subprocess.run(train_cmd, check=True)
        
        _, _, latest_folder = get_latest_run_info(args.output_root, args.exp_name)
        model_path = os.path.join(latest_folder, "models", "Single_PAConv_last.t7")
        teacher_paths[tag] = model_path
        print(f">>> [SUCCESS] {tag} 생성 완료. 10초 대기...\n")
        time.sleep(10)

    # ---------------------------------------------------------
    # 단계 2: 각 티처별로 DeepPA 정규화 유무(OFF/ON) 실험 진행 (총 4회)
    # ---------------------------------------------------------
    for t_tag, t_path in teacher_paths.items():
        sub_experiments = [
            {"tag": f"{t_tag}_NormOFF", "norm": "False", "desc": f"[{t_tag}] 기반 제자 학습 - 정규화 OFF"},
            {"tag": f"{t_tag}_NormON",  "norm": "True",  "desc": f"[{t_tag}] 기반 제자 학습 - 정규화 ON"}
        ]

        for s_exp in sub_experiments:
            tag, norm, desc = s_exp["tag"], s_exp["norm"], s_exp["desc"]
            print(f"\n{'='*60}\n 🧪 [STUDENT EXPERIMENT] {desc}\n{'='*60}")

            # [Train]
            train_cmd = [sys.executable, "train.py"] + base_args + [
                "--model", "DeepPA_auto",
                "--use_loss_norm", norm,
                "--use_warmup", "False",
                "--freeze_paconv", "True",
                "--pretrained_paconv_path", t_path,
                "--user_tag", tag
            ]
            subprocess.run(train_cmd, check=True)

            # [Eval]
            run_id, train_len, _ = get_latest_run_info(args.output_root, args.exp_name)
            eval_cmd = [sys.executable, "eval_all.py"] + base_args + [
                "--model", "DeepPA_auto", "--run_id", run_id, "--user_tag", tag
            ]
            if train_len and "--train_len" not in base_args: eval_cmd.extend(["--train_len", train_len])
            if "--Eval_DataType" not in base_args: eval_cmd.extend(["--Eval_DataType", "test"])
            
            subprocess.run(eval_cmd, check=True)
            print(f">>> [SUCCESS] {tag} 완료! 10초 대기...\n")
            time.sleep(10)

    print("\n🎉 [ALL COMPLETED] 2종 티처 및 4종 제자 실험이 모두 종료되었습니다!")