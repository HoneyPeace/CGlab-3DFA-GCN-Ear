import os
import sys
import subprocess
import re
import time
from My_args import parser

def get_latest_run_info(output_root, exp_name):
    """지정된 실험 폴더에서 가장 최근에 생성(수정)된 폴더를 찾아 run_id와 train_len을 추출합니다."""
    project_dir = os.path.join(output_root, exp_name)
    if not os.path.exists(project_dir):
        return None, None
    
    subdirs = [os.path.join(project_dir, d) for d in os.listdir(project_dir) 
               if os.path.isdir(os.path.join(project_dir, d))]
    
    if not subdirs:
        return None, None
        
    def get_actual_mtime(folder):
        latest_time = os.path.getmtime(folder) 
        for root, dirs, files in os.walk(folder):
            for f in files:
                file_path = os.path.join(root, f)
                try:
                    latest_time = max(latest_time, os.path.getmtime(file_path))
                except OSError:
                    pass
        return latest_time

    latest_subdir = max(subdirs, key=get_actual_mtime)
    folder_name = os.path.basename(latest_subdir)
    
    try:
        run_id = folder_name.split('_')[-1]
    except:
        run_id = None
        
    match = re.search(r'_train(\d+)', folder_name)
    train_len = match.group(1) if match else None
    
    return run_id, train_len

def filter_args(args_list, keys_to_remove):
    """사용자가 입력한 커맨드에서 스크립트가 강제로 제어할 인자들을 미리 솎아냅니다."""
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
    
    # 원본 명령어 캡처 후, 충돌 방지를 위해 강제 제어할 파라미터 필터링
    raw_user_args = sys.argv[1:]
    keys_to_override = ['--use_loss_norm', '--user_tag', '--run_id']
    base_args = filter_args(raw_user_args, keys_to_override)
    
    print("===============================================================")
    print(" 🚀 [ABLATION PIPELINE START] 스케일링 정규화(Loss Norm) 효과 비교 실험")
    print("===============================================================\n")

    # 🌟 비교할 두 가지 실험 세팅 (태그명, 정규화 사용 여부)
    experiments = [
        {"tag": "Norm_OFF", "use_loss_norm": "False", "desc": "1. 정규화 끄기 (Raw Loss 날것 그대로 학습)"},
        {"tag": "Norm_ON",  "use_loss_norm": "True",  "desc": "2. 정규화 켜기 (모든 로스 1.0 체급으로 맞춤)"}
    ]

    for exp in experiments:
        tag = exp["tag"]
        use_loss_norm = exp["use_loss_norm"]
        desc = exp["desc"]
        
        print(f"\n{'='*60}")
        print(f" 🧪 [EXPERIMENT START] {desc}")
        print(f"{'='*60}")

        # ---------------------------------------------------------
        # 1. 학습 (train.py) 실행
        # ---------------------------------------------------------
        print(f">>> [PHASE 1] Executing train.py for [{tag}]...")
        train_cmd = [sys.executable, "train.py"] + base_args + ["--use_loss_norm", use_loss_norm, "--user_tag", tag]
        
        try:
            subprocess.run(train_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n🚨 [ERROR] {tag} 학습 중 오류가 발생했습니다: {e}")
            sys.exit(1)

        # ---------------------------------------------------------
        # 2. 방금 생성된 폴더 정보 찾기
        # ---------------------------------------------------------
        run_id, train_len = get_latest_run_info(args.output_root, args.exp_name)
        if not run_id:
            print(f"\n🚨 [ERROR] {tag} 학습 폴더를 찾을 수 없습니다.")
            sys.exit(1)

        print(f"\n>>> [PHASE 2] Executing eval_all.py for [{tag}]... (Run ID: {run_id})")

        # ---------------------------------------------------------
        # 3. 평가 (eval_all.py) 실행
        # ---------------------------------------------------------
        eval_cmd = [sys.executable, "eval_all.py"] + base_args + ["--run_id", run_id, "--user_tag", tag]
        
        if train_len and "--train_len" not in base_args:
            eval_cmd.extend(["--train_len", train_len])
        if "--Eval_DataType" not in base_args:
            eval_cmd.extend(["--Eval_DataType", "test"])
            
        if "--model_epoch" not in base_args:
            if getattr(args, 'model', 'deeppa_auto').lower() != 'deeppa_auto':
                eval_cmd.extend(["--model_epoch", f"Single_{args.model}_last.t7"])

        try:
            subprocess.run(eval_cmd, check=True)
            print(f">>> [SUCCESS] {tag} 실험이 완료되었습니다!\n")
        except subprocess.CalledProcessError as e:
            print(f"\n🚨 [ERROR] {tag} 평가 중 오류가 발생했습니다: {e}")
            sys.exit(1)
            
        # GPU 메모리 휴식 타임 (10초 대기)
        print("⏳ 다음 실험을 위해 10초간 대기합니다...")
        time.sleep(10)

    print("\n===============================================================")
    print(" 🎉 [ALL EXPERIMENTS COMPLETED] 정규화(Norm) OFF vs ON 실험이 모두 종료되었습니다!")
    print(" 📂 결과 폴더에서 Norm_OFF 와 Norm_ON 엑셀 파일을 비교해 보세요.")
    print("===============================================================")