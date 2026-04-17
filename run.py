import os
import sys
import subprocess
import re
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
        
    # 🔥 하위 폴더/파일까지 싹 다 뒤져서 '가장 마지막으로 쓰여진 시간'을 찾음
    def get_actual_mtime(folder):
        latest_time = os.path.getmtime(folder)  # 기본 폴더 생성 시간
        for root, dirs, files in os.walk(folder):
            for f in files:
                file_path = os.path.join(root, f)
                try:
                    # 파일들의 수정 시간을 비교해 가장 최신 시간을 갱신
                    latest_time = max(latest_time, os.path.getmtime(file_path))
                except OSError:
                    pass
        return latest_time

    # 껍데기가 아닌 진짜 내부 파일 기준으로 가장 최근 폴더 찾기
    latest_subdir = max(subdirs, key=get_actual_mtime)
    folder_name = os.path.basename(latest_subdir)
    
    # 1. 폴더명 맨 끝의 run_id 추출
    try:
        run_id = folder_name.split('_')[-1]
    except:
        run_id = None
        
    # 2. 폴더명에서 실제 학습 데이터 개수(train_len) 자동 추출
    match = re.search(r'_train(\d+)', folder_name)
    train_len = match.group(1) if match else None
    
    return run_id, train_len

if __name__ == "__main__":
    # 사용자가 입력한 명령어 아규먼트 파싱
    args = parser.parse_args()
    
    # 전달받은 원본 명령어 그대로 캡처 (python run.py 부분을 제외한 나머지)
    user_args = sys.argv[1:]
    
    print("===============================================================")
    print(" 🚀 [PIPELINE START] Auto Training & Evaluation Pipeline 가동")
    print("===============================================================\n")

    # ---------------------------------------------------------
    # 1. 학습 (train.py) 실행
    # ---------------------------------------------------------
    print(">>> [PHASE 1] Executing train.py...")
    train_cmd = [sys.executable, "train.py"] + user_args
    try:
        subprocess.run(train_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n🚨 [ERROR] 학습 중 오류가 발생하여 파이프라인을 중단합니다: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 2. 방금 생성된 폴더의 run_id 및 train_len 찾기
    # ---------------------------------------------------------
    run_id, train_len = get_latest_run_info(args.output_root, args.exp_name)
    if not run_id:
        print("\n🚨 [ERROR] 방금 생성된 학습 폴더(run_id)를 찾을 수 없어 평가를 진행할 수 없습니다.")
        sys.exit(1)

    print(f"\n>>> [PHASE 2] Executing eval_all.py... (Detected Run ID: {run_id} | Train Len: {train_len})")

    # ---------------------------------------------------------
    # 3. 평가 (eval_all.py) 실행
    # ---------------------------------------------------------
    eval_cmd = [sys.executable, "eval.py"] + user_args + ["--run_id", run_id]
    
    # 추출한 train_len을 강제로 주입하여 폴더 탐색 실패 원천 차단
    if train_len and "--train_len" not in user_args:
        eval_cmd.extend(["--train_len", train_len])
    
    # 평가 시 데이터 타입이 강제로 덮어씌워지지 않도록 기본값(test) 보장
    if "--Eval_DataType" not in user_args:
        eval_cmd.extend(["--Eval_DataType", "test"])

    # 🌟 [디펜스 포인트: 평가 파일 자동 매핑 완벽 동기화]
    if "--model_epoch" not in user_args:
        target_model = args.model.lower()
        if target_model in ['deeppa_frozen', 'deeppa_auto']:
            # Frozen 모드는 Frozen_Hybrid_last.t7을 로드
            eval_cmd.extend(["--model_epoch", "Frozen_Hybrid_last.t7"])
        elif target_model == 'deeppa_e2e':
            # E2E 모드는 E2E_Hybrid_last.t7을 로드
            eval_cmd.extend(["--model_epoch", "E2E_Hybrid_last.t7"])
        else:
            # 단일 모델(PAConv 등)은 Single_ 이름표를 붙여서 로드
            eval_cmd.extend(["--model_epoch", f"Single_{args.model}_last.t7"])

    try:
        subprocess.run(eval_cmd, check=True)
        print("\n===============================================================")
        print(" 🎉 [PIPELINE SUCCESS] 학습 및 평가 파이프라인이 성공적으로 완료되었습니다!")
        print("===============================================================")
    except subprocess.CalledProcessError as e:
        print(f"\n🚨 [ERROR] 평가 중 오류가 발생했습니다: {e}")