import os
import sys
import subprocess
import re
from My_args import parser

def get_latest_run_info(output_root, exp_name):
    """지정된 실험 폴더에서 가장 최근에 생성된 폴더를 찾아 run_id와 train_len을 추출합니다."""
    project_dir = os.path.join(output_root, exp_name)
    if not os.path.exists(project_dir):
        return None, None
    
    # 폴더 내의 모든 하위 디렉토리 탐색
    subdirs = [os.path.join(project_dir, d) for d in os.listdir(project_dir) 
               if os.path.isdir(os.path.join(project_dir, d))]
    
    if not subdirs:
        return None, None
        
    # 수정 시간(mtime) 기준으로 가장 최근에 만들어진 폴더 찾기
    latest_subdir = max(subdirs, key=os.path.getmtime)
    folder_name = os.path.basename(latest_subdir)
    
    # 1. 폴더명 맨 끝의 run_id 추출
    try:
        run_id = folder_name.split('_')[-1]
    except:
        run_id = None
        
    # 2. 🔥 폴더명에서 실제 학습 데이터 개수(train_len) 자동 추출 (예: _train209_ -> 209)
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
    eval_cmd = [sys.executable, "eval_all.py"] + user_args + ["--run_id", run_id]
    
    # 🔥 추출한 train_len을 강제로 주입하여 폴더 탐색 실패 원천 차단
    if train_len and "--train_len" not in user_args:
        eval_cmd.extend(["--train_len", train_len])
    
    # 평가 시 데이터 타입이 강제로 덮어씌워지지 않도록 기본값(test) 보장
    if "--Eval_DataType" not in user_args:
        eval_cmd.extend(["--Eval_DataType", "test"])

    try:
        subprocess.run(eval_cmd, check=True)
        print("\n===============================================================")
        print(" 🎉 [PIPELINE SUCCESS] 학습 및 평가 파이프라인이 성공적으로 완료되었습니다!")
        print("===============================================================")
    except subprocess.CalledProcessError as e:
        print(f"\n🚨 [ERROR] 평가 중 오류가 발생했습니다: {e}")