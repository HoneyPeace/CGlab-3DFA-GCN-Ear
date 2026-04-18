'''
@Author: Yuan Wang (Modified by Researcher)
@File: run_frozen.py
@Description: 
[원클릭 자동화 파이프라인 (One-Click Auto Pipeline)]
1. Stage 1 (PAConv): 내비게이션 모델 단독 학습 (히트맵 가이드라인 생성)
2. Stage 1 평가 (eval.py)
3. Bridge: 생성된 PAConv 가중치를 Pretrained 폴더로 자동 복사
4. Stage 2 (DeepPA_Frozen): PAConv를 얼린 상태에서 120층 초심층망 학습 (0.47mm 정밀 타격)
5. Stage 2 평가 (eval.py) -> txt 및 엑셀 결과 자동 추출
'''

import os
import sys
import subprocess
import re
import shutil
from My_args import parser

def get_latest_run_info(output_root, exp_name):
    """지정된 실험 폴더에서 가장 최근에 생성(수정)된 폴더를 찾아 run_id, train_len, 폴더 경로를 추출합니다."""
    project_dir = os.path.join(output_root, exp_name)
    if not os.path.exists(project_dir):
        return None, None, None
    
    subdirs = [os.path.join(project_dir, d) for d in os.listdir(project_dir) 
               if os.path.isdir(os.path.join(project_dir, d))]
    
    if not subdirs:
        return None, None, None
        
    # 하위 폴더/파일까지 싹 다 뒤져서 '가장 마지막 시간' 찾기 (안전장치)
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
    
    return run_id, train_len, latest_subdir

if __name__ == "__main__":
    args = parser.parse_args()
    user_args = sys.argv[1:]
    
    print("===============================================================")
    print(" 🚀 [AUTO PIPELINE START] PAConv -> DeepPA 원스톱 자동화 가동")
    print("===============================================================\n")

    # 🌟 기존 명령어에서 모델, 태그 등 충돌을 일으키는 파라미터를 안전하게 필터링
    filtered_args = []
    skip_next = False
    for arg in user_args:
        if skip_next:
            skip_next = False
            continue
        if arg in ["--model", "--use_direct_regression", "--model_epoch", "--run_id", "--user_tag"]:
            skip_next = True
            continue
        filtered_args.append(arg)

    # =========================================================================
    # [PHASE 1] PAConv 사전 학습 및 평가 (Stage 1)
    # =========================================================================
    print(">>> [PHASE 1-A] Executing train.py for PAConv...")
    paconv_train_cmd = [sys.executable, "train.py"] + filtered_args + [
        "--model", "PAConv", 
        "--use_direct_regression", "False", 
        "--user_tag", "Stage1_PAConv"
    ]
    try:
        subprocess.run(paconv_train_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n🚨 [ERROR] PAConv 학습 중 오류가 발생하여 파이프라인을 중단합니다: {e}")
        sys.exit(1)

    p1_run_id, p1_train_len, p1_dir = get_latest_run_info(args.output_root, args.exp_name)
    if not p1_run_id:
        print("\n🚨 [ERROR] PAConv 학습 폴더를 찾을 수 없어 평가를 진행할 수 없습니다.")
        sys.exit(1)

    # 🌟 [수정포인트] eval_all.py -> eval.py 로 변경 완료
    print(f"\n>>> [PHASE 1-B] Executing eval.py for PAConv... (Detected Run ID: {p1_run_id} | Train Len: {p1_train_len})")
    paconv_eval_cmd = [sys.executable, "eval.py"] + filtered_args + [
        "--model", "PAConv", 
        "--run_id", p1_run_id, 
        "--model_epoch", "Single_PAConv_last.t7"
    ]
    if p1_train_len and "--train_len" not in filtered_args:
        paconv_eval_cmd.extend(["--train_len", p1_train_len])
    if "--Eval_DataType" not in filtered_args:
        paconv_eval_cmd.extend(["--Eval_DataType", "test"])
        
    try:
        subprocess.run(paconv_eval_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n🚨 [ERROR] PAConv 평가 중 오류가 발생했습니다: {e}")
        sys.exit(1)

    # =========================================================================
    # [BRIDGE] 가중치 파일 자동 복사 (train.py가 찾을 수 있도록)
    # =========================================================================
    print("\n>>> [BRIDGE] Preparing Pretrained Weights for DeepPA...")
    source_model_path = os.path.join(p1_dir, "models", "Single_PAConv_last.t7")
    target_model_dir = os.path.join(args.output_root, "PAConv_Pretrained", "models")
    os.makedirs(target_model_dir, exist_ok=True)
    
    target_model_path = os.path.join(target_model_dir, "Single_PAConv_last.t7")
    try:
        shutil.copy(source_model_path, target_model_path)
        print(f"  └─ 📦 완료! PAConv 가중치를 성공적으로 브릿지했습니다. ({target_model_path})")
    except FileNotFoundError:
        print(f"\n🚨 [ERROR] 브릿지 실패! PAConv 모델 파일을 찾을 수 없습니다: {source_model_path}")
        sys.exit(1)

    # =========================================================================
    # [PHASE 2] DeepPA Frozen 학습 및 평가 (Stage 2)
    # =========================================================================
    print("\n>>> [PHASE 2-A] Executing train.py for DeepPA Frozen...")
    deeppa_train_cmd = [sys.executable, "train.py"] + filtered_args + [
        "--model", "deeppa_frozen", 
        "--use_direct_regression", "True",
        "--user_tag", "Stage2_DeepPA"
    ]
    try:
        subprocess.run(deeppa_train_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n🚨 [ERROR] DeepPA 학습 중 오류가 발생하여 파이프라인을 중단합니다: {e}")
        sys.exit(1)

    p2_run_id, p2_train_len, p2_dir = get_latest_run_info(args.output_root, args.exp_name)
    if not p2_run_id:
        print("\n🚨 [ERROR] DeepPA 학습 폴더를 찾을 수 없어 평가를 진행할 수 없습니다.")
        sys.exit(1)

    # 🌟 [수정포인트] eval_all.py -> eval.py 로 변경 완료
    print(f"\n>>> [PHASE 2-B] Executing eval.py for DeepPA... (Detected Run ID: {p2_run_id} | Train Len: {p2_train_len})")
    deeppa_eval_cmd = [sys.executable, "eval.py"] + filtered_args + [
        "--model", "deeppa_frozen", 
        "--run_id", p2_run_id, 
        "--model_epoch", "Frozen_Hybrid_last.t7"
    ]
    if p2_train_len and "--train_len" not in filtered_args:
        deeppa_eval_cmd.extend(["--train_len", p2_train_len])
    if "--Eval_DataType" not in filtered_args:
        deeppa_eval_cmd.extend(["--Eval_DataType", "test"])
        
    try:
        subprocess.run(deeppa_eval_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n🚨 [ERROR] DeepPA 평가 중 오류가 발생했습니다: {e}")
        sys.exit(1)

    print("\n===============================================================")
    print(" 🎉 [PIPELINE SUCCESS] 1단계(PAConv) + 2단계(DeepPA) 원스톱 파이프라인이 성공적으로 완료되었습니다!")
    print("===============================================================")