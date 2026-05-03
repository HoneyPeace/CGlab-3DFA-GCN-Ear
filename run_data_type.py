# @Author: Researcher Park Pyeong-hwa & AI Assistant
# @File: run_frozen.py
# @Description: 
# [Ablation Study 풀-오토메이션 매니저 - train.py 호출용]
# 1. Stage 1 (PAConv): 베이스라인 모델 1회 학습 (train.py 호출)
# 2. Bridge: PAConv 가중치 자동 복사
# 3. Stage 2 (DeepPA_Frozen): 'none', 'raw', 'compressed' 3종 연속 학습 및 평가
# ==============================================================================

import os
import sys
import subprocess
import re
import shutil
from My_args import parser

def get_latest_run(output_root, exp_name, tag=None, required_model=None):
    """지정된 실험 폴더에서 특정 태그와 모델 파일을 가진 가장 최근 폴더를 찾습니다."""
    project_dir = os.path.join(output_root, exp_name)
    if not os.path.exists(project_dir):
        return None, None, None
    
    subdirs = [os.path.join(project_dir, d) for d in os.listdir(project_dir) 
               if os.path.isdir(os.path.join(project_dir, d))]
    
    if tag:
        subdirs = [d for d in subdirs if tag in os.path.basename(d)]
        
    if not subdirs:
        return None, None, None
        
    subdirs.sort(key=os.path.getmtime, reverse=True)
    
    for subdir in subdirs:
        if required_model:
            if not os.path.exists(os.path.join(subdir, "models", required_model)):
                continue 
        
        folder_name = os.path.basename(subdir)
        try: run_id = folder_name.split('_')[-1]
        except: run_id = None
            
        match = re.search(r'_train(\d+)', folder_name)
        train_len = match.group(1) if match else None
        
        return run_id, train_len, subdir
        
    return None, None, None

if __name__ == "__main__":
    args = parser.parse_args()
    user_args = sys.argv[1:]
    
    # 🌟 엔진 스크립트 명시: run_frozen.py가 백그라운드로 돌릴 진짜 학습 파일
    MAIN_SCRIPT = "train.py" 
    
    print("===============================================================")
    print(" 🚀 [ABLATION PIPELINE START] 1 x PAConv + 3 x DeepPA (Ablation)")
    print("===============================================================\n")

    # 🌟 인자 필터링 (자동 주입할 변수들이 중복되지 않도록 방어)
    filtered_args = []
    skip_next = False
    for arg in user_args:
        if skip_next:
            skip_next = False
            continue
        # 구버전(--use_raw_injection)과 신버전(--latent_injection_type) 모두 필터링
        if arg in ["--model", "--model_epoch", "--run_id", "--user_tag", "--latent_injection_type", "--use_raw_injection"]:
            skip_next = True
            continue
        filtered_args.append(arg)

    # =========================================================================
    # [PHASE 1] PAConv 사전 학습 및 평가 (Stage 1) - 1회만 실행
    # =========================================================================
    print(f">>> [PHASE 1] Checking existing PAConv Baseline (using {MAIN_SCRIPT})...")
    p1_tag = "Stage1_PAConv"
    p1_model_name = "Single_PAConv_last.t7" 
    
    p1_run_id, p1_train_len, p1_dir = get_latest_run(args.output_root, args.exp_name, tag=p1_tag, required_model=p1_model_name)

    if p1_dir:
        print(f"  └─ 📦 [SKIP] 완료된 PAConv 발견! (Run ID: {p1_run_id})")
    else:
        print("  └─ 🚀 PAConv 베이스라인 학습 시작...")
        paconv_train_cmd = [sys.executable, MAIN_SCRIPT] + filtered_args + [
            "--model", "paconv", "--user_tag", p1_tag
        ]
        try: subprocess.run(paconv_train_cmd, check=True)
        except subprocess.CalledProcessError as e: sys.exit(1)
        
        p1_run_id, p1_train_len, p1_dir = get_latest_run(args.output_root, args.exp_name, tag=p1_tag, required_model=p1_model_name)

    # 평가 로직 (엑셀 체크)
    excel_exists = False
    if p1_dir:
        excel_exists = any(f.endswith('.xlsx') and 'Results' in f for f in os.listdir(p1_dir))
        
    if not excel_exists:
        print(f"  └─ 🚀 PAConv 평가(eval.py) 시작...")
        paconv_eval_cmd = [sys.executable, "eval.py"] + filtered_args + [
            "--model", "paconv", "--run_id", str(p1_run_id), 
            "--model_epoch", p1_model_name, "--user_tag", p1_tag
        ]
        if p1_train_len and "--train_len" not in filtered_args: paconv_eval_cmd.extend(["--train_len", str(p1_train_len)])
        if "--Eval_DataType" not in filtered_args: paconv_eval_cmd.extend(["--Eval_DataType", "test"])
        try: subprocess.run(paconv_eval_cmd, check=True)
        except subprocess.CalledProcessError as e: sys.exit(1)

    # =========================================================================
    # [BRIDGE] 가중치 파일 자동 복사
    # =========================================================================
    print("\n>>> [BRIDGE] Preparing Pretrained Weights for DeepPA...")
    source_model_path = os.path.join(p1_dir, "models", p1_model_name)
    target_model_dir = os.path.join(args.output_root, "PAConv_Pretrained", "models")
    os.makedirs(target_model_dir, exist_ok=True)
    target_model_path = os.path.join(target_model_dir, p1_model_name)
    shutil.copy(source_model_path, target_model_path)
    print(f"  └─ 📦 완료! PAConv 가중치 브릿지 성공. ({p1_model_name})")

    # =========================================================================
    # [PHASE 2] DeepPA Ablation 세트 연속 실행 (None vs Raw vs Compressed)
    # =========================================================================
    ablation_configs = [
        {"name": "Frozen + None (피처 주입 차단)",      "tag": "Stage2_Frozen_None",  "inj_type": "none"},
        {"name": "Frozen + Raw (1344ch 통째로 주입)",    "tag": "Stage2_Frozen_Raw",   "inj_type": "raw"},
        {"name": "Frozen + Compressed (4단계 압축 주입)", "tag": "Stage2_Frozen_Comp",  "inj_type": "compressed"},
    ]

    TARGET_MODEL = "deeppa_frozen"
    p2_model_name = "DeepPA_Frozen_last.t7" 

    for config in ablation_configs:
        print(f"\n===============================================================")
        print(f" 🧪 [PHASE 2 ABLATION] Running Config: {config['name']}")
        print(f"    - Injection Type : {config['inj_type']}")
        print(f"===============================================================")
        
        # 1. 학습 체크 & 실행
        p2_run_id, p2_train_len, p2_dir = get_latest_run(args.output_root, args.exp_name, tag=config['tag'], required_model=p2_model_name)

        if p2_dir:
            print(f"  └─ 📦 [SKIP] 기존 학습 완료! (Run ID: {p2_run_id})")
        else:
            print(f"  └─ 🚀 {config['name']} 학습 시작...")
            deeppa_train_cmd = [sys.executable, MAIN_SCRIPT] + filtered_args + [
                "--model", TARGET_MODEL, 
                "--user_tag", config['tag'],
                "--latent_injection_type", config['inj_type'], # 🌟 낡은 옵션 버리고 이걸로 주입!
                "--model_epoch", p1_model_name 
            ]
            try: subprocess.run(deeppa_train_cmd, check=True)
            except subprocess.CalledProcessError as e: sys.exit(1)

            p2_run_id, p2_train_len, p2_dir = get_latest_run(args.output_root, args.exp_name, tag=config['tag'], required_model=p2_model_name)

        # 2. 평가 체크 & 실행
        excel_exists = False
        if p2_dir:
            excel_exists = any(f.endswith('.xlsx') and 'Results' in f for f in os.listdir(p2_dir))
            
        if excel_exists:
            print(f"  └─ 📊 [SKIP] 기존 평가 결과(Excel) 발견!")
        else:
            print(f"  └─ 🚀 {config['name']} 평가 시작...")
            deeppa_eval_cmd = [sys.executable, "eval.py"] + filtered_args + [
                "--model", TARGET_MODEL, 
                "--run_id", str(p2_run_id), 
                "--model_epoch", p2_model_name,
                "--user_tag", config['tag'],
                "--latent_injection_type", config['inj_type'] 
            ]
            if p2_train_len and "--train_len" not in filtered_args: deeppa_eval_cmd.extend(["--train_len", str(p2_train_len)])
            if "--Eval_DataType" not in filtered_args: deeppa_eval_cmd.extend(["--Eval_DataType", "test"])
                
            try: subprocess.run(deeppa_eval_cmd, check=True)
            except subprocess.CalledProcessError as e: sys.exit(1)

    print("\n===============================================================")
    print(" 🎉 [PIPELINE SUCCESS] 3가지 Ablation 실험 세트가 모두 전자동으로 완료되었습니다!")
    print("===============================================================")