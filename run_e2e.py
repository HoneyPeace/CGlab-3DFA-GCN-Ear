'''
@Author: Researcher Park Pyeong-hwa & AI Assistant
@File: run_e2e.py
@Description: 
[Ablation Study 풀-오토메이션 파이프라인 - E2E 전용]
* 특징: PAConv 사전 가중치(Pretrained) 없이, 교사와 학생을 맨바닥(Scratch)에서 동시 학습.
* Stage 1 & Bridge가 생략되며 바로 2가지 절제 연구(Raw vs Comp)를 연속 실행합니다.
* 연구원님의 원본 에러 방어 로직 및 subprocess 제어 기능 100% 탑재
'''

import os
import sys
import subprocess
import re
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
    
    print("===============================================================")
    print(" 🚀 [ABLATION PIPELINE START] E2E Mode: 2 x DeepPA_E2E (Scratch)")
    print("===============================================================\n")

    # 🌟 인자 필터링 (자동 주입할 변수들이 중복되지 않도록 방어)
    filtered_args = []
    skip_next = False
    for arg in user_args:
        if skip_next:
            skip_next = False
            continue
        # 🌟 이번 실험의 핵심 타겟인 use_raw_injection을 필터링합니다.
        # E2E는 --model_epoch 자체를 받지 않으므로 여기서도 원천 차단합니다.
        if arg in ["--model", "--model_epoch", "--run_id", "--user_tag", "--use_raw_injection"]:
            skip_next = True
            continue
        filtered_args.append(arg)

    # =========================================================================
    # [E2E ABLATION] PAConv 선행학습 없이 바로 2종 세트 연속 실행
    # =========================================================================
    ablation_configs = [
        {"name": "E2E + Raw (1344ch 통째로 전달)",  "tag": "Stage2_E2E_Raw",   "raw": "True"},
        {"name": "E2E + Compressed (4단계 압축)",   "tag": "Stage2_E2E_Comp",  "raw": "False"},
    ]

    p2_model_name = "deeppa_e2e_last.t7" # E2E 전용 가중치 저장 이름

    for config in ablation_configs:
        print(f"\n===============================================================")
        print(f" 🧪 [E2E ABLATION] Running Config: {config['name']}")
        print(f"    - Raw Injection : {config['raw']}")
        print(f"===============================================================")
        
        # 1. 학습 체크 & 실행
        p2_run_id, p2_train_len, p2_dir = get_latest_run(args.output_root, args.exp_name, tag=config['tag'], required_model=p2_model_name)

        if p2_dir:
            print(f"  └─ 📦 [SKIP] 기존 학습 완료! (Run ID: {p2_run_id})")
        else:
            print(f"  └─ 🚀 {config['name']} 학습 시작 (From Scratch)...")
            # 🌟 사전 가중치(--model_epoch) 주입 생략!
            deeppa_train_cmd = [sys.executable, "train.py"] + filtered_args + [
                "--model", "deeppa_e2e", 
                "--user_tag", config['tag'],
                "--use_raw_injection", config['raw']
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
                "--model", "deeppa_e2e", 
                "--run_id", str(p2_run_id), 
                "--model_epoch", p2_model_name,
                "--user_tag", config['tag'],
                "--use_raw_injection", config['raw'] # 🌟 평가 시 채널 에러 방지를 위해 반드시 유지
            ]
            if p2_train_len and "--train_len" not in filtered_args: deeppa_eval_cmd.extend(["--train_len", str(p2_train_len)])
            if "--Eval_DataType" not in filtered_args: deeppa_eval_cmd.extend(["--Eval_DataType", "test"])
                
            try: subprocess.run(deeppa_eval_cmd, check=True)
            except subprocess.CalledProcessError as e: sys.exit(1)

    print("\n===============================================================")
    print(" 🎉 [PIPELINE SUCCESS] End-to-End 논문 실험 세트가 모두 완료되었습니다!")
    print("===============================================================")