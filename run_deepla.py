# @Author: Researcher Park Pyeong-hwa & AI Assistant
# @File: run_deepla.py
# @Description: DeepLA 단독 모델 4종 Ablation (무정지 자동화)
# ==============================================================================

import os
import sys
import subprocess
import re
from My_args import parser

TRAIN_SCRIPT = "train.py"
EVAL_SCRIPT = "eval.py"

def get_latest_run(output_root, exp_name, tag=None, required_model=None):
    project_dir = os.path.join(output_root, exp_name)
    if not os.path.exists(project_dir): return None, None, None
    subdirs = [os.path.join(project_dir, d) for d in os.listdir(project_dir) if os.path.isdir(os.path.join(project_dir, d))]
    
    # 🌟 대소문자 무시 검색 (폴더명 에러 방지)
    if tag: 
        subdirs = [d for d in subdirs if tag.lower() in os.path.basename(d).lower()]
    if not subdirs: return None, None, None
    subdirs.sort(key=os.path.getmtime, reverse=True)
    
    for subdir in subdirs:
        # 모델명도 대소문자 무시하여 검색
        if required_model:
            model_exists = any(required_model.lower() == f.lower() for f in os.listdir(os.path.join(subdir, "models")) if os.path.exists(os.path.join(subdir, "models")))
            if not model_exists:
                continue 
                
        folder_name = os.path.basename(subdir)
        run_id_match = re.search(r'_(\d+)$', folder_name)
        run_id = run_id_match.group(1) if run_id_match else None
        if run_id is None: continue
        return run_id, None, subdir
    return None, None, None

def check_eval_excel_exists(run_dir):
    if not run_dir or not os.path.exists(run_dir): return False
    return any(f.endswith('.xlsx') and '_Results_ME' in f for f in os.listdir(run_dir))

if __name__ == "__main__":
    # My_args 파서를 이용해 유효성 검사 (입력 인자가 My_args와 호환되는지 확인)
    args, unknown = parser.parse_known_args()
    user_args = sys.argv[1:]
    
    print("===============================================================")
    print(" 🚀 [ABLATION PIPELINE START] DeepLA 단독 모델 무정지 자동화")
    print("===============================================================\n")

    filtered_args = []
    skip_next = False
    for arg in user_args:
        if skip_next: skip_next = False; continue
        # 파이프라인에서 자동으로 덮어쓸 파라미터는 제외
        if arg in ["--model", "--model_epoch", "--run_id", "--user_tag"]:
            skip_next = True; continue
        filtered_args.append(arg)

    ablation_configs = [
        {"model_arg": "deepla_ori",      "name": "1. DeepLA_ORI (HDS + 구조로스)",        "tag": "DeepLA_ORI"},
        {"model_arg": "deepla_all",      "name": "2. DeepLA_ALL (HDS 에폭감소 + Val교대)", "tag": "DeepLA_ALL_Epoch"},
        {"model_arg": "deepla_all_tied", "name": "3. DeepLA_ALL_TIED (HDS/Main 운명공동체)", "tag": "DeepLA_ALL_Tied"},
        {"model_arg": "deepla_progress", "name": "4. DeepLA_PROGRESS (오직 Main/Geom 교대)", "tag": "DeepLA_PROGRESS"},
    ]

    for config in ablation_configs:
        print(f"\n===============================================================")
        print(f" 🧪 [PHASE RUN] Running Config: {config['name']}")
        print(f"===============================================================")
        
        model_filename = f"{config['tag']}_last.t7"
        run_id, _, run_dir = get_latest_run(args.output_root, args.exp_name, tag=config['tag'], required_model=model_filename)

        # ---------------------------------------------------------------------
        # 1. 학습
        # ---------------------------------------------------------------------
        if run_dir and run_id:
            print(f"  └─ 📦 [SKIP] 학습된 가중치 발견! (Run ID: {run_id})")
        else:
            print(f"  └─ 🚀 학습 시작... ({config['model_arg']})")
            train_cmd = [sys.executable, TRAIN_SCRIPT] + filtered_args + [
                "--model", config['model_arg'], "--user_tag", config['tag']
            ]
            try: 
                subprocess.run(train_cmd, check=True)
                run_id, _, run_dir = get_latest_run(args.output_root, args.exp_name, tag=config['tag'], required_model=model_filename)
            except subprocess.CalledProcessError: 
                # 학습 단계에서 메모리 부족 등 치명적 에러가 나면 다음 모델로 넘어감 (기존엔 sys.exit)
                print(f"  🚨 [TRAIN ERROR] 학습 중 오류 발생! 다음 모델 학습으로 PASS 합니다.")
                continue

        # ---------------------------------------------------------------------
        # 2. 평가 (에러 발생 시 절대 멈추지 않음)
        # ---------------------------------------------------------------------
        if check_eval_excel_exists(run_dir):
            print(f"  └─ 📊 [SKIP] 기존 평가 결과 발견!")
        else:
            if run_id is None:
                print(f"  ⚠️ [WARNING] run_id 없음. 평가 건너뛰고 다음 진행.")
                continue

            print(f"  └─ 🚀 평가 시작... ({config['model_arg']})")
            eval_cmd = [sys.executable, EVAL_SCRIPT] + filtered_args + [
                "--model", config['model_arg'], "--run_id", str(run_id), 
                "--model_epoch", model_filename, "--user_tag", config['tag']
            ]
            
            try: 
                subprocess.run(eval_cmd, check=True)
                print(f"  └─ ✅ 평가 완료!")
            except Exception as e: 
                # =================================================================
                # 🌟 [요청 2] 평가 중 어떤 에러가 발생해도 파이프라인은 계속 진행
                # =================================================================
                print(f"  🚨 [EVAL ERROR] 평가 중 오류가 발생했습니다: {str(e)}")
                print(f"  🚨 [PASS] 학습 흐름을 끊지 않기 위해 다음 모델로 바로 넘어갑니다.")
                continue

    print("\n===============================================================")
    print(" 🎉 [PIPELINE SUCCESS] 4종 학습 자동화 세트가 모두 완료되었습니다!")
    print("===============================================================")