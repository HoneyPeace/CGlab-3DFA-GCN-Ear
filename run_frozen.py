# @Author: Researcher Park Pyeong-hwa & AI Assistant
# @File: run_aux_ablation.py
# @Description: 
# [투컴 병렬용 Aux 로스 제어 Ablation 매니저]
# - 터미널에서 입력받은 --latent_injection_type 을 유지한 채,
# - 3가지 Aux 제어 모델(Fixed, Drop, No_Aux)을 연속으로 돌립니다.
# ==============================================================================

import os
import sys
import subprocess
import re
import shutil
import time
import filecmp
from My_args import parser

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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

def set_arg_value(args_list, flag, value):
    updated_args = []
    skip_next = False
    for arg in args_list:
        if skip_next:
            skip_next = False
            continue
        if arg == flag:
            skip_next = True
            continue
        updated_args.append(arg)
    updated_args.extend([flag, value])
    return updated_args

def save_pipeline_command_txt():
    output_dir = os.path.join(os.getcwd(), "debug_outputs")
    os.makedirs(output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(output_dir, f"command_run_frozen_{stamp}.txt")
    command = subprocess.list2cmdline([sys.executable] + sys.argv)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("[Working Directory]\n")
        f.write(os.getcwd() + "\n\n")
        f.write("[Command]\n")
        f.write(command + "\n")
    print(f"[INFO] Pipeline command saved to: {save_path}")

if __name__ == "__main__":
    args = parser.parse_args()
    user_args = sys.argv[1:]
    save_pipeline_command_txt()
    
    MAIN_SCRIPT = "train.py" 
    
    # 🌟 터미널에서 입력한 주입 방식 추출 (기본값: raw)
    current_inj_type = "raw"
    if "--latent_injection_type" in user_args:
        idx = user_args.index("--latent_injection_type")
        if idx + 1 < len(user_args):
            current_inj_type = user_args[idx+1]
    aux_drop_epochs = getattr(args, "aux_drop_epochs", 30)
    aux_drop_tag = f"AuxDrop{aux_drop_epochs}" if "--aux_drop_epochs" in user_args else "AuxDrop"
            
    print("===============================================================")
    print(f" 🚀 [AUX ABLATION PIPELINE] Injection Mode: {current_inj_type.upper()}")
    print("    1 x PAConv + 3 x DeepPA (Aux Fixed / Drop / No_Aux)")
    print("===============================================================\n")

    # 인자 필터링
    filtered_args = []
    skip_next = False
    for arg in user_args:
        if skip_next:
            skip_next = False
            continue
        if arg in ["--model", "--model_epoch", "--run_id", "--user_tag", "--latent_injection_type", "--use_raw_injection"]:
            skip_next = True
            continue
        filtered_args.append(arg)

    if "--need_resample" not in filtered_args:
        filtered_args.extend(["--need_resample", "False"])
        print(">>> [INFO] run_frozen.py: --need_resample False")

    no_resample_args = set_arg_value(filtered_args, "--need_resample", "False")
    tag_prefix = f"{args.user_tag}_" if getattr(args, "user_tag", "") else ""
    stage1_tag_prefix = f"{args.stage1_user_tag}_" if getattr(args, "stage1_user_tag", "") else tag_prefix
    stage1_exp_name = getattr(args, "stage1_exp_name", "") or args.exp_name
    if stage1_exp_name != args.exp_name:
        print(f">>> [INFO] run_frozen.py: Stage1 PAConv source exp = {stage1_exp_name}")

    # =========================================================================
    # [PHASE 1] PAConv 사전 학습 및 평가 (Stage 1)
    # =========================================================================
    print(f">>> [PHASE 1] Checking existing PAConv Baseline (using {MAIN_SCRIPT})...")
    p1_tag = f"{stage1_tag_prefix}Stage1_PAConv"
    p1_model_name = "Single_PAConv_last.t7" 
    
    p1_run_id, p1_train_len, p1_dir = get_latest_run(args.output_root, stage1_exp_name, tag=p1_tag, required_model=p1_model_name)

    if p1_dir:
        print(f"  └─ 📦 [SKIP] 완료된 PAConv 발견! (Run ID: {p1_run_id})")
    else:
        if stage1_exp_name != args.exp_name:
            print(f"[ERROR] Requested Stage1 PAConv not found in exp: {stage1_exp_name}")
            print(f"[ERROR] Stage1 tag: {p1_tag}")
            sys.exit(1)
        print("  └─ 🚀 PAConv 베이스라인 학습 시작...")
        paconv_train_cmd = [sys.executable, MAIN_SCRIPT] + filtered_args + [
            "--model", "paconv_heat", "--user_tag", p1_tag
        ]
        try: subprocess.run(paconv_train_cmd, check=True)
        except subprocess.CalledProcessError as e: sys.exit(1)
        
        p1_run_id, p1_train_len, p1_dir = get_latest_run(args.output_root, stage1_exp_name, tag=p1_tag, required_model=p1_model_name)

    excel_exists = False
    if p1_dir:
        excel_exists = any(f.endswith('.xlsx') and 'Results' in f for f in os.listdir(p1_dir))
        
    if not excel_exists:
        print(f"  └─ 🚀 PAConv 평가(eval.py) 시작...")
        stage1_eval_args = no_resample_args
        if stage1_exp_name != args.exp_name:
            stage1_eval_args = set_arg_value(no_resample_args, "--exp_name", stage1_exp_name)
        paconv_eval_cmd = [sys.executable, "eval.py"] + stage1_eval_args + [
            "--model", "paconv_heat", "--run_id", str(p1_run_id), 
            "--model_epoch", p1_model_name, "--user_tag", p1_tag
        ]
        if p1_train_len and "--train_len" not in stage1_eval_args: paconv_eval_cmd.extend(["--train_len", str(p1_train_len)])
        if "--Eval_DataType" not in stage1_eval_args: paconv_eval_cmd.extend(["--Eval_DataType", "test"])
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
    if os.path.exists(target_model_path) and filecmp.cmp(source_model_path, target_model_path, shallow=False):
        print(f"  [INFO] PAConv bridge target already matches source. Copy skipped. ({p1_model_name})")
    else:
        shutil.copy(source_model_path, target_model_path)
    print(f"  └─ 📦 완료! PAConv 가중치 브릿지 성공. ({p1_model_name})")

    # =========================================================================
    # [PHASE 2] Aux 제어 Ablation 세트 연속 실행 (Fixed vs Drop vs No_Aux)
    # =========================================================================
    # 🌟 연구자님의 3가지 모델로 루프를 돕니다.
    ablation_configs = [
        {"model": "frozen_aux_fixed", "name": "Aux 0.1 고정", "tag": f"{tag_prefix}Stage2_{current_inj_type.upper()}_AuxFixed", "saved_name": "Frozen_Aux_Fixed_last.t7"},
        {"model": "frozen_aux_drop",  "name": f"Aux {aux_drop_epochs}ep linear drop", "tag": f"{tag_prefix}Stage2_{current_inj_type.upper()}_{aux_drop_tag}", "saved_name": "Frozen_Aux_Drop_last.t7"},
        {"model": "frozen_no_aux",    "name": "Aux 완전 배제", "tag": f"{tag_prefix}Stage2_{current_inj_type.upper()}_NoAux", "saved_name": "Frozen_No_Aux_last.t7"},
    ]
    ablation_only = getattr(args, "ablation_only", "all").lower()
    if ablation_only != "all":
        ablation_configs = [config for config in ablation_configs if config["model"] == ablation_only]
        if not ablation_configs:
            raise ValueError(f"Unknown ablation_only model: {ablation_only}")

    for config in ablation_configs:
        print(f"\n===============================================================")
        print(f" 🧪 [PHASE 2 ABLATION] Model: {config['model']} ({config['name']})")
        print(f"    - Injection Type : {current_inj_type}")
        print(f"    - Save Tag       : {config['tag']}")
        print(f"===============================================================")
        
        # 1. 학습 체크 & 실행
        p2_run_id, p2_train_len, p2_dir = get_latest_run(args.output_root, args.exp_name, tag=config['tag'], required_model=config['saved_name'])

        if p2_dir:
            print(f"  └─ 📦 [SKIP] 기존 학습 완료! (Run ID: {p2_run_id})")
        else:
            print(f"  └─ 🚀 {config['name']} 학습 시작...")
            deeppa_train_cmd = [sys.executable, MAIN_SCRIPT] + no_resample_args + [
                "--model", config['model'], 
                "--user_tag", config['tag'],
                "--latent_injection_type", current_inj_type, 
                "--model_epoch", p1_model_name 
            ]
            try: subprocess.run(deeppa_train_cmd, check=True)
            except subprocess.CalledProcessError as e: sys.exit(1)

            p2_run_id, p2_train_len, p2_dir = get_latest_run(args.output_root, args.exp_name, tag=config['tag'], required_model=config['saved_name'])

        # 2. 평가 체크 & 실행
        excel_exists = False
        if p2_dir:
            excel_exists = any(f.endswith('.xlsx') and 'Results' in f for f in os.listdir(p2_dir))
            
        if excel_exists:
            print(f"  └─ 📊 [SKIP] 기존 평가 결과(Excel) 발견!")
        else:
            print(f"  └─ 🚀 {config['name']} 평가 시작...")
            deeppa_eval_cmd = [sys.executable, "eval.py"] + no_resample_args + [
                "--model", config['model'], 
                "--run_id", str(p2_run_id), 
                "--model_epoch", config['saved_name'],
                "--user_tag", config['tag'],
                "--latent_injection_type", current_inj_type 
            ]
            if p2_train_len and "--train_len" not in no_resample_args: deeppa_eval_cmd.extend(["--train_len", str(p2_train_len)])
            if "--Eval_DataType" not in no_resample_args: deeppa_eval_cmd.extend(["--Eval_DataType", "test"])
                
            try: subprocess.run(deeppa_eval_cmd, check=True)
            except subprocess.CalledProcessError as e: sys.exit(1)

    print("\n===============================================================")
    print(f" 🎉 [PIPELINE SUCCESS] {current_inj_type.upper()} 주입 기반 3가지 Aux 모델 실험 완료!")
    print("===============================================================")
