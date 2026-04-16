import os
import sys
import subprocess
import time
import glob
import pandas as pd
from My_args import parser

def filter_user_args(args_list, keys_to_remove):
    """사용자가 터미널에 입력한 인자 중, 스크립트가 실험을 위해 강제 수정할 인자들만 제거합니다."""
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
    # 🌟 1. 사용자가 터미널에 입력한 모든 파라미터를 통째로 낚아챕니다.
    raw_user_args = sys.argv[1:]
    
    # 🌟 2. 실험 제어를 위해 스크립트가 직접 바꿀 인자들만 필터링합니다.
    # (중복 인자 전달로 인한 argparse 에러를 방지합니다)
    protected_keys = ['--use_loss_norm', '--use_rlw_for_heatmap', '--user_tag', '--run_id', '--model']
    base_args = filter_user_args(raw_user_args, protected_keys)
    
    # My_args를 통해 기본 정보만 파싱 (저장 경로 확인용)
    args, _ = parser.parse_known_args()
    exp_name = args.exp_name
    output_root = args.output_root

    print(f"==========================================================================")
    print(f" 🚀 [ALL-PASS ABLATION] 모든 파라미터를 유지하며 정규화 ON 실험 가동")
    print(f"==========================================================================\n")

    experiments = [
        {"tag": "NormON_3Loss_Anchor", "use_rlw": "False", "desc": "1. [Norm-ON] 히트맵 닻 고정 (3-Loss)"},
        {"tag": "NormON_4Loss_Random", "use_rlw": "True",  "desc": "2. [Norm-ON] 히트맵 전면 랜덤 (4-Loss)"}
    ]

    for i, exp in enumerate(experiments):
        tag = exp["tag"]
        use_rlw = exp["use_rlw"]
        
        print(f"\n🧪 {exp['desc']} 시작...")

        # [PHASE 1] 학습 (train.py) 실행
        # 사용자의 모든 인자에 '--use_loss_norm True'와 실험용 RLW 태그만 덧붙입니다.
        resample = "True" if i == 0 else "False"
        train_cmd = [sys.executable, "train.py"] + base_args + [
            "--model", "DeepPA_auto",
            "--use_loss_norm", "True", 
            "--use_rlw_for_heatmap", use_rlw,
            "--user_tag", tag,
            "--need_resample", resample
        ]
        
        print(f">>> [EXEC] {' '.join(train_cmd)}")
        subprocess.run(train_cmd, check=True)

        # [PHASE 2] 방금 생성된 Run ID 자동 추적
        time.sleep(2)
        project_dir = os.path.join(output_root, exp_name)
        # 태그가 포함된 폴더 중 가장 번호가 높은(최신) 폴더를 찾습니다.
        folders = [d for d in os.listdir(project_dir) if os.path.isdir(os.path.join(project_dir, d)) and tag in d]
        run_id = str(max([int(f.split('_')[-1]) for f in folders]))

        # [PHASE 3] 평가 (eval.py) 실행
        # 학습 때 쓴 모든 인자 + 정확한 Run ID를 꽂아줍니다.
        eval_cmd = [sys.executable, "eval.py"] + base_args + [
            "--model", "DeepPA_auto",
            "--run_id", run_id,
            "--user_tag", tag,
            "--Eval_DataType", "test"
        ]
        
        print(f">>> [EXEC] {' '.join(eval_cmd)}")
        subprocess.run(eval_cmd, check=True)
        
        print(f"✅ {tag} 단계 완료. 10초간 대기합니다...")
        time.sleep(10)

    # [STEP 4] 📊 최종 엑셀 병합
    print(f"\n📊 모든 실험이 종료되었습니다. 결과를 통합합니다...")
    excel_pattern = os.path.join(project_dir, "**", "*_Results_ME*.xlsx")
    found_files = glob.glob(excel_pattern, recursive=True)

    if found_files:
        master_sheets = {}
        # 파일명을 기준으로 Anchor가 먼저 오도록 정렬
        for fpath in sorted(found_files, reverse=True): 
            col_label = "3Loss_Anchor" if "3Loss" in fpath else "4Loss_Random"
            with pd.ExcelFile(fpath) as xls:
                for sn in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sn)
                    if sn not in master_sheets:
                        for col in df.columns[1:]: df.rename(columns={col: f"{col}_{col_label}"}, inplace=True)
                        master_sheets[sn] = df
                    else:
                        data_only = df.iloc[:, 1:].copy()
                        for col in data_only.columns: data_only.rename(columns={col: f"{col}_{col_label}"}, inplace=True)
                        master_sheets[sn] = pd.concat([master_sheets[sn], data_only], axis=1)

        master_path = os.path.join(project_dir, f"Consolidated_Report_NormON_{exp_name}.xlsx")
        with pd.ExcelWriter(master_path, engine='openpyxl') as writer:
            for sn, merged_df in master_sheets.items(): merged_df.to_excel(writer, sheet_name=sn, index=False)
        print(f"🎉 통합 보고서 생성 완료: {master_path}")