import os
import sys
import subprocess
import numpy as np
import pandas as pd
import glob
import re
import matplotlib.pyplot as plt

# ==========================================
# 1. 실험 설정 (PAConv 히트맵 전용)
# ==========================================
batch_configs = [
    (1, 1), (2, 1), (4, 1), (8, 1),  
    (8, 2), (8, 4), (8, 8)   
]

exp_name = "PAConv_Heatmap_Sweep" # 실험 폴더명
output_root = "../results"       # 결과 저장 루트
user_tag_base = "PAConv_S1"      # 🌟 디스크에 있는 대문자 태그 유지

# 공통 실행 인자 (eval.py 실행용)
common_args = [
    "--model", "paconv",         # 🌟 argparse 에러 방지를 위해 소문자 유지
    "--train_dataset_name", "train",
    "--test_dataset_name", "test",
    "--sigma", "2.5",            # 스크린샷의 sigma2.5와 일치[cite: 6]
    "--num_points", "8192",      # 스크린샷의 FPS8192와 일치[cite: 6]
    "--dataset_seed", "1",
    "--epochs", "500",
    "--sample_way", "FPS",
    "--exp_name", exp_name,
    "--landmark_num", "36",
    "--in_channels", "7"
]

# ==========================================
# 2. Batch Loop 실행 (평가만 수행)
# ==========================================
for i, (b_size, a_steps) in enumerate(batch_configs):
    eff_batch = b_size * a_steps
    # 🌟 스크린샷의 실제 폴더명 규칙에 따른 태그 생성
    current_tag = f"{user_tag_base}_batch_{eff_batch}_phys{b_size}_acc{a_steps}"
    
    print(f"\n{'='*75}")
    print(f" 🔍 [Evaluation Mode] 유효 배치: {eff_batch} (Folder: {current_tag})")
    print(f"{'='*75}\n")
    
    # [STEP 1] 학습 실행은 이미 완료되었으므로 건너뜁니다.
    # -------------------------------------------------------
    # try:
    #     subprocess.run(train_cmd, check=True)
    # except subprocess.CalledProcessError:
    #     continue

    # [STEP 2] 평가 실행 (eval.py)
    # --user_tag에 대문자가 포함된 실제 폴더 태그를 넣어 경로를 찾게 합니다[cite: 11].
    eval_cmd = [sys.executable, "eval.py"] + common_args + [
        "--batch_size", str(b_size),
        "--accumulation_steps", str(a_steps),
        "--user_tag", current_tag,
        "--run_id", "1",               # 폴더 끝의 _1과 일치[cite: 6]
        "--model_epoch", "Single_PAConv_last.t7", 
        "--Eval_DataType", "test"
    ]
    
    print(f" 📊 [Stage 2: Eval] 배치 {eff_batch} 평가 시작...")
    try:
        # 이미 학습된 가중치를 로드하여 평가를 진행합니다[cite: 11].
        subprocess.run(eval_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n🚨 [Error] Batch {eff_batch} 평가 중 오류 발생: {e}")
        continue

# ==========================================
# 3. 결과 엑셀 파일 수집 및 병합
# ==========================================
print(f"\n{'='*75}")
print(" 📉 [병합 작업] PAConv 배치 결과 마스터 엑셀 생성 중...")
print(f"{'='*75}\n")

project_dir = os.path.join(output_root, exp_name)
# 모든 하위 폴더에서 ME 결과 엑셀 파일을 찾습니다[cite: 11].
excel_pattern = os.path.join(project_dir, "**", "PAConv_Results_ME*.xlsx")
found_files = glob.glob(excel_pattern, recursive=True)

def sort_key(path):
    # 폴더명에서 배치 사이즈 숫자를 추출하여 정렬합니다[cite: 11].
    match = re.search(r"batch_(\d+)_", path)
    return int(match.group(1)) if match else 999

found_files.sort(key=sort_key)

if not found_files:
    print(f"🚨 [Error] 병합할 결과 파일을 찾을 수 없습니다. (Path: {project_dir})")
    sys.exit(1)

final_sheets = {"1_Summary": [], "2_Per_Landmark": [], "3_Top10_Hardest": []}
plot_batches, plot_me_values = [], []

for i, fpath in enumerate(found_files):
    eff_batch = sort_key(fpath)
    col_label = f"Batch_{eff_batch}"
    
    with pd.ExcelFile(fpath) as xls:
        for sn in final_sheets.keys():
            if sn in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sn)
                if sn == "1_Summary":
                    # 요약 시트에서 평균 오차(ME) 값을 추출합니다[cite: 11].
                    try:
                        me_val = float(df[df.iloc[:, 0] == "Average ME (mm)"].iloc[0, 1])
                        plot_batches.append(eff_batch)
                        plot_me_values.append(me_val)
                    except: pass
                
                if i == 0:
                    df.rename(columns={df.columns[1]: f"{df.columns[1]}_{col_label}"}, inplace=True)
                    final_sheets[sn].append(df)
                else:
                    data_only = df.iloc[:, 1:].copy()
                    data_only.columns = [f"{c}_{col_label}" for c in data_only.columns]
                    final_sheets[sn].append(data_only)

# 최종 마스터 엑셀 파일을 저장합니다[cite: 11].
master_path = os.path.join(project_dir, f"PAConv_Batch_Sweep_Results_Final.xlsx")
with pd.ExcelWriter(master_path, engine='openpyxl') as writer:
    for sn, df_list in final_sheets.items():
        if df_list: pd.concat(df_list, axis=1).to_excel(writer, sheet_name=sn, index=False)

# ==========================================
# 4. 📈 배치 사이즈별 ME 변화 그래프 생성
# ==========================================
if plot_batches:
    plt.figure(figsize=(10, 6))
    plt.plot(plot_batches, plot_me_values, marker='o', ls='-', color='royalblue', lw=2)
    plt.xscale('log', base=2); plt.grid(True, which='both', alpha=0.3)
    plt.title(f'PAConv Heatmap ME Trend', fontsize=14, fontweight='bold')
    plt.xlabel('Effective Batch Size'); plt.ylabel('Average ME (mm)')
    plt.xticks(plot_batches, [str(b) for b in plot_batches])
    
    for x, y in zip(plot_batches, plot_me_values):
        plt.text(x, y, f'{y:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.savefig(os.path.join(project_dir, "Batch_ME_Trend_Final.png"), dpi=300)
    print(f"✅ [완료] 통합 결과 및 그래프가 생성되었습니다: {project_dir}")