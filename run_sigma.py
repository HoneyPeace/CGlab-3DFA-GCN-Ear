import os
import sys
import subprocess
import numpy as np
import pandas as pd
import glob
import re
import matplotlib.pyplot as plt

# ==========================================
# 1. 실험 설정
# ==========================================
sigmas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

exp_name = "DeepPA최종정리"
output_root = "./output"
user_tag_base = "sigma"

base_cmd = [
    sys.executable, "run.py",
    "--model", "DeepPA_auto",
    "--train_dataset_name", "train",
    "--test_dataset_name", "test",
    "--num_points", "8192",
    "--batch_size", "8",
    "--accumulation_steps", "4",
    "--dataset_seed", "1",
    "--epochs", "500",
    "--sample_way", "FPS",
    "--k_softargmax", "10",
    "--plane_knn", "5",
    "--exp_name", exp_name,
    "--landmark_num", "36",
    "--in_channels", "7"
]

# ==========================================
# 2. Sigma Loop 실행 (🌟 무조건 리샘플링)
# ==========================================
for sigma in sigmas:
    sigma_str = f"{sigma:.1f}"
    current_tag = f"{user_tag_base}_{sigma_str}"
    
    # 🌟 시그마가 바뀔 때마다 무조건 GT 히트맵을 새로 구워야 하므로 "True" 고정!
    cmd = base_cmd + [
        "--sigma", sigma_str, 
        "--need_resample", "True", 
        "--user_tag", current_tag
    ]
    
    print(f"\n{'='*70}")
    print(f" 🚀 [Sigma Sweep] 실행 시작 | Sigma: {sigma_str} (Resample: True)")
    print(f"{'='*70}\n")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n🚨 [Error] Sigma {sigma_str} 실험 중 오류가 발생했습니다: {e}")
        continue

# ==========================================
# 3. 결과 엑셀 파일 수집 및 병합 (Master Excel 생성)
# ==========================================
print(f"\n{'='*70}")
print(" 📊 [병합 작업] 모든 시그마 결과를 하나의 엑셀 파일로 합칩니다...")
print(f"{'='*70}\n")

project_dir = os.path.join(output_root, exp_name)
excel_pattern = os.path.join(project_dir, "**", "DeepPA_Final_Results_ME*.xlsx")
found_files = glob.glob(excel_pattern, recursive=True)

def sort_key(path):
    match = re.search(r"sigma_(\d+\.\d+)", path)
    return float(match.group(1)) if match else 999.0

found_files.sort(key=sort_key)

if not found_files:
    print("🚨 [Error] 병합할 결과 엑셀 파일을 찾을 수 없습니다.")
    sys.exit(1)

final_sheets = {
    "1_Summary": [],
    "2_Per_Landmark": [],
    "3_Top10_Hardest": [],
    "4_Heatmap_Metrics": []
}

plot_sigmas = []
plot_me_values = []
me_row_index = None

for i, fpath in enumerate(found_files):
    current_sigma = sort_key(fpath)
    col_label = f"Sigma_{current_sigma:.1f}"
    print(f"  - 병합 중: {os.path.basename(fpath)} ({col_label})")
    
    with pd.ExcelFile(fpath) as xls:
        for sn in final_sheets.keys():
            if sn in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sn)
                
                if sn == "1_Summary":
                    if i == 0:
                        me_row_index = df[df.iloc[:, 0] == "Average ME (mm)"].index[0]
                    me_value = float(df.iloc[me_row_index, -1])
                    plot_sigmas.append(current_sigma)
                    plot_me_values.append(me_value)

                if i == 0:
                    for col in df.columns[1:]:
                        df.rename(columns={col: f"{col}_{col_label}"}, inplace=True)
                    final_sheets[sn].append(df)
                else:
                    data_only = df.iloc[:, 1:].copy()
                    for col in data_only.columns:
                        data_only.rename(columns={col: f"{col}_{col_label}"}, inplace=True)
                    final_sheets[sn].append(data_only)

master_name = f"Sigma_Sweep_Consolidated_Results_{exp_name}.xlsx"
master_path = os.path.join(project_dir, master_name)

with pd.ExcelWriter(master_path, engine='openpyxl') as writer:
    for sn, df_list in final_sheets.items():
        if df_list:
            merged_df = pd.concat(df_list, axis=1)
            merged_df.to_excel(writer, sheet_name=sn, index=False)

# ==========================================
# 4. 📈 시그마별 ME 변화 추이 그래프 생성 및 저장
# ==========================================
if plot_sigmas and plot_me_values:
    print(f"\n{'='*70}")
    print(" 📈 [그래프 생성] 시그마(Sigma)에 따른 Average ME 변화 추이 그래프를 그립니다...")
    print(f"{'='*70}\n")

    # 🌟 점 개수가 늘어났으므로 가로 폭을 약간 넓게(12, 6) 조정하여 텍스트 겹침 방지
    plt.figure(figsize=(12, 6))
    
    # 🌟 matplotlib의 plot은 기본적으로 x값의 크기에 비례해서 알아서 간격을 그려줌!
    plt.plot(plot_sigmas, plot_me_values, marker='o', linestyle='-', color='#1f77b4', linewidth=2.5, markersize=8)
    
    plt.title('Average ME Trend by Heatmap Sigma', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Sigma Value', fontsize=14, fontweight='bold')
    plt.ylabel('Average ME (mm)', fontsize=14, fontweight='bold')
    
    # x축 눈금을 실험한 시그마 값으로 출력하되, 비례 간격은 유지됨
    plt.xticks(plot_sigmas, fontsize=10, rotation=45) 
    plt.yticks(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 각 점 위에 수치 텍스트 표시
    y_offset = (max(plot_me_values) - min(plot_me_values)) * 0.02
    for x, y in zip(plot_sigmas, plot_me_values):
        plt.text(x, y + y_offset, 
                 f'{y:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#d62728')

    graph_name = f"Sigma_vs_ME_Trend_{exp_name}.png"
    graph_path = os.path.join(project_dir, graph_name)
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ [완료] 엑셀 병합 및 그래프 저장이 모두 끝났습니다!")
    print(f"📁 마스터 엑셀: {os.path.abspath(master_path)}")
    print(f"🖼️ 시그마 그래프: {os.path.abspath(graph_path)}")
else:
    print("\n🚨 [Warning] 그래프를 그리기 위한 ME 데이터를 추출하지 못했습니다.")