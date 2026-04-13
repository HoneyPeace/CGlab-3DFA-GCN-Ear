import os
import sys
import subprocess
import numpy as np
import pandas as pd
import glob
import re
import matplotlib.pyplot as plt

# ==========================================
# 1. 실험 설정 (Target Norm: 0.5, 1.0, 1.5, 2.0)
# ==========================================
target_norms = [0.5, 1.0, 1.5, 2.0]

exp_name = "DeepPA_TargetNorm_Sweep"
output_root = "./output"
user_tag_base = "tnorm"

# 기본 실행 명령어 (시그마는 2.5로 고정 예시)
base_cmd = [
    sys.executable, "run.py",
    "--model", "DeepPA_auto",
    "--train_dataset_name", "train",
    "--test_dataset_name", "test",
    "--sigma", "2.5",
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
# 2. Target Norm Loop 실행 (데이터 리샘플링 최적화 적용)
# ==========================================
for i, tnorm in enumerate(target_norms):
    tnorm_str = f"{tnorm:.1f}"
    current_tag = f"{user_tag_base}_{tnorm_str}"
    
    # 데이터는 바뀌지 않으므로, 첫 번째 루프(i==0)에서만 데이터를 굽습니다(True).
    resample_flag = "True" if i == 0 else "False"
    
    cmd = base_cmd + [
        "--target_norm", tnorm_str, 
        "--need_resample", resample_flag,
        "--user_tag", current_tag
    ]
    
    print(f"\n{'='*75}")
    print(f" 🚀 [Target Norm Sweep] 실행 | Target Norm: {tnorm_str} (Resample: {resample_flag})")
    print(f" 📂 Tag: {current_tag}")
    print(f"{'='*75}\n")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n🚨 [Error] Target Norm {tnorm_str} 실험 중 오류 발생: {e}")
        continue

# ==========================================
# 3. 결과 엑셀 파일 수집 및 병합
# ==========================================
print(f"\n{'='*75}")
print(" 📊 [병합 작업] 모든 Target Norm 결과를 하나의 엑셀로 합칩니다...")
print(f"{'='*75}\n")

project_dir = os.path.join(output_root, exp_name)
excel_pattern = os.path.join(project_dir, "**", "DeepPA_Final_Results_ME*.xlsx")
found_files = glob.glob(excel_pattern, recursive=True)

# 폴더명에서 tnorm 값을 추출하여 오름차순 정렬
def sort_key(path):
    match = re.search(r"tnorm_(\d+\.\d+)", path)
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

plot_tnorms = []
plot_me_values = []
me_row_index = None

for i, fpath in enumerate(found_files):
    current_tnorm = sort_key(fpath)
    col_label = f"TNorm_{current_tnorm:.1f}"
    print(f"  - 병합 중: {os.path.basename(fpath)} ({col_label})")
    
    with pd.ExcelFile(fpath) as xls:
        for sn in final_sheets.keys():
            if sn in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sn)
                
                if sn == "1_Summary":
                    if i == 0:
                        me_row_index = df[df.iloc[:, 0] == "Average ME (mm)"].index[0]
                    me_value = float(df.iloc[me_row_index, -1])
                    plot_tnorms.append(current_tnorm)
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

master_name = f"TargetNorm_Sweep_Consolidated_Results_{exp_name}.xlsx"
master_path = os.path.join(project_dir, master_name)

with pd.ExcelWriter(master_path, engine='openpyxl') as writer:
    for sn, df_list in final_sheets.items():
        if df_list:
            merged_df = pd.concat(df_list, axis=1)
            merged_df.to_excel(writer, sheet_name=sn, index=False)

# ==========================================
# 4. 📈 Target Norm별 ME 변화 그래프 생성
# ==========================================
if plot_tnorms and plot_me_values:
    print(f"\n{'='*75}")
    print(" 📈 [그래프 생성] Target Norm에 따른 Average ME 변화 그래프를 그립니다...")
    
    plt.figure(figsize=(10, 6))
    plt.plot(plot_tnorms, plot_me_values, marker='D', linestyle='-', color='#ff7f0e', linewidth=2.5, markersize=8)
    
    plt.title('Average ME Trend by Loss Target Norm', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Target Norm Value', fontsize=14, fontweight='bold')
    plt.ylabel('Average ME (mm)', fontsize=14, fontweight='bold')
    
    plt.xticks(plot_tnorms, fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)

    y_offset = (max(plot_me_values) - min(plot_me_values)) * 0.05
    if y_offset == 0: y_offset = 0.01  # 값들이 모두 같을 경우를 대비한 안전 장치

    for x, y in zip(plot_tnorms, plot_me_values):
        plt.text(x, y + y_offset, f'{y:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#d62728')

    graph_path = os.path.join(project_dir, f"TargetNorm_vs_ME_Trend_{exp_name}.png")
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ [완료] 통합 엑셀 및 Target Norm 그래프 저장이 완료되었습니다!")
    print(f"📁 마스터 엑셀: {os.path.abspath(master_path)}")
    print(f"🖼️ 그래프 경로: {os.path.abspath(graph_path)}")