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
batch_configs = [
    (2, 1), (4, 1), (8, 1),  
    (8, 2), (8, 4), (8, 8)   
]

exp_name = "DeepPA최종정리"
output_root = "./output"
user_tag_base = "batch"

# 🌟 base_cmd에서 need_resample을 빼고 루프에서 동적 할당
base_cmd = [
    sys.executable, "run.py",
    "--model", "DeepPA_auto",
    "--train_dataset_name", "train",
    "--test_dataset_name", "test",
    "--sigma", "2.5",
    "--num_points", "8192",
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
# 2. Batch Loop 실행 (🌟 리샘플링 동적 제어)
# ==========================================
for i, (b_size, a_steps) in enumerate(batch_configs):
    eff_batch = b_size * a_steps
    current_tag = f"{user_tag_base}_{eff_batch}_phys{b_size}_acc{a_steps}"
    
    # 🌟 첫 번째 루프(i==0)에서만 True, 나머지는 False!
    resample_flag = "True" if i == 0 else "False"
    
    cmd = base_cmd + [
        "--batch_size", str(b_size), 
        "--accumulation_steps", str(a_steps), 
        "--need_resample", resample_flag,
        "--user_tag", current_tag
    ]
    
    print(f"\n{'='*75}")
    print(f" 🚀 [Batch Sweep] 유효 배치: {eff_batch} (Resample: {resample_flag})")
    print(f"{'='*75}\n")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n🚨 [Error] Batch {eff_batch} 실험 중 오류 발생: {e}")
        continue

# ==========================================
# 3. 결과 엑셀 파일 수집 및 병합
# ==========================================
print(f"\n{'='*75}")
print(" 📊 [병합 작업] 모든 배치 결과를 하나의 마스터 엑셀로 합칩니다...")
print(f"{'='*75}\n")

project_dir = os.path.join(output_root, exp_name)
excel_pattern = os.path.join(project_dir, "**", "DeepPA_Final_Results_ME*.xlsx")
found_files = glob.glob(excel_pattern, recursive=True)

# 폴더명에서 유효 배치 사이즈(batch_숫자)를 추출하여 정렬
def sort_key(path):
    match = re.search(r"batch_(\d+)_", path)
    return int(match.group(1)) if match else 999

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

plot_batches = []
plot_me_values = []
me_row_index = None

for i, fpath in enumerate(found_files):
    eff_batch = sort_key(fpath)
    col_label = f"Batch_{eff_batch}"
    print(f"  - 병합 중: {os.path.basename(fpath)} ({col_label})")
    
    with pd.ExcelFile(fpath) as xls:
        for sn in final_sheets.keys():
            if sn in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sn)
                
                if sn == "1_Summary":
                    if i == 0:
                        # "Average ME (mm)" 행 위치 찾기
                        me_row_index = df[df.iloc[:, 0] == "Average ME (mm)"].index[0]
                    me_value = float(df.iloc[me_row_index, -1])
                    plot_batches.append(eff_batch)
                    plot_me_values.append(me_value)

                # 첫 번째 파일은 메트릭 이름 포함, 이후 파일은 데이터 열만 추가
                if i == 0:
                    for col in df.columns[1:]:
                        df.rename(columns={col: f"{col}_{col_label}"}, inplace=True)
                    final_sheets[sn].append(df)
                else:
                    data_only = df.iloc[:, 1:].copy()
                    for col in data_only.columns:
                        data_only.rename(columns={col: f"{col}_{col_label}"}, inplace=True)
                    final_sheets[sn].append(data_only)

master_name = f"Batch_Sweep_Consolidated_Results_{exp_name}.xlsx"
master_path = os.path.join(project_dir, master_name)

with pd.ExcelWriter(master_path, engine='openpyxl') as writer:
    for sn, df_list in final_sheets.items():
        if df_list:
            merged_df = pd.concat(df_list, axis=1)
            merged_df.to_excel(writer, sheet_name=sn, index=False)

# ==========================================
# 4. 📈 배치 사이즈별 ME 변화 그래프 생성
# ==========================================
if plot_batches and plot_me_values:
    print(f"\n{'='*75}")
    print(" 📈 [그래프 생성] Effective Batch Size에 따른 Average ME 변화 그래프를 그립니다...")
    
    plt.figure(figsize=(10, 6))
    plt.plot(plot_batches, plot_me_values, marker='s', linestyle='--', color='#2ca02c', linewidth=2, markersize=9)
    
    plt.title('Average ME Trend by Effective Batch Size', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Effective Batch Size (Phys Batch x Accumulation)', fontsize=13, fontweight='bold')
    plt.ylabel('Average ME (mm)', fontsize=13, fontweight='bold')
    
    # x축을 로그 스케일로 표시하거나 실제 실험값으로 픽스 (2, 4, 8, 16, 32, 64)
    plt.xscale('log', base=2)
    plt.xticks(plot_batches, labels=[str(b) for b in plot_batches], fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, which='both', linestyle=':', alpha=0.6)

    for x, y in zip(plot_batches, plot_me_values):
        plt.text(x, y + (max(plot_me_values)-min(plot_me_values))*0.03, f'{y:.3f}', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='#d62728')

    graph_path = os.path.join(project_dir, f"Batch_vs_ME_Trend_{exp_name}.png")
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ [완료] 통합 엑셀 및 배치 그래프 저장이 완료되었습니다!")
    print(f"📁 마스터 엑셀: {os.path.abspath(master_path)}")
    print(f"🖼️ 배치 그래프: {os.path.abspath(graph_path)}")