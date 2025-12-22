'''
@Author: Yuan Wang (Modified by Researcher 5)
@File: eval_all.py
@Description: Strict Search + Name Tunneling + Detailed Report + Correct 'Average Std' Calculation
'''

from __future__ import print_function, division
import sys
import torch
import numpy as np
import os
import warnings
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from My_args import parser
from PAConv_model import PAConv
from util import landmark_regression
from augmentations import normalize_data

matplotlib.use('Agg')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.manifold._mds")

args = parser.parse_args()
args.eval = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# [기능] 3각도 히트맵 저장
# -----------------------------------------------------------------------------

def save_multiview_heatmap(points, heatmap, save_dir, sample_name, landmark_idx, prefix):
    fig = plt.figure(figsize=(30, 10))
    views = [
        (131, 90, -100, "Front"),
        (132, 30, 120,  "Side"),
        (133, 45, -45, "Downside")
    ]
    for pos, elev, azim, title in views:
        ax = fig.add_subplot(pos, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=heatmap, cmap='jet', s=15, alpha=0.8)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.axis('off')

    # [핵심 변경] 인덱스(sample_idx) 대신 이름(sample_name) 사용
    filename = f"{prefix}_{sample_name}_L{landmark_idx:02d}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=100, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# 1. 경로 탐색 (Strict Search)
# -----------------------------------------------------------------------------
if not args.run_id:
    print("Error: --run_id required (e.g., '1').")
    sys.exit(1)

project_dir = os.path.join(args.output_root, args.exp_name)
if not os.path.exists(project_dir):
    print(f"Error: Project folder not found: {project_dir}")
    sys.exit(1)

target_prefix = f"FPS{args.num_points}_sigma{args.sigma}"
target_suffix = f"_Run_{args.run_id}"

found_folder = None
candidates = []

for d in os.listdir(project_dir):
    d_path = os.path.join(project_dir, d)
    if not os.path.isdir(d_path): continue

    if d.startswith(target_prefix) and d.endswith(target_suffix):
        if args.tag:
            if f"_{args.tag}_" in d: candidates.append(d)
        else:
            candidates.append(d)

if len(candidates) == 0:
    print(f"Error: Cannot find folder with:")
    print(f"  - Prefix: {target_prefix}")
    print(f"  - Suffix: {target_suffix}")
    print(f"  - Tag   : {args.tag}")
    sys.exit(1)
elif len(candidates) > 1:
    print(f"[Warning] Multiple folders found. Using the first one:")
    for c in candidates: print(f" - {c}")
    found_folder = candidates[0]
else:
    found_folder = candidates[0]

# 학습 정보 파싱
train_batch_str = "Unknown"
train_n_str = "Unknown"
try:
    parts = found_folder.split('_')
    for p in parts:
        if p.startswith('B') and p[1:].isdigit():
            train_batch_str = p[1:]
        elif p.startswith('N') and p[1:].isdigit():
            train_n_str = p[1:]
except:
    pass

run_root = os.path.join(project_dir, found_folder)
model_path = os.path.join(run_root, 'models', args.model_epoch)
data_dir = os.path.join(run_root, 'npy_data')

if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    sys.exit(1)

heatmap_save_dir = os.path.join(run_root, "Pred_Heatmaps")
asc_save_dir = os.path.join(run_root, "Pred_Landmarks")
os.makedirs(heatmap_save_dir, exist_ok=True)
os.makedirs(asc_save_dir, exist_ok=True)

print(f"Target Run  : {found_folder}")
print(f"Loading Data : {data_dir}")

# -----------------------------------------------------------------------------
# 2. 데이터 로드 (Names 포함)
# -----------------------------------------------------------------------------
try:
    shape_sample = np.load(os.path.join(data_dir, f"shape_{args.Eval_DataType}.npy"), allow_pickle=True)
    landmark_all = np.load(os.path.join(data_dir, f"landmark_{args.Eval_DataType}.npy"), allow_pickle=True)
    heatmap_sample = np.load(os.path.join(data_dir, f"Heat_data_{args.Eval_DataType}.npy"), allow_pickle=True)
    names_sample = np.load(os.path.join(data_dir, f"names_{args.Eval_DataType}.npy"), allow_pickle=True)
except FileNotFoundError:
    print(f"Error: Backup NPY files not found in {data_dir}.")
    sys.exit(1)

test_dataset = TensorDataset(
    torch.tensor(shape_sample, dtype=torch.float32),
    torch.tensor(landmark_all, dtype=torch.float32),
    torch.tensor(heatmap_sample, dtype=torch.float32)
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# -----------------------------------------------------------------------------
# 3. 모델 로드 및 평가
# -----------------------------------------------------------------------------
model = PAConv(args, args.landmark_num).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

me_list = []
per_landmark_me_list = []

print(f"\n>>> Starting Evaluation ({args.Eval_DataType})...")

for idx, (point, gt_landmark, heatmap) in enumerate(tqdm(test_loader, desc="Evaluating")):
    point = point.to(device)
    gt_landmark = gt_landmark.to(device)
    current_name = str(names_sample[idx])

    B, N, C = point.shape
    centroid = torch.mean(point, axis=1, keepdim=True)
    point_centered = point - centroid
    scale = torch.max(torch.sqrt(torch.sum(point_centered ** 2, axis=2)), axis=1)[0].view(-1, 1, 1)
    point_norm = point_centered / scale 
    
    with torch.no_grad():
        pred_heatmap = model(point_norm.permute(0, 2, 1)).permute(0, 2, 1)
        pred_landmark_norm = landmark_regression(point_norm[0], pred_heatmap[0], args.regression_point_num, idx)
        pred_landmark = (pred_landmark_norm * scale) + centroid

        if idx % 20 == 0:
            points_np = point[0].cpu().numpy()
            heatmap_np = pred_heatmap[0].cpu().numpy()
            for lm_idx in range(heatmap_np.shape[1]):
                 save_multiview_heatmap(points_np, heatmap_np[:, lm_idx], 
                                        heatmap_save_dir, current_name, lm_idx, "pred")

        pred_np = pred_landmark.cpu().numpy().squeeze()
        gt_np = gt_landmark[0].cpu().numpy().squeeze()
        if pred_np.ndim == 1: pred_np = pred_np.reshape(-1, 3)
        if gt_np.ndim == 1: gt_np = gt_np.reshape(-1, 3)

        dists = np.linalg.norm(pred_np - gt_np, axis=1)
        me = np.mean(dists)
        
        me_list.append(me)
        per_landmark_me_list.append(dists)
        
        np.savetxt(os.path.join(asc_save_dir, f"pred_{current_name}.asc"), pred_np, fmt="%.6f", delimiter=",")

# -----------------------------------------------------------------------------
# 4. 결과 집계 및 상세 리포트 작성
# -----------------------------------------------------------------------------
if len(per_landmark_me_list) > 0:
    per_landmark_me_array = np.stack(per_landmark_me_list, axis=0) # (Samples, Landmarks)
    
    # [검증]
    # 1. 각 랜드마크별로 샘플들의 Error에 대한 Std를 구함 (Landmarks,)
    lm_stds = np.std(per_landmark_me_array, axis=0)
    
    # 2. 각 랜드마크별로 샘플들의 Error에 대한 Mean을 구함 (Landmarks,)
    lm_means = np.mean(per_landmark_me_array, axis=0)
    
    # 3. Average Std = (각 랜드마크 Std들의 평균)
    average_std = np.mean(lm_stds)
    average_me = np.mean(lm_means)
else:
    average_me, average_std = 0.0, 0.0
    lm_means, lm_stds = np.array([]), np.array([])

sr_10 = np.sum(np.array(me_list) < 10.0) / len(me_list) * 100
sr_5  = np.sum(np.array(me_list) < 5.0) / len(me_list) * 100

filename = f"ME{average_me:.4f}_std{average_std:.4f}.txt"
result_txt_path = os.path.join(run_root, filename)

with open(result_txt_path, "w") as f:
    f.write(f"==================================================\n")
    f.write(f"           EVALUATION REPORT\n")
    f.write(f"==================================================\n")
    f.write(f" [Experiment Info]\n")
    f.write(f" Folder Name : {found_folder}\n")
    f.write(f" Run ID      : {args.run_id}\n")
    f.write(f" Tag Info    : {args.tag if args.tag else 'None'}\n")
    f.write(f"--------------------------------------------------\n")
    f.write(f" [Training Configuration]\n")
    f.write(f" Train Batch : {train_batch_str}\n")
    f.write(f" Train Count : {train_n_str} samples\n")
    f.write(f" Model Epoch : {args.model_epoch}\n")
    f.write(f"--------------------------------------------------\n")
    f.write(f" [Evaluation Statistics]\n")
    f.write(f" Eval Count  : {len(me_list)} samples\n")
    f.write(f" Data Type   : {args.Eval_DataType}\n")
    f.write(f" Average ME  : {average_me:.4f} mm\n")
    f.write(f" Average Std : {average_std:.4f} mm\n") # [확인] Average of Stds
    f.write(f" SR @ 10mm   : {sr_10:.2f} %\n")
    f.write(f" SR @ 5mm    : {sr_5:.2f} %\n")
    f.write(f"==================================================\n")
    
    if len(lm_means) > 0:
        f.write("\n>>> Top 5 Hardest Landmarks:\n")
        worst_indices = np.argsort(lm_means)[::-1][:5]
        for i in worst_indices:
            f.write(f"    LM {i:02d}: {lm_means[i]:.3f} ± {lm_stds[i]:.3f} mm\n")
        
        f.write("\n>>> Top 5 Easiest Landmarks:\n")
        best_indices = np.argsort(lm_means)[:5]
        for i in best_indices:
            f.write(f"    LM {i:02d}: {lm_means[i]:.3f} ± {lm_stds[i]:.3f} mm\n")
        
        f.write("\n>>> Per-landmark ME (Mean ± Std):\n")
        for i in range(lm_means.shape[0]):
            f.write(f"    LM {i:02d}: {lm_means[i]:.3f} ± {lm_stds[i]:.3f} mm\n")

print(f"\n[Done] Detailed Report saved to: {run_root}")
print(f"      Filename: {filename}")
print(f"Average ME: {average_me:.4f} ± {average_std:.4f}")