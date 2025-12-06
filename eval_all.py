'''
@Author: Yuan Wang (Modified by Researcher 2)
@File: eval_all.py
@Description: Evaluation script with 'Every 30 samples, All Landmarks' Visualization.
'''

from __future__ import print_function, division
import sys
import torch
import numpy as np
import os
import warnings
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # [추가] 3D Plot용
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from My_args import parser
from PAConv_model import PAConv
from util import landmark_regression
# 정규화 함수 필요
from augmentations import normalize_data

matplotlib.use('Agg')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.manifold._mds")

args = parser.parse_args()
args.eval = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# [기능] 3각도 히트맵 저장 (train.py와 동일한 함수 추가)
# -----------------------------------------------------------------------------
def save_multiview_heatmap(points, heatmap, save_dir, sample_idx, landmark_idx, prefix):
    fig = plt.figure(figsize=(1, 1))
    views = [
        (131, 90, -90, "Front View"),
        (132, 45, -45, "Diagonal View"),
        (133, 0, -90,  "Side View")
    ]
    for pos, elev, azim, title in views:
        ax = fig.add_subplot(pos, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=heatmap, cmap='jet', s=15, alpha=0.8)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.axis('off')

    filename = f"{prefix}_S{sample_idx:03d}_L{landmark_idx:02d}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=100, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# 1. 경로 설정 (Unified Structure)
# -----------------------------------------------------------------------------
# 사용자가 --run_id 1 처럼 숫자만 입력하면 해당 폴더를 찾아서 설정
if not args.run_id:
    print("Error: --run_id required (e.g., '1' for the first experiment).")
    sys.exit(1)

project_dir = os.path.join(args.output_root, args.exp_name)
setting_str = f"FPS{args.num_points}_sigma{args.sigma}"
target_folder_name = f"{setting_str}_{args.run_id}"

# 통합 실행 폴더 경로
run_root = os.path.join(project_dir, target_folder_name)

if not os.path.exists(run_root):
    print(f"Error: Experiment folder not found: {run_root}")
    sys.exit(1)

# 하위 경로 설정
model_path = os.path.join(run_root, 'models', args.model_epoch)
data_dir = os.path.join(run_root, 'npy_data')

if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    sys.exit(1)

# 결과 저장용 폴더 생성
# 1. Prediction Visualization
heatmap_save_dir = os.path.join(run_root, "Pred_Heatmaps")
# 2. Predicted Landmarks (ASC)
asc_save_dir = os.path.join(run_root, "Pred_Landmarks")

os.makedirs(heatmap_save_dir, exist_ok=True)
os.makedirs(asc_save_dir, exist_ok=True)

print(f"Target Run  : {target_folder_name}")
print(f"Loading Model: {args.model_epoch}")
print(f"Loading Data : {data_dir}")

# -----------------------------------------------------------------------------
# 2. 데이터 로드 (Backup된 NPY 사용)
# -----------------------------------------------------------------------------
try:
    shape_sample = np.load(os.path.join(data_dir, f"shape_{args.Eval_DataType}.npy"), allow_pickle=True)
    landmark_all = np.load(os.path.join(data_dir, f"landmark_{args.Eval_DataType}.npy"), allow_pickle=True)
    heatmap_sample = np.load(os.path.join(data_dir, f"Heat_data_{args.Eval_DataType}.npy"), allow_pickle=True)
except FileNotFoundError:
    print(f"Error: Backup data files not found in {data_dir}.")
    sys.exit(1)

test_dataset = TensorDataset(
    torch.tensor(shape_sample, dtype=torch.float32),
    torch.tensor(landmark_all, dtype=torch.float32),
    torch.tensor(heatmap_sample, dtype=torch.float32)
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# -----------------------------------------------------------------------------
# 3. 모델 로드
# -----------------------------------------------------------------------------
model = PAConv(args, args.landmark_num).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# -----------------------------------------------------------------------------
# 4. 평가 루프
# -----------------------------------------------------------------------------
me_list = []
per_landmark_me_list = []

print(f"\n>>> Starting Evaluation ({args.Eval_DataType})...")

for idx, (point, gt_landmark, heatmap) in enumerate(tqdm(test_loader, desc="Evaluating")):
    point = point.to(device)
    gt_landmark = gt_landmark.to(device)
    
    # [1] 정규화 파라미터 계산 (복원용)
    B, N, C = point.shape
    centroid = torch.mean(point, axis=1, keepdim=True)
    point_centered = point - centroid
    m = torch.max(torch.sqrt(torch.sum(point_centered ** 2, axis=2)), axis=1)[0]
    scale = m.view(-1, 1, 1)
    
    # [2] 정규화 수행
    point_norm = point_centered / scale 
    
    with torch.no_grad():
        # [3] 모델 예측
        pred_heatmap_raw = model(point_norm.permute(0, 2, 1))
        pred_heatmap = pred_heatmap_raw.permute(0, 2, 1)

        # [4] 회귀 (Normalized Space)
        pred_landmark_norm = landmark_regression(point_norm[0], pred_heatmap[0], args.regression_point_num, idx)
        
        # [5] 복원 (Denormalization)
        pred_landmark = (pred_landmark_norm * scale) + centroid

        # [Visualization] 30개마다, 모든 랜드마크 히트맵 저장
        if idx % 30 == 0:
            points_np = point[0].cpu().numpy()
            heatmap_np = pred_heatmap[0].cpu().numpy() # (N, L)
            
            # save_multiview_heatmap은 (N,) 형태의 heatmap intensity를 원함
            # 따라서 랜드마크 갯수(L)만큼 루프를 돌며 각각 저장
            for lm_idx in range(heatmap_np.shape[1]): # L
                 save_multiview_heatmap(points_np, heatmap_np[:, lm_idx], 
                                        heatmap_save_dir, idx, lm_idx, "pred")

        # ME 계산
        pred_np = pred_landmark.cpu().numpy()
        gt_np = gt_landmark[0].cpu().numpy()
        if pred_np.ndim == 3: pred_np = pred_np.squeeze(0)
        if gt_np.ndim == 3: gt_np = gt_np.squeeze(0)
            
        dists = np.linalg.norm(pred_np - gt_np, axis=1)
        me = np.mean(dists)
        
        me_list.append(me)
        per_landmark_me_list.append(dists)
        
        # [ASC 저장]
        np.savetxt(os.path.join(asc_save_dir, f"pred_{idx:03d}.asc"), pred_np, fmt="%.6f", delimiter=",")

# -----------------------------------------------------------------------------
# 5. 결과 집계 및 텍스트 저장 (파일명 변경 적용)
# -----------------------------------------------------------------------------
average_me = np.mean(me_list)
std_me = np.std(me_list)
sr_10 = np.sum(np.array(me_list) < 10.0) / len(me_list) * 100
sr_5  = np.sum(np.array(me_list) < 5.0) / len(me_list) * 100

# [수정됨] 파일명을 ME수치_std수치.txt 형식으로 생성
filename = f"ME{average_me:.4f}_std{std_me:.4f}.txt"
result_txt_path = os.path.join(run_root, filename)

with open(result_txt_path, "w") as f:
    f.write(f"==========================================\n")
    f.write(f"   Evaluation Result: {args.exp_name}\n")
    f.write(f"==========================================\n")
    f.write(f" Run ID     : {target_folder_name}\n")
    f.write(f" Model      : {args.model_epoch}\n")
    f.write(f" Data Type  : {args.Eval_DataType}\n")
    f.write(f" Reg Points : {args.regression_point_num}\n")
    f.write(f"------------------------------------------\n")
    f.write(f" Average ME : {average_me:.4f} mm\n")
    f.write(f" Std of ME  : {std_me:.4f} mm\n")
    f.write(f" SR @ 10mm  : {sr_10:.2f} %\n")
    f.write(f" SR @ 5mm   : {sr_5:.2f} %\n")
    f.write(f"==========================================\n")
    
    if len(per_landmark_me_list) > 0:
        per_landmark_me_array = np.stack(per_landmark_me_list, axis=0)
        lm_mean = np.mean(per_landmark_me_array, axis=0)
        lm_std = np.std(per_landmark_me_array, axis=0)
        
        f.write(">>> Top 5 Hardest Landmarks:\n")
        worst_indices = np.argsort(lm_mean)[::-1][:5]
        for i in worst_indices:
            f.write(f"    LM {i:02d}: {lm_mean[i]:.3f} ± {lm_std[i]:.3f} mm\n")

# 화면 출력
print(f"\n[Done] Results saved to: {run_root}")
print(f"      Filename: {filename}")
print(f"Average ME: {average_me:.4f} ± {std_me:.4f}")