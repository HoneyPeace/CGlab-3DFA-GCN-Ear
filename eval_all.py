from __future__ import print_function, division
import sys
import torch
import argparse
import numpy as np
import os
import warnings
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from My_args import parser
from PAConv_model import PAConv
from util import landmark_regression, get_3D_FAN_NME
from augmentations import normalize_data # 정규화 함수 임포트

matplotlib.use('Agg')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.manifold._mds")

args = parser.parse_args()
args.eval = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# 2. 경로 설정 (Smart Load)
# -----------------------------------------------------------------------------
if args.use_split_dataset:
    target_dataset_folder = args.dataset
    print_target_name = args.dataset
else:
    target_dataset_folder = args.train_dataset_name
    print_target_name = args.test_dataset_name

if args.run_id:
    folder_prefix = f"FPS{args.num_points}_sigma{args.sigma}"
    full_folder_name = f"{folder_prefix}_{args.run_id}"
    backup_root = os.path.join(os.path.dirname(args.output_root), 'backup')
    run_root = os.path.join(backup_root, target_dataset_folder, full_folder_name)
    
    args.model_path = os.path.join(run_root, 'models', args.model_epoch)
    base_dir = os.path.join(run_root, 'npy_data')
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        sys.exit(1)
else:
    print("Error: --run_id required.")
    sys.exit(1)

print(f"Loading Model: {args.model_path}")
print(f"Loading Data : {base_dir}")

eval_save_dir = os.path.join(run_root, f"Eval_Result_{args.Eval_DataType}_Reg{args.regression_point_num}")
heatmap_save_dir = os.path.join(eval_save_dir, "Pred_Heatmaps")
asc_save_dir = os.path.join(eval_save_dir, "Pred_Landmarks_ASC")
os.makedirs(eval_save_dir, exist_ok=True)
os.makedirs(heatmap_save_dir, exist_ok=True)
os.makedirs(asc_save_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# 3. 데이터 로드
# -----------------------------------------------------------------------------
try:
    shape_sample = np.load(os.path.join(base_dir, f"shape_{args.Eval_DataType}.npy"), allow_pickle=True)
    landmark_all = np.load(os.path.join(base_dir, f"landmark_{args.Eval_DataType}.npy"), allow_pickle=True)
    heatmap_sample = np.load(os.path.join(base_dir, f"Heat_data_{args.Eval_DataType}.npy"), allow_pickle=True)
except FileNotFoundError:
    print(f"Error: Data files not found.")
    sys.exit(1)

test_dataset = TensorDataset(
    torch.tensor(shape_sample, dtype=torch.float32),
    torch.tensor(landmark_all, dtype=torch.float32),
    torch.tensor(heatmap_sample, dtype=torch.float32)
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = PAConv(args, args.landmark_num).to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()

# -----------------------------------------------------------------------------
# 5. 평가 루프 (Normalization -> Inference -> Denormalization)
# -----------------------------------------------------------------------------
me_list = []
per_landmark_me_list = []

print("\n>>> Starting Evaluation (With Normalization)...")

for idx, (point, gt_landmark, heatmap) in enumerate(tqdm(test_loader, desc="Evaluating")):
    point = point.to(device)
    gt_landmark = gt_landmark.to(device)
    
    # [1] 정규화 파라미터 계산 (복원용)
    # augmentations.py의 normalize_data 로직 역추적
    B, N, C = point.shape
    centroid = torch.mean(point, axis=1, keepdim=True) # 중심점
    point_centered = point - centroid
    m = torch.max(torch.sqrt(torch.sum(point_centered ** 2, axis=2)), axis=1)[0] # 최대 거리(스케일)
    scale = m.view(-1, 1, 1)
    
    # [2] 정규화 수행 (모델 입력용)
    point_norm = point_centered / scale 
    
    with torch.no_grad():
        # [3] 모델 예측 (정규화된 좌표 기반)
        pred_heatmap_raw = model(point_norm.permute(0, 2, 1))
        pred_heatmap = pred_heatmap_raw.permute(0, 2, 1)

        # [4] 회귀 (Normalized 좌표계에서 예측 좌표 추출)
        # 중요: Regression할 때 'normalized shape'를 써야 함
        pred_landmark_norm = landmark_regression(point_norm[0], pred_heatmap[0], args.regression_point_num, idx)
        
        # [5] 복원 (Denormalization) -> 원래 mm 단위로 변환
        # 식: (Pred_Norm * Scale) + Centroid
        pred_landmark = (pred_landmark_norm * scale) + centroid

        # 시각화 (20개마다)
        if idx % 20 == 0:
            points_np = point[0].cpu().numpy()
            heatmap_np = pred_heatmap[0].cpu().numpy()
            colors = heatmap_np[:, 0]
            
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=colors, cmap='jet', s=15, alpha=0.8)
            ax.view_init(elev=90, azim=-90)
            ax.set_title(f"Sample {idx} LM 0 (Front)")
            ax.axis('off')
            plt.savefig(os.path.join(heatmap_save_dir, f"sample{idx:03d}_pred.png"))
            plt.close()

        # ME 계산
        pred_np = pred_landmark.cpu().numpy()
        gt_np = gt_landmark[0].cpu().numpy()
        if pred_np.ndim == 3: pred_np = pred_np.squeeze(0)
        if gt_np.ndim == 3: gt_np = gt_np.squeeze(0)
            
        dists = np.linalg.norm(pred_np - gt_np, axis=1)
        me = np.mean(dists)
        
        me_list.append(me)
        per_landmark_me_list.append(dists)
        
        # ASC 저장
        np.savetxt(os.path.join(asc_save_dir, f"pred_{idx:03d}.asc"), pred_np, fmt="%.6f", delimiter=",")

average_me = np.mean(me_list)
std_me = np.std(me_list)
sr_10 = np.sum(np.array(me_list) < 10.0) / len(me_list) * 100
sr_5  = np.sum(np.array(me_list) < 5.0) / len(me_list) * 100

print(f"\n==========================================")
print(f"   Evaluation Result: {args.exp_name}")
print(f"   (Logic: Normalized Input / Denormalized Pred)")
print(f"==========================================")
print(f" Dataset : {print_target_name}")
print(f" Sigma   : {args.sigma}")
print(f" Reg Pts : {args.regression_point_num}")
print(f"------------------------------------------")
print(f" Average ME : {average_me:.4f} mm")
print(f" Std of ME  : {std_me:.4f} mm")
print(f" SR @ 10mm  : {sr_10:.2f} %")
print(f" SR @ 5mm   : {sr_5:.2f} %")
print(f"==========================================\n")

if len(per_landmark_me_list) > 0:
    per_landmark_me_array = np.stack(per_landmark_me_list, axis=0)
    lm_mean = np.mean(per_landmark_me_array, axis=0)
    lm_std = np.std(per_landmark_me_array, axis=0)
    
    print(">>> Top 5 Hardest Landmarks:")
    worst_indices = np.argsort(lm_mean)[::-1][:5]
    for i in worst_indices:
        print(f"    LM {i:02d}: {lm_mean[i]:.3f} ± {lm_std[i]:.3f} mm")