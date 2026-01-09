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
# [기능] 3각도 히트맵 저장 (이름 문자열 처리를 위해 소폭 수정)
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

    # [수정] sample_idx:03d -> sample_name (문자열 그대로 사용)
    filename = f"{prefix}_{sample_name}_L{landmark_idx:02d}.png"
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
base_str = f"FPS{args.num_points}_sigma{args.sigma}_batch{args.batch_size}_train{args.train_len}"

if args.user_tag and args.user_tag != "":
    setting_str = f"{base_str}_{args.user_tag}"
else:
    setting_str = base_str

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
    
    # [추가] 이름 파일 로드 (파일명 매칭용)
    name_path = os.path.join(data_dir, f"name_{args.Eval_DataType}.npy")
    if os.path.exists(name_path):
        name_sample = np.load(name_path, allow_pickle=True)
        print(f">> Sample names loaded: {len(name_sample)} files.")
    else:
        print(">> [Warning] Name file not found. Using Index instead.")
        name_sample = [f"S{i:03d}" for i in range(len(shape_sample))]

    # [추가] 리포트용 학습 데이터 갯수 확인 (파일이 있을 경우만)
    try:
        train_shape_path = os.path.join(data_dir, "shape_train.npy")
        if os.path.exists(train_shape_path):
            train_len = len(np.load(train_shape_path, allow_pickle=True))
        else:
            train_len = "Unknown (File not found)"
    except:
        train_len = "Error checking"

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
    
    # [추가] 현재 샘플의 실제 이름 가져오기
    real_name = name_sample[idx]
    
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
        if idx % 40 == 0:
            points_np = point[0].cpu().numpy()
            heatmap_np = pred_heatmap[0].cpu().numpy() # (N, L)
            
            # save_multiview_heatmap은 (N,) 형태의 heatmap intensity를 원함
            # 따라서 랜드마크 갯수(L)만큼 루프를 돌며 각각 저장
            for lm_idx in range(heatmap_np.shape[1]): # L
                 # [수정] idx 대신 real_name 전달
                 save_multiview_heatmap(points_np, heatmap_np[:, lm_idx], 
                                        heatmap_save_dir, real_name, lm_idx, "pred")

        # ME 계산
        pred_np = pred_landmark.cpu().numpy()
        gt_np = gt_landmark[0].cpu().numpy()
        if pred_np.ndim == 3: pred_np = pred_np.squeeze(0)
        if gt_np.ndim == 3: gt_np = gt_np.squeeze(0)
            
        dists = np.linalg.norm(pred_np - gt_np, axis=1)
        me = np.mean(dists)
        
        me_list.append(me)
        per_landmark_me_list.append(dists)
        
        # [ASC 저장] [수정] 파일명에 실제 이름 적용
        np.savetxt(os.path.join(asc_save_dir, f"pred_{real_name}.asc"), pred_np, fmt="%.6f", delimiter=",")
# -----------------------------------------------------------------------------
# 5. 결과 집계 및 텍스트 저장 (Paper Table III Matching + Top 5 Analysis)
# -----------------------------------------------------------------------------
if len(per_landmark_me_list) > 0:
    # (Samples, Landmarks) 형태로 변환
    per_landmark_me_array = np.stack(per_landmark_me_list, axis=0)
    
    # [1] 각 랜드마크별 평균(Mean)과 표준편차(Std) 계산
    # shape: (Landmark_num, )
    lm_means = np.mean(per_landmark_me_array, axis=0) # 랜드마크별 ME
    lm_stds = np.std(per_landmark_me_array, axis=0)  # 랜드마크별 Std (샘플 간 편차)
    
    # [2] Final Evaluation Metric 계산 (논문 Table III 방식)
    # Average ME: 랜드마크별 평균들의 평균
    average_me = np.mean(lm_means)
    
    # Mean Std: ★ 랜드마크별 표준편차들의 평균 ★ (논문 방식 일치)
    std_me = np.mean(lm_stds)

else:
    average_me = 0.0
    std_me = 0.0
    lm_means = np.array([])
    lm_stds = np.array([])

# Success Rate (SR) 계산
sr_10 = np.sum(np.array(me_list) < 10.0) / len(me_list) * 100
sr_5  = np.sum(np.array(me_list) < 5.0) / len(me_list) * 100

# 파일명 생성
filename = f"ME{average_me:.4f}_std{std_me:.4f}.txt"
result_txt_path = os.path.join(run_root, filename)

with open(result_txt_path, "w") as f:
    f.write(f"==========================================\n")
    f.write(f"   Evaluation Result: {args.exp_name}\n")
    f.write(f"==========================================\n")
    f.write(f" Run ID      : {target_folder_name}\n")
    f.write(f" Model       : {args.model_epoch}\n")
    f.write(f" Data Type   : {args.Eval_DataType}\n")
    
    # [추가] User Comment (폴더명 태그) 기록 -> 정리용
    # None일 경우를 대비해 처리
    user_comment = args.user_tag if args.user_tag else "None"
    f.write(f" User Comment: {user_comment}\n")
    
    f.write(f" Train Data  : {train_len} samples\n")
    f.write(f" Batch Size  : {args.batch_size}\n")
    f.write(f" Num Points  : {args.num_points}\n")
    f.write(f"------------------------------------------\n")
    f.write(f" Average ME : {average_me:.4f} mm\n")
    f.write(f" Average Std: {std_me:.4f} mm (Mean of Landmark Stds)\n") 
    f.write(f" SR @ 10mm  : {sr_10:.2f} %\n")
    f.write(f" SR @ 5mm   : {sr_5:.2f} %\n")
    f.write(f"==========================================\n")
    
    if len(per_landmark_me_list) > 0:
        # --------------------------------------------------
        # Top 5 Hardest Landmarks (평균 Error가 가장 높은 순)
        # --------------------------------------------------
        f.write("\n>>> Top 5 Hardest Landmarks:\n")
        worst_indices = np.argsort(lm_means)[::-1][:5]
        for i in worst_indices:
            f.write(f"    LM {i:02d}: {lm_means[i]:.3f} ± {lm_stds[i]:.3f} mm\n")
        
        # --------------------------------------------------
        # Top 5 Easiest Landmarks (평균 Error가 가장 낮은 순)
        # --------------------------------------------------
        f.write("\n>>> Top 5 Easiest Landmarks:\n")
        best_indices = np.argsort(lm_means)[:5]
        for i in best_indices:
            f.write(f"    LM {i:02d}: {lm_means[i]:.3f} ± {lm_stds[i]:.3f} mm\n")
        
        # --------------------------------------------------
        # All Landmarks: per-landmark ME ± STD (논문 Table III 형식)
        # --------------------------------------------------
        f.write("\n>>> Per-landmark ME (Mean ± Std):\n")
        for i in range(lm_means.shape[0]):
            f.write(f"    LM {i:02d}: {lm_means[i]:.3f} ± {lm_stds[i]:.3f} mm\n")
            
        f.write(f"    ------------------------------------\n")
        # 논문의 'All' 행과 동일한 계산
        f.write(f"    All  : {average_me:.3f} ± {std_me:.3f} mm\n")

# 화면 출력
print(f"\n[Done] Results saved to: {run_root}")
print(f"      Filename: {filename}")
print(f"Average ME: {average_me:.4f} ± {std_me:.4f}")