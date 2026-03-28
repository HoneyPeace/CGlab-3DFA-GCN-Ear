'''
@Author: Yuan Wang (Modified by Researcher 2 & AI Assistant)
@File: eval_all.py
@Description: Evaluation script with Quantitative Heatmap Metrics & DeepPA Support
'''

from __future__ import print_function, division
import sys
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import warnings
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from My_args import parser

# ==========================================
# 🚀 모델 3대장 모두 임포트
# ==========================================
from DeepLA_model import DeepLA_Wrapper
from DeepPA_model import DeepPA_Wrapper  
from PAConv_model import PAConv          
# ==========================================

from loss import get_differentiable_coords

matplotlib.use('Agg')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.manifold._mds")

args = parser.parse_args()
args.eval = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    filename = f"{prefix}_{sample_name}_L{landmark_idx + 1:02d}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=100, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# 1. 경로 설정
# -----------------------------------------------------------------------------
if not args.run_id:
    print("Error: --run_id required (e.g., '1' for the first experiment).")
    sys.exit(1)

project_dir = os.path.join(args.output_root, args.exp_name)
if args.accumulation_steps > 1:
    batch_str = f"{args.batch_size}x{args.accumulation_steps}"
else:
    batch_str = f"{args.batch_size}"

base_str = f"FPS{args.num_points}_sigma{args.sigma}_batch{batch_str}_train{args.train_len}"

if args.user_tag and args.user_tag != "":
    setting_str = f"{base_str}_{args.user_tag}"
else:
    setting_str = base_str

target_folder_name = f"{setting_str}_{args.run_id}"
run_root = os.path.join(project_dir, target_folder_name)

if not os.path.exists(run_root):
    print(f"Error: Experiment folder not found: {run_root}")
    sys.exit(1)

model_path = os.path.join(run_root, 'models', args.model_epoch)
data_dir = os.path.join(run_root, 'npy_data')

if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    sys.exit(1)

heatmap_save_dir = os.path.join(run_root, "Pred_Heatmaps")
asc_save_dir = os.path.join(run_root, "Pred_Landmarks")

os.makedirs(heatmap_save_dir, exist_ok=True)
os.makedirs(asc_save_dir, exist_ok=True)

print(f"Target Run  : {target_folder_name}")
print(f"Loading Model: {args.model_epoch}")
print(f"Loading Data : {data_dir}")

# -----------------------------------------------------------------------------
# 2. 데이터 로드
# -----------------------------------------------------------------------------
try:
    shape_sample = np.load(os.path.join(data_dir, f"shape_{args.Eval_DataType}.npy"), allow_pickle=True)
    landmark_all = np.load(os.path.join(data_dir, f"landmark_{args.Eval_DataType}.npy"), allow_pickle=True)
    heatmap_sample = np.load(os.path.join(data_dir, f"Heat_data_{args.Eval_DataType}.npy"), allow_pickle=True)
    
    name_path = os.path.join(data_dir, f"name_{args.Eval_DataType}.npy")
    if os.path.exists(name_path):
        name_sample = np.load(name_path, allow_pickle=True)
        print(f">> Sample names loaded: {len(name_sample)} files.")
    else:
        print(">> [Warning] Name file not found. Using Index instead.")
        name_sample = [f"S{i:03d}" for i in range(len(shape_sample))]

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
# 3. 모델 로드 (args.model에 따른 자동 스위칭)
# -----------------------------------------------------------------------------

# =========================================================
# 🧊 DeepPA 평가용 PAConv Prior 로드
# =========================================================
if args.model == 'DeepPA':
    print(">>> [Prior Load] Loading Pre-trained PAConv (0.3mm SOTA) for DeepPA Eval...")
    paconv_prior = PAConv(args, args.landmark_num).to(device)
    
    # train.py와 동일한 PAConv 가중치 절대 경로
    best_paconv_path = os.path.join("..", "PAConv_model", "model_epoch_500.t7") 
    paconv_prior.load_state_dict(torch.load(best_paconv_path, map_location=device))
    paconv_prior.eval()
else:
    paconv_prior = None
# =========================================================

if args.model == 'PAConv':
    model = PAConv(args, args.landmark_num).to(device)
    print(">>> [INFO] 🧠 PAConv 백본을 로드합니다.")
elif args.model == 'DeepLA':
    model = DeepLA_Wrapper(args, args.landmark_num).to(device)
    print(">>> [INFO] 🚀 DeepLA-Net 백본을 로드합니다.")
elif args.model == 'DeepPA':
    model = DeepPA_Wrapper(args, args.landmark_num).to(device)
    print(">>> [INFO] 🔥 DeepPA (PAConv-Guided) 백본을 로드합니다.")
else:
    print(f"Error: 지원하지 않는 모델입니다 -> {args.model}")
    sys.exit(1)

# 평가할 모델 파일 로드
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# -----------------------------------------------------------------------------
# 4. 평가 루프 (시간 측정 및 정량 평가)
# -----------------------------------------------------------------------------
me_list = []
per_landmark_me_list = []

cos_sim_list = []  
iou_list = []
time_list = []

per_landmark_cos_sim_list = []
per_landmark_iou_list = []

print(f"\n>>> Starting Evaluation ({args.Eval_DataType})...")

for idx, (point, gt_landmark, heatmap) in enumerate(tqdm(test_loader, desc="Evaluating")):
    point = point.to(device)
    gt_landmark = gt_landmark.to(device)
    gt_heatmap = heatmap.to(device) 
    
    real_name = name_sample[idx]
    
    B, N, C = point.shape
    centroid = torch.mean(point, axis=1, keepdim=True)
    point_centered = point - centroid
    m = torch.max(torch.sqrt(torch.sum(point_centered ** 2, axis=2)), axis=1)[0]
    scale = m.view(-1, 1, 1)
    point_norm = point_centered / scale 
    
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()

        point_input = point_norm.permute(0, 2, 1)
        
        # =========================================================
        # 🚀 DeepPA 스위칭 로직 (Forward Pass)
        # =========================================================
        if args.model == 'DeepPA':
            prior_hint = paconv_prior(point_input)
            pred_heatmap_raw = model(point_input, prior_heatmap=prior_hint) # (B, K, N)
        else:
            pred_heatmap_raw = model(point_input) # (B, K, N)
        # =========================================================
            
        pred_heatmap = pred_heatmap_raw.permute(0, 2, 1)      # 시각화 및 IoU용 (B, N, K)

        pred_landmark_norm = get_differentiable_coords(point_norm, pred_heatmap_raw, k=args.k_softargmax)
        pred_landmark = (pred_landmark_norm * scale) + centroid

        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        time_list.append(end_time - start_time)

        # -----------------------------------------------------------
        # 히트맵 정량 평가 (Per-Landmark Calculation)
        # -----------------------------------------------------------
        pred_vec = pred_heatmap.permute(0, 2, 1) 
        gt_vec = gt_heatmap.permute(0, 2, 1)     
        
        cos_sim_k = F.cosine_similarity(pred_vec, gt_vec, dim=2) * 100.0
        
        cos_sim_list.append(cos_sim_k.mean().item()) 
        per_landmark_cos_sim_list.append(cos_sim_k.cpu().numpy()) 

        threshold = 0.1
        pred_mask = (pred_heatmap > threshold).float() 
        gt_mask = (gt_heatmap > threshold).float()     
        
        intersection_k = (pred_mask * gt_mask).sum(dim=1) 
        union_k = (pred_mask + gt_mask).clamp(0, 1).sum(dim=1)
        
        iou_k = ((intersection_k + 1e-6) / (union_k + 1e-6)) * 100.0
        
        iou_list.append(iou_k.mean().item()) 
        per_landmark_iou_list.append(iou_k.cpu().numpy()) 
        # -----------------------------------------------------------

        if idx % 40 == 0:
            points_np = point[0].cpu().numpy()
            heatmap_np = pred_heatmap[0].cpu().numpy()
            for lm_idx in range(heatmap_np.shape[1]):
                 save_multiview_heatmap(points_np, heatmap_np[:, lm_idx], 
                                        heatmap_save_dir, real_name, lm_idx, "pred")

        pred_np = pred_landmark.cpu().numpy()
        gt_np = gt_landmark[0].cpu().numpy()
        if pred_np.ndim == 3: pred_np = pred_np.squeeze(0)
        if gt_np.ndim == 3: gt_np = gt_np.squeeze(0)
            
        dists = np.linalg.norm(pred_np - gt_np, axis=1)
        me = np.mean(dists)
        
        me_list.append(me)
        per_landmark_me_list.append(dists)
        
        np.savetxt(os.path.join(asc_save_dir, f"pred_{real_name}.asc"), pred_np, fmt="%.6f", delimiter=",")

# -----------------------------------------------------------------------------
# 5. 결과 집계 및 텍스트 파일 저장
# -----------------------------------------------------------------------------
if len(per_landmark_me_list) > 0:
    per_landmark_me_array = np.stack(per_landmark_me_list, axis=0) 
    lm_means = np.mean(per_landmark_me_array, axis=0)
    lm_stds = np.std(per_landmark_me_array, axis=0)
    average_me = np.mean(lm_means)
    std_me = np.mean(lm_stds)
    
    per_landmark_cos_array = np.vstack(per_landmark_cos_sim_list) 
    lm_cos_means = np.mean(per_landmark_cos_array, axis=0)
    lm_cos_stds = np.std(per_landmark_cos_array, axis=0)
    
    per_landmark_iou_array = np.vstack(per_landmark_iou_list) 
    lm_iou_means = np.mean(per_landmark_iou_array, axis=0)
    lm_iou_stds = np.std(per_landmark_iou_array, axis=0)
    
else:
    average_me, std_me = 0.0, 0.0
    lm_means, lm_stds = np.array([]), np.array([])
    lm_cos_means, lm_cos_stds = np.array([]), np.array([])
    lm_iou_means, lm_iou_stds = np.array([]), np.array([])

sr_10 = np.sum(np.array(me_list) < 10.0) / len(me_list) * 100
sr_5  = np.sum(np.array(me_list) < 5.0) / len(me_list) * 100

avg_cos_sim = np.mean(cos_sim_list) if len(cos_sim_list) > 0 else 0.0
avg_iou = np.mean(iou_list) if len(iou_list) > 0 else 0.0
avg_time = np.mean(time_list) * 1000.0 if len(time_list) > 0 else 0.0

filename = f"ME{average_me:.4f}_std{std_me:.4f}.txt"
result_txt_path = os.path.join(run_root, filename)

if args.accumulation_steps > 1:
    batch_str_log = f"{args.batch_size}x{args.accumulation_steps}"
else:
    batch_str_log = f"{args.batch_size}"

with open(result_txt_path, "w") as f:
    f.write(f"==========================================\n")
    f.write(f"   Evaluation Result: {args.exp_name}\n")
    f.write(f"==========================================\n")
    f.write(f" Run ID      : {target_folder_name}\n")
    f.write(f" Model       : {args.model_epoch}\n")
    f.write(f" Data Type   : {args.Eval_DataType}\n")
    user_comment = args.user_tag if args.user_tag else "None"
    f.write(f" User Comment: {user_comment}\n")
    f.write(f" Train Data  : {train_len} samples\n")
    f.write(f" Batch Size  : {batch_str_log}\n")
    f.write(f" Num Points  : {args.num_points}\n")
    f.write(f"------------------------------------------\n")
    f.write(f" Average ME : {average_me:.4f} mm\n")
    f.write(f" Average Std: {std_me:.4f} mm\n")
    f.write(f" SR @ 10mm  : {sr_10:.2f} %\n")
    f.write(f" SR @ 5mm   : {sr_5:.2f} %\n")
    f.write(f" Avg Time   : {avg_time:.2f} ms/sample\n")
    f.write(f"------------------------------------------\n")
    f.write(f" [Heatmap Quantitative Evaluation (Global)]\n")
    f.write(f" Cosine Sim : {avg_cos_sim:.2f} % (Distribution Match)\n")
    f.write(f" mIoU       : {avg_iou:.2f} % (Region Overlap @ 0.1)\n")
    f.write(f"==========================================\n")
    
    if len(per_landmark_me_list) > 0:
        f.write("\n>>> Top 5 Hardest Landmarks (by ME):\n")
        worst_indices = np.argsort(lm_means)[::-1][:5]
        for i in worst_indices:
            f.write(f"    LM {i+1:02d}: {lm_means[i]:.3f} ± {lm_stds[i]:.3f} mm\n")
        
        f.write("\n>>> Per-landmark ME (Mean ± Std):\n")
        for i in range(lm_means.shape[0]):
            f.write(f"    LM {i+1:02d}: {lm_means[i]:.3f} ± {lm_stds[i]:.3f} mm\n")
        
        f.write("\n>>> Per-landmark Cosine Sim (Mean ± Std):\n")
        for i in range(lm_cos_means.shape[0]):
            f.write(f"    LM {i+1:02d}: {lm_cos_means[i]:.2f} ± {lm_cos_stds[i]:.2f} %\n")

        f.write("\n>>> Per-landmark mIoU (Mean ± Std) @ Th=0.1:\n")
        for i in range(lm_iou_means.shape[0]):
            f.write(f"    LM {i+1:02d}: {lm_iou_means[i]:.2f} ± {lm_iou_stds[i]:.2f} %\n")

        f.write(f"    ------------------------------------\n")
        f.write(f"    All (ME) : {average_me:.3f} ± {std_me:.3f} mm\n")

print(f"\n[Done] Results saved to: {run_root}")
print(f"      Filename: {filename}")
print(f"Average ME: {average_me:.4f} ± {std_me:.4f}")
print(f"Cosine Sim: {avg_cos_sim:.2f}% | mIoU: {avg_iou:.2f}%")
print(f"Avg Time  : {avg_time:.2f} ms")