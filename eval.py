'''
@Author: Yuan Wang (Modified by Researcher & AI Assistant)
@File: eval_all.py
@Description: Direct Regression 맞춤형 평가 스크립트 + 🌟 Excel Export
'''

from __future__ import print_function, division

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import time
import warnings
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from My_args import parser

# 🌟 새로운 아키텍처 임포트
from DeepPA_model import DeepPA_Wrapper  
from PAConv_model import PAConv          

# 🌟 Soft-argmax 로드 (이제 예측 좌표를 뽑을 땐 안 쓰지만, 보조 닻(Anchor)의 정밀도 평가용으로만 씁니다)
from loss import get_differentiable_coords

matplotlib.use('Agg')
warnings.filterwarnings("ignore", category=FutureWarning)

args = parser.parse_args()
args.eval = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_multiview_heatmap(points, heatmap, save_dir, sample_name, landmark_idx, prefix):
    fig = plt.figure(figsize=(30, 10))
    views = [(131, 90, -100, "Front"), (132, 30, 120, "Side"), (133, 45, -45, "Downside")]
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
# 1. 경로 및 데이터 로드 
# -----------------------------------------------------------------------------
if not args.run_id:
    print("Error: --run_id required (e.g., '1').")
    sys.exit(1)

project_dir = os.path.join(args.output_root, args.exp_name)
batch_str = f"{args.batch_size}x{args.accumulation_steps}" if args.accumulation_steps > 1 else f"{args.batch_size}"
base_str = f"FPS{args.num_points}_sigma{args.sigma}_batch{batch_str}_train{args.train_len}"
setting_str = f"{base_str}_{args.user_tag}" if args.user_tag else base_str

target_folder_name = f"{setting_str}_{args.run_id}"
run_root = os.path.join(project_dir, target_folder_name)

if not os.path.exists(run_root):
    print(f"Error: Experiment folder not found: {run_root}")
    sys.exit(1)

data_dir = os.path.join(run_root, 'npy_data')
heatmap_save_dir_base = os.path.join(run_root, "Pred_Heatmaps")
asc_save_dir_base = os.path.join(run_root, "Pred_Landmarks")

try:
    in_channels = getattr(args, 'in_channels', 3)
    shape_filename = f"shape_{args.Eval_DataType}.npy"
    shape_sample = np.load(os.path.join(data_dir, shape_filename), allow_pickle=True)
    landmark_all = np.load(os.path.join(data_dir, f"landmark_{args.Eval_DataType}.npy"), allow_pickle=True)
    heatmap_sample = np.load(os.path.join(data_dir, f"Heat_data_{args.Eval_DataType}.npy"), allow_pickle=True)
    name_path = os.path.join(data_dir, f"name_{args.Eval_DataType}.npy")
    name_sample = np.load(name_path, allow_pickle=True) if os.path.exists(name_path) else [f"S{i:03d}" for i in range(len(shape_sample))]
except FileNotFoundError:
    print(f"Error: Backup data files not found in {data_dir}.")
    sys.exit(1)

test_dataset = TensorDataset(
    torch.tensor(shape_sample, dtype=torch.float32),
    torch.tensor(landmark_all, dtype=torch.float32), # GT는 원본 mm 스케일
    torch.tensor(heatmap_sample, dtype=torch.float32)
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# -----------------------------------------------------------------------------
# 2. Joint 모델 래퍼 정의 (train.py와 동일하게 통일)
# -----------------------------------------------------------------------------
class JointE2EModel(nn.Module):
    def __init__(self, args, landmark_num):
        super().__init__()
        self.stage1_paconv = PAConv(args, landmark_num)
        self.stage2_deeppa = DeepPA_Wrapper(args, landmark_num)
        self.s1_aux_head = nn.Conv1d(64, landmark_num, 1)

    def forward(self, x):
        s1_latent = self.stage1_paconv(x)
        pred_coords, s2_aux_hm = self.stage2_deeppa(x, prior_heatmap=s1_latent)
        s1_aux_hm = self.s1_aux_head(s1_latent)
        return pred_coords, s2_aux_hm, s1_aux_hm

# -----------------------------------------------------------------------------
# 3. 평가 수행 코어 함수
# -----------------------------------------------------------------------------
def evaluate_target_model(eval_name, eval_model):
    print(f"\n==================================================")
    print(f" 🚀 [EVALUATION START] 대상 모델: {eval_name}")
    print(f"==================================================")
    
    current_hm_dir = os.path.join(heatmap_save_dir_base, eval_name)
    current_asc_dir = os.path.join(asc_save_dir_base, eval_name)
    os.makedirs(current_hm_dir, exist_ok=True)
    os.makedirs(current_asc_dir, exist_ok=True)

    me_list, per_landmark_me_list = [], []
    cos_sim_list, iou_list, time_list = [], [], []
    per_landmark_cos_sim_list, per_landmark_iou_list = [], []

    eval_model.eval()

    for idx, (point, gt_landmark, heatmap) in enumerate(tqdm(test_loader, desc=f"Eval {eval_name}")):
        point, gt_landmark, gt_heatmap = point.to(device), gt_landmark.to(device), heatmap.to(device)
        real_name = name_sample[idx]
        B, N, C = point.shape
        
        # [정규화 복원 (Denormalization) 준비]
        point_xyz = point[:, :, :3]
        centroid = torch.mean(point_xyz, axis=1, keepdim=True)
        point_centered = point_xyz - centroid
        m = torch.max(torch.sqrt(torch.sum(point_centered ** 2, axis=2)), axis=1)[0]
        scale = m.view(-1, 1, 1)
        point_norm_xyz = point_centered / scale 
        
        if C == 7: point_norm = torch.cat([point_norm_xyz, point[:, :, 3:]], dim=-1)
        elif C == 6: point_norm = torch.cat([point_norm_xyz, point[:, :, 3:]], dim=-1)
        else: point_norm = point_norm_xyz
        
        with torch.no_grad():
            if device.type == 'cuda': torch.cuda.synchronize()
            start_time = time.time()  # 🌟 모델 투입 직전 타이머 시작!

            point_input = point_norm.permute(0, 2, 1).contiguous()
            
            # 1. 모델에서 결과 3개를 받습니다. (S2 히트맵은 None일 수 있음)
            pred_coords_norm, s2_aux_hm, s1_aux_hm = eval_model(point_input)
            
            if device.type == 'cuda': torch.cuda.synchronize()
            time_list.append(time.time() - start_time)  # 🌟 연산 종료 직후 타이머 스톱!
            
            # 2. 🌟 [핵심 수정] 평가용 히트맵 타겟 결정 
            # Stage 2에 히트맵이 없으면(None), Stage 1의 히트맵으로 위치 정확도를 평가합니다.
            eval_target_hm = s1_aux_hm if s2_aux_hm is None else s2_aux_hm

            # 3. mm 단위 좌표 복구 (Direct Regression의 결과물)
            pred_landmark = (pred_coords_norm * scale) + centroid

            # 4. 히트맵 지표 계산 (이제 eval_target_hm을 사용합니다)
            pred_heatmap = eval_target_hm.permute(0, 2, 1) 
            pred_vec, gt_vec = pred_heatmap.permute(0, 2, 1), gt_heatmap.permute(0, 2, 1)     
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

            if idx % 40 == 0:
                points_np, heatmap_np = point_xyz[0].cpu().numpy(), pred_heatmap[0].cpu().numpy()
                for lm_idx in range(heatmap_np.shape[1]):
                     save_multiview_heatmap(points_np, heatmap_np[:, lm_idx], current_hm_dir, real_name, lm_idx, f"pred_{eval_name}")

            pred_np = pred_landmark.cpu().numpy().squeeze(0)
            gt_np = gt_landmark.cpu().numpy().squeeze(0)
                
            # [Metric: Mean Error (L2 Euclidean Distance)]
            dists = np.linalg.norm(pred_np - gt_np, axis=1)
            me = np.mean(dists)
            
            me_list.append(me)
            per_landmark_me_list.append(dists)
            
            np.savetxt(os.path.join(current_asc_dir, f"{eval_name}_pred_{real_name}.asc"), pred_np, fmt="%.6f", delimiter=",")

    # ------------------ 최종 집계 및 Excel 저장 ------------------
    if len(per_landmark_me_list) > 0:
        per_landmark_me_array = np.stack(per_landmark_me_list, axis=0) 
        lm_means, lm_stds, lm_me_95 = np.mean(per_landmark_me_array, axis=0), np.std(per_landmark_me_array, axis=0), np.percentile(per_landmark_me_array, 95, axis=0)
        average_me, std_me, me_95_global = np.mean(lm_means), np.mean(lm_stds), np.percentile(me_list, 95)
        
        per_landmark_cos_array = np.vstack(per_landmark_cos_sim_list) 
        lm_cos_means, lm_cos_stds, lm_cos_5 = np.mean(per_landmark_cos_array, axis=0), np.std(per_landmark_cos_array, axis=0), np.percentile(per_landmark_cos_array, 5, axis=0)
        
        per_landmark_iou_array = np.vstack(per_landmark_iou_list) 
        lm_iou_means, lm_iou_stds, lm_iou_5 = np.mean(per_landmark_iou_array, axis=0), np.std(per_landmark_iou_array, axis=0), np.percentile(per_landmark_iou_array, 5, axis=0)
    else:
        return

    sr_10 = np.sum(np.array(me_list) < 10.0) / len(me_list) * 100
    sr_5  = np.sum(np.array(me_list) < 5.0) / len(me_list) * 100
    avg_cos_sim, cos_sim_5_global = np.mean(cos_sim_list), np.percentile(cos_sim_list, 5)
    avg_iou, iou_5_global = np.mean(iou_list), np.percentile(iou_list, 5)
    avg_time = np.mean(time_list) * 1000.0

    filename_excel = f"{eval_name}_Results_ME{average_me:.4f}.xlsx"
    result_excel_path = os.path.join(run_root, filename_excel)
    
    summary_data = {
        "지표 (Metric)": [
            "[Metadata]", "Experiment", "Run ID", "Model", 
            "[Metrics]", "Average ME (mm)", "Average Std (mm)", "Avg Time (ms)", 
            "Cosine Sim (Anchor, %)", "95%ile Cosine (%)", "mIoU (Anchor @0.1, %)", 
            "95%ile mIoU (%)", "Success Rate (<10mm, %)", "Success Rate (<5mm, %)"
        ],
        eval_name: [
            "", args.exp_name, target_folder_name, eval_name,
            "", round(average_me, 4), round(std_me, 4), round(avg_time, 2), 
            round(avg_cos_sim, 2), round(cos_sim_5_global, 2), round(avg_iou, 2), 
            round(iou_5_global, 2), round(sr_10, 2), round(sr_5, 2)
        ]
    }
    df_summary = pd.DataFrame(summary_data)

    lm_me_combined = [f"{lm_means[i]:.3f} ± {lm_stds[i]:.3f}" for i in range(lm_means.shape[0])]
    df_landmarks = pd.DataFrame({
        "LM": [f"{i+1}" for i in range(lm_means.shape[0])],
        eval_name: lm_me_combined,
        f"{eval_name} (95%ile)": np.round(lm_me_95, 3)
    })

    worst_indices = np.argsort(lm_means)[::-1][:10]
    df_top10 = pd.DataFrame({
        "순위": [f"{r+1}" for r in range(10)],
        eval_name: [f"LM {i+1:02d} ({lm_means[i]:.3f} ± {lm_stds[i]:.3f})" for i in worst_indices],
        f"{eval_name} (95%ile)": [round(lm_me_95[i], 3) for i in worst_indices]
    })

    with pd.ExcelWriter(result_excel_path, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='1_Summary', index=False)
        df_landmarks.to_excel(writer, sheet_name='2_Per_Landmark', index=False)
        df_top10.to_excel(writer, sheet_name='3_Top10_Hardest', index=False)

    print(f"\n[{eval_name} Done] Excel saved to: {filename_excel}")
    print(f"Average ME: {average_me:.4f} ± {std_me:.4f} (95%ile: {me_95_global:.4f} mm)")

# -----------------------------------------------------------------------------
# 4. 모델 로드 및 평가 분기
# -----------------------------------------------------------------------------
if args.model.lower() == 'deeppa_auto':
    print(">>> [INFO] 🚀 Direct Regression Evaluation Mode")
    model = JointE2EModel(args, args.landmark_num).to(device)
    
    # 두 가중치를 로드하여 Joint 모델 완성
    stage1_path = os.path.join(run_root, 'models', 'Stage1_PAConv_last.t7')
    stage2_path = os.path.join(run_root, 'models', 'Stage2_DeepPA_last.t7')
    
    model.stage1_paconv.load_state_dict(torch.load(stage1_path, map_location=device))
    model.stage2_deeppa.load_state_dict(torch.load(stage2_path, map_location=device))
    
    evaluate_target_model("DeepPA_Direct", model)

print("\n>>> [ALL EVALUATION COMPLETED]")