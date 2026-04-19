'''
@Author: Yuan Wang (Modified by Researcher & AI Assistant)
@File: eval.py
@Description: Unified Evaluation Script (Soft-Argmax 적용 + Stage1 좌표 채점 해금 + TXT/EXCEL 동기화)
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

# 🌟 새로운 아키텍처 및 보간법 임포트
from DeepPA_model import DeepPA_Wrapper  
from PAConv_model import PAConv          
from loss import get_differentiable_coords # 🌟 초정밀 좌표 추출 함수 추가

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
if not getattr(args, 'run_id', None):
    print("Error: --run_id required (e.g., '1').")
    sys.exit(1)

project_dir = os.path.join(args.output_root, args.exp_name)
batch_str = f"{args.batch_size}x{args.accumulation_steps}" if getattr(args, 'accumulation_steps', 1) > 1 else f"{args.batch_size}"
base_str = f"FPS{args.num_points}_sigma{args.sigma}_batch{batch_str}_train{args.train_len}"
setting_str = f"{base_str}_{args.user_tag}" if getattr(args, 'user_tag', None) else base_str

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
    eval_datatype = getattr(args, 'Eval_DataType', 'test')
    shape_filename = f"shape_{eval_datatype}.npy"
    shape_sample = np.load(os.path.join(data_dir, shape_filename), allow_pickle=True)
    landmark_all = np.load(os.path.join(data_dir, f"landmark_{eval_datatype}.npy"), allow_pickle=True)
    heatmap_sample = np.load(os.path.join(data_dir, f"Heat_data_{eval_datatype}.npy"), allow_pickle=True)
    name_path = os.path.join(data_dir, f"name_{eval_datatype}.npy")
    name_sample = np.load(name_path, allow_pickle=True) if os.path.exists(name_path) else [f"S{i:03d}" for i in range(len(shape_sample))]
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
# 2. 🌟 평가용 통합 Hybrid Pipeline (Soft-Argmax 동기화)
# -----------------------------------------------------------------------------
class HybridPipeline_Eval(nn.Module):
    def __init__(self, args, landmark_num, mode='frozen'):
        super().__init__()
        self.mode = mode.lower()
        self.stage1_paconv = PAConv(args, landmark_num)
        
        if self.mode in ['frozen', 'e2e']:
            self.stage2_deeppa = DeepPA_Wrapper(args, landmark_num)

    def forward(self, x):
        if self.mode == 'stage1':
            s1_latent, s1_hm_anchor = self.stage1_paconv(x)
            
            # 🌟 [수정됨] 단일 점이 아닌 상위 K개 점의 무게중심(Soft-Argmax) 추출
            s1_hm_prob = F.softmax(s1_hm_anchor, dim=1)
            points_xyz = x[:, :3, :].permute(0, 2, 1).contiguous()
            pred_coords = get_differentiable_coords(points_xyz, s1_hm_prob, k=getattr(self.stage1_paconv.args, 'k_softargmax', 10))
            
            return pred_coords, s1_hm_prob # mIoU 계산을 위해 확률값 전달
            
        elif self.mode in ['frozen', 'e2e']:
            s1_latent, s1_hm = self.stage1_paconv(x)
            out = self.stage2_deeppa(x, prior_latent=s1_latent, prior_heatmap=s1_hm)
            
            if isinstance(out, tuple): pred_coords = out[0]
            else: pred_coords = out 
                
            return pred_coords, s1_hm

# -----------------------------------------------------------------------------
# 3. 평가 수행 코어 함수
# -----------------------------------------------------------------------------
def evaluate_target_model(eval_name, eval_model, pipeline_mode):
    print(f"\n==================================================")
    print(f" 🚀 [EVALUATION START] 대상 모델: {eval_name} (Mode: {pipeline_mode})")
    print(f"==================================================")
    
    current_hm_dir = os.path.join(heatmap_save_dir_base, eval_name)
    current_asc_dir = os.path.join(asc_save_dir_base, eval_name)
    os.makedirs(current_hm_dir, exist_ok=True)
    os.makedirs(current_asc_dir, exist_ok=True)

    me_list, per_landmark_me_list = [], []
    cos_sim_list, iou_list, time_list = [], []
    per_landmark_cos_sim_list, per_landmark_iou_list = [], []

    eval_model.eval()

    for idx, (point, gt_landmark, heatmap) in enumerate(tqdm(test_loader, desc=f"Eval {eval_name}")):
        point, gt_landmark, gt_heatmap = point.to(device), gt_landmark.to(device), heatmap.to(device)
        real_name = name_sample[idx]
        B, N, C = point.shape
        
        # [정규화 복원 세팅]
        point_xyz = point[:, :, :3]
        centroid = torch.mean(point_xyz, axis=1, keepdim=True)
        point_centered = point_xyz - centroid
        m = torch.max(torch.sqrt(torch.sum(point_centered ** 2, axis=2)), axis=1)[0]
        scale = m.view(-1, 1, 1)
        point_norm_xyz = point_centered / scale 
        
        if C >= 6: point_norm = torch.cat([point_norm_xyz, point[:, :, 3:]], dim=-1)
        else: point_norm = point_norm_xyz
        
        with torch.no_grad():
            if device.type == 'cuda': torch.cuda.synchronize()
            start_time = time.time()  

            point_input = point_norm.permute(0, 2, 1).contiguous()
            
            # 통합 모델 포워딩 (여기서 Soft-Argmax 좌표가 나옵니다)
            pred_coords_norm, s1_aux_hm = eval_model(point_input)
            
            if device.type == 'cuda': torch.cuda.synchronize()
            time_list.append(time.time() - start_time)  
            
            # Stage 1은 이미 Softmax가 적용되어 리턴됨
            if pipeline_mode == 'stage1':
                eval_target_hm = s1_aux_hm
            else:
                eval_target_hm = F.softmax(s1_aux_hm, dim=1) if s1_aux_hm is not None else s1_aux_hm
                
            pred_heatmap = eval_target_hm.permute(0, 2, 1) 
            pred_vec, gt_vec = pred_heatmap.permute(0, 2, 1), gt_heatmap.permute(0, 2, 1)     
            
            # 유사도 평가
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

            # 🌟 [봉인 해제!] 조건문 삭제: Stage 1이든 2든 무조건 3D 좌표 오차(mm)를 채점합니다.
            pred_landmark = (pred_coords_norm * scale) + centroid
            pred_np = pred_landmark.cpu().numpy().squeeze(0)
            gt_np = gt_landmark.cpu().numpy().squeeze(0)
                
            dists = np.linalg.norm(pred_np - gt_np, axis=1)
            me = np.mean(dists)
            me_list.append(me)
            per_landmark_me_list.append(dists)
            np.savetxt(os.path.join(current_asc_dir, f"{eval_name}_pred_{real_name}.asc"), pred_np, fmt="%.6f", delimiter=",")

    # ------------------ 최종 집계 및 저장 ------------------
    avg_cos_sim, cos_sim_5_global = np.mean(cos_sim_list), np.percentile(cos_sim_list, 5)
    avg_iou, iou_5_global = np.mean(iou_list), np.percentile(iou_list, 5)
    avg_time = np.mean(time_list) * 1000.0

    if len(per_landmark_me_list) > 0:
        per_landmark_me_array = np.stack(per_landmark_me_list, axis=0)
        lm_means, lm_stds, lm_me_95 = np.mean(per_landmark_me_array, axis=0), np.std(per_landmark_me_array, axis=0), np.percentile(per_landmark_me_array, 95, axis=0)
        average_me, std_me, me_95_global = np.mean(lm_means), np.mean(lm_stds), np.percentile(me_list, 95)
        sr_10 = np.sum(np.array(me_list) < 10.0) / len(me_list) * 100
        sr_5  = np.sum(np.array(me_list) < 5.0) / len(me_list) * 100
        worst_indices = np.argsort(lm_means)[::-1][:10]
    else:
        average_me, std_me, me_95_global, sr_10, sr_5 = 0.0, 0.0, 0.0, 0.0, 0.0
        lm_means, lm_stds, lm_me_95 = np.zeros(args.landmark_num), np.zeros(args.landmark_num), np.zeros(args.landmark_num)
        worst_indices = np.arange(10)

    filename_excel = f"{eval_name}_Results_ME{average_me:.4f}.xlsx"
    result_excel_path = os.path.join(run_root, filename_excel)
    
    # [1] Excel 저장
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
        "LM": [f"{i+1:02d}" for i in range(lm_means.shape[0])],
        f"{eval_name} (Mean ± Std)": lm_me_combined,
        f"{eval_name} (95%ile)": np.round(lm_me_95, 3)
    })

    df_top10 = pd.DataFrame({
        "순위": [f"{r+1}위" for r in range(10)],
        eval_name: [f"LM {i+1:02d} ({lm_means[i]:.3f} ± {lm_stds[i]:.3f})" for i in worst_indices],
        f"{eval_name} (95%ile)": [round(lm_me_95[i], 3) for i in worst_indices]
    })

    with pd.ExcelWriter(result_excel_path, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='1_Summary', index=False)
        df_landmarks.to_excel(writer, sheet_name='2_Per_Landmark', index=False)
        df_top10.to_excel(writer, sheet_name='3_Top10_Hardest', index=False)

    # [2] 텍스트 파일(TXT)을 엑셀과 100% 동일하게 저장
    txt_path = result_excel_path.replace(".xlsx", ".txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("==================================================\n")
        f.write(f" 🚀 Evaluation Summary: {eval_name} (Run ID: {target_folder_name})\n")
        f.write("==================================================\n\n")
        
        f.write("[1. Summary Metrics]\n")
        f.write(f"- Average Inference Time : {avg_time:.2f} ms\n")
        f.write(f"- Heatmap Cosine Sim     : {avg_cos_sim:.2f} % (95%ile: {cos_sim_5_global:.2f} %)\n")
        f.write(f"- Heatmap mIoU (@0.1)    : {avg_iou:.2f} % (95%ile: {iou_5_global:.2f} %)\n")
        

        f.write(f"- Average ME             : {average_me:.4f} ± {std_me:.4f} mm\n")
        f.write(f"- 95%ile ME              : {me_95_global:.4f} mm\n")
        f.write(f"- Success Rate (<10mm)   : {sr_10:.2f} %\n")
        f.write(f"- Success Rate (<5mm)    : {sr_5:.2f} %\n")
        
        f.write("\n[2. Per-Landmark Errors (All 36)]\n")
        for i in range(lm_means.shape[0]):
            f.write(f"  LM {i+1:02d} : {lm_means[i]:.3f} ± {lm_stds[i]:.3f} mm  (95%ile: {lm_me_95[i]:.3f} mm)\n")
            
        f.write("\n[3. Top 10 Hardest Landmarks (Worst Error)]\n")
        for r, i in enumerate(worst_indices):
            f.write(f"  {r+1}위: LM {i+1:02d} (오차: {lm_means[i]:.3f} ± {lm_stds[i]:.3f} mm)\n")

                
    print(f"\n[{eval_name} Done] Excel saved to: {filename_excel}")
    print(f"  └─ 📄 Text summary perfectly synchronized & saved to: {os.path.basename(txt_path)}")
    print(f"Average ME: {average_me:.4f} ± {std_me:.4f} (95%ile: {me_95_global:.4f} mm)")

# -----------------------------------------------------------------------------
# 4. 모델 로드 및 평가 분기
# -----------------------------------------------------------------------------
model_name_lower = args.model.lower()
if model_name_lower == 'paconv': pipeline_mode = 'stage1'
elif model_name_lower in ['deeppa_frozen', 'deeppa_auto']: pipeline_mode = 'frozen'
elif model_name_lower == 'deeppa_e2e': pipeline_mode = 'e2e'
else: pipeline_mode = 'single_custom' 

print(f">>> [INFO] 🚀 Evaluation Mode: {pipeline_mode.upper()}")
model = HybridPipeline_Eval(args, args.landmark_num, mode=pipeline_mode).to(device)

target_weight_path = os.path.join(run_root, 'models', getattr(args, 'model_epoch', f'{args.model}_last.t7'))

if os.path.exists(target_weight_path):
    print(f"📦 모델 가중치 로드 성공: {os.path.basename(target_weight_path)}")
    model.load_state_dict(torch.load(target_weight_path, map_location=device), strict=False)
else:
    print(f"🚨 [ERROR] 가중치 파일을 찾을 수 없습니다: {target_weight_path}")
    sys.exit(1)

evaluate_target_model(args.model, model, pipeline_mode)

print("\n>>> [ALL EVALUATION COMPLETED]")