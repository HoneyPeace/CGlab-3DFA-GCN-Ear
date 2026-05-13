# @Author: Yuan Wang (Modified by Researcher Park Pyeong-hwa & AI Assistant)
# @File: eval_final.py
# @Description: 
# [S2G 전용 평가 스크립트 - Frozen Aux Ablation 지원 추가]
# - train.py와 100% 동일한 normalize_data 함수 Import 적용 
# - Frozen 기반 Ablation 3종(Drop, Fixed, No_Aux) 평가 라우팅 완벽 지원
# ==============================================================================

from __future__ import print_function, division

import sys
import subprocess
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 🌟 아키텍처 및 학습과 동일한 정규화 로직 임포트
from DeepPA_model import DeepPA_Wrapper  
from PAConv_model import PAConv          
from loss import get_differentiable_coords 
from augmentations import normalize_data  # [핵심] 학습 코드와 정규화 동기화

matplotlib.use('Agg')
warnings.filterwarnings("ignore", category=FutureWarning)

args = parser.parse_args()
args.eval = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def shorten_heatmap_name(name, max_len=24):
    safe_name = ''.join(ch if ch.isalnum() or ch in ('_', '-') else '_' for ch in str(name))
    if len(safe_name) <= max_len:
        return safe_name
    return safe_name[:max_len]

def save_multiview_heatmap(points, heatmap, save_dir, sample_name, landmark_idx, prefix):
    os.makedirs(save_dir, exist_ok=True)
    fig = plt.figure(figsize=(30, 10))
    views = [(131, 90, -100, "Front"), (132, 30, 120, "Side"), (133, 45, -45, "Downside")]
    for pos, elev, azim, title in views:
        ax = fig.add_subplot(pos, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=heatmap, cmap='jet', s=15, alpha=0.8)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.axis('off')
    short_sample_name = shorten_heatmap_name(sample_name, max_len=32)
    filename = f"{prefix}_{short_sample_name}_L{landmark_idx + 1:02d}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=100, bbox_inches='tight')
    plt.close()

def save_eval_command_txt(run_root, eval_name):
    command = subprocess.list2cmdline([sys.executable] + sys.argv)
    safe_name = eval_name.lower()
    base_path = os.path.join(run_root, f'command_eval_{safe_name}.txt')
    save_path = base_path
    if os.path.exists(save_path):
        stamp = time.strftime('%Y%m%d_%H%M%S')
        root, ext = os.path.splitext(base_path)
        save_path = f'{root}_{stamp}{ext}'

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('[Working Directory]\n')
        f.write(os.getcwd() + '\n\n')
        f.write('[Command]\n')
        f.write(command + '\n')
    print(f"[INFO] Evaluation command saved to: {save_path}")

def normalize_sample_name(raw_name):
    if hasattr(raw_name, "item"):
        raw_name = raw_name.item()
    if isinstance(raw_name, bytes):
        return raw_name.decode("utf-8", errors="replace")
    return str(raw_name)

# -----------------------------------------------------------------------------
# 1. 경로 및 데이터 로드 
# -----------------------------------------------------------------------------
if not getattr(args, 'run_id', None):
    print("Error: --run_id required (e.g., '1').")
    sys.exit(1)

project_dir = os.path.abspath(os.path.join(args.output_root, args.exp_name))
batch_str = f"{args.batch_size}x{args.accumulation_steps}" if getattr(args, 'accumulation_steps', 1) > 1 else f"{args.batch_size}"
base_str = f"FPS{args.num_points}_sigma{args.sigma}_batch{batch_str}_train{args.train_len}"
setting_str = f"{base_str}_{args.user_tag}" if getattr(args, 'user_tag', None) else base_str

target_folder_name = f"{setting_str}_{args.run_id}"
run_root = os.path.join(project_dir, target_folder_name)

if not os.path.exists(run_root):
    print(f"Error: Experiment folder not found: {run_root}")
    sys.exit(1)

save_eval_command_txt(run_root, args.model)

data_dir = os.path.join(run_root, 'npy_data')
heatmap_save_dir_base = os.path.join(run_root, "HM")
asc_save_dir_base = os.path.join(run_root, "Pred_Landmarks")

try:
    in_channels = getattr(args, 'in_channels', 7)
    eval_datatype = getattr(args, 'Eval_DataType', 'test')
    
    if in_channels == 7:
        shape_filename = f"shape_{eval_datatype}.npy"
        print(f"   [INFO] 🎯 7-Channel Mode: Loading {shape_filename}")
    elif in_channels == 6:
        shape_filename = f"shape_6ch_{eval_datatype}.npy"
        print(f"   [INFO] 🎯 6-Channel Mode: Loading {shape_filename}")
    else:
        shape_filename = f"shape_{eval_datatype}.npy"
        print(f"   [INFO] 🧊 3-Channel Mode: Loading {shape_filename}")

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
# 2. 평가용 통합 Universal Pipeline (train.py와 동기화)
# -----------------------------------------------------------------------------
class UniversalPipeline_Eval(nn.Module):
    def __init__(self, args, landmark_num, mode='single_paconv'):
        super().__init__()
        self.mode = mode.lower()
        self.args = args
        self.landmark_num = landmark_num
        
        if self.mode in ['frozen', 'finetune', 'e2e']:
            self.stage1_paconv = PAConv(args, landmark_num)
            self.stage2_deeppa = DeepPA_Wrapper(args, landmark_num)
        elif self.mode in ['single_paconv', 'single_paconv_heat']:
            self.model = PAConv(args, landmark_num)
        elif self.mode in ['single_deeppa', 'single_deepla']:
            self.model = DeepPA_Wrapper(args, landmark_num)

    def _coords_from_main_heatmap(self, points_xyz, sem_list, fallback_coords=None):
        if sem_list:
            readout_mode = getattr(self.args, 'train_coord_readout', 'topk').lower()
            if readout_mode == 'heatmap_attn_residual' and fallback_coords is not None:
                return fallback_coords
            k_val = getattr(self.args, 'regression_point_num', 10)
            return get_differentiable_coords(points_xyz, sem_list[-1], k=k_val)
        return fallback_coords

    def forward(self, x):   
        points_xyz = x[:, :3, :].permute(0, 2, 1).contiguous()
        if self.mode in ['single_paconv', 'single_paconv_heat']:
            multi_scale_hints, hm_raw = self.model(x)
            k_val = getattr(self.args, 'regression_point_num', 10)
            pred_coords = get_differentiable_coords(points_xyz, hm_raw, k=k_val)
            return pred_coords, [], hm_raw
            
        elif self.mode in ['single_deeppa', 'single_deepla']:
            out = self.model(x)
            sem_list = out[2] if isinstance(out, tuple) and len(out) > 2 else []
            main_hm = sem_list[-1] if len(sem_list) > 0 else None
            fallback_coords = out[0] if isinstance(out, tuple) else out
            pred_coords = self._coords_from_main_heatmap(points_xyz, sem_list, fallback_coords=fallback_coords)
            return pred_coords, sem_list, main_hm
            
        elif self.mode in ['frozen', 'finetune', 'e2e']:
            multi_scale_hints, s1_hm_raw = self.stage1_paconv(x)
            out = self.stage2_deeppa(x, prior_hints=multi_scale_hints)
            sem_list = out[2] if isinstance(out, tuple) and len(out) > 2 else []
            fallback_coords = out[0] if isinstance(out, tuple) else out
            pred_coords = self._coords_from_main_heatmap(points_xyz, sem_list, fallback_coords=fallback_coords)
            return pred_coords, sem_list, s1_hm_raw

# -----------------------------------------------------------------------------
# 3. 평가 수행 코어 함수
# -----------------------------------------------------------------------------
def evaluate_target_model(eval_name, eval_model, pipeline_mode):
    print(f"\n==================================================")
    print(f" 🚀 [EVALUATION START] 대상 모델: {eval_name} (Mode: {pipeline_mode})")
    print(f"==================================================")
    
    current_hm_dir = os.path.join(heatmap_save_dir_base, shorten_heatmap_name(eval_name, max_len=16))
    current_asc_dir = os.path.join(asc_save_dir_base, eval_name)
    os.makedirs(current_hm_dir, exist_ok=True)
    os.makedirs(current_asc_dir, exist_ok=True)

    me_list, per_landmark_me_list = [], []
    sample_records = []
    cos_sim_list, iou_list, time_list = [], [], []
    
    eval_model.eval()

    for idx, (point, gt_landmark, heatmap) in enumerate(tqdm(test_loader, desc=f"Eval {eval_name}")):
        point, gt_landmark, gt_heatmap = point.to(device), gt_landmark.to(device), heatmap.to(device)
        real_name = normalize_sample_name(name_sample[idx])
        B, N, C = point.shape
        
        # 1. 역정규화를 위한 Centroid와 Scale 계산
        point_xyz = point[:, :, :3]
        centroid = torch.mean(point_xyz, axis=1, keepdim=True)
        point_centered = point_xyz - centroid
        m = torch.max(torch.sqrt(torch.sum(point_centered ** 2, axis=2)), axis=1)[0]
        scale = m.view(-1, 1, 1)
        
        with torch.no_grad():
            if device.type == 'cuda': torch.cuda.synchronize()
            start_time = time.time()  

            point_normal, _ = normalize_data(point, gt_landmark)
            point_input = point_normal.permute(0, 2, 1).contiguous()
            
            # 모델 추론
            pred_coords_norm, sem_list, paconv_or_main_hm = eval_model(point_input)
            
            if device.type == 'cuda': torch.cuda.synchronize()
            time_list.append(time.time() - start_time)  
            
            # [히트맵 품질 추출]
            if pipeline_mode in ['single_paconv', 'single_paconv_heat']:
                eval_target_hm_raw = paconv_or_main_hm
            else:
                eval_target_hm_raw = sem_list[-1] if len(sem_list) > 0 else paconv_or_main_hm
                
            if eval_target_hm_raw is not None:
                eval_target_hm = eval_target_hm_raw
                pred_heatmap = eval_target_hm.permute(0, 2, 1) 
                pred_vec, gt_vec = pred_heatmap.permute(0, 2, 1), gt_heatmap.permute(0, 2, 1)     
                
                cos_sim_k = F.cosine_similarity(pred_vec, gt_vec, dim=2) * 100.0
                cos_sim_list.append(cos_sim_k.mean().item()) 

                threshold = 0.1
                pred_mask = (pred_heatmap > threshold).float() 
                gt_mask = (gt_heatmap > threshold).float()  
                intersection_k = (pred_mask * gt_mask).sum(dim=1) 
                union_k = (pred_mask + gt_mask).clamp(0, 1).sum(dim=1)
                iou_k = ((intersection_k + 1e-6) / (union_k + 1e-6)) * 100.0
                
                iou_list.append(iou_k.mean().item()) 

                if idx % 40 == 0:
                    points_np, heatmap_np = point_xyz[0].cpu().numpy(), pred_heatmap[0].cpu().numpy()
                    for lm_idx in range(heatmap_np.shape[1]):
                         save_multiview_heatmap(points_np, heatmap_np[:, lm_idx], current_hm_dir, real_name, lm_idx, "p")

            # 4. 정밀 스케일 복원 (Denormalization)
            pred_landmark = (pred_coords_norm * scale) + centroid
            
            pred_np = pred_landmark.cpu().numpy().squeeze(0)
            gt_np = gt_landmark.cpu().numpy().squeeze(0)
                
            # 5. 밀리미터 단위 오차(ME) 추출
            dists = np.linalg.norm(pred_np - gt_np, axis=1)
            me = np.mean(dists)
            me_list.append(me)
            per_landmark_me_list.append(dists)
            
            pred_asc_name = f"{eval_name}_pred_{real_name}.asc"
            pred_asc_relpath = os.path.join("Pred_Landmarks", eval_name, pred_asc_name)
            np.savetxt(os.path.join(current_asc_dir, pred_asc_name), pred_np, fmt="%.6f", delimiter=",")

            sample_record = {
                "Index": idx,
                "Sample_Name": real_name,
                "Mean_Error_mm": round(float(me), 6),
                "Pred_ASC": pred_asc_relpath,
            }
            for lm_idx, dist in enumerate(dists):
                sample_record[f"LM{lm_idx + 1:02d}_Error_mm"] = round(float(dist), 6)
            sample_records.append(sample_record)

    # ------------------ 지표 산출 및 리포팅 ------------------
    avg_cos_sim = np.mean(cos_sim_list) if cos_sim_list else 0.0
    cos_sim_5_global = np.percentile(cos_sim_list, 5) if cos_sim_list else 0.0
    avg_iou = np.mean(iou_list) if iou_list else 0.0
    iou_5_global = np.percentile(iou_list, 5) if iou_list else 0.0
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
    
    summary_data = {
        "지표 (Metric)": [
            "[Metadata]", "Experiment", "Run ID", "Model", 
            "[Metrics]", "Average ME (mm)", "Average Std (mm)", "Avg Time (ms)", 
            "Cosine Sim (%), ", "95%ile Cosine (%)", "mIoU (@0.1, %)", 
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

    df_samples = pd.DataFrame(sample_records)

    with pd.ExcelWriter(result_excel_path, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='1_Summary', index=False)
        df_landmarks.to_excel(writer, sheet_name='2_Per_Landmark', index=False)
        df_top10.to_excel(writer, sheet_name='3_Top10_Hardest', index=False)
        df_samples.to_excel(writer, sheet_name='4_Per_Sample', index=False)

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
                
        f.write("\n[4. Per-Sample Mean Error]\n")
        for sample_record in sample_records:
            f.write(
                f"  #{sample_record['Index']:03d} "
                f"{sample_record['Sample_Name']} : "
                f"{sample_record['Mean_Error_mm']:.4f} mm "
                f"({sample_record['Pred_ASC']})\n"
            )

    print(f"\n[{eval_name} Done] Excel saved to: {filename_excel}")
    print(f"  └─ 📄 Text summary perfectly synchronized & saved to: {os.path.basename(txt_path)}")
    print(f"Average ME: {average_me:.4f} ± {std_me:.4f} (95%ile: {me_95_global:.4f} mm)")

# -----------------------------------------------------------------------------
# 4. 모델 로드 및 평가 분기
# -----------------------------------------------------------------------------
model_name_lower = args.model.lower()

if model_name_lower in ['paconv', 'paconv_struct']: pipeline_mode = 'single_paconv'
elif model_name_lower == 'paconv_heat': pipeline_mode = 'single_paconv_heat'
elif model_name_lower == 'deeppa': pipeline_mode = 'single_deeppa' 
elif model_name_lower == 'deeppa_finetune': pipeline_mode = 'finetune'
elif model_name_lower == 'deeppa_e2e': pipeline_mode = 'e2e'

# 🌟 [수정됨] deeppa_frozen_no_heat 등 구버전 이름이 들어와도 안전하게 frozen 모드로 매핑
elif model_name_lower in ['deeppa_frozen', 'deeppa_frozen_no_heat', 'frozen_aux_drop', 'frozen_aux_fixed', 'frozen_no_aux']: 
    pipeline_mode = 'frozen'

# 기존 DeepLA 단독 모델 (추후 단독 실험 시 문제없도록 보존)
elif model_name_lower in ['deepla_ori', 'deepla_all', 'deepla_all_tied', 'deepla_progress']:
    pipeline_mode = 'single_deepla' 
else: raise ValueError(f"Unknown model routing: {model_name_lower}")

print(f">>> [INFO] 🚀 Evaluation Mode: {pipeline_mode.upper()} (Model: {args.model})")

model = UniversalPipeline_Eval(args, args.landmark_num, mode=pipeline_mode).to(device)

target_weight_path = os.path.join(run_root, 'models', getattr(args, 'model_epoch', f'{args.model}_last.t7'))

if os.path.exists(target_weight_path):
    print(f"📦 모델 가중치 로드 성공: {os.path.basename(target_weight_path)}")
    model.load_state_dict(torch.load(target_weight_path, map_location=device), strict=False)
else:
    print(f"🚨 [ERROR] 가중치 파일을 찾을 수 없습니다: {target_weight_path}")
    sys.exit(1)

evaluate_target_model(args.model, model, pipeline_mode)

print("\n>>> [ALL EVALUATION COMPLETED]")
