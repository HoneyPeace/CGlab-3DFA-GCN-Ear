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
from stage1_paconv_bridge import (
    build_original_github_stage1,
    get_stage1_input,
    is_original_github_stage1,
)
from loss import (
    get_differentiable_coords,
    CurvatureSurfaceLoss,
    SoftLocalCurvatureSurfaceLoss,
)
from util import landmark_regression, landmark_regression_original
from xlsx_utils import write_dataframes_to_xlsx
from augmentations import normalize_data  # [핵심] 학습 코드와 정규화 동기화

matplotlib.use('Agg')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.manifold._mds")

args = parser.parse_args()
args.eval = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def shorten_heatmap_name(name, max_len=24):
    safe_name = ''.join(ch if ch.isalnum() or ch in ('_', '-') else '_' for ch in str(name))
    if len(safe_name) <= max_len:
        return safe_name
    return safe_name[:max_len]

def compute_surface_normal_distances_mm(pred_coords_norm, gt_coords_norm, points_norm, scale, k_p2p):
    dist_matrix = torch.cdist(gt_coords_norm, points_norm)
    _, knn_indices = torch.topk(dist_matrix, k_p2p, dim=2, largest=False)
    idx_expanded = knn_indices.unsqueeze(-1).expand(-1, -1, -1, 3)
    points_expanded = points_norm.unsqueeze(1).expand(-1, gt_coords_norm.size(1), -1, -1)
    knn_p2p_gt = torch.gather(points_expanded, 2, idx_expanded)

    center_p2p_gt = knn_p2p_gt.mean(dim=2, keepdim=True)
    centered = knn_p2p_gt - center_p2p_gt
    cov_p2p_gt = torch.matmul(centered.transpose(2, 3), centered)
    _, eigvec_p2p_gt = torch.linalg.eigh(cov_p2p_gt)
    normal_gt = eigvec_p2p_gt[..., 0]

    distance_norm = torch.abs(
        torch.sum(
            (pred_coords_norm - center_p2p_gt.squeeze(2).detach()) * normal_gt.detach(),
            dim=-1,
        )
    )
    return distance_norm * scale.view(-1, 1)

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
    plt.savefig(windows_write_path(os.path.join(save_dir, filename)), dpi=100, bbox_inches='tight')
    plt.close()

def windows_write_path(path):
    if os.name == 'nt':
        abs_path = os.path.abspath(path)
        if len(abs_path) >= 248 and not abs_path.startswith('\\\\?\\'):
            return '\\\\?\\' + abs_path
    return path

def get_eval_output_name(model_name):
    result_tag = getattr(args, 'eval_result_tag', '').strip()
    if not result_tag:
        return model_name
    safe_tag = ''.join(ch if ch.isalnum() or ch in ('_', '-') else '_' for ch in result_tag)
    return f"{model_name}_{safe_tag}"

def save_eval_command_txt(run_root, eval_name):
    command = subprocess.list2cmdline([sys.executable] + sys.argv)
    safe_name = eval_name.lower()
    base_path = os.path.join(run_root, f'command_eval_{safe_name}.txt')
    save_path = base_path
    if os.path.exists(save_path):
        stamp = time.strftime('%Y%m%d_%H%M%S')
        root, ext = os.path.splitext(base_path)
        save_path = f'{root}_{stamp}{ext}'
        if os.name == 'nt' and len(save_path) >= 250:
            save_path = os.path.join(run_root, f'cmd_eval_{stamp}.txt')

    with open(windows_write_path(save_path), 'w', encoding='utf-8') as f:
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

def sample_id_digits(sample_name):
    digits = ''.join(ch for ch in str(sample_name) if ch.isdigit())
    return str(int(digits)) if digits else ""

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

eval_output_name = get_eval_output_name(args.model)
save_eval_command_txt(run_root, eval_output_name)

data_dir = os.path.join(run_root, 'npy_data')
heatmap_save_dir_base = os.path.join(run_root, "HM")
asc_save_dir_base = os.path.join(run_root, "LM")

try:
    in_channels = getattr(args, 'in_channels', 7)
    eval_datatype = getattr(args, 'Eval_DataType', 'test')
    
    if in_channels == 7:
        shape_filename = f"shape_{eval_datatype}.npy"
        print(f"   [INFO] 🎯 7-Channel Mode: Loading {shape_filename}")
    elif in_channels == 6:
        shape_filename = f"shape_6ch_{eval_datatype}.npy"
        print(f"   [INFO] 🎯 6-Channel Mode: Loading {shape_filename}")
    elif in_channels == 3:
        shape_filename = f"shape_3ch_{eval_datatype}.npy"
        print(f"   [INFO] 3-Channel Mode: Loading {shape_filename}")
    elif in_channels == 10:
        shape_filename = f"shape_3ch_{eval_datatype}.npy"
        print(f"   [INFO] XYZ->10-Channel Mode: Loading {shape_filename} and zero-padding to 10ch")
    else:
        shape_filename = f"shape_{eval_datatype}.npy"
        print(f"   [INFO] 🧊 3-Channel Mode: Loading {shape_filename}")

    shape_sample = np.load(os.path.join(data_dir, shape_filename), allow_pickle=True)
    if in_channels == 10:
        if shape_sample.ndim != 3 or shape_sample.shape[-1] != 3:
            raise ValueError(f"[ERROR] XYZ->10 eval expects shape_3ch data with last dim 3, got {shape_sample.shape}")
        pad_channels = in_channels - shape_sample.shape[-1]
        zero_pad = np.zeros((*shape_sample.shape[:-1], pad_channels), dtype=shape_sample.dtype)
        shape_sample = np.concatenate([shape_sample, zero_pad], axis=-1)
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
            if is_original_github_stage1(args):
                self.stage1_paconv = build_original_github_stage1(args, landmark_num)
                print("[INFO] Stage1 PAConv source: original_github (xyz-only latent)")
            else:
                self.stage1_paconv = PAConv(args, landmark_num)
            self.stage2_deeppa = DeepPA_Wrapper(args, landmark_num)
        elif self.mode in ['single_paconv', 'single_paconv_heat']:
            if is_original_github_stage1(args):
                self.model = build_original_github_stage1(args, landmark_num)
                print("[INFO] Single PAConv source: original_github (xyz-only latent)")
            else:
                self.model = PAConv(args, landmark_num)
        elif self.mode in ['single_deeppa', 'single_deepla']:
            self.model = DeepPA_Wrapper(args, landmark_num)

    def _coords_from_main_heatmap(self, points_xyz, sem_list, fallback_coords=None):
        if sem_list:
            readout_mode = getattr(self.args, 'train_coord_readout', 'topk').lower()
            if readout_mode in ['sigmoid_xyz_pool', 'heatmap_attn_residual',
                                'heatmap_attn_residual_feature_only'] and fallback_coords is not None:
                return fallback_coords
            k_val = getattr(self.args, 'regression_point_num', 10)
            return get_differentiable_coords(points_xyz, sem_list[-1], k=k_val)
        return fallback_coords

    def set_residual_limit_norm(self, value):
        for module_name in ['model', 'stage2_deeppa']:
            module = getattr(self, module_name, None)
            if module is not None and hasattr(module, 'set_residual_limit_norm'):
                module.set_residual_limit_norm(value)

    def forward(self, x):   
        points_xyz = x[:, :3, :].permute(0, 2, 1).contiguous()
        if self.mode in ['single_paconv', 'single_paconv_heat']:
            stage1_input = get_stage1_input(self.args, x)
            multi_scale_hints, hm_raw = self.model(stage1_input)
            k_val = getattr(self.args, 'regression_point_num', 10)
            coord_method = getattr(self.args, 'eval_heatmap_coord_method', 'topk').lower()
            if coord_method in ['mds', 'mds_original']:
                regression_fn = landmark_regression_original if coord_method == 'mds_original' else landmark_regression
                pred_coords = torch.cat([
                    regression_fn(
                        points_xyz[sample_idx],
                        hm_raw[sample_idx].permute(1, 0).contiguous(),
                        k_val,
                    ).to(points_xyz.device, dtype=points_xyz.dtype)
                    for sample_idx in range(points_xyz.size(0))
                ], dim=0)
            else:
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
            stage1_input = get_stage1_input(self.args, x)
            multi_scale_hints, s1_hm_raw = self.stage1_paconv(stage1_input)
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
    
    short_eval_name = shorten_heatmap_name(eval_name, max_len=16)
    current_hm_dir = os.path.join(heatmap_save_dir_base, short_eval_name)
    current_asc_dir = os.path.join(asc_save_dir_base, short_eval_name)
    os.makedirs(current_hm_dir, exist_ok=True)
    os.makedirs(current_asc_dir, exist_ok=True)

    landmark_mapping_path = os.path.join(
        run_root,
        f"{short_eval_name}_landmark_mapping_LM01_LM{args.landmark_num:02d}.csv",
    )
    pd.DataFrame({
        "LM": [f"LM{i + 1:02d}" for i in range(args.landmark_num)],
        "Index_1based": list(range(1, args.landmark_num + 1)),
        "Index_0based": list(range(args.landmark_num)),
        "Order_Note": ["Predicted ASC row order / GT landmark order"] * args.landmark_num,
    }).to_csv(windows_write_path(landmark_mapping_path), index=False, encoding="utf-8-sig")

    me_list, per_landmark_me_list = [], []
    sample_records = []
    prediction_rows = []
    sample_253_exports = []
    cos_sim_list, iou_list, time_list = [], [], []

    surface_loss_mode = getattr(args, 'surface_loss_mode', 'topk').lower()
    if surface_loss_mode == 'soft_local':
        surface_criterion = SoftLocalCurvatureSurfaceLoss(
            k_p2p=args.plane_knn,
            k_curv=args.curv_knn,
            alpha=args.curv_alpha,
            beta=args.dir_beta,
            sigma=getattr(args, 'soft_curv_sigma', 0.0),
            min_sigma=getattr(args, 'soft_curv_min_sigma', 1e-4),
        ).to(device)
    else:
        surface_criterion = CurvatureSurfaceLoss(
            k_p2p=args.plane_knn,
            k_curv=args.curv_knn,
            alpha=args.curv_alpha,
            beta=args.dir_beta,
        ).to(device)
    surface_loss_list, surface_p2p_mm_list = [], []
    surface_distance_sample_me_list, surface_distance_lm_list = [], []
    surface_curv_diff_list, surface_dir_diff_list = [], []
    
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

            point_normal, gt_landmark_norm = normalize_data(point, gt_landmark)
            point_input = point_normal.permute(0, 2, 1).contiguous()
            if getattr(args, 'train_coord_readout', 'topk').lower() in [
                'heatmap_attn_residual', 'heatmap_attn_residual_feature_only'
            ]:
                residual_max_mm = float(getattr(args, 'hm_attn_residual_max_mm', 0.0))
                if residual_max_mm > 0.0 and hasattr(eval_model, 'set_residual_limit_norm'):
                    avg_m = torch.mean(m).item()
                    eval_model.set_residual_limit_norm(residual_max_mm / max(float(avg_m), 1e-6))
            
            # 모델 추론
            pred_coords_norm, sem_list, paconv_or_main_hm = eval_model(point_input)
            
            if device.type == 'cuda': torch.cuda.synchronize()
            time_list.append(time.time() - start_time)  

            points_for_surface = point_input[:, :3, :].permute(0, 2, 1).contiguous()
            L_srf, p2p_norm, curv_diff, dir_diff = surface_criterion(
                pred_coords_norm,
                gt_landmark_norm.view_as(pred_coords_norm),
                points_for_surface,
                disable_norm=False,
            )
            surface_p2p_mm = float(p2p_norm.item()) * float(torch.mean(m).item())
            surface_loss_list.append(float(L_srf.item()))
            surface_p2p_mm_list.append(surface_p2p_mm)
            surface_curv_diff_list.append(float(curv_diff.item()))
            surface_dir_diff_list.append(float(dir_diff.item()))
            surface_distance_mm = compute_surface_normal_distances_mm(
                pred_coords_norm,
                gt_landmark_norm.view_as(pred_coords_norm),
                points_for_surface,
                scale.view(-1),
                args.plane_knn,
            )
            surface_distance_np = surface_distance_mm.detach().cpu().numpy().squeeze(0)
            surface_distance_sample_me = float(np.mean(surface_distance_np))
            surface_distance_sample_std = float(np.std(surface_distance_np))
            surface_distance_sample_95 = float(np.percentile(surface_distance_np, 95))
            surface_distance_sample_me_list.append(surface_distance_sample_me)
            surface_distance_lm_list.append(surface_distance_np)
            
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
            
            pred_asc_name = f"p_{shorten_heatmap_name(real_name, max_len=32)}.asc"
            pred_asc_relpath = os.path.join("LM", short_eval_name, pred_asc_name)
            np.savetxt(windows_write_path(os.path.join(current_asc_dir, pred_asc_name)), pred_np, fmt="%.6f", delimiter=",")
            sample_error_95 = float(np.percentile(dists, 95))

            for lm_idx, dist in enumerate(dists):
                prediction_rows.append({
                    "Index": idx,
                    "Sample_Name": real_name,
                    "LM": f"LM{lm_idx + 1:02d}",
                    "LM_Index_1based": lm_idx + 1,
                    "Pred_X": round(float(pred_np[lm_idx, 0]), 6),
                    "Pred_Y": round(float(pred_np[lm_idx, 1]), 6),
                    "Pred_Z": round(float(pred_np[lm_idx, 2]), 6),
                    "GT_X": round(float(gt_np[lm_idx, 0]), 6),
                    "GT_Y": round(float(gt_np[lm_idx, 1]), 6),
                    "GT_Z": round(float(gt_np[lm_idx, 2]), 6),
                    "Error_mm": round(float(dist), 6),
                    "Surface_Distance_mm": round(float(surface_distance_np[lm_idx]), 6),
                    "Pred_ASC": pred_asc_relpath,
                })

            if sample_id_digits(real_name) == "253":
                sample_253_error_name = f"p_{shorten_heatmap_name(real_name, max_len=32)}_errors.csv"
                sample_253_error_path = os.path.join(current_asc_dir, sample_253_error_name)
                pd.DataFrame([
                    {
                        "LM": f"LM{lm_idx + 1:02d}",
                        "Pred_X": round(float(pred_np[lm_idx, 0]), 6),
                        "Pred_Y": round(float(pred_np[lm_idx, 1]), 6),
                        "Pred_Z": round(float(pred_np[lm_idx, 2]), 6),
                        "GT_X": round(float(gt_np[lm_idx, 0]), 6),
                        "GT_Y": round(float(gt_np[lm_idx, 1]), 6),
                        "GT_Z": round(float(gt_np[lm_idx, 2]), 6),
                        "Error_mm": round(float(dists[lm_idx]), 6),
                        "Surface_Distance_mm": round(float(surface_distance_np[lm_idx]), 6),
                    }
                    for lm_idx in range(len(dists))
                ]).to_csv(windows_write_path(sample_253_error_path), index=False, encoding="utf-8-sig")
                sample_253_exports.append({
                    "Sample_Name": real_name,
                    "Pred_ASC": pred_asc_relpath,
                    "Error_CSV": os.path.join("LM", short_eval_name, sample_253_error_name),
                })

            sample_record = {
                "Index": idx,
                "Sample_Name": real_name,
                "Mean_Error_mm": round(float(me), 6),
                "Sample_Error_95ile_mm": round(sample_error_95, 6),
                "Surface_Normal_Distance_mm": round(float(surface_p2p_mm), 6),
                "Surface_Distance_ME_mm": round(surface_distance_sample_me, 6),
                "Surface_Distance_STD_mm": round(surface_distance_sample_std, 6),
                "Surface_Distance_95ile_mm": round(surface_distance_sample_95, 6),
                "Surface_Loss_Norm": round(float(L_srf.item()), 6),
                "Surface_Curvature_Diff": round(float(curv_diff.item()), 6),
                "Surface_Direction_Diff": round(float(dir_diff.item()), 6),
                "Pred_ASC": pred_asc_relpath,
            }
            for lm_idx, dist in enumerate(dists):
                sample_record[f"LM{lm_idx + 1:02d}_Error_mm"] = round(float(dist), 6)
            for lm_idx, surface_dist in enumerate(surface_distance_np):
                sample_record[f"LM{lm_idx + 1:02d}_Surface_Distance_mm"] = round(float(surface_dist), 6)
            sample_records.append(sample_record)

    # ------------------ 지표 산출 및 리포팅 ------------------
    avg_cos_sim = np.mean(cos_sim_list) if cos_sim_list else 0.0
    cos_sim_5_global = np.percentile(cos_sim_list, 5) if cos_sim_list else 0.0
    avg_iou = np.mean(iou_list) if iou_list else 0.0
    iou_5_global = np.percentile(iou_list, 5) if iou_list else 0.0
    avg_time = np.mean(time_list) * 1000.0
    avg_surface_loss = np.mean(surface_loss_list) if surface_loss_list else 0.0
    avg_surface_p2p_mm = np.mean(surface_p2p_mm_list) if surface_p2p_mm_list else 0.0
    avg_surface_curv_diff = np.mean(surface_curv_diff_list) if surface_curv_diff_list else 0.0
    avg_surface_dir_diff = np.mean(surface_dir_diff_list) if surface_dir_diff_list else 0.0
    if surface_distance_lm_list:
        surface_distance_array = np.stack(surface_distance_lm_list, axis=0)
        surface_lm_means = np.mean(surface_distance_array, axis=0)
        surface_lm_stds = np.std(surface_distance_array, axis=0)
        surface_lm_95 = np.percentile(surface_distance_array, 95, axis=0)
        surface_distance_me = float(np.mean(surface_distance_array))
        surface_distance_std = float(np.std(surface_distance_array))
        surface_distance_95 = float(np.percentile(surface_distance_array, 95))
        surface_sample_me_95 = float(np.percentile(surface_distance_sample_me_list, 95))
    else:
        surface_lm_means = np.zeros(args.landmark_num)
        surface_lm_stds = np.zeros(args.landmark_num)
        surface_lm_95 = np.zeros(args.landmark_num)
        surface_distance_me = 0.0
        surface_distance_std = 0.0
        surface_distance_95 = 0.0
        surface_sample_me_95 = 0.0

    if len(per_landmark_me_list) > 0:
        per_landmark_me_array = np.stack(per_landmark_me_list, axis=0)
        lm_means, lm_stds, lm_me_95 = np.mean(per_landmark_me_array, axis=0), np.std(per_landmark_me_array, axis=0), np.percentile(per_landmark_me_array, 95, axis=0)
        flat_me_values = per_landmark_me_array.reshape(-1)
        flat_me_values = flat_me_values[np.isfinite(flat_me_values)]
        sample_me_values = np.array(me_list)
        average_me, std_me = np.mean(flat_me_values), np.std(flat_me_values)
        sample_me_95_global = np.percentile(sample_me_values, 95)
        flat_me_95_global = np.percentile(flat_me_values, 95)
        me_95_global = flat_me_95_global
        flat_me_top5_mean = np.mean(flat_me_values[flat_me_values >= flat_me_95_global])
        sr_2_flat = np.sum(flat_me_values < 2.0) / len(flat_me_values) * 100
        sr_3_flat = np.sum(flat_me_values < 3.0) / len(flat_me_values) * 100
        sr_5_flat = np.sum(flat_me_values < 5.0) / len(flat_me_values) * 100
        fr_2_flat = 100.0 - sr_2_flat
        fr_3_flat = 100.0 - sr_3_flat
        fr_5_flat = 100.0 - sr_5_flat
        sr_10 = np.sum(np.array(me_list) < 10.0) / len(me_list) * 100
        sr_5  = np.sum(np.array(me_list) < 5.0) / len(me_list) * 100
        worst_indices = np.argsort(lm_means)[::-1][:10]
    else:
        average_me, std_me, me_95_global, sr_10, sr_5 = 0.0, 0.0, 0.0, 0.0, 0.0
        sample_me_95_global = 0.0
        flat_me_95_global, flat_me_top5_mean = 0.0, 0.0
        sr_2_flat, sr_3_flat, sr_5_flat = 0.0, 0.0, 0.0
        fr_2_flat, fr_3_flat, fr_5_flat = 0.0, 0.0, 0.0
        lm_means, lm_stds, lm_me_95 = np.zeros(args.landmark_num), np.zeros(args.landmark_num), np.zeros(args.landmark_num)
        worst_indices = np.arange(10)

    filename_excel = f"{eval_name}_Results_ME{average_me:.4f}.xlsx"
    result_excel_path = os.path.join(run_root, filename_excel)
    prediction_csv_path = os.path.join(
        run_root,
        f"{short_eval_name}_pred_landmarks_all_{eval_datatype}.csv",
    )
    pd.DataFrame(prediction_rows).to_csv(windows_write_path(prediction_csv_path), index=False, encoding="utf-8-sig")
    prediction_csv_relpath = os.path.basename(prediction_csv_path)
    landmark_mapping_relpath = os.path.basename(landmark_mapping_path)
    sample_253_asc = sample_253_exports[0]["Pred_ASC"] if sample_253_exports else "not found in this eval split"
    sample_253_error_csv = sample_253_exports[0]["Error_CSV"] if sample_253_exports else "not found in this eval split"
    
    summary_data = {
        "지표 (Metric)": [
            "[Metadata]", "Experiment", "Run ID", "Model", 
            "[Metrics]", "Average ME (mm)", "Average Std (mm)", "Avg Time (ms)", 
            "Cosine Sim (%), ", "95%ile Cosine (%)", "mIoU (@0.1, %)", 
            "95%ile mIoU (%)", "Success Rate (<10mm, %)", "Success Rate (<5mm, %)",
            "[Surface Diagnostics]", "Surface Loss Mode", "Surface Loss (norm)",
            "Surface Normal Distance (mm)", "Surface Distance ME (mm)",
            "Surface Distance Std (mm)", "Surface Distance 95%ile (mm)",
            "Surface Sample-ME 95%ile (mm)", "Surface Curvature Diff", "Surface Direction Diff"
        ],
        eval_name: [
            "", args.exp_name, target_folder_name, eval_name,
            "", round(average_me, 4), round(std_me, 4), round(avg_time, 2), 
            round(avg_cos_sim, 2), round(cos_sim_5_global, 2), round(avg_iou, 2), 
            round(iou_5_global, 2), round(sr_10, 2), round(sr_5, 2),
            "", surface_loss_mode, round(avg_surface_loss, 6),
            round(avg_surface_p2p_mm, 4), round(surface_distance_me, 4),
            round(surface_distance_std, 4), round(surface_distance_95, 4),
            round(surface_sample_me_95, 4), round(avg_surface_curv_diff, 6),
            round(avg_surface_dir_diff, 6)
        ]
    }
    df_summary = pd.DataFrame(summary_data)
    df_summary = pd.DataFrame({
        "Metric": [
            "[Metadata]", "Experiment", "Run ID", "Model", "Eval Split",
            "[Coordinate Error]", "Average ME (mm)", "Average Std (mm)",
            "Average Mode", "P95 ME (flat LM error, mm)",
            "P95 Surface Distance (flat LM, mm)", "Sample-mean P95 ME (legacy, mm)",
            "Top 5% ME Mean (flat LM error, mm)",
            "SR@2mm (flat LM, %)", "SR@3mm (flat LM, %)", "SR@5mm (flat LM, %)",
            "FR@2mm (flat LM, %)", "FR@3mm (flat LM, %)", "FR@5mm (flat LM, %)",
            "SR@10mm (sample mean, %)", "SR@5mm (sample mean, %)",
            "Average Inference Time (ms)",
            "[Heatmap]", "Heatmap Cosine Sim (%)", "95%ile Cosine (%)",
            "mIoU (@0.1, %)", "95%ile mIoU (%)",
            "[Surface Diagnostics]", "Surface Loss Mode", "Surface Loss (norm)",
            "Surface Normal Distance (mm)", "Surface Distance ME (mm)",
            "Surface Distance STD (mm)", "Surface Distance 95%ile (mm)",
            "Surface Sample-ME 95%ile (mm)", "Surface Curvature Diff", "Surface Direction Diff",
            "[Exports]", "Predicted Coordinates CSV", "Landmark Mapping CSV",
            "LM ASC Directory", "Sample 253 Pred ASC", "Sample 253 Error CSV",
            "Coordinate/Scale Note",
        ],
        eval_name: [
            "", args.exp_name, target_folder_name, eval_name, eval_datatype,
            "", round(average_me, 4), round(std_me, 4),
            "flat landmark error", round(me_95_global, 4),
            round(surface_distance_95, 4), round(sample_me_95_global, 4),
            round(flat_me_top5_mean, 4),
            round(sr_2_flat, 2), round(sr_3_flat, 2), round(sr_5_flat, 2),
            round(fr_2_flat, 2), round(fr_3_flat, 2), round(fr_5_flat, 2),
            round(sr_10, 2), round(sr_5, 2),
            round(avg_time, 2),
            "", round(avg_cos_sim, 2), round(cos_sim_5_global, 2),
            round(avg_iou, 2), round(iou_5_global, 2),
            "", surface_loss_mode, round(avg_surface_loss, 6),
            round(avg_surface_p2p_mm, 4), round(surface_distance_me, 4),
            round(surface_distance_std, 4), round(surface_distance_95, 4),
            round(surface_sample_me_95, 4), round(avg_surface_curv_diff, 6),
            round(avg_surface_dir_diff, 6),
            "", prediction_csv_relpath, landmark_mapping_relpath,
            os.path.join("LM", short_eval_name), sample_253_asc, sample_253_error_csv,
            "Predicted ASC/CSV coordinates are denormalized back to the input point-cloud coordinate system.",
        ],
    })

    lm_me_combined = [f"{lm_means[i]:.3f} ± {lm_stds[i]:.3f}" for i in range(lm_means.shape[0])]
    df_landmarks = pd.DataFrame({
        "LM": [f"{i+1:02d}" for i in range(lm_means.shape[0])],
        f"{eval_name} (Mean ± Std)": lm_me_combined,
        f"{eval_name} (95%ile)": np.round(lm_me_95, 3),
        "Surface Distance (Mean +/- Std)": [
            f"{surface_lm_means[i]:.3f} +/- {surface_lm_stds[i]:.3f}"
            for i in range(surface_lm_means.shape[0])
        ],
        "Surface Distance (95%ile)": np.round(surface_lm_95, 3)
    })

    df_top10 = pd.DataFrame({
        "순위": [f"{r+1}위" for r in range(10)],
        eval_name: [f"LM {i+1:02d} ({lm_means[i]:.3f} ± {lm_stds[i]:.3f})" for i in worst_indices],
        f"{eval_name} (95%ile)": [round(lm_me_95[i], 3) for i in worst_indices]
    })
    df_landmarks = pd.DataFrame({
        "LM": [f"LM{i + 1:02d}" for i in range(lm_means.shape[0])],
        "ME_mm": np.round(lm_means, 6),
        "STD_mm": np.round(lm_stds, 6),
        "ME +/- STD (mm)": [f"{lm_means[i]:.3f} +/- {lm_stds[i]:.3f}" for i in range(lm_means.shape[0])],
        "Per-Landmark 95%ile (mm)": np.round(lm_me_95, 6),
        "Surface Distance ME_mm": np.round(surface_lm_means, 6),
        "Surface Distance STD_mm": np.round(surface_lm_stds, 6),
        "Surface Distance ME +/- STD (mm)": [
            f"{surface_lm_means[i]:.3f} +/- {surface_lm_stds[i]:.3f}"
            for i in range(surface_lm_means.shape[0])
        ],
        "Surface Distance 95%ile (mm)": np.round(surface_lm_95, 6),
    })
    df_top10 = pd.DataFrame({
        "Rank": [r + 1 for r in range(len(worst_indices))],
        "LM": [f"LM{i + 1:02d}" for i in worst_indices],
        "ME_mm": [round(float(lm_means[i]), 6) for i in worst_indices],
        "STD_mm": [round(float(lm_stds[i]), 6) for i in worst_indices],
        "ME +/- STD (mm)": [f"{lm_means[i]:.3f} +/- {lm_stds[i]:.3f}" for i in worst_indices],
        "Per-Landmark 95%ile (mm)": [round(float(lm_me_95[i]), 6) for i in worst_indices],
        "Surface Distance ME_mm": [round(float(surface_lm_means[i]), 6) for i in worst_indices],
        "Surface Distance 95%ile (mm)": [round(float(surface_lm_95[i]), 6) for i in worst_indices],
    })

    df_samples = pd.DataFrame(sample_records)

    write_dataframes_to_xlsx(
        windows_write_path(result_excel_path),
        [
            ('1_Summary', df_summary),
            ('2_Per_Landmark', df_landmarks),
            ('3_Top10_Hardest', df_top10),
            ('4_Per_Sample', df_samples),
        ]
    )

    txt_path = result_excel_path.replace(".xlsx", ".txt")
    with open(windows_write_path(txt_path), 'w', encoding='utf-8') as f:
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
        f.write(f"- Average Mode           : flat landmark error\n")
        f.write(f"- P95 ME (flat LM)       : {me_95_global:.4f} mm\n")
        f.write(f"- P95 Surface Distance   : {surface_distance_95:.4f} mm\n")
        f.write(f"- Sample-mean P95 ME (legacy): {sample_me_95_global:.4f} mm\n")
        f.write(f"- Top 5% ME Mean (flat LM): {flat_me_top5_mean:.4f} mm\n")
        f.write(f"- SR@2/3/5mm (flat LM)   : {sr_2_flat:.2f} / {sr_3_flat:.2f} / {sr_5_flat:.2f} %\n")
        f.write(f"- FR@2/3/5mm (flat LM)   : {fr_2_flat:.2f} / {fr_3_flat:.2f} / {fr_5_flat:.2f} %\n")

        f.write("\n[1-1. Surface Diagnostics]\n")
        f.write(f"- Surface Loss Mode          : {surface_loss_mode}\n")
        f.write(f"- Surface Loss (norm)        : {avg_surface_loss:.6f}\n")
        f.write(f"- Surface Normal Distance    : {avg_surface_p2p_mm:.4f} mm\n")
        f.write(f"- Surface Distance ME        : {surface_distance_me:.4f} +/- {surface_distance_std:.4f} mm\n")
        f.write(f"- Surface Distance 95%ile    : {surface_distance_95:.4f} mm\n")
        f.write(f"- Surface Sample-ME 95%ile   : {surface_sample_me_95:.4f} mm\n")
        f.write(f"- Surface Curvature Diff     : {avg_surface_curv_diff:.6f}\n")
        f.write(f"- Surface Direction Diff     : {avg_surface_dir_diff:.6f}\n")

        f.write("\n[1-2. Coordinate Exports]\n")
        f.write(f"- Predicted Coordinates CSV : {prediction_csv_relpath}\n")
        f.write(f"- Landmark Mapping CSV      : {landmark_mapping_relpath}\n")
        f.write(f"- LM ASC Directory          : {os.path.join('LM', short_eval_name)}\n")
        f.write(f"- Sample 253 Pred ASC       : {sample_253_asc}\n")
        f.write(f"- Sample 253 Error CSV      : {sample_253_error_csv}\n")
        f.write("- Coordinate/Scale Note     : denormalized back to the input point-cloud coordinate system\n")
        
        f.write("\n[2. Per-Landmark Errors (All 36)]\n")
        for i in range(lm_means.shape[0]):
            f.write(
                f"  LM {i+1:02d} : {lm_means[i]:.3f} ± {lm_stds[i]:.3f} mm "
                f"(95%ile: {lm_me_95[i]:.3f} mm) | "
                f"Surface: {surface_lm_means[i]:.3f} +/- {surface_lm_stds[i]:.3f} mm "
                f"(95%ile: {surface_lm_95[i]:.3f} mm)\n"
            )
            
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

    if not os.path.exists(result_excel_path) or not os.path.exists(txt_path):
        print("[ERROR] Evaluation artifacts were not written correctly.")
        print(f"[ERROR] Results Excel path: {result_excel_path}")
        print(f"[ERROR] Results text path : {txt_path}")
        sys.exit(1)

    print(f"\n[{eval_name} Done] Excel saved to: {filename_excel}")
    print(f"  └─ 📄 Text summary perfectly synchronized & saved to: {os.path.basename(txt_path)}")
    print(f"Average ME: {average_me:.4f} ± {std_me:.4f} (95%ile: {me_95_global:.4f} mm)")
    print(f"Flat LM metrics: SR@2/3/5={sr_2_flat:.2f}/{sr_3_flat:.2f}/{sr_5_flat:.2f}%, FR@2/3/5={fr_2_flat:.2f}/{fr_3_flat:.2f}/{fr_5_flat:.2f}%, P95 ME={me_95_global:.4f} mm")
    print(f"Surface Distance: {surface_distance_me:.4f} +/- {surface_distance_std:.4f} (95%ile: {surface_distance_95:.4f} mm)")

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

evaluate_target_model(eval_output_name, model, pipeline_mode)

print("\n>>> [ALL EVALUATION COMPLETED]")
