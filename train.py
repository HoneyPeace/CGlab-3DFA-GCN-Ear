# @Author: Yuan Wang (Modified by Researcher & AI Assistant)
# @File: train.py
# @Description: 스케일링 정규화 + 0에폭 전면 4-Way RLW + PAConv 기하학적 풀각성 및 표면로스 4종 세부 로깅
# ==============================================================================
# [NotebookLM을 위한 파이프라인 아키텍처 요약]
# 이 스크립트는 3D 귀 랜드마크 탐지를 위한 Two-stage (PAConv -> DeepPA) 조인트 모델의 메인 훈련 루프입니다.
# 
# 💡 핵심 설계 철학 3가지:
# 1. Scaling Normalization (스케일링 정규화): 학습 전(가상 에폭) 기울기 계산 없이 전체 데이터를 순회하여
#    4대 로스의 정적 체급(Static Error)을 재고 1.0으로 맞추는 고정 배수(Multiplier)를 구합니다.
# 2. True 4-Way RLW (0에폭 진정한 풀각성): 기존의 '히트맵 고정 5%' 꼼수를 완전히 버리고,
#    학습 시작(0에폭)부터 [Heatmap, Coord, Surface, Struct] 4대 로스가 동등하게 랜덤 가중치(RLW)를
#    나눠 가지며 무한 경쟁하도록 설계했습니다. 이를 통해 Stage 1과 2가 극도로 다이나믹하게 수렴합니다.
# 3. Detailed Auxiliary Logging (초정밀 보조 로스 모니터링): 
#    표면 로스(Surface Loss)를 단순 1개 수치가 아닌 [통합, 순수P2P거리, 곡률오차, 방향오차] 4개로 
#    분해하여 엑셀에 기록함으로써, 모델이 형태를 헷갈리는지 방향을 헷갈리는지 완벽히 추적합니다.
# ==============================================================================

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

from init import _init_
from My_args import parser
from dataset import FaceLandmarkData
from loss import AdaptiveWingLoss, get_differentiable_coords, CurvatureSurfaceLoss, compute_structural_loss, dynamic_focal_l1_loss
from util import main_sample
from augmentations import normalize_data, PointcloudScaleAndTranslate

from PAConv_model import PAConv
from DeepLA_model import DeepLA_Wrapper
from DeepPA_model import DeepPA_Wrapper  

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weight_init(m):
    # [의도] 초기 기울기 소실/폭발 방지용 Xavier 및 Kaiming 초기화
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight)

def get_experiment_paths(args, train_len):
    # [NotebookLM 참고] 덮어쓰기 방지를 위한 자동 넘버링 디렉토리 생성
    project_dir = os.path.join(args.output_root, args.exp_name)
    os.makedirs(project_dir, exist_ok=True)

    batch_str = f"{args.batch_size}x{args.accumulation_steps}" if args.accumulation_steps > 1 else f"{args.batch_size}"
    base_str = f"FPS{args.num_points}_sigma{args.sigma}_batch{batch_str}_train{train_len}"
    setting_str = f"{base_str}_{args.user_tag}" if args.user_tag else base_str
    
    count = 1
    while True:
        run_name = f"{setting_str}_{count}"
        run_dir = os.path.join(project_dir, run_name)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            break
        count += 1
    
    paths = {
        'root': run_dir, 'models': os.path.join(run_dir, 'models'),
        'npy_backup': os.path.join(run_dir, 'npy_data'), 'gt_heatmap': os.path.join(run_dir, 'GT_Heatmaps')
    }
    for p in paths.values(): os.makedirs(p, exist_ok=True)
    return paths

def process_data_storage(dataset, prefix, paths):
    # [의도] 학습 데이터 위변조 방지 및 재현성 확보를 위한 NPY 원본 백업
    shape_list, landmark_list, heatmap_list = zip(*[(p.numpy(), l.numpy(), h.numpy()) for p, l, h in dataset])
    np.save(os.path.join(paths['npy_backup'], f"shape_{prefix}.npy"), np.stack(shape_list))
    np.save(os.path.join(paths['npy_backup'], f"landmark_{prefix}.npy"), np.stack(landmark_list))
    np.save(os.path.join(paths['npy_backup'], f"Heat_data_{prefix}.npy"), np.stack(heatmap_list))

class JointE2EModel(nn.Module):
    # [NotebookLM 참고] PAConv와 DeepPA를 직렬 연결하여 동시 역전파가 가능하도록 묶은 컨테이너
    def __init__(self, args, landmark_num):
        super().__init__()
        self.stage1_paconv = PAConv(args, landmark_num)
        self.stage2_deeppa = DeepPA_Wrapper(args, landmark_num)

    def forward(self, x):
        prior_hint = self.stage1_paconv(x)
        pred_heatmap = self.stage2_deeppa(x, prior_heatmap=prior_hint)
        if self.training: return pred_heatmap, prior_hint
        return pred_heatmap

def train(args):
    accum_steps = args.accumulation_steps
    MODE = "SPLIT" if not args.test_dataset_name or (args.train_dataset_name == args.test_dataset_name) else "SEPARATE"

    if args.need_resample:
        main_sample(args.num_points, args.seed, args.sigma, args.sample_way, args.train_dataset_name, args.data_root, partition='train')
        main_sample(args.num_points, args.seed, args.sigma, args.sample_way, args.test_dataset_name, args.data_root, partition='test')

    if MODE == "SPLIT":
        full_dataset = FaceLandmarkData(data_root=args.data_root, partition='trainval', data=args.train_dataset_name, in_channels=args.in_channels)
        train_size = int(len(full_dataset) * 0.7)
        test_size = len(full_dataset) - train_size
        torch.manual_seed(args.dataset_seed)
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    else:
        train_dataset = FaceLandmarkData(data_root=args.data_root, partition='train', data=args.train_dataset_name, in_channels=args.in_channels)
        test_dataset = FaceLandmarkData(data_root=args.data_root, partition='test', data=args.test_dataset_name, in_channels=args.in_channels)

    paths = get_experiment_paths(args, len(train_dataset))
    process_data_storage(train_dataset, "train", paths)
    process_data_storage(test_dataset, "test", paths)

    train_loader = DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, num_workers=0, batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    ScaleAndTranslate = PointcloudScaleAndTranslate()

    def execute_stage(current_model_name, current_epochs, disable_norm=False, prior_model=None, stage_name=""):
        model_name_lower = current_model_name.lower()
        if model_name_lower == 'deeppa_e2e': model = JointE2EModel(args, args.landmark_num).to(device)
        else: model = DeepPA_Wrapper(args, args.landmark_num).to(device)
            
        model.apply(weight_init)
        
        # 🌟 My_args.py 의 알파/베타 값 연동
        surface_criterion = CurvatureSurfaceLoss(k_p2p=args.plane_knn, k_curv=args.curv_knn, alpha=args.curv_alpha, beta=args.dir_beta).to(device)
        criterion = AdaptiveWingLoss() if args.loss == 'adaptive_wing' else torch.nn.MSELoss()
        
        opt = optim.Adam(model.parameters(), lr=args.lr, eps=1e-08, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(opt, T_max=current_epochs) if args.scheduler == 'cos' else StepLR(opt, step_size=40, gamma=0.9)

        excel_log_path = os.path.join(paths['root'], f'Training_Log_{stage_name}.xlsx')
        log_records = []
        best_val_mm = float('inf')
        target_norm = 1.0 
        auto_scales = {'heatmap': -1.0, 'coord': -1.0, 'surface': -1.0, 'struct': -1.0}

        # =========================================================================
        # 🟢 [Phase 2.9] 스케일링 정규화 (가상 에폭 캘리브레이션)
        # =========================================================================
        if not disable_norm:
            print(f"\n🔍 [Scaling Normalization] 스케일링 정규화 및 세부 로스 전수 분석 중...")
            model.eval() 
            
            sum_losses = {'heatmap': 0.0, 'coord': 0.0, 'surface': 0.0, 'struct': 0.0, 'p2p': 0.0, 'curv': 0.0, 'dir': 0.0}
            single_batch_losses = {}
            
            with torch.no_grad():
                for i, (point, landmark, seg) in enumerate(tqdm(train_loader, desc="Calibrating", leave=False)):
                    point, landmark, seg = point.to(device), landmark.to(device), seg.to(device)
                    point_normal, landmark_normal = normalize_data(point, landmark)
                    point_normal, augmented_landmark = ScaleAndTranslate(point_normal, landmark_normal)
                    point_input = point_normal.permute(0, 2, 1).contiguous()
                    
                    pred_heatmap = model(point_input) if model_name_lower != 'deeppa_e2e' else model(point_input)
                    points_for_coords = point_input[:, :3, :].permute(0, 2, 1).contiguous() 
                    pred_coords = get_differentiable_coords(points_for_coords, pred_heatmap, k=args.k_softargmax)
                    
                    l_hm = criterion(pred_heatmap, seg.permute(0, 2, 1).contiguous()).item()
                    l_crd = dynamic_focal_l1_loss(pred_coords, augmented_landmark, gamma=args.focal_gamma).item()
                    
                    l_srf_tensor, l_p2p, l_curv, l_dir = surface_criterion(pred_coords, augmented_landmark, points_for_coords, disable_norm=disable_norm)
                    l_srf = l_srf_tensor.item()
                    
                    l_str = compute_structural_loss(pred_coords, augmented_landmark).item()

                    if i == 0: 
                        single_batch_losses = {
                            'heatmap': l_hm, 'coord': l_crd, 'surface': l_srf, 'struct': l_str,
                            'p2p': l_p2p.item(), 'curv': l_curv.item(), 'dir': l_dir.item()
                        }
                    sum_losses['heatmap'] += l_hm; sum_losses['coord'] += l_crd; sum_losses['surface'] += l_srf; sum_losses['struct'] += l_str
                    sum_losses['p2p'] += l_p2p.item(); sum_losses['curv'] += l_curv.item(); sum_losses['dir'] += l_dir.item()

            num_batches = len(train_loader)
            avg_losses = {k: v / num_batches for k, v in sum_losses.items()}
            auto_scales = {k: target_norm / (avg_losses[k] + 1e-6) for k in ['heatmap', 'coord', 'surface', 'struct']}
            
            df_calib = pd.DataFrame({
                "Metric": ["Raw_Loss (1 Batch)", "Raw_Loss (Full Avg)", "Multiplier (Based on Full)"],
                "Heatmap": [single_batch_losses['heatmap'], avg_losses['heatmap'], auto_scales['heatmap']],
                "Coord": [single_batch_losses['coord'], avg_losses['coord'], auto_scales['coord']],
                "Surface_Unified": [single_batch_losses['surface'], avg_losses['surface'], auto_scales['surface']],
                "Structure": [single_batch_losses['struct'], avg_losses['struct'], auto_scales['struct']],
                "Srf_P2P": [single_batch_losses['p2p'], avg_losses['p2p'], "-"],
                "Srf_Curv": [single_batch_losses['curv'], avg_losses['curv'], "-"],
                "Srf_Dir": [single_batch_losses['dir'], avg_losses['dir'], "-"]
            })
            print("\n📊 [Scaling Normalization Result]\n", df_calib.to_string(index=False))
            torch.cuda.empty_cache()

        # =========================================================================
        # 🟢 [Phase 3] 본 학습 (0 에폭부터 전면 4-Way RLW 훈련)
        # =========================================================================
        for epoch in range(current_epochs):
            model.train() 
            train_loss_norm, train_loss_reversed = 0.0, 0.0
            
            raw_hm, raw_crd, raw_srf, raw_str = 0.0, 0.0, 0.0, 0.0
            s2_p2p_raw, s2_curv_raw, s2_dir_raw = 0.0, 0.0, 0.0
            
            s1_raw_hm, s1_raw_crd, s1_raw_srf, s1_raw_str = 0.0, 0.0, 0.0, 0.0
            s1_p2p_raw, s1_curv_raw, s1_dir_raw = 0.0, 0.0, 0.0 
            
            train_mm = 0.0
            opt.zero_grad() 
            
            with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"{stage_name} Ep {epoch+1:03d} [Train]", unit="batch", leave=False) as tepoch:
                for i, (point, landmark, seg) in tepoch:
                    point, landmark, seg = point.to(device), landmark.to(device), seg.to(device)

                    with torch.no_grad():
                        point_xyz = point[:, :, :3]
                        avg_m = torch.mean(torch.max(torch.sqrt(torch.sum((point_xyz - torch.mean(point_xyz, axis=1, keepdim=True)) ** 2, axis=2)), axis=1)[0]).item()

                    point_normal, landmark_normal = normalize_data(point, landmark)
                    point_normal, augmented_landmark = ScaleAndTranslate(point_normal, landmark_normal)
                    point_input = point_normal.permute(0, 2, 1).contiguous()
                    points_for_coords = point_input[:, :3, :].permute(0, 2, 1).contiguous() 
                    
                    # 🌟🌟 [Stage 1: PAConv 진정한 4-Way RLW 가동] 🌟🌟
                    if model_name_lower == 'deeppa_e2e':
                        pred_heatmap, prior_hint = model(point_input)
                        prior_coords = get_differentiable_coords(points_for_coords, prior_hint, k=args.k_softargmax)
                        
                        loss_hm_s1 = criterion(prior_hint, seg.permute(0, 2, 1).contiguous())
                        loss_crd_s1 = dynamic_focal_l1_loss(prior_coords, augmented_landmark, gamma=args.focal_gamma)
                        loss_srf_s1, p2p_s1, curv_err_s1, dir_err_s1 = surface_criterion(prior_coords, augmented_landmark, points_for_coords, disable_norm=False)
                        loss_str_s1 = compute_structural_loss(prior_coords, augmented_landmark)
                        
                        norm_hm_s1 = loss_hm_s1 * auto_scales['heatmap']
                        norm_crd_s1 = loss_crd_s1 * auto_scales['coord']
                        norm_srf_s1 = loss_srf_s1 * auto_scales['surface']
                        norm_str_s1 = loss_str_s1 * auto_scales['struct']
                        
                        # 🌟 4개 로스 동등하게 랜덤 가중치 분배!
                        rand_w_s1 = torch.rand(4).to(device)
                        rand_w_s1 = rand_w_s1 / rand_w_s1.sum()
                        
                        total_loss_stage1 = (rand_w_s1[0] * norm_hm_s1 + rand_w_s1[1] * norm_crd_s1 + 
                                             rand_w_s1[2] * norm_srf_s1 + rand_w_s1[3] * norm_str_s1)
                        
                        s1_raw_hm += loss_hm_s1.item(); s1_raw_crd += loss_crd_s1.item(); s1_raw_srf += loss_srf_s1.item(); s1_raw_str += loss_str_s1.item()
                        s1_p2p_raw += p2p_s1.item(); s1_curv_raw += curv_err_s1.item(); s1_dir_raw += dir_err_s1.item()
                    else:
                        pred_heatmap = model(point_input)
                        total_loss_stage1 = 0.0

                    # 🌟🌟 [Stage 2: DeepPA 진정한 4-Way RLW 가동] 🌟🌟
                    pred_coords = get_differentiable_coords(points_for_coords, pred_heatmap, k=args.k_softargmax)
                    
                    loss_heatmap = criterion(pred_heatmap, seg.permute(0, 2, 1).contiguous())
                    loss_coord = dynamic_focal_l1_loss(pred_coords, augmented_landmark, gamma=args.focal_gamma)
                    loss_surface, p2p_s2, curv_err_s2, dir_err_s2 = surface_criterion(pred_coords, augmented_landmark, points_for_coords, disable_norm=False)
                    loss_struct = compute_structural_loss(pred_coords, augmented_landmark)
                    
                    norm_heatmap = loss_heatmap * auto_scales['heatmap']
                    norm_coord   = loss_coord   * auto_scales['coord']
                    norm_surface = loss_surface * auto_scales['surface']         
                    norm_struct  = loss_struct  * auto_scales['struct']

                    # 🌟 4개 로스 동등하게 랜덤 가중치 분배!
                    rand_weights = torch.rand(4).to(device)
                    rand_weights = rand_weights / rand_weights.sum()
                    
                    total_loss_stage2 = (rand_weights[0] * norm_heatmap + rand_weights[1] * norm_coord + 
                                         rand_weights[2] * norm_surface + rand_weights[3] * norm_struct)
                    
                    total_loss = total_loss_stage2 + total_loss_stage1
                    
                    # 🌟 물리적 역산 로깅도 4-Way 가중치에 맞춰 수정
                    total_loss_raw_reversed = (rand_weights[0] * loss_heatmap + rand_weights[1] * loss_coord + 
                                               rand_weights[2] * loss_surface + rand_weights[3] * loss_struct)
                    if model_name_lower == 'deeppa_e2e':
                        total_loss_raw_reversed += (rand_w_s1[0] * loss_hm_s1 + rand_w_s1[1] * loss_crd_s1 + 
                                                    rand_w_s1[2] * loss_srf_s1 + rand_w_s1[3] * loss_str_s1)

                    loss = total_loss / accum_steps
                    loss.backward()
                    
                    if (i + 1) % accum_steps == 0:
                        opt.step()
                        opt.zero_grad() 

                    with torch.no_grad():
                        mm_error = F.l1_loss(pred_coords, augmented_landmark).item() * avg_m

                    train_loss_norm += total_loss.item()
                    train_loss_reversed += total_loss_raw_reversed.item()
                    
                    raw_hm += loss_heatmap.item(); raw_crd += loss_coord.item(); raw_srf += loss_surface.item(); raw_str += loss_struct.item()
                    s2_p2p_raw += p2p_s2.item(); s2_curv_raw += curv_err_s2.item(); s2_dir_raw += dir_err_s2.item()
                    
                    train_mm += mm_error
                    vram_str = f"{torch.cuda.max_memory_allocated() / (1024 ** 3):.1f}GB" if torch.cuda.is_available() else "CPU"
                    tepoch.set_postfix(Loss=f"{total_loss.item():.4f}", mm=f"{mm_error:.2f}", VRAM=vram_str)

            num_b = len(train_loader)
            t_loss_n, t_loss_r = train_loss_norm / num_b, train_loss_reversed / num_b
            t_hm_raw, t_crd_raw, t_srf_raw, t_str_raw = raw_hm/num_b, raw_crd/num_b, raw_srf/num_b, raw_str/num_b
            t_mm = train_mm / num_b

            # --- Validation ---
            model.eval()
            val_hm, val_mm = 0.0, 0.0
            
            with torch.no_grad():
                for point, landmark, seg in test_loader:
                    point, landmark, seg = point.to(device), landmark.to(device), seg.to(device)
                    point_xyz = point[:, :, :3]
                    avg_m = torch.mean(torch.max(torch.sqrt(torch.sum((point_xyz - torch.mean(point_xyz, axis=1, keepdim=True)) ** 2, axis=2)), axis=1)[0]).item()
                    point_normal, landmark_normal = normalize_data(point, landmark)
                    point_input = point_normal.permute(0, 2, 1).contiguous()
                    
                    pred_heatmap = model(point_input) if model_name_lower == 'deeppa_e2e' else model(point_input)
                    pred_coords = get_differentiable_coords(point_input[:, :3, :].permute(0, 2, 1).contiguous(), pred_heatmap, k=args.k_softargmax)

                    val_hm += criterion(pred_heatmap, seg.permute(0, 2, 1).contiguous()).item()
                    val_mm += F.l1_loss(pred_coords, landmark_normal).item() * avg_m

            v_hm, v_mm = val_hm/len(test_loader), val_mm/len(test_loader)
            
            print(f" [{stage_name} Ep {epoch+1:03d}] T_Norm: {t_loss_n:.2f} | T_mm: {t_mm:.2f} || V_mm: {v_mm:.2f}")
            print(f"  ├─ [S2_DeepPA] HM: {t_hm_raw:.4f} | Crd: {t_crd_raw:.4f} | Srf(Unified): {t_srf_raw:.4f} | Str: {t_str_raw:.4f}")
            if model_name_lower == 'deeppa_e2e':
                print(f"  └─ [S1_PAConv] HM: {s1_raw_hm/num_b:.4f} | Crd: {s1_raw_crd/num_b:.4f} | Srf(Unified): {s1_raw_srf/num_b:.4f} | Str: {s1_raw_str/num_b:.4f}")

            log_records.append({
                'Epoch': epoch + 1, 'Total_Loss_Norm': t_loss_n, 'Total_Loss_Raw_Phys': t_loss_r,
                'S2_HM_Raw': t_hm_raw, 'S2_Crd_Raw': t_crd_raw, 'S2_Str_Raw': t_str_raw,
                'S2_Srf_Unified': t_srf_raw, 'S2_P2P_Dist': s2_p2p_raw/num_b, 'S2_Curv_Err': s2_curv_raw/num_b, 'S2_Dir_Err': s2_dir_raw/num_b,
                'S1_Aux_HM_Raw': s1_raw_hm/num_b, 'S1_Aux_Crd_Raw': s1_raw_crd/num_b, 'S1_Aux_Str_Raw': s1_raw_str/num_b,
                'S1_Aux_Srf_Unified': s1_raw_srf/num_b, 'S1_Aux_P2P_Dist': s1_p2p_raw/num_b, 'S1_Aux_Curv_Err': s1_curv_raw/num_b, 'S1_Aux_Dir_Err': s1_dir_raw/num_b,
                'Train_mm': t_mm, 'Val_mm': v_mm
            })
            
            with pd.ExcelWriter(excel_log_path, engine='openpyxl') as writer:
                if not disable_norm: df_calib.to_excel(writer, sheet_name='1_Calibration', index=False)
                pd.DataFrame(log_records).to_excel(writer, sheet_name='2_Training_Log', index=False)

            if v_mm < best_val_mm:
                best_val_mm = v_mm
                if model_name_lower == 'deeppa_e2e':
                    torch.save(model.stage1_paconv.state_dict(), os.path.join(paths['models'], 'Stage1_PAConv_model_best.t7'))
                    torch.save(model.stage2_deeppa.state_dict(), os.path.join(paths['models'], 'Stage2_DeepPA_model_best.t7'))
                else:
                    best_save_path = os.path.join(paths['models'], f'{stage_name}_model_best.t7')
                    torch.save(model.state_dict(), best_save_path)

            scheduler.step()
            
        return model 

    print(f"\n=== [Phase 3] Start Auto End-to-End Pipeline ===")
    
    if args.model.lower() == 'deeppa_auto':
        print(f">>> [AUTO MODE] 🔥 0에폭 전면 4-Way RLW 훈련 가동 (Loss Norm: {'ON' if args.use_loss_norm else 'OFF'})")
        execute_stage('deeppa_e2e', args.epochs, disable_norm=not args.use_loss_norm, stage_name=f"E2E_Joint_Norm_ON")
    else:
        execute_stage(args.model, args.epochs, disable_norm=(args.model in ['PAConv', 'PAConv_heat']), stage_name=f"Single_{args.model}")
        
    print(f"\n=== Training Finished. Results at: {paths['root']} ===")

if __name__ == "__main__":
    args = parser.parse_args()
    _init_(args)
    train(args)