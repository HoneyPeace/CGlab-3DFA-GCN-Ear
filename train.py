# @Author: Yuan Wang (Modified by Researcher & AI Assistant)
# @File: train.py
# @Description: 동적 히트맵 선학습(Warm-up) + 스케일링 정규화 + RLW 분기 제어(닻 모드 vs 전면 랜덤)
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

from loss import AdaptiveWingLoss, CurvatureSurfaceLoss, compute_structural_loss, dynamic_focal_l1_loss
from util import main_sample
from augmentations import normalize_data, PointcloudScaleAndTranslate

from PAConv_model import PAConv
from DeepLA_model import DeepLA_Wrapper
from DeepPA_model import DeepPA_Wrapper  

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight)

def get_experiment_paths(args, train_len):
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
    shape_list, landmark_list, heatmap_list = zip(*[(p.numpy(), l.numpy(), h.numpy()) for p, l, h in dataset])
    np.save(os.path.join(paths['npy_backup'], f"shape_{prefix}.npy"), np.stack(shape_list))
    np.save(os.path.join(paths['npy_backup'], f"landmark_{prefix}.npy"), np.stack(landmark_list))
    np.save(os.path.join(paths['npy_backup'], f"Heat_data_{prefix}.npy"), np.stack(heatmap_list))

"""
================================================================================
[NotebookLM을 위한 모듈 요약: 계층적 최적화(Hierarchical Optimization) 및 Ablation]
- Stage 1 (PAConv): 64채널의 전역 기하학 라텐트 피처(Latent Feature) 생성.
- Stage 2 (DeepPA): 256채널 원본 직결 헤드를 통해 최종 다이렉트 좌표 예측.

- 🌟 [Ablation Study: RLW 분기 제어]
  1. 닻(Anchor) 모드 (기본값): 히트맵을 1.0 가중치로 고정하여 위치 감각을 단단히 잡고, 
     기하학 3-Loss(좌표, 표면, 구조)만 RLW로 경쟁시켜 세밀한 형태를 깎아냅니다.
  2. 전면 랜덤 모드 (--use_rlw_for_heatmap): 히트맵마저 RLW에 포함시켜, 모델이 
     특징 공간 정렬(Heatmap)과 기하학 최적화를 완전히 자율적으로 조율하게 합니다.
================================================================================
"""
class JointE2EModel(nn.Module):
    def __init__(self, args, landmark_num):
        super().__init__()
        self.stage1_paconv = PAConv(args, landmark_num)
        self.stage2_deeppa = DeepPA_Wrapper(args, landmark_num)
        self.s1_aux_head = nn.Conv1d(64, landmark_num, 1)

    def forward(self, x):
        s1_latent = self.stage1_paconv(x)
        # 🌟 [수정] Stage 2의 보조 히트맵(s2_aux_hm)은 사용하지 않으므로 무시합니다.
        pred_coords, _ = self.stage2_deeppa(x, prior_heatmap=s1_latent)
        s1_aux_hm = self.s1_aux_head(s1_latent)
        # s2_aux_hm 자리에 None을 반환하여 구조를 유지합니다.
        return pred_coords, None, s1_aux_hm

def train(args):
    accum_steps = args.accumulation_steps
    MODE = "SPLIT" if not args.test_dataset_name or (args.train_dataset_name == args.test_dataset_name) else "SEPARATE"
    
    # RLW 분기 스위치
    use_rlw_for_heatmap = getattr(args, 'use_rlw_for_heatmap', False)

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
        
        surface_criterion = CurvatureSurfaceLoss(k_p2p=args.plane_knn, k_curv=args.curv_knn, alpha=args.curv_alpha, beta=args.dir_beta).to(device)
        criterion = AdaptiveWingLoss() if args.loss == 'adaptive_wing' else torch.nn.MSELoss()
        
        opt = optim.Adam(model.parameters(), lr=args.lr, eps=1e-08, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(opt, T_max=current_epochs) if args.scheduler == 'cos' else StepLR(opt, step_size=40, gamma=0.9)

        excel_log_path = os.path.join(paths['root'], f'Training_Log_{stage_name}.xlsx')
        log_records = []
        target_norm = 1.0 
        auto_scales = {'heatmap': 1.0, 'coord': 1.0, 'surface': 1.0, 'struct': 1.0}

        is_warmup = getattr(args, 'use_warmup', True) and (model_name_lower == 'deeppa_e2e')
        warmup_patience = getattr(args, 'warmup_patience', 10)
        best_warmup_hm = float('inf')
        patience_counter = 0

        df_calib = None 

        # =========================================================================
        # 🟢 [Phase 2.9] 스케일링 정규화 (가상 에폭 캘리브레이션)
        # =========================================================================
        if not disable_norm:
            print(f"\n🔍 [Scaling Normalization] 스케일링 정규화 전수 분석 중...")
            model.eval() 
            sum_losses = {'heatmap': 0.0, 'coord': 0.0, 'surface': 0.0, 'struct': 0.0, 'p2p': 0.0, 'curv': 0.0, 'dir': 0.0}
            single_batch_losses = {}
            
            with torch.no_grad():
                for i, (point, landmark, seg) in enumerate(tqdm(train_loader, desc="Calibrating", leave=False)):
                    point, landmark, seg = point.to(device), landmark.to(device), seg.to(device)
                    point_normal, landmark_normal = normalize_data(point, landmark)
                    point_normal, augmented_landmark = ScaleAndTranslate(point_normal, landmark_normal)
                    point_input = point_normal.permute(0, 2, 1).contiguous()
                    
                    # train.py 약 135행 [Phase 2.9] 내부
                    if model_name_lower == 'deeppa_e2e':
                        # 🌟 [수정] s2_aux_hm은 None이므로 s1_aux_hm(Stage 1)의 로스만 HM 지표로 삼습니다.
                        pred_coords, _, s1_aux_hm = model(point_input)
                        l_hm = criterion(s1_aux_hm, seg.permute(0, 2, 1).contiguous()).item()
                    else:
                        # 단독 모델일 경우에도 다이렉트 좌표 위주로 측정
                        pred_coords, _ = model(point_input)
                        l_hm = 0.0 # 혹은 기존 로직 유지
                    
                    points_for_coords = point_input[:, :3, :].permute(0, 2, 1).contiguous() 
                    
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
            print(f"✅ 정규화 완료 (Multiplier: HM={auto_scales['heatmap']:.2f}, Crd={auto_scales['coord']:.2f}, Srf={auto_scales['surface']:.2f}, Str={auto_scales['struct']:.2f})")
            torch.cuda.empty_cache()

        # =========================================================================
        # 🟢 [Phase 3] 본 학습 (Main Training)
        # =========================================================================
        for epoch in range(current_epochs):
            model.train() 
            train_loss_norm, train_loss_reversed = 0.0, 0.0
            
            raw_hm, raw_crd, raw_srf, raw_str = 0.0, 0.0, 0.0, 0.0
            s2_p2p_raw, s2_curv_raw, s2_dir_raw = 0.0, 0.0, 0.0
            s1_raw_hm = 0.0 
            
            train_mm = 0.0
            opt.zero_grad() 
            
            phase_type = "4-Loss RLW" if use_rlw_for_heatmap else "3-Loss RLW (닻)"
            current_phase_str = "🔥 Warm-up (Heatmap Only)" if is_warmup else f"🚀 Main ({phase_type})"
            
            with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"{stage_name} Ep {epoch+1:03d} [{current_phase_str}]", unit="batch", leave=False) as tepoch:
                for i, (point, landmark, seg) in tepoch:
                    point, landmark, seg = point.to(device), landmark.to(device), seg.to(device)

                    with torch.no_grad():
                        point_xyz = point[:, :, :3]
                        avg_m = torch.mean(torch.max(torch.sqrt(torch.sum((point_xyz - torch.mean(point_xyz, axis=1, keepdim=True)) ** 2, axis=2)), axis=1)[0]).item()

                    point_normal, landmark_normal = normalize_data(point, landmark)
                    point_normal, augmented_landmark = ScaleAndTranslate(point_normal, landmark_normal)
                    point_input = point_normal.permute(0, 2, 1).contiguous()
                    points_for_coords = point_input[:, :3, :].permute(0, 2, 1).contiguous() 
                    
                    # ---------------------------------------------------------
                    # 1. Forward Pass 및 손실 계산 (직접 회귀)
                    # ---------------------------------------------------------
                    # train.py 약 215행 [Phase 3] 본 학습 내부
                    if model_name_lower == 'deeppa_e2e':
                        pred_coords, _, s1_aux_hm = model(point_input)
                        
                        # 🌟 [수정] Stage 1 히트맵만 위치 가이드(Anchor)로 사용합니다.
                        loss_hm_s1 = criterion(s1_aux_hm, seg.permute(0, 2, 1).contiguous())
                        norm_hm_s1 = loss_hm_s1 * auto_scales['heatmap']
                        s1_raw_hm += loss_hm_s1.item()
                        
                        # Stage 2 히트맵 로스 변수는 0으로 초기화 (계산 제외)
                        norm_hm_s2 = torch.tensor(0.0).to(device)
                        loss_hm_s2 = torch.tensor(0.0).to(device)
                    else:
                        pred_coords, s2_aux_hm = model(point_input)
                        loss_hm_s2 = criterion(s2_aux_hm, seg.permute(0, 2, 1).contiguous())
                        norm_hm_s2 = loss_hm_s2 * auto_scales['heatmap']
                        norm_hm_s1, loss_hm_s1 = torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
                        
                    loss_coord = dynamic_focal_l1_loss(pred_coords, augmented_landmark, gamma=args.focal_gamma)
                    loss_surface, p2p_s2, curv_err_s2, dir_err_s2 = surface_criterion(pred_coords, augmented_landmark, points_for_coords, disable_norm=False)
                    loss_struct = compute_structural_loss(pred_coords, augmented_landmark)
                    
                    norm_coord   = loss_coord   * auto_scales['coord']
                    norm_surface = loss_surface * auto_scales['surface']         
                    norm_struct  = loss_struct  * auto_scales['struct']

                    # ---------------------------------------------------------
                    # 2. 로스 융합 🌟 [세련된 구조: Stage 1 앵커 전용] 🌟
                    # ---------------------------------------------------------
                    if is_warmup:
                        # 🔥 [수정] Stage 2 히트맵을 제거하고, Stage 1의 위치 가이드(S1_HM)만 학습합니다.
                        total_loss = 1.0 * norm_hm_s1
                        total_loss_raw_reversed = loss_hm_s1
                    else:
                        if use_rlw_for_heatmap:
                            # [Option A: 4-Loss RLW] S1_HM을 포함하여 모든 로스를 자율 조율
                            # 구성: (1) S1_HM, (2) Coord, (3) Surface, (4) Structural
                            rand_w = torch.rand(4).to(device)
                            rand_w = rand_w / rand_w.sum()
                            
                            total_loss = (rand_w[0] * norm_hm_s1) + \
                                        (rand_w[1] * norm_coord) + \
                                        (rand_w[2] * norm_surface) + \
                                        (rand_w[3] * norm_struct)
                                        
                            total_loss_raw_reversed = (rand_w[0] * loss_hm_s1) + \
                                                    (rand_w[1] * loss_coord) + \
                                                    (rand_w[2] * loss_surface) + \
                                                    (rand_w[3] * loss_struct)
                        else:
                            # [Option B: 3-Loss RLW] S1_HM은 1.0 '닻'으로 고정하고 기하학만 조율 (강력 추천)
                            # 구성: (1.0 fixed) S1_HM, (Random) Coord, Surface, Structural
                            rand_w = torch.rand(3).to(device)
                            rand_w = rand_w / rand_w.sum()
                            
                            total_loss = (1.0 * norm_hm_s1) + \
                                        (rand_w[0] * norm_coord) + \
                                        (rand_w[1] * norm_surface) + \
                                        (rand_w[2] * norm_struct)
                                        
                            total_loss_raw_reversed = loss_hm_s1 + loss_coord + loss_surface + loss_struct
                    
                    loss = total_loss / accum_steps
                    loss.backward()
                    
                    if (i + 1) % accum_steps == 0:
                        opt.step()
                        opt.zero_grad() 

                    with torch.no_grad():
                        mm_error = F.l1_loss(pred_coords, augmented_landmark).item() * avg_m

                    train_loss_norm += total_loss.item()
                    train_loss_reversed += total_loss_raw_reversed.item()
                    
                    raw_hm += loss_hm_s2.item(); raw_crd += loss_coord.item(); raw_srf += loss_surface.item(); raw_str += loss_struct.item()
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
                    # test_loader는 배치 사이즈 1입니다.
                    point, landmark, seg = point.to(device), landmark.to(device), seg.to(device)
                    
                    # 🌟 [수정 1] 현재 샘플(1개)에 맞는 스케일링 변수(avg_m)를 매번 계산해야 합니다.
                    point_xyz = point[:, :, :3]
                    avg_m = torch.mean(torch.max(torch.sqrt(torch.sum((point_xyz - torch.mean(point_xyz, axis=1, keepdim=True)) ** 2, axis=2)), axis=1)[0]).item()
                    
                    # 🌟 [수정 2] '중략'된 정규화 로직을 루프 내부에서 새로 수행하여 point_input(Size 1)을 갱신합니다.
                    point_normal, landmark_normal = normalize_data(point, landmark)
                    point_input = point_normal.permute(0, 2, 1).contiguous() 
                    
                    if model_name_lower == 'deeppa_e2e':
                        # 이제 pred_coords와 s1_aux_hm은 정확히 배치 사이즈 1로 나옵니다.
                        pred_coords, s2_aux_hm, s1_aux_hm = model(point_input)
                        
                        target_seg = seg.permute(0, 2, 1).contiguous()
                        
                        # 🌟 [수정 3] Stage 2 히트맵이 None인 경우에 대한 완벽한 방어 로직
                        if s2_aux_hm is not None:
                            val_hm += (criterion(s2_aux_hm, target_seg) + criterion(s1_aux_hm, target_seg)).item() / 2
                        else:
                            # 세련된 구조(S2 HM 제거)에서는 Stage 1의 닻 로스만 검증 지표로 사용합니다. [cite: 2025-11-05]
                            val_hm += criterion(s1_aux_hm, target_seg).item()
                    else:
                        # 단독 모델(DeepPA_Wrapper 등) 대응
                        pred_coords, s2_aux_hm = model(point_input)
                        if s2_aux_hm is not None:
                            val_hm += criterion(s2_aux_hm, seg.permute(0, 2, 1).contiguous()).item()

                    # 🌟 [수정 4] mm 오차 계산 시에도 현재 루프의 landmark_normal을 사용합니다.
                    val_mm += F.l1_loss(pred_coords, landmark_normal).item() * avg_m
            
            # --- Validation 종료 ---
            # 🌟 [수정 1] 누락된 평균값 계산 로직을 추가합니다.
            # 이 줄이 있어야 하단의 print 문에서 v_mm을 인식할 수 있습니다.
            v_hm = val_hm / len(test_loader)
            v_mm = val_mm / len(test_loader)
            
            # 🌟 [수정 2] 교수님이 좋아하시는 '세련된' 출력 구조로 변경
            # Stage 2에서는 히트맵을 안 쓰므로 HM 항목을 제거하거나 0으로 표시합니다. [cite: 2025-11-05]
            print(f" [{stage_name} Ep {epoch+1:03d}] T_Norm: {t_loss_n:.2f} | T_mm: {t_mm:.2f} || V_mm: {v_mm:.2f}")
            
            # S2는 정밀화(Refinement), S1은 위치(Localization)임을 명시합니다. [cite: 2025-11-05]
            print(f"  ├─ [S2_Refinement] Crd: {t_crd_raw:.4f} | Srf: {t_srf_raw:.4f} | Str: {t_str_raw:.4f}")
            
            if model_name_lower == 'deeppa_e2e':
                # Stage 1의 닻(Anchor) 성적표
                print(f"  └─ [S1_Localization] HM(Anchor): {s1_raw_hm/num_b:.4f}")

            # 🌟 Patience Checker (Plateau 기반 Warm-up 해제)
            if is_warmup:
                if v_hm < best_warmup_hm - 1e-4:
                    best_warmup_hm = v_hm
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                print(f"  🔍 [Warm-up 감시] 현재 Val_HM: {v_hm:.4f} | Best: {best_warmup_hm:.4f} | 정체 누적: {patience_counter}/{warmup_patience}")
                
                if patience_counter >= warmup_patience:
                    is_warmup = False
                    print(f"\n{'='*60}")
                    print(f" 🚀 [전환 발동] 보조 히트맵 닻이 완전히 자리를 잡았습니다!")
                    print(f" 🚀 다음 에폭(Ep {epoch+2})부터 기하학 로스({phase_type})를 전면 개방합니다.")
                    print(f"{'='*60}\n")

            log_records.append({
                'Epoch': epoch + 1, 'Phase': current_phase_str,
                'Total_Loss_Norm': t_loss_n, 'Total_Loss_Raw_Phys': t_loss_r,
                'S2_HM_Raw': t_hm_raw, 'S2_Crd_Raw': t_crd_raw, 'S2_Str_Raw': t_str_raw,
                'S2_Srf_Unified': t_srf_raw, 'S2_P2P_Dist': s2_p2p_raw/num_b, 'S2_Curv_Err': s2_curv_raw/num_b, 'S2_Dir_Err': s2_dir_raw/num_b,
                'S1_Aux_HM_Raw': s1_raw_hm/num_b, 
                'Train_mm': t_mm, 'Val_mm': v_mm
            })
            
            with pd.ExcelWriter(excel_log_path, engine='openpyxl') as writer:
                df_log = pd.DataFrame(log_records)
                df_log.to_excel(writer, sheet_name='2_Training_Log', index=False)
                if df_calib is not None: 
                    df_calib.to_excel(writer, sheet_name='1_Calibration', index=False)

            scheduler.step()

        print(f"\n💾 [Model Save] 모든 학습 완료! 마지막 에폭({current_epochs}) 모델을 저장합니다.")
        if model_name_lower == 'deeppa_e2e':
            torch.save(model.stage1_paconv.state_dict(), os.path.join(paths['models'], 'Stage1_PAConv_last.t7'))
            torch.save(model.stage2_deeppa.state_dict(), os.path.join(paths['models'], 'Stage2_DeepPA_last.t7'))
        else:
            last_save_path = os.path.join(paths['models'], f'{stage_name}_last.t7')
            torch.save(model.state_dict(), last_save_path)
            
        return model 

    print(f"\n=== [Phase 3] Start Auto End-to-End Pipeline ===")
    
    if args.model.lower() == 'deeppa_auto':
        rlw_mode = "4-Loss(전면 랜덤)" if use_rlw_for_heatmap else "3-Loss(닻 고정)"
        print(f">>> [AUTO MODE] 🔥 256ch Raw Latent + {rlw_mode} RLW 다이렉트 회귀 훈련 가동")
        execute_stage('deeppa_e2e', args.epochs, disable_norm=not args.use_loss_norm, stage_name=f"E2E_Joint")
    else:
        execute_stage(args.model, args.epochs, disable_norm=(args.model in ['PAConv', 'PAConv_heat']), stage_name=f"Single_{args.model}")
        
    print(f"\n=== Training Finished. Results at: {paths['root']} ===")

if __name__ == "__main__":
    args = parser.parse_args()
    _init_(args)
    train(args)