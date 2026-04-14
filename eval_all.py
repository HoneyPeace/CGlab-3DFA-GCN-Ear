# @Author: Yuan Wang (Modified by Researcher & AI Assistant)
# @File: train.py
# @Description: 동적 히트맵 선학습 + 스케일링 정규화 + 전면 RLW + 🧊 PAConv Freezing (차원 오류 방어 및 로깅 100% 복원본)
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

class JointE2EModel(nn.Module):
    def __init__(self, args, landmark_num):
        super().__init__()
        self.stage1_paconv = PAConv(args, landmark_num)
        self.stage2_deeppa = DeepPA_Wrapper(args, landmark_num)

    def forward(self, x, is_frozen=False):
        if is_frozen:
            self.stage1_paconv.eval()
            with torch.no_grad():
                prior_hint = self.stage1_paconv(x)
        else:
            prior_hint = self.stage1_paconv(x)
            
        stage2_out = self.stage2_deeppa(x, prior_heatmap=prior_hint)
        if self.training: return stage2_out, prior_hint
        return stage2_out

def train(args):
    accum_steps = args.accumulation_steps
    MODE = "SPLIT" if not args.test_dataset_name or (args.train_dataset_name == args.test_dataset_name) else "SEPARATE"
    
    use_direct = getattr(args, 'use_direct_regression', False)
    is_frozen_paconv = getattr(args, 'freeze_paconv', False)
    pretrained_path = getattr(args, 'pretrained_paconv_path', '')

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

    def execute_stage(current_model_name, current_epochs, disable_norm=False, stage_name=""):
        model_name_lower = current_model_name.lower()
        is_teacher_train = ('paconv' in model_name_lower) and (model_name_lower != 'deeppa_e2e')
        
        # =========================================================================
        # 1. 모델 생성 (PAConv 티처는 무조건 히트맵 원본 모델로 생성!)
        # =========================================================================
        if model_name_lower == 'deeppa_e2e': 
            model = JointE2EModel(args, args.landmark_num).to(device)
            if is_frozen_paconv:
                print(f"\n>>> [INFO] 🧊 PAConv Freezing 활성화! 가중치 로드: {pretrained_path}")
                if os.path.exists(pretrained_path):
                    model.stage1_paconv.load_state_dict(torch.load(pretrained_path, map_location=device))
                else:
                    print(">>> [WARNING] 티처 가중치를 찾을 수 없습니다!")
                for param in model.stage1_paconv.parameters(): param.requires_grad = False
                model.stage1_paconv.eval()
        elif is_teacher_train:
            # 🌟 [차원 충돌 해결] 티처 학습 시에는 무조건 오리지널 PAConv(히트맵)를 사용합니다.
            model = PAConv(args, args.landmark_num).to(device)
            print("\n>>> [INFO] 👨‍🏫 Teacher Mode: PAConv(Heatmap) 모델을 단독 학습합니다.")
        else: 
            model = DeepPA_Wrapper(args, args.landmark_num).to(device)
            
        model.apply(weight_init)
        
        surface_criterion = CurvatureSurfaceLoss(k_p2p=args.plane_knn, k_curv=args.curv_knn, alpha=args.curv_alpha, beta=args.dir_beta).to(device)
        criterion = AdaptiveWingLoss() if args.loss == 'adaptive_wing' else torch.nn.MSELoss()
        
        opt_params = filter(lambda p: p.requires_grad, model.parameters())
        opt = optim.Adam(opt_params, lr=args.lr, eps=1e-08, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(opt, T_max=current_epochs) if args.scheduler == 'cos' else StepLR(opt, step_size=40, gamma=0.9)

        excel_log_path = os.path.join(paths['root'], f'Training_Log_{stage_name}.xlsx')
        log_records = []
        target_norm = 1.0 
        auto_scales = {'heatmap': 1.0, 'coord': 1.0, 'surface': 1.0, 'struct': 1.0}

        # 🌟 동적 선학습(Warm-up) 상태 변수 (프리징이면 강제 차단)
        is_warmup = getattr(args, 'use_warmup', True) and (model_name_lower == 'deeppa_e2e' or is_teacher_train)
        if is_frozen_paconv: is_warmup = False
            
        warmup_patience = getattr(args, 'warmup_patience', 10)
        best_warmup_hm = float('inf')
        patience_counter = 0
        df_calib = None 

        # =========================================================================
        # 🟢 [Phase 2.9] 스케일링 정규화 (Calibration)
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
                    points_for_coords = point_input[:, :3, :].permute(0, 2, 1).contiguous() 
                    
                    # 🌟 [차원 충돌 해결] 모델 타입에 따른 출력값 및 좌표 추출 분리
                    if model_name_lower == 'deeppa_e2e':
                        stage2_out, prior_hint = model(point_input, is_frozen=is_frozen_paconv)
                        pred_coords = stage2_out if use_direct else get_differentiable_coords(points_for_coords, stage2_out, k=args.k_softargmax)
                        l_hm = criterion(prior_hint, seg.permute(0, 2, 1).contiguous()).item()
                    elif is_teacher_train:
                        prior_hint = model(point_input)
                        pred_coords = get_differentiable_coords(points_for_coords, prior_hint, k=args.k_softargmax)
                        l_hm = criterion(prior_hint, seg.permute(0, 2, 1).contiguous()).item()
                    else:
                        stage2_out = model(point_input)
                        pred_coords = stage2_out if use_direct else get_differentiable_coords(points_for_coords, stage2_out, k=args.k_softargmax)
                        l_hm = 1.0 if use_direct else criterion(stage2_out, seg.permute(0, 2, 1).contiguous()).item()
                    
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
            print(f"✅ 정규화 완료 (Multiplier: HM={auto_scales['heatmap']:.2f}, Crd={auto_scales['coord']:.2f}, Srf={auto_scales['surface']:.2f})")
            torch.cuda.empty_cache()

        # =========================================================================
        # 🟢 [Phase 3] 본 학습
        # =========================================================================
        for epoch in range(current_epochs):
            if not is_frozen_paconv: 
                model.train() 
            else: 
                if hasattr(model, 'stage2_deeppa'): model.stage2_deeppa.train()
                else: model.train()
                
            train_loss_norm, train_loss_reversed = 0.0, 0.0
            
            raw_hm, raw_crd, raw_srf, raw_str = 0.0, 0.0, 0.0, 0.0
            s2_p2p_raw, s2_curv_raw, s2_dir_raw = 0.0, 0.0, 0.0
            s1_raw_hm, s1_raw_crd, s1_raw_srf, s1_raw_str = 0.0, 0.0, 0.0, 0.0
            s1_p2p_raw, s1_curv_raw, s1_dir_raw = 0.0, 0.0, 0.0 
            
            train_mm = 0.0
            opt.zero_grad() 
            
            current_phase_str = "🧊 Frozen S1 (Teacher)" if is_frozen_paconv else ("🔥 Warm-up (Heatmap Only)" if is_warmup else "🚀 Main (Full RLW)")
            
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
                    
                    # 🌟 1. 모델 포워딩 및 기본 변수 초기화
                    if model_name_lower == 'deeppa_e2e':
                        stage2_out, prior_hint = model(point_input, is_frozen=is_frozen_paconv)
                        loss_hm_s1 = criterion(prior_hint, seg.permute(0, 2, 1).contiguous())
                        norm_hm_s1 = loss_hm_s1 * auto_scales['heatmap']
                        
                        if use_direct:
                            pred_coords = stage2_out
                            loss_heatmap = torch.tensor(0.0).to(device)
                            norm_heatmap = 0.0
                        else:
                            pred_coords = get_differentiable_coords(points_for_coords, stage2_out, k=args.k_softargmax)
                            loss_heatmap = criterion(stage2_out, seg.permute(0, 2, 1).contiguous())
                            norm_heatmap = loss_heatmap * auto_scales['heatmap']

                    elif is_teacher_train:
                        prior_hint = model(point_input)
                        loss_hm_s1 = criterion(prior_hint, seg.permute(0, 2, 1).contiguous())
                        norm_hm_s1 = loss_hm_s1 * auto_scales['heatmap']
                        
                        pred_coords = get_differentiable_coords(points_for_coords, prior_hint, k=args.k_softargmax)
                        loss_heatmap = torch.tensor(0.0).to(device)
                        norm_heatmap = 0.0

                    else: # Single DeepPA
                        stage2_out = model(point_input)
                        loss_hm_s1 = torch.tensor(0.0).to(device)
                        norm_hm_s1 = 0.0
                        
                        if use_direct:
                            pred_coords = stage2_out
                            loss_heatmap = torch.tensor(0.0).to(device)
                            norm_heatmap = 0.0
                        else:
                            pred_coords = get_differentiable_coords(points_for_coords, stage2_out, k=args.k_softargmax)
                            loss_heatmap = criterion(stage2_out, seg.permute(0, 2, 1).contiguous())
                            norm_heatmap = loss_heatmap * auto_scales['heatmap']

                    # 🌟 2. 3D 기하학 로스 계산
                    loss_coord = dynamic_focal_l1_loss(pred_coords, augmented_landmark, gamma=args.focal_gamma)
                    loss_surface, p2p_s2, curv_err_s2, dir_err_s2 = surface_criterion(pred_coords, augmented_landmark, points_for_coords, disable_norm=False)
                    loss_struct = compute_structural_loss(pred_coords, augmented_landmark)
                    
                    norm_coord   = loss_coord   * auto_scales['coord']
                    norm_surface = loss_surface * auto_scales['surface']         
                    norm_struct  = loss_struct  * auto_scales['struct']

                    total_loss_stage1 = torch.tensor(0.0).to(device)
                    total_loss_stage2 = torch.tensor(0.0).to(device)

                    # 🌟 3. 로스 결합 (완벽 복원된 상세 로깅 로직 및 4-Way RLW)
                    if model_name_lower == 'deeppa_e2e':
                        if is_frozen_paconv:
                            total_loss_stage1 = torch.tensor(0.0).to(device)
                            s1_raw_hm += loss_hm_s1.item() # 모니터링용
                        else:
                            if is_warmup or use_direct:
                                total_loss_stage1 = norm_hm_s1
                                s1_raw_hm += loss_hm_s1.item()
                            else:
                                prior_coords = get_differentiable_coords(points_for_coords, prior_hint, k=args.k_softargmax)
                                loss_crd_s1 = dynamic_focal_l1_loss(prior_coords, augmented_landmark, gamma=args.focal_gamma)
                                loss_srf_s1, p2p_s1, curv_err_s1, dir_err_s1 = surface_criterion(prior_coords, augmented_landmark, points_for_coords, disable_norm=False)
                                loss_str_s1 = compute_structural_loss(prior_coords, augmented_landmark)
                                
                                rand_w_s1 = torch.rand(4).to(device); rand_w_s1 = rand_w_s1 / rand_w_s1.sum()
                                total_loss_stage1 = (rand_w_s1[0]*norm_hm_s1 + rand_w_s1[1]*(loss_crd_s1*auto_scales['coord']) + 
                                                     rand_w_s1[2]*(loss_srf_s1*auto_scales['surface']) + rand_w_s1[3]*(loss_str_s1*auto_scales['struct']))
                                
                                s1_raw_hm += loss_hm_s1.item(); s1_raw_crd += loss_crd_s1.item(); s1_raw_srf += loss_srf_s1.item(); s1_raw_str += loss_str_s1.item()
                                s1_p2p_raw += p2p_s1.item(); s1_curv_raw += curv_err_s1.item(); s1_dir_raw += dir_err_s1.item()

                        if is_warmup:
                            total_loss_stage2 = torch.tensor(0.0).to(device) if use_direct else norm_heatmap
                            total_loss_raw_reversed = loss_hm_s1 if use_direct else (loss_heatmap + loss_hm_s1)
                        else:
                            if use_direct:
                                rand_weights = torch.rand(3).to(device); rand_weights = rand_weights / rand_weights.sum()
                                total_loss_stage2 = (rand_weights[0]*norm_coord + rand_weights[1]*norm_surface + rand_weights[2]*norm_struct)
                                total_loss_raw_reversed = (rand_weights[0]*loss_coord + rand_weights[1]*loss_surface + rand_weights[2]*loss_struct)
                            else:
                                rand_weights = torch.rand(4).to(device); rand_weights = rand_weights / rand_weights.sum()
                                total_loss_stage2 = (rand_weights[0]*norm_heatmap + rand_weights[1]*norm_coord + rand_weights[2]*norm_surface + rand_weights[3]*norm_struct)
                                total_loss_raw_reversed = (rand_weights[0]*loss_heatmap + rand_weights[1]*loss_coord + rand_weights[2]*loss_surface + rand_weights[3]*loss_struct)

                    elif is_teacher_train:
                        if is_warmup:
                            total_loss_stage1 = norm_hm_s1
                            s1_raw_hm += loss_hm_s1.item()
                            total_loss_raw_reversed = loss_hm_s1
                        else:
                            rand_w_s1 = torch.rand(4).to(device); rand_w_s1 = rand_w_s1 / rand_w_s1.sum()
                            total_loss_stage1 = (rand_w_s1[0]*norm_hm_s1 + rand_w_s1[1]*norm_coord + rand_w_s1[2]*norm_surface + rand_w_s1[3]*norm_struct)
                            total_loss_raw_reversed = (rand_w_s1[0]*loss_hm_s1 + rand_w_s1[1]*loss_coord + rand_w_s1[2]*loss_surface + rand_w_s1[3]*loss_struct)

                            s1_raw_hm += loss_hm_s1.item(); s1_raw_crd += loss_coord.item(); s1_raw_srf += loss_surface.item(); s1_raw_str += loss_struct.item()
                            s1_p2p_raw += p2p_s2.item(); s1_curv_raw += curv_err_s2.item(); s1_dir_raw += dir_err_s2.item()

                    else: # Single DeepPA
                        if use_direct:
                            rand_weights = torch.rand(3).to(device); rand_weights = rand_weights / rand_weights.sum()
                            total_loss_stage2 = (rand_weights[0]*norm_coord + rand_weights[1]*norm_surface + rand_weights[2]*norm_struct)
                            total_loss_raw_reversed = (rand_weights[0]*loss_coord + rand_weights[1]*loss_surface + rand_weights[2]*loss_struct)
                        else:
                            rand_weights = torch.rand(4).to(device); rand_weights = rand_weights / rand_weights.sum()
                            total_loss_stage2 = (rand_weights[0]*norm_heatmap + rand_weights[1]*norm_coord + rand_weights[2]*norm_surface + rand_weights[3]*norm_struct)
                            total_loss_raw_reversed = (rand_weights[0]*loss_heatmap + rand_weights[1]*loss_coord + rand_weights[2]*loss_surface + rand_weights[3]*loss_struct)
                    
                    total_loss = total_loss_stage2 + total_loss_stage1
                    
                    if not is_teacher_train:
                        raw_hm += loss_heatmap.item(); raw_crd += loss_coord.item()
                        raw_srf += loss_surface.item(); raw_str += loss_struct.item()
                        s2_p2p_raw += p2p_s2.item(); s2_curv_raw += curv_err_s2.item(); s2_dir_raw += dir_err_s2.item()

                    loss = total_loss / accum_steps
                    loss.backward()
                    
                    if (i + 1) % accum_steps == 0:
                        opt.step()
                        opt.zero_grad() 

                    with torch.no_grad():
                        mm_error = F.l1_loss(pred_coords, augmented_landmark).item() * avg_m

                    train_loss_norm += total_loss.item()
                    train_loss_reversed += total_loss_raw_reversed.item()
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
                    points_for_coords = point_input[:, :3, :].permute(0, 2, 1).contiguous() 
                    
                    if model_name_lower == 'deeppa_e2e':
                        stage2_out, prior_hint = model(point_input, is_frozen=is_frozen_paconv)
                        pred_coords = stage2_out if use_direct else get_differentiable_coords(points_for_coords, stage2_out, k=args.k_softargmax)
                        val_hm += criterion(prior_hint, seg.permute(0, 2, 1).contiguous()).item()
                    elif is_teacher_train:
                        prior_hint = model(point_input)
                        pred_coords = get_differentiable_coords(points_for_coords, prior_hint, k=args.k_softargmax)
                        val_hm += criterion(prior_hint, seg.permute(0, 2, 1).contiguous()).item()
                    else:
                        stage2_out = model(point_input)
                        pred_coords = stage2_out if use_direct else get_differentiable_coords(points_for_coords, stage2_out, k=args.k_softargmax)
                        val_hm += criterion(stage2_out, seg.permute(0, 2, 1).contiguous()).item() if not use_direct else 0.0

                    val_mm += F.l1_loss(pred_coords, landmark_normal).item() * avg_m

            v_hm, v_mm = val_hm/len(test_loader), val_mm/len(test_loader)
            
            # 🌟 [완벽 복원] 연구자님이 좋아하셨던 터미널 상세 계층 로깅
            print(f" [{stage_name} Ep {epoch+1:03d}] T_Norm: {t_loss_n:.2f} | T_mm: {t_mm:.2f} || V_mm: {v_mm:.2f}")
            if not is_teacher_train:
                print(f"  ├─ [S2_DeepPA] HM: {t_hm_raw:.4f} | Crd: {t_crd_raw:.4f} | Srf(Unified): {t_srf_raw:.4f} | Str: {t_str_raw:.4f}")
            if model_name_lower == 'deeppa_e2e' or is_teacher_train:
                prefix = "└─" if not is_teacher_train else "├─"
                print(f"  {prefix} [S1_PAConv] HM: {s1_raw_hm/num_b:.4f} | Crd: {s1_raw_crd/num_b:.4f} | Srf(Unified): {s1_raw_srf/num_b:.4f} | Str: {s1_raw_str/num_b:.4f}")

            # 🌟 Patience Checker
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
                    print(f" 🚀 [전환 발동] 히트맵 학습이 한계(Plateau)에 도달했습니다!")
                    print(f" 🚀 다음 에폭(Ep {epoch+2})부터 기하학 로스(RLW)를 전면 개방합니다.")
                    print(f"{'='*60}\n")

            # 🌟 [완벽 복원] 엑셀 상세 변수 기록
            log_records.append({
                'Epoch': epoch + 1, 'Phase': current_phase_str,
                'Total_Loss_Norm': t_loss_n, 'Total_Loss_Raw_Phys': t_loss_r,
                'S2_HM_Raw': t_hm_raw, 'S2_Crd_Raw': t_crd_raw, 'S2_Str_Raw': t_str_raw,
                'S2_Srf_Unified': t_srf_raw, 'S2_P2P_Dist': s2_p2p_raw/num_b, 'S2_Curv_Err': s2_curv_raw/num_b, 'S2_Dir_Err': s2_dir_raw/num_b,
                'S1_Aux_HM_Raw': s1_raw_hm/num_b, 'S1_Aux_Crd_Raw': s1_raw_crd/num_b, 'S1_Aux_Str_Raw': s1_raw_str/num_b,
                'S1_Aux_Srf_Unified': s1_raw_srf/num_b, 'S1_Aux_P2P_Dist': s1_p2p_raw/num_b, 'S1_Aux_Curv_Err': s1_curv_raw/num_b, 'S1_Aux_Dir_Err': s1_dir_raw/num_b,
                'Train_mm': t_mm, 'Val_mm': v_mm
            })
            
            with pd.ExcelWriter(excel_log_path, engine='openpyxl') as writer:
                df_log = pd.DataFrame(log_records)
                df_log.to_excel(writer, sheet_name='2_Training_Log', index=False)
                if df_calib is not None: 
                    df_calib.to_excel(writer, sheet_name='1_Calibration', index=False)

            scheduler.step()

        # =========================================================================
        print(f"\n💾 [Model Save] 모든 학습 완료! 마지막 에폭({current_epochs}) 모델을 저장합니다.")
        if model_name_lower == 'deeppa_e2e':
            torch.save(model.stage1_paconv.state_dict(), os.path.join(paths['models'], 'Stage1_PAConv_last.t7'))
            torch.save(model.stage2_deeppa.state_dict(), os.path.join(paths['models'], 'Stage2_DeepPA_last.t7'))
        elif is_teacher_train:
            # 🌟 티처 단독 학습 시 파일명 하드코딩으로 확실히 저장
            torch.save(model.state_dict(), os.path.join(paths['models'], 'Single_PAConv_last.t7'))
        else:
            last_save_path = os.path.join(paths['models'], f'{stage_name}_last.t7')
            torch.save(model.state_dict(), last_save_path)
            
        return model 

    print(f"\n=== [Phase 3] Start Auto End-to-End Pipeline ===")
    if args.model.lower() == 'deeppa_auto':
        mode_str = "Frozen PAConv + Direct S2" if is_frozen_paconv else ("Direct Regression" if use_direct else "Heatmap Cascade")
        print(f">>> [AUTO MODE] 🔥 {mode_str} 훈련 가동")
        execute_stage('deeppa_e2e', args.epochs, disable_norm=not args.use_loss_norm, stage_name=f"E2E_Joint")
    else:
        execute_stage(args.model, args.epochs, disable_norm=not args.use_loss_norm, stage_name=f"Single_{args.model}")
        
    print(f"\n=== Training Finished. Results at: {paths['root']} ===")

if __name__ == "__main__":
    args = parser.parse_args()
    _init_(args)
    train(args)