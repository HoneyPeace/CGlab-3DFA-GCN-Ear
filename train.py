# @Author: Yuan Wang (Modified by Researcher & AI Assistant)
# @File: train.py
# @Description: Gradient Accumulation + Virtual Epoch LSB + VRAM Optimization + Excel Logging

import os
import time
import numpy as np
import pandas as pd # 🌟 엑셀 저장을 위한 pandas 추가
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

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

# ==========================================
# 🚀 모델 임포트
# ==========================================
from PAConv_model import PAConv
from DeepLA_model import DeepLA_Wrapper
from DeepPA_model import DeepPA_Wrapper  

# GPU 설정
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

    if args.accumulation_steps > 1:
        batch_str = f"{args.batch_size}x{args.accumulation_steps}"
    else:
        batch_str = f"{args.batch_size}"

    base_str = f"FPS{args.num_points}_sigma{args.sigma}_batch{batch_str}_train{train_len}"
    
    if args.user_tag and args.user_tag != "":
        setting_str = f"{base_str}_{args.user_tag}"
    else:
        setting_str = base_str
    
    count = 1
    while True:
        run_name = f"{setting_str}_{count}"
        run_dir = os.path.join(project_dir, run_name)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            break
        count += 1
    
    paths = {
        'root': run_dir,
        'models': os.path.join(run_dir, 'models'),
        'npy_backup': os.path.join(run_dir, 'npy_data'),
        'gt_heatmap': os.path.join(run_dir, 'GT_Heatmaps')
    }
    
    for k, p in paths.items():
        os.makedirs(p, exist_ok=True)
        
    print(f"\n>>> [Experiment Created]")
    print(f"    ID   : {run_name}")
    print(f"    Path : {run_dir}\n")
    
    return paths

def process_data_storage(dataset, prefix, paths):
    shape_list, landmark_list, heatmap_list = [], [], []
    for i in range(len(dataset)):
        p, l, h = dataset[i]
        shape_list.append(p.numpy())
        landmark_list.append(l.numpy())
        heatmap_list.append(h.numpy())

    shape_arr = np.stack(shape_list)
    landmark_arr = np.stack(landmark_list)
    heatmap_arr = np.stack(heatmap_list)

    np.save(os.path.join(paths['npy_backup'], f"shape_{prefix}.npy"), shape_arr)
    np.save(os.path.join(paths['npy_backup'], f"landmark_{prefix}.npy"), landmark_arr)
    np.save(os.path.join(paths['npy_backup'], f"Heat_data_{prefix}.npy"), heatmap_arr)
    print(f"   [{prefix.upper()}] Backup Saved: {paths['npy_backup']}")

# =========================================================================
# 🌟 End-to-End 조인트 모델 정의
# =========================================================================
class JointE2EModel(nn.Module):
    def __init__(self, args, landmark_num):
        super().__init__()
        self.stage1_paconv = PAConv(args, landmark_num)
        self.stage2_deeppa = DeepPA_Wrapper(args, landmark_num)

    def forward(self, x):
        prior_hint = self.stage1_paconv(x)
        pred_heatmap = self.stage2_deeppa(x, prior_heatmap=prior_hint)
        
        if self.training:
            return pred_heatmap, prior_hint
        return pred_heatmap

def train(args):
    accum_steps = args.accumulation_steps
    
    if not args.test_dataset_name or (args.train_dataset_name == args.test_dataset_name):
        MODE = "SPLIT"
        target_dataset = args.train_dataset_name
        args.test_dataset_name = args.train_dataset_name
    else:
        MODE = "SEPARATE"

    print(f">>> [Gradient Accumulation] Steps: {accum_steps}")

    if args.need_resample:
        print("=== [Phase 1] Data Generation (Initial) ===")
        main_sample(args.num_points, args.seed, args.sigma, args.sample_way, 
                    args.train_dataset_name, args.data_root, partition='train')
        main_sample(args.num_points, args.seed, args.sigma, args.sample_way, 
                    args.test_dataset_name, args.data_root, partition='test')

    print("=== [Phase 2] Loading Data ===")
    
    if MODE == "SPLIT":
        full_dataset = FaceLandmarkData(data_root=args.data_root, partition='trainval', data=args.train_dataset_name, in_channels=args.in_channels)
        train_size = int(len(full_dataset) * 0.7)
        test_size = len(full_dataset) - train_size
        torch.manual_seed(args.dataset_seed)
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    else:
        train_dataset = FaceLandmarkData(data_root=args.data_root, partition='train', data=args.train_dataset_name, in_channels=args.in_channels)
        test_dataset = FaceLandmarkData(data_root=args.data_root, partition='test', data=args.test_dataset_name, in_channels=args.in_channels)

    train_len = len(train_dataset)
    paths = get_experiment_paths(args, train_len)

    print("=== [Phase 2.6] Backing up Data ===")
    process_data_storage(train_dataset, "train", paths)
    process_data_storage(test_dataset, "test", paths)

    train_loader = DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, num_workers=0, batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    
    ScaleAndTranslate = PointcloudScaleAndTranslate()

    # =========================================================================
    # 🚀 핵심 학습 루프 함수
    # =========================================================================
    def execute_stage(current_model_name, current_epochs, disable_norm=False, prior_model=None, stage_name=""):
        print(f"\n{'='*50}")
        print(f" 🚀 [Start Stage] {stage_name} (Epochs: {current_epochs}, Norm Disable: {disable_norm})")
        print(f"{'='*50}")

        model_name_lower = current_model_name.lower()

        if model_name_lower == 'deeppa_e2e':
            model = JointE2EModel(args, args.landmark_num).to(device)
        elif model_name_lower == 'deeppa' and prior_model is None:
            paconv_prior = PAConv(args, args.landmark_num).to(device)
            best_paconv_path = os.path.join("..", "PAConv_model", "model_epoch_500.t7")
            paconv_prior.load_state_dict(torch.load(best_paconv_path))
            paconv_prior.eval()
            for param in paconv_prior.parameters():
                param.requires_grad = False 
            prior_model = paconv_prior
            model = DeepPA_Wrapper(args, args.landmark_num).to(device)
        elif model_name_lower in ['paconv', 'paconv_heat']:
            model = PAConv(args, args.landmark_num).to(device)
        elif model_name_lower == 'deepla':
            model = DeepLA_Wrapper(args, args.landmark_num).to(device)
        elif model_name_lower == 'deeppa':
            model = DeepPA_Wrapper(args, args.landmark_num).to(device)
            
        model.apply(weight_init)
        
        surface_criterion = CurvatureSurfaceLoss(k_p2p=args.plane_knn, k_curv=args.curv_knn, alpha=args.curv_alpha, dir_weight=args.dir_weight).to(device)
        criterion = AdaptiveWingLoss() if args.loss == 'adaptive_wing' else torch.nn.MSELoss()
            
        opt = optim.Adam(model.parameters(), lr=args.lr, eps=1e-08, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(opt, T_max=current_epochs) if args.scheduler == 'cos' else StepLR(opt, step_size=40, gamma=0.9)

        # 🌟 엑셀 로그 데이터 저장용 리스트
        excel_log_path = os.path.join(paths['root'], f'Training_Log_{stage_name}.xlsx')
        log_records = []

        best_val_mm = float('inf')
        target_norm = 1.0 
        auto_scales = {'heatmap': -1.0, 'coord': -1.0, 'surface': -1.0, 'struct': -1.0}

        # =========================================================================
        # 🟢 [Phase 2.9] 가상 에폭: 전체 평균 vs 첫 배치 스케일 비교 및 기록
        # =========================================================================
        if not disable_norm and model_name_lower not in ['paconv', 'paconv_heat']:
            print(f"\n🔍 [Scale Calibration] 스케일 정밀 분석 중... (First Batch vs Full Average)")
            model.eval()
            
            # 비교용 변수
            single_batch_losses = {}
            sum_losses = {'heatmap': 0.0, 'coord': 0.0, 'surface': 0.0, 'struct': 0.0}
            
            with torch.no_grad():
                for i, (point, landmark, seg) in enumerate(tqdm(train_loader, desc="Calibrating", leave=False)):
                    # 데이터 전처리 (실제 학습과 동일하게 진행)
                    point, landmark, seg = point.to(device), landmark.to(device), seg.to(device)
                    point_normal, landmark_normal = normalize_data(point, landmark)
                    point_normal, augmented_landmark = ScaleAndTranslate(point_normal, landmark_normal)
                    point_input = point_normal.permute(0, 2, 1).contiguous()
                    
                    if model_name_lower == 'deeppa_e2e':
                        pred_heatmap = model(point_input)
                    elif current_model_name == 'DeepPA' and prior_model is not None:
                        prior_hint = prior_model(point_input)
                        pred_heatmap = model(point_input, prior_heatmap=prior_hint)
                    else:
                        pred_heatmap = model(point_input)
                    
                    points_for_coords = point_input[:, :3, :].permute(0, 2, 1).contiguous() 
                    pred_coords = get_differentiable_coords(points_for_coords, pred_heatmap, k=args.k_softargmax)
                    
                    # 로스 계산
                    l_hm = criterion(pred_heatmap, seg.permute(0, 2, 1).contiguous()).item()
                    l_crd = dynamic_focal_l1_loss(pred_coords, augmented_landmark, gamma=args.focal_gamma).item()
                    l_srf = surface_criterion(pred_coords, augmented_landmark, points_for_coords, disable_norm=disable_norm).item()
                    l_str = compute_structural_loss(pred_coords, augmented_landmark).item()

                    # 🌟 [첫 번째 배치 데이터 기록]
                    if i == 0:
                        single_batch_losses = {'heatmap': l_hm, 'coord': l_crd, 'surface': l_srf, 'struct': l_str}

                    # 🌟 [전체 누적]
                    sum_losses['heatmap'] += l_hm
                    sum_losses['coord'] += l_crd
                    sum_losses['surface'] += l_srf
                    sum_losses['struct'] += l_str

            # 최종 평균 계산
            num_batches = len(train_loader)
            avg_losses = {k: v / num_batches for k, v in sum_losses.items()}

            # 배수(Multiplier) 계산
            auto_scales_single = {k: target_norm / (v + 1e-6) for k, v in single_batch_losses.items()}
            auto_scales_full = {k: target_norm / (v + 1e-6) for k, v in avg_losses.items()}
            
            # 🔥 [실제 학습에는 '전체 평균' 적용]
            auto_scales = auto_scales_full 

            # 🌟 [엑셀 기록용 비교 데이터 준비]
            calibration_report = {
                "Metric": ["Raw_Loss (1 Batch)", "Raw_Loss (Full Avg)", "Multiplier (Based on 1nd)", "Multiplier (Based on Full)"],
                "Heatmap": [single_batch_losses['heatmap'], avg_losses['heatmap'], auto_scales_single['heatmap'], auto_scales_full['heatmap']],
                "Coord": [single_batch_losses['coord'], avg_losses['coord'], auto_scales_single['coord'], auto_scales_full['coord']],
                "Surface": [single_batch_losses['surface'], avg_losses['surface'], auto_scales_single['surface'], auto_scales_full['surface']],
                "Structure": [single_batch_losses['struct'], avg_losses['struct'], auto_scales_single['struct'], auto_scales_full['struct']]
            }
            df_calib = pd.DataFrame(calibration_report)
            print("\n📊 [Scale Comparison Result]")
            print(df_calib.to_string(index=False))

            torch.cuda.empty_cache()
            print(" 🧹 VRAM 캐시 초기화 완료. 본 학습을 시작합니다.\n")
        else:
            auto_scales['heatmap'], auto_scales['coord'], auto_scales['surface'], auto_scales['struct'] = 1.0, 1.0, 1.0, 1.0

        # =========================================================================
        # 🟢 본 학습 (Real Epoch)
        # =========================================================================
        for epoch in range(current_epochs):
            model.train()
            
            # 누적 변수 세팅
            train_loss_norm, train_loss_reversed = 0.0, 0.0
            raw_hm, raw_crd, raw_srf, raw_str = 0.0, 0.0, 0.0, 0.0
            norm_hm_sum, norm_crd_sum, norm_srf_sum, norm_str_sum = 0.0, 0.0, 0.0, 0.0
            train_mm = 0.0
            
            opt.zero_grad() 
            
            with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"{stage_name} Ep {epoch+1:03d}/{current_epochs} [Train]", unit="batch", leave=False) as tepoch:
                for i, (point, landmark, seg) in tepoch:
                    point, landmark, seg = point.to(device), landmark.to(device), seg.to(device)

                    with torch.no_grad():
                        point_xyz = point[:, :, :3]
                        centroid_tmp = torch.mean(point_xyz, axis=1, keepdim=True)
                        batch_m = torch.max(torch.sqrt(torch.sum((point_xyz - centroid_tmp) ** 2, axis=2)), axis=1)[0]
                        avg_m = torch.mean(batch_m).item()

                    point_normal, landmark_normal = normalize_data(point, landmark)
                    point_normal, augmented_landmark = ScaleAndTranslate(point_normal, landmark_normal)
                    point_input = point_normal.permute(0, 2, 1).contiguous()
                    
                    if model_name_lower == 'deeppa_e2e':
                        pred_heatmap, prior_hint = model(point_input)
                        loss_hm_stage1 = criterion(prior_hint, seg.permute(0, 2, 1).contiguous())
                    elif current_model_name == 'DeepPA' and prior_model is not None:
                        with torch.no_grad():
                            prior_hint = prior_model(point_input)
                        pred_heatmap = model(point_input, prior_heatmap=prior_hint)
                    else:
                        pred_heatmap = model(point_input)
                    
                    points_for_coords = point_input[:, :3, :].permute(0, 2, 1).contiguous() 
                    pred_coords = get_differentiable_coords(points_for_coords, pred_heatmap, k=args.k_softargmax)
                    
                    # 🌟 Raw Loss 계산
                    loss_heatmap = criterion(pred_heatmap, seg.permute(0, 2, 1).contiguous())
                    loss_coord = dynamic_focal_l1_loss(pred_coords, augmented_landmark, gamma=args.focal_gamma)
                    loss_surface = surface_criterion(pred_coords, augmented_landmark, points_for_coords, disable_norm=disable_norm)
                    loss_struct = compute_structural_loss(pred_coords, augmented_landmark)
                    
                    # 🌟 Norm Loss 계산
                    norm_heatmap = loss_heatmap * auto_scales['heatmap']
                    norm_coord   = loss_coord   * auto_scales['coord']
                    norm_surface = loss_surface * auto_scales['surface']         
                    norm_struct  = loss_struct  * auto_scales['struct']

                    if model_name_lower == 'paconv_heat':
                        weights = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
                        total_loss = norm_heatmap
                        total_loss_raw_reversed = loss_heatmap # 역산
                    else:
                        stage1_epochs = current_epochs // 5 
                        if epoch < stage1_epochs:
                            weights = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
                            total_loss = norm_heatmap
                            total_loss_raw_reversed = loss_heatmap
                        else:
                            rand_weights = torch.rand(3).to(device)
                            rand_weights = (rand_weights / rand_weights.sum()) * 0.95
                            
                            # 🌟 실제 훈련용(Norm) Total Loss
                            total_loss = (0.05 * norm_heatmap + 
                                          rand_weights[0] * norm_coord + 
                                          rand_weights[1] * norm_surface + 
                                          rand_weights[2] * norm_struct)
                                          
                            # 🌟 역산용(Reversed) Total Loss: 원본 크기에 가중치만 반영한 가상의 물리적 로스
                            total_loss_raw_reversed = (0.05 * loss_heatmap + 
                                                       rand_weights[0] * loss_coord + 
                                                       rand_weights[1] * loss_surface + 
                                                       rand_weights[2] * loss_struct)
                                                       
                            weights = torch.tensor([0.05, rand_weights[0], rand_weights[1], rand_weights[2]])
                    
                    if model_name_lower == 'deeppa_e2e':
                        total_loss = total_loss + loss_hm_stage1

                    loss = total_loss / accum_steps
                    loss.backward()
                    
                    if (i + 1) % accum_steps == 0:
                        opt.step()
                        opt.zero_grad() 

                    with torch.no_grad():
                        true_l1 = F.l1_loss(pred_coords, augmented_landmark).item()
                        mm_error = true_l1 * avg_m

                    # 변수 누적
                    train_loss_norm += total_loss.item()
                    train_loss_reversed += total_loss_raw_reversed.item()
                    
                    raw_hm += loss_heatmap.item()
                    raw_crd += loss_coord.item()
                    raw_srf += loss_surface.item()
                    raw_str += loss_struct.item()
                    
                    norm_hm_sum += norm_heatmap.item()
                    norm_crd_sum += norm_coord.item()
                    norm_srf_sum += norm_surface.item()
                    norm_str_sum += norm_struct.item()
                    train_mm += mm_error
                    
                    if torch.cuda.is_available():
                        vram_used = torch.cuda.max_memory_allocated() / (1024 ** 3)
                        vram_str = f"{vram_used:.1f}GB"
                    else:
                        vram_str = "CPU"
                    
                    tepoch.set_postfix(Loss=f"{total_loss.item():.4f}", HM=f"{loss_heatmap.item():.4f}", mm=f"{mm_error:.2f}", VRAM=vram_str)

            num_b = len(train_loader)
            t_loss_n = train_loss_norm / num_b
            t_loss_r = train_loss_reversed / num_b
            
            t_hm_raw, t_crd_raw, t_srf_raw, t_str_raw = raw_hm/num_b, raw_crd/num_b, raw_srf/num_b, raw_str/num_b
            t_hm_n, t_crd_n, t_srf_n, t_str_n = norm_hm_sum/num_b, norm_crd_sum/num_b, norm_srf_sum/num_b, norm_str_sum/num_b
            t_mm = train_mm / num_b

            # --- Validation ---
            model.eval()
            val_loss, val_hm, val_mm = 0.0, 0.0, 0.0
            val_sample_count = 0 
            
            with torch.no_grad():
                for i, (point, landmark, seg) in enumerate(test_loader):
                    point, landmark, seg = point.to(device), landmark.to(device), seg.to(device)

                    point_xyz = point[:, :, :3]
                    centroid_tmp = torch.mean(point_xyz, axis=1, keepdim=True)
                    batch_m = torch.max(torch.sqrt(torch.sum((point_xyz - centroid_tmp) ** 2, axis=2)), axis=1)[0]
                    avg_m = torch.mean(batch_m).item()

                    point_normal, landmark_normal = normalize_data(point, landmark)
                    point_input = point_normal.permute(0, 2, 1).contiguous()
                    
                    if model_name_lower == 'deeppa_e2e':
                        pred_heatmap = model(point_input)
                    elif current_model_name == 'DeepPA' and prior_model is not None:
                        prior_hint = prior_model(point_input) 
                        pred_heatmap = model(point_input, prior_heatmap=prior_hint)
                    else:
                        pred_heatmap = model(point_input)
                    
                    points_for_coords = point_input[:, :3, :].permute(0, 2, 1).contiguous() 
                    pred_coords = get_differentiable_coords(points_for_coords, pred_heatmap, k=args.k_softargmax)

                    loss_heatmap = criterion(pred_heatmap, seg.permute(0, 2, 1).contiguous())
                    true_l1 = F.l1_loss(pred_coords, landmark_normal).item()
                    mm_error = true_l1 * avg_m
                    
                    val_hm += loss_heatmap.item()
                    val_mm += mm_error
                    
                    val_sample_count += point.size(0)
                    if val_sample_count >= 30: break

            num_val_batches = i + 1
            v_hm, v_mm = val_hm/num_val_batches, val_mm/num_val_batches

            print(f" [{stage_name} Ep {epoch+1:03d}] Total Norm: {t_loss_n:.4f} | Raw HM: {t_hm_raw:.4f} | T_mm: {t_mm:.2f} || V_mm: {v_mm:.2f}")

            # 🌟 엑셀용 데이터 기록 (Raw vs Norm 완벽 분리)
            w_np = weights.detach().cpu().numpy()
            log_records.append({
                'Epoch': epoch + 1,
                'Stage': stage_name,
                'Total_Loss_Norm': t_loss_n,
                'Total_Loss_Reversed (Raw Physical)': t_loss_r,
                'HM_Raw': t_hm_raw, 'HM_Norm': t_hm_n, 'W_HM': w_np[0],
                'Crd_Raw': t_crd_raw, 'Crd_Norm': t_crd_n, 'W_Crd': w_np[1],
                'Srf_Raw': t_srf_raw, 'Srf_Norm': t_srf_n, 'W_Srf': w_np[2],
                'Str_Raw': t_str_raw, 'Str_Norm': t_str_n, 'W_Str': w_np[3],
                'Train_mm': t_mm,
                'Val_mm': v_mm
            })
            
            # 매 에폭마다 엑셀 덮어쓰기 저장 (중간에 꺼져도 기록 유지)
            pd.DataFrame(log_records).to_excel(excel_log_path, index=False)

            if v_mm < best_val_mm:
                best_val_mm = v_mm
                if model_name_lower == 'deeppa_e2e':
                    torch.save(model.stage1_paconv.state_dict(), os.path.join(paths['models'], 'Stage1_PAConv_model_best.t7'))
                    torch.save(model.stage2_deeppa.state_dict(), os.path.join(paths['models'], 'Stage2_DeepPA_model_best.t7'))
                else:
                    best_save_path = os.path.join(paths['models'], f'{stage_name}_model_best.t7')
                    torch.save(model.state_dict(), best_save_path)

            if (epoch + 1) % 10 == 0:
                if model_name_lower == 'deeppa_e2e':
                    torch.save(model.stage1_paconv.state_dict(), os.path.join(paths['models'], f'Stage1_PAConv_epoch_{epoch+1}.t7'))
                    torch.save(model.stage2_deeppa.state_dict(), os.path.join(paths['models'], f'Stage2_DeepPA_epoch_{epoch+1}.t7'))
                else:
                    filename = f'{stage_name}_model_epoch_{epoch+1}.t7'
                    save_path = os.path.join(paths['models'], filename)
                    torch.save(model.state_dict(), save_path)

            scheduler.step()
            
        return model 

    print(f"\n=== [Phase 3] Start Training with Curriculum Learning ===")
    
    model_type = args.model.lower() 
    
    if model_type == 'deeppa_auto':
        print(f">>> [AUTO MODE] 🔥 자동 End-to-End Joint 파이프라인 가동 (Loss Norm: {'ON' if args.use_loss_norm else 'OFF'})")
        execute_stage(current_model_name='deeppa_e2e', current_epochs=args.epochs, disable_norm=not args.use_loss_norm, prior_model=None, stage_name=f"E2E_Joint_Norm_{'ON' if args.use_loss_norm else 'OFF'}")
    else:
        disable_norm_flag = (args.model in ['PAConv', 'PAConv_heat'])
        execute_stage(current_model_name=args.model, current_epochs=args.epochs, disable_norm=disable_norm_flag, prior_model=None, stage_name=f"Single_{args.model}")
        
    print(f"\n=== Training Finished. Results at: {paths['root']} ===")

if __name__ == "__main__":
    args = parser.parse_args()
    _init_(args)
    train(args)