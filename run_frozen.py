# @Author: Yuan Wang (Modified by Researcher Park Pyeong-hwa & AI Assistant)
# @File: train_all_integrated.py
# @Description: All Pipelines + [CRITICAL FIX] Main_HM / Aux_HM Loss 완벽 분리 적용
# ==============================================================================

import os
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import shutil
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

from init import _init_
from My_args import parser
from dataset import FaceLandmarkData
from loss import AdaptiveWingLoss, CurvatureSurfaceLoss, compute_structural_loss, focal_l1_loss, get_differentiable_coords
from loss import DeepPA_HierarchicalHeatmapLoss 
from util import main_sample
from augmentations import normalize_data, PointcloudScaleAndTranslate, PointcloudJitter

from PAConv_model import PAConv
from DeepPA_model import DeepPA_Wrapper  

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None: torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
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
            os.makedirs(run_dir); break
        count += 1
    
    paths = {'root': run_dir, 'models': os.path.join(run_dir, 'models'), 'npy_backup': os.path.join(run_dir, 'npy_data')}
    for p in paths.values(): os.makedirs(p, exist_ok=True)
    return paths

def process_data_storage(dataset, prefix, paths):
    shape_list, landmark_list, heatmap_list = zip(*[(p.numpy(), l.numpy(), h.numpy()) for p, l, h in dataset])
    np.save(os.path.join(paths['npy_backup'], f"shape_{prefix}.npy"), np.stack(shape_list))
    np.save(os.path.join(paths['npy_backup'], f"landmark_{prefix}.npy"), np.stack(landmark_list))
    np.save(os.path.join(paths['npy_backup'], f"Heat_data_{prefix}.npy"), np.stack(heatmap_list))

class UniversalPipeline(nn.Module):
    def __init__(self, args, landmark_num, mode='single_paconv'):
        super().__init__()
        self.mode = mode.lower()
        self.args = args
        self.landmark_num = landmark_num
        
        if self.mode in ['frozen', 'finetune', 'e2e']:
            self.stage1_paconv = PAConv(args, landmark_num)
            if self.mode == 'frozen':
                for param in self.stage1_paconv.parameters(): param.requires_grad = False
            self.stage2_deeppa = DeepPA_Wrapper(args, landmark_num)
            
        elif self.mode in ['single_paconv', 'single_paconv_heat']: 
            self.model = PAConv(args, landmark_num)
            
        elif self.mode in ['single_deepla', 'single_deeppa']:
            self.model = DeepPA_Wrapper(args, landmark_num)

    def forward(self, x):   
        points_norm_xyz = x[:, :3, :].permute(0, 2, 1).contiguous()
        k_val = getattr(self.args, 'regression_point_num', 10)

        if self.mode in ['single_paconv', 'single_paconv_heat']: 
            multi_scale_hints, hm_raw = self.model(x)
            pred_coords = get_differentiable_coords(points_norm_xyz, hm_raw, k=k_val)
            return pred_coords, [], hm_raw
            
        elif self.mode in ['single_deepla', 'single_deeppa']:
            out = self.model(x) 
            pred_coords = out[0]
            sem_list = out[2] if len(out) > 2 else [] 
            main_hm = sem_list[-1] if len(sem_list) > 0 else None
            return pred_coords, sem_list, main_hm
            
        elif self.mode in ['frozen', 'finetune', 'e2e']:
            if self.mode == 'frozen':
                self.stage1_paconv.eval()
                with torch.no_grad(): 
                    multi_scale_hints, s1_hm_raw = self.stage1_paconv(x)
            else:
                multi_scale_hints, s1_hm_raw = self.stage1_paconv(x)
                
            out = self.stage2_deeppa(x, prior_hints=multi_scale_hints)
            pred_coords = out[0]
            sem_list = out[2] if len(out) > 2 else []
            return pred_coords, sem_list, s1_hm_raw
        
def train(args):
    accum_steps = args.accumulation_steps

    if args.need_resample:
        main_sample(args.num_points, args.seed, args.sigma, args.sample_way, args.train_dataset_name, args.data_root, partition='train')
        main_sample(args.num_points, args.seed, args.sigma, args.sample_way, args.test_dataset_name, args.data_root, partition='test')

    print(f">> [INFO] Loading Separate Datasets: {args.train_dataset_name} (train) & {args.test_dataset_name} (test)")
    
    train_dataset = FaceLandmarkData(data_root=args.data_root, partition='train', data=args.train_dataset_name, in_channels=args.in_channels)
    test_dataset = FaceLandmarkData(data_root=args.data_root, partition='test', data=args.test_dataset_name, in_channels=args.in_channels)

    paths = get_experiment_paths(args, len(train_dataset))
    process_data_storage(train_dataset, "train", paths)
    process_data_storage(test_dataset, "test", paths)

    train_loader = DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, num_workers=0, batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    
    ScaleAndTranslate = PointcloudScaleAndTranslate()
    ApplyJitter = PointcloudJitter(std=0.001)
    
    def execute_stage(current_model_name, current_epochs, stage_name=""):
        model_name_lower = current_model_name.lower()
        
        if model_name_lower == 'paconv': pipeline_mode = 'single_paconv'
        elif model_name_lower == 'paconv_heat': pipeline_mode = 'single_paconv_heat' 
        elif model_name_lower == 'deeppa_frozen_no_heat': pipeline_mode = 'deeppa_frozen_no_heat' 
        elif model_name_lower == 'deeppa_finetune': pipeline_mode = 'finetune'
        elif model_name_lower == 'deeppa_e2e': pipeline_mode = 'e2e'
        elif model_name_lower == 'deeppa': pipeline_mode = 'single_deeppa'
        elif model_name_lower in ['deepla_ori', 'deepla_all', 'deepla_all_tied', 'deepla_progress']: 
            pipeline_mode = 'single_deepla'
        elif model_name_lower in ['deeppa_frozen', 'frozen_aux_drop', 'frozen_aux_fixed', 'frozen_no_aux']:
            pipeline_mode = 'frozen'
        else: raise ValueError(f"Unknown model routing: {model_name_lower}")
            
        model = UniversalPipeline(args, args.landmark_num, mode=pipeline_mode).to(device)
        model.apply(weight_init)
        
        if pipeline_mode in ['frozen', 'finetune']:
            original_paconv_path = os.path.join(args.output_root, "PAConv_Pretrained", "models", "Single_PAConv_last.t7")
            backup_paconv_path = os.path.join(paths['models'], "Backup_Pretrained_PAConv.t7")
            if os.path.exists(original_paconv_path):
                shutil.copy2(original_paconv_path, backup_paconv_path)
                print(f"📦 [Pretrained] Coarse Anchor용 PAConv 로드 완료!")
                model.stage1_paconv.load_state_dict(torch.load(backup_paconv_path, map_location=device), strict=False)
            
        surface_criterion = CurvatureSurfaceLoss(k_p2p=args.plane_knn, k_curv=args.curv_knn, alpha=args.curv_alpha, beta=args.dir_beta).to(device)
        hm_criterion = AdaptiveWingLoss().to(device) 
        hierarchical_hm_loss = DeepPA_HierarchicalHeatmapLoss(hm_criterion).to(device)
        
        opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, eps=1e-08, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(opt, T_max=current_epochs) if getattr(args, 'scheduler', 'cos') == 'cos' else StepLR(opt, step_size=40, gamma=0.9)

        excel_log_path = os.path.join(paths['root'], f'Training_Log_{stage_name}.xlsx')
        log_records = []
        best_val_mm = float('inf')
        
        use_loss_norm = getattr(args, 'use_loss_norm', False)
        target_norm = getattr(args, 'target_norm', 1.0)
        auto_scales = {'coord': 1.0, 'surface': 1.0, 'struct': 1.0}
        
        rho = getattr(args, 'yield_factor', 0.9) 
        lambda_anchor = getattr(args, 'lambda_anchor', 0.1) 
        patience = getattr(args, 'patience', 5)
        # 🌟 args에 설정된 Aux(HDS) 기본 버퍼값을 가져옵니다. (기본값 0.3)
        base_hds_buffer = getattr(args, 'hds_buffer', 0.3)

        weight_PA, weight_DP = 1.0, 0.0   
        stagnation_counter = 0
        stagnation_counter_val = 0
        val_decay = 1.0  

        for epoch in range(current_epochs):
            model.train() 
            if pipeline_mode == 'frozen': model.stage1_paconv.eval()
            train_loss_norm, train_mm = 0.0, 0.0
            t_hm_PA, t_hm_main, t_hm_aux, t_crd, t_srf, t_str = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            avg_w_geom = 0.0
            w_hds_current = 0.0 
            opt.zero_grad() 
            
            with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"{stage_name} Ep {epoch+1:03d}", unit="batch", leave=False) as tepoch:
                for i, (point, landmark, seg) in tepoch:
                    point, landmark, seg = point.to(device), landmark.to(device), seg.to(device)

                    with torch.no_grad():
                        point_xyz = point[:, :, :3]
                        avg_m = torch.mean(torch.max(torch.sqrt(torch.sum((point_xyz - torch.mean(point_xyz, axis=1, keepdim=True)) ** 2, axis=2)), axis=1)[0]).item()

                    point_normal, landmark_normal = normalize_data(point, landmark)
                    point_normal, augmented_landmark = ScaleAndTranslate(point_normal, landmark_normal)
                    if getattr(args, 'use_jitter', False): point_normal = ApplyJitter(point_normal)
                    
                    point_input = point_normal.permute(0, 2, 1).contiguous()
                    points_for_coords = point_input[:, :3, :].permute(0, 2, 1).contiguous() 
                    target_hm = seg.permute(0, 2, 1).contiguous()
                    
                    pred_coords, sem_list, main_hm_or_paconv_hm = model(point_input)

                    L_sem_PA = torch.tensor(0.0).to(device)
                    if pipeline_mode in ['single_paconv', 'single_paconv_heat', 'frozen', 'finetune', 'e2e'] and main_hm_or_paconv_hm is not None:
                        L_sem_PA = hm_criterion(main_hm_or_paconv_hm, target_hm)
                    
                    # 🌟 [치명적 버그 수정] L_sem_DP는 영원히 삭제합니다.
                    L_main_hm, L_aux_hm = hierarchical_hm_loss(sem_list, target_hm)

                    loss_coord = focal_l1_loss(pred_coords, augmented_landmark, gamma=args.focal_gamma)
                    loss_surface, _, _, _ = surface_criterion(pred_coords, augmented_landmark, points_for_coords, disable_norm=False)
                    loss_struct = compute_structural_loss(pred_coords, augmented_landmark)
                    
                    force_geom_calc = (use_loss_norm and epoch == 0 and i == 0)
                    if force_geom_calc:
                        auto_scales['coord'] = target_norm / loss_coord.item() if loss_coord.item() > 0 else 1.0
                        auto_scales['surface'] = target_norm / loss_surface.item() if loss_surface.item() > 0 else 1.0
                        auto_scales['struct'] = target_norm / loss_struct.item() if loss_struct.item() > 0 else 1.0

                    L_pred = (loss_coord * auto_scales['coord']) + (loss_surface * auto_scales['surface']) + (loss_struct * auto_scales['struct'])
                    
                    w_geom_effective = 1.0 
                    w_main = 1.0

                    # =================================================================
                    # 🚀 Single DeepPA / Frozen 환경 맞춤형 로스 및 가중치 제어 (완벽 분리)
                    # =================================================================
                    if pipeline_mode in ['single_deeppa', 'frozen']:
                        warmup_limit = 10
                        w_geom_effective = min(1.0, (epoch / warmup_limit) ** 2) if warmup_limit > 0 else 1.0
                        w_main = 1.0 - (rho * w_geom_effective)
                        
                        if model_name_lower in ['single_deeppa', 'deeppa_frozen']:
                            # 🌟 메인 히트맵과 Aux 히트맵을 분리하고, Aux에는 hds_buffer(예: 0.3) 비중만 줍니다.
                            w_hds_current = base_hds_buffer
                            total_loss = (w_main * L_main_hm) + (w_hds_current * L_aux_hm) + (w_geom_effective * L_pred)
                            
                        elif model_name_lower == 'frozen_aux_drop':
                            w_hds_current = max(0.0, 1.0 - (epoch / 15.0))
                            total_loss = (w_main * L_main_hm) + (w_hds_current * L_aux_hm) + (w_geom_effective * L_pred)
                            
                        elif model_name_lower == 'frozen_aux_fixed':
                            w_hds_current = 0.1
                            total_loss = (w_main * L_main_hm) + (w_hds_current * L_aux_hm) + (w_geom_effective * L_pred)
                            
                        elif model_name_lower == 'frozen_no_aux':
                            w_hds_current = 0.0
                            total_loss = (w_main * L_main_hm) + (w_geom_effective * L_pred)

                    # =================================================================
                    # 🚀 DeepLA 단독 4종 Ablation
                    # =================================================================
                    elif pipeline_mode == 'single_deepla':
                        w_main = val_decay
                        w_geom_effective = 1.0 - val_decay 
                        
                        if model_name_lower == 'deepla_ori':
                            w_hds_current = val_decay
                            total_loss = (w_hds_current * L_aux_hm) + (w_geom_effective * L_pred)
                        elif model_name_lower == 'deepla_all':
                            w_hds_current = base_hds_buffer * (1.0 / (epoch + 1.0))
                            total_loss = (w_hds_current * L_aux_hm) + (w_main * L_main_hm) + (w_geom_effective * L_pred)
                        elif model_name_lower == 'deepla_all_tied':
                            w_hds_current = base_hds_buffer * val_decay
                            total_loss = (w_hds_current * L_aux_hm) + (w_main * L_main_hm) + (w_geom_effective * L_pred)
                        elif model_name_lower == 'deepla_progress':
                            w_hds_current = 0.0
                            total_loss = (w_main * L_main_hm) + (w_geom_effective * L_pred)

                    # =================================================================
                    # [기타 파이프라인]
                    # =================================================================
                    elif pipeline_mode == 'single_paconv_heat': 
                        w_geom_effective = 0.0
                        total_loss = L_sem_PA
                    elif pipeline_mode == 'single_paconv':
                        total_loss = ((1.0 - rho) * L_sem_PA) + (w_geom_effective * L_pred)
                    elif pipeline_mode == 'deeppa_frozen_no_heat':
                        total_loss = L_pred 
                    elif pipeline_mode == 'finetune':
                        w_main = 1.0 - rho
                        w_hds_current = base_hds_buffer
                        total_loss = (lambda_anchor * L_sem_PA) + (w_main * L_main_hm) + (w_hds_current * L_aux_hm) + (w_geom_effective * L_pred)
                    elif pipeline_mode == 'e2e':
                        progress = epoch / current_epochs
                        if weight_DP > weight_PA or ((weight_DP >= 0.9) and (stagnation_counter >= 1)) or (progress >= 0.5):
                            w_geom_effective = (weight_DP ** 2) 
                        else: w_geom_effective = 0.0
                        w_main = 1.0 - rho * w_geom_effective
                        w_hds_current = base_hds_buffer
                        total_loss = w_main * (weight_PA * L_sem_PA + weight_DP * L_main_hm) + (w_hds_current * L_aux_hm) + (w_geom_effective * L_pred)
                                        
                    loss = total_loss / accum_steps
                    loss.backward()
                    
                    if (i + 1) % accum_steps == 0:
                        opt.step(); opt.zero_grad() 

                    with torch.no_grad():
                        mm_error = F.l1_loss(pred_coords, augmented_landmark).item() * avg_m

                    train_loss_norm += total_loss.item()
                    t_hm_PA += L_sem_PA.item(); t_hm_main += L_main_hm.item(); t_hm_aux += L_aux_hm.item()
                    t_crd += loss_coord.item(); t_srf += loss_surface.item(); t_str += loss_struct.item()
                    train_mm += mm_error
                    avg_w_geom += w_geom_effective
                    
                    tepoch.set_postfix(Loss=f"{total_loss.item():.4f}", AuxW=f"{w_hds_current:.2f}")

            num_b = len(train_loader)
            t_loss_n, t_mm, avg_w_geom = train_loss_norm / num_b, train_mm / num_b, avg_w_geom / num_b

            # ----------------------------------------------------
            # 🌟 [Validation]
            # ----------------------------------------------------
            model.eval()
            val_mm_total, val_samples = 0.0, 0
            with torch.no_grad():
                for point, landmark, seg in test_loader:
                    point, landmark, seg = point.to(device), landmark.to(device), seg.to(device)
                    B_val = point.size(0) 
                    point_xyz = point[:, :, :3]
                    avg_m = torch.mean(torch.max(torch.sqrt(torch.sum((point_xyz - torch.mean(point_xyz, axis=1, keepdim=True)) ** 2, axis=2)), axis=1)[0]).item()
                    point_normal, landmark_normal = normalize_data(point, landmark)
                    point_input = point_normal.permute(0, 2, 1).contiguous() 
                    pred_coords = model(point_input)[0]
                    val_mm_total += F.l1_loss(pred_coords, landmark_normal.view_as(pred_coords)).item() * avg_m * B_val
                    val_samples += B_val
            
            v_mm = val_mm_total / val_samples if val_samples > 0 else 0.0                     

            # ========================================================
            # 🔄 [가중치 전환 스케줄러]
            # ========================================================
            if v_mm < best_val_mm:
                best_val_mm = v_mm
                stagnation_counter = 0; stagnation_counter_val = 0 
            else:
                stagnation_counter += 1; stagnation_counter_val += 1 
            
            if pipeline_mode == 'e2e' and stagnation_counter >= patience:
                weight_PA = max(0.1, weight_PA - 0.05) 
                weight_DP = 1.0 - weight_PA
                stagnation_counter = 0
                print(f"\n🔄 [E2E 교대] 모델 정체 감지! PAConv({weight_PA:.2f}) -> DeepPA({weight_DP:.2f})\n")
            
            if pipeline_mode == 'single_deepla' and stagnation_counter_val >= patience:
                val_decay = max(0.1, val_decay - 0.05) 
                stagnation_counter_val = 0
                print(f"\n🔄 [Val-Driven 교대] 정체 감지! Main_HM({val_decay:.2f}) -> Geom_Loss({1.0 - val_decay:.2f}) 비중 이동\n")

           # ----------------------------------------------------
            # 📊 터미널 동적 출력 로깅 (직관성 극대화)
            # ----------------------------------------------------
            inj_type = getattr(args, 'latent_injection_type', 'UNKNOWN').upper()
            
            print(f" [{stage_name} (Inj: {inj_type}) | Ep {epoch+1:03d}] Total_L: {t_loss_n:.4f} | Train_mm: {t_mm:.2f} || Val_mm: {v_mm:.2f} ")
            
            if pipeline_mode in ['frozen', 'single_deeppa', 'single_deepla']:
                print(f"  ├─ ⚙️ Main_W: {w_main:.2f} | Aux(HDS)_W: {w_hds_current:.3f} | Geom_W: {avg_w_geom:.2f}")
                print(f"  └─ 🎯 [Loss] Main_HM: {t_hm_main/num_b:.4f} | Aux_HM: {t_hm_aux/num_b:.4f} | Crd: {t_crd/num_b:.4f} | Srf: {t_srf/num_b:.4f}")
            elif pipeline_mode == 'e2e':
                print(f"  ├─ ⚙️ PA_W: {weight_PA:.2f} | DP_Main_W: {weight_DP * w_main:.2f} | Aux_W: {w_hds_current:.2f}")
                print(f"  └─ 🎯 [Loss] PA_HM: {t_hm_PA/num_b:.4f} | Main_HM: {t_hm_main/num_b:.4f} | Aux_HM: {t_hm_aux/num_b:.4f} | Geom: {(t_crd+t_srf)/num_b:.4f}")

            log_records.append({
                'Epoch': epoch + 1, 'Total_Loss': t_loss_n, 'Val_mm': v_mm, 'Train_mm': t_mm,
                'W_Main_or_Aux': val_decay, 'W_Geom': avg_w_geom, 'W_Aux_HDS': w_hds_current,
                'L_main_hm': t_hm_main/num_b, 'L_aux_hm': t_hm_aux/num_b, 
                'L_coord': t_crd/num_b, 'L_surface': t_srf/num_b
            })
            
            with pd.ExcelWriter(excel_log_path, engine='openpyxl') as writer:
                pd.DataFrame(log_records).to_excel(writer, sheet_name='Training_Log', index=False)

            scheduler.step()

        print(f"\n💾 [Model Save] {stage_name} 학습 완료! 최종 모델을 저장합니다.")
        torch.save(model.state_dict(), os.path.join(paths['models'], f'{stage_name}_last.t7'))
        return model

    print(f"\n=== [Pipeline Start] ===")
    target_model = args.model.lower()
    
    if target_model == 'paconv': execute_stage('paconv', args.epochs, stage_name=f"Single_PAConv")
    elif target_model == 'paconv_heat': execute_stage('paconv_heat', args.epochs, stage_name=f"Single_PAConv_Heat")
    elif target_model == 'deeppa_frozen_no_heat': execute_stage('deeppa_frozen_no_heat', args.epochs, stage_name=f"DeepPA_NoHeat")
    elif target_model == 'deeppa_frozen': execute_stage('deeppa_frozen', args.epochs, stage_name=f"DeepPA_Frozen")
    elif target_model == 'deeppa_finetune': execute_stage('deeppa_finetune', args.epochs, stage_name=f"DeepPA_Finetune")
    elif target_model == 'deeppa_e2e': execute_stage('deeppa_e2e', args.epochs, stage_name=f"DeepPA_E2E")
    elif target_model == 'deeppa': execute_stage('deeppa', args.epochs, stage_name=f"Single_DeepPA")
    elif target_model == 'frozen_aux_drop': execute_stage('frozen_aux_drop', args.epochs, stage_name=f"Frozen_Aux_Drop")
    elif target_model == 'frozen_aux_fixed': execute_stage('frozen_aux_fixed', args.epochs, stage_name=f"Frozen_Aux_Fixed")
    elif target_model == 'frozen_no_aux': execute_stage('frozen_no_aux', args.epochs, stage_name=f"Frozen_No_Aux")
    elif target_model == 'deepla_ori': execute_stage('deepla_ori', args.epochs, stage_name=f"DeepLA_ORI")
    elif target_model == 'deepla_all': execute_stage('deepla_all', args.epochs, stage_name=f"DeepLA_ALL")
    elif target_model == 'deepla_all_tied': execute_stage('deepla_all_tied', args.epochs, stage_name=f"DeepLA_ALL_Tied")
    elif target_model == 'deepla_progress': execute_stage('deepla_progress', args.epochs, stage_name=f"DeepLA_PROGRESS")
    else: print(f"❌ [Error] 지원하지 않는 모델 모드입니다: {target_model}")
        
    print(f"\n=== Training Finished. Results at: {paths['root']} ===")

if __name__ == "__main__":
    args = parser.parse_args()
    _init_(args)
    train(args)