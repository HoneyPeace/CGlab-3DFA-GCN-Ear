# @Author: Yuan Wang (Modified by Researcher)
# @File: train.py
# @Description: 전면 프리징(Freezing) 기반 Task Disentanglement + HDS 커리큘럼 학습
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
🌟 [디펜스 핵심: Frozen Hybrid Model]
- Point 1, 2: PAConv(Stage 1)를 전면 프리징하여 지식을 보존하고 역전파를 차단합니다.
- Point 8: 추출된 힌트는 .detach()를 통해 안전하게 DeepPA(Stage 2)로 주입됩니다.
================================================================================
"""
class FrozenHybridModel(nn.Module):
    def __init__(self, args, landmark_num):
        super().__init__()
        self.stage1_paconv = PAConv(args, landmark_num)
        self.s1_aux_head = nn.Conv1d(64, landmark_num, 1)
        
        # 🌟 [디펜스 포인트 2] PAConv 전면 프리징
        for param in self.stage1_paconv.parameters():
            param.requires_grad = False
        for param in self.s1_aux_head.parameters():
            param.requires_grad = False
            
        self.stage2_deeppa = DeepPA_Wrapper(args, landmark_num)

    def forward(self, x):
        # 🌟 [디펜스 포인트 1] Stage 1은 평가 모드로 동작 (가이드 역할만 수행)
        self.stage1_paconv.eval()
        self.s1_aux_head.eval()
        
        with torch.no_grad():
            s1_latent = self.stage1_paconv(x)
            s1_hm = self.s1_aux_head(s1_latent)
            
        # 🌟 [디펜스 포인트 4, 8] 라텐트 힌트 주입 (detach로 그래디언트 차단)
        # DeepPA_Wrapper가 (pred_coords, spa_loss, sem_list)를 반환하도록 설계됨
        out = self.stage2_deeppa(x, prior_latent=s1_latent.detach(), prior_heatmap=s1_hm_anchor.detach())
        
        # 반환값 호환성 처리 (deeppa_semseg 구조 반영)
        if len(out) >= 3:
            pred_coords, spa_loss, sem_list = out[0], out[1], out[2]
        else:
            pred_coords, spa_loss, sem_list = out[0], torch.tensor(0.0).to(x.device), []
            
        return pred_coords, spa_loss, sem_list, s1_hm

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

    def execute_stage(current_model_name, current_epochs, disable_norm=False, stage_name=""):
        model_name_lower = current_model_name.lower()
        if model_name_lower == 'deeppa_e2e': 
            model = FrozenHybridModel(args, args.landmark_num).to(device)
            # 사전 학습된 PAConv 가중치 로드 (경로를 맞게 수정해주세요)
            paconv_path = os.path.join(args.output_root, "PAConv_Pretrained", "models", "Single_PAConv_last.t7")
            if os.path.exists(paconv_path):
                print(f"📦 [Pretrained Load] 사전 학습된 PAConv 모델을 불러왔습니다.")
                model.stage1_paconv.load_state_dict(torch.save(paconv_path))
            else:
                print(f"⚠️ [Warning] 사전 학습된 PAConv 모델을 찾을 수 없습니다. 랜덤 가중치로 프리징됩니다.")
        else: 
            model = DeepPA_Wrapper(args, args.landmark_num).to(device)
            
        model.apply(weight_init)
        
        surface_criterion = CurvatureSurfaceLoss(k_p2p=args.plane_knn, k_curv=args.curv_knn, alpha=args.curv_alpha, beta=args.dir_beta).to(device)
        criterion = AdaptiveWingLoss() if args.loss == 'adaptive_wing' else torch.nn.MSELoss()
        hm_criterion = nn.BCEWithLogitsLoss() # 히트맵 훈련용 로스
        
        opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, eps=1e-08, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(opt, T_max=current_epochs) if args.scheduler == 'cos' else StepLR(opt, step_size=40, gamma=0.9)

        excel_log_path = os.path.join(paths['root'], f'Training_Log_{stage_name}.xlsx')
        log_records = []
        target_norm = 1.0 
        auto_scales = {'coord': 1.0, 'surface': 1.0, 'struct': 1.0}
        df_calib = None 

        # =========================================================================
        # 🟢 [Phase 3] 본 학습 (Main Training - Deterministic Decay)
        # =========================================================================
        for epoch in range(current_epochs):
            model.train() 
            if model_name_lower == 'deeppa_e2e':
                model.stage1_paconv.eval() # 프리징 모델은 항상 eval 유지
                model.s1_aux_head.eval()
                
            train_loss_norm = 0.0
            t_hm_aux, t_crd, t_srf, t_str = 0.0, 0.0, 0.0, 0.0
            train_mm = 0.0
            opt.zero_grad() 
            
            # 🌟 [디펜스 포인트 7] 결정론적 로스 스케줄링 (Curriculum Learning)
            # 학습 초반에는 HDS 모의고사가 길을 잡고, 에폭이 지날수록 다이렉트 좌표 예측 비중이 1.0에 수렴
            decay_rate = 0.95
            w_sem = 0.3 * (decay_rate ** epoch)
            w_spa = 0.005 * (decay_rate ** epoch)
            w_pred = 1.0 - (w_sem + w_spa)
            
            current_phase_str = f"Frozen + HDS Decay (Pred: {w_pred:.2f})"
            
            with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"{stage_name} Ep {epoch+1:03d}", unit="batch", leave=False) as tepoch:
                for i, (point, landmark, seg) in tepoch:
                    point, landmark, seg = point.to(device), landmark.to(device), seg.to(device)

                    with torch.no_grad():
                        point_xyz = point[:, :, :3]
                        avg_m = torch.mean(torch.max(torch.sqrt(torch.sum((point_xyz - torch.mean(point_xyz, axis=1, keepdim=True)) ** 2, axis=2)), axis=1)[0]).item()

                    point_normal, landmark_normal = normalize_data(point, landmark)
                    point_normal, augmented_landmark = ScaleAndTranslate(point_normal, landmark_normal)
                    point_input = point_normal.permute(0, 2, 1).contiguous()
                    points_for_coords = point_input[:, :3, :].permute(0, 2, 1).contiguous() 
                    target_hm = seg.permute(0, 2, 1).contiguous()
                    
                    # ---------------------------------------------------------
                    # 1. Forward Pass (다이렉트 회귀 + HDS 반환)
                    # ---------------------------------------------------------
                    if model_name_lower == 'deeppa_e2e':
                        pred_coords, spa_loss, sem_list, s1_hm_anchor = model(point_input)
                    else:
                        out = model(point_input)
                        pred_coords, spa_loss, sem_list = out[0], torch.tensor(0.0).to(device), []
                        
                    # ---------------------------------------------------------
                    # 2. 로스 계산 🌟 [디펜스 포인트 5, 6: HDS 모의고사]
                    # ---------------------------------------------------------
                    L_spa = spa_loss if isinstance(spa_loss, torch.Tensor) else torch.tensor(0.0).to(device)
                    
                    L_sem = 0.0
                    if len(sem_list) > 0:
                        for sem_pred in sem_list:
                            L_sem += hm_criterion(sem_pred, target_hm)
                        L_sem = L_sem / len(sem_list)
                    else:
                        L_sem = torch.tensor(0.0).to(device)
                        
                    # [디펜스 포인트 9: 최종 회귀 로스 계산]
                    loss_coord = dynamic_focal_l1_loss(pred_coords, augmented_landmark, gamma=args.focal_gamma)
                    loss_surface, _, _, _ = surface_criterion(pred_coords, augmented_landmark, points_for_coords, disable_norm=False)
                    loss_struct = compute_structural_loss(pred_coords, augmented_landmark)
                    
                    L_pred = loss_coord + loss_surface + loss_struct

                    # ---------------------------------------------------------
                    # 3. 로스 융합 및 역전파 (커리큘럼 가중치 적용)
                    # ---------------------------------------------------------
                    total_loss = (w_sem * L_sem) + (w_spa * L_spa) + (w_pred * L_pred)
                                        
                    loss = total_loss / accum_steps
                    loss.backward()
                    
                    if (i + 1) % accum_steps == 0:
                        opt.step()
                        opt.zero_grad() 

                    with torch.no_grad():
                        mm_error = F.l1_loss(pred_coords, augmented_landmark).item() * avg_m

                    train_loss_norm += total_loss.item()
                    t_hm_aux += L_sem.item(); t_crd += loss_coord.item(); t_srf += loss_surface.item(); t_str += loss_struct.item()
                    train_mm += mm_error
                    
                    vram_str = f"{torch.cuda.max_memory_allocated() / (1024 ** 3):.1f}GB" if torch.cuda.is_available() else "CPU"
                    tepoch.set_postfix(Loss=f"{total_loss.item():.4f}", mm=f"{mm_error:.2f}", VRAM=vram_str)

            num_b = len(train_loader)
            t_loss_n = train_loss_norm / num_b
            t_mm = train_mm / num_b

            # --- Validation ---
            model.eval()
            val_mm = 0.0
            
            with torch.no_grad():
                for point, landmark, seg in test_loader:
                    point, landmark, seg = point.to(device), landmark.to(device), seg.to(device)
                    point_xyz = point[:, :, :3]
                    avg_m = torch.mean(torch.max(torch.sqrt(torch.sum((point_xyz - torch.mean(point_xyz, axis=1, keepdim=True)) ** 2, axis=2)), axis=1)[0]).item()
                    
                    point_normal, landmark_normal = normalize_data(point, landmark)
                    point_input = point_normal.permute(0, 2, 1).contiguous() 
                    
                    if model_name_lower == 'deeppa_e2e':
                        pred_coords, _, _, _ = model(point_input)
                    else:
                        pred_coords = model(point_input)[0]

                    val_mm += F.l1_loss(pred_coords, landmark_normal).item() * avg_m
            
            v_mm = val_mm / len(test_loader)
            
            # --- 성적표 출력 ---
            print(f" [{stage_name} Ep {epoch+1:03d}] Total_L: {t_loss_n:.3f} | Train_mm: {t_mm:.2f} || Val_mm: {v_mm:.2f}")
            print(f"  ├─ [Weights] w_sem: {w_sem:.4f} | w_spa: {w_spa:.4f} | w_pred: {w_pred:.4f}")
            print(f"  ├─ [S2_Pred_Loss] Crd: {t_crd/num_b:.4f} | Srf: {t_srf/num_b:.4f} | Str: {t_str/num_b:.4f}")
            print(f"  └─ [HDS_Aux_Loss] L_sem(Heatmap): {t_hm_aux/num_b:.4f}")

            log_records.append({
                'Epoch': epoch + 1, 'Phase': current_phase_str,
                'Total_Loss': t_loss_n, 'Val_mm': v_mm, 'Train_mm': t_mm,
                'w_sem': w_sem, 'w_pred': w_pred,
                'L_sem_HM': t_hm_aux/num_b, 'L_coord': t_crd/num_b, 'L_surface': t_srf/num_b, 'L_struct': t_str/num_b
            })
            
            with pd.ExcelWriter(excel_log_path, engine='openpyxl') as writer:
                df_log = pd.DataFrame(log_records)
                df_log.to_excel(writer, sheet_name='Training_Log', index=False)

            scheduler.step()

        print(f"\n💾 [Model Save] 학습 완료! 모델을 저장합니다.")
        last_save_path = os.path.join(paths['models'], f'{stage_name}_last.t7')
        
        if model_name_lower == 'deeppa_e2e':
            # 프리징된 PAConv는 굳이 저장할 필요 없으므로 DeepPA 가중치만 저장
            torch.save(model.stage2_deeppa.state_dict(), last_save_path)
        else:
            torch.save(model.state_dict(), last_save_path)
            
        return model 

    print(f"\n=== [Phase 3] Start Task Disentangled Pipeline ===")
    
    if args.model.lower() == 'deeppa_auto':
        print(f">>> [DEFENSE MODE] 🛡️ PAConv 전면 프리징 + HDS 커리큘럼 스케줄링 가동")
        execute_stage('deeppa_e2e', args.epochs, disable_norm=True, stage_name=f"Frozen_Hybrid_DeepPA")
    else:
        execute_stage(args.model, args.epochs, disable_norm=True, stage_name=f"Single_{args.model}")
        
    print(f"\n=== Training Finished. Results at: {paths['root']} ===")

if __name__ == "__main__":
    args = parser.parse_args()
    _init_(args)
    train(args)