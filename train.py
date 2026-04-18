# @Author: Yuan Wang (Modified by Researcher)
# @File: train.py
# @Description: Unified Hybrid Pipeline + CVPR Eq.5 HDS Curriculum + 0.47mm Optimization
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

# 🌟 [수정] dynamic_focal_l1_loss -> focal_l1_loss 로 변경 동기화
from loss import AdaptiveWingLoss, CurvatureSurfaceLoss, compute_structural_loss, focal_l1_loss
from util import main_sample
# 🌟 [수정] PointcloudJitter 임포트 추가
from augmentations import normalize_data, PointcloudScaleAndTranslate, PointcloudJitter

from PAConv_model import PAConv
from DeepLA_model import DeepLA_Wrapper
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

"""
================================================================================
🌟 Unified Hybrid Pipeline
================================================================================
"""
class HybridPipeline(nn.Module):
    def __init__(self, args, landmark_num, mode='frozen'):
        super().__init__()
        self.mode = mode.lower()
        self.stage1_paconv = PAConv(args, landmark_num)
        
        # PAConv가 이제 2개를 반환하므로 aux_head는 불필요함. 
        # (단독 학습 시를 위해 남겨두되, frozen/e2e에서는 사용 안 함)
        self.s1_aux_head = nn.Conv1d(128, landmark_num, 1) 
        
        if self.mode == 'frozen':
            for param in self.stage1_paconv.parameters(): param.requires_grad = False
            for param in self.s1_aux_head.parameters(): param.requires_grad = False
                
        if self.mode in ['frozen', 'e2e']:
            self.stage2_deeppa = DeepPA_Wrapper(args, landmark_num)

    def forward(self, x):   
        if self.mode == 'stage1':
            s1_latent, s1_hm_anchor = self.stage1_paconv(x)
            
            # 🌟 [수정됨] s1_hm_anchor는 (B, 8192, 36)입니다. 
            # 36개의 랜드마크 각각에 대해, 8192개의 점(dim=1) 중 가장 확률이 높은 점을 찾습니다.
            max_idx = torch.argmax(s1_hm_anchor, dim=1) # (B, 36)
            
            # 원본 입력에서 순수 XYZ 좌표 추출 (B, 8192, 3)
            xyz_permuted = x[:, :3, :].permute(0, 2, 1).contiguous()
            
            # 찾아낸 인덱스 번호로 실제 3D 좌표(XYZ)를 뜯어옵니다.
            gather_idx = max_idx.unsqueeze(-1).expand(-1, -1, 3) # (B, 36, 3)
            pred_coords = torch.gather(xyz_permuted, 1, gather_idx) # (B, 36, 3)
            
            return pred_coords, torch.tensor(0.0).to(x.device), [s1_hm_anchor], s1_hm_anchor
            
        elif self.mode == 'frozen':
            self.stage1_paconv.eval()
            with torch.no_grad():
                s1_latent, s1_hm = self.stage1_paconv(x)
            out = self.stage2_deeppa(x, prior_latent=s1_latent.detach(), prior_heatmap=s1_hm.detach())
            pred_coords, spa_loss, sem_list = out[0], out[1] if len(out)>1 else torch.tensor(0.0).to(x.device), out[2] if len(out)>2 else []
            return pred_coords, spa_loss, sem_list, s1_hm
            
        elif self.mode == 'e2e':
            s1_latent, s1_hm = self.stage1_paconv(x)
            out = self.stage2_deeppa(x, prior_latent=s1_latent, prior_heatmap=s1_hm)
            pred_coords, spa_loss, sem_list = out[0], out[1] if len(out)>1 else torch.tensor(0.0).to(x.device), out[2] if len(out)>2 else []
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
    
    # 🌟 [수정] 증강 기법 로드 (Jitter 추가)
    ScaleAndTranslate = PointcloudScaleAndTranslate()
    ApplyJitter = PointcloudJitter(std=0.001)

    def execute_stage(current_model_name, current_epochs, disable_norm=False, stage_name=""):
        model_name_lower = current_model_name.lower()
        
        if model_name_lower == 'paconv': pipeline_mode = 'stage1'
        elif model_name_lower in ['deeppa_frozen', 'deeppa_auto']: pipeline_mode = 'frozen'
        elif model_name_lower == 'deeppa_e2e': pipeline_mode = 'e2e'
        else: pipeline_mode = 'single_custom' 
            
        if pipeline_mode in ['stage1', 'frozen', 'e2e']: 
            model = HybridPipeline(args, args.landmark_num, mode=pipeline_mode).to(device)
            
            if pipeline_mode in ['frozen', 'e2e']:
                paconv_path = os.path.join(args.output_root, "PAConv_Pretrained", "models", "Single_PAConv_last.t7")
                if os.path.exists(paconv_path):
                    print(f"📦 [Pretrained Load] 사전 학습된 PAConv 로드 완료! (Mode: {pipeline_mode.upper()})")
                    model.load_state_dict(torch.load(paconv_path, map_location=device), strict=False)
                else:
                    print(f"⚠️ [Warning] 사전 학습된 PAConv를 찾을 수 없습니다. 랜덤 가중치로 시작합니다.")
        else: 
            model = DeepPA_Wrapper(args, args.landmark_num).to(device)
            
        model.apply(weight_init)
        
        surface_criterion = CurvatureSurfaceLoss(k_p2p=args.plane_knn, k_curv=args.curv_knn, alpha=args.curv_alpha, beta=args.dir_beta).to(device)
        # 🌟 [수정] 논문 Eq.5에 맞춰 BCE -> AdaptiveWingLoss(L_sem) 적용
        hm_criterion = AdaptiveWingLoss().to(device) 
        
        opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, eps=1e-08, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(opt, T_max=current_epochs) if getattr(args, 'scheduler', 'cos') == 'cos' else StepLR(opt, step_size=40, gamma=0.9)

        excel_log_path = os.path.join(paths['root'], f'Training_Log_{stage_name}.xlsx')
        log_records = []
        auto_scales = {'coord': 1.0, 'surface': 1.0, 'struct': 1.0}; target_norm = 1.0
        use_loss_norm = getattr(args, 'use_loss_norm', False)

        for epoch in range(current_epochs):
            model.train() 
                
            train_loss_norm = 0.0; t_hm_aux, t_crd, t_srf, t_str = 0.0, 0.0, 0.0, 0.0; train_mm = 0.0
            opt.zero_grad() 
            
            # 🌟 [수정] CVPR 논문 Eq. 5 (지수 감소 스케줄링) 완벽 구현
            if not getattr(args, 'use_direct_regression', True):
                w_sem, w_spa, w_pred = 1.0, 0.0, 0.0
                current_phase_str = f"STAGE1 Heatmap Only"
            else:
                # 초기 가중치 설정 (alpha=0.3, beta=0.005)
                alpha_init = 0.3
                beta_init = 0.005
                
                # 역수 기반 감쇄 (n = 1 / (epoch + 1))
                decay_n = 1.0 / (epoch + 1)
                
                w_sem = alpha_init ** decay_n
                w_spa = beta_init ** decay_n
                w_pred = 1.0 - (w_sem + w_spa)
                
                # 모델이 충분히 안정화된 후(예: 30에폭)에는 강제로 정밀 타격에 100% 비중
                if epoch > 30:
                    w_sem, w_spa, w_pred = 0.0, 0.0, 1.0
                    
                current_phase_str = f"{pipeline_mode.upper()} CVPR Eq.5 (Pred: {w_pred:.2f})"
            
            with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"{stage_name} Ep {epoch+1:03d}", unit="batch", leave=False) as tepoch:
                for i, (point, landmark, seg) in tepoch:
                    point, landmark, seg = point.to(device), landmark.to(device), seg.to(device)

                    with torch.no_grad():
                        point_xyz = point[:, :, :3]
                        avg_m = torch.mean(torch.max(torch.sqrt(torch.sum((point_xyz - torch.mean(point_xyz, axis=1, keepdim=True)) ** 2, axis=2)), axis=1)[0]).item()

                    point_normal, landmark_normal = normalize_data(point, landmark)
                    point_normal, augmented_landmark = ScaleAndTranslate(point_normal, landmark_normal)
                    
                    # 🌟 [수정] 강건성 확보를 위한 가우시안 지터링 (정답지는 고정)
                    point_normal = ApplyJitter(point_normal)
                    
                    point_input = point_normal.permute(0, 2, 1).contiguous()
                    points_for_coords = point_input[:, :3, :].permute(0, 2, 1).contiguous() 
                    target_hm = seg.permute(0, 2, 1).contiguous()
                    
                    # ---------------------------------------------------------
                    # 1. Forward Pass & PAConv 히트맵 안전 확보
                    # ---------------------------------------------------------
                    if pipeline_mode in ['stage1', 'frozen', 'e2e']:
                        pred_coords, spa_loss, sem_list, s1_hm = model(point_input)
                        
                        sem_list = list(sem_list)
                        if s1_hm is not None:
                            sem_list.append(s1_hm)
                    else:
                        out = model(point_input)
                        pred_coords, spa_loss, sem_list = out[0], torch.tensor(0.0).to(device), []
                        
                    # 2. 로스 계산 (HDS 모의고사)
                    L_spa = spa_loss if isinstance(spa_loss, torch.Tensor) else torch.tensor(0.0).to(device)
                    
                    # ---------------------------------------------------------
                    # 기하학적 1:1 매칭 HDS (Gather 방식)
                    # ---------------------------------------------------------
                    safe_sem_list = []
                    target_list = []

                    for idx, sp in enumerate(sem_list):
                        if sp is None: continue
                        if sp.shape[1] != target_hm.shape[1]: 
                            sp = sp.permute(0, 2, 1).contiguous()
                        
                        curr_N = sp.shape[2]
                        if curr_N == 8192 or curr_N == 2048: # 모델 입력 크기에 따라 분기
                            safe_sem_list.append(sp)
                            target_list.append(target_hm)
                        else:
                            # [TODO]: Gather 방식 활성화 시, 모델에서 indices 리스트를 리턴받아야 함.
                            # 현재 구조에서는 shape 불일치 시 스킵하여 에러 방지
                            continue 

                    if len(safe_sem_list) > 0:
                        L_sem = sum([hm_criterion(s, t) for s, t in zip(safe_sem_list, target_list)]) / len(safe_sem_list)
                    else:
                        L_sem = torch.tensor(0.0).to(device)
                        
                    # 🌟 [수정] 최종 회귀 로스 계산 (Standard Focal L1)
                    loss_coord = focal_l1_loss(pred_coords, augmented_landmark, gamma=args.focal_gamma)
                    loss_surface, _, _, _ = surface_criterion(pred_coords, augmented_landmark, points_for_coords, disable_norm=False)
                    loss_struct = compute_structural_loss(pred_coords, augmented_landmark)

                    if use_loss_norm and epoch == 0 and i == 0 and w_pred > 0:
                        auto_scales['coord'] = target_norm / loss_coord.item() if loss_coord.item() > 0 else 1.0
                        auto_scales['surface'] = target_norm / loss_surface.item() if loss_surface.item() > 0 else 1.0
                        auto_scales['struct'] = target_norm / loss_struct.item() if loss_struct.item() > 0 else 1.0

                    # 🌟 [수정됨] RLW (Random Loss Weighting) 적용 로직
                    if getattr(args, 'use_rlw_for_pred', False):
                        # 매 스텝마다 3개 로스에 대해 합이 1이 되는 랜덤 가중치(Dirichlet 효과) 생성
                        rand_w = F.softmax(torch.randn(3, device=device), dim=0)
                        w_crd, w_srf, w_str = rand_w[0], rand_w[1], rand_w[2]
                    else:
                        # RLW가 꺼져있으면 원래대로 모두 1.0 비율 고정
                        w_crd, w_srf, w_str = 1.0, 1.0, 1.0

                    L_pred = (w_crd * loss_coord * auto_scales['coord']) + \
                             (w_srf * loss_surface * auto_scales['surface']) + \
                             (w_str * loss_struct * auto_scales['struct'])

                    # 3. 로스 융합 및 역전파
                    total_loss = (w_sem * L_sem) + (w_spa * L_spa) + (w_pred * L_pred)
                                        
                    loss = total_loss / accum_steps
                    loss.backward()
                    
                    if (i + 1) % accum_steps == 0:
                        opt.step()
                        opt.zero_grad() 

                    with torch.no_grad():
                        # 🌟 w_pred > 0 조건을 지워서, 1단계에서도 무조건 Train 오차를 계산합니다!
                        mm_error = F.l1_loss(pred_coords, augmented_landmark).item() * avg_m

                    train_loss_norm += total_loss.item()
                    t_hm_aux += L_sem.item(); t_crd += loss_coord.item(); t_srf += loss_surface.item(); t_str += loss_struct.item()
                    train_mm += mm_error
                    
                    vram_str = f"{torch.cuda.max_memory_allocated() / (1024 ** 3):.1f}GB" if torch.cuda.is_available() else "CPU"
                    tepoch.set_postfix(Loss=f"{total_loss.item():.4f}", VRAM=vram_str)

            num_b = len(train_loader)
            t_loss_n = train_loss_norm / num_b
            t_mm = train_mm / num_b

# --- Validation ---
            model.eval()
            val_mm_total = 0.0
            val_samples = 0
            
            with torch.no_grad():
                for point, landmark, seg in test_loader:
                    point, landmark, seg = point.to(device), landmark.to(device), seg.to(device)
                    B_val = point.size(0) 
                    
                    point_xyz = point[:, :, :3]
                    avg_m = torch.mean(torch.max(torch.sqrt(torch.sum((point_xyz - torch.mean(point_xyz, axis=1, keepdim=True)) ** 2, axis=2)), axis=1)[0]).item()
                    
                    # 테스트 시에는 Jitter 등 증강을 주지 않고 순수 Normalize만!
                    point_normal, landmark_normal = normalize_data(point, landmark)
                    point_input = point_normal.permute(0, 2, 1).contiguous() 
                    
                    if pipeline_mode in ['stage1', 'frozen', 'e2e']:
                        pred_coords, _, _, _ = model(point_input)
                    else:
                        pred_coords = model(point_input)[0]

                    # 🌟 조건문(if w_pred > 0)을 완전히 삭제했습니다. 이제 1단계에서도 무조건 오차를 누적합니다!
                    val_mm_total += F.l1_loss(pred_coords, landmark_normal).item() * avg_m * B_val
                    
                    val_samples += B_val
            
            # 전체 샘플에 대한 정확한 가중 평균 오차 계산
            v_mm = val_mm_total / val_samples if val_samples > 0 else 0.0                     

            
# --- 성적표 출력 ---
            if pipeline_mode == 'stage1':
                # 🌟 [수정됨] Stage 1에서도 Train_mm과 Val_mm(Argmax 좌표 오차)을 당당하게 출력합니다!
                print(f" [{stage_name} Ep {epoch+1:03d}] Total_L: {t_loss_n:.3f} | L_sem(HM): {t_hm_aux/num_b:.4f} || Train_mm: {t_mm:.2f} | Val_mm: {v_mm:.2f}")
            else:
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
        torch.save(model.state_dict(), last_save_path)
            
        return model 

    print(f"\n=== [Pipeline Start] ===")
    
    target_model = args.model.lower()
    if target_model == 'paconv':
        print(f">>> [MODE: STAGE 1] 🎯 PAConv 단독 학습 (히트맵 100%)")
        execute_stage('paconv', args.epochs, stage_name=f"Single_PAConv")
    elif target_model in ['deeppa_frozen', 'deeppa_auto']:
        print(f">>> [MODE: FROZEN] ❄️ PAConv 프리징 + HDS 커리큘럼")
        execute_stage('deeppa_frozen', args.epochs, stage_name=f"Frozen_Hybrid")
    elif target_model == 'deeppa_e2e':
        print(f">>> [MODE: E2E] 🔥 PAConv 동시 학습(Joint Opt) + HDS 커리큘럼")
        execute_stage('deeppa_e2e', args.epochs, stage_name=f"E2E_Hybrid")
    else:
        execute_stage(args.model, args.epochs, disable_norm=True, stage_name=f"Single_{args.model}")
        
    print(f"\n=== Training Finished. Results at: {paths['root']} ===")

if __name__ == "__main__":
    args = parser.parse_args()
    _init_(args)
    train(args)