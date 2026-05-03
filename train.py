# @Author: Yuan Wang (Modified by Researcher Park Pyeong-hwa & AI Assistant)
# @File: train_all_integrated.py
# @Description: 순수 학습 루프 (모든 분기/가중치/프린팅 로직은 loss_controller.py로 100% 분리됨)
# ==============================================================================

import os
import sys
import subprocess
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import time
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import pandas as pd
import numpy as np
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

# 🌟 [핵심] 외부 로스 컨트롤러 임포트
from loss_controller import DeepPALossController

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None: torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, (torch.nn.Conv2d, torch.nn.Conv1d)):
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

def _shape_file_name(in_channels, partition):
    if in_channels == 7:
        return f"shape_{partition}.npy"
    if in_channels == 6:
        return f"shape_6ch_{partition}.npy"
    if in_channels == 3:
        return f"shape_3ch_{partition}.npy"
    return f"shape_{partition}.npy"

def resolve_validation_dataset(args):
    val_partition = getattr(args, 'val_partition', 'val')
    val_dataset_name = getattr(args, 'val_dataset_name', '')
    if val_dataset_name:
        return val_dataset_name, val_partition

    candidate_names = [args.train_dataset_name, args.test_dataset_name]
    for candidate_name in candidate_names:
        base_path = os.path.join(args.data_root, f"{candidate_name}-npy")
        shape_path = os.path.join(base_path, _shape_file_name(args.in_channels, val_partition))
        heat_path = os.path.join(base_path, f"Heat_data_{val_partition}.npy")
        land_path = os.path.join(base_path, f"landmark_{val_partition}.npy")
        if os.path.exists(shape_path) and os.path.exists(heat_path) and os.path.exists(land_path):
            return candidate_name, val_partition

    return args.test_dataset_name, 'test'

def backup_npy_split(paths, data_root, data_name, partition, label):
    base_path = os.path.join(data_root, f"{data_name}-npy")
    if not os.path.isdir(base_path):
        print(f"   [WARNING] {label.upper()} NPY backup skipped. Missing folder: {base_path}")
        return

    suffix = f"_{partition}.npy"
    copied_count = 0
    for file_name in os.listdir(base_path):
        if file_name.endswith(suffix):
            src_path = os.path.join(base_path, file_name)
            dst_path = os.path.join(paths['npy_backup'], file_name)
            shutil.copy2(src_path, dst_path)
            copied_count += 1

    print(f"   [{label.upper()}] Backup Saved: {paths['npy_backup']} ({copied_count} files)")

def get_last_checkpoint_name(model_name):
    m_name = model_name.lower()
    if m_name in ['paconv', 'paconv_heat', 'paconv_struct']:
        return 'Single_PAConv_last.t7'
    if m_name == 'frozen_aux_fixed':
        return 'Frozen_Aux_Fixed_last.t7'
    if m_name == 'frozen_aux_drop':
        return 'Frozen_Aux_Drop_last.t7'
    if m_name == 'frozen_no_aux':
        return 'Frozen_No_Aux_last.t7'
    return f'{m_name}_last.t7'

def save_command_txt(paths, model_name):
    command = subprocess.list2cmdline([sys.executable] + sys.argv)
    base_path = os.path.join(paths['root'], f'command_train_{model_name.lower()}.txt')
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
    print(f"[INFO] Training command saved to: {save_path}")

class UniversalPipeline(nn.Module):
    def __init__(self, args, landmark_num, mode='single_deeppa'):
        super().__init__()
        self.mode = mode.lower()
        self.args = args
        self.landmark_num = landmark_num
        
        if self.mode in ['frozen', 'finetune', 'e2e']:
            self.stage1_paconv = PAConv(args, landmark_num)
            if self.mode == 'frozen':
                for param in self.stage1_paconv.parameters(): param.requires_grad = False
            self.stage2_deeppa = DeepPA_Wrapper(args, landmark_num)
            
        # 🌟 PAConv 단독 모드 복원
        elif self.mode in ['single_paconv', 'single_paconv_heat']: 
            self.model = PAConv(args, landmark_num)
            
        else:
            self.model = DeepPA_Wrapper(args, landmark_num)

    def forward(self, x):   
        points_norm_xyz = x[:, :3, :].permute(0, 2, 1).contiguous()
        k_val = getattr(self.args, 'regression_point_num', 10)

        # 🌟 PAConv 단독 모드 복원
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
    val_dataset_name, val_partition = resolve_validation_dataset(args)

    if args.need_resample:
        main_sample(args.num_points, args.seed, args.sigma, args.sample_way, args.train_dataset_name, args.data_root, partition='train')
        main_sample(args.num_points, args.seed, args.sigma, args.sample_way, val_dataset_name, args.data_root, partition=val_partition)
        if not (val_dataset_name == args.test_dataset_name and val_partition == 'test'):
            main_sample(args.num_points, args.seed, args.sigma, args.sample_way, args.test_dataset_name, args.data_root, partition='test')

    print(f">> [INFO] Loading Separate Datasets: {args.train_dataset_name} (train) & {val_dataset_name} ({val_partition})")
    train_dataset = FaceLandmarkData(data_root=args.data_root, partition='train', data=args.train_dataset_name, in_channels=args.in_channels)
    val_dataset = FaceLandmarkData(data_root=args.data_root, partition=val_partition, data=val_dataset_name, in_channels=args.in_channels)

    paths = get_experiment_paths(args, len(train_dataset))
    save_command_txt(paths, m_name)
    backup_npy_split(paths, args.data_root, args.train_dataset_name, 'train', 'train')
    backup_npy_split(paths, args.data_root, val_dataset_name, val_partition, 'val')
    if not (val_dataset_name == args.test_dataset_name and val_partition == 'test'):
        backup_npy_split(paths, args.data_root, args.test_dataset_name, 'test', 'test')

    train_loader = DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, num_workers=0, batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    
    ScaleAndTranslate = PointcloudScaleAndTranslate()
    ApplyJitter = PointcloudJitter(std=0.001)

    m_name = args.model.lower()
    
    # 🌟 1. 아키텍처(파이프라인) 모드 라우팅 (예외 처리 및 완전성 강화)
    if m_name in ['paconv', 'paconv_struct']: 
        pipeline_mode = 'single_paconv'
    elif m_name == 'paconv_heat': 
        pipeline_mode = 'single_paconv_heat'
    elif m_name in ['deeppa_frozen', 'frozen_aux_drop', 'frozen_aux_fixed', 'frozen_no_aux']:
        pipeline_mode = 'frozen'
    elif m_name in ['deepla_ori', 'deepla_decay', 'deepla_all_tied', 'deepla_progress']:
        pipeline_mode = 'single_deepla'
    elif m_name == 'deeppa_e2e':
        pipeline_mode = 'e2e'
    elif m_name == 'single_deeppa':
        pipeline_mode = 'single_deeppa'
    else:
        raise ValueError(f"❌ [Error] 지원하지 않는 모델 모드입니다: {m_name}") # 방어 코드

    print(f"\n=== [Pipeline Start: {m_name.upper()}] ===")
    
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
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs) if getattr(args, 'scheduler', 'cos') == 'cos' else StepLR(opt, step_size=40, gamma=0.9)

    excel_log_path = os.path.join(paths['root'], f'Training_Log_{m_name}.xlsx')
    log_records = []

    # 🌟 2. 외부 로스 컨트롤러 장착 (파라미터 연동 강화)
    loss_controller = DeepPALossController(
        patience=args.patience, 
        base_hds=getattr(args, 'hds_buffer', 0.1), # 하드코딩 제거
        decay_step=getattr(args, 'val_decay_step', 0.05),
        min_heatmap_warmup=getattr(args, 'min_heatmap_warmup', 30),
        use_rlw_for_pred=getattr(args, 'use_rlw_for_pred', False)
    )
    auto_scales = {'pa': -1.0, 'main': -1.0, 'aux': -1.0, 'coord': -1.0, 'surface': -1.0, 'struct': -1.0}

    for epoch in range(args.epochs):
        model.train() 
        if pipeline_mode == 'frozen': model.stage1_paconv.eval()
        
        t_loss_n, t_mm = 0.0, 0.0
        t_hm_PA, t_hm_main, t_hm_aux, t_crd, t_srf, t_str = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        opt.zero_grad() 
        
        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"{m_name.upper()} Ep {epoch+1:03d}", unit="batch", leave=False) as tepoch:
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
                
                pred_coords, sem_list, paconv_hm = model(point_input)

                # 🌟 3. 개별(순수) 로스 계산
                L_pa = torch.tensor(0.0).to(device)
                if pipeline_mode in ['e2e', 'single_paconv', 'single_paconv_heat'] and paconv_hm is not None:
                    L_pa = hm_criterion(paconv_hm, target_hm)
                
                L_main_hm = torch.tensor(0.0).to(device)
                L_aux_hm = torch.tensor(0.0).to(device)
                
                if pipeline_mode not in ['single_paconv', 'single_paconv_heat']:
                     L_main_hm, L_aux_hm = hierarchical_hm_loss(sem_list, target_hm)

                L_crd = focal_l1_loss(pred_coords, augmented_landmark, gamma=args.focal_gamma)
                L_srf, _, _, _ = surface_criterion(pred_coords, augmented_landmark, points_for_coords, disable_norm=False)
                L_str = compute_structural_loss(pred_coords, augmented_landmark)

                if getattr(args, 'use_loss_norm', False):
                    loss_items = {
                        'pa': L_pa,
                        'main': L_main_hm,
                        'aux': L_aux_hm,
                        'coord': L_crd,
                        'surface': L_srf,
                        'struct': L_str,
                    }
                    for key, loss_value in loss_items.items():
                        if auto_scales[key] < 0:
                            value = loss_value.detach().item()
                            auto_scales[key] = args.target_norm / (value + 1e-6) if value > 0.0 else 1.0
                    L_pa = L_pa * auto_scales['pa']
                    L_main_hm = L_main_hm * auto_scales['main']
                    L_aux_hm = L_aux_hm * auto_scales['aux']
                    L_crd = L_crd * auto_scales['coord']
                    L_srf = L_srf * auto_scales['surface']
                    L_str = L_str * auto_scales['struct']
                
                # 🌟 4. 로스 컨트롤러에게 모델명과 개별 로스를 넘겨 최종 로스 산출
                total_loss, weights = loss_controller.compute_loss(m_name, epoch, L_main_hm, L_aux_hm, L_crd, L_srf, L_str, L_pa)
                                    
                loss = total_loss / accum_steps
                loss.backward()
                
                if (i + 1) % accum_steps == 0:
                    opt.step(); opt.zero_grad() 

                with torch.no_grad():
                    mm_error = F.l1_loss(pred_coords, augmented_landmark).item() * avg_m

                t_loss_n += total_loss.item()
                t_hm_PA += L_pa.item(); t_hm_main += L_main_hm.item(); t_hm_aux += L_aux_hm.item()
                t_crd += L_crd.item(); t_srf += L_srf.item(); t_str += L_str.item()
                t_mm += mm_error
                
                tepoch.set_postfix(Loss=f"{total_loss.item():.4f}")

        # ----------------------------------------------------
        # 🌟 5. Validation 평가
        # ----------------------------------------------------
        model.eval()
        val_mm_total, val_samples = 0.0, 0
        with torch.no_grad():
            for point, landmark, seg in val_loader:
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
        # 🔄 6. 컨트롤러에 Val_mm 전달하여 Patience 스케줄러 작동
        # ========================================================
        if loss_controller.update_patience(v_mm, epoch):
            print(f"\n🔄 [Patience Trigger] {loss_controller.patience}에폭 정체! 가중치 전환 -> {loss_controller.val_decay:.2f}\n")

        # ----------------------------------------------------
        # 📊 7. 컨트롤러를 통한 스마트 프린팅 및 엑셀 로깅 (관계로스 Str 포함)
        # ----------------------------------------------------
        num_b = len(train_loader)
        log_record = loss_controller.print_and_get_log(
            m_name, epoch, t_loss_n, t_mm/num_b, v_mm, weights, num_b, 
            t_hm_main, t_hm_aux, t_crd, t_srf, t_str, t_hm_PA
        )
        
        log_records.append(log_record)
        with pd.ExcelWriter(excel_log_path, engine='openpyxl') as writer:
            pd.DataFrame(log_records).to_excel(writer, sheet_name='Training_Log', index=False)

        scheduler.step()

    print(f"\n💾 [Model Save] {m_name} 학습 완료! 최종 모델을 저장합니다.")
    torch.save(model.state_dict(), os.path.join(paths['models'], get_last_checkpoint_name(m_name)))

def execute_all_models(args):
    train(args)
    print(f"\n=== Training Finished ===")

if __name__ == "__main__":
    args = parser.parse_args()
    _init_(args)
    execute_all_models(args)
