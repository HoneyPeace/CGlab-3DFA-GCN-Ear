'''
@Author: Yuan Wang (Modified by Researcher 2 & AI Assistant)
@File: train.py
@Description: Gradient Accumulation + Curriculum Learning + Auto 2-Stage Pipeline (deeppa_auto) + Global Auto-Scaled 4-Loss + 🌟 7-Channel Direct Pipeline
'''

import os
import time
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

import torch
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
# 🚀 모델 임포트 (3대장 모두 준비 완료)
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

def save_multiview_heatmap(points, heatmap, save_dir, sample_idx, landmark_idx, prefix):
    fig = plt.figure(figsize=(30, 10))
    views = [
        (131, 90, -100, "Front"),
        (132, 30, 120,  "Side"),
        (133, 45, -45, "Downside")
    ]
    for pos, elev, azim, title in views:
        ax = fig.add_subplot(pos, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=heatmap, cmap='jet', s=15, alpha=0.8)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.axis('off')

    filename = f"{prefix}_S{sample_idx:03d}_L{landmark_idx:02d}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=100, bbox_inches='tight')
    plt.close()

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

def train(args):
    accum_steps = args.accumulation_steps
    
    if not args.test_dataset_name or (args.train_dataset_name == args.test_dataset_name):
        MODE = "SPLIT"
        target_dataset = args.train_dataset_name
        args.test_dataset_name = args.train_dataset_name
        print(f"\n>>> [MODE] Split Mode (Dataset: {target_dataset})")
    else:
        MODE = "SEPARATE"
        print(f"\n>>> [MODE] Separate Mode")
        print(f"    Train: {args.train_dataset_name} / Test: {args.test_dataset_name}")

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

    print("=== [Phase 2.5] Creating Experiment Paths ===")
    train_len = len(train_dataset)
    paths = get_experiment_paths(args, train_len)

    log_file_path = os.path.join(paths['root'], 'training_log.txt')
    with open(log_file_path, 'w') as f:
        f.write(f"Experiment ID: {paths['root'].split('/')[-1]}\n")
        f.write(f"Model Backbone: {args.model}\n")
        f.write(f"Dataset: Train({args.train_dataset_name}) / Test({args.test_dataset_name})\n")
        f.write("="*160 + "\n")
        f.write(f"{'Epoch':<6} | {'T_Loss':<8} | {'T_HM':<8} | {'T_Crd':<8} | {'T_Srf':<8} | {'T_Str':<8} | {'T_mm':<6} || {'V_Loss':<8} | {'V_HM':<8} | {'V_Crd':<8} | {'V_Srf':<8} | {'V_Str':<8} | {'V_mm':<6} || {'W_HM':<6} | {'W_Crd':<6} | {'W_Srf':<6} | {'W_Str':<6}\n")
        f.write("="*160 + "\n")

    print("=== [Phase 2.6] Backing up Data ===")
    process_data_storage(train_dataset, "train", paths)
    process_data_storage(test_dataset, "test", paths)

    train_loader = DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, num_workers=0, batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    
    ScaleAndTranslate = PointcloudScaleAndTranslate()

    # =========================================================================
    # 🚀 핵심 학습 루프 함수 (Stage 분리를 위한 캡슐화)
    # =========================================================================
    def execute_stage(current_model_name, current_epochs, disable_norm=False, prior_model=None, stage_name=""):
        print(f"\n{'='*50}")
        print(f" 🚀 [Start Stage] {stage_name} (Epochs: {current_epochs}, Norm Disable: {disable_norm})")
        print(f"{'='*50}")

        with open(log_file_path, 'a') as f:
            f.write(f"\n--- [START STAGE] {stage_name} (Epochs: {current_epochs}) ---\n")

        model_name_lower = current_model_name.lower()

        if model_name_lower == 'deeppa' and prior_model is None:
            print(">>> [Prior Load] Loading Pre-trained PAConv from default path...")
            paconv_prior = PAConv(args, args.landmark_num).to(device)
            best_paconv_path = os.path.join("..", "PAConv_model", "model_epoch_500.t7")
            if not os.path.exists(best_paconv_path):
                raise FileNotFoundError(f"🚨 PAConv 모델 파일을 찾을 수 없습니다: {best_paconv_path}")
            paconv_prior.load_state_dict(torch.load(best_paconv_path))
            paconv_prior.eval()
            for param in paconv_prior.parameters():
                param.requires_grad = False 
            prior_model = paconv_prior

        if model_name_lower in ['paconv', 'paconv_heat']:
            model = PAConv(args, args.landmark_num).to(device)
            if model_name_lower == 'paconv_heat':
                print(">>> [INFO] 🔥 PAConv_heat 모드: 1-Loss(히트맵)만 학습합니다.")
        elif model_name_lower == 'deepla':
            model = DeepLA_Wrapper(args, args.landmark_num).to(device)
        elif model_name_lower == 'deeppa':
            model = DeepPA_Wrapper(args, args.landmark_num).to(device)
        else:
            raise ValueError(f"Unknown model: {current_model_name}")
            
        model.apply(weight_init)
        
        surface_criterion = CurvatureSurfaceLoss(
            k_p2p=args.plane_knn, k_curv=args.curv_knn, 
            alpha=args.curv_alpha, dir_weight=args.dir_weight
        ).to(device)

        criterion = AdaptiveWingLoss() if args.loss == 'adaptive_wing' else torch.nn.MSELoss()
            
        if args.use_sgd: 
            opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=args.weight_decay)
        else: 
            opt = optim.Adam(model.parameters(), lr=args.lr, eps=1e-08, weight_decay=args.weight_decay)
        
        scheduler = CosineAnnealingLR(opt, T_max=current_epochs) if args.scheduler == 'cos' else StepLR(opt, step_size=40, gamma=0.9)

        opt.zero_grad() 
        best_val_mm = float('inf')

        target_norm = 1.53
        auto_scales = {'heatmap': -1.0, 'coord': -1.0, 'surface': -1.0, 'struct': -1.0}

        for epoch in range(current_epochs):
            model.train()
            train_loss, train_hm, train_crd, train_srf, train_str, train_mm = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
            with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"{stage_name} Epoch {epoch+1:03d}/{current_epochs} [Train]", unit="batch", leave=False) as tepoch:
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
                    
                    if current_model_name == 'DeepPA' and prior_model is not None:
                        with torch.no_grad():
                            prior_hint = prior_model(point_input)
                        pred_heatmap = model(point_input, prior_heatmap=prior_hint)
                    else:
                        pred_heatmap = model(point_input)
                    
                    points_for_coords = point_input[:, :3, :].permute(0, 2, 1).contiguous() 
                    pred_coords = get_differentiable_coords(points_for_coords, pred_heatmap, k=args.k_softargmax)
                    
                    loss_heatmap = criterion(pred_heatmap, seg.permute(0, 2, 1).contiguous())
                    loss_coord = dynamic_focal_l1_loss(pred_coords, augmented_landmark, gamma=args.focal_gamma)
                    loss_surface = surface_criterion(pred_coords, augmented_landmark, points_for_coords, disable_norm=disable_norm)
                    loss_struct = compute_structural_loss(pred_coords, augmented_landmark)
                    
                    if epoch == 0 and i == 0 and auto_scales['heatmap'] < 0:
                        if disable_norm or model_name_lower in ['paconv', 'paconv_heat']:
                            auto_scales['heatmap'], auto_scales['coord'], auto_scales['surface'], auto_scales['struct'] = 1.0, 1.0, 1.0, 1.0
                        else:
                            auto_scales['heatmap'] = target_norm / (loss_heatmap.item() + 1e-6)
                            auto_scales['coord']   = target_norm / (loss_coord.item() + 1e-6)
                            auto_scales['surface'] = target_norm / (loss_surface.item() + 1e-6)
                            auto_scales['struct']  = target_norm / (loss_struct.item() + 1e-6)

                    norm_heatmap = loss_heatmap * auto_scales['heatmap']
                    norm_coord   = loss_coord   * auto_scales['coord']
                    norm_surface = loss_surface * auto_scales['surface']         
                    norm_struct  = loss_struct  * auto_scales['struct']

                    if model_name_lower == 'paconv_heat':
                        weights = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
                        total_loss = norm_heatmap
                    else:
                        stage1_epochs = current_epochs // 5 
                        if epoch < stage1_epochs:
                            weights = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
                            total_loss = norm_heatmap
                        else:
                            rand_weights = torch.rand(3).to(device)
                            rand_weights = (rand_weights / rand_weights.sum()) * 0.95
                            total_loss = (0.05 * norm_heatmap + 
                                          rand_weights[0] * norm_coord + 
                                          rand_weights[1] * norm_surface + 
                                          rand_weights[2] * norm_struct)
                            weights = torch.tensor([0.05, rand_weights[0], rand_weights[1], rand_weights[2]])
                    
                    loss = total_loss / accum_steps
                    loss.backward()
                    
                    if (i + 1) % accum_steps == 0:
                        opt.step()
                        opt.zero_grad() 

                    with torch.no_grad():
                        true_l1 = F.l1_loss(pred_coords, augmented_landmark).item()
                        mm_error = true_l1 * avg_m

                    train_loss += total_loss.item()
                    train_hm   += loss_heatmap.item()
                    train_crd  += loss_coord.item()
                    train_srf  += loss_surface.item()
                    train_str  += loss_struct.item()
                    train_mm   += mm_error
                    
                    tepoch.set_postfix(Loss=f"{total_loss.item():.4f}", HM=f"{loss_heatmap.item():.4f}", Srf=f"{loss_surface.item():.4f}", mm=f"{mm_error:.2f}")

            t_loss = train_loss / len(train_loader)
            t_hm   = train_hm / len(train_loader)
            t_crd  = train_crd / len(train_loader)
            t_srf  = train_srf / len(train_loader)
            t_str  = train_str / len(train_loader) 
            t_mm   = train_mm / len(train_loader)

            # --- Validation ---
            model.eval()
            val_loss, val_hm, val_crd, val_srf, val_str, val_mm = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
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
                    
                    if current_model_name == 'DeepPA' and prior_model is not None:
                        prior_hint = prior_model(point_input) 
                        pred_heatmap = model(point_input, prior_heatmap=prior_hint)
                    else:
                        pred_heatmap = model(point_input)
                    
                    points_for_coords = point_input[:, :3, :].permute(0, 2, 1).contiguous() 
                    pred_coords = get_differentiable_coords(points_for_coords, pred_heatmap, k=args.k_softargmax)

                    loss_heatmap = criterion(pred_heatmap, seg.permute(0, 2, 1).contiguous())
                    loss_coord = dynamic_focal_l1_loss(pred_coords, landmark_normal, gamma=args.focal_gamma)
                    loss_surface = surface_criterion(pred_coords, landmark_normal, points_for_coords, disable_norm=disable_norm)
                    loss_struct = compute_structural_loss(pred_coords, landmark_normal)
                    
                    true_l1 = F.l1_loss(pred_coords, landmark_normal).item()
                    mm_error = true_l1 * avg_m
                    
                    norm_heatmap = loss_heatmap * auto_scales['heatmap']
                    norm_coord   = loss_coord   * auto_scales['coord']
                    norm_surface = loss_surface * auto_scales['surface'] 
                    norm_struct  = loss_struct  * auto_scales['struct']
                    
                    if model_name_lower == 'paconv_heat':
                        total_loss = norm_heatmap
                    else:
                        stage1_epochs = current_epochs // 5
                        if epoch < stage1_epochs:
                            total_loss = norm_heatmap
                        else:
                            total_loss = 0.25 * norm_heatmap + 0.25 * norm_coord + 0.25 * norm_surface + 0.25 * norm_struct

                    val_loss += total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
                    val_hm   += loss_heatmap.item()
                    val_crd  += loss_coord.item()
                    val_srf  += loss_surface.item()
                    val_str  += loss_struct.item()
                    val_mm   += mm_error
                    
                    val_sample_count += point.size(0)
                    if val_sample_count >= 30: break

            num_val_batches = i + 1
            v_loss, v_hm, v_crd, v_srf, v_str, v_mm = val_loss/num_val_batches, val_hm/num_val_batches, val_crd/num_val_batches, val_srf/num_val_batches, val_str/num_val_batches, val_mm/num_val_batches

            # --- 🌟 터미널 실시간 출력 (4가지 세부 로스 모두 표시) ---
            print(f" [{stage_name} Ep {epoch+1:03d}] T_Loss: {t_loss:.4f} (HM:{t_hm:.4f} Crd:{t_crd:.4f} Srf:{t_srf:.4f} Str:{t_str:.4f}) | T_mm: {t_mm:.2f} || V_mm: {v_mm:.2f}")

            if model_name_lower == 'paconv_heat':
                w_hm, w_crd, w_srf, w_str = 1.0, 0.0, 0.0, 0.0
            else:
                if epoch < (current_epochs // 5):
                    w_hm, w_crd, w_srf, w_str = 1.0, 0.0, 0.0, 0.0
                else:
                    w_np = weights.detach().cpu().numpy()
                    w_hm, w_crd, w_srf, w_str = w_np[0], w_np[1], w_np[2], w_np[3]

            with open(log_file_path, 'a') as f:
                log_line = f"{epoch+1:<6d} | {t_loss:<8.4f} | {t_hm:<8.4f} | {t_crd:<8.4f} | {t_srf:<8.4f} | {t_str:<8.4f} | {t_mm:<6.2f} || {v_loss:<8.4f} | {v_hm:<8.4f} | {v_crd:<8.4f} | {v_srf:<8.4f} | {v_str:<8.4f} | {v_mm:<6.2f} || {w_hm:<6.3f} | {w_crd:<6.3f} | {w_srf:<6.3f} | {w_str:<6.3f}\n"
                f.write(log_line)

            # 🏆 Best Model 저장
            if v_mm < best_val_mm:
                best_val_mm = v_mm
                print(f" 🌟 [{stage_name}] Best Model Saved! Error: {best_val_mm:.4f} mm")
                with open(log_file_path, 'a') as f:
                    f.write(f"  >>> *** {stage_name} Epoch {epoch+1}: Best Model Saved! (Val Error: {best_val_mm:.4f} mm) ***\n")
                best_save_path = os.path.join(paths['models'], f'{stage_name}_model_best.t7')
                torch.save(model.state_dict(), best_save_path)

            if (epoch + 1) % 10 == 0:
                filename = f'{stage_name}_model_epoch_{epoch+1}.t7'
                save_path = os.path.join(paths['models'], filename)
                torch.save(model.state_dict(), save_path)

            scheduler.step()
            
        return model 

    # =========================================================================
    # 🎯 파이프라인 제어기: deeppa_auto 분기 처리
    # =========================================================================
    print(f"\n=== [Phase 3] Start Training with Curriculum Learning ===")
    
    model_type = args.model.lower() 
    
    if model_type == 'deeppa_auto':
        print(f">>> [AUTO MODE] 자동 2-Stage 학습 파이프라인을 가동합니다. (입력: {args.model})")
        
        # 1️⃣ Stage 1: PAConv (정규화 끄기)
        paconv_model = execute_stage(
            current_model_name='PAConv', 
            current_epochs=args.paconv_epochs, 
            disable_norm=True, 
            prior_model=None, 
            stage_name="Stage1_PAConv"
        )
        
        # 백업 폴더에 PAConv 모델 저장
        backup_dir = os.path.join(paths['root'], "deeppa_backup", "paconv_model")
        os.makedirs(backup_dir, exist_ok=True)
        paconv_save_path = os.path.join(backup_dir, "last_paconv.pth")
        torch.save(paconv_model.state_dict(), paconv_save_path)
        print(f"\n>>> [Stage 1 완료] PAConv 모델 백업 완료: {paconv_save_path}")
        
        # 2️⃣ Stage 2: DeepPA (정규화 켜기 + 앞서 학습한 PAConv 전달)
        paconv_model.eval()
        for param in paconv_model.parameters():
            param.requires_grad = False
            
        execute_stage(
            current_model_name='DeepPA', 
            current_epochs=args.deeppa_epochs, 
            disable_norm=False, 
            prior_model=paconv_model, 
            stage_name="Stage2_DeepPA"
        )
        print("\n>>> [AUTO MODE] 전체 파이프라인 성공적으로 종료되었습니다.")
        
    else:
        # 기존 단일 모델 실행 모드
        disable_norm_flag = (args.model in ['PAConv', 'PAConv_heat'])
        execute_stage(
            current_model_name=args.model, 
            current_epochs=args.epochs, 
            disable_norm=disable_norm_flag, 
            prior_model=None, 
            stage_name=f"Single_{args.model}"
        )
        
    print(f"\n=== Training Finished. Results at: {paths['root']} ===")

if __name__ == "__main__":
    args = parser.parse_args()
    _init_(args)
    train(args)