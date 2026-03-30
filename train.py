'''
@Author: Yuan Wang (Modified by Researcher 2 & AI Assistant)
@File: train.py
@Description: Gradient Accumulation + Curriculum Learning + DeepLA Offset Support
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
from loss import AdaptiveWingLoss, get_differentiable_coords, compute_point_to_plane_loss, compute_structural_loss, dynamic_focal_l1_loss
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

    if args.need_resample:
        main_sample(args.num_points, args.seed, args.sigma, args.sample_way, 
                    args.train_dataset_name, args.data_root, partition='train')
        main_sample(args.num_points, args.seed, args.sigma, args.sample_way, 
                    args.test_dataset_name, args.data_root, partition='test')

    print("=== [Phase 2] Loading Data ===")
    
    if MODE == "SPLIT":
        full_dataset = FaceLandmarkData(data_root=args.data_root, partition='trainval', data=args.train_dataset_name)
        train_size = int(len(full_dataset) * 0.7)
        test_size = len(full_dataset) - train_size
        torch.manual_seed(args.dataset_seed)
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    else:
        train_dataset = FaceLandmarkData(data_root=args.data_root, partition='train', data=args.train_dataset_name)
        test_dataset = FaceLandmarkData(data_root=args.data_root, partition='test', data=args.test_dataset_name)

    train_len = len(train_dataset)
    paths = get_experiment_paths(args, train_len)

    process_data_storage(train_dataset, "train", paths)
    process_data_storage(test_dataset, "test", paths)

    train_loader = DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, num_workers=0, batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    
    ScaleAndTranslate = PointcloudScaleAndTranslate()

    print(f"\n>>> [Model Init] Selected Backbone: {args.model}")
    
    if args.model == 'DeepPA':
        paconv_prior = PAConv(args, args.landmark_num).to(device)
        best_paconv_path = os.path.join("..", "PAConv_model", "model_epoch_500.t7")
        paconv_prior.load_state_dict(torch.load(best_paconv_path))
        paconv_prior.eval()
        for param in paconv_prior.parameters(): param.requires_grad = False 
    else:
        paconv_prior = None

    if args.model == 'PAConv' or args.model == 'PAConv_heat':
        model = PAConv(args, args.landmark_num).to(device)
    elif args.model == 'DeepLA':
        model = DeepLA_Wrapper(args, args.landmark_num).to(device)
    elif args.model == 'DeepPA':
        model = DeepPA_Wrapper(args, args.landmark_num).to(device)
        
    model.apply(weight_init)
    
    if args.loss == 'adaptive_wing': criterion = AdaptiveWingLoss()
    else: criterion = torch.nn.MSELoss()
        
    if args.use_sgd: 
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=args.weight_decay)
    else: 
        opt = optim.Adam(model.parameters(), lr=args.lr, eps=1e-08, weight_decay=args.weight_decay)
    
    if args.scheduler == 'cos': scheduler = CosineAnnealingLR(opt, T_max=args.epochs)
    else: scheduler = StepLR(opt, step_size=40, gamma=0.9)

    print(f"\n=== [Phase 3] Start Training ===")
    opt.zero_grad() 
    best_val_mm = float('inf')

    for epoch in range(args.epochs):
        model.train()
        train_loss, train_hm, train_crd, train_srf, train_str, train_mm = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1:03d}/{args.epochs} [Train]", unit="batch") as tepoch:
            for i, (point, landmark, seg) in tepoch:
                point, landmark, seg = point.to(device), landmark.to(device), seg.to(device)

                with torch.no_grad():
                    centroid_tmp = torch.mean(point, axis=1, keepdim=True)
                    batch_m = torch.max(torch.sqrt(torch.sum((point - centroid_tmp) ** 2, axis=2)), axis=1)[0]
                    avg_m = torch.mean(batch_m).item()

                point_normal, landmark_normal = normalize_data(point, landmark)
                point_normal, augmented_landmark = ScaleAndTranslate(point_normal, landmark_normal)
                point_input = point_normal.permute(0, 2, 1)
                points_for_coords = point_normal # [B, N, 3]
                
                # =========================================================
                # 🚀 오프셋 vs 히트맵 예측 및 타겟 생성 스위칭
                # =========================================================
                if args.model == 'DeepLA':
                    # 1. 오프셋 전용 로직
                    out = model(point_input)
                    if isinstance(out, tuple):
                        pred_offsets, spa_loss, sem_loss = out
                    else:
                        pred_offsets = out
                        spa_loss, sem_loss = 0.0, 0.0
                        
                    # 모든 점이 각자 투표한 랜드마크 위치 계산: p_i + offset
                    # pred_votes: [B, N, K, 3]
                    pred_votes = points_for_coords.unsqueeze(2) + pred_offsets
                    # 최종 랜드마크는 모든 점의 의견을 평균 냄
                    pred_coords = pred_votes.mean(dim=1) 
                    
                    # 정답 오프셋 생성: L_j - p_i (거리 벡터)
                    # gt_offsets: [B, N, K, 3]
                    gt_offsets = augmented_landmark.unsqueeze(1) - points_for_coords.unsqueeze(2)
                    
                    # 히트맵 로스 변수를 오프셋 Smooth L1 로스로 대체하여 모니터링
                    loss_heatmap = F.smooth_l1_loss(pred_offsets, gt_offsets)
                    
                    # 구조 보조 로스 적용
                    loss_coord = F.smooth_l1_loss(pred_coords, augmented_landmark)
                    loss_surface = compute_point_to_plane_loss(pred_coords, points_for_coords, k=args.plane_knn)
                    loss_struct = compute_structural_loss(pred_coords, augmented_landmark)
                    
                    if isinstance(spa_loss, torch.Tensor): loss_surface += spa_loss
                    if isinstance(sem_loss, torch.Tensor): loss_struct += sem_loss * 0.1 

                else:
                    # 2. 기존 히트맵 로직 (PAConv, DeepPA 등)
                    if args.model == 'DeepPA':
                        prior_hint = paconv_prior(point_input)
                        pred_heatmap = model(point_input, prior_heatmap=prior_hint)
                    else:
                        pred_heatmap = model(point_input)
                    
                    pred_coords = get_differentiable_coords(points_for_coords, pred_heatmap, k=args.k_softargmax)
                    loss_heatmap = criterion(pred_heatmap, seg.permute(0, 2, 1).contiguous())
                    loss_coord = dynamic_focal_l1_loss(pred_coords, augmented_landmark, gamma=args.focal_gamma)
                    loss_surface = compute_point_to_plane_loss(pred_coords, points_for_coords, k=args.plane_knn)
                    loss_struct = compute_structural_loss(pred_coords, augmented_landmark)
                # =========================================================

                # 커리큘럼 러닝 가중치 적용
                stage1_epochs = args.epochs // 5 
                if epoch < stage1_epochs:
                    weights = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
                    total_loss = loss_heatmap
                else:
                    rand_weights = torch.rand(3).to(device)
                    rand_weights = (rand_weights / rand_weights.sum()) * 0.95
                    total_loss = (0.05 * loss_heatmap + 
                                  rand_weights[0] * loss_coord + 
                                  rand_weights[1] * loss_surface + 
                                  rand_weights[2] * loss_struct)
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
                
                tepoch.set_postfix(Loss=f"{total_loss.item():.4f}", Offset_or_HM=f"{loss_heatmap.item():.4f}", Crd=f"{loss_coord.item():.4f}", mm=f"{mm_error:.2f}")

        t_loss = train_loss / len(train_loader)
        t_hm   = train_hm / len(train_loader)
        t_crd  = train_crd / len(train_loader)
        t_srf  = train_srf / len(train_loader)
        t_str  = train_str / len(train_loader) 
        t_mm   = train_mm / len(train_loader)

        # -------------------------------------------------------------
        # [VALIDATION] 
        # -------------------------------------------------------------
        model.eval()
        val_loss, val_hm, val_crd, val_srf, val_str, val_mm = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        val_sample_count = 0 
        
        with torch.no_grad():
            with tqdm(enumerate(test_loader), desc=f"Epoch {epoch+1:03d}/{args.epochs} [Valid]", leave=False) as vepoch:
                for i, (point, landmark, seg) in vepoch:
                    point, landmark, seg = point.to(device), landmark.to(device), seg.to(device)

                    centroid_tmp = torch.mean(point, axis=1, keepdim=True)
                    batch_m = torch.max(torch.sqrt(torch.sum((point - centroid_tmp) ** 2, axis=2)), axis=1)[0]
                    avg_m = torch.mean(batch_m).item()

                    point_normal, landmark_normal = normalize_data(point, landmark)
                    point_input = point_normal.permute(0, 2, 1)
                    points_for_coords = point_normal

                    # =========================================================
                    # 🚀 Val 루프 (오프셋 vs 히트맵 스위칭)
                    # =========================================================
                    if args.model == 'DeepLA':
                        out = model(point_input)
                        pred_offsets = out[0] if isinstance(out, tuple) else out
                        
                        pred_votes = points_for_coords.unsqueeze(2) + pred_offsets
                        pred_coords = pred_votes.mean(dim=1) 
                        
                        gt_offsets = landmark_normal.unsqueeze(1) - points_for_coords.unsqueeze(2)
                        loss_heatmap = F.smooth_l1_loss(pred_offsets, gt_offsets)
                        loss_coord = F.smooth_l1_loss(pred_coords, landmark_normal)
                        loss_surface = compute_point_to_plane_loss(pred_coords, points_for_coords, k=args.plane_knn)
                        loss_struct = compute_structural_loss(pred_coords, landmark_normal)
                    else:
                        if args.model == 'DeepPA':
                            prior_hint = paconv_prior(point_input)
                            pred_heatmap = model(point_input, prior_heatmap=prior_hint)
                        else:
                            pred_heatmap = model(point_input)
                        
                        pred_coords = get_differentiable_coords(points_for_coords, pred_heatmap, k=args.k_softargmax)
                        loss_heatmap = criterion(pred_heatmap, seg.permute(0, 2, 1).contiguous())
                        loss_coord = dynamic_focal_l1_loss(pred_coords, landmark_normal, gamma=args.focal_gamma)
                        loss_surface = compute_point_to_plane_loss(pred_coords, points_for_coords, k=args.plane_knn)
                        loss_struct = compute_structural_loss(pred_coords, landmark_normal)
                    # =========================================================

                    stage1_epochs = args.epochs // 5
                    if epoch < stage1_epochs:
                        total_loss = loss_heatmap
                    else:
                        total_loss = 0.25 * loss_heatmap + 0.25 * loss_coord + 0.25 * loss_surface + 0.25 * loss_struct

                    val_loss += total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
                    val_hm   += loss_heatmap.item()
                    val_crd  += loss_coord.item()
                    val_srf  += loss_surface.item()
                    val_str  += loss_struct.item()
                    val_mm   += true_l1 = F.l1_loss(pred_coords, landmark_normal).item() * avg_m
                    
                    val_sample_count += point.size(0)
                    if val_sample_count >= 30: break

        num_val_batches = i + 1
        v_loss, v_hm, v_crd, v_srf, v_str, v_mm = val_loss / num_val_batches, val_hm / num_val_batches, val_crd / num_val_batches, val_srf / num_val_batches, val_str / num_val_batches, val_mm / num_val_batches

        print(f" [Train] Total: {t_loss:.4f} | Offset/HM: {t_hm:.4f} | Crd: {t_crd:.4f} | mm: {t_mm:.2f}")
        print(f" [Val]   Total: {v_loss:.4f} | Offset/HM: {v_hm:.4f} | Crd: {v_crd:.4f} | mm: {v_mm:.2f}")
        
        if v_mm < best_val_mm:
            best_val_mm = v_mm
            print(f" 🌟 [Best Model Saved] 최고 성능 갱신! 오차: {best_val_mm:.4f} mm")
            torch.save(model.state_dict(), os.path.join(paths['models'], 'model_best.t7'))

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(paths['models'], f'model_epoch_{epoch+1}.t7'))

        scheduler.step()
    
    print(f"\n=== Training Finished. Results at: {paths['root']} ===")
    print(f"🏆 최종 달성한 최고 성능(Best Validation Error): {best_val_mm:.4f} mm\n")

if __name__ == "__main__":
    args = parser.parse_args()
    _init_(args)
    train(args)