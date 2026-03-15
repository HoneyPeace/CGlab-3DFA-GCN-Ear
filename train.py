'''
@Author: Yuan Wang (Modified by Researcher 2 & AI Assistant)
@File: train.py
@Description: Gradient Accumulation + AWL (4-Loss Hybrid: HM + Crd + Srf + Struct)
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
# [수정] compute_structural_loss 임포트 추가
from loss import AdaptiveWingLoss, get_differentiable_coords, compute_point_to_plane_loss, compute_structural_loss
from util import main_sample
from PAConv_model import PAConv
from augmentations import normalize_data, PointcloudScaleAndTranslate

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# [추가] 불확실성 기반 자동 가중치 조절 모듈 (Kendall et al., CVPR 2018)
# =============================================================================
class AutomaticWeightedLoss(torch.nn.Module):
    def __init__(self, num_losses=4): # [수정] 4개의 로스로 변경
        super(AutomaticWeightedLoss, self).__init__()
        self.params = torch.nn.Parameter(torch.zeros(num_losses, requires_grad=True))

    def forward(self, losses):
        total_loss = 0
        for i, loss in enumerate(losses):
            total_loss += torch.exp(-self.params[i]) * loss + self.params[i]
        return total_loss
# =============================================================================

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

    if prefix == 'test':
        vis_save_dir = os.path.join(paths['gt_heatmap'], prefix)
        os.makedirs(vis_save_dir, exist_ok=True)
        print(f"   [{prefix.upper()}] Saving GT Heatmaps (Every 30th Sample)...")
        for idx in tqdm(range(0, len(shape_arr), 30), desc=f"   Saving GT {prefix}"):
            points_np = shape_arr[idx]
            heatmap_np = heatmap_arr[idx].T
            for lm_idx in range(heatmap_np.shape[0]):
                save_multiview_heatmap(points_np, heatmap_np[lm_idx], vis_save_dir, idx, lm_idx, prefix)
    else:
        print(f"   [{prefix.upper()}] GT Heatmap generation skipped (Requested).")

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
    if accum_steps == 1:
        print("    -> Operating in Standard Mode")
    else:
        print(f"    -> Effective Batch Size: {args.batch_size * accum_steps}")

    if args.need_resample:
        print("=== [Phase 1] Data Generation (Initial) ===")
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

    print("=== [Phase 2.5] Creating Experiment Paths ===")
    train_len = len(train_dataset)
    paths = get_experiment_paths(args, train_len)

    print("=== [Phase 2.6] Backing up Data ===")
    process_data_storage(train_dataset, "train", paths)
    process_data_storage(test_dataset, "test", paths)

    train_loader = DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, num_workers=0, batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    
    ScaleAndTranslate = PointcloudScaleAndTranslate()

    # 6. 모델 및 가중치 모듈 초기화
    model = PAConv(args, args.landmark_num).to(device)
    model.apply(weight_init)
    
    # [수정] 4개의 로스를 담당하도록 초기화
    awl = AutomaticWeightedLoss(num_losses=4).to(device)
    
    if args.loss == 'adaptive_wing': criterion = AdaptiveWingLoss()
    else: criterion = torch.nn.MSELoss()
        
    if args.use_sgd: 
        opt = optim.SGD([
            {'params': model.parameters()},
            {'params': awl.parameters(), 'weight_decay': 0}
        ], lr=args.lr*100, momentum=args.momentum, weight_decay=args.weight_decay)
    else: 
        opt = optim.Adam([
            {'params': model.parameters()},
            {'params': awl.parameters(), 'weight_decay': 0}
        ], lr=args.lr, eps=1e-08, weight_decay=args.weight_decay)
    
    if args.scheduler == 'cos': scheduler = CosineAnnealingLR(opt, T_max=args.epochs)
    else: scheduler = StepLR(opt, step_size=40, gamma=0.9)

    print(f"\n=== [Phase 3] Start Training with 4-Loss AWL ===")
    
    opt.zero_grad() 

    for epoch in range(args.epochs):
        # -------------------------------------------------------------
        # [TRAIN] 학습 루프
        # -------------------------------------------------------------
        model.train()
        # [수정] 구조적 로스(train_str) 변수 추가
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
                pred_heatmap = model(point_input)
                
                points_for_coords = point_input.permute(0, 2, 1)
                pred_coords = get_differentiable_coords(points_for_coords, pred_heatmap, k=args.k_softargmax)

                loss_heatmap = criterion(pred_heatmap, seg.permute(0, 2, 1).contiguous())
                loss_coord = F.l1_loss(pred_coords, augmented_landmark)
                loss_surface = compute_point_to_plane_loss(pred_coords, points_for_coords, k=args.plane_knn)
                
                # [신규 추가] 구조적 위상 로스 계산 (학습 시에는 augmentation된 정답 좌표 사용)
                loss_struct = compute_structural_loss(pred_coords, augmented_landmark)
                
                mm_error = loss_coord.item() * avg_m

                # [수정] 4개의 로스를 AWL에 전달
                total_loss = awl([loss_heatmap, loss_coord, loss_surface, loss_struct])
                
                loss = total_loss / accum_steps
                loss.backward()
                
                if (i + 1) % accum_steps == 0:
                    opt.step()
                    opt.zero_grad() 

                train_loss += total_loss.item()
                train_hm   += loss_heatmap.item()
                train_crd  += loss_coord.item()
                train_srf  += loss_surface.item()
                train_str  += loss_struct.item() # [추가]
                train_mm   += mm_error
                
                tepoch.set_postfix(Loss=f"{total_loss.item():.4f}", HM=f"{loss_heatmap.item():.4f}", Crd=f"{loss_coord.item():.4f}", Srf=f"{loss_surface.item():.4f}", Str=f"{loss_struct.item():.4f}", mm=f"{mm_error:.2f}")

        t_loss = train_loss / len(train_loader)
        t_hm   = train_hm / len(train_loader)
        t_crd  = train_crd / len(train_loader)
        t_srf  = train_srf / len(train_loader)
        t_str  = train_str / len(train_loader) # [추가]
        t_mm   = train_mm / len(train_loader)

        # -------------------------------------------------------------
        # [VALIDATION] 평가 루프
        # -------------------------------------------------------------
        model.eval()
        val_loss, val_hm, val_crd, val_srf, val_str, val_mm = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        with torch.no_grad():
            with tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Epoch {epoch+1:03d}/{args.epochs} [Valid]", unit="batch", leave=False) as vepoch:
                for i, (point, landmark, seg) in vepoch:
                    point, landmark, seg = point.to(device), landmark.to(device), seg.to(device)

                    centroid_tmp = torch.mean(point, axis=1, keepdim=True)
                    batch_m = torch.max(torch.sqrt(torch.sum((point - centroid_tmp) ** 2, axis=2)), axis=1)[0]
                    avg_m = torch.mean(batch_m).item()

                    point_normal, landmark_normal = normalize_data(point, landmark)
                    
                    point_input = point_normal.permute(0, 2, 1)
                    pred_heatmap = model(point_input)
                    
                    points_for_coords = point_input.permute(0, 2, 1)
                    pred_coords = get_differentiable_coords(points_for_coords, pred_heatmap, k=args.k_softargmax)

                    loss_heatmap = criterion(pred_heatmap, seg.permute(0, 2, 1).contiguous())
                    loss_coord = F.l1_loss(pred_coords, landmark_normal)
                    loss_surface = compute_point_to_plane_loss(pred_coords, points_for_coords, k=args.plane_knn)
                    
                    # [신규 추가] 평가 시 구조적 로스 계산 (정규화된 정답 좌표 사용)
                    loss_struct = compute_structural_loss(pred_coords, landmark_normal)
                    
                    mm_error = loss_coord.item() * avg_m
                    
                    # [수정] 4개의 로스 전달
                    total_loss = awl([loss_heatmap, loss_coord, loss_surface, loss_struct])

                    val_loss += total_loss.item()
                    val_hm   += loss_heatmap.item()
                    val_crd  += loss_coord.item()
                    val_srf  += loss_surface.item()
                    val_str  += loss_struct.item() # [추가]
                    val_mm   += mm_error

        v_loss = val_loss / len(test_loader)
        v_hm   = val_hm / len(test_loader)
        v_crd  = val_crd / len(test_loader)
        v_srf  = val_srf / len(test_loader)
        v_str  = val_str / len(test_loader) # [추가]
        v_mm   = val_mm / len(test_loader)

        # -------------------------------------------------------------
        # [PRINT] 결과 및 가중치 출력 
        # -------------------------------------------------------------
        print(f" [Train] 전체 Loss: {t_loss:.4f} | HM: {t_hm:.4f} | Crd: {t_crd:.4f} | Srf: {t_srf:.4f} | Struct: {t_str:.4f} | mm: {t_mm:.2f}")
        print(f" [Val]   전체 Loss: {v_loss:.4f} | HM: {v_hm:.4f} | Crd: {v_crd:.4f} | Srf: {v_srf:.4f} | Struct: {v_str:.4f} | mm: {v_mm:.2f}")
        
        # [수정] 4개의 가중치 출력
        effective_weights = torch.exp(-awl.params).detach().cpu().numpy()
        print(f" [AWL Weights] HM: {effective_weights[0]:.4f} | Crd: {effective_weights[1]:.4f} | Srf: {effective_weights[2]:.4f} | Struct: {effective_weights[3]:.4f}\n")

        if (epoch + 1) % 5 == 0:
            filename = f'model_epoch_{epoch+1}.t7'
            save_path = os.path.join(paths['models'], filename)
            torch.save(model.state_dict(), save_path)

        scheduler.step()
    
    print(f"\n=== Training Finished. Results at: {paths['root']} ===\n")

if __name__ == "__main__":
    args = parser.parse_args()
    _init_(args)
    train(args)