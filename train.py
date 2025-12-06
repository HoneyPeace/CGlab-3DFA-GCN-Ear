'''
@Author: Yuan Wang (Modified by Researcher 2)
@File: train.py
@Description: Normalization Fixed + Unified Folder Structure + Save Every 30 Samples
'''

import os
import time
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

from init import _init_
from My_args import parser
from dataset import FaceLandmarkData
from loss import AdaptiveWingLoss
from util import main_sample
from PAConv_model import PAConv
from augmentations import normalize_data, PointcloudScaleAndTranslate

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

# -----------------------------------------------------------------------------
# [기능 1] 통합 실험 ID 및 경로 생성
# -----------------------------------------------------------------------------
def get_experiment_paths(args):
    """
    구조: results/{exp_name}/{FPS_Sigma_Count}/
          ├── models/
          ├── npy_data/
          └── GT_Heatmaps/
    """
    project_dir = os.path.join(args.output_root, args.exp_name)
    os.makedirs(project_dir, exist_ok=True)

    setting_str = f"FPS{args.num_points}_sigma{args.sigma}"
    
    count = 1
    while True:
        run_name = f"{setting_str}_{count}"
        run_dir = os.path.join(project_dir, run_name)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            break
        count += 1
    
    # 하위 폴더 생성
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

# -----------------------------------------------------------------------------
# [기능 2] 3각도 히트맵 저장
# -----------------------------------------------------------------------------
def save_multiview_heatmap(points, heatmap, save_dir, sample_idx, landmark_idx, prefix):
    fig = plt.figure(figsize=(15, 5))
    views = [
        (131, 90, -90, "Front View"),
        (132, 45, -45, "Diagonal View"),
        (133, 0, -90,  "Side View")
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

# -----------------------------------------------------------------------------
# [기능 3] 데이터 백업 및 GT 시각화
# -----------------------------------------------------------------------------
def process_data_storage(dataset, prefix, paths):
    """
    dataset의 내용을 numpy로 변환하여 backup 폴더에 저장하고,
    GT 히트맵 일부를 시각화하여 GT_Heatmaps 폴더에 저장
    """
    shape_list, landmark_list, heatmap_list = [], [], []
    for i in range(len(dataset)):
        p, l, h = dataset[i]
        shape_list.append(p.numpy())
        landmark_list.append(l.numpy())
        heatmap_list.append(h.numpy())

    shape_arr = np.stack(shape_list)
    landmark_arr = np.stack(landmark_list)
    heatmap_arr = np.stack(heatmap_list)

    # 1. NPY Backup
    np.save(os.path.join(paths['npy_backup'], f"shape_{prefix}.npy"), shape_arr)
    np.save(os.path.join(paths['npy_backup'], f"landmark_{prefix}.npy"), landmark_arr)
    np.save(os.path.join(paths['npy_backup'], f"Heat_data_{prefix}.npy"), heatmap_arr)
    print(f"   [{prefix.upper()}] Backup Saved: {paths['npy_backup']}")

    # 2. GT Visualization
    vis_save_dir = os.path.join(paths['gt_heatmap'], prefix)
    os.makedirs(vis_save_dir, exist_ok=True)

    print(f"   [{prefix.upper()}] Saving GT Heatmaps (Every 30th Sample)...")
    
    # [수정] 0부터 끝까지 30 간격으로 루프 (0, 30, 60...)
    for idx in tqdm(range(0, len(shape_arr), 30), desc=f"   Saving GT {prefix}"):
        points_np = shape_arr[idx]
        heatmap_np = heatmap_arr[idx].T  # (L, N)
        
        # 해당 샘플의 모든 랜드마크 저장
        for lm_idx in range(heatmap_np.shape[0]):
            save_multiview_heatmap(points_np, heatmap_np[lm_idx], vis_save_dir, idx, lm_idx, prefix)

# -----------------------------------------------------------------------------
# [기능 4] 메인 학습 함수
# -----------------------------------------------------------------------------
def train(args):
    # 1. 스마트 모드 감지 (Split vs Separate)
    if not args.test_dataset_name or (args.train_dataset_name == args.test_dataset_name):
        MODE = "SPLIT"
        target_dataset = args.train_dataset_name
        args.test_dataset_name = args.train_dataset_name
        print(f"\n>>> [MODE] Split Mode (Dataset: {target_dataset})")
    else:
        MODE = "SEPARATE"
        print(f"\n>>> [MODE] Separate Mode")
        print(f"    Train: {args.train_dataset_name} / Test: {args.test_dataset_name}")

    # 2. 통합 경로 생성 (results/Ear_Project_Final/FPS.../)
    paths = get_experiment_paths(args)

    # 3. 데이터 생성 (Resample) - util.py의 기능 사용
    # 주의: util.py는 data_root/{dataset}-npy에 저장함. 이를 로드해서 paths['npy_backup']으로 옮길 것임.
    if args.need_resample:
        print("=== [Phase 1] Data Generation (Initial) ===")
        main_sample(args.num_points, args.seed, args.sigma, args.sample_way, args.train_dataset_name, args.data_root)
        if MODE == "SEPARATE":
            main_sample(args.num_points, args.seed, args.sigma, args.sample_way, args.test_dataset_name, args.data_root)

    # 4. 데이터 로드 및 분할
    print("=== [Phase 2] Loading & Backing up Data ===")
    
    if MODE == "SPLIT":
        full_dataset = FaceLandmarkData(data_root=args.data_root, partition='trainval', data=args.train_dataset_name)
        train_size = int(len(full_dataset) * 0.7)
        test_size = len(full_dataset) - train_size
        torch.manual_seed(args.dataset_seed)
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    else:
        train_dataset = FaceLandmarkData(data_root=args.data_root, partition='train', data=args.train_dataset_name)
        test_dataset = FaceLandmarkData(data_root=args.data_root, partition='test', data=args.test_dataset_name)

    # 5. 데이터 백업 수행 (이때 통합 폴더로 복사됨)
    process_data_storage(train_dataset, "train", paths)
    process_data_storage(test_dataset, "test", paths)

    # DataLoader 설정
    train_loader = DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    ScaleAndTranslate = PointcloudScaleAndTranslate()

    # 6. 모델 초기화
    model = PAConv(args, args.landmark_num).to(device)
    model.apply(weight_init)
    
    if args.loss == 'adaptive_wing': criterion = AdaptiveWingLoss()
    else: criterion = torch.nn.MSELoss()
        
    if args.use_sgd: opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=args.weight_decay)
    else: opt = optim.Adam(model.parameters(), lr=args.lr, eps=1e-08, weight_decay=args.weight_decay)
    
    if args.scheduler == 'cos': scheduler = CosineAnnealingLR(opt, T_max=args.epochs)
    else: scheduler = StepLR(opt, step_size=40, gamma=0.9)

    # 7. 학습 루프
    print(f"\n=== [Phase 3] Start Training ===")
    for epoch in range(args.epochs):
        model.train()
        loss_epoch = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch") as tepoch:
            for point, landmark, seg in tepoch:
                point    = point.to(device)            # (B, N, 3)
                landmark = landmark.to(device)         # (B, L, 3)
                seg      = seg.to(device)              # (B, N, L)

                # 1) 정규화
                point_normal = normalize_data(point)    # (B, N, 3)
                # 2) 증강 (Scale & Translate)
                point_normal = ScaleAndTranslate(point_normal)

                # 3) 입력 변환
                point_input = point_normal.permute(0, 2, 1)      # (B, 3, N)

                opt.zero_grad()
                pred_heatmap = model(point_input)                # (B, L, N)

                loss = criterion(pred_heatmap, seg.permute(0, 2, 1).contiguous())

                loss.backward()
                opt.step()
                
                loss_epoch += loss.item()
                tepoch.set_postfix(loss=loss.item())

        avg_loss = loss_epoch / len(train_loader)
        # print(f'Epoch: [{epoch+1}/{args.epochs}] Avg Loss: {avg_loss:.6f}') # tqdm에 통합

        # 모델 저장 (5에폭마다)
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