'''
@Author: Yuan Wang (Modified by Researcher 2)
@File: train.py
@Description: Normalization Fixed + Smart Split/Separate Mode
'''

import os
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

from init import *
from My_args import parser
from dataset import FaceLandmarkData
from loss import AdaptiveWingLoss
from util import main_sample
from PAConv_model import PAConv
from init import _init_

# [중요] 정규화 함수 임포트
from augmentations import normalize_data 

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# [기능 1] 유니크 ID 생성기
# -----------------------------------------------------------------------------
def get_experiment_id(backup_root, dataset_name, setting_str):
    date_str = time.strftime("%Y%m%d")
    base_name = f"{setting_str}_{date_str}"
    count = 1
    while True:
        exp_id = f"{base_name}_{count}"
        check_path = os.path.join(backup_root, dataset_name, exp_id)
        if not os.path.exists(check_path):
            return exp_id
        count += 1

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
        # 점 크기 15, 투명도 0.8로 설정하여 2048개여도 잘 보이게 함
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=heatmap, cmap='jet', s=15, alpha=0.8)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.axis('off')

    filename = f"{prefix}_S{sample_idx:03d}_L{landmark_idx:02d}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=100, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# [기능 3] 데이터 백업 및 시각화 저장
# -----------------------------------------------------------------------------
def process_data_storage(dataset, prefix, paths, args):
    shape_list, landmark_list, heatmap_list = [], [], []
    for i in range(len(dataset)):
        p, l, h = dataset[i]
        shape_list.append(p.numpy())
        landmark_list.append(l.numpy())
        heatmap_list.append(h.numpy())

    shape_arr = np.stack(shape_list)
    landmark_arr = np.stack(landmark_list)
    heatmap_arr = np.stack(heatmap_list)

    # Backup NPY
    np.save(os.path.join(paths['backup_npy'], f"shape_{prefix}.npy"), shape_arr)
    np.save(os.path.join(paths['backup_npy'], f"landmark_{prefix}.npy"), landmark_arr)
    np.save(os.path.join(paths['backup_npy'], f"Heat_data_{prefix}.npy"), heatmap_arr)
    print(f"[{prefix.upper()}] NPY Saved to Backup: {paths['backup_npy']}")

    # Save GT Heatmap Images
    vis_dir_name = f"GT_{prefix.capitalize()}"
    vis_save_dir = os.path.join(paths['heatmap_exp'], vis_dir_name)
    os.makedirs(vis_save_dir, exist_ok=True)

    print(f"[{prefix.upper()}] Saving GT Heatmap Images (Top 5 Samples)...")
    for idx in range(min(5, len(shape_arr))):
        points_np = shape_arr[idx]
        heatmap_np = heatmap_arr[idx].T
        # 모든 랜드마크 저장
        for lm_idx in range(heatmap_np.shape[0]):
            save_multiview_heatmap(points_np, heatmap_np[lm_idx], vis_save_dir, idx, lm_idx, prefix)
    print(f"[{prefix.upper()}] Visualization Saved.")


# -----------------------------------------------------------------------------
# [기능 4] 메인 학습 함수
# -----------------------------------------------------------------------------
def train(args):
    # 1. 경로 설정
    base_root = os.path.dirname(args.output_root)
    heatmap_root = os.path.join(base_root, 'Heatmap')
    backup_root = os.path.join(base_root, 'backup')
    
    # 2. 스마트 모드 감지 (Split vs Separate)
    # 이름이 같거나 Test 이름이 비어있으면 Split Mode
    if not args.test_dataset_name or (args.train_dataset_name == args.test_dataset_name):
        MODE = "SPLIT"
        target_dataset = args.train_dataset_name
        # 이름 통일
        args.test_dataset_name = args.train_dataset_name
        print(f"\n>>> [MODE CHECK] Same Dataset Detected ('{target_dataset}')")
        print(">>> Action: Split Mode (One dataset -> 7:3 Split)")
    else:
        MODE = "SEPARATE"
        print(f"\n>>> [MODE CHECK] Different Datasets Detected.")
        print(f"    - Train: '{args.train_dataset_name}'")
        print(f"    - Test : '{args.test_dataset_name}'")
        print(">>> Action: Separate Mode (Train on A, Test on B)")
        target_dataset = args.train_dataset_name # 저장은 Train셋 이름 기준

    # 3. 실험 ID 및 경로 생성
    setting_str = f"FPS{args.num_points}_sigma{args.sigma}"
    exp_id = get_experiment_id(backup_root, target_dataset, setting_str)
    
    backup_exp_dir = os.path.join(backup_root, target_dataset, exp_id)
    heatmap_exp_dir = os.path.join(heatmap_root, target_dataset, exp_id)
    
    backup_models_dir = os.path.join(backup_exp_dir, 'models')
    backup_npy_dir = os.path.join(backup_exp_dir, 'npy_data')
    
    os.makedirs(backup_models_dir, exist_ok=True)
    os.makedirs(backup_npy_dir, exist_ok=True)
    os.makedirs(heatmap_exp_dir, exist_ok=True)
    
    print(f"\n>>> ID: {exp_id}")
    print(f">>> Backup Path : {backup_exp_dir}")
    print(f">>> Heatmap Path: {heatmap_exp_dir}\n")

    paths = {'backup_npy': backup_npy_dir, 'heatmap_exp': heatmap_exp_dir}

    # 4. 데이터 생성 (Resample)
    if args.need_resample:
        print("=== [Phase 1] Data Generation ===")
        if MODE == "SPLIT":
            # 하나만 생성
            main_sample(args.num_points, args.seed, args.sigma, args.sample_way, args.train_dataset_name)
        else:
            # 둘 다 생성
            main_sample(args.num_points, args.seed, args.sigma, args.sample_way, args.train_dataset_name)
            main_sample(args.num_points, args.seed, args.sigma, args.sample_way, args.test_dataset_name)

    # 5. 데이터 로드 및 저장
    print("=== [Phase 2] Loading & Storing Data ===")
    
    if MODE == "SPLIT":
        # 하나를 로드해서 쪼갬
        full_dataset = FaceLandmarkData(data_root=args.data_root, partition='trainval', data=args.train_dataset_name)
        train_size = int(len(full_dataset) * 0.7)
        test_size = len(full_dataset) - train_size
        torch.manual_seed(args.dataset_seed)
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    else:
        # 각각 로드
        train_dataset = FaceLandmarkData(data_root=args.data_root, partition='train', data=args.train_dataset_name)
        test_dataset = FaceLandmarkData(data_root=args.data_root, partition='test', data=args.test_dataset_name)

    # 데이터 백업 및 시각화 수행
    process_data_storage(train_dataset, "train", paths, args)
    process_data_storage(test_dataset, "test", paths, args)

    train_loader = DataLoader(train_dataset, num_workers=4, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # 6. 모델 설정
    model = PAConv(args, args.landmark_num).to(device)
    model.apply(weight_init)
    
    if args.loss == 'adaptive_wing': criterion = AdaptiveWingLoss()
    else: criterion = torch.nn.MSELoss()
        
    if args.use_sgd: opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=args.weight_decay)
    else: opt = optim.Adam(model.parameters(), lr=args.lr, eps=1e-08, weight_decay=args.weight_decay)
    
    scheduler = StepLR(opt, step_size=40, gamma=0.9)

    # 7. 학습 루프
    print(f"\n=== [Phase 3] Start Training ===")
    for epoch in range(args.epochs):
        model.train()
        loss_epoch = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch") as tepoch:
            for point, landmark, seg in tepoch:
                point = point.to(device)
                landmark = landmark.to(device)
                seg = seg.to(device)
                
                # [복구 완료] 정규화 (Normalization) 적용
                # 데이터를 0~1 단위로 줄여서 학습 효율 극대화
                point_normalized = normalize_data(point)
                
                opt.zero_grad()
                pred_heatmap = model(point_normalized.permute(0, 2, 1))
                loss = criterion(pred_heatmap, seg.permute(0, 2, 1).contiguous())
                loss.backward()
                opt.step()
                
                loss_epoch += loss.item()
                tepoch.set_postfix(loss=loss.item())

        avg_loss = loss_epoch / len(train_loader)
        print(f'Epoch: [{epoch+1}/{args.epochs}] Avg Loss: {avg_loss:.6f}')

        if (epoch + 1) % 5 == 0:
            filename = f'model_epoch_{epoch+1}.t7'
            save_path = os.path.join(backup_models_dir, filename)
            torch.save(model.state_dict(), save_path)
            print(f"Model Saved: {save_path}")
        
        scheduler.step()

if __name__ == "__main__":
    args = parser.parse_args()
    _init_(args)
    train(args)