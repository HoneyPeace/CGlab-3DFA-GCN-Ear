'''
@Author: Researcher 5 (Fixed Logic - Loop Unpacking)
@File: train.py
@Description: Handles variable number of dataset returns (4 items instead of 3).
'''

import os
import sys
import shutil
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from init import _init_
from My_args import parser
from dataset import FaceLandmarkData
from loss import AdaptiveWingLoss
from util import main_sample
from PAConv_model import PAConv
from augmentations import normalize_data, PointcloudScaleAndTranslate

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

def get_experiment_paths(args, train_data_count):
    project_dir = os.path.join(args.output_root, args.exp_name)
    os.makedirs(project_dir, exist_ok=True)

    real_batch = args.batch_size * args.accum_iter
    setting_str = f"FPS{args.num_points}_sigma{args.sigma}_B{real_batch}_N{train_data_count}"
    
    if args.tag:
        setting_str += f"_{args.tag}"
    
    count = 1
    while True:
        run_name = f"{setting_str}_Run_{count}"
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
    for k, p in paths.items(): os.makedirs(p, exist_ok=True)
        
    print(f"\n>>> [Experiment Created]")
    print(f"    ID   : {run_name}")
    print(f"    Path : {os.path.abspath(run_dir)}\n")
    return paths

def process_data_storage(dataset, prefix, paths):
    if len(dataset) == 0: return

    print(f"   [Backup] Copying {prefix} NPY files...")
    if hasattr(dataset, 'data_root') and hasattr(dataset, 'DATA'):
        src_dir = os.path.join(dataset.data_root, f"{dataset.DATA}_npy")
    else:
        src_dir = os.path.join(args.data_root, f"{prefix}_npy")

    dst_dir = paths['npy_backup']
    
    files_to_copy = [f"shape_{prefix}.npy", f"landmark_{prefix}.npy", f"Heat_data_{prefix}.npy", f"names_{prefix}.npy"]
    
    for f in files_to_copy:
        src_file = os.path.join(src_dir, f)
        dst_file = os.path.join(dst_dir, f)
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)

    if prefix == 'test':
        vis_save_dir = paths['gt_heatmap']
        print(f"   [{prefix.upper()}] Saving GT Heatmaps (Every 30th)...")
        for idx in tqdm(range(0, len(dataset), 30), desc=f"   Saving GT"):
            sample = dataset[idx]
            
            if isinstance(sample, (tuple, list)):
                points_raw = sample[0]
                heatmap_raw = sample[2]
            else:
                points_raw = sample
                heatmap_raw = None

            if hasattr(points_raw, 'cpu'):
                points_np = points_raw.detach().cpu().numpy()
            else:
                points_np = points_raw
                
            if hasattr(heatmap_raw, 'cpu'):
                heatmap_np = heatmap_raw.detach().cpu().numpy().T
            elif heatmap_raw is not None:
                heatmap_np = heatmap_raw.T
            else:
                continue

            name = str(idx) 
            
            for lm_idx in range(heatmap_np.shape[0]):
                save_multiview_heatmap(points_np, heatmap_np[lm_idx], vis_save_dir, name, lm_idx, prefix)
    else:
        print(f"   [{prefix.upper()}] Skipping GT visualization for speed.")

def save_multiview_heatmap(points, heatmap, save_dir, sample_name, landmark_idx, prefix):
    fig = plt.figure(figsize=(30, 10))
    views = [(131, 90, -100, "Front"), (132, 30, 120, "Side"), (133, 45, -45, "Downside")]
    for pos, elev, azim, title in views:
        ax = fig.add_subplot(pos, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=heatmap, cmap='jet', s=15, alpha=0.8)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.axis('off')
    filename = f"{prefix}_{sample_name}_L{landmark_idx:02d}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=100, bbox_inches='tight')
    plt.close()

def train(args):
    # 1. 데이터 생성
    if args.need_resample:
        print("=== [Phase 1] Data Generation ===")
        main_sample(args.num_points, args.seed, args.sigma, args.sample_way, 'train', args.data_root)
        main_sample(args.num_points, args.seed, args.sigma, args.sample_way, 'test', args.data_root)

    # 2. 데이터 로드
    print("\n=== [Phase 2] Loading Data from Folders ===")
    train_dataset = FaceLandmarkData(args.data_root, partition='train')
    test_dataset = FaceLandmarkData(args.data_root, partition='test')

    if len(train_dataset) == 0:
        print("Error: Train dataset is empty. Check 'Data/train' folder.")
        return

    print(f"   Train Samples: {len(train_dataset)} (Loaded directly from 'train' folder)")
    print(f"   Test  Samples: {len(test_dataset)} (Loaded directly from 'test' folder)")

    # 3. 실험 폴더 생성
    paths = get_experiment_paths(args, len(train_dataset))

    # 4. 백업 수행
    process_data_storage(train_dataset, "train", paths)
    process_data_storage(test_dataset, "test", paths)

    # 5. DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    ScaleAndTranslate = PointcloudScaleAndTranslate()

    # 6. 모델
    model = PAConv(args, args.landmark_num).to(device)
    model.apply(weight_init)
    
    if args.loss == 'adaptive_wing': criterion = AdaptiveWingLoss()
    else: criterion = torch.nn.MSELoss()
    
    opt = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(opt, step_size=40, gamma=0.9)

    eff_batch = args.batch_size * args.accum_iter
    print(f"\n=== Start Training (Effective Batch: {eff_batch}) ===")

    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch") as tepoch:
            # [★수정됨] 데이터가 3개든 4개든 5개든 상관없이 앞에서 3개만 가져오도록 변경
            for i, batch_data in enumerate(tepoch): 
                # 안전하게 인덱스로 접근
                point = batch_data[0]
                landmark = batch_data[1]
                seg = batch_data[2]
                # 뒤에 있는 이름(name) 정보 등은 학습에 안 쓰므로 무시

                point, seg = point.to(device), seg.to(device)

                point = normalize_data(point)
                point = ScaleAndTranslate(point)
                point_input = point.permute(0, 2, 1)

                pred_heatmap = model(point_input)
                loss = criterion(pred_heatmap, seg.permute(0, 2, 1).contiguous())

                # Gradient Accumulation
                loss = loss / args.accum_iter
                loss.backward()

                if (i + 1) % args.accum_iter == 0:
                    opt.step()
                    opt.zero_grad()
                
                tepoch.set_postfix(loss=loss.item() * args.accum_iter)

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(paths['models'], f'model_epoch_{epoch+1}.t7'))
    
    print(f"\n[Done] Results at: {os.path.abspath(paths['root'])}")

if __name__ == "__main__":
    args = parser.parse_args()
    _init_(args)
    train(args)