'''
@Author: Researcher 5
@File: train.py
@Description: Final Version (Import Fixed + Fast Backup + Gradient Accumulation)
'''

import os
import sys
import shutil # 고속 백업용
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
# [수정 완료] generate_npy_from_folder 제거 (util.py에 없는 함수임)
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
# 실험 폴더 생성
# -----------------------------------------------------------------------------
def get_experiment_paths(args, train_data_count):
    project_dir = os.path.join(args.output_root, args.exp_name)
    os.makedirs(project_dir, exist_ok=True)

    # 폴더명에 실제 효과 배치를 표기 (B * Accum)
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

# -----------------------------------------------------------------------------
# 3각도 히트맵 저장
# -----------------------------------------------------------------------------
def save_multiview_heatmap(points, heatmap, save_dir, sample_name, landmark_idx, prefix):
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

    filename = f"{prefix}_{sample_name}_L{landmark_idx:02d}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=100, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# [최적화] 데이터 백업 (고속 복사 + Train 시각화 건너뛰기)
# -----------------------------------------------------------------------------
def process_data_storage(dataset, prefix, paths):
    if len(dataset) == 0: return

    print(f"   [Fast Backup] Copying {prefix} NPY files...")
    
    src_dir = dataset.npy_dir
    dst_dir = paths['npy_backup']
    
    files_to_copy = [
        f"shape_{prefix}.npy", 
        f"landmark_{prefix}.npy", 
        f"Heat_data_{prefix}.npy", 
        f"names_{prefix}.npy"
    ]
    
    for f in files_to_copy:
        src_file = os.path.join(src_dir, f)
        dst_file = os.path.join(dst_dir, f)
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
        else:
            print(f"     Warning: {f} not found for backup.")

    # GT Visualization (Test만 수행)
    if prefix == 'test':
        vis_save_dir = paths['gt_heatmap'] # 폴더 바로 아래 저장
        print(f"   [{prefix.upper()}] Saving GT Heatmaps (Every 30th)...")
        for idx in tqdm(range(0, len(dataset), 30), desc=f"   Saving GT"):
            points_np = dataset.points[idx]
            heatmap_np = dataset.heatmaps[idx].T
            name = str(dataset.names[idx])
            
            for lm_idx in range(heatmap_np.shape[0]):
                save_multiview_heatmap(points_np, heatmap_np[lm_idx], vis_save_dir, name, lm_idx, prefix)
    else:
        print(f"   [{prefix.upper()}] Skipping GT visualization for speed.")

# -----------------------------------------------------------------------------
# 메인 학습 함수
# -----------------------------------------------------------------------------
def train(args):
    # 1. 데이터 생성 (util.main_sample 사용)
    if args.need_resample:
        main_sample(args.num_points, args.seed, args.sigma, args.sample_way, 'train', args.data_root)
        main_sample(args.num_points, args.seed, args.sigma, args.sample_way, 'test', args.data_root)

    print("\n=== Loading Data ===")
    train_dataset = FaceLandmarkData(args.data_root, partition='train')
    test_dataset = FaceLandmarkData(args.data_root, partition='test')

    if len(train_dataset) == 0:
        print("Error: Train dataset is empty.")
        return

    paths = get_experiment_paths(args, len(train_dataset))

    process_data_storage(train_dataset, "train", paths)
    process_data_storage(test_dataset, "test", paths)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    ScaleAndTranslate = PointcloudScaleAndTranslate()

    model = PAConv(args, args.landmark_num).to(device)
    model.apply(weight_init)
    
    if args.loss == 'adaptive_wing': criterion = AdaptiveWingLoss()
    else: criterion = torch.nn.MSELoss()
    
    opt = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(opt, step_size=40, gamma=0.9)

    eff_batch = args.batch_size * args.accum_iter
    print(f"\n=== Start Training ===")
    print(f"    Physical Batch: {args.batch_size}")
    print(f"    Accumulation  : {args.accum_iter}")
    print(f"    Total Batch   : {eff_batch} (Shown in Folder Name)\n")

    opt.zero_grad() 

    for epoch in range(args.epochs):
        model.train()
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch") as tepoch:
            for i, (point, landmark, seg, name) in enumerate(tepoch):
                point, seg = point.to(device), seg.to(device)

                point = normalize_data(point)
                point = ScaleAndTranslate(point)
                point_input = point.permute(0, 2, 1)

                pred_heatmap = model(point_input)
                loss = criterion(pred_heatmap, seg.permute(0, 2, 1).contiguous())

                # [중요] 그래디언트 누적
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