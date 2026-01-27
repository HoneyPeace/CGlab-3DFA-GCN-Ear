'''
@Author: Yuan Wang (Modified by Researcher 2)
@File: train.py
@Description: Gradient Accumulation + Unified Folder Structure + NO Train Heatmaps
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
from augmentations import normalize_data, PointcloudScaleAndTranslate, PointcloudRotation


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
def get_experiment_paths(args, train_len):
    """
    구조: results/{exp_name}/{FPS_Sigma_Batch_Train_[Tag]_Count}/
    """
    project_dir = os.path.join(args.output_root, args.exp_name)
    os.makedirs(project_dir, exist_ok=True)

    if args.accumulation_steps > 1:
        batch_str = f"{args.batch_size}x{args.accumulation_steps}"
    else:
        batch_str = f"{args.batch_size}"

    base_str = f"FPS{args.num_points}_sigma{args.sigma}_batch{batch_str}_train{train_len}"
    
    # 태그가 있으면 붙이고, 없으면 안 붙임
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

# -----------------------------------------------------------------------------
# [기능 2] 3각도 히트맵 저장
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# [<<<< 수정됨] 법선 벡터 시각화 함수 (Auto-Scaling 적용)
# -----------------------------------------------------------------------------
def save_multiview_normal(points, normals, save_dir, sample_idx, prefix):
    """
    점(Blue)과 법선벡터(Red Arrow)를 시각화 (데이터 크기에 맞춰 화살표 길이 자동 조절)
    """
    fig = plt.figure(figsize=(30, 10))
    
    views = [
        (131, 90, -100, "Front"),
        (132, 30, 120,  "Side"),
        (133, 45, -45, "Downside")
    ]
    
    # [핵심 수정 1] 데이터의 스케일(범위)을 계산하여 화살표 길이를 동적으로 설정
    # 점들이 퍼져있는 전체 크기(Max-Min)의 약 8% 길이로 화살표를 그림
    # 이렇게 하면 데이터가 mm단위든 m단위든 무조건 눈에 보임
    coord_range = np.ptp(points, axis=0).max() 
    arrow_length = coord_range * 0.08          

    # [핵심 수정 2] 너무 빽빽하면 징그러우니 적당히 건너뛰기
    # 점 개수가 4000개라면 step=15 정도로 해서 약 260개만 그리기
    step = 15 
    
    for pos, elev, azim, title in views:
        ax = fig.add_subplot(pos, projection='3d')
        
        # 1. 점 찍기 (파란색)
        # alpha를 0.2로 낮춰서 점을 아주 흐리게 만들고, 화살표를 돋보이게 함
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='b', alpha=0.2)
        
        # 2. 법선 벡터 화살표 그리기 (빨간색)
        ax.quiver(
            points[::step, 0], points[::step, 1], points[::step, 2],    # 시작점
            normals[::step, 0], normals[::step, 1], normals[::step, 2], # 방향
            length=arrow_length,  # [수정됨] 자동 계산된 길이 사용
            normalize=True,       # 벡터 길이를 1로 맞춘 뒤 length를 곱함
            color='r', 
            linewidth=1.5         # [수정됨] 선 두께를 1.5로 키움
        )
        
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{title} - Normals\n(Len: {arrow_length:.2f})") # 타이틀에 길이 정보 표시
        ax.axis('off')
    
    filename = f"{prefix}_Normal_S{sample_idx:03d}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=100, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# [기능 3] 데이터 백업 및 GT 시각화 (수정됨: Train 히트맵 스킵)
# -----------------------------------------------------------------------------
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

    # 1. NPY Backup (이건 무조건 저장해야 나중에 분석 가능)
    np.save(os.path.join(paths['npy_backup'], f"shape_{prefix}.npy"), shape_arr)
    np.save(os.path.join(paths['npy_backup'], f"landmark_{prefix}.npy"), landmark_arr)
    np.save(os.path.join(paths['npy_backup'], f"Heat_data_{prefix}.npy"), heatmap_arr)
    print(f"   [{prefix.upper()}] Backup Saved: {paths['npy_backup']}")

    # 2. GT Visualization
    # [수정된 부분] prefix가 'test'일 때만 히트맵 이미지를 저장합니다.
    if prefix == 'test':
# (1) 히트맵 저장 경로
        vis_save_dir = os.path.join(paths['gt_heatmap'], prefix)
        os.makedirs(vis_save_dir, exist_ok=True)
        
        # (2) [<<<< 수정 2] 노말 벡터 저장 경로 생성 (GT_Normals 폴더)
        normal_save_dir = os.path.join(paths['root'], 'GT_Normals')
        os.makedirs(normal_save_dir, exist_ok=True)

        print(f"   [{prefix.upper()}] Saving GT Heatmaps & Normals (Every 30th Sample)...")
        
        # 30개 간격으로 저장
        for idx in tqdm(range(0, len(shape_arr), 30), desc=f"   Saving GT {prefix}"):
            points_np = shape_arr[idx]
            heatmap_np = heatmap_arr[idx].T
            
            for lm_idx in range(heatmap_np.shape[0]):
                save_multiview_heatmap(points_np, heatmap_np[lm_idx], vis_save_dir, idx, lm_idx, prefix)
# --- [B] [<<<< 수정 2] 3-View 노말 벡터 저장 ---
            if points_np.shape[1] >= 6:
                xyz = points_np[:, :3]
                normals = points_np[:, 3:6]
                save_multiview_normal(xyz, normals, normal_save_dir, idx, prefix)
    else:
        # Train 데이터 등은 저장하지 않음
        print(f"   [{prefix.upper()}] GT Heatmap generation skipped (Requested).")

# -----------------------------------------------------------------------------
# [기능 4] 메인 학습 함수
# -----------------------------------------------------------------------------
def train(args):
    accum_steps = args.accumulation_steps
    
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

    print(f">>> [Gradient Accumulation] Steps: {accum_steps}")
    if accum_steps == 1:
        print("    -> Operating in Standard Mode")
    else:
        print(f"    -> Effective Batch Size: {args.batch_size * accum_steps}")

 # 3. 데이터 생성 (Resample) - [수정] Train/Test 분리 생성 지시
    if args.need_resample:
        print("=== [Phase 1] Data Generation (Initial) ===")
        
        # (1) Train 데이터만 읽어서 -> shape_train.npy 로 저장
        main_sample(args.num_points, args.seed, args.sigma, args.sample_way, 
                    args.train_dataset_name, args.data_root, partition='train')
        
        # (2) Test 데이터만 읽어서 -> shape_test.npy 로 저장
        main_sample(args.num_points, args.seed, args.sigma, args.sample_way, 
                    args.test_dataset_name, args.data_root, partition='test')

    # 4. 데이터 로드 및 분할
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

    # [수정] 데이터 로드 완료 후 폴더 생성
    print("=== [Phase 2.5] Creating Experiment Paths ===")
    train_len = len(train_dataset)
    paths = get_experiment_paths(args, train_len)

    # 5. 데이터 백업 수행 (Train 히트맵 스킵 로직 적용됨)
    print("=== [Phase 2.6] Backing up Data ===")
    process_data_storage(train_dataset, "train", paths)
    process_data_storage(test_dataset, "test", paths)

    # DataLoader 설정
    train_loader = DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    ScaleAndTranslate = PointcloudScaleAndTranslate()
    # Rotation 추가: 0.26라디안 = 약 15도 (정합된 데이터에 안전한 범위)
    #Rotate = PointcloudRotation(angle_range=0.26)
    
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
    
    opt.zero_grad() 

    for epoch in range(args.epochs):
        model.train()
        loss_epoch = 0.0
        
        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch") as tepoch:
            for i, (point, landmark, seg) in tepoch:
                point    = point.to(device)
                landmark = landmark.to(device)
                seg      = seg.to(device)

                point_normal = normalize_data(point)
                point_normal = ScaleAndTranslate(point_normal)
                
                #point_normal = Rotate(point_normal) # Rotation 적용 (XYZ와 Normal 모두 회전함!)
                point_input = point_normal.permute(0, 2, 1)

                pred_heatmap = model(point_input)

                loss = criterion(pred_heatmap, seg.permute(0, 2, 1).contiguous())
                loss = loss / accum_steps
                loss.backward()
                
                if (i + 1) % accum_steps == 0:
                    opt.step()
                    opt.zero_grad() 

                current_loss_val = loss.item() * accum_steps
                loss_epoch += current_loss_val
                tepoch.set_postfix(loss=current_loss_val)

        if (epoch + 1) % 5 == 0:
            filename = f'model_epoch_{epoch+1}.t7'
            save_path = os.path.join(paths['models'], filename)
            torch.save(model.state_dict(), save_path)

        scheduler.step()
    
    print(f"\n=== Training Finished. Results at: {paths['root']} ===\n")

if __name__ == "__main__":
    args = parser.parse_args()
    _init_(args)
    # 태그 입력 받는 부분 삭제됨 (My_args 의존)
    train(args)