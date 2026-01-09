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
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

from init import _init_
from My_args import parser
# from dataset import FaceLandmarkData  <-- 이제 안 씀
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

    setting_str = f"FPS{args.num_points}_sigma{int(args.sigma)}"
    
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
    fig = plt.figure(figsize=(30, 10))
    views = [
        (131, 90, -100, "Front"),
        (132, 30, 120,  "Side"),
        (133, 45, -45, "Downside")
    ]
    for pos, elev, azim, title in views:
        ax = fig.add_subplot(pos, projection='3d')
        # heatmap 값에 따라 색상 매핑 (0~1 사이 값이라 가정)
        p = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=heatmap, cmap='jet', s=15, alpha=0.8, vmin=0, vmax=1)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.axis('off')

    filename = f"{prefix}_S{sample_idx:03d}_L{landmark_idx:02d}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=100, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# [기능 3] 데이터 백업 및 GT 시각화 (수정됨)
# -----------------------------------------------------------------------------
def process_data_storage(dataset, prefix, paths):
    """
    TensorDataset에서 데이터를 꺼내 저장하고 시각화합니다.
    (TensorDataset.tensors 속성을 직접 사용하여 속도 최적화)
    """
    # TensorDataset에서 원본 텐서들을 바로 가져옵니다. (CPU로 이동 후 numpy 변환)
    # dataset.tensors = (shape_tensor, landmark_tensor, heatmap_tensor)
    shape_arr = dataset.tensors[0].numpy()
    landmark_arr = dataset.tensors[1].numpy()
    heatmap_arr = dataset.tensors[2].numpy()

    # 1. NPY Backup
    np.save(os.path.join(paths['npy_backup'], f"shape_{prefix}.npy"), shape_arr)
    np.save(os.path.join(paths['npy_backup'], f"landmark_{prefix}.npy"), landmark_arr)
    np.save(os.path.join(paths['npy_backup'], f"Heat_data_{prefix}.npy"), heatmap_arr)
    print(f"   [{prefix.upper()}] Backup Saved: {paths['npy_backup']}")

    # 2. GT Visualization
    vis_save_dir = os.path.join(paths['gt_heatmap'], prefix)
    os.makedirs(vis_save_dir, exist_ok=True)

    print(f"   [{prefix.upper()}] Saving GT Heatmaps (Every 30th Sample)...")
    
    # 0부터 끝까지 30 간격으로 루프
    # 시각화할 때 시간이 오래 걸릴 수 있으니 너무 많으면 간격을 늘리세요 (현재 30)
    for idx in tqdm(range(0, len(shape_arr), 30), desc=f"   Saving GT {prefix}"):
        points_np = shape_arr[idx]
        heatmap_np = heatmap_arr[idx].T  # (L, N) -> 보통 히트맵 차원이 이렇게 됨을 가정
        
        # 모든 랜드마크를 다 찍으면 너무 많으니 첫 번째(0번) 랜드마크만 샘플로 저장하거나,
        # 꼭 필요한 경우에만 전체 루프를 도세요. 여기선 0번과 10번만 찍도록 예시를 듭니다.
        # (전부 다 찍으려면 range(heatmap_np.shape[0]) 그대로 사용)
        target_lms = [0, 5, 10, 15] if heatmap_np.shape[0] > 15 else range(heatmap_np.shape[0])
        
        for lm_idx in target_lms:
            save_multiview_heatmap(points_np, heatmap_np[lm_idx], vis_save_dir, idx, lm_idx, prefix)

# -----------------------------------------------------------------------------
# [기능 4] 메인 학습 함수
# -----------------------------------------------------------------------------
def train(args):
    # -----------------------------------------------------------
    # [설정 강제] 경로 고정
    # -----------------------------------------------------------
    args.data_root = '../Data'       
    args.output_root = '../results' 
    
    print(f"\n>>> [User Mode] Manual Data Management")
    print(f"    Data Root   : {os.path.abspath(args.data_root)}")
    print(f"    Output Root : {os.path.abspath(args.output_root)}")

    # 1. 실험 폴더 및 ID 생성
    paths = get_experiment_paths(args)

    # 2. 데이터 처리
    if args.need_resample:
        print("\n=== [Phase 1] Loading Data from '../Data/train' & '../Data/test' ===")
        
        # [Train 데이터 로드]
        h_train, s_train, l_train = main_sample(
            args.num_points, args.seed, args.sigma, args.sample_way, "train", args.data_root
        )
        
        # [Test 데이터 로드]
        h_test, s_test, l_test = main_sample(
            args.num_points, args.seed, args.sigma, args.sample_way, "test", args.data_root
        )
        
        # [TensorDataset 생성]
        train_dataset = TensorDataset(
            torch.from_numpy(s_train).float(), 
            torch.from_numpy(l_train).float(), 
            torch.from_numpy(h_train).float()
        )
        test_dataset = TensorDataset(
            torch.from_numpy(s_test).float(), 
            torch.from_numpy(l_test).float(), 
            torch.from_numpy(h_test).float()
        )
        
        print(f"    Train Samples: {len(train_dataset)}")
        print(f"    Test  Samples: {len(test_dataset)}")

    else:
        print("Error: For this manual mode, please set --need_resample True")
        return

    # 3. 데이터 백업 & 시각화
    process_data_storage(train_dataset, "train", paths)
    process_data_storage(test_dataset, "test", paths)

    # 4. DataLoader 설정
    train_loader = DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    ScaleAndTranslate = PointcloudScaleAndTranslate()

    # 5. 모델 초기화
    model = PAConv(args, args.landmark_num).to(device)
    model.apply(weight_init)
    
    if args.loss == 'adaptive_wing': criterion = AdaptiveWingLoss()
    else: criterion = torch.nn.MSELoss()
        
    if args.use_sgd: opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=args.weight_decay)
    else: opt = optim.Adam(model.parameters(), lr=args.lr, eps=1e-08, weight_decay=args.weight_decay)
    
    if args.scheduler == 'cos': scheduler = CosineAnnealingLR(opt, T_max=args.epochs)
    else: scheduler = StepLR(opt, step_size=40, gamma=0.9)

    # 6. 학습 루프
    print(f"\n=== [Phase 3] Start Training ===")
    
    # Loss 기록용 리스트
    loss_history = []

    for epoch in range(args.epochs):
        model.train()
        loss_epoch = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch") as tepoch:
            for point, landmark, seg in tepoch:
                point    = point.to(device)
                landmark = landmark.to(device)
                seg      = seg.to(device)

                # 1) 정규화 & 증강
                point_normal = normalize_data(point)
                point_normal = ScaleAndTranslate(point_normal)

                # 2) 모델 입력 (Permute: B, N, C -> B, C, N)
                point_input = point_normal.permute(0, 2, 1)

                opt.zero_grad()
                pred_heatmap = model(point_input)
                
                # Loss 계산 (seg도 Permute 필요할 수 있음: 모델 출력 shape 확인 필)
                # PAConv 출력: (B, L, N) / seg: (B, N, L) -> seg.permute(0, 2, 1)
                loss = criterion(pred_heatmap, seg.permute(0, 2, 1).contiguous())

                loss.backward()
                opt.step()
                
                loss_epoch += loss.item()
                tepoch.set_postfix(loss=loss.item())
        
        # 에폭 평균 Loss 기록
        avg_loss = loss_epoch / len(train_loader)
        loss_history.append(avg_loss)

        # 모델 저장 (5에폭마다)
        if (epoch + 1) % 5 == 0:
            filename = f'model_epoch_{epoch+1}.t7'
            save_path = os.path.join(paths['models'], filename)
            torch.save(model.state_dict(), save_path)

        scheduler.step()
    
    # 학습 종료 후 Loss 그래프 저장
    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(paths['root'], "loss_curve.png"))
    plt.close()

    print(f"\n=== Training Finished. Results at: {paths['root']} ===\n")

if __name__ == "__main__":
    args = parser.parse_args()
    _init_(args)
    train(args)