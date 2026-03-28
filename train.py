'''
@Author: Yuan Wang (Modified by Researcher 2 & AI Assistant)
@File: train.py
@Description: Gradient Accumulation + Curriculum Learning + DeepPA Switching + Best Model Saving
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
    test_loader = DataLoader(test_dataset, num_workers=0, batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    
    ScaleAndTranslate = PointcloudScaleAndTranslate()

    print(f"\n>>> [Model Init] Selected Backbone: {args.model}")
    
    # =========================================================
    # 🧊 [신규] DeepPA 전용: 얼려진 PAConv Prior 모델 로드
    # =========================================================
    if args.model == 'DeepPA':
        print(">>> [Prior Load] Loading Pre-trained PAConv (0.3mm SOTA)...")
        paconv_prior = PAConv(args, args.landmark_num).to(device)
        
        # 🚨 평화님이 지정해주신 PAConv 폴더 절대 경로 (파일명은 본인 파일명에 맞게 수정하세요!)
        best_paconv_path = os.path.join("..", "PAConv_model", "model_epoch_500.t7")
        
        if not os.path.exists(best_paconv_path):
            raise FileNotFoundError(f"🚨 PAConv 모델 파일을 찾을 수 없습니다! 파일명을 확인해주세요: {best_paconv_path}")
            
        paconv_prior.load_state_dict(torch.load(best_paconv_path))
        paconv_prior.eval() # 평가 모드 고정
        for param in paconv_prior.parameters():
            param.requires_grad = False # 기울기 계산 완전 차단! (메모리 아끼기)
    else:
        paconv_prior = None
    # =========================================================

    # 🚀 메인 학습 모델 스위칭 로직
    if args.model == 'PAConv':
        model = PAConv(args, args.landmark_num).to(device)
    elif args.model == 'DeepLA':
        model = DeepLA_Wrapper(args, args.landmark_num).to(device)
    elif args.model == 'DeepPA':
        model = DeepPA_Wrapper(args, args.landmark_num).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
        
    model.apply(weight_init)
    
    if args.loss == 'adaptive_wing': criterion = AdaptiveWingLoss()
    else: criterion = torch.nn.MSELoss()
        
    if args.use_sgd: 
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=args.weight_decay)
    else: 
        opt = optim.Adam(model.parameters(), lr=args.lr, eps=1e-08, weight_decay=args.weight_decay)
    
    if args.scheduler == 'cos': scheduler = CosineAnnealingLR(opt, T_max=args.epochs)
    else: scheduler = StepLR(opt, step_size=40, gamma=0.9)

    print(f"\n=== [Phase 3] Start Training with Curriculum Learning ===")
    
    opt.zero_grad() 
    
    # 🏆 [신규] 최고 성능을 기록하기 위한 변수 초기화
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
                
                # =========================================================
                # 🚀 스위칭 로직: DeepPA일 때만 Prior를 뽑아서 같이 넣어줌!
                # =========================================================
                point_input = point_normal.permute(0, 2, 1)
                
                # 👇👇 [여기에 추가 1]
                #print("\n>>> [DEBUG 1] 데이터 전처리 완료, PAConv 진입 대기...")
                
                if args.model == 'DeepPA':
                    with torch.no_grad():
                        prior_hint = paconv_prior(point_input)
                    # 👇👇 [여기에 추가 2]
                    #print(">>> [DEBUG 2] PAConv(Prior) 통과 완료! DeepLA 진입 대기...")
                    
                    pred_heatmap = model(point_input, prior_heatmap=prior_hint)
                    # 👇👇 [여기에 추가 3]
                    #print(">>> [DEBUG 3] DeepLA 통과 완료! 3D 좌표 추출 대기...")
                else:
                    pred_heatmap = model(point_input)
                
                points_for_coords = point_input.permute(0, 2, 1)
                pred_coords = get_differentiable_coords(points_for_coords, pred_heatmap, k=args.k_softargmax)
                
                # 👇👇 [여기에 추가 4]
                #print(">>> [DEBUG 4] 3D 좌표 추출 완료! Loss 계산 시작...")
                
                loss_heatmap = criterion(pred_heatmap, seg.permute(0, 2, 1).contiguous())
                # ... (이하 기존 코드 동일)
                loss_coord = dynamic_focal_l1_loss(pred_coords, augmented_landmark, gamma=args.focal_gamma)
                loss_surface = compute_point_to_plane_loss(pred_coords, points_for_coords, k=args.plane_knn)
                loss_struct = compute_structural_loss(pred_coords, augmented_landmark)
                
                stage1_epochs = args.epochs // 5 
                
                if epoch < stage1_epochs:
                    weights = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
                    total_loss = loss_heatmap
                else:
                    weights = torch.rand(4).to(device)
                    weights = weights / weights.sum()
                    total_loss = (0.05 * loss_heatmap + 
                                  weights[1] * loss_coord + 
                                  weights[2] * loss_surface + 
                                  weights[3] * loss_struct)
                
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
                
                tepoch.set_postfix(Loss=f"{total_loss.item():.4f}", HM=f"{loss_heatmap.item():.4f}", Crd=f"{loss_coord.item():.4f}", Srf=f"{loss_surface.item():.4f}", Str=f"{loss_struct.item():.4f}", mm=f"{mm_error:.2f}")

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
                    
                    # =========================================================
                    # 🚀 Val 루프에서도 스위칭 로직 동일하게 적용
                    # =========================================================
                    if args.model == 'DeepPA':
                        prior_hint = paconv_prior(point_input)
                        pred_heatmap = model(point_input, prior_heatmap=prior_hint)
                    else:
                        pred_heatmap = model(point_input)
                    # =========================================================
                    
                    points_for_coords = point_input.permute(0, 2, 1)
                    pred_coords = get_differentiable_coords(points_for_coords, pred_heatmap, k=args.k_softargmax)

                    loss_heatmap = criterion(pred_heatmap, seg.permute(0, 2, 1).contiguous())
                    loss_coord = dynamic_focal_l1_loss(pred_coords, landmark_normal, gamma=args.focal_gamma)
                    loss_surface = compute_point_to_plane_loss(pred_coords, points_for_coords, k=args.plane_knn)
                    loss_struct = compute_structural_loss(pred_coords, landmark_normal)
                    
                    true_l1 = F.l1_loss(pred_coords, landmark_normal).item()
                    mm_error = true_l1 * avg_m
                    
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
                    val_mm   += mm_error
                    
                    val_sample_count += point.size(0)
                    if val_sample_count >= 30:
                        break

        num_val_batches = i + 1
        v_loss = val_loss / num_val_batches
        v_hm   = val_hm / num_val_batches
        v_crd  = val_crd / num_val_batches
        v_srf  = val_srf / num_val_batches
        v_str  = val_str / num_val_batches 
        v_mm   = val_mm / num_val_batches

        # -------------------------------------------------------------
        # [PRINT] 
        # -------------------------------------------------------------
        print(f" [Train] 전체 Loss: {t_loss:.4f} | HM: {t_hm:.4f} | Crd: {t_crd:.4f} | Srf: {t_srf:.4f} | Struct: {t_str:.4f} | mm: {t_mm:.2f}")
        print(f" [Val]   전체 Loss: {v_loss:.4f} | HM: {v_hm:.4f} | Crd: {v_crd:.4f} | Srf: {v_srf:.4f} | Struct: {v_str:.4f} | mm: {v_mm:.2f}")
        
        # 🏆 [신규] 베스트 모델 저장 로직
        if v_mm < best_val_mm:
            best_val_mm = v_mm
            print(f" 🌟 [Best Model Saved] 최고 성능 갱신! 오차: {best_val_mm:.4f} mm")
            best_save_path = os.path.join(paths['models'], 'model_best.t7')
            torch.save(model.state_dict(), best_save_path)

        # 기존 10주기마다 중간 백업 저장 (원하시면 삭제 가능)
        if (epoch + 1) % 10 == 0:
            filename = f'model_epoch_{epoch+1}.t7'
            save_path = os.path.join(paths['models'], filename)
            torch.save(model.state_dict(), save_path)

        scheduler.step()
    
    print(f"\n=== Training Finished. Results at: {paths['root']} ===")
    print(f"🏆 최종 달성한 최고 성능(Best Validation Error): {best_val_mm:.4f} mm\n")

if __name__ == "__main__":
    args = parser.parse_args()
    _init_(args)
    train(args)