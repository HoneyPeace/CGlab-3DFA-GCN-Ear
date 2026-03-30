'''
@Author: Yuan Wang (Modified by Researcher 2 & AI Assistant)
@File: train.py
@Description: Gradient Accumulation + Curriculum Learning + DeepPA Switching + PAConv_heat Support + Best Model Saving + Auto-Scaled Curvature Loss
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
# 🚀 변경점: compute_point_to_plane_loss 대신 AutoScaledCurvatureSurfaceLoss 임포트
from loss import AdaptiveWingLoss, get_differentiable_coords, AutoScaledCurvatureSurfaceLoss, compute_structural_loss, dynamic_focal_l1_loss
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
    # 🧊 DeepPA 전용: 얼려진 PAConv Prior 모델 로드
    # =========================================================
    if args.model == 'DeepPA':
        print(">>> [Prior Load] Loading Pre-trained PAConv (0.3mm SOTA)...")
        paconv_prior = PAConv(args, args.landmark_num).to(device)
        
        best_paconv_path = os.path.join("..", "PAConv_model", "model_epoch_500.t7")
        
        if not os.path.exists(best_paconv_path):
            raise FileNotFoundError(f"🚨 PAConv 모델 파일을 찾을 수 없습니다! 파일명을 확인해주세요: {best_paconv_path}")
            
        paconv_prior.load_state_dict(torch.load(best_paconv_path))
        paconv_prior.eval()
        for param in paconv_prior.parameters():
            param.requires_grad = False 
    else:
        paconv_prior = None
    # =========================================================

    # 🚀 메인 학습 모델 스위칭 로직 (PAConv_heat 추가)
    if args.model == 'PAConv' or args.model == 'PAConv_heat':
        model = PAConv(args, args.landmark_num).to(device)
        if args.model == 'PAConv_heat':
            print(">>> [INFO] 🔥 PAConv_heat 모드 가동: 500에폭 내내 1-Loss(히트맵)만 학습합니다!")
    elif args.model == 'DeepLA':
        model = DeepLA_Wrapper(args, args.landmark_num).to(device)
    elif args.model == 'DeepPA':
        model = DeepPA_Wrapper(args, args.landmark_num).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
        
    model.apply(weight_init)
    
    # 🚀 [신규 추가] 자동 스케일링이 탑재된 표면 곡률 로스 초기화
    # 목표 체급(1.53)에 맞춰 자동으로 스케일을 펌핑해줍니다.
    surface_criterion = AutoScaledCurvatureSurfaceLoss(
        target_norm_val=1.53, 
        k_p2p=args.plane_knn if hasattr(args, 'plane_knn') else 5, 
        k_curv=15, 
        alpha=10.0
    ).to(device)

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
                
                if args.model == 'DeepPA':
                    with torch.no_grad():
                        prior_hint = paconv_prior(point_input)
                    pred_heatmap = model(point_input, prior_heatmap=prior_hint)
                else:
                    pred_heatmap = model(point_input)
                
                points_for_coords = point_input.permute(0, 2, 1)
                pred_coords = get_differentiable_coords(points_for_coords, pred_heatmap, k=args.k_softargmax)
                
                loss_heatmap = criterion(pred_heatmap, seg.permute(0, 2, 1).contiguous())
                loss_coord = dynamic_focal_l1_loss(pred_coords, augmented_landmark, gamma=args.focal_gamma)
                
                # 🚀 변경점: 곡률 기반 하이브리드 로스 계산 (자동 스케일 적용됨)
                loss_surface = surface_criterion(pred_coords, augmented_landmark, points_for_coords)
                
                loss_struct = compute_structural_loss(pred_coords, augmented_landmark)
                
                # =========================================================
                # ⚖️ [Scale Normalization] 스케일 정규화 (황금 밸런스 적용)
                # =========================================================
                # 목표 체급: 약 1.53 수준으로 모두 통일
                norm_heatmap = loss_heatmap * 0.3   # 5.199 * 0.3  ≈ 1.55
                norm_coord   = loss_coord   * 1.0   # 1.533 * 1.0  = 1.53 (기준점)
                # 🚀 변경점: 수동 펌핑(35.0) 제거. surface_criterion이 스스로 1.53으로 맞춰 반환합니다.
                norm_surface = loss_surface         
                norm_struct  = loss_struct  * 5     # 0.327 * 4.7  ≈ 1.53
                
                # 🔍 첫 번째 에폭, 첫 번째 배치에서 순수 스케일 및 정규화 결과 출력
                if epoch == 0 and i == 0:
                    print(f"\n=========================================")
                    print(f" 🔍 [Raw Loss Scale Check]")
                    print(f"  - Heatmap : {loss_heatmap.item():.6f}  (Norm: {norm_heatmap.item():.6f})")
                    print(f"  - Coord   : {loss_coord.item():.6f}  (Norm: {norm_coord.item():.6f})")
                    print(f"  - Surface : {loss_surface.item():.6f}  (Norm: {norm_surface.item():.6f})")
                    print(f"  - Struct  : {loss_struct.item():.6f}  (Norm: {norm_struct.item():.6f})")
                    print(f"=========================================\n")
                # =========================================================

                # =========================================================
                # 🚀 Loss 스위칭 로직 (정규화된 변수 norm_* 사용)
                # =========================================================
                if args.model == 'PAConv_heat':
                    # 🔥 무조건 히트맵만 100% 학습!
                    weights = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
                    total_loss = norm_heatmap
                else:
                    # 1. 나머지 3개의 로스를 위한 랜덤 가중치를 뽑습니다.
                    rand_weights = torch.rand(4).to(device)
                    # 3. 히트맵은 0.05 고정, 나머지는 0.95를 나눠가진 값을 곱해줍니다. (정규화된 로스 사용)
                    total_loss = ( rand_weights[0] * norm_heatmap + 
                                   rand_weights[1] * norm_coord + 
                                   rand_weights[2] * norm_surface + 
                                   rand_weights[3] * norm_struct)
                    
                    # (선택) 터미널 화면 출력을 위해 weights 변수를 예쁘게 다시 조립해 줍니다.
                    weights = torch.tensor([rand_weights[0], rand_weights[1], rand_weights[2], rand_weights[3]])
                # =========================================================
                
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
                    
                    if args.model == 'DeepPA':
                        prior_hint = paconv_prior(point_input)
                        pred_heatmap = model(point_input, prior_heatmap=prior_hint)
                    else:
                        pred_heatmap = model(point_input)
                    
                    points_for_coords = point_input.permute(0, 2, 1)
                    pred_coords = get_differentiable_coords(points_for_coords, pred_heatmap, k=args.k_softargmax)

                    loss_heatmap = criterion(pred_heatmap, seg.permute(0, 2, 1).contiguous())
                    loss_coord = dynamic_focal_l1_loss(pred_coords, landmark_normal, gamma=args.focal_gamma)
                    
                    # 🚀 변경점: Val 루프에서도 클래스 기반 로스 호출
                    loss_surface = surface_criterion(pred_coords, landmark_normal, points_for_coords)
                    
                    loss_struct = compute_structural_loss(pred_coords, landmark_normal)
                    
                    true_l1 = F.l1_loss(pred_coords, landmark_normal).item()
                    mm_error = true_l1 * avg_m
                    
                    # Val 루프도 동일한 정규화 상수 적용
                    norm_heatmap = loss_heatmap * 0.3
                    norm_coord   = loss_coord   * 1.0
                    # 🚀 변경점: 수동 펌핑(37.0) 제거
                    norm_surface = loss_surface 
                    norm_struct  = loss_struct  * 4.7
                    
                    # =========================================================
                    # 🚀 Val 루프 Loss 스위칭 로직 (정규화된 변수 norm_* 사용)
                    # =========================================================
                    if args.model == 'PAConv_heat':
                        total_loss = norm_heatmap
                    else:
                        stage1_epochs = args.epochs // 5
                        if epoch < stage1_epochs:
                            total_loss = norm_heatmap
                        else:
                            total_loss = 0.25 * norm_heatmap + 0.25 * norm_coord + 0.25 * norm_surface + 0.25 * norm_struct
                    # =========================================================

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
        
        # 🚀 에폭별 로스 반영률(%) 모니터링 출력 스위칭
        if args.model == 'PAConv_heat':
            print(f" 📊 [로스 반영률] Heatmap: 100% (오직 히트맵 정답지 구축 중! 🎯)")
        else:
            stage1_epochs = args.epochs // 5
            if epoch < stage1_epochs:
                print(f" 📊 [로스 반영률] Heatmap: 100% | Coord: 0% | Surface: 0% | Struct: 0%  (Stage 1: 번역기 집중 학습 🔥)")
            else:
                w_np = weights.detach().cpu().numpy()
                print(f" 📊 [로스 반영률] Heatmap: {w_np[0]*100:.1f}% | Coord: {w_np[1]*100:.1f}% | Surface: {w_np[2]*100:.1f}% | Struct: {w_np[3]*100:.1f}%  (Stage 2: 4-Loss RLW 완전 해방 🌪️)")

        # 🏆 베스트 모델 저장 로직
        if v_mm < best_val_mm:
            best_val_mm = v_mm
            print(f" 🌟 [Best Model Saved] 최고 성능 갱신! 오차: {best_val_mm:.4f} mm")
            best_save_path = os.path.join(paths['models'], 'model_best.t7')
            torch.save(model.state_dict(), best_save_path)

        # 기존 10주기마다 중간 백업 저장
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