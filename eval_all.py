from __future__ import print_function, division
import sys
import torch
import argparse
import numpy as np
import os
import warnings
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from My_args import parser
from dataset import FaceLandmarkData
from PAConv_model import PAConv
from util import landmark_regression, get_3D_FAN_NME

#3D 적으로 보기 쉬운 각도 ---> 0709 추가
def update_view(angle):
    ax.view_init(elev=35, azim=angle)
    return fig,

#  창 띄우지 않고 이미지 저장만 가능하게 설정
matplotlib.use('Agg')
warnings.filterwarnings("ignore", category=FutureWarning)

# 파라미터 설정
args = parser.parse_args()
args.eval = True

#경로 따라가게 수정 ---> 07/05
if args.model_path == '':
    model_subdir = os.path.join(f"{args.dataset}-npy", f"FPS{args.num_points}_sigma{args.sigma}", "models")
    model_filename = args.model_epoch # 필요한 epoch 번호로 변경 가능
    args.model_path = os.path.join(model_subdir, model_filename)
    print(f" model_path 비어 있어 자동 설정됨: {args.model_path}")

#if args.model_path == '':
#    args.model_path = 'checkpoints/Face_alignment_with_PAConv/Ear296_Korean/models/model_epoch_250.t7'
#    print(f" model_path가 비어 있어 기본값으로 설정됨: {args.model_path}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 모델 불러오기
model = PAConv(args, args.landmark_num).to(device) #총 랜드마크 갯수 일치시켜야됨
model.load_state_dict(torch.load(args.model_path))
model.eval()

regression_point_num = args.regression_point_num

# 경로 설정
base_dir = os.path.join(f"{args.dataset}-npy", f"FPS{args.num_points}_sigma{args.sigma}")
heatmap_dir = os.path.join(base_dir, f"PD_heatmap_{args.Eval_DataType}_regression_{args.regression_point_num}")
os.makedirs(heatmap_dir, exist_ok=True)
asc_dir = os.path.join(base_dir, f"PD_heatmap_{args.Eval_DataType}_regression_{args.regression_point_num}", f"PD_landmark_{args.Eval_DataType}_regression_{args.regression_point_num}")
os.makedirs(asc_dir, exist_ok=True)
# 데이터 로드
shape_sample = np.load(os.path.join(base_dir, f"shape_{args.Eval_DataType}.npy"), allow_pickle=True)
landmark_all = np.load(os.path.join(base_dir, f"landmark_{args.Eval_DataType}.npy"), allow_pickle=True)
heatmap_sample = np.load(os.path.join(base_dir, f"Heat_data_{args.Eval_DataType}.npy"), allow_pickle=True)


test_dataset = TensorDataset(
    torch.tensor(shape_sample, dtype=torch.float32),
    torch.tensor(landmark_all, dtype=torch.float32),
    torch.tensor(heatmap_sample, dtype=torch.float32)
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 추가: ME 계산용 리스트
me_list = []
per_landmark_me_list = []
all_dists_list = []          # 모든 샘플 × 랜드마크 거리 펼친 것 (Table III 'All' 용)

# 평가 시작
total_nme = 0.0
print("\n Evaluating on test set...")
for idx, (point, gt_landmark, heatmap) in enumerate(tqdm(test_loader, desc="Evaluation Progress")):
    point = point.to(device)
    heatmap = heatmap.to(device)
    gt_landmark = gt_landmark.to(device)

    with torch.no_grad():
        pred_heatmap_raw = model(point.permute(0, 2, 1))  # [1, N, 2048]
        pred_heatmap = pred_heatmap_raw.permute(0, 2, 1)  # [1, 2048, N]

        # 디버그 출력
        print(f"\n Sample {idx}")
        print(f"point[0].shape: {point[0].shape}")
        print(f"pred_heatmap[0].shape: {pred_heatmap[0].shape}")
        print(f"gt_landmark[0].shape: {gt_landmark[0].shape}")

        # 예측 히트맵 저장 경로 설정 ---> 07/05
        #reg_num = args.regression_point_num
        #heatmap_dir = os.path.join(base_dir, f"PD_heatmap_regression_{reg_num}")
        #os.makedirs(heatmap_dir, exist_ok=True)

        # 히트맵 시각화 (40개마다)
        if idx % 20 == 0:
            points_np = point[0].cpu().numpy()
            heatmap_np = pred_heatmap[0].cpu().numpy()
            landmark_num = heatmap_np.shape[1]
            for landmark_id in range(min(40, landmark_num)):
                colors = heatmap_np[:, landmark_id]
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                sc = ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=colors, cmap='jet', s=1)
                ax.view_init(elev=90, azim=-90) #위에서 내려다보는 각도
                #ax.view_init(elev=55, azim=235) # tragus 잘보이는 각도
                plt.colorbar(sc, label=f"Heatmap value for landmark {landmark_id}")
                plt.title(f"Sample {idx} - Heatmap (Landmark {landmark_id})")
                #plt.savefig(f"heatmap_visualizations/sample{idx:03d}_landmark{landmark_id}.png")
                plt.savefig(os.path.join(heatmap_dir, f"sample{idx:03d}_landmark{landmark_id}.png"))
                plt.close()

                # 애니메이션 저장 (샘플 0, 랜드마크 11일 때만)
                #if idx == 0 and landmark_id == 11:
                #    fig = plt.figure()
                #    ax = fig.add_subplot(111, projection='3d')
                #    sc = ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=colors, cmap='jet', s=1)
                #    plt.colorbar(sc, label=f"Heatmap value for landmark {landmark_id}")
                #    ax.set_title(f"Rotating Heatmap - Sample {idx}, Landmark {landmark_id}")
                #    ani = animation.FuncAnimation(fig, update_view, frames=range(0, 360, 3), interval=50)
                #    ani_save_path = os.path.join(heatmap_dir, f"sample{idx:03d}_landmark{landmark_id}_rotate.gif")
                #    ani.save(ani_save_path, writer='pillow', fps=20)
                #    plt.close()
                #    print(f" 애니메이션 저장 완료: {ani_save_path}")
   
        # 후처리 및 평가
        pred_landmark = landmark_regression(point[0], pred_heatmap[0], args.regression_point_num, idx)
        nme, _ = get_3D_FAN_NME(pred_landmark, gt_landmark)
        total_nme += nme.item()

        # ME 계산 추가
        pred_np = pred_landmark.cpu().numpy()
        gt_np = gt_landmark[0].cpu().numpy()
        dists = np.linalg.norm(pred_np - gt_np, axis=2)
        me = np.mean(dists)
        me_list.append(me)
        # 추가: 이 샘플의 랜드마크별 거리 저장 (L 길이 1D 배열)
        per_landmark_me_list.append(dists.squeeze(0)) 
        # ② 전체 거리 펼친 리스트 (Table III All 용)
        all_dists_list.append(dists.reshape(-1))        # (L,)

        print(f"[{idx:03d}] NME: {nme.item():.4f} | ME: {me:.4f}")
        #if idx % 10 == 0:
        #    print(f"[{idx:03d}] NME: {nme.item():.4f} | ME: {me:.4f}")

        pred_coords = pred_landmark.squeeze(0).cpu().numpy()  # shape: (landmark_num, 3)
        asc_save_path = os.path.join(asc_dir, f"pred_landmark_{args.Eval_DataType}_{idx:03d}.asc")
        np.savetxt(asc_save_path, pred_coords, fmt="%.6f", delimiter=",")

#  결과 요약
average_nme = total_nme / len(test_loader)
average_me = np.mean(me_list)
std_me = np.std(me_list)
sr = np.sum(np.array(me_list) < 10.0) / len(me_list) * 100

#  최종 출력
print(f"\n Evaluation Complete!")
print(f"🔹 Average NME(not normalized yet): {average_nme:.4f}")
print(f"🔹 Average ME : {average_me:.4f}")
print(f"🔹 Std of ME  : {std_me:.4f}")
print(f"🔹 SR@10mm    : {sr:.2f}%")

#25.11.25
if len(per_landmark_me_list) > 0:
    per_landmark_me_array = np.stack(per_landmark_me_list, axis=0).astype(np.float64)  # (N, L)

    # 랜드마크별 평균 / 표준편차
    per_landmark_mean = np.mean(per_landmark_me_array, axis=0)
    per_landmark_std  = np.std(per_landmark_me_array, axis=0)

    print("\n🔹 Per-landmark ME (mm):   (형식: Mean ± STD)")
    for lid in range(len(per_landmark_mean)):
        print(f"  - Lm {lid:02d}: {per_landmark_mean[lid]:.3f} ± {per_landmark_std[lid]:.3f} mm")

# (2) Table III 'All' 행: 모든 샘플 × 랜드마크 거리 전체에 대한 Mean ± STD
if len(all_dists_list) > 0:
    all_dists = np.concatenate(all_dists_list, axis=0).astype(np.float64)  # (N * L,)
    all_mean = np.mean(all_dists)
    all_std  = np.std(all_dists)
    print(f"\n🔹 All (per-point error, SOTA style): {all_mean:.3f} ± {all_std:.3f} mm")

        
    # 디버깅: 좌표 범위 확인 ---> 06/25 노말라이제이션인지 확인용
print(f"[{idx}] point range: min={point[0].min().item():.3f}, max={point[0].max().item():.3f}")
print(f"[{idx}] pred_landmark range: min={pred_landmark.min().item():.3f}, max={pred_landmark.max().item():.3f}")

# 디버깅 ME list 구성 확인용 ---> 07.04 추가
me_array = np.array(me_list)
nan_count = np.isnan(me_array).sum()
zero_count = np.sum(me_array < 1e-6)
large_count = np.sum(me_array > 100)
bad_indices = np.where(np.isnan(me_array) | (me_array < 1e-6) | (me_array > 100))[0]

print("\n🔍 ME 값 상태 분석")
print(f" - 총 샘플 수: {len(me_array)}")
print(f" - NaN 개수: {nan_count}")
print(f" - 0 또는 너무 작은 값(<1e-6): {zero_count}")
print(f" - 이상치(>100mm) 개수: {large_count}")
print(f" - 문제 있는 인덱스: {bad_indices.tolist()}")

"""
#  NaN 제거 후 유효한 ME 값 기준으로 다시 계산 
me_valid = me_array[~np.isnan(me_array)]
print(f"\n🔹 유효 샘플 기준 평균 ME: {np.mean(me_valid):.4f}")
print(f"🔹 유효 샘플 기준 표준편차: {np.std(me_valid):.4f}")
print(f"🔹 유효 SR@10mm: {(np.sum(me_valid < 10.0) / len(me_valid)) * 100:.2f}%")
"""

"""
            # NME 방식: bounding box 대각선
            min_xyz = torch.min(gt_sample, dim=0)[0]
            max_xyz = torch.max(gt_sample, dim=0)[0]
            norm_diag = torch.norm(max_xyz - min_xyz)
            nme_diag = me_tensor / norm_diag
            total_nme_diag += nme_diag.item()
            nme_list_diag.append(nme_diag.item())
"""