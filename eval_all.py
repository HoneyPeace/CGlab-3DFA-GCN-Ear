'''
@Author: Yuan Wang (Modified by Researcher 2 & AI Assistant)
@File: eval_all.py
@Description: Auto 2-Stage Evaluation Script (PAConv Base vs DeepPA Final) & 7-Channel Compatibility & Full Metrics & 🌟 Excel Export

[NotebookLM을 위한 모듈 요약]
이 스크립트는 3D 랜드마크 검출 모델의 최종 성능을 측정하는 '평가(Evaluation) 파이프라인'입니다.
논문의 실험(Experiments) 섹션 작성을 위한 핵심 코드이며, 다음과 같은 특징을 가집니다.
1. 정성적 평가(Qualitative): 3D 히트맵을 다각도(정면, 측면, 하단)에서 렌더링하여 이미지로 저장.
2. 정량적 평가(Quantitative): Mean Error(mm 거리 오차), Cosine Similarity(분포 유사성), mIoU(영역 겹침) 등 다각도 지표 산출.
3. 자동화(Automation): 'DeepPA_auto' 모드 시 1단계(PAConv)와 2단계(DeepPA)를 연속으로 평가하여 성능 향상폭을 엑셀로 자동 정리함.
'''

from __future__ import print_function, division
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import time
import warnings
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from My_args import parser

# ==========================================
# 🚀 모델 3대장 모두 임포트
# ==========================================
from DeepLA_model import DeepLA_Wrapper
from DeepPA_model import DeepPA_Wrapper  
from PAConv_model import PAConv          
# ==========================================

from loss import get_differentiable_coords

matplotlib.use('Agg')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.manifold._mds")

args = parser.parse_args()
args.eval = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_multiview_heatmap(points, heatmap, save_dir, sample_name, landmark_idx, prefix):
    """
    [정성적 평가 시각화 도구 (Qualitative Visualization)]
    - 목적: 예측된 랜드마크 확률 분포(히트맵)를 3D 공간 상에 점(Point)의 색상(Colormap)으로 매핑하여 저장합니다.
    - 특징: 3D 구조의 특성상 가려지는 부분이 없도록 정면(Front), 측면(Side), 하단(Downside) 3개의 카메라 뷰(View)를 동시 렌더링합니다.
    """
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

    filename = f"{prefix}_{sample_name}_L{landmark_idx + 1:02d}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=100, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# 1. 경로 설정
# -----------------------------------------------------------------------------
if not args.run_id:
    print("Error: --run_id required (e.g., '1' for the first experiment).")
    sys.exit(1)

project_dir = os.path.join(args.output_root, args.exp_name)
batch_str = f"{args.batch_size}x{args.accumulation_steps}" if args.accumulation_steps > 1 else f"{args.batch_size}"
base_str = f"FPS{args.num_points}_sigma{args.sigma}_batch{batch_str}_train{args.train_len}"
setting_str = f"{base_str}_{args.user_tag}" if args.user_tag else base_str

target_folder_name = f"{setting_str}_{args.run_id}"
run_root = os.path.join(project_dir, target_folder_name)

if not os.path.exists(run_root):
    print(f"Error: Experiment folder not found: {run_root}")
    sys.exit(1)

data_dir = os.path.join(run_root, 'npy_data')
heatmap_save_dir_base = os.path.join(run_root, "Pred_Heatmaps")
asc_save_dir_base = os.path.join(run_root, "Pred_Landmarks")

print(f"Target Run  : {target_folder_name}")
print(f"Loading Data : {data_dir}")

# -----------------------------------------------------------------------------
# 2. 데이터 로드 (🌟 7채널 다이나믹 스위칭 적용)
# -----------------------------------------------------------------------------
try:
    in_channels = getattr(args, 'in_channels', 3)
    shape_filename = f"shape_{args.Eval_DataType}.npy"
    
    if in_channels == 7:
        print(f">>> [INFO] 🎯 7-Channel Mode: Loading Geometric Features (XYZ+Dir+Curv) from {shape_filename}")
    elif in_channels == 6:
        print(f">>> [INFO] 🎯 6-Channel Mode: Loading Geometric Features from {shape_filename}")
    else:
        print(f">>> [INFO] 🧊 3-Channel Mode: Loading Standard XYZ from {shape_filename}")
        
    shape_sample = np.load(os.path.join(data_dir, shape_filename), allow_pickle=True)
    landmark_all = np.load(os.path.join(data_dir, f"landmark_{args.Eval_DataType}.npy"), allow_pickle=True)
    heatmap_sample = np.load(os.path.join(data_dir, f"Heat_data_{args.Eval_DataType}.npy"), allow_pickle=True)
    
    name_path = os.path.join(data_dir, f"name_{args.Eval_DataType}.npy")
    if os.path.exists(name_path):
        name_sample = np.load(name_path, allow_pickle=True)
    else:
        name_sample = [f"S{i:03d}" for i in range(len(shape_sample))]

    try:
        train_shape_path = os.path.join(data_dir, "shape_train.npy")
        train_len = len(np.load(train_shape_path, allow_pickle=True)) if os.path.exists(train_shape_path) else "Unknown"
    except:
        train_len = "Error checking"

except FileNotFoundError:
    print(f"Error: Backup data files not found in {data_dir}.")
    sys.exit(1)

test_dataset = TensorDataset(
    torch.tensor(shape_sample, dtype=torch.float32),
    torch.tensor(landmark_all, dtype=torch.float32),
    torch.tensor(heatmap_sample, dtype=torch.float32)
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# -----------------------------------------------------------------------------
# 3. 평가 수행 코어 함수
# -----------------------------------------------------------------------------
def evaluate_target_model(eval_name, eval_model, prior_model=None):
    """
    [핵심 정량 평가 로직]
    평가 지표 (NotebookLM 분석용):
    1. Mean Error (ME): 예측 랜드마크 3D 좌표와 실제 3D 좌표 간의 유클리디안 거리 (단위: mm). 낮을수록 좋음.
    2. Cosine Similarity: 예측 히트맵과 정답 히트맵을 벡터로 보았을 때의 유사도. 분포의 방향성이 얼마나 비슷한지 평가. 높을수록 좋음.
    3. mIoU (Intersection over Union): 예측 히트맵과 정답 히트맵이 임계값(0.1) 이상인 영역이 얼마나 겹치는지 평가. 픽셀 수준의 분할 정확도. 높을수록 좋음.
    """
    print(f"\n==================================================")
    print(f" 🚀 [EVALUATION START] 대상 모델: {eval_name}")
    print(f"==================================================")
    
    current_hm_dir = os.path.join(heatmap_save_dir_base, eval_name)
    current_asc_dir = os.path.join(asc_save_dir_base, eval_name)
    os.makedirs(current_hm_dir, exist_ok=True)
    os.makedirs(current_asc_dir, exist_ok=True)

    me_list, per_landmark_me_list = [], []
    
    # 🔥🔥🔥 바로 이 부분의 오타를 수정했습니다! (변수 3개, 리스트 3개) 🔥🔥🔥
    cos_sim_list, iou_list, time_list = [], [], []
    
    per_landmark_cos_sim_list, per_landmark_iou_list = [], []

    eval_model.eval()
    if prior_model:
        prior_model.eval()

    for idx, (point, gt_landmark, heatmap) in enumerate(tqdm(test_loader, desc=f"Eval {eval_name}")):
        point, gt_landmark, gt_heatmap = point.to(device), gt_landmark.to(device), heatmap.to(device)
        real_name = name_sample[idx]
        B, N, C = point.shape
        
        # [정규화 복원 (Denormalization)]
        # 모델의 스케일 무관성(Scale-invariance)을 위해 정규화했던 데이터를, 
        # 실제 mm 단위의 오차(Mean Error)를 계산하기 위해 다시 원래 스케일과 위치로 되돌립니다.
        point_xyz = point[:, :, :3]
        centroid = torch.mean(point_xyz, axis=1, keepdim=True)
        point_centered = point_xyz - centroid
        m = torch.max(torch.sqrt(torch.sum(point_centered ** 2, axis=2)), axis=1)[0]
        scale = m.view(-1, 1, 1)
        point_norm_xyz = point_centered / scale 
        
        if C == 7:
            point_geom = point[:, :, 3:]
            point_norm = torch.cat([point_norm_xyz, point_geom], dim=-1)
        elif C == 6:
            point_dir = point[:, :, 3:]
            point_norm = torch.cat([point_norm_xyz, point_dir], dim=-1)
        else:
            point_norm = point_norm_xyz
        
        with torch.no_grad():
            if device.type == 'cuda': torch.cuda.synchronize()
            start_time = time.time()

            point_input = point_norm.permute(0, 2, 1).contiguous()
            
            # [Two-stage 추론] prior_model(1단계 PAConv)의 결과를 eval_model(2단계 DeepPA)의 힌트로 제공
            if prior_model is not None:
                prior_hint = prior_model(point_input)
                pred_heatmap_raw = eval_model(point_input, prior_heatmap=prior_hint)
            else:
                pred_heatmap_raw = eval_model(point_input)
                
            pred_heatmap = pred_heatmap_raw.permute(0, 2, 1)

            # 히트맵(확률)에서 실제 3D 좌표 복원 (Soft-argmax 방식 사용)
            pred_landmark_norm = get_differentiable_coords(point_norm_xyz, pred_heatmap_raw, k=args.k_softargmax)
            pred_landmark = (pred_landmark_norm * scale) + centroid # mm 단위로 복구

            if device.type == 'cuda': torch.cuda.synchronize()
            time_list.append(time.time() - start_time)

            # [Metric 1 & 2: Cosine Similarity & mIoU] (히트맵 분포 평가)
            pred_vec, gt_vec = pred_heatmap.permute(0, 2, 1), gt_heatmap.permute(0, 2, 1)     
            cos_sim_k = F.cosine_similarity(pred_vec, gt_vec, dim=2) * 100.0
            cos_sim_list.append(cos_sim_k.mean().item()) 
            per_landmark_cos_sim_list.append(cos_sim_k.cpu().numpy()) 

            threshold = 0.1
            pred_mask, gt_mask = (pred_heatmap > threshold).float(), (gt_heatmap > threshold).float()     
            intersection_k = (pred_mask * gt_mask).sum(dim=1) 
            union_k = (pred_mask + gt_mask).clamp(0, 1).sum(dim=1)
            iou_k = ((intersection_k + 1e-6) / (union_k + 1e-6)) * 100.0
            
            iou_list.append(iou_k.mean().item()) 
            per_landmark_iou_list.append(iou_k.cpu().numpy()) 

            if idx % 40 == 0:
                points_np, heatmap_np = point_xyz[0].cpu().numpy(), pred_heatmap[0].cpu().numpy()
                for lm_idx in range(heatmap_np.shape[1]):
                     save_multiview_heatmap(points_np, heatmap_np[:, lm_idx], current_hm_dir, real_name, lm_idx, f"pred_{eval_name}")

            pred_np = pred_landmark.cpu().numpy().squeeze(0) if pred_landmark.ndim == 3 else pred_landmark.cpu().numpy()
            gt_np = gt_landmark.cpu().numpy().squeeze(0) if gt_landmark.ndim == 3 else gt_landmark.cpu().numpy()
                
            # [Metric 3: Mean Error] 실제 예측 좌표와 정답 좌표 간의 물리적 거리 계산
            dists = np.linalg.norm(pred_np - gt_np, axis=1)
            me = np.mean(dists)
            
            me_list.append(me)
            per_landmark_me_list.append(dists)
            
            np.savetxt(os.path.join(current_asc_dir, f"{eval_name}_pred_{real_name}.asc"), pred_np, fmt="%.6f", delimiter=",")

    # ------------------ 최종 집계 (논문에 들어갈 통계치 산출) ------------------
    if len(per_landmark_me_list) > 0:
        per_landmark_me_array = np.stack(per_landmark_me_list, axis=0) 
        lm_means, lm_stds, lm_me_95 = np.mean(per_landmark_me_array, axis=0), np.std(per_landmark_me_array, axis=0), np.percentile(per_landmark_me_array, 95, axis=0)
        average_me, std_me, me_95_global = np.mean(lm_means), np.mean(lm_stds), np.percentile(me_list, 95)
        
        per_landmark_cos_array = np.vstack(per_landmark_cos_sim_list) 
        lm_cos_means, lm_cos_stds, lm_cos_5 = np.mean(per_landmark_cos_array, axis=0), np.std(per_landmark_cos_array, axis=0), np.percentile(per_landmark_cos_array, 5, axis=0)
        
        per_landmark_iou_array = np.vstack(per_landmark_iou_list) 
        lm_iou_means, lm_iou_stds, lm_iou_5 = np.mean(per_landmark_iou_array, axis=0), np.std(per_landmark_iou_array, axis=0), np.percentile(per_landmark_iou_array, 5, axis=0)
    else:
        return

    # SR (Success Rate): 특정 오차 범위(10mm, 5mm) 내에 들어온 예측의 비율
    sr_10 = np.sum(np.array(me_list) < 10.0) / len(me_list) * 100
    sr_5  = np.sum(np.array(me_list) < 5.0) / len(me_list) * 100
    avg_cos_sim, cos_sim_5_global = np.mean(cos_sim_list), np.percentile(cos_sim_list, 5)
    avg_iou, iou_5_global = np.mean(iou_list), np.percentile(iou_list, 5)
    avg_time = np.mean(time_list) * 1000.0

    batch_str_log = f"{args.batch_size}x{args.accumulation_steps}" if args.accumulation_steps > 1 else f"{args.batch_size}"

    # ------------------ TXT 파일 저장 ------------------
    filename_txt = f"{eval_name}_ME{average_me:.4f}_std{std_me:.4f}.txt"
    result_txt_path = os.path.join(run_root, filename_txt)

    with open(result_txt_path, "w") as f:
        f.write(f"==========================================\n")
        f.write(f"   Evaluation Result: {args.exp_name} ({eval_name})\n")
        f.write(f"==========================================\n")
        f.write(f" Run ID      : {target_folder_name}\n")
        f.write(f" Model       : {eval_name}\n")
        f.write(f" Data Type   : {args.Eval_DataType}\n")
        user_comment = args.user_tag if args.user_tag else "None"
        f.write(f" User Comment: {user_comment}\n")
        f.write(f" Train Data  : {train_len} samples\n")
        f.write(f" Batch Size  : {batch_str_log}\n")
        f.write(f" Num Points  : {args.num_points}\n")
        f.write(f" In Channels : {in_channels}\n")
        f.write(f"------------------------------------------\n")
        f.write(f" Average ME : {average_me:.4f} ± {std_me:.4f} mm\n")
        f.write(f" 95%ile ME  : {me_95_global:.4f} mm\n")
        f.write(f" SR @ 10mm  : {sr_10:.2f} %\n")
        f.write(f" SR @ 5mm   : {sr_5:.2f} %\n")
        f.write(f" Avg Time   : {avg_time:.2f} ms/sample\n")
        f.write(f"------------------------------------------\n")
        f.write(f" [Heatmap Quantitative Evaluation (Global)]\n")
        f.write(f" Cosine Sim : {avg_cos_sim:.2f} % | 95%ile(Bot 5%): {cos_sim_5_global:.2f} %\n")
        f.write(f" mIoU       : {avg_iou:.2f} % | 95%ile(Bot 5%): {iou_5_global:.2f} %\n")
        f.write(f"==========================================\n")
        
        if len(per_landmark_me_list) > 0:
            f.write("\n>>> Per-landmark ME (Mean ± Std | 95%ile):\n")
            for i in range(lm_means.shape[0]):
                f.write(f"    LM {i+1:02d}: {lm_means[i]:.3f} ± {lm_stds[i]:.3f} mm | 95%ile: {lm_me_95[i]:.3f} mm\n")

# ------------------ 🌟 Excel (.xlsx) 파일 완벽 포맷 저장 ------------------
    # [논문 작성 편의성 확보] 산출된 모든 지표를 논문 테이블(Table) 양식에 맞춰 여러 시트(Sheet)로 분할 저장
    filename_excel = f"{eval_name}_Results_ME{average_me:.4f}.xlsx"
    result_excel_path = os.path.join(run_root, filename_excel)
    
    # 1. Summary (사진 1 참고: 메타데이터 + 정량 수치 세로 나열)
    summary_data = {
        "지표 (Metric)": [
            "[Metadata]", "Experiment", "Run ID", "Model", "User Comment",
            "[Metrics]", 
            "Average ME (mm)", 
            "Average Std (mm)", 
            "Avg Time (ms)", 
            "Cosine Sim (%)", 
            "95%ile Cosine (%)", 
            "mIoU (@0.1, %)", 
            "95%ile mIoU (%)", 
            "Success Rate (<10mm, %)", 
            "Success Rate (<5mm, %)",
            "data",
            "point"
        ],
        eval_name: [ # 컬럼명을 모델명으로 지정하여 복붙하기 편하게 설정
            "", args.exp_name, target_folder_name, eval_name, user_comment,
            "",
            round(average_me, 4), 
            round(std_me, 4), 
            round(avg_time, 2), 
            round(avg_cos_sim, 2), 
            round(cos_sim_5_global, 2), 
            round(avg_iou, 2), 
            round(iou_5_global, 2), 
            round(sr_10, 2), 
            round(sr_5, 2),
            args.Eval_DataType,
            args.num_points
        ]
    }
    df_summary = pd.DataFrame(summary_data)

    df_landmarks = pd.DataFrame()
    df_top10 = pd.DataFrame()
    df_heatmap = pd.DataFrame()

    if len(per_landmark_me_list) > 0:
        # 2. Per_Landmark (사진 2 참고: LM 01, 평균±표준편차, 95%값)
        lm_me_combined = [f"{lm_means[i]:.3f} ± {lm_stds[i]:.3f}" for i in range(lm_means.shape[0])]
        landmark_data = {
            "LM (랜드마크)": [f"{i+1}" for i in range(lm_means.shape[0])],
            eval_name: lm_me_combined,
            f"{eval_name} (95%ile)": np.round(lm_me_95, 3)
        }
        df_landmarks = pd.DataFrame(landmark_data)

        # 3. Top 10 Hardest (사진 3 참고: 예측 어려운 순위)
        worst_indices = np.argsort(lm_means)[::-1][:10]
        top10_combined = [f"LM {i+1:02d} ({lm_means[i]:.3f} ± {lm_stds[i]:.3f})" for i in worst_indices]
        top10_95ile = [round(lm_me_95[i], 3) for i in worst_indices]
        top10_data = {
            "순위": [f"{r+1}" for r in range(10)],
            eval_name: top10_combined,
            f"{eval_name} (95%ile)": top10_95ile
        }
        df_top10 = pd.DataFrame(top10_data)

        # 4. Heatmap Metrics (사진 4를 위한 정량화 수치)
        heat_cos_combined = [f"{lm_cos_means[i]:.2f} ± {lm_cos_stds[i]:.2f}" for i in range(lm_means.shape[0])]
        heat_iou_combined = [f"{lm_iou_means[i]:.2f} ± {lm_iou_stds[i]:.2f}" for i in range(lm_means.shape[0])]
        heatmap_data = {
            "LM (랜드마크)": [f"{i+1}" for i in range(lm_means.shape[0])],
            f"{eval_name} (Cos Sim)": heat_cos_combined,
            f"{eval_name} (Cos 5%ile)": np.round(lm_cos_5, 2),
            f"{eval_name} (mIoU)": heat_iou_combined,
            f"{eval_name} (mIoU 5%ile)": np.round(lm_iou_5, 2)
        }
        df_heatmap = pd.DataFrame(heatmap_data)

    # 5. 여러 시트로 분할하여 완벽하게 저장
    with pd.ExcelWriter(result_excel_path, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='1_Summary', index=False)
        if not df_landmarks.empty:
            df_landmarks.to_excel(writer, sheet_name='2_Per_Landmark', index=False)
            df_top10.to_excel(writer, sheet_name='3_Top10_Hardest', index=False)
            df_heatmap.to_excel(writer, sheet_name='4_Heatmap_Metrics', index=False)

    print(f"\n[{eval_name} Done] Results saved to: {run_root}")
    print(f"      TXT   : {filename_txt}")
    print(f"      Excel : {filename_excel}") 
    print(f"Average ME: {average_me:.4f} ± {std_me:.4f} (95%ile: {me_95_global:.4f} mm)")

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 4. 모델 로드 및 평가 분기 (대소문자 무시 적용)
# -----------------------------------------------------------------------------
models_to_eval = []

if args.model.lower() == 'deeppa_auto':
    """
    [Two-stage 자동 평가 모드]
    연구의 핵심인 '1단계 거시적 파악(PAConv) -> 2단계 미시적 정밀 교정(DeepPA)' 파이프라인의
    성능 변화를 명확히 보여주기 위해, 두 모델을 순차적으로 로드하고 각각 평가를 진행합니다.
    """
    print(">>> [INFO] 🔄 Auto Mode: PAConv(Stage1)와 DeepPA(Stage2) 두 모델을 연속으로 평가합니다.")
    
    # 1. PAConv Base 로드
    paconv_model = PAConv(args, args.landmark_num).to(device)
    paconv_path = os.path.join(run_root, 'models', 'Stage1_PAConv_model_best.t7')
    if not os.path.exists(paconv_path):
        paconv_path = os.path.join(run_root, "deeppa_backup", "paconv_model", "last_paconv.pth")
    paconv_model.load_state_dict(torch.load(paconv_path, map_location=device))
    
    # 2. DeepPA Final 로드
    deeppa_model = DeepPA_Wrapper(args, args.landmark_num).to(device)
    deeppa_path = os.path.join(run_root, 'models', 'Stage2_DeepPA_model_best.t7')
    deeppa_model.load_state_dict(torch.load(deeppa_path, map_location=device))
    
    models_to_eval.append(("PAConv_Base", paconv_model, None))
    models_to_eval.append(("DeepPA_Final", deeppa_model, paconv_model))
    
else:
    model_path = os.path.join(run_root, 'models', args.model_epoch)
    
    if args.model == 'DeepPA':
        paconv_prior = PAConv(args, args.landmark_num).to(device)
        paconv_prior.load_state_dict(torch.load(os.path.join("..", "PAConv_model", "model_epoch_500.t7"), map_location=device))
        model = DeepPA_Wrapper(args, args.landmark_num).to(device)
    elif args.model in ['PAConv', 'PAConv_heat']:
        paconv_prior = None
        model = PAConv(args, args.landmark_num).to(device)
    elif args.model == 'DeepLA':
        paconv_prior = None
        model = DeepLA_Wrapper(args, args.landmark_num).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    models_to_eval.append((args.model, model, paconv_prior))

# -----------------------------------------------------------------------------
# 5. 순차적 평가 실행
# -----------------------------------------------------------------------------
for eval_name, target_model, prior_model in models_to_eval:
    evaluate_target_model(eval_name, target_model, prior_model)
    
print("\n>>> [ALL EVALUATION COMPLETED]")