'''
@Author: Yuan Wang (Modified by Researcher & AI Assistant)
@File: My_args.py
@Description: 
[NotebookLM을 위한 핵심 파일 요약]
이 파일은 3D 귀/얼굴 랜드마크 검출 모델의 모든 하이퍼파라미터와 제어 스위치를 정의하는 '통제 센터(Control Center)'입니다.
특히 단순한 3D 좌표(XYZ) 기반 학습을 넘어, **'표면의 곡률과 방향성(Geometric Features)'**을 활용하는 
하이브리드 파이프라인(DeepPA_auto)의 핵심 설정값들이 포함되어 있습니다.
'''

import argparse
import torch

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='3D Ear/Face Landmark Detection')

# =============================================================================
# [1] 기본 설정 (Base Args) & 모델 아키텍처
# =============================================================================
parser.add_argument('--exp_name', type=str, default='Ear_Project_Final', metavar='N', help='Name of the experiment')

# 🔥 'deeppa_auto' 옵션 추가
# [의도] DeepPA_auto는 Two-stage 학습 파이프라인입니다.
# 1단계(PAConv): 전체적인 형태를 보고 랜드마크의 대략적인 위치(Global Context)를 잡음.
# 2단계(DeepPA): 기하학적 로스(Curvature/Direction)를 켜서 표면의 굴곡에 완벽히 밀착되도록 미세 조정(Local Refinement)함.
parser.add_argument('--model', type=str, default='deeppa_frozen', metavar='N', choices=['PAConv_heat', 'PAConv', 'DeepLA', 'DeepPA', 'deeppa_frozen', 'deeppa_e2e'], help='Model to use')
parser.add_argument('--no_cuda', type=str2bool, default=False, help='enables CUDA training')
parser.add_argument('--model_path', type=str, default='', metavar='N', help='Pretrained model path')

parser.add_argument('--dataset', type=str, default='Ear296_Korean', help='Target dataset name')
parser.add_argument('--train_dataset_name', type=str, default='Ear296_Korean', help='Train dataset name')
parser.add_argument('--test_dataset_name', type=str, default='', help='Test dataset name (Optional)')

parser.add_argument('--data_root', type=str, default='../data', help='Root directory of data')
parser.add_argument('--output_root', type=str, default='../results', help='Root directory for results')

# 🌟 입력 채널 설정 (기본 7채널: xyz + 주방향 + 곡률)
# [기하학적 의미] 단순히 3차원 좌표(3차원)만 넣는 것이 아니라, 
# 데이터 전처리 단계에서 추출한 해당 점의 주방향 벡터(Principal Direction, 3차원)와 곡률(Curvature, 1차원)을 
# 추가로 입력하여 모델이 표면의 굴곡을 더 빨리 이해하도록 돕습니다.
parser.add_argument('--in_channels', type=int, default=7, help='Input channels: 3 for (xyz), 7 for (xyz + principal_dir + curvature)')


# =============================================================================
# [2] 학습 설정 (Train Args) & Two-stage 스케줄링
# =============================================================================
parser.add_argument('--eval', type=str2bool, default=False, help='evaluate the model')
parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size', help='Size of batch')
parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size', help='Size of batch')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of episode to train')

# 🔥 Auto 파이프라인 전용 에폭 설정 추가
parser.add_argument('--paconv_epochs', type=int, default=500, help='PAConv stage epochs in auto mode')
parser.add_argument('--deeppa_epochs', type=int, default=500, help='DeepPA stage epochs in auto mode')

parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient Accumulation Steps')

# =============================================================================
# [3] 최적화 설정 (Optimizer Args)
# =============================================================================
parser.add_argument('--loss', type=str, default='adaptive_wing', metavar='N', choices=['mse', 'adaptive_wing'], help='loss function to use')
parser.add_argument('--use_sgd', type=str2bool, default=False, help='Use SGD')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--scheduler', type=str, default='step', metavar='N', choices=['cos', 'step'], help='Scheduler to use')
parser.add_argument('--weight_decay', type=float, default=0, metavar='WD', help='the weight decay')

# =============================================================================
# [4] 데이터 처리 (Data Process Args) & 히트맵 파라미터
# =============================================================================
parser.add_argument('--max_threshold', default=10, type=float, help='the maximum threshold of error_rate')
parser.add_argument('--sample_way', type=str, default='FPS', metavar='sw', choices=['FPS', 'Random', 'CAGQ', 'Geometric'])
parser.add_argument('--need_resample', type=str2bool, default=True, help='Must be True to generate NPY files initially')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')

parser.add_argument('--regression_point_num', type=int, default=10, metavar='RPN', help='points in landmark regression')
parser.add_argument('--dataset_seed', type=int, default=1, metavar='S', help='train/test dataset random seed')
parser.add_argument('--num_points', type=int, default=8192, help='num of points to use')

# [sigma 의도] 3D 가우시안 히트맵의 분산(퍼짐 정도) 설정
parser.add_argument('--sigma', type=float, default=10.0, metavar='Sig', help='Gaussian Variance of heatmap')
parser.add_argument('--k', type=int, default=30, metavar='N', help='Num of nearest neighbors')
parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
parser.add_argument('--landmark_num', type=int, default=36, metavar='L', help='the number of landmark')

# =============================================================================
# [5] 모델 구조 설정 (PAConv Args)
# =============================================================================
parser.add_argument('--calc_scores', type=str, default='softmax', metavar='cs', help='The way to calculate score')
parser.add_argument('--hidden', type=list, default=[[32], [32], [32], [32]], help='the hidden layers of ScoreNet')
parser.add_argument('--num_matrices', type=list, default=[8, 8, 8, 8], help='the number of weight banks')

# =============================================================================
# [6] 기타 설정 (Etc)
# =============================================================================
parser.add_argument('--Eval_DataType', type=str, default="test", help='select npy train, test, sample')
parser.add_argument('--model_epoch', type=str, default="model_epoch_250.t7", help='load trained model file')
parser.add_argument('--run_id', type=str, default='', help='Load specific run from backup (e.g., 1, 2)')
parser.add_argument('--use_split_dataset', type=str2bool, default=True, help='Use split dataset mode')
parser.add_argument('--user_tag', type=str, default='', help='Custom tag added to the folder name')
parser.add_argument('--train_len', type=int, default=209, help='Number of training samples used in folder name')

# =============================================================================
# [7] 🌟 하이브리드 로스 & 3D 표면 페널티 하이퍼파라미터 (loss.py 연동)
# =============================================================================
# [이웃 크기(K) 설정]
parser.add_argument('--k_softargmax', type=int, default=10, help='Top-K points used for Soft-argmax')
parser.add_argument('--plane_knn', type=int, default=5, help='K points for Local Tangent Plane estimation')
parser.add_argument('--curv_knn', type=int, default=30, help='K points for Macroscopic Curvature & Direction estimation')

# [NotebookLM 참고: 기하학적 페널티 승수] 
# 수식: Loss = P2P_Distance * (1 + alpha*Curv_Error + beta*Dir_Error)
# 예측점이 GT의 굴곡(Curvature)이나 뼈대 방향(Direction)을 벗어날 때 오차를 기하급수적으로 증폭시키는 변수입니다.
parser.add_argument('--curv_alpha', type=float, default=10.0, help='[Alpha] Penalty multiplier for curvature magnitude error')
parser.add_argument('--dir_beta', type=float, default=1.0, help='[Beta] Penalty multiplier for eigenvector direction error')

# [Focal Loss 및 스케일 정규화]
parser.add_argument('--focal_gamma', type=float, default=1.0, help='Gamma for dynamic focal loss')
parser.add_argument('--focal_max', type=float, default=10.0, help='Max clamp for focal weights')
parser.add_argument('--use_loss_norm', type=str2bool, default=False, help='Use Initial Loss Normalization')
parser.add_argument('--target_norm', type=float, default=1.0, help='Target scale for Loss Normalization')

# =============================================================================
# [8] 🌟 Ablation Study 및 구조 제어 스위치 (Architecture & Loss Control)
# =============================================================================
# 1. 동적 히트맵 선학습(Warm-up) 설정
parser.add_argument('--use_warmup', type=str2bool, default=False, help='히트맵 로스 정체 기반 자동 선학습 켜기/끄기')
parser.add_argument('--warmup_patience', type=int, default=10, help='몇 에폭 동안 히트맵 로스가 안 떨어지면 RLW로 넘어갈지 결정')

# 2. 다이렉트 좌표 회귀 활성화 플래그 (DeepPA가 히트맵 대신 X,Y,Z를 직접 뱉도록 함)
parser.add_argument('--use_direct_regression', type=str2bool, default=True, help='DeepPA outputs (X,Y,Z) directly instead of heatmap')

# 3. [논문 방어용 Ablation] RLW 분기 제어 스위치
# - False (추천/기본값): 닻(Anchor) 모드. 히트맵을 1.0 가중치로 고정하고 기하학 3-Loss만 RLW로 경쟁시킵니다. (계층적 최적화)
# - True : 전면 랜덤 모드. 히트맵마저 RLW에 포함시켜 모든 4-Loss가 완전히 자율적으로 조율되게 합니다.
parser.add_argument('--use_rlw_for_heatmap', type=str2bool, default=False, help='히트맵 로스를 RLW 풀에 포함시킬지 여부 (False 시 1.0 고정 닻으로 작동)')

# =============================================================================
# [9] 🌟 논문 Equation (5) HDS 파라미터 (디펜스용)
# =============================================================================
parser.add_argument('--hds_alpha', type=float, default=0.3, help='Eq(5) L_sem 시작 가중치 (논문 최적값 0.3)')
parser.add_argument('--hds_beta', type=float, default=0.005, help='Eq(5) L_spa 시작 가중치 (논문 최적값 0.005)')
parser.add_argument('--hds_decay', type=float, default=0.95, help='수렴 안정성을 위한 지수 감쇠율')