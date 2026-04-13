'''
@Author: Yuan Wang (Modified by Researcher 2)
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
parser.add_argument('--model', type=str, default='DeepLA', metavar='N', choices=['PAConv_heat', 'PAConv', 'DeepLA', 'DeepPA', 'DeepPA_auto'], help='Model to use')
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
parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size', help='Size of batch')
parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size', help='Size of batch')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of episode to train')

# 🔥 Auto 파이프라인 전용 에폭 설정 추가
# [의도] --model이 'DeepPA_auto'일 때 작동하며, 1단계(paconv)와 2단계(deeppa)에 각각 몇 에폭씩 할당할지 결정합니다.
parser.add_argument('--paconv_epochs', type=int, default=500, help='PAConv stage epochs in auto mode')
parser.add_argument('--deeppa_epochs', type=int, default=500, help='DeepPA stage epochs in auto mode')

parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient Accumulation Steps')

# =============================================================================
# [3] 최적화 설정 (Optimizer Args)
# =============================================================================
# [참고] 'adaptive_wing' 로스는 loss.py에 정의된 비선형 히트맵 회귀 전용 오차 함수입니다.
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

# [sigma 의도] 3D 공간 상의 정답 랜드마크(GT)를 점 하나(Dirac delta)로 두면 학습이 어려우므로, 
# 주변으로 확률이 퍼지는 가우시안 히트맵을 만듭니다. sigma는 그 퍼지는 범위(분산)를 의미합니다.
parser.add_argument('--sigma', type=float, default=10.0, metavar='Sig', help='Gaussian Variance of heatmap')
parser.add_argument('--k', type=int, default=30, metavar='N', help='Num of nearest neighbors')
parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
parser.add_argument('--landmark_num', type=int, default=40, metavar='L', help='the number of landmark')

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
# [7] 하이브리드 로스 & 3D 투영 하이퍼파라미터 ( loss.py 와 연동됨 )
# =============================================================================
# [가중치 밸런스] 일반적인 유클리디안 거리(alpha)와 표면 수직 거리(beta) 간의 혼합 비율.
parser.add_argument('--alpha_init', type=float, default=0.5, help='Initial weight for Coordinate L1 Loss')
parser.add_argument('--beta_init', type=float, default=0.1, help='Initial weight for Point-to-Plane Surface Loss')

# [이웃 크기(K) 설정의 기하학적 차이]
# - plane_knn (5): 거리가 매우 가까운 5개의 점만 모아 '접평면(Tangent Plane)'과 '법선(Normal)'을 구함. (미시적)
# - curv_knn (30): 넓은 반경의 30개 점을 모아 뼈대의 주방향이나 거시적인 '곡률(Curvature)'을 파악함. (거시적)
parser.add_argument('--k_softargmax', type=int, default=10, help='Top-K points used for Soft-argmax')
parser.add_argument('--plane_knn', type=int, default=5, help='K points for Local Tangent Plane estimation')
parser.add_argument('--curv_knn', type=int, default=30, help='K points for Macroscopic Curvature & Direction estimation')

# [기하학적 페널티 승수] 예측점이 정답 표면의 곡률이나 뼈대 방향을 벗어났을 때, 오차를 얼마나 뻥튀기할 것인가.
parser.add_argument('--curv_alpha', type=float, default=10.0, help='Penalty multiplier for curvature magnitude error')
parser.add_argument('--dir_weight', type=float, default=1.0, help='Penalty multiplier for eigenvector direction error')

# [Focal Loss] 학습 시 계속 못 맞추는 악성 랜드마크에 가중치를 동적으로 더 부여함.
parser.add_argument('--focal_gamma', type=float, default=1.0, help='Gamma for dynamic focal loss')
parser.add_argument('--focal_max', type=float, default=10.0, help='Max clamp for focal weights')

# 👇 loss 스케일 정규화
# [의도] L1 거리, Point-to-Plane 거리, 곡률 오차 등 여러 로스가 섞일 때 단위나 크기가 다르면 학습이 무너짐.
# 이를 방지하기 위해 학습 초기(에폭 0)에 모든 로스 값의 스케일을 target_norm(1.0)에 맞게 강제 정규화하는 기능.
parser.add_argument('--use_loss_norm', type=str2bool, default=True, help='Use Initial Loss Normalization')
parser.add_argument('--target_norm', type=float, default=1.0, help='Target scale for Loss Normalization')