'''
@Author: Yuan Wang (Modified by Researcher)
@File: My_args.py
@Description: 
[NotebookLM을 위한 핵심 파일 요약]
이 파일은 0.47mm 3D 귀/얼굴 랜드마크 검출 모델의 하이퍼파라미터 통제 센터입니다.
불필요한 휴리스틱 튜닝(RLW 등)을 전면 배제하고, CVPR Eq.5 HDS 커리큘럼과 
독립 덧셈형 3D 표면 기하학 로스를 제어하는 변수들로만 무결점하게 구성되었습니다.
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

# 🔥 'deeppa_auto' 및 'deeppa_frozen' 옵션 유지
# 1단계(PAConv): 전체적인 형태를 보고 랜드마크의 대략적인 위치(Global Context) 탐지 (Frozen)
# 2단계(DeepPA): 기하학적 로스를 켜서 표면 굴곡에 완벽히 밀착(Local Refinement)
parser.add_argument('--model', type=str, default='deeppa_frozen', metavar='N', choices=['PAConv_heat', 'PAConv', 'DeepLA', 'DeepPA', 'deeppa_frozen', 'deeppa_e2e'], help='Model to use')
parser.add_argument('--no_cuda', type=str2bool, default=False, help='enables CUDA training')
parser.add_argument('--model_path', type=str, default='', metavar='N', help='Pretrained model path')

parser.add_argument('--dataset', type=str, default='Ear296_Korean', help='Target dataset name')
parser.add_argument('--train_dataset_name', type=str, default='Ear296_Korean', help='Train dataset name')
parser.add_argument('--test_dataset_name', type=str, default='', help='Test dataset name (Optional)')

parser.add_argument('--data_root', type=str, default='../data', help='Root directory of data')
parser.add_argument('--output_root', type=str, default='../results', help='Root directory for results')

# 🌟 입력 채널 설정 (기본 7채널: xyz + 주방향 + 곡률)
# 단순 3D 좌표뿐만 아니라, 오프라인 베이킹된 기하학적 특징(방향/곡률)을 주입하여 모델 수렴을 가속
parser.add_argument('--in_channels', type=int, default=7, help='Input channels: 3 for (xyz), 7 for (xyz + principal_dir + curvature)')


# =============================================================================
# [2] 학습 설정 (Train Args)
# =============================================================================
parser.add_argument('--eval', type=str2bool, default=False, help='evaluate the model')
parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size', help='Size of batch')
parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size', help='Size of batch')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of episode to train')

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
# [4] 데이터 전처리 (Data Process & Heatmap)
# =============================================================================
parser.add_argument('--max_threshold', default=10, type=float, help='the maximum threshold of error_rate')
parser.add_argument('--sample_way', type=str, default='FPS', metavar='sw', choices=['FPS', 'Random', 'CAGQ', 'Geometric'])
parser.add_argument('--need_resample', type=str2bool, default=True, help='Must be True to generate NPY files initially')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')

parser.add_argument('--regression_point_num', type=int, default=10, metavar='RPN', help='points in landmark regression')
parser.add_argument('--dataset_seed', type=int, default=1, metavar='S', help='train/test dataset random seed')
parser.add_argument('--num_points', type=int, default=8192, help='num of points to use')

# 3D 가우시안 히트맵의 분산(퍼짐 정도)
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
parser.add_argument('--model_epoch', type=str, default="Frozen_Hybrid_last.t7", help='load trained model file')
parser.add_argument('--run_id', type=str, default='', help='Load specific run from backup (e.g., 1, 2)')
parser.add_argument('--use_split_dataset', type=str2bool, default=True, help='Use split dataset mode')
parser.add_argument('--user_tag', type=str, default='', help='Custom tag added to the folder name')
parser.add_argument('--train_len', type=int, default=209, help='Number of training samples used in folder name')

# =============================================================================
# [7] 🌟 핵심 방어 논리 1: 하이브리드 표면 페널티 (loss.py 연동)
# =============================================================================
parser.add_argument('--plane_knn', type=int, default=5, help='K points for Local Tangent Plane estimation')
parser.add_argument('--curv_knn', type=int, default=30, help='K points for Macroscopic Curvature & Direction estimation')

# [논문 디펜스 포인트]: 곱셈 교차항 오차를 막기 위한 "독립 선형 덧셈 패널티" 승수
# Loss = P2P_Distance * (1 + alpha * Curv_Error + beta * Dir_Error)
parser.add_argument('--curv_alpha', type=float, default=10.0, help='[Alpha] Penalty multiplier for curvature magnitude error')
parser.add_argument('--dir_beta', type=float, default=1.0, help='[Beta] Penalty multiplier for eigenvector direction error')

# =============================================================================
# [8] 🌟 핵심 방어 논리 2: 최적화 안정성 및 회귀 제어
# =============================================================================
parser.add_argument('--use_direct_regression', type=str2bool, default=True, help='DeepPA가 히트맵 대신 (X,Y,Z) 좌표를 다이렉트로 출력')

# [표준 Focal L1 설정]: 수렴이 불안정한 Dynamic 방식 제거, 검증된 Standard 방식 채택
parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma for Standard Focal L1 loss')
parser.add_argument('--use_loss_norm', type=str2bool, default=False, help='Use Initial Loss Normalization')
parser.add_argument('--target_norm', type=float, default=1.0, help='Target scale for Loss Normalization')

# =============================================================================
# [9] 🌟 핵심 방어 논리 3: DeepLA-Net CVPR Eq. (5) HDS 스케줄링
# =============================================================================
# [논문 디펜스 포인트]: 자의적인 감쇄율 튜닝(Heuristic)을 배제하고, SOTA 논문의 지수 감쇠(n=1/epoch) 수식을 완벽히 차용함.
parser.add_argument('--hds_alpha', type=float, default=0.3, help='Eq(5) L_sem 시작 가중치 (논문 최적값 0.3)')
parser.add_argument('--hds_beta', type=float, default=0.005, help='Eq(5) L_spa 시작 가중치 (논문 최적값 0.005)')