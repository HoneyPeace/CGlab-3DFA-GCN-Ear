'''
@Author: Yuan Wang (Modified by Researcher Park Pyeong-hwa & AI Assistant)
@File: My_args.py
@Description: 
[S2G 3D 랜드마크 탐지 모델 - SOTA 최적화 통제 센터]
- argparse 중복 에러 해결 및 파라미터 단일화 완료
- 8192 해상도에 맞춘 Multi-scale FPS(2048->512->128->32) 정규화 완료
- Universal Pipeline(CW-KD, Yielding, Finetune) 파라미터 동기화
'''

import argparse
import torch
import torch.nn as nn

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='S2G 3D Landmark Detection SOTA Configuration')

# =============================================================================
# [1] 기본 설정 & I/O 
# =============================================================================
parser.add_argument('--exp_name', type=str, default='S2G_Final_Refinement', metavar='N')
# 🌟 [수정 1] 지원하는 5가지 모델 모드로 선택지(choices) 완벽 동기화
parser.add_argument('--model', type=str, default='deeppa_frozen', 
                    choices=['paconv', 'deeppa', 'deeppa_frozen', 'deeppa_finetune', 'deeppa_e2e'])
parser.add_argument('--dataset', type=str, default='Ear296_Korean')
parser.add_argument('--data_root', type=str, default='../data')
parser.add_argument('--output_root', type=str, default='../results')
parser.add_argument('--train_dataset_name', type=str, default='Ear296_Korean')
parser.add_argument('--test_dataset_name', type=str, default='')
parser.add_argument('--run_id', type=str, default='')
parser.add_argument('--user_tag', type=str, default='')
parser.add_argument('--model_epoch', type=str, default="deeppa_frozen_last.t7")

parser.add_argument('--no_cuda', type=str2bool, default=False)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--in_channels', type=int, default=7, help='7(XYZ+Dir+Curv) or 3(XYZ)')
parser.add_argument('--num_points', type=int, default=8192)
parser.add_argument('--landmark_num', type=int, default=36)

# =============================================================================
# [2] 학습 및 최적화 설정
# =============================================================================
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--test_batch_size', type=int, default=1)
parser.add_argument('--accumulation_steps', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--scheduler', type=str, default='step', choices=['cos', 'step'])
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=0.0)

# =============================================================================
# [3] 120층 DeepPA 백본 & 4단계 압축 설정
# =============================================================================
parser.add_argument('--depths', type=list, default=[20, 20, 60, 20])
parser.add_argument('--dims', type=list, default=[64, 128, 256, 512])

# 8192점에 맞는 1/4 Downsampling 비율
parser.add_argument('--npoints', type=list, default=[2048, 512, 128, 32], help='FPS Multi-scale Downsampling')

parser.add_argument('--ks', type=list, default=[20, 20, 20, 20])
parser.add_argument('--nbr_dims', type=list, default=[64, 128, 256, 512])

parser.add_argument('--use_gate', type=str2bool, default=True, help='Enable Gated Residual Fusion')
parser.add_argument('--use_cp', type=str2bool, default=False, help='Gradient Checkpointing')
parser.add_argument('--head_dim', type=int, default=256)
parser.add_argument('--mlp_ratio', type=float, default=2.0)
parser.add_argument('--bn_momentum', type=float, default=0.1)
parser.add_argument('--act', default=nn.GELU)

# =============================================================================
# [4] 자율 로스 게이팅 & 하이브리드 지식 증류 설정
# =============================================================================
# 🌟 [신규] train.py에서 사용하는 양보 계수 및 KD 가중치 파라미터 추가
parser.add_argument('--yield_factor', type=float, default=0.8, help='구조 로스 개방 시 히트맵 비중 양보율 (rho)')
parser.add_argument('--lambda_anchor', type=float, default=0.1, help='Finetune 모드 시 교사(PAConv) 보호 닻 가중치')
parser.add_argument('--lambda_kd', type=float, default=1.0, help='채널별 지식 증류(CW-KD) 기본 가중치')

parser.add_argument('--gating_tau', type=float, default=0.65, help='기하 로스 개방을 위한 히트맵 실력 임계점')
parser.add_argument('--gating_beta', type=float, default=15.0, help='시그모이드 곡선의 가파름')

# =============================================================================
# [5] 하이브리드 표면 기하 로스 & 통제 스위치
# =============================================================================
parser.add_argument('--use_jitter', type=str2bool, default=False, help='가우시안 노이즈 증강 활성화')
# 🌟 [수정 2] 요청하신 대로 RLW와 로스 스케일링 디폴트를 False로 변경
parser.add_argument('--use_loss_norm', type=str2bool, default=False, help='초기값 기반 기하 로스 스케일링 활성화')
parser.add_argument('--use_rlw_for_pred', type=str2bool, default=False, help='기하 로스 전용 Random Loss Weighting 활성화')
parser.add_argument('--target_norm', type=float, default=1.0)

# 🌟 로스 디테일 하이퍼파라미터
parser.add_argument('--regression_point_num', type=int, default=10, help='K points for Soft-Argmax')
parser.add_argument('--plane_knn', type=int, default=5, help='Local Tangent Plane KNN')
parser.add_argument('--curv_knn', type=int, default=30, help='Curvature KNN')
parser.add_argument('--curv_alpha', type=float, default=10.0)
parser.add_argument('--dir_beta', type=float, default=1.0)
parser.add_argument('--focal_gamma', type=float, default=1.0)
parser.add_argument('--hds_alpha', type=float, default=0.3)
parser.add_argument('--hds_beta', type=float, default=0.005)

# =============================================================================
# [6] 데이터 전처리 및 PAConv 내부 파라미터 
# =============================================================================
parser.add_argument('--need_resample', type=str2bool, default=True)
parser.add_argument('--sample_way', type=str, default='FPS')
parser.add_argument('--dataset_seed', type=int, default=1)
parser.add_argument('--sigma', type=float, default=2.5)
parser.add_argument('--train_len', type=int, default=209)
parser.add_argument('--Eval_DataType', type=str, default="test")
parser.add_argument('--eval', type=str2bool, default=False)

parser.add_argument('--k', type=int, default=30, help='PAConv base KNN')
parser.add_argument('--calc_scores', type=str, default='softmax')
parser.add_argument('--hidden', type=list, default=[[32], [32], [32], [32]])
parser.add_argument('--num_matrices', type=list, default=[8, 8, 8, 8])


# =============================================================================
# [DEPRECATED] 미사용 및 구버전 잔재 (사용하지 않음)
# =============================================================================
'''
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--train_dataset_name', type=str, default='Ear296_Korean')
parser.add_argument('--test_dataset_name', type=str, default='')
parser.add_argument('--eval', type=str2bool, default=False)
parser.add_argument('--test_batch_size', type=int, default=1)
parser.add_argument('--accumulation_steps', type=int, default=1)
parser.add_argument('--loss', type=str, default='adaptive_wing')
parser.add_argument('--use_sgd', type=str2bool, default=False)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_threshold', default=10, type=float)
parser.add_argument('--sample_way', type=str, default='FPS')
parser.add_argument('--need_resample', type=str2bool, default=True)
parser.add_argument('--dataset_seed', type=int, default=1)
parser.add_argument('--sigma', type=float, default=10.0)
parser.add_argument('--k', type=int, default=30)
parser.add_argument('--emb_dims', type=int, default=1024)
parser.add_argument('--calc_scores', type=str, default='softmax')
parser.add_argument('--hidden', type=list, default=[[32], [32], [32], [32]])
parser.add_argument('--num_matrices', type=list, default=[8, 8, 8, 8])
parser.add_argument('--Eval_DataType', type=str, default="test")
parser.add_argument('--model_epoch', type=str, default="Frozen_Hybrid_last.t7")
parser.add_argument('--run_id', type=str, default='')
parser.add_argument('--use_split_dataset', type=str2bool, default=True)
parser.add_argument('--user_tag', type=str, default='')
parser.add_argument('--train_len', type=int, default=209)
parser.add_argument('--use_loss_norm', type=str2bool, default=False)
parser.add_argument('--target_norm', type=float, default=1.0)
parser.add_argument('--up_dims', type=list, default=[128, 128, 256, 256])

# 259채널 슬림화 및 다이렉트 회귀 도입으로 폐기된 옵션
parser.add_argument('--use_direct_regression', type=str2bool, default=True)
parser.add_argument('--use_spatial_attention', type=str2bool, default=False)
parser.add_argument('--use_rlw_for_pred', type=str2bool, default=False)
'''