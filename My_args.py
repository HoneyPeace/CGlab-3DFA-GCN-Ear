# @Author: Yuan Wang (Modified by Researcher Park Pyeong-hwa & AI Assistant)
# @File: My_args.py
# @Description: 
# [S2G 3D 랜드마크 탐지 모델 - SOTA 최적화 통제 센터]
# - 🌟 파라미터 통합: latent_injection_type (none / raw / compressed) 도입
# - 🌟 모델 리스트 및 스케줄러 파라미터(HDS 0.1 고정 등) 컨트롤러 동기화
# ==============================================================================

import argparse
import torch
import torch.nn as nn

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='S2G 3D Landmark Detection SOTA Configuration')

# [1] 기본 설정 & I/O 
parser.add_argument('--exp_name', type=str, default='S2G_Final_Refinement', metavar='N')

# 🌟 [수정됨] 컨트롤러 로직과 100% 일치하도록 모델 리스트 업데이트 (deepla_decay 포함)
parser.add_argument('--model', type=str, default='deeppa_frozen', 
                    choices=[
                        'paconv', 'paconv_heat', 'paconv_struct', 'deeppa_frozen', 'deeppa_frozen_no_heat', 'deeppa_finetune', 'deeppa_e2e',
                        'single_deeppa', # 🌟 명시적 추가
                        'deepla_ori', 'deepla_decay', 'deepla_all_tied', 'deepla_progress', # 🌟 deepla_decay 반영
                        'frozen_aux_drop', 'frozen_aux_fixed', 'frozen_no_aux' 
                    ])
parser.add_argument('--dataset', type=str, default='Ear296_Korean')
parser.add_argument('--data_root', type=str, default='../data')
parser.add_argument('--output_root', type=str, default='../results')
parser.add_argument('--train_dataset_name', type=str, default='Ear296_Korean')
parser.add_argument('--val_dataset_name', type=str, default='')
parser.add_argument('--val_partition', type=str, default='val')
parser.add_argument('--test_dataset_name', type=str, default='')
parser.add_argument('--run_id', type=str, default='')
parser.add_argument('--user_tag', type=str, default='')
parser.add_argument('--model_epoch', type=str, default="deeppa_frozen_last.t7")

parser.add_argument('--no_cuda', type=str2bool, default=False) 
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--in_channels', type=int, default=7)
parser.add_argument('--num_points', type=int, default=8192)
parser.add_argument('--landmark_num', type=int, default=36)

# [2] 학습 및 최적화 설정
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--test_batch_size', type=int, default=1)
parser.add_argument('--accumulation_steps', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--scheduler', type=str, default='step', choices=['cos', 'step'])
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=0.0)

# [3] 120층 DeepPA 백본 & 4단계 압축 설정
parser.add_argument('--depths', type=list, default=[20, 20, 60, 20])
parser.add_argument('--dims', type=list, default=[64, 128, 256, 512])
parser.add_argument('--npoints', type=list, default=[2048, 512, 128, 32])
parser.add_argument('--ks', type=list, default=[20, 20, 20, 20])
parser.add_argument('--nbr_dims', type=list, default=[64, 128, 256, 512])

parser.add_argument('--use_gate', type=str2bool, default=True) 
parser.add_argument('--use_cp', type=str2bool, default=False) 
parser.add_argument('--head_dim', type=int, default=256)
parser.add_argument('--mlp_ratio', type=float, default=2.0)
parser.add_argument('--bn_momentum', type=float, default=0.1)
parser.add_argument('--act', default=nn.GELU)

# =============================================================================
# 🌟 [직관성 극대화] 피처 주입 3지 선다형 옵션 도입
# =============================================================================
parser.add_argument('--latent_injection_type', type=str, default='raw', 
                    choices=['none', 'raw', 'compressed'], 
                    help="'none': 주입 없음 / 'raw': 1344ch 통짜 주입 / 'compressed': 각 층별 채널 맞춤 주입")
parser.add_argument('--use_feature_gating', type=str2bool, default=False, 
                    help='True: 주입 시 게이팅 어텐션 사용 / False: 단순 잔차 덧셈')

# [4] 자율 로스 게이팅 & 스케줄링 설정
parser.add_argument('--use_interaction_fusion', type=str2bool, default=True,
                    help='True: use fusion_mlp interaction residual for PAConv hints')
parser.add_argument('--coord_from_heatmap', type=str2bool, default=True,
                    help='True: derive final DeepPA coordinates from main heatmap by differentiable top-k')

parser.add_argument('--yield_factor', type=float, default=0.8)
parser.add_argument('--lambda_anchor', type=float, default=0.1)
parser.add_argument('--gating_tau', type=float, default=0.65)
parser.add_argument('--gating_beta', type=float, default=15.0)

# [5] 하이브리드 표면 기하 로스 & 통제 스위치
parser.add_argument('--use_jitter', type=str2bool, default=False) 
parser.add_argument('--use_loss_norm', type=str2bool, default=False) 
parser.add_argument('--use_rlw_for_pred', type=str2bool, default=False) 
parser.add_argument('--target_norm', type=float, default=1.0)

# 🌟 [수정됨] 컨트롤러 연동 파라미터 (HDS 0.1 고정 반영)
parser.add_argument('--patience', type=int, default=5, help='에폭 정체 대기 한도')
parser.add_argument('--val_decay_step', type=float, default=0.05, help='정체 시 깎이는 메인 가중치 비율')
parser.add_argument('--hds_buffer', type=float, default=0.1, help='Aux(HDS) 기본 고정 가중치 (0.3에서 0.1로 하향)')

parser.add_argument('--min_heatmap_warmup', type=int, default=30,
                    help='Minimum heatmap-only warmup epochs before val-based transition')

parser.add_argument('--regression_point_num', type=int, default=10)
parser.add_argument('--plane_knn', type=int, default=5)
parser.add_argument('--curv_knn', type=int, default=30)
parser.add_argument('--curv_alpha', type=float, default=10.0)
parser.add_argument('--dir_beta', type=float, default=1.0)
parser.add_argument('--focal_gamma', type=float, default=1.0)

# [6] 데이터 전처리 및 PAConv 내부 파라미터 
parser.add_argument('--need_resample', type=str2bool, default=True) 
parser.add_argument('--sample_way', type=str, default='FPS')
parser.add_argument('--dataset_seed', type=int, default=1)
parser.add_argument('--sigma', type=float, default=2.5)
parser.add_argument('--train_len', type=int, default=209)
parser.add_argument('--Eval_DataType', type=str, default="test")
parser.add_argument('--eval', type=str2bool, default=False) 

parser.add_argument('--k', type=int, default=30)
parser.add_argument('--calc_scores', type=str, default='softmax')
parser.add_argument('--hidden', type=list, default=[[32], [32], [32], [32]])
parser.add_argument('--num_matrices', type=list, default=[8, 8, 8, 8])

# =============================================================================
# [DEPRECATED] 미사용 및 구버전 잔재 (사용하지 않음) - 정리 완료
# =============================================================================
'''
# 폐기된 옵션: 259채널 슬림화 및 다이렉트 회귀 도입으로 인한 미사용 파라미터
parser.add_argument('--use_direct_regression', type=str2bool, default=True)
parser.add_argument('--use_spatial_attention', type=str2bool, default=False)
parser.add_argument('--up_dims', type=list, default=[128, 128, 256, 256])
parser.add_argument('--use_split_dataset', type=str2bool, default=True)
parser.add_argument('--max_threshold', default=10, type=float)
parser.add_argument('--loss', type=str, default='adaptive_wing')
parser.add_argument('--use_sgd', type=str2bool, default=False)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--emb_dims', type=int, default=1024)
parser.add_argument('--model_path', type=str, default='')
'''
