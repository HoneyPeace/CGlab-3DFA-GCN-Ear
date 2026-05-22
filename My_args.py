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
parser.add_argument('--train_dataset_name', type=str, default='train')
parser.add_argument('--val_dataset_name', type=str, default='valiation')
parser.add_argument('--val_partition', type=str, default='val')
parser.add_argument('--test_dataset_name', type=str, default='test')
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
parser.add_argument('--stage_downsample_method', type=str, default='fps',
                    choices=['fps', 'grid'],
                    help='DeepPA stage downsampling method. Default fps preserves previous behavior.')
parser.add_argument('--stage_grid_sizes', type=str, default='',
                    help='Comma-separated grid sizes for DeepPA grid downsampling. Empty enables adaptive search.')
parser.add_argument('--stage_grid_search_iters', type=int, default=8,
                    help='Adaptive grid-size search iterations when --stage_downsample_method grid and no grid size is provided.')

parser.add_argument('--use_gate', type=str2bool, default=True) 
parser.add_argument('--use_cp', type=str2bool, default=False) 
parser.add_argument('--head_dim', type=int, default=256)
parser.add_argument('--decoder_fusion', type=str, default='add',
                    choices=['add', 'raw_concat', 'prog_half_final320'],
                    help='DeepPA decoder fusion: add baseline / raw stage concat / progressive half-compress with final 320ch')
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
parser.add_argument('--fusion_residual_base', type=str, default='prior',
                    choices=['deeppa', 'prior'],
                    help="'deeppa': x + fusion_mlp([x, prior]) / 'prior': prior + fusion_mlp([x, prior])")
parser.add_argument('--unfreeze_paconv_in_frozen', type=str2bool, default=False,
                    help='True: allow PAConv parameters to receive gradients in frozen pipeline')
parser.add_argument('--frozen_paconv_lr_scale', type=float, default=1.0,
                    help='Learning-rate scale for PAConv when --unfreeze_paconv_in_frozen is True')
parser.add_argument('--frozen_paconv_hm_weight', type=float, default=0.0,
                    help='PAConv heatmap loss weight used when PAConv is unfrozen in frozen pipeline')
parser.add_argument('--coord_from_heatmap', type=str2bool, default=True,
                    help='True: derive final DeepPA coordinates from main heatmap by differentiable top-k')
parser.add_argument('--train_coord_readout', type=str, default='topk',
                    choices=['topk', 'softargmax', 'soft_ot_topk', 'sigmoid_xyz_pool',
                             'heatmap_attn_residual', 'heatmap_attn_residual_feature_only'],
                    help='Training-only DeepPA coordinate readout for Crd/Srf/Str losses')
parser.add_argument('--heatmap_activation_mode', type=str, default='sigmoid',
                    choices=['sigmoid', 'softmax'],
                    help='Activation used for main/aux DeepPA heatmap outputs')
parser.add_argument('--heatmap_activation_temperature', type=float, default=1.0,
                    help='Point-wise softmax temperature for --heatmap_activation_mode softmax')
parser.add_argument('--paconv_heatmap_activation_mode', type=str, default='raw',
                    choices=['softmax', 'sigmoid', 'raw'],
                    help='Activation used for PAConv heatmap output; default keeps final PAConv heatmap logits raw')
parser.add_argument('--paconv_feature_mode', type=str, default='center_geometry',
                    choices=['center_geometry', 'full_extension'],
                    help='PAConv 7ch edge feature mode: center_geometry uses XYZ relation plus center geometry; full_extension uses center/neighbor/delta for all input channels')
parser.add_argument('--deeppa_feature_mode', type=str, default='center_geometry',
                    choices=['center_geometry', 'full_extension'],
                    help='DeepPA 7ch local feature mode: center_geometry uses XYZ relation plus center geometry; full_extension uses center/neighbor/delta for all input channels')
parser.add_argument('--heatmap_loss_mode', type=str, default='adaptive_wing',
                    choices=['adaptive_wing', 'softmax_ce'],
                    help='Heatmap supervision for main/aux heatmaps')
parser.add_argument('--heatmap_softmax_temperature', type=float, default=1.0,
                    help='Point-wise softmax temperature for --heatmap_loss_mode softmax_ce')
parser.add_argument('--coord_loss_mode', type=str, default='focal_l1',
                    choices=['focal_l1', 'expected_distance', 'ranking'],
                    help='Coordinate/heatmap geometry supervision used as L_coord')
parser.add_argument('--struct_loss_mode', type=str, default='coord',
                    choices=['coord', 'heatmap'],
                    help='Structural loss source: coordinate pairwise distance or heatmap distribution moment')
parser.add_argument('--hm_struct_temperature', type=float, default=1.0,
                    help='Temperature for heatmap-distribution structural loss')
parser.add_argument('--softargmax_temperature', type=float, default=1.0,
                    help='Temperature for soft-argmax/expected-distance/ranking heatmap distributions')
parser.add_argument('--soft_topk_k', type=int, default=0,
                    help='Selected-point count for --train_coord_readout soft_ot_topk; <=0 uses --regression_point_num')
parser.add_argument('--soft_topk_epsilon', type=float, default=0.1,
                    help='Entropic regularization epsilon for OT-based soft top-k readout')
parser.add_argument('--soft_topk_iters', type=int, default=50,
                    help='Sinkhorn iterations for OT-based soft top-k readout')
parser.add_argument('--hm_attn_residual_max_mm', type=float, default=0.0,
                    help='Max heatmap-attention residual correction in mm for --train_coord_readout heatmap_attn_residual; <=0 uses --hm_attn_residual_max_norm')
parser.add_argument('--hm_attn_residual_max_norm', type=float, default=0.0,
                    help='Fallback max residual correction in normalized coordinates for heatmap_attn_residual; <=0 disables residual clamp')
parser.add_argument('--ranking_margin_scale', type=float, default=1.0,
                    help='Distance margin scale for ranking heatmap loss')
parser.add_argument('--ablation_only', type=str, default='all',
                    choices=['all', 'frozen_aux_fixed', 'frozen_aux_drop', 'frozen_no_aux'],
                    help='run_frozen.py only: run one ablation model instead of all')
parser.add_argument('--stage1_user_tag', type=str, default='',
                    help='run_frozen.py only: reuse a PAConv Stage1 tag different from --user_tag')
parser.add_argument('--stage1_exp_name', type=str, default='',
                    help='run_frozen.py only: reuse PAConv Stage1 from a different experiment folder')
parser.add_argument('--stage1_paconv_source', type=str, default='default',
                    choices=['default', 'original_github'],
                    help='frozen/eval only: use default PAConv or original GitHub PAConv as Stage1')
parser.add_argument('--stage1_checkpoint_path', type=str, default='',
                    help='Optional explicit Stage1 PAConv checkpoint path for frozen/finetune/e2e modes')
parser.add_argument('--stage1_use_cuda_extension', type=str2bool, default=False,
                    help='original_github Stage1 only: use PAConv CUDA assemble extension if available')
parser.add_argument('--stage1_in_channels', type=int, choices=[3, 7], default=3,
                    help='original_github Stage1 only: number of input channels to feed/load')

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

parser.add_argument('--aux_drop_epochs', type=int, default=30,
                    help='Epoch count for frozen_aux_drop linear aux heatmap decay')
parser.add_argument('--e2e_aux_mode', type=str, default='fixed',
                    choices=['fixed', 'drop', 'none'],
                    help='deeppa_e2e only: fixed aux weight, linear aux drop, or no aux heatmap loss')
parser.add_argument('--finetune_aux_mode', type=str, default='fixed',
                    choices=['fixed', 'drop', 'none'],
                    help='deeppa_finetune only: fixed aux weight, linear aux drop, or no aux heatmap loss')
parser.add_argument('--e2e_load_paconv_pretrained', type=str2bool, default=False,
                    help='deeppa_e2e only: load PAConv_Pretrained/models/Single_PAConv_last.t7 into stage1 PAConv')
parser.add_argument('--e2e_staged_paconv', type=str2bool, default=False,
                    help='deeppa_e2e only: PAConv-only warmup, then weak PAConv anchor with DeepPA HM/Geom schedule')
parser.add_argument('--e2e_paconv_warmup_epochs', type=int, default=30,
                    help='deeppa_e2e staged PAConv-only warmup epochs')
parser.add_argument('--e2e_paconv_final_weight', type=float, default=0.1,
                    help='deeppa_e2e staged PAConv heatmap anchor weight after warmup')
parser.add_argument('--e2e_paconv_lr_scale', type=float, default=0.1,
                    help='deeppa_e2e PAConv learning-rate scale relative to DeepPA')
parser.add_argument('--use_stagewise_aux_hm', type=str2bool, default=True,
                    help='Compare each auxiliary heatmap with GT heatmap gathered at the same stage point indices')
parser.add_argument('--min_heatmap_warmup', type=int, default=30,
                    help='Minimum heatmap-only warmup epochs before val-based transition')
parser.add_argument('--loss_schedule', type=str, default='val_adaptive',
                    choices=['val_adaptive', 'fixed_three_phase', 'linear_three_phase', 'train_hm_plateau'],
                    help='val_adaptive: existing validation-based transition / fixed_three_phase: fixed heatmap then fixed geometry weights / linear_three_phase: fixed heatmap then linear transition to final weights / train_hm_plateau: train Main_HM plateau-triggered transition')
parser.add_argument('--fixed_heatmap_epochs', type=int, default=60,
                    help='Epoch count for heatmap-only phase when --loss_schedule fixed_three_phase')
parser.add_argument('--fixed_main_weight', type=float, default=0.9,
                    help='Fixed main heatmap weight after fixed_heatmap_epochs')
parser.add_argument('--fixed_geom_weight', type=float, default=0.1,
                    help='Fixed coordinate/surface/structure weight after fixed_heatmap_epochs')
parser.add_argument('--plateau_start_epoch', type=int, default=30,
                    help='Start checking train Main_HM plateau after this epoch when --loss_schedule train_hm_plateau')
parser.add_argument('--plateau_window', type=int, default=20,
                    help='Rolling window size for train Main_HM plateau detection')
parser.add_argument('--plateau_patience', type=int, default=10,
                    help='Number of non-improving rolling windows before plateau-triggered transition')
parser.add_argument('--plateau_threshold', type=float, default=0.01,
                    help='Relative train Main_HM rolling improvement threshold for plateau detection')
parser.add_argument('--plateau_transition_epochs', type=int, default=30,
                    help='Epochs used to ramp from HM-only to fixed_main/fixed_geom after train HM plateau')
parser.add_argument('--plateau_transition_mode', type=str, default='linear',
                    choices=['linear', 'step'],
                    help='Transition shape after train HM plateau: linear ramp or stepwise stages')
parser.add_argument('--plateau_step_size', type=float, default=0.1,
                    help='Geometry weight increment per step when --plateau_transition_mode step')

parser.add_argument('--regression_point_num', type=int, default=10)
parser.add_argument('--eval_heatmap_coord_method', type=str, default='topk',
                    choices=['topk', 'mds'],
                    help='Evaluation heatmap-to-coordinate method for single PAConv heatmap models')
parser.add_argument('--plane_knn', type=int, default=5)
parser.add_argument('--curv_knn', type=int, default=30)
parser.add_argument('--curv_alpha', type=float, default=10.0)
parser.add_argument('--dir_beta', type=float, default=1.0)
parser.add_argument('--surface_loss_mode', type=str, default='topk',
                    choices=['topk', 'soft_local'],
                    help='Surface/curvature loss neighborhood mode')
parser.add_argument('--soft_curv_sigma', type=float, default=0.0,
                    help='Soft local curvature sigma; <=0 uses GT kNN radius as adaptive sigma')
parser.add_argument('--soft_curv_min_sigma', type=float, default=1e-4,
                    help='Minimum sigma clamp for soft local curvature')
parser.add_argument('--focal_gamma', type=float, default=1.0)

# [6] 데이터 전처리 및 PAConv 내부 파라미터 
parser.add_argument('--need_resample', type=str2bool, default=True) 
parser.add_argument('--sample_way', type=str, default='FPS')
parser.add_argument('--geom_batch_size', type=int, default=8,
                    help='Batch size for 7-channel geometric feature baking during resampling')
parser.add_argument('--dataset_seed', type=int, default=1)
parser.add_argument('--sigma', type=float, default=2.5)
parser.add_argument('--train_len', type=int, default=200)
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
