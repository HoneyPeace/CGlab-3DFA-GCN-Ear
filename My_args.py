'''
@Author: Yuan Wang (Modified by Researcher 5)
@File: My_args.py
@Description: Standard Defaults (Relative Path, Batch 32, Accum 1) + Research Params (4096pts, Sigma 3.0)
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
# [1] 기본 설정
# =============================================================================
parser.add_argument('--exp_name', type=str, default='Ear_Project_Final', help='Name of the experiment')
parser.add_argument('--tag', type=str, default='', help='Comment for folder name')
parser.add_argument('--model', type=str, default='PAConv', choices=['EdgeConv', 'PAConv'], help='Model to use')
parser.add_argument('--no_cuda', type=str2bool, default=False, help='enables CUDA training')
parser.add_argument('--model_path', type=str, default='', help='Pretrained model path')

parser.add_argument('--dataset', type=str, default='train', help='Target dataset name')
parser.add_argument('--train_dataset_name', type=str, default='train', help='Train dataset name')
parser.add_argument('--test_dataset_name', type=str, default='test', help='Test dataset name')

# [수정] 상대 경로로 복귀 ('../Data')
parser.add_argument('--data_root', type=str, default='../Data', help='Root directory of data')
parser.add_argument('--output_root', type=str, default='../results', help='Root directory for results')

# =============================================================================
# [2] 학습 설정
# =============================================================================
parser.add_argument('--eval', type=str2bool, default=False, help='evaluate the model')

# [수정] 배치 32 (Standard)
parser.add_argument('--batch_size', type=int, default=32, help='Physical Batch Size')
# [수정] 누적 1 (Standard - No Accumulation)
parser.add_argument('--accum_iter', type=int, default=1, help='Accumulate gradients N times')

parser.add_argument('--test_batch_size', type=int, default=1, help='Size of batch')
parser.add_argument('--epochs', type=int, default=250, help='number of episode to train')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')

# =============================================================================
# [3] 최적화 설정
# =============================================================================
parser.add_argument('--loss', type=str, default='adaptive_wing', choices=['mse', 'adaptive_wing'], help='loss function to use')
parser.add_argument('--use_sgd', type=str2bool, default=False, help='Use SGD')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--scheduler', type=str, default='step', choices=['cos', 'step'], help='Scheduler to use')
parser.add_argument('--weight_decay', type=float, default=0, help='the weight decay')

# =============================================================================
# [4] 데이터 처리
# =============================================================================
parser.add_argument('--max_threshold', default=10, type=float, help='the maximum threshold of error_rate')
parser.add_argument('--sample_way', type=str, default='FPS', choices=['FPS', 'Random', 'CAGQ', 'Geometric'])
parser.add_argument('--need_resample', type=str2bool, default=True, help='Generate NPY files')

parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--regression_point_num', type=int, default=10, help='points in landmark regression')
parser.add_argument('--dataset_seed', type=int, default=1, help='train/test dataset random seed')

# [유지] 연구 핵심 파라미터 (4096, Sigma 3.0)
parser.add_argument('--num_points', type=int, default=4096, help='num of points to use')
parser.add_argument('--sigma', type=float, default=3.0, help='Gaussian Variance of heatmap')

parser.add_argument('--k', type=int, default=30, help='Num of nearest neighbors')
parser.add_argument('--emb_dims', type=int, default=1024, help='Dimension of embeddings')
parser.add_argument('--landmark_num', type=int, default=40, help='the number of landmark')

# =============================================================================
# [5] 모델 구조 설정
# =============================================================================
parser.add_argument('--calc_scores', type=str, default='softmax', help='The way to calculate score')
parser.add_argument('--hidden', type=list, default=[[32], [32], [32], [32]], help='the hidden layers of ScoreNet')
parser.add_argument('--num_matrices', type=list, default=[8, 8, 8, 8], help='the number of weight banks')

# =============================================================================
# [6] 기타 설정
# =============================================================================
parser.add_argument('--Eval_DataType', type=str, default="test", help='select npy train, test, sample')
parser.add_argument('--model_epoch', type=str, default="model_epoch_250.t7", help='load trained model file')
parser.add_argument('--run_id', type=str, default='', help='Load specific run from backup')
parser.add_argument('--use_split_dataset', type=str2bool, default=True, help='Use split dataset mode')