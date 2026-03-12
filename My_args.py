'''
@Author: Yuan Wang (Modified by Researcher 2)
@File: My_args.py
@Description: Added --train_len for explicit folder finding.
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
# [1] 기본 설정 (Base Args)
# =============================================================================
parser.add_argument('--exp_name', type=str, default='Ear_Project_Final', metavar='N', help='Name of the experiment')
parser.add_argument('--model', type=str, default='PAConv', metavar='N', choices=['EdgeConv', 'PAConv'], help='Model to use')
parser.add_argument('--no_cuda', type=str2bool, default=False, help='enables CUDA training')
parser.add_argument('--model_path', type=str, default='', metavar='N', help='Pretrained model path')

parser.add_argument('--dataset', type=str, default='Ear296_Korean', help='Target dataset name')
parser.add_argument('--train_dataset_name', type=str, default='Ear296_Korean', help='Train dataset name')
parser.add_argument('--test_dataset_name', type=str, default='', help='Test dataset name (Optional)')

parser.add_argument('--data_root', type=str, default='../data', help='Root directory of data')
parser.add_argument('--output_root', type=str, default='../results', help='Root directory for results')

# =============================================================================
# [2] 학습 설정 (Train Args)
# =============================================================================
parser.add_argument('--eval', type=str2bool, default=False, help='evaluate the model')
parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size', help='Size of batch')
parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size', help='Size of batch')
parser.add_argument('--epochs', type=int, default=250, metavar='N', help='number of episode to train')
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
# [4] 데이터 처리 (Data Process Args)
# =============================================================================
parser.add_argument('--max_threshold', default=10, type=float, help='the maximum threshold of error_rate')
parser.add_argument('--sample_way', type=str, default='FPS', metavar='sw', choices=['FPS', 'Random', 'CAGQ', 'Geometric'])
parser.add_argument('--need_resample', type=str2bool, default=True, help='Must be True to generate NPY files initially')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')

parser.add_argument('--regression_point_num', type=int, default=10, metavar='RPN', help='points in landmark regression')
parser.add_argument('--dataset_seed', type=int, default=1, metavar='S', help='train/test dataset random seed')
parser.add_argument('--num_points', type=int, default=2048, help='num of points to use')

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

# [추가됨] 평가 시 폴더명을 정확히 찾기 위해 사용 (학습 데이터 개수)
# 예: train.py가 train2290 으로 저장했다면 여기도 2290을 적어야 함
parser.add_argument('--train_len', type=int, default=2290, help='Number of training samples used in folder name')

# =============================================================================
# [새로 추가] 하이브리드 로스 & 3D 투영 하이퍼파라미터
# =============================================================================
parser.add_argument('--alpha_init', type=float, default=0.5, help='Initial weight for Coordinate L1 Loss')
parser.add_argument('--beta_init', type=float, default=0.1, help='Initial weight for Point-to-Plane Surface Loss')
parser.add_argument('--k_softargmax', type=int, default=10, help='Top-K points used for Soft-argmax')
parser.add_argument('--k_knn', type=int, default=10, help='K points for Local Tangent Plane estimation')