'''
@Author: Yuan Wang (Modified by Researcher 2)
@File: My_args.py
@Description: Fixed boolean argument parsing
'''

import argparse
import torch

# [추가] 문자열을 Boolean으로 정확히 변환하는 함수
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='3D Ear/Face Landmark Detection')

# =============================================================================
# [1] 기본 설정
# =============================================================================
parser.add_argument('--exp_name', type=str, default='Ear_Project', metavar='N', help='Name of the experiment')
parser.add_argument('--model', type=str, default='PAConv', metavar='N', choices=['EdgeConv', 'PAConv'], help='Model to use')
parser.add_argument('--no_cuda', type=str2bool, default=False, help='enables CUDA training') # type=bool -> str2bool 변경
parser.add_argument('--model_path', type=str, default='', metavar='N', help='Pretrained model path')

parser.add_argument('--dataset', type=str, default='Ear296_Korean', help='Target dataset name for Split Mode')
parser.add_argument('--train_dataset_name', type=str, default='Ear296_Korean_arg', help='Train folder name (Separate Mode)')
parser.add_argument('--test_dataset_name', type=str, default='', help='Test folder name')
# [수정] type=bool -> type=str2bool
parser.add_argument('--use_split_dataset', type=str2bool, default=False, help='Use 7:3 split for a single dataset')

parser.add_argument('--data_root', type=str, default='../Data', help='Root directory of data')
parser.add_argument('--output_root', type=str, default='../Result', help='Root directory for results')

# =============================================================================
# [2] 학습 설정
# =============================================================================
parser.add_argument('--eval', type=str2bool, default=False, help='evaluate the model') # type 변경
parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size', help='Size of batch')
parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size', help='Size of batch')
parser.add_argument('--epochs', type=int, default=250, metavar='N', help='number of episode to train')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')

# =============================================================================
# [3] 최적화 설정
# =============================================================================
parser.add_argument('--loss', type=str, default='adaptive_wing', metavar='N', choices=['mse', 'adaptive_wing'], help='loss function to use')
parser.add_argument('--use_sgd', type=str2bool, default=False, help='Use SGD') # type 변경
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--scheduler', type=str, default='step', metavar='N', choices=['cos', 'step'], help='Scheduler to use')
parser.add_argument('--weight_decay', type=float, default=0, metavar='WD', help='the weight decay')

# =============================================================================
# [4] 데이터 처리 및 하이퍼파라미터
# =============================================================================
parser.add_argument('--max_threshold', default=10, type=float, help='the maximum threshold of error_rate')
parser.add_argument('--sample_way', type=str, default='FPS', metavar='sw', choices=['FPS', 'Random', 'CAGQ', 'Geometric'])
parser.add_argument('--need_resample', type=str2bool, default=True, help='Must be True to generate NPY files initially') # type 변경
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')

parser.add_argument('--regression_point_num', type=int, default=5, metavar='RPN', help='points in landmark regression')
parser.add_argument('--dataset_seed', type=int, default=1, metavar='S', help='train/test dataset random seed')
parser.add_argument('--num_points', type=int, default=2048, help='num of points to use')

parser.add_argument('--sigma', type=float, default=3.0, metavar='Sig', help='Gaussian Variance of heatmap')
parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors')
parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')

# =============================================================================
# [5] 모델 구조 설정
# =============================================================================
parser.add_argument('--calc_scores', type=str, default='softmax', metavar='cs', help='The way to calculate score')
parser.add_argument('--hidden', type=list, default=[[32], [32], [32], [32]], help='the hidden layers of ScoreNet')
parser.add_argument('--num_matrices', type=list, default=[8, 8, 8, 8], help='the number of weight banks')

# =============================================================================
# [6] 기타 / 평가 설정
# =============================================================================
parser.add_argument('--Eval_DataType', type=str, default="test", help='select npy train, test, sample')
parser.add_argument('--model_epoch', type=str, default="model_epoch_250.t7", help='load trained model file')
parser.add_argument('--landmark_num', type=int, default=40, metavar='L', help='the number of landmark(default: 40)')
parser.add_argument('--run_id', type=str, default='', help='Load specific run from backup (e.g., 20251128_2)')