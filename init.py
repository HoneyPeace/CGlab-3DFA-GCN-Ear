'''
@Author: Yuan Wang (Modified by Researcher 5)
@File: init.py
@Description: Initialize experiment (Unnecessary backups removed)
'''

import torch
import torch.nn as nn

def _init_(args):
    # [수정] 불필요한 Code_Backup 및 전역 Models 폴더 생성 로직 제거
    # 학습 결과와 모델은 train.py가 생성하는 'Run_XXX' 폴더에 통합 저장됩니다.
    print(f">> Experiment initialized: {args.exp_name}")

# 모델 가중치 초기화 함수 (필요 시 사용)
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)